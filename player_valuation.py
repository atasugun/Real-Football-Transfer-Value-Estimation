"""
Football Player Transfer Value Model  (v3)
==========================================
Predicts player market value from player-side attributes,
then produces a valuation matrix across possible league destinations
and transfer windows.

v3 adds: prev_goals + prev_assists (previous season) for Attack & Midfield
         players, sourced from player_stats.csv.

Features used for valuation:
  nationality, league_from, position, age, transfer_season,
  contract_year_left, [prev_goals, prev_assists for Attack/Midfield]

Context scenarios:
  league_to, transfer_window (summer / winter)

Null handling:
  contract_year_left: imputed with median + binary contract_missing flag
  prev_goals/prev_assists: NaN for Defenders/Goalkeepers,
                           NaN (handled natively by XGBoost) if not found
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

DATA_PATH  = os.path.join(os.path.dirname(__file__), "transfers.csv")
STATS_PATH = os.path.join(os.path.dirname(__file__), "player_stats.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "valuation_model.pkl")

DISPLAY_LEAGUES = ["GB1", "ES1", "IT1", "L1", "FR1", "PO1", "NL1", "TR1", "BE1", "RU1"]

ATTACK_MID = {"Attack", "Midfield"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(path: str = DATA_PATH, stats_path: str = STATS_PATH):
    df = pd.read_csv(path)

    df["transfer_date"] = pd.to_datetime(df["transfer_date"])
    df["year"]          = df["transfer_date"].dt.year

    # season → numeric start year  (e.g. "23/24" → 2023)
    df["season_year"] = df["transfer_season"].str[:2].astype(int) + 2000

    # Season before transfer (for stats lookup)
    df["saison"] = df["year"] - 1

    # contract_year_left: missing indicator + impute with median
    df["contract_missing"] = df["contract_year_left"].isnull().astype(int)
    median_contract = df["contract_year_left"].median()
    df["contract_year_left"] = df["contract_year_left"].fillna(median_contract)

    # log-transform market value and target
    df["log_market_value"] = np.log1p(df["market_value_in_eur"])
    df["log_fee"]          = np.log1p(df["transfer_fee"])

    # Engineered features
    df["age_sq"]          = df["age"] ** 2
    df["contract_urgency"] = 1.0 / (df["contract_year_left"] + 0.5)

    season_means = df.groupby("season_year")["log_fee"].mean()
    from numpy.polynomial.polynomial import polyfit
    years = season_means.index.values.astype(float)
    vals  = season_means.values
    c0, c1 = polyfit(years, vals, 1)
    df["season_fee_mean"] = df["season_year"].map(season_means)

    # ── v3: merge goals + assists from player_stats.csv ──────────────────────
    if os.path.exists(stats_path):
        stats = pd.read_csv(stats_path)
        stats = stats.rename(columns={"goals": "prev_goals", "assists": "prev_assists"})
        df = df.merge(stats[["player_id", "saison", "prev_goals", "prev_assists"]],
                      on=["player_id", "saison"], how="left")
        # Defenders & Goalkeepers: set to NaN (goals/assists not predictive)
        mask_non_atk_mid = ~df["position"].isin(ATTACK_MID)
        df.loc[mask_non_atk_mid, "prev_goals"]   = np.nan
        df.loc[mask_non_atk_mid, "prev_assists"] = np.nan

        # Combined goals+assists (only when at least one is available)
        df["goals_assists"] = df["prev_goals"].fillna(0) + df["prev_assists"].fillna(0)
        has_stats = df["prev_goals"].notna() | df["prev_assists"].notna()
        df.loc[~has_stats, "goals_assists"] = np.nan
        df["log_goals_assists"] = np.log1p(df["goals_assists"])

        atk_mid = df["position"].isin(ATTACK_MID)
        cov = df.loc[atk_mid, "prev_goals"].notna().mean() * 100
        print(f"  Goals/assists coverage (Attack+Mid): {cov:.1f}%")
    else:
        print(f"  WARNING: {stats_path} not found — training without goals/assists")
        df["prev_goals"] = np.nan
        df["prev_assists"] = np.nan
        df["goals_assists"] = np.nan
        df["log_goals_assists"] = np.nan

    return df, median_contract, season_means, (c0, c1)


def build_target_encodings(df: pd.DataFrame, target_col: str = "log_fee") -> dict:
    encodings = {}
    global_mean = df[target_col].mean()

    for col in ["nationality", "league_from", "league_to"]:
        agg = df.groupby(col)[target_col].agg(["mean", "count"])
        k = 20
        agg["smoothed"] = (agg["mean"] * agg["count"] + global_mean * k) / (agg["count"] + k)
        encodings[col] = agg["smoothed"].to_dict()
        encodings[f"{col}_default"] = global_mean

    return encodings


def encode_features(df: pd.DataFrame, encodings: dict) -> pd.DataFrame:
    out = df.copy()

    for col in ["nationality", "league_from", "league_to"]:
        out[f"{col}_enc"] = (
            out[col].map(encodings[col]).fillna(encodings[f"{col}_default"])
        )

    out["league_step"] = out["league_to_enc"] - out["league_from_enc"]
    out["is_summer"]   = (out["transfer_window"] == "summer").astype(int)

    pos_dummies = pd.get_dummies(out["position"], prefix="pos")
    out = pd.concat([out, pos_dummies], axis=1)

    return out


def get_feature_cols(df_encoded: pd.DataFrame) -> list:
    pos_cols = [c for c in df_encoded.columns if c.startswith("pos_")]
    return [
        "log_market_value",
        "nationality_enc", "league_from_enc", "league_to_enc", "league_step",
        "season_year", "season_fee_mean",
        "age", "age_sq",
        "contract_year_left", "contract_urgency", "contract_missing",
        "is_summer",
        # v3: performance features (NaN for Defenders/Goalkeepers)
        "prev_goals", "prev_assists", "log_goals_assists",
    ] + pos_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                feature_cols: list, label: str = ""):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.12, random_state=0
    )

    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=15,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.05,
        early_stopping_rounds=100,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    tag = f" [{label}]" if label else ""
    print(f"  Best n_estimators{tag}: {model.best_iteration}")

    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"Feature importances{tag}:")
    print(fi.to_string())
    print()

    return model



def train_lgb_model(X_train: pd.DataFrame, y_train: pd.Series,
                    feature_cols: list, label: str = ""):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.12, random_state=0
    )
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
    )
    tag = f" [{label}]" if label else ""
    print(f"  LGB best n_estimators{tag}: {model.best_iteration_}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 3. POST-PROCESSING MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────
# XGBoost trees saturate at the high end of market value because training data
# is sparse above ~€80M.  For players above this threshold we apply a dampened
# power-law multiplier so the prediction scales with the actual market value.
#
#   multiplier = (market_value / MV_SATURATION) ^ MV_ALPHA
#
# MV_ALPHA=0.55: meaningful correction for elite players at a small R² cost.
# Example: €200M MV → (200/80)^0.55 ≈ 1.65 × base prediction (~€132M)

MV_SATURATION = 80_000_000   # EUR — top of the well-sampled training range
MV_ALPHA      = 0.55         # dampening exponent


def _apply_mv_multiplier(raw_fee: float, market_value_in_eur: float) -> float:
    """Scale up predictions for players whose market value exceeds the training ceiling."""
    if market_value_in_eur <= MV_SATURATION:
        return raw_fee
    multiplier = (market_value_in_eur / MV_SATURATION) ** MV_ALPHA
    return raw_fee * multiplier


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALUATION
# ─────────────────────────────────────────────────────────────────────────────

class PlayerValuator:
    """
    Wraps the trained model + encodings to produce valuations.

    v3 supports optional prev_goals / prev_assists for Attack & Midfield
    players. Pass None (or omit) if unavailable — XGBoost handles NaN natively.
    """

    def __init__(self, model, encodings: dict, feature_cols: list,
                 median_contract: float, all_positions: list,
                 season_means: dict, season_trend: tuple,
                 stats_lookup: dict = None,
                 lgb_model=None):
        self.model           = model
        self.lgb_model       = lgb_model
        self.encodings       = encodings
        self.feature_cols    = feature_cols
        self.median_contract = median_contract
        self.all_positions   = all_positions
        self.season_means    = season_means
        self.season_trend    = season_trend
        # stats_lookup: {(player_id, saison): (goals, assists)}
        self.stats_lookup    = stats_lookup or {}

    def _season_fee_mean(self, season_year: int) -> float:
        if season_year in self.season_means:
            return float(self.season_means[season_year])
        c0, c1 = self.season_trend
        return float(c0 + c1 * season_year)

    def get_player_stats(self, player_id, season_year: int):
        """Look up (goals, assists) for a player in the season before transfer.
        Returns (goals, assists) floats or (None, None) if not found."""
        if player_id is None:
            return None, None
        saison = int(season_year) - 1
        key = (int(player_id), saison)
        entry = self.stats_lookup.get(key)
        if entry is None:
            return None, None
        return entry  # (goals, assists)

    def _make_row(self, nationality, league_from, position, age,
                  season_year, contract_year_left, contract_missing,
                  league_to, is_summer, log_market_value,
                  prev_goals=None, prev_assists=None) -> dict:

        nat_enc = self.encodings["nationality"].get(nationality,  self.encodings["nationality_default"])
        lf_enc  = self.encodings["league_from"].get(league_from,  self.encodings["league_from_default"])
        lt_enc  = self.encodings["league_to"].get(league_to,      self.encodings["league_to_default"])

        # Goals/assists only for Attack & Midfield
        if position in ATTACK_MID:
            g = float(prev_goals)   if prev_goals   is not None and not (isinstance(prev_goals,   float) and np.isnan(prev_goals))   else np.nan
            a = float(prev_assists) if prev_assists  is not None and not (isinstance(prev_assists, float) and np.isnan(prev_assists)) else np.nan
            if not (np.isnan(g) and np.isnan(a)):
                ga = (g if not np.isnan(g) else 0.0) + (a if not np.isnan(a) else 0.0)
                log_ga = np.log1p(ga)
            else:
                ga, log_ga = np.nan, np.nan
        else:
            g = a = ga = log_ga = np.nan

        row = {
            "log_market_value":   log_market_value,
            "nationality_enc":    nat_enc,
            "league_from_enc":    lf_enc,
            "league_to_enc":      lt_enc,
            "league_step":        lt_enc - lf_enc,
            "season_year":        season_year,
            "season_fee_mean":    self._season_fee_mean(season_year),
            "age":                age,
            "age_sq":             age ** 2,
            "contract_year_left": contract_year_left,
            "contract_urgency":   1.0 / (contract_year_left + 0.5),
            "contract_missing":   contract_missing,
            "is_summer":          int(is_summer),
            "prev_goals":         g,
            "prev_assists":       a,
            "log_goals_assists":  log_ga,
        }

        for pos in self.all_positions:
            row[f"pos_{pos}"] = int(pos == position)

        return row

    def value_matrix(self,
                     nationality: str,
                     league_from: str,
                     position: str,
                     age: float,
                     transfer_season: str,
                     market_value_in_eur: float,
                     contract_year_left,
                     display_leagues: list = DISPLAY_LEAGUES,
                     prev_goals=None,
                     prev_assists=None) -> pd.DataFrame:

        season_year = int(transfer_season[:2]) + 2000
        log_mv = np.log1p(market_value_in_eur)
        if contract_year_left is None:
            cyl, cm = self.median_contract, 1
        else:
            cyl, cm = float(contract_year_left), 0

        rows = []
        index = []
        for lg in display_leagues:
            for window, is_summer in [("summer", 1), ("winter", 0)]:
                rows.append(self._make_row(
                    nationality, league_from, position, age,
                    season_year, cyl, cm, lg, is_summer, log_mv,
                    prev_goals, prev_assists,
                ))
                index.append((lg, window))

        X = pd.DataFrame(rows)[self.feature_cols]
        log_preds = self.model.predict(X)
        if self.lgb_model is not None:
            log_preds = (log_preds + self.lgb_model.predict(X)) / 2
        preds = np.expm1(log_preds)
        preds = np.array([_apply_mv_multiplier(p, market_value_in_eur) for p in preds])

        result = pd.DataFrame(
            {"predicted_fee": preds},
            index=pd.MultiIndex.from_tuples(index, names=["league_to", "window"])
        ).unstack("window")[["predicted_fee"]].droplevel(0, axis=1)

        result.columns.name = None
        result = result[["summer", "winter"]]
        result = result.round(0).astype(int)
        return result

    def value_point(self, nationality, league_from, position, age,
                    transfer_season, market_value_in_eur, contract_year_left,
                    league_to, transfer_window,
                    prev_goals=None, prev_assists=None) -> float:
        season_year = int(transfer_season[:2]) + 2000
        log_mv = np.log1p(market_value_in_eur)
        if contract_year_left is None:
            cyl, cm = self.median_contract, 1
        else:
            cyl, cm = float(contract_year_left), 0

        row = self._make_row(
            nationality, league_from, position, age, season_year,
            cyl, cm, league_to, transfer_window == "summer", log_mv,
            prev_goals, prev_assists,
        )
        X = pd.DataFrame([row])[self.feature_cols]
        log_pred = self.model.predict(X)[0]
        if self.lgb_model is not None:
            log_pred = (log_pred + self.lgb_model.predict(X)[0]) / 2
        raw = float(np.expm1(log_pred))
        return _apply_mv_multiplier(raw, market_value_in_eur)

    def save(self, path: str = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "PlayerValuator":
        import sys, types

        class _Remapper(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__" and name == "PlayerValuator":
                    return PlayerValuator
                return super().find_class(module, name)

        with open(path, "rb") as f:
            obj = _Remapper(f).load()
        # backward compat: old pkl may not have stats_lookup
        if not hasattr(obj, "stats_lookup"):
            obj.stats_lookup = {}
        if not hasattr(obj, "lgb_model"):
            obj.lgb_model = None
        print(f"Model loaded from {path}")
        return obj

    @classmethod
    def train(cls, data_path: str = DATA_PATH,
              stats_path: str = STATS_PATH) -> "PlayerValuator":
        print("Loading data...")
        df, median_contract, season_means, season_trend = load_and_prepare(data_path, stats_path)

        print("Building encodings...")
        encodings = build_target_encodings(df)

        print("Encoding features...")
        df_enc = encode_features(df, encodings)
        feature_cols  = get_feature_cols(df_enc)
        all_positions = sorted(df["position"].unique().tolist())

        df_train, df_test = train_test_split(df_enc, test_size=0.15, random_state=42)

        print("Training XGBoost model...")
        model = train_model(df_train[feature_cols], df_train["log_fee"], feature_cols)
        print("Training LightGBM model...")
        lgb_model = train_lgb_model(df_train[feature_cols], df_train["log_fee"], feature_cols)

        # Evaluate ensemble
        log_preds  = (model.predict(df_test[feature_cols]) + lgb_model.predict(df_test[feature_cols])) / 2
        y_test_log = df_test["log_fee"].values
        y_test     = np.expm1(y_test_log)
        y_pred     = np.expm1(log_preds)

        mae     = mean_absolute_error(y_test, y_pred)
        r2      = r2_score(y_test, y_pred)
        mae_log = mean_absolute_error(y_test_log, log_preds)
        r2_log  = r2_score(y_test_log, log_preds)

        SEP = "-" * 50
        print(f"\n{SEP}")
        print("Model evaluation (test set, {:.0f} samples)".format(len(df_test)))
        print(SEP)
        print(f"  R2  (log scale):     {r2_log:.4f}")
        print(f"  MAE (log scale):     {mae_log:.4f}")
        print(f"  MAE (EUR):           EUR {mae:,.0f}")
        print(f"  R2  (EUR):           {r2:.4f}")
        print(f"{SEP}\n")

        # Build stats lookup for inference: {(player_id, saison): (goals, assists)}
        stats_lookup = {}
        if os.path.exists(stats_path):
            stats = pd.read_csv(stats_path)
            for _, row in stats.iterrows():
                key = (int(row["player_id"]), int(row["saison"]))
                g = float(row["goals"])   if pd.notna(row.get("goals"))   else None
                a = float(row["assists"]) if pd.notna(row.get("assists")) else None
                stats_lookup[key] = (g, a)

        valuator = cls(model, encodings, feature_cols, median_contract,
                       all_positions, season_means.to_dict(), season_trend,
                       stats_lookup, lgb_model)
        return valuator


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    val = PlayerValuator.train()
    val.save()

    print("\n" + "="*60)
    print("EXAMPLE: 24-year-old French midfielder from Ligue 1")
    print("         2 years left on contract, 8G+5A last season")
    print("="*60)
    matrix = val.value_matrix(
        nationality="France",
        league_from="FR1",
        position="Midfield",
        age=24,
        transfer_season="24/25",
        market_value_in_eur=15_000_000,
        contract_year_left=2.0,
        prev_goals=8,
        prev_assists=5,
    )
    print(matrix.to_string())
