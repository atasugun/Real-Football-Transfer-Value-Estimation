"""
Transfer Fee Prediction Model v3f
Goals + assists ONLY for Attack players (not Midfield).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path

if not Path("player_stats.csv").exists():
    raise FileNotFoundError("player_stats.csv not found. Run scrape_stats.py first.")

# ── 1. LOAD & MERGE ───────────────────────────────────────────────────────────
df = pd.read_csv('transfers.csv')
df['transfer_date'] = pd.to_datetime(df['transfer_date'])
df['year']  = df['transfer_date'].dt.year
df['month'] = df['transfer_date'].dt.month

df['saison'] = df['year'] - 1

df = df[(df['transfer_fee'] > 0) & (df['market_value_in_eur'] > 0)].copy()
print(f"Base dataset: {len(df):,} transfers")

stats = pd.read_csv('player_stats.csv')
stats = stats.rename(columns={'goals': 'prev_goals', 'assists': 'prev_assists'})

df = df.merge(stats[['player_id', 'saison', 'prev_goals', 'prev_assists']],
              on=['player_id', 'saison'], how='left')

# Only Attack gets goals/assists — Midfield, Defender, Goalkeeper → NaN
mask_not_attack = df['position'] != 'Attack'
df.loc[mask_not_attack, 'prev_goals']   = np.nan
df.loc[mask_not_attack, 'prev_assists'] = np.nan

print(f"Goals coverage (Attack only): "
      f"{df.loc[df['position']=='Attack', 'prev_goals'].notna().mean()*100:.1f}%")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
df['log_fee'] = np.log1p(df['transfer_fee'])
df['log_mv']  = np.log1p(df['market_value_in_eur'])

df['goals_assists'] = df['prev_goals'].fillna(0) + df['prev_assists'].fillna(0)
has_stats = df['prev_goals'].notna() | df['prev_assists'].notna()
df.loc[~has_stats, 'goals_assists'] = np.nan

df['log_goals_assists'] = np.log1p(df['goals_assists'])

# ── 3. ENCODE CATEGORICALS ────────────────────────────────────────────────────
cat_cols = ['position', 'nationality', 'league_from', 'league_to', 'transfer_window']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ── 4. FEATURES ───────────────────────────────────────────────────────────────
FEATURES_BASE = [
    'log_mv', 'age', 'contract_year_left', 'year', 'month',
    'position', 'nationality', 'league_from', 'league_to', 'transfer_window',
]
FEATURES_NEW = ['prev_goals', 'prev_assists', 'log_goals_assists']
FEATURES = FEATURES_BASE + FEATURES_NEW
TARGET = 'log_fee'

# ── 5. TEMPORAL SPLIT ─────────────────────────────────────────────────────────
split_date = df['transfer_date'].quantile(0.8)
train_mask = df['transfer_date'] <= split_date
test_mask  = ~train_mask

X_train, X_test = df.loc[train_mask, FEATURES], df.loc[test_mask, FEATURES]
y_train, y_test = df.loc[train_mask, TARGET],   df.loc[test_mask, TARGET]

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ── 6. EVALUATE ───────────────────────────────────────────────────────────────
def evaluate(name, pred_log, actual_log):
    pred_eur   = np.expm1(pred_log)
    actual_eur = np.expm1(actual_log)
    r2       = r2_score(actual_log, pred_log)
    mae_log  = mean_absolute_error(actual_log, pred_log)
    rmse_log = np.sqrt(mean_squared_error(actual_log, pred_log))
    mae_eur  = mean_absolute_error(actual_eur, pred_eur)
    mape     = np.mean(np.abs((actual_eur - pred_eur) / actual_eur)) * 100
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  R²       (log): {r2:.4f}")
    print(f"  MAE      (log): {mae_log:.4f}")
    print(f"  RMSE     (log): {rmse_log:.4f}")
    print(f"  MAE      (€):   {mae_eur/1e6:.2f}M")
    print(f"  MAPE     (%):   {mape:.1f}%")
    return {'name': name, 'r2': r2, 'mae_log': mae_log,
            'rmse_log': rmse_log, 'mae_eur': mae_eur, 'mape': mape}

results = []

# ── 7. BASELINE ───────────────────────────────────────────────────────────────
tm_pred_log = X_test['log_mv'].values
results.append(evaluate("Transfermarkt Baseline", tm_pred_log, y_test.values))

# ── 8. V1 (no stats) ──────────────────────────────────────────────────────────
lgb_params = {
    'objective': 'regression', 'metric': 'rmse', 'num_leaves': 63,
    'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'min_child_samples': 20, 'lambda_l1': 0.1,
    'lambda_l2': 0.1, 'verbose': -1, 'random_state': 42,
}

dtrain_v1 = lgb.Dataset(X_train[FEATURES_BASE], label=y_train, categorical_feature=cat_cols)
dval_v1   = lgb.Dataset(X_test[FEATURES_BASE],  label=y_test,  categorical_feature=cat_cols, reference=dtrain_v1)
lgb_v1 = lgb.train(lgb_params, dtrain_v1, num_boost_round=2000,
                    valid_sets=[dval_v1],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(period=-1)])
xgb_v1 = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                            reg_lambda=0.1, early_stopping_rounds=50,
                            eval_metric='rmse', random_state=42, verbosity=0)
xgb_v1.fit(X_train[FEATURES_BASE], y_train, eval_set=[(X_test[FEATURES_BASE], y_test)], verbose=False)
v1_ens_log = (lgb_v1.predict(X_test[FEATURES_BASE]) + xgb_v1.predict(X_test[FEATURES_BASE])) / 2
results.append(evaluate("v1 Ensemble (no stats)", v1_ens_log, y_test.values))

# ── 9. V3 (Attack+Mid stats) ──────────────────────────────────────────────────
# Re-run v3 for comparison — reload with Attack+Mid stats
df2 = pd.read_csv('transfers.csv')
df2['transfer_date'] = pd.to_datetime(df2['transfer_date'])
df2['year']  = df2['transfer_date'].dt.year
df2['month'] = df2['transfer_date'].dt.month
df2['saison'] = df2['year'] - 1
df2 = df2[(df2['transfer_fee'] > 0) & (df2['market_value_in_eur'] > 0)].copy()
df2 = df2.merge(stats[['player_id', 'saison', 'prev_goals', 'prev_assists']], on=['player_id','saison'], how='left')
mask2 = ~df2['position'].isin(['Attack', 'Midfield'])
df2.loc[mask2, 'prev_goals'] = np.nan
df2.loc[mask2, 'prev_assists'] = np.nan
df2['log_fee'] = np.log1p(df2['transfer_fee'])
df2['log_mv']  = np.log1p(df2['market_value_in_eur'])
df2['goals_assists'] = df2['prev_goals'].fillna(0) + df2['prev_assists'].fillna(0)
has2 = df2['prev_goals'].notna() | df2['prev_assists'].notna()
df2.loc[~has2, 'goals_assists'] = np.nan
df2['log_goals_assists'] = np.log1p(df2['goals_assists'])
for col in cat_cols:
    le = LabelEncoder()
    df2[col] = le.fit_transform(df2[col].astype(str))
train2 = df2['transfer_date'] <= df2['transfer_date'].quantile(0.8)
test2  = ~train2
X_tr2, X_te2 = df2.loc[train2, FEATURES], df2.loc[test2, FEATURES]
y_tr2, y_te2 = df2.loc[train2, TARGET],   df2.loc[test2, TARGET]
dtrain_v3 = lgb.Dataset(X_tr2, label=y_tr2, categorical_feature=cat_cols)
dval_v3   = lgb.Dataset(X_te2, label=y_te2, categorical_feature=cat_cols, reference=dtrain_v3)
lgb_v3 = lgb.train(lgb_params, dtrain_v3, num_boost_round=2000, valid_sets=[dval_v3],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
xgb_v3 = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                            reg_lambda=0.1, early_stopping_rounds=50,
                            eval_metric='rmse', random_state=42, verbosity=0)
xgb_v3.fit(X_tr2, y_tr2, eval_set=[(X_te2, y_te2)], verbose=False)
v3_ens_log = (lgb_v3.predict(X_te2) + xgb_v3.predict(X_te2)) / 2
results.append(evaluate("v3 Ensemble (Attack+Mid stats)", v3_ens_log, y_te2.values))

# ── 10. V3F (Attack only stats) ───────────────────────────────────────────────
dtrain_f = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
dval_f   = lgb.Dataset(X_test,  label=y_test,  categorical_feature=cat_cols, reference=dtrain_f)
lgb_f = lgb.train(lgb_params, dtrain_f, num_boost_round=2000, valid_sets=[dval_f],
                   callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
xgb_f = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                           reg_lambda=0.1, early_stopping_rounds=50,
                           eval_metric='rmse', random_state=42, verbosity=0)
xgb_f.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
v3f_ens_log = (lgb_f.predict(X_test) + xgb_f.predict(X_test)) / 2
results.append(evaluate("v3f Ensemble (Attack only stats)", v3f_ens_log, y_test.values))

# ── 11. SUMMARY ───────────────────────────────────────────────────────────────
print("\n\n" + "═"*60)
print("  SUMMARY")
print("═"*60)
res_df = pd.DataFrame(results)
res_df['mae_eur_M'] = res_df['mae_eur'] / 1e6
print(res_df[['name','r2','mae_log','mape','mae_eur_M']].to_string(index=False))

# ── 12. KANE CASE STUDY ───────────────────────────────────────────────────────
split_date2 = df['transfer_date'].quantile(0.8)
df_orig = pd.read_csv('transfers.csv')
df_orig['transfer_date'] = pd.to_datetime(df_orig['transfer_date'])
df_orig = df_orig[(df_orig['transfer_fee'] > 0) & (df_orig['transfer_date'] > split_date2)].copy()
df_orig = df_orig.reset_index(drop=True)

print("\n\n  Case study — Attack players:")
print("  " + "─"*75)
worst_names = ['Harry Kane', 'Moisés Caicedo', 'Cole Palmer', 'Jérémy Doku', 'Gonçalo Ramos']
for name in worst_names:
    match = df_orig[df_orig['player_name'] == name]
    if match.empty:
        continue
    idx = match.index[0]
    if idx >= len(v3f_ens_log):
        continue
    actual  = np.expm1(y_test.values[idx])
    tm_pred = np.expm1(X_test['log_mv'].values[idx])
    v1_pred = np.expm1(v1_ens_log[idx])
    v3_pred = np.expm1(v3_ens_log[idx]) if idx < len(v3_ens_log) else float('nan')
    vf_pred = np.expm1(v3f_ens_log[idx])
    goals   = X_test['prev_goals'].values[idx]
    assists = X_test['prev_assists'].values[idx]
    print(f"  {name:22s} | actual={actual/1e6:.1f}M | TM={tm_pred/1e6:.1f}M | "
          f"v1={v1_pred/1e6:.1f}M | v3={v3_pred/1e6:.1f}M | v3f={vf_pred/1e6:.1f}M | "
          f"G={goals:.0f} A={assists:.0f}")
