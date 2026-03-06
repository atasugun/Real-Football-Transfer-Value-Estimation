"""Tuned Random Forest: RandomizedSearchCV on train set, then final eval on test set."""
import io, contextlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from player_valuation import (
    load_and_prepare, build_target_encodings,
    encode_features, get_feature_cols, train_model
)

# ── shared dataset and split ───────────────────────────────────────────────────
df_full, _, _, _ = load_and_prepare()
encodings    = build_target_encodings(df_full)
df_enc       = encode_features(df_full, encodings)
feature_cols = get_feature_cols(df_enc)

df_train, df_test = train_test_split(df_enc, test_size=0.15, random_state=42)

X_train = df_train[feature_cols].values
y_train = df_train["log_fee"].values
X_test  = df_test[feature_cols].values
y_test  = df_test["log_fee"].values
y_true  = np.expm1(y_test)

# ── RandomizedSearchCV (tune on train only, never touches test) ────────────────
param_dist = {
    "n_estimators":     [300, 500],       # fixed low during search for speed
    "max_features":     [0.2, 0.3, 0.4, 0.5, "sqrt"],
    "min_samples_leaf": [1, 2, 3, 5, 8],
    "min_samples_split":[2, 5, 10],
    "max_depth":        [None, 20, 30, 40],
}

print("Running RandomizedSearchCV (n_iter=30, cv=5)... this may take a few minutes")
search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X_train, y_train)

best = search.best_params_
print(f"\nBest params: {best}")
print(f"Best CV R2:  {search.best_score_:.4f}")

# ── Retrain with best params + more trees ─────────────────────────────────────
print("\nRetraining with best params + 2000 trees...")
rf_best = RandomForestRegressor(
    n_estimators=2000,
    max_features=best["max_features"],
    min_samples_leaf=best["min_samples_leaf"],
    min_samples_split=best["min_samples_split"],
    max_depth=best["max_depth"],
    random_state=42,
    n_jobs=-1,
)
rf_best.fit(X_train, y_train)
print("  Done.")

log_preds_rf = rf_best.predict(X_test)
y_pred_rf    = np.expm1(log_preds_rf)

# ── LightGBM for comparison ────────────────────────────────────────────────────
print("Training LightGBM for comparison...")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    lgbm = train_model(df_train[feature_cols], df_train["log_fee"],
                       feature_cols, label="lgbm")
for line in buf.getvalue().splitlines():
    if "Best n_estimators" in line:
        print(line)

log_preds_lgbm = lgbm.predict(df_test[feature_cols])
y_pred_lgbm    = np.expm1(log_preds_lgbm)

# ── helper ────────────────────────────────────────────────────────────────────
def summarise(y_true, y_hat, y_true_log, y_hat_log):
    pct = np.abs(y_hat - y_true) / y_true * 100
    return {
        "R2 (log)":     r2_score(y_true_log, y_hat_log),
        "R2 (EUR)":     r2_score(y_true, y_hat),
        "MAE (EUR)":    mean_absolute_error(y_true, y_hat),
        "Median AE":    median_absolute_error(y_true, y_hat),
        "Median % err": np.median(pct),
        "Within 25%":   (pct < 25).mean() * 100,
        "Within 50%":   (pct < 50).mean() * 100,
    }

rf_stats   = summarise(y_true, y_pred_rf,   y_test, log_preds_rf)
lgbm_stats = summarise(y_true, y_pred_lgbm, y_test, log_preds_lgbm)

# ── overall table ─────────────────────────────────────────────────────────────
SEP = "-" * 62
print(f"\n{SEP}")
print(f"  {'Metric':<24}  {'RF (tuned)':>15}  {'LightGBM':>15}")
print(SEP)
rows = [
    ("R2 (log)",     ".4f"),
    ("R2 (EUR)",     ".4f"),
    ("MAE (EUR)",    ",.0f"),
    ("Median AE",    ",.0f"),
    ("Median % err", "pct"),
    ("Within 25%",   "pct"),
    ("Within 50%",   "pct"),
]
for name, fmt in rows:
    a, b = rf_stats[name], lgbm_stats[name]
    if fmt == "pct":
        print(f"  {name:<24}  {a:>14.1f}%  {b:>14.1f}%")
    elif "," in fmt:
        print(f"  {name:<24}  {a:>15,.0f}  {b:>15,.0f}")
    else:
        print(f"  {name:<24}  {a:>15.4f}  {b:>15.4f}")
print(SEP)

rf_err   = np.abs(y_pred_rf   - y_true)
lgbm_err = np.abs(y_pred_lgbm - y_true)
print(f"\n  LightGBM closer:    {(lgbm_err < rf_err).mean()*100:.1f}% of transfers")
print(f"  RF (tuned) closer:  {(lgbm_err >= rf_err).mean()*100:.1f}% of transfers")

# ── by market value range ──────────────────────────────────────────────────────
y_mv     = df_test["market_value_in_eur"].values
rf_pct   = rf_err   / y_true * 100
lgbm_pct = lgbm_err / y_true * 100
bins     = [0, 1e6, 5e6, 15e6, 50e6, np.inf]
labels   = ["<1M", "1-5M", "5-15M", "15-50M", ">50M"]
mv_band  = pd.cut(y_mv, bins=bins, labels=labels).to_numpy()

print(f"\n  Median % error by market value range:")
print(f"  {'MV Range':<8}  {'n':>5}  {'RF':>8}  {'LightGBM':>9}  {'LGBM wins':>9}")
for b in labels:
    mask = mv_band == b
    if not mask.any():
        continue
    lgbm_wins = (lgbm_err[mask] < rf_err[mask]).mean() * 100
    print(f"  {b:<8}  {mask.sum():>5}  "
          f"{np.median(rf_pct[mask]):>7.1f}%  "
          f"{np.median(lgbm_pct[mask]):>8.1f}%  "
          f"{lgbm_wins:>8.1f}%")

# ── feature importances ────────────────────────────────────────────────────────
fi = pd.Series(rf_best.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n  RF (tuned) feature importances:")
print(fi.to_string())
