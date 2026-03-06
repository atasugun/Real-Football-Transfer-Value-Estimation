"""Neural Network (MLP) vs LightGBM comparison on the same train/test split."""
import io, contextlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from deployment.player_valuation import (
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

# ── MLP: scale features (mandatory for NNs) ───────────────────────────────────
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Inner val split for early stopping
X_tr_s, X_val_s, y_tr, y_val = train_test_split(
    X_train_s, y_train, test_size=0.12, random_state=0
)

print("Training MLP (256-128-64, ReLU, Adam)...")
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.12,
    n_iter_no_change=30,
    random_state=42,
    verbose=False,
)
mlp.fit(X_train_s, y_train)
print(f"  Stopped at iteration: {mlp.n_iter_}")

log_preds_mlp = mlp.predict(X_test_s)
y_pred_mlp    = np.expm1(log_preds_mlp)

# ── LightGBM ───────────────────────────────────────────────────────────────────
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

mlp_stats  = summarise(y_true, y_pred_mlp,  y_test, log_preds_mlp)
lgbm_stats = summarise(y_true, y_pred_lgbm, y_test, log_preds_lgbm)

# ── overall table ─────────────────────────────────────────────────────────────
SEP = "-" * 62
print(f"\n{SEP}")
print(f"  {'Metric':<24}  {'MLP Neural Net':>15}  {'LightGBM':>15}")
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
    a, b = mlp_stats[name], lgbm_stats[name]
    if fmt == "pct":
        print(f"  {name:<24}  {a:>14.1f}%  {b:>14.1f}%")
    elif "," in fmt:
        print(f"  {name:<24}  {a:>15,.0f}  {b:>15,.0f}")
    else:
        print(f"  {name:<24}  {a:>15.4f}  {b:>15.4f}")
print(SEP)

mlp_err  = np.abs(y_pred_mlp  - y_true)
lgbm_err = np.abs(y_pred_lgbm - y_true)
print(f"\n  LightGBM closer:  {(lgbm_err < mlp_err).mean()*100:.1f}% of transfers")
print(f"  MLP closer:       {(lgbm_err >= mlp_err).mean()*100:.1f}% of transfers")

# ── by market value range ──────────────────────────────────────────────────────
y_mv     = df_test["market_value_in_eur"].values
mlp_pct  = mlp_err  / y_true * 100
lgbm_pct = lgbm_err / y_true * 100
bins     = [0, 1e6, 5e6, 15e6, 50e6, np.inf]
labels   = ["<1M", "1-5M", "5-15M", "15-50M", ">50M"]
mv_band  = pd.cut(y_mv, bins=bins, labels=labels).to_numpy()

print(f"\n  Median % error by market value range:")
print(f"  {'MV Range':<8}  {'n':>5}  {'MLP':>8}  {'LightGBM':>9}  {'LGBM wins':>9}")
for b in labels:
    mask = mv_band == b
    if not mask.any():
        continue
    lgbm_wins = (lgbm_err[mask] < mlp_err[mask]).mean() * 100
    print(f"  {b:<8}  {mask.sum():>5}  "
          f"{np.median(mlp_pct[mask]):>7.1f}%  "
          f"{np.median(lgbm_pct[mask]):>8.1f}%  "
          f"{lgbm_wins:>8.1f}%")
