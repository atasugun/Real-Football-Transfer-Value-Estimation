"""Three-way comparison: market_value alone vs single LightGBM vs segmented LightGBM."""
import io, contextlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
from deployment.player_valuation import (
    PlayerValuator, load_and_prepare, build_target_encodings,
    encode_features, get_feature_cols, train_model, MODEL_PATH
)

# ── shared dataset and split (same seed as training) ──────────────────────────
df_full, _, _, _ = load_and_prepare()
encodings    = build_target_encodings(df_full)
df_enc       = encode_features(df_full, encodings)
feature_cols = get_feature_cols(df_enc)

df_train, df_test = train_test_split(df_enc, test_size=0.15, random_state=42)

y_true     = np.expm1(df_test["log_fee"].values)
y_true_log = df_test["log_fee"].values
y_mv       = df_test["market_value_in_eur"].values

# ── single model: train on all training data, suppress verbose output ──────────
print("Training single model for comparison...")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    model_single = train_model(df_train[feature_cols], df_train["log_fee"],
                               feature_cols, label="single")
for line in buf.getvalue().splitlines():
    if "Best n_estimators" in line:
        print(line)

log_preds_sin = model_single.predict(df_test[feature_cols])
y_pred_sin    = np.expm1(log_preds_sin)

# ── segmented model: load from pkl ────────────────────────────────────────────
print("Loading segmented model...")
val = PlayerValuator.load(MODEL_PATH)
log_preds_seg = val._predict(df_test[feature_cols])
y_pred_seg    = np.expm1(log_preds_seg)

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

mv_stats  = summarise(y_true, y_mv,       y_true_log, np.log1p(y_mv))
sin_stats = summarise(y_true, y_pred_sin, y_true_log, log_preds_sin)
seg_stats = summarise(y_true, y_pred_seg, y_true_log, log_preds_seg)

# ── overall table ─────────────────────────────────────────────────────────────
SEP = "-" * 74
print(f"\n{SEP}")
print(f"  {'Metric':<24}  {'Market Value':>13}  {'Single Model':>13}  {'Segmented':>13}")
print(SEP)
metric_rows = [
    ("R2 (log)",     ".4f"),
    ("R2 (EUR)",     ".4f"),
    ("MAE (EUR)",    ",.0f"),
    ("Median AE",    ",.0f"),
    ("Median % err", "pct"),
    ("Within 25%",   "pct"),
    ("Within 50%",   "pct"),
]
for name, fmt in metric_rows:
    a, b, c = mv_stats[name], sin_stats[name], seg_stats[name]
    if fmt == "pct":
        print(f"  {name:<24}  {a:>12.1f}%  {b:>12.1f}%  {c:>12.1f}%")
    elif "," in fmt:
        print(f"  {name:<24}  {a:>13,.0f}  {b:>13,.0f}  {c:>13,.0f}")
    else:
        print(f"  {name:<24}  {a:>13.4f}  {b:>13.4f}  {c:>13.4f}")
print(SEP)

mv_err  = np.abs(y_mv       - y_true)
sin_err = np.abs(y_pred_sin - y_true)
seg_err = np.abs(y_pred_seg - y_true)
print(f"\n  Single   closer than market value: {(sin_err < mv_err).mean()*100:.1f}% of transfers")
print(f"  Segmented closer than market value: {(seg_err < mv_err).mean()*100:.1f}% of transfers")
print(f"  Segmented closer than single:       {(seg_err < sin_err).mean()*100:.1f}% of transfers")

# ── by market value range ──────────────────────────────────────────────────────
mv_pct  = mv_err  / y_true * 100
sin_pct = sin_err / y_true * 100
seg_pct = seg_err / y_true * 100

bins    = [0, 1e6, 5e6, 15e6, 50e6, np.inf]
labels  = ["<1M", "1-5M", "5-15M", "15-50M", ">50M"]
mv_band = pd.cut(y_mv, bins=bins, labels=labels).to_numpy()

print(f"\n  Median % error by market value range:")
print(f"  {'MV Range':<8}  {'n':>5}  {'Market':>8}  {'Single':>8}  {'Segmented':>10}  {'Seg>Sin':>7}")
for b in labels:
    mask = mv_band == b
    if not mask.any():
        continue
    seg_beats_sin = (seg_err[mask] < sin_err[mask]).mean() * 100
    print(f"  {b:<8}  {mask.sum():>5}  "
          f"{np.median(mv_pct[mask]):>7.1f}%  "
          f"{np.median(sin_pct[mask]):>7.1f}%  "
          f"{np.median(seg_pct[mask]):>9.1f}%  "
          f"{seg_beats_sin:>6.1f}%")
