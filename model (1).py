"""
Transfer Fee Prediction Model
Goal: Beat Transfermarkt's market_value_in_eur as a predictor of actual transfer_fee
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# ── 1. LOAD & CLEAN ────────────────────────────────────────────────────────────
df = pd.read_csv('transfers.csv')

# Parse date features
df['transfer_date'] = pd.to_datetime(df['transfer_date'])
df['year']  = df['transfer_date'].dt.year
df['month'] = df['transfer_date'].dt.month

# Keep only rows where both fee and market value are positive
df = df[(df['transfer_fee'] > 0) & (df['market_value_in_eur'] > 0)].copy()
print(f"Dataset: {len(df):,} transfers after filtering")

# Log-transform target and market value (both are heavily right-skewed)
df['log_fee']   = np.log1p(df['transfer_fee'])
df['log_mv']    = np.log1p(df['market_value_in_eur'])

# Ratio feature: actual fee vs market value (in log space = difference)
df['log_fee_mv_ratio'] = df['log_fee'] - df['log_mv']

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
FEATURES = [
    'log_mv',          # Transfermarkt's estimate (log scale)
    'age',
    'contract_year_left',   # NaN kept — LightGBM handles it natively
    'year',
    'month',
    'position',        # categorical
    'nationality',     # categorical
    'league_from',     # categorical
    'league_to',       # categorical
    'transfer_window', # categorical
]

TARGET = 'log_fee'

# Encode categoricals with LabelEncoder (LightGBM accepts int-encoded cats)
cat_cols = ['position', 'nationality', 'league_from', 'league_to', 'transfer_window']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df[FEATURES]
y = df[TARGET]

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
# Temporal split: train on earlier transfers, test on recent ones
split_date = df['transfer_date'].quantile(0.8)
train_mask = df['transfer_date'] <= split_date
test_mask  = ~train_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train up to: {df.loc[train_mask, 'transfer_date'].max().date()}")
print(f"Test  from:  {df.loc[test_mask,  'transfer_date'].min().date()}")

# ── 4. BASELINE: TRANSFERMARKT ────────────────────────────────────────────────
# Transfermarkt's prediction = market_value_in_eur → log_mv in log space
tm_pred_log = X_test['log_mv'].values
tm_pred_eur = np.expm1(tm_pred_log)
actual_eur  = np.expm1(y_test.values)

def evaluate(name, pred_log, actual_log):
    pred_eur   = np.expm1(pred_log)
    actual_eur = np.expm1(actual_log)
    mae_log  = mean_absolute_error(actual_log, pred_log)
    rmse_log = np.sqrt(mean_squared_error(actual_log, pred_log))
    r2_log   = r2_score(actual_log, pred_log)
    mae_eur  = mean_absolute_error(actual_eur, pred_eur)
    mape     = np.mean(np.abs((actual_eur - pred_eur) / actual_eur)) * 100
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  R²       (log): {r2_log:.4f}")
    print(f"  MAE      (log): {mae_log:.4f}")
    print(f"  RMSE     (log): {rmse_log:.4f}")
    print(f"  MAE      (€):   {mae_eur/1e6:.2f}M")
    print(f"  MAPE     (%):   {mape:.1f}%")
    return {'name': name, 'r2': r2_log, 'mae_log': mae_log,
            'rmse_log': rmse_log, 'mae_eur': mae_eur, 'mape': mape}

results = []
results.append(evaluate("Transfermarkt Baseline", tm_pred_log, y_test.values))

# ── 5. LIGHTGBM MODEL ─────────────────────────────────────────────────────────
lgb_params = {
    'objective':        'regression',
    'metric':           'rmse',
    'num_leaves':       63,
    'learning_rate':    0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'min_child_samples': 20,
    'lambda_l1':        0.1,
    'lambda_l2':        0.1,
    'verbose':          -1,
    'random_state':     42,
}

dtrain = lgb.Dataset(X_train, label=y_train,
                     categorical_feature=cat_cols)
dval   = lgb.Dataset(X_test,  label=y_test,
                     categorical_feature=cat_cols, reference=dtrain)

callbacks = [lgb.early_stopping(50, verbose=False),
             lgb.log_evaluation(period=-1)]

lgb_model = lgb.train(
    lgb_params, dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=callbacks
)

lgb_pred_log = lgb_model.predict(X_test)
results.append(evaluate("LightGBM", lgb_pred_log, y_test.values))

# ── 6. XGBOOST MODEL ──────────────────────────────────────────────────────────
# XGBoost also handles NaN natively
xgb_model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    early_stopping_rounds=50,
    eval_metric='rmse',
    random_state=42,
    verbosity=0,
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_pred_log = xgb_model.predict(X_test)
results.append(evaluate("XGBoost", xgb_pred_log, y_test.values))

# ── 7. ENSEMBLE ───────────────────────────────────────────────────────────────
ens_pred_log = (lgb_pred_log + xgb_pred_log) / 2
results.append(evaluate("Ensemble (LGB + XGB)", ens_pred_log, y_test.values))

# ── 8. SUMMARY TABLE ─────────────────────────────────────────────────────────
print("\n\n" + "═"*55)
print("  SUMMARY")
print("═"*55)
res_df = pd.DataFrame(results)
res_df['mae_eur_M'] = res_df['mae_eur'] / 1e6
print(res_df[['name','r2','mae_log','mape','mae_eur_M']].to_string(index=False))

# Improvement over Transfermarkt
best = res_df.iloc[1:]  # exclude baseline
best_idx = best['r2'].idxmax()
best_model = res_df.loc[best_idx]
baseline   = res_df.iloc[0]
print(f"\n  Best model: {best_model['name']}")
print(f"  R² improvement:   {best_model['r2'] - baseline['r2']:+.4f}")
print(f"  MAPE improvement: {baseline['mape'] - best_model['mape']:+.1f}pp")
print(f"  MAE improvement:  {(baseline['mae_eur'] - best_model['mae_eur'])/1e6:+.2f}M €")

# ── 9. FEATURE IMPORTANCE ────────────────────────────────────────────────────
importance = lgb_model.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n\n  LightGBM Feature Importance (by gain):")
print("  " + "─"*35)
for _, row in feat_imp.iterrows():
    bar = "█" * int(row['importance'] / feat_imp['importance'].max() * 20)
    print(f"  {row['feature']:20s} {bar}")

# ── 10. PLOTS ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

actual_log  = y_test.values
actual_eur_ = np.expm1(actual_log)

# (a) Scatter: Actual vs Predicted — Transfermarkt
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(actual_eur_/1e6, tm_pred_eur/1e6, alpha=0.3, s=10, color='gray')
mx = max(actual_eur_.max(), tm_pred_eur.max()) / 1e6
ax1.plot([0, mx], [0, mx], 'r--', lw=1)
ax1.set_xlabel('Actual Fee (M€)')
ax1.set_ylabel('Predicted Fee (M€)')
ax1.set_title('Transfermarkt Baseline')
ax1.set_xlim(0, mx); ax1.set_ylim(0, mx)

# (b) Scatter: Actual vs Predicted — Best model
best_pred_log = lgb_pred_log if best_model['name'] == 'LightGBM' else \
                xgb_pred_log if best_model['name'] == 'XGBoost' else ens_pred_log
best_pred_eur = np.expm1(best_pred_log)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(actual_eur_/1e6, best_pred_eur/1e6, alpha=0.3, s=10, color='steelblue')
ax2.plot([0, mx], [0, mx], 'r--', lw=1)
ax2.set_xlabel('Actual Fee (M€)')
ax2.set_ylabel('Predicted Fee (M€)')
ax2.set_title(f'{best_model["name"]}')
ax2.set_xlim(0, mx); ax2.set_ylim(0, mx)

# (c) Residuals comparison (log scale)
ax3 = fig.add_subplot(gs[0, 2])
tm_resid   = tm_pred_log - actual_log
best_resid = best_pred_log - actual_log
ax3.hist(tm_resid,   bins=60, alpha=0.5, label='Transfermarkt', color='gray')
ax3.hist(best_resid, bins=60, alpha=0.6, label=best_model['name'], color='steelblue')
ax3.axvline(0, color='red', linestyle='--', lw=1)
ax3.set_xlabel('Residual (log scale)')
ax3.set_title('Residual Distribution')
ax3.legend()

# (d) Feature importance
ax4 = fig.add_subplot(gs[1, 0:2])
feat_imp_plot = feat_imp.copy()
ax4.barh(feat_imp_plot['feature'][::-1], feat_imp_plot['importance'][::-1],
         color='steelblue')
ax4.set_xlabel('Importance (gain)')
ax4.set_title('LightGBM Feature Importance')

# (e) Model comparison bar chart
ax5 = fig.add_subplot(gs[1, 2])
names  = [r['name'] for r in results]
r2vals = [r['r2'] for r in results]
colors = ['gray'] + ['steelblue'] * (len(results) - 1)
bars = ax5.bar(names, r2vals, color=colors)
ax5.set_ylabel('R² (log scale)')
ax5.set_title('Model Comparison')
ax5.set_ylim(min(r2vals) * 0.95, 1.0)
for bar, val in zip(bars, r2vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=15, ha='right')

plt.suptitle('Transfer Fee Prediction — Model Evaluation', fontsize=14, fontweight='bold')
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("\n  Plot saved → results.png")
plt.show()
