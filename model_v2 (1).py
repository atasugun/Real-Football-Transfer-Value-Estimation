"""
Transfer Fee Prediction Model v2
- Enhanced feature engineering
- Optuna hyperparameter tuning for LightGBM and XGBoost
- Weighted ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

# ── 1. LOAD & CLEAN ────────────────────────────────────────────────────────────
df = pd.read_csv('transfers.csv')

df['transfer_date'] = pd.to_datetime(df['transfer_date'])
df['year']  = df['transfer_date'].dt.year
df['month'] = df['transfer_date'].dt.month

df = df[(df['transfer_fee'] > 0) & (df['market_value_in_eur'] > 0)].copy()
print(f"Dataset: {len(df):,} transfers")

df['log_fee'] = np.log1p(df['transfer_fee'])
df['log_mv']  = np.log1p(df['market_value_in_eur'])

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
# League prestige tiers (UEFA coefficients / market size)
TOP5  = {'GB1', 'ES1', 'L1', 'IT1', 'FR1'}
MAJOR = {'NL1', 'PO1', 'BE1', 'RU1', 'TR1', 'GR1', 'SC1'}

def league_tier(league):
    if league in TOP5:  return 1
    if league in MAJOR: return 2
    return 3

df['tier_from']  = df['league_from'].map(league_tier)
df['tier_to']    = df['league_to'].map(league_tier)
df['tier_delta'] = df['tier_from'] - df['tier_to']   # >0 = moving to more prestigious league

# Age interactions
df['age_sq']         = df['age'] ** 2
df['log_mv_per_age'] = df['log_mv'] / df['age'].clip(lower=16)

# Contract: missing value is itself informative
df['contract_missing']   = df['contract_year_left'].isna().astype(int)
df['contract_year_left'] = df['contract_year_left'].fillna(-1)   # sentinel for trees

# League pair (e.g. PO1→GB1 has a systematic premium)
df['league_pair'] = df['league_from'].astype(str) + '_' + df['league_to'].astype(str)

# Categorical encoding
cat_cols = ['position', 'nationality', 'league_from', 'league_to',
            'transfer_window', 'league_pair']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── 3. TRAIN / TEST SPLIT (temporal) ──────────────────────────────────────────
split_date = df['transfer_date'].quantile(0.8)
train_mask = df['transfer_date'] <= split_date
test_mask  = ~train_mask

print(f"Train: {train_mask.sum():,} | Test: {test_mask.sum():,}")
print(f"Train up to: {df.loc[train_mask, 'transfer_date'].max().date()}")
print(f"Test  from:  {df.loc[test_mask,  'transfer_date'].min().date()}")

# ── 4. GROUP STATISTICS (computed on train only → no leakage) ─────────────────
# Median fee/MV ratio per destination league — captures league premium
mv_ratio_train = (np.expm1(df.loc[train_mask, 'log_fee'])
                  / np.expm1(df.loc[train_mask, 'log_mv']))
league_to_raw_train = encoders['league_to'].inverse_transform(df.loc[train_mask, 'league_to'])
league_premium_map  = (pd.Series(mv_ratio_train.values, index=league_to_raw_train)
                       .groupby(level=0).median())
global_league_median = league_premium_map.median()

pos_raw_train   = encoders['position'].inverse_transform(df.loc[train_mask, 'position'])
pos_ratio_train = mv_ratio_train
pos_premium_map = (pd.Series(pos_ratio_train.values, index=pos_raw_train)
                   .groupby(level=0).median())
global_pos_median = pos_premium_map.median()

def apply_league_premium(enc_val):
    raw = encoders['league_to'].inverse_transform([enc_val])[0]
    return league_premium_map.get(raw, global_league_median)

def apply_pos_premium(enc_val):
    raw = encoders['position'].inverse_transform([enc_val])[0]
    return pos_premium_map.get(raw, global_pos_median)

df['league_to_premium'] = df['league_to'].apply(apply_league_premium)
df['position_premium']  = df['position'].apply(apply_pos_premium)

# ── 5. FINAL FEATURE SET ──────────────────────────────────────────────────────
FEATURES = [
    # Core
    'log_mv',
    'age',
    'age_sq',
    'log_mv_per_age',
    # Contract
    'contract_year_left',
    'contract_missing',
    # Temporal
    'year',
    'month',
    # League / position
    'tier_from',
    'tier_to',
    'tier_delta',
    'league_to_premium',
    'position_premium',
    # Categoricals
    'position',
    'nationality',
    'league_from',
    'league_to',
    'league_pair',
    'transfer_window',
]

TARGET = 'log_fee'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# ── 6. EVALUATE HELPER ────────────────────────────────────────────────────────
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

# ── 7. TRANSFERMARKT BASELINE ─────────────────────────────────────────────────
tm_pred_log = X_test['log_mv'].values
tm_pred_eur = np.expm1(tm_pred_log)
results.append(evaluate("Transfermarkt Baseline", tm_pred_log, y_test.values))

# ── 8. OPTUNA: TUNE LIGHTGBM ──────────────────────────────────────────────────
lgb_cat_cols = ['position', 'nationality', 'league_from', 'league_to',
                'league_pair', 'transfer_window']

def lgb_cv_rmse(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        dtr  = lgb.Dataset(Xtr, label=ytr, categorical_feature=lgb_cat_cols)
        dval = lgb.Dataset(Xval, label=yval, categorical_feature=lgb_cat_cols,
                           reference=dtr)
        m = lgb.train(
            params, dtr,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(period=-1)]
        )
        scores.append(mean_squared_error(yval, m.predict(Xval)) ** 0.5)
    return np.mean(scores)

def lgb_objective(trial):
    params = {
        'objective':         'regression',
        'metric':            'rmse',
        'verbose':           -1,
        'random_state':      42,
        'num_leaves':        trial.suggest_int('num_leaves', 31, 255),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'feature_fraction':  trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction':  trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq':      trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'lambda_l1':         trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2':         trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'max_depth':         trial.suggest_int('max_depth', 4, 12),
    }
    return lgb_cv_rmse(params)

print("\n  Optuna: tuning LightGBM (50 trials)...")
lgb_study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)

best_lgb_params = {
    'objective': 'regression', 'metric': 'rmse',
    'verbose': -1, 'random_state': 42,
    **lgb_study.best_params
}
print(f"  Best CV RMSE: {lgb_study.best_value:.4f}")
print(f"  Best params:  {lgb_study.best_params}")

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=lgb_cat_cols)
dval   = lgb.Dataset(X_test,  label=y_test,  categorical_feature=lgb_cat_cols,
                     reference=dtrain)
lgb_model = lgb.train(
    best_lgb_params, dtrain,
    num_boost_round=3000,
    valid_sets=[dval],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(period=-1)]
)
lgb_pred_log = lgb_model.predict(X_test)
results.append(evaluate("LightGBM (Optuna)", lgb_pred_log, y_test.values))

# ── 9. OPTUNA: TUNE XGBOOST ───────────────────────────────────────────────────
def xgb_cv_rmse(params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = xgb.XGBRegressor(
            **params,
            n_estimators=2000,
            early_stopping_rounds=40,
            eval_metric='rmse',
            verbosity=0,
        )
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        scores.append(mean_squared_error(yval, m.predict(Xval)) ** 0.5)
    return np.mean(scores)

def xgb_objective(trial):
    params = {
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'random_state':     42,
    }
    return xgb_cv_rmse(params)

print("\n  Optuna: tuning XGBoost (50 trials)...")
xgb_study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

best_xgb_params = {'random_state': 42, **xgb_study.best_params}
print(f"  Best CV RMSE: {xgb_study.best_value:.4f}")

xgb_model = xgb.XGBRegressor(
    **best_xgb_params,
    n_estimators=3000,
    early_stopping_rounds=50,
    eval_metric='rmse',
    verbosity=0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_pred_log = xgb_model.predict(X_test)
results.append(evaluate("XGBoost (Optuna)", xgb_pred_log, y_test.values))

# ── 10. WEIGHTED ENSEMBLE ─────────────────────────────────────────────────────
# Weight inversely proportional to CV RMSE
w_lgb   = 1 / lgb_study.best_value
w_xgb   = 1 / xgb_study.best_value
ens_pred_log = (w_lgb * lgb_pred_log + w_xgb * xgb_pred_log) / (w_lgb + w_xgb)
results.append(evaluate("Weighted Ensemble", ens_pred_log, y_test.values))

# ── 11. SUMMARY ───────────────────────────────────────────────────────────────
print("\n\n" + "═"*58)
print("  SUMMARY")
print("═"*58)
res_df = pd.DataFrame(results)
res_df['mae_eur_M'] = res_df['mae_eur'] / 1e6
print(res_df[['name','r2','mae_log','mape','mae_eur_M']].to_string(index=False))

baseline   = res_df.iloc[0]
best_idx   = res_df.iloc[1:]['r2'].idxmax()
best_model = res_df.loc[best_idx]
print(f"\n  Best model: {best_model['name']}")
print(f"  R² improvement:   {best_model['r2'] - baseline['r2']:+.4f}")
print(f"  MAPE improvement: {baseline['mape'] - best_model['mape']:+.1f}pp")
print(f"  MAE improvement:  {(baseline['mae_eur'] - best_model['mae_eur'])/1e6:+.2f}M €")

# ── 12. FEATURE IMPORTANCE ────────────────────────────────────────────────────
importance = lgb_model.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n\n  LightGBM Feature Importance (by gain):")
print("  " + "─"*40)
for _, row in feat_imp.iterrows():
    bar = "█" * int(row['importance'] / feat_imp['importance'].max() * 25)
    print(f"  {row['feature']:22s} {bar}")

# ── 13. PLOTS ─────────────────────────────────────────────────────────────────
actual_log  = y_test.values
actual_eur_ = np.expm1(actual_log)

best_pred_log = (
    lgb_pred_log if best_model['name'] == 'LightGBM (Optuna)' else
    xgb_pred_log if best_model['name'] == 'XGBoost (Optuna)'  else
    ens_pred_log
)
best_pred_eur = np.expm1(best_pred_log)
mx = max(actual_eur_.max(), tm_pred_eur.max()) / 1e6

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(actual_eur_/1e6, tm_pred_eur/1e6, alpha=0.3, s=10, color='gray')
ax1.plot([0, mx], [0, mx], 'r--', lw=1)
ax1.set_xlabel('Actual Fee (M€)'); ax1.set_ylabel('Predicted Fee (M€)')
ax1.set_title(f'Transfermarkt Baseline\nR²={baseline["r2"]:.3f}  MAPE={baseline["mape"]:.1f}%')
ax1.set_xlim(0, mx); ax1.set_ylim(0, mx)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(actual_eur_/1e6, best_pred_eur/1e6, alpha=0.3, s=10, color='steelblue')
ax2.plot([0, mx], [0, mx], 'r--', lw=1)
ax2.set_xlabel('Actual Fee (M€)'); ax2.set_ylabel('Predicted Fee (M€)')
ax2.set_title(f'{best_model["name"]}\nR²={best_model["r2"]:.3f}  MAPE={best_model["mape"]:.1f}%')
ax2.set_xlim(0, mx); ax2.set_ylim(0, mx)

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(tm_pred_log - actual_log,   bins=60, alpha=0.5, label='Transfermarkt', color='gray')
ax3.hist(best_pred_log - actual_log, bins=60, alpha=0.6, label=best_model['name'], color='steelblue')
ax3.axvline(0, color='red', linestyle='--', lw=1)
ax3.set_xlabel('Residual (log scale)')
ax3.set_title('Residual Distribution')
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[1, 0:2])
ax4.barh(feat_imp['feature'][::-1], feat_imp['importance'][::-1], color='steelblue')
ax4.set_xlabel('Importance (gain)')
ax4.set_title('LightGBM Feature Importance')

ax5 = fig.add_subplot(gs[1, 2])
names  = [r['name'] for r in results]
r2vals = [r['r2'] for r in results]
colors = ['#aaaaaa', '#4c8cbf', '#2e6da4', '#1a4d7a']
bars = ax5.bar(names, r2vals, color=colors)
ax5.set_ylabel('R² (log scale)')
ax5.set_title('Model Comparison')
ax5.set_ylim(min(r2vals) * 0.95, 1.0)
for bar, val in zip(bars, r2vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.xticks(rotation=20, ha='right', fontsize=8)

plt.suptitle('Transfer Fee Prediction v2 — Optuna + Enhanced Features',
             fontsize=14, fontweight='bold')
plt.savefig('results_v2.png', dpi=150, bbox_inches='tight')
print("\n  Plot saved → results_v2.png")
plt.show()
