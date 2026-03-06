import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path

# ── CHECK FILES ──────────────────────────
if not Path("player_stats.csv").exists() or not Path("transfers.csv").exists():
    raise FileNotFoundError("player_stats.csv veya transfers.csv bulunamadı!")

# ── LOAD DATA ────────────────────────────
df = pd.read_csv('transfers.csv')
df['transfer_date'] = pd.to_datetime(df['transfer_date'])
df['year']  = df['transfer_date'].dt.year
df['month'] = df['transfer_date'].dt.month
df['saison'] = df['year'] - 1
df = df[(df['transfer_fee'] > 0) & (df['market_value_in_eur'] > 0)].copy()

stats = pd.read_csv('player_stats.csv')
stats = stats.rename(columns={'goals':'prev_goals','assists':'prev_assists'})
df = df.merge(stats[['player_id','saison','prev_goals','prev_assists']],
              on=['player_id','saison'], how='left')

mask_not_atk_mid = ~df['position'].isin(['Attack','Midfield'])
df.loc[mask_not_atk_mid, ['prev_goals','prev_assists']] = np.nan

# ── FEATURE ENGINEERING ─────────────────
df['log_fee'] = np.log1p(df['transfer_fee'])
df['log_mv']  = np.log1p(df['market_value_in_eur'])
df['goals_assists'] = df['prev_goals'].fillna(0) + df['prev_assists'].fillna(0)
has_stats = df['prev_goals'].notna() | df['prev_assists'].notna()
df.loc[~has_stats, 'goals_assists'] = np.nan
df['log_goals_assists'] = np.log1p(df['goals_assists'])

# ── ENCODE CATEGORICALS ─────────────────
cat_cols = ['position','nationality','league_from','league_to','transfer_window']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ── FEATURES & TARGET ───────────────────
FEATURES = [
    'log_mv', 'age', 'contract_year_left', 'year', 'month',
    'position', 'nationality', 'league_from', 'league_to', 'transfer_window',
    'prev_goals','prev_assists','log_goals_assists'
]
TARGET = 'log_fee'

# ── TEMPORAL SPLIT ──────────────────────
split_date = df['transfer_date'].quantile(0.8)
train_mask = df['transfer_date'] <= split_date
test_mask  = ~train_mask

X_train, X_test = df.loc[train_mask, FEATURES], df.loc[test_mask, FEATURES]
y_train, y_test = df.loc[train_mask, TARGET], df.loc[test_mask, TARGET]

# ── EVALUATE FUNCTION ───────────────────
def evaluate(pred_log, actual_log):
    pred_eur = np.expm1(pred_log)
    actual_eur = np.expm1(actual_log)
    r2 = r2_score(actual_log, pred_log)
    mae_eur = mean_absolute_error(actual_eur, pred_eur)
    mape = np.mean(np.abs((actual_eur - pred_eur) / actual_eur)) * 100
    return r2, mae_eur, mape

# ── TRANSFERMARKT BASELINE ─────────────
tm_pred_log = X_test['log_mv'].values
r2_tm, mae_tm, mape_tm = evaluate(tm_pred_log, y_test.values)

# ── V3 ENSEMBLE ────────────────────────
lgb_params = {
    'objective':'regression', 'metric':'rmse', 'num_leaves':63,
    'learning_rate':0.05, 'feature_fraction':0.8, 'bagging_fraction':0.8,
    'bagging_freq':5, 'min_child_samples':20, 'lambda_l1':0.1, 'lambda_l2':0.1,
    'verbose':-1, 'random_state':42
}

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
dval = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_cols, reference=dtrain)

lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=2000,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

xgb_model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    early_stopping_rounds=50, eval_metric='rmse', random_state=42, verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

v3_pred_log = (lgb_model.predict(X_test) + xgb_model.predict(X_test)) / 2
r2_v3, mae_v3, mape_v3 = evaluate(v3_pred_log, y_test.values)

# ── SEPARATE BAR CHARTS ────────────────
metrics = ['R²', 'MAE (€M)', 'MAPE (%)']
tm_values = [r2_tm, mae_tm/1e6, mape_tm]
v3_values = [r2_v3, mae_v3/1e6, mape_v3]
colors = ['gray', 'seagreen']

for i, metric in enumerate(metrics):
    fig, ax = plt.subplots(figsize=(5,4))
    bars = ax.bar(['Transfermarkt','v3 Ensemble'], [tm_values[i], v3_values[i]], color=colors)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0,3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()