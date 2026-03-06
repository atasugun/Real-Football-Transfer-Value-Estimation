# Transfer Scout

A data-driven football transfer value estimator. Search any active player, fetch their live stats from Transfermarkt, and get a predicted transfer fee across 10 leagues and both transfer windows.

## Live Demo

[transfer-scout.onrender.com](https://transfer-scout.onrender.com) *(update with your actual URL)*

---

## What it does

- Search 8,800+ active football players by name
- Fetches live market value, club, and league from Transfermarkt on selection
- Predicts transfer fee for every combination of destination league × transfer window (summer/winter)
- Model accounts for age, position, nationality, contract status, previous season goals/assists, and league prestige

---

## Model

**Algorithm:** XGBoost + LightGBM ensemble (predictions averaged)
**Target:** log(transfer_fee) — log-transformed to handle right-skewed fee distribution
**Training data:** 8,865 historical transfers from Transfermarkt (2000–2024)

### Features

| Feature | Description |
|---|---|
| log_market_value | Log-transformed Transfermarkt valuation |
| age, age² | Player age with quadratic term |
| contract_year_left | Years remaining on contract |
| contract_urgency | 1 / (contract_year_left + 0.5) |
| contract_missing | Binary flag for missing contract data |
| nationality_enc | Smoothed target encoding of nationality |
| league_from_enc | Smoothed target encoding of current league |
| league_to_enc | Smoothed target encoding of destination league |
| league_step | Prestige difference between leagues |
| season_year | Transfer season start year |
| season_fee_mean | Average log fee for that season |
| is_summer | 1 = summer window, 0 = winter window |
| prev_goals | Goals scored previous season (Attack/Midfield only) |
| prev_assists | Assists previous season (Attack/Midfield only) |
| log_goals_assists | Log(goals + assists) combined stat |
| pos_* | One-hot position dummies |

### Performance

| Metric | Value |
|---|---|
| R² (log scale) | ~0.79 |
| MAE | ~EUR 2.3M |

### Data challenges

- **Missing values:** `contract_year_left` missing for 42% of rows — imputed with median + binary flag. `prev_goals`/`prev_assists` missing for ~40% of attackers/midfielders — passed as NaN (handled natively by XGBoost/LightGBM).
- **Imbalance:** Transfer fees are heavily right-skewed (most under €5M, some over €100M). Solved by predicting log(fee).
- **Outliers:** Tree models saturate above ~€80M market value. A post-processing power-law multiplier is applied for elite players above this threshold.
- **Inconsistencies:** Nationality encoding issues normalized at load time. High-cardinality categoricals (50+ nationalities, 20+ leagues) handled with smoothed target encoding.

---

## Project structure

```
app.py                        Flask web application
player_valuation.py           Model training, inference, PlayerValuator class
scrape_player.py              Transfermarkt live data scraper
static/                       CSS + JS
templates/                    HTML template
data/
  valuation_model.pkl         Trained model (XGBoost + LightGBM)
  players_live.json           Active player index with market values
  tm_urls.json                Player ID -> Transfermarkt URL map
  transfers.csv               Historical transfer data (training)
  player_stats.csv            Season goals/assists by player
  players.csv                 Full player profiles (source for build scripts)
  player_valuations.csv       Market value snapshots (source for build scripts)
scripts/
  build_live_players.py       Builds data/players_live.json from CSV sources
  update_player_status.py     Marks retired players in data/players.csv
```

---

## Running locally

```bash
pip install -r requirements.txt
python player_valuation.py   # retrain model (optional)
python app.py                # starts on http://localhost:5000
```

---

## Deployment

Hosted on Render. The `Procfile` starts the app with gunicorn:

```
web: gunicorn -w 1 --bind 0.0.0.0:$PORT app:app
```

To redeploy with fresh player data:

```bash
python scripts/build_live_players.py   # refresh data/players_live.json
git add data/players_live.json
git commit -m "refresh player data"
git push
```

---

## Data source

Player and transfer data sourced from [Transfermarkt](https://www.transfermarkt.com) via the open Kaggle archive by David Cereijo.
