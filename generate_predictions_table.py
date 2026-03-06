"""
generate_predictions_table.py
------------------------------
Runs the v3 XGBoost model on every row in transfers.csv and outputs
a colour-coded HTML table: predictions_table.html
"""

import numpy as np
import pandas as pd
from player_valuation import (
    load_and_prepare, build_target_encodings, encode_features,
    get_feature_cols, _apply_mv_multiplier, PlayerValuator,
    DATA_PATH, STATS_PATH
)

# ── Run predictions ────────────────────────────────────────────────────────────
print("Loading data...")
df_raw = pd.read_csv(DATA_PATH)
df, median_contract, season_means, season_trend = load_and_prepare(DATA_PATH, STATS_PATH)
encodings = build_target_encodings(df)
df_enc = encode_features(df, encodings)
feature_cols = get_feature_cols(df_enc)

val = PlayerValuator.load()

print("Running predictions on all rows...")
log_preds = val.model.predict(df_enc[feature_cols])
raw_preds = np.expm1(log_preds)
mv_col    = np.expm1(df_enc["log_market_value"].values)
adj_preds = np.array([_apply_mv_multiplier(p, mv) for p, mv in zip(raw_preds, mv_col)])

# ── Build display dataframe ────────────────────────────────────────────────────
out = df_raw[["player_name", "nationality", "position", "age",
              "league_from", "league_to", "transfer_season",
              "transfer_window", "market_value_in_eur", "transfer_fee"]].copy()

out["predicted_fee"] = adj_preds
out["actual_fee"]    = out["transfer_fee"]

# Error ratio: predicted / actual  (1.0 = perfect)
out["ratio"] = out["predicted_fee"] / out["actual_fee"].replace(0, np.nan)

def fmt_eur(v):
    if pd.isna(v) or v == 0:
        return "—"
    if v >= 1e8:  return f"€{v/1e6:.0f}M"
    if v >= 1e7:  return f"€{v/1e6:.1f}M"
    if v >= 1e6:  return f"€{v/1e6:.2f}M"
    if v >= 1e3:  return f"€{v/1e3:.0f}K"
    return f"€{v:,.0f}"

def ratio_color(r):
    """Green = accurate, yellow = moderate error, red = large error."""
    if pd.isna(r):
        return "#2a2a3a"
    if 0.8 <= r <= 1.25:   return "rgba(0,230,118,0.18)"   # green — within 25%
    if 0.6 <= r <= 1.6:    return "rgba(255,214,0,0.15)"   # yellow — within 60%
    if 0.4 <= r <= 2.5:    return "rgba(255,140,0,0.15)"   # orange — large error
    return "rgba(255,70,70,0.18)"                           # red — very large error

def ratio_badge(r):
    if pd.isna(r): return '<span class="badge badge-na">N/A</span>'
    pct = (r - 1) * 100
    sign = "+" if pct >= 0 else ""
    if abs(pct) <= 25:   cls = "badge-good"
    elif abs(pct) <= 60: cls = "badge-ok"
    elif abs(pct) <= 150: cls = "badge-warn"
    else:                cls = "badge-bad"
    return f'<span class="badge {cls}">{sign}{pct:.0f}%</span>'

# Sort by actual fee descending
out = out.sort_values("actual_fee", ascending=False).reset_index(drop=True)

# ── Build HTML ─────────────────────────────────────────────────────────────────
rows_html = []
for _, row in out.iterrows():
    bg = ratio_color(row["ratio"])
    badge = ratio_badge(row["ratio"])
    rows_html.append(f"""
    <tr style="background:{bg}">
      <td class="name">{row['player_name']}</td>
      <td>{row['nationality']}</td>
      <td><span class="pos pos-{row['position'].lower().replace(' ','-')}">{row['position']}</span></td>
      <td class="num">{int(row['age']) if not pd.isna(row['age']) else '—'}</td>
      <td>{row['league_from']}</td>
      <td>{row['league_to']}</td>
      <td>{row['transfer_season']}</td>
      <td class="cap">{row['transfer_window']}</td>
      <td class="num">{fmt_eur(row['market_value_in_eur'])}</td>
      <td class="num actual">{fmt_eur(row['actual_fee'])}</td>
      <td class="num pred">{fmt_eur(row['predicted_fee'])}</td>
      <td class="num">{badge}</td>
    </tr>""")

rows_str = "\n".join(rows_html)

# Summary stats
within_25  = (out["ratio"].between(0.8, 1.25)).sum()
within_50  = (out["ratio"].between(0.667, 1.5)).sum()
total_valid = out["ratio"].notna().sum()

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Transfer Scout — Predictions Table</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0b1220;
    color: #c8d8c0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    padding: 24px;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 700;
    color: #e8f4e8;
    margin-bottom: 6px;
  }}
  .subtitle {{ color: #6b8f6b; margin-bottom: 20px; font-size: 13px; }}

  .stats-bar {{
    display: flex; gap: 24px; margin-bottom: 20px; flex-wrap: wrap;
  }}
  .stat {{
    background: #111e15;
    border: 1px solid #1e3a22;
    border-radius: 8px;
    padding: 10px 18px;
  }}
  .stat-val {{ font-size: 20px; font-weight: 700; color: #00e676; }}
  .stat-lbl {{ font-size: 11px; color: #6b8f6b; margin-top: 2px; }}

  .legend {{
    display: flex; gap: 14px; margin-bottom: 16px; flex-wrap: wrap; align-items: center;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #8aaa8a; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 3px; }}

  .search-wrap {{ margin-bottom: 14px; }}
  #searchBox {{
    background: #111e15; border: 1px solid #1e3a22; border-radius: 6px;
    color: #c8d8c0; padding: 8px 14px; font-size: 13px; width: 280px;
    outline: none;
  }}
  #searchBox:focus {{ border-color: #00e676; }}

  .table-wrap {{
    overflow-x: auto;
    border-radius: 10px;
    border: 1px solid #1e3a22;
  }}
  table {{
    width: 100%; border-collapse: collapse; min-width: 1000px;
  }}
  thead th {{
    background: #111e15;
    color: #6b8f6b;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid #1e3a22;
    position: sticky; top: 0; z-index: 2;
    cursor: pointer; user-select: none;
    white-space: nowrap;
  }}
  thead th:hover {{ color: #00e676; }}
  thead th.sort-asc::after  {{ content: " ▲"; color: #00e676; }}
  thead th.sort-desc::after {{ content: " ▼"; color: #00e676; }}

  tbody tr {{ border-bottom: 1px solid rgba(30,58,34,0.5); transition: filter 0.1s; }}
  tbody tr:hover {{ filter: brightness(1.25); }}
  td {{ padding: 8px 12px; }}
  td.name {{ font-weight: 600; color: #e8f4e8; white-space: nowrap; }}
  td.num  {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.cap  {{ text-transform: capitalize; }}
  td.actual {{ color: #e8f4e8; font-weight: 600; }}
  td.pred   {{ color: #00e676; font-weight: 600; }}

  .pos {{
    display: inline-block; padding: 2px 7px; border-radius: 4px;
    font-size: 11px; font-weight: 700;
  }}
  .pos-attack    {{ background: rgba(255,80,80,0.2);  color: #ff8080; }}
  .pos-midfield  {{ background: rgba(0,180,255,0.2);  color: #60c8ff; }}
  .pos-defender  {{ background: rgba(0,230,118,0.2);  color: #00e676; }}
  .pos-goalkeeper{{ background: rgba(200,160,0,0.2);  color: #ffd700; }}

  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 700; white-space: nowrap;
  }}
  .badge-good {{ background: rgba(0,230,118,0.2);  color: #00e676; }}
  .badge-ok   {{ background: rgba(255,214,0,0.2);  color: #ffd700; }}
  .badge-warn {{ background: rgba(255,140,0,0.2);  color: #ffa040; }}
  .badge-bad  {{ background: rgba(255,70,70,0.2);  color: #ff6060; }}
  .badge-na   {{ background: rgba(100,100,100,0.2); color: #888; }}
</style>
</head>
<body>

<h1>⚽ Transfer Scout — Full Predictions Table</h1>
<p class="subtitle">v3 XGBoost model predictions vs actual transfer fees across all {len(out):,} transfers · sorted by actual fee</p>

<div class="stats-bar">
  <div class="stat">
    <div class="stat-val">{len(out):,}</div>
    <div class="stat-lbl">Total transfers</div>
  </div>
  <div class="stat">
    <div class="stat-val">{within_25/total_valid*100:.1f}%</div>
    <div class="stat-lbl">Within ±25% of actual</div>
  </div>
  <div class="stat">
    <div class="stat-val">{within_50/total_valid*100:.1f}%</div>
    <div class="stat-lbl">Within ±50% of actual</div>
  </div>
  <div class="stat">
    <div class="stat-val">0.8217</div>
    <div class="stat-lbl">R² score (test set)</div>
  </div>
</div>

<div class="legend">
  <span style="font-size:12px;color:#6b8f6b;font-weight:600;">Error color:</span>
  <div class="legend-item"><div class="legend-dot" style="background:rgba(0,230,118,0.35)"></div> Within ±25%</div>
  <div class="legend-item"><div class="legend-dot" style="background:rgba(255,214,0,0.3)"></div> Within ±60%</div>
  <div class="legend-item"><div class="legend-dot" style="background:rgba(255,140,0,0.3)"></div> Within ±150%</div>
  <div class="legend-item"><div class="legend-dot" style="background:rgba(255,70,70,0.35)"></div> &gt;150% error</div>
</div>

<div class="search-wrap">
  <input id="searchBox" type="text" placeholder="Filter by player name..." oninput="filterTable(this.value)" />
</div>

<div class="table-wrap">
<table id="mainTable">
  <thead>
    <tr>
      <th onclick="sortTable(0)">Player</th>
      <th onclick="sortTable(1)">Nat.</th>
      <th onclick="sortTable(2)">Pos.</th>
      <th onclick="sortTable(3)">Age</th>
      <th onclick="sortTable(4)">From</th>
      <th onclick="sortTable(5)">To</th>
      <th onclick="sortTable(6)">Season</th>
      <th onclick="sortTable(7)">Window</th>
      <th onclick="sortTable(8)" style="text-align:right">Market Value</th>
      <th onclick="sortTable(9)" style="text-align:right">Actual Fee</th>
      <th onclick="sortTable(10)" style="text-align:right">Predicted Fee</th>
      <th onclick="sortTable(11)" style="text-align:right">Error</th>
    </tr>
  </thead>
  <tbody id="tableBody">
{rows_str}
  </tbody>
</table>
</div>

<script>
let sortCol = -1, sortDir = 1;

function filterTable(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('#tableBody tr').forEach(tr => {{
    tr.style.display = tr.cells[0].textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

function sortTable(col) {{
  const tbody = document.getElementById('tableBody');
  const rows  = Array.from(tbody.rows).filter(r => r.style.display !== 'none');
  if (sortCol === col) sortDir *= -1; else {{ sortCol = col; sortDir = 1; }}

  rows.sort((a, b) => {{
    let av = a.cells[col].textContent.trim();
    let bv = b.cells[col].textContent.trim();
    const an = parseFloat(av.replace(/[^0-9.-]/g,''));
    const bn = parseFloat(bv.replace(/[^0-9.-]/g,''));
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * sortDir;
    return av.localeCompare(bv) * sortDir;
  }});

  rows.forEach(r => tbody.appendChild(r));

  document.querySelectorAll('thead th').forEach((th, i) => {{
    th.classList.remove('sort-asc','sort-desc');
    if (i === col) th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
  }});
}}
</script>
</body>
</html>"""

with open("predictions_table.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Saved: predictions_table.html")
print(f"Within +-25%: {within_25/total_valid*100:.1f}%")
print(f"Within +-50%: {within_50/total_valid*100:.1f}%")
