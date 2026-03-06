"""
Transfer Scout - Flask Web Application
Serves the Transfer Value Estimator UI backed by the XGBoost valuation model.
"""

import os
import re
import gc
import sys
import json
import time
import unicodedata
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deployment.player_valuation import PlayerValuator, DISPLAY_LEAGUES, DATA_PATH, MODEL_PATH
from scrape_player import fetch_player_live

app = Flask(__name__, template_folder="templates", static_folder="static")

# Rate limiter — protects /api/fetch_live (Transfermarkt scraper)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],          # no global limit; set per-route
    storage_uri="memory://",
)

# ── League metadata ────────────────────────────────────────────────────────────
LEAGUE_META = {
    "GB1": {"name": "Premier League",  "country": "England",     "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "iso": "gb-eng"},
    "ES1": {"name": "La Liga",          "country": "Spain",       "flag": "🇪🇸",        "iso": "es"},
    "IT1": {"name": "Serie A",          "country": "Italy",       "flag": "🇮🇹",        "iso": "it"},
    "L1":  {"name": "Bundesliga",       "country": "Germany",     "flag": "🇩🇪",        "iso": "de"},
    "FR1": {"name": "Ligue 1",          "country": "France",      "flag": "🇫🇷",        "iso": "fr"},
    "PO1": {"name": "Primeira Liga",    "country": "Portugal",    "flag": "🇵🇹",        "iso": "pt"},
    "NL1": {"name": "Eredivisie",       "country": "Netherlands", "flag": "🇳🇱",        "iso": "nl"},
    "TR1": {"name": "Super Lig",        "country": "Turkey",      "flag": "🇹🇷",        "iso": "tr"},
    "BE1": {"name": "Pro League",       "country": "Belgium",     "flag": "🇧🇪",        "iso": "be"},
    "RU1": {"name": "Premier Liga",     "country": "Russia",      "flag": "🇷🇺",        "iso": "ru"},
    "SC1": {"name": "Scottish Prem.",   "country": "Scotland",    "flag": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",   "iso": "gb-sct"},
    "GR1": {"name": "Super League",     "country": "Greece",      "flag": "🇬🇷",        "iso": "gr"},
    "UKR1":{"name": "Premier League",   "country": "Ukraine",     "flag": "🇺🇦",        "iso": "ua"},
    "CZ1": {"name": "Czech Liga",       "country": "Czechia",     "flag": "🇨🇿",        "iso": "cz"},
    "DK1": {"name": "Superliga",        "country": "Denmark",     "flag": "🇩🇰",        "iso": "dk"},
    "SWE1":{"name": "Allsvenskan",      "country": "Sweden",      "flag": "🇸🇪",        "iso": "se"},
    "NOR1":{"name": "Eliteserien",      "country": "Norway",      "flag": "🇳🇴",        "iso": "no"},
    "SUI1":{"name": "Super League",     "country": "Switzerland", "flag": "🇨🇭",        "iso": "ch"},
    "AUT1":{"name": "Bundesliga",       "country": "Austria",     "flag": "🇦🇹",        "iso": "at"},
    "ARG1":{"name": "Liga Profesional", "country": "Argentina",   "flag": "🇦🇷",        "iso": "ar"},
    "BRA1":{"name": "Serie A",          "country": "Brazil",      "flag": "🇧🇷",        "iso": "br"},
    "MLS": {"name": "MLS",              "country": "USA",         "flag": "🇺🇸",        "iso": "us"},
    "JAP1":{"name": "J1 League",        "country": "Japan",       "flag": "🇯🇵",        "iso": "jp"},
}

N_PER_PLAYER   = len(DISPLAY_LEAGUES) * 2   # 10 leagues × 2 windows = 20
LIVE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "players_live.json")
PLAYERS_CSV    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "players.csv")


# Characters that don't decompose via NFD but have obvious ASCII equivalents
_CHAR_MAP = str.maketrans({
    'ł': 'l', 'Ł': 'l',
    'ø': 'o', 'Ø': 'o',
    'ß': 'ss', 'ẞ': 'ss',
    'ð': 'd', 'Ð': 'd',
    'þ': 'th', 'Þ': 'th',
    'æ': 'ae', 'Æ': 'ae',
    'œ': 'oe', 'Œ': 'oe',
    'đ': 'd', 'Đ': 'd',
    'ŧ': 't', 'Ŧ': 't',
    'ŋ': 'n', 'Ŋ': 'n',
    'ĸ': 'k',
    'ı': 'i', 'İ': 'i',  # Turkish dotless-i / dotted-I
    '\u2019': '',  # right single quote (O'Brien)
    '\u2018': '',  # left single quote
    "'": '',       # apostrophe
    '-': ' ',      # hyphen → space (Mohamed Al-Sallawe → al sallawe)
})

def normalize_name(s: str) -> str:
    """
    Robust accent/character-insensitive normalization for player search.
    Handles: accented chars, ł/ø/ß/ð/æ/etc., apostrophes, hyphens.
    """
    # 1. Apply manual char map first (covers non-decomposable specials)
    s = s.translate(_CHAR_MAP)
    # 2. NFD decompose + strip combining marks (é→e, ñ→n, ü→u, etc.)
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    ).lower()
    # 3. Remove anything that's not a-z, 0-9, or space
    s = re.sub(r"[^a-z0-9 ]", "", s)
    # 4. Collapse whitespace
    return " ".join(s.split())


def format_fee(fee: int) -> str:
    if fee >= 100_000_000:
        return f"€{fee / 1_000_000:.0f}M"
    elif fee >= 10_000_000:
        return f"€{fee / 1_000_000:.1f}M"
    elif fee >= 1_000_000:
        return f"€{fee / 1_000_000:.2f}M"
    elif fee >= 1_000:
        return f"€{fee / 1_000:.0f}K"
    return f"€{fee:,}"


# ── Batch pre-computation ──────────────────────────────────────────────────────

def build_players_index(val: PlayerValuator, df: pd.DataFrame) -> dict:
    """
    Vectorised: build one giant feature matrix covering every
    (player × league_to × window) combination, run a single
    model.predict(), then reshape into a per-player dict.

    Returns  {player_name: {
                "max_value":   int,
                "best_league": str,
                "best_window": str,
                "summer":      {league: int, ...},
                "winter":      {league: int, ...},
              }, ...}
    """
    t0 = time.time()

    # Keep only rows we can fully score
    required = ["player_name", "nationality", "position", "league_from",
                "age", "market_value_in_eur", "transfer_season"]
    valid = df.dropna(subset=required).copy()
    valid = valid[valid["market_value_in_eur"] > 0]
    valid = valid[valid["position"].isin(val.all_positions)]

    n = len(valid)
    print(f"Pre-computing valuations for {n} players × {N_PER_PLAYER} scenarios "
          f"= {n * N_PER_PLAYER:,} predictions...")

    # Build the full matrix in one go
    all_rows  = []
    player_names = []   # length = n * N_PER_PLAYER

    for _, p in valid.iterrows():
        season_year = int(str(p["transfer_season"])[:2]) + 2000
        log_mv      = np.log1p(p["market_value_in_eur"])
        cyl_raw     = p.get("contract_year_left")
        if pd.notna(cyl_raw):
            cyl, cm = float(cyl_raw), 0
        else:
            cyl, cm = val.median_contract, 1

        # v3: look up goals/assists for Attack/Midfield players
        player_id = p.get("player_id")
        prev_goals, prev_assists = val.get_player_stats(player_id, season_year)

        for lg in DISPLAY_LEAGUES:
            for _, is_summer in [("summer", 1), ("winter", 0)]:
                all_rows.append(val._make_row(
                    p["nationality"], p["league_from"], p["position"],
                    p["age"], season_year, cyl, cm,
                    lg, is_summer, log_mv,
                    prev_goals, prev_assists,
                ))
                player_names.append(p["player_name"])

    X        = pd.DataFrame(all_rows)[val.feature_cols]
    log_pred = val.model.predict(X)
    preds    = np.expm1(log_pred).astype(int)

    # Reshape into per-player dict
    windows  = ["summer", "winter"]
    index    = {}

    for i, (name, row_data) in enumerate(zip(player_names, all_rows)):
        lg     = DISPLAY_LEAGUES[(i // 2) % len(DISPLAY_LEAGUES)]
        window = windows[i % 2]
        val_i  = int(preds[i])

        if name not in index:
            index[name] = {"summer": {}, "winter": {}}
        index[name][window][lg] = val_i

    # Compute per-player summary
    for name, data in index.items():
        all_vals = list(data["summer"].values()) + list(data["winter"].values())
        best_val = max(all_vals)
        data["max_value"] = best_val
        for w in ("summer", "winter"):
            for lg, v in data[w].items():
                if v == best_val:
                    data["best_league"] = lg
                    data["best_window"] = w
                    break

    elapsed = time.time() - t0
    print(f"Done -- {len(index)} players valued in {elapsed:.1f}s")
    return index


# ── Transfermarkt URL lookup (player_id -> tm_url) ────────────────────────────
_tm_url_map: dict = {}
_tm_urls_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tm_urls.json")
if os.path.exists(_tm_urls_json):
    with open(_tm_urls_json, "r", encoding="utf-8") as _f:
        _tm_url_map = {int(k): v for k, v in json.load(_f).items()}
    print(f"Loaded {len(_tm_url_map):,} Transfermarkt URLs from tm_urls.json")
elif os.path.exists(PLAYERS_CSV):
    _pcsv = pd.read_csv(PLAYERS_CSV, usecols=["player_id", "url"], low_memory=False)
    _pcsv = _pcsv.dropna(subset=["url"])
    _tm_url_map = dict(zip(_pcsv["player_id"].astype(int), _pcsv["url"].astype(str)))
    del _pcsv
    print(f"Loaded {len(_tm_url_map):,} Transfermarkt URLs from players.csv")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading valuation model...")
valuator = PlayerValuator.load(MODEL_PATH)

# ── Historical data (always needed for SEASONS dropdown + fallback) ───────────
print("Loading transfers.csv...")
_df_hist = pd.read_csv(DATA_PATH)
_df_hist["_season_sort"] = _df_hist["transfer_season"].str[:2].astype(int)
_hist_sorted = _df_hist.sort_values("_season_sort", ascending=False)
POSITIONS = ["Attack", "Defender", "Goalkeeper", "Midfield"]
SEASONS   = sorted(_df_hist["transfer_season"].unique().tolist(), reverse=True)

# ── Live player data (players_live.json) with fallback to transfers.csv ───────
data_source    = "historical"
data_freshness = None

if os.path.exists(LIVE_DATA_PATH):
    print(f"Loading players_live.json...")
    with open(LIVE_DATA_PATH, "r", encoding="utf-8") as f:
        _live = json.load(f)

    players_df = pd.DataFrame(_live)
    data_source = "live"
    val_dates = [p["val_date"] for p in _live if p.get("val_date")]
    data_freshness = max(val_dates) if val_dates else None
    print(f"  {len(players_df):,} active players  (market values up to {data_freshness})")

    ALL_NATIONALITIES = sorted(players_df["nationality"].dropna().unique().tolist())
    ALL_LEAGUES       = sorted(players_df["league_from"].dropna().unique().tolist())
else:
    print("players_live.json not found -- run build_live_players.py first.")
    print("Falling back to transfers.csv...")
    players_df = _hist_sorted.drop_duplicates("player_name").reset_index(drop=True)
    ALL_NATIONALITIES = sorted(_df_hist["nationality"].dropna().unique().tolist())
    ALL_LEAGUES       = sorted(_df_hist["league_from"].dropna().unique().tolist())

# ── Vectorised batch prediction for all players ───────────────────────────────
players_index = build_players_index(valuator, players_df)

# ── Flat list used by /api/search and /api/players ────────────────────────────
_players_list = []
for _, row in players_df.iterrows():
    name = row["player_name"]
    if name not in players_index:
        continue
    idx       = players_index[name]
    best_meta = LEAGUE_META.get(idx.get("best_league", ""), {})

    cyl = row.get("contract_year_left")
    # v3: look up this player's goals/assists for the stats tooltip
    pid = row.get("player_id")
    season_year = int(str(row.get("transfer_season", SEASONS[0]))[:2]) + 2000
    pg, pa = valuator.get_player_stats(pid, season_year)

    _players_list.append({
        "_name_norm":         normalize_name(name),
        "name":               name,
        "player_id":          int(pid) if pd.notna(pid) else None,
        "nationality":        str(row.get("nationality", "")),
        "position":           str(row.get("position", "")),
        "league_from":        str(row.get("league_from", "")),
        "from_club":          str(row.get("from_club", row.get("from_club_name", ""))),
        "age":                float(row["age"]) if pd.notna(row.get("age")) else None,
        "market_value":       float(row["market_value_in_eur"]) if pd.notna(row.get("market_value_in_eur")) else 0,
        "contract_year_left": float(cyl) if pd.notna(cyl) else None,
        "season":             str(row.get("transfer_season", SEASONS[0])),
        "image_url":          str(row.get("image_url", "")),
        "val_date":           str(row.get("val_date", "")) if pd.notna(row.get("val_date")) else "",
        "tm_url":             _tm_url_map.get(int(pid) if pd.notna(pid) else -1, ""),
        # v3: performance stats (None if not available)
        "prev_goals":         pg,
        "prev_assists":       pa,
        # pre-computed predictions
        "max_predicted":      idx["max_value"],
        "max_predicted_fmt":  format_fee(idx["max_value"]),
        "best_league":        idx.get("best_league", ""),
        "best_league_name":   best_meta.get("name", idx.get("best_league", "")),
        "best_league_flag":   best_meta.get("flag", ""),
        "best_window":        idx.get("best_window", ""),
    })

del players_df
gc.collect()
print(f"Ready -- {len(_players_list):,} players indexed ({data_source} data).")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "nationalities":   ALL_NATIONALITIES,
        "leagues":         ALL_LEAGUES,
        "display_leagues": DISPLAY_LEAGUES,
        "league_meta":     LEAGUE_META,
        "positions":       POSITIONS,
        "seasons":         SEASONS,
        "data_source":     data_source,
        "data_freshness":  data_freshness,
        "player_count":    len(_players_list),
    })



@app.route("/api/search")
def search():
    q = normalize_name(request.args.get("q", "").strip())
    if len(q) < 2:
        return jsonify([])

    q_tokens = q.split()

    # Tier 1: full query is a substring of the normalized name
    tier1 = []
    # Tier 2: every query token appears somewhere in the normalized name
    tier2 = []

    for p in _players_list:
        norm = normalize_name(p["name"])
        if q in norm:
            tier1.append(p)
        elif all(qt in norm for qt in q_tokens):
            tier2.append(p)

    return jsonify((tier1 + tier2)[:8])



@app.route("/api/fetch_live", methods=["POST"])
@limiter.limit("15 per minute; 60 per hour")
def fetch_live():
    """
    Fetch live market value, goals, and assists from Transfermarkt for a player.
    Body: { player_id, tm_url }
    Returns: { success, market_value?, goals?, assists?, club?, cached }
    """
    data = request.get_json(force=True)
    try:
        player_id = int(data["player_id"])
        tm_url    = data.get("tm_url") or _tm_url_map.get(player_id, "")
        if not tm_url:
            return jsonify({"success": False, "error": "No Transfermarkt URL for this player"}), 404

        # saison=2024 = 2024/25 season (the previous season for 25/26 transfers)
        live = fetch_player_live(player_id, tm_url, saison=2024)
        if not live:
            return jsonify({"success": False, "error": "Could not retrieve data from Transfermarkt"}), 502

        return jsonify({"success": True, **live})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/valuate", methods=["POST"])
def valuate():
    data = request.get_json(force=True)
    try:
        raw_cyl = data.get("contract_year_left")
        cyl = float(raw_cyl) if raw_cyl not in (None, "", "null") else None

        # v3: resolve goals/assists
        # Priority: explicit prev_goals/prev_assists in request > player_id lookup
        def _parse_stat(val):
            if val in (None, "", "null"):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        prev_goals   = _parse_stat(data.get("prev_goals"))
        prev_assists = _parse_stat(data.get("prev_assists"))

        # If not provided explicitly, try looking up by player_id
        if prev_goals is None and prev_assists is None:
            player_id = data.get("player_id")
            if player_id is not None:
                season_year = int(str(data["season"])[:2]) + 2000
                prev_goals, prev_assists = valuator.get_player_stats(player_id, season_year)

        matrix = valuator.value_matrix(
            nationality=data["nationality"],
            league_from=data["league_from"],
            position=data["position"],
            age=float(data["age"]),
            transfer_season=data["season"],
            market_value_in_eur=float(data["market_value"]),
            contract_year_left=cyl,
            prev_goals=prev_goals,
            prev_assists=prev_assists,
        )

        result     = {}
        all_values = []
        for lg in DISPLAY_LEAGUES:
            s = int(matrix.loc[lg, "summer"])
            w = int(matrix.loc[lg, "winter"])
            all_values.extend([s, w])
            m = LEAGUE_META.get(lg, {"name": lg, "country": "", "flag": ""})
            result[lg] = {**m, "summer": s, "winter": w,
                          "summer_fmt": format_fee(s), "winter_fmt": format_fee(w)}

        return jsonify({
            "success":   True,
            "data":      result,
            "leagues":   DISPLAY_LEAGUES,
            "max_value": max(all_values),
            "min_value": min(all_values),
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


if __name__ == "__main__":
    import platform
    port = int(os.getenv("PORT", 5000))
    if platform.system() == "Windows":
        from waitress import serve
        print(f"Serving on http://localhost:{port}  (waitress)")
        serve(app, host="0.0.0.0", port=port, threads=4)
    else:
        # Linux/Mac — use gunicorn via Procfile in production;
        # this branch is only hit when running directly with `python app.py`
        debug = os.getenv("FLASK_DEBUG", "0") == "1"
        app.run(debug=debug, port=port, use_reloader=False)
