"""
build_live_players.py
---------------------
Merges the Transfermarkt dataset archive into players_live.json with
up-to-date market values, ages, contracts, and leagues for every active player.

Sources
  players.csv          — profile, position, nationality, contract, club
  player_valuations.csv — market value snapshots (most recent = current)

Output
  transfer_data/players_live.json   (loaded by app.py at startup)

Usage
  python build_live_players.py
  python build_live_players.py --archive "C:/path/to/archive"
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Defaults (edit if your archive lives elsewhere) ───────────────────────────
DEFAULT_ARCHIVE = Path(__file__).parent   # CSVs live alongside this script
OUTPUT_PATH     = Path(__file__).parent / "players_live.json"

CURRENT_SEASON  = "25/26"
TODAY           = datetime.now()

VALID_POSITIONS = {"Attack", "Defender", "Goalkeeper", "Midfield"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def calc_age(dob_str) -> float | None:
    try:
        dob = pd.to_datetime(dob_str)
        return round((TODAY - dob).days / 365.25, 1)
    except Exception:
        return None


def calc_contract_years(exp_str) -> float | None:
    try:
        exp = pd.to_datetime(exp_str)
        return round(max(0.0, (exp - TODAY).days / 365.25), 2)
    except Exception:
        return None


def fmt(v: float) -> str:
    """Human-readable EUR amount for logging."""
    if v >= 1e8:  return f"€{v/1e6:.0f}M"
    if v >= 1e6:  return f"€{v/1e6:.1f}M"
    if v >= 1e3:  return f"€{v/1e3:.0f}K"
    return f"€{v:,.0f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def build(archive_dir: Path) -> list[dict]:
    print(f"Archive: {archive_dir}")

    # ── Load files ────────────────────────────────────────────────────────────
    players_path    = archive_dir / "players.csv"
    valuations_path = archive_dir / "player_valuations.csv"

    if not players_path.exists():
        raise FileNotFoundError(f"Not found: {players_path}")
    if not valuations_path.exists():
        raise FileNotFoundError(f"Not found: {valuations_path}")

    print("Loading players.csv …")
    players = pd.read_csv(players_path, low_memory=False, encoding="utf-8")
    print(f"  {len(players):,} rows")

    print("Loading player_valuations.csv …")
    valuations = pd.read_csv(valuations_path, low_memory=False)
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    print(f"  {len(valuations):,} rows  (up to {valuations['date'].max().strftime('%Y-%m-%d')})")

    # ── Most-recent valuation per player ──────────────────────────────────────
    print("Finding latest valuation per player …")
    latest = (
        valuations.dropna(subset=["date"])
        .sort_values("date")
        .groupby("player_id", as_index=False)
        .last()[["player_id", "date", "market_value_in_eur",
                 "player_club_domestic_competition_id"]]
    )
    latest.columns = ["player_id", "val_date", "val_market_value", "val_league"]
    print(f"  {len(latest):,} players have at least one valuation")

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = players.merge(latest, on="player_id", how="left")

    # Prefer the most-recent valuation market value; fall back to players.csv column
    df["market_value_final"] = np.where(
        df["val_market_value"].notna() & (df["val_market_value"] > 0),
        df["val_market_value"],
        df["market_value_in_eur"],
    )
    # Prefer latest-valuation league (most current club); fall back to players.csv
    df["league_final"] = df["val_league"].fillna(df["current_club_domestic_competition_id"])

    # ── Filter to active players with all required data ───────────────────────
    before = len(df)
    # Use the explicit is_retired flag; treat missing/non-True values as active
    if "is_retired" in df.columns:
        retired_mask = df["is_retired"].astype(str).str.strip().str.lower() == "true"
        df = df[~retired_mask]
    df = df[df["position"].isin(VALID_POSITIONS)]
    df = df[df["market_value_final"] > 0]
    df = df.dropna(subset=["country_of_citizenship", "date_of_birth", "league_final", "name"])
    print(f"\nActive players with complete data: {len(df):,}  (filtered from {before:,})")

    # ── Compute age and contract ──────────────────────────────────────────────
    df = df.copy()
    df["age_computed"]      = df["date_of_birth"].apply(calc_age)
    df["contract_computed"] = df["contract_expiration_date"].apply(calc_contract_years)

    df = df.dropna(subset=["age_computed"])
    df = df[df["age_computed"].between(14, 46)]
    print(f"After age filter (14-46): {len(df):,}")

    # ── Position distribution ─────────────────────────────────────────────────
    print("\nPosition breakdown:")
    for pos, cnt in df["position"].value_counts().items():
        print(f"  {pos:<12} {cnt:>5}")

    # ── League coverage ───────────────────────────────────────────────────────
    print("\nTop 15 leagues:")
    for lg, cnt in df["league_final"].value_counts().head(15).items():
        print(f"  {lg:<8} {cnt:>5}")

    # ── Build output list ─────────────────────────────────────────────────────
    print("\nBuilding output …")
    output = []
    for _, row in df.iterrows():
        cyl = row.get("contract_computed")
        output.append({
            "player_id":          int(row["player_id"]),
            "player_name":        str(row["name"]),
            "nationality":        str(row["country_of_citizenship"]),
            "position":           str(row["position"]),
            "sub_position":       str(row.get("sub_position", "")),
            "league_from":        str(row["league_final"]),
            "from_club":          str(row.get("current_club_name", "") or ""),
            "age":                float(row["age_computed"]),
            "market_value_in_eur": float(row["market_value_final"]),
            "contract_year_left": float(cyl) if (cyl is not None and not np.isnan(cyl)) else None,
            "transfer_season":    CURRENT_SEASON,
            "val_date":           str(row["val_date"])[:10] if pd.notna(row.get("val_date")) else None,
            "image_url":          str(row.get("image_url", "") or ""),
        })

    # ── Stats ─────────────────────────────────────────────────────────────────
    mvs = [p["market_value_in_eur"] for p in output]
    print(f"\nMarket value stats:")
    print(f"  Min:    {fmt(min(mvs))}")
    print(f"  Median: {fmt(float(np.median(mvs)))}")
    print(f"  Max:    {fmt(max(mvs))}")

    with_contract = sum(1 for p in output if p["contract_year_left"] is not None)
    print(f"\nContract known: {with_contract:,} / {len(output):,} ({100*with_contract/len(output):.0f}%)")

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE),
                        help="Path to the archive folder containing players.csv etc.")
    args = parser.parse_args()

    archive_dir = Path(args.archive)
    data = build(archive_dir)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    print(f"\nSaved {len(data):,} players  ->  {OUTPUT_PATH}")
    print("Now run:  python app.py")


if __name__ == "__main__":
    main()
