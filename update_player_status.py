"""
update_player_status.py
------------------------
Fetches active/retired status from Transfermarkt for players in players.csv
whose last_season is between 2020-2024 (ambiguous — may have moved to
untracked leagues or retired).

Adds / updates an `is_retired` boolean column in players.csv.

Usage:
    python update_player_status.py              # process ambiguous range
    python update_player_status.py --all        # process every player with a url
    python update_player_status.py --min 2021 --max 2023
"""

import argparse
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

PLAYERS_CSV = Path(__file__).parent / "players.csv"
DELAY       = 1.2   # seconds between requests (be polite)
TIMEOUT     = 12

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com/",
})


def fetch_status(tm_url: str) -> str | None:
    """
    Returns 'active', 'retired', or None (could not determine).
    Checks the info-table 'Current club' value on the TM profile page.
    """
    try:
        resp = _session.get(tm_url, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for lbl_el in soup.select(".info-table__content--regular"):
            lbl = lbl_el.get_text(strip=True).rstrip(":").lower()
            if lbl == "current club":
                val_el = lbl_el.find_next_sibling()
                val = val_el.get_text(strip=True).lower() if val_el else ""
                return "retired" if val == "retired" else "active"

        # Fallback: check for "Retired" text anywhere in page header
        if soup.find(string=re.compile(r"\bRetired\b")):
            return "retired"

        return "active"   # has a club but we didn't find the label
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Process all players with a URL")
    parser.add_argument("--min", type=int, default=2020, help="Min last_season (default 2020)")
    parser.add_argument("--max", type=int, default=2024, help="Max last_season (default 2024)")
    args = parser.parse_args()

    print(f"Loading {PLAYERS_CSV} ...")
    df = pd.read_csv(PLAYERS_CSV, low_memory=False)
    print(f"  {len(df):,} rows, columns: {df.columns.tolist()}")

    # Initialise column if missing
    if "is_retired" not in df.columns:
        df["is_retired"] = pd.NA

    # Select rows to process
    has_url = df["url"].notna() & (df["url"].str.strip() != "")
    if args.all:
        mask = has_url
    else:
        mask = has_url & df["last_season"].between(args.min, args.max)

    # Skip rows already filled in
    mask = mask & df["is_retired"].isna()

    targets = df[mask].copy()
    print(f"  Players to fetch: {len(targets):,}")
    if targets.empty:
        print("Nothing to do.")
        return

    updated = 0
    for i, (idx, row) in enumerate(targets.iterrows(), 1):
        tm_url = str(row["url"]).replace("transfermarkt.co.uk", "transfermarkt.com")
        name   = row.get("name", row.get("player_name", f"id={row['player_id']}"))
        status = fetch_status(tm_url)

        df.at[idx, "is_retired"] = (status == "retired") if status is not None else pd.NA

        flag = "RETIRED" if status == "retired" else ("active" if status else "?")
        print(f"  [{i:>5}/{len(targets)}] {name:<30} {flag}")

        updated += 1

        # Save every 50 rows so progress isn't lost on interruption
        if updated % 50 == 0:
            df.to_csv(PLAYERS_CSV, index=False)
            print(f"    -- checkpoint saved ({updated} processed) --")

        time.sleep(DELAY)

    df.to_csv(PLAYERS_CSV, index=False)
    retired_count = (df["is_retired"] == True).sum()
    active_count  = (df["is_retired"] == False).sum()
    print(f"\nDone. Saved to {PLAYERS_CSV}")
    print(f"  Retired: {retired_count:,}  |  Active: {active_count:,}  |  Unknown: {df['is_retired'].isna().sum():,}")


if __name__ == "__main__":
    main()
