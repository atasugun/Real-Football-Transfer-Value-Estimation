"""
scrape_player.py
----------------
Fetches live market value, goals, and assists for a player from Transfermarkt.

Each result is cached in memory for CACHE_TTL seconds to avoid hammering the site.
"""

import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_TTL = 3600   # seconds (1 hour)
_cache: dict = {}  # (player_id, saison) -> (timestamp, data)

# ── HTTP session ──────────────────────────────────────────────────────────────
_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.transfermarkt.com/",
})

TIMEOUT = 10  # seconds per request


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slug_from_url(tm_url: str) -> str:
    """Extract slug from a transfermarkt URL.
    e.g. 'https://www.transfermarkt.co.uk/harry-kane/profil/spieler/132098'
         -> 'harry-kane'
    """
    m = re.search(r"transfermarkt\.[^/]+/([^/]+)/profil/spieler/", tm_url)
    return m.group(1) if m else "player"


def _parse_market_value(text: str) -> float | None:
    """Parse '€75.00m' / '€750k' / '€1.20bn' into EUR float."""
    text = text.strip().replace("\xa0", "").replace(",", ".")
    m = re.search(r"[\d]+\.?[\d]*", text)
    if not m:
        return None
    v = float(m.group())
    tl = text.lower()
    if "bn" in tl or "b" in tl:
        v *= 1_000_000_000
    elif "m" in tl:
        v *= 1_000_000
    elif "k" in tl:
        v *= 1_000
    return v


def _parse_date(text: str):
    """Try to parse a date string in various formats. Returns a date or None."""
    text = text.strip()
    # Exact match
    for fmt in ("%b %d, %Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    # Extract from mixed text — "Jan 28, 1994 (31)"
    m = re.search(r"(\w{3}\s+\d{1,2},\s+\d{4})", text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%b %d, %Y").date()
        except ValueError:
            pass
    # Extract DD.MM.YYYY
    m = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d.%m.%Y").date()
        except ValueError:
            pass
    # Extract DD/MM/YYYY — Transfermarkt combined label format e.g. "Contract expires:30/06/2026"
    m = re.search(r"(\d{2}/\d{2}/\d{4})", text)
    if m:
        try:
            return datetime.strptime(m.group(1), "%d/%m/%Y").date()
        except ValueError:
            pass
    return None


def _fetch_profile(player_id: int, slug: str) -> dict:
    """Scrape profile page for current market value, club, age, and contract."""
    url = f"https://www.transfermarkt.com/{slug}/profil/spieler/{player_id}"
    resp = _session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    result = {}
    today = datetime.today().date()

    # Market value — usually in a <div class="tm-player-market-value-development__current-value">
    mv_el = soup.select_one(".tm-player-market-value-development__current-value")
    if not mv_el:
        mv_el = soup.find("a", class_=re.compile("market-value", re.I))
    if not mv_el:
        mv_el = soup.find(string=re.compile(r"€\s*\d"))
        if mv_el:
            mv_el = mv_el.parent
    if mv_el:
        raw = mv_el.get_text(" ", strip=True)
        mv = _parse_market_value(raw)
        if mv and mv > 0:
            result["market_value"] = mv

    # Current club and league (competition)
    club_el = soup.select_one(".data-header__club a")
    if club_el:
        result["club"] = club_el.get_text(strip=True)

    # League/competition ID — scoped to .data-header to avoid picking up cup competitions
    data_header = soup.select_one(".data-header")
    comp_link = (data_header or soup).find("a", href=re.compile(r"/startseite/wettbewerb/"))
    if comp_link:
        m = re.search(r"/wettbewerb/([A-Za-z0-9]+)", comp_link["href"])
        if m:
            result["league_from"] = m.group(1)

    # Build a clean label->value lookup from the info-table (primary source).
    # Structure: <span class="info-table__content--regular">Label:</span>
    #            <span class="info-table__content--bold">Value</span>
    info = {}
    for lbl_el in soup.select(".info-table__content--regular"):
        lbl = lbl_el.get_text(strip=True).rstrip(":").lower()
        val_el = lbl_el.find_next_sibling()
        if val_el:
            info[lbl] = val_el.get_text(strip=True)

    # Age — prefer info-table "date of birth/age", fall back to .data-header__label
    dob_raw = info.get("date of birth/age") or info.get("date of birth") or info.get("born")
    if dob_raw:
        dob = _parse_date(dob_raw)
        if dob:
            result["age"] = round((today - dob).days / 365.25, 1)
    if "age" not in result:
        for span in soup.select(".data-header__label"):
            lbl = span.get_text(strip=True).lower()
            if "date of birth" in lbl or "born" in lbl:
                dob = _parse_date(span.get_text(" ", strip=True))
                if dob:
                    result["age"] = round((today - dob).days / 365.25, 1)
                break
    if "age" not in result:
        bd_el = soup.find(attrs={"itemprop": "birthDate"})
        if bd_el:
            dob = _parse_date(bd_el.get("content") or bd_el.get_text(strip=True))
            if dob:
                result["age"] = round((today - dob).days / 365.25, 1)

    # Contract expiry — prefer "contract there expires" (loaned players, parent-club contract)
    # over plain "contract expires" (current club contract).
    contract_raw = info.get("contract there expires") or info.get("contract expires")
    if contract_raw:
        exp = _parse_date(contract_raw)
        if exp:
            result["contract_year_left"] = max(0.0, round((exp - today).days / 365.25, 2))

    # Active / retired — "Current club: Retired" on TM profile means retired
    current_club_val = info.get("current club", "").strip().lower()
    result["is_retired"] = current_club_val == "retired"

    return result


def _fetch_stats(player_id: int, slug: str, saison: int = 2024) -> dict:
    """
    Scrape stats page for a specific season's goals and assists.
    saison=2024 → 2024/25 season (Transfermarkt convention).

    Transfermarkt tfoot column layout (0-indexed):
      0: label ("Total 24/25:")
      1: blank (club)
      2: appearances
      3: goals          ← we want this
      4: assists        ← and this
      5: own goals
      6: yellow cards
      ...
    """
    url = (
        f"https://www.transfermarkt.com/{slug}"
        f"/leistungsdaten/spieler/{player_id}/plus/1/saison/{saison}"
    )
    resp = _session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tfoot = soup.select_one("table.items tfoot")
    if not tfoot:
        return {}

    cells = [td.get_text(strip=True) for td in tfoot.find_all("td")]
    # Need at least 5 cells: label, blank, apps, goals, assists
    if len(cells) < 5:
        return {}

    def _int(s):
        try:
            return int(s.replace("'", "").replace(".", "").strip())
        except (ValueError, AttributeError):
            return None

    goals   = _int(cells[3])
    assists = _int(cells[4])

    result = {}
    if goals   is not None and goals   >= 0: result["goals"]   = goals
    if assists is not None and assists >= 0: result["assists"] = assists
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_player_live(player_id: int, tm_url: str, saison: int = 2024) -> dict:
    """
    Returns live data for a player from Transfermarkt.
    Result dict may contain: market_value, goals, assists, club, age, contract_year_left.
    Empty dict on failure. Results are cached for CACHE_TTL seconds.

    saison=2024 → fetches 2024/25 season stats (previous season for 25/26 transfers).
    """
    player_id = int(player_id)
    cache_key = (player_id, saison)

    # Check cache
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            return data

    data = {}
    slug = _slug_from_url(tm_url or "")
    if not slug or slug == "player":
        slug = f"player-{player_id}"

    try:
        profile = _fetch_profile(player_id, slug)
        data.update(profile)
    except Exception as e:
        logger.warning(f"Profile fetch failed for player {player_id}: {e}")

    try:
        stats = _fetch_stats(player_id, slug, saison=saison)
        data.update(stats)
    except Exception as e:
        logger.warning(f"Stats fetch failed for player {player_id}: {e}")

    _cache[cache_key] = (time.time(), data)
    return data


def clear_cache(player_id: int = None):
    if player_id is None:
        _cache.clear()
    else:
        _cache.pop(int(player_id), None)
