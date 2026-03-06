/* Transfer Scout — frontend logic */

// ── State ────────────────────────────────────────────────────────────────────
let meta          = {};
let currentPlayer = null;
let lastMatrix    = null;
let focusedIdx    = -1;

// summer = March(2) through October(9), winter otherwise
const activeWindow = new Date().getMonth() >= 2 && new Date().getMonth() <= 9 ? "summer" : "winter";

// ── DOM refs ─────────────────────────────────────────────────────────────────
const searchInput    = document.getElementById("searchInput");
const clearBtn       = document.getElementById("clearBtn");
const searchDropdown = document.getElementById("searchDropdown");
const playerSection  = document.getElementById("playerSection");
const matrixSection  = document.getElementById("matrixSection");
const valuateForm    = document.getElementById("valuateForm");
const valuateBtn     = document.getElementById("valuateBtn");
const matrixGrid     = document.getElementById("matrixGrid");
const matrixSubtitle = document.getElementById("matrixSubtitle");
const backBtn        = document.getElementById("backBtn");
const viewSearch     = document.getElementById("viewSearch");
const viewCustom     = document.getElementById("viewCustom");
const customForm     = document.getElementById("customForm");
const customBtn      = document.getElementById("customBtn");

// ── Utility ──────────────────────────────────────────────────────────────────
function isoShort(info) {
  if (!info?.iso) return "";
  return info.iso.split("-").pop().slice(0, 2).toUpperCase();
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function fmtMarketValue(v) {
  if (!v) return "—";
  if (v >= 1e8)  return `€${(v/1e6).toFixed(0)}M`;
  if (v >= 1e7)  return `€${(v/1e6).toFixed(1)}M`;
  if (v >= 1e6)  return `€${(v/1e6).toFixed(2)}M`;
  if (v >= 1e3)  return `€${(v/1e3).toFixed(0)}K`;
  return `€${v.toLocaleString()}`;
}

function initials(name) {
  return name.split(" ").map(w => w[0]).filter(Boolean).slice(0, 2).join("").toUpperCase();
}

let _toastTimer = null;
function showToast(msg, persistent = false) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.remove("toast-info");
  t.classList.add("show");
  if (_toastTimer) { clearTimeout(_toastTimer); _toastTimer = null; }
  if (!persistent) _toastTimer = setTimeout(() => t.classList.remove("show"), 3500);
}
function showInfoToast(msg) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add("show", "toast-info");
  if (_toastTimer) { clearTimeout(_toastTimer); _toastTimer = null; }
}
function hideToast() {
  if (_toastTimer) { clearTimeout(_toastTimer); _toastTimer = null; }
  const t = document.querySelector(".toast");
  if (t) { t.classList.remove("show"); t.classList.remove("toast-info"); }
}

// Mirror of the server-side normalize_name()
const _charMap = {
  'ł':'l','Ł':'l','ø':'o','Ø':'o','ß':'ss','ẞ':'ss',
  'ð':'d','Ð':'d','þ':'th','Þ':'th','æ':'ae','Æ':'ae',
  'œ':'oe','Œ':'oe','đ':'d','Đ':'d','\u2019':'','\u2018':'',"'":"",
  '\u0131':'i','\u0130':'i'  // Turkish ı/İ
};
function normalizeName(s) {
  // 1. Manual char map
  s = s.replace(/[łŁøØßẞðÐþÞæÆœŒđĐ\u2019\u2018'\u0131\u0130]/g, c => _charMap[c] ?? c);
  // 2. NFD + strip combining marks
  s = s.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  // 3. Hyphens → space, remove non-alphanumeric except space
  s = s.replace(/-/g, " ").replace(/[^a-zA-Z0-9 ]/g, "");
  // 4. Lowercase + collapse whitespace
  return s.toLowerCase().replace(/\s+/g, " ").trim();
}

// ── Bootstrap meta ────────────────────────────────────────────────────────────
async function loadMeta() {
  try {
    const res = await fetch("/api/meta");
    meta = await res.json();
    populateDropdowns();
    applyMetaBadge();
  } catch {
    showToast("Failed to load metadata — is the server running?");
  }
}

function applyMetaBadge() {
  const sub = document.querySelector(".search-section .section-sub");
  if (sub && meta.player_count) {
    sub.textContent = `Search across ${meta.player_count.toLocaleString()} current players`;
  }

  const badge = document.querySelector(".header-badge");
  if (badge) {
    if (meta.data_source === "live" && meta.data_freshness) {
      badge.textContent = `Transfer Fee Estimation`;
      badge.style.background = "rgba(0,230,118,0.12)";
      badge.style.color = "#00e676";
      badge.style.borderColor = "rgba(0,230,118,0.3)";
    }
  }
}

function populateDropdowns() {
  // Helper to fill a <select> with options
  function fillNat(sel) {
    meta.nationalities.forEach(n => {
      const o = document.createElement("option"); o.value = o.textContent = n; sel.appendChild(o);
    });
  }
  function fillLeague(sel) {
    meta.leagues.forEach(l => {
      const o = document.createElement("option"); o.value = l;
      const info = meta.league_meta[l];
      o.textContent = info ? `${isoShort(info)} ${info.name}` : l;
      sel.appendChild(o);
    });
  }
  function fillSeason(sel) {
    meta.seasons.forEach(s => {
      const o = document.createElement("option"); o.value = o.textContent = s; sel.appendChild(o);
    });
    if (meta.seasons.length) sel.value = meta.seasons[0];
  }

  // Search form
  fillNat(document.getElementById("f-nationality"));
  fillSeason(document.getElementById("f-season"));

  // Custom form
  fillNat(document.getElementById("c-nationality"));
  fillLeague(document.getElementById("c-league"));
  fillSeason(document.getElementById("c-season"));
}

// ── View switching ─────────────────────────────────────────────────────────────
document.getElementById("viewSwitcher").addEventListener("click", e => {
  const btn = e.target.closest(".view-btn");
  if (!btn) return;
  const view = btn.dataset.view;
  document.querySelectorAll(".view-btn").forEach(b => b.classList.toggle("active", b === btn));
  viewSearch.hidden = view !== "search";
  viewCustom.hidden = view !== "custom";
  document.querySelector(".main").classList.toggle("view-custom", view === "custom");
  if (view === "search") {
    playerSection.hidden = true;
    matrixSection.hidden = true;
  }
  if (view === "custom") {
    playerSection.hidden = true;
    matrixSection.hidden = true;
  }
});

// ── Search ────────────────────────────────────────────────────────────────────
searchInput.addEventListener("input", debounce(onSearchInput, 200));
searchInput.addEventListener("keydown", onSearchKeydown);
clearBtn.addEventListener("click", clearSearch);
document.addEventListener("click", e => {
  if (!e.target.closest(".search-wrapper")) closeDropdown();
});

function onSearchInput() {
  const q = searchInput.value.trim();
  clearBtn.classList.toggle("visible", q.length > 0);
  if (q.length < 2) { closeDropdown(); return; }
  fetchSuggestions(q);
}

async function fetchSuggestions(q) {
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(normalizeName(q))}`);
    renderDropdown(await res.json());
  } catch { closeDropdown(); }
}

function renderDropdown(players) {
  searchDropdown.innerHTML = "";
  focusedIdx = -1;
  if (!players.length) { closeDropdown(); return; }

  players.forEach((p, i) => {
    const li = document.createElement("li");
    li.className = "dropdown-item";
    li.setAttribute("role", "option");
    li.dataset.idx = i;

    li.innerHTML = `
      <div class="dropdown-avatar">${initials(p.name)}</div>
      <div class="dropdown-info">
        <div class="dropdown-name">${p.name}</div>
        <div class="dropdown-meta">${p.nationality} &middot; ${p.position}</div>
      </div>
    `;
    li.addEventListener("mousedown", e => { e.preventDefault(); selectPlayer(p); });
    searchDropdown.appendChild(li);
  });
  searchDropdown.style.display = "block";
}

function onSearchKeydown(e) {
  const items = searchDropdown.querySelectorAll(".dropdown-item");
  if (!items.length) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    focusedIdx = Math.min(focusedIdx + 1, items.length - 1);
    updateFocus(items);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    focusedIdx = Math.max(focusedIdx - 1, 0);
    updateFocus(items);
  } else if (e.key === "Enter") {
    e.preventDefault();
    const target = focusedIdx >= 0 ? items[focusedIdx] : items[0];
    if (target) target.dispatchEvent(new Event("mousedown"));
  } else if (e.key === "Escape") {
    closeDropdown();
  }
}

function updateFocus(items) {
  items.forEach((it, i) => it.classList.toggle("focused", i === focusedIdx));
  if (focusedIdx >= 0) items[focusedIdx].scrollIntoView({ block: "nearest" });
}

function closeDropdown() {
  searchDropdown.innerHTML = "";
  searchDropdown.style.display = "none";
  focusedIdx = -1;
}

function clearSearch() {
  searchInput.value = "";
  clearBtn.classList.remove("visible");
  closeDropdown();
  searchInput.focus();
}

// ── Contract snapping ────────────────────────────────────────────────────────
function snapContract(v) {
  const n = parseFloat(v);
  if (isNaN(n) || n <= 0) return 0;
  if (n < 0.5) return n;            // sub-6-month: keep decimal (0.2, 0.3 …)
  return Math.floor(n * 2) / 2;     // >= 6 months: snap to 0.5 increments
}
["f-contract", "c-contract"].forEach(id => {
  document.getElementById(id).addEventListener("blur", function () {
    if (this.value === "") return;
    const snapped = snapContract(this.value);
    this.value = isNaN(snapped) ? "" : snapped;
  });
});

// ── Custom number spinners ────────────────────────────────────────────────────
function initCustomSpinners() {
  document.querySelectorAll(".form-group input[type=number]").forEach(input => {
    const wrap = document.createElement("div");
    wrap.className = "num-wrap";
    input.parentNode.insertBefore(wrap, input);
    wrap.appendChild(input);

    const spins = document.createElement("div");
    spins.className = "num-spins";

    const upBtn = document.createElement("button");
    upBtn.type = "button";
    upBtn.className = "num-spin";
    upBtn.setAttribute("tabindex", "-1");
    upBtn.setAttribute("aria-label", "Increase");
    upBtn.innerHTML = `<svg viewBox="0 0 10 6" width="10" height="6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="1,5 5,1 9,5"/></svg>`;

    const downBtn = document.createElement("button");
    downBtn.type = "button";
    downBtn.className = "num-spin";
    downBtn.setAttribute("tabindex", "-1");
    downBtn.setAttribute("aria-label", "Decrease");
    downBtn.innerHTML = `<svg viewBox="0 0 10 6" width="10" height="6" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="1,1 5,5 9,1"/></svg>`;

    spins.appendChild(upBtn);
    spins.appendChild(downBtn);
    wrap.appendChild(spins);

    const isContract = input.id === "f-contract" || input.id === "c-contract";
    const step  = parseFloat(input.step)  || 1;
    const min   = input.min !== "" ? parseFloat(input.min) : -Infinity;
    const max   = input.max !== "" ? parseFloat(input.max) :  Infinity;

    function nudge(dir) {
      let v = parseFloat(input.value);
      if (isNaN(v)) v = min !== -Infinity ? min : 0;
      v = Math.round((v + dir * step) * 1e9) / 1e9; // float precision fix
      v = Math.max(min, Math.min(max, v));
      if (isContract) v = snapContract(v);
      input.value = v;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      input.dispatchEvent(new Event("change", { bubbles: true }));
    }

    upBtn.addEventListener("click",   () => nudge(+1));
    downBtn.addEventListener("click", () => nudge(-1));
  });
}

// ── Back button ───────────────────────────────────────────────────────────────
backBtn.addEventListener("click", () => {
  matrixSection.hidden = true;
  playerSection.hidden = true;
  currentPlayer = null;
  clearSearch();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// ── Select player ─────────────────────────────────────────────────────────────
function populateFormFromPlayer(p) {
  setSelect("f-nationality", p.nationality);
  setSelect("f-position", p.position);
  setLeague(p.league_from);
  setSelect("f-season", p.season || meta.seasons?.[0]);
  document.getElementById("f-age").value      = p.age ? Math.round(p.age) : "";
  document.getElementById("f-market").value   = p.market_value || "";
  document.getElementById("f-contract").value = p.contract_year_left != null ? snapContract(p.contract_year_left) : "";
}

async function selectPlayer(p) {
  currentPlayer = p;
  closeDropdown();
  searchInput.value = p.name;
  clearBtn.hidden = false;

  // Clear previous results immediately
  matrixSection.hidden = true;
  playerSection.hidden = true;

  // Populate form with cached data immediately
  populateFormFromPlayer(p);

  // Fetch live data from Transfermarkt, update form, then valuate
  if (p.player_id && p.tm_url) {
    showToast("Calculating, hang tight…", true);
    try {
      const res  = await fetch("/api/fetch_live", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_id: p.player_id, tm_url: p.tm_url }),
      });
      const live = await res.json();
      if (live.success) {
        if (live.market_value && live.market_value > 0) {
          document.getElementById("f-market").value = live.market_value;
          currentPlayer.market_value = live.market_value;
        }
        if (live.age != null) {
          document.getElementById("f-age").value = Math.round(live.age);
          currentPlayer.age = live.age;
        }
        if (live.contract_year_left != null) {
          document.getElementById("f-contract").value = snapContract(live.contract_year_left);
          currentPlayer.contract_year_left = live.contract_year_left;
        }
        if (live.league_from) { setLeague(live.league_from); currentPlayer.league_from = live.league_from; }
        if (live.goals   != null) currentPlayer.prev_goals   = live.goals;
        if (live.assists != null) currentPlayer.prev_assists = live.assists;
      }
    } catch (_) { /* fall through to cached data */ }
  }

  await doValuate();
  hideToast();
}

function setSelect(id, val) {
  const sel = document.getElementById(id);
  if (!sel) return;
  const opt = [...sel.options].find(o => o.value === val || o.textContent.startsWith(val));
  if (opt) sel.value = opt.value;
}

function setLeague(code) {
  document.getElementById("f-league").value = code || "";
  const info = meta.league_meta?.[code];
  const display = document.getElementById("f-league-display");
  if (display) display.textContent = info ? isoShort(info) + " " + info.name : (code || "—");
}

// ── Valuate (shared) ──────────────────────────────────────────────────────────
async function doValuate() {
  const btnLabel = valuateBtn.querySelector(".btn-label");
  valuateBtn.disabled = true;
  valuateBtn.classList.add("loading");
  if (btnLabel) btnLabel.textContent = "Calculating…";

  const payload = {
    nationality:        document.getElementById("f-nationality").value,
    position:           document.getElementById("f-position").value,
    league_from:        document.getElementById("f-league").value,
    age:                document.getElementById("f-age").value,
    market_value:       document.getElementById("f-market").value,
    contract_year_left: document.getElementById("f-contract").value || null,
    season:             document.getElementById("f-season").value,
    // v3: pass player_id so server can look up goals/assists
    player_id:          currentPlayer?.player_id ?? null,
  };

  try {
    const res  = await fetch("/api/valuate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Unknown error");
    lastMatrix = data;
    renderMatrix(data);
    playerSection.hidden = true;
    matrixSection.hidden = false;
    matrixSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    valuateBtn.disabled = false;
    valuateBtn.classList.remove("loading");
    if (btnLabel) btnLabel.textContent = "Estimate Transfer Value";
  }
}

valuateForm.addEventListener("submit", e => { e.preventDefault(); doValuate(); });

// ── Custom player form ────────────────────────────────────────────────────────
customForm.addEventListener("submit", async e => {
  e.preventDefault();

  const btnLabel = customBtn.querySelector(".btn-label");
  customBtn.disabled = true;
  customBtn.classList.add("loading");
  if (btnLabel) btnLabel.textContent = "Calculating…";

  const cyl = document.getElementById("c-contract").value;
  const goalsRaw   = document.getElementById("c-goals").value;
  const assistsRaw = document.getElementById("c-assists").value;
  const payload = {
    nationality:        document.getElementById("c-nationality").value,
    position:           document.getElementById("c-position").value,
    league_from:        document.getElementById("c-league").value,
    age:                document.getElementById("c-age").value,
    market_value:       document.getElementById("c-market").value,
    contract_year_left: cyl || null,
    season:             document.getElementById("c-season").value,
    // v3: goals/assists (only sent when visible and filled in)
    prev_goals:   goalsRaw   !== "" ? goalsRaw   : null,
    prev_assists: assistsRaw !== "" ? assistsRaw : null,
  };

  // Update matrix subtitle with the optional name
  const customName = document.getElementById("c-name").value.trim() || "Custom Player";

  try {
    const res  = await fetch("/api/valuate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Unknown error");
    lastMatrix = data;
    renderMatrix(data, customName);
    matrixSection.hidden = false;
    matrixSection.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    customBtn.disabled = false;
    customBtn.classList.remove("loading");
    if (btnLabel) btnLabel.textContent = "Estimate Transfer Value";
  }
});

// ── Render matrix ─────────────────────────────────────────────────────────────
function renderMatrix(data, nameOverride) {
  matrixGrid.innerHTML = "";

  // Detect which form was used: custom or search
  const isCustom = !!nameOverride;
  const prefix   = isCustom ? "c-" : "f-";

  const nat  = document.getElementById(`${prefix}nationality`).value;
  const pos  = document.getElementById(`${prefix}position`).value;
  const age  = document.getElementById(`${prefix}age`).value;
  const mv   = fmtMarketValue(parseFloat(document.getElementById(`${prefix}market`).value));
  const from = document.getElementById(`${prefix}league`).value;
  const lgInfo = meta.league_meta?.[from];
  const fromLabel = lgInfo ? `${isoShort(lgInfo)} ${lgInfo.name}` : from;
  const nameLabel = nameOverride || currentPlayer?.name || "";

  // Contract years left
  const cylRaw = document.getElementById(`${prefix}contract`).value;
  const cylStr = cylRaw !== "" && cylRaw !== null
    ? `${parseFloat(cylRaw).toFixed(1)}yr contract`
    : "contract unknown";

  // Goals / assists (search: from currentPlayer; custom: from form inputs)
  let gaStr = "";
  const posVal = document.getElementById(`${prefix}position`).value;
  if (posVal === "Attack" || posVal === "Midfield") {
    let g, a;
    if (isCustom) {
      g = document.getElementById("c-goals").value;
      a = document.getElementById("c-assists").value;
    } else {
      g = currentPlayer?.prev_goals;
      a = currentPlayer?.prev_assists;
    }
    const hasG = g !== "" && g !== null && g !== undefined;
    const hasA = a !== "" && a !== null && a !== undefined;
    if (hasG || hasA) {
      gaStr = ` · ${hasG ? Math.round(g) + "G" : "?G"} ${hasA ? Math.round(a) + "A" : "?A"}`;
    }
  }

  matrixSubtitle.textContent =
    `${nameLabel ? nameLabel + " · " : ""}${nat} · ${pos} · Age ${age} · ${mv} · ${cylStr}${gaStr} · From ${fromLabel}`;

  const isSummer = activeWindow === "summer";
  const max = data.max_value, min = data.min_value, range = max - min || 1;
  let topVal = 0;
  data.leagues.forEach(lg => {
    const v = isSummer ? data.data[lg].summer : data.data[lg].winter;
    if (v > topVal) topVal = v;
  });

  data.leagues.forEach(lg => {
    const d    = data.data[lg];
    const val  = isSummer ? d.summer : d.winter;
    const fmt  = isSummer ? d.summer_fmt : d.winter_fmt;
    const pct  = ((val - min) / range * 100).toFixed(1);
    const tier = getTier((val - min) / range);
    const isTop = val === topVal;

    const row = document.createElement("div");
    row.className = `matrix-row${isTop ? " is-top" : ""}`;
    row.innerHTML = `
      <div class="matrix-league">
        <img class="league-flag" src="https://flagcdn.com/w40/${d.iso || 'un'}.png" alt="${d.country}" loading="lazy" />
        <div class="league-names">
          <div class="league-name">${d.name}</div>
          <div class="league-country">${d.country}</div>
        </div>
      </div>
      <div class="matrix-values">
        <div class="val-block">
          <span class="val-amount ${tier.cls}">${fmt}</span>
          <div class="val-bar-wrap"><div class="val-bar ${tier.bar}" style="width:${pct}%"></div></div>
        </div>
      </div>`;
    matrixGrid.appendChild(row);
  });
}

function getTier(r) {
  if (r > 0.75) return { cls: "tier-top",  bar: "bar-top"  };
  if (r > 0.50) return { cls: "tier-high", bar: "bar-high" };
  if (r > 0.25) return { cls: "tier-mid",  bar: "bar-mid"  };
  return              { cls: "tier-low",  bar: "bar-low"  };
}


// ── v3: Show/hide goals & assists based on position ───────────────────────────
function updateStatsVisibility(positionValue) {
  const show = positionValue === "Attack" || positionValue === "Midfield";
  document.querySelectorAll(".stats-field").forEach(el => {
    el.classList.toggle("stats-hidden", !show);
  });
}
const cPosSelect = document.getElementById("c-position");
cPosSelect.addEventListener("change", () => updateStatsVisibility(cPosSelect.value));
// Set initial state
updateStatsVisibility(cPosSelect.value);

// ── Init ──────────────────────────────────────────────────────────────────────
loadMeta();
initCustomSpinners();
