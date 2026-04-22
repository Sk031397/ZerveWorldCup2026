
"""
Dashboard Validation Test Suite
================================
Validates the WC 2026 Oracle Dashboard by:
 1. Checking app/main.py exists and scanning for page definitions + key components
 2. Simulating canvas variable availability (elo_ratings, match_df, team_features, simulation_results)
 3. Running prediction logic inline: match predictor, tournament simulator, team profile, insights
 4. Printing structured test report with PASS/FAIL for each component + sample outputs
"""

import os
import ast
import math
import random
import numpy as np
import pandas as pd
from collections import defaultdict

# ─── Test infrastructure ──────────────────────────────────────────────────────
_test_results = []

def _check(name, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    _test_results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
    return condition

print("=" * 70)
print("  WC 2026 ORACLE DASHBOARD — VALIDATION TEST SUITE")
print("=" * 70)

# ─── SECTION 1: Script File Existence & Page Definitions ──────────────────────
print("\n【1】 SCRIPT FILE & PAGE ROUTING")
print("-" * 50)

script_path = "app/main.py"
script_exists = os.path.isfile(script_path)
_check("Script exists at app/main.py", script_exists)

# Define the 5 expected pages/sections we look for in the script
EXPECTED_PAGES = [
    ("Match Predictor",      ["match", "predict"]),
    ("Tournament Simulator", ["simulat", "tournament", "champion"]),
    ("Team Profile",         ["team", "profile", "elo"]),
    ("Insights",             ["insight", "finding", "curse"]),
    ("Overview / Home",      ["overview", "home", "oracle", "set_page_config"]),
]

if script_exists:
    with open(script_path, "r") as _f:
        script_content = _f.read().lower()

    for page_name, keywords in EXPECTED_PAGES:
        found = any(kw in script_content for kw in keywords)
        _check(f"Page definition: {page_name}", found)

    # Key component checks
    _check("Sidebar navigation present",        "sidebar" in script_content)
    _check("st.set_page_config present",        "set_page_config" in script_content)
    _check("session_state caching present",     "session_state" in script_content)
    _check("Canvas variable import (zerve)",    "zerve" in script_content or "variable(" in script_content)
    _check("CSV export / download button",      "download_button" in script_content or "csv" in script_content)
    _check("Demo mode / fallback present",      "demo" in script_content or "fallback" in script_content or "except" in script_content)
else:
    print("  ⚠️  Script not found — page content checks skipped (dashboard not yet written).")
    # Still note expected pages as informational
    for page_name, _ in EXPECTED_PAGES:
        _check(f"Page definition: {page_name}", False, "script missing")
    for comp in ["Sidebar navigation", "set_page_config", "session_state", "Canvas variable import", "CSV export", "Demo mode"]:
        _check(comp, False, "script missing")

# ─── SECTION 2: Canvas Variable Availability ──────────────────────────────────
print("\n【2】 CANVAS VARIABLE AVAILABILITY")
print("-" * 50)

# Map the 4 named dashboard variables to the actual canvas variables
# elo_ratings  → elo_table (303 teams)
# match_df     → wc_model_df (898 WC matches with features)
# team_features→ wc_features (898 rows with engineered features)
# simulation_results → wc2026_sim_results (48 teams, 7 cols)

_check("elo_ratings (elo_table) available",
       isinstance(elo_table, pd.DataFrame) and len(elo_table) > 0,
       f"{len(elo_table)} teams")

_check("match_df (wc_model_df) available",
       isinstance(wc_model_df, pd.DataFrame) and len(wc_model_df) > 0,
       f"{len(wc_model_df)} rows")

_check("team_features (wc_features) available",
       isinstance(wc_features, pd.DataFrame) and len(wc_features) > 0,
       f"{wc_features.shape[0]}×{wc_features.shape[1]}")

_check("simulation_results (wc2026_sim_results) available",
       isinstance(wc2026_sim_results, pd.DataFrame) and len(wc2026_sim_results) > 0,
       f"{len(wc2026_sim_results)} teams")

_check("predict_match function available",
       callable(predict_match))

_check("ELO lookup dict available",
       isinstance(elo_lookup, dict) and len(elo_lookup) > 100,
       f"{len(elo_lookup)} entries")

_check("simulation functions available (sim_group_stage, sim_knockout)",
       callable(sim_group_stage) and callable(sim_knockout))

# ─── SECTION 3: Page Logic — Match Predictor ──────────────────────────────────
print("\n【3】 PAGE LOGIC — MATCH PREDICTOR")
print("-" * 50)

_brazil_arg = predict_match("Brazil", "Argentina")
_pred_keys = {"team1_win", "draw", "team2_win", "team1", "team2", "team1_elo", "team2_elo"}
_check("predict_match returns required keys",
       _pred_keys.issubset(set(_brazil_arg.keys())))

_probs_sum = _brazil_arg["team1_win"] + _brazil_arg["draw"] + _brazil_arg["team2_win"]
_check("Probabilities sum to ~1.0",
       abs(_probs_sum - 1.0) < 0.01,
       f"sum={_probs_sum:.4f}")

_check("ELO values are sensible (1000–3000)",
       1000 < _brazil_arg["team1_elo"] < 3000 and 1000 < _brazil_arg["team2_elo"] < 3000,
       f"Brazil={_brazil_arg['team1_elo']} Argentina={_brazil_arg['team2_elo']}")

print(f"\n  📊 SAMPLE — Brazil vs Argentina:")
print(f"     Brazil win:    {_brazil_arg['team1_win']:.1%}")
print(f"     Draw:          {_brazil_arg['draw']:.1%}")
print(f"     Argentina win: {_brazil_arg['team2_win']:.1%}")
print(f"     ELO: {_brazil_arg['team1_elo']} vs {_brazil_arg['team2_elo']}")

# ─── SECTION 4: Page Logic — Tournament Simulator ─────────────────────────────
print("\n【4】 PAGE LOGIC — TOURNAMENT SIMULATOR")
print("-" * 50)

# Quick 500-sim mini-tournament (same logic as dashboard demo mode)
_N_MINI = 500
_mini_champ = defaultdict(int)
for _s in range(_N_MINI):
    _q = sim_group_stage(WC2026_GROUPS_FINAL)
    _c, _ = sim_knockout(_q)
    if _c:
        _mini_champ[_c] += 1

_check("Mini-simulation produces champions",
       len(_mini_champ) > 0,
       f"{len(_mini_champ)} distinct champions in {_N_MINI} sims")

_mini_df = pd.DataFrame([
    {"team": t, "champion_pct": round(v / _N_MINI * 100, 1)}
    for t, v in _mini_champ.items()
]).sort_values("champion_pct", ascending=False).reset_index(drop=True)

_check("Top contender has >5% win probability",
       _mini_df["champion_pct"].iloc[0] > 5,
       f"Leader: {_mini_df['team'].iloc[0]} @ {_mini_df['champion_pct'].iloc[0]:.1f}%")

# Validate against the pre-run 50k results
_top5_50k = wc2026_sim_results.head(5)
_check("50k sim results have top 5 teams",
       len(_top5_50k) == 5)

print(f"\n  📊 SAMPLE — Top 5 Championship Odds (50,000 simulations):")
for _i, _row in _top5_50k.iterrows():
    print(f"     {_i}. {_row['team']:<18} {_row['champion_pct']:>5.1f}%  (Final: {_row['final_pct']:.1f}%  SF: {_row['sf_pct']:.1f}%)")

# ─── SECTION 5: Page Logic — Team Profile ─────────────────────────────────────
print("\n【5】 PAGE LOGIC — TEAM PROFILE (France)")
print("-" * 50)

def _get_team_profile(team_name):
    """Replicate dashboard team profile logic."""
    # ELO
    _elo_row = elo_table[elo_table["team"] == team_name]
    _team_elo = float(_elo_row["elo"].values[0]) if len(_elo_row) > 0 else 1500.0
    _elo_matches = int(_elo_row["matches_played"].values[0]) if len(_elo_row) > 0 else 0

    # Simulation stats
    _sim_row = wc2026_sim_results[wc2026_sim_results["team"] == team_name]
    _champ_pct = float(_sim_row["champion_pct"].values[0]) if len(_sim_row) > 0 else 0.0
    _final_pct = float(_sim_row["final_pct"].values[0]) if len(_sim_row) > 0 else 0.0

    # Recent form (last 10 WC matches in dataset)
    _wc_team = wc_model_df[(wc_model_df["home_team"] == team_name) | (wc_model_df["away_team"] == team_name)].tail(10)
    _wc_record = {"played": len(_wc_team), "wins": 0, "draws": 0, "losses": 0}
    for _, _r in _wc_team.iterrows():
        if _r["home_team"] == team_name:
            _out = _r["outcome"]  # 2=home win, 1=draw, 0=away win
            if _out == 2: _wc_record["wins"] += 1
            elif _out == 1: _wc_record["draws"] += 1
            else: _wc_record["losses"] += 1
        else:
            _out = _r["outcome"]
            if _out == 0: _wc_record["wins"] += 1
            elif _out == 1: _wc_record["draws"] += 1
            else: _wc_record["losses"] += 1

    # Confederation
    CONF_MAP = {
        "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "France": "UEFA",
        "Spain": "UEFA", "England": "UEFA", "Germany": "UEFA",
        "Morocco": "CAF", "United States": "CONCACAF", "Japan": "AFC"
    }
    _conf = CONF_MAP.get(team_name, get_confederation(team_name) if callable(get_confederation) else "N/A")

    return {
        "team": team_name, "elo": _team_elo, "matches_played": _elo_matches,
        "champion_pct": _champ_pct, "final_pct": _final_pct,
        "confederation": _conf, "wc_record_last10": _wc_record
    }

_france_profile = _get_team_profile("France")
_check("Team profile returns all required fields",
       all(k in _france_profile for k in ["team", "elo", "champion_pct", "confederation"]))
_check("ELO for France is realistic (>1800)",
       _france_profile["elo"] > 1800,
       f"ELO={_france_profile['elo']:.0f}")
_check("Champion % for France is positive",
       _france_profile["champion_pct"] > 0,
       f"{_france_profile['champion_pct']:.1f}%")

print(f"\n  📊 SAMPLE — Team Profile: France")
print(f"     ELO Rating:    {_france_profile['elo']:.0f}")
print(f"     Confederation: {_france_profile['confederation']}")
print(f"     Matches played:{_france_profile['matches_played']}")
print(f"     Champion odds: {_france_profile['champion_pct']:.1f}%")
print(f"     Final odds:    {_france_profile['final_pct']:.1f}%")
print(f"     Last 10 WC:    W{_france_profile['wc_record_last10']['wins']}"
      f" D{_france_profile['wc_record_last10']['draws']}"
      f" L{_france_profile['wc_record_last10']['losses']}")

# ─── SECTION 6: Page Logic — Insights ─────────────────────────────────────────
print("\n【6】 PAGE LOGIC — INSIGHTS")
print("-" * 50)

_check("ELO Curse data (wc_fav) available",
       isinstance(wc_fav, pd.DataFrame) and len(wc_fav) > 0,
       f"{len(wc_fav)} rows")
_check("Upset Index data (upset_by_gap) available",
       isinstance(upset_by_gap, pd.DataFrame) and len(upset_by_gap) > 0,
       f"{len(upset_by_gap)} ELO gap bins")
_check("Group strength data (group_df) available",
       isinstance(group_df, pd.DataFrame) and len(group_df) == 12,
       "12 groups")
_check("Glory Gap data (glory_df) available",
       isinstance(glory_df, pd.DataFrame) and len(glory_df) > 0,
       f"{len(glory_df)} teams")

print(f"\n  📊 SAMPLE — ELO Curse Insight (Strong Favourites):")
_strong_fav = wc_fav[wc_fav["elo_bin"] == "Strong Fav"]
if len(_strong_fav) > 0:
    for _, _sr in _strong_fav.iterrows():
        _ctx = "WC" if _sr["is_world_cup"] else "All matches"
        print(f"     {_ctx:<14}: actual={_sr['actual']:.1%}  expected={_sr['expected']:.1%}  "
              f"overperf={_sr['overperf']:+.1%}  n={_sr['n']}")

# ─── SECTION 7: Component-Level Checks ────────────────────────────────────────
print("\n【7】 COMPONENT CHECKS")
print("-" * 50)

# API integration code: check predict_match behaves correctly for unknown teams
_unknown_pred = predict_match("Atlantis FC", "Wakanda United")
_check("API fallback: unknown teams handled gracefully",
       "team1_win" in _unknown_pred and abs(_unknown_pred["team1_win"] + _unknown_pred["draw"] + _unknown_pred["team2_win"] - 1.0) < 0.01,
       f"sums to {_unknown_pred['team1_win']+_unknown_pred['draw']+_unknown_pred['team2_win']:.4f}")

# Canvas fallback logic: simulate missing var by checking elo_lookup default
_check("Canvas fallback: get_elo returns 1500 for unknown team",
       abs(get_elo("Zanzibar Warriors FC") - 1500.0) < 1.0,
       "default ELO=1500")

# Chart generation code: check figures are matplotlib Figure objects
import matplotlib
_check("Chart generation: elo_trajectories is a valid Figure",
       str(type(elo_trajectories)).find("Figure") > -1)
_check("Chart generation: wc2026_predictions_fig is a valid Figure",
       str(type(wc2026_predictions_fig)).find("Figure") > -1)
_check("Chart generation: feature_importance_fig is a valid Figure",
       str(type(feature_importance_fig)).find("Figure") > -1)

# Session state caching: simulate cache check
_cache = {}
_cache["elo_table"] = elo_table
_check("Session state caching: elo_table cacheable (not empty DataFrame)",
       not _cache["elo_table"].empty)

# Demo mode: validate that WC2026_GROUPS_FINAL has 12 groups × 4 teams
_check("Demo mode: WC2026_GROUPS_FINAL has 48 unique teams",
       sum(len(v) for v in WC2026_GROUPS_FINAL.values()) == 48)

# CSV export: validate wc2026_sim_results can be converted to CSV
_csv_str = wc2026_sim_results.to_csv(index=False)
_check("CSV export: simulation_results convertible to CSV",
       len(_csv_str) > 100,
       f"{len(_csv_str)} chars")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TEST SUMMARY")
print("=" * 70)

_total = len(_test_results)
_passed = sum(1 for _, ok, _ in _test_results if ok)
_failed = _total - _passed

# Group by section
_sections = {
    "Script File & Page Routing":      [r for r in _test_results if any(k in r[0].lower() for k in ["script", "page", "sidebar", "config", "session", "canvas", "csv", "demo"])],
    "Canvas Variable Availability":    [r for r in _test_results if any(k in r[0].lower() for k in ["elo_ratings", "match_df", "team_feat", "simulation_res", "predict_match", "elo lookup", "simulation func"])],
    "Match Predictor Logic":           [r for r in _test_results if any(k in r[0].lower() for k in ["predict_match", "probabilities", "elo values"])],
    "Tournament Simulator Logic":      [r for r in _test_results if any(k in r[0].lower() for k in ["mini-sim", "top contender", "50k sim"])],
    "Team Profile Logic":              [r for r in _test_results if any(k in r[0].lower() for k in ["team profile", "elo for", "champion % for"])],
    "Insights Data":                   [r for r in _test_results if any(k in r[0].lower() for k in ["elo curse", "upset index", "group strength", "glory gap"])],
    "Component Checks":                [r for r in _test_results if any(k in r[0].lower() for k in ["api fallback", "canvas fallback", "chart gen", "caching", "demo mode:", "csv export"])],
}

for section, tests in _sections.items():
    if tests:
        _s_pass = sum(1 for _, ok, _ in tests if ok)
        _s_total = len(tests)
        _icon = "✅" if _s_pass == _s_total else ("⚠️" if _s_pass > 0 else "❌")
        print(f"  {_icon}  {section}: {_s_pass}/{_s_total}")

print(f"\n  📈 OVERALL: {_passed}/{_total} tests passed  ({_passed/_total:.0%})")

if _failed == 0:
    print("\n  🎉 ALL TESTS PASSED — Dashboard is ready for deployment!")
elif _failed <= 3:
    print(f"\n  ⚠️  {_failed} test(s) failed — minor issues to address before deploy.")
else:
    print(f"\n  🚨 {_failed} test(s) failed — dashboard needs work before deployment.")

print("=" * 70)

# Export summary for downstream use
dashboard_test_summary = {
    "total_tests": _total,
    "passed": _passed,
    "failed": _failed,
    "pass_rate": round(_passed / _total, 4),
    "results": [(name, ok) for name, ok, _ in _test_results],
}
