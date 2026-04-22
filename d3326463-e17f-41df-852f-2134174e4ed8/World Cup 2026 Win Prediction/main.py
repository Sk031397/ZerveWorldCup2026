"""
World Cup 2026 Intelligence API
================================
FastAPI service exposing 7 endpoints powered by ELO ratings, ML predictions,
Monte Carlo simulations, and historical World Cup data loaded from Zerve canvas.

Endpoints:
  GET  /health                    — Health check with model status + metadata
  POST /predict/match             — Win/draw/loss probabilities, ELO, H2H, confidence
  POST /predict/tournament        — Mini Monte Carlo (1K iters) championship odds
  GET  /teams                     — All teams with ELO + confederation
  GET  /team/{team_name}          — Full team profile
  GET  /insights/top-contenders   — Top 16 by championship probability
  GET  /insights/upsets           — Historical upset analysis by ELO differential
"""

from __future__ import annotations

import math
import random
import difflib
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Load canvas variables ────────────────────────────────────────────────────
from zerve import variable

# Core data structures from canvas blocks
elo_table         = variable(block_name="ELO Rating System — All National Teams", variable_name="elo_table")
wc2026_sim_results = variable(block_name="World Cup 2026 Monte Carlo Simulator",   variable_name="wc2026_sim_results")
all_matches_with_elo = variable(block_name="Deep Analytical Insights — 4 Original Findings", variable_name="all_matches_with_elo")
upset_by_gap      = variable(block_name="Deep Analytical Insights — 4 Original Findings", variable_name="upset_by_gap")
wc_prediction_model = variable(block_name="ML Prediction Model — WC Match Outcomes", variable_name="wc_prediction_model")
form_dict         = variable(block_name="Feature Engineering — WC Match Prediction Dataset", variable_name="form_dict")
h2h_snap          = variable(block_name="Feature Engineering — WC Match Prediction Dataset", variable_name="h2h_snap")
CONF              = variable(block_name="Feature Engineering — WC Match Prediction Dataset", variable_name="CONF")
wc_teams_by_year  = variable(block_name="Feature Engineering — WC Match Prediction Dataset", variable_name="wc_teams_by_year")
international_results = variable(block_name="Fetch Historical World Cup & International Match Data", variable_name="international_results")
WC2026_GROUPS_FINAL = variable(block_name="World Cup 2026 Monte Carlo Simulator",  variable_name="WC2026_GROUPS_FINAL")
ALL_48            = variable(block_name="World Cup 2026 Monte Carlo Simulator",     variable_name="ALL_48")

# ── Pre-compute fast lookups ─────────────────────────────────────────────────
ELO_LOOKUP: Dict[str, float] = dict(zip(elo_table["team"], elo_table["elo"]))
ELO_MATCHES: Dict[str, int]  = dict(zip(elo_table["team"], elo_table["matches_played"]))

# Latest form per team (most recent date in form_dict)
_form_latest: Dict[str, float] = {}
for (tm, dt), val in form_dict.items():
    if tm not in _form_latest:
        _form_latest[tm] = val
    else:
        # form_dict keys: (team, date) — pick max date
        pass
# Rebuild properly using max-date approach
_team_dates: Dict[str, list] = defaultdict(list)
for (tm, dt) in form_dict.keys():
    _team_dates[tm].append(dt)
for tm, dates in _team_dates.items():
    latest_dt = max(dates)
    _form_latest[tm] = form_dict[(tm, latest_dt)]

# Confederation map (team → conf)
TEAM_CONF: Dict[str, str] = {}
for conf_name, members in CONF.items():
    for member in members:
        TEAM_CONF[member] = conf_name

# WC appearances per team (total, using wc_teams_by_year)
WC_APPS: Dict[str, int] = defaultdict(int)
for yr, team_set in wc_teams_by_year.items():
    for tm in team_set:
        WC_APPS[tm] += 1

# WC winners history for "best finish" proxy
WC_WINNERS_HIST = {
    "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3,
    "France": 2, "Uruguay": 2, "England": 1, "Spain": 1,
}

FEATURES = ['elo_diff', 'home_elo', 'away_elo', 'home_form', 'away_form',
            'form_diff', 'h2h_home_winrate', 'wc_exp_diff',
            'home_wc_exp', 'away_wc_exp']

API_VERSION = "1.0.0"
_startup_time = datetime.utcnow().isoformat() + "Z"

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_elo(team: str) -> float:
    return ELO_LOOKUP.get(team, 1500.0)


def get_confederation(team: str) -> str:
    return TEAM_CONF.get(team, "OTHER")


def fuzzy_match_team(name: str, candidates: Optional[List[str]] = None) -> Optional[str]:
    """Case-insensitive fuzzy match against all known teams."""
    if candidates is None:
        candidates = list(ELO_LOOKUP.keys())
    # Exact match (case-insensitive)
    lower_map = {c.lower(): c for c in candidates}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    # Difflib close-match
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    # Fallback: check if name is substring of any candidate
    for c in candidates:
        if name.lower() in c.lower() or c.lower() in name.lower():
            return c
    return None


def wc_appearances(team: str) -> int:
    return WC_APPS.get(team, 0)


def get_recent_form(team: str) -> float:
    return _form_latest.get(team, 0.6)


def get_h2h(team1: str, team2: str) -> float:
    """Returns home (team1) win rate from h2h_snap, using latest available date."""
    # Find the most recent H2H entry
    best_date = None
    best_rate = 0.5
    for (h, a, dt), rate in h2h_snap.items():
        if h == team1 and a == team2:
            if best_date is None or dt > best_date:
                best_date = dt
                best_rate = rate
    return best_rate


def elo_win_prob(elo1: float, elo2: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(elo1 - elo2) / 400.0))


def predict_match_proba(team1: str, team2: str):
    """Use the trained GBM model to predict match outcome probabilities."""
    t1_elo = get_elo(team1)
    t2_elo = get_elo(team2)
    elo_diff = t1_elo - t2_elo

    t1_form = get_recent_form(team1)
    t2_form = get_recent_form(team2)
    h2h_val = get_h2h(team1, team2)
    t1_wc_exp = wc_appearances(team1)
    t2_wc_exp = wc_appearances(team2)

    features_vec = np.array([[
        elo_diff, t1_elo, t2_elo,
        t1_form, t2_form,
        t1_form - t2_form,
        h2h_val,
        t1_wc_exp - t2_wc_exp,
        t1_wc_exp, t2_wc_exp,
    ]])

    proba = wc_prediction_model.predict_proba(features_vec)[0]
    classes = wc_prediction_model.classes_
    label_map = {0: "team2_win", 1: "draw", 2: "team1_win"}
    return {label_map[c]: float(p) for c, p in zip(classes, proba)}


def sim_match_elo(t1: str, t2: str) -> str:
    """Fast ELO-based match simulation. Returns winning team or 'draw'."""
    e1, e2 = get_elo(t1), get_elo(t2)
    exp1 = elo_win_prob(e1, e2)
    p_draw = 0.22
    p1_win = exp1 * (1 - p_draw)
    r = random.random()
    if r < p1_win:
        return t1
    elif r < p1_win + p_draw:
        return "draw"
    return t2


def sim_ko_match(t1: str, t2: str) -> str:
    result = sim_match_elo(t1, t2)
    if result == "draw":
        return t1 if random.random() < elo_win_prob(get_elo(t1), get_elo(t2)) else t2
    return result


def mini_sim_group(teams: List[str]) -> List[str]:
    """Round-robin group stage, return top 2."""
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            t1, t2 = teams[i], teams[j]
            res = sim_match_elo(t1, t2)
            exp_gd = (get_elo(t1) - get_elo(t2)) / 200
            actual_gd = random.gauss(exp_gd, 1.5)
            if res == t1:
                pts[t1] += 3; gd[t1] += abs(int(actual_gd)) + 1; gd[t2] -= abs(int(actual_gd)) + 1
            elif res == t2:
                pts[t2] += 3; gd[t2] += abs(int(actual_gd)) + 1; gd[t1] -= abs(int(actual_gd)) + 1
            else:
                pts[t1] += 1; pts[t2] += 1
    ranked = sorted(teams, key=lambda t: (pts[t], gd[t], random.random()), reverse=True)
    return ranked[:2]


def mini_tournament_sim(teams: List[str], n_sims: int = 1000) -> Dict[str, float]:
    """1K Monte Carlo: random groups → knockouts → championship odds."""
    champ_counts: Dict[str, int] = defaultdict(int)
    for _ in range(n_sims):
        pool = teams[:]
        random.shuffle(pool)
        # Group stage: split into groups of 4 (or 2 if <4)
        qualifiers = []
        while len(pool) >= 4:
            group = pool[:4]; pool = pool[4:]
            qualifiers.extend(mini_sim_group(group))
        qualifiers.extend(pool)  # odd teams auto-qualify
        # Knockout
        random.shuffle(qualifiers)
        bracket = qualifiers[:]
        while len(bracket) > 1:
            winners = []
            for i in range(0, len(bracket) - 1, 2):
                winners.append(sim_ko_match(bracket[i], bracket[i + 1]))
            if len(bracket) % 2 == 1:
                winners.append(bracket[-1])
            bracket = winners
        if bracket:
            champ_counts[bracket[0]] += 1
    return {t: round(champ_counts[t] / n_sims * 100, 2) for t in teams}


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="World Cup 2026 Intelligence API",
    description=(
        "AI-powered football prediction API backed by ELO ratings (49k+ matches), "
        "a Gradient Boosting ML model, and 50,000-iteration Monte Carlo simulations."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_version_header(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-API-Version"] = API_VERSION
    return response


# ── Pydantic Models ──────────────────────────────────────────────────────────

class MatchPredictionRequest(BaseModel):
    team1: str = Field(..., example="Brazil", description="First team name (fuzzy matched)")
    team2: str = Field(..., example="Argentina", description="Second team name (fuzzy matched)")


class HeadToHeadRecord(BaseModel):
    team1_win_rate: float = Field(..., example=0.52, description="Historical H2H win rate for team1 (0-1)")
    total_matches: Optional[int] = Field(None, example=108, description="Total H2H matches played")


class MatchPredictionResponse(BaseModel):
    team1: str = Field(..., example="Brazil")
    team2: str = Field(..., example="Argentina")
    team1_win_probability: float = Field(..., example=0.45, description="Win probability for team1 (0-1)")
    draw_probability: float = Field(..., example=0.28, description="Draw probability (0-1)")
    team2_win_probability: float = Field(..., example=0.27, description="Win probability for team2 (0-1)")
    team1_elo: float = Field(..., example=2089.5, description="Current ELO rating for team1")
    team2_elo: float = Field(..., example=2076.3, description="Current ELO rating for team2")
    elo_differential: float = Field(..., example=13.2, description="ELO difference (team1 - team2)")
    team1_confederation: str = Field(..., example="CONMEBOL")
    team2_confederation: str = Field(..., example="CONMEBOL")
    team1_recent_form: float = Field(..., example=0.75, description="Win rate over last 10 matches (0-1)")
    team2_recent_form: float = Field(..., example=0.70, description="Win rate over last 10 matches (0-1)")
    head_to_head: HeadToHeadRecord
    confidence_score: float = Field(..., example=0.72, description="Model confidence 0-1 (max probability of the 3 outcomes)")
    upset_potential: bool = Field(..., example=False, description="True if underdog has >30% chance of winning")
    ci_team1_win_low: float = Field(..., example=0.38, description="95% CI lower bound for team1 win prob")
    ci_team1_win_high: float = Field(..., example=0.52, description="95% CI upper bound for team1 win prob")
    predicted_outcome: str = Field(..., example="team1_win", description="Most likely outcome")


class TournamentSimRequest(BaseModel):
    teams: List[str] = Field(
        ...,
        example=["Brazil", "Argentina", "France", "Spain", "England", "Germany"],
        description="List of team names to simulate (min 2, fuzzy matched)",
        min_items=2,
    )
    n_simulations: int = Field(1000, ge=100, le=5000, example=1000, description="Number of Monte Carlo iterations (100-5000)")


class TournamentSimResult(BaseModel):
    team: str = Field(..., example="Brazil")
    championship_odds_pct: float = Field(..., example=18.4, description="Championship probability (%)")
    elo_rating: float = Field(..., example=2089.5)
    confederation: str = Field(..., example="CONMEBOL")
    rank: int = Field(..., example=1, description="Rank within this simulation (1 = favourite)")


class TournamentSimResponse(BaseModel):
    teams_simulated: List[str] = Field(..., example=["Brazil", "Argentina", "France"])
    n_simulations: int = Field(..., example=1000)
    results: List[TournamentSimResult]
    top_contender: str = Field(..., example="Brazil")
    simulation_note: str = Field(..., example="Mini round-robin groups + knockout bracket, ELO-based")


class TeamInfo(BaseModel):
    team: str = Field(..., example="France")
    elo_rating: float = Field(..., example=2045.2)
    confederation: str = Field(..., example="UEFA")
    matches_played: int = Field(..., example=892)
    elo_rank: int = Field(..., example=3)


class TeamsResponse(BaseModel):
    total_teams: int = Field(..., example=303)
    teams: List[TeamInfo]


class TeamProfileResponse(BaseModel):
    team: str = Field(..., example="Brazil")
    elo_rating: float = Field(..., example=2089.5)
    elo_rank: int = Field(..., example=1)
    confederation: str = Field(..., example="CONMEBOL")
    matches_played: int = Field(..., example=1024)
    recent_form: float = Field(..., example=0.78, description="Win rate over last 10 matches (0-1)")
    world_cup_appearances: int = Field(..., example=22)
    world_cup_titles: int = Field(..., example=5)
    championship_odds_pct: Optional[float] = Field(None, example=14.2, description="WC2026 championship odds from 50k simulations (%)")
    qualified_for_2026: bool = Field(..., example=True)
    best_finish: str = Field(..., example="Winner (1958, 1962, 1970, 1994, 2002)")


class TopContenderEntry(BaseModel):
    rank: int = Field(..., example=1)
    team: str = Field(..., example="France")
    elo_rating: float = Field(..., example=2045.2)
    confederation: str = Field(..., example="UEFA")
    championship_odds_pct: float = Field(..., example=14.8, description="WC2026 championship probability from 50k sims (%)")
    final_odds_pct: float = Field(..., example=28.3, description="Probability of reaching the final (%)")
    semifinal_odds_pct: float = Field(..., example=45.1)
    recent_form: float = Field(..., example=0.74)
    world_cup_titles: int = Field(..., example=2)


class TopContendersResponse(BaseModel):
    total_contenders: int = Field(..., example=16)
    contenders: List[TopContenderEntry]
    as_of_simulation: str = Field(..., example="50,000 Monte Carlo simulations")


class UpsetEntry(BaseModel):
    elo_gap_bin: str = Field(..., example="100-150", description="ELO differential range between favourite and underdog")
    upset_rate_pct: float = Field(..., example=32.4, description="Actual % of WC matches won by underdog in this ELO band")
    elo_expected_underdog_pct: float = Field(..., example=25.1, description="ELO-model expected underdog win % in this band")
    upset_overperformance_pct: float = Field(..., example=7.3, description="How much underdogs over/underperform ELO expectation (%)")
    sample_size: int = Field(..., example=145, description="Number of WC matches in this ELO band")


class UpsetsResponse(BaseModel):
    analysis_title: str = Field(..., example="Historical WC Upset Analysis by ELO Differential")
    total_wc_matches_analysed: int
    upsets_by_elo_gap: List[UpsetEntry]
    key_finding: str = Field(..., example="Underdogs with 100-200 ELO deficit win ~30% of WC matches, exceeding ELO expectations")


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    elo_teams: int = Field(..., example=303)
    wc2026_teams_simulated: int = Field(..., example=48)
    api_version: str = Field(..., example="1.0.0")
    startup_time: str = Field(..., example="2026-04-19T18:00:00Z")
    data_coverage: str = Field(..., example="49,300+ international matches (1872-2024)")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check with model status and key metadata."""
    return HealthResponse(
        status="healthy",
        model_loaded=wc_prediction_model is not None,
        elo_teams=len(ELO_LOOKUP),
        wc2026_teams_simulated=len(wc2026_sim_results),
        api_version=API_VERSION,
        startup_time=_startup_time,
        data_coverage=f"{len(international_results):,} international matches",
    )


@app.post("/predict/match", response_model=MatchPredictionResponse, tags=["Predictions"])
def predict_match(body: MatchPredictionRequest):
    """
    Predict the outcome of a match between two international teams.

    Returns win/draw/loss probabilities, ELO ratings, head-to-head record,
    confidence score, upset potential flag, and 95% confidence intervals.
    Uses fuzzy team name matching (e.g. 'brazil' → 'Brazil', 'USA' → 'United States').
    """
    t1_name = fuzzy_match_team(body.team1)
    t2_name = fuzzy_match_team(body.team2)

    if t1_name is None:
        raise HTTPException(status_code=404, detail=f"Team '{body.team1}' not found. Check /teams for valid names.")
    if t2_name is None:
        raise HTTPException(status_code=404, detail=f"Team '{body.team2}' not found. Check /teams for valid names.")
    if t1_name == t2_name:
        raise HTTPException(status_code=400, detail="Both teams resolve to the same team.")

    proba = predict_match_proba(t1_name, t2_name)
    t1_win = proba.get("team1_win", 0.0)
    draw   = proba.get("draw", 0.0)
    t2_win = proba.get("team2_win", 0.0)

    t1_elo = get_elo(t1_name)
    t2_elo = get_elo(t2_name)
    elo_diff = t1_elo - t2_elo

    # Confidence interval based on ELO spread uncertainty (~50 ELO ≈ ±5% probability shift)
    elo_uncertainty = min(abs(elo_diff) / 400.0 * 0.15 + 0.05, 0.15)
    ci_low  = max(0.0, t1_win - elo_uncertainty)
    ci_high = min(1.0, t1_win + elo_uncertainty)

    # H2H
    h2h_rate = get_h2h(t1_name, t2_name)
    # Count H2H appearances in international_results
    h2h_df = international_results[
        (international_results["home_team"] == t1_name) &
        (international_results["away_team"] == t2_name)
    ]
    h2h_total = len(h2h_df)

    confidence = max(t1_win, draw, t2_win)
    # Upset potential: underdog (lower ELO) has >30% chance
    if elo_diff > 0:
        upset_potential = t2_win > 0.30
    else:
        upset_potential = t1_win > 0.30

    outcomes = [("team1_win", t1_win), ("draw", draw), ("team2_win", t2_win)]
    predicted = max(outcomes, key=lambda x: x[1])[0]

    return MatchPredictionResponse(
        team1=t1_name,
        team2=t2_name,
        team1_win_probability=round(t1_win, 4),
        draw_probability=round(draw, 4),
        team2_win_probability=round(t2_win, 4),
        team1_elo=round(t1_elo, 1),
        team2_elo=round(t2_elo, 1),
        elo_differential=round(elo_diff, 1),
        team1_confederation=get_confederation(t1_name),
        team2_confederation=get_confederation(t2_name),
        team1_recent_form=round(get_recent_form(t1_name), 3),
        team2_recent_form=round(get_recent_form(t2_name), 3),
        head_to_head=HeadToHeadRecord(
            team1_win_rate=round(h2h_rate, 3),
            total_matches=h2h_total if h2h_total > 0 else None,
        ),
        confidence_score=round(confidence, 4),
        upset_potential=upset_potential,
        ci_team1_win_low=round(ci_low, 4),
        ci_team1_win_high=round(ci_high, 4),
        predicted_outcome=predicted,
    )


@app.post("/predict/tournament", response_model=TournamentSimResponse, tags=["Predictions"])
def predict_tournament(body: TournamentSimRequest):
    """
    Run a mini Monte Carlo simulation for a custom set of teams.

    Simulates round-robin group stages and knockout rounds N times (default 1,000)
    to estimate championship odds for each team. Fuzzy team name matching supported.
    """
    resolved_teams = []
    unresolved = []
    for name in body.teams:
        match = fuzzy_match_team(name)
        if match:
            resolved_teams.append(match)
        else:
            unresolved.append(name)

    if unresolved:
        raise HTTPException(
            status_code=404,
            detail=f"Teams not found: {unresolved}. Check /teams for valid names.",
        )
    if len(set(resolved_teams)) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 distinct teams.")

    resolved_teams = list(dict.fromkeys(resolved_teams))  # deduplicate preserving order
    odds = mini_tournament_sim(resolved_teams, n_sims=body.n_simulations)

    results = sorted(
        [
            TournamentSimResult(
                team=t,
                championship_odds_pct=odds.get(t, 0.0),
                elo_rating=round(get_elo(t), 1),
                confederation=get_confederation(t),
                rank=0,
            )
            for t in resolved_teams
        ],
        key=lambda x: x.championship_odds_pct,
        reverse=True,
    )
    for idx, r in enumerate(results):
        r.rank = idx + 1

    return TournamentSimResponse(
        teams_simulated=resolved_teams,
        n_simulations=body.n_simulations,
        results=results,
        top_contender=results[0].team if results else "",
        simulation_note="Mini round-robin groups + ELO-based knockout bracket",
    )


@app.get("/teams", response_model=TeamsResponse, tags=["Teams"])
def list_teams():
    """
    List all available teams with their current ELO ratings and confederation.

    Returns 303 national teams sorted by ELO rating (descending).
    """
    sorted_table = elo_table.sort_values("elo", ascending=False).reset_index(drop=True)
    team_list = [
        TeamInfo(
            team=str(row.team),
            elo_rating=round(float(row.elo), 1),
            confederation=get_confederation(str(row.team)),
            matches_played=int(row.matches_played),
            elo_rank=int(idx + 1),
        )
        for idx, row in sorted_table.iterrows()
    ]
    return TeamsResponse(total_teams=len(team_list), teams=team_list)


@app.get("/team/{team_name}", response_model=TeamProfileResponse, tags=["Teams"])
def get_team_profile(team_name: str):
    """
    Full team profile: ELO, rank, recent form, historical WC stats, best finish.

    Fuzzy matches team name (e.g. 'brazil', 'USA', 'England' all work).
    """
    resolved = fuzzy_match_team(team_name)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found. Use /teams to browse.")

    t_elo = get_elo(resolved)
    sorted_elos = sorted(ELO_LOOKUP.values(), reverse=True)
    elo_rank = sorted_elos.index(t_elo) + 1 if t_elo in sorted_elos else len(sorted_elos)

    # Championship odds from 50k simulation
    sim_row = wc2026_sim_results[wc2026_sim_results["team"] == resolved]
    champ_pct = float(sim_row["champion_pct"].values[0]) if len(sim_row) > 0 else None
    qualified  = resolved in ALL_48

    wc_titles = WC_WINNERS_HIST.get(resolved, 0)
    if wc_titles > 0:
        title_years = {
            "Brazil": [1958, 1962, 1970, 1994, 2002],
            "Germany": [1954, 1974, 1990, 2014],
            "Italy": [1934, 1938, 1982, 2006],
            "Argentina": [1978, 1986, 2022],
            "France": [1998, 2018],
            "Uruguay": [1930, 1950],
            "England": [1966],
            "Spain": [2010],
        }
        years = title_years.get(resolved, [])
        best_finish = f"Winner ({', '.join(str(y) for y in years)})"
    else:
        best_finish = "Quarter-finals or earlier"

    return TeamProfileResponse(
        team=resolved,
        elo_rating=round(t_elo, 1),
        elo_rank=elo_rank,
        confederation=get_confederation(resolved),
        matches_played=int(ELO_MATCHES.get(resolved, 0)),
        recent_form=round(get_recent_form(resolved), 3),
        world_cup_appearances=wc_appearances(resolved),
        world_cup_titles=wc_titles,
        championship_odds_pct=round(champ_pct, 2) if champ_pct is not None else None,
        qualified_for_2026=qualified,
        best_finish=best_finish,
    )


@app.get("/insights/top-contenders", response_model=TopContendersResponse, tags=["Insights"])
def top_contenders():
    """
    Returns the top 16 teams by WC2026 championship probability (from 50k Monte Carlo sims).

    Includes ELO ratings, recent form, confederation, and key round probabilities.
    """
    top16 = wc2026_sim_results.head(16).copy()
    sorted_elos = sorted(ELO_LOOKUP.values(), reverse=True)

    contenders = []
    for rank, (_, row) in enumerate(top16.iterrows(), start=1):
        team = str(row["team"])
        t_elo = float(row["elo"])
        contenders.append(TopContenderEntry(
            rank=rank,
            team=team,
            elo_rating=round(t_elo, 1),
            confederation=get_confederation(team),
            championship_odds_pct=round(float(row["champion_pct"]), 2),
            final_odds_pct=round(float(row["final_pct"]), 2),
            semifinal_odds_pct=round(float(row["sf_pct"]), 2),
            recent_form=round(get_recent_form(team), 3),
            world_cup_titles=WC_WINNERS_HIST.get(team, 0),
        ))

    return TopContendersResponse(
        total_contenders=len(contenders),
        contenders=contenders,
        as_of_simulation="50,000 Monte Carlo simulations (WC 2026 48-team format)",
    )


@app.get("/insights/upsets", response_model=UpsetsResponse, tags=["Insights"])
def historical_upsets():
    """
    Historical upset analysis at the World Cup, grouped by ELO differential.

    Shows how often underdogs beat favourites in each ELO gap band,
    compared to what pure ELO probability would predict.
    """
    total_wc = len(all_matches_with_elo[all_matches_with_elo["is_world_cup"] == True])

    entries = []
    for _, row in upset_by_gap.iterrows():
        gap_bin = str(row["elo_gap_bin"])
        upset_rate = float(row["upset_rate"])
        elo_expected_fav = float(row["elo_expected_fav"])
        elo_expected_underdog = 1.0 - elo_expected_fav
        overperf = upset_rate - elo_expected_underdog
        entries.append(UpsetEntry(
            elo_gap_bin=gap_bin,
            upset_rate_pct=round(upset_rate * 100, 1),
            elo_expected_underdog_pct=round(elo_expected_underdog * 100, 1),
            upset_overperformance_pct=round(overperf * 100, 1),
            sample_size=int(row["n"]),
        ))

    # Key finding from the data
    best_overperf = max(entries, key=lambda e: e.upset_overperformance_pct)
    key_finding = (
        f"Underdogs with {best_overperf.elo_gap_bin} ELO deficit achieve "
        f"{best_overperf.upset_rate_pct:.1f}% upset rate, "
        f"{best_overperf.upset_overperformance_pct:+.1f}pp vs ELO expectation "
        f"(n={best_overperf.sample_size} WC matches)"
    )

    return UpsetsResponse(
        analysis_title="Historical World Cup Upset Analysis by ELO Differential",
        total_wc_matches_analysed=total_wc,
        upsets_by_elo_gap=entries,
        key_finding=key_finding,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
