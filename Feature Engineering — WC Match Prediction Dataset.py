
import pandas as pd
import numpy as np
from collections import defaultdict

# ── Re-derive ELO snapshots at match time ─────────────────────────────────────
# We'll use the pre_match_elo_df computed in the ELO block

# ── Recent form builder (last N games before a given date) ───────────────────
def build_recent_form(results_df, window=10):
    """Returns dict: (team, date) -> win_rate over last `window` games."""
    # Build a flat per-team match list
    home_df = results_df[['date','home_team','away_team','home_result']].copy()
    home_df.columns = ['date','team','opponent','result']
    
    away_df = results_df[['date','home_team','away_team','home_result']].copy()
    away_df['result'] = away_df['home_result'].map({'W':'L','D':'D','L':'W'})
    away_df = away_df[['date','away_team','home_team','result']].copy()
    away_df.columns = ['date','team','opponent','result']
    
    all_games = pd.concat([home_df, away_df]).sort_values('date').reset_index(drop=True)
    all_games['win_pts'] = all_games['result'].map({'W':1,'D':0.5,'L':0})
    
    # Group by team and compute rolling window
    form_dict = {}
    for team, grp in all_games.groupby('team'):
        grp = grp.sort_values('date').reset_index(drop=True)
        for i, row in grp.iterrows():
            past = grp[grp['date'] < row['date']].tail(window)
            win_rate = past['win_pts'].mean() if len(past) > 0 else 0.5
            form_dict[(team, row['date'])] = win_rate
    return form_dict

# ── Head-to-head record ───────────────────────────────────────────────────────
def build_h2h(results_df):
    """Returns dict: (home, away) -> home_win_rate in H2H."""
    h2h = defaultdict(lambda: [0, 0, 0])  # [home_wins, draws, away_wins]
    sorted_df = results_df.sort_values('date')
    h2h_snap = {}  # (home, away, date) -> record_at_that_point
    for _, row in sorted_df.iterrows():
        h, a = row['home_team'], row['away_team']
        key = (h, a)
        # snapshot BEFORE this match
        rec = h2h[key][:]
        total = sum(rec)
        h2h_snap[(h, a, row['date'])] = rec[0] / total if total > 0 else 0.5
        # update
        if row['home_result'] == 'W': h2h[key][0] += 1
        elif row['home_result'] == 'D': h2h[key][1] += 1
        else: h2h[key][2] += 1
    return h2h_snap

# ── WC appearances per team ───────────────────────────────────────────────────
wc_only = world_cup_matches[world_cup_matches['year'].isin(
    [1930,1934,1938,1950,1954,1958,1962,1966,1970,1974,
     1978,1982,1986,1990,1994,1998,2002,2006,2010,2014,2018,2022]
)].copy()

wc_teams_by_year = {}
for yr in sorted(wc_only['year'].unique()):
    sub = wc_only[wc_only['year'] == yr]
    teams = set(sub['home_team'].tolist() + sub['away_team'].tolist())
    wc_teams_by_year[yr] = teams

def wc_appearances_before(team, year):
    return sum(1 for y, teams in wc_teams_by_year.items() if y < year and team in teams)

# ── Confederation groupings ───────────────────────────────────────────────────
CONF = {
    'UEFA': ['England','France','Germany','Spain','Italy','Portugal','Netherlands',
             'Belgium','Switzerland','Croatia','Denmark','Sweden','Poland','Serbia',
             'Austria','Czech Republic','Hungary','Scotland','Turkey','Norway',
             'Ukraine','Greece','Slovakia','Slovenia','Albania','Kosovo',
             'Bosnia-Herzegovina','Finland','Wales','Romania','Bulgaria','Georgia',
             'North Macedonia','Montenegro','Iceland','Luxembourg','Azerbaijan',
             'Armenia','Cyprus','Malta','Liechtenstein','San Marino','Andorra',
             'Moldova','Belarus','Kazakhstan','Ireland','Northern Ireland'],
    'CONMEBOL': ['Brazil','Argentina','Uruguay','Colombia','Chile','Paraguay',
                 'Ecuador','Peru','Bolivia','Venezuela'],
    'CONCACAF': ['Mexico','United States','Canada','Costa Rica','Honduras',
                 'Panama','Jamaica','Trinidad and Tobago','Haiti','Curacao'],
    'CAF': ['Morocco','Senegal','Nigeria','Ghana','Ivory Coast','Cameroon',
            'Egypt','Tunisia','Algeria','South Africa','Mali'],
    'AFC': ['Japan','South Korea','Australia','Iran','Saudi Arabia',
            'Qatar','Iraq','Jordan','Uzbekistan','Oman'],
    'OFC': ['New Zealand','Fiji','Papua New Guinea'],
}
CONF_ELO = {}  # computed below

def get_confederation(team):
    for conf, members in CONF.items():
        if team in members:
            return conf
    return 'OTHER'

# ── Build feature dataset from World Cup matches (1966–2022) ─────────────────
# Only proper WC tournament matches (not qualifiers)
PROPER_WC_YEARS = [1966,1970,1974,1978,1982,1986,1990,1994,1998,2002,2006,2010,2014,2018,2022]

print("Building recent form dictionary… (may take ~30s)")
form_dict = build_recent_form(international_results)
print("Building H2H dictionary…")
h2h_snap = build_h2h(international_results)
print("Done. Building feature matrix…")

wc_proper = world_cup_matches[world_cup_matches['year'].isin(PROPER_WC_YEARS)].copy()
wc_proper = wc_proper.dropna(subset=['home_score','away_score'])

# Merge pre-match ELOs
wc_features = wc_proper.merge(
    pre_match_elo_df[['date','home_team','away_team','home_elo_pre','away_elo_pre','elo_diff']],
    on=['date','home_team','away_team'], how='left'
)

rows = []
for _, row in wc_features.iterrows():
    h, a, d, yr = row['home_team'], row['away_team'], row['date'], row['year']
    
    home_elo = row.get('home_elo_pre', BASE_ELO)
    away_elo = row.get('away_elo_pre', BASE_ELO)
    elo_diff = (home_elo or BASE_ELO) - (away_elo or BASE_ELO)
    
    home_form = form_dict.get((h, d), 0.5)
    away_form = form_dict.get((a, d), 0.5)
    
    h2h_val = h2h_snap.get((h, a, d), 0.5)
    
    home_wc_exp = wc_appearances_before(h, yr)
    away_wc_exp = wc_appearances_before(a, yr)
    
    home_conf = get_confederation(h)
    away_conf = get_confederation(a)
    
    # Outcome label (from home team's perspective; WC is neutral)
    gh, ga = row['home_goals'], row['away_goals']
    if gh > ga:     outcome = 2   # home win
    elif gh == ga:  outcome = 1   # draw
    else:           outcome = 0   # away win
    
    rows.append({
        'date': d, 'year': yr,
        'home_team': h, 'away_team': a,
        'home_goals': gh, 'away_goals': ga,
        'elo_diff': elo_diff,
        'home_elo': home_elo or BASE_ELO,
        'away_elo': away_elo or BASE_ELO,
        'home_form': home_form,
        'away_form': away_form,
        'form_diff': home_form - away_form,
        'h2h_home_winrate': h2h_val,
        'home_wc_exp': home_wc_exp,
        'away_wc_exp': away_wc_exp,
        'wc_exp_diff': home_wc_exp - away_wc_exp,
        'home_conf': home_conf,
        'away_conf': away_conf,
        'outcome': outcome,
    })

wc_model_df = pd.DataFrame(rows)
print(f"\nFeature matrix shape: {wc_model_df.shape}")
print(f"Outcome distribution:\n{wc_model_df['outcome'].value_counts().rename({2:'Home Win',1:'Draw',0:'Away Win'})}")
print(f"\nSample features:")
print(wc_model_df[['home_team','away_team','elo_diff','home_form','away_form','h2h_home_winrate','outcome']].head(8).to_string(index=False))

print("\n✅ Feature engineering complete.")
