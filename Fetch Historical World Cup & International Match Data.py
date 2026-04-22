
import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ── 1. International results (Kaggle public dataset via GitHub mirror) ──────
URL_RESULTS = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
URL_SHOOTOUTS = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"

print("Fetching international results…")
results_raw = pd.read_csv(URL_RESULTS)
shootouts_raw = pd.read_csv(URL_SHOOTOUTS)

print(f"  Raw results: {results_raw.shape[0]:,} rows, {results_raw.shape[1]} cols")
print(f"  Columns: {list(results_raw.columns)}")

# ── 2. Clean & enrich ────────────────────────────────────────────────────────
results_raw['date'] = pd.to_datetime(results_raw['date'])
results_raw['year'] = results_raw['date'].dt.year

# Tournament importance weights (for ELO K-factor scaling)
importance_map = {
    'FIFA World Cup': 4.0,
    'UEFA Euro': 3.0,
    'Copa América': 3.0,
    'Africa Cup of Nations': 2.5,
    'AFC Asian Cup': 2.5,
    'CONCACAF Gold Cup': 2.0,
    'FIFA Confederations Cup': 2.0,
    'UEFA Nations League': 1.5,
    'Friendly': 1.0,
}
def get_importance(tournament):
    for key, weight in importance_map.items():
        if key in str(tournament):
            return weight
    return 1.2  # default for other competitive

results_raw['importance'] = results_raw['tournament'].apply(get_importance)
results_raw['is_world_cup'] = results_raw['tournament'].str.contains('FIFA World Cup', na=False)
results_raw['is_neutral'] = results_raw['neutral']

# ── 3. World Cup subset ──────────────────────────────────────────────────────
wc_matches = results_raw[results_raw['is_world_cup']].copy()
print(f"\nWorld Cup matches: {len(wc_matches):,}")
print(f"  Years: {sorted(wc_matches['year'].unique())}")

# ── 4. Add match outcome from home team's perspective ───────────────────────
results_raw['home_goals'] = results_raw['home_score']
results_raw['away_goals'] = results_raw['away_score']
results_raw['goal_diff'] = results_raw['home_goals'] - results_raw['away_goals']
results_raw['home_result'] = np.select(
    [results_raw['goal_diff'] > 0, results_raw['goal_diff'] == 0, results_raw['goal_diff'] < 0],
    ['W', 'D', 'L']
)

# Merge shootout info — in a WC shootout, the paper score is 0-0 but the winner advances
shootouts_clean = shootouts_raw[['date','home_team','away_team','winner']].copy()
shootouts_clean['date'] = pd.to_datetime(shootouts_clean['date'])

results_raw = results_raw.merge(
    shootouts_clean.rename(columns={'winner':'shootout_winner'}),
    on=['date','home_team','away_team'],
    how='left'
)

wc_matches = results_raw[results_raw['is_world_cup']].copy()

# ── 5. World Cup editions summary ────────────────────────────────────────────
wc_editions = wc_matches.groupby('year').agg(
    matches=('date','count'),
    avg_goals=('home_goals', lambda x: (x + wc_matches.loc[x.index,'away_goals']).mean()),
    teams=('home_team', lambda x: pd.concat([x, wc_matches.loc[x.index,'away_team']]).nunique())
).reset_index()

print("\nWorld Cup editions summary:")
print(wc_editions.to_string(index=False))

# ── 6. All unique national teams ─────────────────────────────────────────────
all_teams = pd.Series(
    pd.concat([results_raw['home_team'], results_raw['away_team']]).unique()
).sort_values().reset_index(drop=True)
print(f"\nTotal unique national teams in dataset: {len(all_teams)}")

# Export key variables
international_results = results_raw
world_cup_matches = wc_matches
wc_summary = wc_editions
all_national_teams = all_teams

print("\n✅ Data loaded successfully.")
print(f"   international_results: {international_results.shape}")
print(f"   world_cup_matches:     {world_cup_matches.shape}")
