
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

# ── ELO Parameters ────────────────────────────────────────────────────────────
BASE_ELO    = 1500
K_BASE      = 32        # Base K-factor
HOME_ADV    = 100       # ELO points added for home advantage
MIN_MATCHES = 5         # Min matches before ELO is considered reliable

# ── Goal difference multiplier (FIFA-style) ───────────────────────────────────
def gd_multiplier(goal_diff):
    """Amplify ELO swing for large wins."""
    gd = abs(goal_diff)
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    if gd == 3: return 1.75
    return 1.75 + (gd - 3) * 0.125

# ── Process matches chronologically ──────────────────────────────────────────
matches = international_results.dropna(subset=['home_score','away_score']).copy()
matches = matches.sort_values('date').reset_index(drop=True)

elo = defaultdict(lambda: BASE_ELO)
match_count = defaultdict(int)

# Store snapshots: {team: [(date, elo), ...]}
elo_history = defaultdict(list)

# Also store every match's pre-match ELOs (for feature engineering later)
pre_match_elo_rows = []

for _, row in matches.iterrows():
    h, a = row['home_team'], row['away_team']
    neutral = row['is_neutral']
    importance = row['importance']

    elo_h = elo[h]
    elo_a = elo[a]

    # Home advantage (not applied in neutral venues)
    elo_h_adj = elo_h + (0 if neutral else HOME_ADV)

    # Expected scores
    exp_h = 1 / (1 + 10 ** ((elo_a - elo_h_adj) / 400))
    exp_a = 1 - exp_h

    # Actual scores
    gh, ga = row['home_goals'], row['away_goals']
    if gh > ga:
        act_h, act_a = 1.0, 0.0
    elif gh == ga:
        act_h, act_a = 0.5, 0.5
    else:
        act_h, act_a = 0.0, 1.0

    gd_mult = gd_multiplier(gh - ga)
    K = K_BASE * importance * gd_mult

    # Pre-match snapshot for features
    pre_match_elo_rows.append({
        'date': row['date'],
        'home_team': h,
        'away_team': a,
        'home_elo_pre': elo_h,
        'away_elo_pre': elo_a,
        'elo_diff': elo_h_adj - elo_a,
    })

    # Update ELO
    elo[h] = elo_h + K * (act_h - exp_h)
    elo[a] = elo_a + K * (act_a - exp_a)

    match_count[h] += 1
    match_count[a] += 1

    # Record history every match
    elo_history[h].append((row['date'], elo[h]))
    elo_history[a].append((row['date'], elo[a]))

# ── Final ELO leaderboard ─────────────────────────────────────────────────────
elo_table = pd.DataFrame([
    {'team': t, 'elo': round(v, 1), 'matches_played': match_count[t]}
    for t, v in elo.items()
    if match_count[t] >= MIN_MATCHES
]).sort_values('elo', ascending=False).reset_index(drop=True)
elo_table.index += 1  # 1-indexed rank

print("=== CURRENT ELO LEADERBOARD (Top 30) ===")
print(elo_table.head(30).to_string())

# Pre-match ELO dataframe (for feature engineering)
pre_match_elo_df = pd.DataFrame(pre_match_elo_rows)

# ── Visualize ELO trajectories — top 16 teams ─────────────────────────────────
top16 = elo_table.head(16)['team'].tolist()

# Zerve palette
PALETTE = [
    '#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF','#ffd400',
    '#1F77B4','#9467BD','#8C564B','#C49C94','#E377C2','#F7B6D2',
    '#17b26a','#f04438','#5DA5DA','#FAA43A'
]

elo_trajectories = plt.figure(figsize=(16, 9))
ax = elo_trajectories.add_subplot(111)
ax.set_facecolor('#1D1D20')
elo_trajectories.patch.set_facecolor('#1D1D20')

for i, team in enumerate(top16):
    hist = pd.DataFrame(elo_history[team], columns=['date','elo'])
    hist = hist[hist['date'] >= '1970-01-01']  # Modern era
    ax.plot(hist['date'], hist['elo'], linewidth=1.8,
            color=PALETTE[i], alpha=0.85, label=team)

ax.set_title('ELO Rating Trajectories — Top 16 National Teams (1970–2025)',
             color='#fbfbff', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('Year', color='#909094', fontsize=11)
ax.set_ylabel('ELO Rating', color='#909094', fontsize=11)
ax.tick_params(colors='#909094')
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#909094')
ax.set_xlim(pd.Timestamp('1970-01-01'), pd.Timestamp('2026-12-31'))
ax.set_ylim(1300, 2100)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.grid(True, alpha=0.15, color='#909094')
leg = ax.legend(loc='upper left', fontsize=8, framealpha=0.3,
                facecolor='#1D1D20', labelcolor='#fbfbff',
                ncol=2, borderpad=0.5)
plt.tight_layout()
print("\n✅ ELO system built. Trajectories chart ready.")
print(f"   Teams rated: {len(elo_table)}")
print(f"   Top 5: {elo_table.head(5)[['team','elo']].to_string(index=False)}")
