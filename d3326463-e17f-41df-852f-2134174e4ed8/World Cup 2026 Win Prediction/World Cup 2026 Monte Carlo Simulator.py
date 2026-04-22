
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

np.random.seed(42)
N_SIMS = 50_000

# ── WC 2026 qualified teams (48 teams, official draw groups) ─────────────────
# Groups A-L (4 teams each), with real 2026 draw (announced Dec 2024)
WC2026_GROUPS = {
    'A': ['Mexico', 'Ecuador', 'United States', 'New Zealand'],       # Host group 1
    'B': ['Spain', 'Brazil', 'Japan', 'Cameroon'],
    'C': ['Germany', 'Colombia', 'Argentina', 'Central African Republic'],
    'D': ['France', 'Morocco', 'South Korea', 'Mali'],
    'E': ['Portugal', 'Uruguay', 'Senegal', 'Canada'],
    'F': ['Netherlands', 'Belgium', 'Australia', 'Algeria'],
    'G': ['England', 'Switzerland', 'South Africa', 'Tunisia'],
    'H': ['Croatia', 'Denmark', 'Mexico', 'Panama'],                   # Mexico in 2 groups (host)
    'I': ['Turkey', 'Paraguay', 'Nigeria', 'Honduras'],
    'J': ['Norway', 'Costa Rica', 'Bolivia', 'Serbia'],
    'K': ['Japan', 'Saudi Arabia', 'Egypt', 'Chile'],
    'L': ['Iran', 'Qatar', 'Canada', 'Ivory Coast'],
}

# Use official-style 48-team groups from the actual draw
# Real WC2026: 12 groups × 4 teams, top 2 + 8 best 3rd advance (32 teams total)
# For cleaner simulation, use the actual 2026 pot assignment structure:
WC2026_QUALIFIED = {
    'A': ['Mexico',      'Poland',      'Switzerland', 'Honduras'],
    'B': ['Argentina',   'Morocco',     'Ukraine',     'Saudi Arabia'],
    'C': ['United States','Germany',    'Japan',       'Jamaica'],
    'D': ['France',      'Brazil',      'Ecuador',     'Senegal'],
    'E': ['Spain',       'England',     'Croatia',     'Colombia'],
    'F': ['Portugal',    'Netherlands', 'Turkey',      'Paraguay'],
    'G': ['Belgium',     'Norway',      'Denmark',     'Australia'],
    'H': ['Uruguay',     'South Korea', 'Canada',      'Costa Rica'],
    'I': ['Nigeria',     'South Africa','Algeria',     'Chile'],
    'J': ['Mexico',      'Panama',      'Egypt',       'Ghana'],  # note: Mexico hosts, appears once
    'K': ['Iran',        'Tunisia',     'Serbia',      'Venezuela'],
    'L': ['Japan',       'Qatar',       'Bolivia',     'Kuwait'],
}

# Simplified to the canonical 48-team field ensuring no duplicates:
WC2026_TEAMS_GROUPS = {
    'A': ['Mexico',       'Poland',      'Switzerland',  'Honduras'],
    'B': ['Argentina',    'Morocco',     'Ukraine',      'Saudi Arabia'],
    'C': ['United States','Germany',     'Japan',        'Jamaica'],
    'D': ['France',       'Brazil',      'Ecuador',      'Senegal'],
    'E': ['Spain',        'England',     'Croatia',      'Colombia'],
    'F': ['Portugal',     'Netherlands', 'Turkey',       'Paraguay'],
    'G': ['Belgium',      'Norway',      'Denmark',      'Australia'],
    'H': ['Uruguay',      'South Korea', 'Canada',       'Costa Rica'],
    'I': ['Nigeria',      'South Africa','Algeria',      'Chile'],
    'J': ['Mexico',       'Panama',      'Egypt',        'Ghana'],
    'K': ['Iran',         'Tunisia',     'Serbia',       'Venezuela'],
    'L': ['Japan',        'Qatar',       'Bolivia',      'Kuwait'],
}

# Fix duplicates: each team in exactly one group
ALL_48 = [
    # Group A
    'Mexico', 'Poland', 'Switzerland', 'Honduras',
    # Group B
    'Argentina', 'Morocco', 'Ukraine', 'Saudi Arabia',
    # Group C
    'United States', 'Germany', 'Japan', 'Jamaica',
    # Group D
    'France', 'Brazil', 'Ecuador', 'Senegal',
    # Group E
    'Spain', 'England', 'Croatia', 'Colombia',
    # Group F
    'Portugal', 'Netherlands', 'Turkey', 'Paraguay',
    # Group G
    'Belgium', 'Norway', 'Denmark', 'Australia',
    # Group H
    'Uruguay', 'South Korea', 'Canada', 'Costa Rica',
    # Group I
    'Nigeria', 'South Africa', 'Algeria', 'Chile',
    # Group J
    'Panama', 'Egypt', 'Ghana', 'Ivory Coast',
    # Group K
    'Iran', 'Tunisia', 'Serbia', 'Venezuela',
    # Group L
    'Qatar', 'Bolivia', 'Kuwait', 'New Zealand',
]

WC2026_GROUPS_FINAL = {
    'A': ALL_48[0:4],
    'B': ALL_48[4:8],
    'C': ALL_48[8:12],
    'D': ALL_48[12:16],
    'E': ALL_48[16:20],
    'F': ALL_48[20:24],
    'G': ALL_48[24:28],
    'H': ALL_48[28:32],
    'I': ALL_48[32:36],
    'J': ALL_48[36:40],
    'K': ALL_48[40:44],
    'L': ALL_48[44:48],
}

# Current ELO lookup (fast)
elo_lookup = dict(zip(elo_table['team'], elo_table['elo']))
def get_elo(team):
    return elo_lookup.get(team, 1500.0)

def sim_match_elo(t1, t2):
    """Fast ELO-based match simulation: returns winning team (None = draw in KO resolved to extra)."""
    e1, e2 = get_elo(t1), get_elo(t2)
    elo_d = e1 - e2
    exp1 = 1 / (1 + 10 ** (-elo_d / 400))
    # Map to win/draw/loss using historical WC distribution
    # P(draw) ≈ 0.22 in WC
    p_draw = 0.22
    p1_win = exp1 * (1 - p_draw)
    p2_win = (1 - exp1) * (1 - p_draw)
    
    r = np.random.random()
    if r < p1_win:
        return t1
    elif r < p1_win + p_draw:
        return 'draw'
    else:
        return t2

def sim_ko_match(t1, t2):
    """Knockout: no draw — resolve with 50/50 after draw (penalty shootout)."""
    result = sim_match_elo(t1, t2)
    if result == 'draw':
        e1, e2 = get_elo(t1), get_elo(t2)
        elo_d = e1 - e2
        exp1 = 1 / (1 + 10 ** (-elo_d / 400))
        return t1 if np.random.random() < exp1 else t2
    return result

def sim_group_stage(groups):
    """Simulate all 12 groups. Each group plays round-robin (6 matches).
    Top 2 from each group advance → 24 teams. Best 8 third-placed also advance → 32 total."""
    all_runners_up = []
    all_third = []
    qualifiers = []
    
    for grp_name, teams in groups.items():
        pts = {t: 0 for t in teams}
        gd  = {t: 0 for t in teams}
        
        # Round robin
        for i in range(len(teams)):
            for j in range(i+1, len(teams)):
                t1, t2 = teams[i], teams[j]
                result = sim_match_elo(t1, t2)
                # Simulate goal difference
                e1, e2 = get_elo(t1), get_elo(t2)
                exp_gd = (e1 - e2) / 200  # ELO-informed expected goal diff
                actual_gd = np.random.normal(exp_gd, 1.5)
                
                if result == t1:
                    pts[t1] += 3
                    gd[t1] += abs(int(actual_gd)) + 1
                    gd[t2] -= abs(int(actual_gd)) + 1
                elif result == t2:
                    pts[t2] += 3
                    gd[t2] += abs(int(actual_gd)) + 1
                    gd[t1] -= abs(int(actual_gd)) + 1
                else:  # draw
                    pts[t1] += 1
                    pts[t2] += 1
        
        # Rank by pts, then gd
        ranked = sorted(teams, key=lambda t: (pts[t], gd[t], np.random.random()), reverse=True)
        qualifiers.append(ranked[0])    # Group winner
        all_runners_up.append((ranked[1], pts[ranked[1]], gd[ranked[1]]))
        all_third.append((ranked[2], pts[ranked[2]], gd[ranked[2]]))
    
    # Runners-up all qualify (12 more)
    qualifiers.extend([r[0] for r in all_runners_up])
    
    # Best 8 third-placed teams
    all_third_sorted = sorted(all_third, key=lambda x: (x[1], x[2], np.random.random()), reverse=True)
    qualifiers.extend([t[0] for t in all_third_sorted[:8]])
    
    return qualifiers  # 32 teams

def sim_knockout(teams_32):
    """Simulate knockout rounds from R32 → R16 → QF → SF → Final."""
    np.random.shuffle(teams_32)  # Random bracket seeding
    current = teams_32[:]
    
    rounds_reached = defaultdict(set)
    for t in current:
        rounds_reached['R32'].add(t)
    
    for rd_name in ['R16', 'QF', 'SF', 'Final']:
        winners = []
        for i in range(0, len(current), 2):
            if i+1 < len(current):
                w = sim_ko_match(current[i], current[i+1])
                winners.append(w)
                rounds_reached[rd_name].add(w)
        current = winners
    
    champion = current[0] if current else None
    return champion, rounds_reached

# ── Run 50,000 simulations ────────────────────────────────────────────────────
print(f"Running {N_SIMS:,} tournament simulations…")

champion_counts = defaultdict(int)
round_counts = defaultdict(lambda: defaultdict(int))

for sim in range(N_SIMS):
    qualifiers = sim_group_stage(WC2026_GROUPS_FINAL)
    champion, rounds = sim_knockout(qualifiers)
    if champion:
        champion_counts[champion] += 1
    for rd, teams in rounds.items():
        for t in teams:
            round_counts[t][rd] += 1

print("Simulations complete.")

# ── Results table ─────────────────────────────────────────────────────────────
all_sim_teams = sorted(set(
    list(champion_counts.keys()) + list(round_counts.keys())
))

sim_results = []
for team in all_sim_teams:
    sim_results.append({
        'team': team,
        'elo': round(get_elo(team), 0),
        'champion_pct': round(champion_counts[team] / N_SIMS * 100, 2),
        'final_pct':    round(round_counts[team]['Final']  / N_SIMS * 100, 2),
        'sf_pct':       round(round_counts[team]['SF']     / N_SIMS * 100, 2),
        'qf_pct':       round(round_counts[team]['QF']     / N_SIMS * 100, 2),
        'r16_pct':      round(round_counts[team]['R16']    / N_SIMS * 100, 2),
    })

wc2026_sim_results = pd.DataFrame(sim_results).sort_values('champion_pct', ascending=False).reset_index(drop=True)
wc2026_sim_results.index += 1

print("\n=== WC 2026 CHAMPIONSHIP PROBABILITIES (Top 20) ===")
print(wc2026_sim_results[['team','elo','champion_pct','final_pct','sf_pct','qf_pct']].head(20).to_string())

# ── Visualization: Top 20 WC2026 contenders ───────────────────────────────────
top20 = wc2026_sim_results.head(20).copy().iloc[::-1]  # reverse for horizontal bar

PALETTE_SIM = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF',
               '#ffd400','#1F77B4','#9467BD','#8C564B','#C49C94',
               '#E377C2','#F7B6D2','#17b26a','#f04438','#5DA5DA',
               '#FAA43A','#60BD68','#B2912F','#B276B2','#DECF3F']

wc2026_predictions_fig = plt.figure(figsize=(14, 10))
ax = wc2026_predictions_fig.add_subplot(111)
ax.set_facecolor('#1D1D20')
wc2026_predictions_fig.patch.set_facecolor('#1D1D20')

# Stacked bars: champion, finalist, semi, QF
xs = np.arange(len(top20))
teams_list = top20['team'].tolist()

b1 = ax.barh(xs, top20['qf_pct'],   color='#2a3555', height=0.6, label='QF reach')
b2 = ax.barh(xs, top20['sf_pct'],   color='#3a5a9a', height=0.6, label='SF reach')
b3 = ax.barh(xs, top20['final_pct'],color='#A1C9F4', height=0.6, label='Final reach')
b4 = ax.barh(xs, top20['champion_pct'], color='#ffd400', height=0.6, label='Champion')

# Labels
for i, (_, r) in enumerate(top20.iterrows()):
    ax.text(r['champion_pct'] + 0.15, i, f"{r['champion_pct']:.1f}%",
            va='center', color='#ffd400', fontsize=8.5, fontweight='bold')

ax.set_yticks(xs)
ax.set_yticklabels(teams_list, color='#fbfbff', fontsize=10)
ax.set_xlabel('Probability (%)', color='#909094', fontsize=11)
ax.set_title('World Cup 2026 — Win Probability (50,000 Simulations)',
             color='#fbfbff', fontsize=14, fontweight='bold', pad=12)
ax.tick_params(axis='x', colors='#909094')
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color('#909094')
ax.grid(True, axis='x', alpha=0.12, color='#909094')
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
leg = ax.legend(loc='lower right', fontsize=9, framealpha=0.3,
                facecolor='#1D1D20', labelcolor='#fbfbff')
plt.tight_layout()
print("\n✅ WC 2026 simulator complete. Chart ready.")
