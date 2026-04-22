
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

BG = '#1D1D20'
FG = '#fbfbff'
SEC = '#909094'
PALETTE = ['#A1C9F4','#FFB482','#8DE5A1','#FF9F9B','#D0BBFF',
           '#ffd400','#1F77B4','#9467BD','#8C564B','#C49C94',
           '#E377C2','#F7B6D2','#17b26a','#f04438']

# ═══════════════════════════════════════════════════════════════════════
# INSIGHT 1: The ELO Curse — Favourite Underperformance at World Cups
# ═══════════════════════════════════════════════════════════════════════
# Compare actual win rate vs ELO-expected win rate in WC vs friendly/competitive

def elo_expected_win(elo_diff):
    """Pure ELO expected win probability (no draw)."""
    return 1 / (1 + 10 ** (-elo_diff / 400))

# Build a dataset with ELO at match time (from pre_match_elo_df)
all_matches_with_elo = international_results.merge(
    pre_match_elo_df[['date','home_team','away_team','home_elo_pre','away_elo_pre','elo_diff']],
    on=['date','home_team','away_team'], how='inner'
).dropna(subset=['home_elo_pre','away_elo_pre','home_result'])

all_matches_with_elo['expected_home_win'] = elo_expected_win(all_matches_with_elo['elo_diff'])
all_matches_with_elo['actual_home_win'] = (all_matches_with_elo['home_result'] == 'W').astype(float)
all_matches_with_elo['upset'] = all_matches_with_elo['actual_home_win'] < all_matches_with_elo['expected_home_win']

# Bin by ELO difference
all_matches_with_elo['elo_bin'] = pd.cut(all_matches_with_elo['elo_diff'],
    bins=[-600,-300,-150,-75,0,75,150,300,600],
    labels=['Strong Underdog','Underdog','Slight Underdog','Even',
            'Slight Fav','Favourite','Strong Fav','Dominant'])

# Compare WC vs non-WC
wc_perf = all_matches_with_elo.groupby(['elo_bin','is_world_cup']).agg(
    actual=('actual_home_win','mean'),
    expected=('expected_home_win','mean'),
    n=('actual_home_win','count')
).reset_index()
wc_perf['overperf'] = wc_perf['actual'] - wc_perf['expected']

# Only look at 'favourite' side (positive ELO diff)
fav_bins = ['Slight Fav','Favourite','Strong Fav','Dominant']
wc_fav = wc_perf[wc_perf['elo_bin'].isin(fav_bins)].copy()

elo_curse_fig = plt.figure(figsize=(12, 6))
ax = elo_curse_fig.add_subplot(111)
ax.set_facecolor(BG); elo_curse_fig.patch.set_facecolor(BG)

x = np.arange(len(fav_bins))
w = 0.35
wc_over = wc_fav[wc_fav['is_world_cup']==True].set_index('elo_bin')['overperf'].reindex(fav_bins).fillna(0)
non_wc_over = wc_fav[wc_fav['is_world_cup']==False].set_index('elo_bin')['overperf'].reindex(fav_bins).fillna(0)

ax.bar(x - w/2, non_wc_over.values * 100, width=w, color='#A1C9F4', label='All Matches', alpha=0.85)
ax.bar(x + w/2, wc_over.values * 100, width=w, color='#f04438', label='World Cup Only', alpha=0.85)
ax.axhline(0, color=SEC, linewidth=1, linestyle='--', alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(fav_bins, color=FG, fontsize=10)
ax.set_ylabel('Overperformance vs ELO Expectation (%)', color=SEC, fontsize=10)
ax.set_title("The ELO Curse: Favourites Underperform at World Cups\n(Actual win rate minus ELO-expected win rate, by favourite strength)",
             color=FG, fontsize=12, fontweight='bold', pad=10)
ax.tick_params(axis='y', colors=SEC)
ax.spines[['top','right']].set_visible(False)
ax.spines[['bottom','left']].set_color(SEC)
ax.grid(True, axis='y', alpha=0.12, color=SEC)
ax.legend(fontsize=10, framealpha=0.3, facecolor=BG, labelcolor=FG)
plt.tight_layout()

print("=== INSIGHT 1: ELO CURSE ===")
print(wc_fav[['elo_bin','is_world_cup','actual','expected','overperf','n']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════
# INSIGHT 2: Upset Index — Historical Upset Frequency by ELO Gap
# ═══════════════════════════════════════════════════════════════════════
wc_only_elo = all_matches_with_elo[all_matches_with_elo['is_world_cup']].copy()
wc_only_elo['abs_elo_diff'] = wc_only_elo['elo_diff'].abs()
wc_only_elo['is_underdog_side'] = wc_only_elo['elo_diff'] > 0  # home is fav

# For each match, did the underdog win or draw?
wc_only_elo['upset_result'] = np.where(
    wc_only_elo['elo_diff'] > 0,
    (wc_only_elo['actual_home_win'] < 0.5).astype(float),   # home is fav, upset = non-win
    (wc_only_elo['actual_home_win'] > 0.5).astype(float)    # away is fav, upset = home wins
)

wc_only_elo['elo_gap_bin'] = pd.cut(wc_only_elo['abs_elo_diff'],
    bins=[0,50,100,150,200,300,400,1000],
    labels=['0-50','50-100','100-150','150-200','200-300','300-400','400+'])

upset_by_gap = wc_only_elo.groupby('elo_gap_bin').agg(
    upset_rate=('upset_result','mean'),
    n=('upset_result','count'),
    elo_expected_fav=('expected_home_win','mean')
).reset_index()

upset_fig = plt.figure(figsize=(12, 6))
ax2 = upset_fig.add_subplot(111)
ax2.set_facecolor(BG); upset_fig.patch.set_facecolor(BG)

_bins = upset_by_gap['elo_gap_bin'].astype(str).tolist()
_x = np.arange(len(_bins))

bars = ax2.bar(_x, upset_by_gap['upset_rate']*100, color='#FFB482', alpha=0.9, width=0.5, label='Actual Upset Rate')
ax2.plot(_x, (1 - upset_by_gap['elo_expected_fav'])*100, 'o--',
         color='#ffd400', linewidth=2, markersize=7, label='ELO-Expected Underdog Win%')

for i, (bar, n) in enumerate(zip(bars, upset_by_gap['n'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'n={n}', ha='center', va='bottom', color=SEC, fontsize=8)

ax2.set_xticks(_x)
ax2.set_xticklabels(_bins, color=FG, fontsize=10)
ax2.set_xlabel('ELO Gap (Favourite − Underdog)', color=SEC, fontsize=10)
ax2.set_ylabel('Upset Rate (%)', color=SEC, fontsize=10)
ax2.set_title("Upset Index: How Often Do Underdogs Win at the World Cup?\n(Grouped by ELO gap — yellow = what ELO predicts, orange = reality)",
              color=FG, fontsize=12, fontweight='bold', pad=10)
ax2.tick_params(axis='y', colors=SEC)
ax2.spines[['top','right']].set_visible(False)
ax2.spines[['bottom','left']].set_color(SEC)
ax2.grid(True, axis='y', alpha=0.12, color=SEC)
ax2.legend(fontsize=10, framealpha=0.3, facecolor=BG, labelcolor=FG)
plt.tight_layout()

print("\n=== INSIGHT 2: UPSET INDEX ===")
print(upset_by_gap[['elo_gap_bin','upset_rate','n','elo_expected_fav']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════
# INSIGHT 3: Groups of Death 2026 — ELO Strength Heatmap
# ═══════════════════════════════════════════════════════════════════════
group_strength = []
for grp, teams in WC2026_GROUPS_FINAL.items():
    elos = [get_elo(t) for t in teams]
    avg_elo = np.mean(elos)
    top2_elo = np.mean(sorted(elos, reverse=True)[:2])
    elo_range = max(elos) - min(elos)
    group_strength.append({
        'group': grp,
        'teams': ', '.join(teams),
        'avg_elo': round(avg_elo, 0),
        'top2_avg_elo': round(top2_elo, 0),
        'elo_range': round(elo_range, 0),
        'competition_index': round(avg_elo * (1 - elo_range/2000), 0),
    })

group_df = pd.DataFrame(group_strength).sort_values('avg_elo', ascending=False)

groups_fig = plt.figure(figsize=(14, 7))
ax3 = groups_fig.add_subplot(111)
ax3.set_facecolor(BG); groups_fig.patch.set_facecolor(BG)

grp_labels = group_df['group'].tolist()
_gx = np.arange(len(grp_labels))

# Color intensity by avg ELO
elo_norm = (group_df['avg_elo'] - group_df['avg_elo'].min()) / (group_df['avg_elo'].max() - group_df['avg_elo'].min())
bar_colors = plt.cm.YlOrRd(elo_norm.values * 0.8 + 0.2)

bars3 = ax3.bar(_gx, group_df['avg_elo'], color=bar_colors, width=0.6, alpha=0.9)

# Label each group with team names
for i, (bar, row) in enumerate(zip(bars3, group_df.itertuples())):
    teams_short = '\n'.join(row.teams.split(', '))
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
             f'{row.avg_elo:.0f}', ha='center', va='bottom', color='#ffd400',
             fontsize=9, fontweight='bold')
    ax3.text(bar.get_x() + bar.get_width()/2, 1200,
             teams_short, ha='center', va='bottom', color=FG, fontsize=6.5, alpha=0.9)

ax3.set_xticks(_gx)
ax3.set_xticklabels([f"Group {g}" for g in grp_labels], color=FG, fontsize=10)
ax3.set_ylabel('Average ELO Rating', color=SEC, fontsize=10)
ax3.set_ylim(1150, 2200)
ax3.set_title("Groups of Death — 2026 World Cup Group Strength by Average ELO\n(Higher = tougher group; labels show avg ELO)",
              color=FG, fontsize=12, fontweight='bold', pad=10)
ax3.tick_params(axis='y', colors=SEC)
ax3.spines[['top','right']].set_visible(False)
ax3.spines[['bottom','left']].set_color(SEC)
ax3.grid(True, axis='y', alpha=0.10, color=SEC)
plt.tight_layout()

print("\n=== INSIGHT 3: GROUPS OF DEATH ===")
print(group_df[['group','teams','avg_elo','elo_range']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════
# INSIGHT 4: Brazil's Paradox — Most WC Wins, But Falling ELO
# WC Titles vs Current ELO — the "glory gap"
# ═══════════════════════════════════════════════════════════════════════
wc_winners = {
    'Brazil': 5, 'Germany': 4, 'Italy': 4, 'Argentina': 3,
    'France': 2, 'Uruguay': 2, 'England': 1, 'Spain': 1,
    'Netherlands': 0, 'Portugal': 0, 'Belgium': 0, 'Croatia': 0,
    'Morocco': 0, 'Norway': 0, 'Colombia': 0, 'Ecuador': 0,
}

glory_df = pd.DataFrame([
    {'team': t, 'wc_titles': w, 'current_elo': get_elo(t)}
    for t, w in wc_winners.items()
]).sort_values('wc_titles', ascending=False)

# ELO of teams with titles at their peak vs now
glory_df['elo_rank'] = glory_df['current_elo'].rank(ascending=False).astype(int)
glory_df['glory_gap'] = glory_df['wc_titles'] * 400 - (glory_df['current_elo'] - 1500)

glory_fig = plt.figure(figsize=(13, 7))
ax4 = glory_fig.add_subplot(111)
ax4.set_facecolor(BG); glory_fig.patch.set_facecolor(BG)

# Scatter: titles vs ELO
scatter_colors = [PALETTE[i % len(PALETTE)] for i in range(len(glory_df))]
sc = ax4.scatter(glory_df['wc_titles'], glory_df['current_elo'],
                 s=[300 + e*50 for e in glory_df['wc_titles']],
                 c=scatter_colors, alpha=0.9, zorder=5, edgecolors='white', linewidth=0.5)

for _, row in glory_df.iterrows():
    offset_x = 0.05
    offset_y = 8 if row['current_elo'] < 2100 else -25
    ax4.annotate(row['team'], (row['wc_titles'], row['current_elo']),
                 xytext=(row['wc_titles'] + offset_x, row['current_elo'] + offset_y),
                 color=FG, fontsize=9, ha='left')

ax4.set_xlabel('World Cup Titles', color=SEC, fontsize=11)
ax4.set_ylabel('Current ELO Rating', color=SEC, fontsize=11)
ax4.set_title("The Glory Gap: Historical Titles vs. Current Strength\n(Bubble size = titles; rising powers like Spain & Norway lack titles)",
              color=FG, fontsize=12, fontweight='bold', pad=10)
ax4.tick_params(colors=SEC)
ax4.spines[['top','right']].set_visible(False)
ax4.spines[['bottom','left']].set_color(SEC)
ax4.grid(True, alpha=0.12, color=SEC)
ax4.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax4.yaxis.set_major_formatter(mticker.ScalarFormatter())
plt.tight_layout()

print("\n=== INSIGHT 4: GLORY GAP ===")
print(glory_df[['team','wc_titles','current_elo','elo_rank']].to_string(index=False))
print("\n✅ All 4 deep insights complete.")
