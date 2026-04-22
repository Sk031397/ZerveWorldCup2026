
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from itertools import combinations

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════════════
BG_BVI = '#1D1D20'
FG_BVI = '#fbfbff'
SEC_BVI = '#909094'
MAX_ENTROPY_3 = np.log2(3)  # 1.58496 bits — theoretical max for 3-outcome event

# Confederation mapping for WC2026 teams
CONFEDERATION_MAP = {
    'Mexico': 'CONCACAF', 'Poland': 'UEFA', 'Switzerland': 'UEFA', 'Honduras': 'CONCACAF',
    'Argentina': 'CONMEBOL', 'Morocco': 'CAF', 'Ukraine': 'UEFA', 'Saudi Arabia': 'AFC',
    'United States': 'CONCACAF', 'Germany': 'UEFA', 'Japan': 'AFC', 'Jamaica': 'CONCACAF',
    'France': 'UEFA', 'Brazil': 'CONMEBOL', 'Ecuador': 'CONMEBOL', 'Senegal': 'CAF',
    'Spain': 'UEFA', 'England': 'UEFA', 'Croatia': 'UEFA', 'Colombia': 'CONMEBOL',
    'Portugal': 'UEFA', 'Netherlands': 'UEFA', 'Turkey': 'UEFA', 'Paraguay': 'CONMEBOL',
    'Belgium': 'UEFA', 'Norway': 'UEFA', 'Denmark': 'UEFA', 'Australia': 'AFC',
    'Uruguay': 'CONMEBOL', 'South Korea': 'AFC', 'Canada': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Nigeria': 'CAF', 'South Africa': 'CAF', 'Algeria': 'CAF', 'Chile': 'CONMEBOL',
    'Panama': 'CONCACAF', 'Egypt': 'CAF', 'Ghana': 'CAF', 'Ivory Coast': 'CAF',
    'Iran': 'AFC', 'Tunisia': 'CAF', 'Serbia': 'UEFA', 'Venezuela': 'CONMEBOL',
    'Qatar': 'AFC', 'Bolivia': 'CONMEBOL', 'Kuwait': 'AFC', 'New Zealand': 'OFC',
}

# ══════════════════════════════════════════════════════════════════════
# STEP 1: ELO-based 3-outcome probabilities (same model as simulator)
# ══════════════════════════════════════════════════════════════════════
def elo_3way_probs(team1_elo, team2_elo, p_draw_base=0.22):
    """
    Compute win/draw/loss probabilities from ELO ratings.
    Uses the established p_draw=0.22 WC calibration from the simulator.
    Returns (p_win1, p_draw, p_win2).
    """
    elo_d = team1_elo - team2_elo
    exp1 = 1 / (1 + 10 ** (-elo_d / 400))  # ELO expected win (no draw)
    p_win1 = exp1 * (1 - p_draw_base)
    p_win2 = (1 - exp1) * (1 - p_draw_base)
    return p_win1, p_draw_base, p_win2


def shannon_entropy_3(p1, p2, p3):
    """Shannon entropy in bits for a 3-outcome distribution. Max = log2(3) ≈ 1.585."""
    probs = np.array([p1, p2, p3])
    probs = probs[probs > 0]  # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


def kelly_edge(p_win, decimal_odds=None):
    """
    Kelly Criterion implied edge assuming fair-odds market.
    If market odds were perfectly fair (odds = 1/p), Kelly fraction = 0.
    Here we compute edge = p_win - (1 / implied_fair_odds) to show theoretical edge
    if backing the favourite vs. fair market.
    
    For simplicity: edge = p1_win - p2_win (net directional edge of the stronger side).
    Kelly fraction (normalized) = edge / (1/p_win) ≈ p_win - (1-p_win) when b=1
    This represents: how much of your bankroll Kelly says to bet if the market
    priced the match fairly. Near 0 = near-zero edge = highest uncertainty/value.
    """
    # Under fair odds: back team1, Kelly f* = (b*p - q) / b where b = (1/p)-1, q=1-p
    b = (1 / p_win) - 1 if p_win < 1 else 0
    kelly_f = (b * p_win - (1 - p_win)) / b if b > 0 else 0
    return float(kelly_f)


def upset_potential(elo_diff_abs, upset_by_gap_df):
    """
    Look up historical WC upset rate from pre-computed upset_by_gap data.
    Returns the historical base rate for the ELO gap band.
    """
    bins = [0, 50, 100, 150, 200, 300, 400, np.inf]
    labels = ['0-50', '50-100', '100-150', '150-200', '200-300', '300-400', '400+']
    for i in range(len(bins)-1):
        if bins[i] <= elo_diff_abs < bins[i+1]:
            band = labels[i]
            row = upset_by_gap_df[upset_by_gap_df['elo_gap_bin'].astype(str) == band]
            if len(row):
                return float(row['upset_rate'].values[0])
    return 0.30  # default fallback


# ══════════════════════════════════════════════════════════════════════
# STEP 2: Build historical volatility index per ELO band
# (std dev of outcome entropy across historical WC matches in that band)
# ══════════════════════════════════════════════════════════════════════
# Compute entropy for each historical WC match (using pre-match ELOs)
wc_hist = wc_only_elo[['home_elo_pre', 'away_elo_pre', 'elo_diff',
                         'abs_elo_diff', 'elo_gap_bin', 'upset_result']].dropna().copy()

wc_hist['h_p1'], wc_hist['h_pd'], wc_hist['h_p2'] = zip(*wc_hist.apply(
    lambda r: elo_3way_probs(r['home_elo_pre'], r['away_elo_pre']), axis=1
))
wc_hist['hist_entropy'] = wc_hist.apply(
    lambda r: shannon_entropy_3(r['h_p1'], r['h_pd'], r['h_p2']), axis=1
)

# Volatility = std dev of outcome entropy within each ELO gap band
volatility_by_band = wc_hist.groupby('elo_gap_bin', observed=True)['hist_entropy'].agg(
    hist_vol_mean='mean', hist_vol_std='std', hist_n='count'
).reset_index()
volatility_by_band['hist_vol_std'] = volatility_by_band['hist_vol_std'].fillna(0)

print("Historical entropy by ELO gap band (WC matches):")
print(volatility_by_band.to_string(index=False))
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Compute BVI for all 48×47/2 = 1,128 unique group-stage matchups
# (All possible intra-group matchups: 12 groups × C(4,2) = 6 matches = 72 matches total)
# Plus ALL cross-group potential matchups = 1,128 total unique pairs
# ══════════════════════════════════════════════════════════════════════
bvi_rows = []

for team1, team2 in combinations(ALL_48, 2):
    e1 = elo_lookup.get(team1, 1500.0)
    e2 = elo_lookup.get(team2, 1500.0)
    abs_diff = abs(e1 - e2)

    # 1. ELO-based 3-way probabilities
    p_w1, p_d, p_w2 = elo_3way_probs(e1, e2)

    # 2. Shannon entropy (bits) — max = 1.585 for 3 outcomes
    entropy = shannon_entropy_3(p_w1, p_d, p_w2)

    # 3. Kelly implied edge (using stronger team's win prob)
    stronger_p = max(p_w1, p_w2)
    kelly_fraction = kelly_edge(stronger_p)

    # 4. ELO differential
    elo_diff_val = abs_diff

    # 5. Historical upset rate for this ELO band
    hist_upset_rate = upset_potential(abs_diff, upset_by_gap)

    # 6. Historical volatility bonus (how unpredictable is this ELO band historically?)
    bins = [0, 50, 100, 150, 200, 300, 400, np.inf]
    labels_b = ['0-50', '50-100', '100-150', '150-200', '200-300', '300-400', '400+']
    band_label = '0-50'
    for bi in range(len(bins)-1):
        if bins[bi] <= abs_diff < bins[bi+1]:
            band_label = labels_b[bi]
            break
    vol_row = volatility_by_band[volatility_by_band['elo_gap_bin'].astype(str) == band_label]
    hist_vol = float(vol_row['hist_vol_std'].values[0]) if len(vol_row) else 0.0

    # 7. Betting Value Index (BVI):
    # BVI = 0.45 × (entropy / MAX_ENTROPY) 
    #     + 0.25 × (1 - |elo_diff| / 600) clipped to [0,1]   ← ELO closeness
    #     + 0.20 × hist_upset_rate                            ← historical upset volatility
    #     + 0.10 × hist_vol                                   ← intra-band entropy variance
    elo_closeness = max(0, 1 - abs_diff / 600)
    bvi = (0.45 * (entropy / MAX_ENTROPY_3)
         + 0.25 * elo_closeness
         + 0.20 * hist_upset_rate
         + 0.10 * hist_vol)

    # Confederation tags
    conf1 = CONFEDERATION_MAP.get(team1, 'Other')
    conf2 = CONFEDERATION_MAP.get(team2, 'Other')
    matchup_type = 'Same Conf' if conf1 == conf2 else 'Cross Conf'

    # Is this a group-stage actual match?
    is_group_match = False
    team1_group = None
    team2_group = None
    for grp, gteams in WC2026_GROUPS_FINAL.items():
        if team1 in gteams:
            team1_group = grp
        if team2 in gteams:
            team2_group = grp
    if team1_group and team1_group == team2_group:
        is_group_match = True

    bvi_rows.append({
        'team1': team1,
        'team2': team2,
        'team1_elo': round(e1, 1),
        'team2_elo': round(e2, 1),
        'elo_diff': round(elo_diff_val, 1),
        'p_team1_win': round(p_w1, 4),
        'p_draw': round(p_d, 4),
        'p_team2_win': round(p_w2, 4),
        'entropy_bits': round(entropy, 4),
        'entropy_pct_max': round(entropy / MAX_ENTROPY_3 * 100, 1),
        'kelly_fraction': round(kelly_fraction, 4),
        'hist_upset_rate': round(hist_upset_rate, 4),
        'elo_closeness': round(elo_closeness, 4),
        'hist_vol': round(hist_vol, 4),
        'bvi': round(bvi, 4),
        'conf1': conf1,
        'conf2': conf2,
        'matchup_type': matchup_type,
        'is_group_match': is_group_match,
        'elo_band': band_label,
    })

betting_value_df = pd.DataFrame(bvi_rows).sort_values('bvi', ascending=False).reset_index(drop=True)
betting_value_df.index += 1

print(f"✅ BVI computed for {len(betting_value_df):,} unique WC 2026 matchup pairs")
print(f"   Shannon entropy max: {MAX_ENTROPY_3:.4f} bits")
print(f"   BVI range: {betting_value_df['bvi'].min():.4f} – {betting_value_df['bvi'].max():.4f}")
print()

# ══════════════════════════════════════════════════════════════════════
# STEP 4: TOP 20 HIGHEST-BVI MATCHES
# ══════════════════════════════════════════════════════════════════════
top20_bvi = betting_value_df.head(20).copy()

print("=" * 100)
print("TOP 20 HIGHEST BETTING VALUE MATCHES — WC 2026 (Ranked by BVI)")
print("=" * 100)
print(f"{'Rank':<5} {'Match':<35} {'ELO1':>6} {'ELO2':>6} {'ΔELO':>6} "
      f"{'P(W1)':>7} {'P(D)':>7} {'P(W2)':>7} {'H(bits)':>8} {'H%max':>7} "
      f"{'Kelly':>7} {'UpstRt':>7} {'BVI':>7} {'Type':<12} {'Group?'}")
print("-" * 100)
for _, row in top20_bvi.iterrows():
    match_str = f"{row['team1']} vs {row['team2']}"
    group_tag = "✓ GRP" if row['is_group_match'] else ""
    print(f"{row.name:<5} {match_str:<35} {row['team1_elo']:>6.0f} {row['team2_elo']:>6.0f} "
          f"{row['elo_diff']:>6.0f} {row['p_team1_win']:>7.3f} {row['p_draw']:>7.3f} "
          f"{row['p_team2_win']:>7.3f} {row['entropy_bits']:>8.4f} {row['entropy_pct_max']:>7.1f}% "
          f"{row['kelly_fraction']:>7.4f} {row['hist_upset_rate']:>7.3f} {row['bvi']:>7.4f} "
          f"{row['matchup_type']:<12} {group_tag}")

# ══════════════════════════════════════════════════════════════════════
# STEP 5: GOLDEN ZONE — entropy > 1.4 bits (near-maximum uncertainty)
# ══════════════════════════════════════════════════════════════════════
golden_zone_df = betting_value_df[betting_value_df['entropy_bits'] > 1.4].copy()
golden_zone_group = golden_zone_df[golden_zone_df['is_group_match']].copy()

print(f"\n{'='*70}")
print(f"🏆 THE GOLDEN ZONE — Matches with Entropy > 1.4 bits (Near-Max Uncertainty)")
print(f"{'='*70}")
print(f"Total matchups in golden zone (all): {len(golden_zone_df)}")
print(f"Actual group-stage matches in golden zone: {len(golden_zone_group)}")
print(f"\nGroup-stage golden zone matches (ranked by BVI):")
print("-" * 80)
for _, row in golden_zone_group.sort_values('bvi', ascending=False).iterrows():
    print(f"  Group {row.name if row['is_group_match'] else '?'} | "
          f"{row['team1']} vs {row['team2']} | "
          f"ELO: {row['team1_elo']:.0f} vs {row['team2_elo']:.0f} | "
          f"Entropy: {row['entropy_bits']:.4f} bits ({row['entropy_pct_max']:.1f}% of max) | "
          f"BVI: {row['bvi']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# STEP 6: HISTORICAL UPSET BASE RATES BY ELO BAND
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("📊 HISTORICAL WC UPSET BASE RATES BY ELO BAND")
print(f"{'='*70}")
print(f"{'ELO Gap Band':<15} {'Upset Rate':>12} {'ELO-Expected':>14} {'Excess Upset':>14} {'Matches':>8}")
print("-" * 65)
for _, row in upset_by_gap.iterrows():
    excess = row['upset_rate'] - (1 - row['elo_expected_fav'])
    print(f"{str(row['elo_gap_bin']):<15} {row['upset_rate']*100:>11.1f}% "
          f"{(1-row['elo_expected_fav'])*100:>13.1f}% "
          f"{excess*100:>+13.1f}% {int(row['n']):>8}")

# ══════════════════════════════════════════════════════════════════════
# STEP 7: CONFEDERATION & MATCHUP TYPE ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("⚽ CONFEDERATION MATCHUP UNPREDICTABILITY SUMMARY")
print(f"{'='*70}")

# By matchup confederation pair
conf_analysis = betting_value_df.groupby(['conf1','conf2']).agg(
    avg_bvi=('bvi','mean'),
    avg_entropy=('entropy_bits','mean'),
    avg_upset_rate=('hist_upset_rate','mean'),
    n_matchups=('bvi','count')
).reset_index().sort_values('avg_bvi', ascending=False)

# Only show cross-conf or meaningful same-conf rows
print(f"\nTop Confederation Matchup Types by Average BVI:")
print(f"{'Conf1':<12} {'Conf2':<12} {'Avg BVI':>9} {'Avg Entropy':>12} {'Avg Upset%':>11} {'N':>6}")
print("-" * 65)
for _, row in conf_analysis.head(15).iterrows():
    print(f"{row['conf1']:<12} {row['conf2']:<12} {row['avg_bvi']:>9.4f} "
          f"{row['avg_entropy']:>12.4f} {row['avg_upset_rate']*100:>10.1f}% {int(row['n_matchups']):>6}")

# By matchup type (Same vs Cross confederation)
print(f"\nSame-Conf vs Cross-Conf Unpredictability:")
mc_analysis = betting_value_df.groupby('matchup_type').agg(
    avg_bvi=('bvi','mean'),
    avg_entropy=('entropy_bits','mean'),
    max_bvi=('bvi','max'),
    n=('bvi','count')
).reset_index()
print(mc_analysis.to_string(index=False))

# ELO band unpredictability
print(f"\nAverage BVI and Entropy by ELO Gap Band:")
band_summary = betting_value_df.groupby('elo_band').agg(
    avg_bvi=('bvi','mean'),
    avg_entropy=('entropy_bits','mean'),
    avg_kelly=('kelly_fraction','mean'),
    n=('bvi','count')
).reset_index()
# Order bands correctly
band_order = ['0-50','50-100','100-150','150-200','200-300','300-400','400+']
band_summary['elo_band'] = pd.Categorical(band_summary['elo_band'], categories=band_order, ordered=True)
band_summary = band_summary.sort_values('elo_band')
print(f"{'ELO Band':<12} {'Avg BVI':>9} {'Avg Entropy':>12} {'Avg Kelly':>11} {'N':>6}")
print("-" * 50)
for _, row in band_summary.iterrows():
    print(f"{str(row['elo_band']):<12} {row['avg_bvi']:>9.4f} {row['avg_entropy']:>12.4f} "
          f"{row['avg_kelly']:>11.4f} {int(row['n']):>6}")

print(f"\n✅ Betting Value Index analysis complete — {len(betting_value_df):,} matchups scored.")
