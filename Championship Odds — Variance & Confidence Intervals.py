
import numpy as np
import pandas as pd
from collections import defaultdict

# ── Setup ──────────────────────────────────────────────────────────────────────
# We have 50K binary Bernoulli trials per team (champion = 1, else = 0).
# Each team's championship outcome across N_SIMS sims follows Binomial(n, p).
# Variance of a proportion = p*(1-p)/n  →  std dev = sqrt(p*(1-p)/n)
# 95% CI (Wilson score, more accurate for small p) = p ± 1.96 * sqrt(p*(1-p)/n)

N = N_SIMS  # 50,000

# Filter to teams with at least 1 championship (champion_pct > 0)
champ_teams = wc2026_sim_results[wc2026_sim_results['champion_pct'] > 0].copy()
champ_teams = champ_teams.reset_index(drop=True)

# ── Core statistical calculations ─────────────────────────────────────────────
# p = proportion of sims won as champion
champ_teams['p'] = champ_teams['champion_pct'] / 100.0
champ_teams['n_wins'] = (champ_teams['p'] * N).round().astype(int)

# Variance of the binomial proportion estimator: p*(1-p)/N
champ_teams['variance'] = champ_teams['p'] * (1 - champ_teams['p']) / N

# Standard deviation of the estimated proportion
champ_teams['std_dev'] = np.sqrt(champ_teams['variance'])

# 95% CI using Wilson score interval (recommended for proportions, esp. small p)
# z = 1.95996 for 95%
z = 1.95996
n_val = N
p_val = champ_teams['p']

# Wilson score interval:
# centre = (p + z²/2n) / (1 + z²/n)
# half_width = z * sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n)
denom = 1 + z**2 / n_val
centre = (p_val + z**2 / (2 * n_val)) / denom
half_width = (z * np.sqrt(p_val * (1 - p_val) / n_val + z**2 / (4 * n_val**2))) / denom

champ_teams['ci_lower_pct'] = ((centre - half_width) * 100).clip(lower=0).round(3)
champ_teams['ci_upper_pct'] = ((centre + half_width) * 100).round(3)
champ_teams['std_dev_pct']  = (champ_teams['std_dev'] * 100).round(4)
champ_teams['variance_pct'] = (champ_teams['variance'] * 10000).round(6)  # in pct² units

# ── Volatility tiers ───────────────────────────────────────────────────────────
# Volatility = relative uncertainty = std_dev / mean  (coefficient of variation)
# Higher CV → less "pin-down-able" by the model.
# Tier thresholds calibrated to WC simulation context:
#   - Stable:   CV < 0.15  (model is confident)
#   - Moderate: CV 0.15–0.30
#   - Volatile: CV > 0.30  (high model uncertainty)
champ_teams['cv'] = champ_teams['std_dev_pct'] / champ_teams['champion_pct']

def volatility_label(cv):
    if cv < 0.15:
        return '🟢 Stable'
    elif cv < 0.30:
        return '🟡 Moderate'
    else:
        return '🔴 Volatile'

champ_teams['volatility'] = champ_teams['cv'].apply(volatility_label)

# Sort by variance descending (most uncertain first)
variance_ranked = champ_teams.sort_values('variance_pct', ascending=False).reset_index(drop=True)
variance_ranked.index += 1

# ── Rename for display ─────────────────────────────────────────────────────────
display_cols = ['team', 'champion_pct', 'std_dev_pct', 'ci_lower_pct', 'ci_upper_pct', 'volatility']
display_df = variance_ranked[display_cols].copy()
display_df.columns = ['Team', 'Mean Champ %', 'Std Dev %', 'CI Lower %', 'CI Upper %', 'Volatility']

# Also rank by mean for reference
mean_ranked = champ_teams.sort_values('p', ascending=False).reset_index(drop=True)
mean_ranked.index += 1

# ── Print: Ranked by Variance (Most Uncertain First) ──────────────────────────
print("=" * 85)
print("  WC 2026 — CHAMPIONSHIP ODDS VARIANCE ANALYSIS  (50,000 Monte Carlo Simulations)")
print("  95% Confidence Intervals via Wilson Score Interval")
print("=" * 85)
print(f"\n{'Rank':<5} {'Team':<20} {'Mean %':>8} {'Std Dev':>8} {'CI Lower':>9} {'CI Upper':>9} {'Volatility':<15} {'CV':>6}")
print("-" * 85)

for idx, row in variance_ranked.iterrows():
    print(f"{idx:<5} {row['team']:<20} {row['champion_pct']:>7.2f}% {row['std_dev_pct']:>7.4f}% "
          f"{row['ci_lower_pct']:>8.3f}% {row['ci_upper_pct']:>8.3f}% "
          f"{row['volatility']:<18} {row['cv']:>6.3f}")

# ── Summary statistics ─────────────────────────────────────────────────────────
stable_n   = (variance_ranked['volatility'] == '🟢 Stable').sum()
moderate_n = (variance_ranked['volatility'] == '🟡 Moderate').sum()
volatile_n = (variance_ranked['volatility'] == '🔴 Volatile').sum()

print("\n" + "=" * 85)
print("  VOLATILITY TIER SUMMARY")
print("=" * 85)
print(f"  🟢 Stable   (CV < 0.15):  {stable_n:>2} teams  — Model is highly confident in these odds")
print(f"  🟡 Moderate (CV 0.15–0.30):{moderate_n:>2} teams  — Reasonable model certainty")
print(f"  🔴 Volatile (CV > 0.30):  {volatile_n:>2} teams  — High model uncertainty / long-shot teams")

# ── Top 5 most uncertain ────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("  TOP 5 MOST VOLATILE (Highest Std Dev) — Most Uncertain Championship Odds")
print("=" * 85)
top5_vol = variance_ranked.head(5)
for idx, row in top5_vol.iterrows():
    ci_width = row['ci_upper_pct'] - row['ci_lower_pct']
    print(f"  {idx}. {row['team']:<18} Mean: {row['champion_pct']:>5.2f}%  "
          f"Std: {row['std_dev_pct']:.4f}%  CI: [{row['ci_lower_pct']:.3f}% – {row['ci_upper_pct']:.3f}%]  "
          f"CI width: {ci_width:.3f}pp")

# ── Top 5 most stable ───────────────────────────────────────────────────────────
print("\n" + "=" * 85)
print("  TOP 5 MOST STABLE — Most Confident Championship Odds")
print("=" * 85)
top5_stable = variance_ranked.tail(5).iloc[::-1]
for idx, row in top5_stable.iterrows():
    ci_width = row['ci_upper_pct'] - row['ci_lower_pct']
    print(f"  {idx}. {row['team']:<18} Mean: {row['champion_pct']:>5.2f}%  "
          f"Std: {row['std_dev_pct']:.4f}%  CI: [{row['ci_lower_pct']:.3f}% – {row['ci_upper_pct']:.3f}%]  "
          f"CI width: {ci_width:.3f}pp")

print("\n" + "=" * 85)
print(f"  Total teams with ≥1 championship in {N:,} sims: {len(champ_teams)}")
print(f"  Narrowest CI (most precise): {variance_ranked.iloc[-1]['team']} "
      f"({variance_ranked.iloc[-1]['ci_upper_pct'] - variance_ranked.iloc[-1]['ci_lower_pct']:.3f}pp wide)")
print(f"  Widest CI (least precise):   {variance_ranked.iloc[0]['team']} "
      f"({variance_ranked.iloc[0]['ci_upper_pct'] - variance_ranked.iloc[0]['ci_lower_pct']:.3f}pp wide)")
print("=" * 85)

# Store for downstream use
variance_stats = variance_ranked.copy()
