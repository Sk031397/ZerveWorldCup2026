
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Feature matrix ────────────────────────────────────────────────────────────
FEATURES = ['elo_diff', 'home_elo', 'away_elo', 'home_form', 'away_form',
            'form_diff', 'h2h_home_winrate', 'wc_exp_diff',
            'home_wc_exp', 'away_wc_exp']

df = wc_model_df.dropna(subset=FEATURES + ['outcome']).copy()
X = df[FEATURES].values
y = df['outcome'].values  # 0=away win, 1=draw, 2=home win

print(f"Training samples: {len(X)}")

# ── Cross-validation ─────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)

cv_acc  = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
cv_ll   = cross_val_score(model, X, y, cv=skf, scoring='neg_log_loss')

print(f"\n5-Fold Cross-Validation:")
print(f"  Accuracy:  {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
print(f"  Log-loss:  {(-cv_ll).mean():.3f} ± {(-cv_ll).std():.3f}")

# ── Train on full dataset ─────────────────────────────────────────────────────
model.fit(X, y)
train_preds = model.predict(X)
train_proba = model.predict_proba(X)

print(f"\nFull-data Accuracy: {accuracy_score(y, train_preds):.3f}")
print(f"Full-data Log-loss: {log_loss(y, train_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y, train_preds, 
      target_names=['Away Win','Draw','Home Win']))

# ── Baseline: always predict home win ─────────────────────────────────────────
baseline_acc = (y == 2).mean()
print(f"Baseline (always Home Win): {baseline_acc:.3f}")
print(f"Model lift over baseline:   +{(cv_acc.mean() - baseline_acc):.3f}")

# ── Feature importances ───────────────────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)

feature_importance_fig = plt.figure(figsize=(10, 6))
ax = feature_importance_fig.add_subplot(111)
ax.set_facecolor('#1D1D20')
feature_importance_fig.patch.set_facecolor('#1D1D20')

colors = ['#A1C9F4' if v < feat_imp.max()*0.7 else '#ffd400' for v in feat_imp.values]
bars = ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor='none', height=0.6)

ax.set_title('Feature Importance — WC Match Outcome Predictor',
             color='#fbfbff', fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Importance', color='#909094', fontsize=10)
ax.tick_params(colors='#fbfbff', labelsize=10)
ax.spines[['top','right','bottom','left']].set_color('#909094')
ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax.grid(True, axis='x', alpha=0.15, color='#909094')
plt.tight_layout()

# ── predict_match() function ──────────────────────────────────────────────────
def predict_match(team1, team2, current_elo=None, current_form=None):
    """
    Predict World Cup match between team1 (home/team1) and team2.
    Returns dict: {team1_win, draw, team2_win} probabilities.
    Uses current ELO ratings from elo_table.
    """
    _elo_t = current_elo or elo_table
    
    t1_row = _elo_t[_elo_t['team'] == team1]
    t2_row = _elo_t[_elo_t['team'] == team2]
    
    t1_elo = float(t1_row['elo'].values[0]) if len(t1_row) > 0 else 1500.0
    t2_elo = float(t2_row['elo'].values[0]) if len(t2_row) > 0 else 1500.0
    
    elo_d = t1_elo - t2_elo
    
    # Use average WC-era form features
    t1_form = current_form.get(team1, 0.6) if current_form else 0.6
    t2_form = current_form.get(team2, 0.6) if current_form else 0.6
    
    t1_wc_exp = wc_appearances_before(team1, 2030)  # proxy for "current"
    t2_wc_exp = wc_appearances_before(team2, 2030)
    
    features_vec = np.array([[
        elo_d, t1_elo, t2_elo,
        t1_form, t2_form,
        t1_form - t2_form,
        0.5,  # neutral H2H prior
        t1_wc_exp - t2_wc_exp,
        t1_wc_exp, t2_wc_exp,
    ]])
    
    proba = model.predict_proba(features_vec)[0]
    label_map = {0: 'team2_win', 1: 'draw', 2: 'team1_win'}
    classes = model.classes_
    result = {label_map[c]: round(float(p), 4) for c, p in zip(classes, proba)}
    result['team1'] = team1
    result['team2'] = team2
    result['team1_elo'] = round(t1_elo, 1)
    result['team2_elo'] = round(t2_elo, 1)
    return result

# Quick sanity checks
tests = [('Brazil', 'Argentina'), ('Spain', 'England'), ('France', 'Morocco')]
print("\n=== Prediction Sanity Checks ===")
for t1, t2 in tests:
    r = predict_match(t1, t2)
    print(f"  {t1} vs {t2}: {t1}={r['team1_win']:.0%}  D={r['draw']:.0%}  {t2}={r['team2_win']:.0%}  (ELO: {r['team1_elo']} vs {r['team2_elo']})")

wc_prediction_model = model
print("\n✅ Model trained and predict_match() ready.")
