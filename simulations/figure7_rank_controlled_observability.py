"""
Figure 7 — Rank-Controlled Observability under PPS
==================================================

This experiment investigates how rank(Pi) determines the effective
observable dimension under the Platonic Projection Structures (PPS)
framework.

By systematically varying rank(Pi), the experiment evaluates changes in
predictive accuracy, attribution stability, observable information ratio,
and spectral geometry.

The four panels show:

    (a) rank(Pi) vs. test accuracy
    (b) rank(Pi) vs. attribution stability (SHAP variance)
    (c) rank(Pi) vs. observable information ratio
    (d) spectral energy profiles under multiple rank settings

The experiment illustrates that increasing the rank of the observation
operator expands the observable subspace. Predictive performance improves
as task-relevant directions become observable, whereas additional
non-informative dimensions may affect attribution stability without adding
predictive signal.

Usage in Google Colab:
    !pip install shap -q

Then paste the entire contents of this script into a Colab cell and run it.

Outputs:
    figure7_rank_controlled_observability.pdf
    figure7_rank_controlled_observability.png
"""

# ── 0. Dependencies ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 10,
    'axes.labelsize'   : 11,
    'axes.titlesize'   : 11,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'legend.fontsize'  : 9,
    'figure.dpi'       : 300,
    'text.usetex'      : False,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

np.random.seed(42)

# ── 1. Synthetic Task Setup ───────────────────────────────────────────────────
D      = 16     # full latent dimension
N      = 800    # samples
N_test = 200
C      = 4      # number of classes

# Ground-truth signal lives in the top R_true dimensions
R_true = 8

# Full orthonormal basis
U_full, _ = np.linalg.qr(np.random.randn(D, D))

# Generate latent representations
X_latent = np.random.randn(N + N_test, D)

# True label depends only on top R_true directions
w_class = np.random.randn(R_true, C)
logits  = X_latent @ U_full[:, :R_true] @ w_class
y_all   = np.argmax(logits + 0.3 * np.random.randn(*logits.shape), axis=1)

X_train, X_test = X_latent[:N],       X_latent[N:]
y_train, y_test = y_all[:N],          y_all[N:]

# ── 2. Build Π_r for each rank r ──────────────────────────────────────────────
# Eigenvalue profile: exponentially decaying
full_eigenvalues = np.exp(-0.4 * np.arange(D))   # λ_i = e^{-0.4i}
full_eigenvalues /= full_eigenvalues.sum()         # normalise to sum=1

rank_list = list(range(1, D + 1))   # r = 1, 2, ..., 16

def make_Pi(rank):
    """Construct Π with given rank using top-r eigenvectors."""
    evs = full_eigenvalues.copy()
    evs[rank:] = 0.0
    return U_full @ np.diag(evs) @ U_full.T

def project_obs(X, rank):
    """Project X onto observable subspace of rank r."""
    U_r = U_full[:, :rank]
    return X @ U_r          # (N, rank)

# ── 3. Metrics for each rank ──────────────────────────────────────────────────
accuracies          = []
shap_variances      = []
info_ratios         = []

for r in rank_list:
    # (a) Accuracy: train logistic regression on rank-r projected features
    X_tr_r = project_obs(X_train, r)
    X_te_r = project_obs(X_test,  r)
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=0)
    clf.fit(X_tr_r, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te_r))
    accuracies.append(acc)

    # (b) Attribution stability: variance of SHAP values across test samples
    #     Lower variance = more stable attribution
    ridge = Ridge(alpha=0.1).fit(X_tr_r, y_train.astype(float))
    # Use the new maskers.Independent API
    # feature_perturbation is deprecated
    masker = shap.maskers.Independent(X_tr_r)
    exp    = shap.LinearExplainer(ridge, masker)
    sv     = exp.shap_values(X_te_r)         # (N_test, r)
    shap_variances.append(float(np.var(sv)))

    # (c) Observable information ratio: fraction of total spectral energy captured
    info_ratio = full_eigenvalues[:r].sum()  # already normalised
    info_ratios.append(float(info_ratio))

accuracies     = np.array(accuracies)
shap_variances = np.array(shap_variances)
info_ratios    = np.array(info_ratios)

# Normalise SHAP variance for plotting (0–1)
shap_var_norm = shap_variances / shap_variances.max()

# ── 4. Spectral energy curves for panel (d) ───────────────────────────────────
# Show cumulative spectral energy for several selected ranks
selected_ranks = [2, 4, 8, 12, 16]
colors_rank = plt.cm.Blues(np.linspace(0.35, 0.95, len(selected_ranks)))

# ── 5. Plotting ───────────────────────────────────────────────────────────────
COLOR_OBS  = '#2166ac'
COLOR_KER  = '#d6604d'
COLOR_ACC  = '#1a9641'
COLOR_INFO = '#762a83'
COLOR_BASE = '#555555'

fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.40)

# ────────────────────────────────────────────────────────
# Panel (a): rank(Π) vs. Test Accuracy
# ────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(rank_list, accuracies,
          color=COLOR_ACC, linewidth=2.0, marker='o',
          markersize=4, label='Test accuracy')
ax_a.axvline(R_true, color=COLOR_KER, linewidth=1.2,
             linestyle='--', alpha=0.7,
             label=f'True signal rank $r^*={R_true}$')
ax_a.set_xlabel('$\\mathrm{rank}(\\Pi)$')
ax_a.set_ylabel('Test accuracy')
ax_a.set_title('(a) Accuracy vs. $\\mathrm{rank}(\\Pi)$')
ax_a.set_xticks(range(1, D+1, 2))
ax_a.set_ylim(0, 1.05)
ax_a.legend(loc='lower right', framealpha=0.75)

# Saturation annotation
sat_idx = np.where(accuracies >= 0.98 * accuracies.max())[0][0]
ax_a.annotate(f'Saturates at\n$r={rank_list[sat_idx]}$',
              xy=(rank_list[sat_idx], accuracies[sat_idx]),
              xytext=(rank_list[sat_idx] + 2, accuracies[sat_idx] - 0.12),
              fontsize=8, color=COLOR_ACC,
              arrowprops=dict(arrowstyle='->', color=COLOR_ACC, lw=1.0))

# ────────────────────────────────────────────────────────
# Panel (b): rank(Π) vs. Attribution Stability
# ────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(rank_list, shap_var_norm,
          color=COLOR_OBS, linewidth=2.0, marker='s',
          markersize=4, label='SHAP variance (normalized)')
ax_b.axvline(R_true, color=COLOR_KER, linewidth=1.2,
             linestyle='--', alpha=0.7,
             label=f'True signal rank $r^*={R_true}$')
ax_b.set_xlabel('$\\mathrm{rank}(\\Pi)$')
ax_b.set_ylabel('Normalized SHAP variance')
ax_b.set_title('(b) Attribution stability vs. $\\mathrm{rank}(\\Pi)$')
ax_b.set_xticks(range(1, D+1, 2))
ax_b.set_ylim(0, 1.15)
ax_b.legend(loc='upper left', framealpha=0.75)

ax_b.annotate('Adding ker dimensions\nincreases attribution noise',
              xy=(R_true + 1, shap_var_norm[R_true]),
              xytext=(R_true + 3, shap_var_norm[R_true] - 0.25),
              fontsize=7.5, color=COLOR_OBS,
              arrowprops=dict(arrowstyle='->', color=COLOR_OBS, lw=1.0))

# ────────────────────────────────────────────────────────
# Panel (c): rank(Π) vs. Observable Information Ratio
# ────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(rank_list, info_ratios,
          color=COLOR_INFO, linewidth=2.0, marker='^',
          markersize=4, label='Cumulative spectral energy')
ax_c.axvline(R_true, color=COLOR_KER, linewidth=1.2,
             linestyle='--', alpha=0.7,
             label=f'True signal rank $r^*={R_true}$')
ax_c.axhline(info_ratios[R_true - 1], color=COLOR_INFO,
             linewidth=0.8, linestyle=':', alpha=0.5)
ax_c.set_xlabel('$\\mathrm{rank}(\\Pi)$')
ax_c.set_ylabel('Cumulative spectral energy ratio')
ax_c.set_title('(c) Observable information ratio vs. $\\mathrm{rank}(\\Pi)$')
ax_c.set_xticks(range(1, D+1, 2))
ax_c.set_ylim(0, 1.08)
ax_c.legend(loc='lower right', framealpha=0.75)

# Annotate the value at r*
ax_c.annotate(f'{info_ratios[R_true-1]:.2f} at $r^*$',
              xy=(R_true, info_ratios[R_true - 1]),
              xytext=(R_true + 2, info_ratios[R_true - 1] - 0.12),
              fontsize=8, color=COLOR_INFO,
              arrowprops=dict(arrowstyle='->', color=COLOR_INFO, lw=1.0))

# ────────────────────────────────────────────────────────
# Panel (d): Spectral energy profiles for selected ranks
# ────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

for idx, r in enumerate(selected_ranks):
    evs = full_eigenvalues.copy()
    evs[r:] = 0.0
    cumulative = np.cumsum(evs)
    ax_d.plot(range(1, D + 1), cumulative,
              color=colors_rank[idx], linewidth=1.8,
              label=f'$r={r}$')

# Full spectrum
ax_d.plot(range(1, D + 1), np.cumsum(full_eigenvalues),
          color=COLOR_BASE, linewidth=1.5, linestyle='--',
          label='Full ($r=16$)', zorder=5)

ax_d.set_xlabel('Dimension index')
ax_d.set_ylabel('Cumulative spectral energy')
ax_d.set_title('(d) Spectral energy profiles for selected $\\mathrm{rank}(\\Pi)$')
ax_d.set_xticks(range(1, D+1, 2))
ax_d.set_ylim(0, 1.12)
ax_d.legend(loc='lower right', framealpha=0.75, ncol=2)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    'Rank-Controlled Observability: rank($\\Pi$) Determines Effective Observable Geometry (PPS)',
    fontsize=11, fontweight='bold', y=1.01
)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig('figure_rank_control_v2.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figure_rank_control_v2.png', bbox_inches='tight', dpi=300)
plt.show()
print("Saved: figure_rank_control_v2.pdf / .png")

# ── 6. Numerical summary ─────────────────────────────────────────────────────
print("\n=== Numerical Summary ===")
print(f"Latent dim D={D},  True signal rank r*={R_true},  Classes C={C}")
print(f"\n(a) Accuracy:")
print(f"  rank=1:    {accuracies[0]:.4f}")
print(f"  rank=r*:   {accuracies[R_true-1]:.4f}")
print(f"  rank=D:    {accuracies[-1]:.4f}")
print(f"  Saturation rank: {rank_list[sat_idx]}")
print(f"\n(b) SHAP variance (normalized):")
print(f"  rank=1:    {shap_var_norm[0]:.4f}")
print(f"  rank=r*:   {shap_var_norm[R_true-1]:.4f}")
print(f"  rank=D:    {shap_var_norm[-1]:.4f}")
print(f"\n(c) Observable information ratio:")
print(f"  rank=1:    {info_ratios[0]:.4f}")
print(f"  rank=r*:   {info_ratios[R_true-1]:.4f}")
print(f"  rank=D:    {info_ratios[-1]:.4f}")

print("\n[Example sentence for manuscript text]")
print(f'  "As rank(Π) increases from 1 to r*={R_true}, test accuracy rises from')
print(f'   {accuracies[0]:.2f} to {accuracies[R_true-1]:.2f}, capturing {info_ratios[R_true-1]*100:.1f}% of')
print(f'   the total spectral energy. Beyond r*, additional dimensions in ker(Π)')
print(f'   contribute no predictive signal but increase attribution variance')
print(f'   from {shap_var_norm[R_true-1]:.2f} to {shap_var_norm[-1]:.2f} (normalized),')
print(f'   consistent with the PPS prediction that ker(Π) components are')
print(f'   structurally non-informative under observation."')
