"""
Figure 6 — PPS Kernel Observability and Invariance
==================================================

This experiment directly verifies projection-induced observability
and kernel invariance under the Platonic Projection Structures (PPS)
framework.

Panel (a) visualizes the spectral decomposition of the observation
operator Pi into observable and kernel subspaces.

Panel (b) demonstrates that perturbations confined to ker(Pi) leave
the observable O(v) invariant.

Panel (c) shows that perturbations within range(Pi) change O(v).

Panel (d) illustrates the attribution gap between restricted observation
on range(Pi) and full latent-space observation.

Usage in Google Colab:
    !pip install shap -q

Then paste the entire contents of this script into a Colab cell and run it.

Outputs:
    figure6_kernel_observability_invariance.pdf
    figure6_kernel_observability_invariance.png
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
from sklearn.linear_model import Ridge

np.random.seed(42)

# ── 1. Observation Operator Π ────────────────────────────────────────────────
D = 8
R = 4   # rank(Π)
K = D - R

U_full, _ = np.linalg.qr(np.random.randn(D, D))
U_obs = U_full[:, :R]
U_ker = U_full[:, R:]

eigenvalues = np.array([3.0, 2.5, 1.8, 1.2,
                        0.0, 0.0, 0.0, 0.0])

Pi = U_full @ np.diag(eigenvalues) @ U_full.T

def O(v):
    return float(v @ Pi @ v)

# ── 2. Base vector decomposition ─────────────────────────────────────────────
v_base     = np.random.randn(D)
v_obs_base = U_obs @ (U_obs.T @ v_base)
v_ker_base = v_base - v_obs_base
O_base     = O(v_base)

alphas = np.linspace(-3, 3, 300)

# ── 3. Panel (b): ker perturbation -> O invariant ─────────────────────────────
fixed_dir_ker = U_ker @ np.random.randn(K)
fixed_dir_ker /= np.linalg.norm(fixed_dir_ker)

O_ker_perturb = [O(v_obs_base + v_ker_base + a * fixed_dir_ker)
                 for a in alphas]

# ── 4. Panel (c): range perturbation -> O changes────
raw = U_obs @ np.random.randn(R)
# Orthogonalize the perturbation direction against v_obs_base
if np.linalg.norm(v_obs_base) > 1e-12:
    raw -= (raw @ v_obs_base) / (v_obs_base @ v_obs_base) * v_obs_base
fixed_dir_obs = raw / np.linalg.norm(raw)

O_obs_perturb = [O(v_obs_base + a * fixed_dir_obs + v_ker_base)
                 for a in alphas]

# ── 5. Panel (d): SHAP attribution gap ───────────────────────────────────────
N = 600

# Coordinates in the eigenbasis of Pi
X_eig = np.random.randn(N, D)

# Corresponding vectors in the original latent space
X_latent = X_eig @ U_full.T

w_true = np.random.randn(R)

# The target depends only on observable coordinates
y = X_eig[:, :R] @ w_true + 0.05 * np.random.randn(N)

# Restricted model
X_obs = X_eig[:, :R]

m_restr = Ridge(alpha=0.1).fit(X_obs, y)
exp_r = shap.LinearExplainer(
    m_restr,
    X_obs,
    feature_perturbation="correlation_dependent"
)
sv_r = exp_r.shap_values(X_obs)

shap_obs_restr = np.abs(sv_r).mean(axis=0)
shap_ker_restr = np.zeros(K)   # exactly zero by construction

# Full model
m_full = Ridge(alpha=0.1).fit(X_eig, y)
exp_f = shap.LinearExplainer(
    m_full,
    X_eig,
    feature_perturbation="correlation_dependent"
)
sv_f = exp_f.shap_values(X_eig)

mean_sv = np.abs(sv_f).mean(axis=0)

shap_obs_full = mean_sv[:R]
shap_ker_full = mean_sv[R:]

# Normalization
_max = max(shap_obs_restr.max(), shap_obs_full.max())
shap_obs_restr_n = shap_obs_restr / _max
shap_obs_full_n  = shap_obs_full  / _max
shap_ker_full_n  = shap_ker_full  / _max

# ── 6. Plotting ───────────────────────────────────────────────────────────────
COLOR_OBS  = '#2166ac'
COLOR_KER  = '#d6604d'
COLOR_BASE = '#555555'

fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

# ────────────────────────────────────────────────────────
# Panel (a): Eigenvalue spectrum
# ────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
bar_c = [COLOR_OBS if ev > 0 else COLOR_KER for ev in eigenvalues]
ax_a.bar(range(1, D+1), eigenvalues, color=bar_c,
         edgecolor='white', linewidth=0.5, width=0.7)
ax_a.axhline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
ax_a.set_xlabel('Eigenvalue index $i$')
ax_a.set_ylabel('Eigenvalue $\\lambda_i$')
ax_a.set_title('(a) Spectral structure of $\\Pi$')
ax_a.set_xticks(range(1, D+1))
ax_a.set_ylim(-0.3, 4.0)
ax_a.annotate('$\\mathrm{range}(\\Pi)$,  $\\lambda_i > 0$',
              xy=(2.5, 2.6), fontsize=8.5, color=COLOR_OBS, ha='center')
ax_a.annotate('$\\ker(\\Pi)$,  $\\lambda_i = 0$',
              xy=(6.5, 0.28), fontsize=8.5, color=COLOR_KER, ha='center')

# ────────────────────────────────────────────────────────
# Panel (b): ker perturbation -> O remains invariant 
# ────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(alphas, O_ker_perturb,
          color=COLOR_KER, linewidth=2.0,
          label=r'$O(v_\mathrm{obs}+v_\ker+\alpha\,\hat{d}_\ker)$')
ax_b.axhline(O_base, color=COLOR_BASE, linewidth=1.2,
             linestyle='--', label='$O(v_\\mathrm{base})$')
ax_b.fill_between(alphas,
                  O_base - 0.15, O_base + 0.15,
                  alpha=0.12, color=COLOR_BASE)
ax_b.set_xlabel('Perturbation magnitude $\\alpha$')
ax_b.set_ylabel('Observable $O(v)$')
ax_b.set_title('(b) $O(v)$ invariant under $\\ker(\\Pi)$ perturbation')
ax_b.legend(loc='upper right', framealpha=0.75)
margin = max(0.5, (max(O_ker_perturb) - min(O_ker_perturb)) * 0.1 + 0.5)
ax_b.set_ylim(O_base - margin, O_base + margin)

# ────────────────────────────────────────────────────────
# Panel (c): observable-subspace perturbation -> O changes 
# (symmetric parabola around alpha = 0) 
# ────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(alphas, O_obs_perturb,
          color=COLOR_OBS, linewidth=2.0,
          label=r'$O(v_\mathrm{obs}+\alpha\,\hat{d}_\mathrm{obs}+v_\ker)$')
ax_c.axhline(O_base, color=COLOR_BASE, linewidth=1.2,
             linestyle='--', label='$O(v_\\mathrm{base})$')
ax_c.set_xlabel('Perturbation magnitude $\\alpha$')
ax_c.set_ylabel('Observable $O(v)$')
ax_c.set_title('(c) $O(v)$ changes under $\\mathrm{range}(\\Pi)$ perturbation')
ax_c.legend(loc='upper center', framealpha=0.75)

# Annotation positioned in the upper-left to avoid overlapping the curve
y_min_c = min(O_obs_perturb)
y_max_c = max(O_obs_perturb)
ax_c.annotate('Quadratic in $\\alpha$\n(parabolic growth)',
              xy=(alphas[10], O_obs_perturb[10]),
              xytext=(-2.6, y_min_c + (y_max_c - y_min_c) * 0.65),
              fontsize=8, color=COLOR_OBS,
              arrowprops=dict(arrowstyle='->', color=COLOR_OBS, lw=1.0))

# ────────────────────────────────────────────────────────
# Panel (d): SHAP attribution gap with the "kernel" label added 
# ────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

x_obs = np.arange(R)
x_ker = np.arange(R, R + K)
w     = 0.35

ax_d.bar(x_obs - w/2, shap_obs_restr_n,
         w, color=COLOR_OBS, alpha=0.55,
         label='Restricted (range($\\Pi$) only)')
ax_d.bar(x_ker - w/2, shap_ker_restr,
         w, color=COLOR_KER, alpha=0.55)

ax_d.bar(x_obs + w/2, shap_obs_full_n,
         w, color=COLOR_OBS, alpha=1.0,
         label='Full model (all dims)')
ax_d.bar(x_ker + w/2, shap_ker_full_n,
         w, color=COLOR_KER, alpha=1.0)

# obs / ker boundary
ax_d.axvline(R - 0.5, color='black', linewidth=0.9,
             linestyle=':', alpha=0.6)

# Set y-axis limit before adding labels
y_top = max(shap_obs_restr_n.max(), shap_obs_full_n.max(),
            shap_ker_full_n.max()) * 1.22
ax_d.set_ylim(0, y_top)

# "observable" label on the left side of the dotted boundary
ax_d.text(R / 2 - 0.5, y_top * 0.97,
          'observable', ha='center', va='top',
          fontsize=8.5, color=COLOR_OBS, fontstyle='italic')

# "kernel" label on the right side of the dotted boundary
ax_d.text(R + K / 2 - 0.5, y_top * 0.97,
          'kernel', ha='center', va='top',
          fontsize=8.5, color=COLOR_KER, fontstyle='italic')

ax_d.set_xlabel('Latent dimension index')
ax_d.set_ylabel('Normalized mean |SHAP value|')
ax_d.set_title('(d) Attribution gap: restricted vs. full observation')
ax_d.legend(loc='upper right', framealpha=0.75)
ax_d.set_xticks(list(x_obs) + list(x_ker))
ax_d.set_xticklabels(
    [f'$e_{{{i+1}}}$' for i in range(R)] +
    [f'$e_{{{i+R+1}}}$' for i in range(K)],
    fontsize=8.5
)

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle(
    'Projection-Induced Observability: Direct Verification of Kernel Invariance (PPS)',
    fontsize=11, fontweight='bold', y=1.01
)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.savefig('figure6_kernel_observability_invariance.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('figure6_kernel_observability_invariance.png',
            bbox_inches='tight', dpi=300)
plt.show()
print("Saved: figure6_kernel_observability_invariance.pdf / .png")

# ── 7. Numerical summary ─────────────────────────────────────────────────────
dev_ker = max(abs(np.array(O_ker_perturb) - O_base))
dev_obs = max(abs(np.array(O_obs_perturb) - O_base))

print("\n=== Numerical Summary===")
print(f"Latent dim D={D},  rank(Π)={R},  dim(ker)={K}")
print(f"O(v_base)                        = {O_base:.6f}")
print(f"Max |ΔO| under ker perturbation  = {dev_ker:.2e}  (≈ 0, machine precision)")
print(f"Max |ΔO| under obs perturbation  = {dev_obs:.4f}  (>> 0)")
print(f"\nSHAP (normalized):")
print(f"  Restricted – obs dims mean     = {shap_obs_restr_n.mean():.4f}")
print(f"  Restricted – ker dims mean     = {shap_ker_restr.mean():.4f}  (= 0 by construction)")
print(f"  Full model – obs dims mean     = {shap_obs_full_n.mean():.4f}")
print(f"  Full model – ker dims mean     = {shap_ker_full_n.mean():.4f}")
print("\n[Verification of Proposition 1]") 
print(f"  ker perturbation max |ΔO| = {dev_ker:.2e}  → ≈ machine epsilon ✓")
print(f"  obs perturbation max |ΔO| = {dev_obs:.4f}  → >> 0 ✓")
verdict = "Proposition 1 verified ✓" if dev_ker < 1e-10 else \
          "WARNING: ker deviation larger than expected"
print(f"  {verdict}")

print("\n[Example sentence for manuscript text]")
print(f'  "Perturbations confined to ker(Π) induce a maximum observable')
print(f'   deviation of |ΔO| = {dev_ker:.1e}, consistent with Proposition 1,')
print(f'   whereas equivalent perturbations in range(Π) yield |ΔO| = {dev_obs:.2f}."')
