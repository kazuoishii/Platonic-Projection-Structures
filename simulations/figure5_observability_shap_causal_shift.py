"""
Figure 5 — Observability, Attribution, and Causal Influence under PPS

Google Colab compatible version.
Before running in Colab:
    !pip install shap -q

Output:
    figure5_revised.pdf

This experiment illustrates that attribution-based detectability depends
on the observable input structure of a model. A hidden variable receives
zero SHAP attribution by construction, although intervention on the same
variable in a model with access to it reveals substantial output sensitivity.

The experiment is an illustration of the distinction between observability
and causal influence. It should not be interpreted as a direct identification
of a hidden input variable with ker(Pi).
"""

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 1] Installation (run this first in Google Colab)
# ══════════════════════════════════════════════════════════════════════════════
# !pip install shap -q

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 2] Imports and configuration
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 3] Data generation
# ══════════════════════════════════════════════════════════════════════════════
N = 2000
x1     = np.random.randn(N)
x2     = np.random.randn(N)
z_star = np.random.randn(N)          # Sensitive attribute (e.g., race or ZIP code)

noise    = 0.1 * np.random.randn(N)
y_true   = 0.6 * x1 + 0.4 * x2 + 0.5 * z_star + noise
y_binary = (y_true > y_true.mean()).astype(np.float32)

# Visible model input: x1 and x2 only
X_visible = np.stack([x1, x2],         axis=1).astype(np.float32)
# Full model input: x1, x2, and z*
X_full    = np.stack([x1, x2, z_star], axis=1).astype(np.float32)

scaler_v = StandardScaler()
scaler_f = StandardScaler()
X_vis_s  = scaler_v.fit_transform(X_visible).astype(np.float32)
X_full_s = scaler_f.fit_transform(X_full).astype(np.float32)

print("Data generation complete")
print(f"  Number of samples: {N}")
print(f"  Visible入力: {X_vis_s.shape}  (x1, x2)")
print(f"  Full入力:    {X_full_s.shape}  (x1, x2, z*)")

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 4] Model definition and training
# ══════════════════════════════════════════════════════════════════════════════
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU(),
        )
        self.W = nn.Linear(latent_dim, 1, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.W(self.encoder(x))).squeeze()


def train(model, X, y, epochs=300, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    for ep in range(epochs):
        opt.zero_grad()
        nn.BCELoss()(model(Xt), yt).backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            with torch.no_grad():
                loss = nn.BCELoss()(model(Xt), yt).item()
            print(f"  epoch {ep+1:3d}  loss={loss:.4f}")
    return model


print("\n--- Training visible model (x1 and x2 only) ---")
model_vis  = train(MLP(input_dim=2), X_vis_s,  y_binary)

model_vis = train(MLP(input_dim=2), X_vis_s, y_binary)
model_full = train(MLP(input_dim=3), X_full_s, y_binary)

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 5] Eigenvalue spectrum of Pi
# ══════════════════════════════════════════════════════════════════════════════
W_mat          = model_full.W.weight.detach().numpy()   # (1, latent_dim)
Pi             = W_mat.T @ W_mat                        # (latent_dim, latent_dim)
eigenvalues, _ = np.linalg.eigh(Pi)
eigenvalues    = eigenvalues[::-1]                      # Sort in descending order

ker_threshold = 1e-4
ker_dims   = np.where(eigenvalues < ker_threshold)[0]
range_dims = np.where(eigenvalues >= ker_threshold)[0]

print(f"Rank of Pi:    {len(range_dims)}")
print(f"Dimension of ker(Pi): {len(ker_dims)}")
print(f"Eigenvalues:        {np.round(eigenvalues, 4)}")

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 6] SHAP computation
# ══════════════════════════════════════════════════════════════════════════════
def predict_vis(x):
    with torch.no_grad():
        return model_vis(torch.tensor(x, dtype=torch.float32)).numpy().reshape(-1)

def predict_full(x):
    with torch.no_grad():
        return model_full(torch.tensor(x, dtype=torch.float32)).numpy().reshape(-1)

print("Computing SHAP values for the visible model...")
exp_v       = shap.KernelExplainer(predict_vis,  X_vis_s[:100])
shap_v      = exp_v.shap_values(X_vis_s[100:200], nsamples=100)
mean_shap_v = np.abs(shap_v).mean(axis=0)

print("Computing SHAP values for the full model...")
exp_f       = shap.KernelExplainer(predict_full, X_full_s[:100])
shap_f      = exp_f.shap_values(X_full_s[100:200], nsamples=100)
mean_shap_f = np.abs(shap_f).mean(axis=0)

print(f"\nSHAP (Visible) — x1={mean_shap_v[0]:.4f}  x2={mean_shap_v[1]:.4f}")
print(f"SHAP (Full)    — x1={mean_shap_f[0]:.4f}  x2={mean_shap_f[1]:.4f}  z*={mean_shap_f[2]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 7] Causal intervention by flipping z*
# ══════════════════════════════════════════════════════════════════════════════
test_f = X_full_s[200:400].copy()
with torch.no_grad():
    out_orig = model_full(torch.tensor(test_f, dtype=torch.float32)).numpy()

intervened       = test_f.copy()
intervened[:, 2] = -intervened[:, 2]
with torch.no_grad():
    out_int = model_full(torch.tensor(intervened, dtype=torch.float32)).numpy()

delta = np.abs(out_int - out_orig)
print(f"Causal intervention (z* sign flip) — mean Δ={delta.mean():.4f}  max Δ={delta.max():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# [Cell 8] Figure 5: Four-panel visualization
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
fig.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.18, wspace=0.42)

NAVY  = '#1a3a6b'
GREEN = '#2e7d32'
RED   = '#c62828'
DRED  = '#d32f2f'
GREY  = '#9e9e9e'
LGREY = '#78909c'

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#aaaaaa')
    ax.spines['bottom'].set_color('#aaaaaa')
    ax.tick_params(colors='#333333', labelsize=10)
    ax.yaxis.label.set_color('#333333')
    ax.xaxis.label.set_color('#333333')

# (a) Spectral Analysis of Π
ax = axes[0]
latent_dim = len(eigenvalues)
col_eig = [NAVY if v >= ker_threshold else GREY for v in eigenvalues]
ax.bar(range(latent_dim), eigenvalues, color=col_eig, width=0.65, zorder=3)
ax.set_xlabel('Latent Dimension Index', fontsize=11)
ax.set_ylabel('Eigenvalue Magnitude',   fontsize=11)
ax.set_title(r'$\bf{(a)}$ Spectral Analysis of $\Pi$', fontsize=12, pad=10)
ax.set_xticks(range(latent_dim))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
style_ax(ax)

# (b) SHAP: Visible Model
ax = axes[1]
ymax = max(mean_shap_f) * 1.55
feat_v = ['$x_1$', '$x_2$', '$z^*$\n(hidden)']
vals_v = [mean_shap_v[0], mean_shap_v[1], 0.0]
cols_v = [GREEN, GREEN, RED]
ax.bar(feat_v, vals_v, color=cols_v, width=0.5, zorder=3)
ax.text(2, 0.004, 'Not in input\nSHAP = 0',
        ha='center', va='bottom', color=DRED, fontsize=9, fontweight='bold')
ax.set_ylabel('Mean |SHAP Value|', fontsize=11)
ax.set_title('$\\bf{(b)}$ SHAP: Visible Model\n'
             r'($x_1, x_2$ only — $z^*$ hidden)', fontsize=11, pad=10)
ax.set_ylim(0, ymax)
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
style_ax(ax)

# (c) SHAP: Full Model
ax = axes[2]
feat_f = ['$x_1$', '$x_2$', '$z^*$\n(visible)']
vals_f = list(mean_shap_f)
cols_f = [GREEN, GREEN, NAVY]
ax.bar(feat_f, vals_f, color=cols_f, width=0.5, zorder=3)
ax.text(2, mean_shap_f[2] + 0.005,
        f'{mean_shap_f[2]:.3f}\n(detected)',
        ha='center', va='bottom', color=NAVY, fontsize=9, fontweight='bold')
ax.set_ylabel('Mean |SHAP Value|', fontsize=11)
ax.set_title('$\\bf{(c)}$ SHAP: Full Model\n'
             r'($x_1, x_2, z^*$ all visible)', fontsize=11, pad=10)
ax.set_ylim(0, ymax)
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
style_ax(ax)

# (d) Causal Shift in Output
ax = axes[3]
ax.hist(delta, bins=30, color=LGREY, edgecolor='none', alpha=0.9, zorder=3)
ax.axvline(delta.mean(), color=DRED, linestyle='--', linewidth=2,
           label=f'Mean \u0394 = {delta.mean():.2f}', zorder=4)
ax.set_xlabel('\u0394 Output Probability', fontsize=11)
ax.set_ylabel('Count',                     fontsize=11)
ax.set_title(r'$\bf{(d)}$ Causal Shift in Output', fontsize=12, pad=10)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
style_ax(ax)

# Bottom annotation
fig.text(
    0.5, 0.01,
    r'(b) $z^*$ hidden $\rightarrow$ SHAP = 0 (undetectable)     '
    r'(c) $z^*$ visible $\rightarrow$ SHAP $\neq$ 0 (detectable)     '
    r'(d) yet causal influence persists regardless',
    ha='center', fontsize=10, color='#444444', style='italic'
)

# Save to the current working directory (Google Colab)
out_path = 'figure5_revised.pdf'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\n✓ Saved successfully → {out_path}")
print("In Google Colab, download the file from the 'Files' tab in the left sidebar.")
