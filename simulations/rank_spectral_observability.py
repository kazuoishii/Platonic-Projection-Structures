import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
D, n = 32, 5000
X = rng.normal(size=(n, D))
ranks = np.arange(1, D + 1)
energy = []

spectrum = np.exp(-np.linspace(0, 4, D))
for r in ranks:
    lam = np.r_[spectrum[:r], np.zeros(D-r)]
    Pi = np.diag(lam)
    obs = np.einsum("ni,ij,nj->n", X, Pi, X)
    energy.append(obs.mean())

energy = np.asarray(energy)
energy /= energy[-1]

plt.figure()
plt.plot(ranks, energy, marker="o")
plt.xlabel("rank(Pi)")
plt.ylabel("Normalized mean observable energy")
plt.title("Rank-controlled observable geometry under PPS")
plt.tight_layout()
plt.savefig("rank_spectral_observability.png", dpi=200)
print("Saved rank_spectral_observability.png")
