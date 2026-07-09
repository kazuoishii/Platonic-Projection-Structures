import numpy as np
from pps import quadratic_observable

rng = np.random.default_rng(2026)
D, R, n = 8, 4, 1000
Pi = np.diag([2.0, 1.5, 1.0, 0.5] + [0.0] * (D - R))
v = rng.normal(size=D)
base = quadratic_observable(v, Pi)

kernel_changes = []
range_changes = []
for _ in range(n):
    dker = np.r_[np.zeros(R), rng.normal(size=D-R)]
    drange = np.r_[rng.normal(size=R), np.zeros(D-R)]
    kernel_changes.append(abs(quadratic_observable(v+dker, Pi) - base))
    range_changes.append(abs(quadratic_observable(v+drange, Pi) - base))

print("Mean absolute observable change")
print("kernel perturbation:", np.mean(kernel_changes))
print("range perturbation :", np.mean(range_changes))
assert np.max(kernel_changes) < 1e-10
