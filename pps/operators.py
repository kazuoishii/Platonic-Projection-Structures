import numpy as np

def quadratic_observable(v, Pi):
    v = np.asarray(v, dtype=float)
    Pi = np.asarray(Pi, dtype=float)
    return float(v @ Pi @ v)

def effective_rank(Pi, tol=1e-10):
    vals = np.linalg.eigvalsh(np.asarray(Pi, dtype=float))
    scale = max(1.0, float(np.max(np.abs(vals))))
    return int(np.sum(vals > tol * scale))

def eigenspaces(Pi, tol=1e-10):
    Pi = np.asarray(Pi, dtype=float)
    vals, vecs = np.linalg.eigh(Pi)
    scale = max(1.0, float(np.max(np.abs(vals))))
    mask = vals > tol * scale
    return vals, vecs[:, mask], vecs[:, ~mask]
