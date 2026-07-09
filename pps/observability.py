import numpy as np
from .operators import eigenspaces

def decompose_observable_kernel(v, Pi, tol=1e-10):
    v = np.asarray(v, dtype=float)
    _, U_obs, U_ker = eigenspaces(Pi, tol=tol)
    v_obs = U_obs @ (U_obs.T @ v) if U_obs.size else np.zeros_like(v)
    v_ker = U_ker @ (U_ker.T @ v) if U_ker.size else np.zeros_like(v)
    return v_obs, v_ker
