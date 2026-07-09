import numpy as np
from pps import quadratic_observable, decompose_observable_kernel

Pi = np.diag([2.0, 1.0, 0.0])
v = np.array([1.0, 2.0, 3.0])

v_obs, v_ker = decompose_observable_kernel(v, Pi)
print("v:", v)
print("observable component:", v_obs)
print("kernel component:", v_ker)
print("O(v) =", quadratic_observable(v, Pi))
print("O(v_obs) =", quadratic_observable(v_obs, Pi))
