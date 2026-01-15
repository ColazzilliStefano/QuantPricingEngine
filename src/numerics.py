import numba
import numpy as np

@numba.jit(nopython=True)
def hw_exact_loop(num_steps: int, num_paths: int, r0: float, analytical_mean_vector: np.ndarray,
                   exp_k_dt_arr: np.ndarray, stoch_vol_arr: np.ndarray, dW_matrix: np.ndarray) -> np.ndarray:
    """JIT-compiled loop for the exact discretization scheme."""
    rates = np.zeros((num_steps + 1, num_paths))
    rates[0, :] = r0
    x_prev = np.zeros(num_paths)

    for t in range(1, num_steps + 1):
        decay = exp_k_dt_arr[t-1]
        vol = stoch_vol_arr[t-1]
        for p in range(num_paths):
            x_curr = x_prev[p] * decay + vol * dW_matrix[t-1, p]
            rates[t, p] = x_curr + analytical_mean_vector[t]
            x_prev[p] = x_curr
    return rates

@numba.jit(nopython=True)
def hw_euler_loop(num_steps: int, num_paths: int, r0: float,
                   theta_vector: np.ndarray, kappa: float, dt: float,
                   sigma_sqrt_dt: float, dW_matrix: np.ndarray) -> np.ndarray:
    """JIT-compiled loop for the Euler-Maruyama discretization scheme."""
    rates = np.zeros((num_steps + 1, num_paths))
    rates[0, :] = r0
    for t in range(num_steps):
        for p in range(num_paths):
            dr = (theta_vector[t] - kappa * rates[t, p]) * dt + sigma_sqrt_dt * dW_matrix[t, p]
            rates[t+1, p] = rates[t, p] + dr
    return rates
