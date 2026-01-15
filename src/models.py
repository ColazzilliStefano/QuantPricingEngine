import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Callable, Optional, Tuple, Dict, Any
from .numerics import hw_exact_loop, hw_euler_loop
from .curves import NSSModel

class HullWhiteEngine:
    def __init__(self, kappa: float, sigma: float,
                 forward_curve_func: Callable,
                 forward_curve_prime_func: Callable,
                 spot_curve_func: Callable,
                 curve_params: np.ndarray,
                 r0: float, seed: int = 42):
        self.kappa = kappa
        self.sigma = sigma
        self.r0 = r0
        self.f_forward = forward_curve_func
        self.f_prime = forward_curve_prime_func
        self.P_spot_market = spot_curve_func
        self.curve_params = curve_params
        self.rng = np.random.Generator(np.random.PCG64(seed))
        print(f"HullWhiteEngine Initialized: k={self.kappa:.4f}, s={self.sigma:.4f}, r0={self.r0:.4f}")

    def _get_theta_vector(self, time_axis: np.ndarray) -> np.ndarray:
        t = np.maximum(time_axis, 1e-6)
        f_0_t = self.f_forward(t, self.curve_params)
        f_prime_0_t = self.f_prime(t, self.curve_params)
        term1 = f_prime_0_t
        term2 = self.kappa * f_0_t
        term3 = (self.sigma**2 / (2.0 * self.kappa)) * (1.0 - np.exp(-2.0 * self.kappa * t))
        return term1 + term2 + term3

    def _get_analytical_mean_vector(self, time_axis: np.ndarray) -> np.ndarray:
        forward_curve = self.f_forward(time_axis, self.curve_params)
        convexity_adj = (self.sigma**2 / (2.0 * self.kappa**2)) * (1.0 - np.exp(-self.kappa * time_axis))**2
        analytical_mean = forward_curve + convexity_adj
        if not np.isclose(analytical_mean[0], self.r0):
            print(f"WARNING: E[r(0)] ({analytical_mean[0]:.6f}) != r0 ({self.r0:.6f}).")
        return analytical_mean

    def run_simulation_exact(self, T: float, num_steps: int, num_paths: int, event_dates: Optional[np.ndarray] = None):
        print(f"Starting MC Simulation (Exact Scheme) - T={T}")
        base_grid = np.linspace(0, T, num_steps + 1)
        if event_dates is not None:
            full_grid = np.unique(np.concatenate((base_grid, event_dates)))
            time_axis = full_grid[full_grid <= T + 1e-9]
        else:
            time_axis = base_grid

        num_steps = len(time_axis) - 1
        dt_array = np.diff(time_axis)

        analytical_mean_vector = self._get_analytical_mean_vector(time_axis)
        exp_k_dt_vec = np.exp(-self.kappa * dt_array)
        stoch_vol_vec = self.sigma * np.sqrt((1.0 - np.exp(-2.0 * self.kappa * dt_array)) / (2.0 * self.kappa))
        dW_matrix = self.rng.normal(0.0, 1.0, (num_steps, num_paths))

        rates = hw_exact_loop(num_steps, num_paths, self.r0, analytical_mean_vector, exp_k_dt_vec, stoch_vol_vec, dW_matrix)
        return rates, time_axis, analytical_mean_vector

    def run_simulation_euler(self, T: float, num_steps: int, num_paths: int):
        print(f"Starting MC Simulation (Euler Scheme) - {num_steps} steps")
        dt = T / num_steps
        time_axis = np.linspace(0, T, num_steps + 1)
        theta_vector = self._get_theta_vector(time_axis)
        sigma_sqrt_dt = self.sigma * np.sqrt(dt)
        dW_matrix = self.rng.normal(0.0, 1.0, (num_steps, num_paths))
        rates = hw_euler_loop(num_steps, num_paths, self.r0, theta_vector[:-1], self.kappa, dt, sigma_sqrt_dt, dW_matrix)
        return rates, time_axis

    def analyze_results(self, simulated_rates: np.ndarray, time_axis: np.ndarray, analytical_mean: np.ndarray) -> Dict[str, Any]:
        final_rates = simulated_rates[-1, :]
        return {
            "analytical_mean": analytical_mean,
            "simulated_mean": np.mean(simulated_rates, axis=1),
            "final_rates": final_rates,
            "final_mean": np.mean(final_rates),
            "final_median": np.median(final_rates)
        }

    # ANALYTIC PRICING
    def B(self, t: float, T: float) -> float:
        return (1.0 - np.exp(-self.kappa * (T - t))) / self.kappa

    def _A(self, t: float, T: float) -> float:
        P_0_T = self.P_spot_market(np.array([T]), self.curve_params)[0]
        P_0_t = self.P_spot_market(np.array([t]), self.curve_params)[0]
        f_0_t = self.f_forward(np.array([t]), self.curve_params)[0]
        B_val = self.B(t, T)
        convexity = (self.sigma**2 / (4 * self.kappa)) * (1 - np.exp(-2 * self.kappa * t)) * (B_val**2)
        return np.exp(np.log(P_0_T / P_0_t) + B_val * f_0_t - convexity)

    def price_zcb_analytic(self, t: float, T: float, r_t: float = None) -> float:
        if r_t is None: r_t = self.r0
        if t >= T: return 1.0
        P_market_T = NSSModel.P_spot(np.array([T]), self.curve_params)[0]
        P_market_t = NSSModel.P_spot(np.array([t]), self.curve_params)[0]
        B_val = self.B(t, T)
        f_0_t = self.f_forward(np.array([t]), self.curve_params)[0]
        adjustment = (self.sigma**2 / (4 * self.kappa)) * (B_val**2) * (1 - np.exp(-2 * self.kappa * t))
        return (P_market_T / P_market_t) * np.exp(-B_val * (r_t - f_0_t) - adjustment)

    def price_option_on_zcb_analytic(self, t_eval: float, T_expiry: float, T_maturity: float,
                                     strike: float, option_type: str = 'call') -> float:
        P_t_S = self.price_zcb_analytic(t_eval, T_expiry)
        P_t_T = self.price_zcb_analytic(t_eval, T_maturity)
        Sigma_P = (self.sigma / self.kappa) * \
                  (1.0 - np.exp(-self.kappa * (T_maturity - T_expiry))) * \
                  np.sqrt((1.0 - np.exp(-2.0 * self.kappa * (T_expiry - t_eval))) / (2.0 * self.kappa))

        if Sigma_P < 1e-9:
            intrinsic_value = P_t_T - strike * P_t_S
            return max(0.0, intrinsic_value) if option_type == 'call' else max(0.0, -intrinsic_value)

        d1 = (np.log(P_t_T / (P_t_S * strike)) + 0.5 * Sigma_P**2) / Sigma_P
        d2 = d1 - Sigma_P

        if option_type == 'call':
            return P_t_T * norm.cdf(d1) - strike * P_t_S * norm.cdf(d2)
        else:
            return strike * P_t_S * norm.cdf(-d2) - P_t_T * norm.cdf(-d1)

    # COMPLEX INSTRUMENTS
    def price_swaption_jamshidian(self, T_expiry: float, T_tenor: float, K_strike: float) -> float:
        payment_times = np.arange(T_expiry + 1.0, T_expiry + T_tenor + 1.0, 1.0)
        def objective(r):
            price_sum = 0.0
            for T_i in payment_times:
                coupon = K_strike if T_i != payment_times[-1] else (1.0 + K_strike)
                A_i = self._A(T_expiry, T_i)
                B_i = self.B(T_expiry, T_i)
                price_sum += coupon * A_i * np.exp(-B_i * r)
            return price_sum - 1.0

        try:
            r_star = brentq(objective, -0.5, 0.5)
        except ValueError:
            print(f"Jamshidian Error: r* not found")
            return 0.0

        swaption_price = 0.0
        for T_i in payment_times:
            coupon = K_strike if T_i != payment_times[-1] else (1.0 + K_strike)
            A_i = self._A(T_expiry, T_i)
            B_i = self.B(T_expiry, T_i)
            K_i = A_i * np.exp(-B_i * r_star)
            put_price = self.price_option_on_zcb_analytic(0, T_expiry, T_i, strike=K_i, option_type='put')
            swaption_price += coupon * put_price
        return swaption_price

    def price_coupon_bond(self, t_eval: float, coupon_times: np.ndarray, coupon_amounts: np.ndarray, face_value: float) -> float:
        price = 0.0
        for t_pay, c_amount in zip(coupon_times, coupon_amounts):
            if t_pay > t_eval:
                price += self.price_zcb_analytic(t_eval, t_pay) * c_amount
        if len(coupon_times) > 0 and coupon_times[-1] > t_eval:
            price += self.price_zcb_analytic(t_eval, coupon_times[-1]) * face_value
        return price

    def price_irs(self, t_eval: float, t_start: float, t_end: float, fixed_rate: float, freq: float = 0.5, notional: float = 100.0, is_payer: bool = True) -> float:
        val_float = notional * (self.price_zcb_analytic(t_eval, t_start) - self.price_zcb_analytic(t_eval, t_end))
        payment_dates = np.arange(t_start + freq, t_end + 0.001, freq)
        val_fixed = 0.0
        coupon = fixed_rate * freq * notional
        for t_pay in payment_dates:
            val_fixed += coupon * self.price_zcb_analytic(t_eval, t_pay)
        return (val_float - val_fixed) if is_payer else (val_fixed - val_float)

    def price_cap_floor(self, t_eval: float, start: float, end: float, strike: float, freq: float = 0.5, notional: float = 100.0, is_cap: bool = True) -> float:
        pay_dates = np.arange(start + freq, end + 0.001, freq)
        fix_dates = pay_dates - freq
        total_val = 0.0
        opt_type = 'put' if is_cap else 'call'

        for t_fix, t_pay in zip(fix_dates, pay_dates):
            tau = t_pay - t_fix
            strike_bond = 1.0 / (1.0 + strike * tau)
            opt_val = self.price_option_on_zcb_analytic(t_eval, t_fix, t_pay, strike_bond, opt_type)
            total_val += opt_val * (1.0 + strike * tau) * notional
        return total_val

    def price_zcb_mc(self, T: float, num_steps: int = 100, num_paths: int = 5000, use_exact_scheme: bool = True) -> Tuple[float, float]:
        if use_exact_scheme:
                simulated_rates, time_axis, _ = self.run_simulation_exact(T, num_steps, num_paths)
        else:
                simulated_rates, time_axis = self.run_simulation_euler(T, num_steps, num_paths)
        dt_array = np.diff(time_axis)
        integral_r_paths = np.sum(simulated_rates[:-1, :] * dt_array[:, None], axis=0)
        discount_factors = np.exp(-integral_r_paths)
        return np.mean(discount_factors), np.std(discount_factors) / np.sqrt(num_paths)

    # PDE PRICING
    def price_zcb_pde(self, T, Nspace=400, Ntime=1000) -> Tuple[float, np.ndarray, np.ndarray]:
        print(f"Starting PDE Solver for ZCB T={T}")
        dt = T / Ntime
        vol_approx = self.sigma * np.sqrt(T)
        r_max = self.r0 + 6 * vol_approx
        r_min = self.r0 - 6 * vol_approx
        r, dr = np.linspace(r_min, r_max, Nspace, retstep=True)
        drr = dr * dr
        sig2 = self.sigma**2
        t_grid = np.linspace(0, T, Ntime + 1)
        theta_vec = self._get_theta_vector(t_grid)

        V = np.zeros((Nspace, Ntime + 1))
        V[:, -1] = 1.0

        for n in range(Ntime - 1, -1, -1):
            t_curr = t_grid[n]
            drift = theta_vec[n] - self.kappa * r[1:-1]
            max_p, min_p = np.maximum(drift, 0), np.minimum(drift, 0)

            a = min_p * (dt/dr) - 0.5 * (dt/drr) * sig2
            b = 1 + dt * r[1:-1] + (dt/drr) * sig2 + (dt/dr) * (max_p - min_p)
            c = -max_p * (dt/dr) - 0.5 * (dt/drr) * sig2

            D = sparse.diags([a[1:], b, c[:-1]], [-1, 0, 1], shape=(Nspace-2, Nspace-2)).tocsc()

            val_min = self.price_zcb_analytic(t_curr, T, r[0])
            val_max = self.price_zcb_analytic(t_curr, T, r[-1])
            V[0, n], V[-1, n] = val_min, val_max

            rhs = V[1:-1, n+1].copy()
            rhs[0] -= a[0] * val_min
            rhs[-1] -= c[-1] * val_max

            try: V[1:-1, n] = spsolve(D, rhs)
            except: V[1:-1, n] = V[1:-1, n+1]

        from scipy import interpolate
        f_val = interpolate.interp1d(r, V[:, 0], kind='cubic')
        return float(f_val(self.r0)), r, V

    def price_callable_bond_pde(self, T_maturity: float, coupon_times: np.ndarray, coupon_amounts: np.ndarray,
                                call_times: np.ndarray, call_price: float, face_value: float = 100.0,
                                Nspace: int = 100, Ntime: int = 1000) -> float:
        print(f"Starting PDE Solver for Callable Bond T={T_maturity}")
        r_max, r_min = max(self.r0 * 4, 0.6), -0.20
        r, dr = np.linspace(r_min, r_max, Nspace, retstep=True)
        t, dt = np.linspace(0, T_maturity, Ntime, retstep=True)
        V = np.full(Nspace, face_value)
        sig2, drr = self.sigma**2, dr * dr
        theta_vec = self._get_theta_vector(t)
        dt_tol = dt / 1.5

        for n in range(Ntime - 2, -1, -1):
            t_current = t[n]
            drift = theta_vec[n] - self.kappa * r[1:-1]
            max_p, min_p = np.maximum(drift, 0), np.minimum(drift, 0)
            a = min_p * (dt/dr) - 0.5 * (dt/drr) * sig2
            b = 1 + dt * r[1:-1] + (dt/drr) * sig2 + (dt/dr) * (max_p - min_p)
            c = -max_p * (dt/dr) - 0.5 * (dt/drr) * sig2
            D = sparse.diags([a[1:], b, c[:-1]], [-1, 0, 1], shape=(Nspace-2, Nspace-2)).tocsc()
            offset = np.zeros(Nspace-2)
            offset[0] = a[0] * V[0]; offset[-1] = c[-1] * V[-1]
            try: V[1:-1] = spsolve(D, (V[1:-1] - offset))
            except: return np.nan

            for tc, c_amt in zip(coupon_times, coupon_amounts):
                if abs(tc - t_current) < dt_tol: V += c_amt

            is_call_date = False
            for t_call in call_times:
                if abs(t_call - t_current) < dt_tol:
                    is_call_date = True
                    break
            if is_call_date: V = np.minimum(V, call_price)

        return np.interp(self.r0, r, V)
