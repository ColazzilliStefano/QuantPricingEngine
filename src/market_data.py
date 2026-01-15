import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from fredapi import Fred
from .curves import NSSModel

def download_yield_data(fred_api: 'Fred', series: Dict[str, str], start: str,
                        end: pd.Timestamp, maturity_map: dict) -> Tuple[pd.DataFrame, np.ndarray]:
    yield_data = pd.DataFrame()
    for maturity, series_id in series.items():
            s = fred_api.get_series(series_id, observation_start=start, observation_end=end)
            s.name = maturity
            yield_data = pd.concat([yield_data, s], axis=1)

    yield_data = yield_data.ffill().dropna() / 100
    ordered_cols = sorted(yield_data.columns, key=lambda x: maturity_map.get(x, float('inf')))
    yield_data = yield_data[ordered_cols]
    maturities_numeric = np.array([maturity_map[col] for col in ordered_cols])
    print("Data downloaded and cleaned.")
    return yield_data, maturities_numeric

def run_nss_calibration_cycle(yield_data: pd.DataFrame,
                          maturities_numeric: np.ndarray,
                          param_bounds: tuple) -> Tuple[Dict[pd.Timestamp, NSSModel], pd.DataFrame]:
    print("Starting NSS calibration cycle")
    calibrated_models = {}
    failed_dates = []

    first_yields = yield_data.iloc[0].values
    rate_short, rate_long = first_yields[0], first_yields[-1]
    rate_mean = np.mean(first_yields)
    slope = rate_short - rate_long

    p0_standard = np.array([first_yields[-1], slope, 0.0, 0.0, 1.5, 5.0])
    guess_humped = np.array([rate_long, slope, 0.05, -0.02, 1.0, 4.0])
    guess_flat = np.array([rate_mean, 0.0, 0.0, 0.0, 2.0, 9.0])

    guesses_to_try = [p0_standard, guess_humped, guess_flat]
    best_initial_mse = float('inf')
    current_params = p0_standard.copy()

    for guess in guesses_to_try:
        m_try, res_try = NSSModel.calibrate(maturities_numeric, first_yields, guess, param_bounds)
        if m_try and res_try.fun < best_initial_mse:
            best_initial_mse = res_try.fun
            current_params = m_try.params
    print(f"  Best initial MSE found: {best_initial_mse:.8f}")

    for date, row in yield_data.iterrows():
        model, result = NSSModel.calibrate(maturities_numeric, row.values, current_params, param_bounds)

        is_unstable = False
        if model:
            if model.b1 <= -0.145 or model.b1 >= 0.145: is_unstable = True
            if model.b2 >= 0.29 or model.b2 <= -0.19: is_unstable = True
            if model.l1 <= 0.15 or model.l1 >= 4.9: is_unstable = True

        if model and not is_unstable:
            calibrated_models[date] = model
            current_params = model.params
        else:
            rescue_model, rescue_res = NSSModel.calibrate(maturities_numeric, row.values, p0_standard, param_bounds)
            if rescue_model:
                calibrated_models[date] = rescue_model
                current_params = rescue_model.params
            else:
                print(f"ATTENTION: saved failed {date.date()}")
                failed_dates.append(date)
                current_params = p0_standard

    print(f"Calibration completed. Done: {len(calibrated_models)} | Fail: {len(failed_dates)}")

    if calibrated_models:
        params_data = {date: m.params for date, m in calibrated_models.items()}
        params_df = pd.DataFrame.from_dict(params_data, orient='index',columns=['b0', 'b1', 'b2', 'b3', 'l1', 'l2'])
        params_df['short_rate'] = params_df['b0'] + params_df['b1']
    else:
        params_df = pd.DataFrame()
    return calibrated_models, params_df

def calibrate_hw_parameters_from_history(short_rate_series: pd.Series, dt: float = 1.0/252.0) -> Tuple[float, float]:
    print(f"[CALIBRATION] Estimating Hull-White parameters from history ({len(short_rate_series)} obs)...")
    rates = short_rate_series.values
    dr = rates[1:] - rates[:-1]
    r_t = rates[:-1]

    slope, intercept = np.polyfit(r_t, dr, 1)
    kappa_est = -slope / dt
    sigma_est = np.std(dr - (intercept + slope * r_t)) / np.sqrt(dt)

    if kappa_est < 1e-4:
        print("[WARNING] Estimated Kappa is negative or null (Mean Aversion). Forced to 0.01.")
        kappa_est = 0.01

    print(f"Historical Kappa: {kappa_est:.4f} | Historical Sigma: {sigma_est:.4f}")
    return kappa_est, sigma_est

def calibration_objective_Q(params, market_swaptions, curve_funcs, r0_market):
    k_trial, s_trial = params
    if k_trial < 1e-4 or s_trial < 1e-4: return 1e9

    total_error = 0.0
    for swaption in market_swaptions:
        T_exp = swaption['expiry']
        vol_mkt = swaption['vol_normal']
        hw_vol_model = s_trial * (1.0 - np.exp(-k_trial * T_exp)) / k_trial
        total_error += (hw_vol_model - vol_mkt)**2
    return total_error
