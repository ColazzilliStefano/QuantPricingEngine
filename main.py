import numpy as np
import pandas as pd
from scipy.optimize import minimize
from google.colab import userdata
from fredapi import Fred
import config
from src.market_data import download_yield_data, run_nss_calibration_cycle, calibrate_hw_parameters_from_history, calibration_objective_Q
from src.curves import NSSModel
from src.models import HullWhiteEngine
from src.visualization import NSSVisualizer, HullWhiteVisualizer
from tests.validation import run_professional_unit_tests

def main():
    print("="*80)
    print("   QUANTITATIVE PRICING ENGINE - HULL-WHITE (Q-MEASURE)")
    print(f"   Output Directory: {config.CHART_DIR}")
    print("="*80)

    # 1. DATA INGESTION
    print("\n[1] DATA INGESTION & YIELD CURVE CONSTRUCTION")
    fred = None
    try:
        api_key = userdata.get('FRED_API_KEY')
        if api_key:
            fred = Fred(api_key=api_key)
            print("    -> FRED API Connected.")
        else: print("    -> WARNING: FRED API Key missing.")
    except Exception as e: print(f"    -> WARNING: FRED Connection issue ({e})")

    end_date = pd.Timestamp.now()
    try:
        yield_data, maturities_numeric = download_yield_data(fred, config.SERIES_IDS, config.START_DATE, end_date, config.MATURITY_MAP)
        print(f"    -> Data Loaded: {len(yield_data)} records.")
    except Exception as e:
        print(f"    -> CRITICAL ERROR: Data Download Failed. {e}")
        return

    # NSS CALIBRATION
    try:
        calibrated_models, params_df = run_nss_calibration_cycle(yield_data, maturities_numeric, config.PARAM_BOUNDS)
    except TypeError: return

    if not calibrated_models: return

    latest_date = yield_data.index[-1]
    latest_model = calibrated_models[latest_date]
    params_nss_safe = np.array(latest_model.params).flatten()
    print(f"    -> NSS Curve Fitted @ {latest_date.date()}")

    # VISUALIZATION
    try:
        nss_viz = NSSVisualizer(config.CHART_DIR)
        latest_yields = yield_data.loc[latest_date].values
        date_str = str(latest_date.date())
        maturity_labels = yield_data.columns.tolist()
        nss_viz.plot_nss_fit(latest_model, latest_yields, maturities_numeric, date_str)
        nss_viz.plot_fit_residuals(latest_model, latest_yields, maturities_numeric, maturity_labels, date_str)
        print("    -> PLOTS: Saved.")
    except: pass

    # 2. HULL-WHITE CALIBRATION
    print("\n[2] HULL-WHITE CALIBRATION")
    short_rate_col = '3M' if '3M' in yield_data.columns else yield_data.columns[0]
    r0_market = yield_data.iloc[-1][short_rate_col]

    try: nss_r0 = NSSModel.nss_yield(np.array([0.0001]), *params_nss_safe)[0]
    except: nss_r0 = 0.04
    spread = r0_market - nss_r0
    print(f"    -> Spread Adjustment: {spread*10000:.2f} bps")

    # Implied Vol Calibration
    calibration_basket = [
        {'expiry': 1.0, 'tenor': 5.0, 'vol_normal': 0.0085},
        {'expiry': 5.0, 'tenor': 5.0, 'vol_normal': 0.0080},
        {'expiry': 10.0,'tenor': 10.0,'vol_normal': 0.0075}
    ]
    def curve_spot_dummy(t, *args): return NSSModel.P_spot(t, params_nss_safe) * np.exp(-spread*t)

    res_Q = minimize(calibration_objective_Q, [0.03, 0.01], args=(calibration_basket, {'spot': curve_spot_dummy}, r0_market),
                      method='L-BFGS-B', bounds=((0.001, 2.0), (0.0001, 0.20)))
    kappa_implied, sigma_implied = res_Q.x
    print(f"    -> CALIBRATED PARAMETERS (PRICING): Kappa={kappa_implied:.4f}, Sigma={sigma_implied:.4f}")

    # 3. ENGINE INITIALIZATION
    def shifted_forward_curve(t, *args): return NSSModel.f_forward(t, params_nss_safe) + spread
    def shifted_spot_curve(t, *args): return NSSModel.P_spot(t, params_nss_safe) * np.exp(-spread * t)
    def shifted_forward_prime(t, *args): return NSSModel.f_forward_prime(t, params_nss_safe)

    engine = HullWhiteEngine(kappa_implied, sigma_implied, shifted_forward_curve, shifted_forward_prime, shifted_spot_curve, params_nss_safe, r0_market)

    # 4. SIMULATION & PRICING
    print("\n[4] SIMULATION & PRICING")
    T_sim = 10
    simulated_rates, time_axis, analytical_mean = engine.run_simulation_exact(T=T_sim, num_steps=int(252*T_sim), num_paths=10000)

    hw_viz = HullWhiteVisualizer(config.CHART_DIR)
    analysis = engine.analyze_results(simulated_rates, time_axis, analytical_mean)
    hw_viz.plot_simulation_results(simulated_rates, analysis, time_axis, T_sim)

    # Pricing Examples
    print(f"    -> Swaption 1Yx5Y (3%):  {engine.price_swaption_jamshidian(1.0, 5.0, 0.03):.4f}")
    print("    -> Pricing Callable Bond (10Y, 5%, Call@100 from Y5)...")
    times = np.arange(1.0, 11.0)
    val_call = engine.price_callable_bond_pde(10.0, times, np.full(10, 5.0), np.arange(5.0,10.0), 100.0, Nspace=100, Ntime=500)
    print(f"       Value: {val_call:.4f}")

    # Plots
    try:
        hw_viz.plot_instrument_sensitivity(engine, "Swaption")
        hw_viz.plot_instrument_sensitivity(engine, "Callable")
    except: pass

    # Unit Tests
    run_professional_unit_tests(engine)

if __name__ == "__main__":
    main()
