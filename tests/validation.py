import numpy as np

def run_professional_unit_tests(engine, T_val=5.0):
    print("-" * 60)
    print("[7] UNIT TEST SUITE - MODEL VALIDATION")
    print("-" * 60)
    STEPS_PER_YEAR = 252

    print("[TEST 1] Martingale Check (Forward Rate Consistency)")
    T_sim = T_val
    steps = int(STEPS_PER_YEAR * T_sim)
    paths = 10000
    rates, t_grid, _ = engine.run_simulation_exact(T_sim, steps, paths)
    mc_mean_T = np.mean(rates[-1, :])
    analytical_mean_curve = engine._get_analytical_mean_vector(t_grid)
    theory_mean_T = analytical_mean_curve[-1]
    diff_bps = (mc_mean_T - theory_mean_T) * 10000

    print(f"   -> MC Mean  {T_val}Y:     {mc_mean_T:.6f}")
    print(f"   -> Theory Mean  {T_val}Y: {theory_mean_T:.6f}")
    print(f"   -> Divergence:         {diff_bps:.4f} bps")
    if abs(diff_bps) < 1.0: print("   -> STATUS: PASSED (Drift is correct)")
    else: print("   -> STATUS: FAILED (Drift bias detected)")

    print("\n[TEST 2] Volatility/Variance Check")
    mc_var_T = np.var(rates[-1, :])
    k, s = engine.kappa, engine.sigma
    theory_var_T = (s**2 / (2*k)) * (1 - np.exp(-2*k*T_val))
    ratio = mc_var_T / theory_var_T

    print(f"   -> MC Variance:     {mc_var_T:.8f}")
    print(f"   -> Theory Variance: {theory_var_T:.8f}")
    print(f"   -> Ratio (Target 1.0): {ratio:.4f}")
    if 0.95 < ratio < 1.05: print("   -> STATUS: PASSED (Diffusion is correct)")
    else: print("   -> STATUS: WARNING (Check Random Number Generator)")

    print("\n[TEST 3] ZCB Pricing")
    P_mc, stderr = engine.price_zcb_mc(T_val, steps, 10000, use_exact_scheme=True)
    relative_err = stderr / P_mc
    print(f"   -> ZCB Price (MC): {P_mc:.6f}")
    print(f"   -> Std Error:      {stderr:.6f} ({relative_err:.2%})")
    if relative_err < 0.001: print("   -> STATUS: PASSED (Convergence is tight)")
    else: print("   -> STATUS: WEAK (Need more paths)")
