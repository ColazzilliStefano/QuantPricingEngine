import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, List
import pandas as pd

class NSSVisualizer:
    def __init__(self, chart_dir: str = "charts"):
        self.chart_dir = chart_dir
        if not os.path.exists(self.chart_dir):
            os.makedirs(self.chart_dir)

    def _save_and_show(self, fig: plt.Figure, filename: str):
        path = os.path.join(self.chart_dir, filename)
        try:
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"[IO] Chart saved: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save {filename}: {e}")
        plt.close(fig)

    def plot_nss_fit(self, model, current_yields: np.ndarray, maturities: np.ndarray, date_str: str):
        print(f"Generating NSS Fit chart for {date_str}")
        tau_smooth = np.linspace(min(maturities), max(maturities), 500)
        yields_nss_smooth = model.predict(tau_smooth)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(maturities, current_yields * 100, 'o', label='Market Data (Observed)', color='navy', markersize=8, alpha=0.7)
        ax.plot(tau_smooth, yields_nss_smooth * 100, '-', label='NSS Theoretical Fit', color='firebrick', linewidth=2)
        ax.set_title(f'Nelson-Siegel-Svensson Yield Curve Fit ({date_str})', fontweight='bold')
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Yield (%)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        self._save_and_show(fig, "1_nss_fit.png")

    def plot_fit_residuals(self, model, current_yields: np.ndarray, maturities: np.ndarray, maturity_labels: List[str], date_str: str):
        print(f"Analyzing NSS Fit Residuals for {date_str}")
        model_yields = model.predict(maturities)
        errors_bps = (current_yields - model_yields) * 10000
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(maturity_labels, errors_bps, width=0.6, color='darkred', alpha=0.7, edgecolor='black')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_title(f"Calibration Error (Residuals) by Maturity ({date_str})", fontweight='bold')
        ax.set_ylabel("Error (bps) [Market - Model]")
        mae = np.mean(np.abs(errors_bps))
        ax.text(0.02, 0.95, f"MAE: {mae:.2f} bps", transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self._save_and_show(fig, "2_nss_residuals.png")

class HullWhiteVisualizer:
    def __init__(self, chart_dir: str = "charts"):
        self.chart_dir = chart_dir
        if not os.path.exists(self.chart_dir):
            os.makedirs(self.chart_dir)

    def _save_and_show(self, fig, filename):
        path = os.path.join(self.chart_dir, filename)
        try:
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Plot saved in: {path}")
        except Exception as e:
            print(f"[WARNING] Impossible save the plot: {e}")
        plt.close(fig)

    def plot_simulation_results(self, simulated_rates: np.ndarray, analysis: Dict[str, Any], time_axis: np.ndarray, T: float, num_paths_to_plot: int = 500):
        print("Generation of Monte Carlo simulation plot")
        y_min, y_max = -0.025, 0.1
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [3, 1]})
        plot_paths = min(simulated_rates.shape[1], num_paths_to_plot)
        axes[0].set_ylim(y_min, y_max)
        axes[0].plot(time_axis, simulated_rates[:, :plot_paths], alpha=0.1, linewidth=0.8)
        axes[0].plot(time_axis, analysis["analytical_mean"], color='red', linestyle='--', linewidth=2.5, label='Analytical Mean E[r(t)]')
        axes[0].plot(time_axis, analysis["simulated_mean"], color='black', linestyle=':', linewidth=2.5, label='Simulazion Mean MC')
        axes[0].set_title("Hull-White: Monte Carlo Simulation & Mean Reversion")
        axes[0].legend(loc='upper left')
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[1].set_ylim(y_min, y_max)
        axes[1].hist(analysis["final_rates"], bins=70, density=True, orientation='horizontal', alpha=0.7, color='skyblue', edgecolor='white')
        axes[1].set_title(f"Final Distribution (T={T}y)")
        axes[1].axhline(analysis["final_mean"], color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {analysis["final_mean"]:.2%}')
        axes[1].legend(loc='upper right')
        plt.tight_layout()
        self._save_and_show(fig, "hw_mc_simulation_results.png")

    def plot_instrument_sensitivity(self, engine, instrument_type: str = "Swap"):
        print(f"Sensitivity Analysis (PV vs r0) for: {instrument_type}")
        r_min, r_max = -0.05, 0.10
        r_range = np.linspace(r_min, r_max, 100)
        prices = []
        original_r0 = engine.r0
        T_test, Notional = 5.0, 100.0
        try:
            for r_test in r_range:
                engine.r0 = r_test
                if instrument_type == "ZCB": p = engine.price_zcb_analytic(0, T_test) * Notional
                elif instrument_type == "Coupon Bond": p = engine.price_coupon_bond(0, np.arange(1.0, T_test + 0.01, 1.0), np.full(len(np.arange(1.0, T_test + 0.01, 1.0)), 0.03 * Notional), face_value=Notional)
                elif instrument_type == "Swap": p = engine.price_irs(0, 0, T_test, fixed_rate=0.03, notional=Notional, is_payer=True)
                elif instrument_type == "Cap": p = engine.price_cap_floor(0, 0, T_test, strike=0.04, notional=Notional, is_cap=True)
                elif instrument_type == "Swaption": p = engine.price_swaption_jamshidian(T_expiry=1.0, T_tenor=5.0, K_strike=0.03) * Notional
                elif instrument_type == "Callable":
                    p = engine.price_callable_bond_pde(10.0, np.arange(1.0, 11.0), np.full(10, 0.05 * Notional), np.arange(5.0, 10.0), Notional, face_value=Notional, Nspace=200, Ntime=1000)
                else: raise ValueError
                prices.append(p)
        except Exception as e: print(f"[ERROR] {e}")
        finally: engine.r0 = original_r0

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r_range, prices, lw=2.5, color='blue', label=f"PV {instrument_type}")
        ax.axvline(original_r0, color='red', ls='--', alpha=0.8, label=f"Current r0 ({original_r0:.2%})")
        ax.set_title(f"Price Sensitivity Analysis: {instrument_type}", fontsize=14)
        ax.set_xlabel("Instantaneous Short Rate (r0)")
        ax.set_ylabel("Present Value (€)")
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
        self._save_and_show(fig, f"hw_sensitivity_{instrument_type}.png")
