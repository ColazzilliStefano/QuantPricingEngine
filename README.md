# QuantPricingEngine: Hull-White 1-Factor Model

### **A Modular Python Framework for Interest Rate Derivatives Pricing**

This repository hosts a high-performance pricing engine based on the **Hull-White One-Factor Model**. It engineers the full interest rate modeling lifecycle: from **Yield Curve Construction** (Nelson-Siegel-Svensson) to **Model Calibration** ($\mathbb{P}$ vs $\mathbb{Q}$ measure), **Monte Carlo Simulation**, and **Finite Difference (PDE) Pricing**.

The project demonstrates a hybrid architecture capable of pricing linear products (Swaps), non-linear European options (Caps, Swaptions), and path-dependent American/Bermudan instruments (Callable Bonds).



## Project Context & Workflow

This library demonstrates a modern **AI-Augmented Development** approach, designed to maximize code robustness, performance, and architectural cleanliness.

* **Role of the Author (Financial Architect):** Defined the mathematical models, pricing strategies (e.g., Jamshidian decomposition, PDE boundary conditions), calibration logic, and performed the final financial validation of results.
* **Role of the AI (Code Optimization):** Utilized as an interactive coding partner to enforce strict **PEP-8 standards**, optimize numerical loops via **Numba/LLVM**, and implement clean **Model-View-Controller** patterns.

This workflow simulates a real-world quant desk environment where theoretical models must be rapidly translated into production-grade, bug-free code.



## Theoretical Framework

### 1. Yield Curve Construction (Static State)
The framework builds a continuous discount factor curve $P(0,T)$ from discrete market data using the **Nelson-Siegel-Svensson (NSS)** model. This ensures the differentiability required for the Hull-White drift term.

$$R(\tau) = \beta_0 + \beta_1 \frac{1-e^{-\tau/\lambda_1}}{\tau/\lambda_1} + \beta_2 \left(\frac{1-e^{-\tau/\lambda_1}}{\tau/\lambda_1} - e^{-\tau/\lambda_1}\right) + \beta_3 \left(\frac{1-e^{-\tau/\lambda_2}}{\tau/\lambda_2} - e^{-\tau/\lambda_2}\right)$$

* **Calibration Strategy:** Implements a **"Smart Rolling & Rescue"** algorithm. It uses $t_{-1}$ optimal parameters as the initial guess for time $t$. This guarantees parameter stability over time-series, preventing artificial P&L volatility caused by curve fitting jumps.

### 2. Hull-White Dynamics (Dynamic State)
The short rate $r_t$ evolves under the Risk-Neutral Measure $\mathbb{Q}$ according to:

$$dr_t = [\theta(t) - \kappa r_t]dt + \sigma dW_t$$

* **$\theta(t)$ (Drift Adjustment):** Calculated analytically from the NSS Forward Curve and its slope to ensure exact fit to current market prices (No-Arbitrage condition).
* **$\kappa$ (Mean Reversion):** Determines how fast rates return to the central tendency.
* **$\sigma$ (Volatility):** Determines the width of the rate distribution.



## Calibration Strategy: $\mathbb{P}$ vs $\mathbb{Q}$

A critical distinction is made between historical analysis and pricing requirements:

1.  **Historical Calibration ($\mathbb{P}$-Measure):**
    * **Purpose:** Regime Analysis and Risk Management.
    * **Method:** Linear regression on historical short-rate time series ($r_{t+\Delta t} - r_t$).
    * **Output:** Assesses the **Volatility Risk Premium** by benchmarking historical realizations against implied market pricing.

2.  **Implied Calibration ($\mathbb{Q}$-Measure):**
    * **Purpose:** Pricing (to match market prices of liquid derivatives).
    * **Method:** **Synthetic Volatility Surface**. The calibration module is agnostic to data sources; it accepts a volatility surface (e.g., Swaption Matrix) and minimizes the Least Squares difference using **Bachelier's formula** and **Jamshidian's Decomposition**.


## Hybrid Pricing Architecture

The library implements a **triangulation approach** to pricing, selecting the optimal numerical method based on the instrument's payoff structure.

| Method | Instrument Scope | Rationale |
| :--- | :--- | :--- |
| **Analytical** | ZCBs, Caps/Floors, Swaptions | Uses **Jamshidian's Decomposition** for Swaptions. Closed-form solutions eliminate variance error and reduce compute time to microseconds. |
| **Monte Carlo** | Path-Dependent / Complex Payoffs | **Numba JIT-compiled** engine. Implements both **Euler-Maruyama** and **Exact** discretization schemes. Bypasses Python GIL to achieve C++ level performance on large-scale simulations ($10^5+$ paths). |
| **PDE Solver** | Bermudan / American (Callable Bonds) | **Crank-Nicolson** finite difference scheme on a spatial grid. Provides deterministic stability for backward induction, capturing early-exercise boundaries with superior accuracy compared to regression-based MC. |



## Validation & QA

The model includes a professional validation suite to ensure mathematical consistency:

* **Martingale Test:** Verifies that the drift of the simulated rates matches the theoretical forward curve (Drift Consistency). *Typical Error target: < 1 bps.*
* **Variance Ratio:** Compares the Monte Carlo variance against the analytical variance $\frac{\sigma^2}{2\kappa}(1-e^{-2\kappa t})$.
* **Pricing Convergence:** Compares MC Price vs Analytical Price for Zero Coupon Bonds.



## Repository Structure
### Tech Stack
* **Core:** Python 3.x
* **Numerics:** NumPy, SciPy
* **HPC:** Numba (LLVM-based JIT Compilation)
* **Data:** Pandas, FredAPI (Federal Reserve Economic Data)
* **Visualization:** Matplotlib



### Disclaimer
This software is for educational and academic purposes. It is not intended for use in live trading environments without further rigorous testing and data integration.
