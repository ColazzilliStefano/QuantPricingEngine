import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional

class NSSModel:
    """
    Encapsulates the logic of the Nelson-Siegel-Svensson (NSS) model
    and its derived functions (ZCB prices, forward rates).
    """
    def __init__(self, b0: float, b1: float, b2: float, b3: float, l1: float, l2: float):
        # params vector: [b0, b1, b2, b3, l1, l2]
        # beta0 (b0): Long-term level (limit as tau -> infinity)
        # beta1 (b1): Short-term component (slope)
        # beta2 (b2): Medium-term curvature (hump)
        # beta3 (b3): Second curvature parameter (Svensson extension)
        # lambda1 (l1): Decay factor for the first hump
        # lambda2 (l2): Decay factor for the second hump
        self.b0, self.b1, self.b2, self.b3, self.l1, self.l2 = b0, b1, b2, b3, l1, l2

    @property   #convenience allowing to always attribute the same value to the function, allows writing 'params' instead of 'params()' later
    def params(self) -> np.ndarray:  #defines parameters in a matrix
        return np.array([self.b0, self.b1, self.b2, self.b3, self.l1, self.l2])

    @property
    def short_rate(self) -> float:  # short rate as instantaneous rate today.
        return self.b0 + self.b1

    @property
    def long_rate(self) -> float:  #Long Rate
        return self.b0


# 2.CORE MATHEMATICAL

    @staticmethod  #allows performing generic pure operations, without considering case data
    def nss_yield(tau: np.ndarray, b0: float, b1: float, b2: float, b3: float, l1: float, l2: float) -> np.ndarray: #heart of the model. Defines the nss curve as a positive nd matrix given by the sum of the 4 terms
        tau = np.where(tau == 0, 1e-6, tau)
        t1 = b0
        t2 = b1 * ((1 - np.exp(-tau / l1)) / (tau / l1))
        t3 = b2 * ((1 - np.exp(-tau / l1)) / (tau / l1) - np.exp(-tau / l1))
        t4 = b3 * ((1 - np.exp(-tau / l2)) / (tau / l2) - np.exp(-tau / l2))
        return t1 + t2 + t3 + t4


    def predict(self, tau: np.ndarray) -> np.ndarray:
        return self.nss_yield(tau, self.b0, self.b1, self.b2, self.b3, self.l1, self.l2)


    @staticmethod
    def _objective_function(params: np.ndarray, maturities: np.ndarray, market_yields: np.ndarray) -> float:
        """
        Objective function minimized by the algorithm.
        Calculates the Weighted Sum of Squared Errors (Weighted SSE).
        Logic: Inverse Duration Weighting (1/Maturity).
        """
        model_yields = NSSModel.nss_yield(maturities, *params)  # 1. Calculation of model yields with current parameters
        residuals = model_yields - market_yields  # 2. Calculation of residuals (Pure error)
        weights = 1.0 / (maturities + 1e-6)  # Weights based on inverse of maturity (Duration Proxy)
        weights = weights / np.mean(weights)  # Weight normalization
        weighted_error = np.sum(weights * residuals**2)  # 3. Calculation Weighted SSE

        return weighted_error


    @classmethod  #ensures the function below receives not the instance, but the class object -> builds the model (class)
    def calibrate(cls, maturities: np.ndarray, market_yields: np.ndarray,
                  p0_initial: np.ndarray, bounds: tuple) -> Tuple[Optional['NSSModel'], 'OptimizeResult']:

            result = minimize(cls._objective_function, p0_initial, args=(maturities, market_yields),
                            method='L-BFGS-B', bounds=bounds)
            if result.success:
                return cls(*result.x), result
            else:
                return None, result


# 3.FINANCIAL OUTPUT

    @staticmethod
    def P_spot(T: np.ndarray, params_nss: np.ndarray) -> np.ndarray:   #spot rate function
        T = np.where(T == 0, 1e-6, T)
        yield_spot = NSSModel.nss_yield(T, *params_nss)
        return np.exp(-yield_spot * T)

    @staticmethod
    def f_forward(T: np.ndarray, params_nss: np.ndarray) -> np.ndarray:  #foward rate funciton
        b0, b1, b2, b3, l1, l2 = params_nss
        T = np.where(T == 0, 1e-6, T)
        t1 = b0
        t2 = b1 * np.exp(-T / l1)
        t3 = b2 * (T / l1) * np.exp(-T / l1)
        t4 = b3 * (T / l2) * np.exp(-T / l2)
        return t1 + t2 + t3 + t4

    @staticmethod
    def f_forward_prime(T: np.ndarray, params_nss: np.ndarray) -> np.ndarray:  #function of forward rate prime
        b0, b1, b2, b3, l1, l2 = params_nss
        T = np.where(T == 0, 1e-6, T)
        dt1 = -b1/l1 * np.exp(-T/l1)
        dt2 = (b2/l1) * np.exp(-T/l1) - (b2*T/l1**2) * np.exp(-T/l1)
        dt3 = (b3/l2) * np.exp(-T/l2) - (b3*T/l2**2) * np.exp(-T/l2)
        return dt1 + dt2 + dt3
