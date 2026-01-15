import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from .curves import NSSModel # Needed for P_spot inside helper

def fwd_swap_rate_and_annuity(T_expiry, T_tenor, params_nss):
    payment_times = np.arange(T_expiry + 1, T_expiry + T_tenor + 1)
    annuity = np.sum(NSSModel.P_spot(payment_times, params_nss))
    p_start = NSSModel.P_spot(T_expiry, params_nss)
    p_end = NSSModel.P_spot(T_expiry + T_tenor, params_nss)
    fwd_swap_rate = (p_start - p_end) / (annuity + 1e-9)
    return fwd_swap_rate, annuity

def price_swaption_bachelier(F, K, T_expiry, vol_normal, annuity, option_type='payer'):
    d = (F - K) / (vol_normal * np.sqrt(T_expiry) + 1e-9)
    if option_type == 'payer':
        price = annuity * ((F - K) * norm.cdf(d) + vol_normal * np.sqrt(T_expiry) * norm.pdf(d))
    else:
        price = annuity * ((K - F) * norm.cdf(-d) + vol_normal * np.sqrt(T_expiry) * norm.pdf(-d))
    return price

def implied_vol_bachelier(target_price, F, K, T_expiry, annuity, option_type='payer'):
    def objective(vol):
        return price_swaption_bachelier(F, K, T_expiry, vol, annuity, option_type) - target_price
    try:
        return brentq(objective, 1e-6, 0.5)
    except ValueError:
        return 0.0
