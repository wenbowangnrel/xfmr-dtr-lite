"""
hotspot.py - Top-oil and hot-spot temperature calculations.

Implements the IEEE C57.91-2011 Annex C difference equation thermal model.
Two levels of calculation are exposed:

  1. steady_state_top_oil_rise()   -- ultimate top-oil rise at a given load (no time)
  2. steady_state_hot_spot_rise()  -- ultimate hot-spot rise at a given load (no time)
  3. top_oil_step()                -- one time-step of top-oil temperature (dynamic)
  4. hot_spot_step()               -- one time-step of hot-spot temperature (dynamic)

Functions 1 and 2 answer: "where is the transformer heading?"
Functions 3 and 4 answer: "where is the transformer right now?"

The caller is responsible for the time loop. See example.py for usage.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# Steady-state (ultimate) calculations
# ---------------------------------------------------------------------------

def steady_state_top_oil_rise(params: dict, load_factor: float) -> float:
    """Ultimate top-oil rise over ambient at a given load factor.

    This is the temperature the top-oil would reach if load_factor were held
    constant forever. It does not account for thermal lag.

    IEEE C57.91-2011 Equation (2):
        Delta_theta_o = Delta_theta_or * ((1 + R * K^2) / (1 + R))^x

    Parameters
    ----------
    params      : transformer parameter dict loaded from params/*.json
    load_factor : per-unit load, e.g. 1.0 = rated, 1.2 = 120% overload

    Returns
    -------
    Top-oil rise over ambient in degC.
    """
    K = max(load_factor, 0.0)
    R = params["load_loss_ratio"]       # ratio of load loss to no-load loss
    x = params["oil_exponent"]          # oil exponent
    delta_theta_or = params["top_oil_rise_c"]  # rated top-oil rise

    return delta_theta_or * (((1.0 + R * K ** 2) / (1.0 + R)) ** x)


def steady_state_hot_spot_rise(params: dict, load_factor: float) -> float:
    """Ultimate hot-spot rise over top-oil at a given load factor.

    This is the hot-spot gradient the winding would reach if load_factor were
    held constant forever. It does not account for winding thermal lag.

    IEEE C57.91-2011 Equation (3):
        Delta_theta_h = Delta_theta_hr * H * K^y

    Parameters
    ----------
    params      : transformer parameter dict loaded from params/*.json
    load_factor : per-unit load

    Returns
    -------
    Hot-spot rise over top-oil in degC.
    """
    K = max(load_factor, 0.0)
    y = params["winding_exponent"]          # winding exponent
    delta_theta_hr = params["hot_spot_rise_c"]  # rated hot-spot rise
    H = params["hot_spot_factor"]           # hot-spot factor

    return delta_theta_hr * H * (K ** y)


# ---------------------------------------------------------------------------
# Dynamic (single time-step) calculations
# ---------------------------------------------------------------------------

def top_oil_step(
    params: dict,
    load_factor: float,
    ambient_temp_c: float,
    prev_top_oil_temp_c: float | None,
    dt_hours: float = 1.0,
) -> float:
    """Advance top-oil temperature by one time step.

    Uses the IEEE C57.91 Annex C exponential approach toward the
    ultimate (steady-state) top-oil temperature.

    If prev_top_oil_temp_c is None (first step), returns the steady-state
    temperature directly -- i.e. assumes the transformer starts fully
    thermally settled at the given load.

    IEEE C57.91-2011 Annex C difference equation:
        d(theta_o) = (dt / (k11 * tau_o)) * (theta_o_ultimate - theta_o_current)
    which is equivalent to the exponential:
        theta_o(t) = theta_o(t-1) + (1 - exp(-dt / (k11 * tau_o)))
                     * (theta_o_ultimate - theta_o(t-1))

    Parameters
    ----------
    params               : transformer parameter dict
    load_factor          : per-unit load for this time step
    ambient_temp_c       : ambient temperature in degC for this time step
    prev_top_oil_temp_c  : top-oil temperature at end of previous step (degC).
                           Pass None for the very first step.
    dt_hours             : time step size in hours (default 1.0)

    Returns
    -------
    Top-oil temperature in degC at end of this time step.
    """
    ultimate_rise = steady_state_top_oil_rise(params, load_factor)
    ultimate_temp = ambient_temp_c + ultimate_rise

    if prev_top_oil_temp_c is None:
        return ultimate_temp

    tau = params["thermal_time_constant_hours"]   # tau_o
    k11 = params["k11"]
    alpha = 1.0 - math.exp(-dt_hours / (k11 * tau))

    return prev_top_oil_temp_c + alpha * (ultimate_temp - prev_top_oil_temp_c)


def hot_spot_step(
    params: dict,
    load_factor: float,
    top_oil_temp_c: float,
    prev_hot_spot_temp_c: float | None,
    dt_hours: float = 1.0,
) -> float:
    """Advance hot-spot temperature by one time step.

    Uses the IEEE C57.91 Annex C exponential approach toward the
    ultimate (steady-state) hot-spot gradient over top-oil.

    If prev_hot_spot_temp_c is None (first step), returns the steady-state
    hot-spot temperature directly.

    IEEE C57.91-2011 Annex C difference equation:
        theta_h(t) = top_oil(t) + gradient(t)
        gradient(t) = gradient(t-1) + (1 - exp(-dt / (k22 * tau_w)))
                      * (gradient_ultimate - gradient(t-1))

    Note: top_oil_temp_c must be the value already computed for this same
    time step via top_oil_step().

    Parameters
    ----------
    params                 : transformer parameter dict
    load_factor            : per-unit load for this time step
    top_oil_temp_c         : top-oil temperature for this step (degC)
    prev_hot_spot_temp_c   : hot-spot temperature at end of previous step (degC).
                             Pass None for the very first step.
    dt_hours               : time step size in hours (default 1.0)

    Returns
    -------
    Hot-spot temperature in degC at end of this time step.
    """
    ultimate_gradient = steady_state_hot_spot_rise(params, load_factor)

    if prev_hot_spot_temp_c is None:
        return top_oil_temp_c + ultimate_gradient

    prev_gradient = prev_hot_spot_temp_c - top_oil_temp_c

    tau_w = params["winding_time_constant_hours"]  # tau_w
    k22 = params["k22"]
    alpha = 1.0 - math.exp(-dt_hours / (k22 * tau_w))

    gradient = prev_gradient + alpha * (ultimate_gradient - prev_gradient)

    return top_oil_temp_c + gradient
