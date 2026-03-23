"""
aging.py - Insulation aging and loss of life calculations.

Implements the IEEE C57.91-2011 Arrhenius-based aging model for oil-filled
transformers with thermally upgraded kraft paper insulation.

Two functions are exposed:

  1. aging_rate()    -- acceleration factor at a single hot-spot temperature
  2. loss_of_life()  -- total aging consumed over a hot-spot temperature series

The core idea is that insulation ages faster at higher temperatures following
an Arrhenius (exponential) relationship. IEEE C57.91 defines 110 C as the
reference hot-spot temperature where the aging rate equals 1.0 (normal rate).
Above 110 C the rate is greater than 1.0 (accelerated aging).
Below 110 C the rate is less than 1.0 (slower aging).

Rule of thumb from IEEE C57.91: every 6 C rise above 110 C doubles the aging rate.
"""

from __future__ import annotations

import math


# IEEE C57.91-2011 Section 7: reference hot-spot temperature for normal aging
REFERENCE_HOT_SPOT_TEMP_C = 110.0

# IEEE C57.91-2011 Section 7: Arrhenius activation energy constant for
# thermally upgraded kraft paper insulation
ACTIVATION_ENERGY_CONSTANT = 15000.0


def aging_rate(
    hot_spot_temp_c: float,
    reference_temp_c: float = REFERENCE_HOT_SPOT_TEMP_C,
    activation_energy: float = ACTIVATION_ENERGY_CONSTANT,
) -> float:
    """Aging acceleration factor at a given hot-spot temperature.

    Returns how many times faster (or slower) the insulation is aging
    compared to the normal rate at reference_temp_c.

    IEEE C57.91-2011 Equation (9) -- Arrhenius model:
        F_AA = exp(B / (theta_ref + 273) - B / (theta_h + 273))

    where B is the activation energy constant (15000 for upgraded kraft paper).

    Examples:
        aging_rate(110)  → 1.0   (normal rate, reference temperature)
        aging_rate(116)  → ~2.0  (aging twice as fast, +6 C rule)
        aging_rate(98)   → ~0.5  (aging half as fast, below reference)

    Parameters
    ----------
    hot_spot_temp_c : hot-spot temperature in degC for this time step
    reference_temp_c : reference hot-spot temperature in degC (default 110 C)
    activation_energy : Arrhenius constant B (default 15000)

    Returns
    -------
    Dimensionless aging acceleration factor (1.0 = normal rate).
    """
    return math.exp(
        activation_energy / (reference_temp_c + 273.0)
        - activation_energy / (hot_spot_temp_c + 273.0)
    )


def loss_of_life(
    hot_spot_temps_c: list[float],
    params: dict,
    dt_hours: float = 1.0,
) -> dict:
    """Compute insulation aging consumed over a hot-spot temperature series.

    Integrates the aging acceleration factor over time to get total equivalent
    aging hours consumed, then expresses this as a percentage of normal
    transformer life.

    IEEE C57.91-2011 Equation (10):
        Aging consumed = sum(F_AA(i) * dt) for all time steps i

    Parameters
    ----------
    hot_spot_temps_c : list of hot-spot temperatures in degC, one per time step
    params           : transformer parameter dict (used for normal_life_hours)
    dt_hours         : time step size in hours (default 1.0)

    Returns
    -------
    dict with three keys:
        "equivalent_aging_factor"  -- mean aging rate over the period (dimensionless)
        "consumed_life_hours"      -- equivalent aging hours consumed
        "loss_of_life_percent"     -- consumed hours as % of normal transformer life
    """
    factors = [aging_rate(t) for t in hot_spot_temps_c]

    n = len(factors)
    if n == 0:
        return {
            "equivalent_aging_factor": 0.0,
            "consumed_life_hours": 0.0,
            "loss_of_life_percent": 0.0,
        }

    mean_factor = sum(factors) / n
    consumed_hours = sum(factors) * dt_hours
    normal_life_hours = params["normal_life_hours"]
    loss_percent = 100.0 * consumed_hours / normal_life_hours

    return {
        "equivalent_aging_factor": mean_factor,
        "consumed_life_hours": consumed_hours,
        "loss_of_life_percent": loss_percent,
    }
