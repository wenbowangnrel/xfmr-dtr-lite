"""
thermal_limit.py - Thermal loading limit calculation.

Answers the question: given current ambient temperature, what is the highest
load factor this transformer can sustain without exceeding its thermal limits?

This uses steady-state (ultimate) temperatures only -- it finds the load where
the transformer would eventually settle at exactly the limit. It does not
account for how long the transformer can exceed the limit transiently before
the thermal mass runs out (that is captured by the time-stepping in hotspot.py).

IEEE C57.91-2011 loading categories used as default limits:
  Normal:              max hot-spot 110 C, max top-oil 95 C
  Planned overload:    max hot-spot 130 C, max top-oil 105 C
  Long-time emergency: max hot-spot 140 C, max top-oil 110 C
  Short-time emergency: max hot-spot 180 C, max top-oil 115 C

See params/*.json for the limits stored with each transformer.
"""

from __future__ import annotations

import numpy as np

from .hotspot import steady_state_top_oil_rise, steady_state_hot_spot_rise


def thermal_loading_limit(
    params: dict,
    ambient_temp_c: float,
    max_hot_spot_temp_c: float | None = None,
    max_top_oil_temp_c: float | None = None,
    search_min: float = 0.1,
    search_max: float = 3.0,
    search_step: float = 0.01,
) -> float:
    """Find the highest steady-state load factor within thermal limits.

    Sweeps load factor from search_min to search_max in steps of search_step.
    Returns the last load factor where both top-oil and hot-spot stay within
    their limits. Stops as soon as either limit is exceeded.

    If max_hot_spot_temp_c or max_top_oil_temp_c are not provided, the values
    stored in the params dict (max_hot_spot_temp_c, max_top_oil_temp_c) are
    used -- these correspond to normal loading limits from IEEE C57.91 Table 3.

    Parameters
    ----------
    params              : transformer parameter dict loaded from params/*.json
    ambient_temp_c      : current ambient temperature in degC
    max_hot_spot_temp_c : hot-spot temperature limit in degC (optional)
    max_top_oil_temp_c  : top-oil temperature limit in degC (optional)
    search_min          : lowest load factor to consider (default 0.1 pu)
    search_max          : highest load factor to consider (default 3.0 pu)
    search_step         : resolution of the sweep (default 0.01 pu)

    Returns
    -------
    Highest load factor in per-unit that keeps both temperatures within limits.
    """
    if max_hot_spot_temp_c is None:
        max_hot_spot_temp_c = params["max_hot_spot_temp_c"]
    if max_top_oil_temp_c is None:
        max_top_oil_temp_c = params["max_top_oil_temp_c"]

    load_factors = np.arange(search_min, search_max + search_step, search_step)
    last_ok = search_min

    for K in load_factors:
        top_oil_temp = ambient_temp_c + steady_state_top_oil_rise(params, float(K))
        hot_spot_temp = top_oil_temp + steady_state_hot_spot_rise(params, float(K))

        if top_oil_temp <= max_top_oil_temp_c and hot_spot_temp <= max_hot_spot_temp_c:
            last_ok = float(K)
        else:
            break

    return last_ok
