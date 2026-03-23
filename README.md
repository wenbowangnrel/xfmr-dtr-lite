# transformer-dtr-lite

A lightweight Python implementation of the **IEEE C57.91-2011** thermal model for mineral-oil-immersed transformers. Computes top-oil temperature, hot-spot temperature, insulation aging rate, and thermal loading limit using the Annex C difference equations.

---

## What it does

| Capability | Function | Standard reference |
|---|---|---|
| Top-oil temperature (dynamic) | `top_oil_step()` | IEEE C57.91 Annex C Eq. C.1–C.2 |
| Hot-spot temperature (dynamic) | `hot_spot_step()` | IEEE C57.91 Annex C Eq. C.3–C.4 |
| Steady-state top-oil rise | `steady_state_top_oil_rise()` | IEEE C57.91 Eq. 2 |
| Steady-state hot-spot rise | `steady_state_hot_spot_rise()` | IEEE C57.91 Eq. 3 |
| Insulation aging rate F_AA | `aging_rate()` | IEEE C57.91 Eq. 1 |
| Loss of life over a period | `loss_of_life()` | IEEE C57.91 Section 5 |
| Thermal loading limit | `thermal_loading_limit()` | IEEE C57.91 Table 3 |

---

## Installation

```bash
git clone https://github.com/wenbowangnrel/transformer-dtr-lite.git
cd transformer-dtr-lite
pip install -e .
```

Install notebook dependencies (for `validation_with_IEEE_std.ipynb`):

```bash
pip install -r requirements.txt
```

---

## Quick start

```python
import json
from transformer_dtr_lite import top_oil_step, hot_spot_step, aging_rate

# load transformer parameters
with open("params/service_transformer.json") as f:
    params = json.load(f)

# simulate one hour at K=0.8, 30 °C ambient, starting from 30 °C
top_oil  = top_oil_step(params, load_factor=0.8, ambient_temp_c=30.0,
                         prev_top_oil_temp_c=30.0, dt_hours=1.0)
hot_spot = hot_spot_step(params, load_factor=0.8, top_oil_temp_c=top_oil,
                          prev_hot_spot_temp_c=30.0, dt_hours=1.0)

print(f"Top-oil:  {top_oil:.1f} °C")
print(f"Hot-spot: {hot_spot:.1f} °C")
print(f"F_AA:     {aging_rate(hot_spot):.4f}")
```

Run the full 24-hour example:

```bash
python example.py
```

---

## Transformer parameters

Pre-built parameter files are in `params/`. All values are from IEEE C57.91-2011.

| File | Rating | Cooling | Source |
|---|---|---|---|
| `ieee_annex_c_187mva_odaf.json` | 187 MVA | ODAF | IEEE C57.91 Annex C (validated) |
| `power_transformer.json` | 25 MVA | ONAN | IEEE Table 4, typical values |
| `service_transformer.json` | 500 kVA | ONAN | IEEE Table 4, typical values |
| `service_transformer_50kva.json` | 50 kVA | ONAN | IEEE Table 4, typical values |

All ONAN transformers are 65 °C-rise class: at K=1 and 30 °C ambient the hot-spot reaches exactly 110 °C (F_AA = 1.0), which is the IEEE normal aging reference point.

Key parameters that differ by cooling type (IEEE Table 4):

| Parameter | ONAN | ODAF |
|---|---|---|
| Oil exponent `n` | 0.8 | 1.0 |
| Winding exponent `y = 2m` | 1.6 | 2.0 |
| Time-constant correction `k11` | 0.5 | 1.0 |

See `params/transformer_params_comparison.csv` for a full side-by-side comparison of all four transformers.

---

## Validation

`validation_with_IEEE_std.ipynb` validates the model against three levels of reference data:

- **Section 1** — Analytical checks: exact Arrhenius values, rated steady-state rises
- **Section 2** — Convergence: dynamic simulation converges to steady-state
- **Section 2b** — ONAN design point: both ONAN transformers reach 110 °C at K=1
- **Section 3** — IEEE C57.91-2011 Annex C Table C.1: 24-hour Normal Load cycle, 187 MVA ODAF

RMS error vs IEEE Table C.1: **0.13 °C** (systematic positive bias from rounded Annex C coefficients).

---

## Project structure

```
transformer-dtr-lite/
├── src/transformer_dtr_lite/
│   ├── hotspot.py          # top-oil and hot-spot temperature steps
│   ├── aging.py            # F_AA aging rate and loss-of-life
│   └── thermal_limit.py    # thermal loading limit solver
├── params/
│   ├── ieee_annex_c_187mva_odaf.json
│   ├── power_transformer.json
│   ├── service_transformer.json
│   ├── service_transformer_50kva.json
│   └── transformer_params_comparison.csv
├── data/
│   └── example_profile.csv
├── validation_with_IEEE_std.ipynb
├── example.py
├── pyproject.toml
└── requirements.txt
```

---

## Reference

IEEE Std C57.91-2011, *IEEE Guide for Loading Mineral-Oil-Immersed Transformers and Step-Voltage Regulators*, IEEE, 2012.
