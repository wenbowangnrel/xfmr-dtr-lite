"""
example.py - Runs both IEEE example transformers through a 24-hour profile.

Loads transformer parameters from params/*.json and the operating profile
from data/example_profile.csv, then:

  1. Steps through each hour computing top-oil and hot-spot temperatures
  2. Computes the thermal loading limit at each hour
  3. Computes loss of life over the full 24-hour period
  4. Prints a summary table
  5. Saves a plot to data/example_results.png

Run from the repo root:
    python example.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from transformer_dtr_lite.hotspot import top_oil_step, hot_spot_step
from transformer_dtr_lite.thermal_limit import thermal_loading_limit
from transformer_dtr_lite.aging import loss_of_life


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PARAMS_DIR = ROOT / "params"
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_params(filename: str) -> dict:
    """Load transformer parameters from a JSON file in params/."""
    with open(PARAMS_DIR / filename) as f:
        return json.load(f)


def load_profile(filename: str) -> list[dict]:
    """Load operating profile from a CSV file in data/."""
    rows = []
    with open(DATA_DIR / filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "hour": int(row["hour"]),
                "load_factor": float(row["load_factor"]),
                "ambient_temp_c": float(row["ambient_temp_c"]),
                "solar_radiation_w_m2": float(row["solar_radiation_w_m2"]),
                "wind_speed_m_s": float(row["wind_speed_m_s"]),
            })
    return rows


def simulate(params: dict, profile: list[dict], dt_hours: float = 1.0) -> list[dict]:
    """Run the transformer through the profile, one hour at a time.

    This loop is intentionally written out here rather than hidden inside
    a library function, so you can see exactly how the time-stepping works:

        for each hour:
            1. compute top-oil temperature (chasing ultimate value with tau_o)
            2. compute hot-spot temperature (chasing ultimate value with tau_w)
            3. compute thermal loading limit at this ambient

    The previous temperatures are passed forward each step -- that is the
    transformer's thermal memory.
    """
    results = []
    prev_top_oil = None
    prev_hot_spot = None

    for row in profile:
        load = row["load_factor"]
        ambient = row["ambient_temp_c"]

        # step 1: top-oil temperature this hour
        top_oil = top_oil_step(
            params=params,
            load_factor=load,
            ambient_temp_c=ambient,
            prev_top_oil_temp_c=prev_top_oil,
            dt_hours=dt_hours,
        )

        # step 2: hot-spot temperature this hour
        hot_spot = hot_spot_step(
            params=params,
            load_factor=load,
            top_oil_temp_c=top_oil,
            prev_hot_spot_temp_c=prev_hot_spot,
            dt_hours=dt_hours,
        )

        # step 3: thermal loading limit at this ambient (steady-state ceiling)
        limit = thermal_loading_limit(
            params=params,
            ambient_temp_c=ambient,
        )

        results.append({
            "hour": row["hour"],
            "load_factor": load,
            "ambient_temp_c": ambient,
            "top_oil_temp_c": top_oil,
            "hot_spot_temp_c": hot_spot,
            "thermal_limit_pu": limit,
        })

        # carry forward thermal state to next step
        prev_top_oil = top_oil
        prev_hot_spot = hot_spot

    return results


def print_summary(name: str, results: list[dict], aging: dict) -> None:
    """Print a summary table to the console."""
    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    print(f"  {'Hour':>4}  {'Load':>6}  {'Ambient':>8}  {'Top-Oil':>8}  {'Hot-Spot':>9}  {'Limit':>7}")
    print(f"  {'':>4}  {'[pu]':>6}  {'[°C]':>8}  {'[°C]':>8}  {'[°C]':>9}  {'[pu]':>7}")
    print(f"  {'-' * 57}")
    for r in results:
        print(
            f"  {r['hour']:>4}  {r['load_factor']:>6.2f}  "
            f"{r['ambient_temp_c']:>8.1f}  {r['top_oil_temp_c']:>8.1f}  "
            f"{r['hot_spot_temp_c']:>9.1f}  {r['thermal_limit_pu']:>7.2f}"
        )
    print(f"  {'-' * 57}")
    print(f"  Equivalent aging factor : {aging['equivalent_aging_factor']:.4f}")
    print(f"  Consumed life hours     : {aging['consumed_life_hours']:.4f} h")
    print(f"  Loss of life            : {aging['loss_of_life_percent']:.6f} %")


def plot_results(
    profile: list[dict],
    power_results: list[dict],
    service_results: list[dict],
    output_path: Path,
) -> None:
    """Save a three-panel plot of temperatures, loading limits, and load profile."""
    hours = [r["hour"] for r in profile]
    ambient = [r["ambient_temp_c"] for r in profile]
    load = [r["load_factor"] for r in profile]

    power_top_oil = [r["top_oil_temp_c"] for r in power_results]
    power_hot_spot = [r["hot_spot_temp_c"] for r in power_results]
    power_limit = [r["thermal_limit_pu"] for r in power_results]

    service_top_oil = [r["top_oil_temp_c"] for r in service_results]
    service_hot_spot = [r["hot_spot_temp_c"] for r in service_results]
    service_limit = [r["thermal_limit_pu"] for r in service_results]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # panel 1: temperatures
    axes[0].plot(hours, ambient, label="Ambient", color="tab:green", linestyle="--")
    axes[0].plot(hours, power_top_oil, label="Power top-oil", color="tab:blue")
    axes[0].plot(hours, service_top_oil, label="Service top-oil", color="tab:orange")
    axes[0].set_ylabel("Temperature [°C]")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Transformer DTR Lite — IEEE C57.91 Example Parameters")
    axes[0].grid(True, alpha=0.3)

    # panel 2: hot-spot temperatures
    axes[1].plot(hours, power_hot_spot, label="Power hot-spot", color="tab:blue")
    axes[1].plot(hours, service_hot_spot, label="Service hot-spot", color="tab:orange")
    axes[1].axhline(110, linestyle=":", color="black", linewidth=1, label="110°C reference")
    axes[1].set_ylabel("Hot-spot [°C]")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # panel 3: load profile vs thermal limits
    axes[2].plot(hours, load, label="Applied load", color="tab:gray", linewidth=2)
    axes[2].plot(hours, power_limit, label="Power thermal limit", color="tab:blue", linestyle="--")
    axes[2].plot(hours, service_limit, label="Service thermal limit", color="tab:orange", linestyle="--")
    axes[2].set_ylabel("Per-unit [pu]")
    axes[2].set_xlabel("Hour of day")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # load IEEE example parameters
    power_params = load_params("power_transformer.json")
    service_params = load_params("service_transformer.json")

    # load 24-hour operating profile
    profile = load_profile("example_profile.csv")

    # simulate both transformers
    power_results = simulate(power_params, profile)
    service_results = simulate(service_params, profile)

    # compute loss of life over the 24-hour period
    power_aging = loss_of_life(
        hot_spot_temps_c=[r["hot_spot_temp_c"] for r in power_results],
        params=power_params,
    )
    service_aging = loss_of_life(
        hot_spot_temps_c=[r["hot_spot_temp_c"] for r in service_results],
        params=service_params,
    )

    # print summary tables
    print_summary("IEEE Power Transformer — 25 MVA ONAN", power_results, power_aging)
    print_summary("IEEE Service Transformer — 500 kVA ONAN", service_results, service_aging)

    # save plot
    plot_results(profile, power_results, service_results, DATA_DIR / "example_results.png")


if __name__ == "__main__":
    main()
