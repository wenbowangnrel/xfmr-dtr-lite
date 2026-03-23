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


def simulate_fine(params: dict, profile: list[dict], dt_minutes: float = 5.0) -> list[dict]:
    """High-resolution simulation (sub-hourly steps) to visualise thermal lag.

    The profile is hourly; this function sub-steps each hour at dt_minutes
    resolution, holding load and ambient constant within each hour (step
    function). The result is a fine-grained time series that shows the
    gradual temperature rise lagging behind each instantaneous load change.

    Parameters
    ----------
    params      : transformer parameter dict
    profile     : hourly operating profile from load_profile()
    dt_minutes  : time step size in minutes (default 5 min)

    Returns
    -------
    List of dicts with keys: time_hours, load_factor, hot_spot_temp_c
    """
    dt_hours = dt_minutes / 60.0
    steps_per_hour = round(1.0 / dt_hours)
    results = []
    prev_top_oil = None
    prev_hot_spot = None

    for row in profile:
        load = row["load_factor"]
        ambient = row["ambient_temp_c"]
        hour_start = float(row["hour"])

        for i in range(steps_per_hour):
            t = hour_start + i * dt_hours

            top_oil = top_oil_step(
                params=params,
                load_factor=load,
                ambient_temp_c=ambient,
                prev_top_oil_temp_c=prev_top_oil,
                dt_hours=dt_hours,
            )
            hot_spot = hot_spot_step(
                params=params,
                load_factor=load,
                top_oil_temp_c=top_oil,
                prev_hot_spot_temp_c=prev_hot_spot,
                dt_hours=dt_hours,
            )

            results.append({
                "time_hours": t,
                "load_factor": load,
                "hot_spot_temp_c": hot_spot,
            })

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
    power_fine: list[dict] | None = None,
    service_fine: list[dict] | None = None,
    daily_profile: list[dict] | None = None,
    daily_power_results: list[dict] | None = None,
    daily_service_results: list[dict] | None = None,
) -> None:
    """Save a three-panel plot.

    Panel 1: hot-spot (fine resolution, step load) — shows thermal lag.
    Panel 2: thermal loading limit + ambient — step profile.
    Panel 3: hot-spot + load for a realistic daily variation profile.
    """
    hours   = [r["hour"] for r in profile]
    ambient = [r["ambient_temp_c"] for r in profile]
    load    = [r["load_factor"] for r in profile]

    power_limit   = [r["thermal_limit_pu"] for r in power_results]
    service_limit = [r["thermal_limit_pu"] for r in service_results]

    n_panels = 3 if daily_profile is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=False)
    if n_panels == 2:
        axes = list(axes) + [None]
    fig.suptitle("Transformer DTR Lite — IEEE C57.91 Example Parameters", fontsize=12)

    # ── Panel 1: hot-spot temperature (left) + applied load (right) ───────────
    ax1 = axes[0]

    if power_fine is not None and service_fine is not None:
        # Fine-resolution curves show the thermal lag clearly
        fine_t_pwr = [r["time_hours"]    for r in power_fine]
        fine_h_pwr = [r["hot_spot_temp_c"] for r in power_fine]
        fine_t_svc = [r["time_hours"]    for r in service_fine]
        fine_h_svc = [r["hot_spot_temp_c"] for r in service_fine]

        ax1.plot(fine_t_pwr, fine_h_pwr, label="Power xfmr hot-spot",   color="tab:blue")
        ax1.plot(fine_t_svc, fine_h_svc, label="Service xfmr hot-spot", color="tab:orange")

        # Load drawn as a staircase: each hourly value is held constant until
        # the next hour, then jumps instantaneously -- making the thermal lag
        # of the smooth hot-spot curves visually obvious
        ax1r = ax1.twinx()
        ax1r.step(hours, load, where="post", color="tab:gray", linewidth=1.5,
                  linestyle="--", label="Applied load (step)")
    else:
        power_hot_spot   = [r["hot_spot_temp_c"] for r in power_results]
        service_hot_spot = [r["hot_spot_temp_c"] for r in service_results]
        ax1.plot(hours, power_hot_spot,   label="Power xfmr hot-spot",   color="tab:blue")
        ax1.plot(hours, service_hot_spot, label="Service xfmr hot-spot", color="tab:orange")
        ax1r = ax1.twinx()
        ax1r.plot(hours, load, color="tab:gray", linewidth=1.5, linestyle="--", label="Applied load")

    ax1.axhline(110, linestyle=":", color="black", linewidth=1, label="110 °C reference")
    ax1.set_ylabel("Hot-spot temperature [°C]")
    ax1.grid(True, alpha=0.3)

    ax1r.set_ylabel("Load factor [pu]", color="tab:gray")
    ax1r.tick_params(axis="y", labelcolor="tab:gray")
    ax1r.set_ylim(0, max(load) * 1.5)

    # combined legend from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1r, labels1r = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, labels1 + labels1r, loc="upper left", fontsize=8.5)

    # ── Panel 2: thermal loading limit (left) + ambient temperature (right) ───
    ax2 = axes[1]
    ax2.plot(hours, power_limit,   label="Power xfmr thermal limit",   color="tab:blue",   linestyle="--")
    ax2.plot(hours, service_limit, label="Service xfmr thermal limit", color="tab:orange", linestyle="--")
    ax2.set_ylabel("Thermal loading limit [pu]")
    ax2.set_xlabel("Hour of day")
    ax2.grid(True, alpha=0.3)

    ax2r = ax2.twinx()
    ax2r.plot(hours, ambient, color="tab:green", linewidth=1.5, linestyle="-.", label="Ambient temperature")
    ax2r.set_ylabel("Ambient temperature [°C]", color="tab:green")
    ax2r.tick_params(axis="y", labelcolor="tab:green")

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2r, labels2r = ax2r.get_legend_handles_labels()
    ax2.legend(lines2 + lines2r, labels2 + labels2r, loc="upper left", fontsize=8.5)

    # ── Panel 3: daily variation — hot-spot (left) + load (right) ─────────────
    if daily_profile is not None and daily_power_results is not None and daily_service_results is not None:
        d_hours   = [r["hour"] for r in daily_profile]
        d_load    = [r["load_factor"] for r in daily_profile]
        d_pwr_hs  = [r["hot_spot_temp_c"] for r in daily_power_results]
        d_svc_hs  = [r["hot_spot_temp_c"] for r in daily_service_results]

        ax3 = axes[2]
        ax3.plot(d_hours, d_pwr_hs, label="Power xfmr hot-spot",   color="tab:blue")
        ax3.plot(d_hours, d_svc_hs, label="Service xfmr hot-spot", color="tab:orange")
        ax3.axhline(110, linestyle=":", color="black", linewidth=1, label="110 °C reference")
        ax3.set_ylabel("Hot-spot temperature [°C]")
        ax3.set_xlabel("Hour of day")
        ax3.grid(True, alpha=0.3)

        ax3r = ax3.twinx()
        ax3r.plot(d_hours, d_load, color="tab:gray", linewidth=1.5, linestyle="--",
                  label="Applied load (daily)")
        ax3r.set_ylabel("Load factor [pu]", color="tab:gray")
        ax3r.tick_params(axis="y", labelcolor="tab:gray")
        ax3r.set_ylim(0, max(d_load) * 1.5)

        lines3, labels3 = ax3.get_legend_handles_labels()
        lines3r, labels3r = ax3r.get_legend_handles_labels()
        ax3.legend(lines3 + lines3r, labels3 + labels3r, loc="upper left", fontsize=8.5)

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

    # simulate both transformers (hourly for summary table)
    power_results = simulate(power_params, profile)
    service_results = simulate(service_params, profile)

    # high-resolution simulation (15-min steps) to visualise thermal lag in plot
    power_fine = simulate_fine(power_params, profile, dt_minutes=15.0)
    service_fine = simulate_fine(service_params, profile, dt_minutes=15.0)

    # daily variation profile (smooth ramp — realistic 24-hour operation)
    daily_profile = load_profile("daily_profile.csv")
    daily_power_results = simulate(power_params, daily_profile)
    daily_service_results = simulate(service_params, daily_profile)

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

    # save plot (pass fine results so panel 1 shows thermal lag)
    plot_results(
        profile, power_results, service_results,
        DATA_DIR / "example_results.png",
        power_fine=power_fine,
        service_fine=service_fine,
        daily_profile=daily_profile,
        daily_power_results=daily_power_results,
        daily_service_results=daily_service_results,
    )


if __name__ == "__main__":
    main()
