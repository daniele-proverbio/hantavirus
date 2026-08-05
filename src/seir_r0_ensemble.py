"""
seir_r0_ensemble.py
===================
Run the SEIR small-world model for an ensemble of R0 values drawn from a
Gaussian distribution N(R0_mean, R0_std), and plot the resulting S, E, I, R
trajectories.

Usage
-----
    python seir_r0_ensemble.py                          # defaults
    python seir_r0_ensemble.py --R0-mean 2.5 --R0-std 0.5 --n-samples 30
    python seir_r0_ensemble.py --R0-mean 3.0 --R0-std 1.0 --n-samples 50 \\
        --nodes 1000 --incubation 5 --infectious 7 --k 6

Parameters
----------
  --R0-mean   float   Mean of the Gaussian R0 distribution  (default: 2.5)
  --R0-std    float   Std  of the Gaussian R0 distribution  (default: 0.5)
  --n-samples int     Number of R0 values to draw           (default: 20)
  --nodes     int     Network size                          (default: 500)
  --incubation float  Mean incubation period (days)         (default: 5.0)
  --infectious float  Mean infectious period (days)         (default: 7.0)
  --k         int     Watts-Strogatz degree                 (default: 4)
  --p         float   Rewiring probability                  (default: 0.1)
  --seed      int     Master random seed                    (default: 42)
  --R0-min    float   Hard lower clip for sampled R0 values (default: 0.1)
  --out       str     Optional path to save the figure      (default: None)
  --no-plot   flag    Suppress the interactive plot window
"""

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Accept either the original or the re-packaged module name
try:
    from seir_network_model import run_seir
except ImportError:
    try:
        from epidemic_seir_smallworld import run_seir
    except ImportError:
        sys.exit(
            "Cannot find seir_network_model.py or epidemic_seir_smallworld.py. "
            "Place one of them in the same directory as this script."
        )

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError as exc:
    sys.exit(f"matplotlib and numpy are required: {exc}")


# ---------------------------------------------------------------------------
# 1.  Sample R0 values from a Gaussian
# ---------------------------------------------------------------------------

def sample_r0_gaussian(
    mean: float,
    std: float,
    n: int,
    r0_min: float = 0.1,
    seed: int = 42,
) -> list[float]:
    """
    Draw ``n`` R0 values from N(mean, std²), clipped to [r0_min, ∞).

    Negative or near-zero R0 values are biologically meaningless and would
    crash the simulator, so values below ``r0_min`` are resampled until the
    full ``n`` valid draws are collected.
    """
    rng    = random.Random(seed)
    values = []
    while len(values) < n:
        v = rng.gauss(mean, std)
        if v >= r0_min:
            values.append(v)
    return values


# ---------------------------------------------------------------------------
# 2.  Run ensemble
# ---------------------------------------------------------------------------

def run_ensemble(
    r0_values: list[float],
    n_nodes: int,
    incubation_days: float,
    infectious_days: float,
    k: int,
    p_rewire: float,
    base_seed: int,
) -> list[dict]:
    """
    Run one SEIR simulation per R0 value.

    Returns a list of result dicts, each with keys:
      'R0', 'S', 'E', 'I', 'R'  – the R0 used and the four daily series.
    """
    results = []
    for idx, R0 in enumerate(r0_values):
        S, E, I, R, _ = run_seir(
            n_nodes         = n_nodes,
            R0              = R0,
            incubation_days = incubation_days,
            infectious_days = infectious_days,
            k               = k,
            p_rewire        = p_rewire,
            seed            = base_seed + idx,
        )
        results.append({"R0": R0, "S": S, "E": E, "I": I, "R": R})
        print(f"  [{idx+1:>3}/{len(r0_values)}]  R0={R0:.3f}  "
              f"duration={len(I)} days  peak_I={max(I)}")
    return results


# ---------------------------------------------------------------------------
# 3.  Plotting
# ---------------------------------------------------------------------------

def plot_ensemble(
    results:         list[dict],
    r0_mean:         float,
    r0_std:          float,
    n_nodes:         int,
    incubation_days: float,
    infectious_days: float,
    out_path:        str | None = None,
) -> None:
    """
    Four-panel figure: one panel per compartment (S, E, I, R).

    Each trajectory is coloured by its R0 value using a diverging colormap
    centred on R0_mean, so trajectories from low-R0 draws are visually
    distinct from high-R0 draws.  A colourbar is drawn on the right.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             constrained_layout=True)
    fig.suptitle(
        f"SEIR ensemble  —  "
        f"$R_0 \\sim \\mathcal{{N}}({r0_mean},\\,{r0_std}^2)$,  "
        f"$n={len(results)}$ trajectories\n"
        f"Network: $N={n_nodes}$ nodes,  "
        f"$\\sigma^{{-1}}={incubation_days}$ d,  "
        f"$\\gamma^{{-1}}={infectious_days}$ d",
        fontsize=13, y=1.01,
    )

    compartments = [
        ("S", "Susceptible",  axes[0, 0], "#1565C0"),
        ("E", "Exposed",      axes[0, 1], "#E65100"),
        ("I", "Infectious",   axes[1, 0], "#B71C1C"),
        ("R", "Recovered",    axes[1, 1], "#2E7D32"),
    ]

    # Colourmap: map each R0 to a colour
    r0_vals = [res["R0"] for res in results]
    r0_min_val, r0_max_val = min(r0_vals), max(r0_vals)
    # Use a diverging map centred on R0_mean
    cmap     = plt.cm.coolwarm
    r0_range = max(r0_max_val - r0_mean, r0_mean - r0_min_val, 1e-6)
    norm     = mcolors.TwoSlopeNorm(
        vmin  = r0_mean - r0_range,
        vcenter = r0_mean,
        vmax  = r0_mean + r0_range,
    )

    # Draw trajectories
    for key, label, ax, _ in compartments:
        for res in results:
            series = [v / n_nodes for v in res[key]]   # normalise to [0,1]
            days   = list(range(len(series)))
            color  = cmap(norm(res["R0"]))
            ax.plot(days, series, color=color, alpha=0.55, linewidth=1.2)

        # Overlay the trajectory whose R0 is closest to the mean
        closest = min(results, key=lambda r: abs(r["R0"] - r0_mean))
        series  = [v / n_nodes for v in closest[key]]
        ax.plot(range(len(series)), series,
                color="black", linewidth=2.2, linestyle="--",
                label=f"$R_0 \\approx$ {closest['R0']:.2f} (closest to mean)",
                zorder=5)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Day", fontsize=10)
        ax.set_ylabel("Fraction of population", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=8, loc="upper right")

    # Shared colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label("$R_0$ value", fontsize=11)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure saved to: {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 4.  Summary statistics
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], n_nodes: int) -> None:
    """Print a compact table of per-trajectory statistics."""
    w = 70
    print(f"\n{'='*w}")
    print(f"  {'R0':>7}  {'Duration':>9}  {'Peak I':>8}  "
          f"{'Day Peak I':>10}  {'Attack %':>9}")
    print(f"  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*9}")
    for res in sorted(results, key=lambda r: r["R0"]):
        I = res["I"]
        peak_I   = max(I)
        peak_day = I.index(peak_I)
        total    = n_nodes - res["S"][-1]
        print(f"  {res['R0']:>7.3f}  {len(I)-1:>9}  "
              f"{peak_I:>8}  {peak_day:>10}  "
              f"{total/n_nodes*100:>8.1f}%")
    print(f"{'='*w}\n")


# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEIR ensemble over R0 ~ N(mean, std²)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--R0-mean",    type=float, default=2.5, dest="r0_mean",
                        help="Mean of the Gaussian R0 distribution")
    parser.add_argument("--R0-std",     type=float, default=0.5, dest="r0_std",
                        help="Std of the Gaussian R0 distribution")
    parser.add_argument("--n-samples",  type=int,   default=20,  dest="n_samples",
                        help="Number of R0 values to draw")
    parser.add_argument("--nodes",      type=int,   default=500)
    parser.add_argument("--incubation", type=float, default=5.0,
                        help="Mean incubation period (days)")
    parser.add_argument("--infectious", type=float, default=7.0,
                        help="Mean infectious period (days)")
    parser.add_argument("--k",          type=int,   default=4,
                        help="Watts-Strogatz ring-lattice degree")
    parser.add_argument("--p",          type=float, default=0.1,
                        help="Edge rewiring probability")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--R0-min",     type=float, default=0.1, dest="r0_min",
                        help="Hard lower clip for sampled R0 values")
    parser.add_argument("--out",        type=str,   default=None,
                        help="Path to save the figure (e.g. ensemble.png)")
    parser.add_argument("--no-plot",    action="store_true", dest="no_plot",
                        help="Suppress the interactive plot window")
    args = parser.parse_args()

    # ── Sample R0 values ──────────────────────────────────────────────────
    print(f"\n  Sampling {args.n_samples} R0 values from "
          f"N({args.r0_mean}, {args.r0_std}²) …")
    r0_values = sample_r0_gaussian(
        mean   = args.r0_mean,
        std    = args.r0_std,
        n      = args.n_samples,
        r0_min = args.r0_min,
        seed   = args.seed,
    )
    print(f"  R0 range: [{min(r0_values):.3f}, {max(r0_values):.3f}]  "
          f"mean={sum(r0_values)/len(r0_values):.3f}\n")

    # ── Run ensemble ──────────────────────────────────────────────────────
    print(f"  Running {args.n_samples} SEIR simulations …\n")
    results = run_ensemble(
        r0_values       = r0_values,
        n_nodes         = args.nodes,
        incubation_days = args.incubation,
        infectious_days = args.infectious,
        k               = args.k,
        p_rewire        = args.p,
        base_seed       = args.seed + 1000,
    )

    # ── Print summary ─────────────────────────────────────────────────────
    print_summary(results, args.nodes)

    # ── Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_ensemble(
            results         = results,
            r0_mean         = args.r0_mean,
            r0_std          = args.r0_std,
            n_nodes         = args.nodes,
            incubation_days = args.incubation,
            infectious_days = args.infectious,
            out_path        = args.out,
        )
