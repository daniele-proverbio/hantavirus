"""
plot_ss_multi_r0.py
===================
Standalone plotting function that compares superspreading distributions
(attack fraction and SS event count) across three different R0 values,
side-by-side and non-overlapping.

For each R0, a SweepResult (produced by sweep_kd()) must be supplied.
The function produces a figure with two rows:

  Row 1 – attack-fraction distributions, one column per k_d value,
           three coloured box-plots side-by-side (one per R0) within
           each column.
  Row 2 – superspreading event distributions, same layout.

A summary trend panel is also appended on the right of each row showing
mean ± std vs k_d (log scale) for all three R0 values simultaneously.

Public API
----------
    from plot_ss_multi_r0 import plot_multi_r0

    results = {
        2.0: sweep_kd(kd_values=[...], R0=2.0, ...),
        2.5: sweep_kd(kd_values=[...], R0=2.5, ...),
        3.5: sweep_kd(kd_values=[...], R0=3.5, ...),
    }
    plot_multi_r0(results, out_path="comparison.png")

Requirements
------------
  - All three SweepResult objects must have the same kd_values (same
    length and same order).  A ValueError is raised otherwise.
  - matplotlib, numpy (already required by seir_superspreading_sweep).

CLI (self-contained demo)
-------------------------
    python plot_ss_multi_r0.py \\
        --r0-values "2.0 2.5 3.5" \\
        --kd-values "0.05 0.1 0.5 1.0 5.0 20.0" \\
        --nodes 500 --reps 20 --horizon 200 --out comparison.png
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
except ImportError as exc:
    sys.exit(f"matplotlib and numpy are required: {exc}")

try:
    from seir_superspreading_sweep import SweepResult, sweep_kd, SS_THRESHOLD
except ImportError as exc:
    sys.exit(
        "Cannot import seir_superspreading_sweep.py.  "
        "Place it in the same directory as plot_ss_multi_r0.py.\n"
        f"Details: {exc}"
    )


# ---------------------------------------------------------------------------
# Colour palette: one colour per R0 value (assigned in order)
# ---------------------------------------------------------------------------
_PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate(results: dict[float, SweepResult]) -> list[float]:
    """
    Check that all SweepResult objects share the same kd_values.
    Returns the shared kd_values list.
    """
    r0_list = sorted(results.keys())
    if len(r0_list) < 2:
        raise ValueError("Need at least two R0 values to compare.")
    ref_kd = results[r0_list[0]].kd_values
    for r0 in r0_list[1:]:
        if results[r0].kd_values != ref_kd:
            raise ValueError(
                f"kd_values mismatch between R0={r0_list[0]} and R0={r0}: "
                f"{ref_kd} vs {results[r0].kd_values}"
            )
    return ref_kd


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_multi_r0(
    results:      dict[float, SweepResult],
    out_path:     Optional[str] = None,
    fig_width:    float = 16.0,
    fig_height:   float = 9.0,
    violin_width: float = 0.20,
    show_points:  bool  = True,
    ss_threshold: int   = SS_THRESHOLD,
) -> None:
    """
    Compare superspreading distributions across multiple R0 values using
    side-by-side violin plots, one panel per k_d value.

    Parameters
    ----------
    results : dict[float, SweepResult]
        Keys are R0 values (floats); values are SweepResult objects
        returned by sweep_kd().  All must share the same kd_values.
        Typically three entries, but any number ≥ 2 is accepted.
    out_path : str or None
        If given, the figure is saved to this path (PNG or PDF).
    fig_width, fig_height : float
        Figure dimensions in inches (default 16 × 9).
    violin_width : float
        Width of each individual violin.  Total group width per panel is
        violin_width × n_r0 plus small gaps (default 0.20).
    show_points : bool
        Overlay individual replicate values as scatter points on the
        violins (default True).  Disable for large n_replicates.
    ss_threshold : int
        Displayed in axis labels (default: value from seir_superspreading_sweep).

    Layout
    ------
    Two rows × (n_kd + 1) columns:
      Rows:    [0] attack fraction   [1] SS event count
      Columns: [0..n_kd-1] grouped violin plots, one per k_d value
               [n_kd]      trend lines mean ± std vs k_d (log x-axis)

    Within each column, R0 values are placed side-by-side with a small
    gap — no overlap between violins.
    """
    kd_values = _validate(results)
    r0_list   = sorted(results.keys())
    n_r0      = len(r0_list)
    n_kd      = len(kd_values)
    colours   = _PALETTE[:n_r0]

    # ── Figure layout  (tighter spacing vs previous version) ─────────────
    fig = plt.figure(figsize=(fig_width, fig_height))
    col_widths = [1.0] * n_kd + [1.25]
    gs = gridspec.GridSpec(
        2, n_kd + 1, figure=fig,
        width_ratios = col_widths,
        hspace       = 0.22,   # tighter vertical gap between rows
        wspace       = 0.12,   # tighter horizontal gap between columns
        left=0.055, right=0.97, top=0.88, bottom=0.06,
    )

    af_axes: list[plt.Axes] = []
    ss_axes: list[plt.Axes] = []
    for j in range(n_kd):
        af_axes.append(fig.add_subplot(gs[0, j]))
        ss_axes.append(fig.add_subplot(gs[1, j]))

    ax_trend_af = fig.add_subplot(gs[0, n_kd])
    ax_trend_ss = fig.add_subplot(gs[1, n_kd])

    # ── Legend patches ────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=colours[idx], alpha=0.75,
                       label=f"R₀ = {r0}")
        for idx, r0 in enumerate(r0_list)
    ]

    # ── x-positions: side-by-side within each panel ───────────────────────
    gap     = 0.04
    total_w = n_r0 * violin_width + (n_r0 - 1) * gap
    offsets = [
        -total_w / 2 + i * (violin_width + gap) + violin_width / 2
        for i in range(n_r0)
    ]

    # ── Violin plots: one k_d column at a time ────────────────────────────
    rng_jitter = np.random.default_rng(42)

    for j, kd in enumerate(kd_values):
        ax_af = af_axes[j]
        ax_ss = ss_axes[j]

        for idx, r0 in enumerate(r0_list):
            kr  = results[r0].kd_results[j]
            x   = offsets[idx]
            col = colours[idx]

            af_vals = np.array(kr.attack_rates,     dtype=float)
            ss_vals = np.array(kr.ss_event_totals,  dtype=float)

            for ax, vals in [(ax_af, af_vals), (ax_ss, ss_vals)]:
                # Only draw a violin when there is meaningful spread;
                # with all-identical values violinplot raises an error.
                spread = vals.max() - vals.min()

                if spread > 1e-9 and len(vals) >= 3:
                    vp = ax.violinplot(
                        vals,
                        positions  = [x],
                        widths     = [violin_width],
                        showmedians = True,
                        showextrema = True,
                        showmeans   = False,
                    )
                    # Style each part of the violin
                    for body in vp["bodies"]:
                        body.set_facecolor(col)
                        body.set_alpha(0.45)
                        body.set_edgecolor(col)
                        body.set_linewidth(0.6)
                    for part_key in ("cmedians", "cmins", "cmaxes", "cbars"):
                        if part_key in vp:
                            vp[part_key].set_edgecolor(col)
                            vp[part_key].set_linewidth(
                                1.6 if part_key == "cmedians" else 0.8
                            )
                else:
                    # Degenerate case: draw a horizontal line at the single value
                    ax.hlines(vals[0], x - violin_width / 2,
                              x + violin_width / 2,
                              colors=col, linewidths=1.5)

                # Scatter overlay of individual points
                if show_points:
                    jitter = rng_jitter.uniform(
                        -violin_width * 0.28,
                         violin_width * 0.28,
                        size=len(vals),
                    )
                    ax.scatter(
                        x + jitter, vals,
                        color=col, s=12, alpha=0.6,
                        linewidths=0, zorder=3,
                    )

        # Per-column formatting
        for ax in (ax_af, ax_ss):
            ax.set_xlim(-total_w / 2 - 0.12, total_w / 2 + 0.12)
            ax.set_xticks([])
            ax.set_title(f"$k_d$ = {kd:.2g}", fontsize=10, pad=3)
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(axis="y", alpha=0.22, linewidth=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

        ax_af.set_ylim(-0.04, 1.06)

        # y-axis label only on the first column
        if j == 0:
            ax_af.set_ylabel("Attack fraction ($C_{200}/N$)", fontsize=12)
            ax_ss.set_ylabel(f"SS events  (≥{ss_threshold})", fontsize=12)
        else:
            ax_af.set_ylabel("")
            ax_ss.set_ylabel("")
            # Hide y-tick labels on non-first columns to reduce clutter
            ax_af.set_yticklabels([])
            ax_ss.set_yticklabels([])

        # Legend on the last k_d column
        if j == n_kd - 1:
            ax_af.legend(handles=legend_patches, fontsize=8,
                         loc="lower left", framealpha=0.7,
                         borderpad=0.4, labelspacing=0.3)

    # ── Trend panels ──────────────────────────────────────────────────────
    kd_arr = np.array(kd_values, dtype=float)

    for idx, r0 in enumerate(r0_list):
        col      = colours[idx]
        means_af = [results[r0].kd_results[j].mean_af for j in range(n_kd)]
        stds_af  = [results[r0].kd_results[j].std_af  for j in range(n_kd)]
        means_ss = [results[r0].kd_results[j].mean_ss for j in range(n_kd)]
        stds_ss  = [results[r0].kd_results[j].std_ss  for j in range(n_kd)]

        ax_trend_af.errorbar(
            kd_arr, means_af, yerr=stds_af,
            fmt="o-", color=col, linewidth=1.6,
            capsize=3, capthick=1.1, markersize=5,
            label=f"R₀ = {r0}", alpha=0.88,
        )
        ax_trend_af.fill_between(
            kd_arr,
            [max(0.0, m - s) for m, s in zip(means_af, stds_af)],
            [min(1.0, m + s) for m, s in zip(means_af, stds_af)],
            alpha=0.10, color=col,
        )

        ax_trend_ss.errorbar(
            kd_arr, means_ss, yerr=stds_ss,
            fmt="s-", color=col, linewidth=1.6,
            capsize=3, capthick=1.1, markersize=5,
            label=f"R₀ = {r0}", alpha=0.88,
        )
        ax_trend_ss.fill_between(
            kd_arr,
            [max(0.0, m - s) for m, s in zip(means_ss, stds_ss)],
            [m + s for m, s in zip(means_ss, stds_ss)],
            alpha=0.10, color=col,
        )

    for ax in (ax_trend_af, ax_trend_ss):
        ax.set_xscale("log")
        ax.set_xlabel("$k_d$  (log scale)", fontsize=12)
        ax.grid(alpha=0.22, linewidth=0.5)
        ax.legend(fontsize=8, loc="best", framealpha=0.7,
                  borderpad=0.4, labelspacing=0.3)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax_trend_af.set_ylabel("Mean attack fraction", fontsize=12)
    ax_trend_af.set_ylim(-0.04, 1.06)
    ax_trend_af.set_title("Trend: mean ± std", fontsize=10, pad=3)

    ax_trend_ss.set_ylabel(f"Mean SS events  (≥{ss_threshold})", fontsize=12)
    ax_trend_ss.set_ylim(bottom=-0.3)
    ax_trend_ss.set_title("Trend: mean ± std", fontsize=10, pad=3)

    # ── Super-title ───────────────────────────────────────────────────────
    r0_str = "  |  ".join(f"R₀={r0}" for r0 in r0_list)
    sweep0 = next(iter(results.values()))
    #fig.suptitle(f"Superspreading distributions: {r0_str}     "f"N={sweep0.n_nodes},  k={sweep0.k_network},  "f"horizon={sweep0.horizon}d,  {sweep0.n_replicates} reps/k_d",fontsize=11, y=0.97,)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.show()


# ---------------------------------------------------------------------------
# CLI – self-contained demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side SS distribution comparison across three R0 values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--r0-values",  type=str, default="2.0 2.5 3.5",
                        help="Space-separated R0 values (exactly 2 or 3)")
    parser.add_argument("--kd-values",  type=str,
                        default="0.05 0.1 0.5 1.0 5.0 20.0",
                        help="Space-separated k_d values")
    parser.add_argument("--nodes",      type=int,   default=500)
    parser.add_argument("--incubation", type=float, default=5.0)
    parser.add_argument("--infectious", type=float, default=7.0)
    parser.add_argument("--k",          type=int,   default=8,
                        help="Watts-Strogatz degree")
    parser.add_argument("--p",          type=float, default=0.1)
    parser.add_argument("--reps",       type=int,   default=20,
                        help="Replicates per k_d per R0")
    parser.add_argument("--horizon",    type=int,   default=200)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--out",        type=str,   default=None,
                        help="Path to save figure (e.g. comparison.png)")
    parser.add_argument("--no-plot",    action="store_true", dest="no_plot")
    parser.add_argument("--quiet",      action="store_true")
    args = parser.parse_args()

    r0_values = [float(x) for x in args.r0_values.split()]
    kd_values = [float(x) for x in args.kd_values.split()]

    if len(r0_values) < 2:
        sys.exit("Please supply at least two R0 values.")

    results: dict[float, SweepResult] = {}
    for i, r0 in enumerate(r0_values):
        if not args.quiet:
            print(f"\n  Running R0 = {r0}  ({len(kd_values)} k_d values "
                  f"× {args.reps} reps) …")
        results[r0] = sweep_kd(
            kd_values       = kd_values,
            R0              = r0,
            n_nodes         = args.nodes,
            n_replicates    = args.reps,
            incubation_days = args.incubation,
            infectious_days = args.infectious,
            k               = args.k,
            p_rewire        = args.p,
            horizon         = args.horizon,
            base_seed       = args.seed + i * 100_000,
            verbose         = not args.quiet,
        )

    if not args.no_plot:
        plot_multi_r0(results, out_path=args.out)
