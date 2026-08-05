"""
seir_superspreading_sweep.py
============================
Sweep the dispersion parameter k_d of the negative-binomial offspring
distribution and quantify, for each value:

  - The distribution of superspreading events across repeated simulations
  - The distribution of attack rates across repeated simulations

Superspreading model
--------------------
The number of secondary infections caused by one infectious node on one day
is drawn from a Gamma-Poisson (negative-binomial) distribution, implemented
in two steps:

  1. Draw individual reproductive number:
         ν  ~  Gamma(shape = k_d, scale = R0 / k_d)
     so that E[ν] = R0 and Var[ν] = R0² / k_d.

  2. Attempt to infect each susceptible neighbour independently.
     The per-edge daily transmission probability for this node and day is:
         β_i = 1 − exp(−ν / infectious_days)
     capped at 1.  This converts the continuous Gamma draw into a
     consistent per-edge Bernoulli probability.

     When k_d → ∞ the Gamma distribution collapses to a point mass at R0
     and the model reduces to the homogeneous SEIR (no superspreading).
     When k_d is small (e.g. 0.1) the variance is high and a small fraction
     of nodes drive nearly all transmission.

Superspreading event definition
--------------------------------
A node is classified as a superspreader on a given day if it causes ≥
SS_THRESHOLD (default 4) new exposures among its neighbours on that day.
Total superspreading events per simulation = sum over all days and all nodes.

Attack rate
-----------
    AF(horizon) = (N − S(min(horizon, T_extinction))) / N

Public API
----------
    from seir_superspreading_sweep import (
        run_seir_ss,
        sweep_kd,
        plot_distributions,
        SweepResult,
    )

    results = sweep_kd(
        kd_values       = [0.1, 0.5, 1.0, 5.0, 20.0],
        R0              = 2.5,
        n_nodes         = 1000,
        n_replicates    = 30,
        horizon         = 200,
    )
    plot_distributions(results)

CLI
---
    python seir_superspreading_sweep.py                          # defaults
    python seir_superspreading_sweep.py \\
        --kd-values "0.05 0.1 0.3 0.5 1.0 2.0 5.0 20.0" \\
        --R0 3.0 --nodes 1000 --reps 30 --horizon 200 --plot
"""

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from seir_network_model import watts_strogatz
except ImportError:
    sys.exit(
        "Cannot find seir_network_model.py. "
        "Place it in the same directory as this script."
    )

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError as exc:
    sys.exit(f"numpy and matplotlib are required: {exc}")


# ---------------------------------------------------------------------------
# State codes
# ---------------------------------------------------------------------------
_S = 0
_E = 1
_I = 2
_R = 3

# Superspreading threshold: a node that exposes >= SS_THRESHOLD neighbours
# in one day is counted as a superspreader.
SS_THRESHOLD = 4


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Outcome of a single simulation run."""
    attack_rate:    float        # fraction ever infected up to horizon
    ss_events:      int          # total superspreading events (days × nodes)
    ss_by_day:      list[int]    # superspreading events per day
    daily_new_exp:  list[int]    # new exposures per day (incidence)


@dataclass
class KdResult:
    """Aggregate over n_replicates runs at one k_d value."""
    k_disp:           float
    attack_rates:     list[float]   = field(default_factory=list)
    ss_event_totals:  list[int]     = field(default_factory=list)
    # Derived statistics (populated by sweep_kd after all reps finish)
    mean_af:          float = 0.0
    std_af:           float = 0.0
    mean_ss:          float = 0.0
    std_ss:           float = 0.0
    median_af:        float = 0.0
    median_ss:        float = 0.0


@dataclass
class SweepResult:
    """Full output of sweep_kd()."""
    kd_results:      list[KdResult]
    kd_values:       list[float]
    R0:              float
    n_nodes:         int
    n_replicates:    int
    horizon:         int
    incubation_days: float
    infectious_days: float
    k_network:       int
    p_rewire:        float


# ---------------------------------------------------------------------------
# Core single-run simulator
# ---------------------------------------------------------------------------

def run_seir_ss(
    n_nodes:         int,
    R0:              float,
    k_disp:          float,
    incubation_days: float = 5.0,
    infectious_days: float = 7.0,
    k:               int   = 8,
    p_rewire:        float = 0.1,
    horizon:         int   = 200,
    seed:            int   = 42,
) -> SimResult:
    """
    Single SEIR simulation on a Watts-Strogatz network with Gamma-distributed
    offspring numbers (negative-binomial superspreading).

    Superspreading parameterisation
    --------------------------------
    Each infectious node i on each day draws its individual reproductive
    potential ν_i from:

        ν_i ~ Gamma(shape = k_disp, scale = R0 / k_disp)

    so that E[ν_i] = R0 and Var[ν_i] = R0² / k_disp.  This is the
    Lloyd-Smith et al. (2005) negative-binomial offspring distribution with
    mean R0 and dispersion k_disp.

    The per-edge daily transmission probability for node i is then:

        β_i = 1 − exp(−ν_i / infectious_days)

    which is derived by treating ν_i as a Poisson rate for the total number
    of transmissions over the infectious period, then allocating it uniformly
    across infectious_days and converting to a daily Bernoulli probability
    via the complementary CDF of the exponential.  This ensures that the
    expected total transmissions across the entire infectious period equals
    ν_i, consistent with the R0 parameterisation.

    Parameters
    ----------
    n_nodes         : int    network size (default 1000)
    R0              : float  basic reproduction number
    k_disp          : float  Gamma dispersion parameter
                             small → high heterogeneity (strong superspreading)
                             large → near-homogeneous (k_disp → ∞ = standard SEIR)
    incubation_days : float  mean incubation period σ⁻¹ (default 5.0)
    infectious_days : float  mean infectious period γ⁻¹ (default 7.0)
    k               : int    Watts-Strogatz ring-lattice degree (default 8)
    p_rewire        : float  edge rewiring probability (default 0.1)
    horizon         : int    maximum days to simulate (default 200)
    seed            : int    random seed (default 42)

    Returns
    -------
    SimResult with attack_rate, ss_events, ss_by_day, daily_new_exp.
    """
    rng         = random.Random(seed)
    adj         = watts_strogatz(n_nodes, k=k, p=p_rewire, seed=seed)
    mean_degree = sum(len(v) for v in adj.values()) / n_nodes

    sigma = 1.0 / incubation_days
    gamma = 1.0 / infectious_days

    # Gamma parameters for ν_i: shape=k_disp, scale=R0/k_disp
    gamma_shape = k_disp
    gamma_scale = R0 / k_disp

    state                          = [_S] * n_nodes
    state[rng.randint(0, n_nodes - 1)] = _E

    ss_by_day:    list[int] = []
    new_exp_day:  list[int] = []

    t = 0
    while True:
        E  = state.count(_E)
        I  = state.count(_I)

        if (E == 0 and I == 0) or t >= horizon:
            break

        new_state   = state[:]
        ss_today    = 0
        new_exp_t   = 0

        for node in range(n_nodes):

            if state[node] == _S:
                # Susceptible: exposed by any I neighbour
                for nb in adj[node]:
                    if state[nb] == _I:
                        # Draw this neighbour's individual beta for today
                        nu  = rng.gammavariate(gamma_shape, gamma_scale)
                        b   = min(1.0 - math.exp(-nu / infectious_days), 1.0)
                        if rng.random() < b:
                            new_state[node] = _E
                            break   # one exposure per day suffices

            elif state[node] == _E:
                if rng.random() < sigma:
                    new_state[node] = _I

            elif state[node] == _I:
                # Draw this node's individual ν for today
                nu  = rng.gammavariate(gamma_shape, gamma_scale)
                b_i = min(1.0 - math.exp(-nu / infectious_days), 1.0)

                exposures = 0
                for nb in adj[node]:
                    if new_state[nb] == _S and rng.random() < b_i:
                        new_state[nb] = _E
                        exposures    += 1

                new_exp_t += exposures
                if exposures >= SS_THRESHOLD:
                    ss_today += 1

                if rng.random() < gamma:
                    new_state[node] = _R

        ss_by_day.append(ss_today)
        new_exp_day.append(new_exp_t)
        state = new_state
        t    += 1

    S_final     = state.count(_S)
    attack_rate = (n_nodes - S_final) / n_nodes
    ss_total    = sum(ss_by_day)

    return SimResult(
        attack_rate   = attack_rate,
        ss_events     = ss_total,
        ss_by_day     = ss_by_day,
        daily_new_exp = new_exp_day,
    )


# ---------------------------------------------------------------------------
# Sweep over k_d values
# ---------------------------------------------------------------------------

def sweep_kd(
    kd_values:       list[float],
    R0:              float = 2.5,
    n_nodes:         int   = 1000,
    n_replicates:    int   = 30,
    incubation_days: float = 5.0,
    infectious_days: float = 7.0,
    k:               int   = 8,
    p_rewire:        float = 0.1,
    horizon:         int   = 200,
    base_seed:       int   = 42,
    verbose:         bool  = True,
) -> SweepResult:
    """
    Run n_replicates SEIR simulations for each k_d in kd_values and
    collect the distribution of superspreading events and attack rates.

    Parameters
    ----------
    kd_values    : list[float]
        Dispersion parameters to sweep. Typical values:
          0.05–0.5  → strong superspreading (SARS-like)
          0.5–2.0   → moderate superspreading (MERS-like)
          5–∞       → near-homogeneous (seasonal flu-like)
    R0           : float   basic reproduction number (default 2.5)
    n_nodes      : int     network size (default 1000)
    n_replicates : int     stochastic replicates per k_d value (default 30)
    incubation_days : float mean incubation period (days, default 5.0)
    infectious_days : float mean infectious period (days, default 7.0)
    k            : int     Watts-Strogatz degree (default 8)
    p_rewire     : float   rewiring probability (default 0.1)
    horizon      : int     max days per simulation (default 200)
    base_seed    : int     master seed; run (i,r) uses base_seed+i*1000+r
    verbose      : bool    print progress bar (default True)

    Returns
    -------
    SweepResult
        Contains a KdResult for each k_d value, with the full distributions
        (attack_rates and ss_event_totals) plus summary statistics.
    """
    kd_results: list[KdResult] = []
    n_kd        = len(kd_values)
    total       = n_kd * n_replicates
    done        = 0

    for i, kd in enumerate(kd_values):
        kr = KdResult(k_disp=kd)

        for rep in range(n_replicates):
            seed   = base_seed + i * 1000 + rep
            result = run_seir_ss(
                n_nodes         = n_nodes,
                R0              = R0,
                k_disp          = kd,
                incubation_days = incubation_days,
                infectious_days = infectious_days,
                k               = k,
                p_rewire        = p_rewire,
                horizon         = horizon,
                seed            = seed,
            )
            kr.attack_rates.append(result.attack_rate)
            kr.ss_event_totals.append(result.ss_events)
            done += 1

            if verbose:
                pct = done / total * 100
                bar = "█" * int(pct / 2)
                print(f"\r  [{bar:<50}] {pct:5.1f}%  "
                      f"k_d={kd:.3f}  rep={rep+1}/{n_replicates}  "
                      f"AF={result.attack_rate:.3f}  SS={result.ss_events}",
                      end="", flush=True)

        # Compute summary statistics
        kr.mean_af   = float(np.mean(kr.attack_rates))
        kr.std_af    = float(np.std(kr.attack_rates, ddof=1)) if n_replicates > 1 else 0.0
        kr.median_af = float(np.median(kr.attack_rates))
        kr.mean_ss   = float(np.mean(kr.ss_event_totals))
        kr.std_ss    = float(np.std(kr.ss_event_totals, ddof=1)) if n_replicates > 1 else 0.0
        kr.median_ss = float(np.median(kr.ss_event_totals))
        kd_results.append(kr)

    if verbose:
        print()

    return SweepResult(
        kd_results      = kd_results,
        kd_values       = list(kd_values),
        R0              = R0,
        n_nodes         = n_nodes,
        n_replicates    = n_replicates,
        horizon         = horizon,
        incubation_days = incubation_days,
        infectious_days = infectious_days,
        k_network       = k,
        p_rewire        = p_rewire,
    )


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(sweep: SweepResult) -> None:
    """Print a formatted summary table of sweep results."""
    w   = 76
    div = "=" * w
    print(f"\n{div}")
    print(f"  SEIR Superspreading Sweep  "
          f"(R0={sweep.R0}, N={sweep.n_nodes}, "
          f"horizon={sweep.horizon}d, {sweep.n_replicates} reps)")
    print(div)
    print(f"  {'k_d':>8}  {'mean AF':>9}  {'std AF':>8}  "
          f"{'median AF':>10}  {'mean SS':>9}  {'std SS':>8}  {'median SS':>10}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*10}")
    for kr in sweep.kd_results:
        print(f"  {kr.k_disp:>8.3f}  "
              f"{kr.mean_af:>9.3f}  {kr.std_af:>8.3f}  {kr.median_af:>10.3f}  "
              f"{kr.mean_ss:>9.1f}  {kr.std_ss:>8.1f}  {kr.median_ss:>10.1f}")
    print(div + "\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distributions(
    sweep:    SweepResult,
    out_path: Optional[str] = None,
) -> None:
    """
    Three-panel figure:

    Panel 1 (top-left):  Violin/box plots of attack-rate distributions
                         for each k_d value.
    Panel 2 (top-right): Violin/box plots of total superspreading event
                         distributions for each k_d value.
    Panel 3 (bottom):    Scatter of mean ± std for both metrics vs k_d,
                         on a log x-axis to spread the k_d values visually. -< Not used
    """
    kd_values = sweep.kd_values
    n_kd      = len(kd_values)
    kd_labels = [f"{v:.2g}" for v in kd_values]

    af_data = [kr.attack_rates    for kr in sweep.kd_results]
    ss_data = [kr.ss_event_totals for kr in sweep.kd_results]

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3, 2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    #ax3 = fig.add_subplot(gs[1, 0])
    #ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1: attack rate distributions ────────────────────────────────
    vp1 = ax1.violinplot(af_data, positions=range(n_kd),
                         showmedians=True, showextrema=True)
    for body in vp1["bodies"]:
        body.set_facecolor("#2196F3")
        body.set_alpha(0.5)
    vp1["cmedians"].set_color("#0D47A1")
    vp1["cmedians"].set_linewidth(2)
    # Overlay individual points
    for j, vals in enumerate(af_data):
        ax1.scatter([j] * len(vals), vals,
                    color="#0D47A1", s=18, alpha=0.6, zorder=3)
    ax1.set_xticks(range(n_kd))
    ax1.set_xticklabels(kd_labels, fontsize=9)
    ax1.set_xlabel("Dispersion parameter k_d", fontsize=11)
    ax1.set_ylabel("Attack fraction", fontsize=11)
    ax1.set_title("Distribution of attack fractions", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(np.mean([kr.mean_af for kr in sweep.kd_results]),
                color="grey", lw=0.8, ls=":", alpha=0.6)

    # ── Panel 2: superspreading event distributions ────────────────────────
    ss_max = max(max(v) for v in ss_data) if any(ss_data) else 1
    if ss_max > 0:
        vp2 = ax2.violinplot(ss_data, positions=range(n_kd),
                             showmedians=True, showextrema=True)
        for body in vp2["bodies"]:
            body.set_facecolor("#F44336")
            body.set_alpha(0.5)
        vp2["cmedians"].set_color("#B71C1C")
        vp2["cmedians"].set_linewidth(2)
    for j, vals in enumerate(ss_data):
        ax2.scatter([j] * len(vals), vals,
                    color="#B71C1C", s=18, alpha=0.6, zorder=3)
    ax2.set_xticks(range(n_kd))
    ax2.set_xticklabels(kd_labels, fontsize=10)
    ax2.set_xlabel("Dispersion parameter k_d", fontsize=14)
    ax2.set_ylabel(f"Total SS events (≥{SS_THRESHOLD} offspring/node/day)",
                   fontsize=14)
    #ax2.set_title("Distribution of superspreading events", fontsize=11)
    ax2.set_ylim(bottom=-0.5)
    ax2.grid(axis="y", alpha=0.3)


    # ── Panel 3: mean AF ± std vs k_d (log x-axis) ────────────────────────
    #means_af = [kr.mean_af for kr in sweep.kd_results]
    #stds_af  = [kr.std_af  for kr in sweep.kd_results]
    #ax3.errorbar(kd_values, means_af, yerr=stds_af,fmt="o-", color="#2196F3", linewidth=2,capsize=4, capthick=1.5, markersize=7,label="Mean ± Std")
    #ax3.fill_between(kd_values,[m - s for m, s in zip(means_af, stds_af)],[m + s for m, s in zip(means_af, stds_af)],alpha=0.15, color="#2196F3")
    #ax3.set_xscale("log")
    #ax3.set_xlabel("k_d  (log scale)", fontsize=11)
    #ax3.set_ylabel("Mean attack fraction", fontsize=11)
    #ax3.set_title("Mean attack fraction vs k_d", fontsize=11)
    #ax3.set_ylim(-0.05, 1.05)
    #ax3.grid(alpha=0.3)
    #ax3.legend(fontsize=9)


    # ── Panel 4: mean SS events ± std vs k_d (log x-axis) ────────────────
    #means_ss = [kr.mean_ss for kr in sweep.kd_results]
    #stds_ss  = [kr.std_ss  for kr in sweep.kd_results]
    #ax4.errorbar(kd_values, means_ss, yerr=stds_ss,fmt="s-", color="#F44336", linewidth=2,capsize=4, capthick=1.5, markersize=7,label="Mean ± Std")
    #ax4.fill_between(kd_values,[max(0, m - s) for m, s in zip(means_ss, stds_ss)],[m + s for m, s in zip(means_ss, stds_ss)], alpha=0.15, color="#F44336")
    #ax4.set_xscale("log")
    #ax4.set_xlabel("k_d  (log scale)", fontsize=11)
    #ax4.set_ylabel(f"Mean SS events (≥{SS_THRESHOLD} offspring)", fontsize=11)
    #ax4.set_title("Mean superspreading events vs k_d", fontsize=11)
    #ax4.set_ylim(bottom=-0.5)
    #ax4.grid(alpha=0.3)
    #ax4.legend(fontsize=9)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_KD = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEIR superspreading sweep: SS events and attack rates vs k_d",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kd-values",   type=str,
                        default=" ".join(str(v) for v in DEFAULT_KD),
                        help="Space-separated dispersion parameter values")
    parser.add_argument("--R0",          type=float, default=2.5)
    parser.add_argument("--nodes",       type=int,   default=1000)
    parser.add_argument("--incubation",  type=float, default=5.0,
                        help="Mean incubation period (days)")
    parser.add_argument("--infectious",  type=float, default=7.0,
                        help="Mean infectious period (days)")
    parser.add_argument("--k",           type=int,   default=8,
                        help="Watts-Strogatz ring-lattice degree")
    parser.add_argument("--p",           type=float, default=0.1,
                        help="Edge rewiring probability")
    parser.add_argument("--reps",        type=int,   default=30,
                        help="Stochastic replicates per k_d value")
    parser.add_argument("--horizon",     type=int,   default=200,
                        help="Maximum simulation days per run")
    parser.add_argument("--ss-threshold", type=int,  default=4,
                        dest="ss_threshold",
                        help="Min offspring per node per day to count as SS event")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         type=str,   default=None,
                        help="Path to save the figure")
    parser.add_argument("--no-plot",     action="store_true", dest="no_plot")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    kd_values = [float(x) for x in args.kd_values.split()]

    # Allow user to override the module-level threshold
    if args.ss_threshold != SS_THRESHOLD:
        import seir_superspreading_sweep as _self
        _self.SS_THRESHOLD = args.ss_threshold

    n_total = len(kd_values) * args.reps
    print(f"\n  Sweeping {len(kd_values)} k_d values × {args.reps} reps "
          f"= {n_total} simulations")
    print(f"  R0={args.R0}  N={args.nodes}  k={args.k}  "
          f"horizon={args.horizon}d  SS threshold ≥ {args.ss_threshold}\n")

    sweep = sweep_kd(
        kd_values       = kd_values,
        R0              = args.R0,
        n_nodes         = args.nodes,
        n_replicates    = args.reps,
        incubation_days = args.incubation,
        infectious_days = args.infectious,
        k               = args.k,
        p_rewire        = args.p,
        horizon         = args.horizon,
        base_seed       = args.seed,
        verbose         = not args.quiet,
    )

    print_report(sweep)

    if not args.no_plot:
        try:
            plot_distributions(sweep, out_path=args.out)
        except ImportError:
            print("matplotlib not installed – skipping plot.")
