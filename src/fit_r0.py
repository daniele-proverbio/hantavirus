"""
fit_r0.py  –  Fit R0 to empirical epidemic data using the SEIR small-world model
==================================================================================
Public API (importable)
-----------------------
The primary entry point for external scripts is ``fit_r0()``, which returns a
``FitResult`` dataclass.  Supporting helpers are also importable:

    from fit_r0 import fit_r0, load_data, save_fitted_curve, plot_fit, FitResult

FitResult fields
----------------
  R0_estimate   float        – point estimate (ABC: posterior median; grid: best RMSE)
  R0_ci_low     float|None   – 5th-percentile of ABC posterior  (None for grid)
  R0_ci_high    float|None   – 95th-percentile of ABC posterior (None for grid)
  best_distance float        – normalised RMSE of the best-fit simulation
  method        str          – "abc" or "grid"
  I_fitted      list[float]  – ensemble-averaged I series at R0_estimate
  R_fitted      list[float]  – ensemble-averaged R series at R0_estimate
  days          list[int]    – day indices matching the observed series
  I_obs         list[int]    – observed I series (echoed for convenience)
  R_obs         list[int]    – observed R series (echoed for convenience)
  # ABC-only diagnostics (empty lists for grid)
  R0_all        list[float]  – all sampled R0 values
  dist_all      list[float]  – corresponding distances
  epsilon       float|None   – acceptance threshold used
  n_accepted    int          – number of accepted draws

Strategy
--------
Because the SEIR model on a stochastic network has no closed-form likelihood,
two complementary methods are available via the ``method`` parameter:

  "abc"   (default)  Approximate Bayesian Computation – rejection sampling.
                     Draws R0 ~ Uniform(R0_min, R0_max), runs the SEIR model,
                     computes a summary-statistic distance, and accepts if
                     distance < ε.  Returns a posterior with credible interval.
                     ε is auto-calibrated unless set explicitly.

  "grid"             Grid search over [R0_min, R0_max].  Each candidate is
                     evaluated over n_replicates runs to smooth stochastic
                     noise.  Faster but returns only a point estimate.

Input data
----------
CSV with at least one column named 'I' (daily infectious count).
Optional columns: 'day' (index), 'R' (recovered, improves fit quality).

  day,I,R
  0,1,0
  1,1,0
  2,3,0
  ...

"""

import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 0.  Import the simulation (need companion functions)
# ---------------------------------------------------------------------------

# Allow running from any directory by adding the script's own folder to path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from epidemic_seir_smallworld import run_seir
except ImportError as exc:
    sys.exit(
        "Cannot import epidemic_seir_smallworld.py.  "
        "Make sure it is in the same directory as fit_r0.py.\n"
        f"Details: {exc}"
    )


# ---------------------------------------------------------------------------
# 0b.  Public result type
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """
    All outputs from a single ``fit_r0()`` call.

    Attributes
    ----------
    R0_estimate : float
        Point estimate of R0.
        ABC  → posterior median of accepted draws.
        Grid → R0 with lowest RMSE.
    R0_ci_low : float or None
        5th-percentile of the ABC posterior.  None when method="grid".
    R0_ci_high : float or None
        95th-percentile of the ABC posterior.  None when method="grid".
    best_distance : float
        Normalised RMSE of the best-fit simulation against the observed data.
    method : str
        "abc" or "grid".
    I_fitted : list[float]
        Ensemble-averaged simulated I series at R0_estimate.
    R_fitted : list[float]
        Ensemble-averaged simulated R series at R0_estimate.
    days : list[int]
        Day indices (mirrors the input series).
    I_obs : list[int]
        Observed I series (echoed for convenience).
    R_obs : list[int]
        Observed R series (echoed for convenience).
    R0_all : list[float]
        All sampled R0 values (ABC only; empty for grid).
    dist_all : list[float]
        Corresponding distances for every draw (ABC only; empty for grid).
    epsilon : float or None
        Acceptance threshold used by ABC.  None for grid.
    n_accepted : int
        Number of accepted ABC draws.  0 for grid.
    """
    R0_estimate:   float
    R0_ci_low:     Optional[float]
    R0_ci_high:    Optional[float]
    best_distance: float
    method:        str
    I_fitted:      list
    R_fitted:      list
    days:          list
    I_obs:         list
    R_obs:         list
    R0_all:        list  = field(default_factory=list)
    dist_all:      list  = field(default_factory=list)
    epsilon:       Optional[float] = None
    n_accepted:    int   = 0


# ---------------------------------------------------------------------------
# 1.  Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[list[int], list[int], list[int]]:
    """
    Load empirical time series from a CSV file.

    Required column : 'I'  – daily infectious (or incident) count
    Optional columns: 'day', 'R'

    Returns (days, I_obs, R_obs).
    R_obs is all-zeros when the column is absent.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the CSV is malformed or has fewer than 3 time points.
    """
    days:  list[int] = []
    I_obs: list[int] = []
    R_obs: list[int] = []

    with open(path, newline="") as fh:
        reader  = csv.DictReader(fh)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "i" not in headers:
            raise ValueError(
                f"CSV must contain a column named 'I'.  Found: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader):
            row_lower = {k.strip().lower(): v.strip() for k, v in row.items()}
            try:
                I_obs.append(int(float(row_lower["i"])))
            except (ValueError, KeyError):
                raise ValueError(
                    f"Non-numeric value in column 'I' at row {row_idx + 2}."
                )
            days.append(int(row_lower.get("day", str(row_idx))))
            R_obs.append(int(float(row_lower.get("r", "0"))))

    if len(I_obs) < 3:
        raise ValueError("Need at least 3 time points to fit.")

    return days, I_obs, R_obs


# ---------------------------------------------------------------------------
# 2.  Simulation wrapper: ensemble average
# ---------------------------------------------------------------------------

def simulate_ensemble(
    R0: float,
    n_nodes: int,
    incubation_days: float,
    infectious_days: float,
    superspreading: bool,
    k_disp: float,
    k: int,
    p_rewire: float,
    n_replicates: int,
    obs_length: int,
    base_seed: int = 0,
) -> tuple[list[float], list[float]]:
    """
    Run `n_replicates` stochastic SEIR simulations and return the
    element-wise mean I and R series, each truncated / padded to
    exactly `obs_length` time steps.
    """
    I_acc = [0.0] * obs_length
    R_acc = [0.0] * obs_length

    for rep in range(n_replicates):
        _, _, I_sim, R_sim, _ = run_seir(
            n_nodes         = n_nodes,
            R0              = R0,
            incubation_days = incubation_days,
            infectious_days = infectious_days,
            superspreading  = superspreading,
            k_disp          = k_disp,
            k               = k,
            p_rewire        = p_rewire,
            seed            = base_seed + rep,
        )
        for t in range(obs_length):
            I_acc[t] += I_sim[t] if t < len(I_sim) else 0
            R_acc[t] += R_sim[t] if t < len(R_sim) else 0

    I_mean = [v / n_replicates for v in I_acc]
    R_mean = [v / n_replicates for v in R_acc]
    return I_mean, R_mean


# ---------------------------------------------------------------------------
# 3.  Distance / loss functions
# ---------------------------------------------------------------------------

def distance(
    I_obs: list[int],
    R_obs: list[int],
    I_sim: list[float],
    R_sim: list[float],
    n_nodes: int,
    use_R: bool,
) -> float:
    """
    Normalised root-mean-square error between observed and simulated curves.

    Both I and R series are normalised by n_nodes so they sit in [0,1],
    making the distance scale-free and comparable across population sizes.
    If use_R is True and R_obs is non-trivial, the distance is the
    average of the I-RMSE and R-RMSE; otherwise only I is used.
    """
    T   = len(I_obs)
    N   = float(n_nodes)

    rmse_I = math.sqrt(
        sum((I_obs[t] / N - I_sim[t] / N) ** 2 for t in range(T)) / T
    )

    if use_R and any(r > 0 for r in R_obs):
        rmse_R = math.sqrt(
            sum((R_obs[t] / N - R_sim[t] / N) ** 2 for t in range(T)) / T
        )
        return (rmse_I + rmse_R) / 2.0

    return rmse_I


# ---------------------------------------------------------------------------
# 4a.  Grid-search fit (to get a quick and rough estimate)
# ---------------------------------------------------------------------------

def fit_grid(
    I_obs: list[int],
    R_obs: list[int],
    n_nodes: int,
    incubation_days: float,
    infectious_days: float,
    superspreading: bool,
    k_disp: float,
    k: int,
    p_rewire: float,
    n_replicates: int,
    R0_min: float,
    R0_max: float,
    grid_points: int,
    verbose: bool,
) -> tuple[float, float, list[float], list[float], list[float]]:
    """
    Evaluate the RMSE on a regular grid of R0 values and return the minimiser.

    Returns (best_R0, best_rmse, R0_grid, rmse_grid, [I_fit, R_fit]).
    """
    T        = len(I_obs)
    use_R    = any(r > 0 for r in R_obs)
    step     = (R0_max - R0_min) / (grid_points - 1)
    R0_grid  = [R0_min + i * step for i in range(grid_points)]
    rmse_grid: list[float] = []

    best_R0   = R0_grid[0]
    best_rmse = float("inf")
    best_I: list[float] = []
    best_R: list[float] = []

    for idx, R0 in enumerate(R0_grid):
        I_sim, R_sim = simulate_ensemble(
            R0, n_nodes, incubation_days, infectious_days,
            superspreading, k_disp, k, p_rewire, n_replicates, T,
        )
        d = distance(I_obs, R_obs, I_sim, R_sim, n_nodes, use_R)
        rmse_grid.append(d)

        if d < best_rmse:
            best_rmse = d
            best_R0   = R0
            best_I    = I_sim
            best_R    = R_sim

        if verbose:
            bar  = "█" * int(40 * (idx + 1) / grid_points)
            pad  = " " * (40 - len(bar))
            print(f"\r  [{bar}{pad}]  R0={R0:.3f}  rmse={d:.4f}", end="", flush=True)

    if verbose:
        print()

    return best_R0, best_rmse, R0_grid, rmse_grid, best_I, best_R


# ---------------------------------------------------------------------------
# 4b.  ABC rejection-sampling fit
# ---------------------------------------------------------------------------

def fit_abc(
    I_obs: list[int],
    R_obs: list[int],
    n_nodes: int,
    incubation_days: float,
    infectious_days: float,
    superspreading: bool,
    k_disp: float,
    k: int,
    p_rewire: float,
    n_replicates: int,
    R0_min: float,
    R0_max: float,
    n_samples: int,
    epsilon: Optional[float],
    accept_quantile: float,
    verbose: bool,
    seed: int,
) -> tuple[float, float, float, float, list[float], list[float], list[float], list[float]]:
    """
    ABC rejection sampler.

    1. Draw R0 ~ Uniform(R0_min, R0_max).
    2. Simulate and compute distance.
    3. Collect all (R0, distance) pairs from n_samples draws.
    4. If epsilon is None, auto-set it to the accept_quantile-th percentile
       of distances (e.g. best 20%).
    5. Return posterior summary statistics and the best-fit curve.

    Returns
    -------
    (R0_median, R0_lo, R0_hi, best_R0, best_dist,
     R0_all, dist_all, I_best, R_best)
    """
    rng   = random.Random(seed)
    T     = len(I_obs)
    use_R = any(r > 0 for r in R_obs)

    R0_all:   list[float] = []
    dist_all: list[float] = []

    best_dist = float("inf")
    best_I:   list[float] = []
    best_R:   list[float] = []
    best_R0   = (R0_min + R0_max) / 2.0

    for s in range(n_samples):
        R0_try = rng.uniform(R0_min, R0_max)
        I_sim, R_sim = simulate_ensemble(
            R0_try, n_nodes, incubation_days, infectious_days,
            superspreading, k_disp, k, p_rewire, n_replicates, T,
            base_seed=s * 100,
        )
        d = distance(I_obs, R_obs, I_sim, R_sim, n_nodes, use_R)
        R0_all.append(R0_try)
        dist_all.append(d)

        if d < best_dist:
            best_dist = d
            best_R0   = R0_try
            best_I    = I_sim
            best_R    = R_sim

        if verbose:
            bar = "█" * int(40 * (s + 1) / n_samples)
            pad = " " * (40 - len(bar))
            print(f"\r  [{bar}{pad}]  sample {s+1}/{n_samples}  "
                  f"best_R0={best_R0:.3f}  d={best_dist:.4f}",
                  end="", flush=True)

    if verbose:
        print()

    # Auto-calibrate epsilon if not given
    if epsilon is None:
        sorted_dists = sorted(dist_all)
        cutoff_idx   = max(1, int(len(sorted_dists) * accept_quantile))
        epsilon      = sorted_dists[cutoff_idx - 1]

    accepted = [r for r, d in zip(R0_all, dist_all) if d <= epsilon]

    if not accepted:
        # Fallback: just use the best single draw
        accepted = [best_R0]

    accepted.sort()
    n_acc   = len(accepted)
    R0_med  = accepted[n_acc // 2]
    R0_lo   = accepted[max(0, int(0.05 * n_acc))]
    R0_hi   = accepted[min(n_acc - 1, int(0.95 * n_acc))]

    return (R0_med, R0_lo, R0_hi, best_R0, best_dist,
            R0_all, dist_all, best_I, best_R)


# ---------------------------------------------------------------------------
# 5.  Reporting
# ---------------------------------------------------------------------------

def print_grid_report(
    best_R0: float, best_rmse: float,
    incubation_days: float, infectious_days: float,
    n_nodes: int, n_replicates: int,
) -> None:
    w = 54
    d = "=" * w
    print(f"\n{d}")
    print(f"  R0 Fit  –  Grid Search")
    print(d)
    print(f"  Best R0            : {best_R0:.4f}")
    print(f"  RMSE (normalised)  : {best_rmse:.6f}")
    print(f"  Nodes              : {n_nodes}")
    print(f"  Incubation (days)  : {incubation_days}")
    print(f"  Infectious (days)  : {infectious_days}")
    print(f"  Replicates / R0    : {n_replicates}")
    print(f"{d}\n")


def print_abc_report(
    R0_med: float, R0_lo: float, R0_hi: float,
    best_dist: float, n_accepted: int, n_samples: int,
    incubation_days: float, infectious_days: float,
    n_nodes: int, n_replicates: int,
) -> None:
    w = 54
    d = "=" * w
    print(f"\n{d}")
    print(f"  R0 Fit  –  Approximate Bayesian Computation")
    print(d)
    print(f"  R0 estimate (median)   : {R0_med:.4f}")
    print(f"  90% credible interval  : [{R0_lo:.4f}, {R0_hi:.4f}]")
    print(f"  Best distance          : {best_dist:.6f}")
    print(f"  Accepted / Total draws : {n_accepted} / {n_samples}")
    print(f"  Nodes                  : {n_nodes}")
    print(f"  Incubation (days)      : {incubation_days}")
    print(f"  Infectious (days)      : {infectious_days}")
    print(f"  Replicates / draw      : {n_replicates}")
    print(f"{d}\n")


# ---------------------------------------------------------------------------
# 6.  CSV export of fitted curve
# ---------------------------------------------------------------------------

def save_fitted_curve(
    path: str,
    days: list[int],
    I_obs: list[int],
    R_obs: list[int],
    I_fit: list[float],
    R_fit: list[float],
) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "I_observed", "R_observed", "I_fitted", "R_fitted"])
        for t, day in enumerate(days):
            writer.writerow([
                day,
                I_obs[t],
                R_obs[t],
                f"{I_fit[t]:.2f}",
                f"{R_fit[t]:.2f}",
            ])
    print(f"  Fitted curve saved to: {path}")


# ---------------------------------------------------------------------------
# 7.  Public API  –  fit_r0()
# ---------------------------------------------------------------------------

def fit_r0(
    I_obs: list[int],
    *,
    R_obs: Optional[list[int]]   = None,
    days:  Optional[list[int]]   = None,
    # Model parameters
    n_nodes:         int   = 1000,
    incubation_days: float = 5.0,
    infectious_days: float = 7.0,
    superspreading:  bool  = False,
    k_disp:          float = 0.1,
    k:               int   = 4,
    p_rewire:        float = 0.1,
    # Search bounds
    R0_min: float = 0.5,
    R0_max: float = 6.0,
    # Method
    method: str = "abc",
    # ABC parameters
    n_samples:       int            = 500,
    epsilon:         Optional[float]= None,
    accept_quantile: float          = 0.20,
    # Grid parameters
    grid_points: int = 30,
    # Shared
    n_replicates: int = 10,
    seed:         int = 42,
    verbose:      bool = True,
) -> FitResult:
    """
    Fit R0 to an observed epidemic time series using the SEIR small-world model.

    This is the main public entry point.  All internal helpers (``fit_abc``,
    ``fit_grid``, ``simulate_ensemble``, …) remain available but are considered
    implementation details.

    Parameters
    ----------
    I_obs : list[int]
        Observed daily infectious counts (required).
    R_obs : list[int], optional
        Observed daily recovered counts.  When provided (and non-zero), it is
        included in the distance metric, which tightens the R0 estimate.
        Pass ``None`` or ``[]`` to use only I.
    days : list[int], optional
        Day indices for each time point.  Defaults to ``[0, 1, 2, …]``.
    n_nodes : int
        Number of nodes in the small-world network (default 1000).
    incubation_days : float
        Mean incubation period in days – σ⁻¹ (default 5.0).
    infectious_days : float
        Mean infectious period in days – γ⁻¹ (default 7.0).
    superspreading : bool
        Enable individual-level infectiousness heterogeneity (default False).
    k_disp : float
        Gamma dispersion parameter for superspreading (default 0.1).
    k : int
        Ring-lattice degree of the Watts-Strogatz network (default 4).
    p_rewire : float
        Edge rewiring probability (default 0.1).
    R0_min, R0_max : float
        Search bounds for R0 (defaults 0.5 – 6.0).
    method : {"abc", "grid"}
        Fitting algorithm.  "abc" returns a credible interval; "grid" is
        faster but returns only a point estimate (default "abc").
    n_samples : int
        [ABC] Number of random R0 draws (default 500).
    epsilon : float or None
        [ABC] Acceptance threshold ε.  Auto-calibrated when None (default).
    accept_quantile : float
        [ABC] Fraction of draws to accept when auto-calibrating ε (default 0.20).
    grid_points : int
        [Grid] Number of candidate R0 values (default 30).
    n_replicates : int
        Stochastic runs averaged per R0 candidate (default 10).
    seed : int
        Master random seed (default 42).
    verbose : bool
        Print a progress bar during fitting (default True).

    Returns
    -------
    FitResult
        Dataclass with R0_estimate, credible interval, fitted curves, and
        full ABC diagnostics.  See ``FitResult`` docstring for all fields.

    Examples
    --------
    Basic usage with a list of counts::

        from fit_r0 import fit_r0

        I_obs = [1, 1, 3, 8, 20, 45, 80, 95, 85, 60, 35, 15, 5, 1]
        result = fit_r0(I_obs, n_nodes=500, incubation_days=5,
                        infectious_days=7, n_samples=200)
        print(f"R0 = {result.R0_estimate:.2f}  "
              f"90% CI [{result.R0_ci_low:.2f}, {result.R0_ci_high:.2f}]")

    Loading from a CSV file::

        from fit_r0 import fit_r0, load_data

        days, I_obs, R_obs = load_data("outbreak.csv")
        result = fit_r0(I_obs, R_obs=R_obs, days=days, n_nodes=1000)
    """
    if method not in ("abc", "grid"):
        raise ValueError(f"method must be 'abc' or 'grid', got {method!r}")
    if len(I_obs) < 3:
        raise ValueError("I_obs must have at least 3 time points.")

    # Normalise optional inputs
    R_obs_safe: list[int] = list(R_obs) if R_obs else [0] * len(I_obs)
    days_safe:  list[int] = list(days)  if days  else list(range(len(I_obs)))

    common = dict(
        n_nodes         = n_nodes,
        incubation_days = incubation_days,
        infectious_days = infectious_days,
        superspreading  = superspreading,
        k_disp          = k_disp,
        k               = k,
        p_rewire        = p_rewire,
        n_replicates    = n_replicates,
        R0_min          = R0_min,
        R0_max          = R0_max,
        verbose         = verbose,
    )

    if method == "grid":
        best_R0, best_rmse, _, _, I_fit, R_fit = fit_grid(
            I_obs, R_obs_safe, grid_points=grid_points, **common
        )
        return FitResult(
            R0_estimate   = best_R0,
            R0_ci_low     = None,
            R0_ci_high    = None,
            best_distance = best_rmse,
            method        = "grid",
            I_fitted      = I_fit,
            R_fitted      = R_fit,
            days          = days_safe,
            I_obs         = list(I_obs),
            R_obs         = R_obs_safe,
        )

    # ABC
    (R0_med, R0_lo, R0_hi, _best_R0, best_dist,
     R0_all, dist_all, I_best, R_best) = fit_abc(
        I_obs, R_obs_safe,
        n_samples       = n_samples,
        epsilon         = epsilon,
        accept_quantile = accept_quantile,
        seed            = seed,
        **common,
    )

    # Compute the epsilon that was actually used (for diagnostics)
    if epsilon is not None:
        eps_used = epsilon
    else:
        sorted_d  = sorted(dist_all)
        cutoff    = max(1, int(len(sorted_d) * accept_quantile))
        eps_used  = sorted_d[cutoff - 1]
    n_accepted = sum(1 for d in dist_all if d <= eps_used)

    return FitResult(
        R0_estimate   = R0_med,
        R0_ci_low     = R0_lo,
        R0_ci_high    = R0_hi,
        best_distance = best_dist,
        method        = "abc",
        I_fitted      = I_best,
        R_fitted      = R_best,
        days          = days_safe,
        I_obs         = list(I_obs),
        R_obs         = R_obs_safe,
        R0_all        = R0_all,
        dist_all      = dist_all,
        epsilon       = eps_used,
        n_accepted    = n_accepted,
    )


# ---------------------------------------------------------------------------
# 8.  Standalone plot helper (importable)
# ---------------------------------------------------------------------------

def plot_fit(result: FitResult) -> None:
    """
    Display a matplotlib figure for a ``FitResult``.

    Shows observed vs fitted I (and R when non-trivial), plus an ABC
    posterior histogram when ``result.method == "abc"``.

    Raises ``ImportError`` if matplotlib is not installed.
    """
    import matplotlib.pyplot as plt

    T          = len(result.I_obs)
    days       = result.days
    has_R      = any(r > 0 for r in result.R_obs)
    is_abc     = result.method == "abc" and bool(result.R0_all)
    n_panels   = 2 if is_abc else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(13 if is_abc else 8, 5))
    ax_curve  = axes[0] if is_abc else axes

    # Observed vs fitted
    ax_curve.scatter(days, result.I_obs, color="#F44336", zorder=5,
                     label="I observed", s=40)
    ax_curve.plot(days, result.I_fitted[:T], color="#F44336", linewidth=2,
                  linestyle="--", label="I fitted")
    if has_R:
        ax_curve.scatter(days, result.R_obs, color="#4CAF50", zorder=5,
                         label="R observed", s=40, marker="^")
        ax_curve.plot(days, result.R_fitted[:T], color="#4CAF50", linewidth=2,
                      linestyle="--", label="R fitted")

    ci_str = ""
    if result.R0_ci_low is not None:
        ci_str = f"  90% CI [{result.R0_ci_low:.2f}, {result.R0_ci_high:.2f}]"
    ax_curve.set_title(f"Best fit  R₀ = {result.R0_estimate:.3f}{ci_str}")
    ax_curve.set_xlabel("Day")
    ax_curve.set_ylabel("Count")
    ax_curve.legend()
    ax_curve.grid(alpha=0.3)

    # ABC posterior
    if is_abc:
        ax_post = axes[1]
        ax_post.hist(result.R0_all, bins=30, color="#9E9E9E", alpha=0.6,
                     label="All draws")
        accepted = [r for r, d in zip(result.R0_all, result.dist_all)
                    if d <= (result.epsilon or float("inf"))]
        ax_post.hist(accepted, bins=20, color="#2196F3", alpha=0.8,
                     label=f"Accepted (ε={result.epsilon:.4f})")
        ax_post.axvline(result.R0_estimate, color="#F44336", linewidth=2,
                        label=f"Median R₀={result.R0_estimate:.3f}")
        if result.R0_ci_low is not None:
            ax_post.axvline(result.R0_ci_low,  color="#F44336", linewidth=1,
                            linestyle=":", label=f"5th pct={result.R0_ci_low:.3f}")
            ax_post.axvline(result.R0_ci_high, color="#F44336", linewidth=1,
                            linestyle=":", label=f"95th pct={result.R0_ci_high:.3f}")
        ax_post.set_xlabel("R0")
        ax_post.set_ylabel("Count")
        ax_post.set_title("ABC posterior")
        ax_post.legend(fontsize=8)
        ax_post.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 9.  Entry point 
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit R0 to empirical SEIR data via ABC or grid search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--data",        required=True,
                        help="Path to CSV file with empirical data  "
                             "(required column: 'I';  optional: 'day', 'R')")

    # ── Model parameters (fixed during fitting) ───────────────────────────
    parser.add_argument("--nodes",       type=int,   default=1000,
                        help="Number of nodes in the network")
    parser.add_argument("--incubation",  type=float, default=5.0,
                        help="Mean incubation period (days)")
    parser.add_argument("--infectious",  type=float, default=7.0,
                        help="Mean infectious period (days)")
    parser.add_argument("--k",           type=int,   default=4,
                        help="Ring-lattice degree")
    parser.add_argument("--p",           type=float, default=0.1,
                        help="Edge rewiring probability")

    # ── Superspreading (forwarded to run_seir unchanged) ──────────────────
    parser.add_argument("--superspreading", action="store_true",
                        help="Enable superspreading heterogeneity during fitting")
    parser.add_argument("--k-disp",      type=float, default=0.1,
                        dest="k_disp",
                        help="Superspreading dispersion parameter k")

    # ── Search bounds for R0 ──────────────────────────────────────────────
    parser.add_argument("--R0-min",      type=float, default=0.5,
                        dest="R0_min",
                        help="Lower bound of R0 search range")
    parser.add_argument("--R0-max",      type=float, default=6.0,
                        dest="R0_max",
                        help="Upper bound of R0 search range")

    # ── Fitting method ────────────────────────────────────────────────────
    parser.add_argument("--method",      choices=["abc", "grid"], default="abc",
                        help="Fitting method: 'abc' (Approximate Bayesian "
                             "Computation, recommended) or 'grid' (grid search)")

    # ABC-specific
    parser.add_argument("--n-samples",   type=int,   default=500,
                        dest="n_samples",
                        help="[ABC] Number of random R0 draws")
    parser.add_argument("--epsilon",     type=float, default=None,
                        help="[ABC] Acceptance threshold ε.  "
                             "If omitted, auto-set to the best "
                             "--accept-quantile fraction of distances.")
    parser.add_argument("--accept-quantile", type=float, default=0.20,
                        dest="accept_quantile",
                        help="[ABC] Fraction of draws to accept when "
                             "auto-calibrating ε  (e.g. 0.20 = best 20%%)")

    # Grid-specific
    parser.add_argument("--grid-points", type=int,   default=30,
                        dest="grid_points",
                        help="[Grid] Number of R0 values to evaluate")

    # Shared
    parser.add_argument("--replicates",  type=int,   default=10,
                        help="Stochastic replicates per R0 evaluation "
                             "(higher = smoother but slower)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--out",         type=str,   default=None,
                        help="Optional path to save the fitted curve CSV")
    parser.add_argument("--plot",        action="store_true",
                        help="Show a matplotlib comparison plot")
    parser.add_argument("--quiet",       action="store_true",
                        help="Suppress progress output")

    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────
    try:
        days, I_obs, R_obs = load_data(args.data)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error loading data: {exc}")

    print(f"\n  Loaded {len(I_obs)} time points from '{args.data}'")
    print(f"  Peak I_obs = {max(I_obs)} on day {I_obs.index(max(I_obs))}\n")

    if args.method == "grid":
        print("  Running grid search …")
    else:
        print("  Running ABC rejection sampler …")

    # ── Run fitting via public API ─────────────────────────────────────────
    result = fit_r0(
        I_obs           = I_obs,
        R_obs           = R_obs,
        days            = days,
        n_nodes         = args.nodes,
        incubation_days = args.incubation,
        infectious_days = args.infectious,
        superspreading  = args.superspreading,
        k_disp          = args.k_disp,
        k               = args.k,
        p_rewire        = args.p,
        R0_min          = args.R0_min,
        R0_max          = args.R0_max,
        method          = args.method,
        n_samples       = args.n_samples,
        epsilon         = args.epsilon,
        accept_quantile = args.accept_quantile,
        grid_points     = args.grid_points,
        n_replicates    = args.replicates,
        seed            = args.seed,
        verbose         = not args.quiet,
    )

    # ── Print report ───────────────────────────────────────────────────────
    if result.method == "grid":
        print_grid_report(
            result.R0_estimate, result.best_distance,
            args.incubation, args.infectious,
            args.nodes, args.replicates,
        )
    else:
        print_abc_report(
            result.R0_estimate,
            result.R0_ci_low,
            result.R0_ci_high,
            result.best_distance,
            result.n_accepted,
            args.n_samples,
            args.incubation,
            args.infectious,
            args.nodes,
            args.replicates,
        )

    # ── Export ─────────────────────────────────────────────────────────────
    if args.out:
        save_fitted_curve(
            args.out, result.days, result.I_obs, result.R_obs,
            result.I_fitted, result.R_fitted,
        )

    # ── Plot ───────────────────────────────────────────────────────────────
    if args.plot:
        try:
            plot_fit(result)
        except ImportError:
            print("matplotlib not installed – skipping plot.")
