"""
fit_infec_per.py
================
Fit the infectious period T_I = 1/γ of the SEIR small-world network model
against empirical epidemic data using Approximate Bayesian Computation (ABC).

R0 and the incubation period are treated as known (user-supplied).
Optionally, the user may also supply the observed serial interval (SI), which
adds a second term to the distance metric and tightens the posterior when the
I-series alone is weakly informative about T_I.

Serial interval constraint
--------------------------
For a SEIR model the expected serial interval is:

    SI = T_E + T_I / 2                  (approximate, see §3 of description)

so SI ≈ incubation_days + infectious_days / 2.

When ``serial_interval`` is provided, a second distance term is added:

    d_SI = |SI_candidate - SI_obs| / SI_obs

The composite distance is a weighted sum:

    d_total = (1 − w_si) · d_curve + w_si · d_SI

where ``si_weight`` (default 0.3) controls the relative importance of the
serial-interval constraint.  Set ``si_weight = 0`` to ignore the SI entirely
(equivalent to not passing ``serial_interval``).

Input CSV format (if used)
----------------
Required column: 'I'  (daily infectious count)
Optional columns: 'day', 'R'

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
# Simulation import (requires companioin functions)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from seir_network_model import run_seir
except ImportError:
    try:
        from epidemic_seir_smallworld import run_seir
    except ImportError:
        sys.exit(
            "Cannot find seir_network_model.py or epidemic_seir_smallworld.py.\n"
            "Place one in the same directory as fit_infec_per.py."
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InfecPeriodResult:
    """
    Output of ``fit_infectious_period()``.

    Fields
    ------
    TI_estimate   : float        – point estimate of T_I (ABC median or grid best)
    TI_ci_low     : float|None   – 5th-percentile of ABC posterior  (None for grid)
    TI_ci_high    : float|None   – 95th-percentile of ABC posterior (None for grid)
    best_distance : float        – composite distance at TI_estimate
    method        : str          – "abc" or "grid"
    R0            : float        – user-supplied R0 (echoed)
    incubation_days : float      – user-supplied T_E (echoed)
    serial_interval : float|None – user-supplied SI (echoed; None if not given)
    si_weight     : float        – weight applied to the SI distance term
    I_fitted      : list[float]  – ensemble-averaged fitted I series
    R_fitted      : list[float]  – ensemble-averaged fitted R series
    days          : list[int]    – day indices (mirrors observed)
    I_obs         : list[int]    – observed I (echoed)
    R_obs         : list[int]    – observed R (echoed)
    TI_all        : list[float]  – all ABC draws (empty for grid)
    dist_all      : list[float]  – composite distances for all draws (empty for grid)
    dist_curve    : list[float]  – curve-only RMSE for all draws (empty for grid)
    dist_si       : list[float]  – SI distances for all draws (empty for grid)
    epsilon       : float|None   – ABC acceptance threshold (None for grid)
    n_accepted    : int          – accepted draws (0 for grid)
    implied_SI    : float        – SI implied by TI_estimate: T_E + T_I/2
    """
    TI_estimate:     float
    TI_ci_low:       Optional[float]
    TI_ci_high:      Optional[float]
    best_distance:   float
    method:          str
    R0:              float
    incubation_days: float
    serial_interval: Optional[float]
    si_weight:       float
    I_fitted:        list
    R_fitted:        list
    days:            list
    I_obs:           list
    R_obs:           list
    TI_all:          list            = field(default_factory=list)
    dist_all:        list            = field(default_factory=list)
    dist_curve:      list            = field(default_factory=list)
    dist_si:         list            = field(default_factory=list)
    epsilon:         Optional[float] = None
    n_accepted:      int             = 0
    implied_SI:      float           = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[list[int], list[int], list[int]]:
    """
    Load epidemic time series from a CSV file.

    Required column : 'I'  – daily infectious count
    Optional columns: 'day', 'R'

    Returns
    -------
    (days, I_obs, R_obs)

    Raises
    ------
    FileNotFoundError  – path does not exist
    ValueError         – malformed CSV or fewer than 3 time points
    """
    days:  list[int] = []
    I_obs: list[int] = []
    R_obs: list[int] = []

    with open(path, newline="") as fh:
        reader  = csv.DictReader(fh)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        if "i" not in headers:
            raise ValueError(
                f"CSV must contain column 'I'. Found: {reader.fieldnames}"
            )
        for row_idx, row in enumerate(reader):
            r = {k.strip().lower(): v.strip() for k, v in row.items()}
            try:
                I_obs.append(int(float(r["i"])))
            except (ValueError, KeyError):
                raise ValueError(f"Non-numeric 'I' at row {row_idx + 2}.")
            days.append(int(r.get("day", str(row_idx))))
            R_obs.append(int(float(r.get("r", "0"))))

    if len(I_obs) < 3:
        raise ValueError("Need at least 3 time points to fit.")
    return days, I_obs, R_obs


# ---------------------------------------------------------------------------
# Ensemble simulation at a given T_I
# ---------------------------------------------------------------------------

def _ensemble_ti(
    infectious_days: float,
    R0:              float,
    incubation_days: float,
    n_nodes:         int,
    superspreading:  bool,
    k_disp:          float,
    k:               int,
    p_rewire:        float,
    n_replicates:    int,
    obs_length:      int,
    base_seed:       int = 0,
) -> tuple[list[float], list[float]]:
    """
    Run n_replicates SEIR simulations at the given infectious_days and
    return the ensemble-averaged I and R series, each of length obs_length.
    """
    I_acc = [0.0] * obs_length
    R_acc = [0.0] * obs_length
    for rep in range(n_replicates):
        _, _, I_s, R_s, _ = run_seir(
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
            I_acc[t] += I_s[t] if t < len(I_s) else 0
            R_acc[t] += R_s[t] if t < len(R_s) else 0
    return [v / n_replicates for v in I_acc], [v / n_replicates for v in R_acc]


# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------

def _curve_rmse(
    I_obs: list[int],
    R_obs: list[int],
    I_sim: list[float],
    R_sim: list[float],
    n_nodes: int,
) -> float:
    """
    Normalised RMSE over I (and R when available).
    Both series are divided by n_nodes so the metric is scale-free in [0,1].
    When R_obs is non-trivial, the distance is the mean of I-RMSE and R-RMSE.
    """
    T = len(I_obs)
    N = float(n_nodes)
    ri = math.sqrt(sum((I_obs[t] / N - I_sim[t] / N) ** 2 for t in range(T)) / T)
    if any(r > 0 for r in R_obs):
        rr = math.sqrt(sum((R_obs[t] / N - R_sim[t] / N) ** 2 for t in range(T)) / T)
        return (ri + rr) / 2.0
    return ri


def _si_distance(
    infectious_days: float,
    incubation_days: float,
    serial_interval: float,
) -> float:
    """
    Relative absolute error between the candidate-implied serial interval and
    the observed one.

    For a SEIR model the mean serial interval is approximated as:

        SI ≈ T_E + T_I / 2

    The factor 1/2 arises because the infector, on average, transmits halfway
    through their infectious period (uniform sojourn time approximation).
    The relative error keeps the SI distance on the same [0,1] scale as the
    normalised curve RMSE, making the composite distance interpretable.
    """
    si_candidate = incubation_days + infectious_days / 2.0
    return abs(si_candidate - serial_interval) / serial_interval


def _composite_distance(
    I_obs:           list[int],
    R_obs:           list[int],
    I_sim:           list[float],
    R_sim:           list[float],
    n_nodes:         int,
    infectious_days: float,
    incubation_days: float,
    serial_interval: Optional[float],
    si_weight:       float,
) -> tuple[float, float, float]:
    """
    Compute (d_total, d_curve, d_si).

    d_total = (1 − w_si) · d_curve + w_si · d_SI

    When serial_interval is None or si_weight == 0, d_SI = 0 and
    d_total = d_curve exactly.
    """
    d_curve = _curve_rmse(I_obs, R_obs, I_sim, R_sim, n_nodes)

    if serial_interval is not None and si_weight > 0.0:
        d_si    = _si_distance(infectious_days, incubation_days, serial_interval)
        d_total = (1.0 - si_weight) * d_curve + si_weight * d_si
    else:
        d_si    = 0.0
        d_total = d_curve

    return d_total, d_curve, d_si


# ---------------------------------------------------------------------------
# Grid search over T_I (simple and quick fitting scheme, to check that a value can be found and get a rough quick estimate)
# ---------------------------------------------------------------------------

def _fit_ti_grid(
    I_obs:           list[int],
    R_obs:           list[int],
    R0:              float,
    incubation_days: float,
    n_nodes:         int,
    superspreading:  bool,
    k_disp:          float,
    k:               int,
    p_rewire:        float,
    n_replicates:    int,
    TI_min:          float,
    TI_max:          float,
    grid_points:     int,
    serial_interval: Optional[float],
    si_weight:       float,
    verbose:         bool,
) -> tuple[float, float, list[float], list[float], list[float], list[float]]:
    """
    Grid search for T_I.

    Returns (best_TI, best_dist, TI_grid, dist_grid, I_fit, R_fit).
    """
    T    = len(I_obs)
    step = (TI_max - TI_min) / max(grid_points - 1, 1)
    grid = [TI_min + i * step for i in range(grid_points)]

    best_TI, best_d = grid[0], float("inf")
    best_I:  list[float] = []
    best_R:  list[float] = []
    dist_grid: list[float] = []

    for idx, ti in enumerate(grid):
        I_sim, R_sim = _ensemble_ti(
            ti, R0, incubation_days, n_nodes, superspreading, k_disp,
            k, p_rewire, n_replicates, T, base_seed=idx * 100,
        )
        d_total, _, _ = _composite_distance(
            I_obs, R_obs, I_sim, R_sim, n_nodes,
            ti, incubation_days, serial_interval, si_weight,
        )
        dist_grid.append(d_total)
        if d_total < best_d:
            best_d, best_TI, best_I, best_R = d_total, ti, I_sim, R_sim

        if verbose:
            bar = "█" * int(40 * (idx + 1) / grid_points)
            print(f"\r  [{bar:<40}]  T_I={ti:.2f} d  dist={d_total:.5f}",
                  end="", flush=True)

    if verbose:
        print()

    return best_TI, best_d, grid, dist_grid, best_I, best_R


# ---------------------------------------------------------------------------
# ABC rejection sampler over T_I
# ---------------------------------------------------------------------------

def _fit_ti_abc(
    I_obs:           list[int],
    R_obs:           list[int],
    R0:              float,
    incubation_days: float,
    n_nodes:         int,
    superspreading:  bool,
    k_disp:          float,
    k:               int,
    p_rewire:        float,
    n_replicates:    int,
    TI_min:          float,
    TI_max:          float,
    n_samples:       int,
    epsilon:         Optional[float],
    accept_quantile: float,
    serial_interval: Optional[float],
    si_weight:       float,
    verbose:         bool,
    seed:            int,
) -> tuple[
    float, float, float,        # TI_med, TI_lo, TI_hi
    float, float,               # best_dist, best_TI
    list[float], list[float],   # I_best, R_best
    list[float], list[float], list[float],  # TI_all, dist_all, dist_curve_all
    list[float], int, float,    # dist_si_all, n_accepted, eps_used
]:
    """
    ABC rejection sampler for T_I.

    Draws T_I ~ Uniform(TI_min, TI_max), runs the ensemble, computes the
    composite distance (curve RMSE + optional SI penalty), and accepts when
    distance ≤ ε.  ε is auto-calibrated if not provided.
    """
    T   = len(I_obs)
    rng = random.Random(seed)

    TI_all:         list[float] = []
    dist_all:       list[float] = []
    dist_curve_all: list[float] = []
    dist_si_all:    list[float] = []

    best_d,  best_TI  = float("inf"), (TI_min + TI_max) / 2.0
    best_I:  list[float] = []
    best_R:  list[float] = []

    for s in range(n_samples):
        ti_try = rng.uniform(TI_min, TI_max)

        I_sim, R_sim = _ensemble_ti(
            ti_try, R0, incubation_days, n_nodes, superspreading, k_disp,
            k, p_rewire, n_replicates, T, base_seed=s * 100,
        )
        d_total, d_curve, d_si = _composite_distance(
            I_obs, R_obs, I_sim, R_sim, n_nodes,
            ti_try, incubation_days, serial_interval, si_weight,
        )

        TI_all.append(ti_try)
        dist_all.append(d_total)
        dist_curve_all.append(d_curve)
        dist_si_all.append(d_si)

        if d_total < best_d:
            best_d, best_TI, best_I, best_R = d_total, ti_try, I_sim, R_sim

        if verbose:
            bar = "█" * int(40 * (s + 1) / n_samples)
            print(f"\r  [{bar:<40}]  {s+1}/{n_samples}"
                  f"  best T_I={best_TI:.2f} d  d={best_d:.5f}",
                  end="", flush=True)

    if verbose:
        print()

    # Auto-calibrate ε
    if epsilon is None:
        sd      = sorted(dist_all)
        epsilon = sd[max(1, int(len(sd) * accept_quantile)) - 1]

    accepted = sorted([ti for ti, d in zip(TI_all, dist_all) if d <= epsilon])
    if not accepted:
        accepted = [best_TI]

    n        = len(accepted)
    TI_med   = accepted[n // 2]
    TI_lo    = accepted[max(0, int(0.05 * n))]
    TI_hi    = accepted[min(n - 1, int(0.95 * n))]
    n_accept = sum(1 for d in dist_all if d <= epsilon)

    return (TI_med, TI_lo, TI_hi, best_d, best_TI,
            best_I, best_R,
            TI_all, dist_all, dist_curve_all, dist_si_all,
            n_accept, epsilon)


# ---------------------------------------------------------------------------
# Public API – fit_infectious_period
# ---------------------------------------------------------------------------

def fit_infectious_period(
    I_obs: list[int],
    *,
    R_obs:           Optional[list[int]]   = None,
    days:            Optional[list[int]]   = None,
    R0:              float = 2.5,
    incubation_days: float = 5.0,
    n_nodes:         int   = 1000,
    superspreading:  bool  = False,
    k_disp:          float = 0.1,
    k:               int   = 4,
    p_rewire:        float = 0.1,
    TI_min:          float = 1.0,
    TI_max:          float = 21.0,
    serial_interval: Optional[float] = None,
    si_weight:       float = 0.30,
    method:          str   = "abc",
    n_samples:       int   = 500,
    epsilon:         Optional[float] = None,
    accept_quantile: float = 0.20,
    grid_points:     int   = 40,
    n_replicates:    int   = 10,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> InfecPeriodResult:
    """
    Fit the mean infectious period T_I = 1/γ to empirical data via ABC or
    grid search, treating R0 and the incubation period as known.

    Parameters
    ----------
    I_obs : list[int]
        Observed daily infectious counts (required).
    R_obs : list[int], optional
        Observed daily recovered counts.  Included in the distance metric
        when non-zero, tightening the T_I estimate.
    days : list[int], optional
        Day indices.  Defaults to [0, 1, 2, …].
    R0 : float
        Known basic reproduction number (default 2.5).
    incubation_days : float
        Known mean incubation period T_E in days (default 5.0).
    n_nodes : int
        Network size (default 1000).
    superspreading : bool
        Enable Gamma-modulated individual β (default False).
    k_disp : float
        Superspreading dispersion parameter (default 0.1).
    k : int
        Watts-Strogatz ring-lattice degree (default 4).
    p_rewire : float
        Edge rewiring probability (default 0.1).
    TI_min, TI_max : float
        Prior bounds for T_I in days (default 1.0 – 21.0).
    serial_interval : float or None
        Observed mean serial interval in days.  When given, a second
        distance term d_SI = |SI_candidate − SI_obs| / SI_obs is added,
        where SI_candidate = T_E + T_I / 2.  This acts as a soft constraint
        that concentrates the posterior around T_I values consistent with the
        observed SI.  Pass None to use only the epidemic curve (default None).
    si_weight : float
        Weight w in [0,1] for the SI distance term:
            d_total = (1 − w) · d_curve + w · d_SI
        Only used when serial_interval is provided (default 0.30).
    method : {"abc", "grid"}
        Fitting algorithm.  "abc" returns a credible interval (default).
        "grid" is faster but returns only a point estimate.
    n_samples : int
        [ABC] Number of T_I draws (default 500).
    epsilon : float or None
        [ABC] Acceptance threshold; auto-calibrated when None.
    accept_quantile : float
        [ABC] Fraction of draws to accept for auto-ε (default 0.20).
    grid_points : int
        [Grid] Number of T_I candidates (default 40).
    n_replicates : int
        Stochastic replicates per T_I candidate (default 10).
    seed : int
        Master random seed (default 42).
    verbose : bool
        Print progress bar (default True).

    Returns
    -------
    InfecPeriodResult

    Examples
    --------
    Minimal usage::

        result = fit_infectious_period(
            I_obs=[1, 3, 8, 20, 45, 80, 95, 85, 60, 35, 15, 5, 1],
            R0=2.5, incubation_days=5.0, n_nodes=500,
        )
        print(result.TI_estimate, result.TI_ci_low, result.TI_ci_high)

    With serial interval constraint::

        result = fit_infectious_period(
            I_obs=I_obs, R_obs=R_obs, days=days,
            R0=2.5, incubation_days=5.0,
            serial_interval=7.0, si_weight=0.3,
        )

    Notes
    -----
    The serial interval provides an independent constraint on T_I because it
    encodes information about transmission timing that is not captured by the
    shape of the epidemic curve alone.  However, si_weight should be chosen
    carefully: a very high weight (> 0.6) effectively ignores the epidemic
    curve and fits the SI only.  Values between 0.2 and 0.4 give a balanced
    constraint in typical settings.
    """
    if method not in ("abc", "grid"):
        raise ValueError(f"method must be 'abc' or 'grid', got {method!r}")
    if len(I_obs) < 3:
        raise ValueError("I_obs must have at least 3 time points.")
    if TI_min <= 0:
        raise ValueError(f"TI_min must be > 0, got {TI_min}")
    if TI_max <= TI_min:
        raise ValueError(f"TI_max ({TI_max}) must be > TI_min ({TI_min})")
    if not 0.0 <= si_weight <= 1.0:
        raise ValueError(f"si_weight must be in [0,1], got {si_weight}")

    R_obs_s = list(R_obs) if R_obs else [0] * len(I_obs)
    days_s  = list(days)  if days  else list(range(len(I_obs)))

    if verbose:
        si_str = (f"  SI constraint: SI_obs={serial_interval:.2f} d  "
                  f"weight={si_weight}"
                  if serial_interval is not None else "  No SI constraint")
        print(f"\n  Fitting T_I  |  R0={R0}  T_E={incubation_days} d  "
              f"T_I ∈ [{TI_min}, {TI_max}] d")
        print(si_str)
        print(f"  Method: {method}  "
              f"{'n_samples=' + str(n_samples) if method=='abc' else 'grid_points=' + str(grid_points)}"
              f"  replicates={n_replicates}\n")

    common = dict(
        R0=R0, incubation_days=incubation_days, n_nodes=n_nodes,
        superspreading=superspreading, k_disp=k_disp,
        k=k, p_rewire=p_rewire, n_replicates=n_replicates,
        serial_interval=serial_interval, si_weight=si_weight, verbose=verbose,
    )

    if method == "grid":
        best_TI, best_d, _, _, I_fit, R_fit = _fit_ti_grid(
            I_obs, R_obs_s, TI_min=TI_min, TI_max=TI_max,
            grid_points=grid_points, **common,
        )
        implied_SI = incubation_days + best_TI / 2.0
        return InfecPeriodResult(
            TI_estimate=best_TI, TI_ci_low=None, TI_ci_high=None,
            best_distance=best_d, method="grid",
            R0=R0, incubation_days=incubation_days,
            serial_interval=serial_interval, si_weight=si_weight,
            I_fitted=I_fit, R_fitted=R_fit,
            days=days_s, I_obs=list(I_obs), R_obs=R_obs_s,
            implied_SI=implied_SI,
        )

    # ABC
    (TI_med, TI_lo, TI_hi, best_d, _best_TI,
     best_I, best_R,
     TI_all, dist_all, dist_curve_all, dist_si_all,
     n_accept, eps_used) = _fit_ti_abc(
        I_obs, R_obs_s, TI_min=TI_min, TI_max=TI_max,
        n_samples=n_samples, epsilon=epsilon,
        accept_quantile=accept_quantile, seed=seed,
        **common,
    )
    implied_SI = incubation_days + TI_med / 2.0
    return InfecPeriodResult(
        TI_estimate=TI_med, TI_ci_low=TI_lo, TI_ci_high=TI_hi,
        best_distance=best_d, method="abc",
        R0=R0, incubation_days=incubation_days,
        serial_interval=serial_interval, si_weight=si_weight,
        I_fitted=best_I, R_fitted=best_R,
        days=days_s, I_obs=list(I_obs), R_obs=R_obs_s,
        TI_all=TI_all, dist_all=dist_all,
        dist_curve=dist_curve_all, dist_si=dist_si_all,
        epsilon=eps_used, n_accepted=n_accept,
        implied_SI=implied_SI,
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_fitted_curve(
    path:   str,
    result: InfecPeriodResult,
) -> None:
    """Write observed and fitted I/R series to CSV."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "I_observed", "R_observed", "I_fitted", "R_fitted"])
        for t, day in enumerate(result.days):
            w.writerow([
                day,
                result.I_obs[t],
                result.R_obs[t],
                f"{result.I_fitted[t]:.2f}",
                f"{result.R_fitted[t]:.2f}",
            ])
    print(f"  Saved fitted curve: {path}")


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def _print_report(result: InfecPeriodResult) -> None:
    w = 58; d = "=" * w
    print(f"\n{d}")
    si_str = (f"{result.serial_interval:.2f} d  (weight={result.si_weight})"
              if result.serial_interval else "not used")
    if result.method == "grid":
        print(f"  Infectious Period Fit  –  Grid Search\n{d}")
        print(f"  Best T_I           : {result.TI_estimate:.4f} d")
        print(f"  RMSE (composite)   : {result.best_distance:.6f}")
    else:
        print(f"  Infectious Period Fit  –  ABC\n{d}")
        print(f"  T_I estimate (med) : {result.TI_estimate:.4f} d")
        print(f"  90% credible int.  : "
              f"[{result.TI_ci_low:.4f}, {result.TI_ci_high:.4f}] d")
        print(f"  Best distance      : {result.best_distance:.6f}")
        print(f"  Accepted / Total   : {result.n_accepted}")
    print(f"  R0 (given)         : {result.R0}")
    print(f"  T_E (given)        : {result.incubation_days:.2f} d")
    print(f"  Serial interval    : {si_str}")
    print(f"  Implied SI         : {result.implied_SI:.3f} d  "
          f"(= T_E + T_I/2)")
    print(f"{d}\n")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_fit(
    result:   InfecPeriodResult,
    out_path: Optional[str] = None,
) -> None:
    """
    Two- or three-panel figure:
      Left  – observed vs fitted I/R curves
      Right – ABC posterior histogram of T_I (ABC only)
      Far right – composite vs curve-only distance scatter (when SI used)

    Raises ImportError if matplotlib is not available.
    """
    import matplotlib.pyplot as plt

    T       = len(result.I_obs)
    days    = result.days
    has_R   = any(r > 0 for r in result.R_obs)
    is_abc  = result.method == "abc" and bool(result.TI_all)
    has_si  = result.serial_interval is not None and is_abc

    n_panels = 1 + int(is_abc) + int(has_si)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ── Panel 1: observed vs fitted ──────────────────────────────────────
    ax = axes[0]
    ax.scatter(days, result.I_obs, color="#F44336", s=30, zorder=5,
               label="I observed")
    ax.plot(days, result.I_fitted[:T], color="#F44336", lw=2, ls="--",
            label="I fitted")
    if has_R:
        ax.scatter(days, result.R_obs, color="#4CAF50", s=30, marker="^",
                   zorder=5, label="R observed")
        ax.plot(days, result.R_fitted[:T], color="#4CAF50", lw=2, ls="--",
                label="R fitted")
    ci_str = (f"  90% CI [{result.TI_ci_low:.2f}, {result.TI_ci_high:.2f}] d"
              if result.TI_ci_low is not None else "")
    ax.set_title(f"T_I = {result.TI_estimate:.2f} d{ci_str}\n"
                 f"(R₀={result.R0}, T_E={result.incubation_days} d, "
                 f"implied SI={result.implied_SI:.2f} d)", fontsize=10)
    ax.set_xlabel("Day"); ax.set_ylabel("Count")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── Panel 2: ABC posterior ───────────────────────────────────────────
    if is_abc:
        ax2  = axes[1]
        eps  = result.epsilon or float("inf")
        ax2.hist(result.TI_all, bins=30, color="#9E9E9E", alpha=0.5,
                 label="All draws")
        accepted = [ti for ti, d in zip(result.TI_all, result.dist_all)
                    if d <= eps]
        ax2.hist(accepted, bins=20, color="#2196F3", alpha=0.8,
                 label=f"Accepted (ε={eps:.4f})")
        ax2.axvline(result.TI_estimate, color="#F44336", lw=2,
                    label=f"Median={result.TI_estimate:.2f} d")
        if result.TI_ci_low is not None:
            ax2.axvline(result.TI_ci_low,  color="#F44336", lw=1, ls=":")
            ax2.axvline(result.TI_ci_high, color="#F44336", lw=1, ls=":")
        if result.serial_interval:
            # Mark the T_I value implied by the observed SI
            ti_from_si = 2.0 * (result.serial_interval - result.incubation_days)
            if result.TI_ci_low is not None:
                ax2.axvline(ti_from_si, color="#FF9800", lw=1.5, ls="--",
                            label=f"T_I from SI={ti_from_si:.2f} d")
        ax2.set_xlabel("T_I (days)"); ax2.set_ylabel("Count")
        ax2.set_title("ABC posterior of T_I", fontsize=10)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── Panel 3: SI distance decomposition ───────────────────────────────
    if has_si:
        ax3 = axes[2]
        eps = result.epsilon or float("inf")
        colours = ["#2196F3" if d <= eps else "#BDBDBD"
                   for d in result.dist_all]
        ax3.scatter(result.dist_curve, result.dist_si,
                    c=colours, s=18, alpha=0.7, linewidths=0)
        ax3.set_xlabel("Curve RMSE  d_curve", fontsize=10)
        ax3.set_ylabel("SI distance  d_SI", fontsize=10)
        ax3.set_title(f"Distance decomposition\n"
                      f"blue=accepted, grey=rejected  (w={result.si_weight})",
                      fontsize=10)
        ax3.grid(alpha=0.3)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit infectious period T_I via ABC or grid search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data",        required=True,
                        help="CSV with column 'I' (required), 'day', 'R' (optional)")

    # Fixed model parameters
    parser.add_argument("--R0",          type=float, required=True,
                        help="Known basic reproduction number")
    parser.add_argument("--incubation",  type=float, required=True,
                        dest="incubation_days",
                        help="Known mean incubation period T_E (days)")
    parser.add_argument("--nodes",       type=int,   default=1000)
    parser.add_argument("--k",           type=int,   default=4,
                        help="Watts-Strogatz degree")
    parser.add_argument("--p",           type=float, default=0.1,
                        help="Rewiring probability")
    parser.add_argument("--superspreading", action="store_true")
    parser.add_argument("--k-disp",      type=float, default=0.1, dest="k_disp")

    # T_I search bounds
    parser.add_argument("--TI-min",      type=float, default=1.0,  dest="TI_min",
                        help="Lower bound for T_I prior (days)")
    parser.add_argument("--TI-max",      type=float, default=21.0, dest="TI_max",
                        help="Upper bound for T_I prior (days)")

    # Serial interval constraint
    parser.add_argument("--serial-interval", type=float, default=None,
                        dest="serial_interval",
                        help="Observed mean serial interval (days); "
                             "adds SI penalty to the distance metric")
    parser.add_argument("--si-weight",   type=float, default=0.30,
                        dest="si_weight",
                        help="Weight w for the SI distance term  "
                             "(d = (1−w)·d_curve + w·d_SI)")

    # Method
    parser.add_argument("--method",      choices=["abc", "grid"], default="abc")
    parser.add_argument("--n-samples",   type=int,   default=500, dest="n_samples")
    parser.add_argument("--epsilon",     type=float, default=None)
    parser.add_argument("--accept-quantile", type=float, default=0.20,
                        dest="accept_quantile")
    parser.add_argument("--grid-points", type=int,   default=40, dest="grid_points")
    parser.add_argument("--replicates",  type=int,   default=10)
    parser.add_argument("--seed",        type=int,   default=42)

    # Output
    parser.add_argument("--out",         type=str,   default=None,
                        help="CSV path for fitted curve")
    parser.add_argument("--plot",        action="store_true")
    parser.add_argument("--quiet",       action="store_true")

    args = parser.parse_args()

    try:
        days, I_obs, R_obs = load_data(args.data)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")

    print(f"\n  Loaded {len(I_obs)} points from '{args.data}'  "
          f"(peak I={max(I_obs)} on day {I_obs.index(max(I_obs))})")

    result = fit_infectious_period(
        I_obs            = I_obs,
        R_obs            = R_obs,
        days             = days,
        R0               = args.R0,
        incubation_days  = args.incubation_days,
        n_nodes          = args.nodes,
        k                = args.k,
        p_rewire         = args.p,
        superspreading   = args.superspreading,
        k_disp           = args.k_disp,
        TI_min           = args.TI_min,
        TI_max           = args.TI_max,
        serial_interval  = args.serial_interval,
        si_weight        = args.si_weight,
        method           = args.method,
        n_samples        = args.n_samples,
        epsilon          = args.epsilon,
        accept_quantile  = args.accept_quantile,
        grid_points      = args.grid_points,
        n_replicates     = args.replicates,
        seed             = args.seed,
        verbose          = not args.quiet,
    )

    _print_report(result)

    if args.out:
        save_fitted_curve(args.out, result)

    if args.plot:
        try:
            plot_fit(result, out_path=None)
        except ImportError:
            print("matplotlib not installed – skipping plot.")
