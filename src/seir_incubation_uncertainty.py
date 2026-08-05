"""
seir_incubation_uncertainty.py
==============================
Three functions that explore how uncertainty in the incubation period
propagates through SEIR epidemic dynamics.

Function 1 – sweep_incubation_range()
    Runs the network SEIR model for a fixed grid of incubation period values.
    Each value in the grid is treated as a deterministic parameter.

Function 2 – sample_incubation_gamma_network()
    Runs the network SEIR model repeatedly; at each run the incubation period
    is drawn fresh from Gamma(shape, scale), so the ensemble reflects the full
    parametric uncertainty.

Function 3 – seir_delay_gamma_ode()
    Simulates a mean-field SEIR compartment model where the E→I transition
    is governed by a delay distribution.  The Gamma distribution is
    approximated as a chain of n_stages identical exponential sub-stages
    (the linear chain trick), which gives exact Gamma(n_stages, scale/n_stages)
    delay distributions.  Each run draws (shape, scale) from the user-supplied
    Gamma prior and integrates the ODE forward.

All three functions return a list of result dicts and accept an optional
``plot`` flag that renders trajectory panels immediately.

"""

import argparse
import math
import os
import random
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from seir_network_model import run_seir    # Companion functions
except ImportError:
    try:
        from epidemic_seir_smallworld import run_seir
    except ImportError:
        sys.exit(
            "Cannot find seir_network_model.py or epidemic_seir_smallworld.py.\n"
            "Place one of them in the same directory as this script."
        )

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from scipy.integrate import solve_ivp
except ImportError as exc:
    sys.exit(f"numpy, matplotlib and scipy are required:  {exc}")


# ============================================================================
# Shared plotting helper
# ============================================================================

def plot_results(
    results:    list[dict],
    color_key:  str        = "incubation_days",
    color_label: str       = "Incubation period (days)",
    n_nodes:    int        = 1,          # set to 1 if series are already fractions
    normalise:  bool       = True,
    out_path:   Optional[str] = None,
) -> None:
    """
    Four-panel S/E/I/R plot for any of the three result lists.

    Trajectories are coloured by ``color_key`` (a field in each result dict).
    The trajectory closest to the median colour-key value is overlaid in black.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    #fig.suptitle(title, fontsize=13, y=1.02)

    compartments = [
        ("S", "Susceptible",  axes[0, 0]),
        ("E", "Exposed",      axes[0, 1]),
        ("I", "Infectious",   axes[1, 0]),
        ("R", "Removed",    axes[1, 1]),
    ]

    cvals   = [r[color_key] for r in results]
    cmin, cmax = min(cvals), max(cvals)
    crange  = max(cmax - cmin, 1e-9)
    cmap    = plt.cm.plasma
    norm    = mcolors.Normalize(vmin=cmin, vmax=cmax)

    median_c = sorted(cvals)[len(cvals) // 2]
    ref      = min(results, key=lambda r: abs(r[color_key] - median_c))

    scale = n_nodes if normalise else 1

    for key, label, ax in compartments:
        for res in results:
            series = np.array(res[key], dtype=float) / scale
            days   = np.arange(len(series))
            color  = cmap(norm(res[color_key]))
            ax.plot(days, series, color=color, alpha=0.45, linewidth=1.1)

        # Reference trajectory (median colour value)
        ref_s = np.array(ref[key], dtype=float) / scale
        ax.plot(np.arange(len(ref_s)), ref_s,
                color="black", linewidth=2.2, linestyle="-",
                label=f"{color_label} ≈ {ref[color_key]:.2f} (median)",
                zorder=5)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Day", fontsize=10)
        ax.set_ylabel("Fraction of population" if normalise else "Count",
                      fontsize=10)
        ax.set_ylim(-0.02, 1.05 if normalise else None)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=8)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(color_label, fontsize=11)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.show()


# ============================================================================
# Function 1 – sweep_incubation_range
# ============================================================================

def sweep_incubation_range(
    incubation_values: list[float],
    R0:                float = 2.5,
    n_nodes:           int   = 500,
    infectious_days:   float = 7.0,
    k:                 int   = 4,
    p_rewire:          float = 0.1,
    superspreading:    bool  = False,
    k_disp:            float = 0.1,
    base_seed:         int   = 42,
    plot:              bool  = False,
    out_path:          Optional[str] = None,
) -> list[dict]:
    """
    Run the SEIR network model for each value in ``incubation_values``.

    Each run uses the same network topology and R0; only the incubation period
    (and therefore β, which depends on infectious_days only, not on σ) varies.

    Parameters
    ----------
    incubation_values : list[float]
        Grid of mean incubation periods (days) to sweep over.
    R0 : float
        Basic reproduction number (default 2.5).
    n_nodes : int
        Network size (default 500).
    infectious_days : float
        Mean infectious period γ⁻¹, fixed across all runs (default 7.0).
    k : int
        Watts-Strogatz ring-lattice degree (default 4).
    p_rewire : float
        Edge rewiring probability (default 0.1).
    superspreading : bool
        Enable individual-level β heterogeneity (default False).
    k_disp : float
        Gamma dispersion for superspreading (default 0.1).
    base_seed : int
        Seed for run i = base_seed + i (default 42).
    plot : bool
        If True, display a four-panel trajectory plot (default False).
    out_path : str or None
        If given, save the plot to this path.

    Returns
    -------
    list[dict]
        One dict per incubation value, with keys:
          'incubation_days' float  – the parameter used
          'S', 'E', 'I', 'R'      – daily compartment series (counts)
          'peak_I'          int    – maximum daily I
          'peak_day'        int    – day of peak I
          'attack_rate'     float  – fraction ever infected
    """
    results = []

    for idx, incub in enumerate(incubation_values):
        S, E, I, R, _ = run_seir(
            n_nodes         = n_nodes,
            R0              = R0,
            incubation_days = incub,
            infectious_days = infectious_days,
            k               = k,
            p_rewire        = p_rewire,
            superspreading  = superspreading,
            k_disp          = k_disp,
            seed            = base_seed + idx,
        )
        peak_I   = max(I)
        peak_day = I.index(peak_I)
        attack   = (n_nodes - S[-1]) / n_nodes

        results.append({
            "incubation_days": incub,
            "S": S, "E": E, "I": I, "R": R,
            "peak_I": peak_I, "peak_day": peak_day,
            "attack_rate": attack,
        })

    if plot or out_path:
        plot_results(
            results,
            color_key="incubation_days",
            color_label="Incubation period σ⁻¹ (days)",
            n_nodes=n_nodes,
            out_path=out_path,
        )

    return results


# ============================================================================
# Function 2 – sample_incubation_gamma_network
# ============================================================================

def sample_incubation_gamma_network(
    gamma_shape:     float,
    gamma_scale:     float,
    n_runs:          int   = 30,
    R0:              float = 2.5,
    n_nodes:         int   = 500,
    infectious_days: float = 7.0,
    k:               int   = 4,
    p_rewire:        float = 0.1,
    superspreading:  bool  = False,
    k_disp:          float = 0.1,
    base_seed:       int   = 42,
    plot:            bool  = False,
    out_path:        Optional[str] = None,
) -> list[dict]:
    """
    Run the SEIR network model ``n_runs`` times, drawing the incubation period
    afresh from Gamma(shape, scale) at each run.

    The incubation period is treated as an uncertain parameter: each run
    represents one possible realisation of the true σ⁻¹.  The ensemble
    captures full parametric uncertainty, unlike Function 1 which only
    evaluates a fixed grid.

    The Gamma distribution is parameterised as:
        σ⁻¹ ~ Gamma(shape=gamma_shape, scale=gamma_scale)
        E[σ⁻¹] = gamma_shape × gamma_scale
        Var[σ⁻¹] = gamma_shape × gamma_scale²

    Parameters
    ----------
    gamma_shape : float
        Shape parameter α of the Gamma distribution.
    gamma_scale : float
        Scale parameter θ of the Gamma distribution (mean = α·θ).
    n_runs : int
        Number of Monte Carlo runs (default 30).
    R0 : float
        Basic reproduction number (default 2.5).
    n_nodes : int
        Network size (default 500).
    infectious_days : float
        Mean infectious period γ⁻¹, fixed (default 7.0).
    k, p_rewire, superspreading, k_disp
        Network and superspreading parameters (same as run_seir).
    base_seed : int
        Master seed; run i uses base_seed + i for both the Gamma draw
        and the network simulation (default 42).
    plot : bool
        Display trajectory plot (default False).
    out_path : str or None
        Save plot to this path if given.

    Returns
    -------
    list[dict]
        One dict per run, with keys:
          'incubation_days' float  – the σ⁻¹ drawn this run
          'S', 'E', 'I', 'R'      – daily compartment series (counts)
          'peak_I', 'peak_day', 'attack_rate'
    """
    rng = random.Random(base_seed)

    results = []
    for run in range(n_runs):
        # Draw incubation period from Gamma; ensure it is positive
        incub = rng.gammavariate(gamma_shape, gamma_scale)
        incub = max(incub, 0.5)          # safety floor: at least half a day

        S, E, I, R, _ = run_seir(
            n_nodes         = n_nodes,
            R0              = R0,
            incubation_days = incub,
            infectious_days = infectious_days,
            k               = k,
            p_rewire        = p_rewire,
            superspreading  = superspreading,
            k_disp          = k_disp,
            seed            = base_seed + run,
        )
        peak_I   = max(I)
        peak_day = I.index(peak_I)
        attack   = (n_nodes - S[-1]) / n_nodes

        results.append({
            "incubation_days": incub,
            "S": S, "E": E, "I": I, "R": R,
            "peak_I": peak_I, "peak_day": peak_day,
            "attack_rate": attack,
        })

    if plot or out_path:
        plot_results(
            results,
            color_key="incubation_days",
            color_label="Sampled incubation period σ⁻¹ (days)",
            n_nodes=n_nodes,
            out_path=out_path,
        )

    return results


# ============================================================================
# Function 3 – seir_delay_gamma_ode_sensitivity (sensitivity analysis)
# ============================================================================

def _build_gamma_chain_rhs(
    n_stages:        int,
    stage_rate:      float,     # rate through each sub-stage = n_stages / mean_incub
    beta:            float,
    gamma:           float,
    N:               float,
) -> callable:
    """
    Return the right-hand side of the linear-chain SEIR ODE.

    The E compartment is split into ``n_stages`` sequential sub-stages
    E_1 → E_2 → … → E_n, each with rate ``stage_rate``.  This gives an
    E→I delay that follows Gamma(n_stages, 1/stage_rate) exactly.

    State vector layout:
        y[0]           = S
        y[1..n_stages] = E_1, …, E_n
        y[n_stages+1]  = I
        y[n_stages+2]  = R
    """
    def rhs(t, y):
        S  = y[0]
        E  = y[1 : n_stages + 1]
        I  = y[n_stages + 1]
        R  = y[n_stages + 2]

        dS   = -beta * S * I / N
        dE   = np.zeros(n_stages)
        dE[0] = beta * S * I / N - stage_rate * E[0]
        for j in range(1, n_stages):
            dE[j] = stage_rate * E[j - 1] - stage_rate * E[j]
        dI   = stage_rate * E[-1] - gamma * I
        dR   = gamma * I

        return np.concatenate([[dS], dE, [dI], [dR]])

    return rhs


def seir_delay_gamma_ode_sensitivity(
    gamma_shape:     float,
    gamma_scale:     float,
    n_runs:          int   = 30,
    R0:              float = 2.5,
    n_nodes:         int   = 500,
    infectious_days: float = 7.0,
    k_network:       int   = 4,          # used only to compute β via mean degree
    p_rewire:        float = 0.1,
    t_max:           int   = 300,
    dt:              float = 0.5,
    n_stages:        Optional[int] = None,
    base_seed:       int   = 42,
    plot:            bool  = False,
    out_path:        Optional[str] = None,
) -> list[dict]:
    """
    Simulate a mean-field SEIR ODE with a Gamma-distributed incubation delay.

    The E→I transition is modelled as a gamma-distributed delay using the
    **linear chain trick**: the E compartment is decomposed into ``n_stages``
    sequential exponential sub-stages, each with rate n_stages/σ⁻¹.  When
    n_stages sub-stages are chained, the total sojourn time in E follows
    Gamma(n_stages, σ⁻¹/n_stages), matching the target distribution.

    At each run the incubation period σ⁻¹ is drawn from
    Gamma(gamma_shape, gamma_scale), so the ensemble captures uncertainty in
    both the value of σ⁻¹ and the resulting epidemic trajectory.

    The transmission rate β is derived from R0 using the mean degree of a
    Watts-Strogatz network with parameters (k_network, p_rewire), consistent
    with the network simulations in Functions 1 and 2:
        β = R0 / (⟨k⟩ · infectious_days)

    Parameters
    ----------
    gamma_shape : float
        Shape α of the Gamma prior on the incubation period.
    gamma_scale : float
        Scale θ of the Gamma prior (mean incubation = α·θ days).
    n_runs : int
        Number of Monte Carlo ODE integrations (default 30).
    R0 : float
        Basic reproduction number (default 2.5).
    n_nodes : int
        Population size N (default 500).
    infectious_days : float
        Mean infectious period γ⁻¹ (default 7.0).
    k_network : int
        Watts-Strogatz degree used to compute β (default 4).
    p_rewire : float
        Rewiring probability used to compute mean degree (default 0.1).
    t_max : int
        Length of integration window in days (default 300).
    dt : float
        Output time step in days (default 0.5).
    n_stages : int or None
        Number of linear-chain sub-stages.  If None, set to
        round(gamma_shape) so the sub-stage Gamma matches the prior mean
        and variance as closely as possible.
    base_seed : int
        Master seed; run i uses base_seed + i for the Gamma draw.
    plot : bool
        Display trajectory plot (default False).
    out_path : str or None
        Save plot to this path if given.

    Returns
    -------
    list[dict]
        One dict per run, with keys:
          'incubation_days' float       – the σ⁻¹ drawn this run
          'n_stages'        int         – sub-stages used
          't'               np.ndarray  – time points
          'S', 'E', 'I', 'R' np.ndarray – compartment trajectories (counts)
          'peak_I'          float       – maximum I
          'peak_day'        float       – day of peak I
          'attack_rate'     float       – fraction ever infected
    """
    rng = random.Random(base_seed)

    # Mean degree of a Watts-Strogatz graph ≈ k (rewiring preserves degree)
    mean_degree = k_network
    gamma_rate  = 1.0 / infectious_days

    # n_stages: default to round(gamma_shape) so chain Gamma ≈ prior Gamma
    n_st = n_stages if n_stages is not None else max(1, round(gamma_shape))

    # In the mean-field ODE the force of infection is β·I/N (per-capita contact
    # rate times transmission probability).  The network formula gives the
    # per-edge β; scaling by mean_degree recovers the per-capita rate so that
    # R0 = β_ode / γ, matching the standard mean-field relationship.
    beta = R0 * gamma_rate   # per-capita rate: R0 = β / γ

    t_eval = np.arange(0, t_max + dt, dt)

    results = []
    for run in range(n_runs):
        incub = rng.gammavariate(gamma_shape, gamma_scale)
        incub = max(incub, 0.5)

        stage_rate = n_st / incub   # each sub-stage drains at this rate

        rhs = _build_gamma_chain_rhs(
            n_stages   = n_st,
            stage_rate = stage_rate,
            beta       = beta,
            gamma      = gamma_rate,
            N          = float(n_nodes),
        )

        # Initial conditions: one exposed individual, rest susceptible
        y0           = np.zeros(n_st + 3)
        y0[0]        = n_nodes - 1    # S
        y0[1]        = 1.0            # E_1 (first sub-stage)
        # y0[2..n_st] = 0             # other E sub-stages
        # y0[n_st+1]  = 0             # I
        # y0[n_st+2]  = 0             # R

        sol = solve_ivp(
            rhs,
            t_span = (0, t_max),
            y0     = y0,
            t_eval = t_eval,
            method = "RK45",
            rtol   = 1e-6,
            atol   = 1e-8,
        )

        S_arr = sol.y[0]
        E_arr = sol.y[1 : n_st + 1].sum(axis=0)   # sum all E sub-stages
        I_arr = sol.y[n_st + 1]
        R_arr = sol.y[n_st + 2]
        t_arr = sol.t

        peak_I   = float(I_arr.max())
        peak_day = float(t_arr[I_arr.argmax()])
        attack   = float(n_nodes - S_arr[-1]) / n_nodes

        results.append({
            "incubation_days": incub,
            "n_stages":        n_st,
            "t":               t_arr,
            "S":               S_arr,
            "E":               E_arr,
            "I":               I_arr,
            "R":               R_arr,
            "peak_I":          peak_I,
            "peak_day":        peak_day,
            "attack_rate":     attack,
        })

    if plot or out_path:
        _plot_ode_results_sensitivity(
            results,
            n_nodes=n_nodes,
            out_path=out_path,
        )

    return results

# ============================================================================
# Function 4 – seir_delay_gamma_ode_unc  (uncertainty analysis)
# ============================================================================
 
def _build_heterogeneous_chain_rhs(
    stage_rates: list[float],
    beta:        float,
    gamma:       float,
    N:           float,
) -> callable:
    """
    Return the RHS of a linear-chain SEIR ODE where each sub-stage has its
    own independently sampled draining rate.
 
    State vector layout:
        y[0]              = S
        y[1 .. n_stages]  = E_1, E_2, …, E_n   (one per stage)
        y[n_stages + 1]   = I
        y[n_stages + 2]   = R
 
    The sojourn time in stage j is Exp(stage_rates[j]), so the total sojourn
    across all stages is the sum of independent (generally non-identical)
    exponentials — a hypo-exponential distribution.  When all rates are
    identical and drawn from Gamma(α, θ), the ensemble of total sojourn
    times approximates a Gamma mixture, capturing genuine within-run
    uncertainty in the delay distribution.
    """
    n  = len(stage_rates)
    rates = np.array(stage_rates, dtype=float)
 
    def rhs(t, y):
        S  = y[0]
        E  = y[1: n + 1]
        I  = y[n + 1]
        R  = y[n + 2]
 
        new_exposures = beta * S * I / N
 
        dS        = -new_exposures
        dE        = np.empty(n)
        dE[0]     = new_exposures - rates[0] * E[0]
        for j in range(1, n):
            dE[j] = rates[j - 1] * E[j - 1] - rates[j] * E[j]
        dI = rates[-1] * E[-1] - gamma * I
        dR = gamma * I
 
        return np.concatenate([[dS], dE, [dI], [dR]])
 
    return rhs
 
 
def seir_delay_gamma_ode_unc(
    gamma_shape:     float,
    gamma_scale:     float,
    n_runs:          int   = 30,
    n_stages:        int   = 5,
    R0:              float = 2.5,
    n_nodes:         int   = 500,
    infectious_days: float = 7.0,
    t_max:           int   = 300,
    dt:              float = 0.5,
    base_seed:       int   = 42,
    plot:            bool  = False,
    out_path:        Optional[str] = None,
) -> list[dict]:
    """
    Uncertainty analysis: SEIR ODE where the total incubation period is drawn
    from a Gamma distribution and then subdivided equally across sub-stages.
 
    The user specifies a Gamma distribution for the **total** incubation period
    σ⁻¹ (the full E→I sojourn time).  At each run one total period is drawn:
 
        σ⁻¹^(r)  ~  Gamma(α, θ)
 
    This total is split equally across ``n_stages`` sequential sub-stages:
 
        τ_j^(r)  =  σ⁻¹^(r) / n_stages,   j = 1, …, n_stages
 
    Each sub-stage drains at rate:
 
        λ_j^(r)  =  1 / τ_j^(r)  =  n_stages / σ⁻¹^(r)
 
    So all stages have the same rate within a run (an Erlang chain), but the
    shared rate varies between runs because σ⁻¹ is uncertain.  The effective
    incubation reported for each run equals the drawn σ⁻¹ exactly.
 
    This is the correct interpretation: the Gamma prior describes the total
    latent period, not the duration of a single sub-stage.  Drawing n_stages
    independent periods and summing them would inflate the effective incubation
    by a factor of n_stages, which is the bug this version corrects.
 
    Mathematical formulation
    ------------------------
    At run r:
 
        σ⁻¹^(r) ~ Gamma(α, θ)
        λ^(r)   = n_stages / σ⁻¹^(r)       (shared rate for all stages)
 
        dS/dt    = −β S I / N
        dE_j/dt  =  λ E_{j-1}  −  λ E_j    (j = 1, …, n_stages; E_0 ≡ new exposures)
        dI/dt    =  λ E_{n}  −  γ I
        dR/dt    =  γ I
 
    The total sojourn in E follows Erlang(n_stages, λ^(r)), whose mean is
    n_stages / λ^(r) = σ⁻¹^(r) — matching the drawn value exactly.
 
    Prior moments
    -------------
        E[σ⁻¹]   = α · θ
        Var[σ⁻¹] = α · θ²
        Std[σ⁻¹] = √α · θ
 
    Parameters
    ----------
    gamma_shape : float
        Shape α of the Gamma prior on the total incubation period.
    gamma_scale : float
        Scale θ (E[σ⁻¹] = α·θ days).
    n_runs : int
        Number of Monte Carlo ODE integrations (default 30).
    n_stages : int
        Number of E sub-stages (Erlang shape; default 5).
        More stages → narrower within-run delay distribution.
    R0 : float
        Basic reproduction number; β = R0 · γ (default 2.5).
    n_nodes : int
        Population size N (default 500).
    infectious_days : float
        Mean infectious period γ⁻¹ (default 7.0).
    t_max : int
        ODE integration horizon in days (default 300).
    dt : float
        Output time step (default 0.5 days).
    base_seed : int
        Master seed (default 42).
    plot : bool
        Display a four-panel S/E/I/R plot (default False).
    out_path : str or None
        Save the plot to this path if given.
 
    Returns
    -------
    list[dict]
        One dict per run with keys:
          'total_incub'   float        – the σ⁻¹ drawn this run (days)
          'stage_period'  float        – per-sub-stage period = σ⁻¹ / n_stages
          'stage_rate'    float        – shared draining rate = n_stages / σ⁻¹
          'eff_incub'     float        – equals total_incub (sanity check)
          't'             np.ndarray   – time array
          'S','E','I','R' np.ndarray   – compartment trajectories (counts)
          'E_stages'      np.ndarray   – shape (n_stages, T) individual sub-stages
          'peak_I'        float        – maximum infectious count
          'peak_day'      float        – day of peak
          'attack_rate'   float        – fraction ever infected
    """
    if n_stages < 1:
        raise ValueError("n_stages must be ≥ 1.")
 
    rng        = random.Random(base_seed)
    gamma_rate = 1.0 / infectious_days
    beta       = R0 * gamma_rate          # mean-field: R0 = β / γ
    t_eval     = np.arange(0, t_max + dt, dt)
 
    mean_incub_prior = gamma_shape * gamma_scale
    std_incub_prior  = math.sqrt(gamma_shape) * gamma_scale
 
    results = []
 
    for run in range(n_runs):
        # 1. Draw the TOTAL incubation period from the user's Gamma prior.
        total_incub = max(rng.gammavariate(gamma_shape, gamma_scale), 0.1)
 
        # 2. Subdivide equally: each of the n_stages sub-stages has the same
        #    period τ = total_incub / n_stages and rate λ = n_stages / total_incub.
        #    This keeps eff_incub = n_stages × τ = total_incub exactly.
        stage_period = total_incub / n_stages
        stage_rate   = n_stages / total_incub        # = 1 / stage_period
 
        # All stages share the same rate in this run (Erlang chain).
        stage_rates  = [stage_rate] * n_stages
 
        rhs = _build_heterogeneous_chain_rhs(
            stage_rates = stage_rates,
            beta        = beta,
            gamma       = gamma_rate,
            N           = float(n_nodes),
        )
 
        y0    = np.zeros(n_stages + 3)
        y0[0] = n_nodes - 1    # S
        y0[1] = 1.0            # E_1
 
        sol = solve_ivp(
            rhs,
            t_span = (0, t_max),
            y0     = y0,
            t_eval = t_eval,
            method = "RK45",
            rtol   = 1e-6,
            atol   = 1e-8,
        )
 
        S_arr    = sol.y[0]
        E_stages = sol.y[1: n_stages + 1]
        E_arr    = E_stages.sum(axis=0)
        I_arr    = sol.y[n_stages + 1]
        R_arr    = sol.y[n_stages + 2]
        t_arr    = sol.t
 
        peak_I   = float(I_arr.max())
        peak_day = float(t_arr[I_arr.argmax()])
        attack   = float(n_nodes - S_arr[-1]) / n_nodes
 
        results.append({
            "total_incub":  total_incub,
            "stage_period": stage_period,
            "stage_rate":   stage_rate,
            "eff_incub":    total_incub,      # = n_stages × stage_period
            "t":            t_arr,
            "S":            S_arr,
            "E":            E_arr,
            "E_stages":     E_stages,
            "I":            I_arr,
            "R":            R_arr,
            "peak_I":       peak_I,
            "peak_day":     peak_day,
            "attack_rate":  attack,
        })

 
    if plot or out_path:
        _plot_ode_results_unc(
            results,
            n_nodes  = n_nodes,
            out_path = out_path,
        )
 
    return results

####
# PLOTS
####

def _plot_ode_results_sensitivity(
    results:  list[dict],
    n_nodes:  int,
    out_path: Optional[str] = None,
) -> None:
    """
    Four-panel plot for the ODE results (Function 3).
    Uses continuous time arrays (not integer day indices).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    compartments = [
        ("S", "Susceptible",  axes[0, 0]),
        ("E", "Exposed (all sub-stages)", axes[0, 1]),
        ("I", "Infectious",   axes[1, 0]),
        ("R", "Recovered",    axes[1, 1]),
    ]

    cvals = [r["incubation_days"] for r in results]
    cmin, cmax = min(cvals), max(cvals)
    norm  = mcolors.Normalize(vmin=cmin, vmax=cmax)
    cmap  = plt.cm.plasma

    median_c = sorted(cvals)[len(cvals) // 2]
    ref      = min(results, key=lambda r: abs(r["incubation_days"] - median_c))

    for key, label, ax in compartments:
        for res in results:
            series = np.array(res[key]) / n_nodes
            color  = cmap(norm(res["incubation_days"]))
            ax.plot(res["t"], series, color=color, alpha=0.45, linewidth=1.1)

        ref_s = np.array(ref[key]) / n_nodes
        ax.plot(ref["t"], ref_s,
                color="black", linewidth=2.2, linestyle="--",
                label=f"σ⁻¹ ≈ {ref['incubation_days']:.2f} d (median)",
                zorder=5)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Day", fontsize=10)
        ax.set_ylabel("Fraction of population", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=8)

    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Sampled incubation period σ⁻¹ (days)", fontsize=11)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")

    plt.show()


def _plot_ode_results_unc(
    results:  list[dict],
    n_nodes:  int,
    out_path: Optional[str] = None,
) -> None:
    """
    Four-panel plot for the ODE results (Function 3).
    Uses continuous time arrays (not integer day indices).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
 
    compartments = [
        ("S", "Susceptible",  axes[0, 0]),
        ("E", "Exposed (all sub-stages)", axes[0, 1]),
        ("I", "Infectious",   axes[1, 0]),
        ("R", "Recovered",    axes[1, 1]),
    ]
 
    cvals = [r["eff_incub"] for r in results]
    cmin, cmax = min(cvals), max(cvals)
    norm  = mcolors.Normalize(vmin=cmin, vmax=cmax)
    cmap  = plt.cm.plasma
 
    median_c = sorted(cvals)[len(cvals) // 2]
    ref      = min(results, key=lambda r: abs(r["eff_incub"] - median_c))
 
    for key, label, ax in compartments:
        for res in results:
            series = np.array(res[key]) / n_nodes
            color  = cmap(norm(res["eff_incub"]))
            ax.plot(res["t"], series, color=color, alpha=0.45, linewidth=1.1)
 
        ref_s = np.array(ref[key]) / n_nodes
        ax.plot(ref["t"], ref_s,
                color="black", linewidth=2.2, linestyle="--",
                label=f"eff. σ⁻¹ ≈ {ref['eff_incub']:.2f} d (median)",
                zorder=5)
 
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Day", fontsize=10)
        ax.set_ylabel("Fraction of population", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=8)
 
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Effective incubation period n·E[λ⁻¹] (days)", fontsize=11)
 
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
 
    plt.show()

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEIR incubation-period uncertainty – three analysis functions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--func",         choices=["1", "2", "3", "4", "all"],
                        default="all",
                        help="Which function(s) to run")
    # Shared model parameters
    parser.add_argument("--R0",           type=float, default=2.5)
    parser.add_argument("--nodes",        type=int,   default=500)
    parser.add_argument("--infectious",   type=float, default=7.0,
                        help="Mean infectious period (days)")
    parser.add_argument("--k",            type=int,   default=4,
                        help="Watts-Strogatz degree")
    parser.add_argument("--p",            type=float, default=0.1,
                        help="Rewiring probability")
    parser.add_argument("--seed",         type=int,   default=42)
    # Function 1 parameters
    parser.add_argument("--incub-values", type=str,
                        default="1 2 3 5 7 10 14",
                        help="[Func 1] Space-separated incubation values (days)")
    # Functions 2 & 3 parameters
    parser.add_argument("--gamma-shape",  type=float, default=5.0,
                        dest="gamma_shape",
                        help="[Func 2/3] Shape α of Gamma prior on σ⁻¹")
    parser.add_argument("--gamma-scale",  type=float, default=1.0,
                        dest="gamma_scale",
                        help="[Func 2/3] Scale θ of Gamma prior on σ⁻¹")
    parser.add_argument("--n-runs",       type=int,   default=20,
                        dest="n_runs",
                        help="[Func 2/3] Number of Monte Carlo runs")
    # Function 3 parameters
    parser.add_argument("--t-max",        type=int,   default=300,
                        dest="t_max",
                        help="[Func 3] ODE integration horizon (days)")
    parser.add_argument("--n-stages",     type=int,   default=None,
                        dest="n_stages",
                        help="[Func 3] Linear-chain sub-stages (default=round(gamma_shape))")
    # Output
    parser.add_argument("--plot",         action="store_true",
                        help="Display plots")
    parser.add_argument("--out-prefix",   type=str,   default=None,
                        dest="out_prefix",
                        help="Prefix for saved figure paths, e.g. 'figs/run1'")
    args = parser.parse_args()

    incub_values = [float(x) for x in args.incub_values.split()]

    def _out(suffix):
        return f"{args.out_prefix}_{suffix}.png" if args.out_prefix else None

    run_all = args.func == "all"

    if run_all or args.func == "1":
        sweep_incubation_range(
            incubation_values = incub_values,
            R0                = args.R0,
            n_nodes           = args.nodes,
            infectious_days   = args.infectious,
            k                 = args.k,
            p_rewire          = args.p,
            base_seed         = args.seed,
            plot              = args.plot,
            out_path          = _out("func1"),
        )

    if run_all or args.func == "2":
        sample_incubation_gamma_network(
            gamma_shape     = args.gamma_shape,
            gamma_scale     = args.gamma_scale,
            n_runs          = args.n_runs,
            R0              = args.R0,
            n_nodes         = args.nodes,
            infectious_days = args.infectious,
            k               = args.k,
            p_rewire        = args.p,
            base_seed       = args.seed,
            plot            = args.plot,
            out_path        = _out("func2"),
        )

    if run_all or args.func == "3":
        seir_delay_gamma_ode_sensitivity(
            gamma_shape     = args.gamma_shape,
            gamma_scale     = args.gamma_scale,
            n_runs          = args.n_runs,
            R0              = args.R0,
            n_nodes         = args.nodes,
            infectious_days = args.infectious,
            k_network       = args.k,
            p_rewire        = args.p,
            t_max           = args.t_max,
            n_stages        = args.n_stages,
            base_seed       = args.seed,
            plot            = args.plot,
            out_path        = _out("func3"),
        )

    if run_all or args.func == "4":
        seir_delay_gamma_ode_unc(
            gamma_shape     = args.gamma_shape,
            gamma_scale     = args.gamma_scale,
            n_runs          = args.n_runs,
            n_stages        = args.n_stages if args.n_stages else 5,
            R0              = args.R0,
            n_nodes         = args.nodes,
            infectious_days = args.infectious,
            t_max           = args.t_max,
            base_seed       = args.seed,
            plot            = args.plot,
            out_path        = _out("func4"),
        )
