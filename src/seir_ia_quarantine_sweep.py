"""
seir_ia_quarantine_sweep.py
===========================
Compute and plot the average attack fraction of the SEIR + asymptomatic (IA)
+ quarantine (Q) network model over two 2-D parameter grids:

  Grid 1:  eta  × theta   (quarantine rate vs asymptomatic fraction)
  Grid 2:  eta  × T_s     (quarantine rate vs IA→I transition period)

Model structure  (per the agreed description)
---------------------------------------------
Compartments:  S, E, I, IA, Q, R

  S  → E    : transmission from I or IA neighbours at rates β_I, β_IA
  E  → I    : fraction θ,   rate σ   = 1 / incubation_days
  E  → IA   : fraction (1−θ), rate a·σ  = a / incubation_days
  IA → I    : rate  1 / T_s          (T_s = incubation_days − incubation_days/a)
  IA → R    : rate  γ   = 1 / infectious_days   (recovers without symptoms)
  I  → Q    : rate  η   (quarantine of symptomatic; ONLY I nodes are quarantined)
  I  → R    : rate  γ   (if not quarantined this step)
  Q  → R    : rate  γ
  IA nodes do NOT get quarantined; they continue to infect freely.

Attack fraction definition
--------------------------
  AF = (N − S(T_end)) / N

i.e. the fraction of the population that was ever infected (passed through E).

Public API
----------
    from seir_ia_quarantine_sweep import (
        run_seir_ia_q,
        sweep_eta_theta,
        sweep_eta_ts,
        plot_heatmap,
    )

CLI
---
    python seir_ia_quarantine_sweep.py                       # defaults
    python seir_ia_quarantine_sweep.py \\
        --eta-values  "0.0 0.05 0.10 0.20 0.30 0.50" \\
        --theta-values "0.1 0.3 0.5 0.7 0.9" \\
        --ts-values    "1 2 3 4 5 6" \\
        --R0 2.5 --nodes 500 --reps 10 --plot
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
    from seir_network_model import watts_strogatz, draw_infectiousness
except ImportError:
    sys.exit(
        "Cannot find seir_network_model.py. "
        "Place it in the same directory as this script."
    )

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError as exc:
    sys.exit(f"matplotlib and numpy are required: {exc}")


# ---------------------------------------------------------------------------
# State codes  (plain integers – no enum overhead in tight loop)
# ---------------------------------------------------------------------------
_S  = 0   # Susceptible
_E  = 1   # Exposed
_I  = 2   # Infectious symptomatic
_IA = 3   # Infectious asymptomatic
_Q  = 4   # Quarantined (from I only)
_R  = 5   # Recovered


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def run_seir_ia_q(
    n_nodes:          int,
    R0:               float,
    incubation_days:  float = 5.0,
    infectious_days:  float = 7.0,
    theta:            float = 0.5,
    a:                float = 2.0,
    eta:              float = 0.0,
    beta_ratio:       float = 1.0,
    t_intervention:   int   = 0,
    superspreading:   bool  = False,
    k_disp:           float = 0.1,
    k:                int   = 4,
    p_rewire:         float = 0.1,
    seed:             int   = 42,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    """
    Discrete-time SEIR + asymptomatic (IA) + quarantine (Q) on a
    Watts-Strogatz small-world network.

    Compartment flow
    ----------------
    S  → E    per infectious neighbour (I or IA), rates β_I and β_IA
    E  → I    daily probability  θ · σ          (σ = 1/incubation_days)
    E  → IA   daily probability  (1−θ) · a · σ  (faster incubation)
    IA → I    daily probability  1/T_s           (T_s = inc*(1−1/a))
    IA → R    daily probability  γ               (asymptomatic recovery)
    I  → Q    daily probability  η               (quarantine; only I nodes)
    I  → R    daily probability  (1−η) · γ
    Q  → R    daily probability  γ
    IA nodes are NEVER quarantined.

    Competing hazards for E
    -----------------------
    On each day an E node first draws for E→IA  (prob (1−θ)·a·σ),
    then, if that fails, draws for E→I (prob θ·σ).
    The complementary structure ensures that neither branch is drawn
    from a probability > 1 for any valid (θ, a, σ).

    Parameters
    ----------
    n_nodes         : int    network size
    R0              : float  basic reproduction number (pre-intervention)
    incubation_days : float  mean incubation period T_E (days)
    infectious_days : float  mean infectious period T_I (days)
    theta           : float  fraction E→I (symptomatic); (1−θ) go E→IA
    a               : float  acceleration factor for asymptomatic incubation
                             E→IA period = incubation_days / a  (a > 1 → faster)
    eta             : float  daily quarantine rate for I nodes  (0 = no quarantine)
    beta_ratio      : float  β_IA / β_I  (asymptomatic relative infectiousness)
                             default 1.0 means IA is as infectious as I
    t_intervention  : int    day quarantine begins (default 0 = from day 1)
    superspreading  : bool   Gamma-modulated individual β
    k_disp          : float  dispersion for superspreading
    k               : int    Watts-Strogatz ring-lattice degree
    p_rewire        : float  edge rewiring probability
    seed            : int    random seed

    Returns
    -------
    (S, E, I, IA, Q, R)  – six daily-count lists; sum = n_nodes at all times.
    """
    rng         = random.Random(seed)
    adj         = watts_strogatz(n_nodes, k=k, p=p_rewire, seed=seed)
    mean_degree = sum(len(v) for v in adj.values()) / n_nodes

    # β_I derived from R0.  β_IA scaled by beta_ratio.
    # R0 = (θ·β_I + (1−θ)·β_IA) · <k> · T_I  in the mixed model,
    # so β_I = R0 / (<k>·T_I·(θ + (1−θ)·beta_ratio))
    effective_beta_factor = theta + (1.0 - theta) * beta_ratio
    beta_I  = R0 / (mean_degree * infectious_days * effective_beta_factor)
    beta_IA = beta_ratio * beta_I

    sigma   = 1.0 / incubation_days          # E→I rate (symptomatic branch)
    sigma_a = a / incubation_days            # E→IA rate (asymptomatic branch, faster)
    gamma   = 1.0 / infectious_days          # I/IA/Q → R rate

    # T_s: period IA→I.  incubation_days/a + T_s = incubation_days
    # → T_s = incubation_days * (1 - 1/a).  Guard against a=1 (T_s=0 → instant).
    T_s = incubation_days * (1.0 - 1.0 / a) if a > 1.0 else 0.0
    alpha1 = (1.0 / T_s) if T_s > 0.0 else float("inf")

    # Initial state: one exposed patient zero
    state                     = [_S] * n_nodes
    state[rng.randint(0, n_nodes - 1)] = _E

    S_list:  list[int] = []
    E_list:  list[int] = []
    I_list:  list[int] = []
    IA_list: list[int] = []
    Q_list:  list[int] = []
    R_list:  list[int] = []

    t = 0
    while True:
        S  = state.count(_S);  E  = state.count(_E)
        I  = state.count(_I);  IA = state.count(_IA)
        Q  = state.count(_Q);  R  = state.count(_R)

        S_list.append(S);  E_list.append(E)
        I_list.append(I);  IA_list.append(IA)
        Q_list.append(Q);  R_list.append(R)

        # Stop when no latent or infectious nodes remain
        if E == 0 and I == 0 and IA == 0:
            break

        intervening = (t >= t_intervention)
        new_state   = state[:]

        for node in range(n_nodes):

            if state[node] == _S:
                # Attempt infection by each I or IA neighbour independently
                for nb in adj[node]:
                    nb_state = state[nb]
                    if nb_state == _I:
                        b = draw_infectiousness(beta_I, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = _E
                            break
                    elif nb_state == _IA:
                        b = draw_infectiousness(beta_IA, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = _E
                            break

            elif state[node] == _E:
                # Competing hazards: E→IA checked first, then E→I
                # E→IA: fraction (1-θ) with rate sigma_a
                if rng.random() < (1.0 - theta) * sigma_a:
                    new_state[node] = _IA
                elif rng.random() < theta * sigma:
                    new_state[node] = _I

            elif state[node] == _I:
                # I transmits to susceptible neighbours
                for nb in adj[node]:
                    if new_state[nb] == _S:
                        b = draw_infectiousness(beta_I, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[nb] = _E
                # I→Q or I→R (quarantine only if intervention active)
                if intervening and eta > 0.0 and rng.random() < eta:
                    new_state[node] = _Q
                elif rng.random() < gamma:
                    new_state[node] = _R

            elif state[node] == _IA:
                # IA transmits (not quarantinable)
                for nb in adj[node]:
                    if new_state[nb] == _S:
                        b = draw_infectiousness(beta_IA, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[nb] = _E
                # IA→I (become symptomatic) or IA→R (recover asymptomatically)
                # Competing: IA→I at rate alpha1, IA→R at rate gamma
                if alpha1 == float("inf"):
                    # a=1 edge case: instant symptom onset
                    new_state[node] = _I
                elif rng.random() < min(1.0 / (T_s + 1.0 / gamma) / (1.0 / T_s), 1.0):
                    # Sequential Bernoulli: IA→I first
                    if rng.random() < (1.0 / T_s) / (1.0 / T_s + gamma):
                        new_state[node] = _I
                    else:
                        new_state[node] = _R
                else:
                    # Simplified: on each day IA either progresses to I
                    # with prob proportional to alpha1, or recovers with prob gamma
                    r_draw = rng.random()
                    p_to_I = min(1.0 / T_s, 1.0) if T_s > 0 else 1.0
                    p_to_R = gamma
                    if r_draw < p_to_I:
                        new_state[node] = _I
                    elif r_draw < p_to_I + p_to_R:
                        new_state[node] = _R

            elif state[node] == _Q:
                if rng.random() < gamma:
                    new_state[node] = _R
            # _R is absorbing

        state = new_state
        t    += 1

    return S_list, E_list, I_list, IA_list, Q_list, R_list


# ---------------------------------------------------------------------------
# Cleaner IA transition helper — replace the nested logic above
# ---------------------------------------------------------------------------

def _run_seir_ia_q_clean(
    n_nodes:         int,
    R0:              float,
    incubation_days: float,
    infectious_days: float,
    theta:           float,
    a:               float,
    T_s:             float,
    eta:             float,
    beta_ratio:      float,
    t_intervention:  int,
    superspreading:  bool,
    k_disp:          float,
    k:               int,
    p_rewire:        float,
    seed:            int,
    horizon:         int = 200,
) -> float:
    """
    Internal single-run simulator that returns the attack fraction at day
    ``horizon``.

    The simulation runs until day ``horizon`` OR until no active cases
    remain (E = I = IA = 0), whichever comes first.  The attack fraction
    is always evaluated at the earlier of the two stopping conditions:

        AF = (N − S(t_stop)) / N,   t_stop = min(horizon, T_end)

    This means:
      - If the epidemic ends before ``horizon``, AF reflects the true
        final size (no further infections possible after extinction).
      - If the epidemic is still active at ``horizon``, AF is a snapshot
        of cumulative infections up to that day — a partial attack fraction.

    Uses clean sequential Bernoulli logic for all competing hazards.
    """
    rng         = random.Random(seed)
    adj         = watts_strogatz(n_nodes, k=k, p=p_rewire, seed=seed)
    mean_degree = sum(len(v) for v in adj.values()) / n_nodes

    effective_beta_factor = theta + (1.0 - theta) * beta_ratio
    beta_I  = R0 / (mean_degree * infectious_days * effective_beta_factor)
    beta_IA = beta_ratio * beta_I

    sigma   = 1.0 / incubation_days
    sigma_a = a   / incubation_days
    gamma   = 1.0 / infectious_days

    rate_ia_to_i = (1.0 / T_s) if T_s > 1e-9 else 1e6
    p_ia_to_i    = min(rate_ia_to_i, 1.0)
    p_ia_to_r    = min(gamma, 1.0)

    state                     = [_S] * n_nodes
    state[rng.randint(0, n_nodes - 1)] = _E

    t = 0
    while True:
        E  = state.count(_E)
        I  = state.count(_I)
        IA = state.count(_IA)

        # Stop at natural extinction OR when the horizon is reached
        if E == 0 and I == 0 and IA == 0:
            break
        if t >= horizon:
            break

        intervening = (t >= t_intervention)
        new_state   = state[:]

        for node in range(n_nodes):
            ns = state[node]

            if ns == _S:
                for nb in adj[node]:
                    nb_s = state[nb]
                    if nb_s == _I:
                        b = draw_infectiousness(beta_I, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = _E
                            break
                    elif nb_s == _IA:
                        b = draw_infectiousness(beta_IA, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = _E
                            break

            elif ns == _E:
                if rng.random() < (1.0 - theta) * min(sigma_a, 1.0):
                    new_state[node] = _IA
                elif rng.random() < theta * min(sigma, 1.0):
                    new_state[node] = _I

            elif ns == _I:
                for nb in adj[node]:
                    if new_state[nb] == _S:
                        b = draw_infectiousness(beta_I, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[nb] = _E
                if intervening and eta > 0.0 and rng.random() < min(eta, 1.0):
                    new_state[node] = _Q
                elif rng.random() < gamma:
                    new_state[node] = _R

            elif ns == _IA:
                for nb in adj[node]:
                    if new_state[nb] == _S:
                        b = draw_infectiousness(beta_IA, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[nb] = _E
                if rng.random() < p_ia_to_i:
                    new_state[node] = _I
                elif rng.random() < p_ia_to_r:
                    new_state[node] = _R

            elif ns == _Q:
                if rng.random() < gamma:
                    new_state[node] = _R

        state = new_state
        t    += 1

    S_final = state.count(_S)
    return (n_nodes - S_final) / n_nodes


# ---------------------------------------------------------------------------
# Grid sweeps
# ---------------------------------------------------------------------------

def sweep_eta_theta(
    eta_values:      list[float],
    theta_values:    list[float],
    R0:              float = 2.5,
    n_nodes:         int   = 500,
    incubation_days: float = 5.0,
    infectious_days: float = 7.0,
    a:               float = 2.0,
    beta_ratio:      float = 1.0,
    t_intervention:  int   = 0,
    horizon:         int   = 200,
    superspreading:  bool  = False,
    k_disp:          float = 0.1,
    k:               int   = 6,
    p_rewire:        float = 0.1,
    n_replicates:    int   = 10,
    base_seed:       int   = 42,
    verbose:         bool  = True,
) -> np.ndarray:
    """
    Sweep (eta, theta) and return mean attack fraction up to ``horizon`` days.

    T_s is derived internally from a and incubation_days:
        T_s = incubation_days * (1 − 1/a)

    Parameters
    ----------
    horizon : int
        Maximum number of simulation days.  The attack fraction is
        evaluated at min(horizon, T_extinction).  Default 200.

    Returns
    -------
    np.ndarray of shape (len(eta_values), len(theta_values))
    Row i → eta_values[i],  column j → theta_values[j].
    """
    n_eta   = len(eta_values)
    n_theta = len(theta_values)
    result  = np.zeros((n_eta, n_theta))
    T_s     = incubation_days * (1.0 - 1.0 / a) if a > 1.0 else 0.0
    total   = n_eta * n_theta
    done    = 0

    for i, eta in enumerate(eta_values):
        for j, theta in enumerate(theta_values):
            af_sum = 0.0
            for rep in range(n_replicates):
                seed = base_seed + i * 10_000 + j * 100 + rep
                af   = _run_seir_ia_q_clean(
                    n_nodes, R0, incubation_days, infectious_days,
                    theta, a, T_s, eta, beta_ratio, t_intervention,
                    superspreading, k_disp, k, p_rewire, seed,
                    horizon=horizon,
                )
                af_sum += af
            result[i, j] = af_sum / n_replicates
            done += 1
            if verbose:
                pct = done / total * 100
                bar = "█" * int(pct / 2)
                print(f"\r  [{bar:<50}] {pct:5.1f}%  "
                      f"η={eta:.2f}  θ={theta:.2f}  "
                      f"AF={result[i,j]:.3f}",
                      end="", flush=True)

    if verbose:
        print()
    return result


def sweep_eta_ts(
    eta_values:      list[float],
    ts_values:       list[float],
    theta:           float = 0.5,
    R0:              float = 2.5,
    n_nodes:         int   = 500,
    incubation_days: float = 5.0,
    infectious_days: float = 7.0,
    beta_ratio:      float = 1.0,
    t_intervention:  int   = 0,
    horizon:         int   = 200,
    superspreading:  bool  = False,
    k_disp:          float = 0.1,
    k:               int   = 4,
    p_rewire:        float = 0.1,
    n_replicates:    int   = 10,
    base_seed:       int   = 42,
    verbose:         bool  = True,
) -> np.ndarray:
    """
    Sweep (eta, T_s) and return mean attack fraction up to ``horizon`` days.

    The asymptomatic acceleration factor a is derived from T_s:
        a = incubation_days / (incubation_days − T_s)
    which ensures  incubation_days/a + T_s = incubation_days.
    T_s must satisfy 0 < T_s < incubation_days.

    Parameters
    ----------
    horizon : int
        Maximum number of simulation days.  The attack fraction is
        evaluated at min(horizon, T_extinction).  Default 200.

    Returns
    -------
    np.ndarray of shape (len(eta_values), len(ts_values))
    Row i → eta_values[i],  column j → ts_values[j].
    """
    n_eta = len(eta_values)
    n_ts  = len(ts_values)
    result = np.zeros((n_eta, n_ts))
    total  = n_eta * n_ts
    done   = 0

    for i, eta in enumerate(eta_values):
        for j, T_s in enumerate(ts_values):
            if T_s <= 0 or T_s >= incubation_days:
                raise ValueError(
                    f"T_s={T_s} is out of range (0, incubation_days={incubation_days})."
                )
            a = incubation_days / (incubation_days - T_s)

            af_sum = 0.0
            for rep in range(n_replicates):
                seed = base_seed + i * 10_000 + j * 100 + rep
                af   = _run_seir_ia_q_clean(
                    n_nodes, R0, incubation_days, infectious_days,
                    theta, a, T_s, eta, beta_ratio, t_intervention,
                    superspreading, k_disp, k, p_rewire, seed,
                    horizon=horizon,
                )
                af_sum += af
            result[i, j] = af_sum / n_replicates
            done += 1
            if verbose:
                pct = done / total * 100
                bar = "█" * int(pct / 2)
                print(f"\r  [{bar:<50}] {pct:5.1f}%  "
                      f"η={eta:.2f}  T_s={T_s:.2f}  "
                      f"AF={result[i,j]:.3f}",
                      end="", flush=True)

    if verbose:
        print()
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap_asymptomatic(
    data:       np.ndarray,
    x_values:   list[float],
    y_values:   list[float],
    x_label:    str,
    y_label:    str,
    title:      str,
    out_path:   Optional[str] = None,
    annotate:   bool          = True,
) -> None:
    """
    Render an annotated heatmap of the attack fraction.

    data shape : (len(y_values), len(x_values))
    x-axis     → x_values (columns)
    y-axis     → y_values (rows), displayed bottom-to-top
    """
    # Flip rows so y increases upward
    data_plot    = data[::-1]
    y_labels_rev = y_values[::-1]

    fig, ax = plt.subplots(figsize=(11,7))

    im = ax.imshow(
        data_plot,
        cmap   = "YlOrRd",
        aspect = "auto",
        vmin   = 0.0,
        vmax   = 1.0,
        origin = "upper",
    )

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([f"{v:.2f}" for v in x_values], fontsize=10)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([f"{v:.3f}" for v in y_labels_rev], fontsize=10)
    ax.set_xlabel(x_label, fontsize=14, labelpad=8)
    ax.set_ylabel(y_label, fontsize=14, labelpad=8)
    ax.set_title(title, fontsize=12, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Attack fraction $C_{200}/N$", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
    )

    # Cell annotations
    if annotate and data_plot.size <= 120:
        thresh = 0.55
        for i in range(data_plot.shape[0]):
            for j in range(data_plot.shape[1]):
                val   = data_plot[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.show()


def plot_both_heatmaps(
    data_et:      np.ndarray,
    eta_values:   list[float],
    theta_values: list[float],
    data_ets:     np.ndarray,
    ts_values:    list[float],
    R0:           float,
    n_nodes:      int,
    incubation_days: float,
    infectious_days: float,
    n_replicates: int,
    theta_fixed:  float,
    a:            float,
    horizon:      int   = 200,
    out_path:     Optional[str] = None,
) -> None:
    """Side-by-side heatmaps for the two sweeps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def _draw(ax, data, x_vals, y_vals, xlabel, ylabel, title):
        d_plot   = data[::-1]
        y_rev    = y_vals[::-1]
        im = ax.imshow(d_plot, cmap="YlOrRd", aspect="auto",
                       vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in x_vals], fontsize=9)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in y_rev], fontsize=9)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attack fraction", fontsize=10)
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
        )
        if d_plot.size <= 120:
            thresh = 0.55
            for i in range(d_plot.shape[0]):
                for j in range(d_plot.shape[1]):
                    val = d_plot[i, j]
                    col = "white" if val > thresh else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.5, color=col, fontweight="bold")

    T_s_derived = incubation_days * (1.0 - 1.0 / a)
    _draw(ax1, data_et, theta_values, eta_values,
          xlabel="θ  (symptomatic fraction)",
          ylabel="η  (quarantine rate of I)",
          title=(f"Grid 1: η × θ  [horizon = {horizon} days]\n"
                 f"R₀={R0}, N={n_nodes}, T_E={incubation_days}d, "
                 f"T_I={infectious_days}d, a={a} (T_s={T_s_derived:.1f}d), "
                 f"{n_replicates} reps"))

    _draw(ax2, data_ets, ts_values, eta_values,
          xlabel="T_s  (IA → I period, days)",
          ylabel="η  (quarantine rate of I)",
          title=(f"Grid 2: η × T_s  [horizon = {horizon} days]\n"
                 f"R₀={R0}, N={n_nodes}, T_E={incubation_days}d, "
                 f"T_I={infectious_days}d, θ={theta_fixed}, "
                 f"{n_replicates} reps"))

    fig.suptitle(
        f"Mean attack fraction at day {horizon} – "
        f"SEIR + Asymptomatic (IA) + Quarantine (Q)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary_table(
    data:      np.ndarray,
    row_vals:  list[float],
    col_vals:  list[float],
    row_label: str,
    col_label: str,
) -> None:
    cw = 8
    hdr = f"  {row_label[:6]+'\\'+col_label[:5]:>12}  " + \
          "".join(f"{v:>{cw}.2f}" for v in col_vals)
    print(hdr)
    print("  " + "-" * len(hdr))
    for i, rv in enumerate(row_vals):
        row = f"  {rv:>12.3f}  " + \
              "".join(f"{data[i,j]:>{cw}.3f}" for j in range(len(col_vals)))
        print(row)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_ETA    = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
DEFAULT_THETA  = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DEFAULT_TS     = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Attack-fraction heatmaps for SEIR + IA + Q model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eta-values",   type=str,
                        default=" ".join(str(v) for v in DEFAULT_ETA),
                        help="Space-separated η values")
    parser.add_argument("--theta-values", type=str,
                        default=" ".join(str(v) for v in DEFAULT_THETA),
                        help="Space-separated θ values (Grid 1)")
    parser.add_argument("--ts-values",    type=str,
                        default=" ".join(str(v) for v in DEFAULT_TS),
                        help="Space-separated T_s values in days (Grid 2)")
    parser.add_argument("--theta-fixed",  type=float, default=0.5,
                        dest="theta_fixed",
                        help="Fixed θ used in Grid 2 (η × T_s)")
    parser.add_argument("--R0",           type=float, default=2.5)
    parser.add_argument("--nodes",        type=int,   default=500)
    parser.add_argument("--incubation",   type=float, default=5.0,
                        help="Mean incubation period T_E (days)")
    parser.add_argument("--infectious",   type=float, default=7.0,
                        help="Mean infectious period T_I (days)")
    parser.add_argument("--a",            type=float, default=2.0,
                        help="Asymptomatic incubation accelerator "
                             "(E→IA period = incubation/a; used in Grid 1)")
    parser.add_argument("--beta-ratio",   type=float, default=1.0,
                        dest="beta_ratio",
                        help="β_IA / β_I (relative asymptomatic infectiousness)")
    parser.add_argument("--t-int",        type=int,   default=0,
                        dest="t_intervention",
                        help="Day quarantine intervention begins")
    parser.add_argument("--k",            type=int,   default=4)
    parser.add_argument("--p",            type=float, default=0.1)
    parser.add_argument("--reps",         type=int,   default=10,
                        help="Stochastic replicates per cell")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--out",          type=str,   default=None,
                        help="Path to save the combined figure")
    parser.add_argument("--horizon",      type=int,   default=200,
                        help="Day at which the attack fraction is evaluated "
                             "(simulation also stops early at epidemic extinction)")
    parser.add_argument("--no-plot",      action="store_true", dest="no_plot",
                        help="Suppress the interactive plot window")
    parser.add_argument("--quiet",        action="store_true")
    args = parser.parse_args()

    eta_values   = [float(x) for x in args.eta_values.split()]
    theta_values = [float(x) for x in args.theta_values.split()]
    ts_values    = [float(x) for x in args.ts_values.split()]

    # Validate T_s values against incubation_days
    bad_ts = [t for t in ts_values if not (0 < t < args.incubation)]
    if bad_ts:
        sys.exit(f"T_s values {bad_ts} are outside (0, {args.incubation}). "
                 f"Each T_s must satisfy 0 < T_s < incubation_days.")

    total_runs = (len(eta_values) * len(theta_values) +
                  len(eta_values) * len(ts_values)) * args.reps
    print(f"\n  Grid 1 (η × θ):  {len(eta_values)} × {len(theta_values)} "
          f"× {args.reps} reps = {len(eta_values)*len(theta_values)*args.reps} runs")
    print(f"  Grid 2 (η × T_s): {len(eta_values)} × {len(ts_values)} "
          f"× {args.reps} reps = {len(eta_values)*len(ts_values)*args.reps} runs")
    print(f"  Total: {total_runs} simulations\n")

    common = dict(
        R0=args.R0, n_nodes=args.nodes,
        incubation_days=args.incubation, infectious_days=args.infectious,
        beta_ratio=args.beta_ratio, t_intervention=args.t_intervention,
        horizon=args.horizon,
        k=args.k, p_rewire=args.p,
        n_replicates=args.reps, base_seed=args.seed,
        verbose=not args.quiet,
    )

    print("  Running Grid 1: η × θ …")
    data_et = sweep_eta_theta(
        eta_values, theta_values, a=args.a, **common,
    )

    print("\n  Running Grid 2: η × T_s …")
    data_ets = sweep_eta_ts(
        eta_values, ts_values, theta=args.theta_fixed, **common,
    )

    print("\n  Grid 1 – Mean attack fraction (rows=η, cols=θ):")
    print_summary_table(data_et, eta_values, theta_values, "eta", "theta")

    print("  Grid 2 – Mean attack fraction (rows=η, cols=T_s):")
    print_summary_table(data_ets, eta_values, ts_values, "eta", "T_s")

    if not args.no_plot:
        plot_both_heatmaps(
            data_et, eta_values, theta_values,
            data_ets, ts_values,
            R0=args.R0, n_nodes=args.nodes,
            incubation_days=args.incubation, infectious_days=args.infectious,
            n_replicates=args.reps,
            theta_fixed=args.theta_fixed, a=args.a,
            horizon=args.horizon,
            out_path=args.out,
        )
