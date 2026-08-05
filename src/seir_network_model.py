"""
seir_network_model.py
=====================
SEIR epidemic simulation on a Watts-Strogatz small-world network
with optional superspreading and piecewise-constant R(t) schedule.

Public API
----------
    from seir_network_model import run_seir, run_seir_piecewise
    from seir_network_model import watts_strogatz, draw_infectiousness

run_seir(n_nodes, R0, ...)
    Constant-R0 simulation.  Returns (S, E, I, R, ss_events) daily series.

run_seir_piecewise(n_nodes, Rt_schedule, ...)
    Time-varying simulation.  Rt_schedule is a list of (day, R) breakpoints.
    Returns the same five series.

Compartments
------------
  S  Susceptible   – can be infected
  E  Exposed       – infected, incubating, not yet infectious
  I  Infectious    – actively spreading
  R  Recovered     – immune

Parameters (shared by both simulators)
---------------------------------------
  n_nodes          int    number of network nodes
  incubation_days  float  mean incubation period  σ = 1/incubation_days
  infectious_days  float  mean infectious period  γ = 1/infectious_days
  k                int    Watts-Strogatz ring-lattice degree (default 4)
  p_rewire         float  edge rewiring probability (default 0.1)
  seed             int    random seed (default 42)
  superspreading   bool   enable Gamma-modulated per-node β (default False)
  k_disp           float  Gamma dispersion for superspreading (default 0.1)

Superspreading model
--------------------
Each infectious node draws a daily multiplier:
    m ~ Gamma(shape=k_disp, scale=1/k_disp)   →  E[m]=1, Var[m]=1/k_disp
    β_i = 1 − exp(−m · β)   (capped at 1)

E[m]=1 preserves the population-mean β and therefore R0.
Small k_disp → high variance → strong superspreading (SARS/COVID-like ≈ 0.1).
Large k_disp → near-homogeneous (seasonal flu ≈ 1.0).

CLI
---
    python seir_network_model.py --nodes 1000 --R0 2.5 --incubation 5 --infectious 7
    python seir_network_model.py --nodes 1000 --R0 2.5 --superspreading --k-disp 0.1 --plot

References
----------
Lloyd-Smith et al. (2005) Nature 438, 355–359  (superspreading / negative-binomial)
Watts & Strogatz (1998) Nature 393, 440–442    (small-world network)
"""

import argparse
import math
import random
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class State(IntEnum):
    S = 0
    E = 1
    I = 2
    R = 3


# ---------------------------------------------------------------------------
# Network builder
# ---------------------------------------------------------------------------

def watts_strogatz(
    n: int,
    k: int = 4,
    p: float = 0.1,
    seed: int = 42,
) -> dict[int, set[int]]:
    """
    Build a Watts-Strogatz small-world graph.

    Parameters
    ----------
    n : int   – number of nodes
    k : int   – initial ring-lattice degree (must be even)
    p : float – edge rewiring probability  (0 = ring lattice, 1 = random graph)

    Returns
    -------
    dict[int, set[int]]
        Adjacency sets: adj[i] = set of neighbours of node i.
    """
    rng = random.Random(seed)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % n
            adj[i].add(nb)
            adj[nb].add(i)

    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                nb = (i + j) % n
                candidates = [v for v in range(n) if v != i and v not in adj[i]]
                if candidates:
                    new_nb = rng.choice(candidates)
                    adj[i].discard(nb); adj[nb].discard(i)
                    adj[i].add(new_nb); adj[new_nb].add(i)

    return adj


# ---------------------------------------------------------------------------
# Superspreading helper
# ---------------------------------------------------------------------------

def draw_infectiousness(
    base_beta: float,
    superspreading: bool,
    k_disp: float,
    rng: random.Random,
) -> float:
    """
    Per-node, per-day transmission probability.

    Without superspreading → base_beta (constant).
    With superspreading    → β_i = 1 − exp(−m · base_beta),
                             m ~ Gamma(k_disp, 1/k_disp), capped at 1.
    """
    if not superspreading:
        return base_beta
    m = rng.gammavariate(k_disp, 1.0 / k_disp)
    return min(1.0 - math.exp(-m * base_beta), 1.0)


# ---------------------------------------------------------------------------
# Constant-R0 SEIR simulator
# ---------------------------------------------------------------------------

def run_seir(
    n_nodes: int,
    R0: float,
    incubation_days: float = 5.0,
    infectious_days: float = 7.0,
    superspreading: bool = False,
    k_disp: float = 0.1,
    k: int = 4,
    p_rewire: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """
    Discrete-time SEIR model on a Watts-Strogatz network with constant R0.

    Transition probabilities
    ------------------------
    S → E : per I-neighbour independently, with probability β_i per day
            β derived as  R0 / (mean_degree × infectious_days)
    E → I : daily probability σ = 1 / incubation_days
    I → R : daily probability γ = 1 / infectious_days

    Parameters
    ----------
    n_nodes         : int   – network size
    R0              : float – basic reproduction number
    incubation_days : float – mean incubation period (days)
    infectious_days : float – mean infectious period (days)
    superspreading  : bool  – enable individual β heterogeneity
    k_disp          : float – Gamma dispersion for superspreading
    k               : int   – Watts-Strogatz degree
    p_rewire        : float – rewiring probability
    seed            : int   – random seed

    Returns
    -------
    (S, E, I, R, ss_events)  – five lists, one value per simulated day.
    ss_events[t] = number of nodes that caused ≥ 5 new exposures on day t
                   (always zero when superspreading=False).
    """
    SS_THRESHOLD = 5

    rng         = random.Random(seed)
    adj         = watts_strogatz(n_nodes, k=k, p=p_rewire, seed=seed)
    mean_degree = sum(len(v) for v in adj.values()) / n_nodes
    base_beta   = R0 / (mean_degree * infectious_days)
    sigma       = 1.0 / incubation_days
    gamma       = 1.0 / infectious_days

    state                    = [State.S] * n_nodes
    state[rng.randint(0, n_nodes - 1)] = State.E

    S_list: list[int] = []
    E_list: list[int] = []
    I_list: list[int] = []
    R_list: list[int] = []
    ss_list: list[int] = []

    while True:
        S = state.count(State.S); E = state.count(State.E)
        I = state.count(State.I); R = state.count(State.R)
        S_list.append(S); E_list.append(E)
        I_list.append(I); R_list.append(R)

        if E == 0 and I == 0:
            ss_list.append(0)
            break

        new_state = state[:]
        ss_today  = 0

        for node in range(n_nodes):
            if state[node] == State.S:
                for nb in adj[node]:
                    if state[nb] == State.I:
                        b = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = State.E
                            break
            elif state[node] == State.E:
                if rng.random() < sigma:
                    new_state[node] = State.I
            elif state[node] == State.I:
                b         = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                exposures = 0
                for nb in adj[node]:
                    if new_state[nb] == State.S and rng.random() < b:
                        new_state[nb] = State.E
                        exposures    += 1
                if superspreading and exposures >= SS_THRESHOLD:
                    ss_today += 1
                if rng.random() < gamma:
                    new_state[node] = State.R

        ss_list.append(ss_today)
        state = new_state

    return S_list, E_list, I_list, R_list, ss_list


# ---------------------------------------------------------------------------
# Piecewise-R(t) SEIR simulator
# ---------------------------------------------------------------------------

def run_seir_piecewise(
    n_nodes: int,
    Rt_schedule: list[tuple[int, float]],
    incubation_days: float = 5.0,
    infectious_days: float = 7.0,
    superspreading: bool = False,
    k_disp: float = 0.1,
    k: int = 4,
    p_rewire: float = 0.1,
    seed: int = 42,
    n_days: Optional[int] = None,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """
    Discrete-time SEIR model with a piecewise-constant R(t) schedule.

    Parameters
    ----------
    n_nodes       : int
        Number of network nodes.
    Rt_schedule   : list of (day, R) tuples
        Sorted breakpoints.  (t_start, R) means R(t) = R for all t ≥ t_start
        until the next breakpoint.
        Example: [(0, 3.0), (30, 1.2), (60, 0.8)]
          → R=3.0 for days 0–29, R=1.2 for days 30–59, R=0.8 from day 60.
    incubation_days, infectious_days, superspreading, k_disp, k, p_rewire, seed
        Same as run_seir().
    n_days : int or None
        If given, run exactly n_days steps, padding with zeros after extinction.

    Returns
    -------
    (S, E, I, R, ss_events) – same format as run_seir().
    """
    if not Rt_schedule:
        raise ValueError("Rt_schedule must contain at least one (day, R) entry.")

    SS_THRESHOLD = 3
    rng          = random.Random(seed)
    adj          = watts_strogatz(n_nodes, k=k, p=p_rewire, seed=seed)
    mean_degree  = sum(len(v) for v in adj.values()) / n_nodes
    gamma        = 1.0 / infectious_days
    sigma        = 1.0 / incubation_days
    schedule     = sorted(Rt_schedule, key=lambda x: x[0])

    def _beta(t: int) -> float:
        R_now = schedule[0][1]
        for t_start, R_val in schedule:
            if t >= t_start:
                R_now = R_val
            else:
                break
        return R_now / (mean_degree * infectious_days)

    state                    = [State.S] * n_nodes
    state[rng.randint(0, n_nodes - 1)] = State.E

    S_list: list[int] = []
    E_list: list[int] = []
    I_list: list[int] = []
    R_list: list[int] = []
    ss_list: list[int] = []

    t = 0
    while True:
        S = state.count(State.S); E = state.count(State.E)
        I = state.count(State.I); R = state.count(State.R)
        S_list.append(S); E_list.append(E)
        I_list.append(I); R_list.append(R)

        if E == 0 and I == 0:
            ss_list.append(0)
            if n_days is None or t >= n_days - 1:
                break
            while len(I_list) < n_days:
                S_list.append(S_list[-1]); E_list.append(0)
                I_list.append(0);          R_list.append(R_list[-1])
                ss_list.append(0)
            break

        if n_days is not None and t >= n_days - 1:
            ss_list.append(0)
            break

        base_beta = _beta(t)
        new_state = state[:]
        ss_today  = 0

        for node in range(n_nodes):
            if state[node] == State.S:
                for nb in adj[node]:
                    if state[nb] == State.I:
                        b = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = State.E
                            break
            elif state[node] == State.E:
                if rng.random() < sigma:
                    new_state[node] = State.I
            elif state[node] == State.I:
                b         = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                exposures = 0
                for nb in adj[node]:
                    if new_state[nb] == State.S and rng.random() < b:
                        new_state[nb] = State.E
                        exposures    += 1
                if superspreading and exposures >= SS_THRESHOLD:
                    ss_today += 1
                if rng.random() < gamma:
                    new_state[node] = State.R

        ss_list.append(ss_today)
        state = new_state
        t    += 1

    return S_list, E_list, I_list, R_list, ss_list


# ---------------------------------------------------------------------------
# SEIR + intervention simulator  (SEIR-IS-Q model)
# ---------------------------------------------------------------------------

def run_seir_intervention(
    n_nodes:          int,
    R0:               float,
    incubation_days:  float = 5.0,
    infectious_days:  float = 7.0,
    t_intervention:   int   = 30,
    chi:              float | list[float] = 0.1,
    eta:              float | list[float] = 0.1,
    superspreading:   bool  = False,
    k_disp:           float = 0.1,
    k:                int   = 4,
    p_rewire:         float = 0.1,
    seed:             int   = 42,
) -> tuple[
    list[int], list[int], list[int], list[int],
    list[int], list[int], list[int]
]:
    """
    SEIR epidemic with contact-tracing / isolation intervention.

    Extends the standard SEIR model with two new absorbing compartments that
    become active on day ``t_intervention``:

      IS  (Isolated-Susceptible/Exposed)
          Nodes in state E are detected and isolated at daily rate χ.
          They leave the transmission network immediately — they can neither
          receive nor pass on infection.  Biologically this represents
          contact-tracing of exposed individuals or voluntary quarantine of
          those who suspect exposure.

      Q   (Quarantined-Infectious)
          Nodes in state I are detected and quarantined at daily rate η.
          They stop transmitting immediately.  This represents identification
          and isolation of symptomatic cases.

    Both IS and Q nodes eventually recover.  Once in IS or Q, a node recovers
    at the same daily rate γ = 1/infectious_days as an ordinary I node (a
    conservative assumption; in practice isolated individuals may recover at
    the same rate regardless of isolation status).

    Transition diagram
    ------------------
                          t < t_int           t ≥ t_int
      S  → E   β per I-neighbour (same throughout)
      E  → I   σ = 1/incubation_days         (same)
      E  → IS  —                              χ (daily, post-intervention)
      I  → R   γ = 1/infectious_days         (same)
      I  → Q   —                              η (daily, post-intervention)
      IS → R   γ                              (same rate as I→R)
      Q  → R   γ                              (same rate as I→R)

    Competition between E→I and E→IS (post-intervention)
    ------------------------------------------------------
    On each day, an E node faces two competing hazards: advancing to I (rate σ)
    and being isolated (rate χ).  These are modelled as sequential independent
    Bernoulli trials per day:
      - With probability χ  → node moves to IS (checked first).
      - Else with probability σ → node moves to I.
    The resulting daily E→IS probability is χ, and E→I probability is (1−χ)·σ.
    The same sequential logic applies to I→Q vs I→R.

    Scalar vs array rates
    ---------------------
    ``chi`` and ``eta`` may each be:

    - A scalar float: the same rate applies on every day from t_intervention
      onward.
    - A list/array of floats: element [j] is the rate on day
      (t_intervention + j).  If the epidemic outlasts the array, the last
      element is held constant.  This allows time-varying intervention
      intensity (e.g. ramping up testing capacity or relaxing restrictions).

    Parameters
    ----------
    n_nodes         : int
        Number of nodes in the network.
    R0              : float
        Basic reproduction number.
    incubation_days : float
        Mean incubation period σ⁻¹ (default 5.0).
    infectious_days : float
        Mean infectious period γ⁻¹ (default 7.0).
    t_intervention  : int
        Day on which χ and η become active (0-indexed, default 30).
        Before this day the model is standard SEIR.
    chi             : float or list[float]
        Daily E→IS isolation rate (scalar or time-varying array).
        Must be in [0, 1].
    eta             : float or list[float]
        Daily I→Q quarantine rate (scalar or time-varying array).
        Must be in [0, 1].
    superspreading  : bool
        Enable Gamma-modulated individual β (default False).
    k_disp          : float
        Gamma dispersion for superspreading (default 0.1).
    k               : int
        Watts-Strogatz ring-lattice degree (default 4).
    p_rewire        : float
        Edge rewiring probability (default 0.1).
    seed            : int
        Random seed (default 42).

    Returns
    -------
    (S, E, I, R, IS, Q, ss_events)
        Seven daily-count lists.  All seven sum to n_nodes at every time step.
        ss_events[t] = number of I-nodes that caused ≥ 3 exposures on day t
                       (always 0 when superspreading=False).
    """
    # ── Validate and normalise chi / eta to callable index functions ─────────
    def _rate_at(rates, day_offset: int) -> float:
        """Return rate for (t_intervention + day_offset), clamped to [0,1]."""
        if isinstance(rates, (int, float)):
            return float(min(max(rates, 0.0), 1.0))
        idx = min(day_offset, len(rates) - 1)
        return float(min(max(rates[idx], 0.0), 1.0))

    SS_THRESHOLD = 3

    rng         = random.Random(seed)
    adj         = watts_strogatz(n_nodes, k=k, p=p_rewire, seed=seed)
    mean_degree = sum(len(v) for v in adj.values()) / n_nodes
    base_beta   = R0 / (mean_degree * infectious_days)
    sigma       = 1.0 / incubation_days
    gamma       = 1.0 / infectious_days

    # Extended state codes (integers, no enum to keep it lightweight)
    _S  = 0   # Susceptible
    _E  = 1   # Exposed
    _I  = 2   # Infectious
    _R  = 3   # Recovered
    _IS = 4   # Isolated (from E)
    _Q  = 5   # Quarantined (from I)

    state                    = [_S] * n_nodes
    state[rng.randint(0, n_nodes - 1)] = _E

    S_list:  list[int] = []
    E_list:  list[int] = []
    I_list:  list[int] = []
    R_list:  list[int] = []
    IS_list: list[int] = []
    Q_list:  list[int] = []
    ss_list: list[int] = []

    t = 0
    while True:
        S  = state.count(_S);  E  = state.count(_E)
        I  = state.count(_I);  R  = state.count(_R)
        IS = state.count(_IS); Q  = state.count(_Q)

        S_list.append(S);   E_list.append(E)
        I_list.append(I);   R_list.append(R)
        IS_list.append(IS); Q_list.append(Q)

        # Stop when no active transmitters remain
        if E == 0 and I == 0:
            ss_list.append(0)
            break

        # Intervention rates for this day
        intervening = (t >= t_intervention)
        day_offset  = max(t - t_intervention, 0)
        chi_t = _rate_at(chi, day_offset) if intervening else 0.0
        eta_t = _rate_at(eta, day_offset) if intervening else 0.0

        new_state = state[:]
        ss_today  = 0

        for node in range(n_nodes):

            if state[node] == _S:
                # Transmission: any I-neighbour may expose this node
                for nb in adj[node]:
                    if state[nb] == _I:
                        b = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = _E
                            break

            elif state[node] == _E:
                # Post-intervention: E→IS takes priority; else E→I
                if intervening and rng.random() < chi_t:
                    new_state[node] = _IS
                elif rng.random() < sigma:
                    new_state[node] = _I

            elif state[node] == _I:
                # Transmit to susceptible neighbours
                b         = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                exposures = 0
                for nb in adj[node]:
                    if new_state[nb] == _S and rng.random() < b:
                        new_state[nb] = _E
                        exposures    += 1
                if superspreading and exposures >= SS_THRESHOLD:
                    ss_today += 1
                # Post-intervention: I→Q takes priority; else I→R
                if intervening and rng.random() < eta_t:
                    new_state[node] = _Q
                elif rng.random() < gamma:
                    new_state[node] = _R

            elif state[node] == _IS:
                # Isolated: no transmission; recovers at same rate as I
                if rng.random() < gamma:
                    new_state[node] = _R

            elif state[node] == _Q:
                # Quarantined: no transmission; recovers at same rate as I
                if rng.random() < gamma:
                    new_state[node] = _R

        ss_list.append(ss_today)
        state = new_state
        t    += 1

    return S_list, E_list, I_list, R_list, IS_list, Q_list, ss_list


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_summary(
    n_nodes: int, R0: float,
    incubation_days: float, infectious_days: float,
    superspreading: bool, k_disp: float,
    S: list[int], E: list[int], I: list[int], R: list[int], ss: list[int],
) -> None:
    """Print a formatted summary table to stdout."""
    peak_I  = max(I); peak_day = I.index(peak_I)
    w = 54; d = "=" * w
    print(f"\n{d}\n  SEIR Epidemic – Small-World Network\n{d}")
    print(f"  Nodes              : {n_nodes}")
    print(f"  R0                 : {R0}")
    print(f"  Incubation period  : {incubation_days} days")
    print(f"  Infectious period  : {infectious_days} days")
    print(f"  Superspreading     : {'ON  (k=' + str(k_disp) + ')' if superspreading else 'OFF'}")
    print(d)
    print(f"  Duration           : {len(S) - 1} days")
    print(f"  Peak exposed  (E)  : {max(E):,}  (day {E.index(max(E))})")
    print(f"  Peak infectious(I) : {peak_I:,}  (day {peak_day})")
    total = n_nodes - S[-1]
    print(f"  Total ever infected: {total:,}  ({total/n_nodes*100:.1f}%)")
    if superspreading and any(ss):
        print(f"  Superspreader evts : {sum(ss):,} total  (peak {max(ss)} on day {ss.index(max(ss))})")
    print(f"{d}\n")


def ascii_chart(
    E: list[int], I: list[int],
    ss: list[int] | None = None,
    width: int = 60, height: int = 15,
) -> None:
    """ASCII chart of E/I curves with optional superspreader markers."""
    n_days  = len(I)
    max_val = max(max(E), max(I), 1)

    def resample(series, cols):
        out = []
        for c in range(cols):
            idx = c * (len(series) - 1) / max(cols - 1, 1)
            lo  = int(idx); hi = min(lo + 1, len(series) - 1)
            out.append(series[lo] * (1 - (idx - lo)) + series[hi] * (idx - lo))
        return out

    cols   = min(n_days, width)
    E_draw = resample(E, cols)
    I_draw = resample(I, cols)

    legend = "░ Exposed  █ Infectious" + ("  ▲ Superspreader day" if ss and any(ss) else "")
    print(f"  E/I curves  ({legend}):")
    print(f"  {max_val:>6} ┐")
    for row in range(height, 0, -1):
        thr  = max_val * row / height
        line = "".join("█" if I_draw[t] >= thr else ("░" if E_draw[t] >= thr else " ")
                       for t in range(cols))
        print(f"  {'':>6} │{line}")
    if ss and any(ss):
        ss_draw = resample(ss, cols)
        print(f"  {'0':>6} └" + "".join("▲" if v >= 0.5 else "─" for v in ss_draw))
    else:
        print(f"  {'0':>6} └" + "─" * cols)
    print(f"         Day 0{' ' * max(cols - 9, 0)}Day {n_days - 1}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEIR epidemic on a Watts-Strogatz small-world network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nodes",          type=int,   default=500)
    parser.add_argument("--R0",             type=float, default=2.5)
    parser.add_argument("--incubation",     type=float, default=5.0,
                        help="Mean incubation period (days)")
    parser.add_argument("--infectious",     type=float, default=7.0,
                        help="Mean infectious period (days)")
    parser.add_argument("--superspreading", action="store_true",
                        help="Enable individual-level infectiousness heterogeneity")
    parser.add_argument("--k-disp",         type=float, default=0.1, dest="k_disp",
                        help="Superspreading dispersion k (~0.1 COVID, ~1 flu)")
    parser.add_argument("--k",              type=int,   default=4,
                        help="Ring-lattice degree")
    parser.add_argument("--p",              type=float, default=0.1,
                        help="Edge rewiring probability")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--plot",           action="store_true",
                        help="Show matplotlib plot (requires matplotlib)")
    args = parser.parse_args()

    S, E, I, R, ss = run_seir(
        n_nodes=args.nodes, R0=args.R0,
        incubation_days=args.incubation, infectious_days=args.infectious,
        superspreading=args.superspreading, k_disp=args.k_disp,
        k=args.k, p_rewire=args.p, seed=args.seed,
    )

    print_summary(args.nodes, args.R0, args.incubation, args.infectious,
                  args.superspreading, args.k_disp, S, E, I, R, ss)
    ascii_chart(E, I, ss if args.superspreading else None)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            days = list(range(len(S)))
            has_ss = args.superspreading and any(ss)
            fig = plt.figure(figsize=(11, 6))
            gs  = gridspec.GridSpec(2 if has_ss else 1, 1,
                                    height_ratios=([4, 1] if has_ss else [1]),
                                    hspace=0.4)
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(days, S, label="S", color="#2196F3", lw=2)
            ax1.plot(days, E, label="E", color="#FF9800", lw=2, ls="--")
            ax1.plot(days, I, label="I", color="#F44336", lw=2)
            ax1.plot(days, R, label="R", color="#4CAF50", lw=2)
            ss_tag = f", SS k={args.k_disp}" if args.superspreading else ""
            ax1.set_title(f"SEIR Small-World  (N={args.nodes}, R₀={args.R0}, "
                          f"σ⁻¹={args.incubation}d, γ⁻¹={args.infectious}d{ss_tag})")
            ax1.set_ylabel("Count"); ax1.legend(); ax1.grid(alpha=0.3)
            if has_ss:
                ax2 = fig.add_subplot(gs[1], sharex=ax1)
                ax2.bar(days, ss, color="#9C27B0", alpha=0.75)
                ax2.set_xlabel("Day"); ax2.set_ylabel("SS events"); ax2.grid(alpha=0.3)
            else:
                ax1.set_xlabel("Day")
            plt.tight_layout(); plt.show()
        except ImportError:
            print("matplotlib not installed – skipping plot.")
