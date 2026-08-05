"""
SEIR Epidemic Simulation on a Watts-Strogatz Small-World Network with Optional Superspreading
=================================================================
Default States
------
  S  Susceptible: can be infected
  E  Exposed    : infected but not yet infectious (incubation period)
  I  Infectious : actively spreading the disease
  R  Recovered  : immune, no longer infectious (either recovered or dead)

Core parameters
---------------
  n_nodes          : int   : number of nodes (persons) in the network
  R0               : float : basic reproduction number
  incubation_days  : float : mean incubation period (α = 1/incubation_days)
  infectious_days  : float : mean infectious period  (γ = 1/infectious_days)

Superspreading (optional, --superspreading flag)
------------------------------------------------
Individual infectiousness heterogeneity is modelled with a negative-binomial
offspring distribution (Lloyd-Smith et al., Nature 2005).

Each infectious node draws a personal infectiousness multiplier m from:
    m ~ Gamma(shape=k_disp, scale=1/k_disp)   →  E[m]=1, Var[m]=1/k_disp

Its effective per-edge daily transmission probability becomes:
    β_i = 1 − exp(−m · β)          (capped at 1)

Small k_disp → high dispersion → strong superspreading (a few nodes drive
               most transmission; the majority barely infect anyone).
Large k_disp → low dispersion  → approaches the homogeneous SEIR baseline.

The global mean β is preserved in expectation, so R0 is unchanged; only
its *variance* across individuals increases.

  --superspreading        toggle (flag, default: off)
  --k-disp FLOAT          dispersion parameter k (default: 0.1,
                          matching SARS-CoV-1 / early COVID-19 estimates; see Wegehaupt2023)
"""

import math
import random
import argparse
from enum import IntEnum

# ---------------------------------------------------------------------------
# 0.  Enumerate states 
# ---------------------------------------------------------------------------

class State(IntEnum):
    S = 0   # Susceptible
    E = 1   # Exposed  (incubating, not yet infectious)
    I = 2   # Infectious
    R = 3   # Recovered


# ---------------------------------------------------------------------------
# 1.  Watts-Strogatz small-world network
# ---------------------------------------------------------------------------

def watts_strogatz(
    n: int,
    k: int = 4,
    p: float = 0.1,
    seed: int = 42,
) -> dict[int, set[int]]:
    """
    Build a Watts-Strogatz small-world graph.

    n : number of nodes
    k : each node starts connected to k nearest neighbours (must be even)
    p : rewiring probability (0 = ring lattice, 1 = random graph)

    Returns adjacency sets: adj[i] = set of neighbours of node i.
    """
    rng = random.Random(seed)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}

    # Step 1 – ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % n
            adj[i].add(nb)
            adj[nb].add(i)

    # Step 2 – rewiring
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p:
                nb = (i + j) % n
                candidates = [v for v in range(n) if v != i and v not in adj[i]]
                if candidates:
                    new_nb = rng.choice(candidates)
                    adj[i].discard(nb)
                    adj[nb].discard(i)
                    adj[i].add(new_nb)
                    adj[new_nb].add(i)

    return adj


# ---------------------------------------------------------------------------
# 2.  Superspreading helper
# ---------------------------------------------------------------------------

def draw_infectiousness(
    base_beta: float,
    superspreading: bool,
    k_disp: float,
    rng: random.Random,
) -> float:
    """
    Return the effective per-edge, per-day transmission probability for one
    infectious node on one day.

    Without superspreading -> always base_beta (homogeneous).
    With superspreading    -> draw multiplier m ~ Gamma(k_disp, 1/k_disp),
                             then β_i = 1 − exp(−m · base_beta), capped at 1, as explained above.

    The Gamma parameterisation ensures E[m] = 1, so the population-average β
    and hence R0 are preserved; only the individual variance grows as k_disp
    decreases.
    """
    if not superspreading:
        return base_beta
    # Python's random.gammavariate(alpha, beta) → Gamma(shape=alpha, scale=beta)
    m = rng.gammavariate(k_disp, 1.0 / k_disp)
    return min(1.0 - math.exp(-m * base_beta), 1.0)


# ---------------------------------------------------------------------------
# 3.  SEIR simulation on a network
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
    Discrete-time SEIR model on a Watts-Strogatz small-world network.

    Each day:
      S -> E  each I-neighbour independently attempts transmission with
              probability β_i (constant when superspreading=False; drawn
              from a Gamma-modulated distribution otherwise).
      E -> I  with daily probability α = 1 / incubation_days.
      I -> R  with daily probability γ = 1 / infectious_days.

    Returns
    -------
    (S_series, E_series, I_series, R_series, ss_events_series)
    The first four are daily compartment counts.
    ss_events_series[t] = number of nodes that caused ≥ SS_THRESHOLD new exposures on day t (counts superspreading); always all-zeros when superspreading=False.
    """
    SS_THRESHOLD = 3.5   # min new exposures in one day to count as superspreader (arbitrary, since no consensus definition exists: I use a value >3α from the mean)

    rng = random.Random(seed)
    adj = watts_strogatz(n_nodes, k=k, p=p_rewire, seed=seed)

    mean_degree = sum(len(v) for v in adj.values()) / n_nodes
    base_beta   = R0 / (mean_degree * infectious_days)
    sigma       = 1.0 / incubation_days
    gamma       = 1.0 / infectious_days

    state        = [State.S] * n_nodes
    patient_zero = rng.randint(0, n_nodes - 1)
    state[patient_zero] = State.E

    S_list:  list[int] = []
    E_list:  list[int] = []
    I_list:  list[int] = []
    R_list:  list[int] = []
    ss_list: list[int] = []   # superspreader event counts per day

    while True:
        S = state.count(State.S)
        E = state.count(State.E)
        I = state.count(State.I)
        R = state.count(State.R)

        S_list.append(S)
        E_list.append(E)
        I_list.append(I)
        R_list.append(R)

        if E == 0 and I == 0:
            ss_list.append(0)
            break

        new_state = state[:]
        ss_today  = 0

        for node in range(n_nodes):

            if state[node] == State.S:
                # Each infectious neighbour makes an independent attempt
                for nb in adj[node]:
                    if state[nb] == State.I:
                        b = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                        if rng.random() < b:
                            new_state[node] = State.E
                            break   # one exposure per day is enough

            elif state[node] == State.E:
                if rng.random() < sigma:
                    new_state[node] = State.I

            elif state[node] == State.I:
                # Draw this node's personal β for today
                b = draw_infectiousness(base_beta, superspreading, k_disp, rng)
                exposures = 0
                for nb in adj[node]:
                    if new_state[nb] == State.S and rng.random() < b:
                        new_state[nb] = State.E
                        exposures += 1
                if superspreading and exposures >= SS_THRESHOLD:
                    ss_today += 1
                if rng.random() < gamma:
                    new_state[node] = State.R

        ss_list.append(ss_today)
        state = new_state

    return S_list, E_list, I_list, R_list, ss_list


# ---------------------------------------------------------------------------
# 4.  Reporting (for the initial tests)
# ---------------------------------------------------------------------------

def print_report(
    n_nodes: int,
    R0: float,
    incubation_days: float,
    infectious_days: float,
    superspreading: bool,
    k_disp: float,
    S: list[int],
    E: list[int],
    I: list[int],
    R: list[int],
    ss: list[int],
) -> None:
    peak_I      = max(I)
    peak_day    = I.index(peak_I)
    peak_E      = max(E)
    total_inf   = n_nodes - S[-1]
    attack_rate = total_inf / n_nodes * 100

    w   = 54
    div = "=" * w
    print(f"\n{div}")
    print(f"  SEIR Epidemic – Small-World Network")
    print(div)
    print(f"  Nodes              : {n_nodes}")
    print(f"  R0                 : {R0}")
    print(f"  Incubation period  : {incubation_days} days")
    print(f"  Infectious period  : {infectious_days} days")
    if superspreading:
        print(f"  Superspreading     : ON  (dispersion k = {k_disp})")
    else:
        print(f"  Superspreading     : OFF (homogeneous β)")
    print(div)
    print(f"  Duration           : {len(S) - 1} days")
    print(f"  Peak exposed  (E)  : {peak_E:,}  (day {E.index(peak_E)})")
    print(f"  Peak infectious(I) : {peak_I:,}  (day {peak_day})")
    print(f"  Total ever infected: {total_inf:,}  ({attack_rate:.1f}% attack rate)")
    if superspreading:
        total_ss   = sum(ss)
        peak_ss    = max(ss)
        peak_ss_dy = ss.index(peak_ss)
        print(f"  Superspreader evts : {total_ss:,} total  "
              f"(peak {peak_ss} on day {peak_ss_dy})")
    print(f"{div}\n")


def ascii_chart(
    E: list[int],
    I: list[int],
    ss: list[int] | None = None,
    width: int = 60,
    height: int = 15,
) -> None:
    """
    Dual ASCII chart: exposed (░) and infectious (█).
    When superspreader event counts (ss) are supplied, days with at least one
    event are marked with ▲ on the bottom axis instead of ─.
    Time axis is downsampled when the epidemic outlasts `width` columns.
    """
    n_days  = len(I)
    max_val = max(max(E), max(I), 1)

    def resample(series: list[int], cols: int) -> list[float]:
        out = []
        for c in range(cols):
            idx      = c * (len(series) - 1) / max(cols - 1, 1)
            lo, hi   = int(idx), min(int(idx) + 1, len(series) - 1)
            frac     = idx - lo
            out.append(series[lo] * (1 - frac) + series[hi] * frac)
        return out

    cols   = min(n_days, width)
    E_draw = resample(E, cols)
    I_draw = resample(I, cols)

    legend = "░ Exposed  █ Infectious"
    if ss and any(ss):
        legend += "  ▲ Superspreader day"
    print(f"  E/I curves over time  ({legend}):")
    print(f"  {max_val:>6} ┐")
    for row in range(height, 0, -1):
        threshold = max_val * row / height
        line = ""
        for t in range(cols):
            if I_draw[t] >= threshold:
                line += "█"
            elif E_draw[t] >= threshold:
                line += "░"
            else:
                line += " "
        print(f"  {'':>6} │{line}")

    # Bottom axis: mark superspreader event days with ▲
    if ss and any(ss):
        ss_draw  = resample(ss, cols)
        axis_row = "".join("▲" if v >= 0.5 else "─" for v in ss_draw)
        print(f"  {'0':>6} └{axis_row}")
    else:
        print(f"  {'0':>6} └" + "─" * cols)

    pad = max(cols - 9, 0)
    print(f"         Day 0{' ' * pad}Day {n_days - 1}\n")


# ---------------------------------------------------------------------------
# 5.  Entry point (for intial tests) 
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SEIR epidemic on a Watts-Strogatz small-world network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core epidemic parameters
    parser.add_argument("--nodes",          type=int,   default=500,
                        help="Number of nodes in the network")
    parser.add_argument("--R0",             type=float, default=2.5,
                        help="Basic reproduction number")
    parser.add_argument("--incubation",     type=float, default=5.0,
                        help="Mean incubation period (days, E→I)")
    parser.add_argument("--infectious",     type=float, default=7.0,
                        help="Mean infectious period  (days, I→R)")

    # ── Superspreading toggle ──────────────────────────────────────────────
    parser.add_argument("--superspreading", action="store_true",
                        help="Enable individual infectiousness heterogeneity "
                             "(negative-binomial offspring distribution). "
                             "OFF by default.")
    parser.add_argument("--k-disp",         type=float, default=0.1,
                        dest="k_disp",
                        help="Dispersion parameter k (only used when "
                             "--superspreading is set). "
                             "Smaller = more heterogeneous. "
                             "~0.1 → SARS/COVID-like; ~1 → seasonal-flu-like.")
    # ──────────────────────────────────────────────────────────────────────

    # Network parameters
    parser.add_argument("--k",              type=int,   default=4,
                        help="Initial ring-lattice degree")
    parser.add_argument("--p",              type=float, default=0.1,
                        help="Edge rewiring probability")
    # Misc
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--plot",           action="store_true",
                        help="Show matplotlib plot (requires matplotlib)")
    args = parser.parse_args()

    S, E, I, R, ss = run_seir(
        n_nodes         = args.nodes,
        R0              = args.R0,
        incubation_days = args.incubation,
        infectious_days = args.infectious,
        superspreading  = args.superspreading,
        k_disp          = args.k_disp,
        k               = args.k,
        p_rewire        = args.p,
        seed            = args.seed,
    )

    print_report(
        args.nodes, args.R0, args.incubation, args.infectious,
        args.superspreading, args.k_disp,
        S, E, I, R, ss,
    )
    ascii_chart(E, I, ss if args.superspreading else None)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            days = list(range(len(S)))
            has_ss_events = args.superspreading and any(ss)

            if has_ss_events:
                fig = plt.figure(figsize=(11, 6))
                gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.4)
                ax1 = fig.add_subplot(gs[0])
                ax2 = fig.add_subplot(gs[1], sharex=ax1)
            else:
                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax2 = None

            ax1.plot(days, S, label="Susceptible (S)", color="#2196F3", linewidth=2)
            ax1.plot(days, E, label="Exposed     (E)", color="#FF9800", linewidth=2, linestyle="--")
            ax1.plot(days, I, label="Infectious  (I)", color="#F44336", linewidth=2)
            ax1.plot(days, R, label="Recovered   (R)", color="#4CAF50", linewidth=2)
            ax1.set_ylabel("Count")
            ss_tag = f", superspreading k={args.k_disp}" if args.superspreading else ""
            ax1.set_title(
                f"SEIR on Small-World Network  "
                f"(N={args.nodes}, R₀={args.R0}, "
                f"α⁻¹={args.incubation}d, γ⁻¹={args.infectious}d{ss_tag})"
            )
            ax1.legend()
            ax1.grid(alpha=0.3)

            if ax2 is not None:
                ax2.bar(days, ss, color="#9C27B0", alpha=0.75,
                        label="Superspreader events / day")
                ax2.set_xlabel("Day")
                ax2.set_ylabel("Events")
                ax2.legend(fontsize=8)
                ax2.grid(alpha=0.3)
            else:
                ax1.set_xlabel("Day")

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not installed – skipping plot.")
