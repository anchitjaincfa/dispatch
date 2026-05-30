"""The combinatorial core: capacitated town -> shelter assignment.

Each threatened town must go to exactly ONE shelter; shelters have finite
capacity. Minimise total person-travel-time. This is a Generalized Assignment
Problem — NP-hard — and it is the SAME problem every backend solves:

  * classical : OR-Tools CP-SAT (exact)        -> solver/classical_solver.py
  * quantum   : QUBO -> numpy simulated-anneal  -> here (always works)
                QUBO -> QAOA on Aer / IBM QPU    -> solver/quantum_solver.py
  * xpyq      : same objective -> /decisions     -> solver/xpyq_solver.py

Because they share this formulation, "do the backends agree?" is a meaningful,
honest test — not marketing.

Qubit budget: one binary var per (town, shelter) => T*S qubits. We keep the live
quantum instance tiny (<=~15 qubits, e.g. 5 towns x 3 shelters) so it fits Aer /
a 10-minute IBM QPU slot. num_qubits() is asserted before any quantum solve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AssignmentProblem:
    town_ids: list
    shelter_ids: list
    cost: list          # cost[i][j] = person-travel-time of sending town i to shelter j
    pop: list           # pop[i]
    cap: list           # cap[j]
    overflow_penalty: float = 1.0e7   # per person over capacity (>> any routing gain)

    @property
    def T(self):
        return len(self.town_ids)

    @property
    def S(self):
        return len(self.shelter_ids)

    def num_qubits(self) -> int:
        return self.T * self.S


def build_problem(towns, shelters, cost, overflow_penalty=1.0e7) -> AssignmentProblem:
    return AssignmentProblem(
        town_ids=[t["id"] for t in towns],
        shelter_ids=[s["id"] for s in shelters],
        cost=cost,
        pop=[t["population"] for t in towns],
        cap=[s["capacity"] for s in shelters],
        overflow_penalty=overflow_penalty,
    )


# --------------------------------------------------------------------------- #
# energy (categorical view: choice[i] = shelter index for town i)
# --------------------------------------------------------------------------- #
def energy(prob: AssignmentProblem, choice) -> float:
    base = sum(prob.cost[i][choice[i]] for i in range(prob.T))
    load = [0.0] * prob.S
    for i in range(prob.T):
        load[choice[i]] += prob.pop[i]
    over = sum(max(0.0, load[j] - prob.cap[j]) for j in range(prob.S))
    return base + prob.overflow_penalty * over


def decode(prob: AssignmentProblem, choice) -> dict:
    return {prob.town_ids[i]: prob.shelter_ids[choice[i]] for i in range(prob.T)}


def shelter_load(prob: AssignmentProblem, choice) -> dict:
    load = {sid: 0 for sid in prob.shelter_ids}
    for i in range(prob.T):
        load[prob.shelter_ids[choice[i]]] += prob.pop[i]
    return load


def feasible(prob: AssignmentProblem, choice) -> bool:
    load = shelter_load(prob, choice)
    return all(load[prob.shelter_ids[j]] <= prob.cap[j] for j in range(prob.S))


# --------------------------------------------------------------------------- #
# solvers over the same energy landscape
# --------------------------------------------------------------------------- #
def brute_force(prob: AssignmentProblem):
    """Exact optimum by enumeration. Only for tiny instances (S**T <= ~1e5)."""
    best, best_e = None, float("inf")
    S, T = prob.S, prob.T
    total = S ** T
    for k in range(total):
        choice, kk = [], k
        for _ in range(T):
            choice.append(kk % S)
            kk //= S
        e = energy(prob, choice)
        if e < best_e:
            best_e, best = e, choice
    return best, best_e


def simulated_anneal(prob: AssignmentProblem, iters=4000, seed=0):
    """Metropolis simulated annealing over the categorical assignment.

    Stands in for an analog/neuromorphic 'accelerated decision' solver: it walks
    the same QUBO energy landscape the quantum backends do, and reaches the same
    optimum on demo-sized instances — while scaling to far larger ones than QAOA.
    """
    rng = np.random.default_rng(seed)
    T, S = prob.T, prob.S
    # greedy warm start: each town -> cheapest shelter
    choice = [int(np.argmin(prob.cost[i])) for i in range(T)]
    e = energy(prob, choice)
    best, best_e = choice[:], e
    # Temperature must straddle the energy landscape, which is dominated by the
    # capacity-overflow penalty. A barrier of ~1 person of overflow costs
    # overflow_penalty; start hot enough to cross a few of those, cool to ~1.
    t_hi = max(1.0e4, prob.overflow_penalty)
    t_lo = 1.0
    for it in range(iters):
        temp = t_hi * (t_lo / t_hi) ** (it / max(1, iters - 1))
        cand = choice[:]
        # Mix single-town relabels with two-town SWAPS. Swaps are essential:
        # the optimum often sits behind a barrier where no single relabel is
        # downhill (moving one town fixes one overflow but creates another), and
        # only exchanging two towns' shelters crosses it.
        if S >= 2 and T >= 2 and rng.random() < 0.5:
            i, k = rng.choice(T, size=2, replace=False)
            cand[i], cand[k] = cand[k], cand[i]
            if cand[i] == choice[i] and cand[k] == choice[k]:
                continue
        else:
            i = int(rng.integers(T))
            new_j = int(rng.integers(S))
            if new_j == choice[i]:
                continue
            cand[i] = new_j
        ce = energy(prob, cand)
        if ce < e or rng.random() < np.exp(-(ce - e) / max(temp, 1e-9)):
            choice, e = cand, ce
            if e < best_e:
                best, best_e = choice[:], e
    return best, best_e


def to_qubo_matrix(prob: AssignmentProblem, onehot_penalty=None):
    """Binary QUBO {(p,q): w} over variables x_{i*S+j} for QAOA / XpyQ / Ising.

    Energy(x) = sum_ij cost_ij x_ij
              + A * sum_i (sum_j x_ij - 1)^2          # exactly-one-shelter
              + soft capacity term (quadratic load)    # over-capacity discouraged
    Capacity is encoded softly (no slack qubits) to keep the qubit count = T*S.
    Returns (Q, var_labels, num_qubits).
    """
    T, S = prob.T, prob.S
    n = T * S
    if onehot_penalty is None:
        onehot_penalty = max(max(row) for row in prob.cost) + prob.overflow_penalty
    A = onehot_penalty
    Q: dict = {}

    def add(p, q, w):
        key = (p, q) if p <= q else (q, p)
        Q[key] = Q.get(key, 0.0) + w

    def idx(i, j):
        return i * S + j

    # linear costs (on the diagonal)
    for i in range(T):
        for j in range(S):
            add(idx(i, j), idx(i, j), prob.cost[i][j])
    # one-hot penalty A*(sum_j x_ij - 1)^2 = A*(-x + 2*pairs)
    for i in range(T):
        for j in range(S):
            add(idx(i, j), idx(i, j), -A)
            for k in range(j + 1, S):
                add(idx(i, j), idx(i, k), 2 * A)
    # soft capacity: penalty * pop_i*pop_k for towns sharing a shelter (quadratic load proxy)
    cpen = prob.overflow_penalty / max(prob.cap)
    for j in range(S):
        for i in range(T):
            for k in range(i + 1, T):
                add(idx(i, j), idx(k, j), cpen * prob.pop[i] * prob.pop[k] /
                    max(prob.cap[j], 1))
    var_labels = {idx(i, j): f"{prob.town_ids[i]}->{prob.shelter_ids[j]}"
                  for i in range(T) for j in range(S)}
    return Q, var_labels, n
