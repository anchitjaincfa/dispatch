"""Scaling benchmark — the honest-measurement artifact.

The Luma rubric asks for 'honest measurement, compared against classical
baselines.' This sweeps the town->shelter assignment from small to large and
times each backend, so we can SHOW (not claim) where exact methods slow down and
why re-solve latency — the binding product constraint — motivates accelerated
hardware. No overclaiming: the annealer is heuristic; we report its optimality
gap on sizes where brute force is still tractable.
"""
from __future__ import annotations

import time

import numpy as np

from . import qubo


def _random_problem(T, S, seed):
    rng = np.random.default_rng(seed)
    pop = [int(rng.integers(500, 2500)) for _ in range(T)]
    total = sum(pop)
    cap = [int(total / S * 1.25) for _ in range(S)]          # feasible w/ slack
    cost = [[float(pop[i] * rng.integers(60, 1800)) for _ in range(S)] for i in range(T)]
    return qubo.build_problem(
        [{"id": f"t{i}", "population": pop[i]} for i in range(T)],
        [{"id": f"s{j}", "capacity": cap[j]} for j in range(S)],
        cost)


def _cpsat_ms(prob):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return None
    t = time.perf_counter()
    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{i}_{j}") for j in range(prob.S)] for i in range(prob.T)]
    for i in range(prob.T):
        m.AddExactlyOne(x[i])
    for j in range(prob.S):
        m.Add(sum(int(prob.pop[i]) * x[i][j] for i in range(prob.T)) <= int(prob.cap[j]))
    m.Minimize(sum(int(prob.cost[i][j]) * x[i][j]
                   for i in range(prob.T) for j in range(prob.S)))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = 5.0
    s.Solve(m)
    return (time.perf_counter() - t) * 1000


def run(sizes=((4, 3), (6, 3), (8, 4), (10, 4), (12, 5), (16, 6), (20, 8)), seed=7):
    rows = []
    for T, S in sizes:
        prob = _random_problem(T, S, seed)
        n = T * S

        t = time.perf_counter()
        choice, e = qubo.simulated_anneal(prob, iters=4000, seed=seed)
        anneal_ms = (time.perf_counter() - t) * 1000

        cpsat_ms = _cpsat_ms(prob)

        brute_ms, gap = None, None
        if S ** T <= 2_000_000:                       # exact only while tractable
            t = time.perf_counter()
            _, opt_e = qubo.brute_force(prob)
            brute_ms = (time.perf_counter() - t) * 1000
            gap = round(e - opt_e, 1)

        rows.append({"vars": n, "T": T, "S": S,
                     "anneal_ms": round(anneal_ms, 1),
                     "cpsat_ms": round(cpsat_ms, 1) if cpsat_ms else None,
                     "brute_ms": round(brute_ms, 1) if brute_ms else None,
                     "anneal_gap": gap})
    return rows


if __name__ == "__main__":
    for r in run():
        print(r)
