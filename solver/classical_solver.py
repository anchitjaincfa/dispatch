"""Classical backend — the always-works baseline AND the live re-solve path.

Town->shelter assignment via OR-Tools CP-SAT (exact, hard capacity). If CP-SAT is
absent or the instance is capacity-infeasible, falls back to a greedy
nearest-feasible assignment so the map NEVER fails to produce a plan.
"""
from __future__ import annotations

from core import qubo
from core.pipeline import solve_scenario

from .base import QUBOProblem, SolveResult


def greedy_assign(prob):
    """Nearest-feasible: most-populous towns first -> cheapest shelter with room."""
    remaining = list(prob.cap)
    order = sorted(range(prob.T), key=lambda i: -prob.pop[i])
    choice = [0] * prob.T
    for i in order:
        best_j, best_c = None, float("inf")
        for j in range(prob.S):
            if remaining[j] >= prob.pop[i] and prob.cost[i][j] < best_c:
                best_c, best_j = prob.cost[i][j], j
        if best_j is None:                      # no room anywhere -> cheapest, overflow
            best_j = min(range(prob.S), key=lambda j: prob.cost[i][j])
        else:
            remaining[best_j] -= prob.pop[i]
        choice[i] = best_j
    return choice, {"method": "greedy-nearest-feasible"}


def cp_sat_assign(prob):
    try:
        from ortools.sat.python import cp_model
    except Exception:
        return greedy_assign(prob)
    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(prob.S)] for i in range(prob.T)]
    for i in range(prob.T):
        model.AddExactlyOne(x[i])
    for j in range(prob.S):
        model.Add(sum(int(prob.pop[i]) * x[i][j] for i in range(prob.T)) <= int(prob.cap[j]))
    SCALE = 1
    model.Minimize(sum(int(round(prob.cost[i][j] * SCALE)) * x[i][j]
                       for i in range(prob.T) for j in range(prob.S)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1.0
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        choice = [next(j for j in range(prob.S) if solver.Value(x[i][j]) == 1)
                  for i in range(prob.T)]
        return choice, {"method": "or-tools-cp-sat",
                        "optimal": status == cp_model.OPTIMAL}
    return greedy_assign(prob)   # hard-capacity infeasible -> soft greedy


class ClassicalSolver:
    name = "classical"

    def __init__(self, roadnet):
        self.roadnet = roadnet

    def available(self) -> bool:
        return True

    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult:
        return solve_scenario(self.roadnet, problem.metadata, cp_sat_assign,
                              backend_name="classical")
