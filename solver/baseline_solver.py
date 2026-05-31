"""CAD closest-unit baseline — what live dispatch does TODAY (the foil to beat).

Each town goes to its NEAREST shelter, capacity-blind, so popular shelters overflow.
Same scenario, same roads, same map — but visibly worse (capacity violations, people
unsheltered) than the DISPATCH global optimizer. It is deliberately NOT a real
optimizer; that contrast is the "it's really optimizing" moment.
"""
from __future__ import annotations

from core import qubo
from core.pipeline import solve_scenario

from .base import QUBOProblem, SolveResult


def baseline_assign(prob):
    return (qubo.greedy_assign(prob),
            {"method": "CAD closest-unit (nearest shelter, capacity-blind)"})


class BaselineSolver:
    name = "baseline"

    def __init__(self, roadnet):
        self.roadnet = roadnet

    def available(self) -> bool:
        return True

    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult:
        return solve_scenario(self.roadnet, problem.metadata, baseline_assign,
                              backend_name="baseline")
