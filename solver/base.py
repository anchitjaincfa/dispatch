"""DISPATCH solver interface.

Every backend (mock, OR-Tools, QAOA/Aer, XPYQ) implements this SAME interface.
The UI and core never import a concrete solver directly -- they depend only on
the `Solver` protocol below. This is what makes the front-end-first approach
work: the app talks to MockSolver now, and you swap in real solvers later
WITHOUT touching the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class QUBOProblem:
    """A combinatorial assignment problem, expressed generically.

    For the mock/classical phases you mostly care about `metadata`, which
    carries the human-readable scenario. The QUBO fields (`Q`, `num_vars`,
    `var_labels`) become important in Phase 4 when you formulate the real
    quantum mapping.
    """
    Q: dict[tuple[int, int], float] = field(default_factory=dict)
    num_vars: int = 0
    var_labels: dict[int, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class SolveResult:
    """The output every backend must return, so the UI can render uniformly."""
    # crew_id -> list of [lng, lat] points forming the crew's route
    crew_routes: dict[str, list[list[float]]] = field(default_factory=dict)
    # town_id -> shelter_id assignment
    evac_assignment: dict[str, str] = field(default_factory=dict)
    # town_id -> list of [lng, lat] points forming the evacuation path
    evac_routes: dict[str, list[list[float]]] = field(default_factory=dict)
    objective: float = 0.0
    backend: str = "mock"            # "mock"|"ortools"|"qaoa_aer"|"xpyq"
    wall_ms: float = 0.0
    feasible: bool = True
    extra: dict = field(default_factory=dict)   # qubit count, shots, gap, notes


@runtime_checkable
class Solver(Protocol):
    name: str

    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult:
        ...

    def available(self) -> bool:
        ...
