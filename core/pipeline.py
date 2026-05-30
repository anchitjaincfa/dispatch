"""The shared solve pipeline — every backend funnels through here, differing
ONLY in how it solves the town->shelter assignment (the `assign_fn`).

assign_fn(problem: AssignmentProblem) -> (choice: list[int], meta: dict)
  choice[i] = shelter index chosen for town i
  meta      = backend notes (method, qubits, gap, wall, ...) -> SolveResult.extra
"""
from __future__ import annotations

import time

from . import evacuate, hazard, qubo, routing


def solve_scenario(roadnet, scenario, assign_fn, backend_name="classical"):
    from solver.base import SolveResult   # lazy import avoids package import cycle

    t0 = time.perf_counter()
    blocked = hazard.blocked_nodes(roadnet, scenario)

    # crew routing (classical in every backend)
    crew_assign, crew_routes = routing.assign_crews(roadnet, scenario, blocked)

    # evacuation: build the QUBO core, solve it via the pluggable backend
    prob = evacuate.build_evac_problem(roadnet, scenario, blocked)
    choice, meta = assign_fn(prob)
    assignment = qubo.decode(prob, choice)
    evac_routes, evac_tts, evac_comp = evacuate.reconstruct_evac_routes(
        roadnet, scenario, assignment, blocked)

    obj_person_sec = qubo.energy(prob, choice)
    feasible = qubo.feasible(prob, choice)
    loads = qubo.shelter_load(prob, choice)
    est_clear_min = round(max(evac_tts.values(), default=0.0) / 60.0, 1)

    wall = (time.perf_counter() - t0) * 1000.0
    extra = {
        "blocked_nodes": len(blocked),
        "shelter_load": loads,
        "shelter_cap": {s["id"]: s["capacity"] for s in scenario["shelters"]},
        "est_clear_min": est_clear_min,
        "crew_assignment": crew_assign,
        "endangered": hazard.endangered_towns(scenario),
        "compromised_evacs": [tid for tid, c in evac_comp.items() if c],
        "qubits": prob.num_qubits(),
        **meta,
    }
    return SolveResult(
        crew_routes=crew_routes,
        evac_assignment=assignment,
        evac_routes=evac_routes,
        objective=round(obj_person_sec / 60.0, 1),   # person-minutes
        backend=backend_name,
        wall_ms=round(wall, 1),
        feasible=feasible,
        extra=extra,
    )
