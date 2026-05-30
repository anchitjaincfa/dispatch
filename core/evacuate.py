"""Evacuation: build the capacitated town->shelter assignment problem (the QUBO
core) and reconstruct the on-road evacuation routes once an assignment is chosen.

The *solving* of the assignment is pluggable (classical CP-SAT / annealer / QAOA
/ XpyQ) — see the solver backends. This module only builds the problem and draws
the resulting routes on real streets.
"""
from __future__ import annotations

from . import qubo


def build_evac_problem(roadnet, scenario, blocked):
    """Cost[i][j] = population_i * travel_time(town_i -> shelter_j), fire-aware."""
    towns = scenario["towns"]
    shelters = scenario["shelters"]
    coords = [t["coord"] for t in towns] + [s["coord"] for s in shelters]
    mat = roadnet.matrix(coords, blocked)
    nT = len(towns)
    cost = [[t["population"] * mat[i][nT + j] for j in range(len(shelters))]
            for i, t in enumerate(towns)]
    return qubo.build_problem(towns, shelters, cost)


def reconstruct_evac_routes(roadnet, scenario, assignment, blocked):
    """assignment {town_id: shelter_id} -> (routes {tid: polyline},
    travel_times {tid: seconds}, compromised {tid: bool}) on real roads."""
    town_by_id = {t["id"]: t for t in scenario["towns"]}
    sh_by_id = {s["id"]: s for s in scenario["shelters"]}
    routes, tts, comp = {}, {}, {}
    for tid, sid in assignment.items():
        r = roadnet.route(town_by_id[tid]["coord"], sh_by_id[sid]["coord"], blocked)
        routes[tid], tts[tid], comp[tid] = r.polyline, r.travel_time, r.compromised
    return routes, tts, comp
