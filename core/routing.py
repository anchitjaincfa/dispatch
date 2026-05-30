"""Crew routing — assign fire crews from stations to defensible points and route
them there along real roads, avoiding the fire. Risk-weighted: a crew will
travel further for a higher-risk point.

This stays classical in every backend (it's the road-routing half of the honest
hybrid). The combinatorial QUBO core is the town->shelter assignment (qubo.py).
"""
from __future__ import annotations


def assign_crews(roadnet, scenario, blocked):
    """Greedy risk-first assignment of live crews to defensible points.

    Returns (assignment {crew_id: dp_id}, routes {crew_id: polyline}).
    """
    dead = set(scenario.get("dead_crews", []))
    crews = [s for s in scenario["stations"] if s["id"] not in dead]
    dps = sorted(scenario["defensible"], key=lambda d: -d["risk"])
    if not crews or not dps:
        return {}, {}

    crew_coords = [c["coord"] for c in crews]
    dp_coords = [d["coord"] for d in dps]
    # travel-time matrix crews x dps (fire-aware)
    mat = roadnet.matrix(crew_coords + dp_coords, blocked)
    nC = len(crews)

    def tt(ci, dj):
        return mat[ci][nC + dj]

    assignment, routes = {}, {}
    used_crew = set()
    # cover highest-risk points first with their best free crew
    for dj, dp in enumerate(dps):
        best_ci, best_cost = None, float("inf")
        for ci, crew in enumerate(crews):
            if ci in used_crew:
                continue
            cost = tt(ci, dj) / max(dp["risk"], 0.1)
            if cost < best_cost:
                best_cost, best_ci = cost, ci
        if best_ci is None:
            break
        used_crew.add(best_ci)
        crew = crews[best_ci]
        assignment[crew["id"]] = dp["id"]
        routes[crew["id"]] = roadnet.route(crew["coord"], dp["coord"], blocked).polyline
    return assignment, routes
