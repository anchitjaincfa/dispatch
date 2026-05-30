"""MockSolver -- BUILD FIRST.

Returns canned-but-plausible crew routes and evacuation assignments instantly,
so the entire front end (map, fire, routes, panels, self-healing motion) works
before any real optimizer exists.

The "intelligence" here is intentionally fake but reactive: it greedily assigns
each crew to the nearest threatened defensible point and each town to the
nearest shelter with remaining capacity, routing around the current fire by a
simple midpoint-deflection heuristic. That's enough to make the map visibly
"heal" when you move the fire, close a road, or remove a crew -- which is the
whole point of Phase F.

Replace this with ORToolsSolver in Phase 2. The UI will not change.
"""
from __future__ import annotations

import math
import time

from .base import QUBOProblem, SolveResult


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _deflect_around_fire(p0: list[float], p1: list[float], fire: list[float],
                         fire_radius: float) -> list[list[float]]:
    """Return a poly-line from p0 to p1 that bends away from the fire center.

    If the straight segment's midpoint is inside the fire radius, push the
    midpoint outward (perpendicular-ish) so the drawn route appears to avoid
    the hazard. Purely cosmetic -- real hazard-aware routing comes in Phase 2.
    """
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]
    d = _dist(mid, fire)
    if d >= fire_radius or d == 0:
        return [p0, p1]
    # unit vector from fire -> midpoint, pushed out past the radius
    ux, uy = (mid[0] - fire[0]) / d, (mid[1] - fire[1]) / d
    push = (fire_radius - d) + fire_radius * 0.4
    bent = [mid[0] + ux * push, mid[1] + uy * push]
    return [p0, bent, p1]


class MockSolver:
    name = "mock"

    def available(self) -> bool:
        return True

    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult:
        t0 = time.perf_counter()
        m = problem.metadata
        stations = m["stations"]            # [{id, name, coord:[lng,lat]}]
        defensible = m["defensible"]        # [{id, coord, risk}]
        towns = m["towns"]                  # [{id, name, coord, population}]
        shelters = m["shelters"]            # [{id, name, coord, capacity}]
        fire = m["fire"]["center"]          # [lng, lat]
        fire_radius = m["fire"]["radius"]
        dead_crews = set(m.get("dead_crews", []))
        closed_roads = m.get("closed_roads", [])  # list of [lng,lat] midpoints to "avoid"

        # --- crew routing: nearest still-threatened defensible point ---
        crew_routes: dict[str, list[list[float]]] = {}
        # only defensible points within ~2x fire radius are "threatened"
        threatened = sorted(
            defensible,
            key=lambda dp: _dist(dp["coord"], fire),
        )
        idx = 0
        for st in stations:
            crew_id = st["id"]
            if crew_id in dead_crews:
                continue
            if not threatened:
                continue
            target = threatened[idx % len(threatened)]
            idx += 1
            route = _deflect_around_fire(st["coord"], target["coord"], fire, fire_radius)
            # nudge around any closed-road midpoints too
            for cr in closed_roads:
                if _dist([(route[0][0] + route[-1][0]) / 2,
                          (route[0][1] + route[-1][1]) / 2], cr) < fire_radius * 0.6:
                    route = _deflect_around_fire(st["coord"], target["coord"], cr, fire_radius)
            crew_routes[crew_id] = route

        # --- evacuation: nearest shelter with remaining capacity ---
        remaining = {s["id"]: s["capacity"] for s in shelters}
        evac_assignment: dict[str, str] = {}
        evac_routes: dict[str, list[list[float]]] = {}
        # evacuate most-threatened (closest to fire) towns first
        for town in sorted(towns, key=lambda t: _dist(t["coord"], fire)):
            best = None
            best_d = float("inf")
            for s in shelters:
                if remaining[s["id"]] < town["population"]:
                    continue
                d = _dist(town["coord"], s["coord"])
                if d < best_d:
                    best_d, best = d, s
            if best is None:  # no capacity left anywhere -> nearest shelter, infeasible flag
                best = min(shelters, key=lambda s: _dist(town["coord"], s["coord"]))
            else:
                remaining[best["id"]] -= town["population"]
            evac_assignment[town["id"]] = best["id"]
            evac_routes[town["id"]] = _deflect_around_fire(
                town["coord"], best["coord"], fire, fire_radius)

        feasible = all(v >= 0 for v in remaining.values())
        objective = sum(
            _dist(t["coord"], next(s for s in shelters if s["id"] == evac_assignment[t["id"]])["coord"])
            for t in towns
        )

        wall = (time.perf_counter() - t0) * 1000.0
        return SolveResult(
            crew_routes=crew_routes,
            evac_assignment=evac_assignment,
            evac_routes=evac_routes,
            objective=round(objective, 4),
            backend=self.name,
            wall_ms=round(wall, 2),
            feasible=feasible,
            extra={"note": "MOCK results -- canned heuristic, not a real optimizer",
                   "shelter_remaining": remaining},
        )
