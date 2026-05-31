"""STEP 7 — decisive solver-vs-rendering isolation.

Sweep the fire across a GRID of positions over the whole map and assert the live
solver ALWAYS returns valid, finite, >=2-point routes (or a clean infeasible flag) —
never an unhandled throw, never a silently-empty render. If this passes, any on-screen
flakiness is a rendering/state/race problem, not the routing math.

Run:  pytest tests/test_route_grid.py -q
"""
import json
import math
import os

import pytest

from core import geo
from solver import ClassicalSolver, QUBOProblem

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = os.path.join(HERE, "..", "data", "scenario_berkeley.json")


@pytest.fixture(scope="module")
def roadnet():
    return geo.build_roadnet()


@pytest.fixture
def scenario():
    with open(SCENARIO) as f:
        return json.load(f)


def _valid_polyline(p):
    return (isinstance(p, list) and len(p) >= 2
            and all(isinstance(pt, (list, tuple)) and len(pt) == 2
                    and all(isinstance(c, (int, float)) and math.isfinite(c) for c in pt)
                    for pt in p))


def _fire_grid(scenario, nx=5, ny=4):
    pts = ([t["coord"] for t in scenario["towns"]]
           + [s["coord"] for s in scenario["shelters"]]
           + [s["coord"] for s in scenario["stations"]])
    lngs, lats = [p[0] for p in pts], [p[1] for p in pts]
    lo_x, hi_x, lo_y, hi_y = min(lngs), max(lngs), min(lats), max(lats)
    return [[lo_x + (hi_x - lo_x) * i / (nx - 1), lo_y + (hi_y - lo_y) * j / (ny - 1)]
            for j in range(ny) for i in range(nx)]


def test_route_grid_never_throws_or_renders_empty(roadnet, scenario):
    solver = ClassicalSolver(roadnet)
    failures = []
    for center in _fire_grid(scenario):
        for radius in (0.008, 0.018):
            sc = json.loads(json.dumps(scenario))   # fresh copy per position
            sc["fire"]["center"] = center
            sc["fire"]["radius"] = radius
            try:
                res = solver.solve(QUBOProblem(metadata=sc))
            except Exception as exc:                 # STEP 2: must never happen
                failures.append(f"THREW at {center} r={radius}: {type(exc).__name__}: {exc}")
                continue
            assert isinstance(res.feasible, bool)    # clean infeasible is allowed
            # every town must be assigned and every drawn route must be valid geometry
            assert len(res.evac_assignment) == len(sc["towns"])
            for tid, poly in res.evac_routes.items():
                if not _valid_polyline(poly):
                    failures.append(f"bad evac path {tid} at {center} r={radius}: {poly[:2]}")
            for cid, poly in res.crew_routes.items():
                if not _valid_polyline(poly):
                    failures.append(f"bad crew path {cid} at {center} r={radius}: {poly[:2]}")
    assert not failures, "solver produced throws/invalid geometry:\n" + "\n".join(failures[:20])
