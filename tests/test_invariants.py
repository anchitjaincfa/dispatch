"""Fast pytest invariants (classical + annealer only; QAOA is exercised separately).

Run:  pytest -q
"""
import json
import os

import pytest

from core import geo, hazard, qubo
from solver import ClassicalSolver, QuantumSolver, QUBOProblem

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = os.path.join(HERE, "..", "data", "scenario_berkeley.json")


@pytest.fixture(scope="module")
def roadnet():
    return geo.build_roadnet()


@pytest.fixture
def scenario():
    with open(SCENARIO) as f:
        return json.load(f)


def test_graph_is_real_and_routable(roadnet):
    assert roadnet.simp.number_of_nodes() > 500
    r = roadnet.route([-122.213, 37.847], [-122.272, 37.868])
    assert len(r.polyline) >= 2


def test_every_town_assigned_and_feasible(roadnet, scenario):
    res = ClassicalSolver(roadnet).solve(QUBOProblem(metadata=scenario))
    assert len(res.evac_assignment) == len(scenario["towns"])
    assert res.feasible, "baseline scenario must be capacity-feasible"


def test_capacity_respected(roadnet, scenario):
    res = ClassicalSolver(roadnet).solve(QUBOProblem(metadata=scenario))
    loads, caps = res.extra["shelter_load"], res.extra["shelter_cap"]
    for sid in caps:
        assert loads[sid] <= caps[sid]


def test_backends_agree_on_assignment(roadnet, scenario):
    a = ClassicalSolver(roadnet).solve(QUBOProblem(metadata=dict(scenario)))
    b = QuantumSolver(roadnet, method="annealer").solve(QUBOProblem(metadata=dict(scenario)))
    assert a.evac_assignment == b.evac_assignment


def test_qubit_budget(roadnet, scenario):
    res = ClassicalSolver(roadnet).solve(QUBOProblem(metadata=scenario))
    assert res.extra["qubits"] <= 20


def test_live_resolve_latency(roadnet, scenario):
    import time
    hazard.advance_fire(scenario)
    t = time.perf_counter()
    ClassicalSolver(roadnet).solve(QUBOProblem(metadata=scenario))
    assert (time.perf_counter() - t) * 1000 < 500


def test_annealer_optimal_on_demo_instance(roadnet, scenario):
    from core import evacuate
    blocked = hazard.blocked_nodes(roadnet, scenario)
    prob = evacuate.build_evac_problem(roadnet, scenario, blocked)
    choice, e = qubo.simulated_anneal(prob)
    _, opt = qubo.brute_force(prob)
    assert abs(e - opt) < 1e-6, "annealer should hit the exact optimum at demo size"


def test_fire_blocks_nodes(roadnet, scenario):
    base = len(hazard.blocked_nodes(roadnet, scenario))
    scenario["fire"]["radius"] *= 1.8
    assert len(hazard.blocked_nodes(roadnet, scenario)) > base
