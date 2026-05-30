"""XpyQ backend — submit the SAME town->shelter assignment to XpyQ Cloud's
accelerated `/decisions` optimizer.

XpyQ is invite-only/NDA-gated, so the exact request schema isn't public. This is
a clean, env-driven adapter: set XPYQ_API_KEY (and optionally XPYQ_ENDPOINT) and
drop the real request body into `_submit` — it's a ~10-line edit confirmed from
the logged-in platform. With no key, it transparently falls back to the numpy
annealer and labels the result honestly, so the backend selector always works.

Model (from the public site): you submit a Python decision/objective function +
data; XpyQ runs it on RRAM/neuromorphic backends and returns the decision. Here
the decision is the assignment vector minimising person-travel-time under
shelter-capacity constraints.
"""
from __future__ import annotations

import os

from core import qubo
from core.pipeline import solve_scenario

from .base import QUBOProblem, SolveResult

XPYQ_ENDPOINT = os.environ.get("XPYQ_ENDPOINT", "https://xpyq.vercel.app/api/decisions")


def _submit(prob, api_key):
    """POST the assignment problem to XpyQ /decisions. Returns choice list.

    NOTE: adjust the payload/parse to the real schema from the logged-in platform.
    """
    import requests

    payload = {
        "problem": "capacitated_assignment",
        "objective": "minimize sum_i pop_i * cost[i][assign_i]",
        "cost": prob.cost,
        "pop": prob.pop,
        "cap": prob.cap,
        "constraints": ["each town -> exactly one shelter",
                        "sum_i pop_i [assign_i==j] <= cap_j"],
    }
    r = requests.post(XPYQ_ENDPOINT, json=payload,
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
    r.raise_for_status()
    data = r.json()
    # expected: {"assignment": [shelter_index_per_town], ...}
    choice = data["assignment"]
    assert len(choice) == prob.T
    return choice


def xpyq_assign(prob):
    api_key = os.environ.get("XPYQ_API_KEY")
    if api_key:
        try:
            choice = _submit(prob, api_key)
            return choice, {"method": "XpyQ /decisions (live)", "endpoint": XPYQ_ENDPOINT}
        except Exception as exc:
            choice, e = qubo.simulated_anneal(prob, iters=4000, seed=0)
            return choice, {"method": "annealer (XpyQ call failed)",
                            "xpyq_error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    # no credentials -> honest local stand-in
    choice, e = qubo.simulated_anneal(prob, iters=4000, seed=0)
    return choice, {"method": "annealer (XpyQ stand-in — set XPYQ_API_KEY for live)",
                    "energy": round(e, 1)}


class XpyQSolver:
    name = "xpyq"

    def __init__(self, roadnet):
        self.roadnet = roadnet

    def available(self) -> bool:
        return bool(os.environ.get("XPYQ_API_KEY"))

    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult:
        return solve_scenario(self.roadnet, problem.metadata, xpyq_assign,
                              backend_name="xpyq")
