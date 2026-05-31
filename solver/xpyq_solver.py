"""XpyQ backend — REAL matrix-compute on XpyQ Cloud's hardware boards.

XpyQ is a purpose-built matrix-compute platform — NOT a QUBO/Ising optimizer (there
is no optimization endpoint). You POST a string of Python code that does matrix-heavy
math; it routes the ops onto its boards and returns stdout text.

So we use it honestly for the matrix-heavy core of a SPECTRAL RELAXATION of our exact
town->shelter assignment QUBO:

  1. build the symmetric QUBO matrix M (quadratic form x^T M x == assignment energy),
  2. ship M to XpyQ and run `linalg.eigh(M)` ON ITS BOARDS (bills ~1 credit),
  3. round the lowest-eigenvalue eigenvector per-town into a shelter assignment,
  4. keep it if feasible; otherwise fall back to the local annealer.

This is a genuine hardware computation on the SAME problem (not a fake endpoint) and
surfaces the real run stats (credits, duration, boards). With no XPYQ_KEY it falls
back to the numpy annealer and labels the result honestly, so the selector always works.

API (verified 2026-05-30):
  base  https://xpyq-lib-production.up.railway.app
  auth  Authorization: Bearer xpyq_live_...
  POST  /api/v1/compute/runs           {code, name} -> {run_id, status}
  GET   /api/v1/compute/runs/{run_id}   -> {status, stdout, duration_ms, credits_charged, boards_used}
  runtime namespace is pre-imported (from_numpy, linalg.eigh, ...); do NOT import xpyq.
"""
from __future__ import annotations

import json
import os
import re
import time

from core import qubo
from core.pipeline import solve_scenario

from .base import QUBOProblem, SolveResult

XPYQ_BASE = os.environ.get("XPYQ_BASE", "https://xpyq-lib-production.up.railway.app")


def _api_key():
    return os.environ.get("XPYQ_KEY") or os.environ.get("XPYQ_API_KEY")


def _symmetric_matrix(prob):
    """Dense symmetric M with x^T M x == qubo.energy form (diag = linear+penalty,
    off-diagonal split in half so the quadratic form matches)."""
    Q, _, n = qubo.to_qubo_matrix(prob)
    M = [[0.0] * n for _ in range(n)]
    for (p, q), w in Q.items():
        if p == q:
            M[p][p] += w
        else:
            M[p][q] += w / 2.0
            M[q][p] += w / 2.0
    return M, n


def _xpyq_run(code, api_key, timeout_s=25):
    """Submit code, poll until terminal, return the result dict. Raises on failure."""
    import requests

    h = {"Authorization": f"Bearer {api_key}"}
    run = requests.post(f"{XPYQ_BASE}/api/v1/compute/runs", headers=h,
                        json={"code": code, "name": "dispatch-evac-relax"},
                        timeout=15)
    run.raise_for_status()
    run_id = run.json()["run_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(f"{XPYQ_BASE}/api/v1/compute/runs/{run_id}",
                         headers=h, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") in ("completed", "failed", "timed_out", "cancelled"):
            return data
        time.sleep(0.4)
    raise TimeoutError("XpyQ run did not reach a terminal state in time")


def _relax_code(M):
    """Python the XpyQ runtime executes: eigendecompose M on the boards, then print the
    eigenvector of the SMALLEST eigenvalue. Verified runtime quirks (2026-05-30):
      * runtime has numpy + from_numpy + linalg pre-imported (do NOT import xpyq),
      * linalg.eigh returns (w, v) Matrices; .numpy() yields the full (evals, evecs) tuple,
      * eigenvalues are NOT sorted ascending — pick argmin explicitly."""
    return (
        "import numpy as np\n"
        f"A = from_numpy(np.array({M!r}, dtype=float))\n"
        "w, v = linalg.eigh(A)\n"
        "t = v.numpy()\n"
        "evals = np.asarray(t[0]); evecs = np.asarray(t[1])\n"
        "imin = int(np.argmin(evals))\n"
        "print('EVEC', [float(x) for x in evecs[:, imin]])\n"
    )


def _round_eigenvector(vec, prob):
    """Capacity-aware rounding of the XpyQ relaxation eigenvector into a shelter
    assignment. The eigenvector gives each town a preference score per shelter; we
    place the most-decisive towns first into their best shelter that still has room.
    Eigenvector sign is arbitrary, so try both and keep the lower-energy result."""
    S, T = prob.S, prob.T
    best = None
    for sign in (1.0, -1.0):
        scores = [[sign * vec[i * S + j] for j in range(S)] for i in range(T)]
        rem = list(prob.cap)
        choice = [0] * T
        # assign decisive towns (biggest score spread) first
        order = sorted(range(T), key=lambda i: -(max(scores[i]) - min(scores[i])))
        for i in order:
            ranked = sorted(range(S), key=lambda j: -scores[i][j])
            placed = next((j for j in ranked if rem[j] >= prob.pop[i]), None)
            choice[i] = placed if placed is not None else ranked[0]
            rem[choice[i]] -= prob.pop[i]
        e = qubo.energy(prob, choice)
        if best is None or e < best[1]:
            best = (choice, e)
    return best


def xpyq_assign(prob):
    api_key = _api_key()
    if not api_key:
        choice, e = qubo.simulated_anneal(prob, iters=4000, seed=0)
        return choice, {"method": "annealer (XpyQ stand-in — set XPYQ_KEY for live)",
                        "energy": round(e, 1)}
    try:
        M, n = _symmetric_matrix(prob)
        res = _xpyq_run(_relax_code(M), api_key)
        if res.get("status") != "completed":
            raise RuntimeError(f"XpyQ status={res.get('status')}: {str(res.get('stdout'))[:80]}")
        m = re.search(r"EVEC\s*(\[.*\])", res.get("stdout", ""))
        if not m:
            raise RuntimeError(f"no eigenvector in stdout: {str(res.get('stdout'))[:80]}")
        vec = json.loads(m.group(1))
        xq_choice, xq_e = _round_eigenvector(vec, prob)

        ann_choice, ann_e = qubo.simulated_anneal(prob, iters=4000, seed=0)
        used_xpyq = qubo.feasible(prob, xq_choice)
        choice = xq_choice if used_xpyq else ann_choice
        return choice, {
            "method": "XpyQ matrix-compute · linalg.eigh spectral relaxation (real boards)",
            "xpyq_credits": res.get("credits_charged"),
            "xpyq_ms": res.get("duration_ms"),
            "xpyq_boards": res.get("boards_used"),
            "xpyq_qubits": n,
            "xpyq_relax_energy": round(xq_e, 1),
            "xpyq_drove_plan": used_xpyq,
            "classical_energy": round(ann_e, 1),
        }
    except Exception as exc:
        choice, e = qubo.simulated_anneal(prob, iters=4000, seed=0)
        return choice, {"method": "annealer (XpyQ call failed)",
                        "xpyq_error": f"{type(exc).__name__}: {str(exc)[:100]}"}


class XpyQSolver:
    name = "xpyq"

    def __init__(self, roadnet):
        self.roadnet = roadnet

    def available(self) -> bool:
        return bool(_api_key())

    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult:
        return solve_scenario(self.roadnet, problem.metadata, xpyq_assign,
                              backend_name="xpyq")
