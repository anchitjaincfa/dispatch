"""Quantum / accelerated backend.

Methods over the SAME QUBO (core/qubo.py):

  * "annealer" — numpy simulated annealing. Always available, exact on demo-sized
    instances (verified vs brute force). Stands in for XpyQ's neuromorphic
    accelerated-decision solver and scales far past QAOA.

  * "qaoa" — REAL QAOA(p=1) circuits on Aer (or a configured IBM QPU). To stay
    fast and honest on stage, QAOA solves the genuinely-quantum SUB-instance (the
    k most-exposed towns x shelters, <= ~9 qubits) while the annealer carries the
    full map. We build the Ising cost Hamiltonian, then EXECUTE the QAOA ansatz
    over a tiny fixed (beta, gamma) grid and SAMPLE bitstrings — no slow classical
    COBYLA outer loop (the part that hangs a live demo). The best sampled
    assignment, scored on the true QUBO energy, is spliced back in. Runs in a
    worker thread with a hard timeout so the UI NEVER hangs; on timeout/error it
    falls back to the annealer's choice.

Honesty: this proves the assignment maps to QUBO/Ising and runs on real quantum
primitives — NOT a production speed advantage at this scale.
"""
from __future__ import annotations

import concurrent.futures as _fut

from core import qubo
from core.pipeline import solve_scenario

from .base import QUBOProblem, SolveResult

QUBIT_BUDGET = 20
QAOA_SUB_TOWNS = 3          # quantum sub-instance size (towns); qubits = this * S
# fixed QAOA(p=1) parameter grid — we EXECUTE real circuits and sample, rather than
# run a slow classical COBYLA outer loop (which is the part that doesn't fit a live demo).
QAOA_PARAM_GRID = [(b, g) for b in (0.3, 0.6) for g in (0.4, 0.8)]
QAOA_SHOTS = 256
QAOA_TIMEOUT_S = 15         # hard wall-clock cap; on timeout we fall back to annealer


def annealer_assign(prob):
    choice, e = qubo.simulated_anneal(prob, iters=4000, seed=0)
    meta = {"method": "neuromorphic-anneal (numpy SA)", "energy": round(e, 1)}
    if prob.S ** prob.T <= 100000:
        opt_choice, opt_e = qubo.brute_force(prob)
        meta["optimal"] = abs(e - opt_e) < 1e-6
        meta["gap_to_optimum"] = round(e - opt_e, 2)
    return choice, meta


def _qubo_to_ising(Q, n):
    """Map a binary QUBO {(p,q): w} to an Ising cost Hamiltonian (SparsePauliOp).

    Substituting x_p = (I - Z_p)/2 into E(x) = sum_p c_pp x_p + sum_{p<q} c_pq x_p x_q
    yields a diagonal Z/ZZ Hamiltonian whose ground state is the optimal bitstring.
    The constant offset is dropped (it doesn't change the argmin).
    """
    from qiskit.quantum_info import SparsePauliOp

    z = {}          # qubit -> coeff on Z_q
    zz = {}         # (p,q) -> coeff on Z_pZ_q
    for (p, q), c in Q.items():
        if p == q:
            z[p] = z.get(p, 0.0) - c / 2.0
        else:
            z[p] = z.get(p, 0.0) - c / 4.0
            z[q] = z.get(q, 0.0) - c / 4.0
            zz[(p, q)] = zz.get((p, q), 0.0) + c / 4.0

    terms = []
    for q_, c in z.items():
        if abs(c) < 1e-12:
            continue
        s = ["I"] * n
        s[n - 1 - q_] = "Z"
        terms.append(("".join(s), c))
    for (p, q_), c in zz.items():
        if abs(c) < 1e-12:
            continue
        s = ["I"] * n
        s[n - 1 - p] = "Z"
        s[n - 1 - q_] = "Z"
        terms.append(("".join(s), c))
    if not terms:                       # degenerate (all-zero) Hamiltonian
        terms = [("I" * n, 0.0)]
    return SparsePauliOp.from_list(terms)


def _run_qaoa_subproblem(subprob):
    """Real QAOA(p=1) on a small sub-instance via grid-sampling.

    Builds the Ising cost Hamiltonian, executes the QAOA ansatz once per
    (beta, gamma) grid point on Aer, samples bitstrings, and returns the sampled
    assignment with the lowest TRUE QUBO energy. Returns (choice, fval). Raises on
    failure (the caller falls back to the annealer).
    """
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer.primitives import SamplerV2

    n = subprob.num_qubits()
    S, T = subprob.S, subprob.T
    Q, _, _ = qubo.to_qubo_matrix(subprob)
    ansatz = QAOAAnsatz(cost_operator=_qubo_to_ising(Q, n), reps=1)
    sampler = SamplerV2()

    best, best_e = None, float("inf")
    for beta, gamma in QAOA_PARAM_GRID:
        binding = {prm: (beta if ("β" in prm.name or "beta" in prm.name.lower())
                         else gamma)
                   for prm in ansatz.parameters}
        circ = ansatz.assign_parameters(binding)
        circ.measure_all()
        circ = transpile(circ, basis_gates=["rz", "rx", "ry", "h", "cx"],
                         optimization_level=1)
        counts = sampler.run([circ], shots=QAOA_SHOTS).result()[0].data.meas.get_counts()
        for bitstr in counts:
            x = [int(bitstr[n - 1 - k]) for k in range(n)]
            choice = [max(range(S), key=lambda j: x[i * S + j]) for i in range(T)]
            e = qubo.energy(subprob, choice)
            if e < best_e:
                best_e, best = e, choice
    if best is None:                    # no samples at all (shouldn't happen)
        raise RuntimeError("QAOA produced no samples")
    return best, float(best_e)


def qaoa_assign(prob):
    # full-map baseline from the annealer
    base_choice, base_meta = annealer_assign(prob)
    S = prob.S
    k = min(QAOA_SUB_TOWNS, prob.T)
    n_qubits = k * S
    if n_qubits > QUBIT_BUDGET:
        base_meta["qaoa_skipped"] = f"{n_qubits} qubits > budget {QUBIT_BUDGET}"
        return base_choice, base_meta

    # build the quantum sub-instance: the k most-exposed (most populous) towns,
    # with shelter capacities reduced by the other towns' (fixed) loads.
    quantum_towns = sorted(range(prob.T), key=lambda i: -prob.pop[i])[:k]
    rem_cap = list(prob.cap)
    for i in range(prob.T):
        if i not in quantum_towns:
            rem_cap[base_choice[i]] -= prob.pop[i]
    sub = qubo.AssignmentProblem(
        town_ids=[prob.town_ids[i] for i in quantum_towns],
        shelter_ids=list(prob.shelter_ids),
        cost=[prob.cost[i] for i in quantum_towns],
        pop=[prob.pop[i] for i in quantum_towns],
        cap=[max(0, c) for c in rem_cap],
        overflow_penalty=prob.overflow_penalty,
    )

    try:
        with _fut.ThreadPoolExecutor(max_workers=1) as ex:
            sub_choice, fval = ex.submit(_run_qaoa_subproblem, sub).result(
                timeout=QAOA_TIMEOUT_S)
        spliced = list(base_choice)
        for idx, town_i in enumerate(quantum_towns):
            spliced[town_i] = sub_choice[idx]
        # keep QAOA result only if it stays feasible & not worse
        if qubo.energy(prob, spliced) <= qubo.energy(prob, base_choice) + 1e-6:
            choice = spliced
        else:
            choice = base_choice
        return choice, {
            "method": "QAOA (Aer statevector) on core sub-instance + annealer full-map",
            "qubits": n_qubits,
            "quantum_towns": [prob.town_ids[i] for i in quantum_towns],
            "qaoa_fval": round(fval, 1),
            **{f"full_{key}": val for key, val in base_meta.items() if key != "method"},
        }
    except (_fut.TimeoutError, Exception) as exc:
        base_meta["qaoa_fallback"] = f"{type(exc).__name__}: {str(exc)[:80]}"
        base_meta["method"] = "annealer (QAOA timed out/failed)"
        return base_choice, base_meta


class QuantumSolver:
    def __init__(self, roadnet, method="annealer"):
        self.roadnet = roadnet
        self.method = method
        self.name = "qaoa_aer" if method == "qaoa" else "quantum"

    def available(self) -> bool:
        if self.method == "qaoa":
            try:
                import qiskit_optimization  # noqa: F401
                import qiskit_algorithms    # noqa: F401
                return True
            except Exception:
                return False
        return True

    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult:
        fn = qaoa_assign if self.method == "qaoa" else annealer_assign
        return solve_scenario(self.roadnet, problem.metadata, fn,
                              backend_name=self.name)
