# 🔥 DISPATCH — The Map That Saves Lives

**Live, agentic wildfire disaster response.** Describe a crisis in plain English; an
agent places it on a real road map, routes the fire crews, and evacuates the towns to
shelters along real streets. Then move the fire — and the **entire plan re-solves live,
in ~140 ms**, while a dispatcher-voice agent narrates every decision.

Built for the **XpyQ Quantum AI Hackathon 1.0** (Berkeley). Tracks: _Applied AI &
Agentic Systems × Routing/Scheduling._

> **The thesis:** in a wildfire, the first hour decides who lives — and the binding
> constraint isn't compute power, it's **how fast the plan can re-solve when reality
> changes**. DISPATCH makes the plan keep up with the fire.

---

## What it does

- **Plain-English crisis in.** An operator brief ("wildfire near Hwy 24, wind pushing
  west toward Forest Park, 5 crews, 3 shelters") is parsed by a Claude agent into a
  structured incident grounded to map entities (deterministic fallback if no API key).
- **Two coupled optimizations on a real road graph:**
  1. **Crew routing** — assign/route fire crews from stations to the most-threatened
     defensible points.
  2. **Evacuation** — capacitated assignment of town populations to shelters
     (a Generalized Assignment Problem — NP-hard), plus hazard-aware paths that bend
     around the fire.
- **Live re-solve.** Advance the fire, close a road, or lose a crew, and crew + evac
  routes re-solve on the blocked graph in ~140 ms; the dispatcher narrates the delta.
- **One problem, four backends.** Every backend solves the **same** town→shelter QUBO,
  so _"do the backends agree?"_ is a real correctness test, not marketing.

## The honest quantum story

The combinatorial core is mapped to a **QUBO / Ising Hamiltonian** and solved four ways
behind one `Solver` interface:

| Backend | Method | Role |
|---|---|---|
| `classical` | OR-Tools CP-SAT (exact) | Always-on baseline + the live re-solve path |
| `accelerated` | numpy simulated annealing (swap-move) | Stand-in for XpyQ's **neuromorphic** optimizer; scales past QAOA, hits the exact optimum at demo size |
| `qaoa` | **real QAOA(p=1) circuits on Aer**, 9 qubits | Solves the genuinely-quantum sub-instance; grid-samples bitstrings, **hits the exact optimum in ~0.5 s** |
| `xpyq` | POST to XpyQ `/decisions` (env-driven) | Same objective on XpyQ's accelerator; falls back to the annealer without a key |

**We do not claim quantum speed advantage.** At demo scale OR-Tools wins, and we say so
on stage. What we claim is defensible: the assignment maps cleanly to a QUBO, it runs on
real quantum primitives and recovers the optimum, and **re-solve latency — not solve
quality — is the binding product constraint.** Quantum is _never_ on the live re-solve
path; it runs only on an explicit "solve this snapshot." We always print the qubit count.

The scaling artifact (`python -m core.bench`) shows this honestly: exact enumeration
blows up (0.3 → 270 → ~4,900 ms → intractable) while the heuristic core stays flat and
**optimal** (gap 0.0), motivating accelerated hardware _upstream_ of the live loop.

## Architecture

```
UI / map  ──►  agent  ──►  optimization core  ──►  swappable solver backends  ──►  data
(Streamlit    (Claude     (QUBO build + route       (classical / annealer /        (cached OSM
 + pydeck)     parse/      reconstruction)           QAOA+Aer / XpyQ)               road graph)
               narrate)
```

The **solver interface is the key design decision** (`solver/base.py`): the UI and core
never import a concrete backend, so you can swap mock → real → quantum with zero UI
changes.

```
dispatch/
  app.py                 Streamlit ops-console UI
  core/   geo · hazard · routing · evacuate · qubo · pipeline · bench
  solver/ base · classical_solver · quantum_solver · xpyq_solver · mock_solver
  agent/  parse · narrate            data/  scenario_berkeley.json · region_graph.json
  tests/  test_invariants · smoke    demo/  script.md
```

## Run it

```bash
cd dispatch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py            # opens the ops console on a free Carto basemap (no token)
```

Optional environment:
- `ANTHROPIC_API_KEY` — live Claude crisis-parsing + dispatcher narration (deterministic
  fallback otherwise).
- `XPYQ_API_KEY` (+ optional `XPYQ_ENDPOINT`) — live XpyQ `/decisions` backend.

## Tests

```bash
pytest -q                 # 8 invariants: feasibility, capacity, backends-agree,
                          # qubit budget (<=20), live re-solve <500ms, annealer optimality
python tests/smoke.py     # end-to-end across all four backends + perturbations
python -m core.bench      # the honest scaling artifact
```

**Status:** 8/8 invariants pass, `smoke.py` ALL CHECKS PASSED, all four backends agree
(objective ≈ 75,668 person-minutes, est. clearance 14.2 min) on the default scenario.

## The scenario

**Oakland–Berkeley Hills, CA — the 1991 Tunnel Fire footprint.** 5 towns (9,000
residents), 3 shelters, 5 fire crews, on the real OpenStreetMap road network (cached
offline; no live calls on the demo path). Local resonance for a Berkeley-judged room.

## Honesty guardrails

- Always state the qubit count when showing QAOA.
- Never imply quantum is faster than OR-Tools at this scale — say the opposite first.
- The framing is _"re-solve latency is the product constraint,"_ never "quantum advantage."
- Every dependency degrades gracefully (OR-Tools → greedy, qiskit → annealer, Claude →
  templates, XpyQ → annealer); nothing live sits on the critical path.
