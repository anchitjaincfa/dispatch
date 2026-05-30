# DISPATCH — Claude Code Handoff & Build Brief

> Paste this file into a fresh Claude Code session at the root of an empty
> project directory (save it as `CLAUDE.md` so it's auto-loaded). It is the
> complete, self-contained context for building DISPATCH. Assume no prior
> knowledge.

---

## ⛳ FIRST THING — before any coding

**Step 1: Check my email for XPYQ hackathon details.**
I'm at the XPYQ "Quantum AI Hackathon 1.0" right now. Before building, pull the
latest hackathon info from my Gmail (`anchit.jain@berkeley.edu`) so your build
matches the real rules and platform. Search for: `XPYQ`, `XpyQ Hackathon`,
`Quantum AI Hackathon`, `Devpost`. Specifically confirm:

- **Submission requirements & deadline** (Devpost): https://xpyq-hackathon.devpost.com/
- **Judging criteria** (see §6 below for my current understanding — verify it).
- **What XpyQ Cloud actually exposes as a solver primitive** — read the docs at
  https://docs.xpyq.io/ and the platform at https://xpyq.vercel.app/. THIS IS THE
  SINGLE MOST IMPORTANT UNKNOWN (see §5). Does it expose a QUBO/Ising/annealing
  endpoint, or fast-MILP, or something else? The answer reshapes the quantum core.
- Any time limits, team rules, or required tech.

Write what you find into a `HACKATHON_NOTES.md` at repo root, then continue.

**Known details already confirmed from my inbox (2026-05-30):**
- Event: QUANTUM AI HACKATHON 1.0 — "AI agents meet qubits, powered by XpyQ."
- When/where: May 30, 9:00 AM PDT, 2168 Shattuck Ave, FL 2, Berkeley, CA.
  Runs into May 31 (~12:30 PM checkpoint per calendar). Treat it as a ~1-day hack.
- Platform: https://xpyq.vercel.app/
- XpyQ Cloud docs: https://docs.xpyq.io/
- IBM Quantum (credits available): https://quantum.ibm.com/
- Submit on Devpost: https://xpyq-hackathon.devpost.com/
- Slack workspace: XpyQ Hackathon 1.0.

**Step 2: Build the UI FIRST** (front-end-first). Details in §7. Do not start
with the optimizer — start with a clickable UI shell on mock data so we can see
and demo the product early.

---

## 1. What we're building

**DISPATCH** — a map-based AI agent for live wildfire disaster response.
You describe a crisis in plain English ("wildfire spreading toward these towns,
4 crews, 3 shelters"); an agent parses it, places the scene on a map, and an
optimizer routes fire crews and plans evacuations. The hero moment: the user
**drags the fire across the map** and every crew route + evacuation route
**re-solves live** while the agent narrates each decision in dispatcher language.

Tracks (remix): 01 Agentic optimization × 03 Routing & scheduling.

**Two coupled optimizations on a road graph:**
1. Crew routing — assign/route crews from stations to threatened defensible
   points (a VRP variant; the NP-hard core, intended quantum/XpyQ target).
2. Evacuation — capacitated assignment of town populations to shelters +
   hazard-aware paths that avoid the fire.

The fire perturbs the graph: blocks nearby roads, raises nearby town priority.
Every perturbation triggers a re-solve.

---

## 2. Build order (IMPORTANT — UI first)

1. **Phase 0 — env & email check.** Do the email check above. Set up the repo
   (§9), pin deps, verify imports, cache a road graph. Kick off XpyQ docs review.
2. **Phase F — FRONT END FIRST.** Build the full UI shell against a `MockSolver`
   and hardcoded mock scenario, so the product looks/feels finished before any
   real optimizer exists. This is the priority. (§7)
3. **Phase 1 — real graph & entities.** Swap mock coords for a real cached OSMnx
   graph + real towns/stations/shelters. Same UI.
4. **Phase 2 — real classical solve (OR-Tools) + self-healing.** The insurance
   build: a working self-healing classical map. Protect it.
5. **Phase 3 — agent.** Claude NL crisis parsing + dispatcher narration.
6. **Phase 4 — quantum/XpyQ core** behind the solver interface (per §5 finding).
7. **Phase 5 — polish + scaling story. Phase 6 — demo prep + recorded backup.**

Stop at each phase, show a working artifact, and commit before proceeding.

---

## 3. Architecture

Strict separation: UI ↔ agent ↔ optimization core ↔ swappable solver backends ↔
data. The solver abstraction is the key design decision — it's what lets the UI
be built first (talks to `MockSolver`) and lets quantum slot in later without UI
changes.

| Layer | Responsibility | Tech |
|---|---|---|
| UI / map | render graph, fire, routes; capture interactions; trigger re-solves | Streamlit + pydeck/deck.gl |
| Agent | NL parsing, narration, tool-calling | Anthropic Claude |
| Core | build VRP & assignment models; QUBO; route reconstruction | OR-Tools, NetworkX, Qiskit Optimization |
| Solver backends | solve the combinatorial core behind one interface | Mock / OR-Tools / QAOA+Aer / XpyQ |
| Data | cached road graph, towns, stations, shelters | OSMnx, GeoPandas, .graphml |

**Re-solve loop:** user action → perturb graph (block edges, raise priorities)
→ re-solve via selected backend (default OR-Tools, warm-started, <~100 ms) →
reconstruct routes classically → re-render → agent narrates the delta.

**Latency rule:** quantum is NEVER on the live re-solve path. Live loop = warm
OR-Tools. Quantum/XpyQ runs only on an explicit "solve this snapshot" button.

---

## 4. The swappable solver interface (build this contract first)

All backends implement the same `Solver` protocol. The UI/core never import a
concrete solver directly.

```python
# solver/base.py
from dataclasses import dataclass, field
from typing import Protocol

@dataclass
class QUBOProblem:
    Q: dict = field(default_factory=dict)
    num_vars: int = 0
    var_labels: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)   # the scenario lives here

@dataclass
class SolveResult:
    crew_routes: dict = field(default_factory=dict)      # crew_id -> [[lng,lat],...]
    evac_assignment: dict = field(default_factory=dict)  # town_id -> shelter_id
    evac_routes: dict = field(default_factory=dict)      # town_id -> [[lng,lat],...]
    objective: float = 0.0
    backend: str = "mock"
    wall_ms: float = 0.0
    feasible: bool = True
    extra: dict = field(default_factory=dict)

class Solver(Protocol):
    name: str
    def solve(self, problem: QUBOProblem, time_budget_ms: int = 1000) -> SolveResult: ...
    def available(self) -> bool: ...
```

Backends: `MockSolver` (build first — canned reactive heuristic), `ORToolsSolver`
(mandatory baseline + live path), `QAOASolver` (Aer, tiny instance), `XpyQSolver`
(per §5).

---

## 5. Hour-0 critical unknown: what does XpyQ Cloud expose?

Read https://docs.xpyq.io/ during Phase 0 and decide:

- **QUBO/Ising/annealing endpoint** → make XpyQ the headline backend; demote
  QAOA/Aer to "mapping validated on gate model."
- **Fast-MILP / classical accelerator** → skip the QUBO detour for the main
  story; formulate MILP, run on XpyQ, frame as "accelerator enables live
  re-solve."
- **Unknown/unreachable** → ship OR-Tools + QAOA/Aer; keep XpyQ wired behind the
  interface, ready.

Record the decision in `HACKATHON_NOTES.md`. Don't block the classical build on it.

**QUBO sizing reality:** QAOA needs 1 qubit/binary; Aer simulation caps ~25–30
qubits. Keep any quantum demo instance ≤ ~20 qubits (e.g. 4–5 town × 3 shelter
assignment). Print the exact qubit count. The quantum backend proves the mapping
+ the latency thesis — NOT a speed advantage over OR-Tools. Stay honest about this.

---

## 6. Judging criteria (my current understanding — VERIFY against Devpost)

Five "ways to win": (1) Technical ambition, (2) Real-world usefulness, (3)
Creative use of XpyQ — "did the weird hardware earn its place," (4) Demo quality
— "make us feel it in 60s," (5) Frontier weirdness — "ship something that
shouldn't have worked." Confirm exact criteria/weights on Devpost.

The honest risk: criterion #3. OR-Tools is fast at demo scale, so quantum can
look decorative. Mitigate by (a) confirming XpyQ's primitive and using it for
real, (b) framing the win as "re-solve latency is the binding product
constraint," and (c) never implying quantum advantage you can't defend.

---

## 7. Phase F — the UI to build first (front-end-first spec)

Build a single Streamlit + pydeck app (`app.py`) that talks ONLY to `MockSolver`
and a hardcoded `data/mock_scenario.json`. Goal: a clickable product that LOOKS
finished before any real optimizer exists.

**Layout:**
- Top bar: "DISPATCH" mark; status pill "SOLVER ONLINE · backend: mock"; a
  backend selector (mock / OR-Tools / QAOA / XpyQ — only mock wired).
- Main map (~70%): stylized/real map with fire, stations, shelters, towns,
  defensible points, crew routes, evac routes. Floating hint "click to move the
  fire — routes re-solve."
- Right rail (~30%): "Dispatcher" panel (3 metric chips: towns at risk / crews
  routed / capacity; plus a narration log) and an "Evacuation plan" list
  (Town → Shelter rows; red when a shelter is over capacity). A decorative NL
  crisis-input box labeled "parsed by the agent (Phase 3)."

**Interactions (work against the mock):**
- Move the fire (click-to-place; stepwise drag — NOT continuous, to stay smooth
  in Streamlit) → re-call MockSolver → redraw all routes, update chips, append a
  dispatcher narration line.
- Buttons: "Advance fire toward towns" (step + grow), "Close a road", "Lose
  nearest crew", "Reset".
- Routes should visibly bend around the fire.

**Visual tokens (dark "wildfire ops console"):**
- bg `#0a0f0d`; panels `#0f1714`, borders `#1d2f28`; map `#0d1411`, grid
  `#15241d`, roads `#26413a`.
- fire `#ff6a18` core / `#ffd24a` hot center → transparent; crews/stations
  `#3b82f6`; evac routes `#f59e0b` (dashed); towns `#eab308`; shelters `#22c55e`.
- text `#cfe3da`; muted `#5e7a70`; danger `#ff7a5c`; online dot `#4ade80` (pulse).
- UI font: Space Grotesk / system sans. Data/log font: Space Mono / monospace.
- Rounded panels (~10px), 1px borders, fire-only glow.

**Basemap = FREE, no token.** Use pydeck's default Carto basemap (`map_provider="carto"`,
`map_style="dark"`). Do NOT require a Mapbox token.

---

## 8. Starter code is provided

I have a working Phase F starter (separate `dispatch_starter/` folder I'll drop
in). It already contains: `solver/base.py`, `solver/mock_solver.py`,
`solver/__init__.py`, `data/mock_scenario.json`, `app.py`, `requirements.txt`,
`README.md`. It runs with `streamlit run app.py`. Start from it, verify it runs,
then extend per §2. If it's not present, build Phase F from §7.

---

## 9. Repo structure

```
dispatch/
  app.py                  # Streamlit entry + map UI
  CLAUDE.md               # this file
  HACKATHON_NOTES.md      # your email + docs findings
  README.md  requirements.txt
  data/  region.graphml  towns.csv  shelters.csv  stations.csv  mock_scenario.json
  core/  scenario.py  routing.py  evacuation.py  qubo.py
  solver/  base.py  mock_solver.py  ortools_solver.py  qaoa_solver.py  xpyq_solver.py
  agent/  parse.py  narrate.py
  viz/  layers.py  scaling.py
  tests/  test_solvers_agree.py  test_perturbation.py  test_parse.py
  demo/  script.md  seed_scenario.json  backup_demo.mp4
```

---

## 10. Testing (a phase isn't done until these pass)

- Backends agree on the tiny instance (OR-Tools vs QAOA, or stated gap).
- Perturbation safety: no route ever traverses a blocked/fire edge.
- Capacity respected: no shelter over capacity; every town assigned.
- Latency: live OR-Tools re-solve < ~100 ms on the demo instance.
- NL parse validates against a JSON schema.
- Quantum sizing: assert qubit count ≤ Aer budget before solving.

---

## 11. Demo & honesty guardrails

60s demo: type crisis → map populates → move fire (heal #1) → close road / lose
crew (heal #2) → "solve snapshot on XpyQ/quantum" with honest numbers → scaling
view ("latency is the product"). Always state the qubit count; never claim
quantum speed advantage; record a backup video.

---

## 12. Working rules

- Do the email check FIRST; write `HACKATHON_NOTES.md`.
- Build the UI FIRST (Phase F) against MockSolver; reuse that exact UI everywhere.
- Never break the classical path; commit at every green checkpoint.
- Small testable increments; print real timings/qubit counts/feasibility.
- If short on time, ship the latest green checkpoint and adjust the demo script.
