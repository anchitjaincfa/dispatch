# DISPATCH — Demo Script

**Event:** Quantum AI Hackathon 1.0 (XpyQ), Berkeley. Sun final = **10-min demo + 5-min Q&A**.
**One-liner:** _"Describe a wildfire in plain English. An agent puts it on the real
map, routes the fire crews, and evacuates 9,000 people to shelters — then you drag
the fire and the entire plan re-solves live, in ~140 ms, while the dispatcher
narrates. The same assignment runs on OR-Tools, an annealer, and real QAOA qubits."_

**The thesis the whole demo defends:** _re-solve latency is the binding product
constraint — not a production quantum speed advantage._ We never overclaim.

---

## 0 · Pre-flight (do this BEFORE you're on stage)

```bash
cd ~/dispatch
source .venv/bin/activate
export ANTHROPIC_API_KEY=...        # optional: live Claude narration; app degrades gracefully without it
python tests/smoke.py               # must print "ALL CHECKS PASSED" — your go/no-go
streamlit run app.py                # leave it running; browser full-screen, zoom so the right rail is readable
```

Checklist:
- [ ] Browser full-screen on the map; sidebar + right rail both visible.
- [ ] Backend selector reads **`classical (OR-Tools)`** (top of the map: `SOLVER ONLINE · backend: ortools · ~140 ms · 15 qubits`).
- [ ] `python -c "from core import bench"` works (you'll run the scaling artifact at the end).
- [ ] **Backup video** queued in a second tab (see §6). If Streamlit wedges, cut to it without apology.
- [ ] Have `data/scenario_berkeley.json` facts in your head: **5 towns / 9,000 residents, 3 shelters, 5 crews**, Oakland–Berkeley Hills (1991 Tunnel Fire footprint).

---

## 1 · Run-of-show (target ~6 min, leaves slack in the 10-min slot)

### Beat 1 — Cold open (0:00–0:45) · "this is a real place"
- Map already loaded. Gesture at it: _"This is the Oakland–Berkeley Hills — the exact
  footprint of the 1991 Tunnel Fire that killed 25 people. Real OpenStreetMap roads,
  no API token. Five towns, nine thousand residents. Three shelters. Five fire crews."_
- Point to the legend: 🔵 crew routes · 🟠 evac routes · 🔴 fire · 🟢 shelters · 🟡 towns.
- _"Everything you'll see routes on those real streets."_

### Beat 2 — The agent parses a crisis (0:45–1:30) · "plain English in"
- In the sidebar **Incident** box the operator brief is pre-filled. Read it aloud:
  _"Wildfire ignited in the Oakland Hills near Hwy 24, wind pushing it west toward
  Forest Park and Claremont. 5 crews staged, 3 shelters online."_
- Click **`⚙ Parse with agent`**.
- Point at the caption that appears: _"The agent pulled the structure out — crews,
  shelters, towns at risk — and grounded it to map entities."_ (Claude if the key is
  set; a deterministic parser otherwise — **say so**, it's part of the honesty story.)
- Right rail: read the three chips — **Towns at risk · Crews routed · Capacity: OK**.
  _"Plan's already solved and feasible: every town has a shelter, no shelter over capacity."_

### Beat 3 — THE HERO LOOP: drag the fire, watch it heal (1:30–2:45) · 75 seconds, the moment
This is the demo. Narrate the **wall-time** out loud each time — that number is the point.
1. Click **`▸▸ Advance fire`** once. The fire grows toward the towns; threatened towns
   redden. Dispatcher log prints: _"Wind shift — fire advanced. Re-solved in ~140 ms.
   Rerouted …"_ → say: **"That whole re-solve — crew routes, evacuation routes, shelter
   assignment — happened in about a seventh of a second."**
2. Use the arrow nudges **◀ ▲ ▶ ▼** to walk the fire two steps **west toward Forest Park /
   Claremont**. Each click re-solves and the orange evac routes visibly **bend around the
   fire**. _"Routes are avoiding the burn — these aren't redrawn lines, the graph is
   re-solved on blocked roads every step."_
3. Click **`⊘ Close a road`**. Log: _"Road closed. Crews and evacuees re-routing around
   the closure (~140 ms)."_ → _"A road just failed. It heals — same latency."_
4. Click **`✕ Lose crew`** (nearest crew to the fire goes grey). Log: _"Crew lost contact.
   Reassigned coverage across remaining crews."_ → _"We just lost the closest crew. Still
   feasible. Still ~140 ms."_
- Land the line: **"This is the product. The binding constraint in disaster response
  isn't compute power — it's how fast the plan can re-solve when reality changes."**

### Beat 4 — The honest quantum reveal (2:45–4:15) · "the qubits earn their place"
- Switch the backend selector to **`qaoa (Aer qubits)`**. Top-of-map pill updates to
  `backend: qaoa_aer · ~0.5 s · 9 qubits`.
- _"Same problem. Now the genuinely-quantum core — the three most-exposed towns against
  three shelters — is solved by **real QAOA circuits on nine qubits**, executed on Aer,
  sampling bitstrings. It runs in about half a second and **hits the exact optimum.**"_
- Open the right-rail expander **`backend detail / honesty`** and read it literally:
  _"Quantum and accelerated backends solve the SAME town→shelter QUBO as OR-Tools."_
- **Say the honest line out loud — do not let a judge say it first:** _"At this scale
  OR-Tools wins on speed. We are NOT claiming quantum advantage. What we're claiming is
  the mapping is real, it runs on real quantum primitives, and all our backends agree."_
- Flip through **`classical`**, **`accelerated (annealer)`**, **`qaoa`** — point out the
  evacuation plan and the objective are the **same** (`obj ≈ 75,668 person-min, clearance
  14.2 min`). _"Four backends, one answer. That's why 'do the backends agree?' is a real
  test for us, not marketing."_
- Name XpyQ: _"`xpyq (/decisions)` is wired to XpyQ's neuromorphic optimization backend
  behind the same interface — env-driven, drop-in. That's the post-GPU hardware this
  hackathon is about, and our QUBO is exactly its target workload."_

### Beat 5 — The scaling artifact (4:15–5:30) · what the rubric explicitly asked for
- Cut to a terminal: `python -m core.bench`. It prints a row per instance size
  (`vars`, `anneal_ms`, `cpsat_ms`, `brute_ms`, `anneal_gap`).
- Read the **three honest curves** off the table — don't overclaim, the numbers carry it:
  - **`brute_ms` (exact enumeration) blows up:** ~0.3 ms at 12 vars → **270 ms at 32 →
    ~4,900 ms at 40 → intractable beyond.** _"This is the NP-hard core, not a slogan —
    truly-exact enumeration explodes."_
  - **`cpsat_ms` (OR-Tools) stays fast** (~4–11 ms) across these sizes. **Be honest:**
    _"OR-Tools is a strong classical baseline and it does NOT fall over at demo scale —
    that's exactly why our quantum claim is about mapping and latency, not beating it here."_
  - **`anneal_gap = 0.0`** wherever brute force is still tractable. _"Our annealer — the
    stand-in for XpyQ's neuromorphic backend — stays flat (~30–44 ms) and hits the exact
    optimum every time. We report the gap precisely so nothing is hand-waved."_
- Land it: _"The artifact shows where exact methods die, that our heuristic core stays
  optimal as it scales, and why accelerated hardware belongs **upstream** of the ~140 ms
  live loop — never on it."_

### Beat 6 — Close (5:30–6:00)
- _"DISPATCH: plain-English crisis in, a live-healing evacuation map out. One QUBO,
  four backends that agree, real qubits in the loop, and an honest scaling story. The
  map that saves lives — and the latency that makes it usable when seconds matter."_

---

## 2 · The numbers to have memorized (all real, from this build)
| Fact | Value |
|---|---|
| Scene | Oakland–Berkeley Hills, 1991 Tunnel Fire footprint |
| Population modeled | 5 towns, **9,000** residents |
| Shelters / crews | 3 shelters (cap 3,600 / 3,400 / 2,800) · 5 crews |
| Objective (all backends) | **≈ 75,668 person-minutes**, est. clearance **14.2 min** |
| Live re-solve | **~140 ms** (classical, warm) — fire advance / road close / lost crew |
| QAOA | **9 qubits**, ~0.5 s, **hits exact optimum** |
| Qubit budget | ≤ 20 (asserted before any quantum solve) |
| Tests | 8/8 pytest invariants + `smoke.py` ALL CHECKS PASSED |

---

## 3 · Q&A prep (the 5 minutes after — anticipate the attacks)

- **"Isn't the quantum part decorative? OR-Tools already wins."**
  _"Correct, and we say so on stage. We're not entered for quantum speed advantage. The
  claim is three things, all defensible: the assignment maps cleanly to a QUBO/Ising
  Hamiltonian, it executes on real qubits and recovers the optimum, and re-solve latency
  — not solve quality — is the binding product constraint. The scaling curve is where
  accelerated hardware earns its place."_

- **"How is this 'agentic'?"**
  _"The agent ingests an unstructured operator brief, extracts structured incident
  parameters, grounds them to map entities, selects the solve, and narrates each delta in
  dispatcher language. Claude when the key's set; a deterministic fallback otherwise."_

- **"Real roads or a toy graph?"**
  _"Real OpenStreetMap road network for the Oakland–Berkeley Hills, cached offline. Every
  route is a real street path; the fire blocks real nodes and routes re-solve around them."_

- **"What's the QAOA actually doing — is it a real circuit?"**
  _"Real QAOA(p=1). We build the Ising cost Hamiltonian from the QUBO, run the ansatz over
  a small (β,γ) grid on Aer, sample bitstrings, and keep the one with the lowest true
  energy. We dropped the COBYLA outer loop on purpose — the classical optimizer was the
  thing that hung; the circuit isn't."_

- **"Why an annealer too?"**
  _"It's our stand-in for XpyQ's neuromorphic optimization backend and it scales past what
  Aer can simulate. It walks the same QUBO landscape — single relabels plus two-town swaps
  so it escapes capacity-overflow local minima — and hits the same optimum at demo size."_

- **"Does it ever produce an infeasible plan?"**
  _"Capacity is enforced and surfaced: the right rail shows per-shelter load bars and an
  OVER badge, and any town with only a fire-adjacent exit is flagged. The baseline scenario
  is feasible and the tests assert it."_

---

## 4 · Honesty guardrails (non-negotiable — these protect the score)
- **Always** state the qubit count when showing QAOA.
- **Never** imply quantum is faster than OR-Tools at this scale. Say the opposite first.
- The framing is **"re-solve latency is the product constraint,"** never "quantum advantage."
- If a backend falls back (QAOA timeout, no Anthropic key), **name it** — graceful
  degradation is a feature here, not something to hide.

---

## 5 · Prize alignment (where to lean)
- **Best Applied AI System** — the agentic crisis→plan→narration loop on real infrastructure.
- **Specialized Compute (best use of XpyQ)** — QUBO is exactly the neuromorphic backend's
  target; one `Solver` interface, drop-in `/decisions` adapter.
- **Quantum Advantage Award** — real QAOA qubits + honest scaling artifact (lead with honesty).
- **Most Creative / Grand** — "drag the fire, the map heals live" is the 75-second hook.

---

## 6 · If it breaks (backup plan)
- **Streamlit wedges / map won't render:** cut to the backup screen-recording (second tab).
  Narrate over it identically — the beats are the same.
- **No Anthropic key / Claude down:** the parser and narrator fall back to deterministic
  text. Say _"running the offline fallback"_ and continue; the plan is unaffected.
- **QAOA slow on the venue laptop:** it runs in a worker thread with a hard timeout and
  falls back to the annealer — the UI never hangs. If it falls back live, say so and move on.
- **Worst case:** lead with `python tests/smoke.py` output — "ALL CHECKS PASSED" plus the
  four-backends-agree line is itself a credible 30-second proof.

> **Record the backup video before code freeze (Sat ~21:00).** One clean pass of Beats
> 1–4 is enough. Save it as `demo/backup_demo.mp4`.
