# HACKATHON_NOTES — DISPATCH @ XpyQ Quantum AI Hackathon 1.0

_Findings that shape the build. Updated 2026-05-30._

## Event (confirmed from Gmail + Luma)
- **QUANTUM AI HACKATHON 1.0 — "AI agents meet qubits, powered by XpyQ."**
- Host: Berkeley Gateway Accelerator. Organizers: Kyle Valiton, Dilip Vasudevan,
  Ayushman Dey, August Bernberg.
- Where: 2168 Shattuck Ave, FL 2, Berkeley, CA. ~10 teams, 139 confirmed.
- When: **Sat May 30 09:00–21:00** (code freeze ~21:00), **Sun May 31 09:00–12:30**
  (final submissions + 10-min demo + 5-min Q&A). Awards dinner the following Wed.
- Prizes: **Grand $2,000** + dinner; Runner-up; 4× Track Champions; **Quantum
  Advantage Award**; **Specialized Compute Award (best use of XpyQ)**; **Best
  Applied AI System**; Most Creative.

## Submission logistics (from org email, Ayushman Dey, 2026-05-30 18:22Z)
- Platform: https://xpyq.vercel.app/  (invite-only / **NDA-gated** — Anchit has access)
- Slack: xpyqhackathon10
- Bug/feature sheet: Google Sheet (shared in email)
- **IBM Quantum**: https://quantum.cloud.ibm.com/ — free tier = **10 minutes** real QPU.
- Devpost: https://xpyq-hackathon.devpost.com/ (verify exact rubric/deadline there)

## Tracks — DISCREPANCY TO VERIFY ON DEVPOST/SLACK
- Design docs assumed: "01 Agentic optimization × 03 Routing & scheduling."
- **Current Luma page lists finance-flavored tracks**: Trading & Execution,
  Portfolio & Alpha, Risk/Fraud/Compliance, **Applied AI & Agentic Systems**;
  **bonus areas: supply-chain scheduling, energy-grid dispatch, graph intelligence,
  scientific simulation.**
- DISPATCH (wildfire crew routing + evac) is **not finance** → target
  **Track 4 (Applied AI & Agentic) + the "dispatch/scheduling" bonus**, aim at
  **Best Applied AI System + Specialized Compute + Quantum Advantage**.

## Judging (Luma): "clear thinking, honest measurement, working prototypes,
compare against classical baselines." → **OR-Tools-as-baseline is a REQUIRED
feature, not a liability.** The scaling benchmark (classical wall-time vs instance
size) is the artifact judges are explicitly asking for.

## XpyQ — what it actually is (the #1 unknown, resolved as far as public site allows)
Heterogeneous **"post-GPU" compute cloud**, three backends:
- **RRAM analog in-memory crossbar** → one-pass matrix-vector multiply (MVM),
  ~256µs, 5pJ/MAC (FPGA-emulated today).
- **Neuromorphic / Josephson-junction spiking** → *explicitly "optimization and
  pattern recognition"* → **our QUBO/assignment target.**
- **Superconducting quantum** → research-stage.
- SDK model: *"write a Python function, deploy; an LLM ensemble auto-optimizes it
  against real hardware cost models."* Submission endpoint observed: **`/decisions`**.
- It is **NOT** a clean D-Wave-style QUBO REST endpoint. Treat it as: submit a
  Python decision/objective function → get an optimized assignment back.

### Decision (records the §5 call from CLAUDE.md)
**Hybrid, three honest backends behind one `Solver` interface:**
1. `classical` — OR-Tools VRP / greedy. Always-on baseline + live re-solve path.
2. `quantum` — assignment core → QUBO → our numpy simulated-annealer (always works);
   QAOA on Aer / one real IBM Quantum run (≤20 qubits) when libs/credits present.
3. `xpyq` — same objective submitted to XpyQ `/decisions` (env-driven adapter;
   drop in real request shape from logged-in platform).
Quantum is **never** on the live drag path. Honest framing: prove the mapping +
that *re-solve speed is the binding product constraint*, not production quantum advantage.

## Engineering de-risk decisions (why the build deviates from the design doc)
- **Dropped OSMnx** (GDAL/geopandas = #1 install trap). Road graph via:
  cached JSON → Overpass HTTP fetch (`requests`, no GDAL) → synthetic grid fallback.
  Same real roads on the same real map (pydeck + free Carto basemap, no token).
- Every external dep degrades gracefully: ortools→greedy, qiskit→numpy annealer,
  Anthropic→deterministic parser/narrator, XpyQ→"unavailable" badge.
- Default scenario = **Oakland–Berkeley Hills (1991 Tunnel Fire footprint)** —
  local resonance for a Berkeley-judged room.

## Env reality on this machine
- Only system **Python 3.9.6** (no homebrew/pyenv). venv at `~/dispatch/.venv`.
- Heavy deps (ortools, qiskit) installed best-effort; app must run without them.
