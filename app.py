"""DISPATCH — live agentic wildfire response. The map that saves lives.

Run:  streamlit run app.py

Real Oakland–Berkeley Hills road network (OpenStreetMap, cached offline) on a free
Carto dark basemap. Describe a crisis -> the agent parses it -> crews route to
defensible points and towns evacuate to shelters along real streets. Perturb the
scene (advance the fire, close a road, lose a crew) and the whole plan RE-SOLVES
live (~100 ms) while the dispatcher narrates. Swap the solver backend
(classical / accelerated annealer / QAOA / XpyQ) — the UI never changes.
"""
from __future__ import annotations

import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv():
    """Minimal, dependency-free .env loader (KEY=VALUE per line). Does not override
    variables already set in the real environment. Enables ANTHROPIC_API_KEY /
    XPYQ_API_KEY without exporting them or adding python-dotenv."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and key not in os.environ:
                os.environ[key] = val


_load_dotenv()

import streamlit as st
import streamlit.components.v1 as components

from agent import narrate, parse
from core import geo, hazard
from solver import (ClassicalSolver, QuantumSolver, XpyQSolver, QUBOProblem)

SCENARIO_PATH = os.path.join(HERE, "data", "scenario_berkeley.json")
SCENARIO_PATH = os.path.join(HERE, "data", "scenario_berkeley.json")

st.set_page_config(page_title="DISPATCH", layout="wide", page_icon="🔥")

# ----------------------------- design tokens --------------------------------
st.markdown("""
<style>
  .stApp {background: #0a0f0d;}
  .block-container {padding-top: 1.0rem; padding-bottom: 0.4rem; max-width: 100%;}
  .d-title {font-family: 'Space Grotesk', system-ui, sans-serif; font-size: 1.9rem;
            font-weight: 800; color: #ff6a18; letter-spacing: 1px; margin: 0;}
  .d-sub {color: #5e7a70; margin-top: -4px; font-size: 0.8rem; letter-spacing: 0.5px;}
  .pill {display:inline-block; background:#0f1714; border:1px solid #1d2f28;
         border-radius: 20px; padding: 4px 12px; font-family: ui-monospace, monospace;
         font-size: 0.74rem; color:#cfe3da;}
  .dot {height:8px; width:8px; background:#4ade80; border-radius:50%; display:inline-block;
        margin-right:6px; box-shadow:0 0 6px #4ade80;}
  .narr {background:#0f1714; color:#cfe3da; border-left:3px solid #ff6a18;
         padding:8px 12px; border-radius:8px; font-family: ui-monospace, monospace;
         font-size:0.8rem; margin-bottom:5px;}
  .chip {background:#0f1714; border:1px solid #1d2f28; border-radius:10px;
         padding:10px 12px; text-align:center;}
  .chip .v {font-size:1.5rem; font-weight:800; color:#cfe3da; font-family:'Space Grotesk',sans-serif;}
  .chip .k {font-size:0.68rem; color:#5e7a70; text-transform:uppercase; letter-spacing:0.5px;}
  .evac-ok {color:#cfe3da;} .evac-bad {color:#ff7a5c; font-weight:700;}
  .capbar {height:7px; background:#15241d; border-radius:4px; overflow:hidden; margin-top:3px;}
  .capfill {height:7px; border-radius:4px;}
  h1,h2,h3,h4 {color:#cfe3da;}
  section[data-testid="stSidebar"] {background:#0c1310;}
</style>
""", unsafe_allow_html=True)


# ----------------------------- resources ------------------------------------
@st.cache_resource(show_spinner="Loading Oakland–Berkeley Hills road network…")
def get_roadnet():
    return geo.build_roadnet()


def base_scenario():
    with open(SCENARIO_PATH) as f:
        return json.load(f)


def init_state():
    if "scenario" not in st.session_state:
        st.session_state.scenario = base_scenario()
        st.session_state.log = []
        st.session_state.prev = None
        st.session_state.event = "initial"
        st.session_state.parsed = None


def log(msg):
    st.session_state.log.insert(0, {"ts": time.strftime("%H:%M:%S"), "text": msg})
    st.session_state.log = st.session_state.log[:4]


def reset():
    st.session_state.scenario = base_scenario()
    st.session_state.prev = None
    st.session_state.event = "reset"
    st.session_state.log = []


HERO_PATH = os.path.join(HERE, "hero", "dispatch_drag.html")


@st.cache_data
def hero_html():
    with open(HERO_PATH) as f:
        return f.read()


# Custom bidirectional component: real Carto basemap + a DRAGGABLE fire marker.
# On drag-release the marker streams its lng/lat back to Python, which re-solves on
# the selected real backend and passes the updated routes back to be drawn.
_DRAGMAP_DIR = os.path.join(HERE, "components", "dispatch_map")
_dispatch_map_component = components.declare_component("dispatch_map", path=_DRAGMAP_DIR)


def dispatch_map(data, key=None):
    return _dispatch_map_component(data=data, key=key, default=None)


@st.cache_data(show_spinner="Sweeping instance sizes…")
def run_scaling():
    """The honest scaling artifact (cached; ~5s once). Exact enumeration vs CP-SAT
    vs the accelerated annealer across growing QUBO sizes."""
    from core import bench
    return bench.run()


roadnet = get_roadnet()
init_state()
sc = st.session_state.scenario

# ----------------------------- header ---------------------------------------
hc1, hcv, hc2, hc3 = st.columns([1.9, 1.5, 1.4, 1.0])
with hc1:
    st.markdown('<div class="d-title">▣ DISPATCH</div>', unsafe_allow_html=True)
    st.markdown('<div class="d-sub">WILDFIRE OPS CONSOLE · THE MAP THAT SAVES LIVES</div>',
                unsafe_allow_html=True)
with hcv:
    view_mode = st.radio("view", ["🗺 Real solver", "🔥 Live drag"],
                         horizontal=True, label_visibility="collapsed")
_drag_view = "Live drag" in view_mode
backend_label = "classical (OR-Tools)"
use_claude = False
if not _drag_view:           # these controls are inert in the mock drag view — hide them
    with hc2:
        backend_label = st.selectbox(
            "Solver backend",
            ["classical (OR-Tools)", "accelerated (annealer)", "qaoa (Aer qubits)",
             "xpyq (/decisions)"],
            index=0, label_visibility="collapsed")
    with hc3:
        use_claude = st.toggle("Claude narration",
                               value=bool(os.environ.get("ANTHROPIC_API_KEY")))

# ----- 2A: Live-drag hero view — true draggable fire (prototype, mock data) ---
if "Live drag" in view_mode:
    st.markdown(
        '<span class="pill">🔥 LIVE DRAG · drag the fire — routes re-solve under your '
        'finger. Interaction model on mock data; the real four-backend solver runs in '
        'the 🗺 Real solver view.</span>', unsafe_allow_html=True)
    components.html(hero_html(), height=760, scrolling=False)
    st.stop()


def make_solver(label):
    if label.startswith("classical"):
        return ClassicalSolver(roadnet)
    if label.startswith("accelerated"):
        return QuantumSolver(roadnet, method="annealer")
    if label.startswith("qaoa"):
        return QuantumSolver(roadnet, method="qaoa")
    return XpyQSolver(roadnet)


solver = make_solver(backend_label)

# ----------------------------- sidebar: agent + controls --------------------
with st.sidebar:
    st.markdown("### ◤ Incident")
    crisis = st.text_area(
        "operator brief", height=110,
        value="Wildfire ignited in the Oakland Hills near Hwy 24, wind pushing it "
              "west toward Forest Park and Claremont. 5 crews staged, 3 shelters online. "
              "Protect the towns and get everyone out.")
    if st.button("⚙ Parse with agent", use_container_width=True):
        st.session_state.parsed = parse.parse_crisis(crisis, sc)
    if st.session_state.parsed:
        p = st.session_state.parsed
        st.caption(f"parsed by **{p.source}** agent → "
                   f"crews: {p.num_crews or len(sc['stations'])} · "
                   f"shelters: {p.num_shelters or len(sc['shelters'])} · "
                   f"at risk: {', '.join(p.towns_at_risk) or '—'}")

    st.markdown("### ◤ Perturb the scene")
    c1, c2 = st.columns(2)
    if c1.button("▸▸ Advance fire", use_container_width=True):
        hazard.advance_fire(sc); st.session_state.event = "advance_fire"
    if c2.button("⊘ Close a road", use_container_width=True):
        import random
        f = sc["fire"]["center"]
        sc.setdefault("closed_roads", []).append(
            [f[0] + random.uniform(-0.015, 0.015), f[1] + random.uniform(-0.015, 0.015)])
        st.session_state.event = "close_road"
    live = [s for s in sc["stations"] if s["id"] not in sc.get("dead_crews", [])]
    if c1.button("✕ Lose crew", use_container_width=True) and live:
        import math
        f = sc["fire"]["center"]
        nearest = min(live, key=lambda s: math.hypot(s["coord"][0]-f[0], s["coord"][1]-f[1]))
        sc.setdefault("dead_crews", []).append(nearest["id"]); st.session_state.event = "lose_crew"
    if c2.button("⟲ Reset", use_container_width=True):
        reset(); sc = st.session_state.scenario

    st.caption("Move the fire (routes re-solve live):")
    n1, n2, n3 = st.columns(3)
    step = 0.004
    if n2.button("▲", use_container_width=True):
        hazard.nudge_fire(sc, dlat=step); st.session_state.event = "advance_fire"
    if n1.button("◀", use_container_width=True):
        hazard.nudge_fire(sc, dlng=-step); st.session_state.event = "advance_fire"
    if n3.button("▶", use_container_width=True):
        hazard.nudge_fire(sc, dlng=step); st.session_state.event = "advance_fire"
    if n2.button("▼", use_container_width=True):
        hazard.nudge_fire(sc, dlat=-step); st.session_state.event = "advance_fire"
    g1, g2 = st.columns(2)
    if g1.button("＋ grow fire", use_container_width=True):
        sc["fire"]["radius"] = min(sc["fire"]["radius"]*1.18, 0.035); st.session_state.event = "advance_fire"
    if g2.button("－ shrink", use_container_width=True):
        sc["fire"]["radius"] = max(sc["fire"]["radius"]*0.85, 0.004); st.session_state.event = "advance_fire"

# ----------------------------- solve ----------------------------------------
res = solver.solve(QUBOProblem(metadata=sc))
event = st.session_state.event
line = narrate.narrate(event, st.session_state.prev, res, sc, use_claude=use_claude)
if line and (not st.session_state.log or st.session_state.log[0]["text"] != line):
    log(line)
st.session_state.prev = res
st.session_state.event = "tick"

# ----------------------------- layout ---------------------------------------
map_col, panel = st.columns([3, 1.15])
with map_col:
    backend_short = res.backend
    st.markdown(
        f'<span class="pill"><span class="dot"></span>SOLVER ONLINE · backend: '
        f'<b>{backend_short}</b> · {res.wall_ms:.0f} ms · {res.extra.get("qubits","–")} qubits</span> '
        f'&nbsp;<span class="pill">{sc["region"]}</span>', unsafe_allow_html=True)
    map_data = {
        "center": sc["center"], "zoom": sc.get("zoom", 12.2),
        "fire": {"center": sc["fire"]["center"], "radius": sc["fire"]["radius"]},
        "towns": [{"id": t["id"], "name": t["name"], "coord": t["coord"]} for t in sc["towns"]],
        "shelters": [{"id": s["id"], "name": s["name"], "coord": s["coord"]} for s in sc["shelters"]],
        "stations": [{"id": s["id"], "name": s["name"], "coord": s["coord"]} for s in sc["stations"]],
        "defensible": [{"id": d["id"], "coord": d["coord"]} for d in sc["defensible"]],
        "crew_routes": res.crew_routes,
        "evac_routes": res.evac_routes,
        "endangered": [t["id"] for t in res.extra.get("endangered", [])],
        "dead_crews": sc.get("dead_crews", []),
    }
    fire_pos = dispatch_map(map_data, key="dragmap")
    if isinstance(fire_pos, dict) and fire_pos.get("lng") is not None:
        pos = [fire_pos["lng"], fire_pos["lat"]]
        cur = sc["fire"]["center"]
        if abs(pos[0] - cur[0]) > 1e-6 or abs(pos[1] - cur[1]) > 1e-6:
            sc["fire"]["center"] = pos
            st.session_state.event = "advance_fire"
            st.rerun()
    st.caption("🖱 drag the fire — re-solves on the selected backend · 🔵 crew routes · "
               "🟠 evac routes · 🟢 shelters · 🔵 stations · 🟡 towns (red = threatened) · ◇ defensible")

with panel:
    end = res.extra.get("endangered", [])
    a, b, c = st.columns(3)
    a.markdown(f'<div class="chip"><div class="v">{len(end)}</div><div class="k">Towns at risk</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="chip"><div class="v">{len(res.crew_routes)}</div><div class="k">Crews routed</div></div>', unsafe_allow_html=True)
    cap_badge = "OK" if res.feasible else "OVER"
    c.markdown(f'<div class="chip"><div class="v">{cap_badge}</div><div class="k">Capacity</div></div>', unsafe_allow_html=True)

    st.markdown("##### Dispatcher")
    if not st.session_state.log:
        st.markdown('<div class="narr">Awaiting first action. Use the controls to move the fire.</div>', unsafe_allow_html=True)
    for ln in st.session_state.log:
        st.markdown(
            f'<div class="narr"><span style="color:#5e7a70;font-size:0.66rem;'
            f'letter-spacing:0.5px">{ln["ts"]} · DISPATCH</span><br>{ln["text"]}</div>',
            unsafe_allow_html=True)

    st.markdown("##### Evacuation plan "
                f'<span style="color:#4ade80">{"FEASIBLE" if res.feasible else ""}</span>'
                f'<span style="color:#ff7a5c">{"" if res.feasible else "OVER CAPACITY"}</span>',
                unsafe_allow_html=True)
    name_t = {t["id"]: t["name"] for t in sc["towns"]}
    name_s = {s["id"]: s["name"] for s in sc["shelters"]}
    comp = set(res.extra.get("compromised_evacs", []))
    e_loads, e_caps = res.extra.get("shelter_load", {}), res.extra.get("shelter_cap", {})
    over_sh = {sid for sid in e_caps if e_loads.get(sid, 0) > e_caps[sid]}
    _b = "border-radius:5px;padding:1px 6px;font-size:0.6rem;font-weight:700"
    routed = f'<span style="background:rgba(34,197,94,.12);color:#22c55e;border:1px solid rgba(34,197,94,.35);{_b}">ROUTED</span>'
    overcap = f'<span style="background:rgba(255,122,92,.15);color:#ff7a5c;border:1px solid rgba(255,122,92,.45);{_b}">OVER CAP</span>'
    for tid, sid in res.evac_assignment.items():
        cls = "evac-bad" if tid in comp else "evac-ok"
        flag = " ⚠ fire-adjacent" if tid in comp else ""
        badge = overcap if sid in over_sh else routed
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:3px"><span class="{cls}">{name_t[tid]} → {name_s[sid]}{flag}'
            f'</span>{badge}</div>', unsafe_allow_html=True)

    st.markdown("##### Shelter capacity")
    loads = res.extra.get("shelter_load", {}); caps = res.extra.get("shelter_cap", {})
    for s in sc["shelters"]:
        ld, cp = loads.get(s["id"], 0), caps.get(s["id"], 1)
        pct = min(100, int(100*ld/max(cp, 1)))
        col = "#ff7a5c" if ld > cp else ("#eab308" if pct > 85 else "#22c55e")
        st.markdown(f'<div style="font-size:0.72rem;color:#5e7a70">{s["name"]} · {ld}/{cp}</div>'
                    f'<div class="capbar"><div class="capfill" style="width:{pct}%;background:{col}"></div></div>',
                    unsafe_allow_html=True)

    with st.expander("backend detail / honesty"):
        st.write({"method": res.extra.get("method"), "qubits": res.extra.get("qubits"),
                  "optimal": res.extra.get("optimal", res.extra.get("full_optimal")),
                  "est_clear_min": res.extra.get("est_clear_min"),
                  "blocked_nodes": res.extra.get("blocked_nodes"),
                  "wall_ms": res.wall_ms})
        st.caption("Quantum/accelerated backends solve the SAME town→shelter QUBO as "
                   "OR-Tools. This proves the mapping and that re-solve speed is the "
                   "binding constraint — not a production quantum speed advantage at this scale.")

# ----------------------------- scaling artifact -----------------------------
with st.expander("📈 Scaling — why latency is the product (honest benchmark)"):
    st.caption("Sweeps the town→shelter QUBO from small to large. Exact enumeration "
               "blows up (the core is genuinely NP-hard); CP-SAT stays fast at demo "
               "scale; the accelerated annealer stays flat AND hits the optimum "
               "(gap 0). That's why accelerated hardware belongs UPSTREAM of the "
               "~140 ms live loop — never on it.")
    if st.button("Run scaling benchmark", help="~5s, cached after the first run"):
        st.session_state.bench = run_scaling()
    rows = st.session_state.get("bench")
    if rows:
        import altair as alt
        import pandas as pd
        df = pd.DataFrame(rows)
        long = df.melt(id_vars=["vars"], value_vars=["brute_ms", "cpsat_ms", "anneal_ms"],
                       var_name="method", value_name="ms").dropna()
        label = {"brute_ms": "exact enumeration", "cpsat_ms": "OR-Tools CP-SAT",
                 "anneal_ms": "accelerated annealer"}
        long["method"] = long["method"].map(label)
        chart = (alt.Chart(long).mark_line(point=True).encode(
                    x=alt.X("vars:Q", title="QUBO variables (T × S)"),
                    y=alt.Y("ms:Q", title="solve time (ms, log)",
                            scale=alt.Scale(type="log")),
                    color=alt.Color("method:N", title="",
                                    scale=alt.Scale(
                                        domain=["exact enumeration", "OR-Tools CP-SAT",
                                                "accelerated annealer"],
                                        range=["#ff7a5c", "#3b82f6", "#22c55e"])))
                 .properties(height=300))
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df[["vars", "T", "S", "brute_ms", "cpsat_ms", "anneal_ms",
                         "anneal_gap"]], hide_index=True, use_container_width=True)
        st.caption("`anneal_gap` is 0.0 wherever brute force is still tractable — the "
                   "heuristic core stays exact as it scales.")
    else:
        st.caption("Click **Run scaling benchmark** to generate the chart.")
