"""The fire — the dynamic core that perturbs the graph.

Pure functions over a scenario dict + a RoadNet. Never mutate the base graph,
so RESET and re-solve are always clean. The fire does two things:
  1. marks road nodes inside its radius as blocked (handled by RoadNet)
  2. raises the priority / time-pressure of nearby towns
"""
from __future__ import annotations

import math


def _dist_deg(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def blocked_nodes(roadnet, scenario):
    """All road nodes inside the fire OR inside any closed-road blob."""
    fire = scenario["fire"]
    blocked = roadnet.blocked_nodes(fire["center"], fire["radius"])
    for cr in scenario.get("closed_roads", []):
        blocked |= roadnet.blocked_nodes(cr, fire["radius"] * 0.45)
    return blocked


def threat_level(town, fire) -> float:
    """0..1 — how threatened a town is (1 = inside the fire)."""
    d = _dist_deg(town["coord"], fire["center"])
    r = fire["radius"]
    if d <= r:
        return 1.0
    # decays to 0 by ~3x the fire radius
    return max(0.0, 1.0 - (d - r) / (2.0 * r))


def time_to_impact_min(town, fire, spread_deg_per_min=0.0015) -> float:
    """Rough minutes until the fire edge reaches the town (defensible Q&A model)."""
    d = _dist_deg(town["coord"], fire["center"]) - fire["radius"]
    if d <= 0:
        return 0.0
    return round(d / spread_deg_per_min, 1)


def endangered_towns(scenario, threshold=0.05):
    """Towns with non-trivial threat, most-threatened first, with impact ETAs."""
    fire = scenario["fire"]
    out = []
    for t in scenario["towns"]:
        lvl = threat_level(t, fire)
        if lvl >= threshold:
            out.append({**t, "threat": round(lvl, 3),
                        "eta_min": time_to_impact_min(t, fire)})
    out.sort(key=lambda t: (-t["threat"], t["eta_min"]))
    return out


def advance_fire(scenario, towns_pull=0.25, growth=1.15, max_radius=0.03):
    """Wind shift: nudge the fire toward the centroid of the towns and grow it."""
    f = scenario["fire"]
    tx = sum(t["coord"][0] for t in scenario["towns"]) / len(scenario["towns"])
    ty = sum(t["coord"][1] for t in scenario["towns"]) / len(scenario["towns"])
    f["center"][0] += (tx - f["center"][0]) * towns_pull
    f["center"][1] += (ty - f["center"][1]) * towns_pull
    f["radius"] = min(f["radius"] * growth, max_radius)
    return scenario


def nudge_fire(scenario, dlng=0.0, dlat=0.0):
    f = scenario["fire"]
    f["center"][0] += dlng
    f["center"][1] += dlat
    return scenario
