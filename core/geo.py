"""DISPATCH geo layer — the real road map, WITHOUT the OSMnx/GDAL install trap.

The map is the product. This module produces a routable road graph for a real
region and draws routes along real streets. It gets the road data three ways,
in order of preference, so it can NEVER block the demo:

  1. cached JSON  (data/region_graph.json) — committed, fully offline
  2. Overpass API (plain HTTP via `requests`, no GDAL) — fetches real OSM roads
  3. synthetic grid over the real bounding box — last-resort, still routable

Downstream (hazard, routing, evacuate, viz) only sees a NetworkX graph whose
nodes carry x=lng / y=lat and whose edges carry length (m) and travel_time (s).
It never knows or cares which of the three produced it.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
CACHE_PATH = os.path.join(DATA, "region_graph.json")

# Oakland–Berkeley Hills — the 1991 Tunnel Fire footprint. bbox = (S, W, N, E).
REGION = {
    "name": "Oakland–Berkeley Hills, CA",
    "center": [-122.218, 37.858],
    "zoom": 12.2,
    "bbox": [37.820, -122.275, 37.905, -122.160],  # south, west, north, east
}

DEFAULT_SPEED_MS = 11.0  # ~40 km/h on hilly residential/arterial roads


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def haversine_m(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    """Great-circle distance in metres between two [lng,lat] points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _add_edge(g: nx.Graph, u, v, ux, uy, vx, vy, speed=DEFAULT_SPEED_MS):
    length = haversine_m(ux, uy, vx, vy)
    if length <= 0:
        return
    g.add_edge(u, v, length=length, travel_time=length / speed)


# --------------------------------------------------------------------------- #
# graph sources
# --------------------------------------------------------------------------- #
def fetch_overpass(bbox, timeout=25) -> nx.Graph:
    """Fetch the real drivable road network from OpenStreetMap via Overpass.

    Lightweight: just `requests` + JSON parsing. No GDAL/geopandas.
    Raises on any failure so the caller can fall back.
    """
    import requests

    s, w, n, e = bbox
    query = f"""
    [out:json][timeout:{timeout}];
    (way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|motorway_link|primary_link|secondary_link"]
        ({s},{w},{n},{e}););
    out body; >; out skel qt;
    """
    headers = {"User-Agent": "DISPATCH-hackathon/1.0 (anchit.jain@berkeley.edu)"}
    for url in ("https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter"):
        try:
            # GET with a User-Agent — Overpass 406s anonymous POSTs.
            r = requests.get(url, params={"data": query}, headers=headers,
                             timeout=timeout + 20)
            r.raise_for_status()
            return _graph_from_overpass_json(r.json())
        except Exception:
            continue
    raise RuntimeError("Overpass fetch failed on all mirrors")


def _graph_from_overpass_json(payload: dict) -> nx.Graph:
    nodes = {el["id"]: (el["lon"], el["lat"])
             for el in payload["elements"] if el["type"] == "node"}
    g = nx.Graph()
    speed_by_hw = {
        "motorway": 28, "trunk": 22, "primary": 17, "secondary": 14,
        "tertiary": 12, "residential": 9, "unclassified": 9,
    }
    for el in payload["elements"]:
        if el["type"] != "way":
            continue
        hw = el.get("tags", {}).get("highway", "residential").split("_link")[0]
        speed = speed_by_hw.get(hw, DEFAULT_SPEED_MS)
        refs = el.get("nodes", [])
        for a, b in zip(refs, refs[1:]):
            if a in nodes and b in nodes:
                ax, ay = nodes[a]
                bx, by = nodes[b]
                if a not in g:
                    g.add_node(a, x=ax, y=ay)
                if b not in g:
                    g.add_node(b, x=bx, y=by)
                _add_edge(g, a, b, ax, ay, bx, by, speed)
    if g.number_of_edges() == 0:
        raise RuntimeError("Overpass returned no drivable edges")
    # keep the largest connected component so routing always has a path
    comp = max(nx.connected_components(g), key=len)
    return g.subgraph(comp).copy()


def synthetic_grid(bbox, nx_cells=24, ny_cells=20, jitter=0.18) -> nx.Graph:
    """A plausible, fully-routable road grid over the real bbox (last resort).

    Renders as a believable street network on the real basemap. Includes a few
    diagonal 'arterials' so routes aren't visibly Manhattan-perfect.
    """
    s, w, n, e = bbox
    g = nx.Graph()
    dlng = (e - w) / (nx_cells - 1)
    dlat = (n - s) / (ny_cells - 1)
    # deterministic pseudo-jitter so the map is stable across runs
    def jit(i, j, salt):
        v = math.sin((i * 73 + j * 19 + salt * 7) * 0.5)
        return v

    def node_xy(i, j):
        x = w + i * dlng + jit(i, j, 1) * dlng * jitter
        y = s + j * dlat + jit(i, j, 2) * dlat * jitter
        return x, y

    for i in range(nx_cells):
        for j in range(ny_cells):
            x, y = node_xy(i, j)
            g.add_node((i, j), x=x, y=y)
    for i in range(nx_cells):
        for j in range(ny_cells):
            x, y = g.nodes[(i, j)]["x"], g.nodes[(i, j)]["y"]
            if i + 1 < nx_cells:
                nx2 = g.nodes[(i + 1, j)]
                _add_edge(g, (i, j), (i + 1, j), x, y, nx2["x"], nx2["y"])
            if j + 1 < ny_cells:
                ny2 = g.nodes[(i, j + 1)]
                _add_edge(g, (i, j), (i, j + 1), x, y, ny2["x"], ny2["y"])
            # sparse diagonals as arterials
            if (i + j) % 4 == 0 and i + 1 < nx_cells and j + 1 < ny_cells:
                d = g.nodes[(i + 1, j + 1)]
                _add_edge(g, (i, j), (i + 1, j + 1), x, y, d["x"], d["y"], speed=20)
    comp = max(nx.connected_components(g), key=len)
    return g.subgraph(comp).copy()


# --------------------------------------------------------------------------- #
# cache (offline-safe)
# --------------------------------------------------------------------------- #
def save_graph_json(g: nx.Graph, path=CACHE_PATH, source="unknown"):
    data = {
        "source": source,
        "nodes": [[str(i), d["x"], d["y"]] for i, d in g.nodes(data=True)],
        "edges": [[str(u), str(v), d["length"], d["travel_time"]]
                  for u, v, d in g.edges(data=True)],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_graph_json(path=CACHE_PATH) -> tuple[nx.Graph, str]:
    with open(path) as f:
        data = json.load(f)
    g = nx.Graph()
    for i, x, y in data["nodes"]:
        g.add_node(i, x=x, y=y)
    for u, v, length, tt in data["edges"]:
        g.add_edge(u, v, length=length, travel_time=tt)
    return g, data.get("source", "cache")


def build_or_load_graph(prefer_cache=True, allow_network=True) -> tuple[nx.Graph, str]:
    """Return (graph, source). Tries cache -> Overpass -> synthetic."""
    if prefer_cache and os.path.exists(CACHE_PATH):
        try:
            g, src = load_graph_json()
            if g.number_of_edges() > 0:
                return g, f"cache({src})"
        except Exception:
            pass
    if allow_network:
        try:
            g = fetch_overpass(REGION["bbox"])
            save_graph_json(g, source="overpass")
            return g, "overpass"
        except Exception:
            pass
    g = synthetic_grid(REGION["bbox"])
    save_graph_json(g, source="synthetic")
    return g, "synthetic"


# --------------------------------------------------------------------------- #
# snapping + routing
# --------------------------------------------------------------------------- #
def snap(g: nx.Graph, lng: float, lat: float):
    """Nearest graph node to a [lng,lat] point."""
    best, bestd = None, float("inf")
    for nid, d in g.nodes(data=True):
        dd = (d["x"] - lng) ** 2 + (d["y"] - lat) ** 2
        if dd < bestd:
            bestd, best = dd, nid
    return best


def node_xy(g: nx.Graph, nid):
    d = g.nodes[nid]
    return [d["x"], d["y"]]


# --------------------------------------------------------------------------- #
# graph simplification — contract degree-2 chains so routing runs on
# intersections only (44k nodes -> ~4k), while edges remember the full street
# geometry so drawn routes still trace real roads. This is what keeps the live
# re-solve under ~100 ms.
# --------------------------------------------------------------------------- #
def simplify_graph(g_full: nx.Graph) -> nx.Graph:
    g = nx.Graph()
    for u, v, d in g_full.edges(data=True):
        g.add_node(u, **g_full.nodes[u])
        g.add_node(v, **g_full.nodes[v])
        g.add_edge(u, v, length=d["length"], travel_time=d["travel_time"], nodes=[u, v])

    work = [n for n in g.nodes if g.degree(n) == 2]
    while work:
        n = work.pop()
        if n not in g or g.degree(n) != 2:
            continue
        a, b = list(g.neighbors(n))
        if a == b:                       # stub/loop — drop it
            g.remove_node(n)
            continue
        if g.has_edge(a, b):             # would create a parallel edge — keep n
            continue
        ea, eb = g[a][n], g[n][b]
        na = ea["nodes"][:] if ea["nodes"][-1] == n else ea["nodes"][::-1]   # ...->n
        nb = eb["nodes"][:] if eb["nodes"][0] == n else eb["nodes"][::-1]     # n->...
        merged = na + nb[1:]
        g.add_edge(a, b, length=ea["length"] + eb["length"],
                   travel_time=ea["travel_time"] + eb["travel_time"], nodes=merged)
        g.remove_node(n)
        for x in (a, b):
            if x in g and g.degree(x) == 2:
                work.append(x)
    return g


class RoadNet:
    """Fast routing over a simplified graph, with full-street polyline expansion
    and fire-aware edge pruning. This is the object the solver actually uses."""

    def __init__(self, g_full: nx.Graph, source: str = "unknown"):
        import numpy as np
        self.source = source
        self.coords = {nid: [d["x"], d["y"]] for nid, d in g_full.nodes(data=True)}
        self.simp = simplify_graph(g_full)
        # numpy arrays of original-node coords (for vectorised fire test)
        self._all_ids = list(self.coords.keys())
        self._all_xy = np.array([self.coords[i] for i in self._all_ids], dtype=float)
        # numpy arrays of simplified-node coords (for vectorised snap)
        self._simp_ids = list(self.simp.nodes)
        self._simp_xy = np.array([self.coords[i] for i in self._simp_ids], dtype=float)
        # map every ORIGINAL node id -> the simplified edges that pass through it,
        # so fire -> blocked-edge lookup is O(#blocked) instead of O(#edges).
        self._node2edges: dict = {}
        self._tt0: dict = {}
        for u, v, d in self.simp.edges(data=True):
            for nid in d["nodes"]:
                self._node2edges.setdefault(nid, set()).add((u, v))
            self._tt0[(u, v)] = self._tt0[(v, u)] = d["travel_time"]
        self._BIG = 1e9

    # --- snapping (vectorised) ---
    def snap(self, lng: float, lat: float):
        import numpy as np
        d = (self._simp_xy[:, 0] - lng) ** 2 + (self._simp_xy[:, 1] - lat) ** 2
        return self._simp_ids[int(np.argmin(d))]

    # --- hazard -> blocked original nodes / simplified edges ---
    def blocked_nodes(self, fire_center, fire_radius_deg) -> set:
        import numpy as np
        d = np.hypot(self._all_xy[:, 0] - fire_center[0],
                     self._all_xy[:, 1] - fire_center[1])
        idx = np.nonzero(d <= fire_radius_deg)[0]
        return {self._all_ids[i] for i in idx}

    def _blocked_edges(self, blocked: set):
        if not blocked:
            return set()
        out = set()
        n2e = self._node2edges
        for nid in blocked:
            e = n2e.get(nid)
            if e:
                out |= e
        return out

    def _penalize(self, blocked_edges):
        """Temporarily raise blocked edges' travel_time to BIG (in place).
        Returns a restore-list. Avoids edge_subgraph view overhead — Dijkstra
        runs on the plain graph, just steering around the fire. Always restore."""
        saved = []
        for u, v in blocked_edges:
            d = self.simp[u][v]
            saved.append((u, v, d["travel_time"]))
            d["travel_time"] = self._BIG
        return saved

    def _restore(self, saved):
        for u, v, tt in saved:
            self.simp[u][v]["travel_time"] = tt

    def _expand(self, simp_path):
        poly = []
        for p, q in zip(simp_path, simp_path[1:]):
            seq = self.simp[p][q]["nodes"]
            if seq[0] != p:
                seq = seq[::-1]
            for nid in seq:
                xy = self.coords[nid]
                if not poly or poly[-1] != xy:
                    poly.append(xy)
        return poly

    # --- routing (fire-aware) ---
    def route(self, src_coord, dst_coord, blocked: set | None = None) -> "RouteResult":
        blocked = blocked or set()
        s, t = self.snap(*src_coord), self.snap(*dst_coord)
        if s == t:
            return RouteResult([list(src_coord), list(dst_coord)], 0.0, False)
        be = self._blocked_edges(blocked)
        saved = self._penalize(be)
        try:
            path = nx.shortest_path(self.simp, s, t, weight="travel_time")
        except Exception:
            self._restore(saved)
            return RouteResult([list(src_coord), list(dst_coord)], 0.0, True)
        self._restore(saved)
        # real travel time from base weights; compromised if it had to use a blocked edge
        real_tt = sum(self._tt0[(a, b)] for a, b in zip(path, path[1:]))
        compromised = any((a, b) in be or (b, a) in be for a, b in zip(path, path[1:]))
        return RouteResult(self._expand(path), real_tt, compromised)

    # --- travel-time matrix among key coords (fire-aware) ---
    def matrix(self, coords, blocked: set | None = None):
        blocked = blocked or set()
        saved = self._penalize(self._blocked_edges(blocked))
        snapped = [self.snap(*c) for c in coords]
        BIG = 1e6
        mat = [[0.0] * len(coords) for _ in coords]
        for i, s in enumerate(snapped):
            try:
                dist = nx.single_source_dijkstra_path_length(
                    self.simp, s, weight="travel_time", cutoff=self._BIG / 2)
            except nx.NodeNotFound:
                dist = {}
            for j, t in enumerate(snapped):
                v = dist.get(t, BIG)
                mat[i][j] = 0.0 if i == j else min(v, BIG)
        self._restore(saved)
        return mat


def build_roadnet(prefer_cache=True, allow_network=True) -> RoadNet:
    g, src = build_or_load_graph(prefer_cache, allow_network)
    return RoadNet(g, source=src)


@dataclass
class RouteResult:
    polyline: list          # [[lng,lat], ...] along real edges
    travel_time: float      # seconds
    compromised: bool       # True if it had to pass near the hazard (no clean path)


def route(g: nx.Graph, src_coord, dst_coord, blocked_nodes: set | None = None) -> RouteResult:
    """Shortest travel-time path between two coords, avoiding blocked (fire) nodes.

    Strategy: route on the graph minus blocked nodes. If that disconnects src/dst,
    fall back to a heavily-penalised path on the full graph and flag it compromised
    (so the UI/agent can say 'only route left runs near the fire').
    """
    blocked_nodes = blocked_nodes or set()
    s = snap(g, *src_coord)
    t = snap(g, *dst_coord)
    if s == t:
        return RouteResult([list(src_coord), list(dst_coord)], 0.0, False)

    clean = [n for n in (set(g.nodes) - blocked_nodes)]
    if s in blocked_nodes:
        blocked_nodes = blocked_nodes - {s}
    if t in blocked_nodes:
        blocked_nodes = blocked_nodes - {t}
    sub = g.subgraph(set(g.nodes) - blocked_nodes)
    try:
        path = nx.shortest_path(sub, s, t, weight="travel_time")
        tt = nx.shortest_path_length(sub, s, t, weight="travel_time")
        return RouteResult([node_xy(g, n) for n in path], tt, False)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    # penalised fallback: never traverse a blocked node if avoidable, but never hang
    def wfn(u, v, d):
        pen = 50.0 if (u in blocked_nodes or v in blocked_nodes) else 1.0
        return d["travel_time"] * pen
    try:
        path = nx.shortest_path(g, s, t, weight=wfn)
        tt = sum(g[a][b]["travel_time"] for a, b in zip(path, path[1:]))
        compromised = any(n in blocked_nodes for n in path)
        return RouteResult([node_xy(g, n) for n in path], tt, compromised)
    except Exception:
        return RouteResult([list(src_coord), list(dst_coord)], 0.0, True)


def travel_time_matrix(g: nx.Graph, coords: list, blocked_nodes: set | None = None) -> list:
    """Symmetric-ish travel-time matrix (seconds) among a small set of key coords.

    Runs one Dijkstra per source over the fire-pruned graph. Designed for ~10-25
    key points (stations/towns/shelters/defensible), keeping a re-solve well under
    100 ms. Unreachable pairs get a large finite penalty (not inf) so solvers stay happy.
    """
    blocked_nodes = blocked_nodes or set()
    sub = g.subgraph(set(g.nodes) - blocked_nodes)
    snapped = [snap(g, *c) for c in coords]
    BIG = 1e6
    mat = [[0.0] * len(coords) for _ in coords]
    for i, s in enumerate(snapped):
        try:
            dist = nx.single_source_dijkstra_path_length(sub, s, weight="travel_time")
        except nx.NodeNotFound:
            dist = {}
        for j, t in enumerate(snapped):
            mat[i][j] = 0.0 if i == j else dist.get(t, BIG)
    return mat
