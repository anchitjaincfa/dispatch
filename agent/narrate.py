"""Dispatcher narration from STRUCTURED DIFFS only (restate, never invent numbers).

Deterministic templates are primary; Claude polishes the phrasing when a key is
present. Every number in the narration comes from the SolveResult, so the agent
can't hallucinate a clearance time or a population.
"""
from __future__ import annotations

import os


def _name(scenario, key, _id):
    for item in scenario[key]:
        if item["id"] == _id:
            return item["name"]
    return _id


def narrate_event(event, before, after, scenario) -> str:
    """One dispatcher sentence describing what the re-solve changed."""
    a = after.extra
    clear = a.get("est_clear_min")
    feas = "feasible" if after.feasible else "OVER CAPACITY"
    comp = a.get("compromised_evacs", [])

    if event == "initial":
        end = a.get("endangered", [])
        top = end[0]["name"] if end else "the corridor"
        return (f"Plan set. {len(after.evac_assignment)} towns routed to shelters, "
                f"{len(after.crew_routes)} crews staged. {top} is most exposed. "
                f"Estimated clearance {clear} min — plan is {feas}.")

    if event == "advance_fire":
        moved = _diff_assignment(before, after, scenario)
        base = (f"Wind shift — fire advanced. Re-solved in {after.wall_ms:.0f} ms. ")
        if moved:
            base += f"Rerouted {', '.join(moved)}. "
        if comp:
            base += (f"{', '.join(_name(scenario,'towns',t) for t in comp)} now has "
                     f"only a fire-adjacent exit. ")
        return base + f"New clearance {clear} min — {feas}."

    if event == "close_road":
        return (f"Road closed. Crews and evacuees re-routing around the closure "
                f"({after.wall_ms:.0f} ms). Clearance {clear} min — {feas}.")

    if event == "lose_crew":
        return (f"Crew lost contact. Reassigned coverage across remaining "
                f"{len(after.crew_routes)} crews. Clearance {clear} min — {feas}.")

    if event == "reset":
        return "Scenario reset to initial conditions."

    return f"Re-solved in {after.wall_ms:.0f} ms — clearance {clear} min, {feas}."


def _diff_assignment(before, after, scenario):
    if before is None:
        return []
    out = []
    for tid, sid in after.evac_assignment.items():
        old = before.evac_assignment.get(tid)
        if old and old != sid:
            out.append(f"{_name(scenario,'towns',tid)} → {_name(scenario,'shelters',sid)}")
    return out


def polish_with_claude(line, after) -> str:
    """Optionally smooth the phrasing — but only restate, never add numbers."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return line
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=("You are a fire dispatcher. Rephrase the status into one calm, "
                    "radio-style sentence. Do NOT add, remove, or change any number "
                    "or name. Restate only."),
            messages=[{"role": "user", "content": line}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip() or line
    except Exception:
        return line


def narrate(event, before, after, scenario, use_claude=True) -> str:
    line = narrate_event(event, before, after, scenario)
    return polish_with_claude(line, after) if use_claude else line
