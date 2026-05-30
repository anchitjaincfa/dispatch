"""Agent layer — natural-language crisis -> structured problem.

Deterministic parser is PRIMARY (regex + known-entity matching) so the product
runs with no API key and the demo never depends on a live LLM call. When
ANTHROPIC_API_KEY is set, Claude refines the parse via tool-calling/structured
output — this is the Track-01 "agent calls the optimizer as a tool" half.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class ParsedCrisis:
    num_crews: int | None = None
    num_shelters: int | None = None
    towns_at_risk: list = field(default_factory=list)
    fire_hint: str = ""
    source: str = "deterministic"
    raw: str = ""


def _num_before(text, *keywords):
    for kw in keywords:
        m = re.search(r"(\d+)\s+" + kw, text, re.I)
        if m:
            return int(m.group(1))
    return None


def parse_deterministic(text, scenario) -> ParsedCrisis:
    town_names = [t["name"] for t in scenario["towns"]]
    mentioned = [n for n in town_names if n.split(" /")[0].split(" (")[0].lower()
                 in text.lower()]
    fire_hint = ""
    m = re.search(r"(north|south|east|west|near|toward[s]?)\s+[\w/ ]+", text, re.I)
    if m:
        fire_hint = m.group(0).strip()
    return ParsedCrisis(
        num_crews=_num_before(text, "crews?", "fire crews?", "engines?"),
        num_shelters=_num_before(text, "shelters?"),
        towns_at_risk=mentioned,
        fire_hint=fire_hint,
        source="deterministic",
        raw=text,
    )


def parse_with_claude(text, scenario) -> ParsedCrisis | None:
    """Refine the parse with Claude structured output. Returns None on any failure."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        tool = {
            "name": "report_crisis",
            "description": "Structured extraction of a wildfire dispatch crisis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "num_crews": {"type": "integer"},
                    "num_shelters": {"type": "integer"},
                    "towns_at_risk": {"type": "array", "items": {"type": "string"}},
                    "fire_hint": {"type": "string"},
                },
            },
        }
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=400,
            tools=[tool],
            tool_choice={"type": "tool", "name": "report_crisis"},
            messages=[{"role": "user", "content":
                       f"Towns in play: {[t['name'] for t in scenario['towns']]}.\n"
                       f"Dispatcher said: {text!r}. Extract the crisis."}],
        )
        for block in msg.content:
            if block.type == "tool_use":
                d = block.input
                return ParsedCrisis(
                    num_crews=d.get("num_crews"),
                    num_shelters=d.get("num_shelters"),
                    towns_at_risk=d.get("towns_at_risk", []),
                    fire_hint=d.get("fire_hint", ""),
                    source="claude", raw=text)
    except Exception:
        return None
    return None


def parse_crisis(text, scenario) -> ParsedCrisis:
    return parse_with_claude(text, scenario) or parse_deterministic(text, scenario)
