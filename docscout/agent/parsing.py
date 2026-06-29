"""Parse model text output into a DocScout Action.

The model is asked to emit exactly one action block of the form:

    ACTION: search
    query: permission sync delay

    ACTION: read
    doc_id: doc_03
    section_id: 4

    ACTION: expand
    doc_id: doc_03
    section_id: 4
    direction: right

    ACTION: answer
    text: 5 minutes
    evidence: doc_03:4, doc_03:5

Parsing is deliberately lenient (case-insensitive ACTION, optional fields) so a
weak base model still produces a usable action at R0.
"""

from __future__ import annotations

import re

from docscout.types import Action, ActionType

_ACTION_RE = re.compile(r"ACTION\s*:\s*([a-zA-Z]+)", re.IGNORECASE)
_KV_RE = re.compile(r"^\s*([a-zA-Z_]+)\s*:\s*(.*?)\s*$")


def _parse_fields(text: str, after: int) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text[after:].splitlines():
        m = _KV_RE.match(line)
        if not m:
            continue
        key, val = m.group(1).lower(), m.group(2).strip()
        if key == "action":
            break  # next action block
        fields[key] = val
    return fields


def parse_action(text: str) -> Action:
    """Parse a single action from model output. Defaults to a no-op search on failure."""
    m = _ACTION_RE.search(text or "")
    if not m:
        return Action(ActionType.SEARCH, {"query": ""})  # caller/env handles empty
    name = m.group(1).lower()
    start = m.end()
    fields = _parse_fields(text, start)

    if name == "search":
        # lenient: accept `query:`, `q:`, or any single field's value as the query
        q = fields.get("query") or fields.get("q")
        if not q:
            vals = [v for v in fields.values()]
            q = vals[0] if vals else ""
        return Action(ActionType.SEARCH, {"query": q.strip().strip('"')})
    if name == "read":
        if "idx" in fields:
            return Action(ActionType.READ, {"idx": fields.get("idx", "")})
        # LENIENT: "read(1)" with idx in brackets
        br_match = re.search(r'read\s*\(\s*(\d+)\s*\)', text, re.IGNORECASE)
        if br_match:
            return Action(ActionType.READ, {"idx": br_match.group(1)})
        # LENIENT: "read(doc_04.3)" or "read(doc_04.3, section_1)"
        doc_match = re.search(r'read\s*\(\s*([^.]+?)\.(\d+)', text, re.IGNORECASE)
        if doc_match:
            return Action(ActionType.READ, {"doc_id": doc_match.group(1).strip(),
                                            "section_id": doc_match.group(2).strip()})
        return Action(ActionType.READ, {"doc_id": fields.get("doc_id", ""),
                                        "section_id": fields.get("section_id", "")})
    # LENIENT: bare number 1-9 = read candidate by index (helps base models that
    # don't follow the ACTION format but emit a candidate number)
    import re as _re2
    m2 = _re2.search(r'\b([1-9])\b', text or "")
    if m2 and not name:
        return Action(ActionType.READ, {"idx": m2.group(1)})
    if name == "expand":
        return Action(ActionType.EXPAND, {"doc_id": fields.get("doc_id", ""),
                                          "section_id": fields.get("section_id", ""),
                                          "direction": fields.get("direction", "right")})
    if name == "answer":
        ev = []
        raw_ev = fields.get("evidence", "")
        for part in raw_ev.split(","):
            if ":" in part:
                d, s = part.split(":", 1)
                ev.append({"doc_id": d.strip(), "section_id": s.strip()})
        return Action(ActionType.ANSWER, {"text": fields.get("text", ""), "evidence": ev})
    return Action(ActionType.SEARCH, {"query": ""})
