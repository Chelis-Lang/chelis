"""Prove bounded, unchanged forwarding of native protocol input slots.

Spec/11 section 1.3 requires validating each supplied DLPack keyword. This
helper checks its exact formal input to exact validator argument edge. Only
whole-value copies and moves with unique, dominating writes are supported;
casts, field projections, reference escapes and helper calls are not proofs.
The caller still owes registration, validator semantics and owner authority.
"""

from collections import defaultdict
from dataclasses import dataclass

from capacity_census_native_flow import (
    CallStage,
    NativeFlowEvidenceError,
    _blocks,
    _call_identity,
    _definition,
    _dominators,
    _identity,
    _place,
)


@dataclass(frozen=True)
class ArgumentSlot:
    local: int
    argument: int

    def __post_init__(self):
        if type(self.local) is not int or self.local < 1:
            raise ValueError("argument forwarding needs a positive input local")
        if type(self.argument) is not int or self.argument < 0:
            raise ValueError("argument forwarding needs a nonnegative argument index")


def _require(condition, message):
    if not condition:
        raise NativeFlowEvidenceError(message)


def _shape(value):
    _require(isinstance(value, dict) and value.get("open") is False,
             "forwarded argument has an open or missing type")
    shape = value.get("shape")
    _require(isinstance(shape, dict), "forwarded argument has no type shape")
    return shape


def _verify(raw, entry, stage, slots):
    _require(raw.get("format") == 3 and raw.get("scope") == "native-bindings",
             "argument forwarding needs native compiler evidence")
    _require(raw.get("errors") == [], "collector errors prevent argument forwarding")
    _require(isinstance(stage, CallStage) and slots, "missing argument forwarding obligations")
    _require(len({s.local for s in slots}) == len(slots)
             and len({s.argument for s in slots}) == len(slots),
             "duplicate input or argument forwarding obligation")
    bodies = [b for b in raw["bodies"] if _identity(b["definition"]) == entry]
    _require(len(bodies) == 1, "missing or ambiguous argument forwarding entry")
    body = bodies[0]
    _, adjacency = _blocks(body)
    dominators = _dominators(adjacency, 0)
    inputs = {item["local"]: item["type"] for item in body["formal_inputs"]}
    _require(len(inputs) == len(body["formal_inputs"]), "duplicate formal input")
    calls = [c for c in raw["calls"] if _definition(c["caller"]) == entry
             and c.get("kind") == "resolved"
             and _call_identity(c) == (stage.declared, stage.resolved)]
    _require(len(calls) == 1, "missing or ambiguous exact argument validator")
    call = calls[0]
    _require(call["block"] in dominators, "argument validator is unreachable")
    flows = [f for f in raw["flows"] if _definition(f["caller"]) == entry]
    writes = defaultdict(list)
    for flow in flows:
        writes[_place(flow["destination"]).local].append(flow)

    for slot in slots:
        _require(slot.local in inputs, "missing forwarded formal input")
        _require(slot.argument < len(call["arguments"])
                 and slot.argument < len(call["formal_inputs"]),
                 "missing forwarded validator argument")
        expected = _shape(inputs[slot.local])
        argument = call["arguments"][slot.argument]
        _require(argument.get("kind") in {"copy", "move"},
                 "validator argument substitutes a constant or non-value input")
        _require(_shape(call["formal_inputs"][slot.argument]) == expected,
                 "validator parameter changes the input type")
        value = argument["place"]
        consumer = call
        visited = set()
        while True:
            place = _place(value)
            _require(not place.projection and _shape(value["type"]) == expected,
                     "forwarded input changes its whole-value type or projection")
            _require(place.local not in visited, "cyclic argument forwarding")
            visited.add(place.local)
            if place.local in inputs:
                _require(place.local == slot.local and not writes[place.local],
                         "forwarded argument replaces or overwrites its formal input")
                break
            owners = writes[place.local]
            _require(len(owners) == 1, "forwarded temporary lacks one exact write")
            owner = owners[0]
            _require(owner["rvalue"] == "Use" and len(owner["sources"]) == 1
                     and not _place(owner["destination"]).projection,
                     "argument forwarding is not a whole-value copy or move")
            block, statement = owner["block"], owner["statement"]
            _require(type(statement) is int and type(consumer["statement"]) is int,
                     "forwarding statement location is malformed")
            _require(block in dominators.get(consumer["block"], set())
                     and (block != consumer["block"] or statement < consumer["statement"]),
                     "forwarded temporary write does not precede its use")
            value, consumer = owner["sources"][0], owner

        # A whole-value path cannot certify an input also exposed through a
        # reference or an unrelated call. This bounded rule makes no alias claim.
        for flow in flows:
            if not any(_place(source).local in visited for source in flow["sources"]):
                continue
            is_validator = (flow["rvalue"] == "call-result"
                            and flow["block"] == call["block"]
                            and flow["statement"] == call["statement"])
            _require(is_validator or (flow["rvalue"] == "Use"
                     and len(flow["sources"]) == 1
                     and not _place(flow["sources"][0]).projection),
                     "forwarded input escapes through an unowned operation")


def argument_forwarding_problems(raw, entry, stage, slots):
    """An empty problem list proves only the supplied forwarding obligations."""
    try:
        _verify(raw, entry, stage, slots)
        return []
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return [f"native argument forwarding failed: {error}"]
