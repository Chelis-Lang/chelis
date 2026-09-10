#!/usr/bin/env python3
"""Check bounded native adapter obligations in format-4 compiler evidence.

This module consumes a provenance-bound compiler record. It verifies exact
constructor ownership and one reviewed direct-call/place-flow chain. It does
not classify a binding, issue numeric authority, or claim general MIR dataflow
or alias soundness.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
from typing import Any, Iterable


class NativeFlowEvidenceError(ValueError):
    """The compiler record cannot support the bounded obligation."""


@dataclass(frozen=True, order=True)
class DefinitionIdentity:
    stable_crate_id: str
    def_id: str
    def_path_hash: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DefinitionIdentity":
        if not isinstance(record, dict):
            raise NativeFlowEvidenceError("definition is not an object")
        values = []
        for key in ("stable_crate_id", "def_id", "def_path_hash"):
            value = record.get(key)
            if not isinstance(value, str) or not value:
                raise NativeFlowEvidenceError(f"definition has invalid {key}")
            values.append(value)
        return cls(*values)


@dataclass(frozen=True)
class CallStage:
    declared: DefinitionIdentity
    resolved: DefinitionIdentity


@dataclass(frozen=True)
class ConstructorOwnership:
    carrier: DefinitionIdentity
    allowed_owners: frozenset[DefinitionIdentity]
    required_owners: frozenset[DefinitionIdentity]

    def __post_init__(self):
        if not self.allowed_owners:
            raise ValueError("constructor ownership needs an allowed owner")
        if not self.required_owners <= self.allowed_owners:
            raise ValueError("required constructor owner is not allowed")


@dataclass(frozen=True)
class NativeReceiver:
    """One exact immutable receiver whose fields already have owner obligations."""

    local: int
    carrier: DefinitionIdentity

    def __post_init__(self):
        if type(self.local) is not int or self.local < 1:
            raise ValueError("native receiver needs a positive input local")


@dataclass(frozen=True)
class AdapterFlowObligation:
    entry: DefinitionIdentity
    stages: tuple[CallStage, ...]
    constructors: tuple[ConstructorOwnership, ...]
    native_types: frozenset[DefinitionIdentity]
    success_variant: DefinitionIdentity | None = None
    control_calls: tuple[CallStage, ...] = ()
    receiver: NativeReceiver | None = None

    def __post_init__(self):
        if not self.stages:
            raise ValueError("native adapter chain needs at least one stage")
        carriers = [rule.carrier for rule in self.constructors]
        if len(carriers) != len(set(carriers)):
            raise ValueError("duplicate native constructor ownership rule")
        if not set(carriers) <= self.native_types:
            raise ValueError("constructor carrier is absent from native types")
        if self.receiver is not None and self.receiver.carrier not in self.native_types:
            raise ValueError("receiver carrier is absent from native types")


@dataclass(frozen=True, order=True)
class _Place:
    local: int
    projection: tuple[str, ...]


def _identity(record: Any) -> DefinitionIdentity:
    return DefinitionIdentity.from_record(record)


def _definition(value: Any) -> DefinitionIdentity:
    if not isinstance(value, dict):
        raise NativeFlowEvidenceError("missing containing definition")
    return _identity(value.get("definition"))


def _native_body_identity(value: Any) -> tuple:
    if not isinstance(value, dict):
        raise NativeFlowEvidenceError("missing constructor body evidence")
    identity = _identity(value.get("definition"))
    kind, ancestors = value.get("kind"), value.get("ancestors")
    substitutions, opened = value.get("substitutions"), value.get("open_type_or_const")
    if (not isinstance(kind, str) or not kind or not isinstance(ancestors, list)
            or not isinstance(substitutions, list) or type(opened) is not bool):
        raise NativeFlowEvidenceError("incomplete constructor body identity")
    parents = tuple(_identity(parent) for parent in ancestors)
    if identity in parents or len(parents) != len(set(parents)):
        raise NativeFlowEvidenceError("cyclic constructor body ancestry")
    return identity, kind, parents, json.dumps(substitutions, sort_keys=True), opened


def _native_constructor_evidence(raw: dict) -> tuple[set, dict, set]:
    """Join the mandatory raw-constructor census to its actual body instances.

    An exposed constructor is a capability, not proof that construction ran.
    Callers reject exposures of guarded carriers. Other exposures remain data;
    this helper grants no authority to their result types or numeric payloads.
    """
    if (not isinstance(raw, dict) or raw.get("format") != 4
            or raw.get("scope") != "native-bindings" or raw.get("errors") != []
            or not isinstance(raw.get("bodies"), list)
            or not isinstance(raw.get("constructor_uses"), list)):
        raise NativeFlowEvidenceError("native constructor_uses require complete format-4 evidence")
    instances, definitions = set(), {}
    for body in raw["bodies"]:
        key = _native_body_identity(body)
        identity, kind, parents, _, _ = key
        if key in instances:
            raise NativeFlowEvidenceError("duplicate constructor body instance")
        instances.add(key)
        lexical = kind, parents
        if identity in definitions and definitions[identity] != lexical:
            raise NativeFlowEvidenceError("conflicting constructor body ancestry")
        definitions[identity] = lexical

    required = {"definition", "carrier", "variant_definition", "kind", "arguments",
                "formal_inputs", "formal_result", "fields", "caller", "block",
                "statement", "operand_index", "context", "cast", "operand", "source"}
    observed, exposed = set(), set()
    for row in raw["constructor_uses"]:
        if not isinstance(row, dict) or set(row) != required:
            raise NativeFlowEvidenceError("incomplete native constructor use")
        caller = _native_body_identity(row["caller"])
        if caller not in instances:
            raise NativeFlowEvidenceError("constructor caller differs from its actual body")
        if (any(type(row[k]) is not int or row[k] < 0
                for k in ("block", "statement", "operand_index"))
                or row["context"] not in {"statement", "terminator"}):
            raise NativeFlowEvidenceError("invalid native constructor use location")
        occurrence = caller, row["block"], row["statement"], row["operand_index"]
        if occurrence in observed:
            raise NativeFlowEvidenceError("duplicate native constructor use occurrence")
        observed.add(occurrence)
        if row["kind"] not in {"Ctor(Struct, Fn)", "Ctor(Variant, Fn)"}:
            raise NativeFlowEvidenceError("native use is not a raw function constructor")
        _identity(row["definition"])
        _identity(row["variant_definition"])
        carrier = _identity(row["carrier"])
        result = row["formal_result"]
        if (not isinstance(result, dict) or not isinstance(result.get("shape"), dict)
                or result["shape"].get("tag") != "nominal"
                or _identity(result.get("nominal")) != carrier
                or _identity(result["shape"].get("definition")) != carrier):
            raise NativeFlowEvidenceError("constructor result differs from its carrier")
        inputs, fields = row["formal_inputs"], row["fields"]
        if (not isinstance(row["arguments"], list) or not isinstance(inputs, list)
                or not isinstance(fields, list) or len(inputs) != len(fields)):
            raise NativeFlowEvidenceError("constructor fields differ from its signature")
        field_ids = set()
        for item, formal in zip(fields, inputs):
            if (not isinstance(item, dict) or set(item) != {"definition", "type"}
                    or not isinstance(item["type"], dict) or not isinstance(formal, dict)
                    or not isinstance(formal.get("shape"), dict)
                    or item["type"].get("shape") != formal["shape"]):
                raise NativeFlowEvidenceError("constructor field type differs from its signature")
            field_id = _identity(item["definition"])
            if field_id in field_ids:
                raise NativeFlowEvidenceError("duplicate native constructor field")
            field_ids.add(field_id)
        cast = row["cast"]
        if cast is not None and (not isinstance(cast, dict) or set(cast) != {"kind", "target"}
                                 or not isinstance(cast["kind"], str)
                                 or not isinstance(cast["target"], dict)):
            raise NativeFlowEvidenceError("malformed native constructor cast")
        if not isinstance(row["operand"], dict) or not isinstance(row["source"], dict):
            raise NativeFlowEvidenceError("missing native constructor operand or source")
        exposed.add(carrier)
    return instances, definitions, exposed


def _place(value: Any) -> _Place:
    if not isinstance(value, dict):
        raise NativeFlowEvidenceError("place is not an object")
    local = value.get("local")
    projection = value.get("projection")
    if type(local) is not int or local < 0:
        raise NativeFlowEvidenceError("place has invalid local")
    if not isinstance(projection, list) or not all(
        isinstance(part, str) for part in projection
    ):
        raise NativeFlowEvidenceError("place has invalid projection")
    return _Place(local, tuple(projection))


def _contains_definition(value: Any, identities: frozenset[DefinitionIdentity]) -> bool:
    if isinstance(value, dict):
        if {"stable_crate_id", "def_id", "def_path_hash"} <= value.keys():
            if _identity(value) in identities:
                return True
        return any(_contains_definition(child, identities) for child in value.values())
    if isinstance(value, list):
        return any(_contains_definition(child, identities) for child in value)
    return False


def _call_identity(call: dict[str, Any]) -> tuple[DefinitionIdentity, DefinitionIdentity] | None:
    callee = call.get("callee")
    if not isinstance(callee, dict):
        return None
    resolved = callee.get("resolved")
    if not isinstance(resolved, dict):
        return None
    return _definition(callee), _definition(resolved)


def _native_bearing_call(
    call: dict[str, Any], native_types: frozenset[DefinitionIdentity]
) -> bool:
    return any(
        _contains_definition(call.get(key), native_types)
        for key in (
            "callable_type",
            "arguments",
            "formal_inputs",
            "formal_result",
            "destination",
        )
    )


def _direct_native_value(
    value: Any, native_types: frozenset[DefinitionIdentity]
) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("type"), dict):
        raise NativeFlowEvidenceError("place has invalid type evidence")
    nominal = value["type"].get("nominal")
    return isinstance(nominal, dict) and _identity(nominal) in native_types


def _reachable(adjacency: dict[int, set[int]], start: int) -> set[int]:
    found = set()
    pending = [start]
    while pending:
        block = pending.pop()
        if block in found:
            continue
        found.add(block)
        pending.extend(adjacency.get(block, ()))
    return found


def _dominators(adjacency: dict[int, set[int]], entry: int) -> dict[int, set[int]]:
    reachable = _reachable(adjacency, entry)
    predecessors = {block: set() for block in reachable}
    for source, targets in adjacency.items():
        if source not in reachable:
            continue
        for target in targets:
            if target in reachable:
                predecessors[target].add(source)
    dominators = {block: set(reachable) for block in reachable}
    dominators[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in sorted(reachable - {entry}):
            incoming = predecessors[block]
            common = set.intersection(*(dominators[pred] for pred in incoming)) if incoming else set()
            updated = {block} | common
            if updated != dominators[block]:
                dominators[block] = updated
                changed = True
    return dominators


def _feeds(available: _Place, source: _Place) -> bool:
    return (
        available.local == source.local
        and len(available.projection) <= len(source.projection)
        and source.projection[: len(available.projection)] == available.projection
    )


def _place_reaches(
    start: _Place,
    target: _Place,
    edges: Iterable[tuple[_Place, _Place, int]],
    allowed_blocks: set[int],
) -> bool:
    usable = [edge for edge in edges if edge[2] in allowed_blocks]
    pending = deque([start])
    found = {start}
    while pending:
        available = pending.popleft()
        if _feeds(available, target):
            return True
        for source, destination, _ in usable:
            if _feeds(available, source) and destination not in found:
                found.add(destination)
                pending.append(destination)
    return False


def _branch_value_reaches_return(
    start: _Place,
    origin: int,
    return_place: _Place,
    blocks: dict[int, dict[str, Any]],
    adjacency: dict[int, set[int]],
    dominators: dict[int, set[int]],
    edges: Iterable[tuple[_Place, _Place, int]],
) -> bool:
    """Prove a value reaches the return place inside its own CFG branch.

    Rust MIR commonly joins Result::Err and Result::Ok branches at one Return
    block.  The successful branch therefore cannot globally dominate Return.
    Keep value propagation inside blocks dominated by the successful origin,
    then separately require that the origin can reach a Return terminator.
    """

    if origin not in adjacency or origin not in dominators:
        return False
    reachable = _reachable(adjacency, origin)
    if not any(
        block in reachable and record.get("terminator") == "Return"
        for block, record in blocks.items()
    ):
        return False
    branch_blocks = {
        block
        for block in reachable
        if block in dominators and origin in dominators[block]
    }
    return _place_reaches(start, return_place, edges, branch_blocks)


def _blocks(body: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, set[int]]]:
    records = body.get("blocks")
    if not isinstance(records, list) or not records:
        raise NativeFlowEvidenceError("entry body has no blocks")
    blocks = {}
    adjacency = {}
    for record in records:
        if not isinstance(record, dict) or type(record.get("block")) is not int:
            raise NativeFlowEvidenceError("entry body has malformed block")
        block = record["block"]
        successors = record.get("successors")
        if block in blocks or not isinstance(successors, list) or not all(
            type(target) is int for target in successors
        ):
            raise NativeFlowEvidenceError("entry body has malformed CFG")
        blocks[block] = record
        adjacency[block] = set(successors)
    if 0 not in blocks or any(
        target not in blocks for targets in adjacency.values() for target in targets
    ):
        raise NativeFlowEvidenceError("entry body has incomplete CFG")
    return blocks, adjacency


def _flow_edges(
    flows: list[Any],
    entry: DefinitionIdentity,
    sanctioned_call_destinations: set[_Place],
) -> tuple[list[tuple[_Place, _Place, int]], list[dict[str, Any]]]:
    edges = []
    selected = []
    for flow in flows:
        if not isinstance(flow, dict):
            raise NativeFlowEvidenceError("malformed native flow")
        if _definition(flow.get("caller")) != entry:
            continue
        block = flow.get("block")
        sources = flow.get("sources")
        if type(block) is not int or not isinstance(sources, list):
            raise NativeFlowEvidenceError("malformed entry flow")
        destination = _place(flow.get("destination"))
        if (
            flow.get("rvalue") == "call-result"
            and destination not in sanctioned_call_destinations
        ):
            selected.append(flow)
            continue
        for source in sources:
            edges.append((_place(source), destination, block))
        selected.append(flow)
    return edges, selected


def _implementation_problems(
    raw: dict[str, Any], obligation: AdapterFlowObligation
) -> list[str]:
    if not isinstance(raw, dict):
        raise NativeFlowEvidenceError("native flow evidence is not an object")
    if raw.get("format") != 4 or raw.get("scope") != "native-bindings":
        raise NativeFlowEvidenceError("native flow needs format-4 native-bindings evidence")
    for key in ("bodies", "calls", "aggregates", "constructor_uses", "flows", "errors"):
        if not isinstance(raw.get(key), list):
            raise NativeFlowEvidenceError(f"native flow evidence has invalid {key}")
    if raw["errors"]:
        return [f"collector errors prevent native flow verification: {raw['errors']!r}"]

    problems = []
    rules = {rule.carrier: rule for rule in obligation.constructors}
    instances, _, exposed = _native_constructor_evidence(raw)
    for carrier in sorted(exposed & rules.keys()):
        problems.append(f"guarded constructor function value is exposed: {carrier.def_path_hash}")
    observed_owners: dict[DefinitionIdentity, set[DefinitionIdentity]] = defaultdict(set)
    allowed_aggregate_destinations = []
    for aggregate in raw["aggregates"]:
        if not isinstance(aggregate, dict):
            raise NativeFlowEvidenceError("malformed native aggregate")
        _identity(aggregate.get("variant_definition"))
        if not isinstance(aggregate.get("arguments"), list):
            raise NativeFlowEvidenceError("native aggregate has malformed arguments")
        carrier = _identity(aggregate.get("definition"))
        rule = rules.get(carrier)
        if rule is None:
            continue
        caller = _native_body_identity(aggregate.get("caller"))
        if caller not in instances:
            raise NativeFlowEvidenceError("constructor caller differs from its actual body")
        owner, kind, _, _, _ = caller
        observed_owners[carrier].add(owner)
        if kind not in {"Fn", "AssocFn"}:
            problems.append("constructor must belong to a direct governing function body")
        elif owner not in rule.allowed_owners:
            problems.append(
                f"constructor owner is not allowed for {carrier.def_path_hash}: {owner.def_path_hash}"
            )
        elif owner == obligation.entry:
            allowed_aggregate_destinations.append(_place(aggregate.get("destination")))
    for rule in obligation.constructors:
        missing = rule.required_owners - observed_owners[rule.carrier]
        for owner in sorted(missing):
            problems.append(
                f"missing required constructor owner for {rule.carrier.def_path_hash}: {owner.def_path_hash}"
            )

    entry_bodies = []
    for body in raw["bodies"]:
        if not isinstance(body, dict):
            raise NativeFlowEvidenceError("malformed native body")
        if _identity(body.get("definition")) == obligation.entry:
            entry_bodies.append(body)
    if len(entry_bodies) != 1:
        return problems + [f"expected exactly one entry body, found {len(entry_bodies)}"]
    entry_body = entry_bodies[0]
    blocks, adjacency = _blocks(entry_body)
    reachable_from_entry = _reachable(adjacency, 0)
    dominators = _dominators(adjacency, 0)

    formal_inputs = entry_body.get("formal_inputs")
    if not isinstance(formal_inputs, list):
        raise NativeFlowEvidenceError("entry body has malformed formal inputs")
    seen_inputs = set()
    receiver_place = None
    for item in formal_inputs:
        if not isinstance(item, dict) or type(item.get("local")) is not int:
            raise NativeFlowEvidenceError("entry body has malformed input local")
        local = item["local"]
        if local < 1 or local in seen_inputs:
            raise NativeFlowEvidenceError("entry body has invalid or repeated input local")
        seen_inputs.add(local)
        if obligation.receiver is not None and local == obligation.receiver.local:
            ty = item.get("type", {})
            if not isinstance(ty, dict):
                raise NativeFlowEvidenceError("native receiver has malformed type")
            shape = ty.get("shape", {})
            if not isinstance(shape, dict):
                raise NativeFlowEvidenceError("native receiver has malformed shape")
            inner = shape.get("inner", {})
            if not isinstance(inner, dict):
                raise NativeFlowEvidenceError("native receiver has malformed referent")
            if (
                ty.get("open") is not False
                or shape.get("tag") != "reference"
                or shape.get("mutable") is not False
                or inner.get("tag") != "nominal"
                or inner.get("arguments") != []
                or _identity(inner.get("definition")) != obligation.receiver.carrier
            ):
                problems.append("native input differs from its exact shared receiver")
            else:
                receiver_place = _Place(local, ())
        elif _contains_definition(item, obligation.native_types):
            problems.append("unaccounted native-bearing entry input")
    if obligation.receiver is not None and obligation.receiver.local not in seen_inputs:
        problems.append("missing exact native receiver input local")

    entry_calls = []
    for call in raw["calls"]:
        if not isinstance(call, dict):
            raise NativeFlowEvidenceError("malformed native call")
        if _definition(call.get("caller")) == obligation.entry:
            entry_calls.append(call)

    stage_calls = []
    claimed = set()
    for index, stage in enumerate(obligation.stages):
        matches = [
            (call_index, call)
            for call_index, call in enumerate(entry_calls)
            if call.get("kind") == "resolved"
            and _call_identity(call) == (stage.declared, stage.resolved)
        ]
        if len(matches) != 1:
            problems.append(
                f"missing exact stage {index} or ambiguous exact stage: found {len(matches)}"
            )
            stage_calls.append(None)
            continue
        call_index, call = matches[0]
        claimed.add(call_index)
        stage_calls.append(call)

    required_controls = Counter(obligation.control_calls)
    observed_controls: Counter[CallStage] = Counter()
    control_calls = []
    for call_index, call in enumerate(entry_calls):
        identity = _call_identity(call)
        if call.get("kind") != "resolved" or identity is None:
            continue
        stage = CallStage(*identity)
        if observed_controls[stage] >= required_controls[stage]:
            continue
        observed_controls[stage] += 1
        claimed.add(call_index)
        control_calls.append(call)
    for control, count in sorted(
        (item for item in required_controls.items() if observed_controls[item[0]] != item[1]),
        key=lambda item: item[0].declared,
    ):
        problems.append(
            "missing exact Result/Try control call "
            f"{control.declared.def_path_hash}: expected {count}, "
            f"found {observed_controls[control]}"
        )

    for index, call in enumerate(entry_calls):
        if index not in claimed and _native_bearing_call(call, obligation.native_types):
            problems.append(
                f"unaccounted native-bearing call at bb{call.get('block')}: {call.get('kind')!r}"
            )

    complete_stages = [call for call in stage_calls if call is not None]
    sanctioned_call_destinations = {
        _place(call.get("destination")) for call in complete_stages + control_calls
    }
    edges, entry_flows = _flow_edges(
        raw["flows"], obligation.entry, sanctioned_call_destinations
    )
    for call in complete_stages:
        if type(call.get("block")) is not int or call["block"] not in reachable_from_entry:
            problems.append("required stage is unreachable from the entry block")
        if call.get("target") is None or type(call.get("target")) is not int:
            problems.append("required stage has no normal continuation target")

    if len(complete_stages) == len(obligation.stages):
        if receiver_place is not None:
            first = complete_stages[0]
            arguments = [
                _place(argument["place"])
                for argument in first.get("arguments", [])
                if isinstance(argument, dict) and argument.get("place") is not None
            ]
            if not any(
                _place_reaches(receiver_place, argument, edges,
                               dominators.get(first["block"], set()))
                for argument in arguments
            ):
                problems.append("native receiver does not reach the first stage")
        for index, (previous, current) in enumerate(
            zip(complete_stages, complete_stages[1:]), start=1
        ):
            previous_target = previous["target"]
            current_block = current["block"]
            if (
                previous_target not in adjacency
                or current_block not in dominators
                or previous_target not in dominators[current_block]
                or current_block not in _reachable(adjacency, previous_target)
            ):
                problems.append(f"stage order is not dominated at stage {index}")
                continue
            allowed_blocks = _reachable(adjacency, previous_target) & dominators[current_block]
            previous_destination = _place(previous.get("destination"))
            arguments = [
                _place(argument["place"])
                for argument in current.get("arguments", [])
                if isinstance(argument, dict) and argument.get("place") is not None
            ]
            if not any(
                _place_reaches(previous_destination, argument, edges, allowed_blocks)
                for argument in arguments
            ):
                problems.append(f"stage place provenance is broken at stage {index}")

        final = complete_stages[-1]
        final_target = final["target"]
        return_place = _Place(0, ())
        if obligation.success_variant is None:
            proven_return = False
            if final_target in adjacency:
                for block, record in blocks.items():
                    if (
                        record.get("terminator") == "Return"
                        and block in dominators
                        and final_target in dominators[block]
                    ):
                        allowed_blocks = _reachable(adjacency, final_target) & dominators[block]
                        if _place_reaches(
                            _place(final.get("destination")),
                            return_place,
                            edges,
                            allowed_blocks,
                        ):
                            proven_return = True
                            break
            if not proven_return:
                problems.append("final stage does not dominate a place-proven returned value")
        else:
            direct_return = _branch_value_reaches_return(
                _place(final.get("destination")),
                final_target,
                return_place,
                blocks,
                adjacency,
                dominators,
                edges,
            )
            successes = [
                aggregate
                for aggregate in raw["aggregates"]
                if _definition(aggregate.get("caller")) == obligation.entry
                and _identity(aggregate.get("variant_definition"))
                == obligation.success_variant
                and _contains_definition(aggregate, obligation.native_types)
                and aggregate.get("block") in reachable_from_entry
            ]
            if not successes and not direct_return:
                problems.append("no exact native-bearing successful return aggregate")
            for success in successes:
                block = success.get("block")
                if type(block) is not int:
                    raise NativeFlowEvidenceError("success aggregate has invalid block")
                operand_places = [
                    _place(field["operand"]["place"])
                    for field in success.get("fields", [])
                    if isinstance(field, dict)
                    and isinstance(field.get("operand"), dict)
                    and field["operand"].get("place") is not None
                ]
                stage_proven = (
                    final_target in adjacency
                    and block in dominators
                    and final_target in dominators[block]
                    and any(
                        _place_reaches(
                            _place(final.get("destination")),
                            operand,
                            edges,
                            _reachable(adjacency, final_target) & dominators[block],
                        )
                        for operand in operand_places
                    )
                )
                returned = stage_proven and _branch_value_reaches_return(
                    _place(success.get("destination")),
                    block,
                    return_place,
                    blocks,
                    adjacency,
                    dominators,
                    edges,
                )
                if not stage_proven or not returned:
                    problems.append(
                        f"successful return bypasses final stage at bb{block}"
                    )

    sanctioned_seeds = [
        _place(call.get("destination")) for call in complete_stages
    ] + allowed_aggregate_destinations
    if receiver_place is not None:
        sanctioned_seeds.append(receiver_place)
    all_blocks = set(blocks)
    for flow in entry_flows:
        destination = flow.get("destination")
        if not _direct_native_value(destination, obligation.native_types):
            continue
        place = _place(destination)
        if not any(_place_reaches(seed, place, edges, all_blocks) for seed in sanctioned_seeds):
            problems.append(
                f"unaccounted native-bearing flow to local {place.local}"
            )

    return problems


def native_flow_problems(
    raw: dict[str, Any], obligation: AdapterFlowObligation
) -> list[str]:
    """Return deterministic obligation failures; an empty list is not authority."""

    try:
        return _implementation_problems(raw, obligation)
    except (KeyError, TypeError, ValueError) as error:
        return [f"malformed native flow evidence: {error}"]
