"""Resolve exact constructor scopes in current native compiler evidence.

The caller must select governing methods from the actual registered boundary
and conversion identities. This helper proves lexical construction ownership,
not validation behavior, reachability, or final transport authority.
"""

import json

from capacity_census_native_flow import (
    ConstructorOwnership,
    DefinitionIdentity,
    NativeFlowEvidenceError,
)


def _require(condition, message):
    if not condition:
        raise NativeFlowEvidenceError(message)


def _identity(value):
    return DefinitionIdentity.from_record(value)


def _body_identity(value):
    _require(isinstance(value, dict), "missing constructor body evidence")
    identity = _identity(value.get("definition"))
    kind = value.get("kind")
    ancestors = value.get("ancestors")
    substitutions = value.get("substitutions")
    opened = value.get("open_type_or_const")
    _require(isinstance(kind, str) and kind
             and isinstance(ancestors, list)
             and isinstance(substitutions, list) and type(opened) is bool,
             "incomplete constructor body identity")
    parents = tuple(_identity(parent) for parent in ancestors)
    _require(identity not in parents and len(parents) == len(set(parents)),
             "cyclic constructor body ancestry")
    return identity, kind, parents, json.dumps(substitutions, sort_keys=True), opened


def constructor_scope_ownership(raw, carrier, roots):
    """Return one bounded ownership rule after checking every carrier aggregate.

    ``roots`` maps each independently resolved governing function identity to
    whether its lexical closures may construct this carrier. Every named root
    must own an observed construction. Nested functions do not inherit a root's
    permission. No path/name prefix or inferred impl ordinal grants ownership.
    """
    _require(isinstance(carrier, DefinitionIdentity)
             and isinstance(roots, dict) and roots
             and all(isinstance(root, DefinitionIdentity) and type(closures) is bool
                     for root, closures in roots.items()),
             "invalid constructor scope policy")
    _require(isinstance(raw, dict) and raw.get("format") == 3
             and raw.get("scope") == "native-bindings"
             and raw.get("errors") == []
             and isinstance(raw.get("bodies"), list)
             and isinstance(raw.get("aggregates"), list),
             "constructor scope requires complete native evidence without errors")

    # Substitutions distinguish actual compiler instances. Lexical ancestry
    # belongs to the defining item and must agree across those instances.
    instances = set()
    definitions = {}
    for body in raw["bodies"]:
        key = _body_identity(body)
        identity, kind, parents, _, _ = key
        _require(key not in instances, "duplicate constructor body instance")
        instances.add(key)
        lexical = (kind, parents)
        _require(identity not in definitions or definitions[identity] == lexical,
                 "conflicting constructor body ancestry")
        definitions[identity] = lexical
    for root in roots:
        _require(root in definitions, "missing governing constructor body")
        _require(definitions[root][0] in {"Fn", "AssocFn"},
                 "constructor scope root must be a governing function")

    observed_roots = set()
    observed_owners = set()
    for aggregate in raw["aggregates"]:
        _require(isinstance(aggregate, dict), "malformed native aggregate")
        if _identity(aggregate.get("definition")) != carrier:
            continue
        key = _body_identity(aggregate.get("caller"))
        _require(key in instances, "constructor caller differs from its actual body")
        identity, kind, parents, _, opened = key
        _require(not opened, "open constructor instance cannot establish ownership")
        owner = identity
        nested = False
        while identity not in roots and kind == "Closure":
            _require(parents, "missing constructor closure parent")
            parent, *tail = parents
            _require(parent in definitions, "missing constructor closure parent body")
            kind, actual_parents = definitions[parent]
            _require(actual_parents == tuple(tail),
                     "constructor closure ancestry differs from its parent body")
            identity, parents = parent, actual_parents
            nested = True
        _require(identity in roots and (not nested or roots[identity]),
                 "constructor outside its exact governing scope")
        observed_roots.add(identity)
        observed_owners.add(owner)
    _require(observed_roots == roots.keys(), "missing required constructor scope")
    owners = frozenset(observed_owners)
    return ConstructorOwnership(carrier, owners, owners)
