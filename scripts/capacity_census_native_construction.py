"""Resolve exact, direct constructor owners in current native compiler evidence.

The caller selects governing methods from the actual registered boundary and
conversion identities. This helper rejects raw constructor function values and
construction in closures. It does not prove validation behavior, private field
shape, execution, or final transport authority.
"""

from capacity_census_native_flow import (
    ConstructorOwnership,
    DefinitionIdentity,
    NativeFlowEvidenceError,
    _native_body_identity,
    _native_constructor_evidence,
)


def _require(condition, message):
    if not condition:
        raise NativeFlowEvidenceError(message)


def constructor_scope_ownership(raw, carrier, roots):
    """Check all carrier aggregates and raw-constructor function-value uses.

    ``roots`` is a nonempty set of independently resolved governing function
    identities. Each must construct the carrier directly. Nested functions,
    closures and function-valued constructors inherit no permission from their
    lexical parent; neither paths nor helper names establish ownership.
    """
    _require(isinstance(carrier, DefinitionIdentity)
             and isinstance(roots, (set, frozenset)) and roots
             and all(isinstance(root, DefinitionIdentity) for root in roots),
             "invalid constructor scope policy")
    instances, definitions, exposed = _native_constructor_evidence(raw)
    _require(isinstance(raw.get("aggregates"), list),
             "constructor scope requires complete native aggregate evidence")
    _require(carrier not in exposed, "guarded constructor function value is exposed")
    for root in roots:
        _require(root in definitions, "missing governing constructor body")
        _require(definitions[root][0] in {"Fn", "AssocFn"},
                 "constructor scope root must be a governing function")

    observed = set()
    for aggregate in raw["aggregates"]:
        _require(isinstance(aggregate, dict), "malformed native aggregate")
        if DefinitionIdentity.from_record(aggregate.get("definition")) != carrier:
            continue
        key = _native_body_identity(aggregate.get("caller"))
        _require(key in instances, "constructor caller differs from its actual body")
        owner, kind, _, _, _ = key
        _require(kind in {"Fn", "AssocFn"},
                 "constructor must belong to a direct governing function body")
        _require(owner in roots, "constructor outside its exact governing scope")
        observed.add(owner)
    _require(observed == roots, "missing required constructor scope")
    owners = frozenset(observed)
    return ConstructorOwnership(carrier, owners, owners)
