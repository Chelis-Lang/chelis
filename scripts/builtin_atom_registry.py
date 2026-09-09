"""Read the normative, identity-keyed builtin authority registry.

Discovery is supplied by the compiled declarations, independently of this
registry. No registration is inferred from a name found in prose.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

REGISTRY = "builtin_semantic_identities.md"
ATOM = re.compile(r"^> \*\*(\[05-OP-[1-9][0-9]*\])\*\*", re.MULTILINE)
IDENTITY = re.compile(r"(?:Numeric|Container|Boundary):[A-Za-z][A-Za-z0-9_]*:[A-Za-z][A-Za-z0-9_]*\Z")
ROW = re.compile(r"\| `([^`]+)` \| (\[05-OP-[1-9][0-9]*\]) \|\Z")
INCORPORATION = (
    "The [builtin identity registry](registry/builtin_semantic_identities.md) "
    "is incorporated by reference into each numbered operation atom named in its "
    "Atom column."
)


class RegistryError(ValueError):
    pass


def named_callables(block: str) -> set[str]:
    return set(re.findall(r"`([A-Za-z][A-Za-z0-9_]*)(?:`|\()", block))


def atom_blocks(spec: str) -> dict[str, str]:
    result: dict[str, str] = {}
    current: str | None = None
    for line in spec.splitlines():
        match = ATOM.match(line)
        if match:
            current = match[1]
            if current in result:
                raise RegistryError(f"duplicate normative definition {current}")
            result[current] = line + "\n"
        elif current is not None and line.startswith(">"):
            result[current] += line + "\n"
        else:
            current = None
    return result


def parse_registry(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|") or line in ("| Identity | Atom |", "|---|---|"):
            continue
        match = ROW.fullmatch(line)
        if not match or not IDENTITY.fullmatch(match[1]):
            raise RegistryError(f"invalid identity registration: {line}")
        identity, atom = match.groups()
        if identity in rows:
            raise RegistryError(f"duplicate registration {identity}")
        rows[identity] = atom
    return rows


def validate(discovered: Iterable[str], table: str, spec: str,
             generated: set[str]) -> dict[str, str]:
    identities = tuple(discovered)
    if not identities:
        raise RegistryError("empty discovery universe")
    counts = Counter(identities)
    if any(n != 1 for n in counts.values()):
        raise RegistryError("duplicate discovery identities")
    if any(not IDENTITY.fullmatch(i) for i in identities):
        raise RegistryError("invalid discovered identity")
    rows = parse_registry(table)
    missing, stale = set(identities) - rows.keys(), rows.keys() - set(identities)
    if missing or stale:
        raise RegistryError(f"missing registrations={sorted(missing)}; stale registrations={sorted(stale)}")
    blocks = atom_blocks(spec)
    preamble = " ".join(spec.split("> **[05-OP-", 1)[0].split())
    if preamble.count(INCORPORATION) != 1:
        raise RegistryError("the chapter must incorporate the builtin identity registry once")
    for identity, atom in rows.items():
        if atom not in blocks:
            raise RegistryError(f"missing normative definition {atom} for {identity}")
        if atom not in generated:
            raise RegistryError(f"missing generated membership {atom}")
        block = blocks[atom]
        # The table authors the identity-to-atom relation. Its normative atom
        # must also name the exact callable; unquoted prose and namespaced
        # lookalikes cannot satisfy this structural check. Substantive contract
        # clauses are checked separately by builtin_atom_semantic_contracts.
        operation = identity.split(":")[1]
        if operation not in named_callables(block):
            raise RegistryError(f"{atom} does not govern the callable {identity}")
    return rows
