#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 4B semantic/schema freeze oracle.

Acceptance is exit 0 with the final line ``DTYPE PHASE 4B ORACLE: PASS``.

Usage:

    .venv/bin/python scripts/dtype_phase4b_oracle.py
"""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PASS_LINE = "DTYPE PHASE 4B ORACLE: PASS"
CONTRACT_FILES = (
    "spec/02-surf-syntax.md",
    "spec/03-deep-syntax.md",
    "spec/04-type-system.md",
    "spec/05-risc-primitives.md",
    "spec/design/capability_table.md",
    "spec/design/dtype_semantics.md",
    "spec/design/loud_unsupported.md",
    "spec/design/spec_provenance.md",
    "spec/design/remediation_roadmap.md",
    "docs/investigations/remediation_status_2026_08_04.md",
)
OP_ATOM = re.compile(
    r"^> \*\*\[05-OP-(\d+)\]\*\* `([a-z0-9_]+)", re.MULTILINE
)
ATOM_START = re.compile(
    r"^> \*\*\[(\d{2}-[A-Z]+-\d+)\]\*\*", re.MULTILINE
)
EXPECTED_PHASE4B_OPS = {
    11: "mean",
    12: "max_reduce",
    13: "min_reduce",
    14: "prod_reduce",
    15: "argmax_reduce",
    16: "argmin_reduce",
    17: "wrap_add",
    18: "wrap_sub",
    19: "wrap_mul",
    20: "is_nan",
    21: "is_finite",
    22: "is_infinite",
    23: "cast_saturate",
    24: "cast_wrap",
    25: "to_string",
    26: "and",
    27: "or",
    28: "not",
}

FROZEN_ATOM_DIGESTS = {
    "04-NUM-4": "b9f34c7d9113c094338a828eca087cacdca1c40321e930631eaf874b16ee3b3f",
    "04-NUM-16": "55a463dbe522a0b653255742568038a375cee2cf3b9c113cc53d7b75953b7dfd",
    "05-OP-11": "5723ecadeb87782d930839d557545eaa90d73b1204ba19e5e40f0c3242faef39",
    "05-OP-12": "05c3af79686730379892ad267bfb4b35451213c138cc3f4bae7e123591b377e3",
    "05-OP-13": "52a071e1e02febb3208f15d1b913de8370577fb7649f127d578c4619c7e8df31",
    "05-OP-14": "7da6a49016d65e63223f270f0ef03c091f8ec893e2698c17a1fb5f3e0ca759c8",
    "05-OP-15": "a9ad8cf1423e99bf092e6dd52ce273555cda77b4beb52ff3a78564db1800933e",
    "05-OP-16": "ddced357a351860e42537449d526dfa3ba8cda0a2aa28073d3ea5b81eae2008f",
    "05-OP-17": "1e4cc1b02acf4d527f3396a8acd49083b63bdc877270129b311bce6e3f7bb54d",
    "05-OP-18": "6cf5eaa4e1ab068d667ea1d8cc26ba366694329669cba39443a4878a6d04714a",
    "05-OP-19": "d84cd2202079d852ba918b99e2bae1c964650362dee200942a9addba954bd5e5",
    "05-OP-20": "6facebb9f20f5072eb9e64b568161a9b968a526fd91107cbb87e0bdbe86d2d2f",
    "05-OP-21": "ced775b654a61c4d36d2313191145b3543e55ef75825a5066e04b3c114437545",
    "05-OP-22": "f7c7c00b0fbea5176eb3427b517f5fb9f7434e24caaacd86fc1408455658329a",
    "05-OP-23": "bfc8198f2940f9d81c4dd3069568058974a39c1ddb37d0e51aadbfde226eb8bc",
    "05-OP-24": "d9210188e6698e34ce8a82008db7eff1a34e674b5f92ddf63c77db5fdd5bf79f",
    "05-OP-25": "8f400b1b3a5c1ba7a9f0f717fc6b41612949c708a21c053ef315883ab03652fb",
    "05-OP-26": "90050a454489c33ba0afb9caa41763591f22461eca947dd3525109c97976362f",
    "05-OP-27": "03a81560ae84cb4dd151e57da34d117a9e616a33700957796edea98c2afaf82f",
    "05-OP-28": "9eb81ed515be3e016371f951a75a3b65c4bae2cd8bfbc8de22c510f8e71be56b",
}

# The markers are part of the freeze contract: each must occur exactly once,
# and the end marker is excluded from the digest. Digests are not a self-bless
# mechanism. An intentional change owes the owning spec/design update, every
# consuming contract, and an adversarial mutation before this manifest moves.
FROZEN_REGION_DIGESTS = {
    "numeric value semantics": (
        "spec/04-type-system.md",
        "## 9. Numeric Value Semantics",
        "## 10. Checker Totality",
        "fb4c271dd4701058458ebb9f0bf2847e7d831bc71ee1a090634eb0a4b57ce5f9",
    ),
    "numeric primitive contracts": (
        "spec/05-risc-primitives.md",
        "### 2.1 Elementwise Binary",
        "### 2.4 Movement",
        "907259b6cf97409e4a8a82f2c6d6bed64bf33bce73a151076f24c00681427a6d",
    ),
    "logical builtin contract": (
        "spec/05-risc-primitives.md",
        "### 3.2 Comparison and Logical Operations",
        "### 3.3 Activation Functions",
        "9036535a7ae1af2dfa364eb580a55b0bcb822c0c9a79bc48cc7f1eeaae7a9d93",
    ),
    "multi-axis reduction contract": (
        "spec/04-type-system.md",
        "Multiple **named** axes may be reduced in one call",
        "**Unification (unitary).**",
        "7e8cd63034ea341f5a338ed39c0c339e3c80894fca2cf864b6a3c35d8faaff9f",
    ),
    "window extrema contract": (
        "spec/05-risc-primitives.md",
        "### 2.3.1 Windowed Reduction",
        "### 2.4 Movement",
        "84a55eed44492dfb38e76d81c11a40028d383712ac1ce1d2782a78e021be83ae",
    ),
    "to_string contract section": (
        "spec/05-risc-primitives.md",
        "### 3.6.3 Canonical value-to-string conversion",
        "### 3.7 Host-Lane Data I/O Numeric Operations",
        "b59c3728307e626648d48bb39a5e8c2d6fe487c3c30d7512f739afce38eb5aac",
    ),
    "named lossy cast section": (
        "spec/05-risc-primitives.md",
        "### 3.8 Named Lossy Cast Forms",
        "## 4. Standard Lowerings",
        "959b507e38b24860cf66bb31698e0e25acfbcdb193225a8ee5e17a70c9584c12",
    ),
    "capability schema": (
        "spec/design/capability_table.md",
        "## The two-table design",
        "## Seed dispositions the table must ship with",
        "6f33014f59d784d876c9b5af0f9238de3576502a9d2a0da4f9ef8be54dae352c",
    ),
    "capability seed dispositions": (
        "spec/design/capability_table.md",
        "## Seed dispositions the table must ship with",
        "## New numeric ops before the table lands (added 2026-07-30)",
        "0988ef53973af79d96f91d42ffdf3b8dd96be97c90ffe0676ed15aa06e61c6b4",
    ),
    "Phase 4 handoff": (
        "spec/design/dtype_semantics.md",
        "## Phase 4 - the capability table becomes the permanent guard",
        "## I1. Interlock with loud unsupported ([#730])",
        "ceaeef17c214b61fb634bd7e5feb0cc54df413fd92f0bfe56e9b1102f3b26732",
    ),
    "compiled stdlib consumer": (
        "spec/design/loud_unsupported.md",
        "### LU5 - derived compiled-stdlib acceptance corpus ([#955])",
        "### LU6 - exhaustive checked host-cast planning ([#1150])",
        "ca1ba0f88b6efcac400d21614dfb38746d6315b1719b11f2c679bb9609940b15",
    ),
    "provenance governed surfaces": (
        "spec/design/spec_provenance.md",
        "| Capability-row, Deep-tag, tolerance-row, and diagnostic citations |",
        "| OpenSpec proving inadequate as the trigger for provenance work |",
        "56675ebf30876d376c9eeb1817472381d9cfb0254522f57a3894366762da782c",
    ),
    "roadmap ownership": (
        "spec/design/remediation_roadmap.md",
        "| **v0.19.0 - grounded dtype storage break",
        "| **v0.20.0 - behavior-preserving permanent guards**",
        "7c3ac7dd809b1b69a41631b3738fd3d850ed17c3e775e8e0bb7c4c502c0a6347",
    ),
    "status dtype row": (
        "docs/investigations/remediation_status_2026_08_04.md",
        "| **#729 dtype semantics** |",
        "| **#730 loud unsupported** |",
        "b7f313b76d409e74f0b4aa9f13a1c28bb8f86335973bd1ff446794e0aec0e242",
    ),
}


class OracleError(RuntimeError):
    """The frozen Phase 4B contract is incomplete or internally inconsistent."""


def read(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise OracleError(f"cannot read {relative}: {error}") from error


def require(text: str, fragment: str, label: str, violations: list[str]) -> None:
    if fragment not in text:
        violations.append(f"missing {label}")


def require_all(
    text: str,
    requirements: tuple[tuple[str, str], ...],
    violations: list[str],
) -> None:
    for fragment, label in requirements:
        require(text, fragment, label, violations)


def atom_blocks(text: str) -> dict[str, str]:
    matches = list(ATOM_START.finditer(text))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.start() : end]
    return blocks


def strict_atom_block(text: str, atom: str) -> str:
    starts = [match for match in ATOM_START.finditer(text) if match.group(1) == atom]
    if len(starts) != 1:
        raise OracleError(
            f"frozen normative atom {atom} must occur exactly once, got {len(starts)}"
        )
    start = starts[0].start()
    end = start
    for line in text[start:].splitlines(keepends=True):
        if end > start and ATOM_START.match(line):
            break
        if not line.startswith(">"):
            break
        end += len(line)
    return text[start:end]


def normalize_frozen_block(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def frozen_digest(text: str) -> str:
    return hashlib.sha256(normalize_frozen_block(text).encode("utf-8")).hexdigest()


def frozen_region(text: str, start: str, end: str, label: str) -> str:
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise OracleError(
            f"frozen {label} markers must occur exactly once "
            f"(start={start_count}, end={end_count})"
        )
    start_index = text.index(start)
    end_index = text.index(end)
    if end_index <= start_index:
        raise OracleError(f"frozen {label} end marker precedes its start")
    return text[start_index:end_index]


def validate_frozen_contract(
    docs: dict[str, str], violations: list[str]
) -> None:
    for atom, expected in FROZEN_ATOM_DIGESTS.items():
        relative = (
            "spec/04-type-system.md" if atom.startswith("04-") else "spec/05-risc-primitives.md"
        )
        try:
            actual = frozen_digest(strict_atom_block(docs[relative], atom))
        except OracleError as error:
            violations.append(str(error))
            continue
        if actual != expected:
            violations.append(
                f"frozen normative atom {atom} digest mismatch: "
                f"expected {expected}, got {actual}"
            )

    for label, (relative, start, end, expected) in FROZEN_REGION_DIGESTS.items():
        try:
            block = frozen_region(docs[relative], start, end, label)
        except OracleError as error:
            violations.append(str(error))
            continue
        actual = frozen_digest(block)
        if actual != expected:
            violations.append(
                f"frozen {label} digest mismatch: expected {expected}, got {actual}"
            )


def require_atom(
    blocks: dict[str, str],
    atom: str,
    requirements: tuple[str, ...],
    violations: list[str],
) -> None:
    body = blocks.get(atom)
    if body is None:
        violations.append(f"missing normative atom [{atom}]")
        return
    normalized_body = " ".join(
        line[2:] if line.startswith("> ") else line
        for line in body.splitlines()
    )
    normalized_body = " ".join(normalized_body.split())
    for fragment in requirements:
        normalized_fragment = " ".join(
            line[2:] if line.startswith("> ") else line
            for line in fragment.splitlines()
        )
        normalized_fragment = " ".join(normalized_fragment.split())
        if normalized_fragment not in normalized_body:
            violations.append(f"[{atom}] missing semantic clause: {fragment}")


def validate_normative_contract(
    docs: dict[str, str], violations: list[str]
) -> None:
    spec02 = docs["spec/02-surf-syntax.md"]
    spec03 = docs["spec/03-deep-syntax.md"]
    spec04 = docs["spec/04-type-system.md"]
    spec05 = docs["spec/05-risc-primitives.md"]

    atoms = [(int(number), name) for number, name in OP_ATOM.findall(spec05)]
    counts = Counter(number for number, _name in atoms)
    duplicates = sorted(number for number, count in counts.items() if count != 1)
    if duplicates:
        violations.append(f"duplicate normative OP atoms: {duplicates}")
    for number, expected_name in EXPECTED_PHASE4B_OPS.items():
        matches = [name for atom_number, name in atoms if atom_number == number]
        actual_name = matches[0] if len(matches) == 1 else None
        if actual_name != expected_name:
            violations.append(
                f"[05-OP-{number}] must govern {expected_name}, got "
                f"{actual_name or 'missing'}"
            )
    if any(name == "cast_round" for _number, name in atoms):
        violations.append("cast_round must not be a normative operation atom")

    require_all(
        spec02,
        (("Surf has no infinity or NaN literal.", "Surf literal exclusion"),),
        violations,
    )
    require_all(
        spec03,
        (
            (
                "Canonical Deep contains no non-finite float literal.",
                "Deep literal exclusion",
            ),
        ),
        violations,
    )

    spec04_blocks = atom_blocks(spec04)
    require_atom(
        spec04_blocks,
        "04-NUM-16",
        (
            "active signed-integer or float source",
            "signed-integer target",
            "truncates a finite float toward zero",
            "`-inf` clamps to the target minimum",
            "`NaN` traps `Domain`",
            "congruent to the exact source modulo",
            "A rounding cast is not a separate operation",
            "`round(source)` followed by checked `cast`",
            "[04-NUM-15] governs the selected failure",
        ),
        violations,
    )
    require_all(
        spec04,
        (
            (
                "axis arguments are sorted by their positions in the original operand, "
                "from\nhighest position to lowest",
                "canonical variadic reduction order",
            ),
            (
                "composition owns the exact value, trap, NaN-selection, and adjoint "
                "behavior",
                "canonical composition authority",
            ),
            ("sum_result(p,a)", "sum result precision rule"),
            ("A typed operation-precondition guard", "typed reduction guard"),
        ),
        violations,
    )

    blocks = atom_blocks(spec05)
    atom_requirements: dict[str, tuple[str, ...]] = {
        "05-OP-11": (
            "`f16`, `bf16`, `f32`, or `f64`",
            "composition `div(sum(x, axis), divisor)`",
            "converted once to the sum result dtype",
            "zero-length axis is a type error when statically known",
            "execution-time extent is zero",
            "traps `Domain` as operation\n> `mean` at the result dtype",
            "`expand(g / divisor, original_shape, axis)`",
            "no accumulator parameter of its own",
        ),
        "05-OP-12": (
            "every active signed\n> integer and float tensor dtype",
            "compared without conversion at their stored dtype",
            "first NaN in increasing axis-index order",
            "first stored representation among equal\n> values",
            "execution-time extent is zero",
            "equal positive or negative infinities",
            "tie count `k` is\n> counted exactly as `int64`",
            "`div(g, k)`",
            "full cotangent flows to the\n> first NaN",
            "Integer operands are forward-only and `grad` rejects them",
        ),
        "05-OP-13": (
            "replacing maximum by minimum",
            "first NaN in\n> increasing axis-index order",
            "equal positive or negative\n> infinities",
            "upstream cotangent divided by the number of equal\n> minima",
        ),
        "05-OP-14": (
            "four product lanes initialized to one",
            "element `i` updates lane `i mod 4`",
            "`(p0 * p1) * (p2 * p3)`",
            "overflow is checked at every multiplication",
            "zero-length axis returns the multiplicative identity",
            "reverse-mode\n> derivative of that exact multiplication tree",
            "gradients at\n> zero operands are defined",
        ),
        "05-OP-15": (
            "every active\n> signed integer and float tensor dtype",
            "returns `int64` indices",
            "lowest\n> axis index containing NaN",
            "Comparisons never convert through another dtype",
            "execution-time extent is zero",
            "result dtype `int64`",
            "non-differentiable: `grad` rejects it",
        ),
        "05-OP-16": (
            "contract of [05-OP-15]",
            "lowest NaN index when present",
            "stored value is minimal",
        ),
        "05-OP-17": (
            "two signed-integer\n> scalar operands or two signed-integer tensor operands",
            "congruent to the exact mathematical sum modulo `2^w`",
            "never traps for\n> overflow",
            "`grad` rejects it",
        ),
        "05-OP-18": (
            "contract of [05-OP-17]",
            "exact mathematical difference modulo `2^w`",
        ),
        "05-OP-19": (
            "contract of [05-OP-17]",
            "exact mathematical product modulo `2^w`",
        ),
        "05-OP-20": (
            "`f16`, `bf16`, `f32`, or\n> `f64` scalar or tensor",
            "returns `bool` on the same surface",
            "finalized stored value\n> without conversion",
            "true exactly when that value is NaN",
            "contributes zero cotangent",
            "Integer, `bool`, `string`, and deferred operands\n> are type errors",
        ),
        "05-OP-21": (
            "contract of\n> [05-OP-20]",
            "true exactly for finite stored values",
            "false for NaN and both infinities",
        ),
        "05-OP-22": (
            "contract of\n> [05-OP-20]",
            "true exactly for positive or negative infinity",
            "false for NaN and every finite value",
        ),
        "05-OP-23": (
            "active\n> signed-integer or float source dtype",
            "signed-integer target dtype",
            "reads the source exactly at its\n> stored dtype",
            "finite float is truncated toward zero",
            "Negative infinity returns the target minimum",
            "NaN traps `Domain`",
            "never traps `Overflow`, wraps, or\n> converts through another numeric dtype",
            "`grad` rejects it",
        ),
        "05-OP-24": (
            "signed-integer source and signed-integer target",
            "congruent to the exact stored\n> source modulo `2^target_width`",
            "never traps for overflow, saturates, or\n> converts through a float dtype",
            "`grad` rejects it",
            "There is no `cast_round` operation",
            "`cast(round(source), target)`",
        ),
        "05-OP-25": (
            "`to_string(value) -> result` borrows exactly one value",
            "without consuming it and returns `string`",
            "admits exactly an active numeric, `bool`, or `string` scalar",
            "tensor whose element dtype is an active numeric dtype or `bool`",
            "`List[T]` when `T` is recursively admitted",
            "Every other value type is a type error",
            "returns `value` byte-for-byte unchanged",
            "dimensions `[d0, ..., d_(r-1)]` and `N` elements",
            "all `N` elements when `N <= 32`, otherwise the first 32",
            "A List boundary never truncates or elides elements",
            "String elements are inserted verbatim, without quoting or escaping",
            "non-injective display form, not a serialization",
            "Every lane produces byte-identical text",
            "operation is pure",
            "performs no arithmetic or dtype conversion",
            "non-differentiable (`grad` rejects it)",
            "no accumulator",
        ),
        "05-OP-26": (
            "exactly two `bool` scalars or two `bool` tensors",
            "identical dimensions",
            "evaluate `left` and then `right`",
            "not short-circuiting",
            "true exactly when both operands are true",
            "performs no arithmetic or dtype conversion",
            "has no accumulator",
            "`grad` rejects it",
        ),
        "05-OP-27": (
            "contract of [05-OP-26]",
            "true exactly when either operand is true",
            "applied element-wise for tensors",
        ),
        "05-OP-28": (
            "exactly one `bool` scalar or `bool` tensor",
            "true exactly when `value` is false",
            "Any non-`bool` operand is a type error",
            "performs no arithmetic or dtype conversion",
            "has no accumulator",
            "`grad` rejects it",
        ),
    }
    for atom, requirements in atom_requirements.items():
        require_atom(blocks, atom, requirements, violations)

    require_all(
        spec05,
        (
            (
                "`print`, `to_string`, `to_list`, diagnostics",
                "to_string observation exit",
            ),
            (
                "Not fully implemented; see chelis#1282 and chelis#1059",
                "to_string implementation owners",
            ),
            (
                "The first NaN is the forward result when any NaN is\npresent",
                "window extrema first-NaN forward rule",
            ),
            (
                "including equal positive or negative infinities, so each\n  receives `g / k`",
                "window extrema infinity-tie adjoint",
            ),
            (
                "route the full `g` to the first\n  NaN in row-major window order",
                "window extrema NaN adjoint",
            ),
            ("Not fully implemented; see chelis#1281", "window work owner"),
        ),
        violations,
    )


def validate_schema_and_consumers(
    docs: dict[str, str], violations: list[str]
) -> None:
    capability = docs["spec/design/capability_table.md"]
    plan = docs["spec/design/dtype_semantics.md"]
    loud = docs["spec/design/loud_unsupported.md"]
    provenance = docs["spec/design/spec_provenance.md"]
    roadmap = docs["spec/design/remediation_roadmap.md"]
    status = docs["docs/investigations/remediation_status_2026_08_04.md"]

    require_all(
        capability,
        (
            ("**Status:** Phase 4B schema frozen", "Phase 4B schema status"),
            (
                "(`BuiltinId`, `SurfaceClass`, operand `Prim`, `SemanticParams`)",
                "numeric table key",
            ),
            (
                "`SurfaceClass` is exactly `Scalar | Tensor`",
                "closed SurfaceClass variants",
            ),
            (
                "Supported { signature_rule: SignatureRuleId, result_dtype_rule:",
                "typed Table-A supported cell",
            ),
            (
                "Rejected { op_atom: SpecAtomRef, diagnostic_kind: DiagnosticKind }",
                "typed semantic rejection cell",
            ),
            (
                "`eval | c-host | c-dag | hip | metal`",
                "exact backend product",
            ),
            (
                "(`BuiltinId`, `SiblingDomain`, `SiblingCaseId`, `SemanticParams`)",
                "sibling registry key",
            ),
            (
                "`SiblingDomain` is the closed enum `Container | Boundary`",
                "closed sibling domains",
            ),
            (
                "`ToStringScalar`, `ToStringTensor`, and `ToStringList`",
                "disjoint to_string cases",
            ),
            (
                "Supported` exactly for [05-OP-25]'s scalar, tensor, and "
                "recursively admitted List domains",
                "to_string semantic authority",
            ),
            ("[#1282] owns aligning the pre-table", "to_string checker owner"),
            (
                "Every sibling `Supported` row expands across\n"
                "the same exact backend set",
                "sibling backend product",
            ),
            (
                "(`ExternalCallableFamily`, `CanonicalCallableId`)**",
                "external semantic key",
            ),
            (
                "(`ExternalCallableFamily`,\n`CanonicalCallableId`, "
                "`ExternalTargetContext`)",
                "external target key",
            ),
            (
                "`RuntimeCExport -> c-runtime` and `PyO3Binding ->\n"
                "python-extension`",
                "external target contexts",
            ),
            (
                "Implemented { implementation_id: ExternalImplementationId }",
                "typed external implemented cell",
            ),
            (
                "(`CanonicalEffectRequirement`, `BackendId`)**",
                "effect disposition key",
            ),
            (
                "`Random | Accum | Io | Test | Resource(ResourceId)`",
                "closed effect requirement domain",
            ),
            (
                "Implemented { implementation_id: EffectImplementationId }",
                "typed effect implemented cell",
            ),
            (
                "There is no\nmissing-row, wildcard, or default disposition.",
                "effect no-default rule",
            ),
            (
                "sealed `CompleteEffectDependencies`",
                "completed effect dependency carrier",
            ),
            (
                "explicit `Pure` case",
                "explicit effect purity result",
            ),
            (
                "An exported stdlib definition has no independent external target cell",
                "derived stdlib execution rule",
            ),
            (
                "Unimplemented { issue: #1281, diagnostic_kind: UnsupportedFeature }",
                "reduction implementation owner",
            ),
            (
                "Unimplemented { issue: #170, diagnostic_kind: UnsupportedFeature }",
                "product implementation owner",
            ),
            (
                "Unimplemented { issue: #1284, diagnostic_kind: UnsupportedFeature }",
                "logical implementation owner",
            ),
            ("invokes the 4B, 4C, and 4D oracles", "nested Phase 4B oracle"),
        ),
        violations,
    )
    for stale in ("NumericSurface", "open question 5", "`to_string` x Tensor/List"):
        if stale in capability:
            violations.append(f"stale capability-schema text remains: {stale}")

    require_all(
        plan,
        (
            ("### Phase 4B - semantic and schema freeze", "Phase 4B plan"),
            (
                ".venv/bin/python scripts/dtype_phase4b_oracle.py",
                "Phase 4B oracle command",
            ),
            (PASS_LINE, "Phase 4B oracle success line"),
            (
                "(ExternalCallableFamily, CanonicalCallableId, "
                "ExternalTargetContext)",
                "dtype-plan external target key",
            ),
            (
                "exported-stdlib dependency derivation",
                "dtype-plan stdlib derivation",
            ),
            (
                "(CanonicalEffectRequirement, BackendId)",
                "dtype-plan effect key",
            ),
            (
                "CompleteEffectDependencies::Pure",
                "dtype-plan explicit effect purity",
            ),
            ("invokes the 4B, 4C, and 4D oracles", "dtype-plan final nesting"),
            ("[#170] product-tree/backend work", "dtype-plan product owner"),
            ("[#1281] mean/extrema/argument-reduction", "dtype-plan reduction owner"),
            ("[#1284]\n   owns replacing", "dtype-plan logical owner"),
            (
                "canonical `to_string`, and boolean",
                "dtype-plan to_string semantics",
            ),
            (
                "[#1282] [05-OP-25] `to_string` checker/eval domain",
                "dtype-plan to_string checker owner",
            ),
            (
                "[#1059] compiled tensor/List cells",
                "dtype-plan to_string backend owner",
            ),
        ),
        violations,
    )
    require_all(
        loud,
        (
            (
                "BuiltinId × SiblingDomain × SiblingCaseId × SemanticParams",
                "loud sibling key",
            ),
            (
                "(ExternalCallableFamily, CanonicalCallableId, "
                "ExternalTargetContext)",
                "loud external target key",
            ),
            (
                "generated transitive\n  dependency closure over the checked body",
                "loud stdlib derivation",
            ),
            (
                "(CanonicalEffectRequirement, BackendId)",
                "loud effect key",
            ),
            (
                "CompleteEffectDependencies::Pure",
                "loud explicit effect purity",
            ),
            ("also invokes the Phase 4B oracle", "loud final-oracle nesting"),
        ),
        violations,
    )
    require_all(
        provenance,
        (
            (
                "ExternalCallableFamily × CanonicalCallableId × "
                "ExternalTargetContext",
                "provenance external target key",
            ),
            (
                "generated transitive dependency closure over the checked body",
                "provenance stdlib derivation",
            ),
            (
                "external target dispositions, exact effect dispositions, and "
                "derived exported-stdlib dependencies",
                "provenance governed-surface summary",
            ),
            (
                "exact effect dispositions",
                "provenance effect authority",
            ),
        ),
        violations,
    )
    require_all(
        roadmap,
        (
            ("[#170] owns the product-tree/backend rows", "roadmap product owner"),
            ("[#1281] owns the remaining reduction rows", "roadmap reduction owner"),
            ("Phase 4B froze semantics", "roadmap Phase 4B boundary"),
            (
                "checker/eval domain ([#1282])",
                "roadmap to_string checker owner",
            ),
            ("[05-OP-25] ([#1059])", "roadmap to_string backend owner"),
            (
                "external-target/effect dispositions",
                "roadmap external target authority",
            ),
            ("effect dispositions", "roadmap effect authority"),
            ("non-numeric logical/`where` lowering ([#1284])", "roadmap logical owner"),
            (
                "exported-stdlib dependency closure",
                "roadmap stdlib derivation",
            ),
        ),
        violations,
    )
    require_all(
        status,
        (
            ("This change is Phase 4B", "status Phase 4B statement"),
            ("[05-OP-11..28]", "status Phase 4B atom range"),
            ("#170 owns the product-tree/backend rows", "status product owner"),
            ("#1281 owns the remaining reduction rows", "status reduction owner"),
            ("logical/`where` lowering (#1284)", "status logical owner"),
            (
                "`to_string` checker/eval domain (#1282)",
                "status to_string checker owner",
            ),
            ("C-host cells (#1059)", "status to_string backend owner"),
            ("Of the 127 issues parented", "status parented count"),
            ("#729 | 15 / 53", "status #729 count"),
            (
                "external target dispositions, and exact effect-disposition rows",
                "status Phase 4C external target delivery",
            ),
            (
                "exact effect-disposition rows",
                "status Phase 4C effect delivery",
            ),
            (
                "exported-stdlib dependency closures",
                "status Phase 4D stdlib derivation",
            ),
            (
                "total external target-disposition registry",
                "status external target authority",
            ),
            (
                "derive per-backend executability transitively from their checked\n"
                "   bodies",
                "status stdlib derivation",
            ),
        ),
        violations,
    )


def validate_contract(root: Path = REPO_ROOT) -> None:
    docs = {relative: read(root, relative) for relative in CONTRACT_FILES}
    violations: list[str] = []
    validate_normative_contract(docs, violations)
    validate_schema_and_consumers(docs, violations)
    validate_frozen_contract(docs, violations)
    if violations:
        raise OracleError("; ".join(violations))


def run_oracle(python: str = sys.executable, root: Path = REPO_ROOT) -> None:
    try:
        validate_contract(root)
        subprocess.run(
            (python, "scripts/generate_rejection_registries.py", "--check"),
            cwd=root,
            check=True,
        )
    except OracleError as error:
        raise SystemExit(f"DTYPE PHASE 4B ORACLE: FAIL: {error}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "DTYPE PHASE 4B ORACLE: FAIL: rejection registry disagreement "
            f"(exit {error.returncode})"
        ) from error
    print(PASS_LINE)


if __name__ == "__main__":
    run_oracle()
