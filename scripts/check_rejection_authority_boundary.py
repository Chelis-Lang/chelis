#!/usr/bin/env python3
"""Check the privacy half of the [05-UNS-5] authority boundary.

Registry membership is enforced by Rust construction and the generated
registries. This check locks the complementary property: downstream crates
cannot obtain the scalar citation wrappers or compose a second generic
authority constructor around one. The only public builders accept raw
literals and perform registry validation themselves.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from generate_rejection_registries import ProductionSource, discover_production_sources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/chelis-types/src/unsupported.rs"

ALLOWED_PUBLIC_METHODS = {
    "SpecAtomRef": {"as_str", "registry"},
    "IssueRef": {"number", "registry"},
    "RejectionAuthority": {"kind", "atom", "issue", "hint", "citation"},
}

PUBLIC_FN = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+(?:const\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)

EXPECTED_PUBLIC_FUNCTIONS = Counter(
    {
        "as_str": 1,
        "registry": 2,
        "number": 1,
        "kind": 1,
        "atom": 1,
        "issue": 1,
        "hint": 1,
        "citation": 1,
        "__build_deliberate_rejection": 1,
        "__build_unimplemented_rejection": 1,
        "new": 1,
        "compiled_host_only_builtin": 1,
        "with_span": 1,
    }
)
DIRECT_BUILDER = re.compile(
    r"\b(__build_deliberate_rejection|__build_unimplemented_rejection)\b"
)

DELIBERATE_LITERAL = re.compile(
    r"deliberate_rejection!\(\s*\"(\[[0-9]{2}-[A-Z]+-[0-9]+\])\"",
    re.MULTILINE,
)
UNIMPLEMENTED_LITERAL = re.compile(
    r"unimplemented_rejection!\(\s*([0-9][0-9_]*)",
    re.MULTILINE,
)
RESPONSE_ONLY_ATOMS = {f"[05-UNS-{index}]" for index in range(1, 7)}


def _impl_body(source: str, type_name: str) -> str:
    marker = f"impl {type_name} {{"
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"missing {marker}")
    brace = source.find("{", start)
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise ValueError(f"unterminated {marker}")


def validate_source(source: str) -> list[str]:
    errors: list[str] = []
    public_functions = Counter(PUBLIC_FN.findall(source))
    if public_functions != EXPECTED_PUBLIC_FUNCTIONS:
        for name in sorted(set(public_functions) | set(EXPECTED_PUBLIC_FUNCTIONS)):
            actual = public_functions[name]
            expected = EXPECTED_PUBLIC_FUNCTIONS[name]
            if actual != expected:
                errors.append(
                    f"unsupported.rs public function inventory changed: {name} "
                    f"appears {actual} time(s), expected {expected}"
                )

    for fragment in ("pub mod ", "pub use ", "include!("):
        if fragment in source:
            errors.append(
                f"unsupported.rs exposes an unreviewed module/export edge: {fragment}"
            )

    for type_name, allowed in ALLOWED_PUBLIC_METHODS.items():
        try:
            body = _impl_body(source, type_name)
        except ValueError as error:
            errors.append(str(error))
            continue
        public = set(PUBLIC_FN.findall(body))
        unexpected = sorted(public - allowed)
        if unexpected:
            errors.append(
                f"{type_name} exposes unapproved public constructor/method(s): "
                + ", ".join(unexpected)
            )

    required_private_edges = (
        "const fn new(atom: &'static str)",
        "const fn new(number: u32)",
        "const fn deliberate(",
        "const fn unimplemented(",
    )
    for fragment in required_private_edges:
        if fragment not in source:
            errors.append(f"missing private construction edge: {fragment}")

    required_validating_builders = (
        "pub const fn __build_deliberate_rejection(",
        "let atom = match SpecAtomRef::new(atom)",
        "RejectionAuthority::deliberate(atom, hint)",
        "pub const fn __build_unimplemented_rejection(",
        "let issue = match IssueRef::new(issue)",
        "RejectionAuthority::unimplemented(issue, hint)",
    )
    for fragment in required_validating_builders:
        if fragment not in source:
            errors.append(f"validated macro edge changed or bypassed: {fragment}")
    return errors


def validate_usage_source(path: str, source: str) -> list[str]:
    errors: list[str] = []
    if path != "crates/chelis-types/src/lib.rs":
        for builder in sorted(set(DIRECT_BUILDER.findall(source))):
            errors.append(
                f"{path}: direct authority builder `{builder}` bypasses the canonical "
                "literal macro; use deliberate_rejection! or unimplemented_rejection!"
            )
    for atom in DELIBERATE_LITERAL.findall(source):
        if atom in RESPONSE_ONLY_ATOMS:
            errors.append(
                f"{path}: {atom} governs the response contract, not the semantic case; "
                "cite the deciding atom"
            )
    for raw_issue in UNIMPLEMENTED_LITERAL.findall(source):
        if int(raw_issue.replace("_", "")) == 959:
            errors.append(
                f"{path}: chelis#959 owns diagnostic migration, not implementation of a "
                "rejected capability"
            )
    return errors


def validate_production_usage(
    root: Path = ROOT, sources: list[ProductionSource] | None = None
) -> list[str]:
    """Validate the exact shared production source graph."""
    errors: list[str] = []
    for source in sources if sources is not None else discover_production_sources(root):
        errors.extend(
            validate_usage_source(
                source.path.as_posix(), source.source
            )
        )
    return errors


def main() -> int:
    errors = validate_source(SOURCE.read_text(encoding="utf-8"))
    errors.extend(validate_production_usage())
    if errors:
        for error in errors:
            print(f"REJECTION AUTHORITY BOUNDARY: {error}", file=sys.stderr)
        print("REJECTION AUTHORITY BOUNDARY: FAIL", file=sys.stderr)
        return 1
    print("REJECTION AUTHORITY BOUNDARY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
