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

from generate_rejection_registries import (
    UNIMPLEMENTED_LITERAL,
    _mask_rust_non_code,
    workspace_rust_paths,
)


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
ANY_FN = re.compile(
    r"\b(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:(?:const|async|unsafe|extern(?:\s+\"[^\"]+\")?)\s+)*"
    r"fn\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
ASSOCIATED_CONST = re.compile(
    r"\b(?:pub(?:\([^)]*\))?\s+)?const\s+(?!fn\b)([A-Za-z_][A-Za-z0-9_]*)"
)
PUBLIC_ITEM = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+"
    r"(struct|enum|trait|type|static|union)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
PUBLIC_CONST = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+const\s+(?!fn\b)([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
SENSITIVE_IMPL = re.compile(
    r"\bimpl\b[^\{;]*\b(SpecAtomRef|IssueRef|RejectionAuthority)\b\s*\{"
)
AUTHORITY_ALIAS = re.compile(
    r"\b(?:"
    r"use\b[^;]*\b(?:SpecAtomRef|IssueRef|RejectionAuthority|"
    r"RejectionCitation|__build_deliberate_rejection|"
    r"__build_unimplemented_rejection)\b[^;]*\bas\b[^;]*;|"
    r"type\s+(?:r#)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^;{}]*\b"
    r"(?:SpecAtomRef|IssueRef|RejectionAuthority|RejectionCitation)\b[^;{}]*;"
    r")"
)
PRIVATE_AUTHORITY_CONSTRUCTION = re.compile(
    r"(?:"
    r"\b(?:SpecAtomRef|IssueRef)\s*::\s*new\b|"
    r"\bRejectionAuthority\s*::\s*(?:deliberate|unimplemented)\b|"
    r"\b(?:SpecAtomRef|IssueRef)\s*\(|"
    r"\bRejectionAuthority\s*\{|"
    r"\bRejectionCitation\s*::|"
    r"\bimpl\b[^\{;]*\b(?:SpecAtomRef|IssueRef|RejectionAuthority)\b\s*\{"
    r")"
)
MACRO_RULES_DEFINITION = re.compile(
    r"\bmacro_rules\s*!\s*(?:r#)?([A-Za-z_][A-Za-z0-9_]*)"
)
MACRO_INVOCATION = re.compile(
    r"\b(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*!\s*[\(\[\{]"
)
ATTRIBUTE_NAME = re.compile(
    r"#\s*\[\s*(?:r#)?([A-Za-z_][A-Za-z0-9_]*)"
)
DERIVE_ATTRIBUTE = re.compile(r"#\s*\[\s*derive\s*\(([^\]]*)\)\s*\]")

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
EXPECTED_IMPL_FUNCTIONS = {
    "SpecAtomRef": Counter({"new": 1, "as_str": 1, "registry": 1}),
    "IssueRef": Counter({"new": 1, "number": 1, "registry": 1}),
    "RejectionAuthority": Counter(
        {
            "deliberate": 1,
            "unimplemented": 1,
            "kind": 1,
            "atom": 1,
            "issue": 1,
            "hint": 1,
            "citation": 1,
        }
    ),
}
PRIVATE_AUTHORITY_TOKEN = re.compile(
    r"\b(?:SpecAtomRef|IssueRef|RejectionAuthority|RejectionCitation|"
    r"AuthorityConstructionError|__build_deliberate_rejection|"
    r"__build_unimplemented_rejection)\b"
)
EXPECTED_PRIVATE_AUTHORITY_TOKENS = Counter(
    {
        "SpecAtomRef": 6,
        "IssueRef": 6,
        "RejectionAuthority": 10,
        "RejectionCitation": 12,
        "AuthorityConstructionError": 16,
        "__build_deliberate_rejection": 2,
        "__build_unimplemented_rejection": 2,
    }
)
EXPECTED_PUBLIC_ITEMS = Counter(
    {
        ("struct", "SpecAtomRef"): 1,
        ("struct", "IssueRef"): 1,
        ("enum", "RejectionAuthorityKind"): 1,
        ("struct", "RejectionAuthority"): 1,
        ("enum", "AuthorityConstructionError"): 1,
        ("enum", "UnsupportedKind"): 1,
        ("enum", "Stage"): 1,
        ("struct", "SpanRef"): 1,
        ("struct", "Unsupported"): 1,
    }
)
EXPECTED_SENSITIVE_IMPLS = Counter(
    {"SpecAtomRef": 1, "IssueRef": 1, "RejectionAuthority": 1}
)
EXPECTED_MACRO_DEFINITIONS = Counter(
    {"deliberate_rejection": 1, "unimplemented_rejection": 1}
)
EXPECTED_MACRO_INVOCATIONS = Counter(
    {
        "write": 12,
        "deliberate_rejection": 3,
        "format": 2,
        "panic": 2,
        "assert": 2,
        "assert_eq": 1,
    }
)
EXPECTED_ATTRIBUTES = Counter(
    {
        "derive": 10,
        "doc": 2,
        "macro_export": 2,
        "test": 2,
        "must_use": 1,
        "cfg": 1,
    }
)
EXPECTED_DERIVES = Counter(
    {
        ("Debug", "Clone", "Copy", "PartialEq", "Eq", "Hash"): 5,
        ("Debug", "Clone", "Copy", "PartialEq", "Eq"): 2,
        ("Debug", "Clone", "PartialEq", "Eq"): 2,
        ("Debug", "Clone", "PartialEq", "Eq", "Default"): 1,
    }
)
PRIVATE_COPY_EQ_HASH_DERIVE = (
    r"#\s*\[\s*derive\s*\(\s*Debug\s*,\s*Clone\s*,\s*Copy\s*,\s*"
    r"PartialEq\s*,\s*Eq\s*,\s*Hash\s*\)\s*\]\s*"
)
PRIVATE_AUTHORITY_LAYOUTS = {
    "SpecAtomRef tuple field": re.compile(
        PRIVATE_COPY_EQ_HASH_DERIVE
        + r"pub\s+struct\s+SpecAtomRef\s*\(\s*&'static\s+str\s*\)\s*;"
    ),
    "IssueRef tuple field": re.compile(
        PRIVATE_COPY_EQ_HASH_DERIVE
        + r"pub\s+struct\s+IssueRef\s*\(\s*NonZeroU32\s*\)\s*;"
    ),
    "RejectionCitation enum": re.compile(
        PRIVATE_COPY_EQ_HASH_DERIVE + r"enum\s+RejectionCitation\s*\{\s*"
        r"Atom\s*\(\s*SpecAtomRef\s*\)\s*,\s*"
        r"Issue\s*\(\s*IssueRef\s*\)\s*,?\s*\}"
    ),
    "RejectionAuthority fields": re.compile(
        PRIVATE_COPY_EQ_HASH_DERIVE
        + r"pub\s+struct\s+RejectionAuthority\s*\{\s*"
        r"citation\s*:\s*RejectionCitation\s*,\s*"
        r"hint\s*:\s*&'static\s+str\s*,?\s*\}"
    ),
}
DIRECT_BUILDER = re.compile(
    r"\b(__build_deliberate_rejection|__build_unimplemented_rejection)\b"
)

DELIBERATE_LITERAL = re.compile(
    r"deliberate_rejection!\(\s*\"(\[[0-9]{2}-[A-Z]+-[0-9]+\])\"",
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
    code = _mask_rust_non_code(source)
    if not code.isascii():
        errors.append(
            "unsupported.rs contains a non-ASCII Rust token; the authority owner "
            "must use identifiers the closed lexical inventory can classify"
        )
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
            body = _impl_body(code, type_name)
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

        functions = Counter(ANY_FN.findall(body))
        expected_functions = EXPECTED_IMPL_FUNCTIONS[type_name]
        if functions != expected_functions:
            errors.append(
                f"{type_name} private/public function inventory changed: "
                f"found {dict(sorted(functions.items()))}, expected "
                f"{dict(sorted(expected_functions.items()))}"
            )
        associated_consts = ASSOCIATED_CONST.findall(body)
        if associated_consts:
            errors.append(
                f"{type_name} exposes an unapproved associated const "
                "construction edge: " + ", ".join(sorted(associated_consts))
            )

    private_tokens = Counter(PRIVATE_AUTHORITY_TOKEN.findall(code))
    if private_tokens != EXPECTED_PRIVATE_AUTHORITY_TOKENS:
        errors.append(
            "unsupported.rs private authority token inventory changed: "
            f"found {dict(sorted(private_tokens.items()))}, expected "
            f"{dict(sorted(EXPECTED_PRIVATE_AUTHORITY_TOKENS.items()))}"
        )

    if AUTHORITY_ALIAS.search(code):
        errors.append(
            "unsupported.rs authority alias is forbidden; keep every "
            "construction edge under its canonical identity"
        )

    public_items = Counter(PUBLIC_ITEM.findall(code))
    public_items.update(("const", name) for name in PUBLIC_CONST.findall(code))
    if public_items != EXPECTED_PUBLIC_ITEMS:
        errors.append(
            "unsupported.rs public item inventory changed: "
            f"found {dict(sorted(public_items.items()))}, expected "
            f"{dict(sorted(EXPECTED_PUBLIC_ITEMS.items()))}"
        )

    sensitive_impls = Counter(SENSITIVE_IMPL.findall(code))
    if sensitive_impls != EXPECTED_SENSITIVE_IMPLS:
        errors.append(
            "unsupported.rs protected impl inventory changed: "
            f"found {dict(sorted(sensitive_impls.items()))}, expected "
            f"{dict(sorted(EXPECTED_SENSITIVE_IMPLS.items()))}"
        )

    macro_definitions = Counter(MACRO_RULES_DEFINITION.findall(code))
    macro_invocations = Counter(MACRO_INVOCATION.findall(code))
    if (
        macro_definitions != EXPECTED_MACRO_DEFINITIONS
        or macro_invocations != EXPECTED_MACRO_INVOCATIONS
    ):
        errors.append(
            "unsupported.rs macro inventory changed: "
            f"definitions={dict(sorted(macro_definitions.items()))}, "
            f"invocations={dict(sorted(macro_invocations.items()))}"
        )

    attributes = Counter(ATTRIBUTE_NAME.findall(code))
    derives = Counter(
        tuple(part.strip() for part in match.group(1).split(","))
        for match in DERIVE_ATTRIBUTE.finditer(code)
    )
    if attributes != EXPECTED_ATTRIBUTES or derives != EXPECTED_DERIVES:
        errors.append(
            "unsupported.rs attribute/derive inventory changed: "
            f"attributes={dict(sorted(attributes.items()))}, "
            f"derives={dict(sorted(derives.items()))}"
        )

    for label, pattern in PRIVATE_AUTHORITY_LAYOUTS.items():
        if len(pattern.findall(code)) != 1:
            errors.append(
                f"unsupported.rs private authority layout changed: {label}"
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
    code = _mask_rust_non_code(source)
    if not code.isascii() and PRIVATE_AUTHORITY_TOKEN.search(code):
        errors.append(
            f"{path}: non-ASCII Rust token is forbidden in authority-bearing "
            "production source"
        )
    for builder in sorted(set(DIRECT_BUILDER.findall(code))):
        errors.append(
            f"{path}: direct authority builder `{builder}` bypasses the canonical "
            "literal macro; use deliberate_rejection! or unimplemented_rejection!"
        )
    if AUTHORITY_ALIAS.search(code):
        errors.append(
            f"{path}: authority alias bypasses the canonical construction boundary"
        )
    if PRIVATE_AUTHORITY_CONSTRUCTION.search(code):
        errors.append(
            f"{path}: private authority constructor is confined to unsupported.rs; "
            "use deliberate_rejection! or unimplemented_rejection!"
        )
    for atom in DELIBERATE_LITERAL.findall(source):
        if atom in RESPONSE_ONLY_ATOMS:
            errors.append(
                f"{path}: {atom} governs the response contract, not the semantic case; "
                "cite the deciding atom"
            )
    for match in UNIMPLEMENTED_LITERAL.finditer(code):
        if int(match.group(2).replace("_", "")) == 959:
            errors.append(
                f"{path}: chelis#959 owns diagnostic migration, not implementation of a "
                "rejected capability"
            )
    return errors


def validate_production_usage(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    paths, _ = workspace_rust_paths(root)
    for path in paths:
        relative = path.relative_to(root)
        if relative.as_posix() == "crates/chelis-types/src/unsupported.rs":
            errors.extend(validate_source(path.read_text(encoding="utf-8")))
            continue
        errors.extend(
            validate_usage_source(
                relative.as_posix(), path.read_text(encoding="utf-8")
            )
        )
    return errors


def main() -> int:
    errors = validate_production_usage()
    if errors:
        for error in errors:
            print(f"REJECTION AUTHORITY BOUNDARY: {error}", file=sys.stderr)
        print("REJECTION AUTHORITY BOUNDARY: FAIL", file=sys.stderr)
        return 1
    print("REJECTION AUTHORITY BOUNDARY: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
