#!/usr/bin/env python3
"""Exact pre-Phase-4C builtin/RISC-to-[05-OP-N] closure oracle.

Acceptance is exit 0 with the final line
``DTYPE BUILTIN ATOM CLOSURE ORACLE: PASS``.

The source universe is derived from typed Rust declarations: every
``BuiltinDecl.capability`` domain/case and every ``RiscAtomIdentity``. The
script contains no builtin count, name allowlist, or compatibility escape.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PASS_LINE = "DTYPE BUILTIN ATOM CLOSURE ORACLE: PASS"
BUILTINS_REL = Path("crates/chelis-types/src/builtins.rs")
RISC_REL = Path("crates/chelis-ir/src/dag.rs")
SPEC_REL = Path("spec/05-risc-primitives.md")
GENERATED_REL = Path("crates/chelis-types/src/rejection_registry_generated.rs")

REQUIRED_ATOMS = tuple(f"[05-OP-{number}]" for number in range(1, 40))
ATOM_DEFINITION = re.compile(r"^> \*\*(\[05-OP-([0-9]+)\])\*\*", re.MULTILINE)
ATOM_GRAMMAR = re.compile(r"\[05-OP-[1-9][0-9]*\]\Z")
GENERATED_ATOM = re.compile(r'"(\[05-OP-[1-9][0-9]*\])"')
FORBIDDEN_DEFAULT = re.compile(r"(?:^|[^a-z])(default|fallback|wildcard)(?:[^a-z]|$)")
FORBIDDEN_COMPATIBILITY = re.compile(
    r"(?:legacy|grandfather|compat(?:ibility)?|deprecated|preexisting|old[_ -]?path)",
    re.IGNORECASE,
)


class OracleError(RuntimeError):
    pass


class Domain(str, Enum):
    NUMERIC = "Numeric"
    CONTAINER = "Container"
    BOUNDARY = "Boundary"


@dataclass(frozen=True, order=True)
class CapabilityIdentity:
    builtin: str
    domain: Domain
    case: str

    def render(self) -> str:
        return f"{self.domain.value}:{self.builtin}:{self.case}"


@dataclass(frozen=True)
class SemanticRegistration:
    identity: CapabilityIdentity
    atom: str


# These fragments are semantic mutation killers, not identity allowlists.
# Identity completeness comes only from the compiled-source declarations.
ESSENTIAL_SEMANTICS: dict[str, tuple[str, ...]] = {
    "[05-OP-1]": (
        "operand's own storage width",
        "complete mathematical integer domain",
        "AdRejectionReason::PiecewiseConstant",
        "no accumulator",
    ),
    "[05-OP-2]": (
        "JsonFloat",
        "JsonInt",
        "outside int64 range is a loud `Overflow` error",
        "never an f64 fallback",
    ),
    "[05-OP-3]": ("returns the stored `JsonInt` int64 exactly", "csv_int", "csv_f64", "no cotangent"),
    "[05-OP-4]": ("accepts exactly f64", "accepts exactly int64", "explicit checked cast"),
    "[05-OP-5]": ("exact decimal digits", "parses back to the identical f64", "non-finite `JsonFloat`"),
    "[05-OP-6]": (
        "cast_trunc",
        "truncated toward zero",
        "outside the target range it traps `overflow`",
        "Semantics are identical on scalar and tensor surfaces",
        "non-differentiable",
    ),
    "[05-OP-7]": ("stored extent", "exact `int64`", "negative value", "zero-cotangent adjoint"),
    "[05-OP-8]": ("every active float", "same dtype", "Random call ordinal", "pathwise adjoint"),
    "[05-OP-9]": ("every active tensor element dtype", "no arithmetic", "runtime List shapes", "balanced tree"),
    "[05-OP-10]": ("`width` SHALL be non-negative", "same dtype", "truncated source cells", "non-differentiable"),
    "[05-OP-11]": ("f16", "f64", "zero-length axis", "adjoint"),
    "[05-OP-12]": (
        "every active signed",
        "first NaN",
        "elements that compare equal to the selected non-NaN maximum",
        "highest-original-position-first",
    ),
    "[05-OP-13]": (
        "min_reduce",
        "same dtype",
        "NaN",
        "upstream cotangent divided by the number of equal minima",
    ),
    "[05-OP-14]": ("canonical balanced tree", "zero-length axis", "overflow", "reverse-mode"),
    "[05-OP-15]": ("int64", "lowest axis index", "lowest axis index containing NaN", "grad` rejects"),
    "[05-OP-16]": ("argmin_reduce", "signature", "dtype", "lowest axis index whose stored value is minimal"),
    "[05-OP-17]": ("modulo `2^w`", "never traps for overflow", "no accumulator", "grad` rejects"),
    "[05-OP-18]": ("exact mathematical difference", "modulo `2^w`"),
    "[05-OP-19]": ("exact mathematical product", "modulo `2^w`"),
    "[05-OP-20]": ("stored value without conversion", "NaN", "zero cotangent", "no accumulator"),
    "[05-OP-21]": ("finite stored values", "signed zeros", "subnormals"),
    "[05-OP-22]": ("positive or negative infinity", "false for NaN"),
    "[05-OP-23]": ("cast_saturate", "signed-integer target", "NaN", "clamped to the target's inclusive range"),
    "[05-OP-24]": ("cast_wrap", "modulo", "signed-integer", "non-differentiable"),
    "[05-OP-25]": ("unit", "List", "Dict", "ADT", "resource handles are type errors", "non-differentiable"),
    "[05-OP-26]": ("exactly two `bool`", "not short-circuiting", "no accumulator", "grad` rejects"),
    "[05-OP-27]": ("or(left, right)", "either operand"),
    "[05-OP-28]": ("exactly one `bool`", "non-differentiable"),
    "[05-OP-29]": (
        "exactly a `bool` tensor",
        "int64",
        "dedicated reduction",
        "AdRejectionReason::IntegerIndexOutput",
    ),
    "[05-OP-30]": ("sum_result", "accumulator", "balanced tree", "adjoint"),
    "[05-OP-31]": ("exactly the ten", "chelis_scalar", "dtype", "no authority"),
    "[05-OP-32]": ("recursive-observation", "signature is part of each identity", "int64", "Domain", "outside AD"),
    "[05-OP-33]": (
        "twenty-three final public C callable identities",
        "axes and rank are `int32_t`",
        "row-major",
        "active signed-integer",
        "adjoint",
    ),
    "[05-OP-34]": (
        "exactly these five",
        "JsonInt(int64)",
        "JsonFloat(f64)",
        "without arithmetic, conversion, or float funnel",
    ),
    "[05-OP-35]": (
        "eighty-three final exported stdlib numeric definitions",
        "normal_cdf",
        "sort::sort",
        "test::assert_eq",
    ),
    "[05-OP-36]": (
        "seven language identities",
        "recursively admitted by this equality rule",
        "NaN",
        "zero cotangent",
    ),
    "[05-OP-37]": ("every active float", "scalar `rate: p`", "0 <= rate < 1", "saved mask"),
    "[05-OP-38]": ("exactly these five", "tensor_scan", "process_run", "test_assert_eq_tensor"),
    "[05-OP-39]": ("reduce_window_sum", "ReduceWindowGrad", "window/stride validation", "higher-order"),
}


def _balanced_end(text: str, opening: int, open_char: str, close_char: str) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = 0
    index = opening
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "/" and next_char == "*":
                block_comment += 1
                index += 2
                continue
            if char == "*" and next_char == "/":
                block_comment -= 1
                index += 2
                continue
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = 1
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise OracleError(f"unterminated `{open_char}` block")


def _initializer_block(text: str, marker: str, open_char: str, close_char: str) -> str:
    marker_index = text.find(marker)
    if marker_index < 0:
        raise OracleError(f"missing typed source marker `{marker}`")
    search_from = marker_index + len(marker)
    if marker.startswith("pub const ") or marker.startswith("const "):
        equals = text.find("=", search_from)
        if equals < 0:
            raise OracleError(f"missing `=` after `{marker}`")
        search_from = equals + 1
    opening = text.find(open_char, search_from)
    if opening < 0:
        raise OracleError(f"missing `{open_char}` after `{marker}`")
    end = _balanced_end(text, opening, open_char, close_char)
    return text[opening + 1 : end]


def _rust_struct_blocks(text: str, marker: str) -> tuple[str, ...]:
    blocks: list[str] = []
    position = 0
    while True:
        start = text.find(marker, position)
        if start < 0:
            break
        opening = text.find("{", start + len(marker))
        if opening < 0:
            raise OracleError(f"missing `{{` after `{marker}`")
        end = _balanced_end(text, opening, "{", "}")
        blocks.append(text[start : end + 1])
        position = end + 1
    return tuple(blocks)


def discover_builtin_names(source: str) -> tuple[str, ...]:
    body = _initializer_block(source, "pub const BUILTINS", "[", "]")
    names: list[str] = []
    for block in _rust_struct_blocks(body, "BuiltinDecl"):
        match = re.search(r'\bname:\s*"([a-z][a-z0-9_]*)"\s*,?', block)
        if match is None:
            raise OracleError("BuiltinDecl without one literal canonical `name`")
        names.append(match.group(1))
    duplicates = sorted(name for name, count in Counter(names).items() if count != 1)
    if duplicates:
        raise OracleError(f"duplicate BuiltinDecl names: {duplicates}")
    return tuple(names)


def _domain_constants(source: str) -> dict[str, tuple[Domain, ...]]:
    constants: dict[str, tuple[Domain, ...]] = {
        "NUMERIC_DOMAIN": (Domain.NUMERIC,),
        "CONTAINER_DOMAIN": (Domain.CONTAINER,),
        "BOUNDARY_DOMAIN": (Domain.BOUNDARY,),
        "NUMERIC_CONTAINER_DOMAINS": (Domain.NUMERIC, Domain.CONTAINER),
    }
    pattern = re.compile(
        r"const\s+([A-Z][A-Z0-9_]*)\s*:\s*&\[BuiltinSemanticDomain\]\s*=\s*&\[(.*?)\];",
        re.DOTALL,
    )
    for name, body in pattern.findall(source):
        domains = tuple(Domain(value) for value in re.findall(r"BuiltinSemanticDomain::(\w+)", body))
        if not domains:
            raise OracleError(f"typed domain constant `{name}` is empty")
        if len(set(domains)) != len(domains):
            raise OracleError(f"typed domain constant `{name}` has duplicate domains")
        expected = constants.get(name)
        if expected is not None and expected != domains:
            raise OracleError(
                f"typed domain constant `{name}` disagrees with the closed domain enum"
            )
        constants[name] = domains
    return constants


def _case_constants(source: str) -> dict[str, tuple[tuple[Domain, str], ...]]:
    constants: dict[str, tuple[tuple[Domain, str], ...]] = {}
    pattern = re.compile(
        r"const\s+([A-Z][A-Z0-9_]*)\s*:\s*&\[BuiltinSiblingCaseDecl\]\s*=\s*&\[(.*?)\];",
        re.DOTALL,
    )
    for name, body in pattern.findall(source):
        rows = tuple(
            (Domain(domain), case)
            for domain, case in re.findall(
                r"domain:\s*BuiltinSemanticDomain::(\w+)\s*,\s*"
                r"case:\s*BuiltinSiblingCaseId::(\w+)",
                body,
                re.DOTALL,
            )
        )
        constants[name] = rows
    return constants


def _capability_constants(
    source: str,
    domains: dict[str, tuple[Domain, ...]],
    cases: dict[str, tuple[tuple[Domain, str], ...]],
) -> dict[str, tuple[tuple[Domain, ...], tuple[tuple[Domain, str], ...]]]:
    constants: dict[str, tuple[tuple[Domain, ...], tuple[tuple[Domain, str], ...]]] = {
        "NUMERIC_CAPABILITY": ((Domain.NUMERIC,), ()),
    }
    pattern = re.compile(
        r"const\s+([A-Z][A-Z0-9_]*)\s*:\s*BuiltinCapabilityDecl\s*=\s*"
        r"BuiltinCapabilityDecl\s*\{(.*?)\};",
        re.DOTALL,
    )
    for name, body in pattern.findall(source):
        domain_match = re.search(r"domains:\s*([A-Z][A-Z0-9_]*)", body)
        case_match = re.search(r"sibling_cases:\s*([A-Z][A-Z0-9_]*|&\[\])", body)
        if domain_match is None or case_match is None:
            raise OracleError(f"typed capability constant `{name}` is incomplete")
        domain_name = domain_match.group(1)
        if domain_name not in domains:
            raise OracleError(f"typed capability constant `{name}` cites unknown domains `{domain_name}`")
        case_name = case_match.group(1)
        case_rows = () if case_name == "&[]" else cases.get(case_name)
        if case_rows is None:
            raise OracleError(f"typed capability constant `{name}` cites unknown cases `{case_name}`")
        constants[name] = (domains[domain_name], case_rows)
    return constants


def discover_builtin_capabilities(source: str) -> tuple[CapabilityIdentity, ...]:
    domain_constants = _domain_constants(source)
    case_constants = _case_constants(source)
    capability_constants = _capability_constants(source, domain_constants, case_constants)
    body = _initializer_block(source, "pub const BUILTINS", "[", "]")
    identities: list[CapabilityIdentity] = []
    seen_names: list[str] = []

    for block in _rust_struct_blocks(body, "BuiltinDecl"):
        name_match = re.search(r'\bname:\s*"([a-z][a-z0-9_]*)"\s*,?', block)
        if name_match is None:
            raise OracleError("BuiltinDecl without one literal canonical `name`")
        name = name_match.group(1)
        seen_names.append(name)
        capability_match = re.search(
            r"\bcapability:\s*(.+?)(?=,\s*(?:inference|realizability|shape_class|axis_arguments):)",
            block,
            re.DOTALL,
        )
        if capability_match is None:
            raise OracleError(f"BuiltinDecl `{name}` has no typed `capability` field")
        expression = capability_match.group(1).strip()

        macro = re.fullmatch(
            r"sibling_capability!\(([A-Z][A-Z0-9_]*),\s*(Numeric|Container|Boundary),\s*(\w+)\)",
            expression,
        )
        if macro is not None:
            domain_constant, domain_text, case = macro.groups()
            domain = Domain(domain_text)
            declared_domains = domain_constants.get(domain_constant)
            if declared_domains != (domain,):
                raise OracleError(
                    f"BuiltinDecl `{name}` sibling macro domain `{domain.value}` disagrees with `{domain_constant}`"
                )
            domains = declared_domains
            sibling_cases = ((domain, case),)
        else:
            declaration = capability_constants.get(expression)
            if declaration is None:
                raise OracleError(
                    f"BuiltinDecl `{name}` cites unknown typed capability `{expression}`"
                )
            domains, sibling_cases = declaration

        if not domains:
            raise OracleError(f"BuiltinDecl `{name}` has an empty semantic domain set")
        for domain in domains:
            if domain is Domain.NUMERIC:
                identities.append(CapabilityIdentity(name, domain, "TableA"))
        for domain, case in sibling_cases:
            if domain is Domain.NUMERIC:
                raise OracleError(f"BuiltinDecl `{name}` put sibling case `{case}` in Numeric")
            if domain not in domains:
                raise OracleError(
                    f"BuiltinDecl `{name}` case `{case}` names undeclared domain `{domain.value}`"
                )
            identities.append(CapabilityIdentity(name, domain, case))
        for domain in (Domain.CONTAINER, Domain.BOUNDARY):
            has_domain = domain in domains
            has_case = any(case_domain is domain for case_domain, _case in sibling_cases)
            if has_domain != has_case:
                raise OracleError(
                    f"BuiltinDecl `{name}` must enumerate exact cases for sibling domain `{domain.value}`"
                )

    duplicates = sorted(name for name, count in Counter(seen_names).items() if count != 1)
    if duplicates:
        raise OracleError(f"duplicate BuiltinDecl names: {duplicates}")
    duplicate_identities = sorted(
        identity.render() for identity, count in Counter(identities).items() if count != 1
    )
    if duplicate_identities:
        raise OracleError(f"duplicate BuiltinDecl capability identities: {duplicate_identities}")
    return tuple(identities)


def _strip_rust_comments_and_strings(source: str) -> str:
    output = list(source)
    index = 0
    line_comment = False
    block_comment = 0
    quote: str | None = None
    escaped = False
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            else:
                output[index] = " "
            index += 1
            continue
        if block_comment:
            output[index] = " "
            if char == "/" and next_char == "*":
                output[index + 1] = " "
                block_comment += 1
                index += 2
                continue
            if char == "*" and next_char == "/":
                output[index + 1] = " "
                block_comment -= 1
                index += 2
                continue
            index += 1
            continue
        if quote is not None:
            if char != "\n":
                output[index] = " "
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and next_char == "/":
            output[index] = output[index + 1] = " "
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            output[index] = output[index + 1] = " "
            block_comment = 1
            index += 2
            continue
        if char in {'"', "'"}:
            output[index] = " "
            quote = char
        index += 1
    return "".join(output)


def _discover_enum_variants(source: str, marker: str) -> tuple[str, ...]:
    body = _initializer_block(source, marker, "{", "}")
    code = _strip_rust_comments_and_strings(body)
    variants: list[str] = []
    depth = 0
    for line in code.splitlines():
        if depth == 0:
            match = re.match(r"\s*([A-Z][A-Za-z0-9_]*)\s*(?:,|\{|\()", line)
            if match is not None:
                variants.append(match.group(1))
        depth += line.count("{") + line.count("(") - line.count("}") - line.count(")")
    duplicates = sorted(name for name, count in Counter(variants).items() if count != 1)
    if duplicates:
        raise OracleError(f"duplicate {marker.removeprefix('pub enum ')} variants: {duplicates}")
    return tuple(variants)


def discover_risc_variants(source: str) -> tuple[str, ...]:
    return _discover_enum_variants(source, "pub enum RiscOp")


def discover_risc_atom_identities(source: str) -> tuple[CapabilityIdentity, ...]:
    variants = discover_risc_variants(source)
    if not variants:
        return ()
    identity_variants = _discover_enum_variants(source, "pub enum RiscAtomIdentity")
    if not identity_variants:
        raise OracleError("RiscOp is non-empty but RiscAtomIdentity has no variants")
    impl_index = source.find("impl RiscAtomIdentity")
    if impl_index < 0:
        raise OracleError("RiscOp is non-empty but has no typed RiscAtomIdentity source")

    all_body = _initializer_block(source[impl_index:], "pub const ALL", "[", "]")
    all_variants = tuple(re.findall(r"Self::(\w+)", all_body))
    if all_variants != identity_variants:
        raise OracleError(
            "RiscAtomIdentity::ALL must enumerate every identity exactly once in enum order: "
            f"enum={identity_variants}, ALL={all_variants}"
        )

    function_index = source.find("pub const fn as_str", impl_index)
    if function_index < 0:
        raise OracleError("RiscAtomIdentity has no exhaustive canonical `as_str`")
    opening = source.find("{", function_index)
    end = _balanced_end(source, opening, "{", "}")
    function = source[opening + 1 : end]
    mappings = re.findall(r'Self::(\w+)\s*=>\s*"([A-Za-z][A-Za-z0-9_]*)"', function)
    if not mappings:
        raise OracleError("RiscAtomIdentity `as_str` exposes no canonical identities")
    variant_names = [variant for variant, _canonical in mappings]
    canonical_names = [canonical for _variant, canonical in mappings]
    if tuple(variant_names) != identity_variants:
        raise OracleError(
            "RiscAtomIdentity `as_str` must map every identity exactly once in enum order: "
            f"enum={identity_variants}, as_str={tuple(variant_names)}"
        )
    if len(set(canonical_names)) != len(canonical_names):
        raise OracleError("RiscAtomIdentity `as_str` has a duplicate canonical identity")

    disposition_index = source.find("pub const fn atom_disposition", end)
    if disposition_index < 0:
        raise OracleError("RiscOp has no exhaustive typed `atom_disposition`")
    disposition_opening = source.find("{", disposition_index)
    disposition_end = _balanced_end(source, disposition_opening, "{", "}")
    disposition = _strip_rust_comments_and_strings(
        source[disposition_opening + 1 : disposition_end]
    )
    disposition_variants = tuple(re.findall(r"Self::(\w+)", disposition))
    missing_dispositions = sorted(set(variants) - set(disposition_variants))
    stale_dispositions = sorted(set(disposition_variants) - set(variants))
    duplicate_dispositions = sorted(
        variant for variant, count in Counter(disposition_variants).items() if count != 1
    )
    if missing_dispositions or stale_dispositions or duplicate_dispositions:
        raise OracleError(
            "RiscOp `atom_disposition` must name every variant exactly once: "
            f"missing={missing_dispositions}, stale={stale_dispositions}, "
            f"duplicate={duplicate_dispositions}"
        )
    disposition_identities = tuple(re.findall(r"Id::(\w+)", disposition))
    missing_identities = sorted(set(identity_variants) - set(disposition_identities))
    stale_identities = sorted(set(disposition_identities) - set(identity_variants))
    duplicate_identities = sorted(
        identity for identity, count in Counter(disposition_identities).items() if count != 1
    )
    if missing_identities or stale_identities or duplicate_identities:
        raise OracleError(
            "RiscOp `atom_disposition` must select every RiscAtomIdentity exactly once: "
            f"missing={missing_identities}, stale={stale_identities}, "
            f"duplicate={duplicate_identities}"
        )
    return tuple(
        CapabilityIdentity(canonical, Domain.NUMERIC, "TableA")
        for canonical in canonical_names
    )


def atom_blocks(spec: str) -> dict[str, str]:
    matches = list(ATOM_DEFINITION.finditer(spec))
    counts = Counter(match.group(1) for match in matches)
    duplicates = sorted(atom for atom, count in counts.items() if count != 1)
    if duplicates:
        raise OracleError(f"normative atoms must have exactly one normative definition line: {duplicates}")
    blocks: dict[str, str] = {}
    lines = spec.splitlines(keepends=True)
    offset = 0
    line_at: dict[int, int] = {}
    for index, line in enumerate(lines):
        line_at[offset] = index
        offset += len(line)
    for match in matches:
        start_line = line_at[match.start()]
        block_lines: list[str] = []
        for line in lines[start_line:]:
            if block_lines and ATOM_DEFINITION.match(line):
                break
            if block_lines and not line.startswith(">"):
                break
            block_lines.append(line)
        blocks[match.group(1)] = "".join(block_lines)
    return blocks


def validate_required_atom_numbers(spec: str, expected: tuple[str, ...] = REQUIRED_ATOMS) -> None:
    definitions = [match.group(1) for match in ATOM_DEFINITION.finditer(spec)]
    counts = Counter(definitions)
    problems: list[str] = []
    for atom in expected:
        if counts[atom] == 0:
            problems.append(f"missing required atom {atom}")
        elif counts[atom] != 1:
            problems.append(f"required atom {atom} has {counts[atom]} definition lines")
    if problems:
        raise OracleError("; ".join(problems))


def _code_spans(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"`([^`\n]+)`", text))


def _leading_identity(code: str) -> str | None:
    match = re.match(r"(?:[a-z0-9_/]+::)?([A-Za-z][A-Za-z0-9_]*)\s*(?:\(|\Z)", code.strip())
    return match.group(1) if match else None


def governing_names(block: str) -> frozenset[str]:
    lines = block.splitlines()
    names: set[str] = set()
    if lines:
        first_spans = _code_spans(lines[0])
        if first_spans:
            name = _leading_identity(first_spans[0])
            if name:
                names.add(name)

    exact_paragraph: list[str] = []
    collecting = False
    for line in lines:
        if "governs exactly" in line or "identities and signatures" in line:
            collecting = True
        if collecting:
            exact_paragraph.append(line)
            if line.strip() == ">":
                break
    for code in _code_spans("\n".join(exact_paragraph)):
        name = _leading_identity(code)
        if name:
            names.add(name)

    for line in lines:
        if not line.startswith("> | ") or "---" in line:
            continue
        cells = line[2:].split("|")[1:-1]
        if not cells:
            continue
        spans = _code_spans(cells[0])
        if not spans:
            continue
        name = _leading_identity(spans[0])
        if name:
            names.add(name)
    return frozenset(names)


def derive_registrations(
    discovered: tuple[CapabilityIdentity, ...], spec: str
) -> tuple[SemanticRegistration, ...]:
    blocks = atom_blocks(spec)
    governed = {atom: governing_names(block) for atom, block in blocks.items()}
    registrations: list[SemanticRegistration] = []
    problems: list[str] = []
    for identity in discovered:
        candidates = sorted(atom for atom, names in governed.items() if identity.builtin in names)
        if not candidates:
            problems.append(
                f"missing semantic registration for discovered identity {identity.render()}"
            )
        elif len(candidates) != 1:
            problems.append(
                f"duplicate semantic registration candidates for {identity.render()}: {candidates}"
            )
        else:
            registrations.append(SemanticRegistration(identity, candidates[0]))
    if problems:
        raise OracleError("\n".join(problems))
    return tuple(registrations)


def _normalized(text: str) -> str:
    unquoted = "\n".join(
        line[2:] if line.startswith("> ") else line[1:] if line == ">" else line
        for line in text.splitlines()
    )
    return " ".join(unquoted.split())


def _case_words(identity: CapabilityIdentity) -> tuple[str, ...]:
    if identity.case in {"TableA", "Risc"}:
        return ()
    case_words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|\Z)|[A-Z]?[a-z]+|[0-9]+", identity.case)
    builtin_words = [word for word in identity.builtin.split("_") if word]
    lowered = [word.lower() for word in case_words]
    for word in builtin_words:
        try:
            lowered.remove(word.lower())
        except ValueError:
            pass
    return tuple(word for word in lowered if word not in {"case"})


def validate_closure(
    *,
    discovered: tuple[CapabilityIdentity, ...],
    registrations: tuple[SemanticRegistration, ...],
    spec: str,
    generated_registry: str,
    essential_semantics: dict[str, tuple[str, ...]],
) -> None:
    violations: list[str] = []
    discovered_counts = Counter(discovered)
    for identity, count in discovered_counts.items():
        if count != 1:
            violations.append(f"duplicate discovered capability identity {identity.render()}")

    registration_counts = Counter(registration.identity for registration in registrations)
    for identity, count in registration_counts.items():
        if count != 1:
            violations.append(f"duplicate semantic registration for {identity.render()}")
    discovered_set = set(discovered)
    registered_set = set(registration_counts)
    for identity in sorted(discovered_set - registered_set):
        violations.append(f"missing semantic registration for {identity.render()}")
    for identity in sorted(registered_set - discovered_set):
        violations.append(f"stale semantic registration for {identity.render()}")

    blocks = atom_blocks(spec)
    generated = set(GENERATED_ATOM.findall(generated_registry))
    governed = {atom: governing_names(block) for atom, block in blocks.items()}
    by_builtin: dict[str, list[CapabilityIdentity]] = defaultdict(list)
    for identity in discovered:
        by_builtin[identity.builtin].append(identity)

    for registration in registrations:
        identity = registration.identity
        rendered = identity.render()
        lowered_fields = f"{identity.builtin} {identity.case}".lower()
        if FORBIDDEN_DEFAULT.search(lowered_fields) or "*" in lowered_fields:
            violations.append(f"forbidden default/fallback registration for {rendered}")
        if FORBIDDEN_COMPATIBILITY.search(lowered_fields):
            violations.append(f"forbidden compatibility/age exception for {rendered}")
        if ATOM_GRAMMAR.fullmatch(registration.atom) is None:
            violations.append(
                f"registration for {rendered} must cite exact `[05-OP-N]`, got `{registration.atom}`"
            )
            continue
        if registration.atom not in blocks:
            violations.append(
                f"atom {registration.atom} for {rendered} does not exist as one normative definition line"
            )
            continue
        if registration.atom not in generated:
            violations.append(
                f"generated rejection registry is missing {registration.atom} for {rendered}"
            )
        if identity.builtin not in governed[registration.atom]:
            violations.append(
                f"{registration.atom} does not name exact builtin `{identity.builtin}` in a governing position"
            )
        if len(by_builtin[identity.builtin]) > 1:
            normalized_block = _normalized(blocks[registration.atom]).lower()
            for word in _case_words(identity):
                if re.search(rf"(?<![a-z0-9_]){re.escape(word)}(?![a-z0-9_])", normalized_block) is None:
                    violations.append(
                        f"{registration.atom} does not govern exact case discriminator `{word}` for {rendered}"
                    )

    for atom in sorted({registration.atom for registration in registrations} - set(essential_semantics)):
        violations.append(
            f"registered atom {atom} has no essential-semantic mutation contract"
        )

    for atom, fragments in essential_semantics.items():
        block = blocks.get(atom)
        if block is None:
            violations.append(f"missing essential-semantics atom {atom}")
            continue
        normalized_block = _normalized(block)
        for fragment in fragments:
            if _normalized(fragment) not in normalized_block:
                violations.append(
                    f"{atom} missing essential semantic clause `{fragment}`"
                )

    if violations:
        raise OracleError("\n".join(violations))


def validate_repository(
    root: Path = REPO_ROOT,
    *,
    essential: dict[str, tuple[str, ...]] = ESSENTIAL_SEMANTICS,
) -> tuple[CapabilityIdentity, ...]:
    builtins = (root / BUILTINS_REL).read_text(encoding="utf-8")
    risc = (root / RISC_REL).read_text(encoding="utf-8")
    spec = (root / SPEC_REL).read_text(encoding="utf-8")
    generated = (root / GENERATED_REL).read_text(encoding="utf-8")

    validate_required_atom_numbers(spec)
    discovered = tuple(
        sorted(
            set(discover_builtin_capabilities(builtins))
            | set(discover_risc_atom_identities(risc))
        )
    )
    registrations = derive_registrations(discovered, spec)
    validate_closure(
        discovered=discovered,
        registrations=registrations,
        spec=spec,
        generated_registry=generated,
        essential_semantics=essential,
    )
    return discovered


def main(
    *,
    root: Path = REPO_ROOT,
    essential: dict[str, tuple[str, ...]] = ESSENTIAL_SEMANTICS,
) -> None:
    discovered = validate_repository(root, essential=essential)
    print(f"discovered {len(discovered)} exact Table-A/sibling capability identities")
    print(PASS_LINE)


def _run() -> int:
    try:
        main()
    except (OSError, OracleError) as error:
        print(f"DTYPE BUILTIN ATOM CLOSURE ORACLE: FAIL\n{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
