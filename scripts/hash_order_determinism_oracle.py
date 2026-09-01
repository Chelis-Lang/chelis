#!/usr/bin/env python3
"""Enforce and run the Phase B hash-order determinism contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from pathlib import PurePosixPath
import posixpath
import re
import subprocess
import sys
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(
    r"HashMap|HashSet|hash_map::|hash_set::|FxHashMap|FxHashSet|"
    r"rustc_hash::|fxhash::|ahash::|hashbrown::|indexmap::|(?<!\.)\binclude\b|"
    r"#\s*\[\s*allow\s*\(\s*clippy::disallowed_types\s*\)\s*\]"
)
RAW_STRING_PREFIX_PATTERN = re.compile(r'(?:b|c)?r(?P<hashes>#{0,255})"')
ITEM_PATTERN = re.compile(
    r"\b(?:fn|struct|enum|mod|type|const|static|trait)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
PATH_BEARING_ATTRIBUTE_PATTERN = re.compile(
    r"#\s*\[[^\]]*\bpath\b[^\]]*\]", re.DOTALL
)
DIRECT_PATH_ATTRIBUTE_PATTERN = re.compile(
    r'#\s*\[\s*path\s*=\s*"(?P<target>[A-Za-z0-9_./-]+)"\s*\]', re.DOTALL
)
PATH_MODULE_ITEM_PATTERN = re.compile(
    r"\s*(?:#\s*\[[^\]]*\]\s*)*mod\s+[A-Za-z_][A-Za-z0-9_]*\s*;"
)
MACRO_RULES_PATTERN = re.compile(
    r"\bmacro_rules\s*!\s*(?:r#)?[A-Za-z_][A-Za-z0-9_]*\s*"
    r"(?P<opener>[({\[])"
)
MACRO_ATTRIBUTE_TEMPLATE_PATTERN = re.compile(
    r"#\s*\[[^\]]*\$[^\]]*\]", re.DOTALL
)
DELIMITER_PAIRS = {"(": ")", "[": "]", "{": "}"}
DELIMITER_CLOSERS = frozenset(DELIMITER_PAIRS.values())


@dataclass(frozen=True, order=True)
class AllowedHit:
    path: str
    item: str
    token: str
    count: int
    reason: str


@dataclass(frozen=True, order=True)
class TokenHit:
    path: str
    item: str
    token: str
    line: int


@dataclass(frozen=True, order=True)
class ApprovedBuildScript:
    path: str
    sha256: str
    reason: str


# Each entry is an exact, reviewed item boundary. An entry without a matching
# token is stale and fails just like an unlisted token.
ALLOWED_HITS: tuple[AllowedHit, ...] = (
    AllowedHit(
        "crates/chelis-unord/src/lib.rs",
        "mod raw",
        "#[allow(clippy::disallowed_types)]",
        1,
        "the private wrapper boundary owns the only production lint allowance",
    ),
    AllowedHit(
        "crates/chelis-unord/src/lib.rs",
        "pub(crate) type Map",
        "HashMap",
        1,
        "private raw storage hidden behind the order-free wrapper",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "use std::collections::HashMap as RenamedMap",
        "HashMap",
        1,
        "compile-fail fixture proves renamed imports are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "type ForbiddenAlias",
        "HashMap",
        1,
        "compile-fail fixture proves type aliases are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn fully_qualified",
        "HashMap",
        2,
        "compile-fail fixture proves fully qualified uses are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn module_qualified",
        "hash_map::",
        2,
        "compile-fail fixture proves module-qualified uses are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn module_qualified",
        "HashMap",
        2,
        "compile-fail fixture proves module-qualified uses are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn glob_import",
        "HashSet",
        2,
        "compile-fail fixture proves glob-imported sets are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn aliased_receiver",
        "HashMap",
        1,
        "compile-fail fixture proves aliased receivers are rejected",
    ),
    AllowedHit(
        "crates/chelis-image-id/src/lib.rs",
        "use std::hash::{BuildHasher, Hasher}",
        "hash_map::",
        1,
        "degraded build fingerprints deliberately randomize toward cache misses",
    ),
)


# Build scripts are arbitrary compile-time Rust. Freezing the exact scripts
# makes adding or changing one an explicit review event; the separate
# reserved `include` identifier prevents an approved script from feeding
# generated Rust tokens into a workspace crate through the built-in macro or
# any imported alias. Method selectors such as `cc::Build::include` are not
# macro paths and remain valid.
APPROVED_BUILD_SCRIPTS: tuple[ApprovedBuildScript, ...] = (
    ApprovedBuildScript(
        "crates/chelis-backend-c/build.rs",
        "f5fa2ee687ec90cb391f13d2d5aab07677aab290e499cce0aae642b93b1d04c8",
        "feature detection only; does not generate Rust",
    ),
    ApprovedBuildScript(
        "crates/chelis-conformance/build.rs",
        "a48d0555bbcfa20440b8ebe5aa14b50a762b8f9ab7dc01fa0e563c5ddf38c8dc",
        "validates committed assets only; does not generate Rust",
    ),
    ApprovedBuildScript(
        "crates/chelis-std-bundle/build.rs",
        "4093d541106548e5a28f7269c27e75be8addffad9831e0e13aa37acd498a3a1f",
        "validates committed bundle artifacts only; does not generate Rust",
    ),
    ApprovedBuildScript(
        "tree-sitter-chelis/build.rs",
        "043cbb4029bbd4d9c63016e80d9bc8ab39e7e0f52e2e8752772e4cdbc5bea1cc",
        "compiles committed parser sources only; does not generate Rust",
    ),
)


COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "scripts/check_hash_order_phase_b_compile_fail.py"),
    ("cargo", "nextest", "run", "-p", "chelis-unord", "--no-fail-fast"),
    (
        "cargo",
        "test",
        "-p",
        "chelis-unord",
        "--release",
        "--test",
        "serde_contract",
        "debug_is_deterministic_and_does_not_expose_contents",
        "--",
        "--exact",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "hash_order_cache_bytes",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "test",
        "-p",
        "chelis-cli",
        "--test",
        "stdlib_typecheck_cache_oracle",
        "stale_stdlib_byte_mutation_misses_not_stale_hit",
        "--",
        "--exact",
        "--nocapture",
    ),
    (sys.executable, "scripts/hash_order_phase_a_oracle.py"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-backend-c",
        "--test",
        "codegen_determinism",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "rank_poly_tier3",
        "form3_bias_broadcast_c_is_byte_deterministic",
        "--no-fail-fast",
    ),
)


class HashOrderDeterminismFailure(RuntimeError):
    """The structural or executable determinism contract failed."""


def _strip_comments_and_strings(text: str) -> str:
    """Blank Rust comments and string contents while preserving line layout."""

    output = list(text)
    index = 0
    block_depth = 0
    length = len(text)
    while index < length:
        if block_depth:
            if text.startswith("/*", index):
                output[index : index + 2] = "  "
                block_depth += 1
                index += 2
            elif text.startswith("*/", index):
                output[index : index + 2] = "  "
                block_depth -= 1
                index += 2
            else:
                if text[index] != "\n":
                    output[index] = " "
                index += 1
            continue

        if text.startswith("//", index):
            end = text.find("\n", index)
            if end < 0:
                end = length
            output[index:end] = " " * (end - index)
            index = end
            continue
        if text.startswith("/*", index):
            output[index : index + 2] = "  "
            block_depth = 1
            index += 2
            continue

        raw_match = RAW_STRING_PREFIX_PATTERN.match(text, index)
        if raw_match is not None:
            opener = raw_match.group(0)
            hashes = raw_match.group("hashes")
            terminator = f'"{hashes}'
            end = text.find(terminator, index + len(opener))
            end = length if end < 0 else end + len(terminator)
            for position in range(index, end):
                if text[position] != "\n":
                    output[position] = " "
            index = end
            continue

        quote_offset = 1 if text.startswith(('b"', 'c"'), index) else 0
        if text[index + quote_offset : index + quote_offset + 1] == '"':
            end = index + quote_offset + 1
            escaped = False
            while end < length:
                char = text[end]
                end += 1
                if char == '"' and not escaped:
                    break
                if char == "\\" and not escaped:
                    escaped = True
                else:
                    escaped = False
            for position in range(index, end):
                if text[position] != "\n":
                    output[position] = " "
            index = end
            continue
        index += 1
    return "".join(output)


def _char_literal_end(text: str, index: int) -> int | None:
    """Return the end of a char literal, without mistaking a lifetime for one."""

    quote = index + 1 if text.startswith("b'", index) else index
    if text[quote : quote + 1] != "'":
        return None
    cursor = quote + 1
    escaped = False
    while cursor < len(text) and text[cursor] != "\n":
        char = text[cursor]
        cursor += 1
        if char == "'" and not escaped:
            return cursor
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return None


def _normal_rust_string(
    path: str, text: str, quote_index: int
) -> tuple[str, int]:
    """Decode one ordinary UTF-8 Rust string and return its exclusive end."""

    value: list[str] = []
    cursor = quote_index + 1
    while cursor < len(text):
        char = text[cursor]
        if char == '"':
            return "".join(value), cursor + 1
        if char != "\\":
            value.append(char)
            cursor += 1
            continue

        escape_start = cursor
        cursor += 1
        if cursor >= len(text):
            break
        escaped = text[cursor]
        simple = {
            "0": "\0",
            "t": "\t",
            "n": "\n",
            "r": "\r",
            '"': '"',
            "'": "'",
            "\\": "\\",
        }
        if escaped in simple:
            value.append(simple[escaped])
            cursor += 1
            continue
        if escaped == "\n":
            cursor += 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            continue
        if escaped == "\r" and text[cursor : cursor + 2] == "\r\n":
            cursor += 2
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            continue
        if escaped == "x":
            digits = text[cursor + 1 : cursor + 3]
            if len(digits) == 2 and all(digit in "0123456789abcdefABCDEF" for digit in digits):
                value.append(chr(int(digits, 16)))
                cursor += 3
                continue
        if escaped == "u" and text[cursor + 1 : cursor + 2] == "{":
            close = text.find("}", cursor + 2)
            if close >= 0:
                digits = text[cursor + 2 : close].replace("_", "")
                if 1 <= len(digits) <= 6 and all(
                    digit in "0123456789abcdefABCDEF" for digit in digits
                ):
                    try:
                        value.append(chr(int(digits, 16)))
                    except ValueError:
                        pass
                    else:
                        cursor = close + 1
                        continue
        raise HashOrderDeterminismFailure(
            f"unsupported Rust string escape at {path}:"
            f"{text.count(chr(10), 0, escape_start) + 1}"
        )
    raise HashOrderDeterminismFailure(
        f"unterminated Rust string at {path}:"
        f"{text.count(chr(10), 0, quote_index) + 1}"
    )


def _block_comment_end(path: str, text: str, index: int) -> int:
    """Return the exclusive end of a nested Rust block comment."""

    depth = 1
    cursor = index + 2
    while cursor < len(text):
        if text.startswith("/*", cursor):
            depth += 1
            cursor += 2
        elif text.startswith("*/", cursor):
            depth -= 1
            cursor += 2
            if depth == 0:
                return cursor
        else:
            cursor += 1
    raise HashOrderDeterminismFailure(
        f"unterminated Rust block comment at {path}:"
        f"{text.count(chr(10), 0, index) + 1}"
    )


def _rust_utf8_string_literals(path: str, text: str) -> tuple[str, ...]:
    """Extract source and compiler-desugared UTF-8 string literal tokens."""

    literals: list[str] = []
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            is_outer_doc = text.startswith("///", index) and not text.startswith(
                "////", index
            )
            if is_outer_doc or text.startswith("//!", index):
                literals.append(text[index + 3 : end])
            index = end
            continue
        if text.startswith("/*", index):
            end = _block_comment_end(path, text, index)
            is_outer_doc = text.startswith("/**", index) and not text.startswith(
                "/***", index
            )
            if is_outer_doc or text.startswith("/*!", index):
                literals.append(text[index + 3 : end - 2])
            index = end
            continue

        char_end = _char_literal_end(text, index)
        if char_end is not None:
            index = char_end
            continue

        raw_match = RAW_STRING_PREFIX_PATTERN.match(text, index)
        if raw_match is not None:
            opener = raw_match.group(0)
            hashes = raw_match.group("hashes")
            terminator = f'"{hashes}'
            body_start = index + len(opener)
            body_end = text.find(terminator, body_start)
            if body_end < 0:
                raise HashOrderDeterminismFailure(
                    f"unterminated raw Rust string at {path}:"
                    f"{text.count(chr(10), 0, index) + 1}"
                )
            if opener.startswith("r"):
                literals.append(text[body_start:body_end])
            index = body_end + len(terminator)
            continue

        if text[index] == '"':
            value, index = _normal_rust_string(path, text, index)
            literals.append(value)
            continue
        if text.startswith(('b"', 'c"'), index):
            _, index = _normal_rust_string(path, text, index + 1)
            continue
        index += 1
    return tuple(literals)


def _token_tree_end(path: str, text: str, open_index: int) -> int:
    """Return the exclusive end of one balanced Rust token tree."""

    opener = text[open_index]
    if opener not in DELIMITER_PAIRS:
        raise AssertionError(f"not a token-tree opener: {opener!r}")
    stack = [opener]
    index = open_index + 1
    while index < len(text):
        char_end = _char_literal_end(text, index)
        if char_end is not None:
            index = char_end
            continue
        char = text[index]
        if char in DELIMITER_PAIRS:
            stack.append(char)
        elif char in DELIMITER_CLOSERS:
            expected = DELIMITER_PAIRS[stack[-1]]
            if char != expected:
                raise HashOrderDeterminismFailure(
                    f"unbalanced macro token tree at {path}:"
                    f"{text.count(chr(10), 0, index) + 1}"
                )
            stack.pop()
            if not stack:
                return index + 1
        index += 1
    raise HashOrderDeterminismFailure(
        f"unterminated macro token tree at {path}:"
        f"{text.count(chr(10), 0, open_index) + 1}"
    )


def _reject_macro_generated_attributes(path: str, sanitized: str) -> None:
    """Forbid local macro transcribers from constructing Rust attributes."""

    cursor = 0
    while match := MACRO_RULES_PATTERN.search(sanitized, cursor):
        body_end = _token_tree_end(path, sanitized, match.start("opener"))
        body = sanitized[match.end("opener") : body_end - 1]
        for arrow in re.finditer(r"=>", body):
            transcriber_start = arrow.end()
            while (
                transcriber_start < len(body)
                and body[transcriber_start].isspace()
            ):
                transcriber_start += 1
            if (
                transcriber_start >= len(body)
                or body[transcriber_start] not in DELIMITER_PAIRS
            ):
                continue
            transcriber_end = _token_tree_end(path, body, transcriber_start)
            transcriber = body[transcriber_start + 1 : transcriber_end - 1]
            generated = MACRO_ATTRIBUTE_TEMPLATE_PATTERN.search(transcriber)
            if generated is not None:
                absolute_start = (
                    match.end("opener") + transcriber_start + 1 + generated.start()
                )
                raise HashOrderDeterminismFailure(
                    f"macro-generated external-module attribute risk at {path}:"
                    f"{sanitized.count(chr(10), 0, absolute_start) + 1}; "
                    "local macro transcribers may not interpolate Rust attributes "
                    "because they can synthesize an unscanned #[path] module"
                )
        cursor = body_end


def _item_for_line(lines: Sequence[str], line_index: int) -> str:
    current = lines[line_index].strip()
    if current.startswith("#["):
        for candidate in lines[line_index + 1 :]:
            match = ITEM_PATTERN.search(candidate)
            if match is not None:
                return f"{candidate[:match.start()].strip()} {match.group(0)}".strip()
            if candidate.strip() and not candidate.lstrip().startswith("#"):
                break
    for candidate in reversed(lines[: line_index + 1]):
        match = ITEM_PATTERN.search(candidate)
        if match is not None:
            prefix = candidate[: match.start()].strip()
            return f"{prefix} {match.group(0)}".strip()
        stripped = candidate.strip()
        if stripped.startswith("use "):
            return stripped.rstrip(";")
    return current.rstrip(";")


def token_hits(path: str, text: str) -> tuple[TokenHit, ...]:
    sanitized = _strip_comments_and_strings(text)
    lines = sanitized.splitlines()
    hits: list[TokenHit] = []
    for line_index, line in enumerate(lines):
        for match in TOKEN_PATTERN.finditer(line):
            hits.append(
                TokenHit(
                    path=path,
                    item=_item_for_line(lines, line_index),
                    token=match.group(0),
                    line=line_index + 1,
                )
            )
    return tuple(hits)


def _path_module_targets(path: str, text: str) -> tuple[str, ...]:
    """Return every direct `#[path] mod` target, rejecting opaque forms."""

    sanitized = _strip_comments_and_strings(text)
    _reject_macro_generated_attributes(path, sanitized)
    direct_by_start = {
        match.start(): match for match in DIRECT_PATH_ATTRIBUTE_PATTERN.finditer(text)
    }
    targets: list[str] = []
    for attribute in PATH_BEARING_ATTRIBUTE_PATTERN.finditer(sanitized):
        direct = direct_by_start.get(attribute.start())
        if direct is None or direct.end() != attribute.end():
            raise HashOrderDeterminismFailure(
                f"unsupported path-bearing attribute at {path}:"
                f"{sanitized.count(chr(10), 0, attribute.start()) + 1}; "
                "use one direct #[path = \"relative/file.rs\"] module attribute"
            )
        if PATH_MODULE_ITEM_PATTERN.match(sanitized, attribute.end()) is None:
            raise HashOrderDeterminismFailure(
                f"unsupported #[path] target at {path}:"
                f"{sanitized.count(chr(10), 0, attribute.start()) + 1}; "
                "the attribute must apply directly to an external mod item"
            )
        target = direct.group("target")
        if target.startswith("/"):
            raise HashOrderDeterminismFailure(
                f"absolute #[path] target {target!r} at {path} is forbidden"
            )
        resolved = posixpath.normpath(
            str(PurePosixPath(path).parent.joinpath(target))
        )
        if resolved == ".." or resolved.startswith("../"):
            raise HashOrderDeterminismFailure(
                f"#[path] target {target!r} at {path} escapes the repository"
            )
        if not resolved.endswith(".rs"):
            raise HashOrderDeterminismFailure(
                f"#[path] target {target!r} at {path} is not a Rust source"
            )
        targets.append(resolved)
    return tuple(targets)


def validate_sources(
    sources: Mapping[str, str],
    allowed: Sequence[AllowedHit] = ALLOWED_HITS,
    approved_build_scripts: Sequence[ApprovedBuildScript] = APPROVED_BUILD_SCRIPTS,
) -> None:
    missing_path_targets = sorted(
        (path, target)
        for path, source in sources.items()
        for target in _path_module_targets(path, source)
        if target not in sources
    )
    if missing_path_targets:
        details = "; ".join(
            f"unscanned #[path] target {target} referenced by {path}"
            for path, target in missing_path_targets
        )
        raise HashOrderDeterminismFailure(details)

    approved_build_by_path = {
        entry.path: entry for entry in approved_build_scripts
    }
    if len(approved_build_by_path) != len(approved_build_scripts):
        raise HashOrderDeterminismFailure("duplicate approved build-script entry")
    observed_build_scripts = {
        path: source
        for path, source in sources.items()
        if Path(path).name == "build.rs"
    }
    unexpected_build_scripts = sorted(
        set(observed_build_scripts).difference(approved_build_by_path)
    )
    stale_build_scripts = sorted(
        set(approved_build_by_path).difference(observed_build_scripts)
    )
    changed_build_scripts = sorted(
        path
        for path, source in observed_build_scripts.items()
        if path in approved_build_by_path
        and hashlib.sha256(source.encode("utf-8")).hexdigest()
        != approved_build_by_path[path].sha256
    )
    if unexpected_build_scripts or stale_build_scripts or changed_build_scripts:
        details = [
            *(f"unapproved build script {path}" for path in unexpected_build_scripts),
            *(f"stale approved build-script entry {path}" for path in stale_build_scripts),
            *(f"changed approved build script {path}" for path in changed_build_scripts),
        ]
        raise HashOrderDeterminismFailure("; ".join(details))

    hits = tuple(
        hit
        for path, source in sorted(sources.items())
        for hit in token_hits(path, source)
    )
    allowed_by_identity = {
        (entry.path, entry.item, entry.token): entry for entry in allowed
    }
    duplicate_entries = len(allowed_by_identity) != len(allowed)
    if duplicate_entries:
        raise HashOrderDeterminismFailure("duplicate hash-order allow-list entry")
    invalid_counts = [entry for entry in allowed if entry.count <= 0]
    if invalid_counts:
        raise HashOrderDeterminismFailure("allow-list token counts must be positive")

    unexpected = [
        hit
        for hit in hits
        if (hit.path, hit.item, hit.token) not in allowed_by_identity
    ]
    observed_counts = Counter((hit.path, hit.item, hit.token) for hit in hits)
    stale = [
        entry
        for entry in allowed
        if observed_counts[(entry.path, entry.item, entry.token)] == 0
    ]
    wrong_counts = [
        entry
        for entry in allowed
        if observed_counts[(entry.path, entry.item, entry.token)]
        not in (0, entry.count)
    ]
    if unexpected or stale or wrong_counts:
        details = []
        details.extend(
            f"unlisted token {hit.token!r} at {hit.path}:{hit.line} ({hit.item})"
            for hit in unexpected
        )
        details.extend(
            f"stale allow-list entry {entry.path} ({entry.item}, {entry.token!r}): "
            f"{entry.reason}"
            for entry in stale
        )
        details.extend(
            f"allow-list token count mismatch at {entry.path} ({entry.item}, "
            f"{entry.token!r}): expected {entry.count}, observed "
            f"{observed_counts[(entry.path, entry.item, entry.token)]}"
            for entry in wrong_counts
        )
        raise HashOrderDeterminismFailure("; ".join(details))


def tracked_rust_sources(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    repo_root = repo_root.resolve()
    completed = subprocess.run(
        (
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.rs",
        ),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    sources: dict[str, str] = {}
    for path in completed.stdout.splitlines():
        sources[path] = (repo_root / path).read_text(encoding="utf-8")

    literals_by_owner: dict[str, tuple[str, ...]] = {}
    pending = list(sorted(sources))
    while pending:
        while pending:
            owner = pending.pop()
            source = sources[owner]
            literals_by_owner[owner] = tuple(
                literal
                for literal in _rust_utf8_string_literals(owner, source)
                if literal.endswith(".rs") and "\0" not in literal
            )
            for target in _path_module_targets(owner, source):
                if target in sources:
                    continue
                candidate = repo_root / target
                try:
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(repo_root)
                except (FileNotFoundError, ValueError) as error:
                    raise HashOrderDeterminismFailure(
                        f"#[path] target {target!r} referenced by {owner} is missing or "
                        "escapes the repository"
                    ) from error
                if not resolved.is_file():
                    raise HashOrderDeterminismFailure(
                        f"#[path] target {target!r} referenced by {owner} is not a file"
                    )
                sources[target] = resolved.read_text(encoding="utf-8")
                pending.append(target)

        # A declarative macro may move a string literal from its definition or
        # invocation into an attribute at another source location. Resolve each
        # Rust-source literal against every scanned source directory. Existing
        # in-repository targets form a conservative source-reachability closure;
        # missing ordinary fixture-name strings do not create source inputs.
        source_directories = {
            PurePosixPath(path).parent for path in sources
        }
        discovered: dict[str, str] = {}
        for owner, literals in sorted(literals_by_owner.items()):
            for literal in literals:
                if posixpath.isabs(literal):
                    candidate_bases = (PurePosixPath("."),)
                else:
                    candidate_bases = tuple(sorted(source_directories))
                for base in candidate_bases:
                    candidate = Path(literal) if posixpath.isabs(literal) else repo_root / base / literal
                    try:
                        resolved = candidate.resolve(strict=True)
                    except FileNotFoundError:
                        continue
                    try:
                        relative = resolved.relative_to(repo_root)
                    except ValueError as error:
                        raise HashOrderDeterminismFailure(
                            f"Rust-source literal {literal!r} at {owner} resolves outside "
                            "the repository and cannot be scanned"
                        ) from error
                    if not resolved.is_file():
                        continue
                    target = relative.as_posix()
                    if target not in sources:
                        discovered[target] = resolved.read_text(encoding="utf-8")
        if discovered:
            sources.update(discovered)
            pending.extend(sorted(discovered))
    return sources


def validate(
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    commands: Sequence[Sequence[str]] = COMMANDS,
    source_loader: Callable[[], Mapping[str, str]] = tracked_rust_sources,
    allowed: Sequence[AllowedHit] = ALLOWED_HITS,
    approved_build_scripts: Sequence[ApprovedBuildScript] = APPROVED_BUILD_SCRIPTS,
) -> None:
    validate_sources(source_loader(), allowed, approved_build_scripts)
    for command in commands:
        completed = runner(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            rendered = " ".join(command)
            raise HashOrderDeterminismFailure(
                f"command exited {completed.returncode}: {rendered}"
            )
    print("HASH ORDER DETERMINISM ORACLE: PASS", flush=True)


def validate_tripwire(
    *,
    source_loader: Callable[[], Mapping[str, str]] = tracked_rust_sources,
    allowed: Sequence[AllowedHit] = ALLOWED_HITS,
    approved_build_scripts: Sequence[ApprovedBuildScript] = APPROVED_BUILD_SCRIPTS,
) -> None:
    validate_sources(source_loader(), allowed, approved_build_scripts)
    print("HASH ORDER TOKEN TRIPWIRE: PASS", flush=True)


def main() -> int:
    try:
        if sys.argv[1:] == ["--scan-only"]:
            validate_tripwire()
        elif sys.argv[1:]:
            raise HashOrderDeterminismFailure(
                "usage: hash_order_determinism_oracle.py [--scan-only]"
            )
        else:
            validate()
    except (HashOrderDeterminismFailure, OSError, subprocess.SubprocessError) as error:
        print(f"HASH ORDER DETERMINISM ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
