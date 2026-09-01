#!/usr/bin/env python3
"""Enforce and run the Phase B hash-order determinism contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
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


@dataclass(frozen=True, order=True)
class AllowedHit:
    path: str
    item: str
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
        "the private wrapper boundary owns the only production lint allowance",
    ),
    AllowedHit(
        "crates/chelis-unord/src/lib.rs",
        "pub(crate) type Map",
        "private raw storage hidden behind the order-free wrapper",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "use std::collections::HashMap as RenamedMap",
        "compile-fail fixture proves renamed imports are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "type ForbiddenAlias",
        "compile-fail fixture proves type aliases are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn fully_qualified",
        "compile-fail fixture proves fully qualified uses are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn module_qualified",
        "compile-fail fixture proves module-qualified uses are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn glob_import",
        "compile-fail fixture proves glob-imported sets are rejected",
    ),
    AllowedHit(
        "crates/chelis-unord/tests/compile_fail/disallowed_hash_types/src/main.rs",
        "fn aliased_receiver",
        "compile-fail fixture proves aliased receivers are rejected",
    ),
    AllowedHit(
        "crates/chelis-compiler-api/src/lib.rs",
        "use std::hash::{BuildHasher, Hasher}",
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


def validate_sources(
    sources: Mapping[str, str],
    allowed: Sequence[AllowedHit] = ALLOWED_HITS,
    approved_build_scripts: Sequence[ApprovedBuildScript] = APPROVED_BUILD_SCRIPTS,
) -> None:
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
    allowed_by_item = {(entry.path, entry.item): entry for entry in allowed}
    duplicate_entries = len(allowed_by_item) != len(allowed)
    if duplicate_entries:
        raise HashOrderDeterminismFailure("duplicate hash-order allow-list entry")

    unexpected = [
        hit for hit in hits if (hit.path, hit.item) not in allowed_by_item
    ]
    hit_items = {(hit.path, hit.item) for hit in hits}
    stale = [
        entry for entry in allowed if (entry.path, entry.item) not in hit_items
    ]
    if unexpected or stale:
        details = []
        details.extend(
            f"unlisted token {hit.token!r} at {hit.path}:{hit.line} ({hit.item})"
            for hit in unexpected
        )
        details.extend(
            f"stale allow-list entry {entry.path} ({entry.item}): {entry.reason}"
            for entry in stale
        )
        raise HashOrderDeterminismFailure("; ".join(details))


def tracked_rust_sources() -> dict[str, str]:
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
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sources: dict[str, str] = {}
    for path in completed.stdout.splitlines():
        sources[path] = (REPO_ROOT / path).read_text(encoding="utf-8")
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
