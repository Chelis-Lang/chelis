#!/usr/bin/env python3
"""Fail on runtime-archive lookups outside the runtime bundle (chelis#1354).

`chelis-runtime-bundle` owns the runtime archive: a build embeds the archive it
linked, `stage` writes those bytes, and `chelis runtime export` wraps `stage`
(spec/08-backends.md §2.1). Code that instead finds an archive, by a hashed-name
scan, a `CHELIS_RUNTIME_DIR` or `CHELIS_RUNTIME_LIB` read, a `-lchelis_runtime`
search, Cargo's artifact report or a build-tree path, can link bytes its
consumer was not built with.

The scan reads every tracked or untracked, not ignored, Rust, Python, Nix,
shell, TOML and YAML file outside the two bundle crates. The number of matches
of each pattern in each file must equal the count of a reviewed row. A `lookup`
row names a remaining lookup and the issue that removes it; a `not-lookup` row
says why the text is not one. A new match fails, and so does a row whose count
no longer matches, so the change that removes a lookup also removes its row.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCANNED_SUFFIXES = frozenset({".rs", ".py", ".nix", ".sh", ".toml", ".yml", ".yaml"})
# The runtime's owners, and this guard, whose patterns and samples are data.
OWNER_PREFIXES = ("crates/chelis-runtime-bundle/", "crates/chelis-runtime-bundle-macro/")
GUARD_FILES = frozenset(
    {"scripts/check_runtime_archive_lookups.py", "scripts/test_check_runtime_archive_lookups.py"}
)
DISPOSITIONS = frozenset({"lookup", "not-lookup"})
ISSUE = re.compile(r"chelis#[1-9][0-9]*")


@dataclass(frozen=True)
class Pattern:
    name: str
    # A literal every alternative of `regex` contains; files without it are skipped.
    token: str
    regex: re.Pattern[str]
    # A line the pattern must match, so each pattern is proved live.
    sample: str
    # False: no reviewed row may allow a match.
    reviewable: bool = True


PATTERNS = (
    Pattern(
        "linker-search",
        "chelis_runtime",
        re.compile(r"-l:?(?:lib)?chelis_runtime\b|[\"']-l[\"']\s*,\s*[\"']:?(?:lib)?chelis_runtime\b"),
        'cmd.args(["-L.", "-lchelis_runtime"]);',
    ),
    Pattern(
        "runtime-variable-read",
        "CHELIS_RUNTIME_",
        re.compile(
            r"env::var(?:_os)?\(\s*\"CHELIS_RUNTIME_"
            r"|(?:option_)?env!\(\s*\"CHELIS_RUNTIME_"
            r"|(?:environ\.get|getenv|environ\.setdefault)\(\s*[\"']CHELIS_RUNTIME_"
            r"|environ\[\s*[\"']CHELIS_RUNTIME_"
            r"|[\"']CHELIS_RUNTIME_(?:DIR|LIB)[\"']\s+(?:not\s+)?in\s+(?:os\.)?environ\b"
            r"|\$\{?CHELIS_RUNTIME_(?:DIR|LIB)\b"
            r"|\benv\.CHELIS_RUNTIME_(?:DIR|LIB)\b"
            r"|getEnv\s+\"CHELIS_RUNTIME_"
            # A name bound to the variable's name reads it wherever it is used.
            r"|\b[A-Za-z_]\w*\s*(?::[^=\n]*)?=\s*[\"']CHELIS_RUNTIME_(?:DIR|LIB)[\"']"
        ),
        'let dir = std::env::var_os("CHELIS_RUNTIME_DIR");',
    ),
    Pattern(
        "runtime-lib-variable",
        "CHELIS_RUNTIME_LIB",
        re.compile(r"\bCHELIS_RUNTIME_LIB\b"),
        'CHELIS_RUNTIME_LIB="$PWD/target/debug/libchelis_runtime.a" cargo nextest run',
        reviewable=False,
    ),
    Pattern(
        "archive-name-match",
        "chelis_runtime",
        re.compile(
            r"(?:lib)?chelis_runtime-(?:[*?\[{]|[0-9a-f]{4})"
            r"|(?:lib)?chelis_runtime\*"
            r"|(?:starts_?with|strip_prefix|hasPrefix|fnmatch|r?glob)\s*\(\s*r?[\"'](?:lib)?chelis_runtime"
            r"|[\"'](?:lib)?chelis_runtime[^\"'\n]*[\"']\s+(?:not\s+)?in\s+\w"
        ),
        "archive = next((target / 'debug' / 'deps').glob('libchelis_runtime-*.a'))",
    ),
    Pattern(
        "cargo-artifact-read",
        "chelis_runtime",
        re.compile(
            r"name[\"']?\s*[\])]?\s*==\s*[\"']chelis_runtime[\"']"
            r"|[\"']chelis_runtime[\"']\s*=="
        ),
        'if row["target"]["name"] == "chelis_runtime" {',
    ),
    Pattern(
        "build-tree-archive",
        "libchelis_runtime",
        re.compile(
            r"\b(?:target|deps|debug|release)/[^\s\"']*libchelis_runtime"
            r"|[\"'](?:target|deps|debug|release)[\"'][^\n]*?[\"']libchelis_runtime"
        ),
        "cp target/release/libchelis_runtime.a dist/",
    ),
)
PATTERN_NAMES = {pattern.name: pattern for pattern in PATTERNS}


@dataclass(frozen=True)
class Row:
    path: str
    pattern: str
    count: int
    disposition: str
    reason: str
    tracking: str | None = None


# The reviewed matches. A lookup row's tracking issue owns its removal.
REVIEWED: tuple[Row, ...] = (
    Row(
        "crates/chelis-cli/tests/std_io_pipeline.rs",
        "linker-search",
        1,
        "lookup",
        "an entirely ignored manual gate links `-L. -lchelis_runtime` in its `chelis build` "
        "output directory; linking the staged archive by path needs its manual-only or "
        "manual-gate row in the same change",
        "chelis#1354",
    ),
    Row(
        "crates/chelis-cli/tests/cross_library_semantic_gap_hip_gpu.rs",
        "linker-search",
        1,
        "lookup",
        "an entirely ignored HIP gate links `-L. -lchelis_runtime` in its `chelis build` "
        "output directory; linking the staged archive by path needs its manual-gate row "
        "and wired docs/manual_gates.md entry in the same change",
        "chelis#1354",
    ),
    Row(
        "crates/chelis-cli/tests/issue_1314_json_bigint.rs",
        "cargo-artifact-read",
        1,
        "lookup",
        "copies a separately built ownership-ledger archive over the staged runtime instead "
        "of building its consumer with chelis-runtime/ownership-ledger",
        "chelis#1354",
    ),
    Row(
        "crates/chelis-compiler-api/tests/ownership_support/mod.rs",
        "cargo-artifact-read",
        1,
        "lookup",
        "links a separately built ownership-ledger archive instead of building its consumer "
        "with chelis-runtime/ownership-ledger",
        "chelis#1354",
    ),
    Row(
        "crates/chelis-compiler-api/tests/ownership_support/mod.rs",
        "archive-name-match",
        1,
        "lookup",
        "stages that separately built archive under a content-addressed "
        "`libchelis_runtime-<hash>.a` name",
        "chelis#1354",
    ),
    Row(
        "crates/chelis-compiler-api/tests/ownership_support/mod.rs",
        "build-tree-archive",
        1,
        "not-lookup",
        "a comment explaining why the harness does not link the uplifted build-tree archive",
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "linker-search",
        2,
        "not-lookup",
        "asserts that printed link commands carry no library search",
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "archive-name-match",
        1,
        "not-lookup",
        "plants a stale hashed archive beside the CLI that `chelis build` must not stage",
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "build-tree-archive",
        1,
        "not-lookup",
        "plants a stale hashed archive beside the CLI that `chelis build` must not stage",
    ),
    Row(
        "crates/chelis-python/src/lib.rs",
        "runtime-variable-read",
        1,
        "not-lookup",
        "saves the caller's value around the test that the extension rejects the variable",
    ),
    Row(
        "scripts/compiled_value_ownership_oracle.py",
        "linker-search",
        1,
        "not-lookup",
        "rejects a compile command that searches for the runtime",
    ),
    Row(
        "scripts/compiled_value_ownership_oracle.py",
        "cargo-artifact-read",
        1,
        "not-lookup",
        "requires the CLI's Cargo build to have compiled chelis-runtime with the ledger "
        "feature; the reference digest comes from the CLI's own export",
    ),
    Row(
        "scripts/compiled_value_ownership_oracle.py",
        "runtime-variable-read",
        1,
        "not-lookup",
        "names the variable only to refuse an inherited value",
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "linker-search",
        1,
        "not-lookup",
        "a searched compile command the oracle must reject",
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "archive-name-match",
        1,
        "not-lookup",
        "a decoy Cargo deps archive whose bytes the oracle must not take as the runtime",
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "build-tree-archive",
        1,
        "not-lookup",
        "a decoy Cargo deps archive whose bytes the oracle must not take as the runtime",
    ),
    Row(
        "scripts/test_nix_flake_contract.py",
        "build-tree-archive",
        1,
        "not-lookup",
        "asserts that the release no longer copies the build-tree archive",
    ),
)


def repository_files(root: Path = ROOT) -> list[str]:
    """Every file a developer can see: tracked, plus untracked and not ignored."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")
    return sorted({name for name in listed.split("\0") if name})


def scan(root: Path, names: Iterable[str]) -> dict[tuple[str, str], list[str]]:
    """Map each (file, pattern) with a match to its `path:line: text` locations."""
    found: dict[tuple[str, str], list[str]] = {}
    for name in names:
        if (
            Path(name).suffix not in SCANNED_SUFFIXES
            or name.startswith(OWNER_PREFIXES)
            or name in GUARD_FILES
        ):
            continue
        path = root / name
        # A deleted-but-listed file has no text; a symlink's target is scanned under its own name.
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for pattern in PATTERNS:
            if pattern.token not in text:
                continue
            for match in pattern.regex.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                found.setdefault((name, pattern.name), []).append(
                    f"{name}:{line}: {lines[line - 1].strip()}"
                )
    return found


def row_errors(rows: Sequence[Row]) -> list[str]:
    errors = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.path, row.pattern)
        label = f"row {row.path} [{row.pattern}]"
        if key in seen:
            errors.append(f"{label} is duplicated")
        seen.add(key)
        pattern = PATTERN_NAMES.get(row.pattern)
        if pattern is None:
            errors.append(f"{label} names an unknown pattern")
        elif not pattern.reviewable:
            errors.append(f"{label}: {row.pattern} admits no reviewed rows")
        if row.path.startswith(OWNER_PREFIXES) or row.path in GUARD_FILES:
            errors.append(f"{label} names a path the scan exempts")
        if row.count < 1:
            errors.append(f"{label} must allow at least one match")
        if row.disposition not in DISPOSITIONS:
            errors.append(f"{label} has disposition {row.disposition!r}")
        if not row.reason.strip():
            errors.append(f"{label} has no reason")
        if row.disposition == "lookup" and (
            row.tracking is None or not ISSUE.fullmatch(row.tracking)
        ):
            errors.append(f"{label} is a lookup without a chelis#N tracking issue")
    return errors


def check(
    root: Path = ROOT,
    rows: Sequence[Row] = REVIEWED,
    names: Sequence[str] | None = None,
) -> list[str]:
    """Return every failure; an empty list means the scan equals the reviewed rows."""
    if names is None:
        names = repository_files(root)
    errors = row_errors(rows)
    for prefix in OWNER_PREFIXES:
        if not any(name.startswith(prefix) for name in names):
            errors.append(f"exempt owner {prefix} no longer exists; remove its exemption")
    found = scan(root, names)
    allowed = {(row.path, row.pattern): row for row in rows}
    for key, locations in sorted(found.items()):
        row = allowed.get(key)
        if row is None or len(locations) > row.count:
            baseline = 0 if row is None else row.count
            errors.append(
                f"{key[1]}: {len(locations)} match(es) in {key[0]}, {baseline} reviewed. "
                "Link the archive that `chelis build`, `chelis_runtime_bundle::stage` or "
                "`chelis runtime export` wrote, by its exact path; or, if the text is not "
                f"a lookup, add a reviewed not-lookup row.\n  " + "\n  ".join(locations)
            )
    for key, row in sorted(allowed.items()):
        observed = len(found.get(key, ()))
        if observed < row.count:
            errors.append(
                f"stale row {row.path} [{row.pattern}]: {observed} match(es), {row.count} "
                "reviewed. Shrink or delete the row in the change that removed the match."
            )
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("RUNTIME ARCHIVE LOOKUPS: FAIL", file=sys.stderr)
        return 1
    print("RUNTIME ARCHIVE LOOKUPS: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
