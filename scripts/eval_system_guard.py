#!/usr/bin/env python3
"""Tripwire for recognized direct evaluator host-access syntax.

The eight shipped filesystem/process evaluator builtins route through the
mandatory typed `EvalSystem` policy boundary. This separate source scan
reports conventional direct host calls and imports outside
`runtime/system_adapter.rs`, including the tested `use std as host` and
platform-specific filesystem spellings. It skips test-only modules.

This lexical check is NOT proof that the adapter is the only possible Rust
host-access path: for example, `extern crate std as host` and filesystem
methods on path values derived through `.to_path_buf()` can evade it.
Those cases require type-aware analysis; a PASS only means the recognized
patterns were absent from the scanned production sources.

Usage:

    <managed-python> scripts/eval_system_guard.py

Acceptance is exit 0 with the final line ``eval system guard: PASS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import bisect
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = REPO_ROOT / "crates" / "chelis-compiler-api" / "src" / "runtime"

# Exempt the default adapter from the recognized source patterns.
ALLOWED_ADAPTER_FILE = "system_adapter.rs"


class SourceGuardError(RuntimeError):
    """A source file could not be read or scanned."""


@dataclass(frozen=True)
class Hit:
    """One direct-access occurrence found by `classify_source`."""

    line_number: int
    reason: str


@dataclass(frozen=True)
class Violation:
    path: Path
    hit: Hit
    line_text: str

    def render(self) -> str:
        return f"{self.path}:{self.hit.line_number}: {self.hit.reason}\n    {self.line_text.strip()}"


# ── Comment/string stripping ────────────────────────────────────────────


def strip_comments_and_strings(source: str) -> str:
    """Mask comments and Rust string/character literals, preserving line numbers.

    Raw and byte-raw literals may contain unescaped quotes, comments and host
    API names. This is a lexical prefilter, not a Rust parser or a security
    boundary; the evaluator's runtime policy enforces adapter permissions.
    """
    out: list[str] = []
    n = len(source)

    def raw_literal_end(start: int) -> int | None:
        if start and (source[start - 1].isalnum() or source[start - 1] == "_"):
            return None
        prefix = 1 if source[start] == "r" else 2
        if prefix == 2 and (
            source[start] not in "bc" or source[start + 1 : start + 2] != "r"
        ):
            return None
        hashes = start + prefix
        while hashes < n and source[hashes] == "#":
            hashes += 1
        if hashes == n or source[hashes] != '"':
            return None  # An identifier such as r#type, not a raw literal.
        terminator = '"' + source[start + prefix : hashes]
        close = source.find(terminator, hashes + 1)
        if close == -1:
            raise SourceGuardError("unterminated Rust raw string literal")
        return close + len(terminator)

    i = 0
    while i < n:
        if source[i] in "rbc":
            raw_end = raw_literal_end(i)
            if raw_end is not None:
                out.append(
                    "".join(
                        "\n" if char == "\n" else " " for char in source[i:raw_end]
                    )
                )
                i = raw_end
                continue
        two = source[i : i + 2]
        if two == "//":
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if two == "/*":
            out.append("  ")
            i += 2
            depth = 1
            while i < n and depth > 0:
                if source[i : i + 2] == "/*":
                    out.append("  ")
                    i += 2
                    depth += 1
                elif source[i : i + 2] == "*/":
                    out.append("  ")
                    i += 2
                    depth -= 1
                else:
                    out.append("\n" if source[i] == "\n" else " ")
                    i += 1
            continue
        if source[i] == '"':
            out.append(" ")
            i += 1
            while i < n and source[i] != '"':
                if source[i] == "\\" and i + 1 < n:
                    out.append("  ")
                    i += 2
                    continue
                out.append("\n" if source[i] == "\n" else " ")
                i += 1
            if i < n:
                out.append(" ")
                i += 1
            continue
        if source[i] == "'" and i + 1 < n:
            # A char literal (`'\n'`, `'a'`, `'\u{7f}'`) closes with a
            # second `'` within a short window; a lifetime (`'a`) does
            # not. Only consume as a literal when a close quote appears
            # nearby on the same line -- otherwise leave `'` untouched so
            # a lifetime token is not corrupted.
            close = source.find("'", i + 1, i + 12)
            if close != -1 and "\n" not in source[i:close]:
                out.append(" " * (close - i + 1))
                i = close + 1
                continue
        out.append(source[i])
        i += 1
    return "".join(out)


_CFG_TEST_INLINE_MODULE = re.compile(
    r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*"
    r"(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*\{"
)


def blank_inline_test_modules(stripped: str) -> str:
    """Exclude balanced cfg(test) modules while retaining source line offsets.

    Strings and comments have already been stripped, so only Rust braces
    remain. Raise for a malformed module instead of skipping later code.
    """
    result = list(stripped)
    for match in _CFG_TEST_INLINE_MODULE.finditer(stripped):
        start = match.end() - 1
        depth = 0
        for end in range(start, len(stripped)):
            if stripped[end] == "{":
                depth += 1
            elif stripped[end] == "}":
                depth -= 1
                if depth == 0:
                    for index in range(match.start(), end + 1):
                        if result[index] != "\n":
                            result[index] = " "
                    break
        else:
            raise SourceGuardError("unbalanced inline #[cfg(test)] module")
    return "".join(result)


# ── Direct-access patterns ──────────────────────────────────────────────

_PATH_EFFECT_METHOD = (
    r"(?:canonicalize|exists|try_exists|is_dir|is_file|is_symlink|metadata|"
    r"symlink_metadata|read_dir|read_link)"
)
_DIRECT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bstd\s*::\s*fs\s*::"), "direct `std::fs::` access"),
    (
        re.compile(r"\bstd\s*::\s*os\s*::\s*(?:unix|windows)\s*::\s*fs\s*::"),
        "direct platform filesystem access",
    ),
    (
        re.compile(r"\bstd\s*::\s*process\s*::"),
        "direct `std::process` access",
    ),
    (
        re.compile(
            rf"\bstd\s*::\s*path\s*::\s*(?:Path|PathBuf)\s*::\s*{_PATH_EFFECT_METHOD}\b"
        ),
        "direct `std::path` filesystem call",
    ),
    (
        re.compile(rf"\b(?:Path|PathBuf)\s*::\s*{_PATH_EFFECT_METHOD}\s*\("),
        "qualified `Path` filesystem call",
    ),
)

# `use` statements that bind a local name to a guarded item. A later
# reference through that bound name (`fs::read`, `Command::new`, a
# directly-imported function called bare, an aliased `Path::exists`)
# would not contain the literal `std::...` text the patterns above match,
# so it needs explicit alias tracking. These match across the WHOLE
# stripped text (not per line, see `classify_source`): rustfmt never
# splits a `use` statement's path across lines in practice, but nothing
# stops a hand-written bypass from inserting a newline (or any other
# `\s`) between path segments, and `\s` already matches `\n` in every
# pattern below -- restricting the search to one line at a time would
# silently defeat that on both the `use`-statement patterns and the
# direct-access patterns, so every regex here runs over the full text.
_USE_FS_MODULE = re.compile(r"\buse\s+std\s*::\s*fs\s*(?:as\s+(\w+))?\s*;")
_USE_FS_ITEMS = re.compile(r"\buse\s+std\s*::\s*fs\s*::\s*\{([^}]*)\}\s*;")
_USE_FS_SINGLE_ITEM = re.compile(
    r"\buse\s+std\s*::\s*fs\s*::\s*(\w+)\s*(?:as\s+(\w+))?\s*;"
)
_USE_OS_FS_MODULE = re.compile(
    r"\buse\s+std\s*::\s*os\s*::\s*(?:unix|windows)\s*::\s*fs\s*(?:as\s+(\w+))?\s*;"
)
_USE_OS_FS_ITEMS = re.compile(
    r"\buse\s+std\s*::\s*os\s*::\s*(?:unix|windows)\s*::\s*fs\s*::\s*\{([^}]*)\}\s*;"
)
_USE_OS_FS_SINGLE_ITEM = re.compile(
    r"\buse\s+std\s*::\s*os\s*::\s*(?:unix|windows)\s*::\s*fs\s*::\s*(\w+)\s*(?:as\s+(\w+))?\s*;"
)
_USE_STD_ALIAS = re.compile(r"\buse\s+std\s+as\s+(\w+)\s*;")
_USE_PROCESS_MODULE = re.compile(r"\buse\s+std\s*::\s*process\s*(?:as\s+(\w+))?\s*;")
_USE_COMMAND = re.compile(
    r"\buse\s+std\s*::\s*process\s*::\s*Command\s*(?:as\s+(\w+))?\s*;"
)
_USE_PATH = re.compile(
    r"\buse\s+std\s*::\s*path\s*::\s*(Path|PathBuf)\s*(?:as\s+(\w+))?\s*;"
)
_USE_GUARDED_MODULE = re.compile(r"\buse\s+std\s*::\s*(fs|process)\b")
_USE_OS_FS_IMPORT = re.compile(
    r"\buse\s+std\s*::\s*os\s*::\s*(?:unix|windows)\s*::\s*fs\b"
)
_USE_BRACED_PATH = re.compile(
    r"\buse\s+std\s*::\s*path\s*::\s*\{[^}]*\bPath(?:Buf)?\b"
)
_USE_STD_TREE_START = re.compile(r"\buse\s+std\s*::\s*\{")
_STD_TREE_GUARDED_ROOTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|,)\s*fs\b"), "nested `std` filesystem import"),
    (re.compile(r"(?:^|,)\s*process\b"), "nested `std` process import"),
    (
        re.compile(r"(?:^|,)\s*path\s*::\s*(?:\{\s*)?Path(?:Buf)?\b"),
        "nested `std` path import",
    ),
    (
        re.compile(r"(?:^|,)\s*os\s*::\s*(?:unix|windows)\s*::\s*fs\b"),
        "nested `std` platform filesystem import",
    ),
)


def _iter_std_use_trees(stripped: str) -> list[tuple[int, str]]:
    """Return balanced `use std::{...}` bodies with their source offsets."""
    trees: list[tuple[int, str]] = []
    for match in _USE_STD_TREE_START.finditer(stripped):
        open_brace = match.end() - 1
        depth = 0
        for offset in range(open_brace, len(stripped)):
            token = stripped[offset]
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
                if depth == 0:
                    body_start = open_brace + 1
                    trees.append((body_start, stripped[body_start:offset]))
                    break
    return trees


def _collect_bound_aliases(
    stripped: str,
) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Return (fs_module_names, fs_function_names, process_module_names,
    command_names, path_names)."""
    fs_module_names: set[str] = set()
    fs_function_names: set[str] = set()
    process_module_names: set[str] = set()
    command_names: set[str] = set()
    path_names: set[str] = set()

    for match in _USE_FS_MODULE.finditer(stripped):
        fs_module_names.add(match.group(1) or "fs")
    for match in _USE_OS_FS_MODULE.finditer(stripped):
        fs_module_names.add(match.group(1) or "fs")
    for match in _USE_FS_ITEMS.finditer(stripped):
        for item in match.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            parts = [part.strip() for part in item.split(" as ")]
            fs_function_names.add(parts[-1])
    for match in _USE_OS_FS_ITEMS.finditer(stripped):
        for item in match.group(1).split(","):
            item = item.strip()
            if item:
                fs_function_names.add(item.split(" as ")[-1].strip())
    for match in _USE_FS_SINGLE_ITEM.finditer(stripped):
        fs_function_names.add(match.group(2) or match.group(1))
    for match in _USE_OS_FS_SINGLE_ITEM.finditer(stripped):
        fs_function_names.add(match.group(2) or match.group(1))
    for match in _USE_PROCESS_MODULE.finditer(stripped):
        process_module_names.add(match.group(1) or "process")
    for match in _USE_COMMAND.finditer(stripped):
        command_names.add(match.group(1) or "Command")
    for match in _USE_PATH.finditer(stripped):
        path_names.add(match.group(2) or match.group(1))

    return (
        fs_module_names,
        fs_function_names,
        process_module_names,
        command_names,
        path_names,
    )


def _line_starts(text: str) -> list[int]:
    """Return the character offset each line starts at (1-indexed lines)."""
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def _line_number_for(line_starts: list[int], offset: int) -> int:
    return bisect.bisect_right(line_starts, offset)


def classify_source(text: str) -> list[Hit]:
    """Return every direct-access occurrence in `text` as a `Hit`.

    Pure function over raw Rust source (comments and string/char literals
    are stripped internally), independent of the filesystem -- this is
    what `scripts/test_eval_system_guard.py` exercises directly against
    positive and negative fixtures.

    Every pattern is matched against the FULL stripped text (never one
    line at a time): matching line-by-line would let a bypass evade
    detection just by inserting a newline between path segments (Rust
    does not care where whitespace falls inside a `::`-separated path),
    since `\\s` in every pattern below already matches `\\n`.
    """
    stripped = blank_inline_test_modules(strip_comments_and_strings(text))
    # Keep source line numbers while canonicalizing bound `std` roots for the
    # existing qualified and imported-item checks. An alias with no host
    # access is harmless; only its later guarded paths produce hits.
    for alias in set(_USE_STD_ALIAS.findall(stripped)) - {"std"}:
        stripped = re.sub(rf"\b{re.escape(alias)}\s*::", "std::", stripped)
    line_starts = _line_starts(stripped)
    (
        fs_module_names,
        fs_function_names,
        process_module_names,
        command_names,
        path_names,
    ) = _collect_bound_aliases(stripped)

    hits: list[Hit] = []

    def add(offset: int, reason: str) -> None:
        hits.append(Hit(_line_number_for(line_starts, offset), reason))

    for pattern, reason in _DIRECT_PATTERNS:
        for match in pattern.finditer(stripped):
            add(match.start(), reason)
    for match in _USE_GUARDED_MODULE.finditer(stripped):
        add(match.start(), f"direct `std::{match.group(1)}` import")
    for match in _USE_BRACED_PATH.finditer(stripped):
        add(match.start(), "braced `std::path::Path` import")
    for match in _USE_OS_FS_IMPORT.finditer(stripped):
        add(match.start(), "direct platform filesystem import")
    for name in fs_module_names:
        for match in re.finditer(rf"(?<!\w){re.escape(name)}\s*::", stripped):
            add(match.start(), f"imported filesystem-module alias `{name}::` access")
    for name in fs_function_names:
        for match in re.finditer(rf"(?<![.\w:]){re.escape(name)}\s*\(", stripped):
            add(
                match.start(),
                f"directly imported filesystem function `{name}(...)` call",
            )
    for name in process_module_names:
        # A bare `use std::process;` (or aliased) import: any later use of
        # `<name>::Command` (however it is spelled downstream, `::new(`
        # included) constructs a process exactly the way the fully
        # qualified `std::process::Command` pattern already catches.
        for match in re.finditer(rf"(?<!\w){re.escape(name)}\s*::\s*Command\b", stripped):
            add(
                match.start(),
                f"imported process-module alias `{name}::Command` construction",
            )
    for name in command_names:
        for match in re.finditer(
            rf"(?<!\w){re.escape(name)}\s*::\s*new\s*\(", stripped
        ):
            add(
                match.start(),
                f"imported process alias `{name}::new(...)` construction",
            )
    for name in path_names:
        for match in re.finditer(
            rf"(?<!\w){re.escape(name)}\s*::\s*{_PATH_EFFECT_METHOD}\s*\(", stripped
        ):
            add(match.start(), f"imported path alias `{name}` filesystem call")
    # A method named `exists` is not evidence of a path effect by itself:
    # `Inventory.exists()` is legitimate. Require an explicit Path/PathBuf
    # constructor or a receiver bound to one of those types.
    path_types = "|".join(map(re.escape, sorted({"Path", "PathBuf"} | path_names)))
    path_type = rf"(?:std\s*::\s*path\s*::\s*)?(?:{path_types})"
    for match in re.finditer(
        rf"\b{path_type}\s*::\s*(?:new|from)\s*\([^)]*\)\s*\.\s*{_PATH_EFFECT_METHOD}\s*\(",
        stripped,
    ):
        add(match.start(), "method-form path filesystem call")
    receivers = {
        match.group(1)
        for match in re.finditer(
            rf"\b(\w+)\s*:\s*&?\s*(?:mut\s+)?{path_type}\b", stripped
        )
    }
    receivers.update(
        match.group(1)
        for match in re.finditer(
            rf"\blet\s+(?:mut\s+)?(\w+)\s*=\s*{path_type}\s*::\s*(?:new|from)\s*\(",
            stripped,
        )
    )
    for receiver in receivers:
        for match in re.finditer(
            rf"(?<![\w.]){re.escape(receiver)}\s*\.\s*{_PATH_EFFECT_METHOD}\s*\(",
            stripped,
        ):
            add(match.start(), "method-form path filesystem call")
    for body_start, body in _iter_std_use_trees(stripped):
        for pattern, reason in _STD_TREE_GUARDED_ROOTS:
            for match in pattern.finditer(body):
                add(body_start + match.start(), reason)
    return hits


# ── Module classification (production vs. `#[cfg(test)]`) ──────────────

# `mod x;` optionally preceded by a visibility modifier (`pub`,
# `pub(crate)`, `pub(super)`, ...).
_MOD_DECL = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;\s*$")
_CFG_TEST_ATTR = re.compile(r"^\s*#\[cfg\(test\)\]\s*$")
# Any other attribute line (`#[allow(dead_code)]`, `#[rustfmt::skip]`, ...).
# Rust permits stacking attributes above one item; `#[cfg(test)]` should
# still apply to the `mod` line even when another attribute sits between
# them.
_OTHER_ATTR = re.compile(r"^\s*#\[.*\]\s*$")
# The one-line form: `#[cfg(test)] mod x;`.
_CFG_TEST_MOD_INLINE = re.compile(
    r"^\s*#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;\s*$"
)


def parse_test_only_modules(mod_rs_text: str) -> frozenset[str]:
    """Return the `mod <name>;` names gated by `#[cfg(test)]`.

    These files never ship in a release binary, so a default-adapter
    parity test file legitimately touching real `std::fs` to build its
    fixtures is not a production bypass. Handles the inline form
    (`#[cfg(test)] mod x;`), the two-line form, a `pub`/`pub(crate)`
    visibility modifier on the `mod` line, and an unrelated attribute
    stacked between `#[cfg(test)]` and the `mod` line. An unrecognized
    form stays in the set of files scanned as production; this limited
    module classifier does not establish which code rustc compiles.
    """
    test_only: set[str] = set()
    pending_cfg_test = False
    for line in mod_rs_text.splitlines():
        inline = _CFG_TEST_MOD_INLINE.match(line)
        if inline:
            test_only.add(inline.group(1))
            pending_cfg_test = False
            continue
        if _CFG_TEST_ATTR.match(line):
            pending_cfg_test = True
            continue
        match = _MOD_DECL.match(line)
        if match:
            if pending_cfg_test:
                test_only.add(match.group(1))
            pending_cfg_test = False
            continue
        if pending_cfg_test and _OTHER_ATTR.match(line):
            # A second attribute stacked between `#[cfg(test)]` and the
            # `mod` line does not cancel the pending gate.
            continue
        if line.strip():
            pending_cfg_test = False
    return frozenset(test_only)


def iter_scanned_files(runtime_dir: Path) -> list[Path]:
    if not runtime_dir.is_dir():
        raise SourceGuardError(f"runtime directory not found: {runtime_dir}")
    mod_rs = runtime_dir / "mod.rs"
    try:
        mod_rs_text = mod_rs.read_text(encoding="utf-8")
    except OSError as error:
        raise SourceGuardError(f"cannot read {mod_rs}: {error}") from error
    test_only_modules = parse_test_only_modules(mod_rs_text)

    scanned = []
    for path in sorted(runtime_dir.rglob("*.rs")):
        relative = path.relative_to(runtime_dir)
        if relative == Path(ALLOWED_ADAPTER_FILE):
            continue
        if relative.parent == Path(".") and relative.stem in test_only_modules:
            continue
        scanned.append(path)
    return scanned


def scan_directory(runtime_dir: Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in iter_scanned_files(runtime_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            # An unreadable source file is an error, not an omitted scan.
            raise SourceGuardError(f"cannot read {path}: {error}") from error
        lines = text.splitlines()
        for hit in classify_source(text):
            line_text = lines[hit.line_number - 1] if 0 < hit.line_number <= len(lines) else ""
            violations.append(Violation(path, hit, line_text))
    return violations


def main() -> int:
    try:
        violations = scan_directory(RUNTIME_DIR)
    except SourceGuardError as error:
        print(f"eval system guard: FAIL: {error}", file=sys.stderr)
        return 1
    if violations:
        print(
            "eval system guard: FAIL: recognized direct system access outside "
            f"`{ALLOWED_ADAPTER_FILE}`",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation.render()}", file=sys.stderr)
        return 1
    print("eval system guard: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
