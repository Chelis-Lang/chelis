#!/usr/bin/env python3
"""Fail on runtime-archive lookups outside the runtime bundle (chelis#1354).

`chelis-runtime-bundle` owns the runtime archive: a build embeds the archive it
linked, `stage` writes those bytes, and `chelis runtime export` wraps `stage`
(spec/08-backends.md §2.1). Code that instead finds an archive can link bytes
its consumer was not built with.

A lookup has to name what it looks for, so the guard anchors on the runtime's
names rather than on the syntax around them. It reads every tracked or
untracked, not ignored, Rust, Python, Nix, shell, TOML and YAML file outside the
two bundle crates and matches:

- any mention of `CHELIS_RUNTIME_DIR` (reviewable) or `CHELIS_RUNTIME_LIB` (never
  allowed), whatever reads, sets or names it;
- the runtime's library or Cargo target name as a string literal
  (`"chelis_runtime"`, `'libchelis_runtime'`), which a Cargo artifact read, a
  `#[link]` attribute or a name filter must contain;
- a library search for it: `-lchelis_runtime`, `-l chelis_runtime`, an `"-l"`
  argument followed by the name, `rustc-link-lib` or `#[link(name = ...)]`;
- `libchelis_runtime` other than the exact staged file name
  `libchelis_runtime.a`: a prefix, a hashed name or a glob stem;
- the exact file name inside a glob or `find -name` scan, or joined onto a
  build-tree directory (`target`, `deps`, `debug`, `release`, `profile`, a
  `target*()` helper or `CARGO_TARGET_DIR`), also across a line break;
- the exact file name on a line that probes for it (an existence or file check,
  a shell test, a directory walk or listing, the executable's location or a
  modification time), or bound to a variable that one of the next three lines
  checks for existence;
- a Cargo package id of the runtime (`"chelis-runtime 0.1.0 (...)"`,
  `#chelis-runtime@...`).

Linking the staged archive by path, `out.join("libchelis_runtime.a")`, is none
of these. Each reviewed row pins the exact lines one pattern matches in one file,
so a new match fails, an edited or removed match fails, and the change that
removes a lookup also removes its row. A `lookup` row names the issue that
removes it; a `not-lookup` row says why the text is not one.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
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
# Every alternative of every pattern contains one of these; other files are skipped.
TOKENS = ("chelis_runtime", "chelis-runtime", "CHELIS_RUNTIME_")
DISPOSITIONS = frozenset({"lookup", "not-lookup"})
ISSUE = re.compile(r"chelis#[1-9][0-9]*")
BUILD_TREE = r"(?:target|deps|debug|release)"


@dataclass(frozen=True)
class Pattern:
    name: str
    alternatives: tuple[str, ...]
    # Lines the pattern must match. Each alternative has a sample no other
    # alternative matches, so deleting one fails the guard's tests.
    samples: tuple[str, ...]
    # False: no reviewed row may allow a match.
    reviewable: bool = True
    regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regex", re.compile("|".join(self.alternatives)))


PATTERNS = (
    Pattern(
        "runtime-variable",
        (r"\bCHELIS_RUNTIME_DIR\b",),
        ('runtime = env.get("CHELIS_RUNTIME_DIR")',),
    ),
    Pattern(
        "runtime-lib-variable",
        (r"\bCHELIS_RUNTIME_LIB\b",),
        ('CHELIS_RUNTIME_LIB="$PWD/target/debug/libchelis_runtime.a" cargo nextest run',),
        reviewable=False,
    ),
    Pattern(
        "library-name",
        (r"[\"'](?:lib)?chelis_runtime[\"']",),
        ("if row.get('target', {}).get('name') != 'chelis_runtime':",),
    ),
    Pattern(
        "linker-search",
        (
            r"-l\s*:?(?:lib)?chelis_runtime\b",
            r"[\"']-l[\"']\s*(?:,|\)\s*\.arg\()\s*[\"']:?(?:lib)?chelis_runtime\b",
            r"link-lib=[^\s\"']*chelis_runtime",
            r"#\[link\(\s*name\s*=\s*r?\"(?:lib)?chelis_runtime\"",
        ),
        (
            'cmd.args(["-L.", "-lchelis_runtime"]);',
            'cmd.arg("-L").arg(dir).arg("-l").arg("chelis_runtime");',
            'println!("cargo:rustc-link-lib=static=chelis_runtime");',
            '#[link(name = "chelis_runtime", kind = "static")]',
        ),
    ),
    Pattern(
        "archive-prefix",
        (r"libchelis_runtime(?!\.a(?![\w*?\[{]))",),
        ("const LIB_PREFIX: &str = \"libchelis_runtime\";", "ls target/debug/libchelis_runtime.a*"),
    ),
    Pattern(
        "archive-scan",
        (
            r"\b(?:r?glob|iglob|fnmatch)\s*\([^)]*libchelis_runtime",
            r"\bfind\b[^\n;|]*-i?name\s+[\"']?[^\s\"';|]*libchelis_runtime",
        ),
        (
            'archive = next(Path(target).glob("**/libchelis_runtime.a"))',
            'glob::glob(\n    &format!("{}/**/libchelis_runtime.a", target.display()),\n)',
            "find target -name libchelis_runtime.a -newer Cargo.lock",
        ),
    ),
    Pattern(
        "build-tree-archive",
        (
            rf"\b{BUILD_TREE}/[^\s\"']*libchelis_runtime",
            rf"[\"']{BUILD_TREE}[\"'][^\n]*?[\"']libchelis_runtime",
            rf"[\"']{BUILD_TREE}[\"']\s*\)\s*\.join\(\s*[\"']libchelis_runtime",
            r"\b(?:target|deps|debug|release|profile)\w*(?:\(\))?\s*(?:\.join\(\s*|/\s*)"
            r"[\"']libchelis_runtime",
            r"CARGO_TARGET_DIR[^\n]*libchelis_runtime",
        ),
        (
            "cp target/release/libchelis_runtime.a dist/",
            'archive = os.path.join("target", "debug", "libchelis_runtime.a")',
            'let archive = root\n    .join("debug")\n    .join("libchelis_runtime.a");',
            'let canonical = target_debug_dir().join("libchelis_runtime.a");',
            'cc main.c "$CARGO_TARGET_DIR/libchelis_runtime.a"',
        ),
    ),
    Pattern(
        "archive-probe",
        (
            # The name and a probe on one line: an existence or file check,
            *(
                rf"(?m:^(?=[^\n]*libchelis_runtime\.a)(?=[^\n]*(?:{probe}))[^\n]*)"
                for probe in (
                    r"\.(?:exists|is_file|try_exists)\(\)|\b(?:exists|isfile)\(|\bmetadata\(",
                    # a shell test,
                    r"\[\[?\s+-[ef]\s|\btest\s+-[ef]\s",
                    # a directory walk or listing,
                    r"WalkDir|os\.walk|read_dir|listdir|\bin\s+files\b|file_name\(\)\s*==",
                    # the executable's location,
                    r"current_exe",
                    # or a modification time.
                    r"st_mtime|getmtime|modified\(\)",
                )
            ),
            # A variable bound to the name and probed within the next three lines.
            r"(?m:^[^\S\n]*(?:let\s+(?:mut\s+)?)?(?P<bound>\w+)[^\S\n]*(?::[^=\n]*)?=[^=\n]*"
            r"libchelis_runtime\.a[^\n]*(?:\n[^\n]*){0,3}?"
            r"(?:\b(?P=bound)\s*\.\s*(?:exists|is_file|try_exists)\("
            r"|\b(?:exists|isfile|getmtime|metadata)\(\s*&?(?P=bound)\b))",
        ),
        (
            'if dir.join("libchelis_runtime.a").exists() {',
            'if os.path.isfile(os.path.join(d, "libchelis_runtime.a")):',
            'if fs::metadata(dir.join("libchelis_runtime.a")).is_ok() {',
            '[ -f "$d/libchelis_runtime.a" ] && runtime="$d"',
            'test -f "$d/libchelis_runtime.a" && runtime="$d"',
            'WalkDir::new(&target).into_iter().find(|e| e.file_name() == "libchelis_runtime.a")',
            'if "libchelis_runtime.a" in files:',
            'let beside = std::env::current_exe()?.with_file_name("libchelis_runtime.a");',
            'newest = max((d / "libchelis_runtime.a" for d in dirs), key=lambda p: p.stat().st_mtime)',
            'newest = max((d / "libchelis_runtime.a" for d in dirs), key=os.path.getmtime)',
            'let candidate = dir.join("libchelis_runtime.a");\nif candidate.exists() {',
            'candidate = d / "libchelis_runtime.a"\nif os.path.isfile(candidate):',
        ),
    ),
    Pattern(
        "runtime-package",
        (r"[\"']chelis-runtime[ @]", r"#chelis-runtime@"),
        (
            'if message["package_id"].startswith("chelis-runtime "):',
            "id = f\"path+file://{root}/crates/chelis-runtime#chelis-runtime@{version}\"",
        ),
    ),
)
PATTERN_NAMES = {pattern.name: pattern for pattern in PATTERNS}


@dataclass(frozen=True)
class Row:
    path: str
    pattern: str
    # The stripped text of every line the pattern matches in `path`.
    lines: tuple[str, ...]
    disposition: str
    reason: str
    tracking: str | None = None


# The reviewed matches. A lookup row's tracking issue owns its removal.
REVIEWED: tuple[Row, ...] = (
    Row(
        "crates/chelis-cli/tests/cross_library_semantic_gap_hip_gpu.rs",
        "linker-search",
        lines=(
            '.arg("-lchelis_runtime")',
        ),
        disposition="lookup",
        reason=(
            "an entirely ignored HIP gate links `-L. -lchelis_runtime` in its `chelis build` "
            "output directory; linking the staged archive by path needs its manual-gate row "
            "and wired docs/manual_gates.md entry in the same change"
        ),
        tracking="chelis#1354",
    ),
    Row(
        "crates/chelis-cli/tests/issue_1314_json_bigint.rs",
        "library-name",
        lines=(
            'row["reason"] == "compiler-artifact" && row["target"]["name"] == "chelis_runtime"',
        ),
        disposition="lookup",
        reason=(
            "selects a separately built ownership-ledger archive from Cargo's "
            "compiler-artifact messages and copies it over the staged runtime, instead of "
            "building its consumer with chelis-runtime/ownership-ledger"
        ),
        tracking="chelis#1354",
    ),
    Row(
        "crates/chelis-cli/tests/std_io_pipeline.rs",
        "linker-search",
        lines=(
            'cmd.args(["-L.", "-lchelis_runtime"]);',
        ),
        disposition="lookup",
        reason=(
            "an entirely ignored manual gate links `-L. -lchelis_runtime` in its `chelis "
            "build` output directory; linking the staged archive by path needs its manual-only "
            "or manual-gate row in the same change"
        ),
        tracking="chelis#1354",
    ),
    Row(
        "crates/chelis-compiler-api/tests/ownership_support/mod.rs",
        "archive-prefix",
        lines=(
            'let archive = staged.join(format!("libchelis_runtime-{:016x}.a", hasher.finish()));',
        ),
        disposition="lookup",
        reason=(
            "stages that separately built archive under a content-addressed "
            "`libchelis_runtime-<hash>.a` name"
        ),
        tracking="chelis#1354",
    ),
    Row(
        "crates/chelis-compiler-api/tests/ownership_support/mod.rs",
        "library-name",
        lines=(
            'row["reason"] == "compiler-artifact" && row["target"]["name"] == "chelis_runtime"',
        ),
        disposition="lookup",
        reason=(
            "selects a separately built ownership-ledger archive from Cargo's "
            "compiler-artifact messages instead of building its consumer with "
            "chelis-runtime/ownership-ledger"
        ),
        tracking="chelis#1354",
    ),
    Row(
        "crates/chelis-python/tests/manual_reef_context.rs",
        "runtime-variable",
        lines=(
            "//! built `libchelis_runtime.a` discoverable via `CHELIS_RUNTIME_DIR`. No",
            "//! # bindings installed into py/.venv, runtime staticlib on CHELIS_RUNTIME_DIR",
            '//! export CHELIS_RUNTIME_DIR="$PWD/target/agents/<name>/debug"',
            '#[ignore = "manual acceptance gate (#816): needs the 0.16.1 toolchain + reef registry, uv, and CHELIS_RUNTIME_DIR (libchelis_runtime.a); builds a temp reef project (tens of seconds)"]',
        ),
        disposition="lookup",
        reason=(
            "a manual gate's prerequisites, command block and ignore reason still point the "
            "extension at a runtime directory, which it now rejects"
        ),
        tracking="chelis#2694",
    ),
    Row(
        "bindings/python/chelis/__init__.py",
        "runtime-variable",
        lines=(
            "the changed files; rebuild the extension to continue. A set ``CHELIS_RUNTIME_DIR``",
        ),
        disposition="not-lookup",
        reason=(
            "documents that `compile_and_load` rejects a set variable"
        ),
    ),
    Row(
        "bindings/python/tests/manual_reef_context.py",
        "runtime-variable",
        lines=(
            "unset CHELIS_RUNTIME_DIR                                       # the extension carries its runtime",
        ),
        disposition="not-lookup",
        reason=(
            "manual gate instructions unset the variable, which the extension rejects"
        ),
    ),
    Row(
        "crates/chelis-backend-c/src/host_emit.rs",
        "archive-prefix",
        lines=(
            "// exported by libchelis_runtime; only the declarations are private.",
        ),
        disposition="not-lookup",
        reason=(
            "a comment naming the runtime library"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "archive-prefix",
        lines=(
            'bin_dir.join("deps/libchelis_runtime-ffffffffffffffff.a"),',
        ),
        disposition="not-lookup",
        reason=(
            "plants a stale hashed archive beside the CLI that `chelis build` must not stage"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "archive-probe",
        lines=(
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
        ),
        disposition="not-lookup",
        reason="asserts that `chelis build` staged the archive in its output directory",
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "build-tree-archive",
        lines=(
            'bin_dir.join("deps/libchelis_runtime-ffffffffffffffff.a"),',
        ),
        disposition="not-lookup",
        reason=(
            "plants a stale hashed archive beside the CLI that `chelis build` must not stage"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "linker-search",
        lines=(
            '.stdout(predicate::str::contains("-lchelis_runtime").not())',
            '.stdout(predicate::str::contains("-lchelis_runtime").not())',
        ),
        disposition="not-lookup",
        reason=(
            "asserts that printed link commands carry no library search"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "runtime-variable",
        lines=(
            '.env("CHELIS_RUNTIME_DIR", &runtime_dir)',
            '.stderr(predicate::str::contains("CHELIS_RUNTIME_DIR is set"))',
            '.stderr(predicate::str::contains("Unset CHELIS_RUNTIME_DIR"));',
            '.env_remove("CHELIS_RUNTIME_DIR")',
            '.env_remove("CHELIS_RUNTIME_DIR")',
            '.env("CHELIS_RUNTIME_DIR", dir.path())',
            '.stderr(predicate::str::contains("Unset CHELIS_RUNTIME_DIR"));',
        ),
        disposition="not-lookup",
        reason=(
            "sets the variable or removes it from a test's environment, and asserts that "
            "`chelis build` and the export reject it"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_300_grad_codegen_compiles.rs",
        "archive-probe",
        lines=("runtime.is_file(),",),
        disposition="not-lookup",
        reason="asserts that `chelis build` staged the archive in its build directory",
    ),
    Row(
        "crates/chelis-cli/tests/issue_352_captured_global_c_emit.rs",
        "archive-probe",
        lines=("runtime.is_file(),",),
        disposition="not-lookup",
        reason="asserts that `chelis build` staged the archive in its build directory",
    ),
    Row(
        "crates/chelis-cli/tests/issue_735_device_fence.rs",
        "archive-probe",
        lines=(
            'assert!(output.join("libchelis_runtime.a").is_file(), "{name}");',
        ),
        disposition="not-lookup",
        reason="asserts that `chelis build` staged the archive in its output directory",
    ),
    Row(
        "crates/chelis-cli/tests/ws2b_numeric_identifier_divergence.rs",
        "archive-probe",
        lines=("runtime.is_file(),",),
        disposition="not-lookup",
        reason="asserts that `chelis build` staged the archive in its build directory",
    ),
    Row(
        "crates/chelis-compiler-api/tests/ownership_support/mod.rs",
        "build-tree-archive",
        lines=(
            "/// or not, replaces the uplifted `debug/libchelis_runtime.a` with a new file.",
        ),
        disposition="not-lookup",
        reason=(
            "a comment explaining why the harness does not link the uplifted build-tree "
            "archive"
        ),
    ),
    Row(
        "crates/chelis-python/src/lib.rs",
        "archive-probe",
        lines=(
            'dir.path().join("libchelis_runtime.a").exists(),',
        ),
        disposition="not-lookup",
        reason="asserts that `compile_and_load` staged the archive beside the shared library",
    ),
    Row(
        "crates/chelis-python/src/lib.rs",
        "runtime-variable",
        lines=(
            "// `CHELIS_RUNTIME_DIR` is refused before anything is written, even when it",
            'let prior_runtime = std::env::var_os("CHELIS_RUNTIME_DIR");',
            'std::env::set_var("CHELIS_RUNTIME_DIR", offered.path());',
            'Some(v) => std::env::set_var("CHELIS_RUNTIME_DIR", v),',
            'None => std::env::remove_var("CHELIS_RUNTIME_DIR"),',
            '"a set CHELIS_RUNTIME_DIR was honored: {}",',
            'message.contains("CHELIS_RUNTIME_DIR is set")',
            '&& message.contains("Unset CHELIS_RUNTIME_DIR"),',
        ),
        disposition="not-lookup",
        reason=(
            "a test sets the variable, restores the caller's value and asserts that "
            "`compile_and_load` rejects it"
        ),
    ),
    Row(
        "crates/chelisup/src/runtime_check.rs",
        "runtime-variable",
        lines=(
            '.env_remove("CHELIS_RUNTIME_DIR")',
        ),
        disposition="not-lookup",
        reason=(
            "removes the variable from the release export it runs, which refuses one"
        ),
    ),
    Row(
        "crates/chelisup/tests/common/mod.rs",
        "archive-prefix",
        lines=(
            'root.join("lib/libchelis_runtime.real"),',
            'std::os::unix::fs::symlink("libchelis_runtime.real", &archive).unwrap();',
        ),
        disposition="not-lookup",
        reason=(
            "the link target of the fixture's symlinked-archive refusal case"
        ),
    ),
    Row(
        "crates/chelisup/tests/common/mod.rs",
        "runtime-variable",
        lines=(
            "/// `<dir>` (and, like the real export, refuses a set `CHELIS_RUNTIME_DIR`).",
            '\\x20 if [ -n \\"${{CHELIS_RUNTIME_DIR+set}}\\" ]; then\\n\\',
            "\\x20   echo 'error: CHELIS_RUNTIME_DIR is set' >&2; exit 1\\n\\",
        ),
        disposition="not-lookup",
        reason=(
            "the fake export refuses a set variable, as the real one does"
        ),
    ),
    Row(
        "crates/chelisup/tests/install.rs",
        "archive-probe",
        lines=(
            'assert!(toolchain.join("lib/libchelis_runtime.a").is_file());',
        ),
        disposition="not-lookup",
        reason="asserts that the installed toolchain holds the release's archive",
    ),
    Row(
        "crates/chelisup/tests/install.rs",
        "runtime-variable",
        lines=(
            '("CHELIS_RUNTIME_DIR", "/elsewhere"),',
        ),
        disposition="not-lookup",
        reason=(
            "sets the variable in the caller's environment to show that `install` does not "
            "pass it to the export"
        ),
    ),
    Row(
        "scripts/capacity_census_native_execution.py",
        "runtime-variable",
        lines=(
            'for name in ("CHELIS_RUNTIME_DIR", "CHELIS_CC", "CHELIS_TEST_CC", "CHELIS_HIPCC",',
        ),
        disposition="not-lookup",
        reason=(
            "removes the variable from native execution workers' environments"
        ),
    ),
    Row(
        "scripts/compiled_value_ownership_oracle.py",
        "library-name",
        lines=(
            'elif artifact_target.get("name") == "chelis_runtime":',
        ),
        disposition="not-lookup",
        reason=(
            "requires the CLI's Cargo build to have compiled chelis-runtime with the ledger "
            "feature; the reference digest comes from the CLI's own export"
        ),
    ),
    Row(
        "scripts/compiled_value_ownership_oracle.py",
        "linker-search",
        lines=(
            'argument.startswith("-lchelis_runtime") for argument in command',
        ),
        disposition="not-lookup",
        reason=(
            "rejects a compile command that searches for the runtime"
        ),
    ),
    Row(
        "scripts/compiled_value_ownership_oracle.py",
        "runtime-variable",
        lines=(
            'RUNTIME_DIR_VARIABLE = "CHELIS_RUNTIME_DIR"',
        ),
        disposition="not-lookup",
        reason=(
            "names the variable only to refuse an inherited value"
        ),
    ),
    Row(
        "scripts/installed_artifact_canary.py",
        "library-name",
        lines=(
            '"chelis_runtime", "chelis_runtime_views", "chelis_runtime_dtype",',
        ),
        disposition="not-lookup",
        reason=(
            "the stem of the public header chelis_runtime.h, which a release must ship"
        ),
    ),
    Row(
        "scripts/runtime_representation_phase1.py",
        "runtime-variable",
        lines=(
            "'runtime-directory-rejected': ('CHELIS_RUNTIME_DIR', bad),",
            "execution(xml, selected, failure_text=f'CHELIS_RUNTIME_DIR is set ({bad})')",
        ),
        disposition="not-lookup",
        reason=(
            "a negative control sets the variable and expects `chelis build` to refuse it"
        ),
    ),
    Row(
        "scripts/test_capacity_census_native_execution.py",
        "runtime-variable",
        lines=(
            '"CHELIS_RUNTIME_DIR": "/foreign/runtime", "CHELIS_CC": "/foreign/compiler",',
            'for name in ("CHELIS_RUNTIME_DIR", "CHELIS_CC", "CHELIS_TEST_CC", "CHELIS_HIPCC",',
        ),
        disposition="not-lookup",
        reason=(
            "a foreign value the census must remove, and the names it must clear"
        ),
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "archive-prefix",
        lines=(
            'archive = target / "debug" / "deps" / "libchelis_runtime-0123abcd.a"',
        ),
        disposition="not-lookup",
        reason=(
            "a decoy Cargo deps archive whose bytes the oracle must not take as the runtime"
        ),
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "build-tree-archive",
        lines=(
            'archive = target / "debug" / "deps" / "libchelis_runtime-0123abcd.a"',
        ),
        disposition="not-lookup",
        reason=(
            "a decoy Cargo deps archive whose bytes the oracle must not take as the runtime"
        ),
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "library-name",
        lines=(
            '"target": {"name": "chelis_runtime", "kind": ["staticlib", "rlib"]},',
        ),
        disposition="not-lookup",
        reason=(
            "a fixture Cargo compiler-artifact message for the oracle's ledger feature check"
        ),
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "linker-search",
        lines=(
            'searched = f"cc -O2 {output}/main.c -L{output} -lchelis_runtime -lm -o {output}/main"',
        ),
        disposition="not-lookup",
        reason=(
            "a searched compile command the oracle must reject"
        ),
    ),
    Row(
        "scripts/test_compiled_value_ownership_oracle.py",
        "runtime-variable",
        lines=(
            'if "CHELIS_RUNTIME_DIR" not in environment:',
            'os.environ.pop("CHELIS_RUNTIME_DIR", None)',
            'with self.assertRaisesRegex(oracle.OracleFailure, "Unset CHELIS_RUNTIME_DIR"):',
            'self.context(CHELIS_RUNTIME_DIR="/foreign/runtime")',
            'self.assertNotIn("CHELIS_RUNTIME_DIR", environments[0])',
        ),
        disposition="not-lookup",
        reason=(
            "tests that the oracle refuses an inherited variable and passes none to a run"
        ),
    ),
    Row(
        "scripts/test_nix_flake_contract.py",
        "build-tree-archive",
        lines=(
            'self.assertNotIn("target/release/libchelis_runtime.a", release)',
        ),
        disposition="not-lookup",
        reason=(
            "asserts that the release no longer copies the build-tree archive"
        ),
    ),
    Row(
        "scripts/test_runtime_representation_phase1.py",
        "runtime-variable",
        lines=(
            "watched = ('CHELIS_RUNTIME_DIR', 'CHELIS_TEST_CC')",
            "oracle.os.environ.pop('CHELIS_RUNTIME_DIR', None)",
            "self.assertNotIn('CHELIS_RUNTIME_DIR', oracle.os.environ)",
            "return (f'stderr=\"error: CHELIS_RUNTIME_DIR is set ({bad}), but chelis stages the runtime '",
            "'built into it and never takes one from a directory. Unset CHELIS_RUNTIME_DIR\"')",
            "'runtime-directory-rejected': {'CHELIS_RUNTIME_DIR': str(empty.parent), 'CHELIS_TEST_CC': None},",
            "'missing-c-compiler': {'CHELIS_RUNTIME_DIR': None, 'CHELIS_TEST_CC': str(evidence / 'missing-compiler')},",
            "{'CHELIS_RUNTIME_DIR': None, 'CHELIS_TEST_CC': None})",
        ),
        disposition="not-lookup",
        reason=(
            "the Phase 1 oracle tests' rejection control and environment hygiene"
        ),
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


def scan(root: Path, names: Iterable[str]) -> tuple[dict[tuple[str, str], list[tuple[int, str]]], list[str]]:
    """Map each (file, pattern) with a match to its (line, text) locations.

    Also return the files that could not be read.
    """
    found: dict[tuple[str, str], list[tuple[int, str]]] = {}
    unreadable: list[str] = []
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
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            unreadable.append(f"{name}: cannot be read as UTF-8, so it cannot be checked ({error})")
            continue
        if not any(token in text for token in TOKENS):
            continue
        lines = text.splitlines()
        for pattern in PATTERNS:
            for match in pattern.regex.finditer(text):
                # The line holding the matched name, even when a match spans a line break.
                line = text.count("\n", 0, match.end() - 1) + 1
                found.setdefault((name, pattern.name), []).append((line, lines[line - 1].strip()))
    return found, unreadable


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
        if not row.lines:
            errors.append(f"{label} must pin at least one line")
        if any(not line or line != line.strip() for line in row.lines):
            errors.append(f"{label} pins a line that is empty or not stripped")
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
    found, unreadable = scan(root, names)
    errors.extend(unreadable)
    allowed = {(row.path, row.pattern): row for row in rows}
    for key in sorted(set(found) | set(allowed)):
        name, pattern = key
        locations = found.get(key, [])
        row = allowed.get(key)
        observed = Counter(text for _, text in locations)
        reviewed = Counter(row.lines if row is not None else ())
        unreviewed = observed - reviewed
        if unreviewed:
            listed = [
                f"{name}:{line}: {text}" for line, text in locations if text in unreviewed
            ]
            errors.append(
                f"{pattern}: {sum(unreviewed.values())} unreviewed match(es) in {name}. "
                "Link the archive that `chelis build`, `chelis_runtime_bundle::stage` or "
                "`chelis runtime export` wrote, by its exact path; or, if the text is not "
                "a lookup, add or update a reviewed not-lookup row.\n  " + "\n  ".join(listed)
            )
        missing = reviewed - observed
        if missing:
            errors.append(
                f"stale row {name} [{pattern}]: {sum(missing.values())} reviewed line(s) no "
                "longer match. Update or delete the row in the change that altered them.\n  "
                + "\n  ".join(sorted(missing.elements()))
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
