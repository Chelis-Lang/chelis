#!/usr/bin/env python3
"""Make every line that names the runtime outside its bundle a reviewed line (chelis#1354).

`chelis-runtime-bundle` owns the runtime archive: a build embeds the archive it
linked, `stage` writes those bytes, and `chelis runtime export` wraps `stage`
(spec/08-backends.md §2.1). Code that instead finds an archive can link bytes
its consumer was not built with.

A lookup has to name what it looks for, and telling a lookup from a staged use
needs a reader, so the guard does not classify lines: every line that names
the runtime needs a reviewed row. It reads every tracked or untracked, not
ignored, Rust, Python, Nix, shell, TOML and YAML file outside the two bundle
crates, one line at a time, and matches:

- `libchelis_runtime` in any form: the staged file name, a prefix, a hashed
  name or a glob;
- a runtime variable, `CHELIS_RUNTIME_` followed by a name (C header guards
  ending in `_H` are not variables); `CHELIS_RUNTIME_LIB` admits no row;
- the library or Cargo target name as a string literal, `"chelis_runtime"` or
  with a link kind, `"static:+whole-archive=chelis_runtime"`, which a Cargo
  artifact read, a `#[link]` attribute or a split `-l` argument contains;
- a library search: `-lchelis_runtime`, `-l chelis_runtime`, `-l
  static=chelis_runtime` with or without link modifiers,
  `rustc-link-lib=...chelis_runtime` or `--library chelis_runtime`;
- the bundle's exported names for the archive and the variable,
  `ARCHIVE_FILE_NAME` and `RUNTIME_DIR_VARIABLE`;
- a Cargo package id of the runtime (`"chelis-runtime 0.1.0 (...)"`,
  `.../crates/chelis-runtime#0.18.11`, `#chelis-runtime@...`).

Each reviewed row pins the exact text of the lines one pattern matches in one
file. A new line fails, an edited or removed line fails, and the change that
removes a lookup also removes its row. A `lookup` row names the issue that
removes it; a `not-lookup` row says why the line is not one.

The guard follows no values. A lookup that reaches the runtime only through a
reviewed line, such as a constant naming the archive or a directory computed on
another line, is visible only as that reviewed line. A name assembled from
fragments or held in an unquoted shell variable, a Cargo read keyed only on the
package name `"chelis-runtime"` (the literal every `-p chelis-runtime` argument
shares), the `staticlib` kind or the manifest path, and files outside the
scanned suffixes are not matched either.
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
# Every alternative of every pattern contains one of these; other lines are skipped.
TOKENS = (
    "chelis_runtime",
    "chelis-runtime",
    "CHELIS_RUNTIME_",
    "ARCHIVE_FILE_NAME",
    "RUNTIME_DIR_VARIABLE",
)
DISPOSITIONS = frozenset({"lookup", "not-lookup"})
ISSUE = re.compile(r"chelis#[1-9][0-9]*")


@dataclass(frozen=True)
class Pattern:
    name: str
    alternatives: tuple[str, ...]
    # Lines the pattern must match. Each alternative has a sample no other
    # alternative matches, so deleting one fails the guard's tests. A pattern
    # applies to one line at a time.
    samples: tuple[str, ...]
    # False: no reviewed row may allow a match.
    reviewable: bool = True
    regex: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regex", re.compile("|".join(self.alternatives)))


PATTERNS = (
    Pattern(
        "archive-name",
        (r"libchelis_runtime",),
        ('let runtime = build_dir.join("libchelis_runtime.a");',),
    ),
    Pattern(
        "runtime-variable",
        # Any runtime variable but the one below; C header guards are not variables.
        (r"\bCHELIS_RUNTIME_(?!LIB\b)(?!(?:[A-Z0-9_]*_)?H\b)[A-Z0-9_]+",),
        ('runtime = env.get("CHELIS_RUNTIME_DIR")',),
    ),
    Pattern(
        "runtime-lib-variable",
        (r"\bCHELIS_RUNTIME_LIB\b",),
        ('CHELIS_RUNTIME_LIB="$PWD/target/debug/runtime.a" cargo nextest run',),
        reviewable=False,
    ),
    Pattern(
        "library-name",
        (r"[\"'](?:[a-z]+(?::[-+a-z,]+)?=)?chelis_runtime[\"']",),
        (
            "if row.get('target', {}).get('name') != 'chelis_runtime':",
            'rustflags = ["-l", "static=chelis_runtime"]',
            '.arg("-l").arg("static:+whole-archive=chelis_runtime")',
        ),
    ),
    Pattern(
        "linker-search",
        (
            r"-l[ \t]*(?:[a-z]+(?::[-+a-z,]+)?=)*:?(?:lib)?chelis_runtime\b",
            r"--library[ \t,]+:?(?:lib)?chelis_runtime\b",
        ),
        (
            'cmd.args(["-L.", "-lchelis_runtime"]);',
            'println!("cargo:rustc-link-lib=static=chelis_runtime");',
            'println!("cargo:rustc-link-lib=static:+whole-archive=chelis_runtime");',
            "cc main.c -Wl,--library,chelis_runtime",
        ),
    ),
    Pattern(
        "bundle-constant",
        (r"\b(?:ARCHIVE_FILE_NAME|RUNTIME_DIR_VARIABLE)\b",),
        (
            "let candidate = dir.join(chelis_runtime_bundle::ARCHIVE_FILE_NAME);",
            "runtime = os.environ.get(RUNTIME_DIR_VARIABLE)",
        ),
    ),
    Pattern(
        "runtime-package",
        (r"[\"']chelis-runtime[ @]", r"[#/]chelis-runtime(?:@|#(?![a-z]))"),
        (
            'if message["package_id"].startswith("chelis-runtime "):',
            'spec = "chelis-runtime@0.18.11"',
            'if package_id.endswith("/crates/chelis-runtime#" + version):',
            'spec = f"path+file://{root}#chelis-runtime@{version}"',
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
        ".github/scripts/smoke_macos_manifested_callable.py",
        "archive-name",
        lines=(
            'runtime_a = output_dir / "libchelis_runtime.a"',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its output directory"
        ),
    ),
    Row(
        ".github/scripts/smoke_macos_metal.py",
        "archive-name",
        lines=(
            'str(out_dir / "libchelis_runtime.a"),',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its output directory"
        ),
    ),
    Row(
        ".github/scripts/test_verify_release_smt.py",
        "archive-name",
        lines=(
            '(root / "staging" / "lib" / "libchelis_runtime.a").write_text("x")',
        ),
        disposition="not-lookup",
        reason=(
            "a fixture file in a release staging tree, where the verifier must find no `chelis` binary"
        ),
    ),
    Row(
        ".github/workflows/ecosystem-drift.yml",
        "archive-name",
        lines=(
            'cp "$runtime_export/libchelis_runtime.a" "$staging/lib/"',
            'cp "$tdir/lib/libchelis_runtime.a" "$ctx/lib/"',
            "COPY lib/libchelis_runtime.a /usr/local/lib/libchelis_runtime.a",
        ),
        disposition="not-lookup",
        reason=(
            "copies the exact archive from the compiler's export into the staged toolchain; the container copies that verified package archive to its image context and checks it again against its copied compiler's export"
        ),
    ),
    Row(
        ".github/workflows/release.yml",
        "archive-name",
        lines=(
            'cp "$runtime_export/libchelis_runtime.a" "$staging/lib/"',
            'cp "$runtime_export/libchelis_runtime.a" "$staging/lib/"',
            'cp "$runtime_export/libchelis_runtime.a" "$staging/lib/"',
            'cp "$runtime_export/libchelis_runtime.a" "$staging/lib/"',
            'cp "$runtime_export/libchelis_runtime.a" "$staging/lib/"',
        ),
        disposition="not-lookup",
        reason=(
            "copies the compiler's exact exported archive into each release tree; the package verifier checks the archive and all six headers against that sealed compiler's export"
        ),
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
        "bindings/python/tests/python_wheel_smoke.py",
        "runtime-variable",
        lines=('RUNTIME_DIR_ENV = "CHELIS_RUNTIME_DIR"',),
        disposition="not-lookup",
        reason=(
            "passes the variable name to a separate consumer to test staging rejection before artifact writes; it never selects an archive"
        ),
    ),
    Row(
        "bindings/python/tests/python_wheel_smoke.py",
        "archive-name",
        lines=('archive = artifact_dir / "libchelis_runtime.a"',),
        disposition="not-lookup",
        reason=(
            "links the exact staged wheel archive after checking its SHA-256 against the receipt, without a directory/name search"
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
        "archive-name",
        lines=(
            "// exported by libchelis_runtime; only the declarations are private.",
        ),
        disposition="not-lookup",
        reason=(
            "a comment naming the runtime library"
        ),
    ),
    Row(
        "crates/chelis-backend-c/src/lib.rs",
        "bundle-constant",
        lines=(
            "cmd.arg(dir.join(chelis_runtime_bundle::ARCHIVE_FILE_NAME));",
        ),
        disposition="not-lookup",
        reason=(
            "a test links the archive `stage_runtime` wrote, by exact path"
        ),
    ),
    Row(
        "crates/chelis-compiler-api/src/fp_env_arch.rs",
        "library-name",
        lines=(
            'call.args.is_empty() && segments == ["chelis_runtime", "FpEnvGuard", "enter"]',
        ),
        disposition="not-lookup",
        reason=(
            "an architecture test matches the Rust path `chelis_runtime::FpEnvGuard::enter` "
            "in parsed compiler-api source; it names no archive or library"
        ),
    ),
    Row(
        "crates/chelis-cli/src/lane_check.rs",
        "archive-name",
        lines=(
            'argv.push("out/libchelis_runtime.a".into());',
        ),
        disposition="not-lookup",
        reason=(
            "links the exact archive the preceding `chelis build --output out` staged "
            "for this program in the same isolated directory"
        ),
    ),
    Row(
        "crates/chelis-cli/src/native_build.rs",
        "archive-name",
        lines=(
            'runtime_archive: PathBuf::from("out/libchelis_runtime.a"),',
            'let mut expected = vec![OsStr::new("out/libchelis_runtime.a")];',
        ),
        disposition="not-lookup",
        reason=(
            "a unit test's fixture path for the staged archive; the test compares the printed "
            "link requirements with it and nothing links or searches for an archive"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/build_deep_ingestion.rs",
        "archive-name",
        lines=(
            'let runtime = dir.path().join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "plants a read-only stale archive where `chelis build` stages, which the build must replace"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cbackend_cast_memcpy.rs",
        "archive-name",
        lines=(
            "/// Compile `kernel.c + main.c + libchelis_runtime.a` and run the",
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "documents and links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cbackend_empty_tensor_numel.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cbackend_print_tensor_f64.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cbackend_reshape_memcpy.rs",
        "archive-name",
        lines=(
            "/// Compile `kernel.c + main.c + libchelis_runtime.a` and run the",
            'let runtime = build_dir.join("libchelis_runtime.a");',
            'let runtime = build.path().join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "documents and links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cli.rs",
        "archive-name",
        lines=(
            'cmd.arg(out_dir.join("libchelis_runtime.a"));',
            'cmd.arg(out_dir.join("libchelis_runtime.a"));',
            '"libchelis_runtime.a",',
            '.stdout(predicate::str::contains("libchelis_runtime.a"))',
            'let staged_archive = out_dir.join("libchelis_runtime.a");',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'runtime_dir.join("libchelis_runtime.a"),',
            'bin_dir.join("deps/libchelis_runtime-ffffffffffffffff.a"),',
            'bin_dir.join("libchelis_runtime.a"),',
            'dir.path().join("lib/libchelis_runtime.a"),',
            'let staged_archive = out_dir.join("libchelis_runtime.a");',
            'let archive = export_dir.join("libchelis_runtime.a");',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'out_dir.join("libchelis_runtime.a").display().to_string(),',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            'assert!(out_dir.join("libchelis_runtime.a").exists());',
            '"libchelis_runtime.a",',
            '"libchelis_runtime.a",',
            '"libchelis_runtime.a",',
        ),
        disposition="not-lookup",
        reason=(
            "links and asserts the archive `chelis build` staged in its output directory, compares `chelis runtime export`'s archive with the carried digest, and plants foreign archives beside the CLI and in a rejected runtime directory, which `chelis build` must not take"
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
            "sets the variable or removes it from a test's environment, and asserts that `chelis build` and the export reject it"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/common/mod.rs",
        "archive-name",
        lines=(
            'cmd.arg(out_dir.join("libchelis_runtime.a"));',
        ),
        disposition="not-lookup",
        reason=(
            "`link_generated` links the archive `chelis build` staged in its output directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/cross_library_semantic_gap_hip_gpu.rs",
        "archive-name",
        lines=(
            '.arg(out_dir.join("libchelis_runtime.a"))',
        ),
        disposition="not-lookup",
        reason=(
            "links the exact runtime archive staged by `chelis build` in `out_dir` before the HIP libraries"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_1137_shrink_expand_rank.rs",
        "archive-name",
        lines=(
            '.arg(out_dir.join("libchelis_runtime.a"))',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its output directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_1271_cross_package_ctor_collision.rs",
        "archive-name",
        lines=(
            'command.arg("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_1464_transformed_fail_traps.rs",
        "archive-name",
        lines=(
            '.arg(build_dir.join("libchelis_runtime.a"))',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_1710_c_source_names.rs",
        "archive-name",
        lines=(
            '.arg(output.join("libchelis_runtime.a"))',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its output directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_1771_callable_selected_result_claims.rs",
        "archive-name",
        lines=(
            '.arg("libchelis_runtime.a")',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_1801_wildcard_absorption.rs",
        "archive-name",
        lines=(
            'build_dir.join("libchelis_runtime.a").to_str().unwrap(),',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_2107_generated_declarations.rs",
        "archive-name",
        lines=(
            'command.args(["main.c", "driver.c", "libchelis_runtime.a"]);',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_2318_key_form.rs",
        "archive-name",
        lines=(
            '.arg("libchelis_runtime.a")',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_2442_binder_literal_pattern.rs",
        "archive-name",
        lines=(
            'out_dir.join("libchelis_runtime.a").to_str().unwrap(),',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its output directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_2582_cxx_host_linkage.rs",
        "archive-name",
        lines=(
            '//! declared `extern "C"`, and then the link against `libchelis_runtime.a`',
            '.arg("libchelis_runtime.a")',
            '"`{source}` does not link against libchelis_runtime.a:\\n{}",',
        ),
        disposition="not-lookup",
        reason=(
            "documents and links the archive `chelis build` staged in the output directory it "
            "compiles in, and names it in the link assertion's failure message"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_300_grad_codegen_compiles.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_352_captured_global_c_emit.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_368_grad_concat_windows.rs",
        "archive-name",
        lines=(
            'build_dir.join("libchelis_runtime.a").to_str().unwrap(),',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_513_reshape_shape_derived_grad.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_513_symbolic_axis_adjoints.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
            'let runtime = build_dir.join("libchelis_runtime.a");',
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_520_adt_match_grad.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_549_grad_permute_named_axis.rs",
        "archive-name",
        lines=(
            '.arg("libchelis_runtime.a")',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_551_grad_symbolic_concat_c_build.rs",
        "archive-name",
        lines=(
            "//!     `libchelis_runtime.a` and runs it,",
            "/// `<stem>.c`, the runtime headers, and `libchelis_runtime.a` into `<dir>`.",
            "/// `libchelis_runtime.a`, run the binary, and return stdout. A gcc or",
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "documents and links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_558_shape_value_read.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_579_chained_expand_eval.rs",
        "archive-name",
        lines=(
            '.arg("libchelis_runtime.a")',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_616_runtime_movement_c_parity.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_616_runtime_reshape_c_parity.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_620_static_if_adt_grad.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_631_guarded_forward_concat_c_parity.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_662_forward_fail_grad_scope.rs",
        "archive-name",
        lines=(
            '.arg(build_dir.join("libchelis_runtime.a"))',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_664_runtime_wildcard_consumer_guards.rs",
        "archive-name",
        lines=(
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_735_device_fence.rs",
        "archive-name",
        lines=(
            'assert!(output.join("libchelis_runtime.a").is_file(), "{name}");',
        ),
        disposition="not-lookup",
        reason=(
            "asserts that `chelis build` staged the archive in its output directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/issue_770_uniform_like_affine_parity.rs",
        "archive-name",
        lines=(
            'cmd.arg("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/jit_par_runtime_gap.rs",
        "archive-name",
        lines=(
            'cmd.arg("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/numeric_dtype_adversarial.rs",
        "archive-name",
        lines=(
            'cmd.arg("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/native_build.rs",
        "archive-name",
        lines=(
            '.arg(out.join("libchelis_runtime.a"))',
            '.arg(out.join("libchelis_runtime.a"))',
            'let runtime = out.join("libchelis_runtime.a");',
            'assert!(argv.contains("libchelis_runtime.a\\n"), "{argv}");',
            'let output = dir.path().join("out/libchelis_runtime.a.c");',
            'let runtime = fs::read(dir.path().join("out/libchelis_runtime.a")).unwrap();',
        ),
        disposition="not-lookup",
        reason=(
            "links the exact archive staged by the tested CLI in out, checks its path in native compiler argv, and probes a forbidden artifact collision before checking the carried digest; none selects a runtime from another location"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/rank_poly_tier2.rs",
        "archive-name",
        lines=(
            '.arg("libchelis_runtime.a")',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/rank_poly_tier3.rs",
        "archive-name",
        lines=(
            '.arg("libchelis_runtime.a")',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/reduction_and_bitwise_matrix.rs",
        "archive-name",
        lines=(
            '.arg("libchelis_runtime.a")',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in the output directory it compiles in"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/runtime_extent_slice_a.rs",
        "archive-name",
        lines=(
            'build_dir.join("libchelis_runtime.a").to_str().unwrap(),',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/std_io_pipeline.rs",
        "archive-name",
        lines=(
            'cmd.arg(out_dir.join("libchelis_runtime.a"));',
        ),
        disposition="not-lookup",
        reason=(
            "links the exact runtime archive staged by `chelis build` in `out_dir` before the remaining link flags"
        ),
    ),
    Row(
        "crates/chelis-cli/tests/ws2b_numeric_identifier_divergence.rs",
        "archive-name",
        lines=(
            "//!   links the debug `libchelis_runtime.a`, whose `debug_assert` is active.)",
            'let runtime = build_dir.join("libchelis_runtime.a");',
        ),
        disposition="not-lookup",
        reason=(
            "documents and links the archive `chelis build` staged in its build directory"
        ),
    ),
    Row(
        "crates/chelis-python/src/lib.rs",
        "archive-name",
        lines=(
            'dir.path().join("libchelis_runtime.a").exists(),',
            'let staged = fs::read(dir.path().join("libchelis_runtime.a")).expect("staged archive");',
        ),
        disposition="not-lookup",
        reason=(
            "asserts that `compile_and_load` staged the archive beside the shared library, and reads those staged bytes"
        ),
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
            "a test sets the variable, restores the caller's value and asserts that `compile_and_load` rejects it"
        ),
    ),
    Row(
        "crates/chelis-python/tests/manual_reef_context.rs",
        "runtime-variable",
        lines=(
            "//! unset CHELIS_RUNTIME_DIR",
        ),
        disposition="not-lookup",
        reason=(
            "the manual command clears an inherited override that the Python extension rejects"
        ),
    ),
    Row(
        "crates/chelis-runtime/tests/runtime_dtype_invalid_ffi.rs",
        "runtime-variable",
        lines=(
            'const CHILD_CASE_ENV: &str = "CHELIS_RUNTIME_DTYPE_INVALID_CHILD_CASE";',
        ),
        disposition="not-lookup",
        reason=(
            "names the variable that selects this test's child-process case, not a runtime location"
        ),
    ),
    Row(
        "crates/chelisup/src/cli.rs",
        "archive-name",
        lines=(
            '(libchelis_runtime.a sha256 {archive_sha256})"',
        ),
        disposition="not-lookup",
        reason=(
            "prints the digest of the archive the installed release ships"
        ),
    ),
    Row(
        "crates/chelisup/src/runtime_check.rs",
        "archive-name",
        lines=(
            "//! A release tarball ships `lib/libchelis_runtime.a` and the public runtime",
            'const ARCHIVE: &str = "libchelis_runtime.a";',
        ),
        disposition="not-lookup",
        reason=(
            "names the archive a release ships under `lib/`, which `chelisup` compares with the one the release's `chelis runtime export` writes"
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
        "archive-name",
        lines=(
            "/// `lib/libchelis_runtime.a` in the tarball.",
            "/// Ship `lib/libchelis_runtime.a` as a symlink to a sibling holding",
            '"archive": "libchelis_runtime.a",',
            'let exported_archive = runtime_export.join("libchelis_runtime.a");',
            'let archive = root.join("lib/libchelis_runtime.a");',
            'root.join("lib/libchelis_runtime.real"),',
            'std::os::unix::fs::symlink("libchelis_runtime.real", &archive).unwrap();',
        ),
        disposition="not-lookup",
        reason=(
            "builds fixture release tarballs with a shipped `lib/` archive, an exact fake-export archive, a symlinked archive `chelisup` must refuse, and a fake export receipt"
        ),
    ),
    Row(
        "crates/chelisup/tests/common/mod.rs",
        "runtime-variable",
        lines=(
            "/// refuses a set `CHELIS_RUNTIME_DIR`). Other invocations echo their args.",
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
        "archive-name",
        lines=(
            '"match `chelis runtime export` (libchelis_runtime.a sha256 {})",',
            'assert!(toolchain.join("lib/libchelis_runtime.a").is_file());',
            '"ships lib/libchelis_runtime.a with SHA-256",',
            '"runtime export has no usable libchelis_runtime.a",',
            '"runtime export reports libchelis_runtime.a with SHA-256",',
            '"runtime export has no usable libchelis_runtime.a: libchelis_runtime.a is not a regular file",',
            '"no usable lib/libchelis_runtime.a: lib/libchelis_runtime.a is not a regular file",',
            '"no usable lib/libchelis_runtime.a: lib is not a directory",',
            '"release has no usable lib/libchelis_runtime.a",',
        ),
        disposition="not-lookup",
        reason=(
            "asserts the installed toolchain's archive and `chelisup`'s messages about it"
        ),
    ),
    Row(
        "crates/chelisup/tests/install.rs",
        "runtime-variable",
        lines=(
            '("CHELIS_RUNTIME_DIR", "/elsewhere"),',
        ),
        disposition="not-lookup",
        reason=(
            "sets the variable in the caller's environment to show that `install` does not pass it to the export"
        ),
    ),
    Row(
        "nix/checks.nix",
        "archive-name",
        lines=(
            '"$package/lib/libchelis_runtime.a" \\',
        ),
        disposition="not-lookup",
        reason=(
            "links the exact installed archive in each of the `chelis-runtime` and `chelis` Nix packages, both verified against their compiler's export; the selected package path is not a runtime archive search"
        ),
    ),
    Row(
        "nix/contracts.nix",
        "archive-name",
        lines=(
            '"lib/libchelis_runtime.a"',
            '"lib/libchelis_runtime.a"',
            '"lib/libchelis_runtime.a"',
            '"lib/libchelis_runtime.a"',
        ),
        disposition="not-lookup",
        reason=(
            "lists the archive among the files the Nix toolchain and runtime packages must ship"
        ),
    ),
    Row(
        "nix/packages.nix",
        "archive-name",
        lines=(
            'install -Dm444 "$export_dir/libchelis_runtime.a" $out/lib/libchelis_runtime.a',
            "cp ${runtime}/lib/libchelis_runtime.a $out/lib/libchelis_runtime.a",
        ),
        disposition="not-lookup",
        reason=(
            "installs the archive from this compiler's exact export and copies it into the combined Nix package; both outputs verify the archive and six headers against their compiler exports"
        ),
    ),
    Row(
        "scripts/capacity_census_native_execution.py",
        "archive-name",
        lines=(
            'staged = _digest(library.parent / "libchelis_runtime.a")',
            '_require(isinstance(receipt, dict) and receipt.get("archive") == "libchelis_runtime.a"',
        ),
        disposition="not-lookup",
        reason=(
            "checks the archive staged beside a compiled library against its staging receipt"
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
        "archive-name",
        lines=(
            'RUNTIME_ARCHIVE_NAME = "libchelis_runtime.a"',
        ),
        disposition="not-lookup",
        reason=(
            "names the file the CLI's export and `chelis build` write, which the oracle reads by exact path and compares with the CLI's own export"
        ),
    ),
    Row(
        "scripts/compiled_value_ownership_oracle.py",
        "bundle-constant",
        lines=(
            'RUNTIME_DIR_VARIABLE = "CHELIS_RUNTIME_DIR"',
            "if RUNTIME_DIR_VARIABLE in os.environ:",
            'f"{RUNTIME_DIR_VARIABLE} is set, but chelis build rejects it and this "',
            'f"oracle links only the runtime its CLI carries. Unset {RUNTIME_DIR_VARIABLE}"',
        ),
        disposition="not-lookup",
        reason=(
            "names the variable only to refuse an inherited value"
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
            "requires the CLI's Cargo build to have compiled chelis-runtime with the ledger feature; the reference digest comes from the CLI's own export"
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
        "scripts/check_single_compile.py",
        "archive-name",
        lines=(
            'RUNTIME_ARCHIVE = "libchelis_runtime.a"',
        ),
        disposition="not-lookup",
        reason=(
            "the runtime archive's name, which the single-compile guard compares compile arguments with; it reads no file"
        ),
    ),
    Row(
        "scripts/datetime_lanes.py",
        "archive-name",
        lines=(
            '"out/libchelis_runtime.a", *self.toolchain.link_flags, "-o", "out/case"], app)',
        ),
        disposition="not-lookup",
        reason=(
            "the Std.Datetime differentials' shared C lane links the archive `chelis build --emit-c --output out` staged in the case's output directory, by its exact path"
        ),
    ),
    Row(
        "scripts/test_datetime_lanes.py",
        "archive-name",
        lines=(
            '(out / "libchelis_runtime.a").write_text("")',
            '["cc", "-O1", "-Werror", "-Iout", "out/main.c", "out/libchelis_runtime.a", "-lm", "-o", "out/case"],',
        ),
        disposition="not-lookup",
        reason=(
            "a stub `chelis build` writes an empty archive at the staged path, and the test asserts the C lane links that exact path"
        ),
    ),
    Row(
        "scripts/decimal_differential.py",
        "archive-name",
        lines=(
            '"out/libchelis_runtime.a", *self.toolchain.link_flags, "-o", "out/case"]',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build --output out` staged in the case's output directory, by its exact path"
        ),
    ),
    Row(
        "scripts/installed_artifact_canary.py",
        "archive-name",
        lines=(
            'REQUIRED_FILES = ("bin/chelis", "lib/libchelis_runtime.a", *HEADERS)',
            'archive = inventory["lib/libchelis_runtime.a"]',
            'if digest(output / "libchelis_runtime.a") != archive:',
            'expected = {"schema": "chelis-runtime-staging/1", "archive": "libchelis_runtime.a",',
            'if report["installed_export_sha256"] != inventory["lib/libchelis_runtime.a"]:',
            'installed / "lib/libchelis_runtime.a", *link_flags,',
            'installed / "lib/libchelis_runtime.a", *link_flags, "-o", binary])',
            'if len(words) < 2 or not words[1].endswith("libchelis_runtime.a"):',
        ),
        disposition="not-lookup",
        reason=(
            "compares the installed archive with its compiler's exact export and the staged archive/receipt, then links the checked installed archive by path for native execution with the link requirements the build reported, after checking that line names the runtime archive"
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
        "scripts/nautilus_local_gate.py",
        "archive-name",
        lines=(
            'str(out_dir / "libchelis_runtime.a"),',
        ),
        disposition="not-lookup",
        reason=(
            "links the archive `chelis build` staged in its output directory"
        ),
    ),
    Row(
        "scripts/runtime_representation_phase1.py",
        "archive-name",
        lines=(
            "empty = bad / 'libchelis_runtime.a'",
        ),
        disposition="not-lookup",
        reason=(
            "the empty archive in the directory a negative control offers `chelis build` through the variable, which it must refuse"
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
        "scripts/runtime_bundle_oracle.py",
        "archive-name",
        lines=(
            'ARCHIVE_FILE_NAME = "libchelis_runtime.a"',
            '"older": f"libchelis_runtime-{older_digest[:16]}-oracle.a",',
            '"newer": f"libchelis_runtime-{newer_digest[:16]}-oracle.a",',
        ),
        disposition="not-lookup",
        reason=(
            "the oracle validates the exact archive named by a staging receipt and plants build-tree decoys only for negative execution; it never selects a link input"
        ),
    ),
    Row(
        "scripts/runtime_bundle_oracle.py",
        "bundle-constant",
        lines=(
            'ARCHIVE_FILE_NAME = "libchelis_runtime.a"',
            'RUNTIME_DIR_VARIABLE = "CHELIS_RUNTIME_DIR"',
            "archive = artifacts / ARCHIVE_FILE_NAME",
            'good = context["run_dir"] / "exports" / "cli-development" / ARCHIVE_FILE_NAME',
            '"baseline": ARCHIVE_FILE_NAME,',
            'archive = package / "lib" / ARCHIVE_FILE_NAME',
            'if not (runtime_package / "lib" / ARCHIVE_FILE_NAME).is_file():',
            'crossed = copied_lib / ARCHIVE_FILE_NAME',
            "env.pop(RUNTIME_DIR_VARIABLE, None)",
            "env=command_env(additions={RUNTIME_DIR_VARIABLE: str(override_dir)}),",
            "env=command_env(additions={RUNTIME_DIR_VARIABLE: str(override_dir)}),",
            "if RUNTIME_DIR_VARIABLE not in export_stderr or any(rejected_export.iterdir()):",
            "if RUNTIME_DIR_VARIABLE not in stderr:",
            "path.name == ARCHIVE_FILE_NAME for path in rejected_output.iterdir()",
            "contains=RUNTIME_DIR_VARIABLE,",
            "additions={RUNTIME_DIR_VARIABLE: str(override_dir)},",
            'if (no_rebuild_dir / RECEIPT_FILE_NAME).exists() or (no_rebuild_dir / ARCHIVE_FILE_NAME).exists():',
            'f\'let candidate = build_dir.join("{ARCHIVE_FILE_NAME}");\\n\',',
        ),
        disposition="not-lookup",
        reason=(
            "names the staged archive and forbidden runtime-location variable only to verify receipts/packages and execute rejection or planted-scan controls"
        ),
    ),
    Row(
        "scripts/runtime_bundle_oracle.py",
        "library-name",
        lines=(
            '(arg.startswith("-l") and "chelis_runtime" in arg)',
            'or (arg == "-l" and index + 1 < len(argv) and "chelis_runtime" in argv[index + 1])',
        ),
        disposition="not-lookup",
        reason="the oracle rejects any printed link command that searches for the runtime by library name",
    ),
    Row(
        "scripts/runtime_bundle_oracle.py",
        "runtime-variable",
        lines=(
            'RUNTIME_DIR_VARIABLE = "CHELIS_RUNTIME_DIR"',
        ),
        disposition="not-lookup",
        reason=(
            "the oracle sets the forbidden variable only for the rejection witness and removes inherited values from positive runs"
        ),
    ),
    Row(
        "scripts/test_core_fragment_parity_receipt.py",
        "archive-name",
        lines=(
            '"Compile: clang -O2 -march=native out/k.c out/libchelis_runtime.a "',
        ),
        disposition="not-lookup",
        reason=(
            "a fixture string reproducing the `Compile:` line `chelis build` prints, "
            "used to test the parity receipt's compile-line extraction; the receipt "
            "runs whatever command the build emits and never names the archive itself"
        ),
    ),
    Row(
        "scripts/test_runtime_bundle_oracle.py",
        "bundle-constant",
        lines=(
            "archive = self.root / oracle.ARCHIVE_FILE_NAME",
            "archive = self.root / oracle.ARCHIVE_FILE_NAME",
            "archive = self.root / oracle.ARCHIVE_FILE_NAME",
            "archive = self.root / oracle.ARCHIVE_FILE_NAME",
            "archive = self.root / oracle.ARCHIVE_FILE_NAME",
            '"archive": oracle.ARCHIVE_FILE_NAME,',
            '"archive": oracle.ARCHIVE_FILE_NAME,',
            'f\'let candidate = build_dir.join("{oracle.ARCHIVE_FILE_NAME}");\\n\',',
            '(package / "lib" / oracle.ARCHIVE_FILE_NAME).write_bytes(b"runtime-A")',
            'archive = package / "lib" / oracle.ARCHIVE_FILE_NAME',
            'baseline = lib_dir / oracle.ARCHIVE_FILE_NAME',
        ),
        disposition="not-lookup",
        reason=(
            "uses the oracle's receipt-name constant in byte-integrity, receipt-replay tamper, and planted guard fixtures"
        ),
    ),
    Row(
        "scripts/test_runtime_bundle_oracle.py",
        "linker-search",
        lines=(
            """('cmd.arg("-lchelis_runtime");',),""",
            'build["stdout"] = f"Compile: {command} -lchelis_runtime\\n".encode()',
        ),
        disposition="not-lookup",
        reason="a synthetic guard row proves that even a reviewed manual linker search blocks oracle completion",
    ),
    Row(
        "scripts/test_capacity_census_native_execution.py",
        "archive-name",
        lines=(
            'def receipt_bytes(archive_sha256: str, archive: str = "libchelis_runtime.a") -> bytes:',
            '"compiled/libchelis_runtime.a": self.runtime,',
            'self.rewrite("compiled/libchelis_runtime.a", b"foreign runtime")',
            'self.rewrite("compiled/libchelis_runtime.a", runtime)',
            'candidates = [Path(packet["binaries"][0]["path"]), library.with_name("libchelis_runtime.a"),',
        ),
        disposition="not-lookup",
        reason=(
            "fixture receipts and staged archives beside a compiled library: the census must reject changed bytes and accept a consistently restaged runtime"
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
        "archive-name",
        lines=(
            'archive = target / "debug" / "deps" / "libchelis_runtime-0123abcd.a"',
            '(exported / "libchelis_runtime.a").write_bytes(carried)',
            'stdout = report(exported / "libchelis_runtime.a")',
            'staged = output / "libchelis_runtime.a"',
        ),
        disposition="not-lookup",
        reason=(
            "a decoy Cargo deps archive the oracle must not take, and fixture export and staged archives"
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
        "scripts/test_decimal_differential.py",
        "archive-name",
        lines=(
            '"Compile: clang -O2 -march=native out/main.c out/libchelis_runtime.a -lm -o out/main\\n")',
            '["clang", "-O2", "-march=native", "out/main.c", "out/libchelis_runtime.a", "-lm", "-o", "out/case"])',
            '(cwd / "out" / "libchelis_runtime.a").write_bytes(b"x")',
            'self.assertEqual((image / "out" / "libchelis_runtime.a").exists(), keep)',
        ),
        disposition="not-lookup",
        reason=(
            "a fixture `Compile:` line as `chelis build` prints it, naming the archive it staged, to test that the "
            "harness reruns that command with only its output retargeted; and a fixture staged archive the harness "
            "must delete once it has linked the program, unless artifacts are kept"
        ),
    ),
    Row(
        "scripts/test_check_single_compile.py",
        "archive-name",
        lines=(
            '\'    cc.current_dir(&native).args(["-O2", "order.c", "libchelis_runtime.a", "-lm", "-o"]);\\n\'',
            '\'    Command::new("cc").args(["main.c", "libchelis_runtime.a", "-o", "main"]).status().unwrap();\\n\'',
            '\'        .arg(out.join("libchelis_runtime.a"))\\n\'',
        ),
        disposition="not-lookup",
        reason=(
            "fixture Rust source the single-compile guard scans; nothing reads or links it"
        ),
    ),
    Row(
        "scripts/test_installed_artifact_canary.py",
        "archive-name",
        lines=(
            'for name in ("bin/chelis", "lib/libchelis_runtime.a", "include/chelis_runtime.h"):',
            '(exported / "libchelis_runtime.a").write_bytes(archive)',
            '(packaged / "lib/libchelis_runtime.a").write_bytes(archive)',
            '"archive": "libchelis_runtime.a",',
            '"archive_sha256": canary.digest(exported / "libchelis_runtime.a"),',
            '(packaged / "lib/libchelis_runtime.a").write_bytes(b"!<arch>\\nwrong")',
            'with self.assertRaisesRegex(ValueError, "libchelis_runtime.a"):',
            '(packaged / "lib/libchelis_runtime.a").write_bytes(archive)',
            'for name in ("lib/libchelis_runtime.a", *canary.HEADERS):',
            'sealed = {"schema": "chelis-runtime-staging/1", "archive": "libchelis_runtime.a",',
            '"archive_sha256": inventory["lib/libchelis_runtime.a"], "mode": "sealed",',
            '(output / "libchelis_runtime.a").write_bytes(b"swapped")',
            '"\'out dir/libchelis_runtime.a\' -lm -lpthread -ldl\\n"',
            '"out/libchelis_runtime.a -lm -framework Accelerate\\n"',
        ),
        disposition="not-lookup",
        reason=(
            "constructs a coherent compiler export and package, crosses each public header and the archive to prove rejection, tests the canary's installed/staged archive checks, and feeds its link-requirements parser build lines, with and without a leading SDK assignment, that name the archive"
        ),
    ),
    Row(
        "scripts/verify_runtime_package.py",
        "archive-name",
        lines=('ARCHIVE = "libchelis_runtime.a"',),
        disposition="not-lookup",
        reason=(
            "names the compiler-export archive whose receipt digest and copied package bytes the verifier compares; it never searches for a candidate"
        ),
    ),
    Row(
        "scripts/test_runtime_representation_phase1.py",
        "archive-name",
        lines=(
            "empty = evidence / 'empty-runtime/libchelis_runtime.a'",
        ),
        disposition="not-lookup",
        reason=(
            "the empty archive in the directory the rejection control offers"
        ),
    ),
    Row(
        "scripts/test_runtime_representation_phase1.py",
        "runtime-variable",
        lines=(
            "watched = ('CHELIS_RUNTIME_DIR', 'CHELIS_TEST_CC')",
            "oracle.os.environ.pop('CHELIS_RUNTIME_DIR', None)",
            "self.assertNotIn('CHELIS_RUNTIME_DIR', oracle.os.environ)",
            'return (f\'stderr="error: CHELIS_RUNTIME_DIR is set ({bad}), but chelis stages the runtime \'',
            '\'built into it and never takes one from a directory. Unset CHELIS_RUNTIME_DIR"\')',
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
    """Map each (file, pattern) with a match to the lines it matches.

    Each matching line appears once per pattern, as (line number, stripped text).
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
        for number, line in enumerate(text.splitlines(), 1):
            if not any(token in line for token in TOKENS):
                continue
            for pattern in PATTERNS:
                if pattern.regex.search(line):
                    found.setdefault((name, pattern.name), []).append((number, line.strip()))
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
            listed = []
            for text, count in unreviewed.items():
                places = [str(line) for line, seen in locations if seen == text]
                share = "" if count == len(places) else f" ({count} of these {len(places)} identical lines)"
                listed.append(f"{name}:{', '.join(places)}: {text}{share}")
            errors.append(
                f"{pattern}: {sum(unreviewed.values())} unreviewed line(s) in {name}. Every "
                "line that names the runtime outside its bundle needs a reviewed row. A lookup "
                "must instead link the archive that `chelis build`, "
                "`chelis_runtime_bundle::stage` or `chelis runtime export` wrote, by its exact "
                "path; a line that does so, or is not a lookup, gets a not-lookup row saying "
                "why.\n  " + "\n  ".join(listed)
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
