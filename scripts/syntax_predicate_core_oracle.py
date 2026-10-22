"""Check portable syntax, native contracts, and fixed-revision parity.

Target compilation does not prove device execution or device memory capacity.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import no_std_core_support as support
from no_std_core_support import ROOT, OracleFailure, compile_rejected, consumer, portable, run

BASELINE = "389e368d2d742e08918e00cbae36a8c9d06a9f47"
BASELINE_REPOSITORY = "https://github.com/Chelis-Lang/chelis.git"
CRATES = ("chelis-deep", "chelis-pred", "chelis-surf", "chelis-macros")
CASES = ("identity", "macro", "tensor", "integer-float")
OPERATIONS = ("parse", "format", "expand", "resugar")
FORBIDDEN = (
    "chelis-syntax-host", "miette", "stacker", "chelis-types", "chelis-reef",
    "chelis-runtime", "chelis-compiler-api", "chelis-backend-c",
    "chelis-backend-hip", "chelis-backend-metal", "tokio", "reqwest",
)

PORTABLE_SOURCE = r'''
#![no_std]
extern crate alloc;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use chelis_stack_services::StackServices;

pub fn syntax(source: &str, stack: &impl StackServices) -> Result<Vec<chelis_deep::Expr>, String> {
    let decls = chelis_surf::parser::parse_str(source, stack).map_err(|e| e.to_string())?;
    let deep = chelis_surf::desugar::desugar_program(&decls, stack).map_err(|e| e.to_string())?;
    Ok(chelis_macros::expand_program(&deep, &Default::default())
        .map_err(|e| e.to_string())?.into_exprs())
}

pub fn predicate(source: &str, stack: &impl StackServices) -> Result<chelis_pred::PredAmenability, String> {
    let exprs = chelis_deep::parser::parse_str(source, stack).map_err(|e| e.to_string())?;
    chelis_pred::predicate_in_grammar(&exprs[0]).map_err(|e| e.to_string())?;
    Ok(chelis_pred::classify_predicate(&exprs[0]))
}
'''

HOST_SOURCE = r'''
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use chelis_deep::{Expr, Span};
use chelis_macros::{expand_program, ExpansionOptions};
use chelis_surf::{parser::parse_str, desugar::desugar_program, format::format_source};
use chelis_surf::resugar::{normalize_deep_for_surface_roundtrip, resugar_program};
__NATIVE_IMPORT__

struct Counted;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Counted = Counted;

fn measure(operation: impl Fn()) -> serde_json::Value {
    operation();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    let started = Instant::now();
    for _ in 0..8 { operation(); }
    let elapsed = started.elapsed().as_nanos();
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    serde_json::json!({"iterations": 8, "allocations": allocations,
                      "allocated_bytes": bytes, "nanoseconds": elapsed})
}

fn main() {
    let mut semantic = serde_json::Map::new();
    let mut costs = serde_json::Map::new();
    for (name, source) in [
        ("identity", "def identity(x) = x\n"),
        ("macro", "macro twice(x) = add(x, x)\ndef result() = twice(2.0f32)\n"),
        ("tensor", "def twice(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)\n"),
        ("integer-float", "def result() -> f32 = 9007199791611905f32\n"),
    ] {
        let canonical = format_source(source__STACK__).unwrap();
        assert_eq!(format_source(&canonical__STACK__).unwrap(), canonical);
        let decls = parse_str(source__STACK__).unwrap();
        let deep = desugar_program(&decls__STACK__).unwrap();
        let options = ExpansionOptions::default();
        let expanded = expand_program(&deep, &options).unwrap().into_exprs();
        let recovered = resugar_program(&expanded).unwrap();
        let again = expand_program(&desugar_program(&recovered__STACK__).unwrap(), &options)
            .unwrap().into_exprs();
        assert_eq!(normalize_deep_for_surface_roundtrip(&expanded),
                   normalize_deep_for_surface_roundtrip(&again));
        let printed = chelis_deep::printer::print_canonical(&expanded);
        let reparsed = chelis_deep::parser::parse_and_stamp_file(&printed__STACK__).unwrap();
        assert_eq!(chelis_deep::printer::print_canonical(&reparsed), printed);
        let wire = serde_json::to_string(&expanded).unwrap();
        let decoded: Vec<Expr> = serde_json::from_str(&wire).unwrap();
        assert_eq!(decoded, expanded);
        assert_eq!(serde_json::to_string(&decoded).unwrap(), wire);
        let binary = bincode::serialize(&expanded).unwrap();
        assert_eq!(bincode::deserialize::<Vec<Expr>>(&binary).unwrap(), expanded);
        semantic.insert(name.into(), serde_json::json!({
            "canonical": canonical, "deep": printed, "json": wire, "binary": binary,
            "recovered": chelis_surf::format::format_program(&recovered)
        }));
        let records = [
            ("parse", measure(|| { black_box(parse_str(black_box(source)__STACK__).unwrap()); })),
            ("format", measure(|| { black_box(format_source(black_box(source)__STACK__).unwrap()); })),
            ("expand", measure(|| { black_box(expand_program(black_box(&deep), &options).unwrap()); })),
            ("resugar", measure(|| { black_box(resugar_program(black_box(&expanded)).unwrap()); })),
        ].into_iter().map(|(key, value)| (key.into(), value)).collect();
        costs.insert(name.into(), serde_json::Value::Object(records));
    }
    let predicate = chelis_deep::parser::parse_str(
        "(fn {} (params {} p) (app {} (var {} add) (var {} beta) (app {} (var {} add) (var {} alpha) (var {} beta))))"__STACK__
    ).unwrap();
    chelis_pred::predicate_in_grammar(&predicate[0]).unwrap();
    assert_eq!(chelis_pred::classify_predicate(&predicate[0]), chelis_pred::PredAmenability::Linear);
    assert_eq!(chelis_pred::predicate_free_vars(&predicate[0]), ["beta", "alpha"]);
    let partial = chelis_deep::parser::parse_str(
        "(fn {} (params {} p) (app {} (var {} sqrt) (var {} p)))"__STACK__
    ).unwrap();
    chelis_pred::predicate_in_grammar(&partial[0]).unwrap();
    let rejected = chelis_deep::parser::parse_str(
        "(fn {} (params {} p) (app {} (var {} arbitrary) (var {} p)))"__STACK__
    ).unwrap();
    assert_eq!(chelis_pred::classify_predicate(&rejected[0]), chelis_pred::PredAmenability::Opaque);
    semantic.insert("predicate".into(), serde_json::json!({
        "free_vars": chelis_pred::predicate_free_vars(&predicate[0]),
        "rejection": chelis_pred::predicate_in_grammar(&rejected[0]).unwrap_err().to_string()
    }));
    semantic.insert("errors".into(), serde_json::json!([
        parse_str("def broken( ="__STACK__).unwrap_err().to_string(),
        parse_str("value = 1u64\n"__STACK__).unwrap_err().to_string(),
        chelis_deep::parser::parse_and_stamp("(def {} bad (app {} bare_name))"__STACK__).unwrap_err().to_string(),
        chelis_deep::parser::parse_and_stamp_file("(var {} x)"__STACK__).unwrap_err().to_string()
    ]));
    semantic.insert("span".into(), serde_json::to_value(Span::new(2, 5)).unwrap());
    let path = std::path::Path::new("nn/linear.ch");
    __IDENTITY__::validate_module_path("Demo", "Demo.Nn.Linear", path).unwrap();
    semantic.insert("path_error".into(), serde_json::json!(
        __IDENTITY__::validate_module_path("Demo", "Demo.Other", path).unwrap_err()
    ));
    println!("SYNTAX_SNAPSHOT {}", serde_json::json!({"semantic": semantic, "costs": costs}));
}
'''


def dependencies(workspace: Path, *, native: bool, defaults: bool = False) -> str:
    names = list(CRATES[1:])
    if native:
        names.append("chelis-syntax-host")
    else:
        names.append("chelis-stack-services")
    return "\n".join(
        f'{name} = {{ path = {json.dumps(str(workspace / "crates" / name))}, '
        f'default-features = {str(defaults).lower()} }}'
        for name in names
    ) + "\n"


def snapshot(workspace: Path, *, native: bool) -> dict:
    environment = None
    if not native:
        environment = {
            **support._environment(),
            "CARGO_TARGET_DIR": str(workspace.resolve() / "target"),
        }
    source = HOST_SOURCE.replace("__STACK__", ", &NativeStack" if native else "")
    source = source.replace("__NATIVE_IMPORT__", "use chelis_syntax_host::NativeStack;" if native else "")
    source = source.replace("__IDENTITY__", "chelis_syntax_host::module_identity" if native else "chelis_surf::module_identity")
    if native:
        extra = dependencies(workspace, native=True)
    else:
        extra = "\n".join(
            f'{name} = {{ path = {json.dumps(str(workspace / "crates" / name))} }}'
            for name in CRATES[1:]
        ) + "\n"
    result = consumer(workspace / "crates" / CRATES[0], source,
                      dependencies=extra + 'serde_json = "1"\nbincode = "1"\n',
                      env=environment)
    records = [line.removeprefix("SYNTAX_SNAPSHOT ") for line in result.stdout.splitlines()
               if line.startswith("SYNTAX_SNAPSHOT ")]
    if len(records) != 1:
        raise OracleFailure("missing or repeated syntax snapshot")
    try:
        return json.loads(records[0])
    except json.JSONDecodeError as error:
        raise OracleFailure("invalid syntax snapshot JSON") from error


def _validate_semantics(semantic: object) -> None:
    expected = {*CASES, "predicate", "errors", "span", "path_error"}
    if not isinstance(semantic, dict) or set(semantic) != expected:
        raise OracleFailure("incomplete syntax semantic cases")
    for case in CASES:
        source = semantic[case]
        if not isinstance(source, dict) or set(source) != {
            "canonical", "deep", "json", "binary", "recovered",
        }:
            raise OracleFailure(f"incomplete syntax source fields: {case}")
        if any(not isinstance(source[field], str) or not source[field]
               for field in ("canonical", "deep", "json", "recovered")):
            raise OracleFailure(f"incomplete syntax source text: {case}")
        binary = source["binary"]
        if not isinstance(binary, list) or not binary or any(
            type(byte) is not int or not 0 <= byte <= 255 for byte in binary
        ):
            raise OracleFailure(f"invalid syntax binary evidence: {case}")
    predicate = semantic["predicate"]
    if not isinstance(predicate, dict) or set(predicate) != {"free_vars", "rejection"}:
        raise OracleFailure("incomplete syntax predicate fields")
    free_vars = predicate["free_vars"]
    if not isinstance(free_vars, list) or not free_vars or any(
        not isinstance(name, str) or not name for name in free_vars
    ):
        raise OracleFailure("incomplete syntax predicate free variables")
    if not isinstance(predicate["rejection"], str) or not predicate["rejection"]:
        raise OracleFailure("incomplete syntax predicate rejection")
    errors = semantic["errors"]
    if not isinstance(errors, list) or len(errors) != 4 or any(
        not isinstance(error, str) or not error for error in errors
    ):
        raise OracleFailure("incomplete syntax diagnostic evidence")
    span = semantic["span"]
    if not isinstance(span, dict) or set(span) != {"offset", "len"} or any(
        type(value) is not int or value < 0 for value in span.values()
    ):
        raise OracleFailure("incomplete syntax span evidence")
    if not isinstance(semantic["path_error"], str) or not semantic["path_error"]:
        raise OracleFailure("incomplete syntax path-error evidence")


def compare_snapshots(baseline: dict, candidate: dict) -> None:
    try:
        for record in (baseline, candidate):
            _validate_semantics(record["semantic"])
        if baseline["semantic"] != candidate["semantic"]:
            raise OracleFailure("syntax, diagnostic, or wire parity differs")
        if set(baseline["costs"]) != set(CASES) or set(candidate["costs"]) != set(CASES):
            raise OracleFailure("incomplete syntax cost cases")
        for case in CASES:
            previous = baseline["costs"][case]
            current = candidate["costs"][case]
            if set(previous) != set(OPERATIONS) or set(current) != set(OPERATIONS):
                raise OracleFailure(f"incomplete syntax cost operations: {case}")
            for operation in OPERATIONS:
                old, new = previous[operation], current[operation]
                for record in (old, new):
                    for field in ("iterations", "allocations", "allocated_bytes", "nanoseconds"):
                        if type(record[field]) is not int or record[field] < 0:
                            raise OracleFailure(f"invalid syntax cost measurement: {case}/{operation}")
                    if record["iterations"] != 8:
                        raise OracleFailure(f"unequal syntax cost work: {case}/{operation}")
                for field in ("allocations", "allocated_bytes"):
                    if new[field] > old[field]:
                        raise OracleFailure(f"syntax {field} regression: {case}/{operation}")
    except (KeyError, TypeError) as error:
        raise OracleFailure("incomplete syntax snapshot") from error


def target_graphs() -> None:
    for crate in CRATES:
        for defaults in (False, True):
            portable(crate, f'#![no_std]\npub use {crate.replace("-", "_")}::*;\n',
                     default_features=defaults, forbidden=FORBIDDEN)
    for defaults in (False, True):
        portable(CRATES[0], PORTABLE_SOURCE, default_features=defaults,
                 dependencies=dependencies(ROOT, native=False, defaults=defaults), forbidden=FORBIDDEN)
    for crate in ("chelis-deep", "chelis-surf"):
        portable(crate, f'#![no_std]\npub use {crate.replace("-", "_")}::*;\n',
                 features=("pre-020-pipe-migration",), forbidden=FORBIDDEN)


def negative_controls() -> None:
    compile_rejected(CRATES[0], '#![no_std]\npub fn read() { let _ = std::fs::read("source.dp"); }\n',
                     expected=("E0433", "std"))
    compile_rejected(CRATES[0], '#![no_std]\npub fn parse() { let _ = chelis_deep::parser::parse_str(""); }\n',
                     expected=("E0061",))
    with support._package(CRATES[0], '#![no_std]\npub use chelis_deep::Span;\n',
                          executable=False, features=(), default_features=False,
                          dependencies=dependencies(ROOT, native=True)) as directory:
        result = run(("cargo", "metadata", "--format-version", "1", "--filter-platform", support.TARGET), cwd=directory)
        try:
            support._normal_packages(json.loads(result.stdout), FORBIDDEN)
        except OracleFailure as error:
            if "forbidden target dependency: chelis-syntax-host, stacker" not in str(error):
                raise OracleFailure("host dependency control failed for another reason") from error
        else:
            raise OracleFailure("host dependency control unexpectedly passed")


def native_suites() -> None:
    mandatory = {
        "chelis-deep": (
            "nesting_beyond_the_parser_segment_is_located_and_caller_independent",
            "measured_guard_accepts_the_red_zone_boundary_and_rejects_less",
            "var_through_parse_and_stamp_produces_node",
            "bare_name_via_parse_and_stamp_produces_stamp_error",
        ),
        "chelis-surf": (
            "expression_requests_do_not_share_explicit_spans",
            "source_and_analysis_requests_do_not_share_explicit_spans",
            "direct_parser_releases_deep_reference_type_on_small_stack",
            "direct_parser_rejects_invalid_deep_reference_type_on_small_stack",
        ),
        "chelis-pred": ("classify_linear", "classify_polynomial", "classify_transcendental"),
        "chelis-macros": (
            "macro_introduced_let_binder_is_renamed_and_user_binder_is_not",
            "simple_macro_expands_to_base_tags_with_source_metadata",
        ),
    }
    for crate in (*CRATES, "chelis-unord", "chelis-syntax-host"):
        run(("cargo", "nextest", "run", "-p", crate, "--lib", "--tests", "--ignore-default-filter",
             "-E", "not test(=span_id_thousand_nested_spans_complete_in_under_a_minute)"),
            required_tests=mandatory.get(crate, ()))
    run(("cargo", "nextest", "run", "-p", "chelis-surf", "--test", "pipe_sugar_contract",
         "--features", "pre-020-pipe-migration", "--ignore-default-filter",
         "-E", "test(/^migration_/)"),
        required_tests=("migration_preserves_literal_dtype_and_old_grouping",))
    run(("cargo", "nextest", "run", "-p", "chelis-validate", "--test", "rejected_surf_depth", "--ignore-default-filter"))
    run(("cargo", "nextest", "run", "-p", "chelis-compiler-api", "--test", "pipeline_contract", "--ignore-default-filter"))
    run(("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "style_gate", "--test", "prove_deep_nesting_2425", "--ignore-default-filter"),
        required_tests=("extreme_nesting_is_a_located_parse_diagnostic_in_check_and_prove",
                        "check_fails_on_non_canonical_surf_source"))


def _checkout_baseline(directory: Path) -> Path:
    baseline = directory.resolve()
    run(("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(baseline)))
    present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{commit}}"), cwd=baseline)
    if present.returncode != 0:
        run(("git", "fetch", "--quiet", "--no-tags", "--depth=1",
             BASELINE_REPOSITORY, BASELINE), cwd=baseline)
    run(("git", "checkout", "--quiet", "--detach", BASELINE), cwd=baseline)
    return baseline


def main() -> int:
    target_graphs()
    negative_controls()
    with tempfile.TemporaryDirectory(prefix="chelis-syntax-baseline-") as temporary:
        baseline = _checkout_baseline(Path(temporary) / "baseline")
        previous = snapshot(baseline, native=False)
        current = snapshot(ROOT, native=True)
        compare_snapshots(previous, current)
        print("SYNTAX_COMPARISON " + json.dumps({"baseline": BASELINE,
              "before": previous["costs"], "after": current["costs"]}, sort_keys=True), flush=True)
    native_suites()
    print("syntax core oracle: PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OracleFailure, OSError, ValueError) as error:
        print(f"syntax core oracle: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
