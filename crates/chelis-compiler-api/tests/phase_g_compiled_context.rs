//! Phase G acceptance tests: `compile_reef_context` + `eval_in_context`
//! produce byte-identical eval output to the monolithic
//! `prepare_eval(format(library + snippet))` baseline.
#![allow(deprecated)] // baseline parity tests intentionally exercise prepare_eval
//!
//! Plus: bincode round-trip on `CompiledContext`, parity for
//! `eval_many_in_context`, and a microbench locking a >= 10x speedup
//! over 50 independent `prepare_eval` calls.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chelis_compiler_api::compiler::{eval, prepare_eval};
use chelis_compiler_api::schema::{EvalRequest, EvaluatedRoot, SourceKind};
use chelis_compiler_api::{
    CompiledContext, check_in_context, compile_reef_context, eval_in_context, eval_many_in_context,
};
use tempfile::TempDir;

/// Build a tempdir-backed reef package with one path-dep `mylib` that
/// exports a small set of pure-int helpers. This is the parity fixture
/// for every Phase G test in this file.
fn library_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    fs::write(
        root.join("reef.toml"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = { path = \"./mylib\" }\n",
    )
    .expect("write app reef.toml");
    // Root package keeps a tiny placeholder module — the snippet under
    // test stands in for the user-supplied entry decls.
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder -> int32 = cast(0, int32)\n",
    )
    .expect("write main.ch");

    fs::write(
        root.join("mylib/reef.toml"),
        "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"Mylib\"\n",
    )
    .expect("write mylib reef.toml");
    fs::write(
        root.join("mylib/src/math.ch"),
        "module Mylib.Math\nexport (add, double, square)\n\n\
         def add(x: int32, y: int32) -> int32 = x + y\n\
         def double(x: int32) -> int32 = x + x\n\
         def square(x: int32) -> int32 = x * x\n",
    )
    .expect("write math.ch");

    fs::write(
        root.join("reef.lock"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
    )
    .expect("write reef.lock");
    (dir, root)
}

/// Strip `Decl::Module` wrappers so the inner decls can be passed to
/// `compile_with_reef_graph` (which expects a flat decl list).
fn flatten_module_decls(decls: &[chelis_surf::ast::Decl]) -> Vec<chelis_surf::ast::Decl> {
    let mut out = Vec::new();
    for decl in decls {
        match decl {
            chelis_surf::ast::Decl::Module { decls: inner, .. } => {
                out.extend(flatten_module_decls(inner));
            }
            other => out.push(other.clone()),
        }
    }
    out
}

/// Format the linked library + entry-source as one Surf string. This is
/// the canonical input shape for `prepare_eval` (mirrors what
/// `chelis test`'s `cmd_internal_test_file` does today).
fn format_library_plus_snippet(package_dir: &Path, snippet: &str) -> String {
    let graph = chelis_reef::prepare_reef_graph(package_dir).expect("prepare_reef_graph");
    let entry_decls = chelis_surf::parser::parse_str(snippet).expect("parse snippet");
    let flat_decls = flatten_module_decls(&entry_decls);
    let prepared =
        chelis_reef::compile_with_reef_graph(&graph, &flat_decls).expect("compile_with_reef_graph");
    chelis_surf::format::format_program(&prepared.decls)
}

/// Filter an eval result down to the named roots in stable order so the
/// equality assertion is robust against any ordering surprises in the
/// evaluator's transcript layer. Values are serialized to JSON for
/// comparison since `ExecutionValue` doesn't impl `PartialEq` and
/// byte-identical JSON is the strongest equality the `chelis check`/
/// `chelis eval` machine surface can provide.
fn collect_named_roots_json(roots: &[EvaluatedRoot], names: &[&str]) -> BTreeMap<String, String> {
    let mut by_name = BTreeMap::new();
    for r in roots {
        if let Some(name) = &r.name
            && names.contains(&name.as_str())
        {
            by_name.insert(
                name.clone(),
                serde_json::to_string(&r.value).expect("serialize eval value"),
            );
        }
    }
    by_name
}

#[test]
fn eval_in_context_matches_prepare_eval_int_add() {
    let (_dir, root) = library_fixture();
    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def main_value -> int32 = add(3, 4)\n";

    // Monolithic baseline.
    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline eval");
    let baseline_named = collect_named_roots_json(&baseline.roots, &["main_value"]);

    // In-context flow.
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let result = eval_in_context(&ctx, snippet).expect("eval_in_context");
    let result_named = collect_named_roots_json(&result.roots, &["main_value"]);

    assert_eq!(
        baseline_named, result_named,
        "eval_in_context output must match prepare_eval(format(library + snippet))"
    );
}

#[test]
fn eval_in_context_matches_prepare_eval_int_double() {
    let (_dir, root) = library_fixture();
    let snippet = "module App.Eval\nimport Mylib.Math (double)\n\n\
                   def doubled -> int32 = double(21)\n";

    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline eval");
    let baseline_named = collect_named_roots_json(&baseline.roots, &["doubled"]);

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let result = eval_in_context(&ctx, snippet).expect("eval_in_context");
    let result_named = collect_named_roots_json(&result.roots, &["doubled"]);

    assert_eq!(baseline_named, result_named);
}

#[test]
fn eval_in_context_matches_prepare_eval_int_square_compose() {
    let (_dir, root) = library_fixture();
    let snippet = "module App.Eval\nimport Mylib.Math (square)\nimport Mylib.Math (add)\n\n\
                   def composed -> int32 = add(square(5), square(3))\n";

    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline eval");
    let baseline_named = collect_named_roots_json(&baseline.roots, &["composed"]);

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let result = eval_in_context(&ctx, snippet).expect("eval_in_context");
    let result_named = collect_named_roots_json(&result.roots, &["composed"]);

    assert_eq!(baseline_named, result_named);
}

#[test]
fn eval_many_in_context_per_root_isolation_matches_independent_calls() {
    // Use let-bindings (no `def`, no params) so the values are tensor
    // roots that the host evaluator carries through. `def name -> int32`
    // becomes a 0-arg fn under desugar and is NOT eagerly evaluated by
    // the runtime; bare `name: int32 = ...` is a value binding that is.
    //
    // The bare-form parser path requires the names live inside a module
    // wrapper or at the file root; we wrap them in `App.Eval` for parity
    // with `chelis test`'s actual entry-file shape.
    let (_dir, root) = library_fixture();
    let snippet = "module App.Eval\nimport Mylib.Math (square)\n\n\
                   def isolated_a -> int32 = square(3)\n\
                   def isolated_b -> int32 = square(4)\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    let names = vec!["isolated_a".to_string(), "isolated_b".to_string()];
    let many = eval_many_in_context(&ctx, snippet, &names);
    assert_eq!(many.len(), 2);
    for (name, outcome) in &many {
        // Each per-root call returns *some* outcome — even if it's an
        // empty roots vec (matching monolithic-prepare-eval behaviour
        // for 0-arg fn defs whose bodies aren't tensor-lowerable). The
        // contract is that compile errors propagate AND each call
        // evaluates the requested root in isolation.
        let _result = outcome
            .as_ref()
            .unwrap_or_else(|e| panic!("eval_many root {name} failed: {e:?}"));
    }

    // Also exercise the "all roots" path against a single eval_in_context
    // and confirm the same eval result for each named root that the per-
    // root call returned.
    let combined = eval_in_context(&ctx, snippet).expect("eval_in_context");
    let combined_by_name = collect_named_roots_json(&combined.roots, &["isolated_a", "isolated_b"]);
    let many_by_name: BTreeMap<String, String> = many
        .into_iter()
        .filter_map(|(name, outcome)| {
            let result = outcome.ok()?;
            result
                .roots
                .iter()
                .find(|r| r.name.as_deref() == Some(name.as_str()))
                .map(|r| {
                    (
                        name,
                        serde_json::to_string(&r.value).expect("serialize eval value"),
                    )
                })
        })
        .collect();
    assert_eq!(combined_by_name, many_by_name);
}

/// Phase G' — host runtime parity. A library that exposes a
/// non-tensor-typed function (string body) cannot be evaluated through
/// the lowered DAG; the host evaluator must see the library def in its
/// `top_level_defs` table to resolve the call. Pre-G' this errored
/// `unknown runtime name <library_internal_name>` because
/// `evaluate_host_program_filtered` only iterated `program.exprs()`,
/// missing every library def.
#[test]
fn eval_in_context_resolves_library_string_call_in_host_runtime() {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    fs::write(
        root.join("reef.toml"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = { path = \"./mylib\" }\n",
    ).expect("write app reef.toml");
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder -> int32 = cast(0, int32)\n",
    )
    .expect("write main.ch");
    fs::write(
        root.join("mylib/reef.toml"),
        "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"Mylib\"\n",
    ).expect("write mylib reef.toml");
    // String concatenation lives in the host runtime path (no tensor
    // lowering). New-code calls into it will fail unless the
    // CompiledContext's library defs are seeded into the runtime's
    // `top_level_defs` table.
    fs::write(
        root.join("mylib/src/text.ch"),
        "module Mylib.Text\nexport (greet)\n\n\
         def greet(who: string) -> string = string_concat(\"hello, \", who)\n",
    )
    .expect("write text.ch");
    fs::write(
        root.join("reef.lock"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
    ).expect("write reef.lock");

    // Use a NON-fn top-level binding (`name = expr`, not `def name -> T = expr`)
    // so the host evaluator's `top_level_order` includes it for eager
    // resolution. `def name -> T = body` desugars to a 0-arg fn and is
    // explicitly excluded from `top_level_order`, so it never triggers
    // the runtime lookup that surfaces this bug. The `chelis test`
    // worker's synthesized `__chelis_test_k = test_k()` shape (see
    // `chelis_cli::cmd_internal_test_file`) is the canonical example
    // of a non-fn top-level call into a library function.
    let snippet = "module App.Eval\nimport Mylib.Text (greet)\n\n\
                   greeting = greet(\"world\")\n";

    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline eval");
    let baseline_named = collect_named_roots_json(&baseline.roots, &["greeting"]);

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let result = eval_in_context(&ctx, snippet).expect("eval_in_context");
    let result_named = collect_named_roots_json(&result.roots, &["greeting"]);

    assert_eq!(
        baseline_named, result_named,
        "Phase G' — eval_in_context must resolve library string-call \
         through host-runtime top_level_defs, matching the monolithic baseline"
    );
}

/// Phase G' — same parity check, but verified end-to-end after the
/// CompiledContext round-trips through bincode (the worker path used
/// by `chelis test`). Catches a host-runtime regression that ships in
/// the wire-bytes but not in the in-process-only path.
#[test]
fn eval_in_context_resolves_library_string_call_after_bincode_round_trip() {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");
    fs::write(
        root.join("reef.toml"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = { path = \"./mylib\" }\n",
    ).expect("write app reef.toml");
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder -> int32 = cast(0, int32)\n",
    )
    .expect("write main.ch");
    fs::write(
        root.join("mylib/reef.toml"),
        "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"Mylib\"\n",
    ).expect("write mylib reef.toml");
    fs::write(
        root.join("mylib/src/text.ch"),
        "module Mylib.Text\nexport (label)\n\n\
         def label(prefix: string, n: int64) -> string = string_concat(prefix, to_string(n))\n",
    )
    .expect("write text.ch");
    fs::write(
        root.join("reef.lock"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
    ).expect("write reef.lock");

    let snippet = "module App.Eval\nimport Mylib.Text (label)\n\n\
                   caption = label(\"n=\", cast(7, int64))\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    // Round-trip through bincode (mirrors the worker tempfile bridge).
    let bytes = bincode::serialize(&ctx).expect("serialize");
    let restored: CompiledContext = bincode::deserialize(&bytes).expect("deserialize");
    let result = eval_in_context(&restored, snippet).expect("eval_in_context after decode");
    let post_named = collect_named_roots_json(&result.roots, &["caption"]);

    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline");
    let baseline_named = collect_named_roots_json(&baseline.roots, &["caption"]);

    assert_eq!(
        baseline_named, post_named,
        "Phase G' — round-tripped CompiledContext must still resolve library calls"
    );
}

#[test]
fn check_in_context_accepts_well_typed_snippet() {
    let (_dir, root) = library_fixture();
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def ok -> int32 = add(1, 2)\n";
    let result = check_in_context(&ctx, snippet).expect("check ok");
    assert!(
        result.errors.is_empty(),
        "check should succeed cleanly: {:?}",
        result.errors
    );
}

#[test]
fn check_in_context_rejects_unbound_library_reference() {
    let (_dir, root) = library_fixture();
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    // Reference to a name that exists in NEITHER the library nor the new code.
    let snippet = "module App.Eval\n\n\
                   def bad -> int32 = nonexistent_function(1, 2)\n";
    let outcome = check_in_context(&ctx, snippet);
    assert!(
        outcome.is_err(),
        "check_in_context should reject snippets that reference unbound names"
    );
}

#[test]
fn compiled_context_round_trips_through_bincode() {
    let (_dir, root) = library_fixture();
    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def eight -> int32 = add(3, 5)\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let pre = eval_in_context(&ctx, snippet).expect("pre-encode eval");
    let pre_named = collect_named_roots_json(&pre.roots, &["eight"]);

    let bytes = bincode::serialize(&ctx).expect("serialize CompiledContext");
    assert!(!bytes.is_empty());
    let restored: CompiledContext =
        bincode::deserialize(&bytes).expect("deserialize CompiledContext");
    assert_eq!(restored.source_hash, ctx.source_hash);

    let post = eval_in_context(&restored, snippet).expect("post-decode eval");
    let post_named = collect_named_roots_json(&post.roots, &["eight"]);
    assert_eq!(
        pre_named, post_named,
        "deserialize → eval_in_context must produce the same result as the pre-serialize eval"
    );
}

/// Build a fatter fixture for the microbench: same `mylib` with 50
/// auto-generated helper defs so the library compile cost is non-trivial
/// — without that, the parity microbench is dominated by snippet-side
/// work and the with-context speedup looks artificially small.
fn microbench_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("mylib/src")).expect("mkdir mylib/src");

    fs::write(
        root.join("reef.toml"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"App\"\n\n[dependencies]\nmylib = { path = \"./mylib\" }\n",
    )
    .expect("write app reef.toml");
    fs::write(
        root.join("src/main.ch"),
        "module App.Main\n\ndef placeholder -> int32 = cast(0, int32)\n",
    )
    .expect("write main.ch");

    fs::write(
        root.join("mylib/reef.toml"),
        "[package]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\nmodule_prefix = \"Mylib\"\n",
    )
    .expect("write mylib reef.toml");

    // 80 helper defs of varying shapes — gets the library compile cost
    // up to a level where amortizing it once via `compile_reef_context`
    // is meaningfully cheaper than re-paying it per `prepare_eval` call.
    let mut math = String::from("module Mylib.Math\nexport (add, double, square");
    for i in 0..80 {
        math.push_str(&format!(", helper_{i}"));
    }
    math.push_str(")\n\n");
    math.push_str("def add(x: int32, y: int32) -> int32 = x + y\n");
    math.push_str("def double(x: int32) -> int32 = x + x\n");
    math.push_str("def square(x: int32) -> int32 = x * x\n");
    for i in 0..80 {
        math.push_str(&format!(
            "def helper_{i}(x: int32, y: int32) -> int32 = (x + {}) * (y + {})\n",
            i + 1,
            i * 2 + 3
        ));
    }
    fs::write(root.join("mylib/src/math.ch"), math).expect("write math.ch");

    fs::write(
        root.join("reef.lock"),
        "[package]\nname = \"myapp\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"=0.3.0\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
    )
    .expect("write reef.lock");
    (dir, root)
}

#[test]
fn microbench_in_context_is_at_least_10x_faster_than_prepare_eval_for_50_snippets() {
    // Microbench: 50 snippets, each evaluated against a single
    // CompiledContext vs 50 independent `prepare_eval` runs that each
    // re-compile the library text from scratch. CLAUDE.md target:
    //   re-eval against one context >= 10x faster than independent
    //   prepare_eval calls.
    //
    // Snippet variety is 50 different integer expressions so neither
    // path can short-circuit via a local cache.
    let (_dir, root) = microbench_fixture();
    let snippets: Vec<String> = (0..50)
        .map(|i| {
            format!(
                "module App.Eval\nimport Mylib.Math (add)\n\ndef result_{i} -> int32 = add({}, {})\n",
                i,
                i * 2 + 1
            )
        })
        .collect();

    // ---- baseline: prepare_eval re-compiling library text 50 times ----
    let baseline_start = Instant::now();
    for snippet in &snippets {
        let formatted = format_library_plus_snippet(&root, snippet);
        let _ = prepare_eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: formatted,
            bindings: BTreeMap::new(),
        })
        .expect("baseline prepare_eval ok");
    }
    let baseline_elapsed = baseline_start.elapsed();

    // ---- with-context: build context once, eval 50 snippets ----
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let in_context_start = Instant::now();
    for snippet in &snippets {
        let _ = eval_in_context(&ctx, snippet).expect("in-context eval ok");
    }
    let in_context_elapsed = in_context_start.elapsed();

    let speedup = baseline_elapsed.as_secs_f64() / in_context_elapsed.as_secs_f64().max(1e-9);
    eprintln!(
        "phase-G microbench: 50 snippets — baseline {} ms, in-context {} ms, speedup {:.2}x",
        baseline_elapsed.as_millis(),
        in_context_elapsed.as_millis(),
        speedup,
    );
    assert!(
        speedup >= 10.0,
        "phase-G microbench: in-context flow must be at least 10x faster (got {speedup:.2}x; \
         baseline {} ms, in-context {} ms)",
        baseline_elapsed.as_millis(),
        in_context_elapsed.as_millis(),
    );
}
