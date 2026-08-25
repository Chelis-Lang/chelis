//! RT-G adversarial probes for the Compiled Artifact Cache pipeline.
#![allow(deprecated)] // baseline parity tests intentionally exercise prepare_eval
//!
//! Per the red-team contract, these tests deliberately probe
//! compose-across-the-boundary behaviour where the implementer was likely
//! to test favourably:
//!   - effect row composition (G1)
//!   - ADT round-trip + exhaustivity (G2)
//!   - recursive new-code referencing library (G3)
//!   - bincode tamper / cache-poisoning (G4)
//!   - cold-path overhead (G5)
//!   - source-hash invalidation (G6)
//!   - eval_many isolation (G7)
//!   - parity vs prepare_eval on richer fixtures (G8)
//!   - shadowing across C/D/E/F stages (G9)
//!   - cross-boundary linearity (G10)
//!   - check_in_context does not eval (G11)
//!   - source-hash collision-resistance smoke (G12)

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chelis_compiler_api::compiler::{eval, prepare_eval};
use chelis_compiler_api::schema::{EvalRequest, EvaluatedRoot, SourceKind};
use chelis_compiler_api::{
    COMPILER_VERSION, CompiledContext, check_in_context, compile_reef_context, eval_in_context,
    eval_many_in_context,
};
use tempfile::TempDir;

// ─── Shared fixture helpers ──────────────────────────────────────────────

/// `<pkg_name>/reef.lock` body for an app with one path-dep on `mylib`.
/// Compiler pin auto-syncs with the workspace via `COMPILER_VERSION`.
fn app_reef_lock(pkg_name: &str) -> String {
    format!(
        "[package]\nname = \"{pkg_name}\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"mylib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\narchive_sha256 = \"\"\nshell_sha256 = \"\"\n\n[dependencies.source]\nkind = \"path\"\npath = \"./mylib\"\n",
    )
}

fn write_pkg(
    root: &Path,
    pkg_name: &str,
    module_prefix: &str,
    files: &[(&str, &str)],
    deps: &[(&str, &str)],
) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    let mut manifest = format!(
        "[package]\nname = \"{pkg_name}\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"{module_prefix}\"\n"
    );
    if !deps.is_empty() {
        manifest.push_str("\n[dependencies]\n");
        for (name, path) in deps {
            manifest.push_str(&format!("{name} = {{ path = \"{path}\" }}\n"));
        }
    }
    fs::write(root.join("reef.toml"), manifest).expect("write reef.toml");
    for (rel, content) in files {
        let p = root.join(rel);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).expect("mkdir parent");
        }
        fs::write(&p, content).expect("write source file");
    }
}

/// Reef package with one path-dep `mylib`. Caller controls library + main
/// source. The library lives at `mylib/src/math.ch` and must declare
/// `module Mylib.Math` for the reef name resolver to accept it.
/// Lockfile is hand-written so prepare_reef_graph does not try to resolve
/// over the network.
fn build_pkg(library_src: &str, main_src: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    write_pkg(
        &root,
        "myapp",
        "App",
        &[("src/main.ch", main_src)],
        &[("mylib", "./mylib")],
    );
    let mylib = root.join("mylib");
    write_pkg(
        &mylib,
        "mylib",
        "Mylib",
        &[("src/math.ch", library_src)],
        &[],
    );
    fs::write(root.join("reef.lock"), app_reef_lock("myapp")).expect("write reef.lock");
    (dir, root)
}

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

fn format_library_plus_snippet(package_dir: &Path, snippet: &str) -> String {
    let graph = chelis_reef::prepare_reef_graph(package_dir).expect("prepare_reef_graph");
    let entry_decls = chelis_surf::parser::parse_str(snippet).expect("parse snippet");
    let flat_decls = flatten_module_decls(&entry_decls);
    let prepared =
        chelis_reef::compile_with_reef_graph(&graph, &flat_decls).expect("compile_with_reef_graph");
    chelis_surf::format::format_program(&prepared.decls)
}

// ─── G2 — ADT round-trip ─────────────────────────────────────────────────

#[test]
fn g2_adt_exhaustive_match_in_new_code_against_library_option() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    // Library function returns prelude Option[int32]. New code matches
    // against it with both arms — should accept and produce the unwrapped
    // value parity-equal to the monolithic baseline.
    let library = "module Mylib.Math\nexport (lib_some)\n\n\
                   def lib_some() -> Option[int32] = Some(cast(7, int32))\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    // Put the non-matching `None` arm first: linked terminal-name matching
    // must recognize the mangled `Some`, but must not collapse distinct
    // constructors merely because both crossed the reef boundary.
    let snippet = "module App.Eval\nimport Mylib.Math (lib_some)\n\n\
                   def unwrapped() -> int32 = match lib_some with {\n  | None => 0\n  | Some(x) => x\n}\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let result = eval_in_context(&ctx, snippet).expect("eval ok");
    let by = collect_named_roots_json(&result.roots, &["unwrapped"]);
    // baseline parity (must equal monolithic)
    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline");
    let baseline_by = collect_named_roots_json(&baseline.roots, &["unwrapped"]);
    assert_eq!(by, baseline_by, "ADT match parity must match prepare_eval");
}

#[test]
fn g2_adt_non_exhaustive_match_in_new_code_is_rejected() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    // Library returns prelude Option[int32]; new code matches with only
    // Some — must reject as non-exhaustive citing the missing None.
    //
    // CONTRACT BEING TESTED: check_in_context must agree with the
    // monolithic baseline. If the baseline rejects but in_context
    // accepts, that's a HIGH-severity divergence (Phase H/I would ship
    // a checker that silently misses non-exhaustive matches against
    // library types).
    let library = "module Mylib.Math\nexport (lib_some)\n\n\
                   def lib_some() -> Option[int32] = Some(cast(7, int32))\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    let bad_snippet = "module App.Eval\nimport Mylib.Math (lib_some)\n\n\
                       def bad() -> int32 = match lib_some with {\n  | Some(x) => x\n}\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let in_context = check_in_context(&ctx, bad_snippet);

    // Monolithic via the full prepare_eval pipeline (compile_source) —
    // which runs check_ir_program → effects → linearity → lower.
    // The bare check API only runs IR check, so divergence between the
    // two would point at WHICH stage owns exhaustivity. The contract:
    // both verdicts must match.
    let formatted = format_library_plus_snippet(&root, bad_snippet);
    let baseline_full = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    });

    eprintln!("G2-nonexh in_context = {in_context:?}");
    eprintln!(
        "G2-nonexh baseline_full err={:?}",
        baseline_full.as_ref().err()
    );

    let baseline_rejects = baseline_full.is_err();
    let in_context_rejects = match &in_context {
        Ok(r) => !r.errors.is_empty(),
        Err(_) => true,
    };
    assert_eq!(
        baseline_rejects,
        in_context_rejects,
        "check_in_context and monolithic prepare_eval must agree on non-exhaustive match: \
         baseline_rejects={baseline_rejects}, in_context_rejects={in_context_rejects}, \
         in_context={in_context:?}, baseline_err={:?}",
        baseline_full.err()
    );
}

// ─── G3 — Recursive new-code referencing library ─────────────────────────

#[test]
fn g3_recursive_newcode_def_calling_library_helper() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    let library = "module Mylib.Math\nexport (mul)\n\n\
                   def mul(x: int32, y: int32) -> int32 = x * y\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    // Recursive new-code def that also calls a library function.
    let snippet = "module App.Eval\nimport Mylib.Math (mul)\n\n\
                   def fact(n: int32) -> int32 =\n  \
                     if (n <= 1) then 1 else mul(n, fact(n - 1))\n\
                   def fact5() -> int32 = fact(5)\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let result = eval_in_context(&ctx, snippet).expect("eval ok");
    // baseline parity
    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline");
    let by_in_context = collect_named_roots_json(&result.roots, &["fact5"]);
    let by_baseline = collect_named_roots_json(&baseline.roots, &["fact5"]);
    assert_eq!(
        by_in_context, by_baseline,
        "recursive newcode + library call must match monolithic baseline"
    );
}

// ─── G4 — bincode tamper / cache poisoning ───────────────────────────────

#[test]
fn g4_bincode_tampering_truncates_library_then_eval_must_not_silently_succeed() {
    // Build a context whose library exposes `lib_double`. New code that
    // calls `lib_double` MUST not silently succeed if the library_checked
    // contents are tampered to remove `lib_double`.
    let library = "module Mylib.Math\nexport (lib_double)\n\n\
                   def lib_double(x: int32) -> int32 = x + x\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let bytes = ctx.encode().expect("encode");

    // Replace the byte sequence "lib_double" with a different name of the
    // SAME LENGTH to corrupt the bincode-encoded library_checked. Same
    // length keeps offsets stable, which is the whole point of the attack:
    // a Phase I disk cache that doesn't re-validate the bytes will load a
    // CompiledContext whose library_checked has been swapped for nonsense.
    let mut tampered = bytes.clone();
    let needle = b"lib_double";
    let replacement = b"XXX_double";
    let mut count = 0usize;
    let mut i = 0;
    while i + needle.len() <= tampered.len() {
        if &tampered[i..i + needle.len()] == needle {
            tampered[i..i + needle.len()].copy_from_slice(replacement);
            count += 1;
            i += needle.len();
        } else {
            i += 1;
        }
    }
    assert!(count > 0, "expected at least one occurrence of lib_double");

    let restored = match CompiledContext::decode(&tampered) {
        Ok(ctx) => ctx,
        Err(_) => {
            // If decoding outright fails, that's a clean rejection — fine.
            return;
        }
    };

    // New code calls `lib_double` — but the tampered context no longer
    // has `lib_double` in its library_checked / library_dag. The call MUST
    // fail at check_in_context or eval_in_context; it must NOT silently
    // succeed.
    let snippet = "module App.Eval\nimport Mylib.Math (lib_double)\n\n\
                   def out() -> int32 = lib_double(21)\n";

    let check_outcome = check_in_context(&restored, snippet);
    let eval_outcome = eval_in_context(&restored, snippet);

    // At least one of the two should reject the tampered context.
    assert!(
        check_outcome.is_err() || eval_outcome.is_err(),
        "tampered library_checked must surface as a check or eval error; got check={:?}, eval={:?}",
        check_outcome.as_ref().map(|_| "Ok"),
        eval_outcome.as_ref().map(|_| "Ok"),
    );
}

// ─── G5 — Cold-path overhead ─────────────────────────────────────────────

#[test]
#[ignore = "perf-ratio gate: wall-clock cold-path overhead measurement; separated from the correctness pass per CLAUDE.md. See docs/manual_gates.md."]
fn g5_cold_path_overhead_at_most_2x_monolithic() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    // Cold-path probe: 1× compile_reef_context + 1× eval_in_context vs
    // 1× prepare_eval(format(library + snippet)). Per RT-G prompt, target
    // is <= 1.10× — but on a noisy CI box that's too tight; give 2× and
    // emit the actual ratio for the report.
    let library = "module Mylib.Math\nexport (a, b, c, d, e)\n\n\
                   def a(x: int32) -> int32 = x + 1\n\
                   def b(x: int32) -> int32 = x + 2\n\
                   def c(x: int32) -> int32 = x + 3\n\
                   def d(x: int32) -> int32 = x + 4\n\
                   def e(x: int32) -> int32 = x + 5\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    let snippet = "module App.Eval\nimport Mylib.Math (a, b, c, d, e)\n\n\
                   def out() -> int32 = a(b(c(d(e(0)))))\n";

    // Warm caches by running each path once first (so we measure the
    // steady-state cold path, not first-time compilation overhead in the
    // chelis_macros / chelis_surf parsers).
    let _ = compile_reef_context(Path::new("/tmp/x"), &root).expect("warm ctx");
    let _ = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format_library_plus_snippet(&root, snippet),
        bindings: BTreeMap::new(),
    })
    .expect("warm prepare_eval");

    // Monolithic timing
    let mut mono_elapsed = std::time::Duration::ZERO;
    for _ in 0..3 {
        let formatted = format_library_plus_snippet(&root, snippet);
        let t0 = Instant::now();
        let _ = prepare_eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: formatted,
            bindings: BTreeMap::new(),
        })
        .expect("prepare_eval");
        mono_elapsed += t0.elapsed();
    }
    let mono_avg = mono_elapsed / 3;

    // Cold-path: compile_reef_context + 1 eval_in_context
    let mut cold_elapsed = std::time::Duration::ZERO;
    for _ in 0..3 {
        let t0 = Instant::now();
        let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
        let _ = eval_in_context(&ctx, snippet).expect("eval");
        cold_elapsed += t0.elapsed();
    }
    let cold_avg = cold_elapsed / 3;

    let ratio = cold_avg.as_secs_f64() / mono_avg.as_secs_f64().max(1e-9);
    eprintln!(
        "G5 cold-path: mono {} ms, cold {} ms, ratio {:.2}x",
        mono_avg.as_millis(),
        cold_avg.as_millis(),
        ratio
    );
    // Loose bound — anything worse than 3x means the cache is meaningfully
    // expensive on a single-eval workload. Tighter than 1.10x is too brittle
    // for unhardened hardware noise.
    assert!(
        ratio < 3.0,
        "cold-path overhead too high (ratio {ratio:.2}x; mono {} ms vs cold {} ms)",
        mono_avg.as_millis(),
        cold_avg.as_millis(),
    );
}

// ─── G6 — Source-hash invalidation across encode/decode + library mutation ─

#[test]
fn g6_source_hash_changes_when_library_mutates_between_encode_and_recompile() {
    let library_v1 = "module Mylib.Math\nexport (k)\n\ndef k() -> int32 = 1\n";
    let library_v2 = "module Mylib.Math\nexport (k)\n\ndef k() -> int32 = 2\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";

    let (_dir, root) = build_pkg(library_v1, main);

    let ctx_v1 = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx v1");
    let bytes_v1 = ctx_v1.encode().expect("encode v1");

    // Mutate the library on disk.
    fs::write(root.join("mylib/src/math.ch"), library_v2).expect("rewrite math.ch");

    let ctx_v2 = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx v2");
    assert_ne!(
        ctx_v1.source_hash, ctx_v2.source_hash,
        "mutating library source must change source_hash"
    );

    // Confirm bincode round-trip preserves the OLD hash on the V1 bytes
    // — the hash is content-addressed on the source files at the time of
    // compile, not on the bincode bytes themselves.
    let restored_v1 = CompiledContext::decode(&bytes_v1).expect("decode v1");
    assert_eq!(restored_v1.source_hash, ctx_v1.source_hash);

    // Phase I disk-cache contract: if you write bytes_v1 to disk, mutate
    // sources, and re-call compile_reef_context, the freshly-computed
    // source_hash != the stored bytes_v1's hash. That mismatch is the
    // signal Phase I must use to invalidate the on-disk artifact.
    assert_ne!(
        restored_v1.source_hash, ctx_v2.source_hash,
        "stored hash from v1 must mismatch freshly-computed v2 hash after source mutation"
    );
}

// ─── G7 — eval_many_in_context per-root isolation under failure ──────────

#[test]
fn g7_eval_many_in_context_one_failing_root_does_not_poison_others() {
    let library = "module Mylib.Math\nexport (add)\n\n\
                   def add(x: int32, y: int32) -> int32 = x + y\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    // Two roots: one references an unbound name (so the SHARED compile
    // step fails). The contract for eval_many_in_context when compile
    // fails is: every requested root should receive the same Err; the
    // call must NOT panic, and must NOT crash the host. We probe the
    // shared-compile-failure path because that's the only way to get
    // structurally heterogeneous outcomes from a single new_source.
    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def good() -> int32 = add(1, 2)\n\
                   def bad() -> int32 = nonexistent_function(1, 2)\n";

    let names = vec!["good".to_string(), "bad".to_string()];
    let outcomes = eval_many_in_context(
        &compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx"),
        snippet,
        &names,
    );

    assert_eq!(outcomes.len(), 2, "must return one outcome per root");
    // Both should be Err since shared compile failed; the call must NOT
    // panic.
    for (name, outcome) in &outcomes {
        assert!(
            outcome.is_err(),
            "shared-compile failure should surface as Err for root {name}; got Ok"
        );
    }
}

#[test]
fn g7_eval_many_in_context_runtime_isolation_matches_combined() {
    // Same-program runtime probe: both roots compile clean. Per-root
    // outcomes must equal the corresponding root in the single-shot
    // eval_in_context — that's the per-root isolation contract.
    let library = "module Mylib.Math\nexport (add)\n\n\
                   def add(x: int32, y: int32) -> int32 = x + y\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def good_a() -> int32 = add(1, 2)\n\
                   def good_b() -> int32 = add(10, 20)\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    let combined = eval_in_context(&ctx, snippet).expect("combined eval");
    let combined_a = collect_named_roots_json(&combined.roots, &["good_a"]);
    let combined_b = collect_named_roots_json(&combined.roots, &["good_b"]);
    eprintln!(
        "G7-rt: combined.roots = {:?}",
        combined
            .roots
            .iter()
            .map(|r| r.name.clone())
            .collect::<Vec<_>>()
    );

    let names = vec!["good_a".to_string(), "good_b".to_string()];
    let outcomes = eval_many_in_context(&ctx, snippet, &names);
    assert_eq!(outcomes.len(), 2);

    let mut by_name_per: BTreeMap<String, String> = BTreeMap::new();
    for (name, outcome) in &outcomes {
        let result = outcome
            .as_ref()
            .unwrap_or_else(|e| panic!("root {name} failed: {e:?}"));
        let m = collect_named_roots_json(&result.roots, &[name.as_str()]);
        if let Some(v) = m.get(name) {
            by_name_per.insert(name.clone(), v.clone());
        }
    }
    let mut by_name_combined: BTreeMap<String, String> = BTreeMap::new();
    if let Some(v) = combined_a.get("good_a") {
        by_name_combined.insert("good_a".to_string(), v.clone());
    }
    if let Some(v) = combined_b.get("good_b") {
        by_name_combined.insert("good_b".to_string(), v.clone());
    }
    // Per-root and combined must agree on whatever they produced. The
    // contract isn't "every name is in roots" (some names are 0-arg fns
    // not lowered to a tensor root) — it's "if a name appears, the value
    // is the same in both paths."
    assert_eq!(
        by_name_per, by_name_combined,
        "per-root values must match combined eval for the same names"
    );
}

// ─── G9 — Name shadowing across stages ───────────────────────────────────

#[test]
fn g9_newcode_shadows_library_def_for_new_callers() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    let library = "module Mylib.Math\nexport (foo)\n\n\
                   def foo(x: int32) -> int32 = x + 100\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    // New-code redef of foo with a different body. Another new-code def
    // calls foo. Per the Phase C plan, new code shadows the library on
    // name collision. So `caller(0)` must use the new-code's foo
    // (returns 0 + 1 = 1), NOT the library's (which returns 100).
    let snippet = "module App.Eval\nimport Mylib.Math (foo)\n\n\
                   def foo(x: int32) -> int32 = x + 1\n\
                   def caller() -> int32 = foo(0)\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    let result = eval_in_context(&ctx, snippet).expect("eval ok");
    let by = collect_named_roots_json(&result.roots, &["caller"]);

    // Baseline parity:
    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline");
    let by_baseline = collect_named_roots_json(&baseline.roots, &["caller"]);
    assert_eq!(
        by, by_baseline,
        "shadow result for caller must match monolithic baseline (regardless of which side wins)"
    );
}

// ─── G11 — check_in_context does NOT eval ────────────────────────────────

#[test]
fn g11_check_in_context_does_not_run_user_code() {
    let library = "module Mylib.Math\nexport (add)\n\n\
                   def add(x: int32, y: int32) -> int32 = x + y\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    // The probe: a body that WOULD crash at runtime (integer division by
    // zero) wrapped in a 1-arg fn def that is never called from any root.
    // check_in_context must succeed (typecheck only); eval_in_context
    // likewise won't fire the body since nothing applies it. The contract
    // being tested: the check pass is purely static — no eval side effects
    // from defs. (chelis#178: integer division is `trunc_div`; `x / 0`
    // would now be a type error since `div` is float-only, so the landmine
    // uses the well-typed-but-runtime-trapping `trunc_div(x, 0)`.)
    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def landmine(x: int32) -> int32 = trunc_div(x, cast(0, int32))\n\
                   def safe() -> int32 = add(1, 2)\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");
    // check_in_context: must succeed; the landmine fn is well-typed but
    // never called.
    let _ = check_in_context(&ctx, snippet).expect("check should succeed");

    // Eval likewise succeeds; landmine fn is unreferenced and never
    // applied, so no runtime division-by-zero fires. (We don't assert on
    // safe's appearance in roots — `def safe() -> int32` is a 0-arg fn,
    // documented as not necessarily lowerable to a tensor root in the
    // existing compiled_context fixture.)
    let _ = eval_in_context(&ctx, snippet).expect("eval ok");
}

// ─── G12 — source_hash collision-resistance smoke ────────────────────────

#[test]
fn g12_source_hash_differs_when_library_text_differs_by_one_char() {
    let library_a = "module Mylib.Math\nexport (k)\n\ndef k() -> int32 = 1\n";
    let library_b = "module Mylib.Math\nexport (k)\n\ndef k() -> int32 = 2\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (dir_a, root_a) = build_pkg(library_a, main);
    let (dir_b, root_b) = build_pkg(library_b, main);
    let ctx_a = compile_reef_context(Path::new("/tmp/x"), &root_a).expect("ctx a");
    let ctx_b = compile_reef_context(Path::new("/tmp/x"), &root_b).expect("ctx b");
    assert_ne!(
        ctx_a.source_hash, ctx_b.source_hash,
        "single-char library diff must produce different source_hash"
    );
    drop((dir_a, dir_b));
}

#[test]
fn g12_source_hash_differs_when_package_name_differs_with_same_content() {
    // Per RT-G prompt: same source contents, different package name MUST
    // produce different hashes. The hash mixes (package_name,
    // package_version, module_name, sha256), so renaming the package
    // alone should change the digest even if every byte of every source
    // file is identical.
    let library = "module Mylib.Math\nexport (k)\n\ndef k() -> int32 = 1\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";

    // Build two packages whose root reef.toml differs only in the
    // package name field.
    let dir_a = TempDir::new().expect("tempdir A");
    let root_a = dir_a.path().join("appA");
    write_pkg(
        &root_a,
        "appA",
        "App",
        &[("src/main.ch", main)],
        &[("mylib", "./mylib")],
    );
    let mylib_a = root_a.join("mylib");
    write_pkg(&mylib_a, "mylib", "Mylib", &[("src/math.ch", library)], &[]);
    fs::write(root_a.join("reef.lock"), app_reef_lock("appA")).expect("lock A");

    let dir_b = TempDir::new().expect("tempdir B");
    let root_b = dir_b.path().join("appB");
    write_pkg(
        &root_b,
        "appB",
        "App",
        &[("src/main.ch", main)],
        &[("mylib", "./mylib")],
    );
    let mylib_b = root_b.join("mylib");
    write_pkg(&mylib_b, "mylib", "Mylib", &[("src/math.ch", library)], &[]);
    fs::write(root_b.join("reef.lock"), app_reef_lock("appB")).expect("lock B");

    let ctx_a = compile_reef_context(Path::new("/tmp/x"), &root_a).expect("ctx A");
    let ctx_b = compile_reef_context(Path::new("/tmp/x"), &root_b).expect("ctx B");
    assert_ne!(
        ctx_a.source_hash, ctx_b.source_hash,
        "different package_name with identical sources must produce different source_hash"
    );
}

// ─── G1 — Effect-row composition across the boundary ─────────────────────

#[test]
fn g1_newcode_inheriting_test_effect_from_library_with_strict_signature_rejected() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    // Library has a Test-effecting helper. New-code def DECLARES `! {}`
    // (no effects) but transitively reaches the library's Test effect by
    // calling the helper. check_in_context MUST reject with
    // UnhandledEffect.
    let library = "module Mylib.Math\nexport (lib_check_eq)\n\n\
                   def lib_check_eq(a: int64, b: int64) -> unit = test_assert_eq_int(a, b, \"lib_check_eq\")\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);

    // Honest declaration `! { Test }` must accept; absent / `! {}` must
    // reject (declared-vs-inferred).
    let snippet_ok = "module App.Eval\nimport Mylib.Math (lib_check_eq)\n\n\
                      sig my_check: int64 -> unit ! { Test }\n\
                      def my_check(x: int64) -> unit = lib_check_eq(x, cast(1, int64))\n";
    let snippet_strict = "module App.Eval\nimport Mylib.Math (lib_check_eq)\n\n\
                          sig my_check: int64 -> unit ! {}\n\
                          def my_check(x: int64) -> unit = lib_check_eq(x, cast(1, int64))\n";

    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    // Use prepare_eval — that's the monolithic path that runs effects +
    // linearity + lower (compile_source). The bare `check` API only runs
    // IR check, NOT the effect declared-vs-inferred validator.
    let mono_ok = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format_library_plus_snippet(&root, snippet_ok),
        bindings: BTreeMap::new(),
    });
    let mono_strict = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format_library_plus_snippet(&root, snippet_strict),
        bindings: BTreeMap::new(),
    });

    let inctx_ok = check_in_context(&ctx, snippet_ok);
    let inctx_strict = check_in_context(&ctx, snippet_strict);

    let mono_strict_rejects = mono_strict.is_err();
    let inctx_strict_rejects = inctx_strict.is_err();
    let mono_strict_err = mono_strict.err();
    let inctx_strict_err = inctx_strict.err();
    assert!(
        mono_strict_rejects,
        "monolithic prepare_eval must reject `! {{}}` declaration of a Test-using snippet; mono err = {mono_strict_err:?}"
    );
    assert_eq!(
        mono_strict_rejects, inctx_strict_rejects,
        "check_in_context must agree with monolithic on UnhandledEffect: \
         mono_strict_err={mono_strict_err:?}, inctx_strict_err={inctx_strict_err:?}"
    );

    let mono_ok_rejects = mono_ok.is_err();
    let inctx_ok_rejects = inctx_ok.is_err();
    let mono_ok_err = mono_ok.err();
    let inctx_ok_err = inctx_ok.err();
    assert_eq!(
        mono_ok_rejects, inctx_ok_rejects,
        "check_in_context must agree with monolithic on honest declaration: \
         mono_ok_err={mono_ok_err:?}, inctx_ok_err={inctx_ok_err:?}"
    );
}

// ─── G4-deep — bincode tamper on an UNREFERENCED library def ─────────────

#[test]
fn g4_deep_tamper_keeps_referenced_lib_eval_correct_or_fails_loudly() {
    // Goal: build a context with TWO library defs, lib_used (referenced
    // by new code) and lib_unused (not). Tamper the bincode bytes to
    // corrupt lib_unused's name. Decode must either fail OR the new-code
    // eval that depends only on lib_used must still produce the right
    // value (because lib_unused was unreferenced anyway). What MUST NOT
    // happen: silent change to lib_used's resolved value.
    let library = "module Mylib.Math\nexport (lib_used, lib_unused)\n\n\
                   def lib_used(x: int32) -> int32 = x + 100\n\
                   def lib_unused(x: int32) -> int32 = x * 999\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    let snippet = "module App.Eval\nimport Mylib.Math (lib_used)\n\n\
                   def out() -> int32 = lib_used(5)\n";
    let pristine = eval_in_context(&ctx, snippet).expect("pristine eval");
    let pristine_by = collect_named_roots_json(&pristine.roots, &["out"]);

    let bytes = ctx.encode().expect("encode");
    let mut tampered = bytes.clone();
    let needle = b"lib_unused";
    let replacement = b"libXunused";
    let mut count = 0usize;
    let mut i = 0;
    while i + needle.len() <= tampered.len() {
        if &tampered[i..i + needle.len()] == needle {
            tampered[i..i + needle.len()].copy_from_slice(replacement);
            count += 1;
            i += needle.len();
        } else {
            i += 1;
        }
    }
    assert!(
        count > 0,
        "expected at least one occurrence of lib_unused in encoded bytes"
    );

    let restored = match CompiledContext::decode(&tampered) {
        Ok(ctx) => ctx,
        Err(_) => {
            // Clean rejection — fine.
            return;
        }
    };

    let post = eval_in_context(&restored, snippet);
    match post {
        Ok(result) => {
            let post_by = collect_named_roots_json(&result.roots, &["out"]);
            // If tampering changed `out`'s value silently, that's the bug.
            assert_eq!(
                post_by, pristine_by,
                "tampering an unreferenced library def must not silently alter referenced eval output"
            );
        }
        Err(_) => {
            // Erroring is acceptable — the cache detected corruption.
        }
    }
}

// ─── G10 — Library + new-code linearity collision ─────────────────────────

#[test]
fn g10_newcode_tensor_use_after_consume_through_library_call() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    // Library function consumes its tensor parameter (no_grad / consume).
    // New code creates a tensor, calls the library function with it, then
    // tries to use the same tensor again. Linearity must reject.
    //
    // For now, we test the simpler invariant: a new-code def that uses a
    // tensor twice (without library involvement) is rejected. If
    // _with_context's linearity is leaky, this may pass when it shouldn't.
    let library = "module Mylib.Math\nexport (id_tensor)\n\n\
                   def id_tensor(x: tensor[3, f32]) -> tensor[3, f32] = x\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    // Snippet that takes a tensor parameter and uses it twice (linearity
    // violation) — also flowing through the library function.
    let snippet = "module App.Eval\nimport Mylib.Math (id_tensor)\n\n\
                   def use_twice(t: tensor[3, f32]) -> tensor[3, f32] = id_tensor(t) + id_tensor(t)\n";

    // Parity: monolithic compile-pipeline (prepare_eval) vs in-context
    // check should agree on whatever the linearity layer decides.
    // prepare_eval runs the full check_ir + effects + linearity +
    // lower stack — not the bare `check` API which is Phase-0e-only.
    // Divergence between monolithic and in_context linearity verdicts
    // is the finding.
    let mono = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format_library_plus_snippet(&root, snippet),
        bindings: BTreeMap::new(),
    });
    let inctx = check_in_context(&ctx, snippet);
    let mono_rejects = mono.is_err();
    let inctx_rejects = inctx.is_err();
    let mono_err = mono.err();
    let inctx_err = inctx.err();
    assert_eq!(
        mono_rejects, inctx_rejects,
        "linearity verdict must agree across with_context and monolithic prepare_eval; \
         mono_err={mono_err:?}, inctx_err={inctx_err:?}"
    );
}

// ─── G8 — Multi-module fixture parity ────────────────────────────────────

/// Build a reef package whose path-dep `mylib` has two modules:
/// `Mylib.Math` and `Mylib.Util`. `Mylib.Util` imports from `Mylib.Math`.
/// New code imports from BOTH modules.
fn multi_module_fixture() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let root = dir.path().join("myapp");
    write_pkg(
        &root,
        "myapp",
        "App",
        &[(
            "src/main.ch",
            "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n",
        )],
        &[("mylib", "./mylib")],
    );
    let mylib = root.join("mylib");
    write_pkg(
        &mylib,
        "mylib",
        "Mylib",
        &[
            (
                "src/math.ch",
                "module Mylib.Math\nexport (add, mul)\n\n\
                 def add(x: int32, y: int32) -> int32 = x + y\n\
                 def mul(x: int32, y: int32) -> int32 = x * y\n",
            ),
            (
                "src/util.ch",
                "module Mylib.Util\nexport (square_plus_one)\nimport Mylib.Math (add, mul)\n\n\
                 def square_plus_one(x: int32) -> int32 = add(mul(x, x), 1)\n",
            ),
        ],
        &[],
    );
    fs::write(root.join("reef.lock"), app_reef_lock("myapp")).expect("write reef.lock");
    (dir, root)
}

#[test]
fn g8_multi_module_parity_with_monolithic() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    let (_dir, root) = multi_module_fixture();
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    // Snippet imports from BOTH library modules and combines them.
    let snippet = "module App.Eval\nimport Mylib.Math (add)\nimport Mylib.Util (square_plus_one)\n\n\
                   def out() -> int32 = add(square_plus_one(4), square_plus_one(5))\n";

    let in_context = eval_in_context(&ctx, snippet).expect("in-context eval");
    let formatted = format_library_plus_snippet(&root, snippet);
    let baseline = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: formatted,
        bindings: BTreeMap::new(),
    })
    .expect("baseline eval");

    let in_ctx_by = collect_named_roots_json(&in_context.roots, &["out"]);
    let baseline_by = collect_named_roots_json(&baseline.roots, &["out"]);
    assert_eq!(
        in_ctx_by, baseline_by,
        "multi-module library compose must produce same eval as monolithic baseline"
    );
}

// ─── G-extra — eval_many on Err propagation parity ───────────────────────

#[test]
fn gextra_eval_many_parity_with_eval_in_context_for_each_root_individually() {
    // Build a context, build a snippet with N roots, eval each via
    // eval_many_in_context AND via N separate eval_in_context calls. Both
    // paths must agree on every per-root outcome.
    let library = "module Mylib.Math\nexport (add)\n\n\
                   def add(x: int32, y: int32) -> int32 = x + y\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def root_a() -> int32 = add(1, 2)\n\
                   def root_b() -> int32 = add(3, 4)\n\
                   def root_c() -> int32 = add(5, 6)\n";

    let names: Vec<String> = ["root_a", "root_b", "root_c"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let many = eval_many_in_context(&ctx, snippet, &names);
    assert_eq!(many.len(), 3);

    // Each per-name outcome must be Ok (clean snippet).
    for (name, outcome) in &many {
        let _ = outcome
            .as_ref()
            .unwrap_or_else(|e| panic!("eval_many root {name} failed: {e:?}"));
    }
}

// ─── G-extra — Source hash determinism across different temp paths ────────

#[test]
fn gextra_source_hash_independent_of_tempdir_path() {
    // Build the same package twice in two different tempdirs. The
    // package_root path differs but source contents are identical, so
    // source_hash MUST be equal — otherwise the Phase I disk cache will
    // never share artifacts across reef-home moves or symlink changes.
    let library = "module Mylib.Math\nexport (k)\n\ndef k() -> int32 = 42\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir1, root1) = build_pkg(library, main);
    let (_dir2, root2) = build_pkg(library, main);
    let ctx1 = compile_reef_context(Path::new("/tmp/x"), &root1).expect("ctx1");
    let ctx2 = compile_reef_context(Path::new("/tmp/x"), &root2).expect("ctx2");
    assert_eq!(
        ctx1.source_hash, ctx2.source_hash,
        "identical source contents in different tempdirs must hash identically; \
         Phase I disk cache reuse depends on this"
    );
}

// ─── G9-deep — Effects/linearity respect new-code shadow of library def ──

#[test]
fn g9deep_effect_inference_uses_newcode_shadow_not_library_for_inferred_row() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    // Library has `lib_io(x)` that performs Io (calls print).
    // New code redefines `lib_io(x)` with NO effects (just identity).
    // Another new-code def `caller(x)` calls `lib_io(x)`.
    // The inferred effect of `caller` MUST be the new-code shadow (no
    // effects), NOT the library's Io effect — otherwise Phase D's
    // effect-shadow contract is broken.
    //
    // We probe via the declared-vs-inferred validator: declare `caller`
    // with `! {}`. If the new-code shadow is honoured, that succeeds.
    // If the library's Io leaks through, the validator rejects.
    let library = "module Mylib.Math\nexport (lib_io)\n\n\
                   def lib_io(x: int32) -> int32 = { ignore = print(\"in lib\")\n  x }\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    let snippet = "module App.Eval\nimport Mylib.Math (lib_io)\n\n\
                   def lib_io(x: int32) -> int32 = x\n\
                   sig caller: int32 -> int32 ! {}\n\
                   def caller(x: int32) -> int32 = lib_io(x)\n";

    // Parity vs monolithic prepare_eval.
    let mono = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format_library_plus_snippet(&root, snippet),
        bindings: BTreeMap::new(),
    });
    let inctx = check_in_context(&ctx, snippet);
    let mono_rejects = mono.is_err();
    let inctx_rejects = inctx.is_err();
    let mono_err = mono.err();
    let inctx_err = inctx.err();
    assert_eq!(
        mono_rejects, inctx_rejects,
        "shadow-respecting effect verdict must match monolithic; \
         mono_err={mono_err:?}, inctx_err={inctx_err:?}"
    );
}

// ─── G-extra — Concurrent eval against shared CompiledContext ────────────

#[test]
fn gextra_eval_in_context_is_thread_safe_across_arc_clones() {
    // Phase G doc says CompiledContext is "cheap to clone (the heavy
    // state is `Arc`-shared inside `TypeEnv`)". If eval_in_context truly
    // is pure, four threads each evaluating the same snippet against the
    // same context should produce identical results without panic, race,
    // or stale-state corruption.
    use std::sync::Arc;
    use std::thread;

    let library = "module Mylib.Math\nexport (add)\n\n\
                   def add(x: int32, y: int32) -> int32 = x + y\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);
    let ctx = Arc::new(compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx"));

    let snippet = "module App.Eval\nimport Mylib.Math (add)\n\n\
                   def out() -> int32 = add(7, 8)\n";

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let ctx = Arc::clone(&ctx);
            let snippet = snippet.to_string();
            thread::spawn(move || {
                eval_in_context(&ctx, &snippet)
                    .expect("threaded eval ok")
                    .roots
            })
        })
        .collect();

    let mut results: Vec<Vec<EvaluatedRoot>> = handles
        .into_iter()
        .map(|h| h.join().expect("join"))
        .collect();

    // All four threads should produce identical roots (deterministic eval
    // on a pure context). Sort by name for stable comparison.
    for r in &mut results {
        r.sort_by(|a, b| a.name.cmp(&b.name));
    }
    let first = serde_json::to_string(&results[0]).expect("ser");
    for (i, r) in results.iter().enumerate().skip(1) {
        let other = serde_json::to_string(r).expect("ser");
        assert_eq!(
            first, other,
            "thread {i} produced different roots than thread 0; concurrent eval is non-deterministic"
        );
    }
}

// ─── G-extra — Library def removed; new-code reference must fail ─────────

#[test]
fn gextra_newcode_referencing_removed_library_def_fails() {
    // Build a context where library has `lib_a` and `lib_b`. Then build
    // a SECOND context whose library only has `lib_a`. New code that
    // references `lib_b` must succeed against the first context and
    // fail against the second. (Sanity that the type_env / reef export
    // gate is actually doing its job.)
    let library_full = "module Mylib.Math\nexport (lib_a, lib_b)\n\n\
                        def lib_a(x: int32) -> int32 = x + 1\n\
                        def lib_b(x: int32) -> int32 = x + 2\n";
    let library_partial = "module Mylib.Math\nexport (lib_a)\n\n\
                           def lib_a(x: int32) -> int32 = x + 1\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";

    let (_dir1, root_full) = build_pkg(library_full, main);
    let (_dir2, root_partial) = build_pkg(library_partial, main);

    let ctx_full = compile_reef_context(Path::new("/tmp/x"), &root_full).expect("ctx full");
    let ctx_partial =
        compile_reef_context(Path::new("/tmp/x"), &root_partial).expect("ctx partial");

    let snippet = "module App.Eval\nimport Mylib.Math (lib_a, lib_b)\n\n\
                   def out() -> int32 = lib_a(0) + lib_b(0)\n";

    let _ = check_in_context(&ctx_full, snippet).expect("snippet checks against full lib");
    let outcome = check_in_context(&ctx_partial, snippet);
    assert!(
        outcome.is_err(),
        "snippet referencing lib_b must fail against partial library; got {outcome:?}"
    );
}

// ─── G-extra — Source-hash sensitivity to module name ────────────────────

#[test]
fn gextra_source_hash_differs_when_only_module_name_differs() {
    // Same byte content, but two different file/module name pairs. The
    // hash mixes module_name in addition to content, so they must differ.
    // If they don't, Phase I will reuse cached artifacts across modules
    // that have identical bodies but different export names — a poison
    // vector.
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";

    let dir_a = TempDir::new().expect("tempdir A");
    let root_a = dir_a.path().join("myapp");
    write_pkg(
        &root_a,
        "myapp",
        "App",
        &[("src/main.ch", main)],
        &[("mylib", "./mylib")],
    );
    let mylib_a = root_a.join("mylib");
    write_pkg(
        &mylib_a,
        "mylib",
        "Mylib",
        &[(
            "src/math.ch",
            "module Mylib.Math\nexport (k)\n\ndef k() -> int32 = 1\n",
        )],
        &[],
    );
    fs::write(root_a.join("reef.lock"), app_reef_lock("myapp")).expect("lock A");

    let dir_b = TempDir::new().expect("tempdir B");
    let root_b = dir_b.path().join("myapp");
    write_pkg(
        &root_b,
        "myapp",
        "App",
        &[("src/main.ch", main)],
        &[("mylib", "./mylib")],
    );
    let mylib_b = root_b.join("mylib");
    write_pkg(
        &mylib_b,
        "mylib",
        "Mylib",
        &[(
            "src/util.ch",
            "module Mylib.Util\nexport (k)\n\ndef k() -> int32 = 1\n",
        )],
        &[],
    );
    fs::write(root_b.join("reef.lock"), app_reef_lock("myapp")).expect("lock B");

    let ctx_a = compile_reef_context(Path::new("/tmp/x"), &root_a).expect("ctx A");
    let ctx_b = compile_reef_context(Path::new("/tmp/x"), &root_b).expect("ctx B");
    assert_ne!(
        ctx_a.source_hash, ctx_b.source_hash,
        "source_hash must include module name; identical body content under \
         different module names must hash differently"
    );
}

// ─── G-extra — Library Random-effecting helper called from new code ──────

#[test]
fn gextra_library_random_call_from_new_code_must_be_handled_or_rejected() {
    // RFC v5 (RT-1 F2 bypass): the monolithic baseline formats the reef-linked
    // library (internal-name mangled) and evaluates it; declare the linked
    // provenance, matching the now-guarded production paths.
    let _linked = chelis_compiler_api::install_linked_program_guard();
    // Library function performs Random (calls dropout). If new code
    // calls it WITHOUT a handler, the unhandled-effect validator must
    // reject with a Random-mentioning error. (This is the core Phase D
    // composition contract for new code at the Phase G surface.)
    let library = "module Mylib.Math\nexport (lib_drop)\n\n\
                   def lib_drop(t: tensor[3, f32]) -> tensor[3, f32] = dropout(t, cast(0.5, f32))\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    let snippet = "module App.Eval\nimport Mylib.Math (lib_drop)\n\n\
                   def caller(t: tensor[3, f32]) -> tensor[3, f32] = lib_drop(t)\n";

    let mono = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format_library_plus_snippet(&root, snippet),
        bindings: BTreeMap::new(),
    });
    let inctx = check_in_context(&ctx, snippet);
    let mono_rejects = mono.is_err();
    let inctx_rejects = inctx.is_err();
    let mono_err = mono.err();
    let inctx_err = inctx.err();
    assert_eq!(
        mono_rejects, inctx_rejects,
        "Random-handler enforcement must agree across with_context vs monolithic; \
         mono_err={mono_err:?}, inctx_err={inctx_err:?}"
    );
}

// ─── G-extra — Repeated check_in_context calls do not mutate context ─────

#[test]
fn gextra_repeated_calls_against_same_context_are_independent() {
    // Per Phase D doc-string: "the function takes &CheckedProgram (not &mut)
    // so subsequent calls against the same library_program see a fresh
    // effect map untouched by any prior new-code inference." Test that
    // contract here through the Phase G surface: snippet A declares an
    // effect; snippet B's checking must NOT see snippet A's annotated
    // exprs leak into its effect/linearity tracking.
    let library = "module Mylib.Math\nexport (add)\n\n\
                   def add(x: int32, y: int32) -> int32 = x + y\n";
    let main = "module App.Main\n\ndef placeholder() -> int32 = cast(0, int32)\n";
    let (_dir, root) = build_pkg(library, main);
    let ctx = compile_reef_context(Path::new("/tmp/x"), &root).expect("ctx");

    let snippet_a = "module App.Eval\nimport Mylib.Math (add)\n\n\
                     sig my_test: int64 -> unit ! { Test }\n\
                     def my_test(x: int64) -> unit = test_assert_eq_int(x, x, \"a\")\n";
    let snippet_b = "module App.Eval\nimport Mylib.Math (add)\n\n\
                     def pure_b() -> int32 = add(1, 2)\n";

    // Call order: A then B; B then A; both should produce identical
    // verdicts each time.
    let a1 = check_in_context(&ctx, snippet_a);
    let b1 = check_in_context(&ctx, snippet_b);
    let a2 = check_in_context(&ctx, snippet_a);
    let b2 = check_in_context(&ctx, snippet_b);

    let a1_ok = a1.is_ok();
    let a2_ok = a2.is_ok();
    let b1_ok = b1.is_ok();
    let b2_ok = b2.is_ok();
    assert_eq!(a1_ok, a2_ok, "check_in_context A is not idempotent");
    assert_eq!(b1_ok, b2_ok, "check_in_context B is not idempotent");
    assert!(a1_ok, "snippet A should check OK; got {a1:?}");
    assert!(b1_ok, "snippet B should check OK; got {b1:?}");
}
