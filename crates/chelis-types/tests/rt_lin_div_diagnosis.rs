//! Phase G' linearity divergence root-cause diagnosis.
//!
//! Probes the difference between Path A (direct cache, `compile_reef_context`)
//! and Path B (legacy format-reparse, `compile_with_reef_graph + format_program +
//! prepare_eval`) for the chelis-std `test_linspace_endpoints` shape.
//!
//! Output: pretty-printed annotated_exprs from each path. Manual diff inline.
//! `--nocapture` to dump the diff to stdout.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_program;
use chelis_surf::parser::parse_str;
use chelis_types::{
    build_type_env_from_library, check_ir_program, check_ir_with_context, check_linearity,
    check_linearity_with_context,
};

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

const LIBRARY_SRC: &str = r#"
def assert_shape(t: &tensor[4, f32], expected_n: int64, label: string) -> int32 = {
  actual_n = cast(shape(t, cast(0, int32)), int64)
  rank(t)
}
def assert_close_tensor(actual: &tensor[4, f32], expected: &tensor[4, f32], tol: f32, label: string) -> int32 = rank(actual)
def linspace(start: f32, stop: f32, count: int32) -> tensor[4, f32] = to_tensor([start, stop, start, stop])
"#;

const NEW_SRC: &str = r#"
def test_linspace_endpoints(expected: tensor[4, f32]) -> int32 = {
  actual = linspace(cast(0.0, f32), cast(1.0, f32), cast(3, int32))
  _ = assert_shape(actual, cast(3, int64), "len")
  close = assert_close_tensor(actual, expected, cast(0.000001, f32), "vals")
  _ = drop(actual)
  _ = drop(expected)
  close
}
"#;

fn pretty_print_program(label: &str, exprs: &[chelis_deep::Expr]) -> String {
    let mut out = String::new();
    out.push_str(&format!("=== {label} ===\n"));
    for (i, e) in exprs.iter().enumerate() {
        out.push_str(&format!("--- expr[{i}] ---\n"));
        out.push_str(&print_canonical(std::slice::from_ref(e)));
        out.push('\n');
    }
    out
}

#[test]
fn diagnosis_path_a_vs_path_b_annotated_exprs() {
    // ----- Path A: direct cache -----
    let library_deep = surf_to_deep(LIBRARY_SRC);
    let library_checked_a = check_ir_program(&library_deep)
        .unwrap_or_else(|e| panic!("Path A library check_ir failed: {:?}", e.errors));
    let library_checked_a = check_linearity(&library_checked_a)
        .unwrap_or_else(|e| panic!("Path A library check_linearity failed: {:?}", e));

    let ctx_a = build_type_env_from_library(&library_deep)
        .unwrap_or_else(|e| panic!("Path A type-env build failed: {:?}", e));
    let new_deep_a = surf_to_deep(NEW_SRC);
    let new_checked_a = check_ir_with_context(&ctx_a, &new_deep_a)
        .unwrap_or_else(|e| panic!("Path A new-code check_ir failed: {:?}", e.errors));

    let lin_a = check_linearity_with_context(&library_checked_a, &new_checked_a);
    println!(
        "PATH A (cache) lin verdict: {}",
        match &lin_a {
            Ok(_) => "OK".to_string(),
            Err(errs) => format!("ERR: {} error(s): {:?}", errs.len(), errs),
        }
    );

    // ----- Path B: format-reparse -----
    // Build the linked source the way `compile_with_reef_graph + format_program`
    // would: prepend library decls (after rewrite/desugar — but we'll mirror by
    // re-formatting the surf source). Simplest: concatenate library + new in
    // raw surf, since neither uses reef-internal-name rewriting.
    let combined_src = format!("{LIBRARY_SRC}\n{NEW_SRC}");
    // Mirror format-reparse: parse surf → format → parse again → desugar +
    // expand → check monolithic. The format_program step strips some
    // surface-level annotation forms; we want to see if it affects the
    // annotated_exprs the linearity checker walks.
    let raw_decls = parse_str(&combined_src).expect("combined parse");
    let formatted = format_program(&raw_decls);
    println!("--- Path B formatted source ---\n{formatted}\n--- end ---");
    let reparsed = parse_str(&formatted).expect("reparse formatted");
    let deep_b = chelis_macros::expand_program(
        &desugar_program(&reparsed),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand B")
    .into_exprs();

    let library_checked_b = check_ir_program(&deep_b)
        .unwrap_or_else(|e| panic!("Path B check_ir failed: {:?}", e.errors));
    let lin_b = check_linearity(&library_checked_b);
    println!(
        "PATH B (format-reparse) lin verdict: {}",
        match &lin_b {
            Ok(_) => "OK".to_string(),
            Err(errs) => format!("ERR: {} error(s): {:?}", errs.len(), errs),
        }
    );

    // Pretty-print each path's annotated_exprs.
    let pretty_a_lib = pretty_print_program(
        "PATH A library_checked.annotated_exprs",
        library_checked_a.annotated_exprs(),
    );
    let pretty_a_new = pretty_print_program(
        "PATH A new_checked.annotated_exprs",
        new_checked_a.annotated_exprs(),
    );
    let pretty_b_combined = pretty_print_program(
        "PATH B combined_checked.annotated_exprs",
        library_checked_b.annotated_exprs(),
    );

    println!("{pretty_a_lib}");
    println!("{pretty_a_new}");
    println!("{pretty_b_combined}");

    // Also run Path B's MONOLITHIC walker against Path A's annotated_exprs
    // (library + new concatenated) — so we control for the format-reparse
    // step entirely. If A's combined annotated_exprs accept under
    // monolithic check_linearity but reject under check_linearity_with_context,
    // the divergence is in the walker, NOT the AST.
    let mut combined_a_exprs: Vec<chelis_deep::Expr> = library_checked_a.annotated_exprs().to_vec();
    combined_a_exprs.extend(new_checked_a.annotated_exprs().to_vec());
    // We cannot directly construct a CheckedProgram from annotated_exprs, so
    // instead, take the unannotated library_deep + new_deep_a, concat, and run
    // monolithic check + linearity over THAT. This is the closest analogue to
    // Path B without the surf-format round-trip.
    let mut combined_pre_check: Vec<chelis_deep::Expr> = library_deep.clone();
    combined_pre_check.extend(new_deep_a.clone());
    let combined_checked = check_ir_program(&combined_pre_check)
        .unwrap_or_else(|e| panic!("Combined (unformatted) check_ir failed: {:?}", e.errors));
    let lin_combined = check_linearity(&combined_checked);
    println!(
        "COMBINED (no-format) monolithic lin verdict: {}",
        match &lin_combined {
            Ok(_) => "OK".to_string(),
            Err(errs) => format!("ERR: {} error(s): {:?}", errs.len(), errs),
        }
    );

    let pretty_combined = pretty_print_program(
        "COMBINED (no-format) annotated_exprs",
        combined_checked.annotated_exprs(),
    );
    println!("{pretty_combined}");

    // ----- Probe: do prelude ADTs make the difference? -----
    //
    // Hypothesis: `annotate_ir_program` (called from
    // `check_ir_program`) builds with EMPTY ADT registry — no Cons/Nil
    // — so `to_tensor([..])` body of linspace fails to resolve, leaving
    // linspace's return type as `t-var _`. The `_with_context` path with a
    // non-empty `TypeEnv` (built via `build_type_env_from_library`, which
    // starts from `TypeEnv::empty()` whose prelude ADTs ARE registered)
    // does resolve it.
    //
    // Cross-check: run check_ir_with_context with TypeEnv::empty() on
    // the COMBINED program. TypeEnv::empty() has library_def_count == 0,
    // so it falls back to annotate_ir_program — empty ADT registry.
    // Verdict should match Path B / combined_no_format.
    let combined_checked_via_empty_ctx =
        check_ir_with_context(&chelis_types::TypeEnv::empty(), &combined_pre_check)
            .unwrap_or_else(|e| panic!("Combined via empty ctx check_ir failed: {:?}", e.errors));
    let lin_via_empty_ctx = check_linearity(&combined_checked_via_empty_ctx);
    println!(
        "COMBINED via TypeEnv::empty() ctx lin verdict: {}",
        match &lin_via_empty_ctx {
            Ok(_) => "OK".to_string(),
            Err(errs) => format!("ERR: {} error(s): {:?}", errs.len(), errs),
        }
    );

    // Now the kicker: build a TypeEnv from the library, then run the new
    // code through check_ir_with_context with that NON-EMPTY context.
    // (This is exactly Path A's flow.)
    println!(
        "(Path A is the same: built type_env from library, ran new code with non-empty ctx → REJECT.)"
    );

    // Cross-check #2: run check_ir_with_context with a NON-EMPTY
    // type_env (built from library) on the COMBINED program (library
    // duplicated). This isolates whether the `_with_context` annotation
    // branch alone changes the verdict, independent of new-code-only
    // walking.
    let combined_via_nonempty_ctx = check_ir_with_context(&ctx_a, &combined_pre_check)
        .unwrap_or_else(|e| panic!("Combined via non-empty ctx check_ir failed: {:?}", e.errors));
    let lin_via_nonempty_ctx = check_linearity(&combined_via_nonempty_ctx);
    println!(
        "COMBINED via non-empty (lib) ctx lin verdict: {}",
        match &lin_via_nonempty_ctx {
            Ok(_) => "OK".to_string(),
            Err(errs) => format!("ERR: {} error(s): {:?}", errs.len(), errs),
        }
    );

    // Cross-check #3: run check_ir_with_context with a NON-EMPTY
    // type_env (built from library) on JUST the new code. Same as Path A.
    let new_via_nonempty_ctx =
        check_ir_with_context(&ctx_a, &new_deep_a).expect("new via non-empty ctx clean");
    let lin_new_via_nonempty_ctx_mono = check_linearity(&new_via_nonempty_ctx);
    println!(
        "NEW-only via non-empty (lib) ctx + monolithic lin verdict: {}",
        match &lin_new_via_nonempty_ctx_mono {
            Ok(_) => "OK".to_string(),
            Err(errs) => format!("ERR: {} error(s): {:?}", errs.len(), errs),
        }
    );

    println!(
        "agreement summary: A={:?} B={:?} combined_no_format={:?} combined_via_empty_ctx={:?} combined_via_nonempty_ctx={:?} new_only_via_nonempty_ctx_mono={:?}",
        lin_a.is_ok(),
        lin_b.is_ok(),
        lin_combined.is_ok(),
        lin_via_empty_ctx.is_ok(),
        lin_via_nonempty_ctx.is_ok(),
        lin_new_via_nonempty_ctx_mono.is_ok(),
    );

    // Final: which paths agree?
    println!(
        "agreement: A={:?} B={:?} combined_no_format={:?}",
        lin_a.is_ok(),
        lin_b.is_ok(),
        lin_combined.is_ok()
    );
}
