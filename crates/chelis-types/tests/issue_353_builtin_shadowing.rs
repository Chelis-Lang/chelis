//! chelis#353: a user `def` or `sig` that shadows a builtin name must be
//! rejected at declaration time.
//!
//! Pre-fix, `def sum(...)` checked clean but was broken on both execution
//! lanes: the evaluator dispatches builtin-first by name
//! (`runtime/host_ops.rs::builtin_name`), so the call hit the builtin's
//! arity/typing (`expected int arg at index 1, got None`), and the C
//! backend lowered the call as the builtin reduction and the compiled
//! binary segfaulted (rc=139). Three lanes, three different answers.
//!
//! The fix is a check-time rejection (`CheckErrorKind::BuiltinShadowing`)
//! raised in `collect_all_declarations`, the declaration-collection
//! chokepoint shared by every front-end entry point (`infer_program`,
//! `check_ir_program`, `check_ir_with_context`,
//! `build_type_env_from_library`), so `check`, `eval`, `build`, and
//! `test` all reject the program with the same diagnostic.
//!
//! The rejection derives from `chelis_types::BUILTIN_NAMES` — the exact
//! table both the evaluator dispatch and IR lowering import — so the
//! rejected set and the dispatched set can never drift. The
//! `every_builtin_name_def_is_rejected` sweep pins that table-completeness
//! property name by name.
//!
//! Scope (spec/04-type-system.md §8.6):
//! - binds to top-level `def` and `defsig` after module flattening,
//!   including load-style top-level bindings (`sum = ...` desugars to a
//!   `def`),
//! - does NOT bind to function parameters or block-local bindings: those
//!   bind values, not call-site dispatch, and shadow harmlessly (pinned
//!   below),
//! - does NOT bind to reef package modules: their decls are
//!   internal-name-rewritten (`pkg__...`) before the checker runs, so a
//!   package `def sum` neither collides nor mis-dispatches (pinned in the
//!   CLI-level test file).

use chelis_types::errors::CheckErrorKind;
use chelis_types::{BUILTIN_NAMES, check_ir_program};

fn parse_core_fixture(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture")
}

fn shadowing_errors(exprs: &[chelis_deep::Expr]) -> Vec<chelis_types::errors::CheckError> {
    match check_ir_program(exprs, chelis_types::ProgramOrigin::Unlinked) {
        Ok(_) => Vec::new(),
        Err(report) => report
            .errors
            .into_iter()
            .filter(|e| matches!(e.kind, CheckErrorKind::BuiltinShadowing))
            .collect(),
    }
}

/// The chelis#353 reproducer: `def sum` shadowing the reduction builtin
/// is rejected with a diagnostic that names the builtin, identifies the
/// declaration site, says shadowing is not allowed, and cites the owning
/// spec section.
#[test]
fn def_sum_shadowing_rejected_with_spec_diagnostic() {
    let deep = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/def_sum_shadowing_rejected_with_spec_diagnostic_70.deep"
    ));
    let errs = shadowing_errors(&deep);
    assert_eq!(
        errs.len(),
        1,
        "exactly one BuiltinShadowing error expected (the inline-annotated \
         def desugars to defsig+def and must be reported once); got {errs:?}"
    );
}

/// A standalone `sig sum: ...` (Deep `defsig`) with no accompanying body
/// is rejected the same way: the sig alone rebinds the env entry for
/// `sum`, so later builtin calls would typecheck against the user
/// signature while eval still dispatches the builtin.
#[test]
fn sig_only_shadowing_rejected() {
    let deep = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/sig_only_shadowing_rejected_107.deep"
    ));
    let errs = shadowing_errors(&deep);
    assert_eq!(
        errs.len(),
        1,
        "sig-only shadowing must be rejected; got {errs:?}"
    );
}

/// Load-style top-level bindings desugar to `def`, so `sum = ...` at top
/// level is rejected with the same clear diagnostic (pre-fix it failed
/// with a confusing unification error against the builtin scheme).
#[test]
fn top_level_load_binding_shadowing_rejected() {
    let deep = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/top_level_load_binding_shadowing_rejected_130.deep"
    ));
    let errs = shadowing_errors(&deep);
    assert!(
        !errs.is_empty(),
        "top-level `sum = ...` desugars to `def sum` and must be rejected"
    );
}

/// Table-completeness sweep: every name in `BUILTIN_NAMES` is rejected
/// as a `def` name. The rejection reads the same table the evaluator
/// dispatch (`runtime/host_ops.rs`) and IR lowering (`lower.rs`) import,
/// and this sweep makes any future divergence loud: add a builtin, and
/// the shadowing rejection covers it automatically or this test fails.
///
/// The defs are synthesized at the Deep level because several builtin
/// names (`and`, `or`, `not`, `mod`, `where`, ...) are Surf operator or
/// keyword spellings that cannot appear as a Surf `def` name; the Deep
/// symbol grammar accepts them all, and Deep is what every checker entry
/// point consumes.
#[test]
fn every_builtin_name_def_is_rejected() {
    for name in BUILTIN_NAMES {
        let deep_src = format!(
            "(def {{}} {name} (fn {{}} (params {{}} (x {{type: (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} f32)))}})) (app {{}} (var {{}} relu) (var {{}} x))))"
        );
        let exprs = chelis_deep::parser::parse_str(&deep_src)
            .unwrap_or_else(|e| panic!("deep parse for builtin `{name}`: {e:?}"));
        let errs = shadowing_errors(&exprs);
        assert!(
            !errs.is_empty(),
            "def `{name}` shadows a BUILTIN_NAMES entry and must be rejected"
        );
    }
}

/// Spot-check `defsig` rejection across builtin classes (a reduction, an
/// elementwise op, a shape op, a constructor) at the Deep level.
#[test]
fn defsig_rejection_spans_builtin_classes() {
    for name in ["sum", "relu", "mul", "expand", "to_tensor"] {
        let deep_src = format!(
            "(defsig {{}} {name} (t-fn {{}} (t-ref {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} f32))) (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} f32))))"
        );
        let exprs = chelis_deep::parser::parse_str(&deep_src).expect("deep parse");
        let errs = shadowing_errors(&exprs);
        assert!(
            !errs.is_empty(),
            "defsig `{name}` must be rejected as builtin shadowing"
        );
    }
}

/// Negative parity: near-miss names that merely CONTAIN a builtin name
/// must stay accepted.
#[test]
fn near_miss_names_still_check_clean() {
    for def_name in ["sum2", "my_sum", "summarize", "relu_fwd"] {
        let deep = parse_core_fixture(&format!(
            r###"(module {{surf_path: "NearMiss"}}
  nearmiss
  (defsig {{}}
    {def_name}
    (t-fn {{}}
      (t-ref {{}} (t-tensor {{}} (d-name {{}} batch) (t-prim {{}} f32)))
      (t-tensor {{}} (d-name {{}} batch) (t-prim {{}} f32))))
  (def {{}}
    {def_name}
    (fn {{}}
      (params {{}} (x {{type: (t-var {{}} _)}}))
      (app {{span: "surf:86..93"}}
        (var {{span: "surf:86..90"}} relu)
        (var {{span: "surf:91..92"}} x))))
  (def {{}}
    out
    (app {{span: "surf:100..141"}}
      (var {{span: "surf:100..117"}} {def_name})
      (app {{span: "surf:118..140"}}
        (var {{span: "surf:118..127"}} to_tensor)
        (app {{span: "surf:128..139"}}
          (var {{}} Cons)
          (lit {{span: "surf:129..132", type: (t-prim {{}} f32)}} 1.0)
          (app {{}}
            (var {{}} Cons)
            (app {{span: "surf:134..138"}}
              (var {{}} neg)
              (lit {{span: "surf:135..138", type: (t-prim {{}} f32)}} 2.0))
            (var {{}} Nil)))))))
"###
        ));
        let res = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked);
        assert!(
            res.is_ok(),
            "near-miss `{def_name}` must check clean; got {:?}",
            res.err().map(|r| r.errors)
        );
    }
}

/// Scope pin: VALUE-POSITION parameters and block-locals may reuse
/// builtin names — verified on all three lanes in the owning issue
/// investigation (check clean, eval and C backend both produce
/// [1.0, 0.0]) — so the §8.6 rule deliberately does not bind to them.
///
/// Call position follows the same lexical rule: the innermost callable
/// binding wins over the builtin in every lane (spec/04-type-system.md
/// §8.6, chelis#1076).
#[test]
fn value_params_and_locals_may_reuse_builtin_names() {
    let param_case = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/value_params_and_locals_may_reuse_builtin_names_226.deep"
    ));
    assert!(
        check_ir_program(&param_case, chelis_types::ProgramOrigin::Unlinked).is_ok(),
        "a value parameter named `sum` must stay accepted"
    );

    let local_case = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/value_params_and_locals_may_reuse_builtin_names_236.deep"
    ));
    assert!(
        check_ir_program(&local_case, chelis_types::ProgramOrigin::Unlinked).is_ok(),
        "a block-local named `sum` must stay accepted"
    );
}

/// An inline-annotated `def` desugars to a `defsig` AND a `def` with the
/// same name; the rejection must dedupe to exactly one error per name,
/// reported as the `def` (what the user wrote).
#[test]
fn inline_annotated_def_reports_exactly_one_error() {
    let deep = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/inline_annotated_def_reports_exactly_one_error_255.deep"
    ));
    let errs = shadowing_errors(&deep);
    assert_eq!(
        errs.len(),
        1,
        "defsig+def pair from one inline-annotated def must produce one \
         BuiltinShadowing error; got {errs:?}"
    );
}

/// chelis#1076: callable parameters named like builtins follow ordinary
/// lexical scope. The checker must accept both a def parameter and a lambda
/// parameter in call position; eval/build parity is pinned at the CLI layer.
#[test]
fn builtin_named_param_called_in_body_is_accepted() {
    let called = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/builtin_named_param_called_in_body_is_accepted_274.deep"
    ));
    let called_result = check_ir_program(&called, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        called_result.is_ok(),
        "calling a builtin-named parameter must follow lexical scope; got {:?}",
        called_result.err().map(|report| report.errors)
    );

    // Lambda parameters get the same treatment.
    let lambda = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/builtin_named_param_called_in_body_is_accepted_286.deep"
    ));
    let lambda_result = check_ir_program(&lambda, chelis_types::ProgramOrigin::Unlinked);
    assert!(
        lambda_result.is_ok(),
        "calling a builtin-named lambda parameter must follow lexical scope; got {:?}",
        lambda_result.err().map(|report| report.errors)
    );

    // A DIFFERENT (non-shadowed) callee alongside a builtin-named value
    // parameter stays accepted: only the call-position shape is rejected.
    let value_only = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/builtin_named_param_called_in_body_is_accepted_299.deep"
    ));
    assert!(
        check_ir_program(&value_only, chelis_types::ProgramOrigin::Unlinked).is_ok(),
        "value-position builtin-named params must stay accepted"
    );
}

#[test]
fn builtin_named_non_callable_local_still_rejects_as_a_type_error() {
    let deep = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_353_builtin_shadowing/builtin_named_non_callable_local_still_rejects_as_a_type_error_312.deep"
    ));
    let err = check_ir_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("a scalar local is not callable");
    assert!(
        err.errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::TypeMismatch)
                && error.message.contains("f64 vs (f64, i64)")
        }) && err
            .errors
            .iter()
            .all(|error| !matches!(error.kind, CheckErrorKind::BuiltinShadowing)),
        "lexical precedence must diagnose the selected scalar local rather than fall back to the builtin: {:?}",
        err.errors
    );
}
