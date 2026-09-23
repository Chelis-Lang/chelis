//! Bounded authored-binder repair experiment, not a named value-origin policy.
//! Rigidity expectations follow spec/04 §§4.3, 4.4 and [04-INF-6].
//! Named-query controls preserve the existing source contract; they do not
//! authorize associating arbitrary same-spelled named values globally.

use chelis_deep::parser::parse_str;
use chelis_types::{
    CheckedProgram, InferResult, TypeEnv, build_compiled_library_context, check_ir_program,
    check_ir_with_context, errors::CheckErrorKind,
};

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    parse_str(source).expect("Deep fixture parses")
}

const D: &str = "(d-var {} d)";
const SEVENTEEN: &str = "(d-lit {} 17)";
const FIXED: &str = "(t-tensor {} (d-name {} fixed) (t-prim {} f32))";
const AFFINE: &str = "(deftype {} Affine () (variant {} Affine (field {} gain (t-tensor {} (d-name {} fixed) (t-prim {} f32)))))";

fn tensor(dimensions: &str) -> String {
    format!("(t-tensor {{}} {dimensions} (t-prim {{}} f32))")
}

// Canonical defsig binders and erased parameter annotations match representative.deep.
fn function(
    name: &str,
    binders: &str,
    input: &str,
    gain: &str,
    result: &str,
    body: &str,
) -> String {
    format!(
        "(defsig {{}} {name} ({binders}) (t-fn {{}} {input} {gain} {result}))
         (def {{}} {name} (fn {{}} (params {{}} (x {{type: (t-var {{}} _)}}) (gain {{type: (t-var {{}} _)}})) {body}))"
    )
}

fn multiply(gain: &str, reversed: bool) -> String {
    if reversed {
        format!("(app {{}} (var {{}} mul) {gain} (var {{}} x))")
    } else {
        format!("(app {{}} (var {{}} mul) (var {{}} x) {gain})")
    }
}

fn insertion_control(result: &str, reversed: bool) -> String {
    let gain = "(app {} (var {} insert) (access {} (var {} gain) gain)
        (lit {type: (t-prim {} i32)} 0)
        (app {} (var {} shape) (var {} x) (lit {type: (t-prim {} i32)} 0)))";
    let function = function(
        "checked",
        "a d",
        &tensor("(d-var {} a) (d-var {} d)"),
        "(t-adt {} Affine)",
        &tensor(&format!("(d-var {{}} a) {result}")),
        &multiply(gain, reversed),
    );
    format!("{AFFINE}\n{function}")
}

fn consumer(alias: bool, reduce: bool) -> String {
    let callee = if alias { "f" } else { "aligned" };
    let call = format!("(app {{}} (var {{}} {callee}) (var {{}} x) (var {{}} gain))");
    let body = if reduce {
        format!("(app {{}} (var {{}} sum) {call} (var {{}} fixed))")
    } else {
        call
    };
    let body = if alias {
        format!("(let {{}} (bind {{}} f (var {{}} aligned)) {body})")
    } else {
        body
    };
    let result = if reduce {
        "(t-var {} _)".to_owned()
    } else {
        tensor(SEVENTEEN)
    };
    function(
        if reduce { "reduce" } else { "wrong" },
        "d",
        &tensor(D),
        FIXED,
        &result,
        &body,
    )
}

fn verdict_error(
    result: Result<CheckedProgram, InferResult>,
    accepts: bool,
    label: &str,
) -> Option<String> {
    match result {
        Ok(_) if accepts => None,
        Ok(_) => Some(format!(
            "{label}: accepted a forbidden authored-binder constraint"
        )),
        Err(report) if !accepts => (!report
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)))
        .then(|| format!("{label}: wrong rejection class: {report:?}")),
        Err(report) => Some(format!(
            "{label}: rejected matching/name-query control: {report:?}"
        )),
    }
}

fn assert_verdict(result: Result<CheckedProgram, InferResult>, accepts: bool, label: &str) {
    if let Some(error) = verdict_error(result, accepts, label) {
        panic!("{error}");
    }
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Whole,
    Live,
    Decoded,
}

fn check(library: &str, consumer: &str, route: Route) -> Result<CheckedProgram, InferResult> {
    if matches!(route, Route::Whole) {
        return check_ir_program(&parse(&format!("{library}\n{consumer}")));
    }
    let (context, _) = build_compiled_library_context(&parse(library)).expect("library accepted");
    let context = if matches!(route, Route::Decoded) {
        let bytes = bincode::serialize(&context).unwrap();
        bincode::deserialize::<TypeEnv>(&bytes).unwrap()
    } else {
        context
    };
    check_ir_with_context(&context, &parse(consumer))
}

fn direct_control(result: &str, reversed: bool, adt: bool) -> String {
    let prefix = if adt { AFFINE } else { "" };
    let parameter = if adt { "(t-adt {} Affine)" } else { FIXED };
    let gain = if adt {
        "(access {} (var {} gain) gain)"
    } else {
        "(var {} gain)"
    };
    let function = function(
        "checked",
        "d",
        &tensor(D),
        parameter,
        &tensor(result),
        &multiply(gain, reversed),
    );
    format!("{prefix}\n{function}")
}

#[test]
fn direct_and_adt_bad_results_reject_both_orders() {
    let mut failures = Vec::new();
    for adt in [false, true] {
        for reversed in [false, true] {
            let source = direct_control(SEVENTEEN, reversed, adt);
            failures.extend(verdict_error(
                check_ir_program(&parse(&source)),
                false,
                &source,
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn direct_and_adt_matching_results_accept_both_orders() {
    assert_verdict(
        check_ir_program(&parse(include_str!(
            "fixtures/core/tests/issue_1875_protected_dimension_identity/representative.deep"
        ))),
        true,
        "representative Deep program",
    );
    for adt in [false, true] {
        for reversed in [false, true] {
            let source = direct_control(D, reversed, adt);
            assert_verdict(check_ir_program(&parse(&source)), true, &source);
        }
    }
}

#[test]
fn insertion_bad_results_reject_both_orders() {
    let mut failures = Vec::new();
    for reversed in [false, true] {
        let source = insertion_control(SEVENTEEN, reversed);
        failures.extend(verdict_error(
            check_ir_program(&parse(&source)),
            false,
            &source,
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn insertion_matching_results_accept_both_orders() {
    for reversed in [false, true] {
        let source = insertion_control(D, reversed);
        assert_verdict(check_ir_program(&parse(&source)), true, &source);
    }
}

fn helper(reversed: bool) -> String {
    function(
        "aligned",
        "d",
        &tensor(D),
        FIXED,
        &tensor(D),
        &multiply("(var {} gain)", reversed),
    )
}

#[test]
fn helper_bad_results_reject_all_routes_and_aliases() {
    let mut failures = Vec::new();
    for route in [Route::Whole, Route::Live, Route::Decoded] {
        for reversed in [false, true] {
            for alias in [false, true] {
                failures.extend(verdict_error(
                    check(&helper(reversed), &consumer(alias, false), route),
                    false,
                    &format!("{route:?}, reversed={reversed}, alias={alias}"),
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn helper_named_queries_accept_all_routes_and_aliases() {
    for route in [Route::Whole, Route::Live, Route::Decoded] {
        for reversed in [false, true] {
            for alias in [false, true] {
                assert_verdict(
                    check(&helper(reversed), &consumer(alias, true), route),
                    true,
                    &format!("{route:?}, reversed={reversed}, alias={alias}"),
                );
            }
        }
    }
}

#[test]
fn unrelated_named_argument_does_not_name_returned_axis() {
    let library = function(
        "aligned",
        "d",
        &tensor(D),
        FIXED,
        &tensor(D),
        "(copy {} (var {} x))",
    );
    let consumer = consumer(false, true);
    for route in [Route::Whole, Route::Live, Route::Decoded] {
        assert_verdict(
            check(&library, &consumer, route),
            false,
            &format!("{route:?}"),
        );
    }
}

#[test]
fn return_only_and_distinctness_boundaries() {
    for (source, accepts) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1875_protected_dimension_identity/return_only_distinct.deep"
            ),
            true,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1875_protected_dimension_identity/return_only_same.deep"
            ),
            false,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1875_protected_dimension_identity/return_only_named.deep"
            ),
            true,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1875_protected_dimension_identity/return_only_wildcard.deep"
            ),
            true,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1875_protected_dimension_identity/distinct_return_binder.deep"
            ),
            false,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1875_protected_dimension_identity/distinct_parameter_binders.deep"
            ),
            false,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1875_protected_dimension_identity/distinct_named_dimensions.deep"
            ),
            false,
        ),
    ] {
        assert_verdict(check_ir_program(&parse(source)), accepts, source);
    }
}
