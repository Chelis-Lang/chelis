//! Bounded authored-binder repair experiment, not a named value-origin policy.
//! Rigidity expectations follow spec/04 §§4.3, 4.4 and [04-INF-6].
//! Named-query controls preserve the existing source contract; they do not
//! authorize associating arbitrary same-spelled named values globally.

use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{
    CheckedProgram, InferResult, TypeEnv, build_compiled_library_context, check_ir_program,
    check_ir_with_context, errors::CheckErrorKind,
};

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    desugar_program(&parse_str(source).expect("fixture parses")).expect("Surf fixture must desugar")
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
    let prefix = if adt {
        "type Affine = | Affine { gain: tensor[fixed, f32] }\n"
    } else {
        ""
    };
    let parameter = if adt {
        "gain: Affine"
    } else {
        "gain: tensor[fixed, f32]"
    };
    let gain = if adt { "gain.gain" } else { "gain" };
    let body = if reversed {
        format!("mul({gain}, x)")
    } else {
        format!("mul(x, {gain})")
    };
    format!(
        "{prefix}def checked[d](x: tensor[d, f32], {parameter}) -> tensor[{result}, f32] = {body}"
    )
}

#[test]
fn direct_and_adt_bad_results_reject_both_orders() {
    let mut failures = Vec::new();
    for adt in [false, true] {
        for reversed in [false, true] {
            let source = direct_control("17", reversed, adt);
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
    for adt in [false, true] {
        for reversed in [false, true] {
            let source = direct_control("d", reversed, adt);
            assert_verdict(check_ir_program(&parse(&source)), true, &source);
        }
    }
}

#[test]
fn insertion_bad_results_reject_both_orders() {
    let mut failures = Vec::new();
    for reversed in [false, true] {
        let body = if reversed {
            "mul(insert(gain.gain, 0i32, shape(x, 0i32)), x)"
        } else {
            "mul(x, insert(gain.gain, 0i32, shape(x, 0i32)))"
        };
        let source = format!(
            "type Affine = | Affine {{ gain: tensor[fixed, f32] }}\ndef checked[a, d](x: tensor[a, d, f32], gain: Affine) -> tensor[a, 17, f32] = {body}"
        );
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
        let body = if reversed {
            "mul(insert(gain.gain, 0i32, shape(x, 0i32)), x)"
        } else {
            "mul(x, insert(gain.gain, 0i32, shape(x, 0i32)))"
        };
        let source = format!(
            "type Affine = | Affine {{ gain: tensor[fixed, f32] }}\ndef checked[a, d](x: tensor[a, d, f32], gain: Affine) -> tensor[a, d, f32] = {body}"
        );
        assert_verdict(check_ir_program(&parse(&source)), true, &source);
    }
}

fn helper(reversed: bool) -> String {
    let body = if reversed {
        "mul(gain, x)"
    } else {
        "mul(x, gain)"
    };
    format!(
        "def aligned[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[d, f32] = {body}"
    )
}

#[test]
fn helper_bad_results_reject_all_routes_and_aliases() {
    let mut failures = Vec::new();
    for route in [Route::Whole, Route::Live, Route::Decoded] {
        for reversed in [false, true] {
            for alias in [false, true] {
                let call = if alias {
                    "{ f = aligned\n f(x, gain) }"
                } else {
                    "aligned(x, gain)"
                };
                let consumer = format!(
                    "def wrong[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[17, f32] = {call}"
                );
                failures.extend(verdict_error(
                    check(&helper(reversed), &consumer, route),
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
                let call = if alias {
                    "{ f = aligned\n sum(f(x, gain), fixed) }"
                } else {
                    "sum(aligned(x, gain), fixed)"
                };
                let consumer =
                    format!("def reduce[d](x: tensor[d, f32], gain: tensor[fixed, f32]) = {call}");
                assert_verdict(
                    check(&helper(reversed), &consumer, route),
                    true,
                    &format!("{route:?}, reversed={reversed}, alias={alias}"),
                );
            }
        }
    }
}

#[test]
fn unrelated_named_argument_does_not_name_returned_axis() {
    let library =
        "def aligned[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[d, f32] = copy(x)";
    let consumer =
        "def reduce[d](x: tensor[d, f32], gain: tensor[fixed, f32]) = sum(aligned(x, gain), fixed)";
    for route in [Route::Whole, Route::Live, Route::Decoded] {
        assert_verdict(
            check(library, consumer, route),
            false,
            &format!("{route:?}"),
        );
    }
}

#[test]
fn return_only_and_distinctness_boundaries() {
    for (source, accepts) in [
        (
            "def kept[k](x: tensor[2, f32]) -> tensor[k, f32] = to_tensor([1.0f32, 2.0f32, 3.0f32])",
            true,
        ),
        (
            "def wrong[k](x: tensor[2, f32]) -> tensor[k, f32] = to_tensor([1.0f32, 2.0f32])",
            false,
        ),
        (
            "def kept[m](x: tensor[batch, f32]) -> tensor[m, f32] = copy(x)",
            true,
        ),
        (
            "def kept[k](x: tensor[*, f32]) -> tensor[k, f32] = copy(x)",
            true,
        ),
        (
            "def wrong[n, m](x: tensor[n, f32], y: tensor[batch, f32]) -> tensor[m, f32] = add(x, y)",
            false,
        ),
        (
            "def wrong[a, b](x: tensor[a, f32], y: tensor[b, f32]) -> tensor[a, f32] = mul(x, y)",
            false,
        ),
        (
            "def wrong(x: tensor[fixed, f32], y: tensor[other, f32]) = mul(x, y)",
            false,
        ),
    ] {
        assert_verdict(check_ir_program(&parse(source)), accepts, source);
    }
}
