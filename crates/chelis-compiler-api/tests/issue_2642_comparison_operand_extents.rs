//! chelis#2642: a comparison, a logical operation or a `where` over operands
//! whose extents are known only at run time compiles, and its operand
//! agreement and declared result are checked when it runs, on both lanes.
//!
//! Each `to_tensor` result carries its own run-time extent, so `lt(a, b)`
//! over two of them compares two extents the graph cannot prove equal. The
//! verifier required the operands' extents to be one proved extent and the
//! C build stopped at an internal invariant, while `chelis eval` ran the
//! program. runtime_extents.md C2.3 decides the rule: only a contradiction is
//! refused before execution, and an extent the graph cannot prove keeps its
//! run-time guard. These operations are same-shape producers, as arithmetic
//! is, so the C emitter checks their operands' agreement before indexing
//! them, and a declared result extent on the returned value is checked by
//! the operation that produced it (spec/04 §4.7). `chelis eval` checked no
//! such declared extent on a comparison of literal operands; it now reports
//! the same `eq` guard as compiled C.
//!
//! A logical operation whose operands are comparisons written inline over
//! `to_tensor` results, such as `and(lt(to_tensor(xs), to_tensor(ys)), ...)`
//! with no name bound, is computed on the host, element by element, after the
//! same operand agreement check. Its C build combined the two tensor pointers
//! with a scalar `&&`.
//!
//! Oracle for the positive corpus: the compiled program runs against the
//! `ownership-ledger` runtime, every allocation is finalized with no live
//! owner left, and stdout equals the evaluator's rendering of the same roots.
//!
//! The negative twins give the operands two lengths, or the declared result
//! another extent. Both lanes must fail. A contradiction between literal
//! extents is still refused before anything runs.

mod ownership_support;

use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{
    CompileRequest, CompileTarget, Diagnostic, DiagnosticSpan, EvalRequest, SourceKind,
};
use chelis_vocab::DiagnosticKind;
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

struct Case {
    name: &'static str,
    source: &'static str,
}

const POSITIVE: &[Case] = &[
    // The witness: two independent run-time lengths.
    Case {
        name: "comparison_of_two_lengths",
        source: "module Demo.Main\ndef both(xs: List[i64], ys: List[i64]) -> List[bool] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  to_list(lt(a, b))\n}\nresult = both([1i64, 2i64], [3i64, 0i64])\n",
    },
    // One operand computed from both lengths.
    Case {
        name: "comparison_with_a_computed_operand",
        source: "module Demo.Main\ndef both(xs: List[i64], ys: List[i64]) -> List[bool] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  to_list(lt(a, sub(b, a)))\n}\nresult = both([1i64, 2i64], [3i64, 0i64])\n",
    },
    // Logical operations over comparisons of two lengths.
    Case {
        name: "logical_over_two_lengths",
        source: "module Demo.Main\ndef both(xs: List[i64], ys: List[i64]) -> List[bool] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  to_list(or(not(lt(a, b)), and(gt(a, sub(a, a)), eq(a, b))))\n}\nresult = both([1i64, 2i64, 7i64], [3i64, 2i64, 0i64])\n",
    },
    // A tensor helper whose wildcard parameters each carry their own extent:
    // `where` over its branches and `and` over its operands.
    Case {
        name: "where_over_wildcard_parameters",
        source: "module Demo.Main\ndef f(a: tensor[*, i64], b: tensor[*, i64]) -> tensor[*, i64] = where(lt(a, sub(a, a)), a, b)\ndef run(xs: List[i64], ys: List[i64]) -> List[i64] = to_list(f(to_tensor(xs), to_tensor(ys)))\nresult = run([1i64, -2i64], [3i64, 4i64])\n",
    },
    Case {
        name: "logical_over_wildcard_parameters",
        source: "module Demo.Main\ndef f(a: tensor[*, bool], b: tensor[*, bool]) -> tensor[*, bool] = and(a, b)\ndef run(xs: List[i64], ys: List[i64]) -> List[bool] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  to_list(f(gt(a, sub(a, a)), gt(b, sub(b, b))))\n}\nresult = run([1i64, -2i64], [3i64, 4i64])\n",
    },
    // Logical operations whose operands are comparisons written inline over
    // `to_tensor` results, with no name bound between them: the operands
    // are evaluated first and the logical operation takes their results.
    Case {
        name: "logical_over_inline_comparisons",
        source: "module Demo.Main\ndef run(xs: List[i64], ys: List[i64]) -> List[bool] = to_list(and(lt(to_tensor(xs), to_tensor(ys)), lt(to_tensor(ys), to_tensor(xs))))\nresult = run([1i64, 9i64], [5i64, 6i64])\n",
    },
    Case {
        name: "or_and_not_over_inline_comparisons",
        source: "module Demo.Main\ndef run(xs: List[i64], ys: List[i64]) -> List[bool] = to_list(or(lt(to_tensor(xs), to_tensor(ys)), not(lt(to_tensor(ys), to_tensor(xs)))))\nresult = run([1i64, 9i64], [5i64, 6i64])\n",
    },
    Case {
        name: "logical_over_inline_comparisons_returned",
        source: "module Demo.Main\ndef run(xs: List[i64], ys: List[i64]) -> tensor[*, bool] = and(lt(to_tensor(xs), to_tensor(ys)), lt(to_tensor(ys), to_tensor(xs)))\nresult = to_list(run([1i64, 9i64], [5i64, 6i64]))\n",
    },
    Case {
        name: "logical_over_generic_helper_calls",
        source: "module Demo.Main\ndef less[n](a: &tensor[n, i64], b: &tensor[n, i64]) -> tensor[n, bool] = lt(a, b)\ndef run(xs: List[i64], ys: List[i64]) -> List[bool] = to_list(and(less(to_tensor(xs), to_tensor(ys)), less(to_tensor(ys), to_tensor(xs))))\nresult = run([1i64, 9i64], [5i64, 6i64])\n",
    },
    // The shape a validating column constructor takes: comparisons of a
    // parameter with a helper's result, folded into a failure index.
    Case {
        name: "validated_column_update",
        source: "module Demo.Main\ntype Ds[n] =\n  | Ds { days: tensor[n, i64] }\ndef filled[n](like: &tensor[n, i64], value: i64) -> tensor[n, i64] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))\ndef first_false(flags: List[bool]) -> i64 = fold(fn (acc: (i64, i64), flag: bool) -> if gte(acc.1, 0i64) then acc else if flag then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), flags).1\ndef from_days[n](t: tensor[n, i64]) -> Ds[n] = {\n  bad = first_false(to_list(gte(t, filled(t, -1000i64))))\n  if gte(bad, 0i64) then fail(\"from_days: domain: out of range\") else Ds { days: t }\n}\ndef epoch[n](ds: Ds[n]) -> tensor[n, i64] = ds.days\ndef add_days[n](ds: Ds[n], days: &tensor[n, i64]) -> Ds[n] = {\n  start = epoch(ds)\n  fits = and(lte(days, sub(filled(start, 100i64), start)), gte(days, sub(filled(start, -100i64), start)))\n  bad = first_false(to_list(fits))\n  if gte(bad, 0i64) then fail(\"add_days: overflow: out of range\") else Ds { days: add(start, days) }\n}\ndef run(starts: List[i64], steps: List[i64]) -> List[i64] = to_list(epoch(add_days(from_days(to_tensor(starts)), to_tensor(steps))))\nresult = run([0i64, 5i64, 7i64, 9i64], [1i64, -1i64, 0i64, 0i64])\n",
    },
    // The issue's program with an agreeing call, run-time and literal forms.
    Case {
        name: "declared_result_agrees",
        source: "module Demo.Main\ndef f(a: tensor[*, f32], b: tensor[*, f32]) -> tensor[2, bool] = eq(a, b)\ndef run(xs: List[f32], ys: List[f32]) -> List[bool] = to_list(f(to_tensor(xs), to_tensor(ys)))\nresult = run([1.0, 2.0], [1.0, 5.0])\n",
    },
    Case {
        name: "declared_result_agrees_on_literals",
        source: "module Demo.Main\ndef f(a: tensor[*, f32], b: tensor[*, f32]) -> tensor[2, bool] = eq(a, b)\nout = f(to_tensor([1.0, 2.0]), to_tensor([1.0, 5.0]))\n",
    },
];

/// Each negative twin with the line the evaluator reports and the line the
/// compiled program reports.
const NEGATIVE: &[(Case, &str, &str)] = &[
    (
        Case {
            name: "comparison_lengths_differ",
            source: "module Demo.Main\ndef both(xs: List[i64], ys: List[i64]) -> List[bool] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  to_list(lt(a, b))\n}\nresult = both([1i64, 2i64], [3i64, 0i64, 5i64])\n",
        },
        "lt operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in lt at i64",
        "lt operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in lt at i64",
    ),
    (
        Case {
            name: "logical_lengths_differ",
            source: "module Demo.Main\ndef both(xs: List[i64], ys: List[i64]) -> List[bool] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  to_list(and(gt(a, sub(a, a)), lt(b, sub(b, b))))\n}\nresult = both([1i64, 2i64], [3i64, 0i64, 5i64])\n",
        },
        "and operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in and at i64",
        "and operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in and at i64",
    ),
    (
        Case {
            name: "inline_logical_lengths_differ",
            source: "module Demo.Main\ndef run(xs: List[i64], ys: List[i64]) -> List[bool] = to_list(and(lt(to_tensor(xs), to_tensor(xs)), lt(to_tensor(ys), to_tensor(ys))))\nresult = run([1i64, 2i64], [3i64, 0i64, 5i64])\n",
        },
        "and operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in and at i64",
        "and operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in and at i64",
    ),
    (
        Case {
            name: "where_lengths_differ",
            source: "module Demo.Main\ndef f(a: tensor[*, i64], b: tensor[*, i64]) -> tensor[*, i64] = where(lt(a, sub(a, a)), a, b)\ndef run(xs: List[i64], ys: List[i64]) -> List[i64] = to_list(f(to_tensor(xs), to_tensor(ys)))\nresult = run([1i64, -2i64], [3i64, 4i64, 5i64])\n",
        },
        "where operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in where at i64",
        "where operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in where at i64",
    ),
    (
        Case {
            name: "wildcard_logical_lengths_differ",
            source: "module Demo.Main\ndef f(a: tensor[*, bool], b: tensor[*, bool]) -> tensor[*, bool] = and(a, b)\ndef run(xs: List[i64], ys: List[i64]) -> List[bool] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  to_list(f(gt(a, sub(a, a)), gt(b, sub(b, b))))\n}\nresult = run([1i64, -2i64], [3i64, 4i64, 5i64])\n",
        },
        "and operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in and at i64",
        "and operands disagree at axis 0: lhs [2] has 2, rhs [3] has 3\nnumeric trap: domain in and at i64",
    ),
    // The issue's expected failure: the declared-result guard names `eq`.
    (
        Case {
            name: "declared_result_disagrees",
            source: "module Demo.Main\ndef f(a: tensor[*, f32], b: tensor[*, f32]) -> tensor[2, bool] = eq(a, b)\ndef run(xs: List[f32], ys: List[f32]) -> List[bool] = to_list(f(to_tensor(xs), to_tensor(ys)))\nresult = run([1.0, 2.0, 3.0], [1.0, 5.0, 3.0])\n",
        },
        "extent `2`: claimed = 2, eq axis 0 = 3\nnumeric trap: domain in eq at i64",
        "extent `2`: claimed = 2, eq axis 0 = 3\nnumeric trap: domain in eq at i64",
    ),
    // The issue's own program, whose literal operands `chelis eval` used to
    // compare without checking the declared extent.
    (
        Case {
            name: "declared_result_disagrees_on_literals",
            source: "module Demo.Main\ndef f(a: tensor[*, f32], b: tensor[*, f32]) -> tensor[2, bool] = eq(a, b)\nout = f(to_tensor([1.0, 2.0, 3.0]), to_tensor([1.0, 5.0, 3.0]))\n",
        },
        "extent `2`: claimed = 2, eq axis 0 = 3\nnumeric trap: domain in eq at i64",
        "extent `2`: claimed = 2, eq axis 0 = 3\nnumeric trap: domain in eq at i64",
    ),
];

/// Literal extents that contradict each other are a type error on both
/// lanes, before anything runs.
const CONTRADICTION: Case = Case {
    name: "literal_lengths_contradict",
    source: "module Demo.Main\nresult = lt(to_tensor([1i64, 2i64]), to_tensor([3i64, 0i64, 5i64]))\n",
};

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

/// The evaluator's rendering of every root, in the compiled driver's format.
fn evaluated(source: &str) -> String {
    let result = eval(request(source))
        .unwrap_or_else(|error| panic!("evaluator rejected the case: {error:?}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

/// The evaluator's failure text for a program that must not run.
fn evaluator_failure(source: &str) -> String {
    let error = eval(request(source)).expect_err("the evaluator must refuse the case");
    error
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn outcome(case: &Case, check: impl FnOnce(&str)) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| check(case.source))).map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{}\n  -> {head}", case.name, case.source)
    })
}

// REGRESSION TEST. Before this change the C build of every program here
// stopped at "comparison ... requires exactly matching operand shape" (or
// the logical and `where` forms of that invariant) in ownership lowering.
#[test]
fn comparisons_over_run_time_extents_compile_as_eval_runs_them() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| {
            outcome(case, |source| {
                let expected = evaluated(source);
                let generated = ownership_support::emit(source, case.name);
                let (summary, stdout) = ownership_support::run_program(&generated);
                ownership_support::balanced(&summary);
                assert_eq!(stdout, expected, "compiled output differs from eval");
            })
            .err()
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn disagreeing_extents_still_fail_in_both_lanes() {
    let failures: Vec<String> = NEGATIVE
        .iter()
        .filter_map(|(case, evaluator, compiled)| {
            outcome(case, |source| {
                let refused = evaluator_failure(source);
                assert!(refused.contains(evaluator), "evaluator: {refused}");
                let generated = ownership_support::emit(source, case.name);
                let stderr = ownership_support::run_failure_stderr(&generated, "");
                assert!(stderr.contains(compiled), "compiled: {stderr}");
            })
            .err()
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn contradicting_literal_extents_are_refused_before_running() {
    let call_offset = u64::try_from(
        CONTRADICTION
            .source
            .find("lt(")
            .expect("authored comparison call"),
    )
    .expect("source offset fits u64");
    let assert_refusal = |errors: &[Diagnostic]| {
        let mismatch = errors
            .iter()
            .find(|error| {
                error.kind() == DiagnosticKind::DimensionMismatch
                    && error.message.contains("`lt`")
                    && error.message.contains("argument 2")
                    && error.message.contains("axis 0")
            })
            .unwrap_or_else(|| panic!("the comparison must reject at its operand: {errors:?}"));
        assert_eq!(mismatch.expected.as_deref(), Some("2"), "{errors:?}");
        assert_eq!(mismatch.got.as_deref(), Some("3"), "{errors:?}");
        assert_eq!(
            mismatch.span,
            Some(DiagnosticSpan::Point {
                offset: call_offset
            }),
            "{errors:?}"
        );
    };

    let evaluated = eval(request(CONTRADICTION.source))
        .expect_err("the evaluator must refuse contradictory literal extents");
    assert_refusal(&evaluated.errors);

    let compiled = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: CONTRADICTION.source.to_string(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .err()
    .unwrap_or_else(|| panic!("{}: the C build must refuse", CONTRADICTION.name));
    assert_refusal(&compiled.errors);
}
