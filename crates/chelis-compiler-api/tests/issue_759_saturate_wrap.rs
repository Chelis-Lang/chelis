//! chelis#759: the named lossy casts `cast_saturate` ([05-OP-23]) and
//! `cast_wrap` ([05-OP-24]) over every admitted source and target pair.
//!
//! Each pair runs over an edge grid on four surfaces: a top-level tensor
//! with a borrowed source (the DAG lane), a definition (the typed graph lane),
//! a tensor computed on the host (the C host tensor lane) and scalars (the
//! host scalar lane). `chelis eval` must print the values an
//! independent `i128` oracle computes, and the compiled program must print
//! what eval prints with every allocation finalized. The negative twins are
//! the pairs each atom refuses, NaN under `cast_saturate`, and `grad`.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const INTS: [(&str, u32); 4] = [("i8", 8), ("i16", 16), ("i32", 32), ("i64", 64)];
const FLOATS: [&str; 4] = ["f16", "bf16", "f32", "f64"];

fn bounds(width: u32) -> (i128, i128) {
    let half = 1i128 << (width - 1);
    (-half, half - 1)
}

/// One integer element spelled at `dtype`. The minimum is written as an
/// expression, since its magnitude is not a literal of the dtype.
fn int_element(value: i128, dtype: &str, width: u32) -> String {
    if value == bounds(width).0 {
        format!("sub({}{dtype}, 1{dtype})", value + 1)
    } else {
        format!("{value}{dtype}")
    }
}

/// An integer source's edges: its own extremes and every target's extremes
/// and their neighbours that the source can hold.
fn int_edges(width: u32) -> Vec<i128> {
    let (min, max) = bounds(width);
    let mut values = vec![min, min + 1, -1, 0, 1, max - 1, max];
    for (_, target) in INTS {
        let (lo, hi) = bounds(target);
        values.extend([lo - 1, lo, lo + 1, hi - 1, hi, hi + 1]);
    }
    values.retain(|value| (min..=max).contains(value));
    values.sort_unstable();
    values.dedup();
    values
}

/// A float source's edges, each exact at `dtype`, as (spelling, value). The
/// infinities are computed, since they have no literal.
fn float_edges(dtype: &str) -> Vec<(String, f64)> {
    let mut values = vec![
        -256.0, -129.0, -128.0, -127.5, -1.5, -0.5, 0.0, 0.5, 1.5, 127.5, 128.0, 256.0,
    ];
    values.extend(match dtype {
        // f16 tops out at 65504; 60000 is exact and beyond i16.
        "f16" => vec![-60000.0, 60000.0],
        _ => vec![-1e30, -2147483648.0, 2147483648.0, 1e30],
    });
    if dtype == "f64" {
        // 2^63 is the first f64 above i64::MAX; the next one down is in range.
        values.extend([
            -9223372036854775808.0,
            9223372036854774784.0,
            9223372036854775808.0,
        ]);
    }
    let mut spelled: Vec<(String, f64)> = values
        .into_iter()
        .map(|value| (format!("{value:?}{dtype}"), value))
        .collect();
    spelled.push((format!("div(1.0{dtype}, 0.0{dtype})"), f64::INFINITY));
    spelled.push((format!("div(-1.0{dtype}, 0.0{dtype})"), f64::NEG_INFINITY));
    spelled
}

fn saturate(value: f64, width: u32) -> i128 {
    let (lo, hi) = bounds(width);
    if value == f64::INFINITY {
        hi
    } else if value == f64::NEG_INFINITY {
        lo
    } else {
        (value.trunc() as i128).clamp(lo, hi)
    }
}

fn wrap(value: i128, width: u32) -> i128 {
    let modulus = 1i128 << width;
    let residue = value.rem_euclid(modulus);
    if residue >= modulus / 2 {
        residue - modulus
    } else {
        residue
    }
}

fn shown(values: &[i128]) -> String {
    let items: Vec<String> = values.iter().map(i128::to_string).collect();
    format!("[{}]", items.join(", "))
}

/// One program per (operation, source): every target, on every surface,
/// with the oracle's rendering of each root.
fn program(
    op: &str,
    source: &str,
    elements: &[String],
    expected: &[Vec<i128>],
) -> (String, String) {
    let list = format!("[{}]", elements.join(", "));
    let n = elements.len();
    let mut text = format!(
        "module Demo.Main\n\
         def rebuilt[n](t: &tensor[n, {source}]) -> tensor[n, {source}] = to_tensor(to_list(t))\n\
         x = to_tensor({list})\n"
    );
    let mut want = String::new();
    for ((target, _), values) in INTS.iter().zip(expected) {
        text.push_str(&format!(
            "def g_{target}[n](t: tensor[n, {source}]) -> tensor[n, {target}] = {op}(t, {target})\n\
             def s_{target}(v: {source}) -> {target} = {op}(v, {target})\n\
             top_{target} = to_list({op}(&x, {target}))\n\
             graph_{target} = to_list(g_{target}(to_tensor({list})))\n\
             host_{target} = to_list({op}(rebuilt(to_tensor({list})), {target}))\n\
             scalars_{target} = map(fn (i: i64) -> s_{target}(index(to_list(x), i)), range(0i64, {n}i64))\n"
        ));
        let values = shown(values);
        for root in ["top", "graph", "host", "scalars"] {
            want.push_str(&format!("{root}_{target} = {values}\n"));
        }
    }
    (text, want)
}

/// Every admitted pair: `cast_saturate` from each signed integer and float,
/// `cast_wrap` from each signed integer, each to every signed integer.
fn cases() -> Vec<(String, String, String)> {
    let mut cases = Vec::new();
    for (source, width) in INTS {
        let edges = int_edges(width);
        let elements: Vec<String> = edges
            .iter()
            .map(|v| int_element(*v, source, width))
            .collect();
        for op in ["cast_saturate", "cast_wrap"] {
            let expected: Vec<Vec<i128>> = INTS
                .iter()
                .map(|(_, target)| {
                    edges
                        .iter()
                        .map(|v| match op {
                            "cast_wrap" => wrap(*v, *target),
                            _ => (*v).clamp(bounds(*target).0, bounds(*target).1),
                        })
                        .collect()
                })
                .collect();
            let (text, want) = program(op, source, &elements, &expected);
            cases.push((format!("{op}_{source}"), text, want));
        }
    }
    for source in FLOATS {
        let edges = float_edges(source);
        let elements: Vec<String> = edges.iter().map(|(spelled, _)| spelled.clone()).collect();
        let expected: Vec<Vec<i128>> = INTS
            .iter()
            .map(|(_, target)| edges.iter().map(|(_, v)| saturate(*v, *target)).collect())
            .collect();
        let (text, want) = program("cast_saturate", source, &elements, &expected);
        cases.push((format!("cast_saturate_{source}"), text, want));
    }
    cases
}

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

fn evaluated(source: &str) -> String {
    let result = eval(request(source))
        .unwrap_or_else(|error| panic!("evaluator rejected the case: {error:?}"));
    result
        .roots
        .iter()
        .filter(|root| root.name.as_deref() != Some("x"))
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

fn refusal(source: &str) -> String {
    match eval(request(source)) {
        Ok(_) => String::new(),
        Err(refused) => refused
            .errors
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

// REGRESSION TEST. On `4bb166024` neither name was bound ("unbound
// variable `cast_saturate`").
#[test]
fn every_admitted_pair_evaluates_to_the_oracle() {
    let mut failures = Vec::new();
    for (name, source, want) in cases() {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            assert_eq!(evaluated(&source), want, "eval differs from the oracle");
        }));
        if let Err(payload) = outcome {
            let head: String = panic_message(payload).chars().take(1200).collect();
            failures.push(format!("{name}: {head}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn every_admitted_pair_compiles_as_eval_runs_it() {
    let mut failures = Vec::new();
    for (name, source, _) in cases() {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let expected = evaluated(&source);
            let generated = ownership_support::emit(&source, &name);
            let (summary, stdout) = ownership_support::run_program(&generated);
            ownership_support::balanced(&summary);
            let compiled: String = stdout
                .lines()
                .filter(|line| !line.starts_with("x = "))
                .map(|line| format!("{line}\n"))
                .collect();
            assert_eq!(compiled, expected, "compiled output differs from eval");
        }));
        if let Err(payload) = outcome {
            let head: String = panic_message(payload).chars().take(1200).collect();
            failures.push(format!("{name}: {head}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// [05-OP-23]: NaN traps `Domain` as operation `cast_saturate` at the target
/// dtype, on every surface and in both lanes.
#[test]
fn saturate_traps_domain_on_nan_in_both_lanes() {
    for (surface, body) in [
        (
            "dag_tensor",
            "to_list(cast_saturate(sqrt(to_tensor([1.0f32, -1.0f32])), i16))",
        ),
        (
            "host_tensor",
            "to_list(cast_saturate(to_tensor([1e30f32, div(0.0f32, 0.0f32)]), i16))",
        ),
        ("scalar", "cast_saturate(div(0.0f64, 0.0f64), i16)"),
    ] {
        let source = format!("module Demo.Main\nout = {body}\n");
        let refused = match eval(request(&source)) {
            Ok(result) => panic!("{surface}: NaN must trap, got {:?}", result.roots),
            Err(refused) => refused
                .errors
                .iter()
                .map(|diagnostic| diagnostic.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        assert!(
            refused.contains("numeric trap: domain in cast_saturate at i16"),
            "{surface}: eval reported {refused}"
        );
        let generated = ownership_support::emit(&source, &format!("nan_{surface}"));
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        assert!(
            stderr.contains("numeric trap: domain in cast_saturate at i16"),
            "{surface}: compiled program reported {stderr}"
        );
    }
}

/// The pairs each atom refuses, each beside the admitted twin it differs from
/// by one dtype.
#[test]
fn refused_pairs_are_type_errors() {
    for (call, admitted, fragment) in [
        (
            "cast_saturate(true, i32)",
            "cast_saturate(1i32, i32)",
            "`cast_saturate` source `bool`",
        ),
        (
            "cast_saturate(1.5f32, f64)",
            "cast_saturate(1.5f32, i64)",
            "`cast_saturate` target `f64`",
        ),
        (
            "cast_saturate(1i32, bool)",
            "cast_saturate(1i32, i8)",
            "`cast_saturate` target `bool`",
        ),
        (
            "cast_saturate(\"1\", i32)",
            "cast_saturate(1i64, i32)",
            "cast",
        ),
        (
            "cast_wrap(1.5f32, i32)",
            "cast_wrap(1i64, i32)",
            "`cast_wrap` source `f32`",
        ),
        (
            "cast_wrap(true, i8)",
            "cast_wrap(1i16, i8)",
            "`cast_wrap` source `bool`",
        ),
        (
            "cast_wrap(1i32, f32)",
            "cast_wrap(1i32, i16)",
            "`cast_wrap` target `f32`",
        ),
        (
            "cast_wrap(to_tensor([1.0f64]), i32)",
            "cast_wrap(to_tensor([1i64]), i32)",
            "`cast_wrap` source `f64`",
        ),
    ] {
        let refused = refusal(&format!("module Demo.Main\nout = {call}\n"));
        assert!(refused.contains(fragment), "{call}: {refused:?}");
        assert_eq!(
            refusal(&format!("module Demo.Main\nout = {admitted}\n")),
            "",
            "{admitted} is admitted"
        );
    }
}

/// [05-OP-23]: `cast_saturate` is non-differentiable, so `grad` refuses a
/// gradient through it rather than returning a silent zero. `cast_wrap` reads
/// only integers, which carry no cotangent; its structural rejection is pinned
/// in `chelis-ir`'s grad tests.
#[test]
fn grad_through_cast_saturate_is_refused() {
    let program = "module Demo.Main\n\
                   def loss(x: f32) -> f32 = cast(cast_saturate(x, i32), f32)\n\
                   out = grad(loss)(2.5f32)\n";
    let refused = refusal(program);
    assert!(
        refused.contains("cast_saturate")
            && (refused.contains("non-differentiable") || refused.contains("piecewise constant")),
        "grad must be refused naming the rung and why: {refused:?}"
    );
}
