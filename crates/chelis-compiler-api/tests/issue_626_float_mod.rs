//! chelis#626: float `mod` is C `fmod` in both lanes.
//!
//! [05-OP-64] makes float `mod` the exact remainder with the quotient
//! truncated toward zero: the dividend's sign, a signed zero kept, NaN for an
//! infinite dividend, a zero divisor or a NaN operand, the dividend for an
//! infinite divisor, and `f16`/`bf16` computed at `f32` and narrowed once.
//! Each dtype runs over scalars, a top-level tensor, a definition (the typed
//! graph lane) and operands computed on the host (the C host lane), and the
//! compiled program must print what `chelis eval` prints with every
//! allocation finalized.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// The smallest positive subnormal of each dtype.
fn subnormal(dtype: &str) -> &'static str {
    match dtype {
        "f16" => "5.960464477539063e-8",
        "bf16" => "9.183549615799121e-41",
        "f32" => "1.401298464324817e-45",
        _ => "5e-324",
    }
}

fn operands(dtype: &str) -> (String, String) {
    let tiny = format!("{}{dtype}", subnormal(dtype));
    let inf = format!("div(1.0{dtype}, 0.0{dtype})");
    let nan = format!("div(0.0{dtype}, 0.0{dtype})");
    let large = if matches!(dtype, "f16" | "bf16") {
        "1024.0"
    } else {
        "1e30"
    };
    let lhs = [
        format!("5.5{dtype}"),
        format!("-5.5{dtype}"),
        format!("5.5{dtype}"),
        format!("-5.5{dtype}"),
        format!("7.0{dtype}"),
        inf.clone(),
        format!("3.0{dtype}"),
        format!("neg(0.0{dtype})"),
        format!("-4.0{dtype}"),
        nan.clone(),
        format!("2.0{dtype}"),
        format!("mul({tiny}, 7.0{dtype})"),
        format!("1.0{dtype}"),
        format!("{large}{dtype}"),
    ];
    let rhs = [
        format!("3.0{dtype}"),
        format!("3.0{dtype}"),
        format!("-3.0{dtype}"),
        format!("-3.0{dtype}"),
        format!("0.0{dtype}"),
        format!("2.0{dtype}"),
        inf,
        format!("1.0{dtype}"),
        format!("2.0{dtype}"),
        format!("2.0{dtype}"),
        nan,
        format!("mul({tiny}, 3.0{dtype})"),
        tiny.clone(),
        format!("7.0{dtype}"),
    ];
    (
        format!("[{}]", lhs.join(", ")),
        format!("[{}]", rhs.join(", ")),
    )
}

/// The [05-OP-64] values, in the evaluator's rendering.
fn expected(dtype: &str) -> String {
    let (tiny, last) = match dtype {
        "f16" => ("6e-8", "2.0"),
        "bf16" => ("9e-41", "2.0"),
        "f32" => ("1e-45", "1.0"),
        _ => ("5e-324", "5.0"),
    };
    format!("[2.5, -2.5, 2.5, -2.5, NaN, NaN, 3.0, -0.0, -0.0, NaN, NaN, {tiny}, 0.0, {last}]")
}

fn program(dtype: &str) -> String {
    let (lhs, rhs) = operands(dtype);
    format!(
        "module Demo.Main\n\
         def rebuilt[n](t: &tensor[n, {dtype}]) -> tensor[n, {dtype}] = to_tensor(to_list(t))\n\
         def apply[n](x: tensor[n, {dtype}], y: tensor[n, {dtype}]) -> tensor[n, {dtype}] = mod(&x, &y)\n\
         def scalar(x: {dtype}, y: {dtype}) -> {dtype} = mod(x, y)\n\
         x = to_tensor({lhs})\n\
         y = to_tensor({rhs})\n\
         top = to_list(mod(&x, &y))\n\
         graph = to_list(apply(to_tensor({lhs}), to_tensor({rhs})))\n\
         host = to_list(mod(rebuilt(to_tensor({lhs})), rebuilt(to_tensor({rhs}))))\n\
         scalars = map(fn (i: i64) -> scalar(index(to_list(x), i), index(to_list(y), i)), range(0i64, 14i64))\n"
    )
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
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

// REGRESSION TEST. On `4bb166024` the checker refused every float `mod`
// ("mod requires matching integer arguments, got f32 and f32").
#[test]
fn float_mod_gives_the_fmod_values_in_eval() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let shown = evaluated(&program(dtype));
        let want = expected(dtype);
        for root in ["top", "graph", "host", "scalars"] {
            assert!(
                shown.contains(&format!("{root} = {want}\n")),
                "{dtype} {root}: want {want} in\n{shown}"
            );
        }
    }
}

#[test]
fn float_mod_compiles_as_eval_runs_it() {
    let mut failures = Vec::new();
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = program(dtype);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let expected = evaluated(&source);
            let generated = ownership_support::emit(&source, &format!("fmod_{dtype}"));
            let (summary, stdout) = ownership_support::run_program(&generated);
            ownership_support::balanced(&summary);
            assert_eq!(stdout, expected, "compiled output differs from eval");
        }));
        if let Err(payload) = outcome {
            let head: String = panic_message(payload).chars().take(600).collect();
            failures.push(format!("{dtype}: {head}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// [05-OP-64]: `mod` rejects differentiation at every dtype.
#[test]
fn float_mod_under_grad_is_refused() {
    let source = "module Demo.Main\n\
                  def loss(x: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mod(x, to_tensor([2.0f32, 3.0f32])), 0i32))\n\
                  out = grad(loss)(to_tensor([3.5f32, 4.0f32]))\n";
    let refused = eval(request(source)).expect_err("grad over mod must be refused");
    let refused = refused
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        refused.contains("mod is non-differentiable"),
        "eval reported {refused}"
    );
}
