//! chelis#3101: borrowed tensor operands to `mod`, the bitwise and shift
//! operations, `cast` and `cast_trunc` run in both lanes.
//!
//! spec/05 section 1.3.1 with [05-OP-47], [05-OP-64], [05-OP-63] and [05-OP-6]
//! makes each tensor operand a read-only `&tensor[D,p]` parameter, so
//! `op(&x, &y)` computes what `op(x, y)` computes and leaves `x` and `y`
//! readable. Each program compiles, runs against the `ownership-ledger`
//! runtime with every allocation finalized, and prints what `chelis eval`
//! prints.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn program(body: &str) -> String {
    format!("module Demo.Main\n{body}")
}

fn cases() -> Vec<(String, String)> {
    let mut cases = Vec::new();
    for op in ["mod", "bitand", "bitor", "bitxor", "shl", "shr"] {
        for dtype in ["i8", "i64"] {
            cases.push((
                format!("{op}_{dtype}"),
                program(&format!(
                    "def apply[n](x: tensor[n, {dtype}], y: tensor[n, {dtype}]) -> (tensor[n, {dtype}], tensor[n, {dtype}], tensor[n, {dtype}]) = ({op}(&x, &y), x, y)\n\
                     graph = apply(to_tensor([7{dtype}, -7{dtype}, 5{dtype}]), to_tensor([3{dtype}, 2{dtype}, 1{dtype}]))\n\
                     x = to_tensor([-6{dtype}, 9{dtype}, 4{dtype}])\n\
                     y = to_tensor([4{dtype}, 3{dtype}, 2{dtype}])\n\
                     top = to_list({op}(&x, &y))\n"
                )),
            ));
        }
    }
    cases.push((
        "casts".to_string(),
        program(
            "def apply(x: tensor[3, f32]) -> (tensor[3, f64], tensor[3, i32], tensor[3, f32]) = (cast(&x, f64), cast_trunc(&x, i32), x)\n\
             graph = apply(to_tensor([1.75f32, -2.5f32, 3.0f32]))\n\
             x = to_tensor([1.5f64, -0.5f64])\n\
             top = to_list(cast_trunc(&x, i8))\n\
             widened = to_list(cast(&x, f32))\n",
        ),
    ));
    // An owned source auto-borrows: every rung reads `x`, which stays live
    // for the borrows after it, and the identity cast leaves both the result
    // and `x` owned.
    cases.push((
        "owned_cast_sources".to_string(),
        program(
            "def apply(x: tensor[3, f32]) -> (tensor[3, f64], tensor[3, i32], tensor[3, i16], tensor[3, f32], tensor[3, f32]) = {\n\
             \x20 a = cast(x, f64)\n\
             \x20 b = cast_trunc(x, i32)\n\
             \x20 c = cast_saturate(x, i16)\n\
             \x20 same = cast(x, f32)\n\
             \x20 (a, b, c, same, add(&x, &x))\n\
             }\n\
             def wrap(y: tensor[3, i32]) -> (tensor[3, i8], tensor[3, i32]) = {\n\
             \x20 w = cast_wrap(y, i8)\n\
             \x20 (w, add(&y, &y))\n\
             }\n\
             graph = apply(to_tensor([1.75f32, -2.5f32, 40000.0f32]))\n\
             wrapped = wrap(to_tensor([200i32, -129i32, 5i32]))\n\
             x = to_tensor([1.5f64, -0.5f64])\n\
             top = to_list(cast_trunc(x, i8))\n\
             again = to_list(add(&x, &x))\n",
        ),
    ));
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

// REGRESSION TEST. On `4bb166024` the checker refused every borrowed
// operand here ("mod requires matching integer arguments, got &tensor[..]
// and &tensor[..]", "cast argument 1: expected tensor or numeric/bool
// scalar, got &tensor[..]"), and linearity refused a borrowed cast source
// ("borrow is only valid as a direct call argument").
#[test]
fn borrowed_operands_compile_as_eval_runs_them() {
    let mut failures = Vec::new();
    for (name, source) in cases() {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let expected = evaluated(&source);
            let generated = ownership_support::emit(&source, &name);
            let (summary, stdout) = ownership_support::run_program(&generated);
            ownership_support::balanced(&summary);
            assert_eq!(stdout, expected, "compiled output differs from eval");
        }));
        if let Err(payload) = outcome {
            let head: String = panic_message(payload).chars().take(600).collect();
            failures.push(format!("{name}:\n{source}  -> {head}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// The borrowed call computes the owned call's values.
#[test]
fn borrowed_and_owned_calls_agree() {
    let borrowed = evaluated(&program(
        "x = to_tensor([-7i32, 9i32])\ny = to_tensor([3i32, 2i32])\nr = to_list(mod(&x, &y))\n",
    ));
    assert!(borrowed.contains("r = [-1, 1]"), "{borrowed}");
}
