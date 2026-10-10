//! chelis#3397: `grad` of a function that takes a function-valued argument.
//!
//! `spec/06-transformations.md` section 2.1 gives a function component the
//! gradient type `unit`, and section 2.2 treats a parameter outside `wrt` as
//! a constant of the differentiated call. The evaluator staged every `grad`
//! argument as a tensor input and failed at run time on a function value
//! ("grad/vmap argument 0 must be a tensor or scalar, got Closure { ... }"),
//! and the gradient lowering had no way to pass a callable to the body.
//!
//! Oracle: each program prints the `chelis eval` rendering pinned here, and
//! compiled to C and run against the `ownership-ledger` runtime it prints the
//! same with every allocation finalized. The negative twin names the
//! function parameter in `wrt`, which spec/06 section 2.7 refuses.

mod ownership_support;

use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "module Demo.Main\n\
type Lin[i] =\n  | Lin { w: tensor[i, f32] }\n\
def square_sum(t: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(t, t), 0i32))\n\
def apply_loss(f: (tensor[2, f32]) -> f32, p: tensor[2, f32]) -> f32 = f(p)\n\
def apply_s(f: (f32) -> f32, x: f32) -> f32 = f(x)\n\
def cube_sum(t: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(mul(t, t), t), 0i32))\n\
def apply_ab(f: (tensor[2, f32]) -> f32, a: tensor[2, f32], b: tensor[3, f32]) -> f32 = add(f(a), tensor_to_scalar(sum(mul(b, b), 0i32)))\n\
def lin_loss(p: Lin[2], x: tensor[2, f32]) -> f32 = match p with {\n  | Lin { w } => tensor_to_scalar(sum(mul(w, x), 0i32))\n}\n\
def through[M](loss: M -> tensor[2, f32] -> f32, p: M, x: tensor[2, f32]) -> f32 = loss(p, x)\n";

struct Case {
    name: &'static str,
    body: &'static str,
    eval: &'static str,
}

const POSITIVE: &[Case] = &[
    // The issue's reproducer: d/dp sum(p * p) = 2p.
    Case {
        name: "declared_function_argument",
        body: "g = grad(apply_loss, wrt=p)(square_sum, to_tensor([1.0f32, 2.0f32]))\n",
        eval: "g = tensor(shape=[2], data=[2.0, 4.0])\n",
    },
    // Without `wrt`, only the differentiable parameter is selected.
    Case {
        name: "default_wrt_skips_the_function",
        body: "g = grad(apply_loss)(square_sum, to_tensor([1.0f32, 2.0f32]))\n",
        eval: "g = tensor(shape=[2], data=[2.0, 4.0])\n",
    },
    // An anonymous function argument reads a captured tensor.
    Case {
        name: "closure_argument",
        body: "w = to_tensor([3.0f32, 5.0f32])\ng = grad(apply_loss, wrt=p)(fn (t: tensor[2, f32]) -> tensor_to_scalar(sum(mul(t, w), 0i32)), to_tensor([1.0f32, 2.0f32]))\n",
        eval: "w = tensor(shape=[2], data=[3.0, 5.0])\ng = tensor(shape=[2], data=[3.0, 5.0])\n",
    },
    // A concrete data-value target beside the function argument.
    Case {
        name: "data_value_target",
        body: "def through_lin(loss: Lin[2] -> tensor[2, f32] -> f32, p: Lin[2], x: tensor[2, f32]) -> f32 = loss(p, x)\np0 = Lin { w: to_tensor([1.0f32, 2.0f32]) }\nx0 = to_tensor([3.0f32, 4.0f32])\ng = grad(through_lin, wrt=p)(lin_loss, p0, x0)\n",
        eval: "p0.w = tensor(shape=[2], data=[1.0, 2.0])\nx0 = tensor(shape=[2], data=[3.0, 4.0])\ng = Lin(tensor(shape=[2], data=[3.0, 4.0]))\n",
    },
    // The repair the `vmap` diagnostic names: the mapped function closes
    // over the function instead of receiving it.
    Case {
        name: "vmap_closing_over_the_function",
        body: "v = vmap(fn (x: tensor[2, f32]) -> apply_loss(square_sum, x))(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n",
        eval: "v = tensor(shape=[2], data=[5.0, 25.0])\n",
    },
    // A closure argument capturing a value that is not top-level: a
    // parameter of the enclosing def. d/dx (x * x * c) = 2xc = 20 at c = 5,
    // x = 2; the outer gradients are d/dx 2xc = 2c = 10 and d/dc 2xc = 2x = 4.
    Case {
        name: "closure_capturing_a_parameter",
        body: "def d_apply(c: f32, x: f32) -> f32 = grad(apply_s, wrt=x)(fn (y: f32) -> mul(mul(y, y), c), x)\nv = d_apply(5.0f32, 2.0f32)\ndx = grad(d_apply, wrt=x)(5.0f32, 2.0f32)\ndc = grad(d_apply, wrt=c)(5.0f32, 2.0f32)\n",
        eval: "v = 20.0\ndx = 10.0\ndc = 4.0\n",
    },
    // The tensor form: d/dp sum(p * p * w) = 2pw = [6, 20] at w = [3, 5],
    // p = [1, 2]; through a further grad, d/dw sum(2pw) = 2p = [2, 4].
    Case {
        name: "closure_capturing_a_tensor_parameter",
        body: "def outer(w: tensor[2, f32], p: tensor[2, f32]) -> tensor[2, f32] = grad(apply_loss, wrt=p)(fn (t: tensor[2, f32]) -> tensor_to_scalar(sum(mul(mul(t, t), w), 0i32)), p)\ndef total(w: tensor[2, f32], p: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(outer(w, p), 0i32))\ng = outer(to_tensor([3.0f32, 5.0f32]), to_tensor([1.0f32, 2.0f32]))\ngw = grad(total, wrt=w)(to_tensor([3.0f32, 5.0f32]), to_tensor([1.0f32, 2.0f32]))\n",
        eval: "g = tensor(shape=[2], data=[6.0, 20.0])\ngw = tensor(shape=[2], data=[2.0, 4.0])\n",
    },
    // The closure captures the very parameter that is the `wrt` target; the
    // capture is a constant of the call, so d/dp sum(p * p_captured) is the
    // captured value, [1, 2].
    Case {
        name: "closure_capturing_the_wrt_parameter",
        body: "def self_cap(p: tensor[2, f32]) -> tensor[2, f32] = grad(apply_loss, wrt=p)(fn (t: tensor[2, f32]) -> tensor_to_scalar(sum(mul(t, p), 0i32)), p)\ng = self_cap(to_tensor([1.0f32, 2.0f32]))\n",
        eval: "g = tensor(shape=[2], data=[1.0, 2.0])\n",
    },
    // A function-first `grad` inside a def body, for each `wrt` form, and a
    // multi-name `wrt` written after the function: d/da sum(a^3) = 3a^2 =
    // [3, 12] and d/db sum(b^2) = 2b = [6, 10, 14].
    Case {
        name: "function_first_grad_inside_a_def",
        body: "def inside_b(a: tensor[2, f32], b: tensor[3, f32]) -> tensor[3, f32] = grad(apply_ab, wrt=b)(cube_sum, a, b)\ndef inside_a(a: tensor[2, f32], b: tensor[3, f32]) -> tensor[2, f32] = grad(apply_ab, wrt=a)(cube_sum, a, b)\ndef inside_d(a: tensor[2, f32], b: tensor[3, f32]) -> (tensor[2, f32], tensor[3, f32]) = grad(apply_ab)(cube_sum, a, b)\ngb = inside_b(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 5.0f32, 7.0f32]))\nga = inside_a(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 5.0f32, 7.0f32]))\ngd = inside_d(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 5.0f32, 7.0f32]))\n",
        eval: "gb = tensor(shape=[3], data=[6.0, 10.0, 14.0])\nga = tensor(shape=[2], data=[3.0, 12.0])\ngd.0 = tensor(shape=[2], data=[3.0, 12.0])\ngd.1 = tensor(shape=[3], data=[6.0, 10.0, 14.0])\n",
    },
    Case {
        name: "multi_name_wrt_after_the_function",
        body: "gab = grad(apply_ab, wrt=(a, b))(cube_sum, to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 5.0f32, 7.0f32]))\ngba = grad(apply_ab, wrt=(b, a))(cube_sum, to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 5.0f32, 7.0f32]))\n",
        eval: "gab.0 = tensor(shape=[2], data=[3.0, 12.0])\ngab.1 = tensor(shape=[3], data=[6.0, 10.0, 14.0])\ngba.0 = tensor(shape=[3], data=[6.0, 10.0, 14.0])\ngba.1 = tensor(shape=[2], data=[3.0, 12.0])\n",
    },
    // A type-generic formal instantiated at a tensor and at an f64 scalar:
    // d/dp sum(p^2) * sum(x) = 2p * 7 = [14, 28, 42], and d/dp p^2 x = 2px =
    // 30 at p = 3, x = 5.
    Case {
        name: "generic_formal_at_a_tensor_and_a_scalar",
        body: "def through_t[M](loss: M -> tensor[2, f32] -> f32, p: M, x: tensor[2, f32]) -> f32 = loss(p, x)\ndef through_s[M](loss: M -> f32 -> f32, p: M, x: f32) -> f32 = loss(p, x)\ndef tloss(p: tensor[3, f32], x: tensor[2, f32]) -> f32 = mul(tensor_to_scalar(sum(mul(p, p), 0i32)), tensor_to_scalar(sum(x, 0i32)))\ndef dloss(p: f64, x: f32) -> f32 = mul(cast(mul(p, p), f32), x)\ndef step(q: tensor[3, f32], x: tensor[2, f32]) -> tensor[3, f32] = grad(through_t, wrt=p)(tloss, q, x)\ngq = grad(through_t, wrt=p)(tloss, to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([3.0f32, 4.0f32]))\ngs = step(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([3.0f32, 4.0f32]))\ngd = grad(through_s, wrt=p)(dloss, 3.0f64, 5.0f32)\n",
        eval: "gq = tensor(shape=[3], data=[14.0, 28.0, 42.0])\ngs = tensor(shape=[3], data=[14.0, 28.0, 42.0])\ngd = 30.0\n",
    },
    // The issue's generic form: a type-generic def over a data-value target.
    Case {
        name: "generic_over_a_data_value",
        body: "p0 = Lin { w: to_tensor([1.0f32, 2.0f32]) }\nx0 = to_tensor([3.0f32, 4.0f32])\ng = grad(through, wrt=p)(lin_loss, p0, x0)\n",
        eval: "p0.w = tensor(shape=[2], data=[1.0, 2.0])\nx0 = tensor(shape=[2], data=[3.0, 4.0])\ng = Lin(tensor(shape=[2], data=[3.0, 4.0]))\n",
    },
];

/// spec/06 section 2.7: a function named in `wrt` is a `non_differentiable`
/// type error on both lanes.
const FUNCTION_IN_WRT: (&str, &str) = (
    "g = grad(apply_loss, wrt=f)(square_sum, to_tensor([1.0f32, 2.0f32]))\n",
    "grad `wrt` index 0 is not differentiable",
);

/// spec/06 section 3.6 broadcasts a function-valued `vmap` argument
/// unbatched; the evaluator refuses it with a diagnostic that names that rule
/// and the repair, and the C build refuses it too.
const VMAP_FUNCTION_ARGUMENT: (&str, &str) = (
    "v = vmap(apply_loss)(square_sum, to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]]))\n",
    "host runtime: `vmap(...)` argument 0 is a function value. \
     spec/06-transformations.md section 3.6 broadcasts a non-tensor argument unbatched to \
     every row, and this lane does not yet carry a function-valued `vmap` argument \
     (chelis#3523); close over the function inside the mapped function instead, as in \
     `vmap(fn (x) -> f(g, x))`.",
);

fn source(body: &str) -> String {
    format!("{PRELUDE}{body}")
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

fn check_positive(case: &Case) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        let source = source(case.body);
        let expected = evaluated(&source);
        assert_eq!(expected, case.eval, "eval output");
        let generated = ownership_support::emit(&source, case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{}\n  -> {head}", case.name, source(case.body))
    })
}

// REGRESSION TEST. On `2b8c9acf6` every positive program failed: eval with
// "grad/vmap argument 0 must be a tensor or scalar, got Closure { ... }",
// and the C build with its unresolved-`grad` refusal.
#[test]
fn grad_with_a_function_argument_compiles_as_eval_runs_it() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn a_function_named_in_wrt_is_refused_on_both_lanes() {
    let (body, message) = FUNCTION_IN_WRT;
    let source = source(body);
    let refused = eval(request(&source)).expect_err("the evaluator must refuse");
    let refused = format!("{refused:?}");
    assert!(refused.contains(message), "eval: {refused}");
    let refused = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source,
        target: CompileTarget::C,
        entry_name: Some("fixture".into()),
    })
    .expect_err("the C build must refuse");
    let refused = format!("{refused:?}");
    assert!(refused.contains(message), "C build: {refused}");
}

#[test]
fn a_function_valued_vmap_argument_is_refused_by_name() {
    let (body, message) = VMAP_FUNCTION_ARGUMENT;
    let source = source(body);
    let refused = eval(request(&source)).expect_err("the evaluator must refuse");
    let refused = format!("{refused:?}");
    assert!(refused.contains(message), "eval: {refused}");
    compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source,
        target: CompileTarget::C,
        entry_name: Some("fixture".into()),
    })
    .expect_err("the C build must refuse");
}
