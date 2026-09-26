//! spec/06 §2.10.1: a runtime `if` differentiates the executed branch, and
//! the untaken arm contributes nothing to a gradient. An arm lowered into a
//! `Where` is still computed where its activation is false (spec/10 §3), from
//! operands its checks accept (a float `div` reads 0 / 1, a draw yields
//! zeros), and its adjoints run too. Each contribution an arm's adjoint
//! produces for a cotangent outside the arm is therefore selected against an
//! exact zero by the arm's activation, so a non-finite local derivative of an
//! untaken arm, substituted or genuine, never reaches the gradient.
//!
//! Every row runs `main` through `eval_selected` (a Host-lane root: the host
//! interpreter, which runs the gradient graph as a kernel) and in the native
//! C of the whole program, and where the gradient takes one tensor, a
//! Tensor-lane twin `selected(x)` in the DAG evaluator itself; each must
//! print exactly the expected line. The `chelis eval --file` rows are the
//! same shapes in `chelis-cli`'s `issue_2563_untaken_arm_eval_file`. The expected values are
//! derived by hand: `d/dx x = 1`, `d/dx x log(x) = log(x) + 1`,
//! `d/dx log(x - 1) = 1 / (x - 1)` and `d/dx x log(x - 1) = log(x - 1) +
//! x / (x - 1)`, at f32.
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

mod ownership_support;

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_types::types::Lane;
use std::collections::BTreeMap;

/// An untaken arm whose float `div` computes from substituted operands:
/// `log(0 / 1)` is `-inf`, and the product's adjoint for `x` is `0 * -inf`.
const DIV_SUBSTITUTED: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(div(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

/// An untaken arm whose draw yields zeros: `log(0)` is `-inf`.
const DRAW_SUBSTITUTED: &str = "def loss(x: tensor[4, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(copy(x), 0i32))
  r = if lt(100.0f32, s) then mul(copy(x), log(uniform_like(key_from_seed(1i64), copy(x), 1.0f32, 2.0f32))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[4, f32] = grad(loss)(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
";

/// chelis#2640: an untaken arm whose derivative is genuinely not finite at
/// the primal, `log(x - 1)` at `x = 1`, with no substituted operand.
const GENUINE_LOG: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then log(sub(&x, to_tensor([1.0f32]))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

/// The same through a product, whose adjoint multiplies the zero cotangent
/// by `log(0)`.
const GENUINE_PRODUCT: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(sub(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

/// Per-row activations under `vmap(grad(...))`: row 0 leaves the arm
/// untaken, row 1 takes it.
const VMAP_GRAD_PRODUCT: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(sub(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[2, 1, f32] = vmap(grad(loss))(to_tensor([[1.0f32], [10.0f32]]))
";

const VMAP_GRAD_DIV_SUBSTITUTED: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(5.0f32, s) then mul(&x, log(div(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[2, 1, f32] = vmap(grad(loss))(to_tensor([[1.0f32], [10.0f32]]))
";

/// Per-row activations in the graph `grad` differentiates: a `vmap` whose
/// body branches, inside the loss.
const GRAD_OF_VMAPPED_ARM: &str = "def h(x: tensor[f32]) -> tensor[f32] = if lt(5.0f32, tensor_to_scalar(copy(x))) then mul(&x, log(sub(&x, scalar_to_tensor(1.0f32)))) else copy(x)
def loss(xs: tensor[2, f32]) -> tensor[f32] = sum(vmap(h)(xs), 0i32)
def main() -> tensor[2, f32] = grad(loss)(to_tensor([1.0f32, 10.0f32]))
";

/// A value the vmapped body captures, differentiated through the arm:
/// row 0 contributes `x = 1`, row 1 `log(9)`.
const GRAD_OF_VMAPPED_CAPTURE: &str = "def loss(xs: tensor[2, f32], w: tensor[f32]) -> tensor[f32] = {
  ys = vmap(fn (x: tensor[f32]) -> if lt(5.0f32, tensor_to_scalar(copy(x))) then mul(log(sub(&x, scalar_to_tensor(1.0f32))), copy(w)) else mul(x, copy(w)))(xs)
  sum(ys, 0i32)
}
def main() -> tensor[f32] = grad(loss, wrt=w)(to_tensor([1.0f32, 10.0f32]), scalar_to_tensor(2.0f32))
";

/// Nested per-row activations: the outer row 1 leaves its arm untaken, and
/// its inner `vmap` computes `log` of negative values there.
const GRAD_OF_NESTED_VMAPPED_ARMS: &str = "def h(x: tensor[f32]) -> tensor[f32] = if lt(5.0f32, tensor_to_scalar(copy(x))) then mul(&x, log(sub(&x, scalar_to_tensor(1.0f32)))) else copy(x)
def row(xs: tensor[2, f32]) -> tensor[2, f32] = if lt(0.0f32, tensor_to_scalar(sum(copy(xs), 0i32))) then vmap(h)(xs) else mul(xs, to_tensor([3.0f32, 3.0f32]))
def loss(xss: tensor[2, 2, f32]) -> tensor[f32] = sum(sum(vmap(row)(xss), 0i32), 0i32)
def main() -> tensor[2, 2, f32] = grad(loss)(to_tensor([[1.0f32, 10.0f32], [-1.0f32, -2.0f32]]))
";

/// A taken arm, `x log(x)` at `x = 1`.
const TAKEN_DIV: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then mul(&x, log(div(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

/// A taken arm whose derivative is genuinely infinite: `1 / (x - 1)` at 1.
const TAKEN_GENUINE_LOG: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then log(sub(&x, to_tensor([1.0f32]))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

/// A taken arm whose derivative is genuinely NaN: `-inf + inf` at 1.
const TAKEN_GENUINE_PRODUCT: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then mul(&x, log(sub(&x, to_tensor([1.0f32])))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[1, f32] = grad(loss)(to_tensor([1.0f32]))
";

/// Per row: row 0 takes the infinite arm, row 1 does not.
const VMAP_GRAD_TAKEN_GENUINE_LOG: &str = "def loss(x: tensor[1, f32]) -> tensor[f32] = {
  s = tensor_to_scalar(sum(&x, 0i32))
  r = if lt(s, 5.0f32) then log(sub(&x, to_tensor([1.0f32]))) else copy(x)
  sum(r, 0i32)
}
def main() -> tensor[2, 1, f32] = vmap(grad(loss))(to_tensor([[1.0f32], [10.0f32]]))
";

/// One row: `main`'s expected line, from the host interpreter through
/// `eval_selected` and from the whole program's C, and, where the gradient
/// takes one tensor, its Tensor-lane twin.
struct Row {
    label: &'static str,
    source: &'static str,
    main: &'static str,
    selected: Option<Selected>,
}

/// A Tensor-lane root `selected(x)` appended to the row's source and
/// evaluated at a bound `x`, so the DAG evaluator runs the gradient graph
/// itself.
struct Selected {
    def: &'static str,
    shape: &'static [i64],
    x: &'static [f32],
    expected: &'static str,
}

const ONE: Selected = Selected {
    def: "def selected(x: tensor[1, f32]) -> tensor[1, f32] = grad(loss)(x)\n",
    shape: &[1],
    x: &[1.0],
    expected: "selected = tensor(shape=[1], data=[1.0])",
};

/// An untaken arm contributes nothing: each row returns the else arm's
/// gradient alone.
const UNTAKEN: [Row; 9] = [
    Row {
        label: "div substituted",
        source: DIV_SUBSTITUTED,
        main: "main = tensor(shape=[1], data=[1.0])",
        selected: Some(ONE),
    },
    Row {
        label: "draw substituted",
        source: DRAW_SUBSTITUTED,
        main: "main = tensor(shape=[4], data=[1.0, 1.0, 1.0, 1.0])",
        selected: Some(Selected {
            def: "def selected(x: tensor[4, f32]) -> tensor[4, f32] = grad(loss)(x)\n",
            shape: &[4],
            x: &[1.0, 1.0, 1.0, 1.0],
            expected: "selected = tensor(shape=[4], data=[1.0, 1.0, 1.0, 1.0])",
        }),
    },
    Row {
        label: "genuine log",
        source: GENUINE_LOG,
        main: "main = tensor(shape=[1], data=[1.0])",
        selected: Some(ONE),
    },
    Row {
        label: "genuine product",
        source: GENUINE_PRODUCT,
        main: "main = tensor(shape=[1], data=[1.0])",
        selected: Some(ONE),
    },
    Row {
        label: "vmap(grad) product",
        source: VMAP_GRAD_PRODUCT,
        main: "main = tensor(shape=[2, 1], data=[1.0, 3.3083358])",
        selected: Some(Selected {
            def: "def selected(x: tensor[2, 1, f32]) -> tensor[2, 1, f32] = vmap(grad(loss))(x)\n",
            shape: &[2, 1],
            x: &[1.0, 10.0],
            expected: "selected = tensor(shape=[2, 1], data=[1.0, 3.3083358])",
        }),
    },
    Row {
        label: "vmap(grad) div substituted",
        source: VMAP_GRAD_DIV_SUBSTITUTED,
        main: "main = tensor(shape=[2, 1], data=[1.0, 3.3025851])",
        selected: None,
    },
    Row {
        label: "grad of a vmapped arm",
        source: GRAD_OF_VMAPPED_ARM,
        main: "main = tensor(shape=[2], data=[1.0, 3.3083358])",
        selected: Some(Selected {
            def: "def selected(x: tensor[2, f32]) -> tensor[2, f32] = grad(loss)(x)\n",
            shape: &[2],
            x: &[1.0, 10.0],
            expected: "selected = tensor(shape=[2], data=[1.0, 3.3083358])",
        }),
    },
    Row {
        label: "grad of a vmapped capture",
        source: GRAD_OF_VMAPPED_CAPTURE,
        main: "main = 3.1972246",
        selected: None,
    },
    Row {
        label: "grad of nested vmapped arms",
        source: GRAD_OF_NESTED_VMAPPED_ARMS,
        main: "main = tensor(shape=[2, 2], data=[1.0, 3.3083358, 3.0, 3.0])",
        selected: None,
    },
];

/// A taken arm keeps its true gradient, including a non-finite one: the
/// mask selects on the activation and never hides a value the executed
/// branch computes.
const TAKEN: [Row; 4] = [
    Row {
        label: "taken div",
        source: TAKEN_DIV,
        main: "main = tensor(shape=[1], data=[1.0])",
        selected: Some(ONE),
    },
    Row {
        label: "taken genuine log",
        source: TAKEN_GENUINE_LOG,
        main: "main = tensor(shape=[1], data=[inf])",
        selected: Some(Selected {
            expected: "selected = tensor(shape=[1], data=[inf])",
            ..ONE
        }),
    },
    Row {
        label: "taken genuine product",
        source: TAKEN_GENUINE_PRODUCT,
        main: "main = tensor(shape=[1], data=[NaN])",
        selected: Some(Selected {
            expected: "selected = tensor(shape=[1], data=[NaN])",
            ..ONE
        }),
    },
    Row {
        label: "vmap(grad) taken genuine log",
        source: VMAP_GRAD_TAKEN_GENUINE_LOG,
        main: "main = tensor(shape=[2, 1], data=[inf, 1.0])",
        selected: None,
    },
];

/// The lines `eval_selected` reports for `root`, and the lane it ran in.
fn eval_lines(
    source: &str,
    root: &str,
    bindings: BTreeMap<String, TensorValue>,
) -> Result<(Vec<String>, Option<Lane>), String> {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings,
        },
        &[root.into()],
    )
    .map_err(|error| format!("{:?}", error.errors))?;
    let lane = result
        .manifest
        .entries
        .iter()
        .find(|entry| entry.name == root)
        .map(|entry| entry.lane);
    let lines = result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}",
                root.name.as_deref().unwrap_or("<unnamed>"),
                root.display.as_deref().unwrap_or("<no display>")
            )
        })
        .collect();
    Ok((lines, lane))
}

/// The lines the native C program prints for its roots.
fn c_lines(source: &str, label: &str) -> Vec<String> {
    let generated = ownership_support::emit(source, label);
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    stdout.lines().map(str::to_string).collect()
}

/// Every row in every lane, collected before the assertion so one red row
/// never hides a sibling.
fn check(rows: &[Row]) {
    let mut failures = Vec::new();
    for row in rows {
        let label = row.label;
        let expected = vec![row.main.to_string()];
        match eval_lines(row.source, "main", BTreeMap::new()) {
            Ok((lines, _)) if lines == expected => {}
            outcome => failures.push(format!("{label}: host {outcome:?}")),
        }
        let native = c_lines(row.source, label);
        if native != expected {
            failures.push(format!("{label}: C {native:?}"));
        }
        if let Some(selected) = &row.selected {
            let bindings = BTreeMap::from([(
                "x".to_string(),
                TensorValue {
                    shape: selected.shape.to_vec(),
                    data: wire_values::storage_f32(selected.x.to_vec()),
                },
            )]);
            let source = format!("{}{}", row.source, selected.def);
            match eval_lines(&source, "selected", bindings) {
                Ok((lines, Some(Lane::Tensor))) if lines == [selected.expected] => {}
                outcome => failures.push(format!("{label}: DAG evaluator {outcome:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Evidentiary status: REGRESSION TEST for the substituted rows ("div
/// substituted", "draw substituted", "vmap(grad) div substituted"), NaN at
/// b47fdd7d3, where an untaken arm's substituted operands reached the
/// gradient through its adjoints; and for the genuine rows, NaN on main
/// 7807ca4ff (chelis#2640). The fail-first run is recorded per row in the
/// pull request.
#[test]
fn an_untaken_arm_contributes_nothing_to_a_gradient_in_the_evaluator_and_c() {
    check(&UNTAKEN);
}

/// Evidentiary status: DISPOSITION LOCK (the taken rows hold at b47fdd7d3);
/// it pins that the mask selects on the activation rather than hiding
/// non-finite values.
#[test]
fn a_taken_arm_keeps_its_gradient_in_the_evaluator_and_c() {
    check(&TAKEN);
}
