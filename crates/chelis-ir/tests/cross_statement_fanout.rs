//! Cross-statement aliasing and consuming fan-out. A var-RHS binding
//! (`alias = x`) shares a lowered node with `x`; later borrow uses
//! remain valid without an explicit copy. Consuming fan-out receives
//! copy insertion. These fixtures check linearity, lowered DAG copy
//! counts, evaluator results, and AD parity against explicit copies,
//! including top-level statements without a module wrapper.

use chelis_unord::UnordMap;

use chelis_ir::analysis::analyze_function_copy_cost;
use chelis_ir::dag::{Dag, NodeId, RiscOp};
use chelis_ir::eval::{TensorValue, eval_tensor_with};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::lower::try_lower_program;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::types::Prim;
use chelis_types::{check_linearity, check_typed_program};

/// Full Surf-to-DAG pipeline: parse Surf → desugar to Deep → typecheck →
/// effects → linearity → lower. Returns `Err(error_string)` for any stage
/// that fails so the linearity-only fixtures can pin the exact failure
/// mode.
fn surf_to_dag(source: &str) -> Result<Dag, String> {
    let decls = surf_parse(source).map_err(|e| format!("surf parse: {e:?}"))?;
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep)
        .map_err(|errs| format!("typecheck failed: {:?}", errs.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errs| format!("effects failed: {errs:?}"))?;
    let checked =
        check_linearity(&checked).map_err(|errs| format!("linearity failed: {errs:?}"))?;
    try_lower_program(&checked).map_err(|diag| format!("lowering failed: {diag:?}"))
}

fn find_def_root_by_name(dag: &Dag, name: &str) -> NodeId {
    // For a `def fanout(...) = body`, lowering registers `fanout` (or a
    // scalar-output Store named `fanout`) as a DAG root. Probe by Store
    // name first, otherwise pick the root whose subgraph corresponds to
    // this def via the symbol_table — easier here: just take the last
    // root, which is the most-recently-added top-level def.
    for &root in dag.roots() {
        if let Some(node) = dag.get(root)
            && let RiscOp::Store { name: store_name } = &node.op
            && store_name == name
        {
            return root;
        }
    }
    *dag.roots().last().expect("dag has at least one root")
}

/// Count `RiscOp::Copy` nodes reachable from `root` in `dag`. The cost
/// analyzer in `chelis_ir::analysis::analyze_function_copy_cost` only
/// considers reachable nodes, so this is the spec-aligned copy count.
fn copy_count(dag: &Dag, root: NodeId) -> usize {
    analyze_function_copy_cost(dag, "fanout", root).copy_count
}

/// Convenience: lift a 3-element f64 list into a `TensorValue`.
fn tv_3(data: [f64; 3]) -> TensorValue {
    TensorValue::from_vec(vec![3], data.to_vec())
}

/// Run the forward DAG for `fanout(input_name = value)` and return the
/// resulting tensor. `input_name` is the surf-level parameter name (e.g.
/// `"x"` or `"w"`), which the lowering pass uses as the `Load` op's name.
fn eval_fanout(dag: &Dag, root: NodeId, input_name: &str, input_value: TensorValue) -> TensorValue {
    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert(input_name.to_string(), input_value);
    let values =
        eval_tensor_with(dag, |name| inputs.get(name).cloned()).expect("forward eval succeeds");
    values
        .get(&root)
        .cloned()
        .unwrap_or_else(|| panic!("forward output for fanout root {root:?} missing"))
}

#[test]
fn call_rhs_fanout_control_lowers_without_explicit_copy() {
    // Two realizations of one input require a fork but no explicit
    // source-level copy.
    let source = r#"
module Repro.RealizeFanOut

def fanout(w: tensor[3, f32]) -> tensor[3, f32] = {
  a = realize(w)
  b = realize(w)
  add(a, b)
}
"#;
    let dag = surf_to_dag(source).expect("control surface must lower cleanly");
    let root = find_def_root_by_name(&dag, "fanout");
    // Two consume sites on `w` (realize×2) → exactly one inserted Copy.
    assert_eq!(
        copy_count(&dag, root),
        1,
        "expected exactly one inserted Copy for two realize consumes of `w`"
    );

    // Forward sanity: add(realize(w), realize(w)) = 2*w.
    let out = eval_fanout(&dag, root, "w", tv_3([1.0, 2.0, 3.0]));
    let expected = [2.0_f64, 4.0, 6.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (out.to_f64_lossy_vec()[i] - *want).abs() < 1e-6,
            "control forward[{i}]: expected {want}, got {}",
            out.to_f64_lossy_vec()[i]
        );
    }
}

#[test]
fn var_rhs_let_alias_then_borrow_use_lowers_without_explicit_copy() {
    // `mul` borrows both inputs. `alias` and `x` share the same
    // lowered load, so no consuming fan-out or Copy node is needed.
    let source = r#"
module Repro.FanOut

def fanout(x: tensor[3, f32]) -> tensor[3, f32] = {
  alias = x
  mul(x, alias)
}
"#;
    let dag = surf_to_dag(source).expect("var-RHS let aliasing must lower cleanly");
    let root = find_def_root_by_name(&dag, "fanout");
    // `mul` is a borrow primitive (auto-borrows both args). Source has
    // no consuming fan-out → zero inserted Copy nodes. Per spec §"Copy
    // Insertion", "Borrows do not count as fan-out".
    assert_eq!(
        copy_count(&dag, root),
        0,
        "expected zero Copy nodes: `mul` borrows both args; aliasing is free at the DAG level"
    );

    // AD parity vs the explicit-`copy(x)` workaround. Both rewrites must
    // produce the same gradient because the only difference is an inert
    // `Copy` node in the explicit variant, and `Copy`'s adjoint is the
    // identity.
    let workaround = r#"
module Repro.FanOutWorkaround

def fanout(x: tensor[3, f32]) -> tensor[3, f32] = {
  alias = copy(x)
  mul(x, alias)
}
"#;
    let workaround_dag = surf_to_dag(workaround).expect("explicit-copy form must lower cleanly");
    let workaround_root = find_def_root_by_name(&workaround_dag, "fanout");

    let x = tv_3([1.5, -0.5, 2.0]);
    let target_fwd = eval_fanout(&dag, root, "x", x.clone());
    let workaround_fwd = eval_fanout(&workaround_dag, workaround_root, "x", x.clone());
    assert_eq!(target_fwd.shape, workaround_fwd.shape);
    for i in 0..target_fwd.len() {
        assert!(
            (target_fwd.to_f64_lossy_vec()[i] - workaround_fwd.to_f64_lossy_vec()[i]).abs() < 1e-6,
            "forward parity fail at [{i}]: target={} workaround={}",
            target_fwd.to_f64_lossy_vec()[i],
            workaround_fwd.to_f64_lossy_vec()[i]
        );
    }

    // AD parity. Collapse `fanout(x)` to a scalar by summing and
    // differentiate w.r.t. the `Load { name: "x" }` parameter node.
    let target_x_load = find_load(&dag, "x");
    let workaround_x_load = find_load(&workaround_dag, "x");
    let target_scalar = scalarize_root(&dag, root);
    let workaround_scalar = scalarize_root(&workaround_dag, workaround_root);

    let target_grad = grad_dag_checked(&target_scalar.dag, target_scalar.root, &[target_x_load])
        .expect("target gradient must succeed");
    let workaround_grad = grad_dag_checked(
        &workaround_scalar.dag,
        workaround_scalar.root,
        &[workaround_x_load],
    )
    .expect("workaround gradient must succeed");

    let target_grad_node = target_grad.grad_nodes[&target_x_load];
    let workaround_grad_node = workaround_grad.grad_nodes[&workaround_x_load];

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert("x".to_string(), x);
    let target_vals = eval_tensor_with(&target_grad.dag, |n| inputs.get(n).cloned())
        .expect("target backward eval");
    let workaround_vals = eval_tensor_with(&workaround_grad.dag, |n| inputs.get(n).cloned())
        .expect("workaround backward eval");

    let target_dx = &target_vals[&target_grad_node];
    let workaround_dx = &workaround_vals[&workaround_grad_node];
    assert_eq!(target_dx.shape, workaround_dx.shape);
    for i in 0..target_dx.len() {
        assert!(
            (target_dx.to_f64_lossy_vec()[i] - workaround_dx.to_f64_lossy_vec()[i]).abs() < 1e-6,
            "AD parity fail at [{i}]: target={} workaround={}",
            target_dx.to_f64_lossy_vec()[i],
            workaround_dx.to_f64_lossy_vec()[i]
        );
    }
}

#[test]
fn top_level_no_module_var_rhs_aliased_fan_out_passes_linearity() {
    // Bare top-level `y = x` also creates an aliasing binding, so
    // later borrows of `x` pass linearity without an explicit copy.
    // Function-body fixtures below cover lowered copy counts and AD.
    let source = r#"
x = to_tensor([1.5, 2.7, -0.3], f32)
y = x
a = mul(y, to_tensor([2.0, 2.0, 2.0], f32))
b = mul(x, to_tensor([3.0, 3.0, 3.0], f32))
"#;
    surf_to_dag(source).expect("top-level no-module var-RHS aliasing must pass linearity");

    // A multi-level alias chain obeys the same rule.
    // `y = x; z = y; ...`. Each link records an alias so later
    // borrows of the source remain valid.
    let chained = r#"
x = to_tensor([1.5, 2.7, -0.3], f32)
y = x
z = y
a = mul(z, to_tensor([2.0, 2.0, 2.0], f32))
b = mul(x, to_tensor([3.0, 3.0, 3.0], f32))
"#;
    surf_to_dag(chained).expect("chained top-level var-RHS aliasing must pass linearity");
}

#[test]
fn fn_body_cross_statement_var_rhs_aliased_fan_out_lowers_without_explicit_copy() {
    // Function-body aliases keep later borrows valid, including
    // repeated borrows across let-bindings.
    let source = r#"
module Repro.FanOutFnBody

def fanout(x: tensor[3, f32]) -> tensor[3, f32] = {
  y = x
  a = mul(y, x)
  b = mul(x, x)
  add(a, b)
}
"#;
    let dag = surf_to_dag(source).expect("function-body var-RHS aliasing must lower cleanly");
    let root = find_def_root_by_name(&dag, "fanout");
    // mul/add borrow all args; no non-borrow consuming fan-out source
    // ⇒ zero Copy nodes per spec §"Copy Insertion".
    assert_eq!(
        copy_count(&dag, root),
        0,
        "expected zero Copy nodes; mul/add borrow both args; aliasing is free at the DAG level"
    );

    // AD parity vs the explicit-`copy(x)` rewrite. Copy's adjoint is
    // identity, so gradients must match within 1e-6.
    let workaround = r#"
module Repro.FanOutFnBodyWorkaround

def fanout(x: tensor[3, f32]) -> tensor[3, f32] = {
  y = copy(x)
  a = mul(y, x)
  b = mul(x, x)
  add(a, b)
}
"#;
    let workaround_dag = surf_to_dag(workaround).expect("explicit-copy form must lower cleanly");
    let workaround_root = find_def_root_by_name(&workaround_dag, "fanout");

    let x = tv_3([1.5, -0.5, 2.0]);
    let target_fwd = eval_fanout(&dag, root, "x", x.clone());
    let workaround_fwd = eval_fanout(&workaround_dag, workaround_root, "x", x.clone());
    assert_eq!(target_fwd.shape, workaround_fwd.shape);
    for i in 0..target_fwd.len() {
        assert!(
            (target_fwd.to_f64_lossy_vec()[i] - workaround_fwd.to_f64_lossy_vec()[i]).abs() < 1e-6,
            "forward parity fail at [{i}]: target={} workaround={}",
            target_fwd.to_f64_lossy_vec()[i],
            workaround_fwd.to_f64_lossy_vec()[i]
        );
    }

    let target_x_load = find_load(&dag, "x");
    let workaround_x_load = find_load(&workaround_dag, "x");
    let target_scalar = scalarize_root(&dag, root);
    let workaround_scalar = scalarize_root(&workaround_dag, workaround_root);

    let target_grad = grad_dag_checked(&target_scalar.dag, target_scalar.root, &[target_x_load])
        .expect("target gradient must succeed");
    let workaround_grad = grad_dag_checked(
        &workaround_scalar.dag,
        workaround_scalar.root,
        &[workaround_x_load],
    )
    .expect("workaround gradient must succeed");

    let target_grad_node = target_grad.grad_nodes[&target_x_load];
    let workaround_grad_node = workaround_grad.grad_nodes[&workaround_x_load];

    let mut inputs: UnordMap<String, TensorValue> = UnordMap::new();
    inputs.insert("x".to_string(), x);
    let target_vals = eval_tensor_with(&target_grad.dag, |n| inputs.get(n).cloned())
        .expect("target backward eval");
    let workaround_vals = eval_tensor_with(&workaround_grad.dag, |n| inputs.get(n).cloned())
        .expect("workaround backward eval");

    let target_dx = &target_vals[&target_grad_node];
    let workaround_dx = &workaround_vals[&workaround_grad_node];
    assert_eq!(target_dx.shape, workaround_dx.shape);
    for i in 0..target_dx.len() {
        assert!(
            (target_dx.to_f64_lossy_vec()[i] - workaround_dx.to_f64_lossy_vec()[i]).abs() < 1e-6,
            "AD parity fail at [{i}]: target={} workaround={}",
            target_dx.to_f64_lossy_vec()[i],
            workaround_dx.to_f64_lossy_vec()[i]
        );
    }
}

/// Find the `Load { name }` node in a DAG.
fn find_load(dag: &Dag, name: &str) -> NodeId {
    for node in dag.nodes() {
        if let RiscOp::Load { name: load_name } = &node.op
            && load_name == name
        {
            return node.id;
        }
    }
    panic!(
        "Load(`{name}`) not found in DAG; available ops: {:?}",
        dag.nodes().iter().map(|n| &n.op).collect::<Vec<_>>()
    );
}

struct Scalarized {
    dag: Dag,
    root: NodeId,
}

/// Clone `dag` and sum-reduce `root` (a `[3, f32]` tensor) down to a
/// scalar so `grad_dag_checked` accepts it.
fn scalarize_root(dag: &Dag, root: NodeId) -> Scalarized {
    use chelis_ir::dag::TensorType;
    let mut out = dag.clone();
    let out_decl = out.nodes()[0].owner.decl;
    let scalar = out.add_node(
        out_decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![root],
        TensorType::scalar_f32(),
        None,
    );
    // Promote the scalar to the only root so grad_dag_checked sees a
    // scalar-float output.
    out.set_roots(vec![scalar]);
    Scalarized {
        dag: out,
        root: scalar,
    }
}
