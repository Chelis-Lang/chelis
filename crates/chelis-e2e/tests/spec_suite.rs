// Executable specification test suite for the Chelis language.
//
// Tests organized by language behavior, not by crate. Each test exercises the
// full pipeline or relevant subset and asserts specific expected values.

use std::collections::HashMap;
use std::io::Write;
use std::process::Command;

use chelis_deep::ast::{Atom, Expr};
use chelis_ir::dag::{Dag, NodeId, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_scalar, eval_tensor};
use chelis_ir::grad::grad_dag;
use chelis_types::errors::CheckErrorKind;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

/// Lower Deep source text to a RISC DAG via the types-checked lowering path.
fn lower_deep(src: &str) -> Dag {
    let exprs = chelis_deep::parser::parse_str(src).expect("Deep parse failed");
    let checked = chelis_types::check_phase0e_program(&exprs)
        .unwrap_or_else(|r| panic!("type check failed: {:?}", r.errors));
    chelis_ir::lower::lower_program(&checked)
}

// =========================================================================
// Category 1: Parsing
// =========================================================================

#[test]
fn spec_surf_to_deep_roundtrip() {
    let src = include_str!("../../../examples/mnist.ch");
    let decls = chelis_surf::parser::parse_str(src).expect("Surf parse failed");
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let deep_text = chelis_deep::printer::print_canonical(&deep_exprs);

    // Re-parse the canonical Deep text -- must succeed without error.
    let reparsed =
        chelis_deep::parser::parse_str(&deep_text).expect("re-parsing canonical Deep failed");
    assert!(
        !reparsed.is_empty(),
        "round-tripped Deep must produce at least one expression"
    );
    // Print again and check idempotence.
    let deep_text2 = chelis_deep::printer::print_canonical(&reparsed);
    assert_eq!(
        deep_text, deep_text2,
        "canonical Deep printer is not idempotent"
    );
}

#[test]
fn spec_deep_strict_validates_desugared() {
    let src = include_str!("../../../examples/mnist.ch");
    let decls = chelis_surf::parser::parse_str(src).expect("Surf parse failed");
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);

    let warnings = chelis_deep::validate::validate(&deep_exprs);
    let unknown_tags: Vec<_> = warnings
        .iter()
        .filter(|w| matches!(w.kind, chelis_deep::validate::WarningKind::UnknownTag))
        .collect();
    assert!(
        unknown_tags.is_empty(),
        "desugared output contains unknown tags: {:?}",
        unknown_tags.iter().map(|w| &w.message).collect::<Vec<_>>()
    );
}

#[test]
fn spec_all_executable_examples_parse_and_check() {
    let examples_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .expect("examples/ directory not found");

    let mut count = 0;
    for entry in std::fs::read_dir(&examples_dir).expect("cannot read examples/") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("ch") {
            let src = std::fs::read_to_string(&path).unwrap();
            let result = chelis_surf::parser::parse_str(&src);
            assert!(
                result.is_ok(),
                "failed to parse {}: {:?}",
                path.display(),
                result.err()
            );
            let decls = result.unwrap();
            let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
            assert!(
                !deep_exprs.is_empty(),
                "{} desugared to empty",
                path.display()
            );
            let report = chelis_types::check_phase0e_fitness(&deep_exprs);
            assert!(
                report.errors.is_empty(),
                "{} must remain executable in Phase 0: {:?}",
                path.display(),
                report.errors
            );
            assert!(
                (report.score - 1.0).abs() < 1e-9,
                "{} must keep score 1.0, got {}",
                path.display(),
                report.score
            );
            count += 1;
        }
    }
    assert!(
        count >= 2,
        "expected at least 2 executable .ch example files, found {count}"
    );
}

#[test]
fn spec_deep_3tuple_format() {
    let src = include_str!("../../../examples/hello_tensor.ch");
    let decls = chelis_surf::parser::parse_str(src).expect("Surf parse failed");
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);

    fn check_3tuple(expr: &Expr) {
        match expr {
            Expr::List(list, _) => {
                if list.elements.len() >= 2
                    && let Some(Expr::Atom(Atom::Symbol(_tag), _)) = list.elements.first()
                {
                    // Second element must be a map (metadata).
                    assert!(
                        matches!(list.elements.get(1), Some(Expr::Map(_, _))),
                        "tagged node must have metadata map at element[1], got: {:?}",
                        list.elements.get(1)
                    );
                }
                for child in &list.elements {
                    check_3tuple(child);
                }
            }
            Expr::MetaExpr(meta, _) => {
                check_3tuple(&meta.expr);
                for (_, v) in &meta.entries {
                    check_3tuple(v);
                }
            }
            Expr::Map(map, _) => {
                for (_, v) in &map.entries {
                    check_3tuple(v);
                }
            }
            Expr::Atom(_, _) => {}
        }
    }

    for expr in &deep_exprs {
        check_3tuple(expr);
    }
}

// =========================================================================
// Category 2: Type Checking
// =========================================================================

#[test]
fn spec_correct_program_fitness_1() {
    // Use the Surf pipeline: a well-typed Surf program should get fitness 1.0.
    let surf_src = "def f(x: tensor[n, f32]): tensor[n, f32] = relu(x)";
    let decls = chelis_surf::parser::parse_str(surf_src).expect("Surf parse failed");
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let report = chelis_types::check_program(&deep_exprs);
    assert!(
        (report.score - 1.0).abs() < 1e-9,
        "well-typed program should have fitness 1.0, got {} (errors: {:?})",
        report.score,
        report.errors
    );
    assert!(
        report.errors.is_empty(),
        "well-typed program should have no errors, got: {:?}",
        report.errors
    );
}

#[test]
fn spec_precision_mismatch_is_error() {
    // add(tensor[n, f32], tensor[n, bf16]) should produce PrecisionMismatch.
    let surf_src = r#"
def bad(a: tensor[n, f32], b: tensor[n, bf16]): tensor[n, f32] = add(a, b)
    "#;
    let decls = chelis_surf::parser::parse_str(surf_src).expect("Surf parse failed");
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let result = chelis_types::infer_program(&deep_exprs);
    assert!(!result.errors.is_empty(), "expected type errors");
    let has_precision_error = result
        .errors
        .iter()
        .any(|e| matches!(e.kind, CheckErrorKind::PrecisionMismatch));
    assert!(
        has_precision_error,
        "expected PrecisionMismatch error, got: {:?}",
        result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
}

#[test]
fn spec_dimension_mismatch_is_error() {
    // add(tensor[batch, f32], tensor[seq, f32]) should produce DimensionMismatch.
    let surf_src = r#"
def bad(a: tensor[batch, f32], b: tensor[seq, f32]): tensor[batch, f32] = add(a, b)
    "#;
    let decls = chelis_surf::parser::parse_str(surf_src).expect("Surf parse failed");
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let result = chelis_types::infer_program(&deep_exprs);
    assert!(!result.errors.is_empty(), "expected type errors");
    let has_dim_error = result
        .errors
        .iter()
        .any(|e| matches!(e.kind, CheckErrorKind::DimensionMismatch));
    assert!(
        has_dim_error,
        "expected DimensionMismatch error, got: {:?}",
        result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
}

#[test]
fn spec_unbound_variable_is_error() {
    let src = r#"
        (def {} y (app {} (var {} nonexistent_function) (lit {type: (t-prim {} f32)} 1.0)))
    "#;
    let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
    let result = chelis_types::infer_program(&exprs);
    let has_unbound = result
        .errors
        .iter()
        .any(|e| matches!(e.kind, CheckErrorKind::UnboundVariable));
    assert!(
        has_unbound,
        "expected UnboundVariable error, got: {:?}",
        result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
}

// =========================================================================
// Category 3: Lowering
// =========================================================================

#[test]
fn spec_relu_decomposes_to_max_elem() {
    let src = r#"
        (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
        (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} relu) (var {} x)))
    "#;
    let dag = lower_deep(src);
    let has_max_elem = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::MaxElem));
    let has_const_zero = dag
        .nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::Const { value } if value == 0.0));
    assert!(has_max_elem, "relu should decompose to MaxElem");
    assert!(has_const_zero, "relu should decompose with Const(0)");
    // No standalone Relu op should exist in the DAG.
    let has_no_relu_tag = dag.nodes().iter().all(|n| {
        !matches!(
            &n.op,
            RiscOp::Load { name } if name == "relu"
        )
    });
    assert!(has_no_relu_tag, "RISC DAG must not contain a relu Load");
}

#[test]
fn spec_sub_decomposes_to_add_neg() {
    let src = r#"
        (def {} a (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} a))
        (def {} b (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} b))
        (def {} c (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} sub) (var {} a) (var {} b)))
    "#;
    let dag = lower_deep(src);
    let has_add = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Add));
    let has_neg = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Neg));
    assert!(has_add, "sub should decompose to include Add");
    assert!(has_neg, "sub should decompose to include Neg");
}

#[test]
fn spec_sigmoid_decomposes() {
    let src = r#"
        (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
        (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} sigmoid) (var {} x)))
    "#;
    let dag = lower_deep(src);
    // sigmoid(x) = 1/(1+exp(-x)) decomposes to Exp, Neg, Add, plus div decomposition.
    let has_exp = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Exp));
    let has_neg = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Neg));
    let has_add = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Add));
    let has_log = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Log));
    assert!(has_exp, "sigmoid decomposition should contain Exp");
    assert!(has_neg, "sigmoid decomposition should contain Neg");
    assert!(has_add, "sigmoid decomposition should contain Add");
    assert!(
        has_log,
        "sigmoid decomposition should contain Log (from recip)"
    );
}

#[test]
fn spec_matmul_decomposes() {
    let src = r#"
        (def {} a (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} a))
        (def {} b (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} b))
        (def {} c (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
            (var {} matmul) (var {} a) (var {} b)))
    "#;
    let dag = lower_deep(src);
    let has_expand = dag
        .nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::Expand { .. }));
    let has_mul = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Mul));
    let has_sum = dag
        .nodes()
        .iter()
        .any(|n| matches!(n.op, RiscOp::Sum { .. }));
    assert!(
        has_expand,
        "matmul decomposition should contain Expand nodes"
    );
    assert!(has_mul, "matmul decomposition should contain Mul");
    assert!(has_sum, "matmul decomposition should contain Sum");
}

// =========================================================================
// Category 4: Codegen
// =========================================================================

fn gcc_available() -> bool {
    Command::new("gcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn runtime_src_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../chelis-backend-c/runtime")
        .canonicalize()
        .expect("runtime dir not found")
}

fn compile_and_run_dag(dag: &Dag, func_name: &str) -> String {
    assert!(gcc_available(), "gcc not available -- skipping");
    let result = chelis_backend_c::codegen(dag, func_name);
    let tmp = tempfile::tempdir().unwrap();
    let rt_dir = runtime_src_dir();
    let h_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.h")).unwrap();
    let c_rt = std::fs::read_to_string(rt_dir.join("chelis_runtime.c")).unwrap();

    let write = |name: &str, content: &str| {
        let path = tmp.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    };
    write("chelis_runtime.h", &h_src);
    write("chelis_runtime.c", &c_rt);
    write("model.c", &result.c_source);

    let main_c = format!(
        r#"
#include "chelis_runtime.h"
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main() {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    for (int i = 0; i < outputs[0]->size; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", outputs[0]->data[i]);
    }}
    printf("\n");
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    write("main.c", &main_c);
    let bin_path = tmp.path().join("test_bin");

    let mut cmd = Command::new("gcc");
    cmd.args(["-O2", "-Wall"]);
    cmd.args(&result.compile_flags);
    cmd.arg(tmp.path().join("main.c").to_str().unwrap());
    cmd.arg(tmp.path().join("model.c").to_str().unwrap());
    cmd.arg(tmp.path().join("chelis_runtime.c").to_str().unwrap());
    cmd.args(&result.link_flags);
    cmd.arg("-o");
    cmd.arg(bin_path.to_str().unwrap());
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "gcc compilation failed:\nstderr: {}\nC source:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );

    let run = Command::new(bin_path.to_str().unwrap()).output().unwrap();
    assert!(
        run.status.success(),
        "binary execution failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap().trim().to_string()
}

#[test]
fn spec_generated_c_compiles() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
    dag.add_root(c);

    let result = chelis_backend_c::codegen(&dag, "spec_test");
    assert!(
        !result.c_source.is_empty(),
        "codegen should produce non-empty C source"
    );
    // Actually compile it.
    let _ = compile_and_run_dag(&dag, "spec_test");
}

#[test]
fn spec_add_numerical_correctness() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
    dag.add_root(c);

    let out = compile_and_run_dag(&dag, "spec_add");
    let value: f64 = out.trim().parse().expect("output is not a number");
    assert!(
        (value - 3.0).abs() < 1e-5,
        "add(1,2) should produce 3.0, got {value}"
    );
}

#[test]
fn spec_relu_numerical_correctness() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // relu(const(-1)) -> 0.0
    {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: -1.0 }, vec![], scalar_f32());
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        let r = dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32());
        dag.add_root(r);

        let out = compile_and_run_dag(&dag, "spec_relu_neg");
        let value: f64 = out.trim().parse().unwrap();
        assert!(value.abs() < 1e-5, "relu(-1) should be 0.0, got {value}");
    }
    // relu(const(5)) -> 5.0
    {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        let r = dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32());
        dag.add_root(r);

        let out = compile_and_run_dag(&dag, "spec_relu_pos");
        let value: f64 = out.trim().parse().unwrap();
        assert!(
            (value - 5.0).abs() < 1e-5,
            "relu(5) should be 5.0, got {value}"
        );
    }
}

// =========================================================================
// Category 5: Automatic Differentiation
// =========================================================================

#[test]
fn spec_grad_add_is_one() {
    // f(x,y) = x + y => df/dx = 1, df/dy = 1
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], scalar_f32());
    let out = dag.add_node(RiscOp::Add, vec![x, y], scalar_f32());
    dag.add_root(out);

    let grad_result = grad_dag(&dag, out, &[x, y]).expect("grad_dag failed");
    let dx_node = grad_result.grad_nodes[&x];
    let dy_node = grad_result.grad_nodes[&y];

    let inputs: HashMap<String, f64> = [("x".into(), 3.0), ("y".into(), 7.0)].into_iter().collect();
    let vals = eval_scalar(&grad_result.dag, &inputs);

    assert!(
        (vals[&dx_node] - 1.0).abs() < 1e-9,
        "d(x+y)/dx should be 1.0, got {}",
        vals[&dx_node]
    );
    assert!(
        (vals[&dy_node] - 1.0).abs() < 1e-9,
        "d(x+y)/dy should be 1.0, got {}",
        vals[&dy_node]
    );

    // Verify by finite differences: f(x+h)-f(x-h) / 2h ~ 1.0
    let h = 1e-5;
    let f_plus: HashMap<String, f64> = [("x".into(), 3.0 + h), ("y".into(), 7.0)]
        .into_iter()
        .collect();
    let f_minus: HashMap<String, f64> = [("x".into(), 3.0 - h), ("y".into(), 7.0)]
        .into_iter()
        .collect();
    let v_plus = eval_scalar(&dag, &f_plus);
    let v_minus = eval_scalar(&dag, &f_minus);
    let fd = (v_plus[&out] - v_minus[&out]) / (2.0 * h);
    assert!(
        (fd - 1.0).abs() < 1e-4,
        "finite-difference d(x+y)/dx should be ~1.0, got {fd}"
    );
}

#[test]
fn spec_grad_mul_is_cross() {
    // f(x,y) = x * y => df/dx = y, df/dy = x
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], scalar_f32());
    let out = dag.add_node(RiscOp::Mul, vec![x, y], scalar_f32());
    dag.add_root(out);

    let grad_result = grad_dag(&dag, out, &[x, y]).expect("grad_dag failed");

    let inputs: HashMap<String, f64> = [("x".into(), 3.0), ("y".into(), 5.0)].into_iter().collect();
    let vals = eval_scalar(&grad_result.dag, &inputs);

    let dx = vals[&grad_result.grad_nodes[&x]];
    let dy = vals[&grad_result.grad_nodes[&y]];
    assert!(
        (dx - 5.0).abs() < 1e-9,
        "d(x*y)/dx at y=5 should be 5.0, got {dx}"
    );
    assert!(
        (dy - 3.0).abs() < 1e-9,
        "d(x*y)/dy at x=3 should be 3.0, got {dy}"
    );

    // Finite difference check for df/dx.
    let h = 1e-5;
    let f_p: HashMap<String, f64> = [("x".into(), 3.0 + h), ("y".into(), 5.0)]
        .into_iter()
        .collect();
    let f_m: HashMap<String, f64> = [("x".into(), 3.0 - h), ("y".into(), 5.0)]
        .into_iter()
        .collect();
    let v_p = eval_scalar(&dag, &f_p);
    let v_m = eval_scalar(&dag, &f_m);
    let fd = (v_p[&out] - v_m[&out]) / (2.0 * h);
    assert!(
        (fd - 5.0).abs() < 1e-3,
        "finite-difference d(x*y)/dx at y=5 should be ~5.0, got {fd}"
    );
}

#[test]
fn spec_grad_composed_chain() {
    // f(x) = exp(neg(x)) = exp(-x), df/dx = -exp(-x)
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
    let neg_x = dag.add_node(RiscOp::Neg, vec![x], scalar_f32());
    let out = dag.add_node(RiscOp::Exp, vec![neg_x], scalar_f32());
    dag.add_root(out);

    let grad_result = grad_dag(&dag, out, &[x]).expect("grad_dag failed");
    let dx_node = grad_result.grad_nodes[&x];

    let x_val = 2.0;
    let inputs: HashMap<String, f64> = [("x".into(), x_val)].into_iter().collect();
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let analytic = -(-x_val).exp(); // -exp(-x)
    assert!(
        (vals[&dx_node] - analytic).abs() < 1e-6,
        "d(exp(-x))/dx at x=2 should be {analytic}, got {}",
        vals[&dx_node]
    );

    // Finite differences verification.
    let h = 1e-5;
    let f_p: HashMap<String, f64> = [("x".into(), x_val + h)].into_iter().collect();
    let f_m: HashMap<String, f64> = [("x".into(), x_val - h)].into_iter().collect();
    let v_p = eval_scalar(&dag, &f_p);
    let v_m = eval_scalar(&dag, &f_m);
    let fd = (v_p[&out] - v_m[&out]) / (2.0 * h);
    assert!(
        (fd - analytic).abs() < 1e-4,
        "finite-difference d(exp(-x))/dx should be ~{analytic}, got {fd}"
    );
}

// =========================================================================
// Category 6: Evaluator
// =========================================================================

#[test]
fn spec_eval_matmul_correct() {
    // 2x3 @ 3x2 => 2x2, using known values.
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    // Expected: [[58,64],[139,154]]
    let src = r#"
        (def {} a (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} a))
        (def {} b (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} b))
        (def {} c (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 2) (t-prim {} f32))}
            (var {} matmul) (var {} a) (var {} b)))
    "#;
    let dag = lower_deep(src);

    let mut inputs = HashMap::new();
    inputs.insert(
        "a".into(),
        TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    inputs.insert(
        "b".into(),
        TensorValue::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    );

    let vals = eval_tensor(&dag, &inputs).unwrap();
    let result_id = NodeId(dag.len() - 1);
    let result = &vals[&result_id];

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn spec_eval_softmax_sums_to_one() {
    let src = r#"
        (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
        (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} softmax) (var {} x) (lit {} 0)))
    "#;
    let dag = lower_deep(src);

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    );

    let vals = eval_tensor(&dag, &inputs).unwrap();
    let result_id = NodeId(dag.len() - 1);
    let result = &vals[&result_id];

    let sum: f64 = result.data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax output should sum to 1.0, got {sum}"
    );

    // Check individual values match known softmax([1,2,3]).
    let expected = [0.09003057, 0.24472847, 0.66524096];
    for (i, (actual, target)) in result.data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - target).abs() < 1e-5,
            "softmax[{i}]: expected {target}, got {actual}"
        );
    }
}

#[test]
fn spec_eval_relu_preserves_positive() {
    let src = r#"
        (def {} x (var {type: (t-tensor {} (d-lit {} 5) (t-prim {} f32))} x))
        (def {} y (app {type: (t-tensor {} (d-lit {} 5) (t-prim {} f32))}
            (var {} relu) (var {} x)))
    "#;
    let dag = lower_deep(src);

    let mut inputs = HashMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![5], vec![-1.0, 0.0, 2.0, -3.0, 5.0]),
    );

    let vals = eval_tensor(&dag, &inputs).unwrap();
    let result_id = NodeId(dag.len() - 1);
    let result = &vals[&result_id];

    assert_eq!(result.shape, vec![5]);
    assert_eq!(result.data, vec![0.0, 0.0, 2.0, 0.0, 5.0]);
}
