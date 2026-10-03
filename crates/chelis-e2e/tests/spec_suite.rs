// Tests only: Rust std functions on the clippy disallowed list compute
// reference or input values here; the list holds production code to
// chelis-crmath (chelis#2957).
#![allow(clippy::disallowed_methods)]
// Executable specification test suite for the Chelis language.
//
// Tests organized by language behavior, not by crate. Each test exercises the
// full pipeline or relevant subset and asserts specific expected values.

use chelis_unord::UnordMap;
use std::io::Write;
use std::process::Command;

use chelis_deep::ast::Expr;
use chelis_e2e::pipeline::compile_surf;
use chelis_ir::dag::{Dag, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_scalar, eval_tensor, eval_tensor_roots_with_strict};
use chelis_ir::grad::grad_dag;
use chelis_types::errors::CheckErrorKind;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

// chelis#729 Phase 1: the finite-difference autodiff specs verify grad
// machinery against f64 central differences, so their DAGs are f64-typed
// (an f32-typed DAG now genuinely rounds at f32, which drowns the h=1e-5
// difference quotient in rounding noise; the dtype behavior itself is
// covered by the chelis-cli matrix suites).
fn scalar_f64() -> TensorType {
    TensorType {
        dims: vec![],
        precision: chelis_types::types::Prim::F64,
    }
}

/// Lower Deep source text to a RISC DAG via the types-checked lowering path.
fn lower_deep(src: &str) -> Dag {
    let exprs = chelis_deep::parser::parse_str(src).expect("Deep parse failed");
    let checked = chelis_types::check_ir_program(&exprs)
        .unwrap_or_else(|r| panic!("type check failed: {:?}", r.errors));
    let checked = chelis_effects::check_program(&checked)
        .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
    let checked = chelis_types::check_linearity(&checked)
        .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"));
    chelis_ir::lower::lower_program(&checked)
}

// =========================================================================
// Category 1: Parsing
// =========================================================================

#[test]
fn spec_surf_to_deep_roundtrip() {
    let src = include_str!("../../../examples/mnist.ch");
    let decls = chelis_surf::parser::parse_str(src).expect("Surf parse failed");
    let deep_exprs =
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
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
    let deep_exprs =
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");

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
            let deep_exprs =
                chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
            assert!(
                !deep_exprs.is_empty(),
                "{} desugared to empty",
                path.display()
            );
            let report = chelis_types::check_ir_fitness(&deep_exprs);
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
    let deep_exprs =
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");

    fn check_3tuple(expr: &Expr) {
        // A tagged node is an `Expr::Node`, whose metadata map is a
        // structural field, so the walk checks the nested forms it reaches.
        match expr {
            Expr::MetaExpr(meta, _) => {
                check_3tuple(&meta.expr);
                meta.metadata.visit_syntax(&mut |_, v| check_3tuple(v));
            }
            Expr::Map(map, _) => {
                map.visit_syntax(&mut |_, v| check_3tuple(v));
            }
            Expr::Node(node, _) => {
                node.meta()
                    .visit_syntax(&mut |_, value| check_3tuple(value));
                for child in node.children_iter() {
                    match child {
                        chelis_deep::node::ChildRef::Expr(expr)
                        | chelis_deep::node::ChildRef::Syntax(expr)
                        | chelis_deep::node::ChildRef::Type(expr)
                        | chelis_deep::node::ChildRef::EffectHandler(expr)
                        | chelis_deep::node::ChildRef::Bypass(expr) => check_3tuple(expr),
                        chelis_deep::node::ChildRef::Binder(_)
                        | chelis_deep::node::ChildRef::Selector(_) => {}
                    }
                }
            }
            Expr::BareList(elements, _) => {
                for child in elements {
                    check_3tuple(child);
                }
            }
            Expr::UnknownForm(data) => {
                data.meta.visit_syntax(&mut |_, value| check_3tuple(value));
                for child in &data.children {
                    check_3tuple(child);
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
    let surf_src = "def f[n](x: tensor[n, f32]) -> tensor[n, f32] = relu(x)";
    let decls = chelis_surf::parser::parse_str(surf_src).expect("Surf parse failed");
    let deep_exprs =
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
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
def bad[n](a: tensor[n, f32], b: tensor[n, bf16]) -> tensor[n, f32] = add(a, b)
    "#;
    let decls = chelis_surf::parser::parse_str(surf_src).expect("Surf parse failed");
    let deep_exprs =
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
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
def bad(a: tensor[batch, f32], b: tensor[seq, f32]) -> tensor[batch, f32] = add(a, b)
    "#;
    let decls = chelis_surf::parser::parse_str(surf_src).expect("Surf parse failed");
    let deep_exprs =
        chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
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
        .any(|e| matches!(e.kind, CheckErrorKind::UnboundVariable { .. }));
    assert!(
        has_unbound,
        "expected UnboundVariable error, got: {:?}",
        result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
}

#[test]
fn spec_top_level_initialization_cycle_is_not_a_perfect_wrapperless_check() {
    // [04-INF-4]/[04-INF-7]/[04-INF-8], chelis#1601: this suite calls the
    // wrapper-less public APIs directly, so the initialization report must
    // live in their shared semantic driver rather than only in
    // `check_typed_program`.
    let declarations =
        chelis_surf::parser::parse_str("a = b\nb = a\n").expect("cycle fixture must parse");
    let deep_exprs =
        chelis_surf::desugar::desugar_program(&declarations).expect("Surf fixture must desugar");
    let inferred = chelis_types::infer_program(&deep_exprs);
    assert!(
        inferred
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::CycleDetected)),
        "infer_program must report the top-level initialization cycle: {:?}",
        inferred.errors
    );
    let report = chelis_types::check_program(&deep_exprs);
    assert!(report.score < 1.0, "a cycle cannot receive perfect fitness");
    let diagnostics = |errors: &[chelis_types::errors::CheckError]| {
        errors
            .iter()
            .map(|error| {
                format!(
                    "{}|{}|{}",
                    error.kind.diagnostic_name(),
                    error.message,
                    error.suggestions.join("\u{1f}")
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(diagnostics(&report.errors), diagnostics(&inferred.errors));
}

// =========================================================================
// Category 3: Lowering
// =========================================================================

#[test]
fn spec_relu_survives_as_dedicated_identity() {
    let src = r#"
        (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
        (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} relu) (var {} x)))
    "#;
    let dag = lower_deep(src);
    assert!(dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Relu)));
    assert!(!dag.nodes().iter().any(|n| matches!(n.op, RiscOp::MaxElem)));
}

#[test]
fn spec_sub_lowers_to_direct_identity_without_arithmetic_surrogate() {
    let src = r#"
        (def {} a (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} a))
        (def {} b (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} b))
        (def {} c (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} sub) (var {} a) (var {} b)))
    "#;
    let dag = lower_deep(src);
    assert_eq!(
        dag.nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Sub))
            .count(),
        1,
        "sub should lower to exactly one direct Sub identity"
    );
    assert!(
        dag.nodes()
            .iter()
            .all(|node| !matches!(node.op, RiscOp::Add | RiscOp::Neg)),
        "direct Sub lowering must not reconstruct subtraction as Add/Neg"
    );
}

#[test]
fn spec_sigmoid_decomposes() {
    let src = r#"
        (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
        (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} sigmoid) (var {} x)))
    "#;
    let dag = lower_deep(src);
    // sigmoid(x) = recip(1 + exp(-x)) decomposes to Exp + Neg + Add
    // + Recip (4 ops). An `exp(neg(log(_)))` reciprocal chain would
    // produce a `Log` node; that is a regression — the chain NaNs
    // on non-positive inputs.
    let has_exp = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Exp));
    let has_neg = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Neg));
    let has_add = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Add));
    let has_recip = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Recip));
    let has_log = dag.nodes().iter().any(|n| matches!(n.op, RiscOp::Log));
    assert!(has_exp, "sigmoid decomposition should contain Exp");
    assert!(has_neg, "sigmoid decomposition should contain Neg");
    assert!(has_add, "sigmoid decomposition should contain Add");
    assert!(
        has_recip,
        "sigmoid decomposition should contain Recip (the IEEE reciprocal primitive primitive)"
    );
    assert!(
        !has_log,
        "sigmoid decomposition must not contain Log; a Log node would indicate the historically \
         recip-via-log cascade has returned"
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
    Command::new(chelis_backend_c::toolchain::c_compiler())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn compile_and_run_dag(dag: &Dag, func_name: &str) -> String {
    assert!(gcc_available(), "gcc not available -- skipping");
    let selected = chelis_backend_c::prepare_dag_for_codegen(
        dag.clone(),
        chelis_backend_c::CodegenOptions::default(),
    );
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(selected).unwrap(),
    )
    .unwrap();
    let result = chelis_backend_c::codegen(verified, func_name).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    let write = |name: &str, content: &str| {
        let path = tmp.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    };
    write("model.c", &result.c_source);

    let main_c = format!(
        r#"
#include "chelis_runtime.h"
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main() {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    chelis_read_view output_view = chelis_tensor_read_view(outputs[0]);
    const float *output_data = (const float *)output_view.data;
    for (int64_t i = 0; i < output_view.count; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", output_data[i]);
    }}
    printf("\n");
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    write("main.c", &main_c);
    let bin_path = tmp.path().join("test_bin");

    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    let mut cmd = Command::new(&toolchain.compiler);
    cmd.args(["-O2", "-Wall"]);
    cmd.args(&toolchain.compile_flags);
    cmd.arg(tmp.path().join("main.c").to_str().unwrap());
    cmd.arg(tmp.path().join("model.c").to_str().unwrap());
    cmd.arg(&staged.archive);
    cmd.args(&toolchain.link_flags);
    cmd.arg("-o");
    cmd.arg(bin_path.to_str().unwrap());
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "{} compilation failed:\nstderr: {}\nC source:\n{}",
        toolchain.compiler,
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
    dag.add_root(c);

    let selected = chelis_backend_c::prepare_dag_for_codegen(
        dag.clone(),
        chelis_backend_c::CodegenOptions::default(),
    );
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(selected).unwrap(),
    )
    .unwrap();
    let result = chelis_backend_c::codegen(verified, "spec_test").unwrap();
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let zero = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        let r = dag.add_node(decl, RiscOp::MaxElem, vec![x, zero], scalar_f32(), None);
        dag.add_root(r);

        let out = compile_and_run_dag(&dag, "spec_relu_neg");
        let value: f64 = out.trim().parse().unwrap();
        assert!(value.abs() < 1e-5, "relu(-1) should be 0.0, got {value}");
    }
    // relu(const(5)) -> 5.0
    {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let zero = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        let r = dag.add_node(decl, RiscOp::MaxElem, vec![x, zero], scalar_f32(), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Add, vec![x, y], scalar_f64(), None);
    dag.add_root(out);

    let grad_result = grad_dag(&dag, out, &[x, y]).expect("grad_dag failed");
    let dx_node = grad_result.grad_nodes[&x];
    let dy_node = grad_result.grad_nodes[&y];

    let inputs: UnordMap<String, f64> =
        [("x".into(), 3.0), ("y".into(), 7.0)].into_iter().collect();
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
    let f_plus: UnordMap<String, f64> = [("x".into(), 3.0 + h), ("y".into(), 7.0)]
        .into_iter()
        .collect();
    let f_minus: UnordMap<String, f64> = [("x".into(), 3.0 - h), ("y".into(), 7.0)]
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Mul, vec![x, y], scalar_f64(), None);
    dag.add_root(out);

    let grad_result = grad_dag(&dag, out, &[x, y]).expect("grad_dag failed");

    let inputs: UnordMap<String, f64> =
        [("x".into(), 3.0), ("y".into(), 5.0)].into_iter().collect();
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
    let f_p: UnordMap<String, f64> = [("x".into(), 3.0 + h), ("y".into(), 5.0)]
        .into_iter()
        .collect();
    let f_m: UnordMap<String, f64> = [("x".into(), 3.0 - h), ("y".into(), 5.0)]
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f64(),
        None,
    );
    let neg_x = dag.add_node(decl, RiscOp::Neg, vec![x], scalar_f64(), None);
    let out = dag.add_node(decl, RiscOp::Exp, vec![neg_x], scalar_f64(), None);
    dag.add_root(out);

    let grad_result = grad_dag(&dag, out, &[x]).expect("grad_dag failed");
    let dx_node = grad_result.grad_nodes[&x];

    let x_val = 2.0;
    let inputs: UnordMap<String, f64> = [("x".into(), x_val)].into_iter().collect();
    let vals = eval_scalar(&grad_result.dag, &inputs);
    let analytic = -(-x_val).exp(); // -exp(-x)
    assert!(
        (vals[&dx_node] - analytic).abs() < 1e-6,
        "d(exp(-x))/dx at x=2 should be {analytic}, got {}",
        vals[&dx_node]
    );

    // Finite differences verification.
    let h = 1e-5;
    let f_p: UnordMap<String, f64> = [("x".into(), x_val + h)].into_iter().collect();
    let f_m: UnordMap<String, f64> = [("x".into(), x_val - h)].into_iter().collect();
    let v_p = eval_scalar(&dag, &f_p);
    let v_m = eval_scalar(&dag, &f_m);
    let fd = (v_p[&out] - v_m[&out]) / (2.0 * h);
    assert!(
        (fd - analytic).abs() < 1e-4,
        "finite-difference d(exp(-x))/dx should be ~{analytic}, got {fd}"
    );
}

#[test]
fn spec_vmap_grad_matches_per_example_loop_baseline() {
    let src = r#"
def loss(x: tensor[features, f32]) -> tensor[f32] =
  sum(mul(copy(x), x), 0)

def per_example_grad(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] =
  vmap(grad(loss))(xs)
"#;
    let compiled = compile_surf(src).expect("vmap(grad(...)) program should compile");
    let root = compiled.root_nodes["per_example_grad"];
    let xs = TensorValue::from_vec(
        vec![3, 4],
        vec![
            1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.5, -3.0, 2.0, 1.0, -0.5,
        ],
    );
    let inputs = UnordMap::from([
        ("xs".to_string(), xs.clone()),
        (
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![0.0; 4]),
        ),
    ]);
    let values =
        eval_tensor(&compiled.dag, &inputs).expect("per-example grad evaluation should succeed");
    let actual = values[&root].clone();

    let mut single = Dag::new();
    let single_decl = single.declare("test");
    let x = single.add_node(
        single_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![chelis_ir::dag::DimInfo::Lit(4)],
            precision: chelis_types::types::Prim::F32,
        },
        None,
    );
    let sq = single.add_node(
        single_decl,
        RiscOp::Mul,
        vec![x, x],
        TensorType {
            dims: vec![chelis_ir::dag::DimInfo::Lit(4)],
            precision: chelis_types::types::Prim::F32,
        },
        None,
    );
    let loss = single.add_node(
        single_decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![sq],
        TensorType::scalar_f32(),
        None,
    );
    single.add_root(loss);
    let grad = grad_dag(&single, loss, &[x]).expect("single-example grad baseline should exist");
    let grad_root = grad.grad_nodes[&x];

    let mut expected = Vec::with_capacity(xs.len());
    for example in xs.to_f64_lossy_vec().chunks(4) {
        let baseline_inputs = UnordMap::from([(
            "x".to_string(),
            TensorValue::from_vec(vec![4], example.to_vec()),
        )]);
        let baseline_values = eval_tensor_roots_with_strict(&grad.dag, &[grad_root], |name| {
            baseline_inputs.get(name).cloned()
        })
        .expect("baseline grad evaluation should succeed");
        expected.extend(
            baseline_values[&grad_root]
                .to_f64_lossy_vec()
                .iter()
                .copied(),
        );
    }

    assert_eq!(actual.shape, vec![3, 4]);
    assert_eq!(
        actual.len(),
        expected.len(),
        "per-example gradient element count mismatch"
    );
    for (actual_value, expected_value) in actual.to_f64_lossy_vec().iter().zip(expected.iter()) {
        assert!(
            (actual_value - expected_value).abs() <= 1e-6,
            "expected per-example gradient value {expected_value}, got {actual_value}"
        );
    }
}

#[test]
fn spec_vmap_grad_multiple_wrt_matches_per_example_loop_baseline() {
    let vmapped_src = r#"
def loss(
  x: tensor[4, f32],
  w: tensor[4, f32],
  v: tensor[4, f32]
) -> tensor[f32] =
  sum(mul(x, add(w, v)), 0)

def per_example_grads(
  xs: tensor[3, 4, f32],
  ws: tensor[3, 4, f32],
  vs: tensor[3, 4, f32]
) -> (tensor[3, 4, f32], tensor[3, 4, f32]) =
  vmap(grad(loss, wrt=(w, v)))(xs, ws, vs)
"#;
    let vmapped = compile_surf(vmapped_src).expect("vmapped multi-wrt grad program should compile");
    let dw_root = vmapped.root_nodes["per_example_grads.0"];
    let dv_root = vmapped.root_nodes["per_example_grads.1"];

    let xs = TensorValue::from_vec(
        vec![3, 4],
        vec![
            1.0, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 4.5, -3.0, 2.0, 1.0, -0.5,
        ],
    );
    let ws = TensorValue::from_vec(
        vec![3, 4],
        vec![
            0.25, -0.5, 0.75, 1.25, 0.1, 0.2, 0.3, 0.4, -1.0, 0.0, 1.0, 2.0,
        ],
    );
    let vs = TensorValue::from_vec(
        vec![3, 4],
        vec![
            1.5, -1.0, 0.5, 2.0, 0.3, -0.7, 1.1, -1.3, 2.5, -2.0, 0.25, 0.75,
        ],
    );
    let vmapped_values =
        eval_tensor_roots_with_strict(&vmapped.dag, &[dw_root, dv_root], |name| match name {
            "xs" => Some(xs.clone()),
            "ws" => Some(ws.clone()),
            "vs" => Some(vs.clone()),
            _ => None,
        })
        .expect("vmapped multi-wrt grad evaluation should succeed");
    let actual_dw = vmapped_values[&dw_root].clone();
    let actual_dv = vmapped_values[&dv_root].clone();

    let single_src = r#"
def loss(
  x: tensor[4, f32],
  w: tensor[4, f32],
  v: tensor[4, f32]
) -> tensor[f32] =
  sum(mul(x, add(w, v)), 0)

def grads(x: tensor[4, f32], w: tensor[4, f32], v: tensor[4, f32])
    -> (tensor[4, f32], tensor[4, f32]) =
  grad(loss, wrt=(w, v))(x, w, v)
"#;
    let single =
        compile_surf(single_src).expect("single-example multi-wrt grad program should compile");
    let single_dw_root = single.root_nodes["grads.0"];
    let single_dv_root = single.root_nodes["grads.1"];

    let mut expected_dw = Vec::new();
    let mut expected_dv = Vec::new();
    for batch_index in 0..3 {
        let x = TensorValue::from_vec(
            vec![4],
            xs.to_f64_lossy_vec()[batch_index * 4..(batch_index + 1) * 4].to_vec(),
        );
        let w = TensorValue::from_vec(
            vec![4],
            ws.to_f64_lossy_vec()[batch_index * 4..(batch_index + 1) * 4].to_vec(),
        );
        let v = TensorValue::from_vec(
            vec![4],
            vs.to_f64_lossy_vec()[batch_index * 4..(batch_index + 1) * 4].to_vec(),
        );
        let values =
            eval_tensor_roots_with_strict(&single.dag, &[single_dw_root, single_dv_root], |name| {
                match name {
                    "x" => Some(x.clone()),
                    "w" => Some(w.clone()),
                    "v" => Some(v.clone()),
                    _ => None,
                }
            })
            .expect("single-example multi-wrt grad evaluation should succeed");
        expected_dw.extend(values[&single_dw_root].to_f64_lossy_vec().iter().copied());
        expected_dv.extend(values[&single_dv_root].to_f64_lossy_vec().iter().copied());
    }

    assert_eq!(actual_dw.shape, vec![3, 4]);
    assert_eq!(actual_dv.shape, vec![3, 4]);
    assert_eq!(actual_dw.to_f64_lossy_vec(), expected_dw);
    assert_eq!(actual_dv.to_f64_lossy_vec(), expected_dv);
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

    let mut inputs = UnordMap::new();
    inputs.insert(
        "a".into(),
        TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    inputs.insert(
        "b".into(),
        TensorValue::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
    );

    let vals = eval_tensor(&dag, &inputs).unwrap();
    let result_id = *dag.roots().last().expect("DAG root");
    let result = &vals[&result_id];

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.to_f64_lossy_vec(), vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn spec_eval_softmax_sums_to_one() {
    let src = r#"
        (def {} x (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
        (def {} y (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
            (var {} softmax) (var {} x) (lit {} 0)))
    "#;
    let dag = lower_deep(src);

    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
    );

    let vals = eval_tensor(&dag, &inputs).unwrap();
    let result_id = *dag.roots().last().expect("DAG root");
    let result = &vals[&result_id];

    let sum: f64 = result.to_f64_lossy_vec().iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax output should sum to 1.0, got {sum}"
    );

    // Check individual values match known softmax([1,2,3]).
    let expected = [0.09003057, 0.24472847, 0.66524096];
    for (i, (actual, target)) in result
        .to_f64_lossy_vec()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
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

    let mut inputs = UnordMap::new();
    inputs.insert(
        "x".into(),
        TensorValue::from_vec(vec![5], vec![-1.0, 0.0, 2.0, -3.0, 5.0]),
    );

    let vals = eval_tensor(&dag, &inputs).unwrap();
    let result_id = *dag.roots().last().expect("DAG root");
    let result = &vals[&result_id];

    assert_eq!(result.shape, vec![5]);
    assert_eq!(result.to_f64_lossy_vec(), vec![0.0, 0.0, 2.0, 0.0, 5.0]);
}
