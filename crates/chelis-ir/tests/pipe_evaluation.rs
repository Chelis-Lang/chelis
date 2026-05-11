//! Pipe-stage lowering: pin all four supported callable shapes.
//!
//! `lower_pipe()` in `crates/chelis-ir/src/lower.rs` chains a seed value
//! through stages. Plain function, `vmap`, `grad`, and `vmap(grad)`
//! callable stages all lower via the shared non-pipe construction.
//!
//! This file pins all four shapes:
//!
//! 1. Plain unary builtin (`x |> relu`).
//! 2. Plain callable with extra args (`x |> add(a)` ≡ `add(x, a)`).
//! 3. `x |> grad(f)` ≡ `grad(f)(x)`.
//! 4. `xs |> vmap(grad(f))` ≡ `vmap(grad(f))(xs)`.

use std::collections::HashMap;

use chelis_deep::Expr;
use chelis_ir::dag::{DimInfo, NodeId, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::lower::lower_subexpr_program;
use chelis_types::types::Prim;

fn f32_vec(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn f32_mat(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

fn parse_one(src: &str) -> Expr {
    chelis_deep::parser::parse_str(src)
        .expect("parse failed")
        .into_iter()
        .next()
        .expect("at least one expr")
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 1 — control: `x |> relu` lowers and evaluates as `relu(x)`.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pipe_unary_builtin_stage_lowers_and_evaluates() {
    // (pipe {} (var {} x) (var {} relu))
    let pipe_src = r#"
        (pipe {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
          (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
          (var {} relu))
    "#;
    let pipe_expr = parse_one(pipe_src);
    let scoped = HashMap::from([("x".to_string(), f32_vec(3))]);
    let dag = lower_subexpr_program(&pipe_expr, scoped, HashMap::new(), HashMap::new());

    let inputs = HashMap::from([(
        "x".to_string(),
        TensorValue::from_vec(vec![3], vec![-1.0, 0.0, 2.5]),
    )]);
    let roots: Vec<NodeId> = dag.roots().to_vec();
    assert!(
        !roots.is_empty(),
        "lowered pipe must produce at least one root"
    );
    let values = eval_tensor_roots_with_strict(&dag, &roots, |name| inputs.get(name).cloned())
        .expect("eval succeeds");
    let out = &values[roots.last().unwrap()];
    assert_eq!(out.shape, vec![3]);
    // relu([-1, 0, 2.5]) = [0, 0, 2.5]
    assert_eq!(out.data, vec![0.0, 0.0, 2.5]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 2 — control: `x |> add(a)` lowers and evaluates as `add(x, a)`.
// Surf desugars `x |> add(a)` into a lambda
// `(fn (params __chelis_pipe) (app add __chelis_pipe a))`. Use that shape
// here so this test pins the Plain-callable path through `lower_pipe`.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pipe_plain_callable_with_args_lowers_and_evaluates() {
    let pipe_src = r#"
        (pipe {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
          (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
          (fn {type: (t-fn {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)) (t-tensor {} (d-lit {} 3) (t-prim {} f32)))}
            (params {} (__chelis_pipe {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}))
            (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
              (var {} add)
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} __chelis_pipe)
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} a))))
    "#;
    let pipe_expr = parse_one(pipe_src);
    let scoped = HashMap::from([("x".to_string(), f32_vec(3)), ("a".to_string(), f32_vec(3))]);
    let dag = lower_subexpr_program(&pipe_expr, scoped, HashMap::new(), HashMap::new());

    let inputs = HashMap::from([
        (
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        ),
        (
            "a".to_string(),
            TensorValue::from_vec(vec![3], vec![10.0, 20.0, 30.0]),
        ),
    ]);
    let roots: Vec<NodeId> = dag.roots().to_vec();
    assert!(!roots.is_empty());
    let values = eval_tensor_roots_with_strict(&dag, &roots, |name| inputs.get(name).cloned())
        .expect("eval succeeds");
    let out = &values[roots.last().unwrap()];
    assert_eq!(out.shape, vec![3]);
    assert_eq!(out.data, vec![11.0, 22.0, 33.0]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 3 — primary target: `x |> grad(f)` ≡ `grad(f)(x)`.
//
// f(x: [1]f32) -> f32 = sum(mul(x, x), 0)  →  df/dx = 2*x.
// At x=[3.0], gradient = [6.0].
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pipe_grad_stage_matches_non_pipe_application() {
    let fn_src = r#"
        (fn {}
          (params {}
            (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
          (app {type: (t-tensor {} (t-prim {} f32))}
            (var {} sum)
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (var {} mul)
              (copy {} (var {} x))
              (copy {} (var {} x)))
            (lit {type: (t-prim {} int32)} 0)))
    "#;
    // `x |> grad(f)` -- the seed is the input tensor, the stage is the
    // bare `(grad {} (var {} f))` callable (no lambda wrap; Surf does not
    // wrap `Grad`-stage pipes in a lambda).
    let pipe_src = r#"
        (pipe {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
          (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input)
          (grad {} (var {} loss)))
    "#;
    // Non-pipe reference: `grad(f)(x)`.
    let app_src = r#"
        (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
          (grad {} (var {} loss))
          (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input))
    "#;

    let mut program_defs = HashMap::new();
    program_defs.insert("loss".to_string(), parse_one(fn_src));
    let input_ty = f32_vec(1);
    let scoped = HashMap::from([("input".to_string(), input_ty)]);

    let pipe_dag = lower_subexpr_program(
        &parse_one(pipe_src),
        scoped.clone(),
        HashMap::new(),
        program_defs.clone(),
    );
    let app_dag = lower_subexpr_program(&parse_one(app_src), scoped, HashMap::new(), program_defs);

    let inputs = HashMap::from([(
        "input".to_string(),
        TensorValue::from_vec(vec![1], vec![3.0]),
    )]);

    let pipe_roots: Vec<NodeId> = pipe_dag.roots().to_vec();
    let app_roots: Vec<NodeId> = app_dag.roots().to_vec();
    assert!(!pipe_roots.is_empty(), "pipe DAG must have a root");
    assert!(!app_roots.is_empty(), "app DAG must have a root");

    let pipe_values =
        eval_tensor_roots_with_strict(&pipe_dag, &pipe_roots, |name| inputs.get(name).cloned())
            .expect("pipe eval");
    let app_values =
        eval_tensor_roots_with_strict(&app_dag, &app_roots, |name| inputs.get(name).cloned())
            .expect("app eval");

    let pipe_out = &pipe_values[pipe_roots.last().unwrap()];
    let app_out = &app_values[app_roots.last().unwrap()];
    assert_eq!(pipe_out.shape, app_out.shape);
    assert_eq!(pipe_out.shape, vec![1]);
    // df/dx for f(x) = sum(x*x) at x=[3.0] is [6.0].
    assert!(
        (pipe_out.data[0] - 6.0).abs() < 1e-6,
        "pipe grad result at x=3 must be 6.0, got {:?}",
        pipe_out.data,
    );
    for (p, a) in pipe_out.data.iter().zip(app_out.data.iter()) {
        assert!(
            (p - a).abs() < 1e-6,
            "pipe and non-pipe grad must agree element-wise: pipe={p}, app={a}",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture 4 — secondary target: `xs |> vmap(grad(f))` ≡ `vmap(grad(f))(xs)`.
//
// f(x: [1]f32) -> f32 = sum(mul(x, x), 0).
// vmap(grad(f)) maps over the batch axis. For xs=[[1.0],[2.0],[3.0]] (shape
// [3,1]) the per-batch gradient is [[2.0],[4.0],[6.0]].
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn pipe_vmap_grad_stage_matches_non_pipe_application() {
    let fn_src = r#"
        (fn {}
          (params {}
            (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
          (app {type: (t-tensor {} (t-prim {} f32))}
            (var {} sum)
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (var {} mul)
              (copy {} (var {} x))
              (copy {} (var {} x)))
            (lit {type: (t-prim {} int32)} 0)))
    "#;
    // `xs |> vmap(grad(f))` -- the seed is the batched input, the stage is the
    // bare `(vmap {} (grad {} f) axis=0)` callable.
    let pipe_src = r#"
        (pipe {type: (t-tensor {} (d-lit {} 3) (d-lit {} 1) (t-prim {} f32))}
          (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 1) (t-prim {} f32))} xs)
          (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} int32)} 0)))
    "#;
    let app_src = r#"
        (app {type: (t-tensor {} (d-lit {} 3) (d-lit {} 1) (t-prim {} f32))}
          (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} int32)} 0))
          (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 1) (t-prim {} f32))} xs))
    "#;

    let mut program_defs = HashMap::new();
    program_defs.insert("loss".to_string(), parse_one(fn_src));
    let xs_ty = f32_mat(3, 1);
    let scoped = HashMap::from([("xs".to_string(), xs_ty)]);

    let pipe_dag = lower_subexpr_program(
        &parse_one(pipe_src),
        scoped.clone(),
        HashMap::new(),
        program_defs.clone(),
    );
    let app_dag = lower_subexpr_program(&parse_one(app_src), scoped, HashMap::new(), program_defs);

    let inputs = HashMap::from([(
        "xs".to_string(),
        TensorValue::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]),
    )]);

    let pipe_roots: Vec<NodeId> = pipe_dag.roots().to_vec();
    let app_roots: Vec<NodeId> = app_dag.roots().to_vec();
    assert!(!pipe_roots.is_empty());
    assert!(!app_roots.is_empty());

    let pipe_values =
        eval_tensor_roots_with_strict(&pipe_dag, &pipe_roots, |name| inputs.get(name).cloned())
            .expect("pipe eval");
    let app_values =
        eval_tensor_roots_with_strict(&app_dag, &app_roots, |name| inputs.get(name).cloned())
            .expect("app eval");

    let pipe_out = &pipe_values[pipe_roots.last().unwrap()];
    let app_out = &app_values[app_roots.last().unwrap()];
    assert_eq!(pipe_out.shape, app_out.shape);
    assert_eq!(pipe_out.shape, vec![3, 1]);
    // vmap(grad(sum(x*x)))([[1],[2],[3]]) = [[2],[4],[6]]
    let expected = [2.0, 4.0, 6.0];
    for (i, want) in expected.iter().enumerate() {
        assert!(
            (pipe_out.data[i] - want).abs() < 1e-6,
            "pipe vmap-grad[{i}] must be {want}, got {:?}",
            pipe_out.data,
        );
    }
    for (p, a) in pipe_out.data.iter().zip(app_out.data.iter()) {
        assert!(
            (p - a).abs() < 1e-6,
            "pipe and non-pipe vmap-grad must agree element-wise: pipe={p}, app={a}",
        );
    }
}
