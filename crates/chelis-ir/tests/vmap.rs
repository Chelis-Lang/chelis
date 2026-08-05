use chelis_ir::dag::{Dag, DimExpr, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::vmap::vectorize_axis0;
use chelis_types::check_ir_program;
use chelis_types::types::Prim;
use std::collections::HashMap;

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn mat_f32(m: usize, n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(m), DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn eval_root(dag: &Dag, inputs: &HashMap<String, TensorValue>) -> TensorValue {
    let root = dag.roots()[0];
    let values = eval_tensor_roots_with_strict(dag, &[root], |name| inputs.get(name).cloned())
        .expect("evaluation should succeed");
    values[&root].clone()
}

#[test]
fn vmap_elementwise_vectorizes_axis_zero() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3), None);
    let y = dag.add_node(RiscOp::Neg, vec![x], vec_f32(3), None);
    dag.add_root(y);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let value = eval_root(
        &vmapped,
        &HashMap::from([(
            "x".to_string(),
            TensorValue::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, -5.0, 6.0]),
        )]),
    );
    assert_eq!(value.shape, vec![2, 3]);
    assert_eq!(
        value.to_f64_lossy_vec(),
        vec![-1.0, -2.0, -3.0, -4.0, 5.0, -6.0]
    );
}

#[test]
fn vmap_reduction_shifts_the_reduced_axis() {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let y = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(2),
        None,
    );
    dag.add_root(y);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(2)).expect("vmap should succeed");
    let value = eval_root(
        &vmapped,
        &HashMap::from([(
            "x".to_string(),
            TensorValue::from_vec(
                vec![2, 2, 3],
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 8.0, 9.0,
                ],
            ),
        )]),
    );
    assert_eq!(value.shape, vec![2, 2]);
    assert_eq!(value.to_f64_lossy_vec(), vec![6.0, 15.0, 60.0, 24.0]);
}

#[test]
fn vmap_nested_adds_multiple_batch_axes() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4), None);
    dag.add_root(x);

    let inner = vectorize_axis0(&dag, DimInfo::Lit(3)).expect("inner vmap should succeed");
    let outer = vectorize_axis0(&inner, DimInfo::Lit(2)).expect("outer vmap should succeed");
    let root = outer.roots()[0];
    let node = outer.get(root).expect("root");
    assert_eq!(
        node.output_type,
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        }
    );
}

#[test]
fn vmap_batched_matmul_stays_in_expand_mul_sum_form() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let a_exp = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: DimExpr::Concrete(4),
        },
        vec![a],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let b_exp = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: DimExpr::Concrete(2),
        },
        vec![b],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let prod = dag.add_node(
        RiscOp::Mul,
        vec![a_exp, b_exp],
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![prod],
        mat_f32(2, 4),
        None,
    );
    dag.add_root(out);

    let vmapped = vectorize_axis0(&dag, DimInfo::Lit(5)).expect("vmap should succeed");
    assert!(
        vmapped
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::Expand { axis: 3, .. })),
        "batched matmul should stay in the generic expand->mul->sum decomposition"
    );
}

// =====================================================================
// chelis#524: a non-constant `vmap` mapped (batch) axis must FAIL CLOSED
// at lowering — a loud, LOCATED, FATAL diagnostic — never silently
// default to axis 0.
//
// `resolve_callable_expr_inner`'s `Some("vmap")` arm read the axis with
// `kids.get(1).and_then(extract_usize_value).unwrap_or(0)`, which
// collapsed BOTH "axis arg absent" (the documented vmap default) AND
// "axis arg present but non-constant" to axis 0. A runtime mapped axis
// then silently vectorized the WRONG axis — the same default-to-0
// anti-pattern #364 eliminated for reductions / softmax / gather/scatter.
//
// The surf parser only admits a literal `axis=N`, so a runtime axis is
// reachable through the Deep (`.dp`) input path (or any internal IR
// transform): the program below `chelis check`s clean and pre-fix
// `chelis build` succeeded, silently lowering the `vmap` over axis 0.
// Negative parity with #364's `extract_axis_raw`.
// =====================================================================

/// A `vmap` over `tensor[batch, features]` whose mapped axis is the bound
/// runtime parameter `ax: int32` (a non-constant). Spans are carried on the
/// axis node so the lowering diagnostic is LOCATED.
const VMAP_RUNTIME_AXIS_DEEP: &str = r#"
(defsig {} process
  (t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32))
           (t-tensor {} (d-name {} features) (t-prim {} f32))))
(def {} process
  (fn {} (params {} (x {type: (t-tensor {} (d-name {} features) (t-prim {} f32))}))
    (app {} (var {} relu) (var {} x))))
(defsig {} batch_process
  (t-fn {} (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
           (t-prim {} int32)
           (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
(def {} batch_process
  (fn {} (params {}
           (xs {type: (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))})
           (ax {type: (t-prim {} int32)}))
    (pipe {} (var {} xs)
      (vmap {} (var {} process) (var {span: "dp:vmap-runtime-axis"} ax)))))
"#;

/// The same program with a CONSTANT (`lit`) mapped axis — the negative
/// parity control. Identical except the axis node is a literal, so it must
/// lower cleanly.
const VMAP_CONST_AXIS_DEEP: &str = r#"
(defsig {} process
  (t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32))
           (t-tensor {} (d-name {} features) (t-prim {} f32))))
(def {} process
  (fn {} (params {} (x {type: (t-tensor {} (d-name {} features) (t-prim {} f32))}))
    (app {} (var {} relu) (var {} x))))
(defsig {} batch_process
  (t-fn {} (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
           (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
(def {} batch_process
  (fn {} (params {}
           (xs {type: (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))}))
    (pipe {} (var {} xs)
      (vmap {} (var {} process) (lit {type: (t-prim {} int32)} 0)))))
"#;

fn check_effects_linearity_deep(deep_src: &str) -> chelis_types::CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(deep_src).expect("deep parse");
    let checked = check_ir_program(&exprs)
        .unwrap_or_else(|r| panic!("deep program must check clean: {:?}", r.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    chelis_types::check_linearity(&checked).expect("linearity")
}

fn lower_surf_program(src: &str) -> Result<Dag, chelis_ir::lower::LowerDiagnostic> {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("expand")
    .into_exprs();
    let checked = check_ir_program(&exprs)
        .unwrap_or_else(|r| panic!("surf program must check clean: {:?}", r.errors));
    let checked = chelis_effects::check_program(&checked).expect("effects");
    let checked = chelis_types::check_linearity(&checked).expect("linearity");
    chelis_ir::lower::try_lower_program(&checked)
}

/// REJECT: a present-but-non-constant `vmap` mapped axis must lower to a
/// FATAL, LOCATED diagnostic naming `vmap` and the runtime-axis cause — not
/// silently default to axis 0 (which pre-fix let `chelis build` succeed).
#[test]
fn issue524_runtime_vmap_axis_is_fatal_located_lowering_error() {
    let checked = check_effects_linearity_deep(VMAP_RUNTIME_AXIS_DEEP);
    let diag = chelis_ir::lower::try_lower_program(&checked)
        .expect_err("a runtime vmap axis must REJECT at lowering, not silently default to 0");
    assert!(
        diag.fatal,
        "the runtime-axis reject must be FATAL (not host-fallback-absorbable); got {diag:?}"
    );
    assert!(
        diag.message.contains("vmap") && diag.message.contains("axis"),
        "the diagnostic must name `vmap` and the axis; got: {}",
        diag.message
    );
    assert!(
        diag.message.contains("compile-time")
            && diag
                .message
                .contains("runtime axis must be rejected at check time"),
        "the diagnostic must explain the compile-time-constant requirement (mirroring #364); \
         got: {}",
        diag.message
    );
    // LOCATED: the axis node carried a span, so the diagnostic must point at
    // it — a default-to-0 would have no location at all.
    assert_eq!(
        diag.span_id.as_deref(),
        Some("dp:vmap-runtime-axis"),
        "the reject must be located at the axis sub-expression; got {:?}",
        diag.span_id
    );
}

/// CONTROL (constant axis): the same program with a `lit` axis lowers
/// cleanly — FAIL-CLOSED must not become reject-everything.
#[test]
fn issue524_constant_vmap_axis_still_lowers() {
    let checked = check_effects_linearity_deep(VMAP_CONST_AXIS_DEEP);
    let dag = chelis_ir::lower::try_lower_program(&checked)
        .expect("a constant vmap axis must still lower cleanly");
    assert!(!dag.nodes().is_empty(), "lowered DAG must be non-empty");
}

/// CONTROL (absent axis): the documented `vmap` contract — an OMITTED axis
/// argument defaults to axis 0 — must still hold. `vmap(process)` lowers
/// cleanly (the `None` branch, not the rejected non-constant branch).
#[test]
fn issue524_absent_vmap_axis_defaults_to_zero_and_lowers() {
    let src = "def process(x: tensor[features, f32]) -> tensor[features, f32] = relu(x)\n\
         def batch_process(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] = xs |> vmap(process)\n";
    let dag = lower_surf_program(src).expect("an absent vmap axis must default to 0 and lower");
    assert!(!dag.nodes().is_empty(), "lowered DAG must be non-empty");
}

/// CONTROL (explicit nonzero constant axis via Surf): the canonical named
/// `axis=1` form lowers cleanly.
#[test]
fn issue524_explicit_nonzero_constant_vmap_axis_via_surf_lowers() {
    let src = "def process(x: tensor[features, f32]) -> tensor[features, f32] = relu(x)\n\
         def batch_process(xs: tensor[features, batch, f32]) -> tensor[features, batch, f32] = xs |> vmap(process, axis=1)\n";
    let dag = lower_surf_program(src).expect("an explicit constant vmap axis must lower");
    assert!(!dag.nodes().is_empty(), "lowered DAG must be non-empty");
}
