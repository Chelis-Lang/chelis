//! Phase F: `lower_program_with_context` composes a library DAG with new
//! code without mutating the library and produces evaluator-equivalent
//! results to monolithic `lower_program(library + new)` lowering.
//!
//! Acceptance criteria, mirrored from the Phase F plan:
//! 1. New API reachable from `chelis_ir`.
//! 2. Existing `lower_program` continues to work (covered by the workspace).
//! 3. Evaluator parity for at least 3 library + snippet pairs (this file).
//! 4. Node-ID disjointness — library-max + 1 ≤ min(new-code-node IDs).
//! 5. Library DAG is not mutated (`&context_dag` is `&` only).

use chelis_unord::UnordMap;

use chelis_deep::Expr;
use chelis_ir::dag::{Dag, NodeId};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with};
use chelis_ir::{
    LoweredLibrary, lower_program, lower_program_to_library, lower_program_with_context,
};
use chelis_types::{
    CheckedProgram, TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_linearity,
};

fn parse(src: &str) -> Vec<Expr> {
    chelis_deep::parser::parse_str(src).expect("deep parse")
}

fn check_lib(library_src: &str) -> (Vec<Expr>, TypeEnv, CheckedProgram) {
    let library_exprs = parse(library_src);
    let env = build_type_env_from_library(&library_exprs).expect("library type-checks clean");

    // Build a CheckedProgram for the library by running the standard
    // monolithic IR check check on the library source itself, then the
    // effect + linearity passes — this is the same pipeline `lower_program`
    // expects.
    let checked = check_ir_program(&library_exprs).expect("library IR check clean");
    let checked = chelis_effects::check_program(&checked).expect("library effect clean");
    let checked = check_linearity(&checked).expect("library linearity clean");
    (library_exprs, env, checked)
}

fn check_with_ctx(ctx: &TypeEnv, new_src: &str) -> CheckedProgram {
    let new_exprs = parse(new_src);
    let checked = check_ir_with_context(ctx, &new_exprs).expect("new-code IR check clean");
    let checked = chelis_effects::check_program(&checked).expect("new-code effect clean");
    check_linearity(&checked).expect("new-code linearity clean")
}

fn check_monolithic(combined_src: &str) -> CheckedProgram {
    let exprs = parse(combined_src);
    let checked = check_ir_program(&exprs).expect("monolithic IR check clean");
    let checked = chelis_effects::check_program(&checked).expect("monolithic effect clean");
    check_linearity(&checked).expect("monolithic linearity clean")
}

fn eval_dag_root_values(dag: &Dag, inputs: &UnordMap<String, TensorValue>) -> Vec<TensorValue> {
    let roots: Vec<NodeId> = dag.roots().to_vec();
    let values = eval_tensor_roots_with(dag, &roots, |name| inputs.get(name).cloned())
        .expect("eval succeeds");
    roots
        .iter()
        .map(|id| {
            values
                .get(id)
                .cloned()
                .unwrap_or_else(|| TensorValue::scalar(0.0))
        })
        .collect()
}

fn eval_named_roots(
    dag: &Dag,
    inputs: &UnordMap<String, TensorValue>,
    interesting: &[&str],
) -> UnordMap<String, TensorValue> {
    // Collect the (single) root index per requested name. For a library
    // `(def {} foo body)` the root is the body's NodeId. For tuple-decomposed
    // defs the roots are Store nodes whose `name` is `foo.N`. We iterate roots
    // and pick out matching names by matching the Store name OR (for non-Store
    // roots) by re-running the DAG and indexing under the bare def name in
    // root-document-order.
    use chelis_ir::dag::RiscOp;

    let roots = dag.roots();
    let values = eval_tensor_roots_with(dag, roots, |name| inputs.get(name).cloned())
        .expect("eval succeeds");

    let mut out = UnordMap::new();
    for &root in roots {
        if let Some(node) = dag.get(root)
            && let RiscOp::Store { name } = &node.op
            && interesting.iter().any(|n| *n == name.as_str())
        {
            out.insert(name.as_str().to_string(), values[&root].clone());
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Acceptance: evaluator parity (monolithic vs composed).
// ─────────────────────────────────────────────────────────────────────────────

/// One library + snippet parity case: monolithic `lower_program` and
/// composed `lower_program_with_context` must produce the same last-root
/// shape and byte-identical data, and that data must equal `expected`
/// (the spot-check that guards against an evaluator bug masking BOTH
/// paths into agreement on a wrong value).
struct ParityCase {
    label: &'static str,
    library_src: &'static str,
    new_src: &'static str,
    expected: Vec<f64>,
}

#[test]
fn parity_monolithic_vs_composed_for_library_snippet_pairs() {
    let cases = [
        ParityCase {
            label: "lib_const referenced by new code (lib_const * 3.0)",
            library_src: "
                (def {} lib_const (lit {type: (t-tensor {} (t-prim {} f32))} 7.0))
            ",
            new_src: "
                (def {} y
                  (app {type: (t-tensor {} (t-prim {} f32))}
                       (var {} mul)
                       (var {type: (t-tensor {} (t-prim {} f32))} lib_const)
                       (lit {type: (t-tensor {} (t-prim {} f32))} 3.0)))
            ",
            expected: vec![21.0],
        },
        ParityCase {
            label: "lib function inlined into new code (double(5.0))",
            library_src: "
                (defsig {} double
                  (t-fn {} (t-tensor {} (t-prim {} f32)) (t-tensor {} (t-prim {} f32))))
                (def {} double
                  (fn {} (params {} (x {type: (t-tensor {} (t-prim {} f32))}))
                    (app {type: (t-tensor {} (t-prim {} f32))}
                         (var {} mul)
                         (var {type: (t-tensor {} (t-prim {} f32))} x)
                         (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))))
            ",
            new_src: "
                (def {} a
                  (app {type: (t-tensor {} (t-prim {} f32))}
                       (var {} double)
                       (lit {type: (t-tensor {} (t-prim {} f32))} 5.0)))
            ",
            expected: vec![10.0],
        },
        ParityCase {
            label: "lib value and lib function combined (affine(2.0) = 2.0 * 4.0 + 1.5)",
            library_src: "
                (def {} bias (lit {type: (t-tensor {} (t-prim {} f32))} 1.5))
                (defsig {} affine
                  (t-fn {} (t-tensor {} (t-prim {} f32)) (t-tensor {} (t-prim {} f32))))
                (def {} affine
                  (fn {} (params {} (x {type: (t-tensor {} (t-prim {} f32))}))
                    (app {type: (t-tensor {} (t-prim {} f32))}
                         (var {} add)
                         (app {type: (t-tensor {} (t-prim {} f32))}
                              (var {} mul)
                              (var {type: (t-tensor {} (t-prim {} f32))} x)
                              (lit {type: (t-tensor {} (t-prim {} f32))} 4.0))
                         (var {type: (t-tensor {} (t-prim {} f32))} bias))))
            ",
            new_src: "
                (def {} z
                  (app {type: (t-tensor {} (t-prim {} f32))}
                       (var {} affine)
                       (lit {type: (t-tensor {} (t-prim {} f32))} 2.0)))
            ",
            expected: vec![9.5],
        },
    ];

    for case in &cases {
        let combined_src = format!("{}\n{}", case.library_src, case.new_src);

        // Monolithic baseline.
        let mono_checked = check_monolithic(&combined_src);
        let mono_dag = lower_program(&mono_checked);
        let mono_roots = eval_dag_root_values(&mono_dag, &UnordMap::new());

        // Composed.
        let (_lib_exprs, ctx, lib_checked) = check_lib(case.library_src);
        let library: LoweredLibrary = lower_program_to_library(&lib_checked);
        let new_checked = check_with_ctx(&ctx, case.new_src);
        let composed_dag = lower_program_with_context(&library, &new_checked);
        let composed_roots = eval_dag_root_values(&composed_dag, &UnordMap::new());

        // The new-code root must be present in BOTH and agree.
        let mono_root = mono_roots
            .last()
            .unwrap_or_else(|| panic!("[{}] mono has at least one root", case.label))
            .clone();
        let composed_root = composed_roots
            .last()
            .unwrap_or_else(|| panic!("[{}] composed has at least one root", case.label))
            .clone();
        assert_eq!(
            mono_root.shape, composed_root.shape,
            "[{}] root shape must agree",
            case.label,
        );
        assert_eq!(
            mono_root.to_f64_lossy_vec(),
            composed_root.to_f64_lossy_vec(),
            "[{}] root value must agree byte-for-byte: mono={:?} composed={:?}",
            case.label,
            mono_root.to_f64_lossy_vec(),
            composed_root.to_f64_lossy_vec(),
        );
        // Spot-check the actual numeric expectation so that an
        // evaluator-side bug masking BOTH paths into wrongness doesn't
        // slip past.
        assert_eq!(
            composed_root.to_f64_lossy_vec(),
            case.expected,
            "[{}] composed root value must equal the spot-checked expectation",
            case.label,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Acceptance: node-ID disjointness.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn node_id_disjointness_new_code_above_library_max() {
    let library_src = "
        (def {} lib_a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
        (def {} lib_b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
        (def {} lib_c (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
    ";
    let new_src = "
        (def {} new_d
          (app {type: (t-tensor {} (t-prim {} f32))}
               (var {} add)
               (var {type: (t-tensor {} (t-prim {} f32))} lib_a)
               (var {type: (t-tensor {} (t-prim {} f32))} lib_b)))
    ";

    let (_, ctx, lib_checked) = check_lib(library_src);
    let library = lower_program_to_library(&lib_checked);
    let library_max_id = library
        .dag()
        .nodes()
        .iter()
        .map(|node| node.id.0)
        .max()
        .expect("library has nodes");

    let new_checked = check_with_ctx(&ctx, new_src);
    let composed_dag = lower_program_with_context(&library, &new_checked);

    // All library NodeIds appear unchanged in the composed DAG.
    for lib_node in library.dag().nodes() {
        let in_composed = composed_dag.get(lib_node.id).expect("library node present");
        assert_eq!(
            in_composed.op, lib_node.op,
            "library node {:?}: op must match cloned library DAG",
            lib_node.id,
        );
        assert_eq!(
            in_composed.inputs, lib_node.inputs,
            "library node {:?}: inputs must match",
            lib_node.id,
        );
    }

    // Every node added BY new-code lowering has id strictly greater than
    // the library's max id.
    let composed_max = composed_dag
        .nodes()
        .iter()
        .map(|node| node.id.0)
        .max()
        .expect("composed has nodes");
    assert!(
        composed_max > library_max_id,
        "composed DAG must add new-code nodes above library max; \
         library_max={library_max_id}, composed_max={composed_max}",
    );

    let new_code_min_id = composed_dag
        .nodes()
        .iter()
        .skip(library.dag().len())
        .map(|node| node.id.0)
        .min()
        .expect("composed has new-code nodes beyond library");
    assert!(
        new_code_min_id > library_max_id,
        "every new-code node id must be > library_max ({library_max_id}); \
         observed new_code_min_id={new_code_min_id}",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Acceptance: immutability — `&library` is `&` only.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn library_dag_is_not_mutated_by_lower_program_with_context() {
    let library_src = "
        (def {} lib_x (lit {type: (t-tensor {} (t-prim {} f32))} 11.0))
    ";
    let new_src = "
        (def {} y
          (app {type: (t-tensor {} (t-prim {} f32))}
               (var {} add)
               (var {type: (t-tensor {} (t-prim {} f32))} lib_x)
               (lit {type: (t-tensor {} (t-prim {} f32))} 1.0)))
    ";

    let (_, ctx, lib_checked) = check_lib(library_src);
    let library = lower_program_to_library(&lib_checked);

    let library_len_before = library.dag().len();
    let library_roots_before = library.dag().roots().to_vec();
    let library_node_ops_before: Vec<_> = library
        .dag()
        .nodes()
        .iter()
        .map(|n| (n.id, n.op.clone(), n.inputs.clone()))
        .collect();

    let new_checked = check_with_ctx(&ctx, new_src);
    let _composed_dag = lower_program_with_context(&library, &new_checked);

    // Library carrier must be untouched after composition.
    assert_eq!(library.dag().len(), library_len_before);
    assert_eq!(library.dag().roots(), &library_roots_before[..]);
    let after: Vec<_> = library
        .dag()
        .nodes()
        .iter()
        .map(|n| (n.id, n.op.clone(), n.inputs.clone()))
        .collect();
    assert_eq!(library_node_ops_before, after, "library DAG nodes mutated");

    // And the composed call is repeatable (no hidden state) — running again
    // must produce the same result.
    let again = lower_program_with_context(&library, &new_checked);
    let evaled_a = eval_dag_root_values(&_composed_dag, &UnordMap::new());
    let evaled_b = eval_dag_root_values(&again, &UnordMap::new());
    assert_eq!(
        evaled_a.last().map(|v| v.to_f64_lossy_vec().clone()),
        evaled_b.last().map(|v| v.to_f64_lossy_vec().clone()),
        "repeated composed lowering must be deterministic",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Acceptance: lib node referenced (not duplicated) when new code uses a
// library value def by name.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn library_value_def_is_referenced_not_duplicated() {
    // Library has a single Const node; new code references it three times.
    // A monolithic lowering produces three uses of the same NodeId. The
    // composed lowering must do the same: the NodeId of `lib_v` in the
    // composed DAG is identical to its id in `library.dag`, and the
    // new-code add nodes have it among their inputs.
    let library_src = "
        (def {} lib_v (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
    ";
    let new_src = "
        (def {} use1
          (app {type: (t-tensor {} (t-prim {} f32))}
               (var {} add)
               (var {type: (t-tensor {} (t-prim {} f32))} lib_v)
               (var {type: (t-tensor {} (t-prim {} f32))} lib_v)))
    ";

    let (_, ctx, lib_checked) = check_lib(library_src);
    let library = lower_program_to_library(&lib_checked);
    let lib_v_id = *library
        .symbol_table()
        .get("lib_v")
        .expect("library symbol_table holds top-level def `lib_v`");

    let new_checked = check_with_ctx(&ctx, new_src);
    let composed_dag = lower_program_with_context(&library, &new_checked);

    // `lib_v` occupies the same NodeId in the composed DAG as in the library.
    let lib_v_in_composed = composed_dag.get(lib_v_id).expect("lib_v present");
    let lib_v_in_library = library.dag().get(lib_v_id).expect("lib_v in library");
    assert_eq!(lib_v_in_composed.op, lib_v_in_library.op);

    // At least one new-code node has `lib_v_id` as an input — i.e. the
    // composed DAG references the library node directly rather than
    // duplicating it.
    let referenced = composed_dag
        .nodes()
        .iter()
        .skip(library.dag().len())
        .any(|node| node.inputs.contains(&lib_v_id));
    assert!(
        referenced,
        "expected at least one new-code node to reference library node {lib_v_id:?} as an input; \
         composed DAG: {:?}",
        composed_dag
            .nodes()
            .iter()
            .skip(library.dag().len())
            .map(|n| (n.id, n.op.clone(), n.inputs.clone()))
            .collect::<Vec<_>>(),
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative coverage: `lower_program(library)` and the LoweredLibrary `dag`
// field must agree, so the spec's `context_dag = lower_program(library)`
// invariant holds.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lowered_library_dag_matches_lower_program() {
    let library_src = "
        (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
        (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
        (def {} c
          (app {type: (t-tensor {} (t-prim {} f32))}
               (var {} add)
               (var {type: (t-tensor {} (t-prim {} f32))} a)
               (var {type: (t-tensor {} (t-prim {} f32))} b)))
    ";
    let (_, _, lib_checked) = check_lib(library_src);
    let dag_a = lower_program(&lib_checked);
    let library = lower_program_to_library(&lib_checked);
    assert_eq!(dag_a.len(), library.dag().len());
    assert_eq!(dag_a.roots(), library.dag().roots());
    for (na, nb) in dag_a.nodes().iter().zip(library.dag().nodes().iter()) {
        assert_eq!(na.id, nb.id);
        assert_eq!(na.op, nb.op);
        assert_eq!(na.inputs, nb.inputs);
        assert_eq!(na.output_type, nb.output_type);
    }
}

// Quiet unused-warning for the helper — only the named-roots variant is
// exercised by some compositions; keep the import stable.
#[allow(dead_code)]
fn _suppress_unused() {
    let _ = eval_named_roots;
}
