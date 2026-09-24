//! chelis#1277 Slice B2h: the shared kernel decision, `host_def_kernel`,
//! answers "is this def a kernel" before any lowering runs, and its answer
//! is the one the C host program acts on.
//!
//! Each `None` below names a class the byte-identity corpus found the kernel
//! lowering rejecting AFTER a kernel decision (the C lane then fell through
//! to host code, chelis#1515; the eval lane errored). The decision now names
//! the class itself, so both lanes choose host code for the same reason and
//! no lowering error is caught to decide a lane.
//!
//! EVIDENTIARY STATUS: `a_pure_tensor_def_is_a_kernel`,
//! `an_io_effect_row_is_host` and `a_match_on_a_constructor_literal_stays_a_kernel`
//! are disposition locks (they hold on the tree without the walker too); the
//! four others are regression tests, watched returning `Err` (a failed kernel
//! lowering) on the tree without the walker and `Ok(false)` with it.

use chelis_ir::host::{HostLoweringSession, host_def_kernel};

fn checked_deep(src: &str) -> chelis_types::CheckedProgram {
    let exprs = chelis_deep::parser::parse_str(src).expect("parse Deep fixture");
    let checked = chelis_types::check_ir_program(
        &exprs,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .unwrap_or_else(|report| {
        panic!(
            "type check failed: {:?}",
            report
                .errors
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
        )
    });
    let checked =
        chelis_effects::check_program(&checked, &mut chelis_types::NativeServices::new(None))
            .expect("effects");
    chelis_types::check_linearity(&checked, &mut chelis_types::NativeServices::new(None))
        .expect("linearity")
}

fn decision(src: &str, name: &str) -> Result<bool, String> {
    let program = checked_deep(src);
    chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        host_def_kernel(
            &mut HostLoweringSession::new(&program),
            name,
            None,
            runtime,
            &mut chelis_types::NativeServices::new(None),
        )
    })
    .map(|kernel| kernel.is_some())
    .map_err(|diagnostic| diagnostic.to_string())
}

#[test]
fn a_pure_tensor_def_is_a_kernel() {
    let src = include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/a_pure_tensor_def_is_a_kernel.deep"
    );
    assert_eq!(decision(src, "f"), Ok(true));
}

#[test]
fn an_io_effect_row_is_host() {
    let src = include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/an_io_effect_row_is_host.deep"
    );
    assert_eq!(decision(src, "f"), Ok(false));
}

/// chelis#520 D1: only a compile-time-known constructor scrutinee lowers. The
/// match sits inside a `let` block (the issue_1200 shape): a `match` at the
/// body ROOT is already on the keep-in-host list, so only the nested form
/// reaches the walker.
#[test]
fn a_match_on_a_runtime_scrutinee_is_host_before_lowering() {
    let src = include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/a_match_on_a_runtime_scrutinee_is_host_before_lowering.deep"
    );
    assert_eq!(decision(src, "run"), Ok(false));
}

/// The positive twin: a constructor-literal scrutinee inside the same block
/// stays a kernel, so the class does not swallow the static matches the
/// lowering does select.
#[test]
fn a_match_on_a_constructor_literal_stays_a_kernel() {
    let src = include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/a_match_on_a_constructor_literal_stays_a_kernel.deep"
    );
    assert_eq!(decision(src, "run"), Ok(true));
}

/// chelis#1058: the compiled lowering carries literal window and stride
/// lists only; a runtime list keeps the def in host code on both lanes.
#[test]
fn a_runtime_window_list_is_host_before_lowering() {
    let src = include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/a_runtime_window_list_is_host_before_lowering.deep"
    );
    assert_eq!(decision(src, "f"), Ok(false));
}

/// chelis#776: a `pad` fill that does not resolve statically keeps the def in
/// host code on both lanes.
#[test]
fn a_runtime_pad_fill_is_host_before_lowering() {
    let src = include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/a_runtime_pad_fill_is_host_before_lowering.deep"
    );
    assert_eq!(decision(src, "f"), Ok(false));
}

/// A host-only builtin reached through a `let` inside the body (the
/// issue_1222 shape the corpus found).
#[test]
fn a_host_only_builtin_inside_the_body_is_host_before_lowering() {
    let src = include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/a_host_only_builtin_inside_the_body_is_host_before_lowering.deep"
    );
    assert_eq!(decision(src, "f"), Ok(false));
}

/// Section 4.7 orders interface checks by the declaring signature even when
/// the implementation extracts a tensor helper. Shape-only inputs remain ABI
/// witnesses; an unrelated dead input must not change their relative order.
#[test]
fn tensor_helper_retains_declaring_parameter_order() {
    use chelis_ir::host::{RandomLoweringState, lower_named_tensor_entry_dag};
    let program = checked_deep(include_str!(
        "fixtures/core/tests/issue_1277_kernel_decision/tensor_helper_retains_declaring_parameter_order.deep"
    ));
    for random in [
        None,
        Some(RandomLoweringState {
            seed: Some(7),
            counter: 9,
        }),
    ] {
        let kernel = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            host_def_kernel(
                &mut HostLoweringSession::new(&program),
                "f",
                random,
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap()
        .unwrap();
        assert_eq!(
            kernel
                .inputs
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["b", "z", "a"]
        );
        assert_eq!(kernel.next_random_counter, random.map(|s| s.counter));
        assert_load_order_after_rebuilding(&kernel.dag, &["b", "z", "a"]);
    }
    let dag = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        lower_named_tensor_entry_dag(
            &program,
            "f",
            runtime,
            &mut chelis_types::NativeServices::new(None),
        )
    })
    .unwrap();
    assert_load_order_after_rebuilding(&dag, &["b", "z", "a"]);
}

fn assert_load_order_after_rebuilding(dag: &chelis_ir::Dag, expected: &[&str]) {
    let mut dag = chelis_ir::optimize::dead_code_eliminate(dag);
    chelis_ir::optimize::constant_fold(&mut dag);
    let dag = chelis_ir::optimize::dead_code_eliminate(&dag);
    let names = dag
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            chelis_ir::RiscOp::Load { name } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(names, expected);
}

/// A signatureless subexpression has an assigned deterministic ABI order.
/// Reversing insertion into its unordered scope must not reverse that ABI.
#[test]
fn signatureless_helper_retains_assigned_abi_order() {
    use chelis_ir::dag::{DimInfo, TensorType};
    use chelis_types::types::Prim;
    use chelis_unord::UnordMap;
    let expr = chelis_deep::parser::parse_str("(app {} (var {} add) (var {} z) (var {} a))")
        .unwrap()
        .remove(0);
    for names in [["z", "a"], ["a", "z"]] {
        let mut scope = UnordMap::new();
        for name in names {
            scope.insert(
                name.to_owned(),
                TensorType {
                    dims: vec![DimInfo::Lit(3)],
                    precision: Prim::F32,
                },
            );
        }
        let dag = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
            chelis_ir::lower::try_lower_subexpr_program(
                &expr,
                scope,
                UnordMap::new(),
                UnordMap::new(),
                runtime,
                &mut chelis_types::NativeServices::new(None),
            )
        })
        .unwrap();
        assert_load_order_after_rebuilding(&dag, &["a", "z"]);
    }
}
