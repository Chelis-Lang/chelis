//! First-class HIP Count codegen contract (chelis#1291 / [05-OP-29]).

mod support;

use chelis_backend_hip::codegen_hip_host_program;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::{codegen_hip, lowered_host_program};

const ONE_COUNT_HELPER: &str = "\
mask: tensor[2, 3, bool] = [[true, false, true], [false, true, true]]\n\
rows: tensor[2, int64] = count(&mask, 1)\n";

const TWO_COUNT_HELPERS: &str = "\
mask: tensor[2, 3, bool] = [[true, false, true], [false, true, true]]\n\
rows: tensor[2, int64] = count(&mask, 1)\n\
columns: tensor[3, int64] = count(&mask, 0)\n";

fn ty(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
    TensorType { dims, precision }
}

fn lit_ty(dims: &[usize], precision: Prim) -> TensorType {
    ty(dims.iter().copied().map(DimInfo::Lit).collect(), precision)
}

fn count_dag(input: TensorType, axes: Vec<usize>, output: TensorType) -> Dag {
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load {
            name: "mask".into(),
        },
        vec![],
        input,
        None,
    );
    let count = dag.add_node(RiscOp::Count { axes }, vec![input], output, None);
    dag.add_root(count);
    dag
}

#[test]
fn hip_count_emits_one_dedicated_multi_axis_checked_balanced_kernel() {
    let dag = count_dag(
        lit_ty(&[2, 3, 5], Prim::Bool),
        vec![2, 0],
        lit_ty(&[3], Prim::Int64),
    );
    let generated = codegen_hip(&dag, "count_multi").expect("valid Count emits on HIP");
    let source = &generated.c_source;

    assert!(source.contains("kernel_count_1"), "{source}");
    assert_eq!(source.matches("const char *kernel_count_1_src").count(), 1);
    assert!(source.contains("const int __count_selected[8] = { 1, 0, 1"));
    assert!(
        source
            .contains("for (int __count_axis = input_ndim - 1; __count_axis >= 0; --__count_axis)")
    );
    // chelis#1308's tagged carrier stores bool as `Repr::Bool8`: the kernel
    // reads exactly one byte per element and rejects any byte outside {0, 1}.
    assert!(source.contains("const unsigned char *a"), "{source}");
    assert!(
        source.contains("chelis_count_record_error(count_error, 1)"),
        "{source}"
    );
    assert!(source.contains("long long __count_frame_start[64]"));
    assert!(source.contains("long long __count_frame_len[64]"));
    assert!(source.contains("unsigned char __count_frame_state[64]"));
    assert!(source.contains("long long __count_frame_left[64]"));
    assert!(source.contains("while (__count_sp >= 0)"));
    assert!(source.contains("while ((__count_split << 1) < (unsigned long long)__count_len)"));
    assert!(source.contains("INT64_MAX - __count_right"));
    assert!(source.contains("chelis_count_record_error(count_error, 2)"));
    assert!(source.contains("numeric trap: overflow in count at int64"));
    // Peak bytes: 30 one-byte inputs + 3 int64 outputs + the 4-byte status word.
    assert_eq!(generated.peak_device_bytes_estimate, Some(30 + 3 * 8 + 4));

    for forbidden in [
        "RiscOp::Sum",
        "kernel_sum",
        "kernel_cast",
        "count_host_fallback",
        "M1 fallback stub",
    ] {
        assert!(
            !source.contains(forbidden),
            "HIP Count must not emit forbidden alias/fallback {forbidden:?}:\n{source}"
        );
    }
}

#[test]
fn hip_count_named_fixed_dims_and_empty_selected_extent_keep_the_same_kernel() {
    let named = count_dag(
        ty(
            vec![
                DimInfo::Named("batch".into(), Some(2)),
                DimInfo::Named("seq".into(), Some(0)),
                DimInfo::Named("feature".into(), Some(5)),
            ],
            Prim::Bool,
        ),
        vec![1],
        ty(
            vec![
                DimInfo::Named("batch".into(), Some(2)),
                DimInfo::Named("feature".into(), Some(5)),
            ],
            Prim::Int64,
        ),
    );
    let source = codegen_hip(&named, "count_empty")
        .expect("named fixed Count emits")
        .c_source;
    assert!(source.contains("if (__count_len == 0)"), "{source}");
    assert!(source.contains("__count_value = 0"), "{source}");
    assert!(source.contains("kernel_count_1"), "{source}");
}

#[test]
fn hip_count_grammar_is_rejected_at_the_sealed_ownership_boundary() {
    // The backend only ever receives a verified ownership program, and
    // ownership lowering enforces [05-OP-29]'s operand grammar itself, so a
    // malformed Count never reaches kernel emission. Each case names the
    // lowering invariant that stops it.
    let cases = [
        (
            count_dag(
                lit_ty(&[2, 3], Prim::F32),
                vec![1],
                lit_ty(&[2], Prim::Int64),
            ),
            "requires bool input",
        ),
        (
            count_dag(
                lit_ty(&[2, 3], Prim::Bool),
                vec![1],
                lit_ty(&[2], Prim::Int32),
            ),
            "requires int64 output",
        ),
        (
            count_dag(
                lit_ty(&[2, 3], Prim::Bool),
                vec![0, 1],
                lit_ty(&[], Prim::Int64),
            ),
            "strictly descending",
        ),
        (
            count_dag(
                lit_ty(&[2, 3], Prim::Bool),
                vec![2],
                lit_ty(&[2], Prim::Int64),
            ),
            "axis out of range",
        ),
    ];

    for (dag, needle) in cases {
        let error = match chelis_ir::ownership::lower_dag_ownership(dag) {
            Ok(_) => panic!("malformed Count must not lower to a verified program"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("count at node 1"), "{error}");
        assert!(error.contains(needle), "{error}");
    }
}

#[test]
fn hip_count_rejects_its_device_rank_limit_at_the_backend_boundary() {
    // The rank limit is the device carrier's, not the language's: ownership
    // lowering admits the DAG, and the backend's typed validator reports the
    // chelis#1345 capability cell before any kernel is emitted.
    let dag = count_dag(
        lit_ty(&[1; 9], Prim::Bool),
        vec![8],
        lit_ty(&[1; 8], Prim::Int64),
    );
    let error = match codegen_hip(&dag, "invalid_count") {
        Ok(_) => panic!("an over-rank Count must not reach HIP kernel emission"),
        Err(error) => error,
    };
    let rendered = error.to_string();
    assert!(
        rendered.contains("unimplemented chelis#1345:"),
        "{rendered}"
    );
    assert!(
        rendered.contains("exceeds the HIP device rank limit"),
        "{rendered}"
    );
}

#[test]
fn hip_host_program_externalizes_every_count_helper() {
    let (helpers, verified) = lowered_host_program(ONE_COUNT_HELPER, "count_program");
    let generated = codegen_hip_host_program(&verified, "count_program", helpers)
        .expect("host Count helper emits as HIP");
    assert_eq!(
        generated.device_helpers.len(),
        1,
        "{}",
        generated.host.c_source
    );
    let helper = &generated.device_helpers[0];
    assert!(
        helper.name.contains("count_program__global__tensor_"),
        "{}",
        helper.name
    );
    assert!(
        generated.host.c_source.contains(&format!(
            "void {}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);",
            helper.name
        )),
        "{}",
        generated.host.c_source
    );
    assert!(
        !generated
            .host
            .c_source
            .contains(&format!("static void {}(", helper.name)),
        "the host wrapper must declare, not embed, the device Count helper:\n{}",
        generated.host.c_source
    );
    assert!(
        generated
            .host
            .c_source
            .contains(&format!("{}(", helper.name)),
        "{}",
        generated.host.c_source
    );
    assert!(
        helper.result.c_source.contains("kernel_count_"),
        "{}",
        helper.result.c_source
    );
    assert!(
        helper
            .result
            .c_source
            .contains(&format!("void {}(", helper.name)),
        "{}",
        helper.result.c_source
    );
}

#[test]
fn hip_host_program_isolates_multiple_count_helpers() {
    let (helpers, verified) = lowered_host_program(TWO_COUNT_HELPERS, "two_counts");
    let generated = codegen_hip_host_program(&verified, "two_counts", helpers)
        .expect("each host Count helper emits as HIP");
    assert_eq!(generated.device_helpers.len(), 2);
    assert_ne!(
        generated.device_helpers[0].name,
        generated.device_helpers[1].name
    );
    for helper in &generated.device_helpers {
        assert!(helper.result.c_source.contains("kernel_count_"));
        assert!(
            !generated
                .host
                .c_source
                .contains(&format!("static void {}(", helper.name))
        );
    }
}

#[test]
fn hip_host_program_rejects_an_external_helper_outside_the_manifest() {
    let (mut helpers, verified) = lowered_host_program(ONE_COUNT_HELPER, "count_program");
    for helper in &mut helpers {
        helper.name = format!("{}_renamed", helper.name);
    }
    let error = codegen_hip_host_program(&verified, "count_program", helpers)
        .err()
        .expect("a helper name outside the wrapper manifest must not link by accident");
    let rendered = error.to_string();
    assert!(
        rendered.contains("unknown external host tensor helper"),
        "{rendered}"
    );
    assert!(rendered.contains("deliberate [04-TOT-2]"), "{rendered}");
}
