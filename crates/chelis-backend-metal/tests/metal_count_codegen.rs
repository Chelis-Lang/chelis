//! First-class Metal Count codegen contract (chelis#1291 / [05-OP-29]).

mod support;

use chelis_backend_metal::codegen_metal_host_program;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use support::{codegen_metal, lowered_host_program, try_codegen_metal};

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
fn metal_count_emits_one_dedicated_multi_axis_checked_balanced_kernel() {
    let dag = count_dag(
        lit_ty(&[2, 3, 5], Prim::Bool),
        vec![2, 0],
        lit_ty(&[3], Prim::Int64),
    );
    let generated = codegen_metal(&dag, "count_multi");
    let source = &generated.mm_source;

    assert!(!source.contains("M1 fallback stub"), "{source}");
    assert!(source.contains("kernel void k_count_1"), "{source}");
    assert_eq!(source.matches("kernel void k_count_1").count(), 1);
    assert!(source.contains("constant int __count_selected[8] = { 1, 0, 1"));
    assert!(source.contains("for (int __count_axis = (int)dims.input_ndim - 1;"));
    // The bool carrier is one byte on Metal (`metal_elem_size(Bool) == 1`);
    // the kernel reads it as `device const uchar*`, never MSL `bool` (whose
    // load would normalize a noncanonical byte to true), and rejects any
    // byte outside {0, 1} through status code 1.
    assert!(source.contains("device const uchar* input"), "{source}");
    assert!(!source.contains("device const bool* input"), "{source}");
    assert!(
        source.contains("chelis_count_record_error(count_error, 1)"),
        "{source}"
    );
    assert!(source.contains("array<long, 64> __count_frame_start"));
    assert!(source.contains("array<long, 64> __count_frame_len"));
    assert!(source.contains("array<uchar, 64> __count_frame_state"));
    assert!(source.contains("array<long, 64> __count_frame_left"));
    assert!(source.contains("while (__count_sp >= 0)"));
    assert!(source.contains("while ((__count_split << 1) < (ulong)__count_len)"));
    assert!(source.contains("0x7fffffffffffffffL - __count_right"));
    assert!(source.contains("atomic_compare_exchange_weak_explicit"));
    assert!(source.contains("numeric trap: overflow in count at int64"));
    // The status word is device scratch from the runtime, not a tensor
    // buffer, so it stays outside the no-reuse tensor-allocation plan; its
    // width comes from the emitter's dtype authority, not a header spelling.
    assert!(
        source.contains("chelis_metal_alloc_status_word(sizeof(int32_t))"),
        "{source}"
    );
    assert_eq!(
        source.matches("= chelis_metal_alloc(").count(),
        2,
        "one Load buffer and one Count output buffer:\n{source}"
    );
    // Peak bytes: 30 one-byte inputs + 3 int64 outputs + the 4-byte status word.
    assert_eq!(generated.peak_device_bytes_estimate, Some(30 + 3 * 8 + 4));

    for forbidden in ["k_reduce_sum", "k_cast", "count_host_fallback"] {
        assert!(
            !source.contains(forbidden),
            "Metal Count must not emit forbidden alias/fallback {forbidden:?}:\n{source}"
        );
    }
}

#[test]
fn metal_count_accepts_rank_three_named_fixed_dims_and_empty_selected_extent() {
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
    let source = codegen_metal(&named, "count_empty").mm_source;
    assert!(!source.contains("M1 fallback stub"), "{source}");
    assert!(source.contains("if (__count_len == 0)"), "{source}");
    assert!(source.contains("__count_value = 0"), "{source}");
    assert!(source.contains("kernel void k_count_1"), "{source}");
}

#[test]
fn metal_count_grammar_is_rejected_at_the_sealed_ownership_boundary() {
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
fn metal_count_rejects_its_device_rank_limit_at_the_backend_boundary() {
    // The rank limit is the device carrier's, not the language's: ownership
    // lowering admits the DAG, and the backend's typed validator reports the
    // chelis#1844 capability cell before any kernel is emitted.
    let dag = count_dag(
        lit_ty(&[1; 9], Prim::Bool),
        vec![8],
        lit_ty(&[1; 8], Prim::Int64),
    );
    let error = match try_codegen_metal(&dag, "invalid_count") {
        Ok(_) => panic!("an over-rank Count must not reach Metal kernel emission"),
        Err(error) => error,
    };
    let rendered = error.to_string();
    assert!(
        rendered.contains("unimplemented chelis#1844:"),
        "{rendered}"
    );
    assert!(
        rendered.contains("exceeds the Metal device rank limit"),
        "{rendered}"
    );
}

#[test]
fn metal_count_rejects_extents_beyond_the_uint32_device_index_limit() {
    // Metal's launch geometry and the kernel's `ChelisCountDims` index in
    // `uint`; an input with more than `u32::MAX` elements is a device carrier
    // limit (chelis#1844) the typed validator reports before any kernel is
    // emitted.
    let dag = count_dag(
        lit_ty(&[65_536, 65_537], Prim::Bool),
        vec![1],
        lit_ty(&[65_536], Prim::Int64),
    );
    let error = match try_codegen_metal(&dag, "count_beyond_uint32") {
        Ok(_) => panic!("an input beyond the uint32 index limit must not reach Metal emission"),
        Err(error) => error,
    };
    let rendered = error.to_string();
    assert!(
        rendered.contains("unimplemented chelis#1844:"),
        "{rendered}"
    );
    assert!(
        rendered.contains("exceeds the uint32 device indexing limit"),
        "{rendered}"
    );
}

#[test]
fn metal_host_program_externalizes_every_count_helper_without_a_stub() {
    let (helpers, verified) = lowered_host_program(ONE_COUNT_HELPER, "count_program");
    let generated = codegen_metal_host_program(&verified, "count_program", helpers)
        .expect("host Count helper emits as Metal");
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
    let device = &helper.result.mm_source;
    assert!(device.contains("kernel void k_count_"), "{device}");
    assert!(
        device.contains(&format!("void {}(", helper.name)),
        "{device}"
    );
    assert!(!device.contains("M1 fallback stub"), "{device}");
}

#[test]
fn metal_host_program_isolates_multiple_count_helpers() {
    let (helpers, verified) = lowered_host_program(TWO_COUNT_HELPERS, "two_counts");
    let generated = codegen_metal_host_program(&verified, "two_counts", helpers)
        .expect("each host Count helper emits as Metal");
    assert_eq!(generated.device_helpers.len(), 2);
    assert_ne!(
        generated.device_helpers[0].name,
        generated.device_helpers[1].name
    );
    for helper in &generated.device_helpers {
        assert!(helper.result.mm_source.contains("kernel void k_count_"));
        assert!(
            !generated
                .host
                .c_source
                .contains(&format!("static void {}(", helper.name))
        );
    }
}

#[test]
fn metal_host_program_rejects_an_external_helper_outside_the_manifest() {
    let (mut helpers, verified) = lowered_host_program(ONE_COUNT_HELPER, "count_program");
    for helper in &mut helpers {
        helper.name = format!("{}_renamed", helper.name);
    }
    let error = codegen_metal_host_program(&verified, "count_program", helpers)
        .err()
        .expect("a helper name outside the wrapper manifest must not link by accident");
    let rendered = error.to_string();
    assert!(
        rendered.contains("unknown external host tensor helper"),
        "{rendered}"
    );
    assert!(rendered.contains("deliberate [04-TOT-2]"), "{rendered}");
}

/// Disposition lock on the runtime helper's body, not a regression test: Metal
/// zero-fills a fresh `StorageModeShared` buffer today, so no executable path
/// can observe the `memset`. It is defense against a future reusing
/// allocator, and this lock is what keeps it from being deleted as dead code.
#[test]
fn metal_status_word_helper_zeroes_the_buffer_it_returns() {
    let header = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/runtime/chelis_metal_runtime.h"
    ))
    .expect("Metal runtime header is readable");
    let signature = "static inline id<MTLBuffer> chelis_metal_alloc_status_word(size_t bytes) {";
    let start = header
        .find(signature)
        .expect("the status-word helper keeps its byte-count signature");
    let body = &header[start + signature.len()..];
    let body = &body[..body.find("\n}").expect("helper body closes")];
    assert!(
        body.contains("chelis_metal_alloc(bytes)"),
        "the status word is allocated through the shared allocator at the caller's width:\n{body}"
    );
    assert!(
        body.contains("memset([buf contents], 0, bytes);"),
        "the status word must be zeroed through the untyped contents pointer before it is returned:\n{body}"
    );
    assert!(
        body.find("memset").expect("zeroing present")
            < body.find("return buf").expect("returns the buffer"),
        "zeroing must precede the return:\n{body}"
    );
    assert!(
        !body.contains("sizeof"),
        "the helper spells no width of its own:\n{body}"
    );
}
