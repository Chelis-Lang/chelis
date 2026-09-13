//! Spec08 materialization barrier and spec04 exact dtype preservation.
//! Shipped Realize transports stored bits; admitting it does not admit new arithmetic.
use chelis_compiler_api::compiler::reject_unsupported_hip_ops;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn dag(precision: Prim, materialization: bool) -> Dag {
    let mut dag = Dag::new();
    let ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision,
    };
    let input = dag.add_node(
        RiscOp::Load {
            name: "input".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let (op, inputs) = if materialization {
        (RiscOp::Realize, vec![input])
    } else {
        (RiscOp::Add, vec![input, input])
    };
    let result = dag.add_node(op, inputs, ty, None);
    dag.add_root(result);
    dag
}

fn generated_realize_source(precision: Prim) -> String {
    let raw = dag(precision, true);
    reject_unsupported_hip_ops(&raw).expect("shipped exact-bit Realize must pass the HIP gate");
    let selected = chelis_backend_hip::prepare_dag_for_codegen(raw);
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(selected)
            .expect("shipped exact-bit Realize must lower ownership"),
    )
    .expect("shipped exact-bit Realize must verify ownership");
    chelis_backend_hip::codegen_hip(verified, &format!("realize_{}", precision.name()))
        .expect("shipped exact-bit Realize must reach HIP code generation")
        .c_source
}

#[test]
fn hip_gate_and_codegen_accept_shipped_exact_narrow_float_materialization() {
    for (precision, dtype_macro) in [
        (Prim::Bf16, "CHELIS_DTYPE_BF16"),
        (Prim::F16, "CHELIS_DTYPE_F16"),
    ] {
        let source = generated_realize_source(precision);
        let kernel = format!("kernel_realize_{dtype_macro}");
        assert!(
            source.contains(&format!("void {kernel}(\\n"))
                && source.contains("const unsigned char *a"),
            "Realize must use the raw-byte materialization kernel: {source}"
        );
        assert!(
            source.contains("for (chelis_device_metadata byte = 0; byte < 2; ++byte)"),
            "f16/bf16 Realize must copy exactly two stored bytes per element: {source}"
        );
        assert!(
            source.contains(&format!("chelis_launch_kernel(mod_{kernel}, \"{kernel}\"")),
            "Realize must launch its materialization kernel: {source}"
        );
    }
}

#[test]
fn hip_materialization_does_not_authorize_unimplemented_narrow_float_arithmetic() {
    for precision in [Prim::Bf16, Prim::F16] {
        let error = reject_unsupported_hip_ops(&dag(precision, false))
            .expect_err("Realize support must not admit unimplemented arithmetic");
        assert!(
            error
                .errors
                .iter()
                .any(|diagnostic| diagnostic.message.contains("narrow-float compute")),
            "{error:?}"
        );
    }
}

#[test]
fn hip_generic_dtype_rejection_points_to_the_matrix_without_a_capability_list() {
    let error = reject_unsupported_hip_ops(&dag(Prim::F8e4m3, false))
        .expect_err("f8e4m3 must reach the generic HIP dtype rejection");
    let message = &error.errors[0].message;
    assert!(
        message.contains("spec/04-type-system.md §1.1.3"),
        "{message}"
    );
    for stale in [
        "Supported: f32/f64/bool",
        "planned Realize",
        "dedicated ReLU nodes",
        "spec/04-type-system.md §5.7.1",
    ] {
        assert!(
            !message.contains(stale),
            "generic dtype rejection repeated stale capability prose `{stale}`: {message}"
        );
    }
}
