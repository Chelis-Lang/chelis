mod support;
use chelis_ir::dag::{Dag, DimInfo, FusedInput, FusedStep, FusedStepOp, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::fs;
use std::process::Command;
use support::codegen;

/// Physical storage outlives a logical last use unless the emitter releases it.
/// Cover retained dead slots, exact-capacity recycling, and early explicit Drop.
/// Use scratch-free operators: this plan bounds DAG slots, not kernel scratch.
#[test]
fn physical_slot_bound_covers_executed_allocation_lifetimes() {
    use chelis_ir::ownership::{LiveByteBound, plan_c_storage};

    for (reduce, drop_first, expected_bound, expected_peak, expected_result) in [
        (false, false, 8, 8, "1"),
        (true, false, 24, 24, "-1"),
        (false, true, 20, 16, "2"),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let scalar = TensorType::scalar_f32();
        let first_type = if reduce || drop_first {
            vec_f32(4)
        } else {
            scalar.clone()
        };
        let first = dag.add_node(
            decl,
            RiscOp::synth_const(Prim::F32, 1.0),
            vec![],
            first_type.clone(),
            None,
        );
        let result = if drop_first {
            dag.add_node(decl, RiscOp::Drop, vec![first], first_type, None);
            dag.add_node(
                decl,
                RiscOp::synth_const(Prim::F32, 2.0),
                vec![],
                scalar,
                None,
            )
        } else {
            let op = if reduce {
                RiscOp::MaxReduce { axis: 0 }
            } else {
                RiscOp::Neg
            };
            let second = dag.add_node(decl, op, vec![first], scalar.clone(), None);
            dag.add_node(decl, RiscOp::Neg, vec![second], scalar, None)
        };
        dag.add_root(result);
        let plan = plan_c_storage(support::verified_dag(&dag, Default::default())).unwrap();
        assert_eq!(plan.max_live_bytes(), LiveByteBound::Exact(expected_bound));
        let generated = codegen(&dag, "physical_peak_probe").unwrap();
        // These fixtures have only unique f32 descriptors. Count actual payload
        // allocation/release, not logical owners or cumulative allocation traffic.
        let instrumented = generated
            .c_source
            .replace("chelis_alloc(", "probe_alloc(")
            .replace("chelis_tensor_release(", "probe_release(");
        let prefix = r#"
#include <stdio.h>
#include "chelis_runtime.h"
static int64_t live_bytes = 0, peak_bytes = 0;
static chelis_tensor *probe_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype) {
    chelis_tensor *t = chelis_alloc(rank, shape, dtype);
    live_bytes += chelis_tensor_numel(t) * sizeof(float);
    if (live_bytes > peak_bytes) peak_bytes = live_bytes;
    return t;
}
static void probe_release(chelis_tensor *t) {
    live_bytes -= chelis_tensor_numel(t) * sizeof(float);
    chelis_tensor_release(t);
}
"#;
        let main = r#"
int main(void) {
    chelis_tensor *out[1] = {0};
    physical_peak_probe(NULL, 0, out, 1);
    chelis_read_view view = chelis_tensor_read_view(out[0]);
    printf("peak=%lld result=%g\n", (long long)peak_bytes, (double)*(const float *)view.data);
    probe_release(out[0]);
    return live_bytes != 0;
}
"#;
        let dir = tempfile::tempdir().unwrap();
        let staged = chelis_runtime_bundle::stage(dir.path())
            .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
        let source = dir.path().join("probe.c");
        fs::write(&source, format!("{prefix}\n{instrumented}\n{main}")).unwrap();
        let toolchain = chelis_backend_c::toolchain::test_toolchain(generated.requirements);
        let bin = dir.path().join("probe");
        let compile = Command::new(toolchain.compiler)
            .arg("-O0")
            .arg("-I")
            .arg(dir.path())
            .args(toolchain.compile_flags)
            .arg(source)
            .arg(&staged.archive)
            .args(toolchain.link_flags)
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        assert!(
            compile.status.success(),
            "{}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let run = Command::new(bin).output().unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(
            String::from_utf8(run.stdout).unwrap(),
            format!("peak={expected_peak} result={expected_result}\n")
        );
        assert!(expected_bound >= expected_peak);
    }
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn compile_and_run(
    test_name: &str,
    c_source: &str,
    requirements: chelis_backend_c::toolchain::CodegenRequirements,
) -> String {
    let tmp = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    fs::write(tmp.path().join("model.c"), c_source).unwrap();
    fs::write(
        tmp.path().join("main.c"),
        r#"
#include <stdio.h>
#include "chelis_runtime.h"

void fused_in_place_probe(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

static void print_tensor(const char *label, chelis_tensor *t) {
    chelis_read_view view = chelis_tensor_read_view(t);
    printf("%s", label);
    for (int64_t i = 0; i < chelis_tensor_numel(t); i++) {
        printf(" %.6f", ((const float*)view.data)[i]);
    }
    printf("\n");
}

int main(void) {
    int64_t shape4[1] = { 4 };

    chelis_tensor *x = chelis_alloc(1, shape4, CHELIS_DTYPE_F32);
    chelis_tensor_write *x_guard = chelis_tensor_begin_write(x);
    chelis_write_view x_view = chelis_tensor_write_view(x_guard);
    for (int i = 0; i < 4; i++) ((float*)x_view.data)[i] = (float)(i + 1);
    chelis_tensor_end_write(x_guard);
    chelis_tensor *inputs0[1] = { x };
    chelis_tensor *outputs0[1] = { 0 };
    fused_in_place_probe(inputs0, 1, outputs0, 1);
    print_tensor("contig_out", outputs0[0]);
    print_tensor("contig_input", x);
    chelis_tensor_release(outputs0[0]);
    chelis_tensor_release(x);
    return 0;
}
"#,
    )
    .unwrap();

    let toolchain = chelis_backend_c::toolchain::test_toolchain(requirements);
    let bin = tmp.path().join(test_name);
    let mut cmd = Command::new(&toolchain.compiler);
    cmd.args(["-O2", "-I"]);
    cmd.arg(tmp.path());
    cmd.args(&toolchain.compile_flags);
    cmd.arg(tmp.path().join("main.c"));
    cmd.arg(tmp.path().join("model.c"));
    cmd.arg(&staged.archive);
    cmd.args(&toolchain.link_flags);
    cmd.arg("-o");
    cmd.arg(&bin);
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "C compile failed:\n{}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        c_source
    );

    let run = Command::new(bin).output().unwrap();
    assert!(
        run.status.success(),
        "binary failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}

/// Compile and run a fused chain whose reusable input is a program
/// input, over the canonical contiguous public tensor representation.
///
/// chelis#933: this test used to assert `contig_input 2.0 4.0 6.0 8.0`,
/// i.e. it baked the *mutated* caller buffer into the expected stdout
/// and pinned the defect as correct. A program input belongs to the
/// caller, so the kernel must leave it at `1 2 3 4`. Non-canonical
/// external view metadata is not constructible through the exact public
/// tensor ABI; movement operations materialize canonical tensors instead.
#[test]
fn fused_in_place_compile_run_preserves_canonical_caller_input() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let scale = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let ops = vec![FusedStep {
        op: FusedStepOp::Mul,
        input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
    }];
    let fused = dag.add_node(
        decl,
        RiscOp::FusedElem { ops },
        vec![x, scale],
        vec_f32(4),
        None,
    );
    dag.set_reusable_input(fused, x);
    dag.add_root(fused);

    let result = codegen(&dag, "fused_in_place_probe").unwrap();
    let stdout = compile_and_run(
        "fused_in_place_probe",
        &result.c_source,
        result.requirements,
    );

    assert_eq!(
        stdout,
        "\
contig_out 2.000000 4.000000 6.000000 8.000000
contig_input 1.000000 2.000000 3.000000 4.000000
"
    );
}

/// A program-owned `Copy` with one terminal fused consumer receives the
/// shared reuse token. The emitted C transfers that descriptor, repurposes
/// its exact metadata, computes the right result, and leaves the caller's
/// entry tensor untouched.
#[test]
fn fused_in_place_compile_run_reuses_program_owned_storage() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let owned = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(4), None);
    let scale = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let ops = vec![FusedStep {
        op: FusedStepOp::Mul,
        input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
    }];
    let fused = dag.add_node(
        decl,
        RiscOp::FusedElem { ops },
        vec![owned, scale],
        vec_f32(4),
        None,
    );
    dag.set_reusable_input(fused, owned);
    dag.add_root(fused);

    let result = codegen(&dag, "fused_in_place_probe").unwrap();
    assert!(
        result.c_source.contains("chelis_tensor *t3 = t1;"),
        "the fused output must transfer the token-selected descriptor"
    );
    assert!(
        result
            .c_source
            .contains("chelis_tensor_repurpose(t3, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C(1)), (chelis_scalar[]){ chelis_scalar_from_bits(CHELIS_DTYPE_I64, 4) });"),
        "the transferred descriptor must receive exact output metadata"
    );
    let stdout = compile_and_run(
        "fused_program_owned_reuse_probe",
        &result.c_source,
        result.requirements,
    );

    assert_eq!(
        stdout,
        "\
contig_out 2.000000 4.000000 6.000000 8.000000
contig_input 1.000000 2.000000 3.000000 4.000000
"
    );
}

/// `Drop` destroys its source descriptor as well as releasing the backing
/// storage. A later equal-capacity owner therefore needs a fresh descriptor;
/// the generic slot recycler may not resurrect the released pointer.
#[test]
fn drop_then_equal_capacity_owner_allocates_a_fresh_descriptor() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let disposable = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(4), None);
    dag.add_node(decl, RiscOp::Drop, vec![disposable], vec_f32(4), None);
    let output = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(4), None);
    dag.add_root(output);

    let result = codegen(&dag, "fused_in_place_probe").unwrap();
    let guard_end = result
        .c_source
        .find("chelis_tensor_end_write(t1_write_guard);")
        .expect("Drop must end the source write guard");
    let release = result
        .c_source
        .find("chelis_tensor_release(t1);")
        .expect("Drop must release the source descriptor");
    assert!(
        guard_end < release,
        "Drop must end the guard before release"
    );
    assert!(
        !result.c_source.contains("chelis_tensor *t3 = t1;"),
        "the released descriptor must not be selected for later slot reuse"
    );
    let stdout = compile_and_run(
        "drop_then_equal_capacity_owner_probe",
        &result.c_source,
        result.requirements,
    );

    assert_eq!(
        stdout,
        "\
contig_out 1.000000 2.000000 3.000000 4.000000
contig_input 1.000000 2.000000 3.000000 4.000000
"
    );
}
