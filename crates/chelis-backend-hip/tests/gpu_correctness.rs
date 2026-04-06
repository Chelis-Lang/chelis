//! GPU execution correctness tests (G1-G9).
//!
//! These require a HIP-capable GPU plus `hipcc`/`hiprtc`.
//! They are `#[ignore]` by default — run with:
//!     cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
//!
//! Manual gate per AGENTS.md: not part of default CI.

use chelis_backend_hip::codegen_hip;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_types::types::Prim;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone)]
struct TestInput {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl TestInput {
    fn new(name: &str, shape: &[usize], data: &[f32]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.to_vec(),
        }
    }

    fn evaluator_value(&self) -> TensorValue {
        TensorValue::from_vec(
            self.shape.clone(),
            self.data.iter().copied().map(f64::from).collect(),
        )
    }
}

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn bool_scalar() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::Bool,
    }
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn vec_bool(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Bool,
    }
}

fn mat_f32(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    }
}

fn write_temp_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write temp file");
    path
}

fn hip_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(chelis_backend_hip::runtime_dir())
}

fn cpu_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-backend-c/runtime")
}

fn require_hipcc() {
    let output = Command::new("hipcc")
        .arg("--version")
        .output()
        .unwrap_or_else(|err| panic!("failed to probe hipcc: {err}"));
    assert!(
        output.status.success(),
        "hipcc is required for the manual HIP gate"
    );
}

fn c_shape(shape: &[usize]) -> (usize, Vec<usize>) {
    if shape.is_empty() {
        (1, vec![1])
    } else {
        (shape.len(), shape.to_vec())
    }
}

fn build_main_cpp(
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    inputs: &[TestInput],
) -> String {
    let mut lines = Vec::new();

    if input_labels.is_empty() {
        lines.push("    chelis_tensor **inputs = NULL;".to_string());
    } else {
        lines.push(format!(
            "    chelis_tensor *input_storage[{}] = {{0}};",
            input_labels.len()
        ));
        lines.push("    chelis_tensor **inputs = input_storage;".to_string());
        for (slot, label) in input_labels.iter().enumerate() {
            let input = inputs
                .iter()
                .find(|candidate| &candidate.name == label)
                .unwrap_or_else(|| panic!("missing test input '{label}'"));
            let (ndim, c_dims) = c_shape(&input.shape);
            let dims = c_dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!("    int shape_{slot}[{ndim}] = {{ {dims} }};"));
            lines.push(format!(
                "    input_storage[{slot}] = chelis_alloc({ndim}, shape_{slot}, CHELIS_F32);"
            ));
            for (idx, value) in input.data.iter().enumerate() {
                lines.push(format!(
                    "    input_storage[{slot}]->data[{idx}] = {:.8}f;",
                    value
                ));
            }
        }
    }

    lines.push(format!("    chelis_tensor *outputs[{n_out}] = {{0}};"));
    lines.push(format!(
        "    {func_name}(inputs, {}, outputs, {n_out});",
        input_labels.len()
    ));
    lines.push(format!(
        "    for (int out_idx = 0; out_idx < {n_out}; out_idx++) {{"
    ));
    lines.push("        for (int i = 0; i < outputs[out_idx]->size; i++) {".to_string());
    lines.push("            if (i > 0) printf(\" \");".to_string());
    lines.push("            printf(\"%.6f\", outputs[out_idx]->data[i]);".to_string());
    lines.push("        }".to_string());
    lines.push("        printf(\"\\n\");".to_string());
    lines.push("        chelis_free(outputs[out_idx]);".to_string());
    lines.push("    }".to_string());
    for slot in 0..input_labels.len() {
        lines.push(format!("    chelis_free(input_storage[{slot}]);"));
    }

    format!(
        r#"#include "chelis_runtime.h"
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
{body}
    return 0;
}}
"#,
        body = lines.join("\n")
    )
}

fn compile_and_run_single_output(dag: &Dag, func_name: &str, inputs: &[TestInput]) -> Vec<f32> {
    require_hipcc();
    let result = codegen_hip(dag, func_name);
    assert_eq!(
        result.output_labels.len(),
        1,
        "manual harness currently expects a single output"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    let cpu_rt = cpu_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    write_temp_file(
        tmp.path(),
        "chelis_runtime.h",
        &fs::read_to_string(cpu_rt.join("chelis_runtime.h")).expect("cpu runtime header"),
    );
    write_temp_file(
        tmp.path(),
        "chelis_runtime.c",
        &fs::read_to_string(cpu_rt.join("chelis_runtime.c")).expect("cpu runtime source"),
    );
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    write_temp_file(
        tmp.path(),
        "main.cpp",
        &build_main_cpp(
            func_name,
            &result.input_labels,
            result.output_labels.len(),
            inputs,
        ),
    );

    let bin_path = tmp.path().join("gpu_correctness_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(tmp.path().join("chelis_runtime.c"));
    compile_cmd.args(&result.link_flags);
    compile_cmd.arg("-o");
    compile_cmd.arg(&bin_path);
    let compile = compile_cmd.output().expect("run hipcc");
    assert!(
        compile.status.success(),
        "hipcc failed:\nstderr: {}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );

    let run = Command::new(&bin_path).output().expect("run gpu binary");
    assert!(
        run.status.success(),
        "GPU binary failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .map(|token| token.parse::<f32>().expect("parse output float"))
        .collect()
}

fn expected_single_output(dag: &Dag, inputs: &[TestInput]) -> Vec<f32> {
    let input_map: HashMap<String, TensorValue> = inputs
        .iter()
        .map(|input| (input.name.clone(), input.evaluator_value()))
        .collect();
    let values =
        eval_tensor_roots_with_strict(dag, dag.roots(), |name| input_map.get(name).cloned())
            .expect("evaluator should succeed");
    let root = *dag.roots().first().expect("single root expected");
    values[&root]
        .data
        .iter()
        .copied()
        .map(|x| x as f32)
        .collect()
}

fn assert_close_vec(actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "output length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= 1e-4,
            "mismatch at index {idx}: expected {e}, got {a}"
        );
    }
}

fn assert_gpu_matches_eval(dag: &Dag, func_name: &str, inputs: &[TestInput]) {
    let actual = compile_and_run_single_output(dag, func_name, inputs);
    let expected = expected_single_output(dag, inputs);
    assert_close_vec(&actual, &expected);
}

// ===========================================================================
// G1: add(const(1), const(2)) = 3.0 on GPU
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g1_add_consts_gpu() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let c = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
    dag.add_root(c);

    let actual = compile_and_run_single_output(&dag, "g1_add_consts", &[]);
    assert_eq!(actual, vec![3.0]);
}

// ===========================================================================
// G2: All unary ops GPU == evaluator within tolerance
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_neg_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::Neg, vec![x], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g2_neg",
        &[TestInput::new("x", &[3], &[1.5, -2.0, 0.25])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_exp_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::Exp, vec![x], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g2_exp",
        &[TestInput::new("x", &[3], &[0.0, 1.0, -1.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_log_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::Log, vec![x], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g2_log",
        &[TestInput::new("x", &[3], &[1.0, std::f32::consts::E, 4.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_sin_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::Sin, vec![x], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g2_sin",
        &[TestInput::new(
            "x",
            &[3],
            &[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
        )],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_sqrt_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::Sqrt, vec![x], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g2_sqrt",
        &[TestInput::new("x", &[3], &[1.0, 4.0, 9.0])],
    );
}

// ===========================================================================
// G3: All binary ops GPU == evaluator within tolerance
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g3_add_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g3_add",
        &[
            TestInput::new("x", &[3], &[1.0, 2.0, 3.0]),
            TestInput::new("y", &[3], &[4.0, -1.0, 0.5]),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g3_mul_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::Mul, vec![x, y], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g3_mul",
        &[
            TestInput::new("x", &[3], &[2.0, -3.0, 0.5]),
            TestInput::new("y", &[3], &[4.0, -2.0, 8.0]),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g3_max_elem_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(3));
    let out = dag.add_node(RiscOp::MaxElem, vec![x, y], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g3_max_elem",
        &[
            TestInput::new("x", &[3], &[1.0, 8.0, -2.0]),
            TestInput::new("y", &[3], &[2.0, 3.0, -3.0]),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g3_cmplt_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(4));
    let out = dag.add_node(RiscOp::CmpLt, vec![x, y], vec_bool(4));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g3_cmplt",
        &[
            TestInput::new("x", &[4], &[1.0, 3.0, 5.0, 7.0]),
            TestInput::new("y", &[4], &[2.0, 3.0, 4.0, 8.0]),
        ],
    );
}

// ===========================================================================
// G4: Reduction GPU == evaluator
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g4_sum_reduction_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(2, 3));
    let out = dag.add_node(RiscOp::Sum { axis: 0 }, vec![x], vec_f32(3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g4_sum",
        &[TestInput::new(
            "x",
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g4_max_reduce_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(2, 3));
    let out = dag.add_node(RiscOp::MaxReduce { axis: 1 }, vec![x], vec_f32(2));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g4_max_reduce",
        &[TestInput::new(
            "x",
            &[2, 3],
            &[1.0, 9.0, 3.0, 4.0, 2.0, 6.0],
        )],
    );
}

// ===========================================================================
// G5: expand then add — stride-0 correct on GPU
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g5_expand_add_stride_zero() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let expanded = dag.add_node(RiscOp::Expand { axis: 0, size: 4 }, vec![x], mat_f32(4, 3));
    let c = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], mat_f32(4, 3));
    let out = dag.add_node(RiscOp::Add, vec![expanded, c], mat_f32(4, 3));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g5_expand_add",
        &[TestInput::new("x", &[3], &[1.0, 2.0, 3.0])],
    );
}

// ===========================================================================
// G6: permute then elementwise — reordered strides on GPU
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g6_permute_then_add() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(2, 3));
    let perm = dag.add_node(RiscOp::Permute { axes: vec![1, 0] }, vec![x], mat_f32(3, 2));
    let c = dag.add_node(RiscOp::Const { value: 1.5 }, vec![], mat_f32(3, 2));
    let out = dag.add_node(RiscOp::Add, vec![perm, c], mat_f32(3, 2));
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g6_permute_add",
        &[TestInput::new(
            "x",
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
    );
}

// ===========================================================================
// G7: Host↔device round-trip
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g7_host_device_roundtrip() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let out = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![x],
        vec_f32(4),
    );
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g7_roundtrip",
        &[TestInput::new("x", &[4], &[0.25, -1.5, 2.75, 9.0])],
    );
}

// ===========================================================================
// G8: Multiple kernels in sequence (add then mul)
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g8_multi_kernel_chain() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
    let add = dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());
    let c = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
    let out = dag.add_node(RiscOp::Mul, vec![add, c], scalar_f32());
    dag.add_root(out);

    let actual = compile_and_run_single_output(&dag, "g8_chain", &[]);
    assert_eq!(actual, vec![20.0]);
}

// ===========================================================================
// G9: Load mapping (named inputs)
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g9_load_mapping() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], scalar_f32());
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], scalar_f32());
    let out = dag.add_node(RiscOp::CmpLt, vec![x, y], bool_scalar());
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g9_load_mapping",
        &[
            TestInput::new("x", &[], &[2.0]),
            TestInput::new("y", &[], &[5.0]),
        ],
    );
}
