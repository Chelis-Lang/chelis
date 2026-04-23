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
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use std::collections::HashMap;
use std::env;
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

fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
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

fn cpu_runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn cpu_runtime_library_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("../../target/debug/deps"),
        manifest_dir.join("../../target/release/deps"),
    ];
    if let Ok(dir) = env::var("CHELIS_RUNTIME_DIR") {
        let candidate_dir = PathBuf::from(dir);
        if let Some(path) = fs::read_dir(&candidate_dir).ok().and_then(|entries| {
            entries.flatten().map(|entry| entry.path()).find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
            })
        }) {
            return path;
        }
    }
    for dir in candidates {
        if let Some(path) = fs::read_dir(&dir).ok().and_then(|entries| {
            entries.flatten().map(|entry| entry.path()).find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
            })
        }) {
            return path;
        }
    }
    panic!("could not locate libchelis_runtime.a for backend-hip manual tests");
}

fn copy_runtime_artifacts(dst: &Path) {
    let include = cpu_runtime_include_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        write_temp_file(
            dst,
            header,
            &fs::read_to_string(include.join(header)).unwrap_or_else(|_| panic!("read {header}")),
        );
    }
    fs::copy(cpu_runtime_library_path(), dst.join("libchelis_runtime.a"))
        .expect("copy rust runtime library");
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

fn append_case_lines(
    lines: &mut Vec<String>,
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    inputs: &[TestInput],
    prefix: &str,
) {
    if input_labels.is_empty() {
        lines.push(format!("    chelis_tensor **{prefix}_inputs = NULL;"));
    } else {
        lines.push(format!(
            "    chelis_tensor *{prefix}_input_storage[{}] = {{0}};",
            input_labels.len(),
        ));
        lines.push(format!(
            "    chelis_tensor **{prefix}_inputs = {prefix}_input_storage;"
        ));
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
            lines.push(format!(
                "    int {prefix}_shape_{slot}[{ndim}] = {{ {dims} }};"
            ));
            lines.push(format!(
                "    {prefix}_input_storage[{slot}] = chelis_alloc({ndim}, {prefix}_shape_{slot}, CHELIS_F32);"
            ));
            for (idx, value) in input.data.iter().enumerate() {
                lines.push(format!(
                    "    {prefix}_input_storage[{slot}]->data[{idx}] = {:.8}f;",
                    value
                ));
            }
        }
    }

    lines.push(format!(
        "    chelis_tensor *{prefix}_outputs[{n_out}] = {{0}};"
    ));
    lines.push(format!(
        "    {func_name}({prefix}_inputs, {}, {prefix}_outputs, {n_out});",
        input_labels.len()
    ));
    lines.push(format!(
        "    for (int {prefix}_out_idx = 0; {prefix}_out_idx < {n_out}; {prefix}_out_idx++) {{"
    ));
    lines.push(format!(
        "        for (int {prefix}_i = 0; {prefix}_i < {prefix}_outputs[{prefix}_out_idx]->size; {prefix}_i++) {{"
    ));
    lines.push(format!("            if ({prefix}_i > 0) printf(\" \");"));
    lines.push(format!(
        "            printf(\"%.6f\", {prefix}_outputs[{prefix}_out_idx]->data[{prefix}_i]);"
    ));
    lines.push("        }".to_string());
    lines.push("        printf(\"\\n\");".to_string());
    lines.push(format!(
        "        chelis_free({prefix}_outputs[{prefix}_out_idx]);"
    ));
    lines.push("    }".to_string());
    for slot in 0..input_labels.len() {
        lines.push(format!("    chelis_free({prefix}_input_storage[{slot}]);"));
    }
}

fn build_main_cpp(
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    inputs: &[TestInput],
) -> String {
    let mut lines = Vec::new();
    append_case_lines(&mut lines, func_name, input_labels, n_out, inputs, "case0");

    format!(
        r#"#include "chelis_runtime.h"
extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
{body}
    return 0;
}}
"#,
        body = lines.join("\n")
    )
}

fn build_multi_case_main_cpp(
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    input_cases: &[Vec<TestInput>],
) -> String {
    let mut lines = Vec::new();
    for (idx, inputs) in input_cases.iter().enumerate() {
        append_case_lines(
            &mut lines,
            func_name,
            input_labels,
            n_out,
            inputs,
            &format!("case{idx}"),
        );
    }

    format!(
        r#"#include "chelis_runtime.h"
extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

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
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    copy_runtime_artifacts(tmp.path());
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
    compile_cmd.arg(format!("-L{}", tmp.path().display()));
    compile_cmd.arg("-lchelis_runtime");
    compile_cmd.arg("-lpthread");
    compile_cmd.arg("-ldl");
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

fn compile_and_run_output_cases(
    dag: &Dag,
    func_name: &str,
    input_cases: &[Vec<TestInput>],
) -> Vec<Vec<f32>> {
    require_hipcc();
    let result = codegen_hip(dag, func_name);
    assert_eq!(
        result.output_labels.len(),
        1,
        "manual harness currently expects a single output"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    copy_runtime_artifacts(tmp.path());
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    write_temp_file(
        tmp.path(),
        "main.cpp",
        &build_multi_case_main_cpp(
            func_name,
            &result.input_labels,
            result.output_labels.len(),
            input_cases,
        ),
    );

    let bin_path = tmp.path().join("gpu_correctness_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(format!("-L{}", tmp.path().display()));
    compile_cmd.arg("-lchelis_runtime");
    compile_cmd.arg("-lpthread");
    compile_cmd.arg("-ldl");
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
    String::from_utf8(run.stdout)
        .expect("utf8 stdout")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|token| token.parse::<f32>().expect("parse output float"))
                .collect()
        })
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

fn assert_fused_gpu_matches_unfused_eval(dag: &Dag, func_name: &str, inputs: &[TestInput]) {
    let fused = fuse(dag);
    let actual = compile_and_run_single_output(&fused, func_name, inputs);
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

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g4_symbolic_row_sum_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("seq".into(), None),
        ],
        precision: Prim::F32,
    };
    let out_ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None)],
        precision: Prim::F32,
    };
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty);
    let out = dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], out_ty);
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g4_symbolic_sum",
        &[TestInput::new(
            "x",
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g4_symbolic_softmax_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("seq".into(), None),
        ],
        precision: Prim::F32,
    };
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty.clone());
    let out = chelis_ir::tier2::lower_softmax(&mut dag, x, 1, &x_ty);
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g4_symbolic_softmax",
        &[TestInput::new(
            "x",
            &[2, 3],
            &[1.0, 2.0, 3.0, 0.0, -1.0, 4.0],
        )],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g4_symbolic_matmul_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let a_ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], a_ty.clone());
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], b_ty.clone());
    let out = chelis_ir::tier2::lower_matmul(&mut dag, a, b, &a_ty, &b_ty);
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g4_symbolic_matmul",
        &[
            TestInput::new("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TestInput::new("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g4_symbolic_matmul_gpu_reuses_one_artifact_for_multiple_batch_sizes() {
    let mut dag = Dag::new();
    let a_ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], a_ty.clone());
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], b_ty.clone());
    let out = chelis_ir::tier2::lower_matmul(&mut dag, a, b, &a_ty, &b_ty);
    dag.add_root(out);

    let actual = compile_and_run_output_cases(
        &dag,
        "g4_symbolic_matmul_reuse",
        &[
            vec![
                TestInput::new("a", &[1, 3], &[1.0, 2.0, 3.0]),
                TestInput::new("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ],
            vec![
                TestInput::new("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                TestInput::new("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ],
        ],
    );
    assert_eq!(actual.len(), 2, "expected one output row per input case");
    assert_close_vec(&actual[0], &[22.0, 28.0]);
    assert_close_vec(&actual[1], &[22.0, 28.0, 49.0, 64.0]);
}

// ===========================================================================
// G5: expand then add — stride-0 correct on GPU
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g5_expand_add_stride_zero() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(3));
    let expanded = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::DimExpr::Concrete(4),
        },
        vec![x],
        mat_f32(4, 3),
    );
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

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g10_realize_materializes_view_on_gpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(5));
    let s = dag.add_node(RiscOp::Stride { strides: vec![2] }, vec![x], vec_f32(3));
    let r = dag.add_node(RiscOp::Realize, vec![s], vec_f32(3));
    dag.add_root(r);

    assert_gpu_matches_eval(
        &dag,
        "g10_realize_materializes_view",
        &[TestInput::new("x", &[5], &[1.0, 2.0, 3.0, 4.0, 5.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g11_repeated_load_alias_matches_cpu() {
    let mut dag = Dag::new();
    let x0 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let x1 = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let out = dag.add_node(RiscOp::Add, vec![x0, x1], vec_f32(4));
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g11_repeated_load_alias",
        &[TestInput::new("x", &[4], &[1.0, -2.0, 3.5, 4.25])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g12_reused_slot_respects_logical_size() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 1.0 }, vec![], vec_f32(8));
    let b = dag.add_node(RiscOp::Const { value: 2.0 }, vec![], vec_f32(8));
    let _wide = dag.add_node(RiscOp::Add, vec![a, b], vec_f32(8));
    let small = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], vec_f32(4));
    let out = dag.add_node(RiscOp::Neg, vec![small], vec_f32(4));
    dag.add_root(out);

    assert_gpu_matches_eval(&dag, "g12_reused_slot_logical_size", &[]);
}

// ===========================================================================
// GF1: Fused add→neg on GPU matches CPU
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn gf1_fused_add_neg_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(4));
    let add = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
    let out = dag.add_node(RiscOp::Neg, vec![add], vec_f32(4));
    dag.add_root(out);

    assert_fused_gpu_matches_unfused_eval(
        &dag,
        "gf1_fused_add_neg",
        &[
            TestInput::new("x", &[4], &[1.0, -2.0, 3.5, -4.25]),
            TestInput::new("y", &[4], &[0.5, 4.0, -1.5, 2.25]),
        ],
    );
}

// ===========================================================================
// GF2: Fused 3-way chain on GPU matches CPU
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn gf2_fused_three_way_chain_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(4));
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], vec_f32(4));
    let z = dag.add_node(RiscOp::Load { name: "z".into() }, vec![], vec_f32(4));
    let sum = dag.add_node(RiscOp::Add, vec![x, y], vec_f32(4));
    let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], vec_f32(4));
    let relu = dag.add_node(RiscOp::MaxElem, vec![sum, zero], vec_f32(4));
    let out = dag.add_node(RiscOp::Mul, vec![relu, z], vec_f32(4));
    dag.add_root(out);

    assert_fused_gpu_matches_unfused_eval(
        &dag,
        "gf2_fused_add_relu_mul",
        &[
            TestInput::new("x", &[4], &[1.0, -2.0, 3.0, -4.0]),
            TestInput::new("y", &[4], &[-0.5, 0.5, 2.0, 1.0]),
            TestInput::new("z", &[4], &[2.0, 3.0, -1.5, 4.0]),
        ],
    );
}

// ===========================================================================
// G13: Segmented reductions cover tiny/small/large strategies
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g13_tiny_segmented_sum_matches_eval() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(3, 8));
    let out = dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], vec_f32(3));
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g13_tiny_segmented_sum",
        &[TestInput::new(
            "x",
            &[3, 8],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0, 1.0, 2.0, 3.0, 4.0,
                0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5,
            ],
        )],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g13_small_segmented_max_matches_eval() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(2, 16));
    let out = dag.add_node(RiscOp::MaxReduce { axis: 1 }, vec![x], vec_f32(2));
    dag.add_root(out);

    let data: Vec<f32> = (0..32).map(|i| (i as f32) - 10.0).collect();
    assert_gpu_matches_eval(
        &dag,
        "g13_small_segmented_max",
        &[TestInput::new("x", &[2, 16], &data)],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g13_large_segmented_sum_matches_eval() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], mat_f32(2, 128));
    let out = dag.add_node(RiscOp::Sum { axis: 1 }, vec![x], vec_f32(2));
    dag.add_root(out);

    let data: Vec<f32> = (0..256).map(|i| ((i % 17) as f32) - 8.0).collect();
    assert_gpu_matches_eval(
        &dag,
        "g13_large_segmented_sum",
        &[TestInput::new("x", &[2, 128], &data)],
    );
}

// ===========================================================================
// G14: Staged scalar reductions match evaluator
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g14_staged_scalar_sum_matches_eval() {
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vec_f32(1024));
    let out = dag.add_node(RiscOp::Sum { axis: 0 }, vec![x], scalar_f32());
    dag.add_root(out);

    let data: Vec<f32> = (0..1024).map(|i| ((i % 9) as f32) - 4.0).collect();
    assert_gpu_matches_eval(
        &dag,
        "g14_staged_scalar_sum",
        &[TestInput::new("x", &[1024], &data)],
    );
}

// ===========================================================================
// G15: hipBLAS matmul and non-contiguous fallback match evaluator
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g15_hipblas_matmul_matches_eval() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], mat_f32(2, 3));
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], mat_f32(3, 4));
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::DimExpr::Concrete(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::DimExpr::Concrete(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4));
    let out = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g15_hipblas_matmul",
        &[
            TestInput::new("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TestInput::new(
                "b",
                &[3, 4],
                &[1.0, 0.0, 2.0, 1.0, -1.0, 3.0, 0.5, 2.0, 4.0, -2.0, 1.0, 0.0],
            ),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g15_noncontiguous_matmul_fallback_matches_eval() {
    let mut dag = Dag::new();
    let base_a = dag.add_node(
        RiscOp::Load {
            name: "base_a".into(),
        },
        vec![],
        mat_f32(3, 2),
    );
    let a = dag.add_node(
        RiscOp::Permute { axes: vec![1, 0] },
        vec![base_a],
        mat_f32(2, 3),
    );
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], mat_f32(3, 4));
    let ea = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::DimExpr::Concrete(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
    );
    let eb = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::DimExpr::Concrete(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
    );
    let mul = dag.add_node(RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4));
    let out = dag.add_node(RiscOp::Sum { axis: 1 }, vec![mul], mat_f32(2, 4));
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g15_noncontig_matmul",
        &[
            TestInput::new("base_a", &[3, 2], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            TestInput::new(
                "b",
                &[3, 4],
                &[1.0, 0.0, 2.0, 1.0, -1.0, 3.0, 0.5, 2.0, 4.0, -2.0, 1.0, 0.0],
            ),
        ],
    );
}
