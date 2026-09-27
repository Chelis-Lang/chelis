//! GPU execution correctness tests (G1-G9).
//!
//! These require a HIP-capable GPU plus `hipcc`/`hiprtc`.
//! They are `#[ignore]` by default — run with:
//!     scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
//!
//! The `scripts/hip_test.py` wrapper sets the full hipBLAS env per
//! `docs/local_hip_environment.md`. Running the raw `cargo test ...` command
//! without that wrapper inherits only the systemd `environment.d/hip.conf`
//! settings (the `-isystem` half of `HIPCC_COMPILE_FLAGS_APPEND` and
//! `HSA_OVERRIDE_GFX_VERSION=11.0.0`), which segfaults at process exit for
//! hipBLAS-dependent tests. The panic-site hint below detects this case.
//!
//! Manual gate per AGENTS.md: not part of default CI.

mod support;
use chelis_ir::dag::{
    ComparisonKind, Dag, DimInfo, ExtremaKind, ExtremaOperand, NodeId, RiscOp, RtAxis, RtDim,
    TensorType,
};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_ir::fuse::fuse;
use chelis_types::types::Prim;
use chelis_unord::UnordMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use support::codegen_hip;

const GFX1151_LIB_FRAGMENT: &str = "_rocm_sdk_libraries_gfx1151/lib";
const REQUIRED_HSA_OVERRIDE: &str = "11.5.1";

fn hipblas_env_hint(link_flags: &[String]) -> String {
    let uses_hipblas = link_flags.iter().any(|f| f.contains("hipblas"));
    if !uses_hipblas {
        return String::new();
    }
    let flags = env::var("HIPCC_COMPILE_FLAGS_APPEND").unwrap_or_default();
    let gfx = env::var("HSA_OVERRIDE_GFX_VERSION").unwrap_or_default();
    let ld = env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let missing_l = !flags.contains(GFX1151_LIB_FRAGMENT);
    let wrong_gfx = gfx != REQUIRED_HSA_OVERRIDE;
    let missing_ld = !ld.contains(GFX1151_LIB_FRAGMENT);
    if !(missing_l || wrong_gfx || missing_ld) {
        return String::new();
    }
    format!(
        "\n\nhint: this test uses hipBLAS; the empty output is the signature \
         of a process-exit SIGSEGV from a mismatched ROCm stack. \
         See docs/local_hip_environment.md §3 and re-run via scripts/hip_test.py, \
         or set:\n  \
         HSA_OVERRIDE_GFX_VERSION=11.5.1 (got {gfx:?})\n  \
         LD_LIBRARY_PATH must contain {GFX1151_LIB_FRAGMENT} (got {ld:?})\n  \
         HIPCC_COMPILE_FLAGS_APPEND must contain `-L .../{GFX1151_LIB_FRAGMENT}` (got {flags:?})"
    )
}

fn assert_gpu_binary_success(run: &Output, link_flags: &[String]) {
    if run.status.success() {
        return;
    }
    panic!(
        "GPU binary failed:\nstdout: {}\nstderr: {}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr),
        hipblas_env_hint(link_flags)
    );
}

#[derive(Clone)]
struct TestInput {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
    dtype: Prim,
}

impl TestInput {
    fn new(name: &str, shape: &[usize], data: &[f32]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.to_vec(),
            dtype: Prim::F32,
        }
    }

    fn int64(name: &str, shape: &[usize], data: &[i64]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.iter().map(|value| *value as f32).collect(),
            dtype: Prim::Int64,
        }
    }

    /// WS-A4: i8 input. Stored as f32 in the carrier `data` field for
    /// API symmetry; the harness reinterpret-casts to `int8_t*` before
    /// writing into the runtime-allocated buffer, and `chelis_alloc`
    /// is called with `CHELIS_DTYPE_I8` so the buffer is sized at 1 byte
    /// per element.
    #[allow(dead_code, reason = "WS-A4 manual HIP gate; constructed by i8 tests")]
    fn int8(name: &str, shape: &[usize], data: &[i8]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.iter().map(|value| *value as f32).collect(),
            dtype: Prim::Int8,
        }
    }

    /// WS-A4: i16 input. Same packing convention as `i8`.
    #[allow(dead_code, reason = "WS-A4 manual HIP gate; constructed by i16 tests")]
    fn int16(name: &str, shape: &[usize], data: &[i16]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.iter().map(|value| *value as f32).collect(),
            dtype: Prim::Int16,
        }
    }

    fn int32(name: &str, shape: &[usize], data: &[i32]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.iter().map(|value| *value as f32).collect(),
            dtype: Prim::Int32,
        }
    }

    /// chelis#1291: exact `Bool8` input. The carrier `data` holds 0.0/1.0
    /// for the evaluator; the driver writes one `uint8_t` per element
    /// into a `CHELIS_DTYPE_BOOL` allocation, which is what the Count
    /// kernel reads.
    #[allow(
        dead_code,
        reason = "chelis#1291 manual HIP gate; constructed by Count tests"
    )]
    fn bool8(name: &str, shape: &[usize], data: &[bool]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data
                .iter()
                .map(|value| f32::from(u8::from(*value)))
                .collect(),
            dtype: Prim::Bool,
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

fn vec_i32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int32,
    }
}

fn vec_i64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int64,
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

fn tensor4_f32(a: usize, b: usize, c: usize, d: usize) -> TensorType {
    TensorType {
        dims: vec![
            DimInfo::Lit(a),
            DimInfo::Lit(b),
            DimInfo::Lit(c),
            DimInfo::Lit(d),
        ],
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
                "    int64_t {prefix}_shape_{slot}[{ndim}] = {{ {dims} }};"
            ));
            lines.push(format!(
                "    {prefix}_input_storage[{slot}] = chelis_alloc({ndim}, {prefix}_shape_{slot}, {dtype});",
                dtype = match input.dtype {
                    Prim::F32 => "CHELIS_DTYPE_F32",
                    // WS-A4: narrow signed integer dtypes per spec/04-type-system.md §1.1.
                    Prim::Int8 => "CHELIS_DTYPE_I8",
                    Prim::Int16 => "CHELIS_DTYPE_I16",
                    Prim::Int32 => "CHELIS_DTYPE_I32",
                    Prim::Int64 => "CHELIS_DTYPE_I64",
                    Prim::Bool => "CHELIS_DTYPE_BOOL",
                    other => panic!("unsupported manual HIP test dtype {}", other.name()),
                }
            ));
            lines.push(format!(
                "    chelis_tensor_write *{prefix}_input_guard_{slot} = chelis_tensor_begin_write({prefix}_input_storage[{slot}]);"
            ));
            lines.push(format!(
                "    chelis_write_view {prefix}_input_view_{slot} = chelis_tensor_write_view({prefix}_input_guard_{slot});"
            ));
            for (idx, value) in input.data.iter().enumerate() {
                match input.dtype {
                    // Sibling of #250/#251/#252: exact f32 bit pattern via
                    // `chelis_f32_from_bits` (from the included
                    // `chelis_runtime.h`), not a lossy `{:.8}f` decimal.
                    Prim::F32 => lines.push(format!(
                        "    ((float *){prefix}_input_view_{slot}.data)[{idx}] = chelis_f32_from_bits(0x{bits:08x}u);",
                        bits = value.to_bits()
                    )),
                    // WS-A4: i8/i16 inputs are written via reinterpret cast on
                    // `t->data` so the harness exercises the same memory layout
                    // the generated HIP code reads from.
                    Prim::Int8 => lines.push(format!(
                        "    ((int8_t*){prefix}_input_view_{slot}.data)[{idx}] = {};",
                        *value as i8
                    )),
                    Prim::Int16 => lines.push(format!(
                        "    ((int16_t*){prefix}_input_view_{slot}.data)[{idx}] = {};",
                        *value as i16
                    )),
                    Prim::Int32 => lines.push(format!(
                        "    ((int*){prefix}_input_view_{slot}.data)[{idx}] = {};",
                        *value as i32
                    )),
                    Prim::Int64 => lines.push(format!(
                        "    ((int64_t*){prefix}_input_view_{slot}.data)[{idx}] = {}LL;",
                        *value as i64
                    )),
                    // chelis#1308's `Repr::Bool8`: exactly one byte holding 0 or 1.
                    Prim::Bool => lines.push(format!(
                        "    ((uint8_t*){prefix}_input_view_{slot}.data)[{idx}] = {};",
                        u8::from(*value != 0.0)
                    )),
                    other => panic!("unsupported manual HIP test dtype {}", other.name()),
                }
            }
            lines.push(format!(
                "    chelis_tensor_end_write({prefix}_input_guard_{slot});"
            ));
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
        "        chelis_read_view {prefix}_output_view = chelis_tensor_read_view({prefix}_outputs[{prefix}_out_idx]);"
    ));
    lines.push(format!(
        "        for (int {prefix}_i = 0; {prefix}_i < chelis_tensor_numel({prefix}_outputs[{prefix}_out_idx]); {prefix}_i++) {{"
    ));
    lines.push(format!("            if ({prefix}_i > 0) printf(\" \");"));
    lines.push(format!(
        "            printf(\"%.6f\", ((const float *){prefix}_output_view.data)[{prefix}_i]);"
    ));
    lines.push("        }".to_string());
    lines.push("        printf(\"\\n\");".to_string());
    lines.push(format!(
        "        chelis_tensor_release({prefix}_outputs[{prefix}_out_idx]);"
    ));
    lines.push("    }".to_string());
    for slot in 0..input_labels.len() {
        lines.push(format!(
            "    chelis_tensor_release({prefix}_input_storage[{slot}]);"
        ));
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

fn build_caller_preservation_main_cpp(
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    inputs: &[TestInput],
) -> String {
    assert_eq!(input_labels, &["x"], "caller-preservation probe expects x");
    assert_eq!(n_out, 1, "caller-preservation probe expects one output");
    let input = inputs
        .iter()
        .find(|candidate| candidate.name == "x")
        .expect("caller-preservation probe requires x");
    assert_eq!(input.dtype, Prim::F32);
    assert_eq!(input.shape, vec![4]);
    let initialization = input
        .data
        .iter()
        .enumerate()
        .map(|(index, value)| {
            format!(
                "    ((float *)host_input_view.data)[{index}] = chelis_f32_from_bits(0x{:08x}u);",
                value.to_bits()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"#include "chelis_runtime.h"
#include "chelis_hip_runtime.h"
extern "C" void {func_name}_device(const chelis_device_tensor_owner *const *inputs, int32_t n_in, chelis_device_tensor_owner **outputs, int32_t n_out);

int main(void) {{
    int64_t host_shape[1] = {{ 4 }};
    chelis_scalar device_shape[1] = {{ chelis_scalar_from_bits(CHELIS_DTYPE_I64, 4) }};
    chelis_tensor *host_input = chelis_alloc(1, host_shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *host_input_guard = chelis_tensor_begin_write(host_input);
    chelis_write_view host_input_view = chelis_tensor_write_view(host_input_guard);
{initialization}
    chelis_tensor_end_write(host_input_guard);
    chelis_device_tensor_owner *device_input = chelis_device_tensor_alloc(chelis_metadata_plan_new(chelis_scalar_from_bits(CHELIS_DTYPE_I64, 1), device_shape, chelis_scalar_from_bits(CHELIS_DTYPE_F32, 0)));
    chelis_device_tensor_copy_from_host(device_input, host_input);
    const chelis_device_tensor_owner *device_inputs[1] = {{ device_input }};
    chelis_device_tensor_owner *device_outputs[1] = {{ 0 }};
    {func_name}_device(device_inputs, 1, device_outputs, 1);

    chelis_tensor *host_output = chelis_alloc(1, host_shape, CHELIS_DTYPE_F32);
    chelis_tensor *host_after = chelis_alloc(1, host_shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *output_guard = chelis_tensor_begin_write(host_output);
    chelis_device_tensor_copy_to_host(output_guard, device_outputs[0]);
    chelis_tensor_end_write(output_guard);
    chelis_tensor_write *after_guard = chelis_tensor_begin_write(host_after);
    chelis_device_tensor_copy_to_host(after_guard, device_input);
    chelis_tensor_end_write(after_guard);
    chelis_read_view host_output_view = chelis_tensor_read_view(host_output);
    chelis_read_view host_after_view = chelis_tensor_read_view(host_after);
    for (int i = 0; i < 4; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", ((const float *)host_output_view.data)[i]);
    }}
    printf("\n");
    for (int i = 0; i < 4; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", ((const float *)host_after_view.data)[i]);
    }}
    printf("\n");

    chelis_tensor_release(host_input);
    chelis_tensor_release(host_output);
    chelis_tensor_release(host_after);
    chelis_device_tensor_release(device_outputs[0]);
    chelis_device_tensor_release(device_input);
    return 0;
}}
"#,
        initialization = initialization,
    )
}

fn compile_and_run_output_and_inputs(
    dag: &Dag,
    func_name: &str,
    inputs: &[TestInput],
) -> Vec<Vec<f32>> {
    require_hipcc();
    let result = codegen_hip(dag, func_name).unwrap();
    assert_eq!(result.output_labels.len(), 1);

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    write_temp_file(
        tmp.path(),
        "main.cpp",
        &build_caller_preservation_main_cpp(
            func_name,
            &result.input_labels,
            result.output_labels.len(),
            inputs,
        ),
    );

    let bin_path = tmp.path().join("caller_preservation_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
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
    assert_gpu_binary_success(&run, &result.link_flags);
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

fn compile_and_run_single_output(dag: &Dag, func_name: &str, inputs: &[TestInput]) -> Vec<f32> {
    require_hipcc();
    let result = codegen_hip(dag, func_name).unwrap();
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
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
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
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
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
    assert_gpu_binary_success(&run, &result.link_flags);
    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .map(|token| token.parse::<f32>().expect("parse output float"))
        .collect()
}

/// Compile a direct-arithmetic DAG with float inputs initialized from raw
/// bits and return every root's output bits. This is the manual HIP proof for
/// stored-operand extrema and adjoints, where decimal/tolerance comparison
/// would erase NaN payload and signed-zero evidence.
fn compile_and_run_float_output_bits(
    dag: &Dag,
    func_name: &str,
    prim: Prim,
    inputs: &[(&str, &[u64])],
) -> Vec<Vec<u64>> {
    require_hipcc();
    assert!(matches!(
        prim,
        Prim::F16 | Prim::Bf16 | Prim::F32 | Prim::F64
    ));
    let result = codegen_hip(dag, func_name).unwrap();
    let n = inputs.first().expect("bit harness needs inputs").1.len();
    assert!(inputs.iter().all(|(_, bits)| bits.len() == n));

    let mut setup = vec![format!("    int64_t shape[1] = {{ {n} }};")];
    setup.push(format!(
        "    chelis_tensor *inputs[{}] = {{0}};",
        result.input_labels.len()
    ));
    for (slot, label) in result.input_labels.iter().enumerate() {
        let bits = inputs
            .iter()
            .find_map(|(name, bits)| (*name == label).then_some(*bits))
            .unwrap_or_else(|| panic!("missing bit input {label}"));
        let dtype = match prim {
            Prim::F16 => "CHELIS_DTYPE_F16",
            Prim::Bf16 => "CHELIS_DTYPE_BF16",
            Prim::F32 => "CHELIS_DTYPE_F32",
            Prim::F64 => "CHELIS_DTYPE_F64",
            _ => unreachable!(),
        };
        setup.push(format!(
            "    inputs[{slot}] = chelis_alloc(1, shape, {dtype});"
        ));
        setup.push(format!(
            "    chelis_tensor_write *input_guard_{slot} = chelis_tensor_begin_write(inputs[{slot}]);"
        ));
        setup.push(format!(
            "    chelis_write_view input_view_{slot} = chelis_tensor_write_view(input_guard_{slot});"
        ));
        for (index, value) in bits.iter().enumerate() {
            setup.push(match prim {
                Prim::F16 | Prim::Bf16 => format!(
                    "    ((uint16_t *)input_view_{slot}.data)[{index}] = UINT16_C(0x{value:04x});"
                ),
                Prim::F32 => format!(
                    "    ((float *)input_view_{slot}.data)[{index}] = chelis_f32_from_bits(0x{value:08x}u);"
                ),
                Prim::F64 => format!(
                    "    ((double *)input_view_{slot}.data)[{index}] = chelis_f64_from_bits(0x{value:016x}uLL);"
                ),
                _ => unreachable!(),
            });
        }
        setup.push(format!("    chelis_tensor_end_write(input_guard_{slot});"));
    }
    setup.push(format!(
        "    chelis_tensor *outputs[{}] = {{0}};",
        result.output_labels.len()
    ));
    setup.push(format!(
        "    {func_name}(inputs, {}, outputs, {});",
        result.input_labels.len(),
        result.output_labels.len()
    ));
    setup.push(format!(
        "    for (int out = 0; out < {}; out++) {{",
        result.output_labels.len()
    ));
    setup.push(
        "        chelis_read_view output_view = chelis_tensor_read_view(outputs[out]);".to_string(),
    );
    setup.push("        for (int i = 0; i < chelis_tensor_numel(outputs[out]); i++) {".to_string());
    setup.push("            if (i > 0) printf(\" \" );".to_string());
    setup.push(match prim {
        Prim::F16 | Prim::Bf16 => {
            "            uint16_t bits = ((const uint16_t *)output_view.data)[i]; printf(\"0x%04x\", bits);"
                .to_string()
        }
        Prim::F32 => {
            "            uint32_t bits; memcpy(&bits, &((const float *)output_view.data)[i], sizeof(bits)); printf(\"0x%08x\", bits);"
                .to_string()
        }
        Prim::F64 => {
            "            uint64_t bits; memcpy(&bits, &((const double *)output_view.data)[i], sizeof(bits)); printf(\"0x%016llx\", (unsigned long long)bits);"
                .to_string()
        }
        _ => unreachable!(),
    });
    setup.push("        }".to_string());
    setup.push("        printf(\"\\n\");".to_string());
    setup.push("        chelis_tensor_release(outputs[out]);".to_string());
    setup.push("    }".to_string());
    setup.push(format!(
        "    for (int i = 0; i < {}; i++) chelis_tensor_release(inputs[i]);",
        result.input_labels.len()
    ));

    let main_cpp = format!(
        r#"#include "chelis_runtime.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
{body}
    return 0;
}}
"#,
        body = setup.join("\n")
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    write_temp_file(tmp.path(), "main.cpp", &main_cpp);
    let bin_path = tmp.path().join("gpu_direct_arithmetic_bits");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
    compile_cmd.arg("-lpthread");
    compile_cmd.arg("-ldl");
    compile_cmd.args(&result.link_flags);
    compile_cmd.arg("-o");
    compile_cmd.arg(&bin_path);
    let compile = compile_cmd.output().expect("run hipcc");
    assert!(
        compile.status.success(),
        "hipcc failed:\nstderr: {}\nsource:\n{}\nharness:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.c_source,
        main_cpp
    );

    let run = Command::new(&bin_path)
        .output()
        .expect("run GPU bit binary");
    assert_gpu_binary_success(&run, &result.link_flags);
    String::from_utf8(run.stdout)
        .expect("utf8 stdout")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|token| {
                    u64::from_str_radix(token.trim_start_matches("0x"), 16)
                        .expect("parse output bits")
                })
                .collect()
        })
        .collect()
}

fn compile_and_run_output_cases(
    dag: &Dag,
    func_name: &str,
    input_cases: &[Vec<TestInput>],
) -> Vec<Vec<f32>> {
    require_hipcc();
    let result = codegen_hip(dag, func_name).unwrap();
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
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
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
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
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
    assert_gpu_binary_success(&run, &result.link_flags);
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
    let input_map: UnordMap<String, TensorValue> = inputs
        .iter()
        .map(|input| (input.name.clone(), input.evaluator_value()))
        .collect();
    let values =
        eval_tensor_roots_with_strict(dag, dag.roots(), |name| input_map.get(name).cloned())
            .expect("evaluator should succeed");
    let root = *dag.roots().first().expect("single root expected");
    values[&root]
        .to_f64_lossy_vec()
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 1.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
    dag.add_root(c);

    let actual = compile_and_run_single_output(&dag, "g1_add_consts", &[]);
    assert_eq!(actual, vec![3.0]);
}

// ===========================================================================
// chelis#770: the uniform_like sampler affine is a single correctly-rounded
// FMA (`fmaf(high - low, (float)unit, low)`), bit-identical across the host
// evaluator, the C lane, and this HIP device kernel. gpu_correctness carries
// no other uniform_like value case, so this locks the on-GPU sampler.
//
// Manual gate runner (this whole file is #[ignore] per the module header):
//   scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- \
//       --ignored --test-threads=1
// Expected success: green — the 8 f32 outputs equal the bit patterns below.
// elem[6]/[7] are the tell: the single-rounding FMA gives 0x408f5273 /
// 0x403ec1e7 where a two-rounding `low + span*unit` gives 0x408f5274 /
// 0x403ec1e8, so a %.6f-precision check could not catch a regression — this
// asserts raw f32 bits.
// ===========================================================================

/// Compile a single-output, no-input DAG and return each f32 output element's
/// raw bit pattern. Mirrors `compile_and_run_single_output` but prints exact
/// bits (`%.6f` cannot distinguish the 1-ULP FMA difference this locks).
fn compile_and_run_output_f32_bits(dag: &Dag, func_name: &str) -> Vec<u32> {
    require_hipcc();
    let result = codegen_hip(dag, func_name).unwrap();
    assert_eq!(
        result.output_labels.len(),
        1,
        "bit harness expects a single output"
    );
    assert!(
        result.input_labels.is_empty(),
        "bit harness expects a no-input DAG"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    let main_cpp = format!(
        r#"#include "chelis_runtime.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(nullptr, 0, outputs, 1);
    chelis_read_view output_view = chelis_tensor_read_view(outputs[0]);
    for (int i = 0; i < chelis_tensor_numel(outputs[0]); i++) {{
        if (i > 0) printf(" ");
        float v = ((const float *)output_view.data)[i];
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        printf("0x%08x", bits);
    }}
    printf("\n");
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    write_temp_file(tmp.path(), "main.cpp", &main_cpp);

    let bin_path = tmp.path().join("gpu_correctness_bits_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
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
    assert_gpu_binary_success(&run, &result.link_flags);
    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .map(|tok| {
            let hex = tok.strip_prefix("0x").unwrap_or(tok);
            u32::from_str_radix(hex, 16).expect("parse hex bit pattern")
        })
        .collect()
}

/// f64 sibling of [`compile_and_run_output_f32_bits`]. The output buffer is
/// read through its declared storage width so an f32-widening regression
/// cannot hide behind a lossy harness cast.
fn compile_and_run_output_f64_bits(dag: &Dag, func_name: &str) -> Vec<u64> {
    require_hipcc();
    let result = codegen_hip(dag, func_name).unwrap();
    assert_eq!(result.output_labels.len(), 1);
    assert!(result.input_labels.is_empty());

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    let main_cpp = format!(
        r#"#include "chelis_runtime.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(nullptr, 0, outputs, 1);
    chelis_read_view output_view = chelis_tensor_read_view(outputs[0]);
    const double *data = (const double *)output_view.data;
    for (int i = 0; i < chelis_tensor_numel(outputs[0]); i++) {{
        if (i > 0) printf(" ");
        uint64_t bits;
        memcpy(&bits, &data[i], sizeof(bits));
        printf("0x%016llx", (unsigned long long)bits);
    }}
    printf("\n");
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
    );
    write_temp_file(tmp.path(), "main.cpp", &main_cpp);

    let bin_path = tmp.path().join("gpu_correctness_f64_bits_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
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
    assert_gpu_binary_success(&run, &result.link_flags);
    String::from_utf8(run.stdout)
        .expect("utf8 stdout")
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .map(|token| u64::from_str_radix(token.trim_start_matches("0x"), 16).unwrap())
        .collect()
}

/// The key `key_from_seed(PINNED_KEY_SEED)` has the bits
/// `0x5072f63b9b5fc46b`, the key `key_ref.py of_draw_key(42, 0)` computes,
/// so the `rng_ref.py uniform 42 0` bits pinned below stay valid for an
/// explicit key.
const PINNED_KEY_SEED: i64 = 0x5072_f63b_9b5f_c46b;

/// `uniform_like(key_from_seed(seed), template, 2.0f32, 5.0f32)`, whose key
/// the HIP lane computes at emission.
fn seeded_uniform_like(
    dag: &mut Dag,
    decl: chelis_ir::dag::DeclId,
    template: NodeId,
    ty: TensorType,
    seed: i64,
) -> NodeId {
    let rank0 = |precision| TensorType {
        dims: vec![],
        precision,
    };
    let seed = dag.add_node(
        decl,
        RiscOp::Const {
            value: chelis_types::scalar_from_i64("test", Prim::Int64, seed).unwrap(),
        },
        vec![],
        rank0(Prim::Int64),
        None,
    );
    let low = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        rank0(Prim::F32),
        None,
    );
    let high = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 5.0),
        vec![],
        rank0(Prim::F32),
        None,
    );
    let key = dag.add_node(
        decl,
        RiscOp::KeyFromSeed,
        vec![seed],
        rank0(Prim::Key),
        None,
    );
    dag.add_node(
        decl,
        RiscOp::UniformLike,
        vec![template, low, high, key],
        ty,
        None,
    )
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn uniform_like_fma_gpu_bit_exact_matches_eval_and_c() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let template = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(8).precision, 0.0),
        vec![],
        vec_f32(8),
        None,
    );
    let out = seeded_uniform_like(&mut dag, decl, template, vec_f32(8), PINNED_KEY_SEED);
    dag.add_root(out);

    let actual = compile_and_run_output_f32_bits(&dag, "uniform_like_fma");
    // The key PINNED_KEY_SEED, [2,5), shape [8]: exact-rational evaluations of
    // [05-RNG-1] and [05-OP-8] (`rng_ref.py uniform 42 0 8 2 5 f32`,
    // chelis#2408). elem[6] is where an f64 affine lands 1 ULP away and
    // elem[2] is the FMA-vs-two-rounding tell.
    let expected: Vec<u32> = vec![
        0x401d600b, 0x406ddbc6, 0x401cb39d, 0x40068337, 0x4005ff9a, 0x4044062f, 0x40683468,
        0x401bf5fc,
    ];
    assert_eq!(
        actual, expected,
        "HIP uniform_like FMA output must be bit-exact with eval/C"
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn issue_937_uniform_like_f64_gpu_bit_exact_matches_shared_sampler() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let template = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f64(8).precision, 0.0),
        vec![],
        vec_f64(8),
        None,
    );
    let key = chelis_types::RandomKey::from_seed(
        chelis_types::scalar_from_i64("test", Prim::Int64, PINNED_KEY_SEED).unwrap(),
    )
    .unwrap();
    let out = seeded_uniform_like(&mut dag, decl, template, vec_f64(8), PINNED_KEY_SEED);
    dag.add_root(out);

    let actual = compile_and_run_output_f64_bits(&dag, "uniform_like_f64");
    let bound = |value| chelis_types::scalar_from_f64("test", Prim::F32, value).unwrap();
    let sampled = chelis_types::PreparedUniformLike::new(Prim::F64, 8, bound(2.0), bound(5.0))
        .unwrap()
        .apply(key)
        .unwrap();
    let expected = (0..8)
        .map(|index| sampled.scalar_at(index).as_f64_lossy().to_bits())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

// ===========================================================================
// G2: All unary ops GPU == evaluator within tolerance
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_neg_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Neg, vec![x], vec_f32(3), None);
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g2_neg",
        &[TestInput::new("x", &[3], &[1.5, -2.0, 0.25])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_copy_materializes_and_terminal_drop_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let copied = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(4), None);
    dag.add_node(decl, RiscOp::Drop, vec![x], vec_f32(4), None);
    let out = dag.add_node(decl, RiscOp::Neg, vec![copied], vec_f32(4), None);
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g2_copy_drop",
        &[TestInput::new("x", &[4], &[1.0, -2.0, 3.5, -4.5])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g2_exp_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Exp, vec![x], vec_f32(3), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Log, vec![x], vec_f32(3), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Sin, vec![x], vec_f32(3), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Sqrt, vec![x], vec_f32(3), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(3), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Mul, vec![x, y], vec_f32(3), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::MaxElem, vec![x, y], vec_f32(3), None);
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

fn direct_extrema_gpu_bit_case(prim: Prim, lhs: &[u64; 6], rhs: &[u64; 6], gradient: &[u64; 6]) {
    let ty = TensorType {
        dims: vec![DimInfo::Lit(6)],
        precision: prim,
    };
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        ty.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        ty.clone(),
        None,
    );
    let g = dag.add_node(
        decl,
        RiscOp::Load { name: "g".into() },
        vec![],
        ty.clone(),
        None,
    );
    for op in [RiscOp::MaxElem, RiscOp::MinElem] {
        let root = dag.add_node(decl, op, vec![a, b], ty.clone(), None);
        dag.add_root(root);
    }
    for (kind, operand) in [
        (ExtremaKind::Max, ExtremaOperand::Left),
        (ExtremaKind::Max, ExtremaOperand::Right),
        (ExtremaKind::Min, ExtremaOperand::Left),
        (ExtremaKind::Min, ExtremaOperand::Right),
    ] {
        let root = dag.add_node(
            decl,
            RiscOp::ExtremaAdjoint { kind, operand },
            vec![a, b, g],
            ty.clone(),
            None,
        );
        dag.add_root(root);
    }

    let zero = 0;
    let expected = vec![
        vec![lhs[0], rhs[1], lhs[2], lhs[3], lhs[4], rhs[5]],
        vec![lhs[0], rhs[1], lhs[2], lhs[3], rhs[4], lhs[5]],
        vec![
            gradient[0],
            zero,
            gradient[2],
            gradient[3],
            gradient[4],
            zero,
        ],
        vec![zero, gradient[1], zero, zero, zero, gradient[5]],
        vec![
            gradient[0],
            zero,
            gradient[2],
            gradient[3],
            zero,
            gradient[5],
        ],
        vec![zero, gradient[1], zero, zero, gradient[4], zero],
    ];
    let actual = compile_and_run_float_output_bits(
        &dag,
        &format!("direct_extrema_{}_bits", prim.name()),
        prim,
        &[("a", lhs), ("b", rhs), ("g", gradient)],
    );
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn direct_extrema_and_adjoints_preserve_exact_bits_at_every_float_width_on_gpu() {
    direct_extrema_gpu_bit_case(
        Prim::F16,
        &[0x7e11, 0x3c00, 0, 0x8000, 0x4000, 0x3c00],
        &[0x4000, 0xfe22, 0x8000, 0, 0x3c00, 0x4000],
        &[0x7e33, 0xbc00, 0x3c00, 0x8000, 0x4200, 0xc400],
    );
    direct_extrema_gpu_bit_case(
        Prim::Bf16,
        &[0x7fc1, 0x3f80, 0, 0x8000, 0x4000, 0x3f80],
        &[0x4000, 0xffc2, 0x8000, 0, 0x3f80, 0x4000],
        &[0x7fc3, 0xbf80, 0x3f80, 0x8000, 0x4040, 0xc080],
    );
    direct_extrema_gpu_bit_case(
        Prim::F32,
        &[
            0x7fc1_2345,
            0x3f80_0000,
            0,
            0x8000_0000,
            0x4000_0000,
            0x3f80_0000,
        ],
        &[
            0x4000_0000,
            0xffc5_4321,
            0x8000_0000,
            0,
            0x3f80_0000,
            0x4000_0000,
        ],
        &[
            0x7fc6_789a,
            0xbf80_0000,
            0x3f80_0000,
            0x8000_0000,
            0x4040_0000,
            0xc080_0000,
        ],
    );
    direct_extrema_gpu_bit_case(
        Prim::F64,
        &[
            0x7ff8_1111_2222_3333,
            0x3ff0_0000_0000_0000,
            0,
            0x8000_0000_0000_0000,
            0x4000_0000_0000_0000,
            0x3ff0_0000_0000_0000,
        ],
        &[
            0x4000_0000_0000_0000,
            0xfff8_4444_5555_6666,
            0x8000_0000_0000_0000,
            0,
            0x3ff0_0000_0000_0000,
            0x4000_0000_0000_0000,
        ],
        &[
            0x7ff8_abcd_1234_5678,
            0xbff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x4008_0000_0000_0000,
            0xc010_0000_0000_0000,
        ],
    );
}

fn direct_relu_gpu_bit_case(prim: Prim, x_bits: &[u64; 6], gradient: &[u64; 6]) {
    let ty = TensorType {
        dims: vec![DimInfo::Lit(6)],
        precision: prim,
    };
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let g = dag.add_node(
        decl,
        RiscOp::Load { name: "g".into() },
        vec![],
        ty.clone(),
        None,
    );
    let relu = dag.add_node(decl, RiscOp::Relu, vec![x], ty.clone(), None);
    let adjoint = dag.add_node(decl, RiscOp::ReluAdjoint, vec![x, g], ty, None);
    dag.add_root(relu);
    dag.add_root(adjoint);
    let expected = vec![
        vec![x_bits[0], x_bits[1], 0, 0, x_bits[4], x_bits[5]],
        vec![0, 0, 0, 0, gradient[4], gradient[5]],
    ];
    let actual = compile_and_run_float_output_bits(
        &dag,
        &format!("direct_relu_{}_bits", prim.name()),
        prim,
        &[("x", x_bits), ("g", gradient)],
    );
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn direct_relu_and_adjoint_preserve_exact_bits_at_every_float_width_on_gpu() {
    direct_relu_gpu_bit_case(
        Prim::F16,
        &[0x7e11, 0x8000, 0, 0xbc00, 0x3c00, 1],
        &[0x7e33, 0x7c00, 0x8000, 0xfe33, 0xbc00, 0x7c00],
    );
    direct_relu_gpu_bit_case(
        Prim::Bf16,
        &[0x7fc1, 0x8000, 0, 0xbf80, 0x3f80, 1],
        &[0x7fc3, 0x7f80, 0x8000, 0xffc3, 0xbf80, 0x7f80],
    );
    direct_relu_gpu_bit_case(
        Prim::F32,
        &[0x7fc1_2345, 0x8000_0000, 0, 0xbf80_0000, 0x3f80_0000, 1],
        &[
            0x7fc6_789a,
            0x7f80_0000,
            0x8000_0000,
            0xffc6_789a,
            0xbf80_0000,
            0x7f80_0000,
        ],
    );
    direct_relu_gpu_bit_case(
        Prim::F64,
        &[
            0x7ff8_1111_2222_3333,
            0x8000_0000_0000_0000,
            0,
            0xbff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            1,
        ],
        &[
            0x7ff8_abcd_1234_5678,
            0x7ff0_0000_0000_0000,
            0x8000_0000_0000_0000,
            0xfff8_abcd_1234_5678,
            0xbff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn direct_sub_gpu_matches_own_width_evaluator() {
    for (prim, lhs, rhs, expected) in [
        (
            Prim::F16,
            [0x3c00, 0x8000, 0x7bff, 0xc450, 0xfe01, 0x3c00],
            [0x4000, 0, 0x7bff, 0x4480, 0x3c00, 0x7d01],
            [
                u64::from(
                    half::f16::from_f32(
                        half::f16::from_bits(0x3c00).to_f32()
                            - half::f16::from_bits(0x4000).to_f32(),
                    )
                    .to_bits(),
                ),
                u64::from(
                    half::f16::from_f32(
                        half::f16::from_bits(0x8000).to_f32() - half::f16::from_bits(0).to_f32(),
                    )
                    .to_bits(),
                ),
                u64::from(
                    half::f16::from_f32(
                        half::f16::from_bits(0x7bff).to_f32()
                            - half::f16::from_bits(0x7bff).to_f32(),
                    )
                    .to_bits(),
                ),
                u64::from(
                    half::f16::from_f32(
                        half::f16::from_bits(0xc450).to_f32()
                            - half::f16::from_bits(0x4480).to_f32(),
                    )
                    .to_bits(),
                ),
                0x7e00,
                0x7e00,
            ],
        ),
        (
            Prim::Bf16,
            [0x3f80, 0x8000, 0x7f7f, 0xc10a, 0xffc1, 0x3f80],
            [0x4000, 0, 0x7f7f, 0x4090, 0x3f80, 0x7f81],
            [
                u64::from(
                    half::bf16::from_f32(
                        half::bf16::from_bits(0x3f80).to_f32()
                            - half::bf16::from_bits(0x4000).to_f32(),
                    )
                    .to_bits(),
                ),
                u64::from(
                    half::bf16::from_f32(
                        half::bf16::from_bits(0x8000).to_f32() - half::bf16::from_bits(0).to_f32(),
                    )
                    .to_bits(),
                ),
                u64::from(
                    half::bf16::from_f32(
                        half::bf16::from_bits(0x7f7f).to_f32()
                            - half::bf16::from_bits(0x7f7f).to_f32(),
                    )
                    .to_bits(),
                ),
                u64::from(
                    half::bf16::from_f32(
                        half::bf16::from_bits(0xc10a).to_f32()
                            - half::bf16::from_bits(0x4090).to_f32(),
                    )
                    .to_bits(),
                ),
                0x7fc0,
                0x7fc0,
            ],
        ),
    ] {
        let ty = TensorType {
            dims: vec![DimInfo::Lit(lhs.len())],
            precision: prim,
        };
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            ty.clone(),
            None,
        );
        let out = dag.add_node(decl, RiscOp::Sub, vec![x, y], ty, None);
        dag.add_root(out);
        let actual = compile_and_run_float_output_bits(
            &dag,
            &format!("direct_sub_{}", prim.name()),
            prim,
            &[("x", &lhs), ("y", &rhs)],
        );
        assert_eq!(actual, vec![expected.to_vec()]);
    }

    let mut f32_dag = Dag::new();
    let f32_dag_decl = f32_dag.declare("test");
    let x = f32_dag.add_node(
        f32_dag_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = f32_dag.add_node(
        f32_dag_decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let out = f32_dag.add_node(f32_dag_decl, RiscOp::Sub, vec![x, y], vec_f32(4), None);
    f32_dag.add_root(out);
    assert_gpu_matches_eval(
        &f32_dag,
        "direct_sub_f32",
        &[
            TestInput::new("x", &[4], &[1.0, -0.0, f32::MAX, -17.25]),
            TestInput::new("y", &[4], &[2.0, 0.0, f32::MAX, 4.5]),
        ],
    );

    let mut f64_dag = Dag::new();
    let f64_dag_decl = f64_dag.declare("test");
    let x = f64_dag.add_node(
        f64_dag_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let y = f64_dag.add_node(
        f64_dag_decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let out = f64_dag.add_node(f64_dag_decl, RiscOp::Sub, vec![x, y], vec_f64(4), None);
    f64_dag.add_root(out);
    assert_gpu_matches_eval_f64(
        &f64_dag,
        "direct_sub_f64",
        &[
            TestInputF64::f64("x", &[4], &[1.0, -0.0, f64::MAX, -17.25]),
            TestInputF64::f64("y", &[4], &[2.0, 0.0, f64::MAX, 4.5]),
        ],
        0.0,
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn direct_checked_signed_sub_reports_exact_device_overflow_trap() {
    let ty = TensorType {
        dims: vec![DimInfo::Lit(2)],
        precision: Prim::Int8,
    };
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        ty.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        ty.clone(),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Sub, vec![a, b], ty, None);
    dag.add_root(out);

    let exact = compile_and_run_single_output_typed_i64(
        &dag,
        "direct_checked_sub_i8_exact",
        &[
            TestInput::int8("a", &[2], &[-128, 127]),
            TestInput::int8("b", &[2], &[0, 1]),
        ],
        "int8_t",
        "%lld",
    );
    assert_eq!(exact, vec![-128, 126]);

    let panic = std::panic::catch_unwind(|| {
        compile_and_run_single_output_typed_i64(
            &dag,
            "direct_checked_sub_i8_overflow",
            &[
                TestInput::int8("a", &[2], &[0, -128]),
                TestInput::int8("b", &[2], &[0, 1]),
            ],
            "int8_t",
            "%lld",
        )
    })
    .expect_err("true signed subtraction overflow must trap after device dispatch");
    let message = if let Some(message) = panic.downcast_ref::<String>() {
        message.as_str()
    } else if let Some(message) = panic.downcast_ref::<&str>() {
        message
    } else {
        panic!("unexpected non-string panic from HIP overflow harness");
    };
    assert!(
        message.contains("numeric trap: overflow in sub at i8"),
        "{message}"
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g3_cmplt_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![x, y],
        vec_bool(4),
        None,
    );
    dag.add_root(out);
    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "g3_cmplt",
        &[
            TestInput::new("x", &[4], &[1.0, 3.0, 5.0, 7.0]),
            TestInput::new("y", &[4], &[2.0, 3.0, 4.0, 8.0]),
        ],
        "uint8_t",
        "%lld",
    );
    assert_eq!(actual, vec![1, 0, 0, 1]);
}

// ===========================================================================
// G4: Reduction GPU == evaluator
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g4_sum_reduction_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(3),
        None,
    );
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::MaxReduce { axis: 1 },
        vec![x],
        vec_f32(2),
        None,
    );
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
    let decl = dag.declare("test");
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
    let x = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], x_ty, None);
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        out_ty,
        None,
    );
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
    let decl = dag.declare("test");
    let x_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("seq".into(), None),
        ],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        x_ty.clone(),
        None,
    );
    let out = chelis_ir::tier2::lower_softmax(decl.into(), &mut dag, x, 1, &x_ty, None);
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
    let decl = dag.declare("test");
    let a_ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        a_ty.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        b_ty.clone(),
        None,
    );
    let out = chelis_ir::tier2::lower_matmul(decl.into(), &mut dag, a, b, &a_ty, &b_ty, None);
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
    let decl = dag.declare("test");
    let a_ty = TensorType {
        dims: vec![DimInfo::Named("batch".into(), None), DimInfo::Lit(3)],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        a_ty.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        b_ty.clone(),
        None,
    );
    let out = chelis_ir::tier2::lower_matmul(decl.into(), &mut dag, a, b, &a_ty, &b_ty, None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let expanded = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![x],
        mat_f32(4, 3),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(4, 3).precision, 2.0),
        vec![],
        mat_f32(4, 3),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Add, vec![expanded, c], mat_f32(4, 3), None);
    dag.add_root(out);
    assert_gpu_matches_eval(
        &dag,
        "g5_expand_add",
        &[TestInput::new("x", &[3], &[1.0, 2.0, 3.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g5_input_axis_expand_executes_from_witness_metadata() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let runtime_vector = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let value = dag.add_node(
        decl,
        RiscOp::Load {
            name: "value".into(),
        },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let witness = dag.add_node(
        decl,
        RiscOp::Load {
            name: "witness".into(),
        },
        vec![],
        runtime_vector.clone(),
        None,
    );
    let expanded = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![value, witness],
        runtime_vector,
        None,
    );
    dag.add_root(expanded);

    assert_gpu_matches_eval(
        &dag,
        "g5_input_axis_expand",
        &[
            TestInput::new("value", &[], &[2.0]),
            TestInput::new("witness", &[4], &[9.0, 8.0, 7.0, 6.0]),
        ],
    );
}

// ===========================================================================
// G6: permute then elementwise — reordered strides on GPU
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g6_permute_then_add() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let perm = dag.add_node(
        decl,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![x],
        mat_f32(3, 2),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(mat_f32(3, 2).precision, 1.5),
        vec![],
        mat_f32(3, 2),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Add, vec![perm, c], mat_f32(3, 2), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![x],
        vec_f32(4),
        None,
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 2.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 3.0),
        vec![],
        scalar_f32(),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 4.0),
        vec![],
        scalar_f32(),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Mul, vec![add, c], scalar_f32(), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        scalar_f32(),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![x, y],
        bool_scalar(),
        None,
    );
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(5),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::Stride {
            strides: vec![chelis_ir::dag::RtDim::Lit(2)],
        },
        vec![x],
        vec_f32(3),
        None,
    );
    let r = dag.add_node(decl, RiscOp::Realize, vec![s], vec_f32(3), None);
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
    let decl = dag.declare("test");
    let x0 = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let x1 = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Add, vec![x0, x1], vec_f32(4), None);
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(8).precision, 1.0),
        vec![],
        vec_f32(8),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(8).precision, 2.0),
        vec![],
        vec_f32(8),
        None,
    );
    let _wide = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(8), None);
    let small = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 3.0),
        vec![],
        vec_f32(4),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Neg, vec![small], vec_f32(4), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
    let out = dag.add_node(decl, RiscOp::Neg, vec![add], vec_f32(4), None);
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let z = dag.add_node(
        decl,
        RiscOp::Load { name: "z".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let sum = dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);
    let zero = dag.add_node(
        decl,
        RiscOp::synth_const(vec_f32(4).precision, 0.0),
        vec![],
        vec_f32(4),
        None,
    );
    let relu = dag.add_node(decl, RiscOp::MaxElem, vec![sum, zero], vec_f32(4), None);
    let out = dag.add_node(decl, RiscOp::Mul, vec![relu, z], vec_f32(4), None);
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

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn fused_direct_sub_and_min_gpu_match_unfused_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let z = dag.add_node(
        decl,
        RiscOp::Load { name: "z".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let difference = dag.add_node(decl, RiscOp::Sub, vec![x, y], vec_f32(4), None);
    let out = dag.add_node(decl, RiscOp::MinElem, vec![difference, z], vec_f32(4), None);
    dag.add_root(out);

    assert_fused_gpu_matches_unfused_eval(
        &dag,
        "fused_direct_sub_min",
        &[
            TestInput::new("x", &[4], &[1.0, -2.0, 3.0, -4.0]),
            TestInput::new("y", &[4], &[-0.5, 0.5, 2.0, 1.0]),
            TestInput::new("z", &[4], &[0.0, -3.0, 0.5, -4.0]),
        ],
    );
}

// ===========================================================================
// GF3: Fused in-place fan-in — when the chain marks a program-owned
// intermediate as reusable, the HIP backend aliases the FusedElem output
// view onto that intermediate's device buffer at runtime
// (`chelis_gpu_is_contiguous` guard + `chelis_gpu_alloc_view` onto
// `d_t{reusable}->data`). The GPU result must still match the unfused
// CPU evaluator bit-for-bit-within-tolerance.
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn gf3_program_owned_fused_in_place_fan_in_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    // `(copy(x) + y) * z`, with the program-owned copy marked as the
    // reusable input on the Add step. `fuse` propagates the hint into the
    // new FusedElem node so the shared planner can mint the token.
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let owned_x = dag.add_node(decl, RiscOp::Copy, vec![x], vec_f32(8), None);
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let z = dag.add_node(
        decl,
        RiscOp::Load { name: "z".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let add = dag.add_node(decl, RiscOp::Add, vec![owned_x, y], vec_f32(8), None);
    dag.set_reusable_input(add, owned_x);
    let out = dag.add_node(decl, RiscOp::Mul, vec![add, z], vec_f32(8), None);
    dag.add_root(out);

    assert_fused_gpu_matches_unfused_eval(
        &dag,
        "gf3_program_owned_fused_in_place_fan_in",
        &[
            TestInput::new("x", &[8], &[1.0, -2.0, 3.5, -4.25, 0.5, -0.75, 8.0, -16.0]),
            TestInput::new("y", &[8], &[0.5, 4.0, -1.5, 2.25, -0.25, 1.5, -2.0, 4.0]),
            TestInput::new("z", &[8], &[2.0, 3.0, -1.5, 4.0, -2.5, 1.0, 0.5, -0.5]),
        ],
    );
}

/// Executing a rejected caller-input reuse hint must leave the caller's
/// original device-backed bytes unchanged.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn compiled_value_ownership_caller_bytes_unchanged() {
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
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let fused = dag.add_node(
        decl,
        RiscOp::FusedElem {
            ops: vec![chelis_ir::dag::FusedStep {
                op: chelis_ir::dag::FusedStepOp::Mul,
                input_indices: vec![
                    chelis_ir::dag::FusedInput::External(0),
                    chelis_ir::dag::FusedInput::External(1),
                ],
            }],
        },
        vec![x, scale],
        vec_f32(4),
        None,
    );
    dag.set_reusable_input(fused, x);
    dag.add_root(fused);

    let lines = compile_and_run_output_and_inputs(
        &dag,
        "compiled_value_ownership_caller_bytes_unchanged",
        &[TestInput::new("x", &[4], &[1.0, 2.0, 3.0, 4.0])],
    );
    assert_eq!(
        lines.len(),
        2,
        "one output and one caller input are required"
    );
    assert_close_vec(&lines[0], &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(
        lines[1],
        vec![1.0, 2.0, 3.0, 4.0],
        "the compiled HIP entry point mutated caller-owned bytes"
    );
}

/// The Phase 3 HIP gate executes the shared proof's positive path. A
/// program-owned `Copy` is terminal at the FusedElem, the generated wrapper
/// aliases the output view onto that exact intermediate, the kernel result is
/// correct, and the caller input remains unchanged.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn compiled_value_ownership_program_owned_reuse() {
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
        RiscOp::synth_const(Prim::F32, 2.0),
        vec![],
        vec_f32(4),
        None,
    );
    let fused = dag.add_node(
        decl,
        RiscOp::FusedElem {
            ops: vec![chelis_ir::dag::FusedStep {
                op: chelis_ir::dag::FusedStepOp::Mul,
                input_indices: vec![
                    chelis_ir::dag::FusedInput::External(0),
                    chelis_ir::dag::FusedInput::External(1),
                ],
            }],
        },
        vec![owned, scale],
        vec_f32(4),
        None,
    );
    dag.set_reusable_input(fused, owned);
    dag.add_root(fused);

    let generated = codegen_hip(&dag, "compiled_value_ownership_program_owned_reuse")
        .expect("program-owned reuse must codegen");
    assert!(
        generated
            .c_source
            .contains("o_t3 = chelis_device_tensor_borrow(plan_t3, d_t1->data,"),
        "the executed artifact must contain the token-selected alias path"
    );
    assert!(
        generated
            .c_source
            .contains("if (chelis_gpu_is_contiguous(d_t1)) {"),
        "the runtime path must test the same token-selected source"
    );

    let lines = compile_and_run_output_and_inputs(
        &dag,
        "compiled_value_ownership_program_owned_reuse",
        &[TestInput::new("x", &[4], &[1.0, 2.0, 3.0, 4.0])],
    );
    assert_eq!(
        lines.len(),
        2,
        "one output and one caller input are required"
    );
    assert_close_vec(&lines[0], &[2.0, 4.0, 6.0, 8.0]);
    assert_eq!(lines[1], vec![1.0, 2.0, 3.0, 4.0]);
}

// ===========================================================================
// G13: Segmented reductions cover tiny/small/large strategies
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g13_tiny_segmented_sum_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 8),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(3),
        None,
    );
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 16),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::MaxReduce { axis: 1 },
        vec![x],
        vec_f32(2),
        None,
    );
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 128),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        vec_f32(2),
        None,
    );
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(1024),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![x],
        scalar_f32(),
        None,
    );
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
        None,
    );
    let mul = dag.add_node(decl, RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(2, 4),
        None,
    );
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
#[ignore = "manual gate: requires HIP-capable GPU, hipcc, and hipBLAS"]
fn g15_hipblas_batched_matmul_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        tensor4_f32(2, 2, 2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        tensor4_f32(2, 2, 3, 2),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::BlasMatmul {
            batch_dims: vec![
                chelis_ir::dag::DimExpr::Concrete(2),
                chelis_ir::dag::DimExpr::Concrete(2),
            ],
            m: chelis_ir::dag::DimExpr::Concrete(2),
            n: chelis_ir::dag::DimExpr::Concrete(2),
            k: chelis_ir::dag::DimExpr::Concrete(3),
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a, b],
        tensor4_f32(2, 2, 2, 2),
        None,
    );
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g15_hipblas_batched_matmul",
        &[
            TestInput::new(
                "a",
                &[2, 2, 2, 3],
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 2.0, 3.5, -2.0, 1.0, 0.0, 1.0, -3.0,
                    2.0, 4.0, -1.0, 5.0, -2.0, 0.5, 1.5, 3.0, -4.0,
                ],
            ),
            TestInput::new(
                "b",
                &[2, 2, 3, 2],
                &[
                    1.0, 0.0, -1.0, 2.0, 0.5, 3.0, 2.0, -2.0, 1.0, 1.5, -0.5, 4.0, 3.0, 1.0, 0.0,
                    -1.0, 2.0, 2.5, -3.0, 0.5, 1.5, -2.0, 4.0, 1.0,
                ],
            ),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU, hipcc, and hipBLAS"]
fn g15_hipblas_strided_batched_symbolic_batch_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Lit(2),
            DimInfo::Lit(3),
        ],
        precision: Prim::F32,
    };
    let b_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Lit(3),
            DimInfo::Lit(2),
        ],
        precision: Prim::F32,
    };
    let out_ty = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Lit(2),
            DimInfo::Lit(2),
        ],
        precision: Prim::F32,
    };
    let a = dag.add_node(decl, RiscOp::Load { name: "a".into() }, vec![], a_ty, None);
    let b = dag.add_node(decl, RiscOp::Load { name: "b".into() }, vec![], b_ty, None);
    let out = dag.add_node(
        decl,
        RiscOp::BlasMatmul {
            batch_dims: vec![chelis_ir::dag::DimExpr::Sym("batch".into())],
            m: chelis_ir::dag::DimExpr::Concrete(2),
            n: chelis_ir::dag::DimExpr::Concrete(2),
            k: chelis_ir::dag::DimExpr::Concrete(3),
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a, b],
        out_ty,
        None,
    );
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g15_hipblas_strided_batched_symbolic_batch",
        &[
            TestInput::new(
                "a",
                &[3, 2, 3],
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 2.0, 3.5, -2.0, 1.0, 0.0, 1.0, -3.0,
                    2.0, 4.0, -1.0,
                ],
            ),
            TestInput::new(
                "b",
                &[3, 3, 2],
                &[
                    1.0, 0.0, -1.0, 2.0, 0.5, 3.0, 2.0, -2.0, 1.0, 1.5, -0.5, 4.0, 3.0, 1.0, 0.0,
                    -1.0, 2.0, 2.5,
                ],
            ),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g15_noncontiguous_matmul_fallback_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let base_a = dag.add_node(
        decl,
        RiscOp::Load {
            name: "base_a".into(),
        },
        vec![],
        mat_f32(3, 2),
        None,
    );
    let a = dag.add_node(
        decl,
        RiscOp::Permute { axes: vec![1, 0] },
        vec![base_a],
        mat_f32(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(4),
        },
        vec![a],
        tensor3_f32(2, 3, 4),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(2),
        },
        vec![b],
        tensor3_f32(2, 3, 4),
        None,
    );
    let mul = dag.add_node(decl, RiscOp::Mul, vec![ea, eb], tensor3_f32(2, 3, 4), None);
    let out = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(2, 4),
        None,
    );
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

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_sparse_gather_i64_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let table = dag.add_node(
        decl,
        RiscOp::Load {
            name: "table".into(),
        },
        vec![],
        mat_f32(4, 3),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        vec_i64(3),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Gather { axis: 0 },
        vec![table, indices],
        mat_f32(3, 3),
        None,
    );
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g16_sparse_gather_i64",
        &[
            TestInput::new(
                "table",
                &[4, 3],
                &[
                    1.0, 2.0, 3.0, 10.0, 20.0, 30.0, -1.0, -2.0, -3.0, 7.0, 8.0, 9.0,
                ],
            ),
            TestInput::int64("indices", &[3], &[2, 0, 3]),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_sparse_scatter_add_i32_matches_eval_with_duplicate_indices() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let target = dag.add_node(
        decl,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        mat_f32(3, 2),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        vec_i32(4),
        None,
    );
    let updates = dag.add_node(
        decl,
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        mat_f32(4, 2),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::ScatterAdd { axis: 0 },
        vec![target, indices, updates],
        mat_f32(3, 2),
        None,
    );
    dag.add_root(out);

    assert_gpu_matches_eval(
        &dag,
        "g16_sparse_scatter_add_i32",
        &[
            TestInput::new("target", &[3, 2], &[1.0, 2.0, 10.0, 20.0, -1.0, -2.0]),
            TestInput::int32("indices", &[4], &[1, 0, 1, 2]),
            TestInput::new(
                "updates",
                &[4, 2],
                &[0.5, 1.0, 2.0, 3.0, -4.0, 5.0, 6.0, -7.0],
            ),
        ],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_sparse_scatter_replace_i64_duplicate_indices_are_last_write_wins() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let target = dag.add_node(
        decl,
        RiscOp::Load {
            name: "target".into(),
        },
        vec![],
        mat_f32(3, 2),
        None,
    );
    let indices = dag.add_node(
        decl,
        RiscOp::Load {
            name: "indices".into(),
        },
        vec![],
        vec_i64(2),
        None,
    );
    let updates = dag.add_node(
        decl,
        RiscOp::Load {
            name: "updates".into(),
        },
        vec![],
        mat_f32(2, 2),
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Scatter { axis: 0 },
        vec![target, indices, updates],
        mat_f32(3, 2),
        None,
    );
    dag.add_root(out);

    let inputs = [
        TestInput::new("target", &[3, 2], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        TestInput::int64("indices", &[2], &[1, 1]),
        TestInput::new("updates", &[2, 2], &[5.0, 5.0, 6.0, 6.0]),
    ];
    let expected = vec![0.0, 0.0, 6.0, 6.0, 0.0, 0.0];
    assert_eq!(expected_single_output(&dag, &inputs), expected);
    assert_eq!(
        compile_and_run_single_output(&dag, "g16_sparse_scatter_replace_i64", &inputs),
        expected
    );
}

// ===========================================================================
// G17+ : f64 acceptance suite (WS-A2)
//
// The original `TestInput` is f32-pinned (data is `Vec<f32>`). Rather than
// retrofit every f32 helper, the f64 path uses a parallel small harness
// below. Each f64 test parallels an existing f32 G* test so the coverage
// matrix is documented by name (`g17_add_consts_f64_gpu` mirrors
// `g1_add_consts_gpu`, etc.).
//
// The f64 stdout parser tolerates the wider dynamic range (`%.10f` print
// in the harness vs `%.6f` in the f32 path).
// ===========================================================================

#[derive(Clone)]
struct TestInputF64 {
    name: String,
    shape: Vec<usize>,
    data: Vec<f64>,
    dtype: Prim,
}

impl TestInputF64 {
    fn f64(name: &str, shape: &[usize], data: &[f64]) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            data: data.to_vec(),
            dtype: Prim::F64,
        }
    }

    fn evaluator_value(&self) -> TensorValue {
        TensorValue::from_vec(self.shape.clone(), self.data.clone())
    }
}

fn vec_f64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F64,
    }
}

fn scalar_f64() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::F64,
    }
}

fn mat_f64(rows: usize, cols: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F64,
    }
}

fn append_case_lines_f64(
    lines: &mut Vec<String>,
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    inputs: &[TestInputF64],
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
                "    int64_t {prefix}_shape_{slot}[{ndim}] = {{ {dims} }};"
            ));
            lines.push(format!(
                "    {prefix}_input_storage[{slot}] = chelis_alloc({ndim}, {prefix}_shape_{slot}, {dtype});",
                dtype = match input.dtype {
                    Prim::F64 => "CHELIS_DTYPE_F64",
                    other => panic!("f64 harness expected f64 input, got {}", other.name()),
                }
            ));
            lines.push(format!(
                "    chelis_tensor_write *{prefix}_input_guard_{slot} = chelis_tensor_begin_write({prefix}_input_storage[{slot}]);"
            ));
            lines.push(format!(
                "    chelis_write_view {prefix}_input_view_{slot} = chelis_tensor_write_view({prefix}_input_guard_{slot});"
            ));
            for (idx, value) in input.data.iter().enumerate() {
                // Cast through `double *` because the C struct's
                // `chelis_tensor.data` is typed `float *` historically;
                // the wheel runtime stores
                // f64 in 8-byte slots (`chelis_dtype_size`). Sibling of
                // #250/#251/#252: exact f64 bit pattern via
                // `chelis_f64_from_bits`, not a lossy `{:.17e}` decimal.
                lines.push(format!(
                    "    ((double*){prefix}_input_view_{slot}.data)[{idx}] = chelis_f64_from_bits(0x{bits:016x}uLL);",
                    bits = value.to_bits()
                ));
            }
            lines.push(format!(
                "    chelis_tensor_end_write({prefix}_input_guard_{slot});"
            ));
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
        "        chelis_read_view {prefix}_output_view = chelis_tensor_read_view({prefix}_outputs[{prefix}_out_idx]);"
    ));
    lines.push(format!(
        "        for (int {prefix}_i = 0; {prefix}_i < chelis_tensor_numel({prefix}_outputs[{prefix}_out_idx]); {prefix}_i++) {{"
    ));
    lines.push(format!("            if ({prefix}_i > 0) printf(\" \");"));
    lines.push(format!(
        "            printf(\"%.17e\", ((const double*){prefix}_output_view.data)[{prefix}_i]);"
    ));
    lines.push("        }".to_string());
    lines.push("        printf(\"\\n\");".to_string());
    lines.push(format!(
        "        chelis_tensor_release({prefix}_outputs[{prefix}_out_idx]);"
    ));
    lines.push("    }".to_string());
    for slot in 0..input_labels.len() {
        lines.push(format!(
            "    chelis_tensor_release({prefix}_input_storage[{slot}]);"
        ));
    }
}

fn build_main_cpp_f64(
    func_name: &str,
    input_labels: &[String],
    n_out: usize,
    inputs: &[TestInputF64],
) -> String {
    let mut lines = Vec::new();
    append_case_lines_f64(&mut lines, func_name, input_labels, n_out, inputs, "case0");
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

fn compile_and_run_single_output_f64(
    dag: &Dag,
    func_name: &str,
    inputs: &[TestInputF64],
) -> Vec<f64> {
    require_hipcc();
    let result = codegen_hip(dag, func_name).unwrap();
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
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    write_temp_file(
        tmp.path(),
        "main.cpp",
        &build_main_cpp_f64(
            func_name,
            &result.input_labels,
            result.output_labels.len(),
            inputs,
        ),
    );

    let bin_path = tmp.path().join("gpu_correctness_bin_f64");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
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
        .map(|token| token.parse::<f64>().expect("parse output double"))
        .collect()
}

fn expected_single_output_f64(dag: &Dag, inputs: &[TestInputF64]) -> Vec<f64> {
    let input_map: UnordMap<String, TensorValue> = inputs
        .iter()
        .map(|input| (input.name.clone(), input.evaluator_value()))
        .collect();
    let values =
        eval_tensor_roots_with_strict(dag, dag.roots(), |name| input_map.get(name).cloned())
            .expect("evaluator should succeed");
    let root = *dag.roots().first().expect("single root expected");
    values[&root].to_f64_lossy_vec().clone()
}

fn assert_close_vec_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "output length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at index {idx}: expected {e}, got {a}, tol={tol}"
        );
    }
}

fn assert_gpu_matches_eval_f64(dag: &Dag, func_name: &str, inputs: &[TestInputF64], tol: f64) {
    let actual = compile_and_run_single_output_f64(dag, func_name, inputs);
    let expected = expected_single_output_f64(dag, inputs);
    assert_close_vec_f64(&actual, &expected, tol);
}

// G17 mirrors G1: const + const → 3.0, but in f64.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g17_add_consts_f64_gpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f64().precision, 1.0),
        vec![],
        scalar_f64(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f64().precision, 2.0),
        vec![],
        scalar_f64(),
        None,
    );
    let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f64(), None);
    dag.add_root(c);
    assert_gpu_matches_eval_f64(&dag, "g17_add_consts_f64", &[], 1e-12);
}

// G18 mirrors G2: unary exp on f64 (uses `exp(double)`, not `expf`).
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g18_exp_f64_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let y = dag.add_node(decl, RiscOp::Exp, vec![x], vec_f64(4), None);
    dag.add_root(y);
    assert_gpu_matches_eval_f64(
        &dag,
        "g18_exp_f64",
        &[TestInputF64::f64("x", &[4], &[0.5, 1.0, 2.0, -1.0])],
        1e-12,
    );
}

// G19 mirrors G3: binary mul on f64.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g19_mul_f64_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let z = dag.add_node(decl, RiscOp::Mul, vec![x, y], vec_f64(4), None);
    dag.add_root(z);
    assert_gpu_matches_eval_f64(
        &dag,
        "g19_mul_f64",
        &[
            TestInputF64::f64("x", &[4], &[1.0, 2.5, -3.0, 1e10]),
            TestInputF64::f64("y", &[4], &[4.0, 0.25, -2.0, 1e-10]),
        ],
        1e-12,
    );
}

// G20 mirrors G4: reduce_sum on f64. Tests the WS-A0 default accumulator
// resolution: f64 operand → f64 accumulator → f64 result.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g20_sum_reduction_f64_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(8),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::sum_default(0, Prim::F64).expect("f64 reduce_sum"),
        vec![x],
        scalar_f64(),
        None,
    );
    dag.add_root(s);
    assert_gpu_matches_eval_f64(
        &dag,
        "g20_sum_f64",
        &[TestInputF64::f64(
            "x",
            &[8],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )],
        1e-12,
    );
}

// G20b: max-reduce on f64 — verifies `fmax` (not `fmaxf`) is dispatched.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g20b_max_reduce_f64_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(5),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::MaxReduce { axis: 0 },
        vec![x],
        scalar_f64(),
        None,
    );
    dag.add_root(m);
    assert_gpu_matches_eval_f64(
        &dag,
        "g20b_max_f64",
        &[TestInputF64::f64("x", &[5], &[1.0, -2.0, 3.5, 0.0, 2.5])],
        1e-12,
    );
}

// G20c: min-reduce on f64. The kernel was previously panic("Phase 3j-pre");
// WS-A2 lifts to a real kernel for f32 + f64.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g20c_min_reduce_f64_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(5),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::MinReduce { axis: 0 },
        vec![x],
        scalar_f64(),
        None,
    );
    dag.add_root(m);
    assert_gpu_matches_eval_f64(
        &dag,
        "g20c_min_f64",
        &[TestInputF64::f64("x", &[5], &[1.0, -2.0, 3.5, 0.0, 2.5])],
        1e-12,
    );
}

// G20d: prod-reduce on f64. Similarly lifted in WS-A2.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g20d_prod_reduce_f64_gpu_matches_cpu() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::ProdReduce { axis: 0 },
        vec![x],
        scalar_f64(),
        None,
    );
    dag.add_root(m);
    assert_gpu_matches_eval_f64(
        &dag,
        "g20d_prod_f64",
        &[TestInputF64::f64("x", &[4], &[1.5, 2.0, -0.5, 4.0])],
        1e-12,
    );
}

// G21 mirrors G4-symbolic-matmul on f64. Exercises the
// `chelis_hipblas_dgemm_row_major` dispatch path.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g21_matmul_f64_gpu_matches_cpu() {
    use chelis_ir::dag::DimExpr;

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f64(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f64(3, 4),
        None,
    );
    let mm = dag.add_node(
        decl,
        RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(4),
            DimExpr::Concrete(3),
            Prim::F64,
        )
        .expect("f64 matmul default"),
        vec![a, b],
        mat_f64(2, 4),
        None,
    );
    dag.add_root(mm);
    assert_gpu_matches_eval_f64(
        &dag,
        "g21_matmul_f64",
        &[
            TestInputF64::f64("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TestInputF64::f64(
                "b",
                &[3, 4],
                &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ),
        ],
        1e-10,
    );
}

// WS-A3 bf16 codegen acceptance ("HIP codegen no longer panics on bf16
// matmul") is pinned by
// `crates/chelis-backend-hip/tests/bf16_f16_matmul.rs::
// bf16_matmul_default_accumulator_emits_bf16_gemm_wrapper`, which is a
// strict superset: it asserts codegen succeeds AND emits the
// `chelis_hipblas_bf16_gemm_f32_acc_row_major` wrapper with `-lhipblas`.
// The former `ws_a3_hip_admits_bf16_matmul_at_codegen` here only asserted
// "does not panic" and was removed as a duplicate.

// Negative coverage parallel to the bf16 case: i8 matmul stays rejected
// (WS-A4 admits i8 reduce_sum, but spec/04-type-system.md §5.7.2
// explicitly excludes integer matmul).
#[test]
fn ws_a2_hip_f1_still_rejects_i8_matmul() {
    use chelis_ir::dag::DimExpr;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let i8_ty = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
        precision: Prim::Int8,
    };
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        i8_ty.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        i8_ty.clone(),
        None,
    );
    let mm_op = RiscOp::BlasMatmul {
        batch_dims: vec![],
        m: DimExpr::Concrete(2),
        n: DimExpr::Concrete(2),
        k: DimExpr::Concrete(2),
        accumulator: Prim::Int32,
    };
    let mm = dag.add_node(decl, mm_op, vec![a, b], i8_ty, None);
    dag.add_root(mm);

    let result = std::panic::catch_unwind(|| {
        let _ = codegen_hip(&dag, "ws_a2_hip_i8");
    });
    let payload = result.expect_err("HIP codegen must reject i8 matmul");
    let msg = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&'static str>().copied())
        .unwrap_or("<non-string panic payload>");
    assert!(
        msg.contains("F1: BlasMatmul on operand precision"),
        "expected F1 HIP rejection for i8 matmul, got: {msg}"
    );
    assert!(
        msg.contains("i8"),
        "F1 message should name the rejected precision; got: {msg}"
    );
}

// Positive coverage that codegen runs to completion for f64 matmul (so
// the F1 lift actually admits f64). The kernel is not launched here —
// a full GPU run is the manual gate above. This test just probes that
// the host-source emitter reaches the dgemm helper without panicking.
#[test]
fn ws_a2_hip_f1_admits_f64_matmul_at_codegen() {
    use chelis_ir::dag::DimExpr;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f64(2, 3),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f64(3, 4),
        None,
    );
    let mm = dag.add_node(
        decl,
        RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(4),
            DimExpr::Concrete(3),
            Prim::F64,
        )
        .expect("f64 matmul default"),
        vec![a, b],
        mat_f64(2, 4),
        None,
    );
    dag.add_root(mm);
    let result = codegen_hip(&dag, "ws_a2_codegen_f64_matmul").unwrap();
    assert!(
        result.c_source.contains("chelis_hipblas_dgemm_row_major("),
        "f64 matmul must dispatch to dgemm"
    );
    assert!(
        !result.c_source.contains("chelis_hipblas_sgemm_row_major("),
        "f64 matmul must not silently fall back to sgemm"
    );
    assert!(
        result.link_flags.iter().any(|f| f == "-lhipblas"),
        "dgemm dispatch must request hipBLAS link"
    );
}

// Positive coverage that f64 reduce_sum kernel string is actually
// emitted with the f64 accumulator type. Cheaper than a GPU run; pairs
// with the manual gate above to lock the f32→f64 codegen split.
#[test]
fn ws_a2_hip_reduce_sum_f64_emits_double_accumulator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::sum_default(0, Prim::F64).expect("f64 reduce_sum"),
        vec![x],
        scalar_f64(),
        None,
    );
    dag.add_root(s);
    let result = codegen_hip(&dag, "ws_a2_reduce_sum_f64").unwrap();
    assert!(
        result.c_source.contains("kernel_sum_ax0_f64"),
        "f64 reduce_sum must use the f64-suffixed kernel name"
    );
    assert!(
        result.c_source.contains("double acc = 0.0;"),
        "f64 reduce_sum kernel must declare a double accumulator"
    );
    assert!(
        result.c_source.contains("const double *a"),
        "f64 reduce_sum kernel must read from a double * input"
    );
}

// Positive coverage that lifted reductions emit f64 kernels.
#[test]
fn ws_a2_hip_min_reduce_f64_kernel_emitted() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::MinReduce { axis: 0 },
        vec![x],
        scalar_f64(),
        None,
    );
    dag.add_root(m);
    let result = codegen_hip(&dag, "ws_a2_min_reduce_f64").unwrap();
    assert!(
        result.c_source.contains("kernel_min_ax0_f64"),
        "f64 min_reduce must use the f64-suffixed kernel name"
    );
    assert!(
        result.c_source.contains("fmin(acc"),
        "f64 min_reduce kernel must use fmin (not fminf)"
    );
}

#[test]
fn ws_a2_hip_argmax_f64_kernel_emitted() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f64(4),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::Argmax { axis: 0 },
        vec![x],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_root(m);
    let result = codegen_hip(&dag, "ws_a2_argmax_f64").unwrap();
    assert!(
        result.c_source.contains("kernel_argmax_ax0_f64"),
        "f64 argmax must use the f64-suffixed kernel name"
    );
    assert!(
        result.c_source.contains("long long *out"),
        "argmax output must be long long *"
    );
}

// ============================================================================
// WS-A4: i8 / i16 end-to-end execution tests on GPU.
//
// These mirror the C-backend WS-A4 tests in
// crates/chelis-backend-c/tests/exec_compile.rs. The
// `gpu_*` tests are `#[ignore]` so they only run under the manual
// HIP gate (see file header). The codegen-shape tests below run by
// default in CI so a kernel-name regression is caught without
// requiring a HIP toolchain.
// ============================================================================

/// WS-A4: dtype-aware single-output runner. The output is read as the
/// requested `out_c_ty` (e.g. `int8_t`, `int32_t`) and parsed back as
/// `i64` so callers can assert exact integer equality. This sits next
/// to the f32 `compile_and_run_single_output` rather than retrofitting
/// it so the f32 callers stay byte-identical and the WS-A4 lift stays
/// localized to this section.
#[allow(dead_code, reason = "WS-A4 manual HIP gate; called from i8/i16 tests")]
fn compile_and_run_single_output_typed_i64(
    dag: &Dag,
    func_name: &str,
    inputs: &[TestInput],
    out_c_ty: &str,
    out_printf_spec: &str,
) -> Vec<i64> {
    require_hipcc();
    let result = codegen_hip(dag, func_name).unwrap();
    assert_eq!(
        result.output_labels.len(),
        1,
        "WS-A4 manual harness expects a single output"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);

    let mut input_setup = Vec::new();
    append_case_lines(
        &mut input_setup,
        func_name,
        &result.input_labels,
        result.output_labels.len(),
        inputs,
        "case0",
    );
    // Reuse `append_case_lines`'s prefix (input init + call invocation)
    // and then append a typed read-and-print block. The float-formatted
    // tail emitted by `append_case_lines` is dropped because we only
    // copy lines through the call invocation line.
    let mut prefix_lines = Vec::new();
    for line in &input_setup {
        prefix_lines.push(line.clone());
        if line.contains(&format!("{func_name}(")) {
            break;
        }
    }
    prefix_lines.push(
        "    chelis_read_view case0_output_view = chelis_tensor_read_view(case0_outputs[0]);"
            .to_string(),
    );
    prefix_lines
        .push("    for (int i = 0; i < chelis_tensor_numel(case0_outputs[0]); i++) {".to_string());
    prefix_lines.push("        if (i > 0) printf(\" \");".to_string());
    prefix_lines.push(format!(
        "        printf(\"{out_printf_spec}\", (long long)((const {out_c_ty}*)case0_output_view.data)[i]);"
    ));
    prefix_lines.push("    }".to_string());
    prefix_lines.push("    printf(\"\\n\");".to_string());
    prefix_lines.push("    chelis_tensor_release(case0_outputs[0]);".to_string());
    for slot in 0..result.input_labels.len() {
        prefix_lines.push(format!(
            "    chelis_tensor_release(case0_input_storage[{slot}]);"
        ));
    }

    let main_src = format!(
        r#"#include "chelis_runtime.h"
#include <stdint.h>
extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
{body}
    return 0;
}}
"#,
        body = prefix_lines.join("\n")
    );
    write_temp_file(tmp.path(), "main.cpp", &main_src);

    let bin_path = tmp.path().join("gpu_correctness_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(tmp.path().join("chelis_device_owner.cpp"));
    compile_cmd.arg(&staged.archive);
    compile_cmd.arg("-lpthread");
    compile_cmd.arg("-ldl");
    compile_cmd.args(&result.link_flags);
    compile_cmd.arg("-o");
    compile_cmd.arg(&bin_path);
    let compile = compile_cmd.output().expect("run hipcc");
    assert!(
        compile.status.success(),
        "hipcc failed (WS-A4 typed):\nstderr: {}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );

    let run = Command::new(&bin_path).output().expect("run gpu binary");
    assert!(
        run.status.success(),
        "GPU binary failed (WS-A4 typed):\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .map(|token| token.parse::<i64>().expect("parse output i64"))
        .collect()
}

/// WS-A4: i8 add on GPU. Picks values that exercise the i8 elementwise
/// kernel path, including a planned wrap (127 + 1 = -128).
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn ws_a4_i8_add_gpu_matches_two_complement_wrap() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Add,
        vec![a, b],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    dag.add_root(out);

    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "ws_a4_i8_add",
        &[
            TestInput::int8("a", &[4], &[1, 100, 127, -50]),
            TestInput::int8("b", &[4], &[2, 50, 1, -80]),
        ],
        "int8_t",
        "%lld",
    );
    // i8 wrap-around: 100+50 = 150 → -106; 127+1 = 128 → -128;
    // -50 + -80 = -130 → +126.
    assert_eq!(actual, vec![3, -106, -128, 126]);
}

/// WS-A4: i16 mul on GPU. 1000 * 1000 = 1_000_000 wraps in i16 to
/// 16960 (1_000_000 mod 65536, with high bit clear).
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn ws_a4_i16_mul_gpu_matches_two_complement_wrap() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int16,
        },
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int16,
        },
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Mul,
        vec![a, b],
        TensorType {
            dims: vec![DimInfo::Lit(2)],
            precision: Prim::Int16,
        },
        None,
    );
    dag.add_root(out);

    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "ws_a4_i16_mul",
        &[
            TestInput::int16("a", &[2], &[100, 1000]),
            TestInput::int16("b", &[2], &[100, 1000]),
        ],
        "int16_t",
        "%lld",
    );
    assert_eq!(actual, vec![10000, 16960]);
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn direct_i32_extrema_gpu_select_exact_signed_operands() {
    for (name, op, expected) in [
        (
            "direct_i32_max",
            RiscOp::MaxElem,
            vec![i32::MAX as i64, -1, 0, i32::MAX as i64],
        ),
        (
            "direct_i32_min",
            RiscOp::MinElem,
            vec![i32::MIN as i64, -1, 0, i32::MIN as i64],
        ),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            vec_i32(4),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            vec_i32(4),
            None,
        );
        let out = dag.add_node(decl, op, vec![a, b], vec_i32(4), None);
        dag.add_root(out);
        let actual = compile_and_run_single_output_typed_i64(
            &dag,
            name,
            &[
                TestInput::int32("a", &[4], &[i32::MAX, -1, 0, i32::MIN]),
                TestInput::int32("b", &[4], &[i32::MIN, -1, 0, i32::MAX]),
            ],
            "int32_t",
            "%lld",
        );
        assert_eq!(actual, expected, "{name}");
    }
}

/// WS-A4: i8 reduce_sum on GPU promotes accumulator to i32 per spec
/// §5.7.1. 200 ones at i8 source produce 200 at i32 output (no wrap).
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn ws_a4_i8_reduce_sum_gpu_promotes_to_i32() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(200)],
            precision: Prim::Int8,
        },
        None,
    );
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int8).expect("i8 sum_default must succeed");
    let out = dag.add_node(
        decl,
        sum_op,
        vec![a],
        TensorType {
            dims: vec![],
            precision: Prim::Int32,
        },
        None,
    );
    dag.add_root(out);

    let data: Vec<i8> = vec![1; 200];
    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "ws_a4_i8_reduce_sum",
        &[TestInput::int8("a", &[200], &data)],
        "int32_t",
        "%lld",
    );
    assert_eq!(actual, vec![200], "200 i8 ones must sum to 200 in i32");
}

/// WS-A4: i16 reduce_sum on GPU. 200 i16 1000s = 200_000 (overflows
/// i16 but fits in i32). Mirrors `ws_a4_i8_reduce_sum_gpu_promotes_to_i32`.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn ws_a4_i16_reduce_sum_gpu_promotes_to_i32() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(200)],
            precision: Prim::Int16,
        },
        None,
    );
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int16).expect("i16 sum_default must succeed");
    let out = dag.add_node(
        decl,
        sum_op,
        vec![a],
        TensorType {
            dims: vec![],
            precision: Prim::Int32,
        },
        None,
    );
    dag.add_root(out);

    let data: Vec<i16> = vec![1000; 200];
    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "ws_a4_i16_reduce_sum",
        &[TestInput::int16("a", &[200], &data)],
        "int32_t",
        "%lld",
    );
    assert_eq!(
        actual,
        vec![200_000],
        "200 i16 1000s must sum to 200_000 in i32 (overflows i16)"
    );
}

/// chelis#178: integer `trunc_div` on GPU rounds the quotient toward
/// zero (the native C `/`). Operands `{7, 7, -7, -7} / {2, -2, 2, -2}`
/// ⇒ `{3, -3, -3, 3}`. Every sign combination is exercised so the
/// GPU result is pinned against the spec/05 §2.1 sign-rounding rule.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn chelis_178_trunc_div_i32_gpu_rounds_toward_zero() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i32(4),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_i32(4),
        None,
    );
    let out = dag.add_node(decl, RiscOp::TruncDiv, vec![a, b], vec_i32(4), None);
    dag.add_root(out);

    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "chelis_178_trunc_div_i32",
        &[
            TestInput::int32("a", &[4], &[7, 7, -7, -7]),
            TestInput::int32("b", &[4], &[2, -2, 2, -2]),
        ],
        "int32_t",
        "%lld",
    );
    assert_eq!(
        actual,
        vec![3, -3, -3, 3],
        "trunc_div must round toward zero on GPU"
    );
}

/// chelis#178: integer `floor_div` on GPU rounds the quotient toward
/// −∞ (native `/` plus a remainder-sign correction). The same operands
/// `{7, 7, -7, -7} / {2, -2, 2, -2}` ⇒ `{3, -4, -4, 3}` — differing
/// from trunc on the two mixed-sign cases. Confirms the device
/// sign-correction kernel agrees with the C backend and evaluator.
#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn chelis_178_floor_div_i32_gpu_rounds_toward_neg_inf() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i32(4),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_i32(4),
        None,
    );
    let out = dag.add_node(decl, RiscOp::FloorDiv, vec![a, b], vec_i32(4), None);
    dag.add_root(out);

    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "chelis_178_floor_div_i32",
        &[
            TestInput::int32("a", &[4], &[7, 7, -7, -7]),
            TestInput::int32("b", &[4], &[2, -2, 2, -2]),
        ],
        "int32_t",
        "%lld",
    );
    assert_eq!(
        actual,
        vec![3, -4, -4, 3],
        "floor_div must round toward -inf on GPU"
    );
}

/// WS-A4 codegen-shape test (no GPU required): the dtype-suffixed
/// kernel name appears in the generated HIP source for an i8 add.
/// Runs by default so a kernel-name regression is caught without a
/// HIP toolchain.
#[test]
fn ws_a4_i8_add_emits_dtype_suffixed_kernel_name() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    let out = dag.add_node(
        decl,
        RiscOp::Add,
        vec![a, b],
        TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::Int8,
        },
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "ws_a4_i8_add_codegen").unwrap();
    assert!(
        result.c_source.contains("kernel_add_i8"),
        "i8 add must emit `kernel_add_i8`; got source:\n{}",
        result.c_source
    );
    assert!(
        result.c_source.contains("int8_t"),
        "i8 add kernel source must mention int8_t; got source:\n{}",
        result.c_source
    );
}

/// WS-A4 codegen-shape test (no GPU required): i8 reduce_sum emits a
/// (source, accumulator) suffixed kernel name that resolves at
/// kernel-source-emit time to the integer-promoted template.
#[test]
fn ws_a4_i8_reduce_sum_emits_promoted_kernel_name() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(200)],
            precision: Prim::Int8,
        },
        None,
    );
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int8).expect("i8 sum_default must succeed");
    let out = dag.add_node(
        decl,
        sum_op,
        vec![a],
        TensorType {
            dims: vec![],
            precision: Prim::Int32,
        },
        None,
    );
    dag.add_root(out);

    let result = codegen_hip(&dag, "ws_a4_i8_reduce_sum_codegen").unwrap();
    // Kernel name encodes both source dtype (`_i8`) and accumulator
    // dtype (`_i32`). The unsuffixed `kernel_sum_ax0` would be the f32
    // → f32 path, so its absence is the regression-shield.
    assert!(
        result.c_source.contains("kernel_sum_ax0_i8_i32"),
        "i8 reduce_sum must emit `kernel_sum_ax0_i8_i32`; got source:\n{}",
        result.c_source
    );
    assert!(
        result.c_source.contains("int32_t acc"),
        "i8 reduce_sum kernel source must use int32_t accumulator (per spec §5.7.1); got source:\n{}",
        result.c_source
    );
    assert!(
        result.c_source.contains("(int32_t)a[src_idx]"),
        "i8 reduce_sum kernel source must widen each source element to int32_t before summing; got source:\n{}",
        result.c_source
    );
}

// ===========================================================================
// G16 (WS-8A): pad / shrink GPU == evaluator. The C backend is the numeric
// oracle (spec/08 §2); each case asserts the HIP GPU result matches the
// `chelis-ir` evaluator within tolerance, which is what `assert_gpu_matches_eval`
// checks. Coverage: rank-1 and rank-2 pad/shrink, non-zero fill, and pad over
// a strided (non-contiguous) source so the source-strides path is exercised.
// ===========================================================================

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_pad_1d_zero_fill_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::zero_pad(
            Prim::F32,
            vec![(chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(1))],
        ),
        vec![x],
        vec_f32(6),
        None,
    );
    dag.add_root(p);
    assert_gpu_matches_eval(
        &dag,
        "g16_pad_1d",
        &[TestInput::new("x", &[4], &[1.0, 2.0, 3.0, 4.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_pad_1d_nonzero_fill_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(3),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::pad(
            vec![(chelis_ir::dag::RtDim::Lit(2), chelis_ir::dag::RtDim::Lit(1))],
            chelis_types::scalar_from_f64("pad", Prim::F32, -7.5).unwrap(),
        ),
        vec![x],
        vec_f32(6),
        None,
    );
    dag.add_root(p);
    assert_gpu_matches_eval(
        &dag,
        "g16_pad_1d_nonzero",
        &[TestInput::new("x", &[3], &[10.0, 20.0, 30.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_pad_2d_asymmetric_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(2, 3),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::zero_pad(
            Prim::F32,
            // before/after per axis: row axis (1,0), col axis (0,2) →
            // output is 3x5.
            vec![
                (chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(0)),
                (chelis_ir::dag::RtDim::Lit(0), chelis_ir::dag::RtDim::Lit(2)),
            ],
        ),
        vec![x],
        mat_f32(3, 5),
        None,
    );
    dag.add_root(p);
    assert_gpu_matches_eval(
        &dag,
        "g16_pad_2d",
        &[TestInput::new(
            "x",
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_pad_over_strided_source_matches_eval() {
    // The source is a strided view (every other element), so the pad
    // kernel must read through the source strides, not assume contiguity.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(6),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::Stride {
            strides: vec![chelis_ir::dag::RtDim::Lit(2)],
        },
        vec![x],
        vec_f32(3),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::pad(
            vec![(chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(1))],
            chelis_types::scalar_from_f64("pad", Prim::F32, 9.0).unwrap(),
        ),
        vec![s],
        vec_f32(5),
        None,
    );
    dag.add_root(p);
    assert_gpu_matches_eval(
        &dag,
        "g16_pad_strided_src",
        &[TestInput::new("x", &[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_shrink_1d_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(6),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::Shrink {
            bounds: vec![(chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(5))],
        },
        vec![x],
        vec_f32(4),
        None,
    );
    dag.add_root(s);
    assert_gpu_matches_eval(
        &dag,
        "g16_shrink_1d",
        &[TestInput::new("x", &[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_shrink_2d_matches_eval() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        mat_f32(3, 4),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::Shrink {
            // keep rows [1,3) and cols [0,2) → 2x2 interior crop.
            bounds: vec![
                (chelis_ir::dag::RtDim::Lit(1), chelis_ir::dag::RtDim::Lit(3)),
                (chelis_ir::dag::RtDim::Lit(0), chelis_ir::dag::RtDim::Lit(2)),
            ],
        },
        vec![x],
        mat_f32(2, 2),
        None,
    );
    dag.add_root(s);
    assert_gpu_matches_eval(
        &dag,
        "g16_shrink_2d",
        &[TestInput::new(
            "x",
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )],
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn g16_pad_then_shrink_roundtrip_matches_eval() {
    // pad then shrink the padded margin back off must recover the input;
    // composing the two kernels exercises both launch paths in one DAG.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::zero_pad(
            Prim::F32,
            vec![(chelis_ir::dag::RtDim::Lit(2), chelis_ir::dag::RtDim::Lit(2))],
        ),
        vec![x],
        vec_f32(8),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::Shrink {
            bounds: vec![(chelis_ir::dag::RtDim::Lit(2), chelis_ir::dag::RtDim::Lit(6))],
        },
        vec![p],
        vec_f32(4),
        None,
    );
    dag.add_root(s);
    assert_gpu_matches_eval(
        &dag,
        "g16_pad_shrink_roundtrip",
        &[TestInput::new("x", &[4], &[3.5, -1.0, 2.25, 8.0])],
    );
}

// ===========================================================================
// chelis#1291: dedicated [05-OP-29] Count kernel, GPU == evaluator exactly.
//
// The C lane's agreement with the evaluator is #1287's core receipt
// (`scripts/dtype_count_oracle.py`), so exact agreement with the evaluator
// here is exact agreement with compiled C as well. Every test is part of the
// manual hardware gate:
//
//     scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness count_ -- --ignored --test-threads=1
// ===========================================================================

fn count_dag(input: TensorType, axes: Vec<usize>, output: TensorType) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let mask = dag.add_node(
        decl,
        RiscOp::Load {
            name: "mask".into(),
        },
        vec![],
        input,
        None,
    );
    let count = dag.add_node(decl, RiscOp::Count { axes }, vec![mask], output, None);
    dag.add_root(count);
    dag
}

fn bool_tensor(dims: &[usize]) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision: Prim::Bool,
    }
}

fn i64_tensor(dims: &[usize]) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision: Prim::Int64,
    }
}

/// Deterministic mask: element `i` is true when `(i * 7 + 3) % 5 < 2`.
fn count_mask(len: usize) -> Vec<bool> {
    (0..len).map(|i| (i * 7 + 3) % 5 < 2).collect()
}

fn assert_count_gpu_matches_eval_exactly(dag: &Dag, func_name: &str, mask: TestInput) {
    let actual = compile_and_run_single_output_typed_i64(
        dag,
        func_name,
        std::slice::from_ref(&mask),
        "int64_t",
        "%lld",
    );
    let expected = expected_single_output(dag, std::slice::from_ref(&mask))
        .into_iter()
        .map(|value| value as i64)
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "{func_name}: GPU Count must equal the evaluator exactly"
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn count_positional_multi_axis_gpu_matches_eval() {
    let dag = count_dag(bool_tensor(&[2, 3, 5]), vec![2, 0], i64_tensor(&[3]));
    let mask = count_mask(30);
    assert_count_gpu_matches_eval_exactly(
        &dag,
        "count_positional_multi_axis",
        TestInput::bool8("mask", &[2, 3, 5], &mask),
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn count_named_fixed_axes_gpu_matches_eval() {
    let dag = count_dag(
        TensorType {
            dims: vec![
                DimInfo::Named("batch".into(), Some(2)),
                DimInfo::Named("seq".into(), Some(4)),
                DimInfo::Named("feature".into(), Some(3)),
            ],
            precision: Prim::Bool,
        },
        vec![1],
        TensorType {
            dims: vec![
                DimInfo::Named("batch".into(), Some(2)),
                DimInfo::Named("feature".into(), Some(3)),
            ],
            precision: Prim::Int64,
        },
    );
    let mask = count_mask(24);
    assert_count_gpu_matches_eval_exactly(
        &dag,
        "count_named_fixed_axes",
        TestInput::bool8("mask", &[2, 4, 3], &mask),
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn count_empty_selected_extent_gpu_is_zero() {
    let dag = count_dag(bool_tensor(&[2, 0, 5]), vec![1], i64_tensor(&[2, 5]));
    let actual = compile_and_run_single_output_typed_i64(
        &dag,
        "count_empty_selected_extent",
        &[TestInput::bool8("mask", &[2, 0, 5], &[])],
        "int64_t",
        "%lld",
    );
    assert_eq!(actual, vec![0; 10]);
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn count_odd_leaf_count_gpu_matches_eval() {
    let dag = count_dag(bool_tensor(&[3, 7]), vec![1], i64_tensor(&[3]));
    let mask = count_mask(21);
    assert_count_gpu_matches_eval_exactly(
        &dag,
        "count_odd_leaf_count",
        TestInput::bool8("mask", &[3, 7], &mask),
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn count_large_leaf_count_gpu_matches_eval() {
    // 4097 leaves per output element: an odd count deeper than any
    // power-of-two split, so the explicit balanced-tree stack is exercised
    // well past its first frames.
    let dag = count_dag(bool_tensor(&[2, 4097]), vec![1], i64_tensor(&[2]));
    let mask = count_mask(2 * 4097);
    assert_count_gpu_matches_eval_exactly(
        &dag,
        "count_large_leaf_count",
        TestInput::bool8("mask", &[2, 4097], &mask),
    );
}

#[test]
#[ignore = "manual gate: requires HIP-capable GPU and hipcc"]
fn count_input_with_a_non_bool_payload_traps_at_the_runtime_write_boundary() {
    // A `Bool8` byte outside {0, 1} is not a member of the dtype. The
    // runtime's `chelis_tensor_end_write` rejects it with a loud domain
    // trap before the entry is ever called, which is why the kernel's own
    // status-code-1 check is a backstop rather than the primary guard: no
    // runtime-produced tensor can reach the kernel with such a byte.
    require_hipcc();
    let dag = count_dag(bool_tensor(&[4]), vec![0], i64_tensor(&[]));
    let func_name = "count_non_bool_payload";
    let result = codegen_hip(&dag, func_name).unwrap();

    let tmp = tempfile::tempdir().expect("tempdir");
    let hip_rt = hip_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_hip_runtime.h",
        &fs::read_to_string(hip_rt.join("chelis_hip_runtime.h")).expect("hip runtime header"),
    );
    support::stage_device_runtime(tmp.path());
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.cpp", &result.c_source);
    write_temp_file(
        tmp.path(),
        "main.cpp",
        &format!(
            r#"#include "chelis_runtime.h"
#include <stdint.h>
extern "C" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
    int64_t shape[1] = {{ 4 }};
    chelis_tensor *input_storage[1] = {{ chelis_alloc(1, shape, CHELIS_DTYPE_BOOL) }};
    chelis_tensor_write *guard = chelis_tensor_begin_write(input_storage[0]);
    chelis_write_view view = chelis_tensor_write_view(guard);
    ((uint8_t*)view.data)[0] = 1;
    ((uint8_t*)view.data)[1] = 0;
    ((uint8_t*)view.data)[2] = 2;
    ((uint8_t*)view.data)[3] = 1;
    chelis_tensor_end_write(guard);
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(input_storage, 1, outputs, 1);
    printf("count returned\n");
    return 0;
}}
"#
        ),
    );

    let bin_path = tmp.path().join("count_non_bool_payload_bin");
    let mut compile_cmd = Command::new("hipcc");
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("main.cpp"));
    compile_cmd.arg(tmp.path().join("model.cpp"));
    compile_cmd.arg(&staged.archive);
    compile_cmd.arg("-lpthread");
    compile_cmd.arg("-ldl");
    compile_cmd.args(&result.link_flags);
    compile_cmd.arg("-o");
    compile_cmd.arg(&bin_path);
    let compile = compile_cmd.output().expect("run hipcc");
    assert!(
        compile.status.success(),
        "hipcc failed:
stderr: {}
source:
{}",
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );

    let run = Command::new(&bin_path).output().expect("run gpu binary");
    let stderr = String::from_utf8_lossy(&run.stderr);
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        !run.status.success(),
        "a non-bool payload must abort, but the binary exited cleanly:
stdout: {stdout}
stderr: {stderr}"
    );
    assert!(!stdout.contains("count returned"), "{stdout}");
    assert!(
        stderr.contains("Domain") && stderr.contains("noncanonical byte 2"),
        "expected the runtime Bool8 domain trap in stderr:\n{stderr}"
    );
}
