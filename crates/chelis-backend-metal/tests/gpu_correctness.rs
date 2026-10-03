//! GPU execution correctness tests for the Metal backend.
//!
//! These require an Apple Silicon Mac with a usable Metal device. They
//! are `#[ignore]` by default — run with:
//!
//! ```sh
//! cargo test -p chelis-backend-metal --test gpu_correctness -- --ignored --test-threads=1
//! ```
//!
//! Manual gate (`docs/manual_gates.md`) for the Metal agreement contract in
//! spec/08-backends.md §4.6. Not part of default CI; the macOS smoke step gates compile+link only because GitHub's
//! macos-latest VMs may return null from `MTLCreateSystemDefaultDevice`.
//!
//! Numerical tolerance: MSL's default `exp`/`log`/`sqrt`/`sin` are fast-math
//! variants. We use a Metal-specific tolerance that's wider than HIP's
//! for transcendental-heavy kernels — see ABS_TOL/REL_TOL constants and
//! the per-test relaxations.
// chelis#2957 S7: Rust std transcendental until S7 moves this to chelis-crmath.
#![allow(clippy::disallowed_methods)]

mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
use chelis_types::types::Prim;
use chelis_unord::UnordMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use support::codegen_metal;

/// Exercise raw signed storage through the generated Objective-C++ wrapper
/// and actual MSL, including the host-visible first-negative shift trap.
#[cfg(target_os = "macos")]
fn run_bitwise_gpu_case(
    prim: Prim,
    kinds: &[chelis_types::BitwiseKind],
    rhs: &[i64],
) -> std::process::Output {
    let n = rhs.len();
    assert!(n <= 4);
    let (c_ty, tag, width, min_literal) = match prim {
        Prim::Int8 => ("int8_t", "CHELIS_DTYPE_I8", 8, "-128LL"),
        Prim::Int16 => ("int16_t", "CHELIS_DTYPE_I16", 16, "-32768LL"),
        Prim::Int32 => ("int32_t", "CHELIS_DTYPE_I32", 32, "-2147483648LL"),
        Prim::Int64 => ("int64_t", "CHELIS_DTYPE_I64", 64, "INT64_MIN"),
        _ => panic!("integer bitwise GPU case only"),
    };
    let ty = TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: prim,
    };
    let mut dag = Dag::new();
    let decl = dag.declare("bitwise");
    let lhs = dag.add_node(
        decl,
        RiscOp::Load { name: "lhs".into() },
        vec![],
        ty.clone(),
        None,
    );
    let rhs_node = dag.add_node(
        decl,
        RiscOp::Load { name: "rhs".into() },
        vec![],
        ty.clone(),
        None,
    );
    for &kind in kinds {
        let result = dag.add_node(
            decl,
            RiscOp::Bitwise(kind),
            vec![lhs, rhs_node],
            ty.clone(),
            None,
        );
        dag.add_root(result);
    }
    require_clangxx();
    let generated = codegen_metal(&dag, "bitwise_device");
    assert_eq!(generated.input_labels, ["lhs", "rhs"]);
    assert_eq!(generated.output_labels.len(), kinds.len());
    let temporary = tempfile::tempdir().expect("tempdir");
    let runtime = metal_runtime_src_dir();
    write_temp_file(
        temporary.path(),
        "chelis_metal_runtime.h",
        &fs::read_to_string(runtime.join("chelis_metal_runtime.h")).expect("metal runtime"),
    );
    let staged = chelis_runtime_bundle::stage(temporary.path()).expect("stage runtime");
    write_temp_file(temporary.path(), "model.mm", &generated.mm_source);
    let rhs_values = rhs
        .iter()
        .map(|value| format!("{value}LL"))
        .collect::<Vec<_>>()
        .join(", ");
    let driver = format!(
        r#"#import <Foundation/Foundation.h>
#include "chelis_runtime.h"
#include <stdint.h>
#include <stdio.h>
extern "C" void bitwise_device(chelis_tensor**, int, chelis_tensor**, int);
int main(void) {{
  @autoreleasepool {{
    int64_t shape[1] = {{{n}}};
    int64_t lhs_values[4] = {{-1LL, 1LL, {min_literal}, 8LL}};
    int64_t rhs_values[4] = {{{rhs_values}}};
    chelis_tensor* inputs[2] = {{chelis_alloc(1, shape, {tag}), chelis_alloc(1, shape, {tag})}};
    for (int slot = 0; slot < 2; ++slot) {{
      chelis_tensor_write* guard = chelis_tensor_begin_write(inputs[slot]);
      chelis_write_view view = chelis_tensor_write_view(guard);
      for (int i = 0; i < {n}; ++i) (({c_ty}*)view.data)[i] = ({c_ty})(slot == 0 ? lhs_values[i] : rhs_values[i]);
      chelis_tensor_end_write(guard);
    }}
    chelis_tensor* outputs[{roots}] = {{0}};
    bitwise_device(inputs, 2, outputs, {roots});
    for (int root = 0; root < {roots}; ++root) {{
      chelis_read_view view = chelis_tensor_read_view(outputs[root]);
      printf("OUT %d", root);
      for (int i = 0; i < {n}; ++i) printf(" %lld", (long long)((const {c_ty}*)view.data)[i]);
      printf("\n");
      chelis_tensor_release(outputs[root]);
    }}
    chelis_tensor_release(inputs[0]);
    chelis_tensor_release(inputs[1]);
  }}
  return 0;
}}
"#,
        roots = kinds.len(),
    );
    write_temp_file(temporary.path(), "driver.mm", &driver);
    let binary = temporary.path().join(format!("bitwise_{width}"));
    let build = Command::new("xcrun")
        .args(["-sdk", "macosx", "clang++", "-O2"])
        .args(&generated.compile_flags)
        .arg(temporary.path().join("driver.mm"))
        .arg(temporary.path().join("model.mm"))
        .arg(format!("-I{}", temporary.path().display()))
        .arg(&staged.archive)
        .args(&generated.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("compile Metal bitwise driver");
    assert!(
        build.status.success(),
        "Metal bitwise host compile failed: {}\n{}",
        String::from_utf8_lossy(&build.stderr),
        generated.mm_source
    );
    Command::new(binary)
        .output()
        .expect("run Metal bitwise driver")
}

#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn bitwise_exact_signed_width_gpu_matches_reference() {
    use chelis_types::{BitwiseKind, bitwise_scalar, scalar_from_i64};
    let kinds = [
        BitwiseKind::And,
        BitwiseKind::Or,
        BitwiseKind::Xor,
        BitwiseKind::ShiftLeft,
        BitwiseKind::ShiftRight,
    ];
    for (prim, width, minimum) in [
        (Prim::Int8, 8, i8::MIN as i64),
        (Prim::Int16, 16, i16::MIN as i64),
        (Prim::Int32, 32, i32::MIN as i64),
        (Prim::Int64, 64, i64::MIN),
    ] {
        let lhs = [-1, 1, minimum, 8];
        let rhs = [1, width - 1, width, width + 1];
        let run = run_bitwise_gpu_case(prim, &kinds, &rhs);
        assert!(
            run.status.success(),
            "{}: {}",
            prim.name(),
            String::from_utf8_lossy(&run.stderr)
        );
        let actual = String::from_utf8(run.stdout).expect("utf8 output");
        for (root, kind) in kinds.iter().enumerate() {
            let expected = lhs
                .into_iter()
                .zip(rhs)
                .map(|(a, b)| {
                    bitwise_scalar(
                        *kind,
                        scalar_from_i64("bitwise", prim, a).unwrap(),
                        scalar_from_i64("bitwise", prim, b).unwrap(),
                    )
                    .unwrap()
                    .as_i64_exact()
                    .unwrap()
                    .to_string()
                })
                .collect::<Vec<_>>()
                .join(" ");
            assert!(
                actual
                    .lines()
                    .any(|line| line == format!("OUT {root} {expected}")),
                "{kind:?} {prim:?}: {actual}"
            );
        }
    }
    let empty = run_bitwise_gpu_case(Prim::Int8, &kinds, &[]);
    assert!(
        empty.status.success(),
        "empty bitwise tensors: {}",
        String::from_utf8_lossy(&empty.stderr)
    );
    let output = String::from_utf8(empty.stdout).expect("utf8 empty output");
    assert_eq!(
        output.lines().collect::<Vec<_>>(),
        (0..kinds.len())
            .map(|root| format!("OUT {root}"))
            .collect::<Vec<_>>()
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn bitwise_first_negative_shift_gpu_reports_exact_count() {
    for prim in [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64] {
        let run = run_bitwise_gpu_case(
            prim,
            &[chelis_types::BitwiseKind::ShiftRight],
            &[-3, -7, 1, 2],
        );
        assert!(!run.status.success(), "negative count must trap");
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            stderr.contains("shift amount must be non-negative, got -3"),
            "{prim:?}: {stderr}"
        );
    }
}

// f32 tolerance for Metal vs evaluator agreement. MSL's default
// transcendentals are fast-math; widen vs HIP's tolerance for safety.
// Tighten per-test if a kernel doesn't actually need the slack.
const ABS_TOL: f32 = 1e-4;
const REL_TOL: f32 = 1e-4;

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

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn write_temp_file(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write temp file");
    path
}

fn metal_runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(chelis_backend_metal::runtime_dir())
}

fn require_clangxx() {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "clang++", "--version"])
        .output();
    assert!(
        output.is_ok_and(|o| o.status.success()),
        "xcrun clang++ is required for the Metal manual gate"
    );
}

fn c_shape(shape: &[usize]) -> (usize, Vec<usize>) {
    if shape.is_empty() {
        (0, vec![])
    } else {
        (shape.len(), shape.to_vec())
    }
}

fn build_driver_mm(func_name: &str, input_labels: &[String], inputs: &[TestInput]) -> String {
    let mut body = Vec::new();

    if input_labels.is_empty() {
        body.push("    chelis_tensor **inputs = NULL;".to_string());
    } else {
        body.push(format!(
            "    chelis_tensor *input_storage[{}] = {{0}};",
            input_labels.len(),
        ));
        body.push("    chelis_tensor **inputs = input_storage;".to_string());
        for (slot, label) in input_labels.iter().enumerate() {
            let input = inputs
                .iter()
                .find(|candidate| &candidate.name == label)
                .unwrap_or_else(|| panic!("missing test input '{label}'"));
            let (ndim, c_dims) = c_shape(&input.shape);
            if ndim == 0 {
                body.push(format!(
                    "    input_storage[{slot}] = chelis_alloc(0, NULL, CHELIS_DTYPE_F32);"
                ));
            } else {
                let dims = c_dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                body.push(format!("    int64_t shape_{slot}[{ndim}] = {{ {dims} }};"));
                body.push(format!(
                    "    input_storage[{slot}] = chelis_alloc({ndim}, shape_{slot}, CHELIS_DTYPE_F32);"
                ));
            }
            body.push(format!(
                "    chelis_tensor_write *input_guard_{slot} = chelis_tensor_begin_write(input_storage[{slot}]);"
            ));
            body.push(format!(
                "    chelis_write_view input_view_{slot} = chelis_tensor_write_view(input_guard_{slot});"
            ));
            for (idx, value) in input.data.iter().enumerate() {
                // Sibling of #250/#251/#252: exact f32 bit pattern via
                // `chelis_f32_from_bits` (from the included
                // `chelis_runtime.h`), not a lossy `{:.8}f` decimal.
                let bits = value.to_bits();
                body.push(format!(
                    "    ((float *)input_view_{slot}.data)[{idx}] = chelis_f32_from_bits(0x{bits:08x}u);"
                ));
            }
            body.push(format!("    chelis_tensor_end_write(input_guard_{slot});"));
        }
    }

    body.push("    chelis_tensor *outputs[1] = {0};".to_string());
    body.push(format!(
        "    {func_name}(inputs, {}, outputs, 1);",
        input_labels.len()
    ));
    body.push(
        "    if (outputs[0] == NULL) { fprintf(stderr, \"output 0 is NULL\\n\"); return 2; }"
            .to_string(),
    );
    body.push(
        "    chelis_read_view output_view = chelis_tensor_read_view(outputs[0]);".to_string(),
    );
    body.push("    for (int i = 0; i < chelis_tensor_numel(outputs[0]); i++) {".to_string());
    body.push("        if (i > 0) printf(\" \");".to_string());
    body.push("        printf(\"%.6f\", ((const float *)output_view.data)[i]);".to_string());
    body.push("    }".to_string());
    body.push("    printf(\"\\n\");".to_string());
    body.push("    chelis_tensor_release(outputs[0]);".to_string());
    for slot in 0..input_labels.len() {
        body.push(format!("    chelis_tensor_release(input_storage[{slot}]);"));
    }

    format!(
        r#"#import <Foundation/Foundation.h>
#include "chelis_runtime.h"
#include <stdio.h>

extern "C" void {func_name}(chelis_tensor **inputs, int n_in,
                            chelis_tensor **outputs, int n_out);

int main(void) {{
    @autoreleasepool {{
{body}
    }}
    return 0;
}}
"#,
        body = body.join("\n")
    )
}

/// Build, link, and run the emitted .mm against the supplied inputs.
/// Returns the float values printed by the driver (parsed from stdout).
fn compile_and_run_single_output(dag: &Dag, func_name: &str, inputs: &[TestInput]) -> Vec<f32> {
    require_clangxx();
    let result = codegen_metal(dag, func_name);
    assert_eq!(
        result.output_labels.len(),
        1,
        "M6 manual harness currently expects a single output"
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    let metal_rt = metal_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_metal_runtime.h",
        &fs::read_to_string(metal_rt.join("chelis_metal_runtime.h")).expect("metal runtime header"),
    );
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.mm", &result.mm_source);
    write_temp_file(
        tmp.path(),
        "driver.mm",
        &build_driver_mm(func_name, &result.input_labels, inputs),
    );

    let bin_path = tmp.path().join("metal_correctness_bin");
    let mut compile_cmd = Command::new("xcrun");
    compile_cmd.args(["-sdk", "macosx", "clang++"]);
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("driver.mm"));
    compile_cmd.arg(tmp.path().join("model.mm"));
    compile_cmd.arg(format!("-I{}", tmp.path().display()));
    compile_cmd.arg(&staged.archive);
    compile_cmd.args(&result.link_flags);
    compile_cmd.arg("-o");
    compile_cmd.arg(&bin_path);
    let compile = compile_cmd.output().expect("run clang++");
    assert!(
        compile.status.success(),
        "clang++ failed:\nstderr: {}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.mm_source
    );

    let run = Command::new(&bin_path).output().expect("run metal binary");
    assert!(
        run.status.success(),
        "Metal binary failed:\nstdout: {}\nstderr: {}",
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

/// Run the IR evaluator on the same DAG with the same inputs and return
/// the flattened f32 result for the first DAG root.
fn evaluator_single_output(dag: &Dag, inputs: &[TestInput]) -> Vec<f32> {
    let env: UnordMap<String, TensorValue> = inputs
        .iter()
        .map(|i| (i.name.clone(), i.evaluator_value()))
        .collect();
    let roots = dag.roots();
    assert_eq!(
        roots.len(),
        1,
        "expected exactly one DAG root for these tests"
    );
    let outputs = eval_tensor_roots_with_strict(dag, roots, |name| env.get(name).cloned())
        .expect("evaluator failure on Metal correctness DAG");
    let value = outputs
        .get(&roots[0])
        .unwrap_or_else(|| panic!("evaluator did not produce a value for root {:?}", roots[0]));
    value.to_f64_lossy_vec().iter().map(|&v| v as f32).collect()
}

fn assert_close(actual: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch. Actual={actual:?} expected={expected:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs_err = (a - e).abs();
        let scale = e.abs().max(1.0);
        let rel_err = abs_err / scale;
        if abs_err > abs_tol && rel_err > rel_tol {
            panic!(
                "{label}: index {i} mismatch. Actual={a} expected={e} \
                 abs_err={abs_err} rel_err={rel_err} (tol abs={abs_tol} rel={rel_tol})"
            );
        }
    }
}

// ===========================================================================
// M6 oracle tests (#[ignore])
// ===========================================================================

#[test]
#[ignore]
fn m6_elementwise_add_matches_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let s = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(8), None);
    dag.add_root(s);

    let inputs = vec![
        TestInput::new("a", &[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        TestInput::new("b", &[8], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
    ];
    let actual = compile_and_run_single_output(&dag, "add_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "elementwise add");
}

#[test]
#[ignore]
fn m6_elementwise_mul_matches_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let s = dag.add_node(decl, RiscOp::Mul, vec![a, b], vec_f32(8), None);
    dag.add_root(s);

    let inputs = vec![
        TestInput::new("a", &[8], &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]),
        TestInput::new("b", &[8], &[2.0, 2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5]),
    ];
    let actual = compile_and_run_single_output(&dag, "mul_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "elementwise mul");
}

#[test]
#[ignore]
fn m6_unary_neg_matches_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let n = dag.add_node(decl, RiscOp::Neg, vec![a], vec_f32(4), None);
    dag.add_root(n);

    let inputs = vec![TestInput::new("a", &[4], &[1.5, -2.5, 0.0, 3.25])];
    let actual = compile_and_run_single_output(&dag, "neg_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "unary neg");
}

#[test]
#[ignore]
fn m6_unary_sqrt_matches_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let s = dag.add_node(decl, RiscOp::Sqrt, vec![a], vec_f32(4), None);
    dag.add_root(s);

    let inputs = vec![TestInput::new("a", &[4], &[1.0, 4.0, 9.0, 16.0])];
    let actual = compile_and_run_single_output(&dag, "sqrt_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "unary sqrt");
}

#[test]
#[ignore]
fn m6_chained_elementwise_matches_evaluator() {
    // sqrt(add(mul(a, b), c)) — three kernels, single output; the
    // transcendentals are fenced on device lanes (chelis#2957).
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let c = dag.add_node(
        decl,
        RiscOp::Load { name: "c".into() },
        vec![],
        vec_f32(8),
        None,
    );
    let m = dag.add_node(decl, RiscOp::Mul, vec![a, b], vec_f32(8), None);
    let s = dag.add_node(decl, RiscOp::Add, vec![m, c], vec_f32(8), None);
    let e = dag.add_node(decl, RiscOp::Sqrt, vec![s], vec_f32(8), None);
    dag.add_root(e);

    let inputs = vec![
        TestInput::new("a", &[8], &[0.1; 8]),
        TestInput::new("b", &[8], &[0.2; 8]),
        TestInput::new("c", &[8], &[0.3; 8]),
    ];
    let actual = compile_and_run_single_output(&dag, "chain_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, 1e-3, 1e-3, "chain sqrt(add(mul, c))");
}

#[test]
#[ignore]
fn m6_full_axis_sum_reduction_matches_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(64),
        None,
    );
    let s = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(s);

    let mut data = Vec::with_capacity(64);
    for i in 0..64 {
        data.push((i as f32) * 0.5 - 8.0);
    }
    let inputs = vec![TestInput::new("a", &[64], &data)];
    let actual = compile_and_run_single_output(&dag, "sum_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "full-axis sum");
}

#[test]
#[ignore]
fn m6_full_axis_sum_reduction_non_power_of_two_matches_evaluator() {
    // Regression test for the red-team C1 finding: reduction kernel
    // silently miscomputed when tg_size was not a power of two because
    // the tree-reduction halving loop dropped upper-half entries.
    // Exercise several non-power-of-2 sizes that previously failed:
    // n=50 returned 32, n=100 returned 64, n=200 returned 128.
    for &n in &[33usize, 50, 100, 200, 333, 1000] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            vec_f32(n),
            None,
        );
        let s = dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a],
            TensorType::scalar_f32(),
            None,
        );
        dag.add_root(s);

        // All-ones input → expected sum = n. Easy to verify by inspection.
        let data = vec![1.0f32; n];
        let inputs = vec![TestInput::new("a", &[n], &data)];
        let actual = compile_and_run_single_output(&dag, &format!("sum_{n}"), &inputs);
        let expected = evaluator_single_output(&dag, &inputs);
        assert_close(
            &actual,
            &expected,
            ABS_TOL,
            REL_TOL,
            &format!("non-power-of-2 sum n={n}"),
        );
    }
}

#[test]
#[ignore]
fn m6_full_axis_max_reduction_matches_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(64),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::MaxReduce { axis: 0 },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(m);

    let mut data = Vec::with_capacity(64);
    for i in 0..64 {
        data.push(((i as f32) * 0.7).sin());
    }
    let inputs = vec![TestInput::new("a", &[64], &data)];
    let actual = compile_and_run_single_output(&dag, "max_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "full-axis max");
}

#[test]
#[ignore]
fn m6_full_axis_min_reduction_matches_evaluator() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(64),
        None,
    );
    let m = dag.add_node(
        decl,
        RiscOp::MinReduce { axis: 0 },
        vec![a],
        TensorType::scalar_f32(),
        None,
    );
    dag.add_root(m);

    let mut data = Vec::with_capacity(64);
    for i in 0..64 {
        data.push(((i as f32) * 0.7).cos());
    }
    let inputs = vec![TestInput::new("a", &[64], &data)];
    let actual = compile_and_run_single_output(&dag, "min_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "full-axis min");
}

#[test]
#[ignore]
fn m6_tiled_matmul_matches_evaluator() {
    fn mat_f32(r: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }
    fn tensor3_f32(a: usize, b: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(a), DimInfo::Lit(b), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    // 32 x 16 @ 16 x 16 -> 32 x 16 — exercises tile alignment exactly.
    let m = 32usize;
    let k = 16usize;
    let n = 16usize;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f32(m, k),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f32(k, n),
        None,
    );
    let ea = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 2,
            size: chelis_ir::dag::RtDim::Lit(n),
        },
        vec![a],
        tensor3_f32(m, k, n),
        None,
    );
    let eb = dag.add_node(
        decl,
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::Lit(m),
        },
        vec![b],
        tensor3_f32(m, k, n),
        None,
    );
    let mul = dag.add_node(decl, RiscOp::Mul, vec![ea, eb], tensor3_f32(m, k, n), None);
    let sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 1,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![mul],
        mat_f32(m, n),
        None,
    );
    dag.add_root(sum);

    let mut a_data = Vec::with_capacity(m * k);
    for i in 0..m * k {
        a_data.push(((i as f32) % 7.0) * 0.1);
    }
    let mut b_data = Vec::with_capacity(k * n);
    for i in 0..k * n {
        b_data.push(((i as f32) % 5.0) * 0.2);
    }
    let inputs = vec![
        TestInput::new("a", &[m, k], &a_data),
        TestInput::new("b", &[k, n], &b_data),
    ];
    let actual = compile_and_run_single_output(&dag, "matmul_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    // Matmul accumulates k=16 multiplies; allow modest absolute tolerance.
    assert_close(&actual, &expected, 1e-3, 1e-3, "tiled matmul 32x16x16");
}

// ===========================================================================
// S4.3 — span-attributed program compile-success on Mac (manual gate).
//
// The S4.3 oracle has two halves: structural grep (covered in default-CI
// `tests/span_comments.rs`) and compile-success via `xcrun -sdk macosx
// clang++` (covered here, as `#[ignore]` per the existing M6 manual-gate
// pattern). This test complements the M3 macOS-smoke step, which today
// only exercises a Surf input (metadata-lossy) — span coverage on Mac
// would otherwise fall entirely to S5's `--deep` work.
//
// Manual-gate command (run on a Mac with Xcode CLI tools installed):
//   cargo test -p chelis-backend-metal --test gpu_correctness -- \
//     --ignored --test-threads=1
//
// Asserts (a) `xcrun -sdk macosx clang++` exits 0 against the generated
// `.mm` (load-bearing compile-success), AND (b) the emitted `.mm`
// contains span comments host-side AND inside the embedded MSL kernel
// raw-string literals.
// ===========================================================================

#[test]
#[ignore]
fn m6_span_attributed_program_compiles_and_matches_evaluator() {
    // Hand-craft a span-attributed IR with a mix of (span_id only,
    // merged_spans only, both, neither) — same shape as the structural
    // S4 oracle, but with shapes/ops that all hit the M2/M4 supported
    // surface so compile-success is meaningful.
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(8),
        Some("op.load_a".into()),
    );
    let neg = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![a],
        vec_f32(8),
        Some("op.neg".into()),
    );
    {
        let node = dag.node_mut(neg).unwrap();
        node.merged_spans = vec!["op.merged_b".into(), "op.merged_a".into()];
    }
    let exp = dag.add_node(decl, RiscOp::Abs, vec![neg], vec_f32(8), None);
    {
        // merged_spans only (no canonical) — the defensive case the
        // emitter must still handle correctly.
        let node = dag.node_mut(exp).unwrap();
        node.merged_spans = vec!["op.exp_merged".into()];
    }
    dag.add_root(exp);

    let inputs = vec![TestInput::new(
        "a",
        &[8],
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )];

    // The compile path is the load-bearing assertion: xcrun -sdk macosx
    // clang++ must accept the generated .mm with embedded span
    // comments. compile_and_run_single_output asserts this internally.
    let actual = compile_and_run_single_output(&dag, "span_test", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(
        &actual,
        &expected,
        1e-3,
        1e-3,
        "span-attributed abs(neg(a))",
    );

    // Structural assertion: regenerate the source and verify the spans
    // landed in both host-side and kernel-side. This catches a
    // regression where span emission landed in only one side of the
    // emitter.
    let result = codegen_metal(&dag, "span_test_2");
    let src = &result.mm_source;
    for span in &[
        "op.load_a",
        "op.neg",
        "op.merged_a",
        "op.merged_b",
        "op.exp_merged",
    ] {
        assert!(
            src.contains(&format!("// span: {span}")),
            "missing `// span: {span}` in generated .mm:\n{src}"
        );
    }
    // The Neg kernel must embed the canonical+merged block inside its
    // MSL raw-string literal.
    let neg_kernel_marker = format!("static NSString *const pso_{}_src = @R\"MSL(", neg.0);
    let kernel_block_start = src
        .find(&neg_kernel_marker)
        .expect("Neg kernel raw-string marker missing");
    let kernel_block_end = src[kernel_block_start..]
        .find(")MSL\"")
        .expect("Neg kernel raw-string terminator missing");
    let kernel_block = &src[kernel_block_start..kernel_block_start + kernel_block_end];
    assert!(
        kernel_block.contains("// span: op.neg"),
        "Neg kernel string must embed `// span: op.neg` inside the MSL raw-string literal:\n{kernel_block}"
    );
    assert!(
        kernel_block.contains("// span: op.merged_a")
            && kernel_block.contains("// span: op.merged_b"),
        "Neg kernel string must embed both merged spans:\n{kernel_block}"
    );
}

/// WS-2 macOS compile-and-run gate: Const-rooted f16 and bf16 DAGs must
/// compile under `xcrun clang++ -fobjc-arc` (the .mm has no `half` /
/// `bfloat` host type in scope) and the runtime buffer must hold the
/// pinned IEEE-754 bit pattern. `cfg(target_os = "macos")` rather than
/// just `#[ignore]` so this never silently compiles on Linux runners.
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn metal_const_f16_bf16_compiles_under_objc_arc_and_produces_exact_bits() {
    use chelis_ir::dag::RiscOp;

    fn vec_prec(n: usize, prec: Prim) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: prec,
        }
    }

    fn build_driver_for_uint16_output(func_name: &str, n: usize) -> String {
        format!(
            r#"#import <Foundation/Foundation.h>
#include "chelis_runtime.h"
#include <stdio.h>
#include <stdint.h>

extern "C" void {func_name}(chelis_tensor **inputs, int n_in,
                            chelis_tensor **outputs, int n_out);

int main(void) {{
    @autoreleasepool {{
        chelis_tensor *outputs[1] = {{0}};
        {func_name}(NULL, 0, outputs, 1);
        if (outputs[0] == NULL) {{ fprintf(stderr, "output 0 is NULL\n"); return 2; }}
        chelis_read_view output_view = chelis_tensor_read_view(outputs[0]);
        const uint16_t *bits = (const uint16_t*)output_view.data;
        for (int i = 0; i < {n}; i++) {{
            if (i > 0) printf(" ");
            printf("0x%04X", bits[i]);
        }}
        printf("\n");
        chelis_tensor_release(outputs[0]);
    }}
    return 0;
}}
"#
        )
    }

    fn compile_and_run_uint16_bits(dag: &Dag, func_name: &str, n: usize) -> Vec<u16> {
        require_clangxx();
        let result = codegen_metal(dag, func_name);

        let tmp = tempfile::tempdir().expect("tempdir");
        let metal_rt = metal_runtime_src_dir();
        write_temp_file(
            tmp.path(),
            "chelis_metal_runtime.h",
            &fs::read_to_string(metal_rt.join("chelis_metal_runtime.h"))
                .expect("metal runtime header"),
        );
        let staged = chelis_runtime_bundle::stage(tmp.path())
            .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
        write_temp_file(tmp.path(), "model.mm", &result.mm_source);
        write_temp_file(
            tmp.path(),
            "driver.mm",
            &build_driver_for_uint16_output(func_name, n),
        );

        let bin_path = tmp.path().join("metal_const_bits_bin");
        let mut compile_cmd = Command::new("xcrun");
        compile_cmd.args(["-sdk", "macosx", "clang++"]);
        compile_cmd.arg("-O2");
        compile_cmd.args(&result.compile_flags);
        compile_cmd.arg(tmp.path().join("driver.mm"));
        compile_cmd.arg(tmp.path().join("model.mm"));
        compile_cmd.arg(format!("-I{}", tmp.path().display()));
        compile_cmd.arg(&staged.archive);
        compile_cmd.args(&result.link_flags);
        compile_cmd.arg("-o");
        compile_cmd.arg(&bin_path);
        let compile = compile_cmd.output().expect("run clang++");
        assert!(
            compile.status.success(),
            "clang++ failed (Const-rooted f16/bf16 must compile under -fobjc-arc):\nstderr: {}\nsource:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            result.mm_source
        );

        let run = Command::new(&bin_path).output().expect("run metal binary");
        assert!(
            run.status.success(),
            "Metal binary failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
        stdout
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("")
            .split_whitespace()
            .map(|token| {
                let stripped = token.trim_start_matches("0x");
                u16::from_str_radix(stripped, 16).expect("parse hex u16")
            })
            .collect()
    }

    // Pinned bit-pattern sweep per WS-2 plan. The same 8 (prec, value, bits)
    // tuples that the Linux structural test asserts, but here verified
    // end-to-end by reading the runtime buffer's raw bytes.
    let cases = [
        (Prim::F16, 2.5_f64, 0x4100_u16, "k_f16_2p5"),
        (Prim::Bf16, 2.5, 0x4020, "k_bf16_2p5"),
        (Prim::F16, 1.5, 0x3E00, "k_f16_1p5"),
        (Prim::Bf16, 1.5, 0x3FC0, "k_bf16_1p5"),
        (Prim::F16, -1.0, 0xBC00, "k_f16_neg1"),
        (Prim::Bf16, -1.0, 0xBF80, "k_bf16_neg1"),
        (Prim::F16, 0.0, 0x0000, "k_f16_zero"),
        (Prim::Bf16, 0.0, 0x0000, "k_bf16_zero"),
    ];

    const N: usize = 4;
    for (prec, value, expected_bits, func_name) in cases {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let c = dag.add_node(
            decl,
            RiscOp::synth_const(prec, value),
            vec![],
            vec_prec(N, prec),
            None,
        );
        let stored = dag.add_node(
            decl,
            RiscOp::Store { name: "out".into() },
            vec![c],
            vec_prec(N, prec),
            None,
        );
        dag.add_root(stored);

        let actual = compile_and_run_uint16_bits(&dag, func_name, N);
        assert_eq!(
            actual.len(),
            N,
            "{prec:?}({value}): expected {N} elements, got {}",
            actual.len()
        );
        for (i, &bits) in actual.iter().enumerate() {
            assert_eq!(
                bits, expected_bits,
                "{prec:?}({value}) element {i}: got 0x{bits:04X}, expected 0x{expected_bits:04X}"
            );
        }
    }
}

// ===========================================================================
// WS-8A: pad / shrink GPU == evaluator (manual Mac gate). These mirror the
// HIP `g16_*` oracle cases. Each compiles the emitted .mm against
// `-framework Metal -framework Foundation`, runs it on the Metal device, and
// asserts agreement with the `chelis-ir` evaluator within the Metal f32
// tolerance. `#[ignore]` because they require an Apple Silicon Mac with a
// usable Metal device (see `spec/08-backends.md` §4.6).
// ===========================================================================

fn mat_f32(r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: Prim::F32,
    }
}

#[test]
#[ignore]
fn m6_pad_1d_zero_fill_matches_evaluator() {
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
    let inputs = vec![TestInput::new("x", &[4], &[1.0, 2.0, 3.0, 4.0])];
    let actual = compile_and_run_single_output(&dag, "pad_1d", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "pad 1d zero fill");
}

#[test]
#[ignore]
fn m6_pad_1d_nonzero_fill_matches_evaluator() {
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
    let inputs = vec![TestInput::new("x", &[3], &[10.0, 20.0, 30.0])];
    let actual = compile_and_run_single_output(&dag, "pad_1d_nonzero", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "pad 1d nonzero fill");
}

#[test]
#[ignore]
fn m6_pad_2d_asymmetric_matches_evaluator() {
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
    let inputs = vec![TestInput::new(
        "x",
        &[2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )];
    let actual = compile_and_run_single_output(&dag, "pad_2d", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "pad 2d asymmetric");
}

#[test]
#[ignore]
fn m6_shrink_1d_matches_evaluator() {
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
    let inputs = vec![TestInput::new("x", &[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])];
    let actual = compile_and_run_single_output(&dag, "shrink_1d", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "shrink 1d");
}

#[test]
#[ignore]
fn m6_shrink_2d_matches_evaluator() {
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
    let inputs = vec![TestInput::new(
        "x",
        &[3, 4],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )];
    let actual = compile_and_run_single_output(&dag, "shrink_2d", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(&actual, &expected, ABS_TOL, REL_TOL, "shrink 2d");
}

#[test]
#[ignore]
fn m6_pad_then_shrink_roundtrip_matches_evaluator() {
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
    let inputs = vec![TestInput::new("x", &[4], &[3.5, -1.0, 2.25, 8.0])];
    let actual = compile_and_run_single_output(&dag, "pad_shrink_roundtrip", &inputs);
    let expected = evaluator_single_output(&dag, &inputs);
    assert_close(
        &actual,
        &expected,
        ABS_TOL,
        REL_TOL,
        "pad then shrink roundtrip",
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
//     cargo test -p chelis-backend-metal --test gpu_correctness count_ -- --ignored --test-threads=1
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

/// Driver for one `Bool8` input and one int64 output. `bytes` are written
/// verbatim into the `CHELIS_DTYPE_BOOL` allocation (one byte per element),
/// so a test can also plant a byte outside {0, 1}.
fn build_count_driver_mm(func_name: &str, shape: &[usize], bytes: &[u8]) -> String {
    let mut body = Vec::new();
    let (ndim, dims) = c_shape(shape);
    if ndim == 0 {
        body.push(
            "    chelis_tensor *input_storage[1] = { chelis_alloc(0, NULL, CHELIS_DTYPE_BOOL) };"
                .to_string(),
        );
    } else {
        let dims = dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        body.push(format!("    int64_t shape_0[{ndim}] = {{ {dims} }};"));
        body.push(
            "    chelis_tensor *input_storage[1] = { chelis_alloc(1, shape_0, CHELIS_DTYPE_BOOL) };"
                .replace("chelis_alloc(1,", &format!("chelis_alloc({ndim},")),
        );
    }
    body.push(
        "    chelis_tensor_write *guard = chelis_tensor_begin_write(input_storage[0]);".to_string(),
    );
    body.push("    chelis_write_view view = chelis_tensor_write_view(guard);".to_string());
    for (idx, byte) in bytes.iter().enumerate() {
        body.push(format!("    ((uint8_t *)view.data)[{idx}] = {byte};"));
    }
    body.push("    chelis_tensor_end_write(guard);".to_string());
    body.push("    chelis_tensor *outputs[1] = {0};".to_string());
    body.push(format!("    {func_name}(input_storage, 1, outputs, 1);"));
    body.push(
        "    if (outputs[0] == NULL) { fprintf(stderr, \"output 0 is NULL\\n\"); return 2; }"
            .to_string(),
    );
    body.push(
        "    chelis_read_view output_view = chelis_tensor_read_view(outputs[0]);".to_string(),
    );
    body.push("    for (int i = 0; i < chelis_tensor_numel(outputs[0]); i++) {".to_string());
    body.push("        if (i > 0) printf(\" \");".to_string());
    body.push(
        "        printf(\"%lld\", (long long)((const int64_t *)output_view.data)[i]);".to_string(),
    );
    body.push("    }".to_string());
    body.push("    printf(\"\\n\");".to_string());
    body.push("    chelis_tensor_release(outputs[0]);".to_string());
    body.push("    chelis_tensor_release(input_storage[0]);".to_string());
    format!(
        r#"#import <Foundation/Foundation.h>
#include "chelis_runtime.h"
#include <stdint.h>
#include <stdio.h>

extern "C" void {func_name}(chelis_tensor **inputs, int n_in,
                            chelis_tensor **outputs, int n_out);

int main(void) {{
    @autoreleasepool {{
{body}
    }}
    return 0;
}}
"#,
        body = body.join("\n")
    )
}

/// Build, link, and run a Count DAG over raw `Bool8` bytes. `Ok` carries
/// the int64 output; `Err` carries the failed binary's stderr so a test
/// can assert on the typed trap text.
fn compile_and_run_count(
    dag: &Dag,
    func_name: &str,
    shape: &[usize],
    bytes: &[u8],
) -> Result<Vec<i64>, String> {
    require_clangxx();
    let result = codegen_metal(dag, func_name);
    assert_eq!(result.output_labels.len(), 1);

    let tmp = tempfile::tempdir().expect("tempdir");
    let metal_rt = metal_runtime_src_dir();
    write_temp_file(
        tmp.path(),
        "chelis_metal_runtime.h",
        &fs::read_to_string(metal_rt.join("chelis_metal_runtime.h")).expect("metal runtime header"),
    );
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.mm", &result.mm_source);
    write_temp_file(
        tmp.path(),
        "driver.mm",
        &build_count_driver_mm(func_name, shape, bytes),
    );

    let bin_path = tmp.path().join("metal_count_bin");
    let mut compile_cmd = Command::new("xcrun");
    compile_cmd.args(["-sdk", "macosx", "clang++"]);
    compile_cmd.arg("-O2");
    compile_cmd.args(&result.compile_flags);
    compile_cmd.arg(tmp.path().join("driver.mm"));
    compile_cmd.arg(tmp.path().join("model.mm"));
    compile_cmd.arg(format!("-I{}", tmp.path().display()));
    compile_cmd.arg(&staged.archive);
    compile_cmd.args(&result.link_flags);
    compile_cmd.arg("-o");
    compile_cmd.arg(&bin_path);
    let compile = compile_cmd.output().expect("run clang++");
    assert!(
        compile.status.success(),
        "clang++ failed:\nstderr: {}\nsource:\n{}",
        String::from_utf8_lossy(&compile.stderr),
        result.mm_source
    );

    let run = Command::new(&bin_path).output().expect("run metal binary");
    if !run.status.success() {
        return Err(format!(
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        ));
    }
    let stdout = String::from_utf8(run.stdout).expect("utf8 stdout");
    Ok(stdout
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .split_whitespace()
        .map(|token| token.parse::<i64>().expect("parse output i64"))
        .collect())
}

fn evaluator_count(dag: &Dag, shape: &[usize], mask: &[bool]) -> Vec<i64> {
    let value = TensorValue::from_vec(
        shape.to_vec(),
        mask.iter().map(|bit| f64::from(u8::from(*bit))).collect(),
    );
    let roots = dag.roots();
    let outputs =
        eval_tensor_roots_with_strict(dag, roots, |name| (name == "mask").then(|| value.clone()))
            .expect("evaluator failure on Metal Count DAG");
    outputs[&roots[0]]
        .to_f64_lossy_vec()
        .iter()
        .map(|&v| v as i64)
        .collect()
}

fn assert_count_gpu_matches_eval_exactly(
    dag: &Dag,
    func_name: &str,
    shape: &[usize],
    mask: &[bool],
) {
    let bytes = mask.iter().map(|bit| u8::from(*bit)).collect::<Vec<_>>();
    let actual = compile_and_run_count(dag, func_name, shape, &bytes)
        .unwrap_or_else(|failure| panic!("{func_name}: Metal binary failed:\n{failure}"));
    let expected = evaluator_count(dag, shape, mask);
    assert_eq!(
        actual, expected,
        "{func_name}: GPU Count must equal the evaluator exactly"
    );
}

#[test]
#[ignore]
fn count_positional_multi_axis_gpu_matches_eval() {
    let dag = count_dag(bool_tensor(&[2, 3, 5]), vec![2, 0], i64_tensor(&[3]));
    assert_count_gpu_matches_eval_exactly(
        &dag,
        "count_positional_multi_axis",
        &[2, 3, 5],
        &count_mask(30),
    );
}

#[test]
#[ignore]
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
    assert_count_gpu_matches_eval_exactly(
        &dag,
        "count_named_fixed_axes",
        &[2, 4, 3],
        &count_mask(24),
    );
}

#[test]
#[ignore]
fn count_empty_selected_extent_gpu_is_zero() {
    let dag = count_dag(bool_tensor(&[2, 0, 5]), vec![1], i64_tensor(&[2, 5]));
    let actual = compile_and_run_count(&dag, "count_empty_selected_extent", &[2, 0, 5], &[])
        .unwrap_or_else(|failure| panic!("Metal binary failed:\n{failure}"));
    assert_eq!(actual, vec![0; 10]);
    assert_eq!(evaluator_count(&dag, &[2, 0, 5], &[]), vec![0; 10]);
}

#[test]
#[ignore]
fn count_odd_leaf_count_gpu_matches_eval() {
    let dag = count_dag(bool_tensor(&[3, 7]), vec![1], i64_tensor(&[3]));
    assert_count_gpu_matches_eval_exactly(&dag, "count_odd_leaf_count", &[3, 7], &count_mask(21));
}

#[test]
#[ignore]
fn count_large_leaf_count_gpu_matches_eval() {
    // 4097 leaves per output element: an odd count deeper than any
    // power-of-two split, so the explicit balanced-tree stack is exercised
    // well past its first frames.
    let dag = count_dag(bool_tensor(&[2, 4097]), vec![1], i64_tensor(&[2]));
    assert_count_gpu_matches_eval_exactly(
        &dag,
        "count_large_leaf_count",
        &[2, 4097],
        &count_mask(2 * 4097),
    );
}

#[test]
#[ignore]
fn count_input_with_a_non_bool_payload_traps_at_the_runtime_write_boundary() {
    // A `Bool8` byte outside {0, 1} is not a member of the dtype. The
    // runtime's `chelis_tensor_end_write` rejects it with a loud domain
    // trap before the entry is ever called, which is why the kernel's own
    // status-code-1 check is a backstop rather than the primary guard: no
    // runtime-produced tensor can reach the kernel with such a byte.
    let dag = count_dag(bool_tensor(&[4]), vec![0], i64_tensor(&[]));
    let failure = compile_and_run_count(&dag, "count_non_bool_payload", &[4], &[1, 0, 2, 1])
        .expect_err("a non-bool payload must trap instead of producing a count");
    assert!(
        failure.contains("Domain") && failure.contains("noncanonical byte 2"),
        "expected the runtime Bool8 domain trap in:\n{failure}"
    );
    assert!(!failure.contains("count returned"), "{failure}");
}
