//! Evaluator-agreement tests: compare eval_tensor results with C codegen+compile+run.
//!
//! WS-1 (dtype + Metal cleanup cycle) extension: bf16 / f16 cases are
//! added below. The HIP cross-validation lives behind the
//! `hip-local-gpu` feature flag (`cargo test -p chelis-e2e --features
//! hip-local-gpu`); the default-feature Linux CI runner skips them
//! because hipcc / libhipblas are not present.
//!
//! ## Integer exactness: superseded oracle (chelis#687, chelis#729 Phase 0)
//!
//! This harness is f64-typed end to end (`eval_last -> f64`,
//! `parse_c_output -> f64`, `assert_close`) and covers float cells only
//! (f32/bf16/f16), where a tolerance is legitimate until chelis#732
//! delivers byte-identical rendering. It deliberately gains NO
//! exact-string integer lane: at this DAG level `RiscOp::Const { value:
//! f64 }` cannot even express an exact int64 above 2^53 (chelis#684), so
//! an integer lane here would test the wrong layer. The exact-integer
//! cross-lane oracle is the PR #696 driver family instead:
//! `crates/chelis-cli/tests/precision_matrix.rs` (`eval_lane_str` /
//! `c_lane_str`, verbatim strings) and
//! `crates/chelis-cli/tests/issue_680_int_exactness.rs` (`eval_int` /
//! `parse_out_binding`, exact `i64` parses). Do not add integer rows to
//! THIS file; add them there.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::eval_tensor;
use chelis_types::types::Prim;

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn scalar_ty(prec: Prim) -> TensorType {
    TensorType {
        dims: vec![],
        precision: prec,
    }
}

fn vec_ty(n: usize, prec: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: prec,
    }
}

fn runtime_src_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("chelis-runtime")
        .join("include")
}

/// Resolve the workspace `target/` directory from the running test binary
/// rather than a `CARGO_MANIFEST_DIR`-relative path, so an external
/// `CARGO_TARGET_DIR` (e.g. a concurrent agent building into
/// `target/agents/<name>`) is honored. The binary lives at
/// `<target>/<profile>/deps/<test-bin>`; strip a trailing `deps` component if
/// present, then drop the profile component to reach `<target>`. See
/// chelis#747.
fn target_dir_from_current_exe() -> PathBuf {
    let exe = std::env::current_exe().expect("could not determine current test executable");
    let mut profile_dir = exe
        .parent()
        .expect("test executable should have a parent directory");
    if profile_dir.file_name().and_then(|name| name.to_str()) == Some("deps") {
        profile_dir = profile_dir
            .parent()
            .expect("`deps` directory should have a parent");
    }
    profile_dir
        .parent()
        .map(PathBuf::from)
        .expect("profile directory should have a parent target directory")
}

fn runtime_library_path() -> PathBuf {
    let target_dir = target_dir_from_current_exe();
    for path in [
        target_dir.join("debug/libchelis_runtime.a"),
        target_dir.join("release/libchelis_runtime.a"),
    ] {
        if path.exists() {
            return path;
        }
    }
    for dir in [
        target_dir.join("debug/deps"),
        target_dir.join("release/deps"),
    ] {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
                {
                    return path;
                }
            }
        }
    }
    panic!("runtime lib not found");
}

fn gcc_available() -> bool {
    Command::new(chelis_backend_c::toolchain::c_compiler())
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn write_temp_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

fn c_test_extra_flags() -> Vec<String> {
    std::env::var("CHELIS_C_TEST_EXTRA_FLAGS")
        .ok()
        .map(|flags| {
            flags
                .split_whitespace()
                .map(|flag| flag.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

/// Build a DAG, generate C, compile with gcc, run, return stdout as string.
fn compile_and_run(dag: &Dag, func_name: &str) -> String {
    let result = chelis_backend_c::codegen(dag, func_name).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let rt_dir = runtime_src_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = std::fs::read_to_string(rt_dir.join(header)).unwrap();
        write_temp_file(tmp.path(), header, &src);
    }
    std::fs::copy(
        runtime_library_path(),
        tmp.path().join("libchelis_runtime.a"),
    )
    .unwrap();
    write_temp_file(tmp.path(), "model.c", &result.c_source);

    let main_c = format!(
        r#"
#include "chelis_runtime.h"
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main() {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    for (int i = 0; i < outputs[0]->size; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", outputs[0]->data[i]);
    }}
    printf("\n");
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    write_temp_file(tmp.path(), "main.c", &main_c);
    let bin_path = tmp.path().join("test_bin");

    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    let mut cmd = Command::new(&toolchain.compiler);
    let extra = c_test_extra_flags();
    if !extra.is_empty() {
        cmd.args(&extra);
    }
    cmd.args(["-O2"]);
    cmd.args(&toolchain.compile_flags);
    cmd.arg(tmp.path().join("main.c").to_str().unwrap());
    cmd.arg(tmp.path().join("model.c").to_str().unwrap());
    cmd.arg(format!("-L{}", tmp.path().display()));
    cmd.arg("-lchelis_runtime");
    cmd.args(&toolchain.link_flags);
    cmd.arg("-o");
    cmd.arg(bin_path.to_str().unwrap());
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "{} failed:\nstderr: {}\nC source:\n{}",
        toolchain.compiler,
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );

    let run = Command::new(bin_path.to_str().unwrap()).output().unwrap();
    assert!(
        run.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap().trim().to_string()
}

fn eval_last(dag: &Dag) -> f64 {
    let inputs = HashMap::new();
    let vals = eval_tensor(dag, &inputs).unwrap();
    let last_id = NodeId(dag.len() - 1);
    vals[&last_id].data[0]
}

fn parse_c_output(output: &str) -> f64 {
    output.trim().parse::<f64>().unwrap()
}

fn assert_close(a: f64, b: f64, tol: f64, label: &str) {
    assert!(
        (a - b).abs() < tol,
        "{label}: eval={a}, C={b}, diff={}",
        (a - b).abs()
    );
}

#[test]
fn agreement_add() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
    let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32(), None);
    dag.add_node(RiscOp::Add, vec![a, b], scalar_f32(), None);

    let eval_result = eval_last(&dag);
    let c_result = parse_c_output(&compile_and_run(&dag, "test_add"));
    assert_close(eval_result, c_result, 1e-6, "add(3,4)");
    assert_close(eval_result, 7.0, 1e-6, "add(3,4) expected");
}

#[test]
fn agreement_mul() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32(), None);
    let b = dag.add_node(RiscOp::Const { value: 6.0 }, vec![], scalar_f32(), None);
    dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32(), None);

    let eval_result = eval_last(&dag);
    let c_result = parse_c_output(&compile_and_run(&dag, "test_mul"));
    assert_close(eval_result, c_result, 1e-6, "mul(5,6)");
    assert_close(eval_result, 30.0, 1e-6, "mul(5,6) expected");
}

#[test]
fn agreement_neg() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 7.0 }, vec![], scalar_f32(), None);
    dag.add_node(RiscOp::Neg, vec![a], scalar_f32(), None);

    let eval_result = eval_last(&dag);
    let c_result = parse_c_output(&compile_and_run(&dag, "test_neg"));
    assert_close(eval_result, c_result, 1e-6, "neg(7)");
    assert_close(eval_result, -7.0, 1e-6, "neg(7) expected");
}

#[test]
fn agreement_relu() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }

    // relu(x) = max(x, 0) -- test with negative input
    {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: -2.0 }, vec![], scalar_f32(), None);
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32(), None);

        let eval_result = eval_last(&dag);
        let c_result = parse_c_output(&compile_and_run(&dag, "test_relu_neg"));
        assert_close(eval_result, c_result, 1e-6, "relu(-2)");
        assert_close(eval_result, 0.0, 1e-6, "relu(-2) expected");
    }

    // relu with positive input
    {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32(), None);
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32(), None);
        dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32(), None);

        let eval_result = eval_last(&dag);
        let c_result = parse_c_output(&compile_and_run(&dag, "test_relu_pos"));
        assert_close(eval_result, c_result, 1e-6, "relu(3)");
        assert_close(eval_result, 3.0, 1e-6, "relu(3) expected");
    }
}

#[test]
fn agreement_exp() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32(), None);
    dag.add_node(RiscOp::Exp, vec![a], scalar_f32(), None);

    let eval_result = eval_last(&dag);
    let c_result = parse_c_output(&compile_and_run(&dag, "test_exp"));
    assert_close(eval_result, c_result, 1e-6, "exp(0)");
    assert_close(eval_result, 1.0, 1e-6, "exp(0) expected");
}

/// WS-1: build a DAG whose trailing node is bf16 or f16, compile,
/// run, and read the 16-bit storage back through the runtime's
/// conversion helper. Returns the trailing element as `f64` so the
/// caller can compare against the evaluator.
fn compile_and_run_reduced(dag: &Dag, func_name: &str, is_bf16: bool) -> f64 {
    let result = chelis_backend_c::codegen(dag, func_name).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let rt_dir = runtime_src_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = std::fs::read_to_string(rt_dir.join(header)).unwrap();
        write_temp_file(tmp.path(), header, &src);
    }
    std::fs::copy(
        runtime_library_path(),
        tmp.path().join("libchelis_runtime.a"),
    )
    .unwrap();
    write_temp_file(tmp.path(), "model.c", &result.c_source);
    let convert_fn = if is_bf16 {
        "chelis_bf16_to_f32"
    } else {
        "chelis_f16_to_f32"
    };
    let main_c = format!(
        r#"
#include "chelis_runtime.h"
#include <stdint.h>
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    uint16_t *p = (uint16_t*)outputs[0]->data;
    printf("%.8f\n", {convert_fn}(p[0]));
    chelis_free(outputs[0]);
    return 0;
}}
"#
    );
    write_temp_file(tmp.path(), "main.c", &main_c);
    let bin_path = tmp.path().join("test_bin");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    let mut cmd = Command::new(&toolchain.compiler);
    let extra = c_test_extra_flags();
    if !extra.is_empty() {
        cmd.args(&extra);
    }
    cmd.args(["-O2"]);
    cmd.args(&toolchain.compile_flags);
    cmd.arg(tmp.path().join("main.c").to_str().unwrap());
    cmd.arg(tmp.path().join("model.c").to_str().unwrap());
    cmd.arg(format!("-L{}", tmp.path().display()));
    cmd.arg("-lchelis_runtime");
    cmd.args(&toolchain.link_flags);
    cmd.arg("-o");
    cmd.arg(bin_path.to_str().unwrap());
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "{} failed:\nstderr: {}\nC source:\n{}",
        toolchain.compiler,
        String::from_utf8_lossy(&compile.stderr),
        result.c_source
    );
    let run = Command::new(bin_path.to_str().unwrap()).output().unwrap();
    assert!(
        run.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8(run.stdout).unwrap();
    stdout.trim().parse::<f64>().unwrap()
}

/// WS-1: bf16 add agrees with the evaluator. Tolerance picked to
/// match the C-backend's `BF16_TOL` constant; the operand-level
/// rounding error of bf16(1.5) and bf16(2.5) is exactly zero (both
/// are representable), so the post-add bit pattern equals 4.0 in
/// either path.
#[test]
fn agreement_bf16_add() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.5 },
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 2.5 },
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    dag.add_node(RiscOp::Add, vec![a, b], scalar_ty(Prim::Bf16), None);
    let c_result = compile_and_run_reduced(&dag, "test_bf16_add", true);
    let eval_result = eval_last(&dag);
    assert_close(eval_result, c_result, 1e-2, "bf16 add(1.5, 2.5)");
}

#[test]
fn agreement_f16_add() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Const { value: 1.5 },
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    let b = dag.add_node(
        RiscOp::Const { value: 2.5 },
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    dag.add_node(RiscOp::Add, vec![a, b], scalar_ty(Prim::F16), None);
    let c_result = compile_and_run_reduced(&dag, "test_f16_add", false);
    let eval_result = eval_last(&dag);
    assert_close(eval_result, c_result, 1e-3, "f16 add(1.5, 2.5)");
}

#[test]
fn agreement_bf16_reduce_sum_matches_eval_within_tol() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // Build a fused Const + reduce_sum so the comparison stays
    // self-contained (no Load to thread inputs through the runtime).
    let n = 8;
    let mut dag = Dag::new();
    let c = dag.add_node(
        RiscOp::Const { value: 0.25 },
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    let sum = RiscOp::sum_default(0, Prim::Bf16).expect("sum constructs");
    dag.add_node(sum, vec![c], scalar_ty(Prim::F32), None);
    let result = chelis_backend_c::codegen(&dag, "test_bf16_sum_const").unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let rt_dir = runtime_src_dir();
    for header in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = std::fs::read_to_string(rt_dir.join(header)).unwrap();
        write_temp_file(tmp.path(), header, &src);
    }
    std::fs::copy(
        runtime_library_path(),
        tmp.path().join("libchelis_runtime.a"),
    )
    .unwrap();
    write_temp_file(tmp.path(), "model.c", &result.c_source);
    let main_c = r#"
#include "chelis_runtime.h"
void test_bf16_sum_const(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *outputs[1] = {0};
    test_bf16_sum_const(NULL, 0, outputs, 1);
    printf("%.8f\n", ((float*)outputs[0]->data)[0]);
    chelis_free(outputs[0]);
    return 0;
}
"#;
    write_temp_file(tmp.path(), "main.c", main_c);
    let bin_path = tmp.path().join("test_bin");
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    let mut cmd = Command::new(&toolchain.compiler);
    let extra = c_test_extra_flags();
    if !extra.is_empty() {
        cmd.args(&extra);
    }
    cmd.args(["-O2"]);
    cmd.args(&toolchain.compile_flags);
    cmd.arg(tmp.path().join("main.c").to_str().unwrap());
    cmd.arg(tmp.path().join("model.c").to_str().unwrap());
    cmd.arg(format!("-L{}", tmp.path().display()));
    cmd.arg("-lchelis_runtime");
    cmd.args(&toolchain.link_flags);
    cmd.arg("-o");
    cmd.arg(bin_path.to_str().unwrap());
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "compile failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(bin_path.to_str().unwrap()).output().unwrap();
    assert!(run.status.success());
    let c_result: f64 = String::from_utf8(run.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let eval_result = eval_last(&dag);
    assert_close(eval_result, c_result, 1e-2, "bf16 reduce_sum(0.25 x 8)");
}
