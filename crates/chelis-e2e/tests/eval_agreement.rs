//! Evaluator-agreement tests: compare eval_tensor results with C codegen+compile+run.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use chelis_ir::dag::{Dag, NodeId, RiscOp, TensorType};
use chelis_ir::eval::eval_tensor;

fn scalar_f32() -> TensorType {
    TensorType::scalar_f32()
}

fn runtime_src_dir() -> PathBuf {
    // chelis-backend-c crate contains the runtime
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("chelis-backend-c")
        .join("runtime")
}

fn gcc_available() -> bool {
    Command::new("gcc")
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
    let result = chelis_backend_c::codegen(dag, func_name);

    let tmp = tempfile::tempdir().unwrap();
    let rt_dir = runtime_src_dir();
    let h_src = std::fs::read_to_string(rt_dir.join("chelis_runtime.h")).unwrap();
    let c_rt = std::fs::read_to_string(rt_dir.join("chelis_runtime.c")).unwrap();
    write_temp_file(tmp.path(), "chelis_runtime.h", &h_src);
    write_temp_file(tmp.path(), "chelis_runtime.c", &c_rt);
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

    let mut cmd = Command::new("gcc");
    let extra = c_test_extra_flags();
    if !extra.is_empty() {
        cmd.args(&extra);
    }
    cmd.args(["-O2"]);
    cmd.args(&result.compile_flags);
    cmd.arg(tmp.path().join("main.c").to_str().unwrap());
    cmd.arg(tmp.path().join("model.c").to_str().unwrap());
    cmd.arg(tmp.path().join("chelis_runtime.c").to_str().unwrap());
    cmd.args(&result.link_flags);
    cmd.arg("-o");
    cmd.arg(bin_path.to_str().unwrap());
    let compile = cmd.output().unwrap();
    assert!(
        compile.status.success(),
        "gcc failed:\nstderr: {}\nC source:\n{}",
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
    let a = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 4.0 }, vec![], scalar_f32());
    dag.add_node(RiscOp::Add, vec![a, b], scalar_f32());

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
    let a = dag.add_node(RiscOp::Const { value: 5.0 }, vec![], scalar_f32());
    let b = dag.add_node(RiscOp::Const { value: 6.0 }, vec![], scalar_f32());
    dag.add_node(RiscOp::Mul, vec![a, b], scalar_f32());

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
    let a = dag.add_node(RiscOp::Const { value: 7.0 }, vec![], scalar_f32());
    dag.add_node(RiscOp::Neg, vec![a], scalar_f32());

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
        let x = dag.add_node(RiscOp::Const { value: -2.0 }, vec![], scalar_f32());
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32());

        let eval_result = eval_last(&dag);
        let c_result = parse_c_output(&compile_and_run(&dag, "test_relu_neg"));
        assert_close(eval_result, c_result, 1e-6, "relu(-2)");
        assert_close(eval_result, 0.0, 1e-6, "relu(-2) expected");
    }

    // relu with positive input
    {
        let mut dag = Dag::new();
        let x = dag.add_node(RiscOp::Const { value: 3.0 }, vec![], scalar_f32());
        let zero = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
        dag.add_node(RiscOp::MaxElem, vec![x, zero], scalar_f32());

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
    let a = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], scalar_f32());
    dag.add_node(RiscOp::Exp, vec![a], scalar_f32());

    let eval_result = eval_last(&dag);
    let c_result = parse_c_output(&compile_and_run(&dag, "test_exp"));
    assert_close(eval_result, c_result, 1e-6, "exp(0)");
    assert_close(eval_result, 1.0, 1e-6, "exp(0) expected");
}
