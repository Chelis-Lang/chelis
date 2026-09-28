//! [05-OP-54] Native C range source: exact endpoints, dynamic counts and overflow.
mod common;
mod support;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::fs;
use std::process::{Command, Output};
fn compile_and_run(name: &str, source: &str, harness: &str) -> Output {
    let probe = common::probe_dir(&format!("issue_570_{name}"));
    let dir = probe.path();
    let staged = chelis_runtime_bundle::stage(dir)
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    fs::write(dir.join("kernel.c"), source).expect("write generated C");
    fs::write(dir.join("main.c"), harness).expect("write C harness");
    let binary = dir.join("probe");
    let mut compile = Command::new("cc");
    compile
        .arg("-O2")
        .arg("-std=c11")
        .arg("-I")
        .arg(dir)
        .arg(dir.join("kernel.c"))
        .arg(dir.join("main.c"))
        .arg(&staged.archive)
        .args(["-lm", "-lpthread", "-ldl", "-o"])
        .arg(&binary);
    if cfg!(target_arch = "x86_64") {
        compile.arg("-mavx2");
    }
    let built = compile.output().expect("invoke C compiler");
    assert!(
        built.status.success(),
        "compiled-C fixture `{name}` failed:\n{}\n{source}",
        String::from_utf8_lossy(&built.stderr)
    );
    Command::new(binary).output().expect("run compiled C")
}

fn program() -> Dag {
    program_with_activation(None)
}

fn program_with_activation(active: Option<bool>) -> Dag {
    let mut dag = Dag::new();
    let owner = dag.declare("runtime_iota");
    let scalar = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let start = dag.add_node(
        owner,
        RiscOp::Load {
            name: "start".into(),
        },
        vec![],
        scalar.clone(),
        None,
    );
    let end = dag.add_node(
        owner,
        RiscOp::Load { name: "end".into() },
        vec![],
        scalar,
        None,
    );
    let activation = active.map(|value| {
        dag.add_node(
            owner,
            RiscOp::synth_const(Prim::Bool, if value { 1.0 } else { 0.0 }),
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        )
    });
    let output = dag.add_node(
        chelis_ir::dag::Owner::new(owner, activation),
        RiscOp::Iota,
        vec![start, end],
        TensorType {
            dims: vec![DimInfo::Named("count".into(), None)],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_root(output);
    dag
}

fn harness(body: &str) -> String {
    format!(
        r#"
#include <stdint.h>
#include <stdio.h>
#include "chelis_runtime.h"
extern void runtime_iota(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {body}
}}
"#
    )
}

#[test]
fn runtime_iota_native_exact_values_and_counts() {
    let source = support::codegen(&program(), "runtime_iota")
        .unwrap()
        .c_source;
    let harness = harness(
        r#"
    int64_t starts[] = {0, -2, 7, 9, INT64_C(9007199254740993)};
    int64_t ends[] = {3, 2, 7, 3, INT64_C(9007199254740995)};
    int64_t counts[] = {3, 4, 0, 0, 2};
    for (int row = 0; row < 5; row++) {
        chelis_tensor *inputs[2] = {
            chelis_tensor_entry_borrow(0, NULL, CHELIS_DTYPE_I64, &starts[row], sizeof(int64_t)),
            chelis_tensor_entry_borrow(0, NULL, CHELIS_DTYPE_I64, &ends[row], sizeof(int64_t))
        }, *outputs[1] = {0};
        runtime_iota(inputs, 2, outputs, 1);
        if (chelis_tensor_shape(outputs[0], 0) != counts[row]) return 10;
        const int64_t *values = chelis_tensor_read_view(outputs[0]).data;
        for (int64_t index = 0; index < counts[row]; index++) {
            if (values[index] != starts[row] + index) return 11;
        }
    }
    puts("PASS"); return 0;
"#,
    );
    let result = compile_and_run("exact", &source, &harness);
    assert!(result.status.success(), "{result:?}");
}

#[test]
fn runtime_iota_native_overflow_precedes_allocation() {
    let source = support::codegen(&program(), "runtime_iota")
        .unwrap()
        .c_source;
    let harness = harness(
        r#"
    int64_t start = INT64_MIN, end = INT64_MAX;
    chelis_tensor *inputs[2] = {
        chelis_tensor_entry_borrow(0, NULL, CHELIS_DTYPE_I64, &start, sizeof(start)),
        chelis_tensor_entry_borrow(0, NULL, CHELIS_DTYPE_I64, &end, sizeof(end))
    }, *outputs[1] = {0};
    runtime_iota(inputs, 2, outputs, 1);
    return 0;
"#,
    );
    let result = compile_and_run("overflow", &source, &harness);
    assert!(!result.status.success(), "{result:?}");
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("overflow in range at i64"),
        "{result:?}"
    );
}

#[test]
fn runtime_iota_native_inactive_arm_does_not_read_overflowing_endpoints() {
    let source = support::codegen(&program_with_activation(Some(false)), "runtime_iota")
        .unwrap()
        .c_source;
    let harness = harness(
        r#"
    int64_t start = INT64_MIN, end = INT64_MAX;
    chelis_tensor *inputs[2] = {
        chelis_tensor_entry_borrow(0, NULL, CHELIS_DTYPE_I64, &start, sizeof(start)),
        chelis_tensor_entry_borrow(0, NULL, CHELIS_DTYPE_I64, &end, sizeof(end))
    }, *outputs[1] = {0};
    runtime_iota(inputs, 2, outputs, 1);
    return chelis_tensor_shape(outputs[0], 0) != 0;
"#,
    );
    let result = compile_and_run("inactive", &source, &harness);
    assert!(result.status.success(), "{result:?}");
}
