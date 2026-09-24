//! Compiled-C acceptance leg for chelis#1281.
//!
//! These tests compile and execute generated C. Source-shape assertions are
//! only negative controls for the historical f32/fmaxf/fminf implementations.

#[path = "../../../tests/support/runtime_archive.rs"]
mod runtime_archive;

mod common;
mod support;

use chelis_ir::dag::{Dag, DimInfo, ReduceWindowKind, RiscOp, TensorType};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::tier2;
use chelis_types::types::Prim;
use chelis_types::{RawTensor, StorageView, finalize_tensor};
use chelis_unord::UnordMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::OnceLock;
use support::codegen;

fn tensor(dims: &[usize], precision: Prim) -> TensorType {
    TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    }
}

fn symbolic_vector(name: &str, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision,
    }
}

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

fn target_debug_dir() -> PathBuf {
    std::env::current_exe()
        .expect("current test executable")
        .parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("test executable lives under target/debug/deps")
}

fn newest_runtime_archive(deps: &Path) -> std::io::Result<Option<PathBuf>> {
    let mut newest = None;
    let entries = match fs::read_dir(deps) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let metadata = entry.metadata()?;
            if metadata.len() == 0 {
                continue;
            }
            let modified = metadata.modified()?;
            if newest
                .as_ref()
                .is_none_or(|(current, _): &(std::time::SystemTime, PathBuf)| modified > *current)
            {
                newest = Some((modified, entry.path()));
            }
        }
    }
    Ok(newest.map(|(_, path)| path))
}

fn runtime_lib_path() -> PathBuf {
    if let Some(path) = runtime_archive::explicit() {
        return path;
    }
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let debug = target_debug_dir();
        newest_runtime_archive(&debug.join("deps"))
            .expect("scan runtime archives")
            .or_else(|| {
                // A managed build puts its observed Cargo launcher first on PATH;
                // env!("CARGO") names the real binary, which cannot build a producer.
                let status = Command::new("cargo")
                    .args(["build", "-p", "chelis-runtime", "--lib"])
                    .status()
                    .expect("build chelis-runtime");
                assert!(status.success(), "build chelis-runtime static library");
                newest_runtime_archive(&debug.join("deps")).expect("rescan runtime archives")
            })
            .expect("non-empty libchelis_runtime-*.a exists")
    })
    .clone()
}

fn compile_and_run(name: &str, source: &str, harness: &str) -> Output {
    let probe = common::probe_dir(&format!("issue_1281_{name}"));
    let dir = probe.path();
    fs::write(dir.join("kernel.c"), source).expect("write generated C");
    fs::write(dir.join("main.c"), harness).expect("write C harness");
    let binary = dir.join("probe");
    let mut compile = Command::new("cc");
    compile
        .arg("-O2")
        .arg("-std=c11")
        .arg("-I")
        .arg(runtime_include_dir())
        .arg(dir.join("kernel.c"))
        .arg(dir.join("main.c"))
        .arg(runtime_lib_path())
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

fn global_reduction_dag(precision: Prim) -> Dag {
    let mut dag = Dag::new();
    let input_ty = tensor(&[2, 4], precision);
    let value_ty = tensor(&[2], precision);
    let index_ty = tensor(&[2], Prim::Int64);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], input_ty, None);
    let max = dag.add_node(
        RiscOp::MaxReduce { axis: 1 },
        vec![x],
        value_ty.clone(),
        None,
    );
    let min = dag.add_node(RiscOp::MinReduce { axis: 1 }, vec![x], value_ty, None);
    let argmax = dag.add_node(RiscOp::Argmax { axis: 1 }, vec![x], index_ty.clone(), None);
    let argmin = dag.add_node(RiscOp::Argmin { axis: 1 }, vec![x], index_ty, None);
    for root in [max, min, argmax, argmin] {
        dag.add_root(root);
    }
    dag
}

#[test]
fn global_extrema_and_argument_reductions_preserve_f32_bits_and_first_indices() {
    let source = codegen(&global_reduction_dag(Prim::F32), "issue1281_global_f32")
        .expect("f32 reductions codegen")
        .c_source;
    assert!(
        !source.contains("fmaxf("),
        "max reduction must select a source value"
    );
    assert!(
        !source.contains("fminf("),
        "min reduction must select a source value"
    );
    let harness = r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void issue1281_global_f32(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {
    uint32_t bits[8] = {
        UINT32_C(0xffc12345), UINT32_C(0x7fc54321), UINT32_C(0x40a00000), UINT32_C(0xc0e00000),
        UINT32_C(0x80000000), UINT32_C(0x00000000), UINT32_C(0xbf800000), UINT32_C(0xc0000000)
    };
    float data[8]; memcpy(data, bits, sizeof(bits));
    int64_t shape[2] = {2, 4};
    chelis_tensor *x = chelis_tensor_entry_borrow(2, shape, CHELIS_DTYPE_F32, data, sizeof(data));
    chelis_tensor *inputs[1] = {x}, *outputs[4] = {0};
    issue1281_global_f32(inputs, 1, outputs, 4);
    uint32_t expected_max[2] = {UINT32_C(0xffc12345), UINT32_C(0x80000000)};
    uint32_t expected_min[2] = {UINT32_C(0xffc12345), UINT32_C(0xc0000000)};
    int64_t expected_argmax[2] = {0, 0}, expected_argmin[2] = {0, 3};
    uint32_t got[2];
    memcpy(got, chelis_tensor_read_view(outputs[0]).data, sizeof(got));
    if (memcmp(got, expected_max, sizeof(got))) return 10;
    memcpy(got, chelis_tensor_read_view(outputs[1]).data, sizeof(got));
    if (memcmp(got, expected_min, sizeof(got))) return 11;
    if (memcmp(chelis_tensor_read_view(outputs[2]).data, expected_argmax, sizeof(expected_argmax))) return 12;
    if (memcmp(chelis_tensor_read_view(outputs[3]).data, expected_argmin, sizeof(expected_argmin))) return 13;
    puts("PASS"); return 0;
}
"#;
    let run = compile_and_run("global_f32", &source, harness);
    assert!(
        run.status.success(),
        "f32 exact reductions failed: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn global_i64_extrema_and_argument_reductions_never_convert_through_float() {
    let source = codegen(&global_reduction_dag(Prim::Int64), "issue1281_global_i64")
        .expect("i64 reductions codegen")
        .c_source;
    let harness = r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void issue1281_global_i64(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {
    int64_t data[8] = {
        INT64_C(9007199254740993), INT64_C(9007199254740995), INT64_C(9007199254740994), INT64_C(-9),
        INT64_MIN, INT64_C(-9007199254740993), INT64_C(7), INT64_MAX
    };
    int64_t shape[2] = {2, 4};
    chelis_tensor *x = chelis_tensor_entry_borrow(2, shape, CHELIS_DTYPE_I64, data, sizeof(data));
    chelis_tensor *inputs[1] = {x}, *outputs[4] = {0};
    issue1281_global_i64(inputs, 1, outputs, 4);
    int64_t expected_max[2] = {INT64_C(9007199254740995), INT64_MAX};
    int64_t expected_min[2] = {INT64_C(-9), INT64_MIN};
    int64_t expected_argmax[2] = {1, 3}, expected_argmin[2] = {3, 0};
    if (memcmp(chelis_tensor_read_view(outputs[0]).data, expected_max, sizeof(expected_max))) return 20;
    if (memcmp(chelis_tensor_read_view(outputs[1]).data, expected_min, sizeof(expected_min))) return 21;
    if (memcmp(chelis_tensor_read_view(outputs[2]).data, expected_argmax, sizeof(expected_argmax))) return 22;
    if (memcmp(chelis_tensor_read_view(outputs[3]).data, expected_argmin, sizeof(expected_argmin))) return 23;
    puts("PASS"); return 0;
}
"#;
    let run = compile_and_run("global_i64", &source, harness);
    assert!(
        run.status.success(),
        "i64 exact reductions failed: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn global_extrema_and_argument_reductions_execute_at_every_remaining_storage_width() {
    for (tag, prim, c_type, c_dtype, data, max, min, argmax, argmin) in [
        (
            "f64",
            Prim::F64,
            "double",
            "CHELIS_DTYPE_F64",
            "-0.0,0.0,-1.0,-2.0,9.0,11.0,11.0,10.0",
            "-0.0,11.0",
            "-2.0,9.0",
            "0,1",
            "3,0",
        ),
        (
            "i8",
            Prim::Int8,
            "int8_t",
            "CHELIS_DTYPE_I8",
            "5,7,7,-4,-128,-3,12,127",
            "7,127",
            "-4,-128",
            "1,3",
            "3,0",
        ),
        (
            "i16",
            Prim::Int16,
            "int16_t",
            "CHELIS_DTYPE_I16",
            "5,30000,30000,-4,-32768,-3,12,32767",
            "30000,32767",
            "-4,-32768",
            "1,3",
            "3,0",
        ),
        (
            "i32",
            Prim::Int32,
            "int32_t",
            "CHELIS_DTYPE_I32",
            "5,2000000000,2000000000,-4,INT32_MIN,-3,12,INT32_MAX",
            "2000000000,INT32_MAX",
            "-4,INT32_MIN",
            "1,3",
            "3,0",
        ),
    ] {
        let function = format!("issue1281_global_{tag}");
        let source = codegen(&global_reduction_dag(prim), &function)
            .unwrap_or_else(|error| panic!("{tag} reduction codegen: {error}"))
            .c_source;
        let harness = format!(
            r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} data[8] = {{{data}}};
    {c_type} expected_max[2] = {{{max}}}, expected_min[2] = {{{min}}};
    int64_t expected_argmax[2] = {{{argmax}}}, expected_argmin[2] = {{{argmin}}};
    int64_t shape[2] = {{2,4}};
    chelis_tensor *x = chelis_tensor_entry_borrow(2, shape, {c_dtype}, data, sizeof(data));
    chelis_tensor *inputs[1] = {{x}}, *outputs[4] = {{0}};
    {function}(inputs, 1, outputs, 4);
    if (memcmp(chelis_tensor_read_view(outputs[0]).data, expected_max, sizeof(expected_max))) return 50;
    if (memcmp(chelis_tensor_read_view(outputs[1]).data, expected_min, sizeof(expected_min))) return 51;
    if (memcmp(chelis_tensor_read_view(outputs[2]).data, expected_argmax, sizeof(expected_argmax))) return 52;
    if (memcmp(chelis_tensor_read_view(outputs[3]).data, expected_argmin, sizeof(expected_argmin))) return 53;
    puts("PASS"); return 0;
}}
"#
        );
        let run = compile_and_run(&format!("global_{tag}"), &source, &harness);
        assert!(
            run.status.success(),
            "{tag} exact reductions failed: stdout={} stderr={}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }

    for (tag, prim, dtype, data, max, min) in [
        (
            "f16",
            Prim::F16,
            "CHELIS_DTYPE_F16",
            "0xfe01,0x7e55,0x4500,0xc700,0x8000,0x0000,0xbc00,0xc000",
            "0xfe01,0x8000",
            "0xfe01,0xc000",
        ),
        (
            "bf16",
            Prim::Bf16,
            "CHELIS_DTYPE_BF16",
            "0xffc1,0x7fe5,0x40a0,0xc0e0,0x8000,0x0000,0xbf80,0xc000",
            "0xffc1,0x8000",
            "0xffc1,0xc000",
        ),
    ] {
        let function = format!("issue1281_global_{tag}");
        let source = codegen(&global_reduction_dag(prim), &function)
            .unwrap_or_else(|error| panic!("{tag} reduction codegen: {error}"))
            .c_source;
        let harness = format!(
            r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint16_t data[8] = {{{data}}};
    uint16_t expected_max[2] = {{{max}}}, expected_min[2] = {{{min}}};
    int64_t expected_argmax[2] = {{0,0}}, expected_argmin[2] = {{0,3}};
    int64_t shape[2] = {{2,4}};
    chelis_tensor *x = chelis_tensor_entry_borrow(2, shape, {dtype}, data, sizeof(data));
    chelis_tensor *inputs[1] = {{x}}, *outputs[4] = {{0}};
    {function}(inputs, 1, outputs, 4);
    if (memcmp(chelis_tensor_read_view(outputs[0]).data, expected_max, sizeof(expected_max))) return 54;
    if (memcmp(chelis_tensor_read_view(outputs[1]).data, expected_min, sizeof(expected_min))) return 55;
    if (memcmp(chelis_tensor_read_view(outputs[2]).data, expected_argmax, sizeof(expected_argmax))) return 56;
    if (memcmp(chelis_tensor_read_view(outputs[3]).data, expected_argmin, sizeof(expected_argmin))) return 57;
    puts("PASS"); return 0;
}}
"#
        );
        let run = compile_and_run(&format!("global_{tag}"), &source, &harness);
        assert!(
            run.status.success(),
            "{tag} exact reductions failed: stdout={} stderr={}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

fn mean_dag(input_ty: TensorType) -> (Dag, chelis_ir::NodeId) {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        input_ty.clone(),
        None,
    );
    let mean = tier2::lower_mean(&mut dag, x, 0, &input_ty, None);
    dag.add_root(mean);
    (dag, mean)
}

#[test]
fn global_mean_compiles_to_the_same_f32_sum_then_divide_bits_as_the_evaluator() {
    let input_ty = tensor(&[4], Prim::F32);
    let (dag, mean) = mean_dag(input_ty);
    let storage = finalize_tensor(
        "issue_1281_mean",
        Prim::F32,
        RawTensor::Float(vec![-1.0e-7, 3.0, 16_777_216.0, -33_554_432.0]),
    )
    .expect("f32 mean input");
    let evaluated = eval_tensor(
        &dag,
        &UnordMap::from([("x".to_string(), TensorValue::from_storage(vec![4], storage))]),
    )
    .expect("evaluate exact mean");
    let StorageView::F32(expected) = evaluated[&mean].storage().view() else {
        panic!("f32 mean evaluator result must retain f32 storage");
    };
    let expected_bits = expected[0].to_bits();
    let source = codegen(&dag, "issue1281_global_mean")
        .expect("global mean C codegen")
        .c_source;
    let harness = format!(
        r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void issue1281_global_mean(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float data[4] = {{-1.0e-7f,3.0f,16777216.0f,-33554432.0f}};
    int64_t shape[1] = {{4}};
    chelis_tensor *x = chelis_tensor_entry_borrow(1, shape, CHELIS_DTYPE_F32, data, sizeof(data));
    chelis_tensor *inputs[1] = {{x}}, *outputs[1] = {{0}};
    issue1281_global_mean(inputs, 1, outputs, 1);
    uint32_t got; memcpy(&got, chelis_tensor_read_view(outputs[0]).data, sizeof(got));
    if (got != UINT32_C(0x{expected_bits:08x})) return 70;
    puts("PASS"); return 0;
}}
"#
    );
    let run = compile_and_run("global_mean", &source, &harness);
    assert!(
        run.status.success(),
        "global mean diverged from evaluator: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn global_extrema_adjoints_split_infinity_ties_and_route_only_the_first_nan() {
    for (tag, op, data, expected) in [
        (
            "max_inf",
            RiscOp::MaxReduce { axis: 0 },
            "INFINITY,INFINITY,1.0f",
            "0.5f,0.5f,0.0f",
        ),
        (
            "min_inf",
            RiscOp::MinReduce { axis: 0 },
            "-INFINITY,-INFINITY,1.0f",
            "0.5f,0.5f,0.0f",
        ),
        (
            "max_nan",
            RiscOp::MaxReduce { axis: 0 },
            "NAN,NAN,1.0f",
            "1.0f,0.0f,0.0f",
        ),
        (
            "min_nan",
            RiscOp::MinReduce { axis: 0 },
            "NAN,NAN,1.0f",
            "1.0f,0.0f,0.0f",
        ),
    ] {
        let mut forward = Dag::new();
        let x = forward.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor(&[3], Prim::F32),
            None,
        );
        let reduced = forward.add_node(
            op,
            vec![x],
            TensorType {
                dims: vec![],
                precision: Prim::F32,
            },
            None,
        );
        let differentiated =
            grad_dag_checked(&forward, reduced, &[x]).expect("differentiate float extrema");
        let function = format!("issue1281_grad_{tag}");
        let source = codegen(&differentiated.dag, &function)
            .unwrap_or_else(|error| panic!("{tag} C gradient codegen: {error}"))
            .c_source;
        let harness = format!(
            r#"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float data[3] = {{{data}}}, expected[3] = {{{expected}}};
    int64_t shape[1] = {{3}};
    chelis_tensor *x = chelis_tensor_entry_borrow(1, shape, CHELIS_DTYPE_F32, data, sizeof(data));
    chelis_tensor *inputs[1] = {{x}}, *outputs[2] = {{0}};
    {function}(inputs, 1, outputs, 2);
    if (memcmp(chelis_tensor_read_view(outputs[1]).data, expected, sizeof(expected))) return 71;
    puts("PASS"); return 0;
}}
"#
        );
        let run = compile_and_run(&format!("grad_{tag}"), &source, &harness);
        assert!(
            run.status.success(),
            "{tag} exact extrema adjoint failed: stdout={} stderr={}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

fn empty_reduction_dag(op: RiscOp, result: Prim) -> Dag {
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        symbolic_vector("runtime_extent", Prim::F32),
        None,
    );
    let out = dag.add_node(
        op,
        vec![x],
        TensorType {
            dims: vec![],
            precision: result,
        },
        None,
    );
    dag.add_root(out);
    dag
}

#[test]
fn runtime_empty_global_extrema_and_argument_reductions_trap_domain() {
    for (name, op, result) in [
        ("max_reduce", RiscOp::MaxReduce { axis: 0 }, Prim::F32),
        ("min_reduce", RiscOp::MinReduce { axis: 0 }, Prim::F32),
        ("argmax_reduce", RiscOp::Argmax { axis: 0 }, Prim::Int64),
        ("argmin_reduce", RiscOp::Argmin { axis: 0 }, Prim::Int64),
    ] {
        let function = format!("issue1281_empty_{}", name.replace("_reduce", ""));
        let source = codegen(&empty_reduction_dag(op, result), &function)
            .expect("dynamic empty reduction codegen")
            .c_source;
        let harness = format!(
            r#"
#include <stdint.h>
#include <stdio.h>
#include "chelis_runtime.h"
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    int64_t shape[1] = {{0}};
    chelis_tensor *x = chelis_tensor_entry_borrow(1, shape, CHELIS_DTYPE_F32, NULL, 0);
    chelis_tensor *inputs[1] = {{x}}, *outputs[1] = {{0}};
    {function}(inputs, 1, outputs, 1);
    puts("UNREACHABLE"); return 0;
}}
"#
        );
        let run = compile_and_run(&format!("empty_{name}"), &source, &harness);
        assert!(
            !run.status.success(),
            "{name} accepted a runtime-empty axis"
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stderr).trim(),
            format!("numeric trap: domain in {name} at {}", result.name()),
            "{name} emitted the wrong trap"
        );
    }

    let (mean_dag, _) = mean_dag(symbolic_vector("runtime_extent", Prim::F32));
    let source = codegen(&mean_dag, "issue1281_empty_mean")
        .expect("dynamic empty mean codegen")
        .c_source;
    let harness = r#"
#include <stdint.h>
#include <stdio.h>
#include "chelis_runtime.h"
extern void issue1281_empty_mean(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {
    int64_t shape[1] = {0};
    chelis_tensor *x = chelis_tensor_entry_borrow(1, shape, CHELIS_DTYPE_F32, NULL, 0);
    chelis_tensor *inputs[1] = {x}, *outputs[1] = {0};
    issue1281_empty_mean(inputs, 1, outputs, 1);
    puts("UNREACHABLE"); return 0;
}
"#;
    let run = compile_and_run("empty_mean", &source, harness);
    assert!(!run.status.success(), "mean accepted a runtime-empty axis");
    assert_eq!(
        String::from_utf8_lossy(&run.stderr).trim(),
        "numeric trap: domain in mean at f32"
    );
}

fn window_forward_and_grad_dag(reducer: ReduceWindowKind, precision: Prim) -> Dag {
    let mut dag = Dag::new();
    let x_ty = tensor(&[4], precision);
    let g_ty = tensor(&[3], precision);
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        x_ty.clone(),
        None,
    );
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], g_ty, None);
    let forward = dag.add_node(
        RiscOp::ReduceWindow {
            reducer,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![x],
        tensor(&[3], precision),
        None,
    );
    let gradient = dag.add_node(
        RiscOp::ReduceWindowGrad {
            reducer,
            window_shape: vec![2],
            strides: vec![1],
        },
        vec![x, g],
        x_ty,
        None,
    );
    dag.add_root(forward);
    dag.add_root(gradient);
    dag
}

fn window_forward_matrix_dag(precision: Prim, include_mean: bool) -> Dag {
    let mut dag = Dag::new();
    let x_ty = tensor(&[4], precision);
    let out_ty = tensor(&[3], precision);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty, None);
    for reducer in [ReduceWindowKind::Max, ReduceWindowKind::Min] {
        let out = dag.add_node(
            RiscOp::ReduceWindow {
                reducer,
                window_shape: vec![2],
                strides: vec![1],
            },
            vec![x],
            out_ty.clone(),
            None,
        );
        dag.add_root(out);
    }
    if include_mean {
        let mean = dag.add_node(
            RiscOp::ReduceWindow {
                reducer: ReduceWindowKind::Mean,
                window_shape: vec![2],
                strides: vec![1],
            },
            vec![x],
            out_ty,
            None,
        );
        dag.add_root(mean);
    }
    dag
}

#[test]
fn window_forward_extrema_execute_at_every_integer_and_wide_float_storage_width() {
    for (tag, prim, c_type, dtype, data, max, min, mean) in [
        (
            "f64",
            Prim::F64,
            "double",
            "CHELIS_DTYPE_F64",
            "1.0,2.0,3.0,4.0",
            "2.0,3.0,4.0",
            "1.0,2.0,3.0",
            Some("1.5,2.5,3.5"),
        ),
        (
            "i8",
            Prim::Int8,
            "int8_t",
            "CHELIS_DTYPE_I8",
            "1,2,3,4",
            "2,3,4",
            "1,2,3",
            None,
        ),
        (
            "i16",
            Prim::Int16,
            "int16_t",
            "CHELIS_DTYPE_I16",
            "1,2,3,4",
            "2,3,4",
            "1,2,3",
            None,
        ),
        (
            "i32",
            Prim::Int32,
            "int32_t",
            "CHELIS_DTYPE_I32",
            "1,2,3,4",
            "2,3,4",
            "1,2,3",
            None,
        ),
        (
            "i64",
            Prim::Int64,
            "int64_t",
            "CHELIS_DTYPE_I64",
            "INT64_C(9007199254740993),INT64_C(9007199254740995),INT64_C(9007199254740994),INT64_C(9007199254740996)",
            "INT64_C(9007199254740995),INT64_C(9007199254740995),INT64_C(9007199254740996)",
            "INT64_C(9007199254740993),INT64_C(9007199254740994),INT64_C(9007199254740994)",
            None,
        ),
    ] {
        let function = format!("issue1281_window_{tag}");
        let source = codegen(&window_forward_matrix_dag(prim, mean.is_some()), &function)
            .unwrap_or_else(|error| panic!("{tag} window codegen: {error}"))
            .c_source;
        let mean_decl = mean
            .map(|values| format!("{c_type} expected_mean[3] = {{{values}}};"))
            .unwrap_or_default();
        let mean_check = mean
            .map(|_| {
                "if (memcmp(chelis_tensor_read_view(outputs[2]).data, expected_mean, sizeof(expected_mean))) return 63;"
            })
            .unwrap_or_default();
        let outputs = if mean.is_some() { 3 } else { 2 };
        let harness = format!(
            r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} data[4] = {{{data}}};
    {c_type} expected_max[3] = {{{max}}}, expected_min[3] = {{{min}}};
    {mean_decl}
    int64_t shape[1] = {{4}};
    chelis_tensor *x = chelis_tensor_entry_borrow(1, shape, {dtype}, data, sizeof(data));
    chelis_tensor *inputs[1] = {{x}}, *outputs[3] = {{0}};
    {function}(inputs, 1, outputs, {outputs});
    if (memcmp(chelis_tensor_read_view(outputs[0]).data, expected_max, sizeof(expected_max))) return 61;
    if (memcmp(chelis_tensor_read_view(outputs[1]).data, expected_min, sizeof(expected_min))) return 62;
    {mean_check}
    puts("PASS"); return 0;
}}
"#
        );
        let run = compile_and_run(&format!("window_{tag}"), &source, &harness);
        assert!(
            run.status.success(),
            "{tag} exact window reductions failed: stdout={} stderr={}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

#[test]
fn reduced_float_window_extrema_and_mean_preserve_storage_and_f32_arithmetic() {
    for (tag, prim, dtype, data, max, min, mean) in [
        (
            "f16",
            Prim::F16,
            "CHELIS_DTYPE_F16",
            "0x3c00,0x4000,0x4200,0x4400",
            "0x4000,0x4200,0x4400",
            "0x3c00,0x4000,0x4200",
            "0x3e00,0x4100,0x4300",
        ),
        (
            "bf16",
            Prim::Bf16,
            "CHELIS_DTYPE_BF16",
            "0x3f80,0x4000,0x4040,0x4080",
            "0x4000,0x4040,0x4080",
            "0x3f80,0x4000,0x4040",
            "0x3fc0,0x4020,0x4060",
        ),
    ] {
        let function = format!("issue1281_window_{tag}");
        let source = codegen(&window_forward_matrix_dag(prim, true), &function)
            .unwrap_or_else(|error| panic!("{tag} window codegen: {error}"))
            .c_source;
        let harness = format!(
            r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint16_t data[4] = {{{data}}};
    uint16_t expected_max[3] = {{{max}}}, expected_min[3] = {{{min}}}, expected_mean[3] = {{{mean}}};
    int64_t shape[1] = {{4}};
    chelis_tensor *x = chelis_tensor_entry_borrow(1, shape, {dtype}, data, sizeof(data));
    chelis_tensor *inputs[1] = {{x}}, *outputs[3] = {{0}};
    {function}(inputs, 1, outputs, 3);
    if (memcmp(chelis_tensor_read_view(outputs[0]).data, expected_max, sizeof(expected_max))) return 64;
    if (memcmp(chelis_tensor_read_view(outputs[1]).data, expected_min, sizeof(expected_min))) return 65;
    if (memcmp(chelis_tensor_read_view(outputs[2]).data, expected_mean, sizeof(expected_mean))) return 66;
    puts("PASS"); return 0;
}}
"#
        );
        let run = compile_and_run(&format!("window_{tag}"), &source, &harness);
        assert!(
            run.status.success(),
            "{tag} exact window reductions failed: stdout={} stderr={}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }
}

#[test]
fn window_extrema_forward_and_adjoint_use_first_nan_and_divide_non_nan_ties() {
    let source = codegen(
        &window_forward_and_grad_dag(ReduceWindowKind::Max, Prim::F32),
        "issue1281_window_exact",
    )
    .expect("window max codegen")
    .c_source;
    assert!(!source.contains("fmaxf("), "window max must not drop NaNs");
    assert!(
        !source.contains("fminf("),
        "window adjoint must select the first NaN itself"
    );
    assert!(
        !source.contains("existing tie/NaN arithmetic remains tracked by #1298"),
        "#1281 owns the exact extrema adjoint semantics in this lane"
    );
    let harness = r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern void issue1281_window_exact(chelis_tensor **, int, chelis_tensor **, int);
static chelis_tensor *view(void *data, int64_t n) {
    int64_t shape[1] = {n};
    return chelis_tensor_entry_borrow(1, shape, CHELIS_DTYPE_F32, data, n * 4);
}
int main(void) {
    uint32_t xbits[4] = {UINT32_C(0xffc12345), UINT32_C(0x7fc54321), UINT32_C(0x7f800000), UINT32_C(0x7f800000)};
    float x[4], g[3] = {6.0f, 8.0f, 10.0f}; memcpy(x, xbits, sizeof(x));
    chelis_tensor *xt = view(x, 4), *gt = view(g, 3);
    chelis_tensor *inputs[2] = {xt, gt}, *outputs[2] = {0};
    issue1281_window_exact(inputs, 2, outputs, 2);
    uint32_t expected_forward[3] = {UINT32_C(0xffc12345), UINT32_C(0x7fc54321), UINT32_C(0x7f800000)};
    if (memcmp(chelis_tensor_read_view(outputs[0]).data, expected_forward, sizeof(expected_forward))) return 30;
    float expected_grad[4] = {6.0f, 8.0f, 5.0f, 5.0f};
    if (memcmp(chelis_tensor_read_view(outputs[1]).data, expected_grad, sizeof(expected_grad))) return 31;
    puts("PASS"); return 0;
}
"#;
    let run = compile_and_run("window_extrema", &source, harness);
    assert!(
        run.status.success(),
        "window exact extrema failed: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn overlapping_window_adjoint_combines_contributions_with_balanced_tree_order() {
    let mut dag = Dag::new();
    let x_ty = tensor(&[3, 3], Prim::F32);
    let g_ty = tensor(&[2, 2], Prim::F32);
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        x_ty.clone(),
        None,
    );
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], g_ty, None);
    let out = dag.add_node(
        RiscOp::ReduceWindowGrad {
            reducer: ReduceWindowKind::Max,
            window_shape: vec![2, 2],
            strides: vec![1, 1],
        },
        vec![x, g],
        x_ty,
        None,
    );
    dag.add_root(out);
    let source = codegen(&dag, "issue1281_balanced_overlap")
        .expect("overlap adjoint codegen")
        .c_source;
    let harness = r#"
#include <stdint.h>
#include <stdio.h>
#include "chelis_runtime.h"
extern void issue1281_balanced_overlap(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {
    float x[9] = {1,1,1,1,1,1,1,1,1};
    float g[4] = {4.0e20f, 4.0f, -4.0e20f, 4.0f};
    int64_t xs[2] = {3,3}, gs[2] = {2,2};
    chelis_tensor *xt = chelis_tensor_entry_borrow(2, xs, CHELIS_DTYPE_F32, x, sizeof(x));
    chelis_tensor *gt = chelis_tensor_entry_borrow(2, gs, CHELIS_DTYPE_F32, g, sizeof(g));
    chelis_tensor *inputs[2] = {xt, gt}, *outputs[1] = {0};
    issue1281_balanced_overlap(inputs, 2, outputs, 1);
    const float *actual = chelis_tensor_read_view(outputs[0]).data;
    if (actual[4] != 0.0f) return 40;
    puts("PASS"); return 0;
}
"#;
    let run = compile_and_run("balanced_overlap", &source, harness);
    assert!(
        run.status.success(),
        "window overlap order was not balanced: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
