//! Adversarial execution tests: actually compile AND RUN generated C code.
//! These test numerical correctness, not just source patterns.
//!
//! The generated kernel signature is:
//!   void func(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out)
//! The kernel allocates output tensors internally via chelis_alloc.
//! We link against the chelis_runtime .a to resolve those symbols.

use chelis_backend_c::{CodegenOptions, MathLib};
mod support;
use chelis_ir::ConcreteHostType as HostType;
use chelis_ir::dag::{
    Dag, DimInfo, ExtremaKind, ExtremaOperand, ReduceWindowKind, RiscOp, TensorType,
};
use chelis_ir::fuse::fuse;
use chelis_ir::host::{
    ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind,
    ConcreteHostFunction as HostFunction, ConcreteHostParam as HostParam,
    ConcreteHostProgram as HostProgram, HostFunctionOrigin,
};
use chelis_types::types::Prim;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use support::{codegen, codegen_with_options, emit_host_program};

mod common;

fn checked_indexing_run(source: &str, harness: &str) -> std::process::Output {
    let probe = common::probe_dir("checked_c_indexing");
    let dir = probe.path();
    fs::write(dir.join("kernel.c"), source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();
    let toolchain = chelis_backend_c::toolchain::test_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            needs_blas: source.contains("#include \"chelis_blas.h\""),
            ..Default::default()
        },
    );
    let binary = dir.join("probe");
    let compiled = Command::new(toolchain.compiler)
        .args([
            "-O2",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
        ])
        .args(toolchain.compile_flags)
        .arg("-I")
        .arg(runtime_include_dir())
        .arg(dir.join("kernel.c"))
        .arg(dir.join("main.c"))
        .arg(runtime_lib_path())
        .args(toolchain.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}\n{source}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    Command::new(binary)
        .env("ASAN_OPTIONS", "detect_leaks=0")
        .output()
        .unwrap()
}

fn checked_literal_dag(storage: chelis_types::TensorStorage, extent: usize, output: Prim) -> Dag {
    let mut dag = Dag::new();
    dag.add_node(
        RiscOp::ConstTensor { data: storage },
        vec![],
        TensorType {
            dims: vec![DimInfo::Lit(extent)],
            precision: output,
        },
        None,
    );
    dag
}
#[test]
fn checked_literals_preserve_every_storage_width_and_reject_count_mismatch() {
    use chelis_types::{RawTensor, StorageView, finalize_tensor};
    for dtype in [
        Prim::F32,
        Prim::F64,
        Prim::F16,
        Prim::Bf16,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
    ] {
        let raw = if dtype.is_float() {
            RawTensor::Float(vec![-0.0, 1.5, f64::INFINITY, f64::NAN])
        } else if dtype == Prim::Bool {
            RawTensor::Int(vec![0, 1, 1, 0])
        } else if dtype == Prim::Int64 {
            RawTensor::Int(vec![i64::MIN, 9007199254740993, -1, i64::MAX])
        } else {
            RawTensor::Int(vec![-1, 0, 1, 127])
        };
        let storage = finalize_tensor("const", dtype, raw).unwrap();
        let expected: Vec<u8> = match storage.view() {
            StorageView::F64(v) => v.iter().flat_map(|x| x.to_bits().to_ne_bytes()).collect(),
            StorageView::F32(v) => v.iter().flat_map(|x| x.to_bits().to_ne_bytes()).collect(),
            StorageView::F16(v) => v.iter().flat_map(|x| x.to_bits().to_ne_bytes()).collect(),
            StorageView::Bf16(v) => v.iter().flat_map(|x| x.to_bits().to_ne_bytes()).collect(),
            StorageView::I64(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            StorageView::I32(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            StorageView::I16(v) => v.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            StorageView::I8(v) => v.iter().map(|x| *x as u8).collect(),
            StorageView::Bool(v) => v.to_vec(),
        };
        for extent in [4, 3, 0] {
            let source = codegen(
                &checked_literal_dag(storage.clone(), extent, dtype),
                "literal_probe",
            )
            .unwrap()
            .c_source;
            let bytes = expected
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let harness = format!(
                r#"
#include "chelis_runtime.h"
#include <string.h>
extern void literal_probe(chelis_tensor**,int,chelis_tensor**,int);
int main(void) {{
 chelis_tensor *out[1]={{0}};literal_probe(0,0,out,1);
 unsigned char expected[]={{{bytes}}};
 chelis_read_view v=chelis_tensor_read_view(out[0]);
 if(v.dtype!={dtype} || v.count!=4 || memcmp(v.data,expected,sizeof expected))return 10;
 chelis_tensor_release(out[0]);return 0;
}}
"#,
                dtype = dtype.runtime_dtype().unwrap().c_macro()
            );
            let out = checked_indexing_run(&source, &harness);
            if extent == 4 {
                assert!(out.status.success(), "{dtype:?}: {out:?}");
            } else {
                assert_eq!(out.status.code(), Some(1), "{dtype:?}/{extent}: {out:?}");
                assert!(
                    String::from_utf8_lossy(&out.stderr)
                        .lines()
                        .any(|s| s == "numeric trap: domain in const at int64"),
                    "{out:?}"
                );
            }
        }
        let empty = storage.reuse_gather(&[]);
        let source = codegen(&checked_literal_dag(empty, 0, dtype), "literal_probe")
            .unwrap()
            .c_source;
        let harness = r#"#include "chelis_runtime.h"
extern void literal_probe(chelis_tensor**,int,chelis_tensor**,int);
int main(void){chelis_tensor *out[1]={0};literal_probe(0,0,out,1);if(chelis_tensor_numel(out[0])!=0)return 1;chelis_tensor_release(out[0]);return 0;}"#;
        let out = checked_indexing_run(&source, harness);
        assert!(out.status.success(), "{dtype:?} empty: {out:?}");
    }
}
#[test]
fn checked_literals_reject_inconsistent_ir_storage_dtype() {
    let value = chelis_types::finalize_tensor(
        "const",
        Prim::F32,
        chelis_types::RawTensor::Float(vec![1.0]),
    )
    .unwrap();
    let result = codegen(&checked_literal_dag(value, 1, Prim::Int32), "literal_probe");
    assert!(
        result.is_err(),
        "inconsistent literal storage dtype must reject"
    );
}

fn checked_window_dag(reducer: ReduceWindowKind, gradient: bool) -> Dag {
    let mut dag = Dag::new();
    let input = TensorType {
        dims: ["batch", "height", "width"]
            .into_iter()
            .map(|n| DimInfo::Named(n.into(), None))
            .collect(),
        precision: Prim::F32,
    };
    let result = TensorType {
        dims: vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Lit(2),
            DimInfo::Lit(2),
        ],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        input.clone(),
        None,
    );
    if gradient {
        let gty = TensorType {
            dims: ["gb", "gh", "gw"]
                .into_iter()
                .map(|n| DimInfo::Named(n.into(), None))
                .collect(),
            precision: Prim::F32,
        };
        let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], gty, None);
        dag.add_node(
            RiscOp::ReduceWindowGrad {
                reducer,
                window_shape: vec![2, 2],
                strides: vec![2, 2],
            },
            vec![x, g],
            input,
            None,
        );
    } else {
        dag.add_node(
            RiscOp::ReduceWindow {
                reducer,
                window_shape: vec![2, 2],
                strides: vec![2, 2],
            },
            vec![x],
            result,
            None,
        );
    }
    dag
}
#[test]
fn checked_windows_forward_and_gradient_execute_with_exact_geometry_under_sanitizers() {
    for (kind, reducer) in [
        ReduceWindowKind::Sum,
        ReduceWindowKind::Mean,
        ReduceWindowKind::Max,
        ReduceWindowKind::Min,
    ]
    .into_iter()
    .enumerate()
    {
        for gradient in [false, true] {
            let source = codegen(&checked_window_dag(reducer, gradient), "window_probe")
                .unwrap()
                .c_source;
            for batch in [2, 0] {
                let harness = format!(
                    r#"
#include "chelis_runtime.h"
#include <stdio.h>
extern void window_probe(chelis_tensor**, int, chelis_tensor**, int);
int main(void) {{
    int64_t xs[3] = {{{batch},5,5}}, gs[3] = {{{batch},2,2}};
    chelis_tensor *x = chelis_alloc(3,xs,CHELIS_DTYPE_F32);
    chelis_tensor *g = chelis_alloc(3,gs,CHELIS_DTYPE_F32);
    chelis_tensor_write *w = chelis_tensor_begin_write(x);
    float *xp = (float*)chelis_tensor_write_view(w).data;
    for (int i=0;i<{batch}*25;i++) xp[i]=(float)(i+1);
    chelis_tensor_end_write(w);
    w = chelis_tensor_begin_write(g);
    float *gp = (float*)chelis_tensor_write_view(w).data;
    for (int i=0;i<{batch}*4;i++) gp[i]=(float)(4*(i+1));
    chelis_tensor_end_write(w);
    chelis_tensor *inputs[2]={{x,g}}, *out[1]={{0}};
    window_probe(inputs,{ninputs},out,1);
    if (chelis_tensor_numel(out[0]) != {batch}*{per_batch}) return 10;
    if (chelis_tensor_shape(out[0],1) != {height} || chelis_tensor_shape(out[0],2) != {width}) return 11;
    const float *actual=(const float*)chelis_tensor_read_view(out[0]).data;
    float expected[60]={{0}};
    for(int b=0;b<{batch};b++) for(int h=0;h<2;h++) for(int c=0;c<2;c++) {{
        int first=b*25+h*10+c*2, group=b*4+h*2+c;
        int slots[4]={{first,first+1,first+5,first+6}};
        if ({gradient}) {{
            for(int j=0;j<4;j++) if ({kind}<2 || ({kind}==2 && j==3) || ({kind}==3 && j==0)) expected[slots[j]]+=(float)(4*(group+1))/({kind}==1?4.0f:1.0f);
        }} else {{
            float value={kind}==2?(float)(first+7):{kind}==3?(float)(first+1):(float)(4*first+16);
            expected[group]={kind}==1?value/4.0f:value;
        }}
    }}
    for(int i=0;i<{batch}*{per_batch};i++) if(actual[i]!=expected[i]) {{ fprintf(stderr,"%d %g %g\n",i,actual[i],expected[i]);return 12; }}
    chelis_tensor_release(out[0]);chelis_tensor_release(g);chelis_tensor_release(x);return 0;
}}
"#,
                    ninputs = if gradient { 2 } else { 1 },
                    per_batch = if gradient { 25 } else { 4 },
                    height = if gradient { 5 } else { 2 },
                    width = if gradient { 5 } else { 2 },
                    gradient = usize::from(gradient)
                );
                let output = checked_indexing_run(&source, &harness);
                assert!(
                    output.status.success(),
                    "{kind}/{gradient}/{batch}: {output:?}"
                );
            }
        }
    }
}
#[test]
fn checked_windows_reject_incorrect_runtime_result_and_cotangent_shapes() {
    for gradient in [false, true] {
        let source = codegen(
            &checked_window_dag(ReduceWindowKind::Sum, gradient),
            "window_probe",
        )
        .unwrap()
        .c_source;
        let harness = format!(
            r#"
#include "chelis_runtime.h"
extern void window_probe(chelis_tensor**,int,chelis_tensor**,int);
int main(void) {{
 int64_t xs[3]={{1,{height},5}},gs[3]={{1,1,4}};
 chelis_tensor *x=chelis_alloc(3,xs,CHELIS_DTYPE_F32), *g=chelis_alloc(3,gs,CHELIS_DTYPE_F32);
 chelis_tensor *inputs[2]={{x,g}},*out[1]={{0}};
 window_probe(inputs,{ninputs},out,1);return 99;
}}
"#,
            height = if gradient { 5 } else { 6 },
            ninputs = if gradient { 2 } else { 1 }
        );
        let output = checked_indexing_run(&source, &harness);
        assert_eq!(output.status.code(), Some(1), "{output:?}");
        let op = if gradient {
            "reduce_window_grad"
        } else {
            "reduce_window_sum"
        };
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .any(|s| s == format!("numeric trap: domain in {op} at int64")),
            "{output:?}"
        );
    }
}

fn checked_blas_dag(operand: Prim, output: Prim) -> Dag {
    use chelis_ir::dag::DimExpr;
    let ty = |names: &[&str], precision| TensorType {
        dims: names
            .iter()
            .map(|n| DimInfo::Named((*n).into(), None))
            .collect(),
        precision,
    };
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        ty(&["batch0", "batch1", "m", "k"], operand),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        ty(&["batch0", "batch1", "k", "n"], operand),
        None,
    );
    let node = dag.add_node(
        RiscOp::BlasMatmul {
            batch_dims: vec![DimExpr::Sym("batch0".into()), DimExpr::Sym("batch1".into())],
            m: DimExpr::Sym("m".into()),
            n: DimExpr::Sym("n".into()),
            k: DimExpr::Sym("k".into()),
            accumulator: if operand == Prim::F64 {
                Prim::F64
            } else {
                Prim::F32
            },
        },
        vec![a, b],
        ty(&["batch0", "batch1", "m", "n"], output),
        None,
    );
    dag.add_root(node);
    dag
}

#[test]
fn blas_vendor_dimension_contract_matches_actual_function_prototypes() {
    let source = codegen_with_options(
        &checked_blas_dag(Prim::F32, Prim::F32),
        "vendor_contract",
        CodegenOptions {
            use_blas: true,
            ..Default::default()
        },
    )
    .unwrap()
    .c_source;
    let preamble = source
        .split("/* CHELIS_UNIFORM_HELPERS_BEGIN */")
        .next()
        .unwrap();
    let contract = &preamble[preamble.find("#ifndef CHELIS_C_BLAS_CONTRACT").unwrap()..];
    let probe = common::probe_dir("blas_dimension_contract");
    let compiler = chelis_backend_c::toolchain::c_compiler();
    let compile = |text: &str| {
        fs::write(probe.path().join("contract.c"), text).unwrap();
        Command::new(&compiler)
            .args(["-std=c11", "-Werror", "-c", "-I"])
            .arg(runtime_include_dir())
            .arg(probe.path().join("contract.c"))
            .arg("-o")
            .arg(probe.path().join("contract.o"))
            .output()
            .unwrap()
    };
    let actual = compile(preamble);
    assert!(
        actual.status.success(),
        "{}",
        String::from_utf8_lossy(&actual.stderr)
    );
    if cfg!(target_os = "macos") {
        let ilp64 = compile(&format!("#define ACCELERATE_LAPACK_ILP64 1\n{preamble}"));
        assert!(
            ilp64.status.success(),
            "{}",
            String::from_utf8_lossy(&ilp64.stderr)
        );
    }
    for openblas in [false, true] {
        for (declared, argument, want) in [
            ("int32_t", "int32_t", None),
            ("int64_t", "int64_t", None),
            ("int32_t", "int64_t", Some("dimension contract mismatch")),
            ("int64_t", "int32_t", Some("dimension contract mismatch")),
            ("uint32_t", "uint32_t", Some("dimensions must be signed")),
            (
                "int16_t",
                "int16_t",
                Some("unsupported CBLAS dimension width"),
            ),
        ] {
            let layout = if openblas {
                "CBLAS_ORDER"
            } else {
                "CBLAS_LAYOUT"
            };
            let selector = if openblas {
                format!("#define OPENBLAS_VERSION 1\ntypedef {declared} blasint;\n")
            } else {
                format!("#define CBLAS_INT {declared}\n")
            };
            let declarations=[("s","float"),("d","double")].iter().map(|(prefix,element)|format!("void cblas_{prefix}gemm(enum {layout}, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE, {argument}, {argument}, {argument}, {element}, const {element}*, {argument}, const {element}*, {argument}, {element}, {element}*, {argument});\n")).collect::<String>();
            let fake = format!(
                "#include <stdint.h>\n#undef __APPLE__\n{selector}typedef enum {layout} {{ CblasRowMajor=101 }} {layout};\nenum CBLAS_TRANSPOSE {{ CblasNoTrans=111 }};\n{declarations}{contract}"
            );
            let result = compile(&fake);
            if let Some(reason) = want {
                assert!(!result.status.success(), "{openblas}/{declared}/{argument}");
                assert!(
                    String::from_utf8_lossy(&result.stderr).contains(reason),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
            } else {
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
        }
    }
}

#[test]
fn checked_blas_batches_scratch_and_empty_domains_execute_under_sanitizers() {
    for (operand, output) in [
        (Prim::F32, Prim::F32),
        (Prim::F64, Prim::F64),
        (Prim::F16, Prim::F16),
        (Prim::Bf16, Prim::Bf16),
        (Prim::F16, Prim::F32),
        (Prim::Bf16, Prim::F32),
    ] {
        let source = codegen_with_options(
            &checked_blas_dag(operand, output),
            "checked_blas",
            CodegenOptions {
                use_blas: true,
                ..Default::default()
            },
        )
        .unwrap()
        .c_source;
        // Existing selected nodes require their ABI declarations even when the
        // caller did not request another specialization pass.
        let default_options = codegen(&checked_blas_dag(operand, output), "checked_blas").unwrap();
        assert!(default_options.requirements.needs_blas);
        assert_eq!(default_options.c_source, source);
        let mut sources = vec![source];
        if operand == Prim::F32 && output == Prim::F32 {
            use chelis_ir::dag::DimExpr;
            use chelis_ir::host::{
                HostBlasMatmulSummary, HostTensorHelper, HostTensorInput, HostTensorSpecialization,
            };
            let dag = checked_blas_dag(operand, output);
            let a_ty = dag.nodes()[0].output_type.clone();
            let b_ty = dag.nodes()[1].output_type.clone();
            let out_ty = dag.nodes()[2].output_type.clone();
            let inputs = vec![
                HostTensorInput {
                    name: "a".into(),
                    ty: a_ty.clone(),
                },
                HostTensorInput {
                    name: "b".into(),
                    ty: b_ty.clone(),
                },
            ];
            let params = inputs
                .iter()
                .map(|input| HostParam {
                    name: input.name.clone(),
                    ty: HostType::Tensor(input.ty.clone()),
                })
                .collect::<Vec<_>>();
            let summary = HostBlasMatmulSummary {
                lhs_input: 0,
                rhs_input: 1,
                input_tys: vec![a_ty, b_ty],
                output: out_ty.clone(),
                batch_dims: vec![DimExpr::Sym("batch0".into()), DimExpr::Sym("batch1".into())],
                m: DimExpr::Sym("m".into()),
                n: DimExpr::Sym("n".into()),
                k: DimExpr::Sym("k".into()),
            };
            let body = HostExpr::new(HostExprKind::TensorCall {
                helper: 0,
                args: params
                    .iter()
                    .map(|p| HostExpr::new(HostExprKind::Var(p.name.clone(), p.ty.clone())))
                    .collect(),
                ty: HostType::Tensor(out_ty.clone()),
            });
            let helper = HostTensorHelper {
                name: "matmul_helper".into(),
                dag,
                inputs,
                output: out_ty.clone(),
                specialization: Some(HostTensorSpecialization::BlasMatmul(summary)),
                summary_rejection: None,
            };
            let program = HostProgram {
                globals: vec![],
                global_tensor_helpers: vec![],
                summary_rejections: vec![],
                functions: vec![HostFunction {
                    name: "host_blas".into(),
                    params,
                    ret_ty: HostType::Tensor(out_ty),
                    body,
                    tensor_helpers: vec![helper],
                    origin: HostFunctionOrigin::Authored,
                    specialization: None,
                    summary_rejections: vec![],
                }],
            };
            let mut host_source =
                support::emit_selected_host_program(program, "host_blas_probe").unwrap();
            assert!(
                host_source.contains("blas_plan_"),
                "summary must execute instead of its helper"
            );
            host_source.push_str("\nvoid checked_blas(chelis_tensor **in,int n_in,chelis_tensor **out,int n_out) { (void)n_in; (void)n_out; out[0]=host_blas(in[0],in[1]); }\n");
            sources.push(host_source);
        }
        let input_dtype = operand.runtime_dtype().unwrap().c_macro();
        let output_dtype = output.runtime_dtype().unwrap().c_macro();
        let store = match operand {
            Prim::F32 => "((float*)v.data)[i]=value;",
            Prim::F64 => "((double*)v.data)[i]=value;",
            Prim::F16 => "((uint16_t*)v.data)[i]=chelis_f32_to_f16(value);",
            Prim::Bf16 => "((uint16_t*)v.data)[i]=chelis_f32_to_bf16(value);",
            _ => unreachable!(),
        };
        let read = match output {
            Prim::F32 => "((const float*)v.data)[i]",
            Prim::F64 => "((const double*)v.data)[i]",
            Prim::F16 => "chelis_f16_to_f32(((const uint16_t*)v.data)[i])",
            Prim::Bf16 => "chelis_bf16_to_f32(((const uint16_t*)v.data)[i])",
            _ => unreachable!(),
        };
        for (batches, k) in [(2, 2), (0, 2), (2, 0)] {
            let harness = format!(
                r#"
#include "chelis_runtime.h"
void checked_blas(chelis_tensor **, int, chelis_tensor **, int);
static void fill(chelis_tensor *t) {{
    chelis_tensor_write *g=chelis_tensor_begin_write(t); chelis_write_view v=chelis_tensor_write_view(g);
    for(int64_t i=0;i<v.count;++i) {{ float value=(float)(i%3+1); {store} }}
    chelis_tensor_end_write(g);
}}
int main(void) {{
    chelis_tensor *a=chelis_alloc(4,(int64_t[]){{{batches},2,3,{k}}},{input_dtype});
    chelis_tensor *b=chelis_alloc(4,(int64_t[]){{{batches},2,{k},4}},{input_dtype});
    fill(a); fill(b); chelis_tensor *in[]={{a,b}},*out[1]={{0}};
    checked_blas(in,2,out,1);
    chelis_read_view v=chelis_tensor_read_view(out[0]);
    if(v.count!={batches}*2*3*4 || v.dtype!={output_dtype}) return 2;
    int64_t shape[4]={{{batches},2,3,4}};
    for(int axis=0;axis<4;++axis) if(chelis_tensor_shape(out[0],axis)!=shape[axis]) return 3;
    for(int64_t i=0;i<v.count;++i) {{
        int64_t batch=i/12,row=(i%12)/4,col=i%4; double expected=0;
        for(int64_t t=0;t<{k};++t) expected+=(double)((batch*3*{k}+row*{k}+t)%3+1)*(double)((batch*{k}*4+t*4+col)%3+1);
        if ((double)({read})!=expected) return 4;
    }}
    chelis_tensor_release(out[0]); chelis_tensor_release(a); chelis_tensor_release(b);
    puts("MATMUL PASS"); return 0;
}}
"#
            );
            for source in &sources {
                let result = checked_indexing_run(source, &harness);
                assert!(
                    result.status.success(),
                    "{operand:?}/{output:?} {batches}/{k}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert_eq!(result.stdout, b"MATMUL PASS\n");
            }
        }
    }
}

#[test]
fn checked_c_sparse_mappings_preserve_stored_bits_under_sanitizers() {
    for (prim, dtype, seed) in [
        (Prim::F32, "CHELIS_DTYPE_F32", 0x3f800000_u64),
        (Prim::F64, "CHELIS_DTYPE_F64", 0x3ff0000000000000),
        (Prim::F16, "CHELIS_DTYPE_F16", 0x3c00),
        (Prim::Bf16, "CHELIS_DTYPE_BF16", 0x3f80),
        (Prim::Int8, "CHELIS_DTYPE_I8", 10),
        (Prim::Int16, "CHELIS_DTYPE_I16", 100),
        (Prim::Int32, "CHELIS_DTYPE_I32", 100),
        (Prim::Int64, "CHELIS_DTYPE_I64", 9_007_199_254_740_993),
        (Prim::Bool, "CHELIS_DTYPE_BOOL", 0),
    ] {
        for (index_prim, index_dtype, index_type) in [
            (Prim::Int32, "CHELIS_DTYPE_I32", "int32_t"),
            (Prim::Int64, "CHELIS_DTYPE_I64", "int64_t"),
        ] {
            for (op, base_shape, output_shape, selected, map) in [
                (
                    RiscOp::Gather { axis: 1 },
                    vec![2, 3, 2],
                    vec![2, 2, 2, 2],
                    "2,0,1,2",
                    "4,5,0,1,2,3,4,5,10,11,6,7,8,9,10,11",
                ),
                // Negative map entries select unchanged base cells (-index-1).
                (
                    RiscOp::Scatter { axis: 1 },
                    vec![2, 3, 2],
                    vec![2, 3, 2],
                    "2,0,2,0",
                    "6,7,-3,-4,4,5,14,15,-9,-10,12,13",
                ),
                (
                    RiscOp::ScatterElements { axis: 1 },
                    vec![2, 3],
                    vec![2, 3],
                    "2,0,1,1",
                    "1,-2,0,-4,3,-6",
                ),
            ] {
                let gather = matches!(op, RiscOp::Gather { .. });
                let elements = matches!(op, RiscOp::ScatterElements { .. });
                let ty = |shape: &[usize], precision| TensorType {
                    dims: shape.iter().copied().map(DimInfo::Lit).collect(),
                    precision,
                };
                let mut dag = Dag::new();
                let base = dag.add_node(
                    RiscOp::Load {
                        name: "base".into(),
                    },
                    vec![],
                    ty(&base_shape, prim),
                    None,
                );
                let indices = dag.add_node(
                    RiscOp::Load {
                        name: "indices".into(),
                    },
                    vec![],
                    ty(&[2, 2], index_prim),
                    None,
                );
                let mut inputs = vec![base, indices];
                let update_shape = if elements {
                    vec![2, 2]
                } else {
                    vec![2, 2, 2, 2]
                };
                if !gather {
                    inputs.push(dag.add_node(
                        RiscOp::Load {
                            name: "updates".into(),
                        },
                        vec![],
                        ty(&update_shape, prim),
                        None,
                    ));
                }
                let output = dag.add_node(op, inputs, ty(&output_shape, prim), None);
                dag.add_root(output);
                let generated = codegen(&dag, "checked_sparse").unwrap();
                let dimensions =
                    |s: &[usize]| s.iter().map(usize::to_string).collect::<Vec<_>>().join(",");
                let base_dims = dimensions(&base_shape);
                let update_dims = dimensions(&update_shape);
                let output_dims = dimensions(&output_shape);
                let base_rank = base_shape.len();
                let update_rank = update_shape.len();
                let output_rank = output_shape.len();
                let input_count = if gather { 2 } else { 3 };
                let gather = usize::from(gather);
                let harness = format!(
                    r#"
#include "chelis_runtime.h"
#include <string.h>
void checked_sparse(chelis_tensor **, int, chelis_tensor **, int);
static uint64_t bits(int64_t index, int update) {{
    return {dtype} == CHELIS_DTYPE_BOOL ? (uint64_t)(((index/2+index)%2)^update) : UINT64_C({seed}) + (uint64_t)index + (update ? 32 : 0);
}}
static void fill(chelis_tensor *t, int update) {{
    chelis_tensor_write *g = chelis_tensor_begin_write(t);
    chelis_write_view v = chelis_tensor_write_view(g);
    for (int64_t i=0;i<v.count;++i) {{ uint64_t value=bits(i,update); memcpy((unsigned char*)v.data+i*chelis_dtype_size({dtype}), &value, chelis_dtype_size({dtype})); }}
    chelis_tensor_end_write(g);
}}
int main(void) {{
    chelis_tensor *base=chelis_alloc({base_rank},(int64_t[]){{{base_dims}}},{dtype});
    chelis_tensor *indices=chelis_alloc(2,(int64_t[]){{2,2}},{index_dtype});
    chelis_tensor *updates=chelis_alloc({update_rank},(int64_t[]){{{update_dims}}},{dtype});
    fill(base,0); fill(updates,1);
    chelis_tensor_write *g=chelis_tensor_begin_write(indices);
    {index_type} values[4]={{{selected}}};
    memcpy(chelis_tensor_write_view(g).data,values,sizeof values); chelis_tensor_end_write(g);
    chelis_tensor *in[]={{base,indices,updates}}, *out[1]={{0}};
    checked_sparse(in,{input_count},out,1);
    int64_t shape[]={{{output_dims}}}; int expected[]={{{map}}};
    if (chelis_tensor_rank(out[0])!={output_rank}) return 2;
    for (int a=0;a<{output_rank};++a) if(chelis_tensor_shape(out[0],a)!=shape[a]) return 3;
    chelis_read_view v=chelis_tensor_read_view(out[0]);
    if (v.count != sizeof expected / sizeof expected[0]) return 4;
    for(int64_t i=0;i<v.count;++i) {{
        uint64_t got=0; memcpy(&got,(const unsigned char*)v.data+i*chelis_dtype_size({dtype}),chelis_dtype_size({dtype}));
        int index=expected[i]; uint64_t want=bits(index<0 ? -index-1 : index, !{gather} && index>=0);
        if(got!=want) return 5;
    }}
    chelis_tensor_release(out[0]); chelis_tensor_release(base); chelis_tensor_release(indices); chelis_tensor_release(updates);
    puts("SPARSE PASS"); return 0;
}}
"#
                );
                let result = checked_indexing_run(&generated.c_source, &harness);
                assert!(
                    result.status.success(),
                    "{prim:?}/{index_prim:?}: {}\n{}",
                    String::from_utf8_lossy(&result.stderr),
                    generated.c_source
                );
                assert_eq!(result.stdout, b"SPARSE PASS\n");
            }
        }
    }
}

#[test]
fn checked_c_sparse_empty_and_invalid_domains_execute_under_sanitizers() {
    let ty = |shape: &[usize], precision| TensorType {
        dims: shape.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    for (op, diagnostic) in [
        (RiscOp::Gather { axis: 1 }, "gather"),
        (RiscOp::ScatterAdd { axis: 1 }, "scatter"),
        (RiscOp::Scatter { axis: 1 }, "scatter_replace"),
        (RiscOp::ScatterElements { axis: 1 }, "scatter_elements"),
    ] {
        for (empty, selected) in [(true, 0), (false, -1), (false, 3), (false, 2)] {
            let gather = matches!(op, RiscOp::Gather { .. });
            let elements = matches!(op, RiscOp::ScatterElements { .. });
            let n = if empty { 0 } else { 2 };
            let index_shape = if elements { vec![2, n] } else { vec![n] };
            let output_shape = if gather { vec![2, n] } else { vec![2, 3] };
            let mut dag = Dag::new();
            let base = dag.add_node(
                RiscOp::Load {
                    name: "base".into(),
                },
                vec![],
                ty(&[2, 3], Prim::F32),
                None,
            );
            let indices = dag.add_node(
                RiscOp::Load {
                    name: "indices".into(),
                },
                vec![],
                ty(&index_shape, Prim::Int64),
                None,
            );
            let mut inputs = vec![base, indices];
            if !gather {
                inputs.push(dag.add_node(
                    RiscOp::Load {
                        name: "updates".into(),
                    },
                    vec![],
                    ty(&[2, n], Prim::F32),
                    None,
                ));
            }
            let output = dag.add_node(op.clone(), inputs, ty(&output_shape, Prim::F32), None);
            dag.add_root(output);
            let source = codegen(&dag, "sparse_boundary").unwrap().c_source;
            let index_rank = index_shape.len();
            let index_dims = index_shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let input_count = if gather { 2 } else { 3 };
            let check = if empty && gather {
                "if (v.count != 0 || chelis_tensor_shape(out[0],0)!=2 || chelis_tensor_shape(out[0],1)!=0) return 2;".to_string()
            } else {
                let expected = if empty {
                    "0,1,2,3,4,5"
                } else if gather {
                    "2,2,5,5"
                } else if diagnostic == "scatter" {
                    "0,1,5,3,4,12"
                } else {
                    "0,1,2,3,4,4"
                };
                format!(
                    "float expected[]={{{expected}}}; if(v.count != sizeof expected / sizeof expected[0]) return 2; for(int64_t i=0;i<v.count;++i) if(((const float*)v.data)[i]!=expected[i]) return 3;"
                )
            };
            let harness = format!(
                r#"
#include "chelis_runtime.h"
void sparse_boundary(chelis_tensor **,int,chelis_tensor **,int);
int main(void) {{
    chelis_tensor *base=chelis_alloc(2,(int64_t[]){{2,3}},CHELIS_DTYPE_F32);
    chelis_tensor *indices=chelis_alloc({index_rank},(int64_t[]){{{index_dims}}},CHELIS_DTYPE_I64);
    chelis_tensor *updates=chelis_alloc(2,(int64_t[]){{2,{n}}},CHELIS_DTYPE_F32);
    chelis_tensor_write *g=chelis_tensor_begin_write(base); chelis_write_view w=chelis_tensor_write_view(g);
    for(int64_t i=0;i<w.count;++i) ((float*)w.data)[i]=(float)i; chelis_tensor_end_write(g);
    g=chelis_tensor_begin_write(indices); w=chelis_tensor_write_view(g);
    for(int64_t i=0;i<w.count;++i) ((int64_t*)w.data)[i]={selected}; chelis_tensor_end_write(g);
    g=chelis_tensor_begin_write(updates); w=chelis_tensor_write_view(g);
    for(int64_t i=0;i<w.count;++i) ((float*)w.data)[i]=(float)(i+1); chelis_tensor_end_write(g);
    chelis_tensor *in[]={{base,indices,updates}},*out[1]={{0}}; sparse_boundary(in,{input_count},out,1);
    chelis_read_view v=chelis_tensor_read_view(out[0]); {check}
    chelis_tensor_release(out[0]); chelis_tensor_release(base); chelis_tensor_release(indices); chelis_tensor_release(updates);
    puts("SPARSE BOUNDARY PASS"); return 0;
}}
"#
            );
            let mut sources = vec![source];
            if diagnostic == "scatter" {
                use chelis_ir::host::{
                    HostTensorHelper, HostTensorInput, summarize_sparse_helper_for_test,
                };
                let inputs = vec![
                    HostTensorInput {
                        name: "base".into(),
                        ty: ty(&[2, 3], Prim::F32),
                    },
                    HostTensorInput {
                        name: "indices".into(),
                        ty: ty(&index_shape, Prim::Int64),
                    },
                    HostTensorInput {
                        name: "updates".into(),
                        ty: ty(&[2, n], Prim::F32),
                    },
                ];
                let output = ty(&output_shape, Prim::F32);
                let specialization = summarize_sparse_helper_for_test(&dag, &inputs, &output);
                assert!(matches!(
                    specialization,
                    Some(chelis_ir::host::HostTensorSpecialization::SparseScatterAdd(
                        _
                    ))
                ));
                let params = inputs
                    .iter()
                    .map(|input| HostParam {
                        name: input.name.clone(),
                        ty: HostType::Tensor(input.ty.clone()),
                    })
                    .collect::<Vec<_>>();
                let body = HostExpr::new(HostExprKind::TensorCall {
                    helper: 0,
                    args: params
                        .iter()
                        .map(|param| {
                            HostExpr::new(HostExprKind::Var(param.name.clone(), param.ty.clone()))
                        })
                        .collect(),
                    ty: HostType::Tensor(output.clone()),
                });
                let helper = HostTensorHelper {
                    name: "sparse_add_helper".into(),
                    dag: dag.clone(),
                    inputs,
                    output: output.clone(),
                    specialization,
                    summary_rejection: None,
                };
                let program = HostProgram {
                    globals: vec![],
                    global_tensor_helpers: vec![],
                    summary_rejections: vec![],
                    functions: vec![HostFunction {
                        name: "host_sparse_add".into(),
                        params,
                        ret_ty: HostType::Tensor(output),
                        body,
                        tensor_helpers: vec![helper],
                        origin: HostFunctionOrigin::Authored,
                        specialization: None,
                        summary_rejections: vec![],
                    }],
                };
                let mut host_source = emit_host_program(&program, "host_sparse_boundary").unwrap();
                assert!(host_source.contains("CHELIS_SPARSE_ADD"));
                host_source.push_str("\nvoid sparse_boundary(chelis_tensor **in,int n_in,chelis_tensor **out,int n_out) { (void)n_in; (void)n_out; out[0]=host_sparse_add(in[0],in[1],in[2]); }\n");
                sources.push(host_source);
            }
            for source in sources {
                let result = checked_indexing_run(&source, &harness);
                if !empty && selected != 2 {
                    assert!(!result.status.success());
                    let stderr = String::from_utf8_lossy(&result.stderr);
                    assert!(
                        stderr.contains(&format!("numeric trap: domain in {diagnostic} at int64")),
                        "{stderr}"
                    );
                    assert!(
                        !stderr.contains("AddressSanitizer") && !stderr.contains("runtime error:"),
                        "{stderr}"
                    );
                } else {
                    assert!(
                        result.status.success(),
                        "{diagnostic}, empty={empty}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    assert_eq!(result.stdout, b"SPARSE BOUNDARY PASS\n");
                }
            }
        }
    }
}

#[test]
fn checked_c_reduction_nontrailing_kernels_execute_under_sanitizers() {
    let ty = |dims: &[usize], precision| TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    for (op, precision, expected) in [
        (
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            Prim::F32,
            "9,12,27,30",
        ),
        (RiscOp::MaxReduce { axis: 1 }, Prim::F32, "5,6,11,12"),
        (RiscOp::MinReduce { axis: 1 }, Prim::F32, "1,2,7,8"),
        (RiscOp::ProdReduce { axis: 1 }, Prim::F32, "15,48,693,960"),
        (RiscOp::Argmax { axis: 1 }, Prim::Int64, "2,2,2,2"),
        (RiscOp::Argmin { axis: 1 }, Prim::Int64, "0,0,0,0"),
    ] {
        let mut dag = Dag::new();
        let input = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(&[2, 3, 2], Prim::F32),
            None,
        );
        let output = dag.add_node(op, vec![input], ty(&[2, 2], precision), None);
        dag.add_root(output);
        let generated = codegen(&dag, "checked_reduce").unwrap();
        let native = if precision == Prim::Int64 {
            "int64_t"
        } else {
            "float"
        };
        let harness = format!(
            r#"
#include "chelis_runtime.h"
#include <stdio.h>
void checked_reduce(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    int64_t shape[] = {{2,3,2}};
    chelis_tensor *x = chelis_alloc(3, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    float *data = (float*)chelis_tensor_write_view(guard).data;
    for (int i=0;i<12;i++) data[i]=(float)(i+1);
    chelis_tensor_end_write(guard);
    chelis_tensor *inputs[]={{x}}, *outputs[]={{NULL}};
    checked_reduce(inputs,1,outputs,1);
    {native} expected[]={{ {expected} }};
    const {native} *got = (const {native}*)chelis_tensor_read_view(outputs[0]).data;
    if (chelis_tensor_numel(outputs[0]) != 4) return 2;
    for (int i=0;i<4;i++) if(got[i]!=expected[i]) return 3;
    chelis_tensor_release(outputs[0]); chelis_tensor_release(x);
    puts("REDUCTION PASS"); return 0;
}}
"#
        );
        let result = checked_indexing_run(&generated.c_source, &harness);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(result.stdout, b"REDUCTION PASS\n");
    }
}

#[test]
fn checked_c_reduction_sum_and_count_preserve_empty_groups_under_sanitizers() {
    for count in [false, true] {
        for (shape, result_shape, expected_count) in [
            (vec![2, 0, 3], vec![2, 3], 6),
            (vec![0, 3, 2], vec![0, 2], 0),
        ] {
            let mut dag = Dag::new();
            let precision = if count { Prim::Bool } else { Prim::F64 };
            let result_precision = if count { Prim::Int64 } else { Prim::F64 };
            let input = dag.add_node(
                RiscOp::Load { name: "x".into() },
                vec![],
                TensorType {
                    dims: shape.iter().copied().map(DimInfo::Lit).collect(),
                    precision,
                },
                None,
            );
            let op = if count {
                RiscOp::Count { axes: vec![1] }
            } else {
                RiscOp::Sum {
                    axis: 1,
                    accumulator: Prim::F64,
                }
            };
            let output = dag.add_node(
                op,
                vec![input],
                TensorType {
                    dims: result_shape.iter().copied().map(DimInfo::Lit).collect(),
                    precision: result_precision,
                },
                None,
            );
            dag.add_root(output);
            let generated = codegen(&dag, "checked_empty_reduce").unwrap();
            let dtype = if count {
                "CHELIS_DTYPE_BOOL"
            } else {
                "CHELIS_DTYPE_F64"
            };
            let native = if count { "int64_t" } else { "double" };
            let dimensions = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let harness = format!(
                r#"
#include "chelis_runtime.h"
void checked_empty_reduce(chelis_tensor **,int,chelis_tensor **,int);
int main(void) {{
    int64_t shape[]={{ {dimensions} }};
    chelis_tensor *x=chelis_alloc(3,shape,{dtype}), *inputs[]={{x}}, *outputs[]={{NULL}};
    checked_empty_reduce(inputs,1,outputs,1);
    if (chelis_tensor_numel(outputs[0])!={expected_count}) return 2;
    const {native} *values=(const {native}*)chelis_tensor_read_view(outputs[0]).data;
    for(int i=0;i<{expected_count};i++) if(values[i]!=0) return 3;
    chelis_tensor_release(outputs[0]); chelis_tensor_release(x); return 0;
}}
"#
            );
            let result = checked_indexing_run(&generated.c_source, &harness);
            assert!(
                result.status.success(),
                "count={count}, shape={shape:?}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}

#[test]
fn checked_c_movement_permute_and_expand_preserve_bits_under_sanitizers() {
    use chelis_ir::dag::RtDim;
    // Each expected map is explicit, independent of the production coordinate helpers.
    let cases = [
        // A three-cycle distinguishes a permutation from its inverse.
        (
            vec![2, 2, 2],
            vec![2, 2, 2],
            RiscOp::Permute {
                axes: vec![1, 2, 0],
            },
            vec![0, 4, 1, 5, 2, 6, 3, 7],
        ),
        (
            vec![2, 3],
            vec![3, 2],
            RiscOp::Permute { axes: vec![1, 0] },
            vec![0, 3, 1, 4, 2, 5],
        ),
        (
            vec![2, 1],
            vec![2, 3],
            RiscOp::Expand {
                axis: 1,
                size: RtDim::Lit(3),
            },
            vec![0, 0, 0, 1, 1, 1],
        ),
        (
            vec![2],
            vec![3, 2],
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(3),
            },
            vec![0, 1, 0, 1, 0, 1],
        ),
        (vec![], vec![], RiscOp::Permute { axes: vec![] }, vec![0]),
        (
            vec![],
            vec![3],
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(3),
            },
            vec![0, 0, 0],
        ),
        (
            vec![2, 0, 3],
            vec![3, 2, 0],
            RiscOp::Permute {
                axes: vec![2, 0, 1],
            },
            vec![],
        ),
        (
            vec![2, 1],
            vec![2, 0],
            RiscOp::Expand {
                axis: 1,
                size: RtDim::Lit(0),
            },
            vec![],
        ),
        (
            vec![2, 1, 1, 1, 1, 1, 1, 1, 3],
            vec![3, 1, 1, 1, 1, 1, 1, 1, 2],
            RiscOp::Permute {
                axes: (0..9).rev().collect(),
            },
            vec![0, 3, 1, 4, 2, 5],
        ),
    ];
    for (prim, dtype, seed) in [
        (Prim::F32, "CHELIS_DTYPE_F32", 0x3f80_0000_u64),
        (Prim::F64, "CHELIS_DTYPE_F64", 0x3ff0_0000_0000_0000),
        (Prim::F16, "CHELIS_DTYPE_F16", 0x3c00),
        (Prim::Bf16, "CHELIS_DTYPE_BF16", 0x3f80),
        (Prim::Int64, "CHELIS_DTYPE_I64", 9_007_199_254_740_993),
        (Prim::Int32, "CHELIS_DTYPE_I32", 100),
        (Prim::Int16, "CHELIS_DTYPE_I16", 100),
        (Prim::Int8, "CHELIS_DTYPE_I8", 100),
        (Prim::Bool, "CHELIS_DTYPE_BOOL", 0),
    ] {
        for (input_shape, output_shape, op, expected) in &cases {
            let ty = |shape: &[usize]| TensorType {
                dims: shape.iter().copied().map(DimInfo::Lit).collect(),
                precision: prim,
            };
            let mut dag = Dag::new();
            let input = dag.add_node(
                RiscOp::Load { name: "x".into() },
                vec![],
                ty(input_shape),
                None,
            );
            let output = dag.add_node(op.clone(), vec![input], ty(output_shape), None);
            dag.add_root(output);
            let generated = codegen(&dag, "checked_movement").unwrap();
            assert!(generated.c_source.contains("chelis_movement_index("));
            assert!(generated.c_source.contains("chelis_movement_plan_release("));
            let spell = |values: &[usize]| {
                if values.is_empty() {
                    "0".into()
                } else {
                    values
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            };
            let input_dims = spell(input_shape);
            let output_dims = spell(output_shape);
            let map = spell(expected);
            let rank_in = input_shape.len();
            let rank_out = output_shape.len();
            let count_in = input_shape.iter().product::<usize>();
            let count_out = expected.len();
            let bits = if prim == Prim::Bool {
                "(uint64_t)(i % 2)".into()
            } else {
                format!("UINT64_C({seed}) + (uint64_t)i")
            };
            let harness = format!(
                r#"
#include "chelis_runtime.h"
#include <string.h>
void checked_movement(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    int64_t in_dims[] = {{ {input_dims} }}, out_dims[] = {{ {output_dims} }}, map[] = {{ {map} }};
    chelis_tensor *x = chelis_alloc({rank_in}, in_dims, {dtype});
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    unsigned char *data = (unsigned char *)chelis_tensor_write_view(guard).data;
    size_t width = (size_t)chelis_dtype_size({dtype});
    for (int64_t i = 0; i < {count_in}; ++i) {{
        uint64_t bits = {bits}; memcpy(data + (size_t)i * width, &bits, width);
    }}
    chelis_tensor_end_write(guard);
    chelis_tensor *inputs[] = {{x}}, *outputs[1] = {{NULL}};
    checked_movement(inputs, 1, outputs, 1);
    chelis_read_view in = chelis_tensor_read_view(x), out = chelis_tensor_read_view(outputs[0]);
    if (out.count != {count_out} || out.dtype != {dtype} || chelis_tensor_rank(outputs[0]) != {rank_out}) return 2;
    for (int32_t axis = 0; axis < {rank_out}; ++axis)
        if (chelis_tensor_shape(outputs[0], axis) != out_dims[axis]) return 3;
    for (int64_t i = 0; i < out.count; ++i)
        if (memcmp((const unsigned char *)out.data + (size_t)i * width,
            (const unsigned char *)in.data + (size_t)map[i] * width, width)) return 4;
    chelis_tensor_release(outputs[0]); chelis_tensor_release(x);
    puts("CHECKED MOVEMENT PASS"); return 0;
}}
"#
            );
            let run = checked_indexing_run(&generated.c_source, &harness);
            assert!(
                run.status.success(),
                "{prim:?} {op:?}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            assert_eq!(
                String::from_utf8_lossy(&run.stdout),
                "CHECKED MOVEMENT PASS\n"
            );
            if prim == Prim::Int64
                && (input_shape == &[2, 3] || input_shape == &[2, 1] && count_out > 0)
            {
                let check = generated
                    .c_source
                    .lines()
                    .find(|line| line.contains("chelis_movement_check_target("))
                    .unwrap();
                let allocation = generated
                    .c_source
                    .lines()
                    .find(|line| line.contains("chelis_tensor *t1 = chelis_alloc("))
                    .unwrap();
                assert!(
                    generated.c_source.find(check).unwrap()
                        < generated.c_source.find(allocation).unwrap()
                );
                let invalid = check.replace("CHELIS_DTYPE_I64, 3)", "CHELIS_DTYPE_I64, 4)");
                assert_ne!(invalid, check);
                // Observe the real allocation site without allocating: malformed metadata
                // must trap first. Removing or moving validation exposes that site.
                let marker = format!("exit(78); /* allocation reached */\n{allocation}");
                let instrumented = generated.c_source.replace(allocation, &marker);
                let rejected =
                    checked_indexing_run(&instrumented.replace(check, &invalid), &harness);
                assert!(!rejected.status.success());
                assert!(String::from_utf8_lossy(&rejected.stderr).contains("Domain"));
                for replacement in [String::new(), invalid.clone()] {
                    let bypass = instrumented.replace(check, "");
                    let bypass = bypass.replace(&marker, &format!("{marker}\n{replacement}"));
                    let run = checked_indexing_run(&bypass, &harness);
                    assert_eq!(
                        run.status.code(),
                        Some(78),
                        "missing or late validation reached allocation"
                    );
                }
            }
            if prim == Prim::Int64 && input_shape == &[2, 2, 2] {
                let axis_line = generated
                    .c_source
                    .lines()
                    .find(|line| line.contains("t1_axes[3]"))
                    .unwrap();
                let inverse = generated.c_source.replace(axis_line,
                    "chelis_scalar t1_axes[3] = {chelis_scalar_from_bits(CHELIS_DTYPE_I64, 2), chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0), chelis_scalar_from_bits(CHELIS_DTYPE_I64, 1)};");
                assert_ne!(inverse, generated.c_source);
                let run = checked_indexing_run(&inverse, &harness);
                assert_eq!(
                    run.status.code(),
                    Some(4),
                    "inverse permutation must corrupt the exact result"
                );
            }
            if prim == Prim::Int64 && input_shape == &[2, 3] {
                let anchor = "chelis_movement_index(t1_movement, chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)i))";
                assert!(generated.c_source.contains(anchor));
                let mutant = generated
                    .c_source
                    .replace(anchor, "0 /* unchecked coordinate bypass */");
                let run = checked_indexing_run(&mutant, &harness);
                assert_eq!(
                    run.status.code(),
                    Some(4),
                    "coordinate bypass must corrupt the exact result"
                );
            }
        }
    }
}

#[test]
fn checked_c_movement_affine_maps_preserve_bits_under_sanitizers() {
    use chelis_ir::dag::RtDim;
    for (prim, dtype, seed, fill_bits) in [
        (
            Prim::F32,
            "CHELIS_DTYPE_F32",
            0x3f80_0000_u64,
            0x3f80_0000_u64,
        ),
        (
            Prim::F64,
            "CHELIS_DTYPE_F64",
            0x3ff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
        ),
        (Prim::F16, "CHELIS_DTYPE_F16", 0x3c00, 0x3c00),
        (Prim::Bf16, "CHELIS_DTYPE_BF16", 0x3f80, 0x3f80),
        (Prim::Int64, "CHELIS_DTYPE_I64", 9_007_199_254_740_993, 1),
        (Prim::Int32, "CHELIS_DTYPE_I32", 100, 1),
        (Prim::Int16, "CHELIS_DTYPE_I16", 100, 1),
        (Prim::Int8, "CHELIS_DTYPE_I8", 100, 1),
        (Prim::Bool, "CHELIS_DTYPE_BOOL", 0, 1),
    ] {
        let pairs = |v: &[(usize, usize)]| {
            v.iter()
                .map(|&(a, b)| (RtDim::Lit(a), RtDim::Lit(b)))
                .collect()
        };
        let fill = chelis_types::scalar_from_i64("pad", prim, 1).unwrap();
        let mut padded = vec![-1_i64; 18];
        for (input, output) in [8, 9, 10, 14, 15, 16].into_iter().enumerate() {
            padded[output] = input as i64;
        }
        let cases = [
            (
                vec![2, 3],
                vec![3, 6],
                RiscOp::Pad {
                    padding: pairs(&[(1, 0), (2, 1)]),
                    fill,
                },
                padded,
            ),
            (
                vec![3, 4],
                vec![2, 3],
                RiscOp::Shrink {
                    bounds: pairs(&[(1, 3), (1, 4)]),
                },
                vec![5, 6, 7, 9, 10, 11],
            ),
            (
                vec![3, 5],
                vec![2, 2],
                RiscOp::Stride {
                    strides: vec![RtDim::Lit(2), RtDim::Lit(3)],
                },
                vec![0, 3, 10, 13],
            ),
            (
                vec![0, 3],
                vec![1, 3],
                RiscOp::Pad {
                    padding: pairs(&[(0, 1), (0, 0)]),
                    fill,
                },
                vec![-1; 3],
            ),
            (
                vec![2, 3],
                vec![0, 3],
                RiscOp::Shrink {
                    bounds: pairs(&[(1, 1), (0, 3)]),
                },
                vec![],
            ),
            (
                vec![2, 0],
                vec![1, 0],
                RiscOp::Stride {
                    strides: vec![RtDim::Lit(2), RtDim::Lit(1)],
                },
                vec![],
            ),
            (
                vec![],
                vec![],
                RiscOp::Pad {
                    padding: vec![],
                    fill,
                },
                vec![0],
            ),
            (
                vec![1; 9],
                vec![1; 9],
                RiscOp::Stride {
                    strides: vec![RtDim::Lit(2); 9],
                },
                vec![0],
            ),
        ];
        for (input_shape, output_shape, op, expected) in cases {
            let ty = |shape: &[usize]| TensorType {
                dims: shape.iter().copied().map(DimInfo::Lit).collect(),
                precision: prim,
            };
            let mut dag = Dag::new();
            let x = dag.add_node(
                RiscOp::Load { name: "x".into() },
                vec![],
                ty(&input_shape),
                None,
            );
            let y = dag.add_node(op.clone(), vec![x], ty(&output_shape), None);
            dag.add_root(y);
            let generated = codegen(&dag, "affine_movement").unwrap();
            let spell = |values: Vec<i64>| {
                if values.is_empty() {
                    "0".into()
                } else {
                    values
                        .iter()
                        .map(i64::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                }
            };
            let input_dims = spell(input_shape.iter().map(|&n| n as i64).collect());
            let output_dims = spell(output_shape.iter().map(|&n| n as i64).collect());
            let map = spell(expected.clone());
            let input_rank = input_shape.len();
            let output_rank = output_shape.len();
            let input_count: usize = input_shape.iter().product();
            let output_count = expected.len();
            let bits = if prim == Prim::Bool {
                "(uint64_t)(i % 2)".into()
            } else {
                format!("UINT64_C({seed}) + (uint64_t)i")
            };
            let harness = format!(
                r#"
#include "chelis_runtime.h"
#include <string.h>
void affine_movement(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    int64_t input_shape[] = {{{input_dims}}}, output_shape[] = {{{output_dims}}}, map[] = {{{map}}};
    chelis_tensor *x = chelis_alloc({input_rank}, input_shape, {dtype});
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    unsigned char *data = chelis_tensor_write_view(guard).data;
    size_t width = (size_t)chelis_dtype_size({dtype});
    for (int64_t i=0; i<{input_count}; ++i) {{uint64_t bits = {bits}; memcpy(data + i*width, &bits, width);}}
    chelis_tensor_end_write(guard);
    chelis_tensor *inputs[] = {{x}}, *outputs[] = {{NULL}};
    affine_movement(inputs, 1, outputs, 1);
    chelis_read_view in = chelis_tensor_read_view(x), out = chelis_tensor_read_view(outputs[0]);
    if (out.count != {output_count} || out.dtype != {dtype} || chelis_tensor_rank(outputs[0]) != {output_rank}) return 2;
    for (int axis=0; axis<{output_rank}; ++axis) if (chelis_tensor_shape(outputs[0], axis) != output_shape[axis]) return 3;
    uint64_t fill = UINT64_C({fill_bits});
    for (int64_t i=0; i<out.count; ++i) {{
        const void *expected = map[i] < 0 ? (const void*)&fill : (const unsigned char*)in.data + map[i]*width;
        if (memcmp((const unsigned char*)out.data + i*width, expected, width)) return 4;
    }}
    chelis_tensor_release(outputs[0]); chelis_tensor_release(x);
    puts("AFFINE PASS"); return 0;
}}
"#
            );
            let run = checked_indexing_run(&generated.c_source, &harness);
            assert!(
                run.status.success(),
                "{prim:?} {op:?}: {}",
                String::from_utf8_lossy(&run.stderr)
            );
            assert_eq!(run.stdout, b"AFFINE PASS\n");
        }
    }
}

#[test]
fn checked_c_movement_runtime_affine_bounds_reject_before_allocation() {
    use chelis_ir::dag::RtDim;
    for (name, op, cases) in [
        (
            "pad",
            RiscOp::Pad {
                padding: vec![(RtDim::Node(1), RtDim::Lit(0))],
                fill: chelis_types::scalar_from_i64("pad", Prim::Int64, 1).unwrap(),
            },
            vec![(2, Some(5)), (-1, None), (i64::MAX, None)],
        ),
        (
            "shrink",
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Node(1))],
            },
            vec![(2, Some(2)), (0, None), (-1, None), (4, None)],
        ),
        (
            "stride",
            RiscOp::Stride {
                strides: vec![RtDim::Node(1)],
            },
            vec![(2, Some(2)), (i64::MAX, Some(1)), (0, None), (-1, None)],
        ),
    ] {
        let ty = |dims| TensorType {
            dims,
            precision: Prim::Int64,
        };
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(3)]),
            None,
        );
        let n = dag.add_node(RiscOp::Load { name: "n".into() }, vec![], ty(vec![]), None);
        let y = dag.add_node(
            op,
            vec![x, n],
            ty(vec![DimInfo::Named("result".into(), None)]),
            None,
        );
        dag.add_root(y);
        let generated = codegen(&dag, "dynamic_affine").unwrap();
        for (bound, expected) in cases {
            let expected_count = expected.unwrap_or(0);
            let padding = name == "pad";
            let harness = format!(
                r#"
#include "chelis_runtime.h"
void dynamic_affine(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    int64_t shape[] = {{3}};
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_DTYPE_I64), *n = chelis_alloc(0, NULL, CHELIS_DTYPE_I64);
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    chelis_fill_scalar(guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, 7)); chelis_tensor_end_write(guard);
    guard = chelis_tensor_begin_write(n);
    chelis_fill_scalar(guard, chelis_scalar_from_bits(CHELIS_DTYPE_I64, UINT64_C({bits}))); chelis_tensor_end_write(guard);
    chelis_tensor *inputs[] = {{x,n}}, *outputs[] = {{NULL}};
    dynamic_affine(inputs,2,outputs,1);
    chelis_read_view out = chelis_tensor_read_view(outputs[0]);
    if (out.count != {expected_count}) return 4;
    for (int64_t i=0; i<out.count; ++i) if (((const int64_t*)out.data)[i] != ({pad} && i < {bound} ? 1 : 7)) return 5;
    chelis_tensor_release(outputs[0]); chelis_tensor_release(x); chelis_tensor_release(n);
    puts("DYNAMIC AFFINE PASS"); return 0;
}}
"#,
                bits = bound as u64,
                pad = i32::from(padding)
            );
            let source = if expected.is_none() {
                let allocation = generated
                    .c_source
                    .lines()
                    .find(|line| line.contains(" = chelis_alloc("))
                    .expect("output allocation");
                generated.c_source.replacen(
                    allocation,
                    &format!("exit(78); /* allocation reached */\n{allocation}"),
                    1,
                )
            } else {
                generated.c_source.clone()
            };
            let run = checked_indexing_run(&source, &harness);
            if expected.is_some() {
                assert!(
                    run.status.success(),
                    "{name} {bound}: {}",
                    String::from_utf8_lossy(&run.stderr)
                );
                assert_eq!(run.stdout, b"DYNAMIC AFFINE PASS\n");
            } else {
                assert!(!run.status.success(), "{name} {bound}: {run:?}");
                assert_ne!(
                    run.status.code(),
                    Some(78),
                    "allocation preceded rejection: {name} {bound}"
                );
                let class = if bound == i64::MAX {
                    "overflow"
                } else {
                    "domain"
                };
                assert!(
                    String::from_utf8_lossy(&run.stderr)
                        .ends_with(&format!("numeric trap: {class} in {name} at int64\n")),
                    "{run:?}"
                );
            }
        }
    }
}

#[test]
fn checked_c_indexing_dag_scalar_fused_reuse_and_empty_execute_under_sanitizers() {
    use chelis_ir::dag::{FusedInput, FusedStep, FusedStepOp};
    for prim in [Prim::F32, Prim::F64] {
        for shape in [vec![], vec![4], vec![0], vec![1; 9]] {
            for fused in [false, true] {
                let ty = TensorType {
                    dims: shape.iter().copied().map(DimInfo::Lit).collect(),
                    precision: prim,
                };
                let scalar_ty = if fused {
                    TensorType {
                        dims: vec![],
                        precision: prim,
                    }
                } else {
                    ty.clone()
                };
                let mut dag = Dag::new();
                let input =
                    dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
                let scalar =
                    dag.add_node(RiscOp::Load { name: "s".into() }, vec![], scalar_ty, None);
                let negated = dag.add_node(RiscOp::Neg, vec![input], ty.clone(), None);
                let op = if fused {
                    RiscOp::FusedElem {
                        ops: vec![FusedStep {
                            op: FusedStepOp::Add,
                            input_indices: vec![FusedInput::External(0), FusedInput::External(1)],
                        }],
                    }
                } else {
                    RiscOp::Add
                };
                let result = dag.add_node(op, vec![negated, scalar], ty, None);
                if fused {
                    dag.set_reusable_input(result, negated);
                }
                dag.add_root(result);
                let generated = codegen(&dag, "checked_indexing").unwrap();
                assert!(
                    generated
                        .c_source
                        .contains("chelis_tensor_elementwise_index_step_for_shape(")
                );
                if fused && !shape.contains(&0) {
                    assert!(
                        generated.c_source.contains("chelis_tensor_repurpose("),
                        "fixture must exercise storage reuse"
                    );
                }
                let inputs = generated
                    .input_labels
                    .iter()
                    .map(|name| name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                let elem = if prim == Prim::F32 { "float" } else { "double" };
                let dtype = if prim == Prim::F32 {
                    "CHELIS_DTYPE_F32"
                } else {
                    "CHELIS_DTYPE_F64"
                };
                let dims = if shape.is_empty() {
                    "1".into()
                } else {
                    shape
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let rank = shape.len();
                let count = shape.iter().product::<usize>();
                let scalar_rank = if fused { 0 } else { rank };
                let scalar_count = if fused { 1 } else { count };
                let harness = format!(
                    r#"
#include "chelis_runtime.h"
void checked_indexing(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    int64_t dims[] = {{ {dims} }};
    chelis_tensor *x = chelis_alloc({rank}, dims, {dtype});
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    {elem} *data = ({elem} *)chelis_tensor_write_view(guard).data;
    for (int64_t i = 0; i < {count}; ++i) data[i] = ({elem})(i + 1);
    chelis_tensor_end_write(guard);
    chelis_tensor *s = chelis_alloc({scalar_rank}, dims, {dtype});
    guard = chelis_tensor_begin_write(s);
    for (int64_t i = 0; i < {scalar_count}; ++i) (({elem} *)chelis_tensor_write_view(guard).data)[i] = 10;
    chelis_tensor_end_write(guard);
    chelis_tensor *inputs[] = {{ {inputs} }}, *outputs[1] = {{NULL}};
    checked_indexing(inputs, 2, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    if (view.count != {count} || chelis_tensor_rank(outputs[0]) != {rank}) return 2;
    for (int64_t i = 0; i < view.count; ++i)
        if (((const {elem} *)view.data)[i] != 9 - i) return 3;
    chelis_tensor_release(outputs[0]); chelis_tensor_release(x); chelis_tensor_release(s);
    puts("CHECKED INDEX PASS");
    return 0;
}}
"#
                );
                let run = checked_indexing_run(&generated.c_source, &harness);
                assert!(
                    run.status.success(),
                    "{prim:?} {shape:?} fused={fused}: {}\n{}",
                    String::from_utf8_lossy(&run.stderr),
                    generated.c_source
                );
                assert_eq!(String::from_utf8_lossy(&run.stdout), "CHECKED INDEX PASS\n");
                if fused && prim == Prim::F32 && shape == vec![4] {
                    // Execute the actual emitted fast-path condition with its
                    // scalar protection removed. ASan must observe the scalar
                    // read beyond its allocation; a source-only check is insufficient.
                    let anchor = "t3_input1_step == 1";
                    assert!(generated.c_source.contains(anchor));
                    let bad_fast =
                        checked_indexing_run(&generated.c_source.replace(anchor, "1"), &harness);
                    assert!(!bad_fast.status.success());
                    assert!(String::from_utf8_lossy(&bad_fast.stderr).contains("AddressSanitizer"));

                    // A same-capacity header change must be rejected while the
                    // original shape remains observable. Moving validation past
                    // fused repurpose erases it and makes this witness return success.
                    let lines: Vec<_> = generated
                        .c_source
                        .lines()
                        .filter(|line| {
                            line.contains("const int64_t t3_input") && line.contains("_step =")
                        })
                        .collect();
                    assert_eq!(lines.len(), 2);
                    let steps = format!("{}\n{}\n", lines[0], lines[1]);
                    assert!(generated.c_source.contains(&steps));
                    let change = r#"
    chelis_tensor_end_write(t2_write_guard);
    chelis_tensor_repurpose(t2, chelis_scalar_from_bits(CHELIS_DTYPE_I64, 2),
        (chelis_scalar[]){chelis_scalar_from_bits(CHELIS_DTYPE_I64, 2), chelis_scalar_from_bits(CHELIS_DTYPE_I64, 2)});
    t2_write_guard = chelis_tensor_begin_write(t2);
    t2_data = chelis_tensor_write_view(t2_write_guard).data;
"#;
                    let changed = generated
                        .c_source
                        .replace(&steps, &format!("{change}{steps}"));
                    let run = checked_indexing_run(&changed, &harness);
                    assert!(!run.status.success());
                    assert!(
                        String::from_utf8_lossy(&run.stderr)
                            .contains("Domain: chelis_tensor_elementwise_index_step_for_shape")
                    );
                    let rebound = "t2_data = t3_data;";
                    assert_eq!(changed.matches(rebound).count(), 1);
                    let late = changed
                        .replace(&steps, "")
                        .replace(rebound, &format!("{rebound}\n{steps}"));
                    let run = checked_indexing_run(&late, &harness);
                    assert!(
                        run.status.success(),
                        "late-validation mutation did not erase the witness: {}",
                        String::from_utf8_lossy(&run.stderr)
                    );
                }
            }
        }
    }
}

#[test]
fn checked_c_indexing_host_scalar_projection_and_reversed_domain_execute_under_sanitizers() {
    for builtin in ["add", "max_elem"] {
        for reversed in [false, true] {
            let (lhs, rhs) = if reversed {
                (vec![], vec![4])
            } else {
                (vec![4], vec![])
            };
            let source = emit_host_program(
                &host_binary_program(builtin, lhs, rhs),
                "checked_host_indexing",
            )
            .unwrap();
            assert!(source.contains("chelis_tensor_elementwise_index_step("));
            assert!(!source.contains("chelis_host_indices_to_flat"));
            let args = if reversed { "s, x" } else { "x, s" };
            let expected = if builtin == "add" { "i + 11" } else { "10" };
            let harness = format!(
                r#"
#include "chelis_runtime.h"
chelis_tensor *the_fn(chelis_tensor *, chelis_tensor *);
int main(void) {{
    int64_t dim = 4;
    chelis_tensor *x = chelis_alloc(1, &dim, CHELIS_DTYPE_F32);
    chelis_tensor *s = chelis_alloc(0, NULL, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    float *data = (float *)chelis_tensor_write_view(guard).data;
    for (int i = 0; i < 4; ++i) data[i] = i + 1;
    chelis_tensor_end_write(guard);
    guard = chelis_tensor_begin_write(s);
    *(float *)chelis_tensor_write_view(guard).data = 10;
    chelis_tensor_end_write(guard);
    chelis_tensor *out = the_fn({args});
    chelis_read_view view = chelis_tensor_read_view(out);
    if (view.count != 4) return 2;
    for (int i = 0; i < 4; ++i) if (((const float *)view.data)[i] != {expected}) return 3;
    chelis_tensor_release(out); chelis_tensor_release(x); chelis_tensor_release(s);
    puts("CHECKED HOST PASS");
    return 0;
}}
"#
            );
            let run = checked_indexing_run(&source, &harness);
            let stderr = String::from_utf8_lossy(&run.stderr);
            if reversed {
                assert!(!run.status.success());
                assert!(
                    stderr.contains("Domain: chelis_tensor_elementwise_index_step"),
                    "{stderr}"
                );
            } else {
                assert!(run.status.success(), "{stderr}\n{source}");
                assert_eq!(String::from_utf8_lossy(&run.stdout), "CHECKED HOST PASS\n");
            }
        }
    }
}

fn vec_f32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

/// Host-arch SIMD ISA flag(s) for the compile-run probes.
///
/// `chelis_simd.h` and the generated kernels are arch-aware (`#ifdef
/// __AVX2__` on x86, `#elif defined(__ARM_NEON)` on ARM, scalar
/// fallback otherwise). On x86_64 we pass `-mavx2` to exercise the AVX2
/// path; on aarch64 NEON is a baseline ISA feature (so `__ARM_NEON` is
/// already defined and the NEON path activates with no flag), and
/// `-mavx2` is an `unsupported option` clang error there. Returning an
/// empty vector on non-x86 keeps the probe portable so it runs via NEON
/// (Apple Silicon CI) or the scalar fallback rather than failing the
/// build.
fn simd_isa_flags() -> Vec<String> {
    if cfg!(target_arch = "x86_64") {
        vec!["-mavx2".to_string()]
    } else {
        Vec::new()
    }
}

/// Locate `target/debug/` for this workspace by walking up from the test binary's
/// own location. The test binary lives at `<target>/debug/deps/<binary>`, so
/// the parent of its parent is the debug directory we want.
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    // exe = <target>/debug/deps/<test_bin>
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

/// Ensure `target/debug/libchelis_runtime.a` exists. When `chelis-runtime` is built
/// transitively as a dev-dependency (rather than as the top-level package), cargo
/// only emits the staticlib to `target/debug/deps/libchelis_runtime-<hash>.a` and
/// does not promote it to the conventional `target/debug/libchelis_runtime.a` path.
/// The test gcc invocation links against the conventional path, so this helper
/// copies the hashed artifact into place on first use. Idempotent and
/// thread-safe.
fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    let deps_dir = canonical
        .parent()
        .expect("canonical lib path has no parent")
        .join("deps");
    // First-pass scan of the deps dir.
    let hashed = find_newest_runtime_archive(&deps_dir)?;
    // If cargo's incremental cache reused the rlib without re-emitting
    // the staticlib (observed on CI cold-cache runs against
    // `chelis-runtime` as a transitive dev-dep), force a rebuild of
    // the lib target and rescan. `cargo build -p chelis-runtime --lib`
    // emits both crate-types declared in chelis-runtime/Cargo.toml,
    // producing the `libchelis_runtime-<hash>.a` artifact the
    // gcc-link harness needs.
    let hashed = match hashed {
        Some(path) => path,
        None => {
            std::process::Command::new(env!("CARGO"))
                .args(["build", "-p", "chelis-runtime", "--lib"])
                .status()
                .map_err(|e| std::io::Error::other(format!("cargo build chelis-runtime: {e}")))?;
            find_newest_runtime_archive(&deps_dir)?.ok_or_else(|| {
                std::io::Error::other(format!(
                    "no libchelis_runtime-*.a found in {} after explicit `cargo build -p \
                     chelis-runtime --lib`",
                    deps_dir.display()
                ))
            })?
        }
    };
    if canonical.exists() && canonical.metadata()?.modified()? >= hashed.metadata()?.modified()? {
        return Ok(());
    }
    // Use a PID-suffixed tmp filename so concurrent test binaries (this
    // file and dtype_matrix_bf16_f16.rs both call into this helper, and
    // nextest runs them in parallel) do not race on a shared tmp path.
    // Each process writes its own tmp and renames into the shared
    // canonical location; last writer wins, but the content is
    // identical so the race is harmless. Without the PID, two
    // processes that interleave `fs::copy` and `fs::rename` produce an
    // ENOENT on the second rename because the first rename moved the
    // shared tmp away.
    static NEXT_TEMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let tmp = canonical.with_extension(format!(
        "a.tmp.{}.{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::copy(&hashed, &tmp)?;
    // The rename can still race with another process renaming its own
    // unique tmp into the same canonical path. On POSIX, rename onto an
    // existing file is atomic, so this is fine. If a peer beat us to
    // it, treat NotFound from a follow-up cleanup as benign.
    match fs::rename(&tmp, canonical) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && canonical.exists() => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn find_newest_runtime_archive(deps_dir: &Path) -> std::io::Result<Option<PathBuf>> {
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = match fs::read_dir(deps_dir) {
        Ok(it) => it,
        // Truly cold target dirs may not have `deps/` yet; let the
        // caller fall through to the explicit `cargo build` fallback.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
            let meta = entry.metadata()?;
            let mtime = meta.modified()?;
            match &newest {
                Some((cur, _)) if *cur >= mtime => {}
                _ => newest = Some((mtime, entry.path())),
            }
        }
    }
    Ok(newest.map(|(_, p)| p))
}

fn runtime_lib_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let canonical = target_debug_dir().join("libchelis_runtime.a");
        if let Err(e) = ensure_runtime_static_lib(&canonical) {
            panic!(
                "failed to materialize libchelis_runtime.a at {}: {}",
                canonical.display(),
                e
            );
        }
        canonical
    })
    .clone()
}

/// Write generated C + harness, compile, run, return stdout. None = compile/run failure.
/// Compile and run like [`compile_and_run_kernel`], but return the exit
/// status and BOTH streams instead of `None` on failure.
///
/// chelis#1277: a runtime extent guard's whole observable is a nonzero exit
/// with a trap line on stderr, which the success-only helper discards - it
/// `eprintln!`s stderr and returns `None`, so a caller cannot assert on the
/// trap it was testing for. This shares that helper's compile plumbing rather
/// than duplicating the runtime-library discovery and include copying.
fn compile_and_run_kernel_capturing(
    test_name: &str,
    c_source: &str,
    harness: &str,
) -> (bool, String) {
    // chelis#1492's collision class: a probe path built here rather than
    // through `common::probe_dir` is a shared directory two concurrent tests
    // can land in. `probe_dir_discipline` scans for exactly this line.
    let probe = common::probe_dir(&format!("exec_{test_name}"));
    let dir = probe.path().to_path_buf();
    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();
    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }
    let bin = dir.join("trap_bin");
    let compile = Command::new("gcc")
        .arg("-O2")
        .args(simd_isa_flags())
        .args([
            "-std=c11",
            "-I",
            dir.to_str().unwrap(),
            dir.join("kernel.c").to_str().unwrap(),
            dir.join("main.c").to_str().unwrap(),
            runtime_lib_path().to_str().unwrap(),
            "-lm",
            "-o",
            bin.to_str().unwrap(),
        ])
        .output()
        .expect("failed to invoke gcc");
    // A compile failure is never a skip. The success-only helper returns
    // `None` for it, and a caller that treats `None` as "no toolchain" then
    // passes vacuously on a kernel that did not build - so this one fails
    // loudly instead, and its callers need no escape branch.
    assert!(
        compile.status.success(),
        "COMPILE FAILED [{test_name}]:\n{}\nKernel C:\n{c_source}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&bin).output().expect("failed to run binary");
    let mut text = String::from_utf8_lossy(&run.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.success(), text)
}

fn compile_and_run_kernel(test_name: &str, c_source: &str, harness: &str) -> Option<String> {
    let probe = common::probe_dir(&format!("exec_{test_name}"));
    let dir = probe.path().to_path_buf();

    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let bin = dir.join("test_bin");
    let runtime_lib = runtime_lib_path();

    let compile = Command::new("gcc")
        .arg("-O2")
        .args(simd_isa_flags())
        .args([
            "-std=c11",
            "-I",
            dir.to_str().unwrap(),
            dir.join("kernel.c").to_str().unwrap(),
            dir.join("main.c").to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime_lib.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("failed to invoke gcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        eprintln!("COMPILE FAILED [{test_name}]:\n{stderr}");
        eprintln!("Kernel C:\n{c_source}");
        return None;
    }

    let run = Command::new(&bin).output().expect("failed to run binary");
    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        let stdout = String::from_utf8_lossy(&run.stdout);
        eprintln!("RUN FAILED [{test_name}]\nstdout: {stdout}\nstderr: {stderr}");
        return None;
    }

    Some(String::from_utf8_lossy(&run.stdout).into_owned())
}

// Common harness header: wrap caller-owned storage in an exact tensor descriptor.
const HARNESS_HEADER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "chelis_runtime.h"

static chelis_tensor *make_view_typed_1d(void* data, int64_t n, chelis_dtype dtype) {
    int64_t shape[1] = {n};
    return chelis_tensor_entry_borrow(
        1, shape, dtype, data, n * chelis_dtype_size(dtype)
    );
}

static chelis_tensor *make_view_1d(float* data, int64_t n) {
    return make_view_typed_1d(data, n, CHELIS_DTYPE_F32);
}
"#;

// ---- Test 6 / Item 6: MathLib::None exp kernel ----

#[test]
fn exec_math_none_exp_kernel_correct_output() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(4), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_none",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    assert!(
        !src.contains("CHELIS_HAS_SLEEF"),
        "MathLib::None must not emit Sleef guard"
    );
    assert!(
        !src.contains("chelis_math.h"),
        "MathLib::None must not include chelis_math.h"
    );
    assert!(
        src.contains("expf("),
        "MathLib::None must use scalar expf()"
    );
    assert!(
        src.contains("#pragma omp parallel for simd"),
        "MathLib::None must use Level-1 omp simd"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_exp_none(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[4] = {{0.0f, 1.0f, 2.0f, -1.0f}};
    chelis_tensor *in_t = make_view_1d(in_data, 4);
    chelis_tensor* in_ptr = in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_exp_none(inputs, 1, outputs, 1);

    float expected[4] = {{1.0f, 2.718282f, 7.389056f, 0.367879f}};
    int ok = 1;
    for (int i = 0; i < 4; i++) {{
        float got = ((float*)chelis_tensor_read_view(outputs[0]).data)[i];
        float reldiff = fabsf(got - expected[i]) / (fabsf(expected[i]) + 1e-6f);
        if (reldiff > 1e-4f) {{
            printf("MISMATCH at %d: got %.6f expected %.6f\n", i, got, expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("none_exp", src, &harness) else {
        panic!("MathLib::None exp kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "MathLib::None exp kernel wrong output:\n{output}"
    );
}

// ---- Test 4: Sleef kernel scalar fallback (without -DCHELIS_HAS_SLEEF) ----
//
// FINDING: A single Load->Exp DAG does NOT generate the Sleef path because
// fuse() only fuses chains of length >= 2.  The Sleef path lives in
// emit_fused_elem which is only called for FusedElem nodes.  A single Exp
// goes through emit_unary_func which never emits Sleef.  This test uses a
// 2-op chain (Exp -> Neg) so that fuse() produces a FusedElem node.
#[test]
fn exec_sleef_kernel_scalar_fallback_correct() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(9), None);
    let e = dag.add_node(RiscOp::Exp, vec![a], vec_f32(9), None);
    dag.add_node(RiscOp::Neg, vec![e], vec_f32(9), None); // 2-op chain: fuses into FusedElem
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_sleef",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("#ifdef CHELIS_HAS_SLEEF"),
        "Missing Sleef guard"
    );
    assert!(src.contains("CHELIS_EXPF8("), "Missing CHELIS_EXPF8");
    assert!(src.contains("_mm256_loadu_ps("), "Missing AVX2 load");
    assert!(src.contains("_mm256_storeu_ps("), "Missing AVX2 store");
    assert!(src.contains("for (; __i < "), "Missing scalar tail");
    assert!(src.contains("#else"), "Missing #else");

    // Compile WITHOUT -DCHELIS_HAS_SLEEF: the scalar #else branch runs for all 9 elements.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_exp_sleef(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[9] = {{0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -2.0f, 0.1f, 3.0f, -0.5f}};
    chelis_tensor *in_t = make_view_1d(in_data, 9);
    chelis_tensor* in_ptr = in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_exp_sleef(inputs, 1, outputs, 1);

    int ok = 1;
    for (int i = 0; i < 9; i++) {{
        // The 2-op chain is exp->neg, so expected = -expf(x)
        float expected = -expf(in_data[i]);
        float got = ((float*)chelis_tensor_read_view(outputs[0]).data)[i];
        float reldiff = fabsf(got - expected) / (fabsf(expected) + 1e-6f);
        if (reldiff > 1e-4f) {{
            printf("MISMATCH at %d: got %.6f expected %.6f\n", i, got, expected);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("sleef_exp_neg9", src, &harness) else {
        panic!("Sleef exp->neg kernel scalar fallback failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "Sleef exp->neg kernel scalar fallback wrong:\n{output}"
    );
}

// ---- Test 10: Scalar ReduceSum preserves the canonical tree ----

#[test]
fn exec_reduce_sum_correct_output() {
    // Use TensorType::scalar_f32() (dims=[]) for the output — that is the correct
    // output type for a full-axis reduction producing a scalar.
    let scalar_ty = TensorType::scalar_f32();
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(100),
        None,
    );
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_reduce_sum").unwrap();
    let src = &result.c_source;

    assert!(
        !src.contains("chelis_sum_f32("),
        "Scalar Sum must preserve the canonical tree, including contiguous input:\n{src}"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_reduce_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[100];
    float scalar_sum = 0.0f;
    for (int i = 0; i < 100; i++) {{
        in_data[i] = (float)(i + 1);
        scalar_sum += in_data[i];
    }}
    chelis_tensor *in_t = make_view_1d(in_data, 100);
    chelis_tensor* in_ptr = in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_reduce_sum(inputs, 1, outputs, 1);

    float got = ((float*)chelis_tensor_read_view(outputs[0]).data)[0];
    float diff = fabsf(got - scalar_sum);
    printf("sum(1..100): got=%.2f expected=%.2f diff=%.6f\n", got, scalar_sum, diff);
    printf("%s\n", diff < 0.5f ? "PASS" : "FAIL");
    return diff < 0.5f ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("reduce_sum100", src, &harness) else {
        panic!("ReduceSum kernel failed to compile/run");
    };
    assert!(output.contains("PASS"), "ReduceSum wrong output:\n{output}");
}

#[test]
fn exec_count_multi_axis_matches_exact_int64_result() {
    let tensor_ty = |dims: &[usize], precision| TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(&[2, 3, 2], Prim::Bool),
        None,
    );
    let output = dag.add_node(
        RiscOp::Count { axes: vec![2, 0] },
        vec![input],
        tensor_ty(&[3], Prim::Int64),
        None,
    );
    dag.add_root(output);
    let generated = codegen_with_options(
        &dag,
        "test_count_multi",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .expect("Count C generation");
    assert!(generated.c_source.contains("chelis_int_checked_add"));
    assert!(generated.c_source.contains("CHELIS_DTYPE_BOOL"));
    assert!(!generated.c_source.contains("CHELIS_BOOL"));
    assert!(generated.c_source.contains("chelis_tensor_rank(t0)"));
    assert!(!generated.c_source.contains("t0->ndim"));
    for balanced_tree_fragment in [
        "while (__level_n_1 > 1)",
        "int64_t __left_1 = 2 * __j_1",
        "int64_t __right_1 = __left_1 + 1",
        "chelis_int_checked_add(__level_1[__left_1], __level_1[__right_1]",
    ] {
        assert!(
            generated.c_source.contains(balanced_tree_fragment),
            "Count C must emit the canonical adjacent-pair balanced tree; missing {balanced_tree_fragment:?}"
        );
    }

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_count_multi(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    uint8_t bits[12] = {{1,0,1,1,0,0,1,1,0,1,1,1}};
    int64_t shape[3] = {{2, 3, 2}};
    chelis_tensor* x = chelis_tensor_entry_borrow(3, shape, CHELIS_DTYPE_BOOL, bits, sizeof(bits));
    if (x == NULL) return 2;
    chelis_tensor* inputs[1] = {{x}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_count_multi(inputs, 1, outputs, 1);
    int64_t expected[3] = {{3, 3, 2}};
    int64_t* got = (int64_t*)chelis_tensor_read_view(outputs[0]).data;
    int ok = chelis_tensor_read_view(outputs[0]).dtype == CHELIS_DTYPE_I64 && chelis_tensor_numel(outputs[0]) == 3;
    for (int i = 0; i < 3; i++) if (got[i] != expected[i]) ok = 0;
    chelis_tensor_release(x);
    chelis_tensor_release(outputs[0]);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let output = compile_and_run_kernel("count_multi", &generated.c_source, &harness)
        .expect("Count C kernel compiles and runs");
    assert!(output.contains("PASS"), "wrong Count output: {output}");
}

#[test]
fn exec_count_selected_zero_extent_returns_zero() {
    let tensor_ty = |dims: &[usize], precision| TensorType {
        dims: dims.iter().copied().map(DimInfo::Lit).collect(),
        precision,
    };
    let mut dag = Dag::new();
    let input = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(&[2, 0, 3], Prim::Bool),
        None,
    );
    let output = dag.add_node(
        RiscOp::Count { axes: vec![1] },
        vec![input],
        tensor_ty(&[2, 3], Prim::Int64),
        None,
    );
    dag.add_root(output);
    let generated = codegen_with_options(
        &dag,
        "test_count_empty",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .expect("empty Count C generation");
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_count_empty(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int64_t shape[3] = {{2, 0, 3}};
    chelis_tensor* x = chelis_tensor_entry_borrow(3, shape, CHELIS_DTYPE_BOOL, NULL, 0);
    if (x == NULL) return 2;
    chelis_tensor* inputs[1] = {{x}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_count_empty(inputs, 1, outputs, 1);
    int64_t* got = (int64_t*)chelis_tensor_read_view(outputs[0]).data;
    int ok = chelis_tensor_read_view(outputs[0]).dtype == CHELIS_DTYPE_I64 && chelis_tensor_numel(outputs[0]) == 6;
    for (int i = 0; i < 6; i++) if (got[i] != 0) ok = 0;
    chelis_tensor_release(x);
    chelis_tensor_release(outputs[0]);
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let output = compile_and_run_kernel("count_empty", &generated.c_source, &harness)
        .expect("empty Count C kernel compiles and runs");
    assert!(
        output.contains("PASS"),
        "wrong empty Count output: {output}"
    );
}

// ---- Issue #254: reduce_window_* C-backend numerical parity ----
//
// The `issue_254_reduce_window_emit` tests pin only the *structural*
// shape of the emitted C (which intrinsic, which init literal). Per
// the backend-numerics policy, evaluator-vs-backend agreement needs an
// actual compile-and-run. These two tests close that gap: the emitted
// C is compiled with gcc, run, and checked against the exact values
// the IR evaluator (`chelis_ir::eval::reduce_window`) produces for the
// same 1x1x3x3 input — the canonical oracle per spec §6. Max exercises
// the `fmaxf` / `-INFINITY` path; Mean exercises the windowed-sum +
// `acc /= window_volume` division path.

fn reduce_window_3x3_dag(reducer: ReduceWindowKind, kernel: &str) -> String {
    let in_ty = TensorType {
        dims: [1, 1, 3, 3].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let out_ty = TensorType {
        dims: [1, 1, 2, 2].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    dag.add_node(
        RiscOp::ReduceWindow {
            reducer,
            window_shape: vec![2, 2],
            strides: vec![1, 1],
        },
        vec![x],
        out_ty,
        None,
    );
    codegen(&dag, kernel).unwrap().c_source
}

// Build a contiguous 1x1x3x3 input view holding [[1..9]] row-major.
const RW_HARNESS_4D_HEADER: &str = r#"
static chelis_tensor *make_view_1x1x3x3(float* data) {
    int64_t shape[4] = {1, 1, 3, 3};
    return chelis_tensor_entry_borrow(
        4, shape, CHELIS_DTYPE_F32, data, 9 * (int64_t)sizeof(float)
    );
}
"#;

#[test]
fn exec_reduce_window_max_matches_evaluator_oracle() {
    let src = reduce_window_3x3_dag(ReduceWindowKind::Max, "test_rw_max");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}
extern void test_rw_max(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[9] = {{1,2,3,4,5,6,7,8,9}};
    chelis_tensor *in_t = make_view_1x1x3x3(in_data);
    chelis_tensor* inputs[1] = {{in_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rw_max(inputs, 1, outputs, 1);

    // 2x2 maxes of [[1,2,3],[4,5,6],[7,8,9]]: [5,6,8,9].
    float expected[4] = {{5.0f, 6.0f, 8.0f, 9.0f}};
    int ok = (chelis_tensor_numel(outputs[0]) == 4);
    for (int i = 0; i < 4; i++) {{
        if (fabsf(((float*)chelis_tensor_read_view(outputs[0]).data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)chelis_tensor_read_view(outputs[0]).data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rw_max_3x3", &src, &harness) else {
        panic!("reduce_window_max kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_max C backend diverged from evaluator oracle:\n{output}"
    );
}

#[test]
fn exec_reduce_window_mean_matches_evaluator_oracle() {
    let src = reduce_window_3x3_dag(ReduceWindowKind::Mean, "test_rw_mean");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}
extern void test_rw_mean(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float in_data[9] = {{1,2,3,4,5,6,7,8,9}};
    chelis_tensor *in_t = make_view_1x1x3x3(in_data);
    chelis_tensor* inputs[1] = {{in_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rw_mean(inputs, 1, outputs, 1);

    // 2x2 means: sums [12,16,24,28] / 4 = [3,4,6,7].
    float expected[4] = {{3.0f, 4.0f, 6.0f, 7.0f}};
    int ok = (chelis_tensor_numel(outputs[0]) == 4);
    for (int i = 0; i < 4; i++) {{
        if (fabsf(((float*)chelis_tensor_read_view(outputs[0]).data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)chelis_tensor_read_view(outputs[0]).data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rw_mean_3x3", &src, &harness) else {
        panic!("reduce_window_mean kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_mean C backend diverged from evaluator oracle:\n{output}"
    );
}

// ---- reduce_window adjoint (ReduceWindowGrad) C-backend parity ----

// din = ReduceWindowGrad(x[1,1,3,3], g[1,1,2,2]) with window=[2,2]
// stride=[1,1]. Load "x" is created first (input slot 0), "g" second
// (slot 1), matching the harness `inputs[]` order.
fn reduce_window_grad_dag(reducer: ReduceWindowKind, kernel: &str) -> String {
    let x_ty = TensorType {
        dims: [1, 1, 3, 3].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let g_ty = TensorType {
        dims: [1, 1, 2, 2].into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        x_ty.clone(),
        None,
    );
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], g_ty, None);
    dag.add_node(
        RiscOp::ReduceWindowGrad {
            reducer,
            window_shape: vec![2, 2],
            strides: vec![1, 1],
        },
        vec![x, g],
        x_ty,
        None,
    );
    codegen(&dag, kernel).unwrap().c_source
}

const RW_GRAD_HARNESS_HEADER: &str = r#"
static chelis_tensor *make_view_1x1x2x2(float* data) {
    int64_t shape[4] = {1, 1, 2, 2};
    return chelis_tensor_entry_borrow(
        4, shape, CHELIS_DTYPE_F32, data, 4 * (int64_t)sizeof(float)
    );
}
"#;

#[test]
fn exec_reduce_window_grad_sum_matches_evaluator_oracle() {
    let src = reduce_window_grad_dag(ReduceWindowKind::Sum, "test_rwg_sum");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}{RW_GRAD_HARNESS_HEADER}
extern void test_rwg_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float x_data[9] = {{1,2,3,4,5,6,7,8,9}};
    float g_data[4] = {{1,1,1,1}};
    chelis_tensor *x_t = make_view_1x1x3x3(x_data);
    chelis_tensor *g_t = make_view_1x1x2x2(g_data);
    chelis_tensor* inputs[2] = {{x_t, g_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rwg_sum(inputs, 2, outputs, 1);

    // Sum adjoint with g=ones is the per-position window-cover count for
    // 2x2 windows / stride 1 over a 3x3 grid: corners 1, edges 2, center 4.
    float expected[9] = {{1,2,1, 2,4,2, 1,2,1}};
    int ok = (chelis_tensor_numel(outputs[0]) == 9);
    for (int i = 0; i < 9; i++) {{
        if (fabsf(((float*)chelis_tensor_read_view(outputs[0]).data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)chelis_tensor_read_view(outputs[0]).data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rwg_sum_3x3", &src, &harness) else {
        panic!("reduce_window_grad sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_grad(sum) C backend diverged from evaluator oracle:\n{output}"
    );
}

#[test]
fn exec_reduce_window_grad_max_matches_evaluator_oracle() {
    let src = reduce_window_grad_dag(ReduceWindowKind::Max, "test_rwg_max");
    let harness = format!(
        r#"{HARNESS_HEADER}{RW_HARNESS_4D_HEADER}{RW_GRAD_HARNESS_HEADER}
extern void test_rwg_max(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float x_data[9] = {{1,2,3,4,5,6,7,8,9}};
    float g_data[4] = {{1,1,1,1}};
    chelis_tensor *x_t = make_view_1x1x3x3(x_data);
    chelis_tensor *g_t = make_view_1x1x2x2(g_data);
    chelis_tensor* inputs[2] = {{x_t, g_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    test_rwg_max(inputs, 2, outputs, 1);

    // Each 2x2 window's max (distinct values) routes its g to the argmax:
    // windows pick (1,1),(1,2),(2,1),(2,2) of the 3x3 grid.
    float expected[9] = {{0,0,0, 0,1,1, 0,1,1}};
    int ok = (chelis_tensor_numel(outputs[0]) == 9);
    for (int i = 0; i < 9; i++) {{
        if (fabsf(((float*)chelis_tensor_read_view(outputs[0]).data)[i] - expected[i]) > 1e-5f) {{
            printf("MISMATCH at %d: got %.4f expected %.4f\n", i, ((float*)chelis_tensor_read_view(outputs[0]).data)[i], expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );
    let Some(output) = compile_and_run_kernel("rwg_max_3x3", &src, &harness) else {
        panic!("reduce_window_grad max kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "reduce_window_grad(max) C backend diverged from evaluator oracle:\n{output}"
    );
}

// ---- IEEE-754 corner cases for Div and Recip ----
// Exercise the C-backend codegen (`emit_binary` for Div, `emit_recip`
// for Recip) end-to-end on the four corner cases an
// `exp(neg(log(b)))` decomposition would mishandle: 5/-2, 1/0, -1/0,
// 0/0 for Div; recip(-2) and recip(0) for Recip. Path: chelis IR →
// emitted C → gcc → run.

#[test]
fn exec_div_ieee_corner_cases() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_f32(4), None);
    dag.add_node(RiscOp::Div, vec![a, b], vec_f32(4), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_div_ieee",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <math.h>
extern void test_div_ieee(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float a_data[4] = {{ 5.0f,  1.0f, -1.0f, 0.0f }};
    float b_data[4] = {{-2.0f,  0.0f,  0.0f, 0.0f }};
    chelis_tensor *a_t = make_view_1d(a_data, 4);
    chelis_tensor *b_t = make_view_1d(b_data, 4);
    chelis_tensor* inputs[2] = {{a_t, b_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_div_ieee(inputs, 2, outputs, 1);

    float* o = chelis_tensor_read_view(outputs[0]).data;
    int ok = 1;
    if (o[0] != -2.5f) {{ printf("MISMATCH 5/-2: got %f want -2.5\n", o[0]); ok = 0; }}
    if (!(isinf(o[1]) && o[1] > 0)) {{ printf("MISMATCH 1/0: got %f want +inf\n", o[1]); ok = 0; }}
    if (!(isinf(o[2]) && o[2] < 0)) {{ printf("MISMATCH -1/0: got %f want -inf\n", o[2]); ok = 0; }}
    if (!isnan(o[3])) {{ printf("MISMATCH 0/0: got %f want NaN\n", o[3]); ok = 0; }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("div_ieee", src, &harness) else {
        panic!("Div IEEE kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C-backend Div must produce IEEE results for the four corner cases:\n{output}"
    );
}

#[test]
fn exec_recip_ieee_corner_cases() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(3), None);
    dag.add_node(RiscOp::Recip, vec![a], vec_f32(3), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_recip_ieee",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <math.h>
extern void test_recip_ieee(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float a_data[3] = {{-2.0f, 0.0f, 4.0f}};
    chelis_tensor *a_t = make_view_1d(a_data, 3);
    chelis_tensor* inputs[1] = {{a_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_recip_ieee(inputs, 1, outputs, 1);

    float* o = chelis_tensor_read_view(outputs[0]).data;
    int ok = 1;
    if (o[0] != -0.5f) {{ printf("MISMATCH recip(-2): got %f want -0.5\n", o[0]); ok = 0; }}
    if (!(isinf(o[1]) && o[1] > 0)) {{ printf("MISMATCH recip(0): got %f want +inf\n", o[1]); ok = 0; }}
    if (o[2] != 0.25f) {{ printf("MISMATCH recip(4): got %f want 0.25\n", o[2]); ok = 0; }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("recip_ieee", src, &harness) else {
        panic!("Recip IEEE kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C-backend Recip must produce IEEE results for the corner cases:\n{output}"
    );
}

/// The C scalar type and `CHELIS_*` dtype macro for an integer precision,
/// used to generate width-parametrized exec harnesses (chelis#550 F2). The
/// printf specifier is always `%lld` after a `(long long)` cast so the same
/// format string works for every width.
fn int_c_type_and_dtype(precision: Prim) -> (&'static str, &'static str) {
    match precision {
        Prim::Int8 => ("int8_t", "CHELIS_DTYPE_I8"),
        Prim::Int16 => ("int16_t", "CHELIS_DTYPE_I16"),
        Prim::Int32 => ("int32_t", "CHELIS_DTYPE_I32"),
        Prim::Int64 => ("int64_t", "CHELIS_DTYPE_I64"),
        other => panic!("int_c_type_and_dtype: non-integer precision {other:?}"),
    }
}

fn vec_int(n: usize, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision,
    }
}

/// Shared exec-compile driver for an integer binary-division op on a
/// same-precision operand pair. Builds a two-load DAG, lowers `op`, compiles
/// the kernel, and asserts each output element matches `expected`. The four
/// operand pairs `{7,2},{7,-2},{-7,2},{-7,-2}` exercise every sign
/// combination so floor-vs-truncate rounding is distinguished on the
/// mixed-sign cases. `precision` parametrizes the integer width (chelis#550
/// F2: int8 / int16 / int64 in addition to the original int32).
fn run_int_div_op_exec(
    op: RiscOp,
    fn_name: &str,
    kernel_name: &str,
    expected: [i32; 4],
    precision: Prim,
) {
    let (c_type, dtype_macro) = int_c_type_and_dtype(precision);
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_int(4, precision),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_int(4, precision),
        None,
    );
    dag.add_node(op, vec![a, b], vec_int(4, precision), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        fn_name,
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let [e0, e1, e2, e3] = expected;
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void {fn_name}(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    {c_type} a_data[4] = {{ 7,  7, -7, -7}};
    {c_type} b_data[4] = {{ 2, -2,  2, -2}};
    {c_type} expected[4] = {{ {e0}, {e1}, {e2}, {e3} }};

    chelis_tensor *a_t = make_view_typed_1d(a_data, 4, {dtype_macro});
    chelis_tensor *b_t = make_view_typed_1d(b_data, 4, {dtype_macro});

    chelis_tensor* inputs[2] = {{a_t, b_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    {fn_name}(inputs, 2, outputs, 1);

    int ok = 1;
    {c_type}* o = ({c_type}*)chelis_tensor_read_view(outputs[0]).data;
    if (chelis_tensor_read_view(outputs[0]).dtype != {dtype_macro}) {{
        printf("FAIL: output dtype %d, expected {dtype_macro} (%d)\n",
               chelis_tensor_read_view(outputs[0]).dtype, {dtype_macro});
        ok = 0;
    }}
    for (int i = 0; i < 4 && ok; i++) {{
        if (o[i] != expected[i]) {{
            printf("MISMATCH idx=%d a=%lld b=%lld got=%lld want=%lld\n",
                   i, (long long)a_data[i], (long long)b_data[i],
                   (long long)o[i], (long long)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel(kernel_name, src, &harness) else {
        panic!("{kernel_name} ({precision:?}) kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C-backend {kernel_name} on {precision:?} mismatch:\n{output}"
    );
}

// spec/05-risc-primitives.md §2.1: `trunc_div` uses C/Rust truncating
// semantics (round toward zero). This is what chelis-std's
// `Std.Decimal::normalize` / `decimal_div_nonzero` rely on for scale
// shifts and quotient computation. The C backend emits `int32_t /
// int32_t` which truncates by language definition; this exec-compile
// test pins that contract end-to-end across every sign combination.
// `{7,-7} / {2,-2}` ⇒ `{3, -3, -3, 3}` (round toward zero).
#[test]
fn exec_trunc_div_int32_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i32",
        "trunc_div_i32",
        [3, -3, -3, 3],
        Prim::Int32,
    );
}

// chelis#550 F2: the trunc_div / floor_div emit is width-independent (the
// C backend promotes to int64 internally), but the repo's negative-parity
// bar requires the narrower and wider integer widths be exercised
// end-to-end, not just int32. Same operands / expected results as the int32
// cases above; only the storage precision changes.
#[test]
fn exec_trunc_div_int8_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i8",
        "trunc_div_i8",
        [3, -3, -3, 3],
        Prim::Int8,
    );
}

#[test]
fn exec_trunc_div_int16_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i16",
        "trunc_div_i16",
        [3, -3, -3, 3],
        Prim::Int16,
    );
}

#[test]
fn exec_trunc_div_int64_truncates_toward_zero() {
    run_int_div_op_exec(
        RiscOp::TruncDiv,
        "test_trunc_div_i64",
        "trunc_div_i64",
        [3, -3, -3, 3],
        Prim::Int64,
    );
}

// spec/05-risc-primitives.md §2.1: `floor_div` rounds the quotient
// toward −∞. It agrees with truncate when the operands share a sign and
// differs on the mixed-sign exact-fraction cases:
// `7 floor_div 2 == 3`, `7 floor_div -2 == -4`, `-7 floor_div 2 == -4`,
// `-7 floor_div -2 == 3`. This is the round-toward-−∞ semantics that
// matches Python `//` / torch / JAX / numpy `floor_divide`; the C
// backend realizes it as native `/` plus a remainder-sign correction.
#[test]
fn exec_floor_div_int32_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i32",
        "floor_div_i32",
        [3, -4, -4, 3],
        Prim::Int32,
    );
}

// chelis#550 F2: floor_div across the remaining integer widths. The
// round-toward-−∞ remainder-sign correction must hold at int8 / int16 /
// int64 just as at int32.
#[test]
fn exec_floor_div_int8_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i8",
        "floor_div_i8",
        [3, -4, -4, 3],
        Prim::Int8,
    );
}

#[test]
fn exec_floor_div_int16_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i16",
        "floor_div_i16",
        [3, -4, -4, 3],
        Prim::Int16,
    );
}

#[test]
fn exec_floor_div_int64_rounds_toward_neg_inf() {
    run_int_div_op_exec(
        RiscOp::FloorDiv,
        "test_floor_div_i64",
        "floor_div_i64",
        [3, -4, -4, 3],
        Prim::Int64,
    );
}

/// Compile a kernel + harness exactly like `compile_and_run_kernel`, but
/// return the run `Output` (status + stderr) so a trap test can assert the
/// binary aborts. Panics if COMPILATION fails — a zero-divisor trap is a
/// runtime abort, not a compile error.
fn compile_and_capture_run(test_name: &str, c_source: &str, harness: &str) -> std::process::Output {
    let probe = common::probe_dir(&format!("exec_{test_name}"));
    let dir = probe.path().to_path_buf();
    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let bin = dir.join("test_bin");
    let runtime_lib = runtime_lib_path();
    let compile = Command::new("gcc")
        .arg("-O2")
        .args(simd_isa_flags())
        .args([
            "-std=c11",
            "-I",
            dir.to_str().unwrap(),
            dir.join("kernel.c").to_str().unwrap(),
            dir.join("main.c").to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime_lib.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("failed to invoke gcc");
    assert!(
        compile.status.success(),
        "COMPILE FAILED [{test_name}]:\n{}\nKernel C:\n{c_source}",
        String::from_utf8_lossy(&compile.stderr),
    );

    Command::new(&bin).output().expect("failed to run binary")
}

// chelis#550 F2: a COMPILED floor_div zero-divisor trap. The existing
// backend trap coverage was trunc_div-only; floor_div emits the SAME
// portable `chelis_int_div_guard` and must abort identically. The divisor
// arrives through a runtime Load (`chelis_tensor_read_view(inputs[1]).data`), so gcc cannot
// constant-fold the zero and elide the guard. Spec/05-risc-primitives.md
// §2.1 scopes the `integer division or remainder by zero` trap to the C
// backend (and the evaluator); this is the fail-closed end-to-end proof.
#[test]
fn exec_floor_div_int_zero_divisor_traps() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_int(2, Prim::Int64),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_int(2, Prim::Int64),
        None,
    );
    dag.add_node(RiscOp::FloorDiv, vec![a, b], vec_int(2, Prim::Int64), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_floor_div_trap",
        CodegenOptions {
            math_lib_override: Some(MathLib::None),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;
    // Emit-shape: floor_div must wrap the integer divisor in the portable guard.
    assert!(
        src.contains("chelis_int_div_guard("),
        "integer floor_div must emit the portable zero-divisor guard (#550); \
         emitted C=\n{src}",
    );

    // b_data[1] == 0: the second element divides by zero at runtime.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_floor_div_trap(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int64_t a_data[2] = {{ 10, 7 }};
    int64_t b_data[2] = {{ 2, 0 }};

    chelis_tensor *a_t = make_view_typed_1d(a_data, 2, CHELIS_DTYPE_I64);
    chelis_tensor *b_t = make_view_typed_1d(b_data, 2, CHELIS_DTYPE_I64);

    chelis_tensor* inputs[2] = {{a_t, b_t}};
    chelis_tensor* outputs[1] = {{NULL}};
    test_floor_div_trap(inputs, 2, outputs, 1);

    /* The guard aborts before reaching here; printing PASS would be a bug. */
    printf("PASS\n");
    return 0;
}}
"#
    );

    let run = compile_and_capture_run("floor_div_trap", src, &harness);
    assert!(
        !run.status.success(),
        "floor_div by a runtime zero divisor must trap (abort), not succeed; \
         stdout={}",
        String::from_utf8_lossy(&run.stdout),
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("integer division or remainder by zero"),
        "the C backend trap must emit the canonical diagnostic on stderr; \
         stderr={stderr:?}",
    );
    assert!(
        !String::from_utf8_lossy(&run.stdout).contains("PASS"),
        "no PASS line may print when the program traps; the guard must abort \
         before the kernel returns",
    );
}

// ---- Test 10b: Cross-backend f32 bit-exactness oracle for issue #163 ----
//
// PR #168 review MED #3: pin C-backend / host-evaluator agreement on
// the issue #163 reproducer at the bit level, end-to-end. The
// host-evaluator test in `chelis-compiler-api::runtime` asserts the
// stride-4 ILP cascade result on the same 11-element multiset; this
// test does the same against the C backend's output by compiling the
// generated C with gcc, running it, and verifying the printed result
// is bit-exactly `0x4087012d` (= 4.218893527984619_f32). Without
// this end-to-end test, a future divergence between the evaluator
// lane and the codegen lane (e.g., a subtle lane-assignment shift in
// `chelis_sum_f32` vs `host_runtime::reduce_f32`) would not be caught
// by either layer's own tests.

#[test]
fn exec_reduce_sum_issue_163_repro_is_bit_exact_with_evaluator() {
    let scalar_ty = TensorType::scalar_f32();
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(11), None);
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: chelis_types::types::Prim::F32,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);
    let result = codegen(&dag, "test_issue_163_sum").unwrap();

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
extern void test_issue_163_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // Issue #163 right-pad reflected sequence: the issue's exact
    // reproducer multiset, identical to the host-runtime test in
    // chelis-compiler-api::runtime::host_runtime_sum_f32_uses_pairwise_order_for_issue_163_repro.
    float in_data[11] = {{
        0.49625658988952637f,
        0.7682217955589294f,
        0.08847743272781372f,
        0.13203048706054688f,
        0.30742114782333374f,
        0.6340786814689636f,
        0.30742114782333374f,
        0.13203048706054688f,
        0.08847743272781372f,
        0.7682217955589294f,
        0.49625658988952637f,
    }};
    chelis_tensor *in_t = make_view_1d(in_data, 11);
    chelis_tensor* inputs[1] = {{ in_t }};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{ out_slot }};

    test_issue_163_sum(inputs, 1, outputs, 1);

    float got = ((float*)chelis_tensor_read_view(outputs[0]).data)[0];
    uint32_t got_bits;
    memcpy(&got_bits, &got, sizeof(got_bits));
    // 4.218893527984619_f32 is the stride-4 ILP cascade result the
    // host runtime emits for the same multiset. Bit-exact equality
    // is the whole point of this PR.
    uint32_t expected_bits = 0x4087012d;
    printf("got=%.17g bits=0x%08x expected_bits=0x%08x\n", (double)got, got_bits, expected_bits);
    printf("%s\n", got_bits == expected_bits ? "PASS" : "FAIL");
    return got_bits == expected_bits ? 0 : 1;
}}
"#,
        HARNESS_HEADER = HARNESS_HEADER,
    );

    let src = &result.c_source;
    let Some(output) = compile_and_run_kernel("issue_163_sum", src, &harness) else {
        panic!("issue #163 sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "C backend's stride-4 cascade must match the host evaluator bit-exactly \
         for the issue #163 multiset; got: {output}"
    );
}

// ---- Test 5: Zero-size tensor does not crash ----

#[test]
fn exec_zero_size_tensor_does_not_crash() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_f32(0), None);
    dag.add_node(RiscOp::Exp, vec![a], vec_f32(0), None);
    let dag = fuse(&dag);

    let result = codegen_with_options(
        &dag,
        "test_exp_zero",
        CodegenOptions {
            math_lib_override: Some(MathLib::Sleef),
            ..Default::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_exp_zero(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // zero-element 1D tensor
    float dummy = 0.0f;
    chelis_tensor *in_t = make_view_1d(&dummy, 0);

    chelis_tensor* in_ptr = in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_exp_zero(inputs, 1, outputs, 1);
    printf("zero-size exp returned, output_size=%lld\n", (long long)(outputs[0] ? chelis_tensor_numel(outputs[0]) : -1));
    printf("PASS\n");
    return 0;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("zero_size_exp", src, &harness) else {
        panic!("Zero-size exp kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "Zero-size exp kernel crashed:\n{output}"
    );
}

// ---- Test 8: NaN propagation inconsistency probe ----

#[test]
fn exec_simd_nan_propagation_inconsistency_probe() {
    let include_dir = runtime_include_dir();
    let probe = common::probe_dir("exec_nan_probe");
    let dir = probe.path().to_path_buf();

    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let c_src = r#"
#include <stdio.h>
#include <math.h>
#include "chelis_simd.h"

int main() {
    // NaN at position 0 of 8-wide AVX2 chunk
    float arr_nan_pos0[8] = {NAN, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    // NaN at position 1 of 8-wide AVX2 chunk
    float arr_nan_pos1[8] = {1.0f, NAN, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    // NaN in scalar tail only (position 8, n=9)
    float arr_nan_tail[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, NAN};

    float max_pos0  = chelis_max_f32(arr_nan_pos0, 8);
    float max_pos1  = chelis_max_f32(arr_nan_pos1, 8);
    float max_tail  = chelis_max_f32(arr_nan_tail, 9);

    int nan_pos0  = isnan(max_pos0);
    int nan_pos1  = isnan(max_pos1);
    int nan_tail  = isnan(max_tail);

    printf("max(NaN@pos0,  n=8): is_nan=%d val=%.2f\n", nan_pos0, max_pos0);
    printf("max(NaN@pos1,  n=8): is_nan=%d val=%.2f\n", nan_pos1, max_pos1);
    printf("max(NaN@tail,  n=9): is_nan=%d val=%.2f\n", nan_tail, max_tail);

    if (nan_pos0 != nan_pos1 || nan_pos0 != nan_tail) {
        printf("INCONSISTENT: NaN propagation depends on position\n");
    } else {
        printf("CONSISTENT: propagation=%d\n", nan_pos0);
    }
    return 0;
}
"#;

    fs::write(dir.join("nan_probe.c"), c_src).unwrap();
    let bin = dir.join("nan_bin");

    let compile = Command::new("gcc")
        .arg("-O2")
        .args(simd_isa_flags())
        .args([
            "-std=c11",
            "-I",
            dir.to_str().unwrap(),
            dir.join("nan_probe.c").to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            "-lm",
        ])
        .output()
        .expect("gcc not available");

    assert!(
        compile.status.success(),
        "nan probe compile failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin)
        .output()
        .expect("failed to run nan probe");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    eprintln!("NaN probe output:\n{stdout}");

    // #172: `chelis_max_f32` now PROPAGATES NaN consistently, regardless of
    // whether the NaN lands in an AVX2 lane or the scalar tail, matching
    // `torch.max` (which returns NaN for any NaN-containing slice, at every
    // position). Previously this probe only DOCUMENTED the position-dependent
    // inconsistency (the SIMD max dropped NaN); it now ASSERTS the fixed,
    // torch-aligned behavior: all three positions yield NaN, and the run
    // reports CONSISTENT propagation.
    assert!(
        stdout.contains("max(NaN@pos0,  n=8): is_nan=1"),
        "NaN at AVX2 lane 0 must propagate (#172); probe output:\n{stdout}"
    );
    assert!(
        stdout.contains("max(NaN@pos1,  n=8): is_nan=1"),
        "NaN at AVX2 lane 1 must propagate (#172); probe output:\n{stdout}"
    );
    assert!(
        stdout.contains("max(NaN@tail,  n=9): is_nan=1"),
        "NaN in the scalar tail must propagate (#172); probe output:\n{stdout}"
    );
    assert!(
        stdout.contains("CONSISTENT: propagation=1"),
        "NaN propagation must be position-independent and always-propagate \
         (#172 torch parity); probe output:\n{stdout}"
    );
}

// ---- Test 9: chelis_simd.h compiles as C++ ----

#[test]
fn exec_simd_header_compiles_as_cxx() {
    let include_dir = runtime_include_dir();
    let probe = common::probe_dir("cxx_probe");
    let dir = probe.path().to_path_buf();

    // Copy simd header
    let simd_src = fs::read_to_string(include_dir.join("chelis_simd.h")).unwrap();
    fs::write(dir.join("chelis_simd.h"), &simd_src).unwrap();

    // Write a minimal C++ file that includes it
    let cxx_src = r#"
#include "chelis_simd.h"
int main() { return 0; }
"#;
    fs::write(dir.join("test.cpp"), cxx_src).unwrap();

    let output = Command::new("g++")
        .args(["-std=c++17", "-O2"])
        .args(simd_isa_flags())
        .args([
            "-I",
            dir.to_str().unwrap(),
            dir.join("test.cpp").to_str().unwrap(),
            "-o",
            dir.join("cxx_bin").to_str().unwrap(),
            "-lm",
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            println!("chelis_simd.h compiles cleanly as C++");
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            panic!("chelis_simd.h FAILS to compile as C++:\n{stderr}");
        }
        Err(e) => {
            eprintln!("g++ not available, skipping: {e}");
        }
    }
}

// =====================================================================
// WS-A1 + WS-A4 acceptance oracle: f64 / mixed-dtype / integer
// reduce_sum / matmul / i8 / i16 end-to-end tests.
// =====================================================================
//
// These tests exercise:
//   * WS-A1: f64 BlasMatmul + dgemm dispatch and the dtype-parameterized
//     ReduceSum accumulator (f64 / mixed / integer paths).
//   * WS-A4: i8 / i16 source data through the active dtype set per
//     spec/04-type-system.md §1.1 plus the reduce_sum
//     accumulator-promotion rule per §5.7.1 (i8/i16 → i32). The C
//     backend's `dtype_macro` maps Int8/Int16 to CHELIS_DTYPE_I8 / CHELIS_DTYPE_I16
//     and the runtime allocator sizes their buffers correctly.
//
// They compile generated C against the runtime + BLAS, run it, and
// compare against a hand-computed reference (the evaluator equivalent
// for the fixed-point reduce_sum / matmul cases is the closed-form
// value).

use chelis_ir::dag::DimExpr;

fn vec_f64(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F64,
    }
}

fn mat_f64(r: usize, c: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
        precision: Prim::F64,
    }
}

fn vec_i32(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int32,
    }
}

fn vec_i8(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int8,
    }
}

fn vec_i16(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::Int16,
    }
}

fn scalar_i32() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::Int32,
    }
}

fn scalar_f64() -> TensorType {
    TensorType {
        dims: vec![],
        precision: Prim::F64,
    }
}

/// Locate openblas / accelerate link flags so the matmul tests can
/// link against `cblas_dgemm`. Returns None on platforms where BLAS
/// isn't reachable; tests skip cleanly in that case.
fn blas_link_flags() -> Option<Vec<String>> {
    if cfg!(target_os = "macos") {
        Some(vec!["-framework".into(), "Accelerate".into()])
    } else {
        Some(vec!["-lopenblas".into()])
    }
}

/// Compile + run a kernel that uses BLAS. Same as
/// `compile_and_run_kernel` but also passes the BLAS link flags. The
/// generated C source is expected to have `#include "chelis_blas.h"`.
/// Returns None on compile/run failure (e.g. openblas not installed).
fn compile_and_run_kernel_with_blas(
    test_name: &str,
    c_source: &str,
    harness: &str,
) -> Option<String> {
    let probe = common::probe_dir(&format!("exec_{test_name}"));
    let dir = probe.path().to_path_buf();

    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();

    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }

    let bin = dir.join("test_bin");
    let runtime_lib = runtime_lib_path();
    let blas_flags = blas_link_flags().unwrap_or_default();

    let mut args: Vec<String> = vec!["-O2".into()];
    args.extend(simd_isa_flags());
    args.extend([
        "-std=c11".into(),
        "-I".into(),
        dir.to_str().unwrap().into(),
        dir.join("kernel.c").to_str().unwrap().into(),
        dir.join("main.c").to_str().unwrap().into(),
        "-o".into(),
        bin.to_str().unwrap().into(),
        runtime_lib.to_str().unwrap().into(),
    ]);
    args.extend(blas_flags);
    args.push("-lm".into());
    args.push("-lpthread".into());
    args.push("-ldl".into());

    let compile = Command::new("gcc")
        .args(&args)
        .output()
        .expect("failed to invoke gcc");

    if !compile.status.success() {
        let stderr = String::from_utf8_lossy(&compile.stderr);
        eprintln!("BLAS COMPILE FAILED [{test_name}]:\n{stderr}");
        eprintln!("Kernel C:\n{c_source}");
        return None;
    }

    let run = Command::new(&bin).output().expect("failed to run binary");
    if !run.status.success() {
        let stderr = String::from_utf8_lossy(&run.stderr);
        let stdout = String::from_utf8_lossy(&run.stdout);
        eprintln!("BLAS RUN FAILED [{test_name}]\nstdout: {stdout}\nstderr: {stderr}");
        return None;
    }

    Some(String::from_utf8_lossy(&run.stdout).into_owned())
}

// ---- WS-A1 Test: f64 reduce_sum produces correct value ----
//
// Spec §5.7.1: f64 reduce_sum default accumulator is f64; result is f64.
// Pre-WS-A1 the C backend emitted `chelis_fill_f32` and `float acc =
// 0.0f` regardless of operand precision, silently truncating. After
// WS-A1 the accumulator type and zero literal come from the IR Sum
// node's accumulator (== output precision per §5.7.1), so the loop
// is `double acc = 0.0;` and the fill carries an exact tagged f64 zero.
#[test]
fn ws_a1_exec_f64_reduce_sum_matches_reference() {
    let scalar_ty = scalar_f64();
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f64(100),
        None,
    );
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_reduce_sum_f64").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("CHELIS_DTYPE_F64"),
        "f64 reduce_sum must allocate an f64 output tensor:\n{src}"
    );
    assert!(
        src.contains("__sum_level_"),
        "f64 sum needs a tree at its accumulator width: {src}"
    );
    assert!(
        !src.contains("chelis_fill_f32(") && !src.contains("chelis_fill_f64("),
        "f64 reduce_sum must not retain dtype-specific compatibility fills:\n{src}"
    );
    assert!(
        src.contains("double *__sum_level_"),
        "f64 reduce_sum accumulator must be a double, not float:\n{src}"
    );
    assert!(
        !src.contains("chelis_sum_f32("),
        "f64 reduce_sum must NOT route through the f32-only chelis_sum_f32 SIMD helper:\n{src}"
    );

    // Build harness: pass an f64 array with shape[100] of 1..=100, expect sum=5050.0
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_reduce_sum_f64(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // Allocate exact f64 storage and describe its byte capacity.
    double in_data[100];
    double scalar_sum = 0.0;
    for (int i = 0; i < 100; i++) {{
        in_data[i] = (double)(i + 1);
        scalar_sum += in_data[i];
    }}
    chelis_tensor *in_t = make_view_typed_1d(in_data, 100, CHELIS_DTYPE_F64);

    chelis_tensor* in_ptr = in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_reduce_sum_f64(inputs, 1, outputs, 1);

    double got = ((double*)chelis_tensor_read_view(outputs[0]).data)[0];
    double diff = fabs(got - scalar_sum);
    printf("sum_f64(1..100): got=%.6f expected=%.6f diff=%.12f\n", got, scalar_sum, diff);
    printf("%s\n", diff < 1e-9 ? "PASS" : "FAIL");
    return diff < 1e-9 ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a1_reduce_sum_f64", src, &harness) else {
        panic!("WS-A1 f64 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 f64 reduce_sum wrong output:\n{output}"
    );
}

// ---- WS-A1 Test: i32 reduce_sum produces integer result ----
//
// Spec §5.7.1: i32 reduce_sum default accumulator is i32; result is i32.
// Pre-WS-A1 the emitter would have emitted `float acc = 0.0f` and
// silently truncated/cast the integer values. After WS-A1 the
// accumulator type comes from the IR node, so the loop is
// `int32_t acc = (int32_t)0;` with no float involvement. The result
// must round-trip exact integers (no float widening that would lose
// precision near 2^24).
#[test]
fn ws_a1_exec_i32_reduce_sum_produces_integer_result_no_float_cast() {
    let scalar_ty = scalar_i32();
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i32(5), None);
    dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![a],
        scalar_ty,
        None,
    );
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_reduce_sum_i32").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("CHELIS_DTYPE_I32"),
        "i32 reduce_sum must allocate an i32 output tensor:\n{src}"
    );
    assert!(
        src.contains("int32_t *__sum_level_"),
        "i32 reduce_sum accumulator must be int32_t (integer-exact), not float:\n{src}"
    );
    assert!(
        !src.contains("float *__sum_level_"),
        "i32 reduce_sum must NOT use a float accumulator (silent precision change):\n{src}"
    );
    assert!(
        !src.contains("chelis_sum_f32("),
        "i32 reduce_sum must NOT route through the f32-only SIMD helper:\n{src}"
    );

    // Pick values whose i32 sum fits without overflow; verify exact integer round-trip.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_reduce_sum_i32(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // i32 storage in a 4-byte slot; runtime alloc treats CHELIS_DTYPE_I32 as 4 bytes.
    int32_t in_data[5] = {{16777215, 16777215, 16777215, 16777215, 16777215}};
    // Note: 16777215 = 2^24 - 1. f32 can represent this exactly, but
    // sum * 5 = 83886075, which a float accumulator would round (since
    // 83886075 != round-to-nearest-float). i32 accumulator returns the
    // exact integer.
    int32_t expected_sum = 16777215 * 5;

    chelis_tensor *in_t = make_view_typed_1d(in_data, 5, CHELIS_DTYPE_I32);

    chelis_tensor* in_ptr = in_t;
    chelis_tensor* inputs[1] = {{in_ptr}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_reduce_sum_i32(inputs, 1, outputs, 1);

    if (chelis_tensor_read_view(outputs[0]).dtype != CHELIS_DTYPE_I32) {{
        printf("FAIL: output dtype is %d, expected CHELIS_DTYPE_I32 (%d)\n",
               chelis_tensor_read_view(outputs[0]).dtype, CHELIS_DTYPE_I32);
        return 1;
    }}
    int32_t got = ((int32_t*)chelis_tensor_read_view(outputs[0]).data)[0];
    printf("sum_i32(5x 2^24-1): got=%d expected=%d\n", got, expected_sum);
    printf("%s\n", got == expected_sum ? "PASS" : "FAIL");
    return got == expected_sum ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a1_reduce_sum_i32", src, &harness) else {
        panic!("WS-A1 i32 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 i32 reduce_sum wrong output (integer arithmetic must be exact):\n{output}"
    );
}

// ---- WS-A1 Test: f64 matmul dispatches cblas_dgemm and matches reference ----
//
// Spec §5.7.1: f64 matmul accumulator is f64; result is f64. The C
// backend now dispatches `cblas_dgemm` (not `cblas_sgemm`) for an f64
// BlasMatmul. Reference is a hand-computed product where the values
// are exactly representable in f64 but not in f32, so an accidental
// sgemm dispatch would fail the tolerance check.
#[test]
fn ws_a1_exec_f64_matmul_dispatches_dgemm_and_matches_reference() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        mat_f64(2, 3),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        mat_f64(3, 4),
        None,
    );
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F64,
    )
    .expect("f64 matmul default constructs (spec §5.7.1)");
    dag.add_node(matmul_op, vec![a, b], mat_f64(2, 4), None);

    let result = codegen_with_options(
        &dag,
        "test_matmul_f64",
        CodegenOptions {
            use_blas: true,
            ..CodegenOptions::default()
        },
    )
    .unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("cblas_dgemm("),
        "f64 matmul MUST dispatch cblas_dgemm (the WS-A1 lift):\n{src}"
    );
    assert!(
        !src.contains("cblas_sgemm("),
        "f64 matmul MUST NOT silently dispatch cblas_sgemm (RT-1 F1 finding):\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_F64"),
        "f64 matmul output tensor must use CHELIS_DTYPE_F64 dtype:\n{src}"
    );
    assert!(
        src.contains("(double*)t"),
        "f64 matmul must cast tensor data pointers to double*:\n{src}"
    );
    assert!(
        result.requirements.needs_blas,
        "f64 matmul codegen must surface needs_blas=true so the toolchain links openblas/Accelerate"
    );

    // A[2,3] = [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
    // B[3,4] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    // Reference C[2,4]: row-major matmul. f64 yields exact results for these
    // operand magnitudes; an accidental sgemm dispatch would compute the same
    // values but on truncated f32 inputs, and the test asserts cblas_dgemm
    // appears in the source so that path is unreachable.
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_matmul_f64(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    double a_data[6] = {{1.5, 2.5, 3.5, 4.5, 5.5, 6.5}};
    double b_data[12] = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0}};

    const int64_t a_shape[2] = {{2, 3}};
    chelis_tensor *a_t = chelis_tensor_entry_borrow(
        2, a_shape, CHELIS_DTYPE_F64, a_data, 6 * (int64_t)sizeof(double)
    );

    const int64_t b_shape[2] = {{3, 4}};
    chelis_tensor *b_t = chelis_tensor_entry_borrow(
        2, b_shape, CHELIS_DTYPE_F64, b_data, 12 * (int64_t)sizeof(double)
    );

    chelis_tensor* inputs[2] = {{a_t, b_t}};
    chelis_tensor* out_slot = NULL;
    chelis_tensor* outputs[1] = {{out_slot}};

    test_matmul_f64(inputs, 2, outputs, 1);

    // Reference: hand-computed row-major matmul.
    double expected[8] = {{
        // row 0
        1.5*1 + 2.5*5 + 3.5*9,    // = 1.5 + 12.5 + 31.5 = 45.5
        1.5*2 + 2.5*6 + 3.5*10,   // = 3 + 15 + 35 = 53
        1.5*3 + 2.5*7 + 3.5*11,   // = 4.5 + 17.5 + 38.5 = 60.5
        1.5*4 + 2.5*8 + 3.5*12,   // = 6 + 20 + 42 = 68
        // row 1
        4.5*1 + 5.5*5 + 6.5*9,    // = 4.5 + 27.5 + 58.5 = 90.5
        4.5*2 + 5.5*6 + 6.5*10,   // = 9 + 33 + 65 = 107
        4.5*3 + 5.5*7 + 6.5*11,   // = 13.5 + 38.5 + 71.5 = 123.5
        4.5*4 + 5.5*8 + 6.5*12    // = 18 + 44 + 78 = 140
    }};
    if (chelis_tensor_read_view(outputs[0]).dtype != CHELIS_DTYPE_F64) {{
        printf("FAIL: output dtype is %d, expected CHELIS_DTYPE_F64 (%d)\n",
               chelis_tensor_read_view(outputs[0]).dtype, CHELIS_DTYPE_F64);
        return 1;
    }}
    double* got = (double*)chelis_tensor_read_view(outputs[0]).data;
    int ok = 1;
    for (int i = 0; i < 8; i++) {{
        double diff = fabs(got[i] - expected[i]);
        if (diff > 1e-12) {{
            printf("MISMATCH at %d: got %.15f expected %.15f diff %.15g\n",
                   i, got[i], expected[i], diff);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel_with_blas("ws_a1_matmul_f64", src, &harness) else {
        eprintln!("WS-A1 f64 matmul kernel failed to compile/run (openblas may be unavailable)");
        // Don't panic if openblas is missing on the host; the source-level
        // assertions above already pin the dgemm dispatch shape.
        return;
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 f64 matmul wrong output (cblas_dgemm dispatch) :\n{output}"
    );
}

// ---- WS-A1 Test: mixed-dtype program (f64 tensor + i32 tensor side-by-side) ----
//
// Compiles a program with both f64 and i32 tensors live in the same
// DAG, returning two outputs: an f64 reduce_sum and an i32 reduce_sum.
// Pre-WS-A1 the elem_type fallback to "float" would have silently
// downgraded every non-f32 path; post-WS-A0 elem_type panics on
// unhandled dtypes and post-WS-A1 the f64 path is wired through the
// dtype-parameterized accumulator. This test exercises both dtypes
// round-tripping through the same compiled kernel without one
// silently corrupting the other.
//
// (i32 *index*-indexed Gather is the obvious mixed-dtype shape but
// the existing C-backend Gather code path treats non-i64 indices as
// `float*`-typed data — a pre-WS-A1 limitation that would conflate
// the i32 codegen with an unrelated bug. Side-by-side reductions
// keep the test focused on WS-A1's actual guarantee: that f64 and
// i32 dtype dispatch coexist without aliasing.)
#[test]
fn ws_a1_exec_mixed_f64_tensors_and_i32_indices_compile_and_run() {
    let mut dag = Dag::new();
    let f_values = dag.add_node(
        RiscOp::Load {
            name: "f_values".into(),
        },
        vec![],
        vec_f64(4),
        None,
    );
    let i_values = dag.add_node(
        RiscOp::Load {
            name: "i_values".into(),
        },
        vec![],
        vec_i32(4),
        None,
    );
    // Two reductions: one f64 (default f64 accumulator), one i32
    // (default i32 accumulator). Both Stores so codegen marks both as
    // outputs.
    let f_sum = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![f_values],
        scalar_f64(),
        None,
    );
    let i_sum = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![i_values],
        scalar_i32(),
        None,
    );
    dag.add_node(
        RiscOp::Store {
            name: "f_out".into(),
        },
        vec![f_sum],
        scalar_f64(),
        None,
    );
    dag.add_node(
        RiscOp::Store {
            name: "i_out".into(),
        },
        vec![i_sum],
        scalar_i32(),
        None,
    );

    let result = codegen(&dag, "test_mixed").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("CHELIS_DTYPE_F64"),
        "mixed-dtype program must use CHELIS_DTYPE_F64 for f64 tensors:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I32"),
        "mixed-dtype program must use CHELIS_DTYPE_I32 for i32 tensors:\n{src}"
    );
    assert!(
        src.contains("double *__sum_level_"),
        "mixed-dtype program must use a double accumulator for the f64 sum:\n{src}"
    );
    assert!(
        src.contains("int32_t *__sum_level_"),
        "mixed-dtype program must use an int32_t accumulator for the i32 sum:\n{src}"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void test_mixed(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    double f_data[4] = {{1.5, 2.5, 3.5, 4.5}};
    int32_t i_data[4] = {{100, 200, 300, 400}};
    double f_expected = 1.5 + 2.5 + 3.5 + 4.5;          // 12.0
    int32_t i_expected = 100 + 200 + 300 + 400;         // 1000

    chelis_tensor *f_t = make_view_typed_1d(f_data, 4, CHELIS_DTYPE_F64);
    chelis_tensor *i_t = make_view_typed_1d(i_data, 4, CHELIS_DTYPE_I32);

    chelis_tensor* inputs[2] = {{f_t, i_t}};
    chelis_tensor* outputs[2] = {{NULL, NULL}};

    test_mixed(inputs, 2, outputs, 2);

    int ok = 1;
    if (chelis_tensor_read_view(outputs[0]).dtype != CHELIS_DTYPE_F64) {{
        printf("FAIL: f_out dtype is %d, expected CHELIS_DTYPE_F64\n", chelis_tensor_read_view(outputs[0]).dtype);
        ok = 0;
    }}
    if (chelis_tensor_read_view(outputs[1]).dtype != CHELIS_DTYPE_I32) {{
        printf("FAIL: i_out dtype is %d, expected CHELIS_DTYPE_I32\n", chelis_tensor_read_view(outputs[1]).dtype);
        ok = 0;
    }}
    double f_got = ((double*)chelis_tensor_read_view(outputs[0]).data)[0];
    int32_t i_got = ((int32_t*)chelis_tensor_read_view(outputs[1]).data)[0];
    double f_diff = fabs(f_got - f_expected);
    printf("mixed: f_got=%.6f expected=%.6f diff=%.12f i_got=%d expected=%d\n",
           f_got, f_expected, f_diff, i_got, i_expected);
    if (f_diff > 1e-9) {{ printf("FAIL: f64 sum off\n"); ok = 0; }}
    if (i_got != i_expected) {{ printf("FAIL: i32 sum off\n"); ok = 0; }}
    // Verify input storage was not cross-corrupted by the dual-dtype path.
    if (i_data[0] != 100 || i_data[3] != 400) {{
        printf("FAIL: i32 input storage corrupted\n"); ok = 0;
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a1_mixed_dtype", src, &harness) else {
        panic!("WS-A1 mixed-dtype kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "WS-A1 mixed-dtype output wrong:\n{output}"
    );
}

// ---- WS-A1 Negative parity: integer matmul errors with useful diagnostic ----
//
// Spec §5.7.2: integer matmul (i8/i16/i32/i64) is not admitted in this
// cycle. The IR-level rejection lives in
// `RiscOp::default_matmul_accumulator` (returns Err for integer
// operands), so the matmul constructor cannot even build an integer
// BlasMatmul; the error message must cite the spec and is the
// negative-parity twin of the f64 acceptance test above.
#[test]
fn ws_a1_negative_integer_matmul_rejected_with_spec_citation() {
    // i32 operand: matmul_default must error.
    let err_i32 = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Int32,
    )
    .expect_err("i32 matmul must be rejected per spec §5.7.2");
    assert!(
        err_i32.contains("spec/04-type-system.md §5.7.2"),
        "i32 matmul rejection must cite §5.7.2; got: {err_i32}"
    );
    assert!(
        err_i32.contains("integer"),
        "i32 matmul rejection must say 'integer'; got: {err_i32}"
    );

    // i64 operand: matmul_default must error.
    let err_i64 = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::Int64,
    )
    .expect_err("i64 matmul must be rejected per spec §5.7.2");
    assert!(
        err_i64.contains("spec/04-type-system.md §5.7.2"),
        "i64 matmul rejection must cite §5.7.2; got: {err_i64}"
    );

    // i8 / i16 must also error.
    for prim in [Prim::Int8, Prim::Int16] {
        let err = RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(4),
            DimExpr::Concrete(3),
            prim,
        )
        .expect_err(&format!("{prim:?} matmul must be rejected per spec §5.7.2"));
        assert!(
            err.contains("§5.7.2"),
            "{prim:?} matmul rejection must cite §5.7.2; got: {err}"
        );
    }
}

// ---- WS-A1 / WS-A3: bf16/f16 matmul admitted at IR validation ----
//
// Spec §5.7.1: bf16/f16 matmul accumulator default is f32. WS-A3
// wired the HIP backend's bf16/f16 dispatch through `hipblasGemmEx`
// with `HIPBLAS_COMPUTE_32F`, so the F1 IR validation guard now
// admits bf16/f16. The C backend still rejects bf16/f16 at its own
// F1 guard (no `cblas_*` dispatch yet); this IR-level test pins
// that the validation guard no longer catches the bf16/f16 case.
// Replaces the prior `ws_a1_negative_bf16_f16_matmul_still_blocked_by_f1_guard`
// assertion per the spec-sync rule that lifted-guard tests be
// REPLACED, not silently deleted.
#[test]
fn ws_a3_bf16_f16_matmul_admitted_at_ir_validation() {
    use chelis_ir::verify;
    for prim in [Prim::Bf16, Prim::F16] {
        let mut dag = Dag::new();
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(3)],
            precision: prim,
        };
        let ty_b = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(4)],
            precision: prim,
        };
        let ty_out = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(4)],
            precision: prim,
        };
        let a = dag.add_node(RiscOp::synth_const(ty.precision, 1.0), vec![], ty, None);
        let b = dag.add_node(RiscOp::synth_const(ty_b.precision, 1.0), vec![], ty_b, None);
        let matmul_op = RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(4),
            DimExpr::Concrete(3),
            prim,
        )
        .unwrap_or_else(|e| panic!("{prim:?} matmul default constructs (per spec §5.7.1): {e}"));
        let _ = dag.add_node(matmul_op, vec![a, b], ty_out, None);

        let errors = verify::verify(&dag);
        assert!(
            !errors.iter().any(|m| m.contains("F1: BlasMatmul")),
            "WS-A3 lifted {prim:?} from the F1 guard; {prim:?} BlasMatmul must validate cleanly. \
             Got: {errors:?}"
        );
    }
}

/// Header for i8/i16-typed harnesses. Reuses the runtime tensor view
/// shape but reinterprets `chelis_tensor_read_view(t).data` as the narrow integer pointer.
const WS_A4_HARNESS_HEADER: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "chelis_runtime.h"

static chelis_tensor *make_view_1d_i8(int8_t* data, int n) {
    int64_t shape[1] = {n};
    return chelis_tensor_entry_borrow(
        1, shape, CHELIS_DTYPE_I8, data, n * (int64_t)sizeof(int8_t)
    );
}

static chelis_tensor *make_view_1d_i16(int16_t* data, int n) {
    int64_t shape[1] = {n};
    return chelis_tensor_entry_borrow(
        1, shape, CHELIS_DTYPE_I16, data, n * (int64_t)sizeof(int16_t)
    );
}
"#;

/// Build a `Load → Op → Op` DAG with two same-precision i8 inputs and
/// emit the C source. Covers the i8 add path through the dtype-aware
/// `elem_type` and `dtype_macro`. Tensors are vec_i8(N) so the codegen
/// reinterpret-casts `chelis_tensor_read_view(t).data` to `int8_t*`.
#[test]
fn exec_i8_add_correct_output() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(8), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i8(8), None);
    dag.add_node(RiscOp::Add, vec![a, b], vec_i8(8), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i8_add").unwrap();
    let src = &result.c_source;

    // The kernel must emit `int8_t*` access against `chelis_tensor_read_view(t).data` (not float*).
    assert!(
        src.contains("int8_t"),
        "i8 add codegen must mention int8_t element type; got:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I8"),
        "i8 add codegen must allocate output via CHELIS_DTYPE_I8; got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_add(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int8_t a_data[8] = {{ 1, 2, 3, 4, -5, -6, 7, 8 }};
    int8_t b_data[8] = {{ 10, 20, 30, 40, 50, 60, -70, -80 }};
    int8_t expected[8];
    for (int i = 0; i < 8; i++) {{
        // Two's-complement wrapping at the i8 width; matches the
        // backend's `int8_t + int8_t` semantics.
        expected[i] = (int8_t)((int)a_data[i] + (int)b_data[i]);
    }}
    chelis_tensor *at = make_view_1d_i8(a_data, 8);
    chelis_tensor *bt = make_view_1d_i8(b_data, 8);
    chelis_tensor* in_ptrs[2] = {{ at, bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_add(in_ptrs, 2, outs, 1);
    int ok = 1;
    for (int i = 0; i < 8; i++) {{
        int8_t got = ((int8_t*)chelis_tensor_read_view(outs[0]).data)[i];
        if (got != expected[i]) {{
            printf("MISMATCH at %d: got %d expected %d\n", i, (int)got, (int)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i8_add", src, &harness) else {
        panic!("i8 add kernel failed to compile/run");
    };
    assert!(output.contains("PASS"), "i8 add wrong output:\n{output}");
}

/// i8 + i8 overflow traps at the declared width before C can execute a
/// narrowing conversion. This locks the Phase 3 checked-arithmetic contract
/// at the backend's direct compile-run surface.
#[test]
fn exec_i8_add_overflow_traps() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(2), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i8(2), None);
    dag.add_node(RiscOp::Add, vec![a, b], vec_i8(2), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i8_add_wrap").unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_add_wrap(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // Both elements overflow int8 and must trap before store-back.
    int8_t a_data[2] = {{ 100, 127 }};
    int8_t b_data[2] = {{ 50, 1 }};
    chelis_tensor *at = make_view_1d_i8(a_data, 2);
    chelis_tensor *bt = make_view_1d_i8(b_data, 2);
    chelis_tensor* in_ptrs[2] = {{ at, bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_add_wrap(in_ptrs, 2, outs, 1);
    return 0;
}}
"#
    );

    let run = compile_and_capture_run("ws_a4_i8_add_wrap", src, &harness);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "i8 add overflow must terminate unsuccessfully"
    );
    assert!(
        stderr.contains("numeric trap: overflow in add at int8"),
        "i8 add overflow must use the canonical diagnostic; stderr={stderr:?}"
    );
}

/// The second i8 multiplication overflows after an in-range first element;
/// the kernel must trap rather than partially legitimizing the wrapped row.
#[test]
fn exec_i8_mul_overflow_traps() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(2), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i8(2), None);
    dag.add_node(RiscOp::Mul, vec![a, b], vec_i8(2), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i8_mul").unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_mul(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int8_t a_data[2] = {{ 12, 16 }};
    int8_t b_data[2] = {{ 10, 8 }};
    // 12*10 = 120 fits; 16*8 = 128 overflows int8 and must trap.
    chelis_tensor *at = make_view_1d_i8(a_data, 2);
    chelis_tensor *bt = make_view_1d_i8(b_data, 2);
    chelis_tensor* in_ptrs[2] = {{ at, bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_mul(in_ptrs, 2, outs, 1);
    return 0;
}}
"#
    );

    let run = compile_and_capture_run("ws_a4_i8_mul", src, &harness);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success(),
        "i8 mul overflow must terminate unsuccessfully"
    );
    assert!(
        stderr.contains("numeric trap: overflow in mul at int8"),
        "i8 mul overflow must use the canonical diagnostic; stderr={stderr:?}"
    );
}

/// i16 add: pick values that exercise the int16_t path through the
/// codegen without overflowing. Mirrors `exec_i8_add_correct_output`
/// for the i16 dtype.
#[test]
fn exec_i16_add_correct_output() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i16(4), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], vec_i16(4), None);
    dag.add_node(RiscOp::Add, vec![a, b], vec_i16(4), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i16_add").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("int16_t"),
        "i16 add codegen must mention int16_t element type; got:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I16"),
        "i16 add codegen must allocate output via CHELIS_DTYPE_I16; got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i16_add(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int16_t a_data[4] = {{ 1000, -2000, 30000, -32000 }};
    int16_t b_data[4] = {{ 500, -1000, -20000, 1000 }};
    int16_t expected[4];
    for (int i = 0; i < 4; i++) {{
        expected[i] = (int16_t)((int)a_data[i] + (int)b_data[i]);
    }}
    chelis_tensor *at = make_view_1d_i16(a_data, 4);
    chelis_tensor *bt = make_view_1d_i16(b_data, 4);
    chelis_tensor* in_ptrs[2] = {{ at, bt }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i16_add(in_ptrs, 2, outs, 1);
    int ok = 1;
    for (int i = 0; i < 4; i++) {{
        int16_t got = ((int16_t*)chelis_tensor_read_view(outs[0]).data)[i];
        if (got != expected[i]) {{
            printf("MISMATCH at %d: got %d expected %d\n", i, (int)got, (int)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i16_add", src, &harness) else {
        panic!("i16 add kernel failed to compile/run");
    };
    assert!(output.contains("PASS"), "i16 add wrong output:\n{output}");
}

/// i8 reduce_sum into the spec-default i32 accumulator (the WS-0
/// pinned promotion rule per §5.7.1). 200 ones at i8 source overflows
/// i8 (max +127); the i32 accumulator + i32 result must yield exactly
/// 200, no overflow, no panic.
#[test]
fn exec_i8_reduce_sum_promotes_to_i32() {
    let mut dag = Dag::new();
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], vec_i8(200), None);
    // Use the spec-default constructor so the IR carries the §5.7.1
    // i32 accumulator, not an inline `Prim::Int8` that would fail the
    // verifier's narrowness check.
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int8).expect("i8 sum_default must succeed");
    dag.add_node(sum_op, vec![a], scalar_i32(), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i8_reduce_sum").unwrap();
    let src = &result.c_source;

    // The accumulator type in the emitted C must be int32_t — pinning
    // this catches a regression where the codegen silently picks the
    // operand precision (the F1 footgun class for reductions).
    assert!(
        src.contains("int32_t *__sum_level_"),
        "i8 reduce_sum must accumulate in int32_t (per spec §5.7.1); got:\n{src}"
    );
    assert!(
        src.contains("CHELIS_DTYPE_I32"),
        "i8 reduce_sum output tensor must be allocated via CHELIS_DTYPE_I32; got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_reduce_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int8_t in_data[200];
    for (int i = 0; i < 200; i++) in_data[i] = 1;
    chelis_tensor *in_t = make_view_1d_i8(in_data, 200);
    chelis_tensor* in_ptrs[1] = {{ in_t }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i8_reduce_sum(in_ptrs, 1, outs, 1);

    int32_t got = ((int32_t*)chelis_tensor_read_view(outs[0]).data)[0];
    int32_t expected = 200;
    printf("sum(200 i8 ones): got=%d expected=%d\n", got, expected);
    printf("%s\n", got == expected ? "PASS" : "FAIL");
    return got == expected ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i8_reduce_sum_200", src, &harness) else {
        panic!("i8 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "i8 reduce_sum did not promote to i32 / produced wrong sum:\n{output}"
    );
}

/// i16 reduce_sum into i32: same accumulator-promotion path as i8.
/// 200 i16 values of 1000 each = 200000 — overflows i16 (max +32767)
/// but fits in i32. Pin both the source-int16 path and the i32 result.
#[test]
fn exec_i16_reduce_sum_promotes_to_i32() {
    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i16(200),
        None,
    );
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int16).expect("i16 sum_default must succeed");
    dag.add_node(sum_op, vec![a], scalar_i32(), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i16_reduce_sum").unwrap();
    let src = &result.c_source;

    assert!(
        src.contains("int32_t *__sum_level_"),
        "i16 reduce_sum must accumulate in int32_t (per spec §5.7.1); got:\n{src}"
    );

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i16_reduce_sum(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int16_t in_data[200];
    for (int i = 0; i < 200; i++) in_data[i] = 1000;
    chelis_tensor *in_t = make_view_1d_i16(in_data, 200);
    chelis_tensor* in_ptrs[1] = {{ in_t }};
    chelis_tensor* outs[1] = {{ NULL }};
    test_i16_reduce_sum(in_ptrs, 1, outs, 1);

    int32_t got = ((int32_t*)chelis_tensor_read_view(outs[0]).data)[0];
    int32_t expected = 200000; // overflows i16 (max +32767), fits in i32
    printf("sum(200 i16 1000s): got=%d expected=%d\n", got, expected);
    printf("%s\n", got == expected ? "PASS" : "FAIL");
    return got == expected ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel("ws_a4_i16_reduce_sum_200", src, &harness) else {
        panic!("i16 reduce_sum kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "i16 reduce_sum did not promote to i32 / produced wrong sum:\n{output}"
    );
}

/// Negative test: the spec/04-type-system.md §5.7.1 narrowness rule
/// says a user cannot request a narrower-than-default accumulator.
/// `sum_with_accumulator(_, Int8, Int8)` must error. This locks the
/// IR-side rejection helper that the WS-A0-Fixups verifier installs;
/// the codegen path never sees the malformed IR.
#[test]
fn ws_a4_i8_sum_with_narrower_accumulator_is_ir_error() {
    let err = chelis_ir::dag::RiscOp::sum_with_accumulator(0, Prim::Int8, Prim::Int8)
        .expect_err("i8 reduce_sum with i8 accumulator must be rejected by sum_with_accumulator");
    assert!(
        err.contains("narrower than"),
        "i8 reduce_sum with i8 accumulator must reference the narrowness rule; got: {err}"
    );
    assert!(
        err.contains("§5.7.1"),
        "rejection diagnostic must cite spec §5.7.1; got: {err}"
    );
}

// ============================================================
// #517: emit_cmplt must read its operands through their OWN
// element dtype, not the boolean (f32) output dtype. The cmplt
// signature is `∀D,p. (tensor[D,p], tensor[D,p]) → tensor[D,bool]`,
// so the operands carry the compared precision `p` while the result
// is bool (stored f32). Reading a RUNTIME-PRODUCED int32 / int64 /
// f64 operand through a raw `float*` reinterprets the bit pattern
// (the #347 / #476 bug class). The discriminator is a NEGATIVE
// integer operand: as int32, `-7 < -3` is true; reinterpreting the
// int32 bit pattern 0xFFFFFFF9 / 0xFFFFFFFD as `float` yields NaN, so
// the buggy `float*` read returns false. eval-vs-C parity oracle.
// ============================================================

fn vec_prim(n: usize, p: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: p,
    }
}

/// Build `cmplt(a, b)` over two runtime Load operands of precision
/// `prim`, evaluate the chelis-ir oracle, compile + run the generated
/// C kernel, and assert the C bool output matches the evaluator
/// element-for-element. `c_elem` / `c_dtype` describe the operand's C
/// storage type and runtime dtype tag; values are passed as f64 (exact
/// for the integer and small-double cases used here).
fn run_cmplt_parity(
    tag: &str,
    prim: Prim,
    c_elem: &str,
    c_dtype: &str,
    a_vals: &[f64],
    b_vals: &[f64],
) {
    use chelis_ir::eval::{TensorValue, eval_tensor};
    use chelis_unord::UnordMap;

    let n = a_vals.len();
    assert_eq!(n, b_vals.len(), "operand length mismatch in {tag}");

    let mut dag = Dag::new();
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_prim(n, prim),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_prim(n, prim),
        None,
    );
    let root = dag.add_node(RiscOp::CmpLt, vec![a, b], vec_prim(n, Prim::Bool), None);

    // Evaluator oracle: numeric `a < b` per element (eval.rs CmpLt).
    let mut inputs = UnordMap::new();
    inputs.insert(
        "a".to_string(),
        TensorValue::from_vec(vec![n], a_vals.to_vec()),
    );
    inputs.insert(
        "b".to_string(),
        TensorValue::from_vec(vec![n], b_vals.to_vec()),
    );
    let evaluated = eval_tensor(&dag, &inputs).expect("evaluator must succeed");
    let expected: Vec<f64> = evaluated[&root].to_f64_lossy_vec().clone();
    assert_eq!(expected.len(), n);

    let dag = fuse(&dag);
    let result =
        codegen_with_options(&dag, &format!("cmplt_{tag}"), CodegenOptions::default()).unwrap();
    let src = &result.c_source;

    // Format the operand initializers and the expected bool vector.
    let fmt_vals = |vals: &[f64]| -> String {
        vals.iter()
            .map(|v| {
                if prim == Prim::F64 {
                    format!("{v:?}")
                } else {
                    format!("{}", *v as i64)
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let a_init = fmt_vals(a_vals);
    let b_init = fmt_vals(b_vals);
    let exp_init = expected
        .iter()
        .map(|v| format!("{}", *v as u8))
        .collect::<Vec<_>>()
        .join(", ");

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>

extern void cmplt_{tag}(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    {c_elem} a_data[{n}] = {{{a_init}}};
    {c_elem} b_data[{n}] = {{{b_init}}};
    chelis_tensor *a_t = make_view_typed_1d(a_data, {n}, {c_dtype});
    chelis_tensor *b_t = make_view_typed_1d(b_data, {n}, {c_dtype});
    chelis_tensor* inputs[2] = {{a_t, b_t}};
    chelis_tensor* outputs[1] = {{NULL}};

    cmplt_{tag}(inputs, 2, outputs, 1);

    uint8_t expected[{n}] = {{{exp_init}}};
    int ok = 1;
    if (chelis_tensor_read_view(outputs[0]).dtype != CHELIS_DTYPE_BOOL) {{
        printf("MISMATCH dtype: got %u expected CHELIS_DTYPE_BOOL (%u)\n",
               (unsigned)chelis_tensor_read_view(outputs[0]).dtype, (unsigned)CHELIS_DTYPE_BOOL);
        ok = 0;
    }}
    for (int i = 0; i < {n}; i++) {{
        uint8_t got = ((uint8_t*)chelis_tensor_read_view(outputs[0]).data)[i];
        if (got != expected[i]) {{
            printf("MISMATCH at %d: got %u expected %u\n",
                   i, (unsigned)got, (unsigned)expected[i]);
            ok = 0;
        }}
    }}
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}}
"#
    );

    let Some(output) = compile_and_run_kernel(&format!("cmplt_{tag}"), src, &harness) else {
        panic!("#517 cmplt parity [{tag}]: kernel failed to compile/run");
    };
    assert!(
        output.contains("PASS"),
        "#517 cmplt parity [{tag}]: C backend disagreed with evaluator.\n\
         Generated C:\n{src}\nRun output:\n{output}"
    );
}

/// #517 primary oracle: runtime int32 operands, including the negative
/// values that the `float*` bit-reinterpret gets wrong.
#[test]
fn exec_cmplt_int32_runtime_operands_match_evaluator() {
    run_cmplt_parity(
        "i32",
        Prim::Int32,
        "int32_t",
        "CHELIS_DTYPE_I32",
        &[-7.0, 2.0, -5.0, 10.0, 3.0, -1.0],
        &[-3.0, 10.0, 3.0, 2.0, 3.0, -1.0],
    );
}

/// #517 sweep: int64 operands. The buggy `float*` read also misaligns
/// the 8-byte stride; reading as `int64_t*` is required.
#[test]
fn exec_cmplt_int64_runtime_operands_match_evaluator() {
    run_cmplt_parity(
        "i64",
        Prim::Int64,
        "int64_t",
        "CHELIS_DTYPE_I64",
        &[-7.0, 2.0, -5.0, 100.0, 3.0],
        &[-3.0, 100.0, 3.0, 2.0, 3.0],
    );
}

/// #517 sweep: f64 operands. f64 read through `float*` truncates the
/// 8-byte payload to 4 bytes; reading as `double*` is required.
#[test]
fn exec_cmplt_f64_runtime_operands_match_evaluator() {
    run_cmplt_parity(
        "f64",
        Prim::F64,
        "double",
        "CHELIS_DTYPE_F64",
        &[-7.5, 2.25, -5.0, 10.0, 3.0],
        &[-3.5, 10.0, 3.0, 2.0, 3.0],
    );
}

fn direct_int_sub_case(
    tag: &str,
    prim: Prim,
    c_type: &str,
    c_dtype: &str,
    lhs: &str,
    rhs: &str,
    expected: &str,
) {
    let mut dag = Dag::new();
    let ty = vec_prim(4, prim);
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
    dag.add_node(RiscOp::Sub, vec![a, b], ty, None);
    let function = format!("direct_sub_{tag}");
    let src = codegen(&dag, &function)
        .expect("direct subtraction codegen")
        .c_source;
    assert!(src.contains("chelis_int_checked_sub"), "{tag}: {src}");
    assert!(!src.contains("chelis_int_checked_add("), "{tag}: {src}");

    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
#include <limits.h>

extern void {function}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {{
    {c_type} a_data[4] = {{ {lhs} }};
    {c_type} b_data[4] = {{ {rhs} }};
    {c_type} expected[4] = {{ {expected} }};
    chelis_tensor *a = make_view_typed_1d(a_data, 4, {c_dtype});
    chelis_tensor *b = make_view_typed_1d(b_data, 4, {c_dtype});
    chelis_tensor *inputs[2] = {{ a, b }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    {c_type} *got = ({c_type} *)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < 4; i++) {{
        if (got[i] != expected[i]) return 1;
    }}
    puts("PASS");
    return 0;
}}
"#
    );
    let output = compile_and_run_kernel(&function, &src, &harness)
        .unwrap_or_else(|| panic!("{tag} direct subtraction did not compile and run"));
    assert!(output.contains("PASS"), "{tag}: {output}");
}

#[test]
fn direct_checked_subtraction_executes_exact_boundaries_at_every_signed_width() {
    for case in [
        (
            "i8",
            Prim::Int8,
            "int8_t",
            "CHELIS_DTYPE_I8",
            "-1, INT8_MAX, INT8_MIN, 3",
            "INT8_MIN, 1, -1, -4",
            "INT8_MAX, INT8_MAX - 1, INT8_MIN + 1, 7",
        ),
        (
            "i16",
            Prim::Int16,
            "int16_t",
            "CHELIS_DTYPE_I16",
            "-1, INT16_MAX, INT16_MIN, 3",
            "INT16_MIN, 1, -1, -4",
            "INT16_MAX, INT16_MAX - 1, INT16_MIN + 1, 7",
        ),
        (
            "i32",
            Prim::Int32,
            "int32_t",
            "CHELIS_DTYPE_I32",
            "-1, INT32_MAX, INT32_MIN, 3",
            "INT32_MIN, 1, -1, -4",
            "INT32_MAX, INT32_MAX - 1, INT32_MIN + 1, 7",
        ),
        (
            "i64",
            Prim::Int64,
            "int64_t",
            "CHELIS_DTYPE_I64",
            "-1, INT64_MAX, INT64_MIN, 3",
            "INT64_MIN, 1, -1, -4",
            "INT64_MAX, INT64_MAX - 1, INT64_MIN + 1, 7",
        ),
    ] {
        direct_int_sub_case(case.0, case.1, case.2, case.3, case.4, case.5, case.6);
    }
}

#[test]
fn direct_checked_subtraction_traps_true_overflow_at_every_signed_width() {
    for (tag, prim, c_type, c_dtype, max) in [
        ("i8", Prim::Int8, "int8_t", "CHELIS_DTYPE_I8", "INT8_MAX"),
        (
            "i16",
            Prim::Int16,
            "int16_t",
            "CHELIS_DTYPE_I16",
            "INT16_MAX",
        ),
        (
            "i32",
            Prim::Int32,
            "int32_t",
            "CHELIS_DTYPE_I32",
            "INT32_MAX",
        ),
        (
            "i64",
            Prim::Int64,
            "int64_t",
            "CHELIS_DTYPE_I64",
            "INT64_MAX",
        ),
    ] {
        let mut dag = Dag::new();
        let ty = vec_prim(1, prim);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        dag.add_node(RiscOp::Sub, vec![a, b], ty, None);
        let function = format!("direct_sub_overflow_{tag}");
        let src = codegen(&dag, &function).unwrap().c_source;
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
#include <limits.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} av[1] = {{ {max} }}; {c_type} bv[1] = {{ -1 }};
    chelis_tensor *a = make_view_typed_1d(av, 1, {c_dtype});
    chelis_tensor *b = make_view_typed_1d(bv, 1, {c_dtype});
    chelis_tensor *inputs[2] = {{ a, b }}; chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1); return 0;
}}
"#
        );
        let run = compile_and_capture_run(&function, &src, &harness);
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            !run.status.success(),
            "{tag} overflow unexpectedly succeeded"
        );
        assert!(
            stderr.contains(&format!("numeric trap: overflow in sub at {}", prim.name())),
            "{tag}: {stderr}"
        );
    }
}

#[test]
fn direct_signed_integer_extrema_chains_survive_fusion_and_execute_at_every_width() {
    for (tag, prim, c_type, c_dtype) in [
        ("i8", Prim::Int8, "int8_t", "CHELIS_DTYPE_I8"),
        ("i16", Prim::Int16, "int16_t", "CHELIS_DTYPE_I16"),
        ("i32", Prim::Int32, "int32_t", "CHELIS_DTYPE_I32"),
        ("i64", Prim::Int64, "int64_t", "CHELIS_DTYPE_I64"),
    ] {
        let mut dag = Dag::new();
        let ty = vec_prim(4, prim);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        let c = dag.add_node(RiscOp::Load { name: "c".into() }, vec![], ty.clone(), None);
        let maximum = dag.add_node(RiscOp::MaxElem, vec![a, b], ty.clone(), None);
        let minimum = dag.add_node(RiscOp::MinElem, vec![maximum, c], ty, None);
        dag.add_root(minimum);

        let fused = fuse(&dag);
        assert!(
            fused
                .nodes()
                .iter()
                .all(|node| !matches!(node.op, RiscOp::FusedElem { .. })),
            "{tag}: signed-integer extrema must stay materialized"
        );
        let function = format!("direct_integer_extrema_chain_{tag}");
        let src = codegen(&fused, &function)
            .unwrap_or_else(|error| panic!("{tag}: fused direct extrema codegen failed: {error}"))
            .c_source;
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} a_data[4] = {{ -5, 7, 3, 0 }};
    {c_type} b_data[4] = {{ -4, 7, -9, 5 }};
    {c_type} c_data[4] = {{ -6, 6, 4, 5 }};
    {c_type} expected[4] = {{ -6, 6, 3, 5 }};
    chelis_tensor *a = make_view_typed_1d(a_data, 4, {c_dtype});
    chelis_tensor *b = make_view_typed_1d(b_data, 4, {c_dtype});
    chelis_tensor *c = make_view_typed_1d(c_data, 4, {c_dtype});
    chelis_tensor *inputs[3] = {{ a, b, c }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 3, outputs, 1);
    {c_type} *got = ({c_type} *)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < 4; i++) if (got[i] != expected[i]) return 1;
    puts("PASS");
    return 0;
}}
"#
        );
        let output = compile_and_run_kernel(&function, &src, &harness)
            .unwrap_or_else(|| panic!("{tag}: integer extrema chain did not compile and run"));
        assert!(output.contains("PASS"), "{tag}: {output}");
    }
}

#[test]
fn direct_bool_max_elem_compiles_without_float_classification() {
    let mut dag = Dag::new();
    let ty = vec_prim(4, Prim::Bool);
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
    let maximum = dag.add_node(RiscOp::MaxElem, vec![a, b], ty, None);
    dag.add_root(maximum);

    let function = "direct_bool_max_elem";
    let src = codegen(&dag, function)
        .expect("Bool max_elem codegen")
        .c_source;
    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint8_t a_data[4] = {{ 0, 0, 1, 1 }};
    uint8_t b_data[4] = {{ 0, 1, 0, 1 }};
    uint8_t expected[4] = {{ 0, 1, 1, 1 }};
    chelis_tensor *a = make_view_typed_1d(a_data, 4, CHELIS_DTYPE_BOOL);
    chelis_tensor *b = make_view_typed_1d(b_data, 4, CHELIS_DTYPE_BOOL);
    chelis_tensor *inputs[2] = {{ a, b }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    uint8_t *got = (uint8_t *)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < 4; i++) if (got[i] != expected[i]) return 1;
    puts("PASS");
    return 0;
}}
"#
    );
    let output = compile_and_run_kernel(function, &src, &harness)
        .expect("Bool max_elem generated C must compile and run");
    assert!(output.contains("PASS"), "{output}");
    assert!(src.contains("uint8_t"), "Bool storage must stay byte-typed");
    assert!(
        !src.contains("isnan("),
        "Bool max_elem must not enter the floating extrema branch: {src}"
    );
}

#[test]
fn direct_fused_runtime_shape_mismatch_traps_before_indexing() {
    let runtime_vec = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let n_ty = runtime_vec("n");
    let m_ty = runtime_vec("m");
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        n_ty.clone(),
        None,
    );
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], m_ty, None);
    let c = dag.add_node(
        RiscOp::Load { name: "c".into() },
        vec![],
        n_ty.clone(),
        None,
    );
    let difference = dag.add_node(RiscOp::Sub, vec![a, b], n_ty.clone(), None);
    let minimum = dag.add_node(RiscOp::MinElem, vec![difference, c], n_ty, None);
    dag.add_root(minimum);

    let fused = fuse(&dag);
    assert!(
        fused
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::FusedElem { .. })),
        "float Sub -> MinElem must exercise the fused path"
    );
    let function = "direct_fused_runtime_shape_guard";
    let src = codegen(&fused, function)
        .expect("fused runtime-shape codegen")
        .c_source;
    assert!(
        src.contains("elementwise operand shape mismatch"),
        "fused codegen dropped the deferred operand-shape guard:\n{src}"
    );
    assert!(
        src.contains("if (t1_rank == t2_rank)"),
        "fused codegen must compare every equal-rank external-input pair:\n{src}"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
static chelis_tensor *make_distinct_view(float *data, int64_t *shape) {{
    return chelis_tensor_entry_borrow(
        1, shape, CHELIS_DTYPE_F32, data, shape[0] * (int64_t)sizeof(float)
    );
}}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float a_data[3] = {{ 1, 2, 3 }};
    float b_data[2] = {{ 1, 2 }};
    float c_data[3] = {{ 4, 5, 6 }};
    int64_t a_shape[1] = {{3}}, b_shape[1] = {{2}}, c_shape[1] = {{3}};
    chelis_tensor *a = make_distinct_view(a_data, a_shape);
    chelis_tensor *b = make_distinct_view(b_data, b_shape);
    chelis_tensor *c = make_distinct_view(c_data, c_shape);
    chelis_tensor *inputs[3] = {{ a, b, c }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 3, outputs, 1);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run("direct_fused_runtime_shape_guard", &src, &harness);
    assert!(
        !run.status.success(),
        "mismatched deferred dimensions reached the fused loop; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("elementwise operand shape mismatch"),
        "fused runtime-shape trap emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn direct_positive_rank_mismatch_traps_before_indexing() {
    let mut dag = Dag::new();
    let vector = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        vector.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        vector.clone(),
        None,
    );
    let out = dag.add_node(RiscOp::Add, vec![a, b], vector, None);
    dag.add_root(out);

    let function = "direct_positive_rank_shape_guard";
    let src = codegen(&dag, function)
        .expect("rank-divergent codegen must stay defensive")
        .c_source;
    assert!(
        src.contains("elementwise operand rank mismatch"),
        "generated C omitted the positive-rank mismatch guard:\n{src}"
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
static chelis_tensor *make_ranked_view(
    float *data, int rank, int64_t *shape, int64_t *strides, int64_t size
) {{
    (void)strides;
    return chelis_tensor_entry_borrow(
        rank, shape, CHELIS_DTYPE_F32, data, size * (int64_t)sizeof(float)
    );
}}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float a_data[3] = {{1, 2, 3}};
    float b_data[3] = {{4, 5, 6}};
    int64_t a_shape[1] = {{3}}, a_strides[1] = {{1}};
    int64_t b_shape[2] = {{3, 1}}, b_strides[2] = {{1, 1}};
    chelis_tensor *a = make_ranked_view(a_data, 1, a_shape, a_strides, 3);
    chelis_tensor *b = make_ranked_view(b_data, 2, b_shape, b_strides, 3);
    chelis_tensor *inputs[2] = {{a, b}};
    chelis_tensor *outputs[1] = {{NULL}};
    {function}(inputs, 2, outputs, 1);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run(function, &src, &harness);
    assert!(
        !run.status.success(),
        "positive-rank mismatch reached indexing; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("input `b` expected rank 1, got 2"),
        "rank guard emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

// ---- chelis#1484: the HOST-VALUE lane's elementwise operand guard -------
//
// `direct_positive_rank_mismatch_traps_before_indexing` above covers the
// tensor-DAG emitter (`src/emit.rs`). An elementwise call whose operand
// carries an IO effect (`debug(...)`) is lowered by a different emitter,
// `src/host_emit.rs`, which allocated the result at the LHS rank and then
// fed the TARGET's index vector into each operand's strides with no rank
// comparison at all. The four tests below drive that emitter directly
// through `emit_host_program`, the same entry the CLI's `chelis build`
// uses for host-lane functions.

fn host_tensor(dims: Vec<usize>) -> TensorType {
    TensorType {
        dims: dims.into_iter().map(DimInfo::Lit).collect(),
        precision: Prim::F32,
    }
}

/// A two-parameter host function `the_fn(a, b) = <builtin>(a, b)` whose
/// operands carry the given (possibly divergent) static shapes. `globals`
/// is empty, so `emit_host_program` emits an object-mode translation unit
/// with no `main` and the harness below supplies its own.
fn host_binary_program(builtin: &str, lhs: Vec<usize>, rhs: Vec<usize>) -> HostProgram {
    let lhs_ty = host_tensor(lhs);
    let rhs_ty = host_tensor(rhs);
    let body = HostExpr::new(HostExprKind::Builtin {
        name: builtin.to_string(),
        args: vec![
            HostExpr::new(HostExprKind::Var(
                "a".to_string(),
                HostType::Tensor(lhs_ty.clone()),
            )),
            HostExpr::new(HostExprKind::Var(
                "b".to_string(),
                HostType::Tensor(rhs_ty.clone()),
            )),
        ],
        ty: HostType::Tensor(lhs_ty.clone()),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            name: "the_fn".to_string(),
            params: vec![
                HostParam {
                    name: "a".to_string(),
                    ty: HostType::Tensor(lhs_ty.clone()),
                },
                HostParam {
                    name: "b".to_string(),
                    ty: HostType::Tensor(rhs_ty),
                },
            ],
            ret_ty: HostType::Tensor(lhs_ty),
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
    }
}

/// A `main` that hands `the_fn` two descriptors with the exact ranks,
/// shapes, and strides given, then prints `UNREACHABLE` if the call
/// returns. Both operands are backed by six floats so that a guard-free
/// build reads in bounds and exits 0 rather than crashing: the failure
/// this pins is the silent wrong answer, not the segfault.
const HOST_GUARD_HARNESS: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "chelis_runtime.h"

static chelis_tensor *make_ranked_view(
    float *data, int32_t rank, const int64_t *shape, const int64_t *strides, int64_t size
) {
    (void)strides;
    return chelis_tensor_entry_borrow(
        rank, shape, CHELIS_DTYPE_F32, data, size * (int64_t)sizeof(float)
    );
}

extern chelis_tensor *the_fn(chelis_tensor *, chelis_tensor *);
"#;

/// chelis#1484 REGRESSION TEST: red on the pre-fix tree (the emitted C
/// carried no rank guard and the binary exited 0), green after. A rank-1
/// LHS beside a rank-2 RHS must abort before the result allocation.
#[test]
fn host_lane_positive_rank_mismatch_traps_before_indexing() {
    let program = host_binary_program("add", vec![3], vec![2, 3]);
    let function = "host_positive_rank_guard";
    let src = emit_host_program(&program, function).expect("host-lane codegen");
    assert!(
        src.contains("elementwise operand rank mismatch"),
        "host-lane codegen omitted the positive-rank mismatch guard:\n{src}"
    );
    assert!(
        src.contains("chelis_host_require_elementwise_agreement(__arg0_"),
        "host-lane rank guard must call the shared opaque-tensor guard:\n{src}"
    );
    assert!(
        !src.contains("->rank"),
        "host-lane rank guard must not reopen the opaque tensor descriptor:\n{src}"
    );

    let harness = format!(
        r#"{HOST_GUARD_HARNESS}
int main(void) {{
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[6] = {{1, 2, 3, 4, 5, 6}};
    static const int64_t a_shape[1] = {{3}}, a_strides[1] = {{1}};
    static const int64_t b_shape[2] = {{2, 3}}, b_strides[2] = {{3, 1}};
    chelis_tensor *a = make_ranked_view(a_data, 1, a_shape, a_strides, 3);
    chelis_tensor *b = make_ranked_view(b_data, 2, b_shape, b_strides, 6);
    the_fn(a, b);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run(function, &src, &harness);
    assert!(
        !run.status.success(),
        "host-lane positive-rank mismatch reached indexing; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("elementwise operand rank mismatch"),
        "host-lane rank guard emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// chelis#1484 REGRESSION TEST: red before, green after. `max_elem` and
/// `min_elem` are emitted by `assign_tensor_binary_func_elementwise`, a
/// different function from the `+`/`-`/`*`/`/` one above, and needed its
/// own guard call.
#[test]
fn host_lane_max_elem_positive_rank_mismatch_traps_before_indexing() {
    let program = host_binary_program("max_elem", vec![3], vec![2, 3]);
    let function = "host_positive_rank_guard_max_elem";
    let src = emit_host_program(&program, function).expect("host-lane codegen");
    assert!(
        src.contains("elementwise operand rank mismatch"),
        "host-lane max_elem codegen omitted the positive-rank guard:\n{src}"
    );

    let harness = format!(
        r#"{HOST_GUARD_HARNESS}
int main(void) {{
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[6] = {{6, 5, 4, 3, 2, 1}};
    static const int64_t a_shape[1] = {{3}}, a_strides[1] = {{1}};
    static const int64_t b_shape[2] = {{2, 3}}, b_strides[2] = {{3, 1}};
    chelis_tensor *a = make_ranked_view(a_data, 1, a_shape, a_strides, 3);
    chelis_tensor *b = make_ranked_view(b_data, 2, b_shape, b_strides, 6);
    the_fn(a, b);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run(function, &src, &harness);
    assert!(
        !run.status.success(),
        "host-lane max_elem rank mismatch reached indexing; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("elementwise operand rank mismatch"),
        "host-lane max_elem guard emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// chelis#1484 REGRESSION TEST: red before, green after. At EQUAL positive
/// rank the host lane compared nothing either, so a `[3]` operand beside a
/// `[2]` one read past the shorter operand. The DAG lane has compared
/// shapes axis by axis since chelis#664; this is the host-lane sibling.
#[test]
fn host_lane_equal_rank_shape_mismatch_traps_before_indexing() {
    let program = host_binary_program("add", vec![3], vec![2]);
    let function = "host_equal_rank_shape_guard";
    let src = emit_host_program(&program, function).expect("host-lane codegen");
    assert!(
        src.contains("elementwise operand shape mismatch"),
        "host-lane codegen omitted the equal-rank shape guard:\n{src}"
    );
    assert!(
        src.contains("chelis_host_require_elementwise_agreement(__arg0_"),
        "host-lane shape guard must call the shared opaque-tensor guard:\n{src}"
    );
    assert!(
        !src.contains("->shape"),
        "host-lane shape guard must not reopen the opaque tensor descriptor:\n{src}"
    );

    let harness = format!(
        r#"{HOST_GUARD_HARNESS}
int main(void) {{
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[6] = {{1, 2, 3, 4, 5, 6}};
    static const int64_t a_shape[1] = {{3}}, a_strides[1] = {{1}};
    static const int64_t b_shape[1] = {{2}}, b_strides[1] = {{1}};
    chelis_tensor *a = make_ranked_view(a_data, 1, a_shape, a_strides, 3);
    chelis_tensor *b = make_ranked_view(b_data, 1, b_shape, b_strides, 2);
    the_fn(a, b);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run(function, &src, &harness);
    assert!(
        !run.status.success(),
        "host-lane equal-rank shape mismatch reached indexing; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("elementwise operand shape mismatch"),
        "host-lane shape guard emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// chelis#1484 DISPOSITION LOCK: green before the fix and green after. Its
/// job is to pin that the new guard does not false-abort the agreeing case
/// the host lane is actually for, and that it leaves the computed values
/// alone. Without it, "abort on every binary elementwise op" would pass
/// the three regression tests above.
#[test]
fn host_lane_matching_shapes_still_compute() {
    let program = host_binary_program("add", vec![2, 3], vec![2, 3]);
    let function = "host_matching_shape_control";
    let src = emit_host_program(&program, function).expect("host-lane codegen");

    let harness = format!(
        r#"{HOST_GUARD_HARNESS}
int main(void) {{
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[6] = {{10, 20, 30, 40, 50, 60}};
    static const int64_t shape[2] = {{2, 3}}, strides[2] = {{3, 1}};
    chelis_tensor *a = make_ranked_view(a_data, 2, shape, strides, 6);
    chelis_tensor *b = make_ranked_view(b_data, 2, shape, strides, 6);
    chelis_tensor *out = the_fn(a, b);
    const float *values = (const float *)chelis_tensor_read_view(out).data;
    for (int64_t i = 0; i < chelis_tensor_numel(out); i++) {{
        printf("%.1f\n", (double)values[i]);
    }}
    chelis_tensor_release(a);
    chelis_tensor_release(b);
    chelis_tensor_release(out);
    return 0;
}}
"#
    );
    let run = compile_and_capture_run(function, &src, &harness);
    assert!(
        run.status.success(),
        "matching host-lane shapes must not abort; stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout);
    let got: Vec<String> = stdout.split_whitespace().map(str::to_string).collect();
    assert_eq!(
        got,
        vec!["11.0", "22.0", "33.0", "44.0", "55.0", "66.0"],
        "host-lane add over matching shapes returned the wrong values: {stdout}"
    );
}

fn direct_fused_reduction_runtime_shape_guard_case(reduce_kind: &str) {
    let matrix = |row_name: &str, column_name: &str| TensorType {
        dims: vec![
            DimInfo::Named(row_name.into(), None),
            DimInfo::Named(column_name.into(), None),
        ],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let output_matrix_ty = matrix("rows", "columns");
    let other_matrix_ty = matrix("other_rows", "other_columns");
    let scalar = dag.add_node(
        RiscOp::Load {
            name: "scalar".into(),
        },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let a = dag.add_node(
        RiscOp::Load { name: "a".into() },
        vec![],
        output_matrix_ty.clone(),
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        other_matrix_ty,
        None,
    );
    let shifted = dag.add_node(RiscOp::Add, vec![scalar, a], output_matrix_ty.clone(), None);
    let product = dag.add_node(RiscOp::Mul, vec![shifted, b], output_matrix_ty, None);
    let output_ty = TensorType {
        dims: vec![DimInfo::Named("rows".into(), None)],
        precision: Prim::F32,
    };
    let reduced = match reduce_kind {
        "sum" => dag.add_node(
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![product],
            output_ty,
            None,
        ),
        "max" => dag.add_node(
            RiscOp::MaxReduce { axis: 1 },
            vec![product],
            output_ty,
            None,
        ),
        _ => unreachable!(),
    };
    dag.add_root(reduced);

    let fused = fuse(&dag);
    let fused_node = fused
        .nodes()
        .iter()
        .find(|node| matches!(node.op, RiscOp::FusedElem { .. }))
        .expect("Add -> Mul must form a FusedElem before reduction inlining");
    assert_eq!(
        fused_node.inputs,
        vec![scalar, a, b],
        "the scalar must be first so the guard proves all-pairs tensor comparison"
    );
    assert!(
        chelis_ir::fuse::reduction_inlined_fused_elems(&fused).contains(&fused_node.id),
        "the regression must exercise the reduction-inlined FusedElem path"
    );

    let function = format!("direct_fused_{reduce_kind}_runtime_shape_guard");
    let src = codegen(&fused, &function)
        .expect("fused reduction runtime-shape codegen")
        .c_source;
    let function_body = src
        .split_once(&format!("void {function}("))
        .unwrap_or_else(|| panic!("generated C omitted {function}:\n{src}"))
        .1;
    let guard_offset = function_body
        .find("elementwise operand shape mismatch")
        .unwrap_or_else(|| panic!("inlined {reduce_kind} dropped its shape guard:\n{src}"));
    let allocation_offset = function_body
        .find("chelis_alloc(")
        .unwrap_or_else(|| panic!("inlined {reduce_kind} emitted no output allocation:\n{src}"));
    assert!(
        guard_offset < allocation_offset,
        "inlined {reduce_kind} must guard before allocation or indexing:\n{src}"
    );
    assert!(
        function_body.contains("if (t1_rank == t2_rank)"),
        "scalar-first ordering must still compare the later tensor pair:\n{src}"
    );
    assert_eq!(
        function_body.matches("chelis_alloc(").count(),
        if reduce_kind == "sum" { 2 } else { 1 },
        "fused reduction allocates its result and Sum's checked tree scratch:\n{src}"
    );
    assert!(!function_body.contains(&format!("chelis_tensor *t{} =", fused_node.id.0)));

    let expected = if reduce_kind == "sum" {
        "29.0f, 110.0f"
    } else {
        "16.0f, 49.0f"
    };
    let harness_support = r#"
static chelis_tensor *make_scalar_view(float *data) {
    return chelis_tensor_entry_borrow(
        0, NULL, CHELIS_DTYPE_F32, data, (int64_t)sizeof(float)
    );
}
static chelis_tensor *make_matrix_view(
    float *data, int64_t *shape, int64_t *strides, int64_t backing_elements
) {
    (void)strides;
    return chelis_tensor_entry_borrow(
        2, shape, CHELIS_DTYPE_F32, data,
        backing_elements * (int64_t)sizeof(float)
    );
}
"#;
    let positive_harness = format!(
        r#"{HARNESS_HEADER}{harness_support}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float scalar_data[1] = {{1.0f}};
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[6] = {{2, 3, 4, 5, 6, 7}};
    int64_t a_shape[2] = {{2, 3}}, a_strides[2] = {{3, 1}};
    int64_t b_shape[2] = {{2, 3}}, b_strides[2] = {{3, 1}};
    chelis_tensor *scalar = make_scalar_view(scalar_data);
    chelis_tensor *a = make_matrix_view(a_data, a_shape, a_strides, 6);
    chelis_tensor *b = make_matrix_view(b_data, b_shape, b_strides, 6);
    chelis_tensor *inputs[3] = {{scalar, a, b}};
    chelis_tensor *outputs[1] = {{NULL}};
    {function}(inputs, 3, outputs, 1);
    float expected[2] = {{{expected}}};
    float *got = (float *)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < 2; i++) if (fabsf(got[i] - expected[i]) > 1e-6f) return 1;
    puts("PASS");
    return 0;
}}
"#
    );
    let output = compile_and_run_kernel(
        &format!("direct_fused_{reduce_kind}_shape_positive"),
        &src,
        &positive_harness,
    )
    .unwrap_or_else(|| panic!("inlined {reduce_kind} positive case failed"));
    assert!(output.contains("PASS"), "inlined {reduce_kind}: {output}");

    let mismatch_harness = format!(
        r#"{HARNESS_HEADER}{harness_support}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float scalar_data[1] = {{1.0f}};
    float a_data[6] = {{1, 2, 3, 4, 5, 6}};
    float b_data[4] = {{2, 3, 5, 6}};
    int64_t a_shape[2] = {{2, 3}}, a_strides[2] = {{3, 1}};
    int64_t b_shape[2] = {{2, 2}}, b_strides[2] = {{2, 1}};
    chelis_tensor *scalar = make_scalar_view(scalar_data);
    chelis_tensor *a = make_matrix_view(a_data, a_shape, a_strides, 6);
    chelis_tensor *b = make_matrix_view(b_data, b_shape, b_strides, 4);
    chelis_tensor *inputs[3] = {{scalar, a, b}};
    chelis_tensor *outputs[1] = {{NULL}};
    {function}(inputs, 3, outputs, 1);
    puts("UNREACHABLE");
    return 0;
}}
"#
    );
    let run = compile_and_capture_run(
        &format!("direct_fused_{reduce_kind}_shape_mismatch"),
        &src,
        &mismatch_harness,
    );
    assert!(
        !run.status.success(),
        "inlined {reduce_kind} mismatch reached allocation/indexing"
    );
    assert!(
        String::from_utf8_lossy(&run.stderr).contains("elementwise operand shape mismatch"),
        "inlined {reduce_kind} mismatch emitted the wrong diagnostic: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn direct_fused_sum_runtime_shape_guard_precedes_allocation_and_executes() {
    direct_fused_reduction_runtime_shape_guard_case("sum");
}

#[test]
fn direct_fused_max_reduce_runtime_shape_guard_precedes_allocation_and_executes() {
    direct_fused_reduction_runtime_shape_guard_case("max");
}

#[derive(Clone, Copy)]
struct DirectExtremaBitCase<'a> {
    tag: &'a str,
    prim: Prim,
    c_dtype: &'a str,
    bits_type: &'a str,
    lhs_bits: &'a [u64],
    rhs_bits: &'a [u64],
}

fn direct_extrema_bit_case(
    case: DirectExtremaBitCase<'_>,
    expected_max: &[u64],
    expected_min: &[u64],
) {
    let DirectExtremaBitCase {
        tag,
        prim,
        c_dtype,
        bits_type,
        lhs_bits,
        rhs_bits,
    } = case;
    let n = lhs_bits.len();
    assert_eq!(rhs_bits.len(), n);
    let format_bits = |bits: &[u64]| {
        bits.iter()
            .map(|value| match bits_type {
                "uint64_t" => format!("UINT64_C(0x{value:016x})"),
                "uint32_t" => format!("UINT32_C(0x{value:08x})"),
                _ => format!("UINT16_C(0x{value:04x})"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    for (op_name, op, expected) in [
        ("max", RiscOp::MaxElem, expected_max),
        ("min", RiscOp::MinElem, expected_min),
    ] {
        let mut dag = Dag::new();
        let ty = vec_prim(n, prim);
        let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
        let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
        dag.add_node(op, vec![a, b], ty, None);
        let function = format!("direct_{op_name}_{tag}");
        let src = codegen(&dag, &function).unwrap().c_source;
        assert!(!src.contains("fmaxf("), "{tag}/{op_name}: {src}");
        assert!(!src.contains("fminf("), "{tag}/{op_name}: {src}");

        let (value_type, setup, got) = match prim {
            Prim::F64 => (
                "double",
                "double a_data[N]; double b_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits));",
                "uint64_t got; memcpy(&got, &((double *)chelis_tensor_read_view(outputs[0]).data)[i], sizeof(got));",
            ),
            Prim::F32 => (
                "float",
                "float a_data[N]; float b_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits));",
                "uint32_t got; memcpy(&got, &((float *)chelis_tensor_read_view(outputs[0]).data)[i], sizeof(got));",
            ),
            Prim::F16 | Prim::Bf16 => (
                "uint16_t",
                "uint16_t *a_data = a_bits; uint16_t *b_data = b_bits;",
                "uint16_t got = ((uint16_t *)chelis_tensor_read_view(outputs[0]).data)[i];",
            ),
            _ => unreachable!(),
        };
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
#define N {n}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {bits_type} a_bits[N] = {{ {lhs} }};
    {bits_type} b_bits[N] = {{ {rhs} }};
    {bits_type} expected[N] = {{ {expected} }};
    {setup}
    (void)sizeof({value_type});
    chelis_tensor *a = make_view_typed_1d(a_data, N, {c_dtype});
    chelis_tensor *b = make_view_typed_1d(b_data, N, {c_dtype});
    chelis_tensor *inputs[2] = {{ a, b }}; chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    for (int i = 0; i < N; i++) {{ {got} if (got != expected[i]) return 1; }}
    puts("PASS"); return 0;
}}
"#,
            lhs = format_bits(lhs_bits),
            rhs = format_bits(rhs_bits),
            expected = format_bits(expected),
        );
        let output = compile_and_run_kernel(&function, &src, &harness)
            .unwrap_or_else(|| panic!("{tag}/{op_name} did not compile and run"));
        assert!(output.contains("PASS"), "{tag}/{op_name}: {output}");
    }
}

#[test]
fn direct_extrema_preserve_nan_payloads_and_lhs_signed_zero_at_every_float_width() {
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "f64",
            prim: Prim::F64,
            c_dtype: "CHELIS_DTYPE_F64",
            bits_type: "uint64_t",
            lhs_bits: &[
                0x7ff8_1111_2222_3333,
                0x3ff0_0000_0000_0000,
                0,
                0x8000_0000_0000_0000,
            ],
            rhs_bits: &[
                0x4000_0000_0000_0000,
                0xfff8_4444_5555_6666,
                0x8000_0000_0000_0000,
                0,
            ],
        },
        &[
            0x7ff8_1111_2222_3333,
            0xfff8_4444_5555_6666,
            0,
            0x8000_0000_0000_0000,
        ],
        &[
            0x7ff8_1111_2222_3333,
            0xfff8_4444_5555_6666,
            0,
            0x8000_0000_0000_0000,
        ],
    );
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "f32",
            prim: Prim::F32,
            c_dtype: "CHELIS_DTYPE_F32",
            bits_type: "uint32_t",
            lhs_bits: &[0x7fc1_2345, 0x3f80_0000, 0, 0x8000_0000],
            rhs_bits: &[0x4000_0000, 0xffc5_4321, 0x8000_0000, 0],
        },
        &[0x7fc1_2345, 0xffc5_4321, 0, 0x8000_0000],
        &[0x7fc1_2345, 0xffc5_4321, 0, 0x8000_0000],
    );
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "f16",
            prim: Prim::F16,
            c_dtype: "CHELIS_DTYPE_F16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7e11, 0x3c00, 0, 0x8000],
            rhs_bits: &[0x4000, 0xfe22, 0x8000, 0],
        },
        &[0x7e11, 0xfe22, 0, 0x8000],
        &[0x7e11, 0xfe22, 0, 0x8000],
    );
    direct_extrema_bit_case(
        DirectExtremaBitCase {
            tag: "bf16",
            prim: Prim::Bf16,
            c_dtype: "CHELIS_DTYPE_BF16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7fc1, 0x3f80, 0, 0x8000],
            rhs_bits: &[0x4000, 0xffc2, 0x8000, 0],
        },
        &[0x7fc1, 0xffc2, 0, 0x8000],
        &[0x7fc1, 0xffc2, 0, 0x8000],
    );
}

fn direct_extrema_adjoint_bit_case(
    case: DirectExtremaBitCase<'_>,
    gradient_bits: &[u64],
    expected: [&[u64]; 4],
) {
    let DirectExtremaBitCase {
        tag,
        prim,
        c_dtype,
        bits_type,
        lhs_bits,
        rhs_bits,
    } = case;
    let n = lhs_bits.len();
    assert_eq!(rhs_bits.len(), n);
    assert_eq!(gradient_bits.len(), n);
    assert!(expected.iter().all(|values| values.len() == n));
    let format_bits = |bits: &[u64]| {
        bits.iter()
            .map(|value| match bits_type {
                "uint64_t" => format!("UINT64_C(0x{value:016x})"),
                "uint32_t" => format!("UINT32_C(0x{value:08x})"),
                _ => format!("UINT16_C(0x{value:04x})"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };

    let mut dag = Dag::new();
    let ty = vec_prim(n, prim);
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], ty.clone(), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], ty.clone(), None);
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], ty.clone(), None);
    for (kind, operand) in [
        (ExtremaKind::Max, ExtremaOperand::Left),
        (ExtremaKind::Max, ExtremaOperand::Right),
        (ExtremaKind::Min, ExtremaOperand::Left),
        (ExtremaKind::Min, ExtremaOperand::Right),
    ] {
        let node = dag.add_node(
            RiscOp::ExtremaAdjoint { kind, operand },
            vec![a, b, g],
            ty.clone(),
            None,
        );
        dag.add_root(node);
    }
    let function = format!("direct_extrema_adjoint_{tag}");
    let src = codegen(&dag, &function).unwrap().c_source;
    assert!(
        src.contains("UINT16_C(0)") || src.contains("0.0"),
        "{tag}: {src}"
    );

    let setup = match prim {
        Prim::F64 => {
            "double a_data[N]; double b_data[N]; double g_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits)); memcpy(g_data, g_bits, sizeof(g_bits));"
        }
        Prim::F32 => {
            "float a_data[N]; float b_data[N]; float g_data[N]; memcpy(a_data, a_bits, sizeof(a_bits)); memcpy(b_data, b_bits, sizeof(b_bits)); memcpy(g_data, g_bits, sizeof(g_bits));"
        }
        Prim::F16 | Prim::Bf16 => {
            "uint16_t *a_data = a_bits; uint16_t *b_data = b_bits; uint16_t *g_data = g_bits;"
        }
        _ => unreachable!(),
    };
    let read_got = match prim {
        Prim::F64 => {
            "uint64_t got; memcpy(&got, &((double *)chelis_tensor_read_view(outputs[out]).data)[i], sizeof(got));"
        }
        Prim::F32 => {
            "uint32_t got; memcpy(&got, &((float *)chelis_tensor_read_view(outputs[out]).data)[i], sizeof(got));"
        }
        Prim::F16 | Prim::Bf16 => {
            "uint16_t got = ((uint16_t *)chelis_tensor_read_view(outputs[out]).data)[i];"
        }
        _ => unreachable!(),
    };
    let expected_rows = expected
        .iter()
        .map(|values| format!("{{ {} }}", format_bits(values)))
        .collect::<Vec<_>>()
        .join(", ");
    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
#define N {n}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {bits_type} a_bits[N] = {{ {lhs} }};
    {bits_type} b_bits[N] = {{ {rhs} }};
    {bits_type} g_bits[N] = {{ {gradient} }};
    {bits_type} expected[4][N] = {{ {expected_rows} }};
    {setup}
    chelis_tensor *a = make_view_typed_1d(a_data, N, {c_dtype});
    chelis_tensor *b = make_view_typed_1d(b_data, N, {c_dtype});
    chelis_tensor *g = make_view_typed_1d(g_data, N, {c_dtype});
    chelis_tensor *inputs[3] = {{ a, b, g }};
    chelis_tensor *outputs[4] = {{ NULL, NULL, NULL, NULL }};
    {function}(inputs, 3, outputs, 4);
    for (int out = 0; out < 4; out++) {{
        for (int i = 0; i < N; i++) {{ {read_got} if (got != expected[out][i]) return 1; }}
    }}
    puts("PASS"); return 0;
}}
"#,
        lhs = format_bits(lhs_bits),
        rhs = format_bits(rhs_bits),
        gradient = format_bits(gradient_bits),
    );
    let output = compile_and_run_kernel(&function, &src, &harness)
        .unwrap_or_else(|| panic!("{tag} direct extrema adjoints did not compile and run"));
    assert!(output.contains("PASS"), "{tag}: {output}");
}

#[test]
fn direct_extrema_adjoints_copy_exact_gradient_bits_for_ties_and_nan_selection() {
    let run = |case: DirectExtremaBitCase<'_>, gradient: &[u64]| {
        let zero = 0;
        let max_left = [
            gradient[0],
            zero,
            gradient[2],
            gradient[3],
            gradient[4],
            zero,
        ];
        let max_right = [zero, gradient[1], zero, zero, zero, gradient[5]];
        let min_left = [
            gradient[0],
            zero,
            gradient[2],
            gradient[3],
            zero,
            gradient[5],
        ];
        let min_right = [zero, gradient[1], zero, zero, gradient[4], zero];
        direct_extrema_adjoint_bit_case(
            case,
            gradient,
            [&max_left, &max_right, &min_left, &min_right],
        );
    };

    run(
        DirectExtremaBitCase {
            tag: "f64",
            prim: Prim::F64,
            c_dtype: "CHELIS_DTYPE_F64",
            bits_type: "uint64_t",
            lhs_bits: &[
                0x7ff8_1111_2222_3333,
                0x3ff0_0000_0000_0000,
                0,
                0x8000_0000_0000_0000,
                0x4000_0000_0000_0000,
                0x3ff0_0000_0000_0000,
            ],
            rhs_bits: &[
                0x4000_0000_0000_0000,
                0xfff8_4444_5555_6666,
                0x8000_0000_0000_0000,
                0,
                0x3ff0_0000_0000_0000,
                0x4000_0000_0000_0000,
            ],
        },
        &[
            0x7ff8_abcd_1234_5678,
            0xbff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x4008_0000_0000_0000,
            0xc010_0000_0000_0000,
        ],
    );
    run(
        DirectExtremaBitCase {
            tag: "f32",
            prim: Prim::F32,
            c_dtype: "CHELIS_DTYPE_F32",
            bits_type: "uint32_t",
            lhs_bits: &[
                0x7fc1_2345,
                0x3f80_0000,
                0,
                0x8000_0000,
                0x4000_0000,
                0x3f80_0000,
            ],
            rhs_bits: &[
                0x4000_0000,
                0xffc5_4321,
                0x8000_0000,
                0,
                0x3f80_0000,
                0x4000_0000,
            ],
        },
        &[
            0x7fc6_789a,
            0xbf80_0000,
            0x3f80_0000,
            0x8000_0000,
            0x4040_0000,
            0xc080_0000,
        ],
    );
    run(
        DirectExtremaBitCase {
            tag: "f16",
            prim: Prim::F16,
            c_dtype: "CHELIS_DTYPE_F16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7e11, 0x3c00, 0, 0x8000, 0x4000, 0x3c00],
            rhs_bits: &[0x4000, 0xfe22, 0x8000, 0, 0x3c00, 0x4000],
        },
        &[0x7e33, 0xbc00, 0x3c00, 0x8000, 0x4200, 0xc400],
    );
    run(
        DirectExtremaBitCase {
            tag: "bf16",
            prim: Prim::Bf16,
            c_dtype: "CHELIS_DTYPE_BF16",
            bits_type: "uint16_t",
            lhs_bits: &[0x7fc1, 0x3f80, 0, 0x8000, 0x4000, 0x3f80],
            rhs_bits: &[0x4000, 0xffc2, 0x8000, 0, 0x3f80, 0x4000],
        },
        &[0x7fc3, 0xbf80, 0x3f80, 0x8000, 0x4040, 0xc080],
    );
}

fn host_scalar_relu_program(ty: HostType) -> HostProgram {
    let body = HostExpr::new(HostExprKind::Builtin {
        name: "relu".to_string(),
        args: vec![HostExpr::new(HostExprKind::Var(
            "x".to_string(),
            ty.clone(),
        ))],
        ty: ty.clone(),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: ty.clone(),
            }],
            ret_ty: ty,
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
    }
}

fn host_scalar_relu_reduced_bits_case(tag: &str, ty: HostType, inputs: &[u16], expected: &[u16]) {
    assert_eq!(inputs.len(), expected.len());
    let function = format!("host_scalar_relu_{tag}_bits");
    let src = emit_host_program(&host_scalar_relu_program(ty), &function)
        .expect("ownership-verified host ReLU codegen");
    let format_bits = |bits: &[u16]| {
        bits.iter()
            .map(|value| format!("UINT16_C(0x{value:04x})"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
#define N {n}
extern uint16_t the_fn(uint16_t);
int main(void) {{
    uint16_t inputs[N] = {{ {inputs} }};
    uint16_t expected[N] = {{ {expected} }};
    for (int i = 0; i < N; ++i) {{
        uint16_t got = the_fn(inputs[i]);
        if (got != expected[i]) {{
            fprintf(stderr, "{tag} ReLU bit mismatch at %d: got 0x%04x expected 0x%04x\n",
                    i, (unsigned)got, (unsigned)expected[i]);
            return 1;
        }}
    }}
    return 0;
}}
"#,
        n = inputs.len(),
        inputs = format_bits(inputs),
        expected = format_bits(expected),
    );
    let run = compile_and_capture_run(&function, &src, &harness);
    assert!(
        run.status.success(),
        "{tag} host scalar ReLU changed selected stored bits: {}",
        String::from_utf8_lossy(&run.stderr)
    );
}

/// [05-OP-43] / chelis#1313: the reduced-float HostProgram ABI carries f16
/// as stored `uint16_t` bits. ReLU selects that exact carrier for nonnegative
/// values and every NaN; only a strictly negative numeric input becomes +0.
#[test]
fn host_scalar_relu_f16_preserves_selected_stored_bits() {
    host_scalar_relu_reduced_bits_case(
        "f16",
        HostType::Scalar(Prim::F16),
        &[
            0xbc00, 0x8000, 0x0000, 0x3c00, 0x7e11, 0xfe11, 0x7c01, 0xfc01,
        ],
        &[
            0x0000, 0x8000, 0x0000, 0x3c00, 0x7e11, 0xfe11, 0x7c01, 0xfc01,
        ],
    );
}

/// [05-OP-43] / chelis#1313: bf16 has the same exact selected-stored-value
/// rule, including payload/sign preservation for quiet and signaling NaNs.
#[test]
fn host_scalar_relu_bf16_preserves_selected_stored_bits() {
    host_scalar_relu_reduced_bits_case(
        "bf16",
        HostType::Scalar(Prim::Bf16),
        &[
            0xbf80, 0x8000, 0x0000, 0x3f80, 0x7fc1, 0xffc1, 0x7f91, 0xff91,
        ],
        &[
            0x0000, 0x8000, 0x0000, 0x3f80, 0x7fc1, 0xffc1, 0x7f91, 0xff91,
        ],
    );
}

fn direct_relu_bit_case(case: DirectExtremaBitCase<'_>, expected: [&[u64]; 2]) {
    let DirectExtremaBitCase {
        tag,
        prim,
        c_dtype,
        bits_type,
        lhs_bits: x_bits,
        rhs_bits: g_bits,
    } = case;
    let n = x_bits.len();
    assert_eq!(g_bits.len(), n);
    assert!(expected.iter().all(|values| values.len() == n));
    let format_bits = |bits: &[u64]| {
        bits.iter()
            .map(|value| match bits_type {
                "uint64_t" => format!("UINT64_C(0x{value:016x})"),
                "uint32_t" => format!("UINT32_C(0x{value:08x})"),
                _ => format!("UINT16_C(0x{value:04x})"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };

    let mut dag = Dag::new();
    let ty = vec_prim(n, prim);
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let g = dag.add_node(RiscOp::Load { name: "g".into() }, vec![], ty.clone(), None);
    let relu = dag.add_node(RiscOp::Relu, vec![x], ty.clone(), None);
    let adjoint = dag.add_node(RiscOp::ReluAdjoint, vec![x, g], ty, None);
    dag.add_root(relu);
    dag.add_root(adjoint);
    let function = format!("direct_relu_{tag}");
    let src = codegen(&dag, &function).unwrap().c_source;
    assert!(!src.contains("fmax"), "{tag}: {src}");

    let setup = match prim {
        Prim::F64 => {
            "double x_data[N]; double g_data[N]; memcpy(x_data, x_bits, sizeof(x_bits)); memcpy(g_data, g_bits, sizeof(g_bits));"
        }
        Prim::F32 => {
            "float x_data[N]; float g_data[N]; memcpy(x_data, x_bits, sizeof(x_bits)); memcpy(g_data, g_bits, sizeof(g_bits));"
        }
        Prim::F16 | Prim::Bf16 => "uint16_t *x_data = x_bits; uint16_t *g_data = g_bits;",
        _ => unreachable!(),
    };
    let read_got = match prim {
        Prim::F64 => {
            "uint64_t got; memcpy(&got, &((double *)chelis_tensor_read_view(outputs[out]).data)[i], sizeof(got));"
        }
        Prim::F32 => {
            "uint32_t got; memcpy(&got, &((float *)chelis_tensor_read_view(outputs[out]).data)[i], sizeof(got));"
        }
        Prim::F16 | Prim::Bf16 => {
            "uint16_t got = ((uint16_t *)chelis_tensor_read_view(outputs[out]).data)[i];"
        }
        _ => unreachable!(),
    };
    let expected_rows = expected
        .iter()
        .map(|values| format!("{{ {} }}", format_bits(values)))
        .collect::<Vec<_>>()
        .join(", ");
    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
#define N {n}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {bits_type} x_bits[N] = {{ {x} }};
    {bits_type} g_bits[N] = {{ {g} }};
    {bits_type} expected[2][N] = {{ {expected_rows} }};
    {setup}
    chelis_tensor *x = make_view_typed_1d(x_data, N, {c_dtype});
    chelis_tensor *g = make_view_typed_1d(g_data, N, {c_dtype});
    chelis_tensor *inputs[2] = {{ x, g }};
    chelis_tensor *outputs[2] = {{ NULL, NULL }};
    {function}(inputs, 2, outputs, 2);
    for (int out = 0; out < 2; out++) {{
        for (int i = 0; i < N; i++) {{ {read_got} if (got != expected[out][i]) return 1; }}
    }}
    puts("PASS"); return 0;
}}
"#,
        x = format_bits(x_bits),
        g = format_bits(g_bits),
    );
    let output = compile_and_run_kernel(&function, &src, &harness)
        .unwrap_or_else(|| panic!("{tag} direct ReLU and adjoint did not compile and run"));
    assert!(output.contains("PASS"), "{tag}: {output}");
}

#[test]
fn direct_relu_preserves_input_bits_and_adjoint_uses_strict_positive_mask() {
    let run = |case: DirectExtremaBitCase<'_>| {
        let x = case.lhs_bits;
        let g = case.rhs_bits;
        let forward = [x[0], x[1], 0, 0, x[4], x[5]];
        let adjoint = [0, 0, 0, 0, g[4], g[5]];
        direct_relu_bit_case(case, [&forward, &adjoint]);
    };

    run(DirectExtremaBitCase {
        tag: "f64",
        prim: Prim::F64,
        c_dtype: "CHELIS_DTYPE_F64",
        bits_type: "uint64_t",
        lhs_bits: &[
            0x7ff8_1111_2222_3333,
            0x8000_0000_0000_0000,
            0,
            0xbff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            1,
        ],
        rhs_bits: &[
            0x7ff8_abcd_1234_5678,
            0x7ff0_0000_0000_0000,
            0x8000_0000_0000_0000,
            0xfff8_abcd_1234_5678,
            0xbff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
        ],
    });
    run(DirectExtremaBitCase {
        tag: "f32",
        prim: Prim::F32,
        c_dtype: "CHELIS_DTYPE_F32",
        bits_type: "uint32_t",
        lhs_bits: &[0x7fc1_2345, 0x8000_0000, 0, 0xbf80_0000, 0x3f80_0000, 1],
        rhs_bits: &[
            0x7fc6_789a,
            0x7f80_0000,
            0x8000_0000,
            0xffc6_789a,
            0xbf80_0000,
            0x7f80_0000,
        ],
    });
    run(DirectExtremaBitCase {
        tag: "f16",
        prim: Prim::F16,
        c_dtype: "CHELIS_DTYPE_F16",
        bits_type: "uint16_t",
        lhs_bits: &[0x7e11, 0x8000, 0, 0xbc00, 0x3c00, 1],
        rhs_bits: &[0x7e33, 0x7c00, 0x8000, 0xfe33, 0xbc00, 0x7c00],
    });
    run(DirectExtremaBitCase {
        tag: "bf16",
        prim: Prim::Bf16,
        c_dtype: "CHELIS_DTYPE_BF16",
        bits_type: "uint16_t",
        lhs_bits: &[0x7fc1, 0x8000, 0, 0xbf80, 0x3f80, 1],
        rhs_bits: &[0x7fc3, 0x7f80, 0x8000, 0xffc3, 0xbf80, 0x7f80],
    });
}

// ---- chelis#1277 Slice B: runtime extent guards, driven with runtime inputs ----
//
// A CLI-rooted program cannot observe these guards on C. When `main` calls the
// def on literal shapes, every claim in the inlined kernel is provable from
// literals, so a violation is a check-time type error and no runtime guard is
// reachable. The guard is observable only when the EXPORTED kernel is driven
// with runtime inputs, which is what this harness does. That mirrors the
// eval-lane finding: the lane that can observe the guard is not the lane a
// `.ch` fixture reaches.
//
// A note that is now history rather than a caveat: before #1509,
// `make_view_typed_1d` kept ONE `static` shape array, so a second view
// overwrote the first and both tensors reported the same extent - which would
// have made a mismatch row pass while comparing a value with itself. #1509's
// opaque-tensor cut replaced the struct literal with
// `chelis_tensor_entry_borrow` over a LOCAL shape array, so each view now
// carries its own extent and the shared helper is safe for this row.

/// Two `Load`s declaring one symbolic dim, BOTH read for data. The class is
/// all-interface, so `spec/04-type-system.md` section 4.7 places its guard at
/// entry and names the `load` primitive.
///
/// An earlier version left `p` unread, which is section 4.7's "regardless of
/// data use" case and the stronger row. It is not testable here: the
/// derivation groups `Name` claims among root-reachable witnesses only, so a
/// witness no result reaches forms no class and gets no guard. That gap is a
/// recorded residual owned by B2b, and this row is deliberately the weaker
/// one it can still prove rather than a rewritten row reporting a lock it no
/// longer holds. The entry guard runs in the prologue, before any operation,
/// so consuming `p` in the body does not let the elementwise operand check
/// preempt it - which the disagreeing case below measures rather than
/// assumes.
fn two_witness_dag() -> chelis_ir::dag::Dag {
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    let named = || TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], named(), None);
    let p = dag.add_node(RiscOp::Load { name: "p".into() }, vec![], named(), None);
    let negated = dag.add_node(RiscOp::Neg, vec![x], named(), None);
    let out = dag.add_node(RiscOp::Add, vec![negated, p], named(), None);
    dag.add_root(out);
    dag
}

#[test]
fn an_all_interface_class_traps_at_entry_when_its_witnesses_disagree() {
    let dag = two_witness_dag();
    let result = codegen_with_options(
        &dag,
        "guard_entry",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void guard_entry(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[2] = {{1.0f, 2.0f}};
    float pd[3] = {{1.0f, 2.0f, 3.0f}};
    chelis_tensor* inputs[2] = {{make_view_1d(xd, 2), make_view_1d(pd, 3)}};
    chelis_tensor* outputs[1] = {{NULL}};
    guard_entry(inputs, 2, outputs, 1);
    printf("NO TRAP\n");
    return 0;
}}
"#
    );

    let (ok, out) = compile_and_run_kernel_capturing("guard_entry", &result.c_source, &harness);
    assert!(!ok, "witnesses of `n` disagree at 2 and 3: {out}");
    assert!(
        !out.contains("NO TRAP"),
        "the guard runs at ENTRY, before any other operation: {out}"
    );
    assert!(
        out.contains("numeric trap: domain in load at int64"),
        "section 4.7 makes this an [04-NUM-9] guard naming the `load`: {out}"
    );
    // NOT `out.contains('n')`. That was satisfied by "numeric", "domain in"
    // and "int64" in the trap line itself, so it asserted nothing about the
    // context line it was meant to check: round 1 replaced the whole context
    // `fprintf` with a literal and both driven rows still passed. Section 4.7
    // requires the disagreeing SOURCE NAMES, the AXIS and each OBSERVED
    // VALUE, so assert that content, which cannot survive the line being
    // replaced.
    assert!(
        out.contains("extent `n`"),
        "the context line names the disagreeing claim: {out}"
    );
    assert!(
        out.contains("x axis 0 = 2"),
        "and the canonical witness with its observed extent: {out}"
    );
    assert!(
        out.contains("p axis 0 = 3"),
        "and the disagreeing witness with its observed extent: {out}"
    );
}

/// The positive twin: agreeing witnesses run to completion. Without it, a
/// guard that trapped on every class would satisfy the row above.
#[test]
fn an_all_interface_class_runs_when_its_witnesses_agree() {
    let dag = two_witness_dag();
    let result = codegen_with_options(
        &dag,
        "guard_entry_ok",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void guard_entry_ok(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[2] = {{1.0f, 2.0f}};
    float pd[2] = {{3.0f, 4.0f}};
    chelis_tensor* inputs[2] = {{make_view_1d(xd, 2), make_view_1d(pd, 2)}};
    chelis_tensor* outputs[1] = {{NULL}};
    guard_entry_ok(inputs, 2, outputs, 1);
    printf("RAN %.1f\n", ((float*)chelis_tensor_read_view(outputs[0]).data)[0]);
    return 0;
}}
"#
    );

    let (ok, out) = compile_and_run_kernel_capturing("guard_entry_ok", &result.c_source, &harness);
    assert!(ok, "agreeing witnesses must execute: {out}");
    assert!(out.contains("RAN 2.0"), "and produce neg(x) + p: {out}");
}

// ---- chelis#1277 b2.4: one guard per axis, and the ABI check narrowed ----
//
// Two spellings of one axis reach the prologue: the binding view sees a
// `Load`'s own axis as an `ExternalAxis` member, and the class view sees a
// folded `shape(t, k)` read of that same tensor as an `InputAxis` member.
// Before the dedupe both were emitted, so an axis could be guarded twice or,
// where the claim's other witness is dropped as unused, compared with itself.
// A tautology and a duplicate are the two ways a guard can be present in the
// text and absent in effect, which is why both directions are pinned here
// rather than only the count.

/// A same-rank `Expand` whose operand extent is symbolic, so the unit-extent
/// claim of `spec/05-risc-primitives.md` section 2.4.1 is checked at run time
/// rather than statically refuted.
///
/// `x_dim` is the operand's declared axis-0 extent and `size` is the width the
/// broadcast sets. The operand is the ONLY input, so its axis is an interface
/// value and section 4.7 places the guard at function entry, in declared
/// signature order, before any other operation of the function runs.
fn same_rank_expand_over_symbolic_operand_dag(x_dim: DimInfo, size: usize) -> chelis_ir::dag::Dag {
    use chelis_ir::dag::{Dag, RiscOp, RtDim, TensorType};
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![x_dim],
            precision: Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(size),
        },
        vec![x],
        TensorType {
            dims: vec![DimInfo::Lit(size)],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(out);
    dag
}

/// Oracle row `expand.positional.replacement.non_unit_source_traps.c`.
///
/// `spec/05-risc-primitives.md` section 2.4.1: the same-rank form "is a claim
/// that the operand's extent at `axis` is 1 [...] A symbolic or runtime
/// operand extent at `axis` other than 1 fails that claim's runtime extent
/// guard and traps `Domain`, placed and rendered per
/// `spec/04-type-system.md` section 4.7 and [04-NUM-9]."
///
/// The `<op>` slot is `load`, not `expand`. Section 4.7 fixes it: "for a guard
/// whose operands are all interface values, the `load` primitive of the later
/// witness in signature order". The operand's axis IS the only witness and it
/// is an input tensor's axis, so the class is all-interface.
///
/// The `<prim>` slot is `int64` because the guarded result is an extent under
/// [05-DIM-1] rather than a tensor element, which is why it does not track the
/// tensor's own `f32`.
///
/// EVIDENTIARY STATUS: regression test. On the base the same kernel executes
/// with no guard at all and returns a tensor built by reading index 0 of an
/// axis that has more than one element, which is the `silent_unguarded`
/// baseline this row is recorded at.
#[test]
fn a_runtime_non_unit_source_under_a_same_rank_claim_traps_at_entry_on_c() {
    let dag = same_rank_expand_over_symbolic_operand_dag(DimInfo::Named("n".into(), None), 3);
    let result = codegen_with_options(
        &dag,
        "bcast_entry",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");

    let run = |name: &str, extent: usize| {
        let values = (0..extent)
            .map(|i| format!("{}.0f", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let harness = format!(
            r#"{HARNESS_HEADER}
extern void bcast_entry(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[{extent}] = {{{values}}};
    chelis_tensor* inputs[1] = {{make_view_1d(xd, {extent})}};
    chelis_tensor* outputs[1] = {{NULL}};
    bcast_entry(inputs, 1, outputs, 1);
    printf("RAN %lld %.1f\n",
           (long long)chelis_tensor_shape(outputs[0], 0),
           ((const float*)chelis_tensor_read_view(outputs[0]).data)[2]);
    return 0;
}}
"#
        );
        compile_and_run_kernel_capturing(name, &result.c_source, &harness)
    };

    let (ok, out) = run("bcast_entry_trap", 2);
    assert!(!ok, "an operand extent of 2 refutes the unit claim: {out}");
    assert!(
        out.contains("numeric trap: domain in load at int64"),
        "the trap renders [04-NUM-9] at the extent's dtype, and `<op>` is \
         `load` because the operand's axis is an interface value: {out}"
    );
    assert!(
        out.contains("claimed = 1") && out.contains("x axis 0 = 2"),
        "with the claim, the axis and the value observed for it: {out}"
    );
    assert!(
        !out.contains("RAN "),
        "the guard runs before the broadcast allocates, so nothing prints: {out}"
    );

    let (ok, out) = run("bcast_entry_ok", 1);
    assert!(ok, "a unit operand extent satisfies the claim: {out}");
    assert!(
        out.contains("RAN 3 1.0"),
        "and broadcasts the single element across the declared width: {out}"
    );
}

fn expand_reading_own_axis_dag(x_dim: DimInfo, out_dim: DimInfo) -> chelis_ir::dag::Dag {
    use chelis_ir::dag::{Dag, RiscOp, RtAxis, RtDim, TensorType};
    let mut dag = Dag::new();
    let scalar = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], scalar, None);
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![x_dim],
            precision: Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, x],
        TensorType {
            dims: vec![out_dim],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(out);
    dag
}

/// The claim's only other witness is the axis the canonical value was read
/// from, so the comparison is `n != n`. chelis#1374's emitted kernel carried
/// exactly this: `int64_t n = chelis_tensor_shape(inputs[1], 0);` followed by
/// `if (chelis_tensor_shape(inputs[1], 0) != n)`.
///
/// EVIDENTIARY STATUS: regression test. Measured failing on the emitted C
/// before the dedupe.
#[test]
fn a_member_reading_the_canonical_axis_emits_no_guard() {
    let dag = expand_reading_own_axis_dag(
        DimInfo::Named("n".into(), None),
        DimInfo::Named("n".into(), None),
    );
    let src = codegen(&dag, "selfcmp").expect("codegen").c_source;
    assert!(
        src.contains("int64_t n = chelis_tensor_shape(inputs[1], 0);"),
        "the canonical value is still declared: {src}"
    );
    assert!(
        !src.contains("if (chelis_tensor_shape(inputs[1], 0) != n)"),
        "and nothing compares it with itself: {src}"
    );
}

/// A `Literal` claim over an input axis whose static size the checker
/// resolved to the same literal. The legacy ABI static-dim check and the
/// class guard are then the identical comparison, and the ABI one runs first
/// and `abort()`s, so `spec/04-type-system.md` section 4.7's [04-NUM-9] trap
/// is unreachable. chelis#1377's shape.
///
/// EVIDENTIARY STATUS: regression test. `main` emits both lines in this
/// order; measured before the narrowing.
#[test]
fn a_literal_class_guard_replaces_the_abi_static_dim_check_on_that_axis() {
    let dag = expand_reading_own_axis_dag(DimInfo::Named("n".into(), Some(4)), DimInfo::Lit(4));
    let src = codegen(&dag, "litclaim").expect("codegen").c_source;
    assert!(
        !src.contains("input `x` axis 0 expected 4"),
        "the ABI check no longer preempts the extent guard: {src}"
    );
    assert!(
        src.contains("if (chelis_tensor_shape(inputs[1], 0) != 4)"),
        "the class guards the same axis against the same literal: {src}"
    );
    assert!(
        src.contains("chelis_numeric_trap(\"numeric trap: domain in load at int64\")"),
        "and renders [04-NUM-9]: {src}"
    );
}

/// The narrowing's exact complement, in two directions. Without these the
/// change above could have deleted the ABI check outright and still passed:
/// an axis no class guards keeps it, and a `Name` claim - whose guard
/// compares against another input's RUNTIME extent rather than against the
/// declared size - keeps it too, because dropping it there would lose a
/// comparison rather than rename one.
///
/// EVIDENTIARY STATUS: disposition lock, and measured as one. Reverting the
/// narrowing leaves this row green, because `main` already emits both lines;
/// what it pins is that the narrowing did not widen into a deletion. Its
/// three siblings above are regression tests, measured red on the same
/// revert.
#[test]
fn the_abi_static_dim_check_survives_where_no_literal_class_guards_the_axis() {
    use chelis_ir::dag::{Dag, RiscOp, TensorType};
    let mut dag = Dag::new();
    let ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], ty.clone(), None);
    let out = dag.add_node(RiscOp::Neg, vec![x], ty, None);
    dag.add_root(out);
    let src = codegen(&dag, "noclass").expect("codegen").c_source;
    assert!(
        src.contains("input `x` axis 0 expected 4"),
        "a program with no runtime extent keeps the C ABI boundary check: {src}"
    );

    // A `Name` claim over a statically sized axis: the class guards
    // `inputs[1]` axis 0 against `n`, which is read from another input, so
    // the static comparison is not the same comparison and stays.
    let named = expand_reading_own_axis_dag(
        DimInfo::Named("n".into(), Some(4)),
        DimInfo::Named("n".into(), Some(4)),
    );
    let src = codegen(&named, "nameclaim").expect("codegen").c_source;
    assert!(
        src.contains("input `x` axis 0 expected 4"),
        "a Name claim does not license dropping the static check: {src}"
    );
}

/// Three witnesses of one claim, two of which name the same axis through
/// different sources: `y`'s `Load` axis (an `ExternalAxis` member) and the
/// `expand` size's folded read of `y` (an `InputAxis` member). The axis is
/// guarded once. chelis#1374's fixture with its `x` parameter kept alive,
/// which is the form whose emitted kernel carried the duplicate.
///
/// EVIDENTIARY STATUS: regression test, measured at two occurrences before
/// the dedupe.
#[test]
fn one_axis_reached_by_two_member_spellings_is_guarded_once() {
    use chelis_ir::dag::{Dag, RiscOp, RtAxis, RtDim, TensorType};
    let named = || TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], named(), None);
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], named(), None);
    let widened = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, y],
        named(),
        None,
    );
    let out = dag.add_node(RiscOp::Add, vec![widened, x], named(), None);
    dag.add_root(out);

    let src = codegen(&dag, "dedupe").expect("codegen").c_source;
    let slot_of = |label: &str| {
        let marker = format!("input `{label}` at slot ");
        src.find(&marker)
            .and_then(|at| src[at + marker.len()..].chars().next())
            .unwrap_or_else(|| panic!("`{label}` has an assigned slot"))
    };
    // Both operands are read from the class's own witnesses, so the guard
    // names two slots rather than a declared variable.
    let guard = format!(
        "if (chelis_tensor_shape(inputs[{}], 0) != chelis_tensor_shape(inputs[{}], 0))",
        slot_of("y"),
        slot_of("x"),
    );
    assert_eq!(src.matches(&guard).count(), 1, "one axis, one guard: {src}");
}

/// Two all-interface classes, both mismatching, whose claim names sort in the
/// OPPOSITE order from their assigned slots. `spec/04-type-system.md` section
/// 4.7 runs interface guards "at function entry, in declared signature
/// order", and never "by binding name, hash iteration, or node identity";
/// the legacy `symbolic_bindings` grouped in a `BTreeMap<String, _>` and
/// would report `adim` first.
///
/// This row is driven rather than CLI-rooted for the reason the block header
/// above gives, and the CLI form is worse than merely unobservable here:
/// `def main() = f(...)` over literal tensors inlines `f` into the root, so
/// every extent becomes a literal, the classes disappear, and the program
/// runs to completion printing a tensor. That is a static type error nobody
/// raises, which is S2b's rejection to add, not a guard this slice can place.
///
/// EVIDENTIARY STATUS: regression test. On `main` neither class exists.
#[test]
fn entry_guards_run_in_assigned_slot_order_not_claim_name_order() {
    use chelis_ir::dag::{Dag, RiscOp, TensorType};
    let dim = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let zz = dag.add_node(
        RiscOp::Load { name: "zz".into() },
        vec![],
        dim("zdim"),
        None,
    );
    let aa = dag.add_node(
        RiscOp::Load { name: "aa".into() },
        vec![],
        dim("adim"),
        None,
    );
    let p = dag.add_node(RiscOp::Load { name: "p".into() }, vec![], dim("zdim"), None);
    let q = dag.add_node(RiscOp::Load { name: "q".into() }, vec![], dim("adim"), None);
    // All four witnesses are READ, and both classes still have to reach one
    // result, so the `adim` pair is reduced to a scalar and broadcast back
    // over `zdim`. The unread form is section 4.7's stronger case and is not
    // testable here; see `two_witness_dag`.
    let zsum = dag.add_node(RiscOp::Add, vec![zz, p], dim("zdim"), None);
    let asum = dag.add_node(RiscOp::Add, vec![aa, q], dim("adim"), None);
    let folded = dag.add_node(
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![asum],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let spread = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: chelis_ir::dag::RtDim::InputAxis {
                tensor: 1,
                axis: chelis_ir::dag::RtAxis::Lit(0),
            },
        },
        vec![folded, zsum],
        dim("zdim"),
        None,
    );
    let out = dag.add_node(RiscOp::Add, vec![zsum, spread], dim("zdim"), None);
    dag.add_root(out);

    let result = codegen_with_options(
        &dag,
        "entry_order",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void entry_order(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float zd[2] = {{1.0f, 2.0f}};
    float ad[2] = {{1.0f, 2.0f}};
    float pd[3] = {{1.0f, 2.0f, 3.0f}};
    float qd[1] = {{1.0f}};
    chelis_tensor* inputs[4] = {{
        make_view_1d(zd, 2), make_view_1d(ad, 2),
        make_view_1d(pd, 3), make_view_1d(qd, 1)
    }};
    chelis_tensor* outputs[1] = {{NULL}};
    entry_order(inputs, 4, outputs, 1);
    printf("NO TRAP\n");
    return 0;
}}
"#
    );

    let (ok, out) = compile_and_run_kernel_capturing("entry_order", &result.c_source, &harness);
    assert!(!ok, "both classes mismatch: {out}");
    assert!(
        out.contains("extent `zdim`"),
        "`zdim` occupies the earlier slot and is reported first: {out}"
    );
    assert!(
        !out.contains("adim"),
        "`adim` sorts first by name but its guard runs second: {out}"
    );
    assert!(
        out.contains("numeric trap: domain in load at int64"),
        "an all-interface class renders [04-NUM-9]: {out}"
    );
}

/// chelis#1377's C row, driven. A declared `tensor[4, f32]` result over an
/// `expand` sized by a runtime read of an input axis: when that axis is not
/// 4, section 4.7 owes an [04-NUM-9] trap at entry, and until b2.4 the
/// legacy ABI static-dim check aborted first with a rendering the atom does
/// not permit.
///
/// The issue's own reproducer is `def main() = f(...)` over literal tensors,
/// and that half does NOT reach this guard: the root inlines `f`, every
/// extent becomes a literal, and the kernel allocates `{5}` for a
/// `tensor[4]` result with no guard and no rejection. The runtime half is
/// this row; the static half is S2b's rejection, and the two are why the row
/// is recorded `lane_divergent` rather than `silent_unguarded`.
///
/// EVIDENTIARY STATUS: regression test for the rendering and the position -
/// on `main` the same input produces `input `x` axis 0 expected 4, got 5`
/// and `abort()`. The positive twin is a disposition lock.
#[test]
fn a_literal_claim_over_a_runtime_read_traps_at_entry_on_c() {
    let dag = expand_reading_own_axis_dag(DimInfo::Named("n".into(), Some(4)), DimInfo::Lit(4));
    let result = codegen_with_options(
        &dag,
        "lit_entry",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");

    let run = |name: &str, extent: usize| {
        let values = (0..extent)
            .map(|i| format!("{}.0f", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let harness = format!(
            r#"{HARNESS_HEADER}
extern void lit_entry(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float bd[1] = {{7.0f}};
    float xd[{extent}] = {{{values}}};
    int64_t scalar_shape[1] = {{1}};
    chelis_tensor* b = chelis_tensor_entry_borrow(0, scalar_shape, CHELIS_DTYPE_F32, bd, sizeof(float));
    chelis_tensor* inputs[2] = {{b, make_view_1d(xd, {extent})}};
    chelis_tensor* outputs[1] = {{NULL}};
    lit_entry(inputs, 2, outputs, 1);
    printf("RAN %lld\n", (long long)chelis_tensor_shape(outputs[0], 0));
    return 0;
}}
"#
        );
        compile_and_run_kernel_capturing(name, &result.c_source, &harness)
    };

    let (ok, out) = run("lit_entry", 5);
    assert!(!ok, "a declared tensor[4] over a read of 5: {out}");
    assert!(
        out.contains("numeric trap: domain in load at int64"),
        "section 4.7 renders this [04-NUM-9], not the ABI check's abort: {out}"
    );
    assert!(
        out.contains("extent `4`: claimed = 4, x axis 0 = 5"),
        "with the claim, the axis and each observed value: {out}"
    );
    assert!(
        !out.contains("expected 4, got"),
        "and the legacy static-dim check no longer preempts it: {out}"
    );

    let (ok, out) = run("lit_entry_ok", 4);
    assert!(ok, "the agreeing extent must execute: {out}");
    assert!(
        out.contains("RAN 4"),
        "and produce the declared shape: {out}"
    );
}

// ---- chelis#1277 b2.4: the LOCAL half of section 4.7's placement ----

/// A class whose operands are not all interface values takes "the source
/// position of the operation that introduces the guarded extent". Here the
/// claim `n` is witnessed by `x`'s own axis (an interface value) and by the
/// `expand` size's read of a STRIDED tensor, whose extent does not exist
/// until the stride runs. So the class is Local, its guard belongs at the
/// `expand`, and it renders [04-NUM-9] like the entry guards do.
///
/// EVIDENTIARY STATUS: regression test for the RENDERING - `main` emits
/// `chelis: runtime dim `n` mismatch at node 3 axis 0` followed by `abort()`
/// at this site, which [04-NUM-9] does not permit - and a disposition lock
/// for the position, which `main` already gets right through chelis#616's
/// `runtime_dim_sites`.
fn local_class_dag() -> chelis_ir::dag::Dag {
    use chelis_ir::dag::{Dag, RiscOp, RtAxis, RtDim, TensorType};
    let named = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], named("n"), None);
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        named("s"),
        None,
    );
    let out = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, strided],
        named("n"),
        None,
    );
    dag.add_root(out);
    dag
}

#[test]
fn a_local_class_guards_at_its_operation_and_renders_the_numeric_trap() {
    let dag = local_class_dag();
    let result = codegen_with_options(
        &dag,
        "local_guard",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");
    assert!(
        !result.c_source.contains("chelis: runtime dim `n` mismatch"),
        "the legacy rendering is gone: {}",
        result.c_source
    );
    assert!(
        result
            .c_source
            .contains("chelis_numeric_trap(\"numeric trap: domain in insert at int64\")"),
        "and `<op>` names the operation introducing the extent: {}",
        result.c_source
    );

    let run = |name: &str, extent: usize| {
        let values = (0..extent)
            .map(|i| format!("{}.0f", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let harness = format!(
            r#"{HARNESS_HEADER}
extern void local_guard(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[{extent}] = {{{values}}};
    float bd[1] = {{7.0f}};
    int64_t scalar_shape[1] = {{1}};
    chelis_tensor* b = chelis_tensor_entry_borrow(0, scalar_shape, CHELIS_DTYPE_F32, bd, sizeof(float));
    chelis_tensor* inputs[2] = {{make_view_1d(xd, {extent}), b}};
    chelis_tensor* outputs[1] = {{NULL}};
    local_guard(inputs, 2, outputs, 1);
    printf("RAN %lld\n", (long long)chelis_tensor_shape(outputs[0], 0));
    return 0;
}}
"#
        );
        compile_and_run_kernel_capturing(name, &result.c_source, &harness)
    };

    // `stride(x, 2)` yields `ceil(n / 2)`, so the claim `n` and the extent
    // the rank-inserting operation actually reads agree only at n = 1.
    let (ok, out) = run("local_guard", 3);
    assert!(!ok, "ceil(3/2) = 2 is not the claimed 3: {out}");
    assert!(
        out.contains("numeric trap: domain in insert at int64"),
        "the local guard renders [04-NUM-9]: {out}"
    );
    assert!(
        out.contains("extent `n`: claimed = 3,"),
        "with the claim and its observed value: {out}"
    );
    assert!(
        !out.contains("NO TRAP") && !out.contains("RAN "),
        "and preempts the operation it guards: {out}"
    );

    let (ok, out) = run("local_guard_ok", 1);
    assert!(ok, "ceil(1/2) = 1 agrees and must execute: {out}");
    assert!(
        out.contains("RAN 1"),
        "and produce the claimed shape: {out}"
    );
}

/// [04] section 4.7: a declared number is a claim, not evidence about
/// the independent size carrier. Exercise live Expand/Reshape sites with
/// computed scalar sizes and reads of computed tensors, on both host lanes.
#[test]
fn numeric_local_extent_claims_execute_exactly() {
    use chelis_ir::dag::{RtAxis, RtDim};
    use chelis_ir::eval::{TensorValue, eval_tensor_with};
    let mut failures = Vec::new();
    let mut executions = 0;
    for resolved_name in [false, true] {
        for reshape in [false, true] {
            for tensor_size in [false, true] {
                let mut dag = Dag::new();
                let claim = if resolved_name {
                    DimInfo::Named("n".into(), Some(4))
                } else {
                    DimInfo::Lit(4)
                };
                let x = dag.add_node(
                    RiscOp::Load { name: "x".into() },
                    vec![],
                    TensorType {
                        dims: vec![claim.clone()],
                        precision: Prim::F32,
                    },
                    None,
                );
                let integer = TensorType {
                    dims: vec![],
                    precision: Prim::Int64,
                };
                let delta = dag.add_node(
                    RiscOp::Load {
                        name: "delta".into(),
                    },
                    vec![],
                    integer.clone(),
                    None,
                );
                let (size, carrier) = if tensor_size {
                    let strided = dag.add_node(
                        RiscOp::Stride {
                            strides: vec![RtDim::Node(1)],
                        },
                        vec![x, delta],
                        TensorType {
                            dims: vec![DimInfo::Named("computed".into(), None)],
                            precision: Prim::F32,
                        },
                        None,
                    );
                    (
                        strided,
                        RtDim::InputAxis {
                            tensor: 1,
                            axis: RtAxis::Lit(0),
                        },
                    )
                } else {
                    let read =
                        dag.add_node(RiscOp::Shape { axis: 0 }, vec![x], integer.clone(), None);
                    let sum = dag.add_node(RiscOp::Add, vec![read, delta], integer, None);
                    (sum, RtDim::Node(1))
                };
                let operand = if reshape {
                    x
                } else {
                    dag.add_node(
                        RiscOp::Sum {
                            axis: 0,
                            accumulator: Prim::F32,
                        },
                        vec![x],
                        TensorType {
                            dims: vec![],
                            precision: Prim::F32,
                        },
                        None,
                    )
                };
                let op = if reshape {
                    RiscOp::Reshape {
                        new_shape: vec![carrier],
                    }
                } else {
                    RiscOp::Expand {
                        axis: 0,
                        size: carrier,
                    }
                };
                let root = dag.add_node(
                    op,
                    vec![operand, size],
                    TensorType {
                        dims: vec![claim],
                        precision: Prim::F32,
                    },
                    None,
                );
                dag.add_root(root);
                let generated = codegen_with_options(
                    &dag,
                    "numeric_claim",
                    CodegenOptions {
                        use_blas: false,
                        math_lib_override: Some(MathLib::None),
                        static_entry: false,
                    },
                )
                .expect("the runtime carrier is supported");
                assert_eq!(generated.input_labels, ["x", "delta"]);
                for good in [true, false] {
                    let delta_value = if tensor_size {
                        if good { 1 } else { 2 }
                    } else if good {
                        0
                    } else {
                        1
                    };
                    let observed = if good {
                        4
                    } else if tensor_size {
                        2
                    } else {
                        5
                    };
                    let name =
                        format!("numeric_local_{resolved_name}_{reshape}_{tensor_size}_{good}");
                    let op = if reshape { "reshape" } else { "insert" };
                    let trap = format!("numeric trap: domain in {op} at int64");
                    let context = format!("node {} axis 0 = {observed}", root.0);
                    let expected = if reshape {
                        vec![1.0, 2.0, 3.0, 4.0]
                    } else {
                        vec![10.0; 4]
                    };
                    let evaluation = eval_tensor_with(&dag, |input| match input {
                        "x" => Some(TensorValue::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0])),
                        "delta" => Some(TensorValue::scalar(f64::from(delta_value))),
                        _ => None,
                    });
                    match evaluation {
                        Ok(values) if good => {
                            let actual = &values[&root];
                            if actual.shape != [4] || actual.to_f64_lossy_vec() != expected {
                                failures
                                    .push(format!("{name}.eval: wrong shape/value: {actual:?}"));
                            }
                        }
                        Err(error)
                            if !good
                                && error.lines().any(|line| line == trap)
                                && error.contains("claimed = 4")
                                && error.contains(&context) => {}
                        other => failures.push(format!(
                            "{name}.eval: expected good={good}, observed {other:?}"
                        )),
                    }
                    let expected_c = if reshape {
                        "{1,2,3,4}"
                    } else {
                        "{10,10,10,10}"
                    };
                    let harness = format!(
                        r#"{HARNESS_HEADER}
extern void numeric_claim(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);
int main(void) {{
    float xd[4] = {{1,2,3,4}};
    int64_t delta = {delta_value};
    chelis_tensor* inputs[2] = {{make_view_1d(xd, 4), chelis_tensor_entry_borrow(0, NULL, CHELIS_DTYPE_I64, &delta, sizeof(delta))}};
    chelis_tensor* outputs[1] = {{NULL}};
    numeric_claim(inputs, 2, outputs, 1);
    if (chelis_tensor_rank(outputs[0]) != 1 || chelis_tensor_shape(outputs[0], 0) != 4) return 41;
    chelis_tensor* dense = chelis_contiguous(outputs[0]);
    chelis_read_view view = chelis_tensor_read_view(dense);
    float expected[4] = {expected_c};
    if (view.dtype != CHELIS_DTYPE_F32 || view.count != 4) return 42;
    for (int i = 0; i < 4; i++) if (((const float*)view.data)[i] != expected[i]) return 43;
    puts("EXACT SHAPE AND VALUES");
    return 0;
}}
"#
                    );
                    let (ok, output) =
                        compile_and_run_kernel_capturing(&name, &generated.c_source, &harness);
                    if good {
                        if !ok || output.trim() != "EXACT SHAPE AND VALUES" {
                            failures.push(format!(
                                "{name}.c: expected exact execution, ok={ok}: {output}"
                            ));
                        }
                    } else if ok
                        || !output.lines().any(|line| line == trap)
                        || !output.contains("claimed = 4")
                        || !output.contains(&context)
                    {
                        failures.push(format!(
                            "{name}.c: expected {trap} with {context}, ok={ok}: {output}"
                        ));
                    }
                    executions += 2;
                }
            }
        }
    }
    assert_eq!(executions, 32, "every pair must execute both lanes");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A mixed class keeps the static ABI check on its resolved declaring input.
/// The entry-class narrowing must not erase it merely because the class also
/// has local carrier members.
#[test]
fn an_interface_member_of_a_mixed_class_is_still_checked() {
    use chelis_ir::dag::{Dag, RiscOp, RtAxis, RtDim, TensorType};
    let resolved = || TensorType {
        dims: vec![DimInfo::Named("n".into(), Some(4))],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], resolved(), None);
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    // Interface member: an `expand` sized by a folded read of the INPUT `x`.
    let from_input = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, x],
        resolved(),
        None,
    );
    // Local member: an `expand` sized by a folded read of a COMPUTED tensor.
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        TensorType {
            dims: vec![DimInfo::Named("s".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let from_computed = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, strided],
        resolved(),
        None,
    );
    let out = dag.add_node(
        RiscOp::Add,
        vec![from_input, from_computed],
        resolved(),
        None,
    );
    dag.add_root(out);

    let result = codegen_with_options(
        &dag,
        "mixed_class",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");
    assert!(
        result.c_source.contains("input `x` axis 0 expected 4"),
        "the caller's obligation on the interface member survives: {}",
        result.c_source
    );

    let harness = format!(
        r#"{HARNESS_HEADER}
extern void mixed_class(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[5] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}};
    float bd[1] = {{7.0f}};
    int64_t scalar_shape[1] = {{1}};
    chelis_tensor* b = chelis_tensor_entry_borrow(0, scalar_shape, CHELIS_DTYPE_F32, bd, sizeof(float));
    chelis_tensor* inputs[2] = {{b, make_view_1d(xd, 5)}};
    chelis_tensor* outputs[1] = {{NULL}};
    mixed_class(inputs, 2, outputs, 1);
    printf("NO TRAP\n");
    return 0;
}}
"#
    );
    let (ok, out) = compile_and_run_kernel_capturing("mixed_class", &result.c_source, &harness);
    assert!(
        !ok,
        "a caller passing 5 for a declared 4 must not run: {out}"
    );
    assert!(
        !out.contains("NO TRAP"),
        "and must not reach the body: {out}"
    );
}

/// A literal claim reading a symbolic input, in a class placed Local by a
/// computed co-member. No static ABI check covers the input; the local
/// carrier consumer must check its independently supplied extent.
fn symbolic_input_mixed_class_dag() -> chelis_ir::dag::Dag {
    use chelis_ir::dag::{Dag, RiscOp, RtAxis, RtDim, TensorType};
    let four = || TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let from_input = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, x],
        four(),
        None,
    );
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        TensorType {
            dims: vec![DimInfo::Named("s".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let from_computed = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, strided],
        four(),
        None,
    );
    let out = dag.add_node(RiscOp::Add, vec![from_input, from_computed], four(), None);
    dag.add_root(out);
    dag
}

#[test]
fn a_literal_claim_on_a_symbolic_input_in_a_local_class_traps() {
    let dag = symbolic_input_mixed_class_dag();
    let result = codegen_with_options(
        &dag,
        "sym_mixed",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");
    // The symbolic input supplies no static ABI size check. The operation
    // still owes its local literal claim before allocation or indexing.
    assert!(!result.c_source.contains("input `x` axis 0 expected"));
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void sym_mixed(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[5] = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}};
    float bd[1] = {{7.0f}};
    int64_t scalar_shape[1] = {{1}};
    chelis_tensor* b = chelis_tensor_entry_borrow(0, scalar_shape, CHELIS_DTYPE_F32, bd, sizeof(float));
    chelis_tensor* inputs[2] = {{make_view_1d(xd, 5), b}};
    chelis_tensor* outputs[1] = {{NULL}};
    sym_mixed(inputs, 2, outputs, 1);
    printf("NO TRAP shape=%lld\n", (long long)chelis_tensor_shape(outputs[0], 0));
    return 0;
}}
"#
    );
    let (ok, out) = compile_and_run_kernel_capturing("sym_mixed", &result.c_source, &harness);
    assert!(!ok, "the local literal claim must trap: {out}");
    assert!(
        out.lines()
            .any(|line| line == "numeric trap: domain in insert at int64")
            && out.contains("claimed = 4")
            && out.contains("node 2 axis 0 = 5")
            && !out.contains("NO TRAP"),
        "the failure must identify the literal claim and observed carrier: {out}"
    );
}

/// An unresolved symbolic input supplies an independently observed extent
/// under a resolved named claim. A computed co-member places the class Local;
/// the resolved result metadata must not exempt its interface member either.
#[test]
fn an_interface_member_with_a_resolved_dim_keeps_its_site_in_a_local_class() {
    use chelis_ir::dag::{Dag, RiscOp, RtAxis, RtDim, TensorType};
    let open = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let resolved = || TensorType {
        dims: vec![DimInfo::Named("n".into(), Some(4))],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let x = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], open("n"), None);
    let y = dag.add_node(RiscOp::Load { name: "y".into() }, vec![], open("m"), None);
    let b = dag.add_node(
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    // Interface member: the size folds a read of the INPUT `y`, while the
    // claim `n` is declared from `x`, so the comparison is between two
    // different inputs rather than a value with itself.
    let from_input = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, y],
        resolved(),
        None,
    );
    let strided = dag.add_node(
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        open("s"),
        None,
    );
    let from_computed = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::InputAxis {
                tensor: 1,
                axis: RtAxis::Lit(0),
            },
        },
        vec![b, strided],
        resolved(),
        None,
    );
    let out = dag.add_node(
        RiscOp::Add,
        vec![from_input, from_computed],
        resolved(),
        None,
    );
    dag.add_root(out);

    let result = codegen_with_options(
        &dag,
        "iface_resolved",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("codegen");
    assert!(
        !result.c_source.contains("axis 0 expected"),
        "no ABI static-dim check covers a symbolic declared dim: {}",
        result.c_source
    );
    assert!(
        result.c_source.contains("node 3 axis 0 = %lld"),
        "the interface member of a Local class keeps its site: {}",
        result.c_source
    );
}

/// Spec/04 section 4.7: simultaneous local guards follow declaration order,
/// even when reshape reverses the claimed axes. Each lane owes the exact first
/// failure independently; the matching control must preserve shape and values.
#[test]
fn local_reshape_guards_follow_declaration_order() {
    use chelis_ir::dag::RtDim;
    use chelis_ir::eval::{TensorValue, eval_tensor_with};
    let mut dag = Dag::new();
    let n = DimInfo::Named("n".into(), Some(4));
    let m = DimInfo::Named("m".into(), Some(2));
    let x = dag.add_node(
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![n.clone(), m.clone()],
            precision: Prim::F32,
        },
        None,
    );
    let int = TensorType {
        dims: vec![],
        precision: Prim::Int64,
    };
    let a = dag.add_node(RiscOp::Load { name: "a".into() }, vec![], int.clone(), None);
    let b = dag.add_node(RiscOp::Load { name: "b".into() }, vec![], int.clone(), None);
    let ashape = dag.add_node(RiscOp::Shape { axis: 1 }, vec![x], int.clone(), None);
    let bshape = dag.add_node(RiscOp::Shape { axis: 0 }, vec![x], int.clone(), None);
    let ac = dag.add_node(RiscOp::Add, vec![ashape, a], int.clone(), None);
    let bc = dag.add_node(RiscOp::Add, vec![bshape, b], int, None);
    let root = dag.add_node(
        RiscOp::Reshape {
            new_shape: vec![RtDim::Node(1), RtDim::Node(2)],
        },
        vec![x, ac, bc],
        TensorType {
            dims: vec![m, n],
            precision: Prim::F32,
        },
        None,
    );
    dag.add_root(root);
    let generated = codegen_with_options(
        &dag,
        "order_probe",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .unwrap();
    assert_eq!(generated.input_labels, ["x", "a", "b"]);
    for (delta_a, delta_b) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        let good = delta_a == 0 && delta_b == 0;
        let eval = eval_tensor_with(&dag, |name| match name {
            "x" => Some(TensorValue::from_vec(
                vec![4, 2],
                (1..=8).map(f64::from).collect(),
            )),
            "a" => Some(TensorValue::scalar(f64::from(delta_a))),
            "b" => Some(TensorValue::scalar(f64::from(delta_b))),
            _ => None,
        });
        let harness = format!(
            r#"{HARNESS_HEADER}
extern void order_probe(chelis_tensor**,int,chelis_tensor**,int);
int main(void) {{ float x[8]={{1,2,3,4,5,6,7,8}}; int64_t sh[2]={{4,2}},da={delta_a},db={delta_b};
chelis_tensor* inputs[3]={{chelis_tensor_entry_borrow(2,sh,CHELIS_DTYPE_F32,x,sizeof(x)),chelis_tensor_entry_borrow(0,NULL,CHELIS_DTYPE_I64,&da,sizeof(da)),chelis_tensor_entry_borrow(0,NULL,CHELIS_DTYPE_I64,&db,sizeof(db))}};
chelis_tensor* outputs[1]={{NULL}}; order_probe(inputs,3,outputs,1);
if(chelis_tensor_rank(outputs[0])!=2 || chelis_tensor_shape(outputs[0],0)!=2 || chelis_tensor_shape(outputs[0],1)!=4) return 41;
chelis_read_view v=chelis_tensor_read_view(outputs[0]); if(v.count!=8) return 42;
for(int i=0;i<8;i++) if(((const float*)v.data)[i]!=i+1) return 43;
puts("EXACT"); return 0; }}"#
        );
        let (ok, c) = compile_and_run_kernel_capturing(
            "local_reshape_guard_order",
            &generated.c_source,
            &harness,
        );
        if good {
            let v = eval.unwrap();
            assert_eq!(v[&root].shape, [2, 4]);
            assert_eq!(
                v[&root].to_f64_lossy_vec(),
                (1..=8).map(f64::from).collect::<Vec<_>>()
            );
            assert!(ok, "{c}");
            assert_eq!(c.trim(), "EXACT");
        } else {
            let e = eval.unwrap_err();
            assert!(!ok, "mismatching claims must fail: {c}");
            let (claim, required, axis, observed) = if delta_b != 0 {
                ("n", 4, 1, 5)
            } else {
                ("m", 2, 0, 3)
            };
            let context = format!(
                "extent {claim}: claimed = {required}, node {} axis {axis} = {observed}",
                root.0
            );
            for (lane, diagnostic) in [("eval", e), ("c", c)] {
                let diagnostic = diagnostic.replace('`', "");
                assert!(
                    diagnostic
                        .lines()
                        .any(|line| line == "numeric trap: domain in reshape at int64"),
                    "{lane}: {diagnostic}"
                );
                assert!(
                    diagnostic.lines().any(|line| line == context),
                    "{lane}: expected {context}, observed {diagnostic}"
                );
            }
        }
    }
}
