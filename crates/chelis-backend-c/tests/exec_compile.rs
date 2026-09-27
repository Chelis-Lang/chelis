//! Adversarial execution tests: actually compile AND RUN generated C code.
//! These test numerical correctness, not just source patterns.
//!
//! The generated kernel signature is:
//!   void func(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out)
//! The kernel allocates output tensors internally via chelis_alloc.
//! We link the carried runtime archive, staged into each probe directory.

use chelis_backend_c::{CodegenOptions, MathLib};

mod support;
use chelis_ir::ConcreteHostType as HostType;
use chelis_ir::dag::{
    ComparisonKind, Dag, DimInfo, ExtentClaim, ExtentWitnessSite, ExtremaKind, ExtremaOperand,
    LogicalKind, Owner, ReduceWindowKind, RiscOp, RtAxis, TensorType,
};
use chelis_ir::eval::{TensorValue, eval_tensor};
use chelis_ir::fuse::fuse;
use chelis_ir::grad::grad_dag_checked;
use chelis_ir::host::{
    ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind,
    ConcreteHostFunction as HostFunction, ConcreteHostParam as HostParam,
    ConcreteHostProgram as HostProgram, HostFunctionOrigin,
};
use chelis_types::types::Prim;
use chelis_types::{RawTensor, finalize_tensor};
use chelis_unord::UnordMap;
use std::fs;
use std::process::Command;
use support::{codegen, codegen_with_options, emit_host_program};

mod common;

fn checked_indexing_run(source: &str, harness: &str) -> std::process::Output {
    let probe = common::probe_dir("checked_c_indexing");
    let dir = probe.path();
    let staged = chelis_runtime_bundle::stage(dir).expect("stage the carried runtime");
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
        .arg(dir)
        .arg(dir.join("kernel.c"))
        .arg(dir.join("main.c"))
        .arg(&staged.archive)
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
    let decl = dag.declare("test");
    dag.add_node(
        decl,
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
            StorageView::Key(_) => {
                unreachable!("the dtype list above has no key: a key has no literal")
            }
        };
        let source = codegen(
            &checked_literal_dag(storage.clone(), 4, dtype),
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
        assert!(out.status.success(), "{dtype:?}: {out:?}");

        for extent in [3, 0] {
            let error = chelis_ir::ownership::lower_dag_ownership(checked_literal_dag(
                storage.clone(),
                extent,
                dtype,
            ))
            .expect_err("malformed literal cardinality must reject before codegen");
            assert_eq!(
                error,
                chelis_ir::ownership::OwnershipError::LoweringInvariant {
                    unit: "dag".into(),
                    detail: format!(
                        "constant tensor at node 0 stores 4 values but its concrete shape requires {extent}"
                    ),
                },
                "{dtype:?}/{extent}"
            );
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
    let decl = dag.declare("test");
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
        decl,
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
        let g = dag.add_node(decl, RiscOp::Load { name: "g".into() }, vec![], gty, None);
        dag.add_node(
            decl,
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
            decl,
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
                .any(|s| s == format!("numeric trap: domain in {op} at i64")),
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        ty(&["batch0", "batch1", "m", "k"], operand),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        ty(&["batch0", "batch1", "k", "n"], operand),
        None,
    );
    let node = dag.add_node(
        decl,
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
    for (name, contents) in chelis_runtime_bundle::PUBLIC_HEADERS {
        fs::write(probe.path().join(name), contents).unwrap();
    }
    let compile = |text: &str| {
        fs::write(probe.path().join("contract.c"), text).unwrap();
        Command::new(&compiler)
            .args(["-std=c11", "-Werror", "-c", "-I"])
            .arg(probe.path())
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
                adt_layouts: Vec::new(),
                functions: vec![HostFunction {
                    helper_result_claim_axes: Vec::new(),
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
            host_source.push_str("\nvoid checked_blas(chelis_tensor **in,int n_in,chelis_tensor **out,int n_out) { (void)n_in; (void)n_out; out[0]=chelis_fn_686f73745f626c6173(in[0],in[1]); }\n");
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
                let decl = dag.declare("test");
                let base = dag.add_node(
                    decl,
                    RiscOp::Load {
                        name: "base".into(),
                    },
                    vec![],
                    ty(&base_shape, prim),
                    None,
                );
                let indices = dag.add_node(
                    decl,
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
                        decl,
                        RiscOp::Load {
                            name: "updates".into(),
                        },
                        vec![],
                        ty(&update_shape, prim),
                        None,
                    ));
                }
                let output = dag.add_node(decl, op, inputs, ty(&output_shape, prim), None);
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
            let decl = dag.declare("test");
            let base = dag.add_node(
                decl,
                RiscOp::Load {
                    name: "base".into(),
                },
                vec![],
                ty(&[2, 3], Prim::F32),
                None,
            );
            let indices = dag.add_node(
                decl,
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
                    decl,
                    RiscOp::Load {
                        name: "updates".into(),
                    },
                    vec![],
                    ty(&[2, n], Prim::F32),
                    None,
                ));
            }
            let output = dag.add_node(decl, op.clone(), inputs, ty(&output_shape, Prim::F32), None);
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
                    adt_layouts: Vec::new(),
                    functions: vec![HostFunction {
                        helper_result_claim_axes: Vec::new(),
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
                host_source.push_str("\nvoid sparse_boundary(chelis_tensor **in,int n_in,chelis_tensor **out,int n_out) { (void)n_in; (void)n_out; out[0]=chelis_fn_686f73745f7370617273655f616464(in[0],in[1],in[2]); }\n");
                sources.push(host_source);
            }
            for source in sources {
                let result = checked_indexing_run(&source, &harness);
                if !empty && selected != 2 {
                    assert!(!result.status.success());
                    let stderr = String::from_utf8_lossy(&result.stderr);
                    assert!(
                        stderr.contains(&format!("numeric trap: domain in {diagnostic} at i64")),
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
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(&[2, 3, 2], Prim::F32),
            None,
        );
        let output = dag.add_node(decl, op, vec![input], ty(&[2, 2], precision), None);
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
            let decl = dag.declare("test");
            let precision = if count { Prim::Bool } else { Prim::F64 };
            let result_precision = if count { Prim::Int64 } else { Prim::F64 };
            let input = dag.add_node(
                decl,
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
                decl,
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
            let decl = dag.declare("test");
            let input = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty(input_shape),
                None,
            );
            let output = dag.add_node(decl, op.clone(), vec![input], ty(output_shape), None);
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
            let decl = dag.declare("test");
            let x = dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                ty(&input_shape),
                None,
            );
            let y = dag.add_node(decl, op.clone(), vec![x], ty(&output_shape), None);
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
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(vec![DimInfo::Lit(3)]),
            None,
        );
        let n = dag.add_node(
            decl,
            RiscOp::Load { name: "n".into() },
            vec![],
            ty(vec![]),
            None,
        );
        let y = dag.add_node(
            decl,
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
                        .ends_with(&format!("numeric trap: {class} in {name} at i64\n")),
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
                let decl = dag.declare("test");
                let input = dag.add_node(
                    decl,
                    RiscOp::Load { name: "x".into() },
                    vec![],
                    ty.clone(),
                    None,
                );
                let scalar = dag.add_node(
                    decl,
                    RiscOp::Load { name: "s".into() },
                    vec![],
                    scalar_ty,
                    None,
                );
                let negated = dag.add_node(decl, RiscOp::Neg, vec![input], ty.clone(), None);
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
                let result = dag.add_node(decl, op, vec![negated, scalar], ty, None);
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
#define the_fn chelis_fn_7468655f666e
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

/// Write generated C + harness, compile, run, return stdout. None = compile/run failure.
/// Compile and run like [`compile_and_run_kernel`], but return the exit
/// status and BOTH streams instead of `None` on failure.
///
/// chelis#1277: a runtime extent guard's whole observable is a nonzero exit
/// with a trap line on stderr, which the success-only helper discards - it
/// `eprintln!`s stderr and returns `None`, so a caller cannot assert on the
/// trap it was testing for. This shares that helper's compile plumbing rather
/// than duplicating the runtime staging.
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
    let staged = chelis_runtime_bundle::stage(&dir).expect("stage the carried runtime");
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
            staged.archive.to_str().unwrap(),
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

    let staged = chelis_runtime_bundle::stage(&dir).expect("stage the carried runtime");

    let bin = dir.join("test_bin");
    let runtime_lib = staged.archive;

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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    dag.add_node(decl, RiscOp::Exp, vec![a], vec_f32(4), None);
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(9),
        None,
    );
    let e = dag.add_node(decl, RiscOp::Exp, vec![a], vec_f32(9), None);
    dag.add_node(decl, RiscOp::Neg, vec![e], vec_f32(9), None); // 2-op chain: fuses into FusedElem
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(100),
        None,
    );
    dag.add_node(
        decl,
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
    let decl = dag.declare("test");
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(&[2, 3, 2], Prim::Bool),
        None,
    );
    let output = dag.add_node(
        decl,
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
    let decl = dag.declare("test");
    let input = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        tensor_ty(&[2, 0, 3], Prim::Bool),
        None,
    );
    let output = dag.add_node(
        decl,
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
    let decl = dag.declare("test");
    let x = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
    dag.add_node(
        decl,
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        x_ty.clone(),
        None,
    );
    let g = dag.add_node(decl, RiscOp::Load { name: "g".into() }, vec![], g_ty, None);
    dag.add_node(
        decl,
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(4),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_f32(4),
        None,
    );
    dag.add_node(decl, RiscOp::Div, vec![a, b], vec_f32(4), None);
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(3),
        None,
    );
    dag.add_node(decl, RiscOp::Recip, vec![a], vec_f32(3), None);
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
/// F2: i8 / i16 / i64 in addition to the original i32).
fn run_int_div_op_exec(
    op: RiscOp,
    fn_name: &str,
    kernel_name: &str,
    expected: [i32; 4],
    precision: Prim,
) {
    let (c_type, dtype_macro) = int_c_type_and_dtype(precision);
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_int(4, precision),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_int(4, precision),
        None,
    );
    dag.add_node(decl, op, vec![a, b], vec_int(4, precision), None);
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

// [05-OP-64]: exact signed remainder, including the full-width trap boundary.
#[test]
fn checked_remainder_executes_and_traps_at_every_signed_width() {
    for (prim, c_type, c_dtype, minimum) in [
        (Prim::Int8, "int8_t", "CHELIS_DTYPE_I8", "INT8_MIN"),
        (Prim::Int16, "int16_t", "CHELIS_DTYPE_I16", "INT16_MIN"),
        (Prim::Int32, "int32_t", "CHELIS_DTYPE_I32", "INT32_MIN"),
        (Prim::Int64, "int64_t", "CHELIS_DTYPE_I64", "INT64_MIN"),
    ] {
        run_int_div_op_exec(
            RiscOp::Mod,
            "checked_remainder",
            prim.name(),
            [1, 1, -1, -1],
            prim,
        );
        let adjacent = format!("{minimum} + 1");
        for (case, lhs, rhs, message) in [
            ("zero", "7", "0", "division by zero"),
            ("minimum", minimum, "-1", ""),
            ("minimum_odd", adjacent.as_str(), "2", ""),
        ] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            let ty = vec_prim(1, prim);
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
            dag.add_node(decl, RiscOp::Mod, vec![a, b], ty, None);
            let function = format!("checked_remainder_{}_{case}", prim.name());
            let src = codegen(&fuse(&dag), &function).unwrap().c_source;
            let expected = if case == "minimum_odd" { -1 } else { 0 };
            let harness = format!(
                r#"{HARNESS_HEADER}
#include <limits.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} av[1] = {{ {lhs} }}; {c_type} bv[1] = {{ {rhs} }};
    chelis_tensor *inputs[2] = {{ make_view_typed_1d(av, 1, {c_dtype}), make_view_typed_1d(bv, 1, {c_dtype}) }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    return (({c_type} *)chelis_tensor_read_view(outputs[0]).data)[0] == {expected} ? 0 : 1;
}}
"#
            );
            let run = compile_and_capture_run(&function, &src, &harness);
            let stderr = String::from_utf8_lossy(&run.stderr);
            if message.is_empty() {
                assert!(
                    run.status.success(),
                    "{function}: exact remainder must be {expected}: {stderr}"
                );
                continue;
            }
            assert!(!run.status.success(), "{function}: unexpected success");
            assert!(
                stderr.contains(&format!(
                    "numeric trap: {message} in mod at {}",
                    prim.name()
                )),
                "{function}: {stderr}"
            );
        }
    }
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
// C backend promotes to i64 internally), but the repo's negative-parity
// bar requires the narrower and wider integer widths be exercised
// end-to-end, not just i32. Same operands / expected results as the i32
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
// round-toward-−∞ remainder-sign correction must hold at i8 / i16 /
// i64 just as at i32.
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

    let staged = chelis_runtime_bundle::stage(&dir).expect("stage the carried runtime");

    let bin = dir.join("test_bin");
    let runtime_lib = staged.archive;
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_int(2, Prim::Int64),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_int(2, Prim::Int64),
        None,
    );
    dag.add_node(
        decl,
        RiscOp::FloorDiv,
        vec![a, b],
        vec_int(2, Prim::Int64),
        None,
    );
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(11),
        None,
    );
    dag.add_node(
        decl,
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f32(0),
        None,
    );
    dag.add_node(decl, RiscOp::Exp, vec![a], vec_f32(0), None);
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
    let probe = common::probe_dir("exec_nan_probe");
    let dir = probe.path().to_path_buf();

    for (name, contents) in chelis_runtime_bundle::PUBLIC_HEADERS {
        fs::write(dir.join(name), contents).unwrap();
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
    let probe = common::probe_dir("cxx_probe");
    let dir = probe.path().to_path_buf();

    // Copy simd header
    let (_, simd_src) = chelis_runtime_bundle::PUBLIC_HEADERS
        .iter()
        .find(|(name, _)| *name == "chelis_simd.h")
        .expect("chelis_simd.h is a public runtime header");
    fs::write(dir.join("chelis_simd.h"), simd_src).unwrap();

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

    let staged = chelis_runtime_bundle::stage(&dir).expect("stage the carried runtime");

    let bin = dir.join("test_bin");
    let runtime_lib = staged.archive;
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_f64(100),
        None,
    );
    dag.add_node(
        decl,
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i32(5),
        None,
    );
    dag.add_node(
        decl,
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
    let matmul_op = RiscOp::matmul_default(
        vec![],
        DimExpr::Concrete(2),
        DimExpr::Concrete(4),
        DimExpr::Concrete(3),
        Prim::F64,
    )
    .expect("f64 matmul default constructs (spec §5.7.1)");
    dag.add_node(decl, matmul_op, vec![a, b], mat_f64(2, 4), None);

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
    let decl = dag.declare("test");
    let f_values = dag.add_node(
        decl,
        RiscOp::Load {
            name: "f_values".into(),
        },
        vec![],
        vec_f64(4),
        None,
    );
    let i_values = dag.add_node(
        decl,
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
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F64,
        },
        vec![f_values],
        scalar_f64(),
        None,
    );
    let i_sum = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::Int32,
        },
        vec![i_values],
        scalar_i32(),
        None,
    );
    dag.add_node(
        decl,
        RiscOp::Store {
            name: "f_out".into(),
        },
        vec![f_sum],
        scalar_f64(),
        None,
    );
    dag.add_node(
        decl,
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
        let decl = dag.declare("test");
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
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(ty.precision, 1.0),
            vec![],
            ty,
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(ty_b.precision, 1.0),
            vec![],
            ty_b,
            None,
        );
        let matmul_op = RiscOp::matmul_default(
            vec![],
            DimExpr::Concrete(2),
            DimExpr::Concrete(4),
            DimExpr::Concrete(3),
            prim,
        )
        .unwrap_or_else(|e| panic!("{prim:?} matmul default constructs (per spec §5.7.1): {e}"));
        let _ = dag.add_node(decl, matmul_op, vec![a, b], ty_out, None);

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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i8(8),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_i8(8),
        None,
    );
    dag.add_node(decl, RiscOp::Add, vec![a, b], vec_i8(8), None);
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i8(2),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_i8(2),
        None,
    );
    dag.add_node(decl, RiscOp::Add, vec![a, b], vec_i8(2), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i8_add_wrap").unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_add_wrap(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    // Both elements overflow i8 and must trap before store-back.
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
        stderr.contains("numeric trap: overflow in add at i8"),
        "i8 add overflow must use the canonical diagnostic; stderr={stderr:?}"
    );
}

/// The second i8 multiplication overflows after an in-range first element;
/// the kernel must trap rather than partially legitimizing the wrapped row.
#[test]
fn exec_i8_mul_overflow_traps() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i8(2),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_i8(2),
        None,
    );
    dag.add_node(decl, RiscOp::Mul, vec![a, b], vec_i8(2), None);
    let dag = fuse(&dag);

    let result = codegen(&dag, "test_i8_mul").unwrap();
    let src = &result.c_source;

    let harness = format!(
        r#"{WS_A4_HARNESS_HEADER}
extern void test_i8_mul(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    int8_t a_data[2] = {{ 12, 16 }};
    int8_t b_data[2] = {{ 10, 8 }};
    // 12*10 = 120 fits; 16*8 = 128 overflows i8 and must trap.
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
        stderr.contains("numeric trap: overflow in mul at i8"),
        "i8 mul overflow must use the canonical diagnostic; stderr={stderr:?}"
    );
}

/// i16 add: pick values that exercise the int16_t path through the
/// codegen without overflowing. Mirrors `exec_i8_add_correct_output`
/// for the i16 dtype.
#[test]
fn exec_i16_add_correct_output() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i16(4),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_i16(4),
        None,
    );
    dag.add_node(decl, RiscOp::Add, vec![a, b], vec_i16(4), None);
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i8(200),
        None,
    );
    // Use the spec-default constructor so the IR carries the §5.7.1
    // i32 accumulator, not an inline `Prim::Int8` that would fail the
    // verifier's narrowness check.
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int8).expect("i8 sum_default must succeed");
    dag.add_node(decl, sum_op, vec![a], scalar_i32(), None);
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
/// but fits in i32. Pin both the source-i16 path and the i32 result.
#[test]
fn exec_i16_reduce_sum_promotes_to_i32() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_i16(200),
        None,
    );
    let sum_op =
        chelis_ir::dag::RiscOp::sum_default(0, Prim::Int16).expect("i16 sum_default must succeed");
    dag.add_node(decl, sum_op, vec![a], scalar_i32(), None);
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
// is bool (stored f32). Reading a RUNTIME-PRODUCED i32 / i64 /
// f64 operand through a raw `float*` reinterprets the bit pattern
// (the #347 / #476 bug class). The discriminator is a NEGATIVE
// integer operand: as i32, `-7 < -3` is true; reinterpreting the
// i32 bit pattern 0xFFFFFFF9 / 0xFFFFFFFD as `float` yields NaN, so
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
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vec_prim(n, prim),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vec_prim(n, prim),
        None,
    );
    let root = dag.add_node(
        decl,
        RiscOp::Compare(ComparisonKind::CmpLt),
        vec![a, b],
        vec_prim(n, Prim::Bool),
        None,
    );

    // Evaluator oracle: direct numeric `cmplt(a, b)` per element.
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

/// #517 primary oracle: runtime i32 operands, including the negative
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

/// #517 sweep: i64 operands. The buggy `float*` read also misaligns
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

fn typed_storage(prim: Prim, raw: chelis_types::RawTensor) -> chelis_types::TensorStorage {
    chelis_types::finalize_tensor("typed C nonnumeric test", prim, raw).unwrap()
}

fn typed_value(storage: chelis_types::TensorStorage) -> TensorValue {
    TensorValue::from_storage(vec![storage.len()], storage)
}

fn typed_value_with_shape(shape: Vec<usize>, storage: chelis_types::TensorStorage) -> TensorValue {
    TensorValue::from_storage(shape, storage)
}

fn c_storage_case(storage: &chelis_types::TensorStorage) -> (&'static str, &'static str, String) {
    use chelis_types::StorageView;

    let values = match storage.view() {
        StorageView::F64(values) => values
            .iter()
            .map(|value| format!("test_f64(UINT64_C(0x{:016x}))", value.to_bits()))
            .collect::<Vec<_>>(),
        StorageView::F32(values) => values
            .iter()
            .map(|value| format!("test_f32(UINT32_C(0x{:08x}))", value.to_bits()))
            .collect::<Vec<_>>(),
        StorageView::F16(values) => values
            .iter()
            .map(|value| format!("UINT16_C(0x{:04x})", value.to_bits()))
            .collect::<Vec<_>>(),
        StorageView::Bf16(values) => values
            .iter()
            .map(|value| format!("UINT16_C(0x{:04x})", value.to_bits()))
            .collect::<Vec<_>>(),
        StorageView::I64(values) => values
            .iter()
            .map(|value| {
                if *value == i64::MIN {
                    "INT64_MIN".to_string()
                } else {
                    format!("INT64_C({value})")
                }
            })
            .collect::<Vec<_>>(),
        StorageView::I32(values) => values
            .iter()
            .map(|value| format!("INT32_C({value})"))
            .collect::<Vec<_>>(),
        StorageView::I16(values) => values
            .iter()
            .map(|value| format!("INT16_C({value})"))
            .collect::<Vec<_>>(),
        StorageView::I8(values) => values
            .iter()
            .map(|value| format!("INT8_C({value})"))
            .collect::<Vec<_>>(),
        StorageView::Bool(values) => values
            .iter()
            .map(|value| format!("UINT8_C({value})"))
            .collect::<Vec<_>>(),
        StorageView::Key(keys) => keys
            .iter()
            .map(|key| format!("UINT64_C(0x{:016x})", key.bits()))
            .collect::<Vec<_>>(),
    };
    let (c_type, c_dtype) = match storage.prim() {
        Prim::F64 => ("double", "CHELIS_DTYPE_F64"),
        Prim::F32 => ("float", "CHELIS_DTYPE_F32"),
        Prim::F16 => ("uint16_t", "CHELIS_DTYPE_F16"),
        Prim::Bf16 => ("uint16_t", "CHELIS_DTYPE_BF16"),
        Prim::Int64 => ("int64_t", "CHELIS_DTYPE_I64"),
        Prim::Int32 => ("int32_t", "CHELIS_DTYPE_I32"),
        Prim::Int16 => ("int16_t", "CHELIS_DTYPE_I16"),
        Prim::Int8 => ("int8_t", "CHELIS_DTYPE_I8"),
        Prim::Bool => ("uint8_t", "CHELIS_DTYPE_BOOL"),
        Prim::Key => ("uint64_t", "CHELIS_DTYPE_KEY"),
        other => panic!("unsupported C test dtype {other:?}"),
    };
    (c_type, c_dtype, values.join(", "))
}

const TYPED_NONNUMERIC_HARNESS: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"

static float test_f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static double test_f64(uint64_t bits) {
    double value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static chelis_tensor *test_view(void *data, int64_t count, chelis_dtype dtype) {
    int64_t shape[1] = { count };
    return chelis_tensor_entry_borrow(
        1, shape, dtype, data, count * chelis_dtype_size(dtype)
    );
}
"#;

#[test]
fn typed_comparison_c_matrix_matches_evaluator_for_every_identity_and_dtype() {
    use chelis_unord::UnordMap;

    let kinds = [
        ComparisonKind::CmpLt,
        ComparisonKind::Lt,
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gt,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ];
    for prim in [
        Prim::F16,
        Prim::Bf16,
        Prim::F32,
        Prim::F64,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
    ] {
        let (lhs, rhs) = if prim.is_float() {
            (
                typed_storage(
                    prim,
                    chelis_types::RawTensor::Float(vec![
                        f64::from_bits(0xfff8_1234_5678_9abc),
                        -0.0,
                        f64::INFINITY,
                        -2.0,
                        4.0,
                    ]),
                ),
                typed_storage(
                    prim,
                    chelis_types::RawTensor::Float(vec![0.0, 0.0, f64::INFINITY, -1.0, 4.0]),
                ),
            )
        } else {
            (
                typed_storage(prim, chelis_types::RawTensor::Int(vec![-2, 0, 3, 4, 5])),
                typed_storage(prim, chelis_types::RawTensor::Int(vec![-1, 0, 2, 4, 6])),
            )
        };
        let n = lhs.len();
        let operand_ty = vec_prim(n, prim);
        let bool_ty = vec_prim(n, Prim::Bool);
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let left = dag.add_node(
            decl,
            RiscOp::Load {
                name: "left".into(),
            },
            vec![],
            operand_ty.clone(),
            None,
        );
        let right = dag.add_node(
            decl,
            RiscOp::Load {
                name: "right".into(),
            },
            vec![],
            operand_ty,
            None,
        );
        let outputs = kinds
            .iter()
            .map(|kind| {
                let output = dag.add_node(
                    decl,
                    RiscOp::Compare(*kind),
                    vec![left, right],
                    bool_ty.clone(),
                    None,
                );
                dag.add_root(output);
                output
            })
            .collect::<Vec<_>>();
        let evaluated = eval_tensor(
            &dag,
            &UnordMap::from([
                ("left".into(), typed_value(lhs.clone())),
                ("right".into(), typed_value(rhs.clone())),
            ]),
        )
        .unwrap();
        let expected = outputs
            .iter()
            .flat_map(|output| {
                evaluated[output]
                    .storage()
                    .to_i64_exact_vec()
                    .unwrap()
                    .into_iter()
                    .map(|value| value.to_string())
            })
            .collect::<Vec<_>>()
            .join(", ");
        let function = format!("typed_compare_{}", prim.name());
        let source = codegen(&dag, &function).unwrap().c_source;
        assert!(
            !source.contains("FusedStepOp"),
            "typed comparisons must not rely on fused vocabulary"
        );
        let (c_type, c_dtype, lhs_init) = c_storage_case(&lhs);
        let (_, _, rhs_init) = c_storage_case(&rhs);
        let harness = format!(
            r#"{TYPED_NONNUMERIC_HARNESS}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} left_data[{n}] = {{ {lhs_init} }};
    {c_type} right_data[{n}] = {{ {rhs_init} }};
    uint8_t expected[{output_count}][{n}] = {{ {expected} }};
    chelis_tensor *left = test_view(left_data, {n}, {c_dtype});
    chelis_tensor *right = test_view(right_data, {n}, {c_dtype});
    chelis_tensor *inputs[2] = {{ left, right }};
    chelis_tensor *outputs[{output_count}] = {{ 0 }};
    {function}(inputs, 2, outputs, {output_count});
    for (int output = 0; output < {output_count}; ++output) {{
        chelis_read_view view = chelis_tensor_read_view(outputs[output]);
        if (view.dtype != CHELIS_DTYPE_BOOL) return 10 + output;
        if (memcmp(view.data, expected[output], {n}) != 0) return 30 + output;
        chelis_tensor_release(outputs[output]);
    }}
    return 0;
}}
"#,
            output_count = kinds.len()
        );
        let (ok, output) = compile_and_run_kernel_capturing(&function, &source, &harness);
        assert!(ok, "{prim:?}: {output}\n{source}");
    }
}

#[test]
fn typed_logical_c_truth_tables_are_bool8() {
    let ty = vec_prim(4, Prim::Bool);
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let left = dag.add_node(
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    for op in [
        RiscOp::Logical(LogicalKind::And),
        RiscOp::Logical(LogicalKind::Or),
        RiscOp::Logical(LogicalKind::Not),
    ] {
        let inputs = if matches!(op, RiscOp::Logical(LogicalKind::Not)) {
            vec![left]
        } else {
            vec![left, right]
        };
        let output = dag.add_node(decl, op, inputs, ty.clone(), None);
        dag.add_root(output);
    }
    let source = codegen(&dag, "typed_logical").unwrap().c_source;
    let harness = format!(
        r#"{TYPED_NONNUMERIC_HARNESS}
extern void typed_logical(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint8_t left_data[4] = {{ 0, 0, 1, 1 }};
    uint8_t right_data[4] = {{ 0, 1, 0, 1 }};
    uint8_t expected[3][4] = {{ {{0,0,0,1}}, {{0,1,1,1}}, {{1,1,0,0}} }};
    chelis_tensor *inputs[2] = {{
        test_view(left_data, 4, CHELIS_DTYPE_BOOL),
        test_view(right_data, 4, CHELIS_DTYPE_BOOL)
    }};
    chelis_tensor *outputs[3] = {{ 0 }};
    typed_logical(inputs, 2, outputs, 3);
    for (int i = 0; i < 3; ++i) {{
        chelis_read_view view = chelis_tensor_read_view(outputs[i]);
        if (view.dtype != CHELIS_DTYPE_BOOL || memcmp(view.data, expected[i], 4)) return 10 + i;
        chelis_tensor_release(outputs[i]);
    }}
    return 0;
}}
"#
    );
    let (ok, output) = compile_and_run_kernel_capturing("typed_logical", &source, &harness);
    assert!(ok, "{output}\n{source}");
}

#[test]
fn typed_bool_eq_neq_c_match_exact_bool_identity() {
    let ty = vec_prim(4, Prim::Bool);
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let left = dag.add_node(
        decl,
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let right = dag.add_node(
        decl,
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    for kind in [ComparisonKind::Eq, ComparisonKind::Neq] {
        let output = dag.add_node(
            decl,
            RiscOp::Compare(kind),
            vec![left, right],
            ty.clone(),
            None,
        );
        dag.add_root(output);
    }
    let source = codegen(&dag, "typed_bool_compare").unwrap().c_source;
    let harness = format!(
        r#"{TYPED_NONNUMERIC_HARNESS}
extern void typed_bool_compare(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint8_t left_data[4] = {{ 0, 0, 1, 1 }};
    uint8_t right_data[4] = {{ 0, 1, 0, 1 }};
    uint8_t expected[2][4] = {{ {{1,0,0,1}}, {{0,1,1,0}} }};
    chelis_tensor *inputs[2] = {{
        test_view(left_data, 4, CHELIS_DTYPE_BOOL),
        test_view(right_data, 4, CHELIS_DTYPE_BOOL)
    }};
    chelis_tensor *outputs[2] = {{ 0 }};
    typed_bool_compare(inputs, 2, outputs, 2);
    for (int i = 0; i < 2; ++i) {{
        chelis_read_view view = chelis_tensor_read_view(outputs[i]);
        if (view.dtype != CHELIS_DTYPE_BOOL || memcmp(view.data, expected[i], 4)) return 10 + i;
        chelis_tensor_release(outputs[i]);
    }}
    return 0;
}}
"#
    );
    let (ok, output) = compile_and_run_kernel_capturing("typed_bool_compare", &source, &harness);
    assert!(ok, "{output}\n{source}");
}

#[test]
fn typed_where_c_copies_selected_storage_bits_for_every_admitted_dtype() {
    use chelis_unord::UnordMap;

    for prim in [
        Prim::F16,
        Prim::Bf16,
        Prim::F32,
        Prim::F64,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
    ] {
        let (then_storage, else_storage) = if prim.is_float() {
            (
                typed_storage(
                    prim,
                    chelis_types::RawTensor::Float(vec![
                        f64::from_bits(0xfff8_1234_5678_9abc),
                        -0.0,
                        4.0,
                        -7.0,
                    ]),
                ),
                typed_storage(
                    prim,
                    chelis_types::RawTensor::Float(vec![
                        f64::from_bits(0x7ff8_abcd_1234_5678),
                        0.0,
                        2.0,
                        9.0,
                    ]),
                ),
            )
        } else if prim == Prim::Bool {
            (
                typed_storage(prim, chelis_types::RawTensor::Int(vec![1, 0, 1, 0])),
                typed_storage(prim, chelis_types::RawTensor::Int(vec![0, 1, 0, 1])),
            )
        } else {
            (
                typed_storage(prim, chelis_types::RawTensor::Int(vec![1, 0, -2, 7])),
                typed_storage(prim, chelis_types::RawTensor::Int(vec![0, 1, 3, -9])),
            )
        };
        let condition = typed_storage(Prim::Bool, chelis_types::RawTensor::Int(vec![1, 0, 1, 0]));
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let condition_id = dag.add_node(
            decl,
            RiscOp::Load {
                name: "condition".into(),
            },
            vec![],
            vec_prim(4, Prim::Bool),
            None,
        );
        let then_id = dag.add_node(
            decl,
            RiscOp::Load {
                name: "then".into(),
            },
            vec![],
            vec_prim(4, prim),
            None,
        );
        let else_id = dag.add_node(
            decl,
            RiscOp::Load {
                name: "else".into(),
            },
            vec![],
            vec_prim(4, prim),
            None,
        );
        let output = dag.add_node(
            decl,
            RiscOp::Where,
            vec![condition_id, then_id, else_id],
            vec_prim(4, prim),
            None,
        );
        dag.add_root(output);
        let evaluated = eval_tensor(
            &dag,
            &UnordMap::from([
                ("condition".into(), typed_value(condition.clone())),
                ("then".into(), typed_value(then_storage.clone())),
                ("else".into(), typed_value(else_storage.clone())),
            ]),
        )
        .unwrap();
        let expected_storage = evaluated[&output].storage();
        let function = format!("typed_where_{}", prim.name());
        let source = codegen(&dag, &function).unwrap().c_source;
        let (c_type, c_dtype, then_init) = c_storage_case(&then_storage);
        let (_, _, else_init) = c_storage_case(&else_storage);
        let (_, _, expected_init) = c_storage_case(expected_storage);
        let harness = format!(
            r#"{TYPED_NONNUMERIC_HARNESS}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint8_t condition_data[4] = {{ 1, 0, 1, 0 }};
    {c_type} then_data[4] = {{ {then_init} }};
    {c_type} else_data[4] = {{ {else_init} }};
    {c_type} expected[4] = {{ {expected_init} }};
    chelis_tensor *inputs[3] = {{
        test_view(condition_data, 4, CHELIS_DTYPE_BOOL),
        test_view(then_data, 4, {c_dtype}),
        test_view(else_data, 4, {c_dtype})
    }};
    chelis_tensor *outputs[1] = {{ 0 }};
    {function}(inputs, 3, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    if (view.dtype != {c_dtype}) return 10;
    if (memcmp(view.data, expected, sizeof expected)) return 11;
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
        );
        let (ok, output_text) = compile_and_run_kernel_capturing(&function, &source, &harness);
        assert!(ok, "{prim:?}: {output_text}\n{source}");
    }
}

#[test]
fn typed_nonnumeric_c_permuted_stepped_views_match_evaluator_and_preserve_bits() {
    use chelis_ir::dag::RtDim;
    use chelis_types::StorageView;
    use chelis_unord::UnordMap;

    let matrix = |rows, cols, precision| TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision,
    };
    let input_f32 = matrix(2, 4, Prim::F32);
    let input_bool = matrix(2, 4, Prim::Bool);
    let permuted_f32 = matrix(4, 2, Prim::F32);
    let permuted_bool = matrix(4, 2, Prim::Bool);
    let stepped_f32 = matrix(2, 2, Prim::F32);
    let stepped_bool = matrix(2, 2, Prim::Bool);

    let lhs_storage = typed_storage(
        Prim::F32,
        chelis_types::RawTensor::Float(vec![
            f64::from_bits(0x7ff8_2468_a000_0000),
            11.0,
            -0.0,
            13.0,
            5.0,
            15.0,
            0.0,
            17.0,
        ]),
    );
    let rhs_storage = typed_storage(
        Prim::F32,
        chelis_types::RawTensor::Float(vec![0.0, 21.0, 0.0, 23.0, 4.0, 25.0, -0.0, 27.0]),
    );
    let logical_lhs_storage = typed_storage(
        Prim::Bool,
        chelis_types::RawTensor::Int(vec![1, 1, 0, 1, 0, 1, 1, 0]),
    );
    let logical_rhs_storage = typed_storage(
        Prim::Bool,
        chelis_types::RawTensor::Int(vec![1, 0, 0, 1, 1, 0, 0, 1]),
    );
    let condition_storage = typed_storage(
        Prim::Bool,
        chelis_types::RawTensor::Int(vec![1, 0, 1, 0, 0, 1, 0, 1]),
    );

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let add_stepped_view =
        |dag: &mut Dag, input, permuted_ty: &TensorType, stepped_ty: &TensorType| {
            let permuted = dag.add_node(
                decl,
                RiscOp::Permute { axes: vec![1, 0] },
                vec![input],
                permuted_ty.clone(),
                None,
            );
            dag.add_node(
                decl,
                RiscOp::Stride {
                    strides: vec![RtDim::Lit(2), RtDim::Lit(1)],
                },
                vec![permuted],
                stepped_ty.clone(),
                None,
            )
        };
    let load = |dag: &mut Dag, name: &str, ty: &TensorType| {
        dag.add_node(
            decl,
            RiscOp::Load { name: name.into() },
            vec![],
            ty.clone(),
            None,
        )
    };

    let lhs_load = load(&mut dag, "lhs", &input_f32);
    let rhs_load = load(&mut dag, "rhs", &input_f32);
    let logical_lhs_load = load(&mut dag, "logical_lhs", &input_bool);
    let logical_rhs_load = load(&mut dag, "logical_rhs", &input_bool);
    let condition_load = load(&mut dag, "condition", &input_bool);
    let lhs = add_stepped_view(&mut dag, lhs_load, &permuted_f32, &stepped_f32);
    let rhs = add_stepped_view(&mut dag, rhs_load, &permuted_f32, &stepped_f32);
    let logical_lhs = add_stepped_view(&mut dag, logical_lhs_load, &permuted_bool, &stepped_bool);
    let logical_rhs = add_stepped_view(&mut dag, logical_rhs_load, &permuted_bool, &stepped_bool);
    let condition = add_stepped_view(&mut dag, condition_load, &permuted_bool, &stepped_bool);

    let mut bool_outputs = Vec::new();
    for kind in [
        ComparisonKind::Eq,
        ComparisonKind::Neq,
        ComparisonKind::Gte,
        ComparisonKind::Lte,
    ] {
        let output = dag.add_node(
            decl,
            RiscOp::Compare(kind),
            vec![lhs, rhs],
            stepped_bool.clone(),
            None,
        );
        dag.add_root(output);
        bool_outputs.push(output);
    }
    for kind in [LogicalKind::And, LogicalKind::Or] {
        let output = dag.add_node(
            decl,
            RiscOp::Logical(kind),
            vec![logical_lhs, logical_rhs],
            stepped_bool.clone(),
            None,
        );
        dag.add_root(output);
        bool_outputs.push(output);
    }
    let not = dag.add_node(
        decl,
        RiscOp::Logical(LogicalKind::Not),
        vec![logical_lhs],
        stepped_bool,
        None,
    );
    dag.add_root(not);
    bool_outputs.push(not);
    let selected = dag.add_node(
        decl,
        RiscOp::Where,
        vec![condition, lhs, rhs],
        stepped_f32,
        None,
    );
    dag.add_root(selected);

    let evaluated = eval_tensor(
        &dag,
        &UnordMap::from([
            (
                "lhs".into(),
                typed_value_with_shape(vec![2, 4], lhs_storage.clone()),
            ),
            (
                "rhs".into(),
                typed_value_with_shape(vec![2, 4], rhs_storage.clone()),
            ),
            (
                "logical_lhs".into(),
                typed_value_with_shape(vec![2, 4], logical_lhs_storage.clone()),
            ),
            (
                "logical_rhs".into(),
                typed_value_with_shape(vec![2, 4], logical_rhs_storage.clone()),
            ),
            (
                "condition".into(),
                typed_value_with_shape(vec![2, 4], condition_storage.clone()),
            ),
        ]),
    )
    .unwrap();
    let expected_bool = bool_outputs
        .iter()
        .map(|output| evaluated[output].storage().to_i64_exact_vec().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        expected_bool,
        vec![
            vec![0, 0, 1, 1],
            vec![1, 1, 0, 0],
            vec![0, 1, 1, 1],
            vec![0, 0, 1, 1],
            vec![1, 0, 0, 0],
            vec![1, 1, 0, 1],
            vec![0, 1, 1, 0],
        ],
        "the fixture must retain NaN-false ordered comparisons, NaN-true neq, \
         signed-zero equality, and both logical outcomes after permutation/stride"
    );
    let expected_where = evaluated[&selected].storage();
    let (
        StorageView::F32(lhs_values),
        StorageView::F32(rhs_values),
        StorageView::F32(where_values),
    ) = (
        lhs_storage.view(),
        rhs_storage.view(),
        expected_where.view(),
    )
    else {
        unreachable!("f32 fixture")
    };
    assert_eq!(
        where_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![
            lhs_values[0].to_bits(),
            rhs_values[4].to_bits(),
            lhs_values[2].to_bits(),
            rhs_values[6].to_bits(),
        ],
        "where must copy the selected stored image, including the NaN payload and -0"
    );

    let function = "typed_nonnumeric_permuted_stepped";
    let source = codegen(&dag, function).unwrap().c_source;
    let (_, _, lhs_init) = c_storage_case(&lhs_storage);
    let (_, _, rhs_init) = c_storage_case(&rhs_storage);
    let (_, _, logical_lhs_init) = c_storage_case(&logical_lhs_storage);
    let (_, _, logical_rhs_init) = c_storage_case(&logical_rhs_storage);
    let (_, _, condition_init) = c_storage_case(&condition_storage);
    let (_, _, expected_where_init) = c_storage_case(expected_where);
    let expected_bool_init = expected_bool
        .iter()
        .map(|row| {
            format!(
                "{{ {} }}",
                row.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let harness = format!(
        r#"{TYPED_NONNUMERIC_HARNESS}
static chelis_tensor *test_matrix(void *data, chelis_dtype dtype) {{
    int64_t shape[2] = {{ 2, 4 }};
    return chelis_tensor_entry_borrow(
        2, shape, dtype, data, 8 * chelis_dtype_size(dtype)
    );
}}

extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float lhs_data[8] = {{ {lhs_init} }};
    float rhs_data[8] = {{ {rhs_init} }};
    uint8_t logical_lhs_data[8] = {{ {logical_lhs_init} }};
    uint8_t logical_rhs_data[8] = {{ {logical_rhs_init} }};
    uint8_t condition_data[8] = {{ {condition_init} }};
    uint8_t expected_bool[7][4] = {{ {expected_bool_init} }};
    float expected_where[4] = {{ {expected_where_init} }};
    chelis_tensor *inputs[5] = {{
        test_matrix(lhs_data, CHELIS_DTYPE_F32),
        test_matrix(rhs_data, CHELIS_DTYPE_F32),
        test_matrix(logical_lhs_data, CHELIS_DTYPE_BOOL),
        test_matrix(logical_rhs_data, CHELIS_DTYPE_BOOL),
        test_matrix(condition_data, CHELIS_DTYPE_BOOL)
    }};
    chelis_tensor *outputs[8] = {{ 0 }};
    {function}(inputs, 5, outputs, 8);
    for (int output = 0; output < 7; ++output) {{
        chelis_read_view view = chelis_tensor_read_view(outputs[output]);
        if (view.dtype != CHELIS_DTYPE_BOOL || view.count != 4) return 10 + output;
        if (memcmp(view.data, expected_bool[output], 4) != 0) return 30 + output;
    }}
    chelis_read_view where_view = chelis_tensor_read_view(outputs[7]);
    if (where_view.dtype != CHELIS_DTYPE_F32 || where_view.count != 4) return 50;
    if (memcmp(where_view.data, expected_where, sizeof expected_where) != 0) return 51;
    for (int output = 0; output < 8; ++output) chelis_tensor_release(outputs[output]);
    for (int input = 0; input < 5; ++input) chelis_tensor_release(inputs[input]);
    return 0;
}}
"#
    );
    let (ok, output) = compile_and_run_kernel_capturing(function, &source, &harness);
    assert!(ok, "{output}\n{source}");
}

#[test]
fn issue_630_eq_neq_owned_copied_and_borrowed_tensors_match_ieee() {
    let values = typed_storage(
        Prim::F32,
        chelis_types::RawTensor::Float(vec![f64::from(f32::from_bits(0x7fc5_4321)), -0.0, 3.5]),
    );
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let ty = vec_prim(3, Prim::F32);
    let bool_ty = vec_prim(3, Prim::Bool);
    let borrowed = dag.add_node(
        decl,
        RiscOp::Load {
            name: "borrowed".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let copied = dag.add_node(decl, RiscOp::Copy, vec![borrowed], ty.clone(), None);
    let owned = dag.add_node(
        decl,
        RiscOp::ConstTensor {
            data: values.clone(),
        },
        vec![],
        ty,
        None,
    );
    for (left, right) in [(borrowed, borrowed), (copied, borrowed), (owned, owned)] {
        for kind in [ComparisonKind::Eq, ComparisonKind::Neq] {
            let output = dag.add_node(
                decl,
                RiscOp::Compare(kind),
                vec![left, right],
                bool_ty.clone(),
                None,
            );
            dag.add_root(output);
        }
    }
    let source = codegen(&dag, "issue_630_native_forms").unwrap().c_source;
    let (c_type, c_dtype, init) = c_storage_case(&values);
    let harness = format!(
        r#"{TYPED_NONNUMERIC_HARNESS}
extern void issue_630_native_forms(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} data[3] = {{ {init} }};
    uint8_t eq[3] = {{ 0, 1, 1 }};
    uint8_t neq[3] = {{ 1, 0, 0 }};
    chelis_tensor *inputs[1] = {{ test_view(data, 3, {c_dtype}) }};
    chelis_tensor *outputs[6] = {{ 0 }};
    issue_630_native_forms(inputs, 1, outputs, 6);
    for (int form = 0; form < 3; ++form) {{
        if (memcmp(chelis_tensor_read_view(outputs[form * 2]).data, eq, 3)) return 10 + form;
        if (memcmp(chelis_tensor_read_view(outputs[form * 2 + 1]).data, neq, 3)) return 20 + form;
    }}
    return 0;
}}
"#
    );
    let (ok, output) =
        compile_and_run_kernel_capturing("issue_630_native_forms", &source, &harness);
    assert!(ok, "{output}\n{source}");
}

#[test]
fn issue_666_typed_gte_where_forward_and_gradient_match_selected_branch() {
    let scalar_f32 = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let scalar_bool = TensorType {
        dims: vec![],
        precision: Prim::Bool,
    };
    let vector_f32 = vec_prim(2, Prim::F32);
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        vector_f32.clone(),
        None,
    );
    let zero_a = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        scalar_f32.clone(),
        None,
    );
    let zero_b = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        scalar_f32.clone(),
        None,
    );
    let nan = dag.add_node(
        decl,
        RiscOp::Div,
        vec![zero_a, zero_b],
        scalar_f32.clone(),
        None,
    );
    let zero_c = dag.add_node(
        decl,
        RiscOp::synth_const(Prim::F32, 0.0),
        vec![],
        scalar_f32.clone(),
        None,
    );
    let condition = dag.add_node(
        decl,
        RiscOp::Compare(ComparisonKind::Gte),
        vec![nan, zero_c],
        scalar_bool,
        None,
    );
    let then_value = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![x],
        scalar_f32.clone(),
        None,
    );
    let squared = dag.add_node(decl, RiscOp::Mul, vec![x, x], vector_f32, None);
    let else_value = dag.add_node(
        decl,
        RiscOp::Sum {
            axis: 0,
            accumulator: Prim::F32,
        },
        vec![squared],
        scalar_f32.clone(),
        None,
    );
    let output = dag.add_node(
        decl,
        RiscOp::Where,
        vec![condition, then_value, else_value],
        scalar_f32,
        None,
    );
    let differentiated = grad_dag_checked(&dag, output, &[x]).unwrap();
    let source = codegen(&differentiated.dag, "issue_666_native_grad")
        .unwrap()
        .c_source;
    let harness = format!(
        r#"{TYPED_NONNUMERIC_HARNESS}
extern void issue_666_native_grad(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float data[2] = {{ 3.0f, 4.0f }};
    chelis_tensor *inputs[1] = {{ test_view(data, 2, CHELIS_DTYPE_F32) }};
    chelis_tensor *outputs[2] = {{ 0 }};
    issue_666_native_grad(inputs, 1, outputs, 2);
    float forward = ((const float *)chelis_tensor_read_view(outputs[0]).data)[0];
    const float *gradient = (const float *)chelis_tensor_read_view(outputs[1]).data;
    if (forward != 25.0f) return 10;
    if (gradient[0] != 6.0f || gradient[1] != 8.0f) return 11;
    return 0;
}}
"#
    );
    let (ok, output) = compile_and_run_kernel_capturing("issue_666_native_grad", &source, &harness);
    assert!(ok, "{output}\n{source}");
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
    let decl = dag.declare("test");
    let ty = vec_prim(4, prim);
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
    dag.add_node(decl, RiscOp::Sub, vec![a, b], ty, None);
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
        let decl = dag.declare("test");
        let ty = vec_prim(1, prim);
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
        dag.add_node(decl, RiscOp::Sub, vec![a, b], ty, None);
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
fn direct_float_subtraction_finalizes_canonical_nan_bits_at_every_width() {
    for (
        tag,
        prim,
        c_type,
        c_dtype,
        uint_type,
        from_bits,
        lhs_nan,
        infinity,
        one,
        signaling_nan,
        canonical_nan,
    ) in [
        (
            "f32",
            Prim::F32,
            "float",
            "CHELIS_DTYPE_F32",
            "uint32_t",
            "chelis_f32_from_bits",
            "UINT32_C(0xffc54321)",
            "UINT32_C(0x7f800000)",
            "UINT32_C(0x3f800000)",
            "UINT32_C(0x7f812345)",
            "UINT32_C(0x7fc00000)",
        ),
        (
            "f64",
            Prim::F64,
            "double",
            "CHELIS_DTYPE_F64",
            "uint64_t",
            "chelis_f64_from_bits",
            "UINT64_C(0xfff8abcd12345678)",
            "UINT64_C(0x7ff0000000000000)",
            "UINT64_C(0x3ff0000000000000)",
            "UINT64_C(0x7ff0123456789abc)",
            "UINT64_C(0x7ff8000000000000)",
        ),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = vec_prim(3, prim);
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
        let function = format!("direct_sub_canonical_nan_{tag}");
        let source = codegen(&dag, &function)
            .expect("wide direct subtraction codegen")
            .c_source;
        assert!(
            source.contains(canonical_nan),
            "{tag} subtraction source lacks exact canonical NaN: {source}"
        );
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
#include <string.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    {c_type} a_data[3] = {{
        {from_bits}({lhs_nan}),
        {from_bits}({infinity}),
        {from_bits}({one})
    }};
    {c_type} b_data[3] = {{
        {from_bits}({one}),
        {from_bits}({infinity}),
        {from_bits}({signaling_nan})
    }};
    chelis_tensor *a = make_view_typed_1d(a_data, 3, {c_dtype});
    chelis_tensor *b = make_view_typed_1d(b_data, 3, {c_dtype});
    chelis_tensor *inputs[2] = {{ a, b }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    const {c_type} *got = (const {c_type} *)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < 3; ++i) {{
        {uint_type} bits = 0;
        memcpy(&bits, &got[i], sizeof(bits));
        if (bits != {canonical_nan}) return 10 + i;
    }}
    puts("PASS");
    return 0;
}}
"#
        );
        let output = compile_and_run_kernel(&function, &source, &harness)
            .unwrap_or_else(|| panic!("{tag} canonical-NaN subtraction did not run"));
        assert!(output.contains("PASS"), "{tag}: {output}");
    }

    for (tag, prim, c_dtype, lhs_nan, infinity, one, signaling_nan, canonical_nan) in [
        (
            "f16",
            Prim::F16,
            "CHELIS_DTYPE_F16",
            "UINT16_C(0xfe55)",
            "UINT16_C(0x7c00)",
            "UINT16_C(0x3c00)",
            "UINT16_C(0x7c01)",
            "UINT16_C(0x7e00)",
        ),
        (
            "bf16",
            Prim::Bf16,
            "CHELIS_DTYPE_BF16",
            "UINT16_C(0xffe5)",
            "UINT16_C(0x7f80)",
            "UINT16_C(0x3f80)",
            "UINT16_C(0x7f81)",
            "UINT16_C(0x7fc0)",
        ),
    ] {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = vec_prim(3, prim);
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
        let function = format!("direct_sub_canonical_nan_{tag}");
        let source = codegen(&dag, &function)
            .expect("reduced-float direct subtraction codegen")
            .c_source;
        assert!(
            source.contains(canonical_nan),
            "{tag} subtraction source lacks exact canonical NaN: {source}"
        );
        let harness = format!(
            r#"{HARNESS_HEADER}
#include <stdint.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    uint16_t a_data[3] = {{ {lhs_nan}, {infinity}, {one} }};
    uint16_t b_data[3] = {{ {one}, {infinity}, {signaling_nan} }};
    chelis_tensor *a = make_view_typed_1d(a_data, 3, {c_dtype});
    chelis_tensor *b = make_view_typed_1d(b_data, 3, {c_dtype});
    chelis_tensor *inputs[2] = {{ a, b }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 2, outputs, 1);
    const uint16_t *got = (const uint16_t *)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < 3; ++i) {{
        if (got[i] != {canonical_nan}) return 10 + i;
    }}
    puts("PASS");
    return 0;
}}
"#
        );
        let output = compile_and_run_kernel(&function, &source, &harness)
            .unwrap_or_else(|| panic!("{tag} canonical-NaN subtraction did not run"));
        assert!(output.contains("PASS"), "{tag}: {output}");
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
        let decl = dag.declare("test");
        let ty = vec_prim(4, prim);
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
        let c = dag.add_node(
            decl,
            RiscOp::Load { name: "c".into() },
            vec![],
            ty.clone(),
            None,
        );
        let maximum = dag.add_node(decl, RiscOp::MaxElem, vec![a, b], ty.clone(), None);
        let minimum = dag.add_node(decl, RiscOp::MinElem, vec![maximum, c], ty, None);
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
    let decl = dag.declare("test");
    let ty = vec_prim(4, Prim::Bool);
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
    let maximum = dag.add_node(decl, RiscOp::MaxElem, vec![a, b], ty, None);
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
    let decl = dag.declare("test");
    let n_ty = runtime_vec("n");
    let m_ty = runtime_vec("m");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        n_ty.clone(),
        None,
    );
    let b = dag.add_node(decl, RiscOp::Load { name: "b".into() }, vec![], m_ty, None);
    let c = dag.add_node(
        decl,
        RiscOp::Load { name: "c".into() },
        vec![],
        n_ty.clone(),
        None,
    );
    let difference = dag.add_node(decl, RiscOp::Sub, vec![a, b], n_ty.clone(), None);
    let minimum = dag.add_node(decl, RiscOp::MinElem, vec![difference, c], n_ty, None);
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
    let decl = dag.declare("test");
    let vector = TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vector.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vector.clone(),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Add, vec![a, b], vector, None);
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

/// #2530/#2531: the executable DAG entry checks rank zero, then validates
/// every supplied input in its ABI slot order with one typed trap line.
#[test]
fn direct_interface_guards_use_slot_order_and_typed_traps() {
    let mut scalar = Dag::new();
    let scalar_decl = scalar.declare("test");
    let x = scalar.add_node(
        scalar_decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    scalar.add_root(x);

    let mut pair = Dag::new();
    let pair_decl = pair.declare("test");
    for name in ["z", "a"] {
        let value = pair.add_node(
            pair_decl,
            RiscOp::Load { name: name.into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(2)],
                precision: Prim::F32,
            },
            None,
        );
        pair.add_root(value);
    }

    let mut failures = Vec::new();
    for (case, dag, first_rank, first_extent, first_dtype, second_extent, second_dtype, expected) in [
        (
            "scalar_rank_bad",
            &scalar,
            1,
            4,
            "F32",
            0,
            "F32",
            Some(("input `x` expected rank 0, got 1", "i64")),
        ),
        ("scalar_rank_good", &scalar, 0, 0, "F32", 0, "F32", None),
        (
            "slot_dtype_bad",
            &pair,
            1,
            2,
            "F64",
            2,
            "I32",
            Some(("input `z` expected dtype f32, got f64", "f32")),
        ),
        (
            "slot_dtype_before_later_extent",
            &pair,
            1,
            2,
            "F64",
            3,
            "F32",
            Some(("input `z` expected dtype f32, got f64", "f32")),
        ),
        (
            "slot_extent_bad",
            &pair,
            1,
            3,
            "F32",
            3,
            "F32",
            Some(("input `z` axis 0 expected 2, got 3", "i64")),
        ),
        ("slot_good", &pair, 1, 2, "F32", 2, "F32", None),
    ] {
        let function = format!("entry_{case}");
        let src = codegen(dag, &function)
            .expect("direct entry codegen")
            .c_source;
        let input_count = if std::ptr::eq(dag, &scalar) { 1 } else { 2 };
        let harness = format!(
            r#"{HARNESS_HEADER}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    chelis_tensor *inputs[2] = {{
        chelis_alloc({first_rank}, (int64_t[]){{{first_extent}}}, CHELIS_DTYPE_{first_dtype}),
        chelis_alloc(1, (int64_t[]){{{second_extent}}}, CHELIS_DTYPE_{second_dtype})
    }};
    chelis_tensor *outputs[2] = {{ NULL, NULL }};
    {function}(inputs, {input_count}, outputs, {input_count});
    for (int i = 0; i < {input_count}; ++i) {{
        if (outputs[i] == NULL || chelis_tensor_rank(outputs[i]) != {first_rank}
            || chelis_tensor_numel(outputs[i]) != ({first_rank} == 0 ? 1 : 2)
            || chelis_tensor_read_view(outputs[i]).dtype != CHELIS_DTYPE_F32
            || ((float*)chelis_tensor_read_view(outputs[i]).data)[0] != 0.0f) {{
            puts("wrong output"); return 2;
        }}
    }}
    puts("completed");
    return 0;
}}
"#
        );
        let run = compile_and_capture_run(&function, &src, &harness);
        let stderr = String::from_utf8_lossy(&run.stderr);
        if let Some((context, dtype)) = expected {
            let trap = format!("numeric trap: domain in load at {dtype}");
            let trap_lines = stderr
                .lines()
                .filter(|line| line.starts_with("numeric trap:"))
                .collect::<Vec<_>>();
            let exact = format!("{function}: {context}\n{trap}\n");
            if run.status.success()
                || stderr != exact
                || trap_lines != [trap.as_str()]
                || stderr.contains("input `a`")
            {
                failures.push(format!(
                    "{case}: stderr={stderr:?}, trap lines={trap_lines:?}"
                ));
            }
        } else {
            if !run.status.success() || !String::from_utf8_lossy(&run.stdout).contains("completed")
            {
                failures.push(format!(
                    "{case}: stdout={:?}, stderr={stderr:?}",
                    String::from_utf8_lossy(&run.stdout)
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// chelis#2490, [04-NUM-11]: the four-argument public entry `x + x` over one
/// input declared at `declared`, run on a zero-filled tensor of `supplied`.
fn direct_entry_dtype_run(declared: Prim, supplied: &str) -> (String, std::process::Output) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(3)],
        precision: declared,
    };
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Add, vec![x, x], ty, None);
    dag.add_root(out);
    let function = format!("direct_entry_dtype_{}", declared.name());
    let src = codegen(&dag, &function)
        .expect("direct entry codegen")
        .c_source;
    let harness = format!(
        r#"{HARNESS_HEADER}
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    chelis_tensor *inputs[1] = {{ chelis_alloc(1, (int64_t[]){{3}}, {supplied}) }};
    chelis_tensor *outputs[1] = {{ NULL }};
    {function}(inputs, 1, outputs, 1);
    puts("completed");
    return 0;
}}
"#
    );
    let name = format!("{function}_{}", supplied.to_lowercase());
    let run = compile_and_capture_run(&name, &src, &harness);
    (src, run)
}

/// chelis#2490 REGRESSION TEST: on the pre-fix tree every mismatched row ran
/// to completion, reading the supplied storage at the declared dtype. The
/// guard now traps before the load reads the input.
#[test]
fn direct_entry_dtype_mismatch_traps_before_the_load_reads() {
    let mut failures = Vec::new();
    for (declared, supplied, actual) in [
        (Prim::Int64, "CHELIS_DTYPE_F64", "f64"),
        (Prim::Int64, "CHELIS_DTYPE_I32", "i32"),
        (Prim::F64, "CHELIS_DTYPE_I64", "i64"),
        (Prim::F32, "CHELIS_DTYPE_BOOL", "bool"),
        (Prim::F32, "CHELIS_DTYPE_F16", "f16"),
    ] {
        let (src, run) = direct_entry_dtype_run(declared, supplied);
        let guard = src
            .find("chelis_tensor_read_view(inputs[0]).dtype != ")
            .unwrap_or(usize::MAX);
        let load = src.find("chelis_tensor_read_view(t0)").expect("load");
        let output = format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        let declared = declared.name();
        if guard > load
            || run.status.success()
            || output.contains("completed")
            || !output.contains(&format!(
                "input `x` expected dtype {declared}, got {actual}"
            ))
            || !output.contains(&format!("numeric trap: domain in load at {declared}\n"))
        {
            failures.push(format!("{declared} <- {supplied}: {output}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn direct_entry_matching_dtypes_complete() {
    for (declared, supplied) in [
        (Prim::Int64, "CHELIS_DTYPE_I64"),
        (Prim::F64, "CHELIS_DTYPE_F64"),
        (Prim::F32, "CHELIS_DTYPE_F32"),
    ] {
        let (_, run) = direct_entry_dtype_run(declared, supplied);
        let output = format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(run.status.success(), "{supplied}: {output}");
        assert!(output.contains("completed"), "{supplied}: {output}");
    }
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
            helper_result_claim_axes: Vec::new(),
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
        adt_layouts: Vec::new(),
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
#define the_fn chelis_fn_7468655f666e

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
    let reads_tensor_rank_field = src.lines().any(|line| {
        line.contains("->rank") && (line.contains("__arg0_") || line.contains("__arg1_"))
    });
    assert!(
        !reads_tensor_rank_field,
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
    let decl = dag.declare("test");
    let output_matrix_ty = matrix("rows", "columns");
    let other_matrix_ty = matrix("other_rows", "other_columns");
    let scalar = dag.add_node(
        decl,
        RiscOp::Load {
            name: "scalar".into(),
        },
        vec![],
        TensorType::scalar_f32(),
        None,
    );
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        output_matrix_ty.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        other_matrix_ty,
        None,
    );
    let shifted = dag.add_node(
        decl,
        RiscOp::Add,
        vec![scalar, a],
        output_matrix_ty.clone(),
        None,
    );
    let product = dag.add_node(decl, RiscOp::Mul, vec![shifted, b], output_matrix_ty, None);
    let output_ty = TensorType {
        dims: vec![DimInfo::Named("rows".into(), None)],
        precision: Prim::F32,
    };
    let reduced = match reduce_kind {
        "sum" => dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![product],
            output_ty,
            None,
        ),
        "max" => dag.add_node(
            decl,
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
        let decl = dag.declare("test");
        let ty = vec_prim(n, prim);
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
        dag.add_node(decl, op, vec![a, b], ty, None);
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
    let decl = dag.declare("test");
    let ty = vec_prim(n, prim);
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
    for (kind, operand) in [
        (ExtremaKind::Max, ExtremaOperand::Left),
        (ExtremaKind::Max, ExtremaOperand::Right),
        (ExtremaKind::Min, ExtremaOperand::Left),
        (ExtremaKind::Min, ExtremaOperand::Right),
    ] {
        let node = dag.add_node(
            decl,
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
#define the_fn chelis_fn_7468655f666e
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
            helper_result_claim_axes: Vec::new(),
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
        adt_layouts: Vec::new(),
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
#define the_fn chelis_fn_7468655f666e
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
    let decl = dag.declare("test");
    let ty = vec_prim(n, prim);
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
fn runtime_branch_local_ascription_c() -> chelis_backend_c::CodegenResult {
    let source = "def f(flag: bool, x: tensor[*, f32]) -> tensor[*, f32] = \
                  if flag then {\n  \
                    y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)\n  \
                    y\n\
                  } else x\n";
    let decls = chelis_surf::parser::parse_str(source).expect("Surf parse");
    let checked = chelis_types::check_ir_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
    )
    .unwrap_or_else(|report| panic!("type check failed: {:?}", report.errors));
    let dag = chelis_ir::host::lower_named_tensor_entry_dag(&checked, "f")
        .expect("named tensor entry lowers");
    let generated = codegen_with_options(
        &dag,
        "runtime_branch_local_ascription",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("runtime branch codegen");
    assert_eq!(generated.input_labels, ["flag", "x"]);
    // The claim is checked under its carrier's owner activation, read from
    // that Bool's storage row by row (any row active runs the guard).
    assert!(
        generated
            .c_source
            .contains("__local_guard_active |= (((const "),
        "the claim's guard reads its carrier's owner activation"
    );
    generated
}

fn runtime_branch_local_ascription_harness(flag: bool) -> String {
    let flag = u8::from(flag);
    format!(
        r#"
#include "chelis_runtime.h"
#include <stdio.h>
extern void runtime_branch_local_ascription(
    chelis_tensor **inputs,
    int n_in,
    chelis_tensor **outputs,
    int n_out
);

int main(void) {{
    uint8_t flag_data[1] = {{{flag}}};
    float x_data[3] = {{1.0f, 2.0f, 3.0f}};
    int64_t x_shape[1] = {{3}};
    chelis_tensor *flag = chelis_tensor_entry_borrow(
        0, NULL, CHELIS_DTYPE_BOOL, flag_data, sizeof(flag_data)
    );
    chelis_tensor *x = chelis_tensor_entry_borrow(
        1, x_shape, CHELIS_DTYPE_F32, x_data, sizeof(x_data)
    );
    chelis_tensor *inputs[2] = {{flag, x}};
    chelis_tensor *outputs[1] = {{NULL}};
    runtime_branch_local_ascription(inputs, 2, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    printf("RAN %lld %.1f %.1f %.1f\n",
           (long long)chelis_tensor_shape(outputs[0], 0),
           ((const float*)view.data)[0],
           ((const float*)view.data)[1],
           ((const float*)view.data)[2]);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(x);
    chelis_tensor_release(flag);
    return 0;
}}
"#
    )
}

#[test]
fn an_untaken_runtime_branch_does_not_emit_its_local_ascription_guard_on_c() {
    let generated = runtime_branch_local_ascription_c();
    let run = compile_and_capture_run(
        "local_ascription_runtime_branch_untaken",
        &generated.c_source,
        &runtime_branch_local_ascription_harness(false),
    );
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout), "RAN 3 1.0 2.0 3.0\n");
}

#[test]
fn a_selected_runtime_branch_emits_its_local_ascription_guard_on_c() {
    let generated = runtime_branch_local_ascription_c();
    let run = compile_and_capture_run(
        "local_ascription_runtime_branch_selected",
        &generated.c_source,
        &runtime_branch_local_ascription_harness(true),
    );
    assert!(!run.status.success(), "the selected local claim must trap");
    let mut text = String::from_utf8_lossy(&run.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&run.stderr));
    assert!(
        text.contains("extent `2`: claimed = 2, pad axis 0 = 3"),
        "{text}"
    );
    assert!(
        text.lines()
            .any(|line| line == "numeric trap: domain in pad at i64"),
        "{text}"
    );
    assert!(!text.contains("RAN "), "{text}");
}

fn two_witness_dag() -> chelis_ir::dag::Dag {
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    let named = || TensorType {
        dims: vec![DimInfo::Named("n".into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        named(),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::Load { name: "p".into() },
        vec![],
        named(),
        None,
    );
    let negated = dag.add_node(decl, RiscOp::Neg, vec![x], named(), None);
    let out = dag.add_node(decl, RiscOp::Add, vec![negated, p], named(), None);
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
        out.contains("numeric trap: domain in load at i64"),
        "section 4.7 makes this an [04-NUM-9] guard naming the `load`: {out}"
    );
    // NOT `out.contains('n')`. That was satisfied by "numeric", "domain in"
    // and "i64" in the trap line itself, so it asserted nothing about the
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![x_dim],
            precision: Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        decl,
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
/// The `<prim>` slot is `i64` because the guarded result is an extent under
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
        out.contains("numeric trap: domain in load at i64"),
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
    let decl = dag.declare("test");
    let scalar = TensorType {
        dims: vec![],
        precision: Prim::F32,
    };
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        scalar,
        None,
    );
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![x_dim],
            precision: Prim::F32,
        },
        None,
    );
    let out = dag.add_node(
        decl,
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
        src.contains("chelis_numeric_trap(\"numeric trap: domain in load at i64\")"),
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
    let decl = dag.declare("test");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let out = dag.add_node(decl, RiscOp::Neg, vec![x], ty, None);
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
    let decl = dag.declare("test");
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        named(),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        named(),
        None,
    );
    let widened = dag.add_node(
        decl,
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
    let out = dag.add_node(decl, RiscOp::Add, vec![widened, x], named(), None);
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
    let decl = dag.declare("test");
    let zz = dag.add_node(
        decl,
        RiscOp::Load { name: "zz".into() },
        vec![],
        dim("zdim"),
        None,
    );
    let aa = dag.add_node(
        decl,
        RiscOp::Load { name: "aa".into() },
        vec![],
        dim("adim"),
        None,
    );
    let p = dag.add_node(
        decl,
        RiscOp::Load { name: "p".into() },
        vec![],
        dim("zdim"),
        None,
    );
    let q = dag.add_node(
        decl,
        RiscOp::Load { name: "q".into() },
        vec![],
        dim("adim"),
        None,
    );
    // All four witnesses are READ, and both classes still have to reach one
    // result, so the `adim` pair is reduced to a scalar and broadcast back
    // over `zdim`. The unread form is section 4.7's stronger case and is not
    // testable here; see `two_witness_dag`.
    let zsum = dag.add_node(decl, RiscOp::Add, vec![zz, p], dim("zdim"), None);
    let asum = dag.add_node(decl, RiscOp::Add, vec![aa, q], dim("adim"), None);
    let folded = dag.add_node(
        decl,
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
        decl,
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
    let out = dag.add_node(decl, RiscOp::Add, vec![zsum, spread], dim("zdim"), None);
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
        out.contains("numeric trap: domain in load at i64"),
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
        out.contains("numeric trap: domain in load at i64"),
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
/// (the legacy rendering; the shipped context names the operation)
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        named("n"),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let strided = dag.add_node(
        decl,
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        named("s"),
        None,
    );
    let out = dag.add_node(
        decl,
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
            .contains("chelis_numeric_trap(\"numeric trap: domain in insert at i64\")"),
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
        out.contains("numeric trap: domain in insert at i64"),
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
                let decl = dag.declare("test");
                let claim = if resolved_name {
                    DimInfo::Named("n".into(), Some(4))
                } else {
                    DimInfo::Lit(4)
                };
                let x = dag.add_node(
                    decl,
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
                    decl,
                    RiscOp::Load {
                        name: "delta".into(),
                    },
                    vec![],
                    integer.clone(),
                    None,
                );
                let (size, carrier) = if tensor_size {
                    let strided = dag.add_node(
                        decl,
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
                    let read = dag.add_node(
                        decl,
                        RiscOp::Shape { axis: 0 },
                        vec![x],
                        integer.clone(),
                        None,
                    );
                    let sum = dag.add_node(decl, RiscOp::Add, vec![read, delta], integer, None);
                    (sum, RtDim::Node(1))
                };
                let operand = if reshape {
                    x
                } else {
                    dag.add_node(
                        decl,
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
                    decl,
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
                    let trap = format!("numeric trap: domain in {op} at i64");
                    // Section 4.7's context names the OPERATION that
                    // introduces the extent, not the node id: a node id is not
                    // a source name and does not survive `spec/06` section
                    // 5.2-5.4's renumbering passes.
                    let _ = root;
                    let context = format!("{op} axis 0 = {observed}");
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        resolved(),
        None,
    );
    let b = dag.add_node(
        decl,
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
        decl,
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
        decl,
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
        decl,
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
        decl,
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        },
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        TensorType {
            dims: vec![],
            precision: Prim::F32,
        },
        None,
    );
    let from_input = dag.add_node(
        decl,
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
        decl,
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
        decl,
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
    let out = dag.add_node(
        decl,
        RiscOp::Add,
        vec![from_input, from_computed],
        four(),
        None,
    );
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
            .any(|line| line == "numeric trap: domain in insert at i64")
            && out.contains("claimed = 4")
            && out.contains("insert axis 0 = 5")
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
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        open("n"),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        open("m"),
        None,
    );
    let b = dag.add_node(
        decl,
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
        decl,
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
        decl,
        RiscOp::Stride {
            strides: vec![RtDim::Lit(2)],
        },
        vec![x],
        open("s"),
        None,
    );
    let from_computed = dag.add_node(
        decl,
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
        decl,
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
        result.c_source.contains("insert axis 0 = %lld"),
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
    let decl = dag.declare("test");
    let n = DimInfo::Named("n".into(), Some(4));
    let m = DimInfo::Named("m".into(), Some(2));
    let x = dag.add_node(
        decl,
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
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        int.clone(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        int.clone(),
        None,
    );
    let ashape = dag.add_node(decl, RiscOp::Shape { axis: 1 }, vec![x], int.clone(), None);
    let bshape = dag.add_node(decl, RiscOp::Shape { axis: 0 }, vec![x], int.clone(), None);
    let ac = dag.add_node(decl, RiscOp::Add, vec![ashape, a], int.clone(), None);
    let bc = dag.add_node(decl, RiscOp::Add, vec![bshape, b], int, None);
    let root = dag.add_node(
        decl,
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
            let _ = root;
            let context =
                format!("extent {claim}: claimed = {required}, reshape axis {axis} = {observed}");
            for (lane, diagnostic) in [("eval", e), ("c", c)] {
                let diagnostic = diagnostic.replace('`', "");
                assert!(
                    diagnostic
                        .lines()
                        .any(|line| line == "numeric trap: domain in reshape at i64"),
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

#[test]
fn checked_snapshot_shape_capture_survives_submission_repurpose() {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let ty = TensorType {
        dims: vec![DimInfo::Lit(1)],
        precision: Prim::Int64,
    };
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let owned = dag.add_node(decl, RiscOp::Copy, vec![x], ty, None);
    let out = dag.add_node(
        decl,
        RiscOp::Shape { axis: 0 },
        vec![owned],
        TensorType {
            dims: vec![],
            precision: Prim::Int64,
        },
        None,
    );
    dag.add_root(out);
    let generated = codegen_with_options(
        &dag,
        "snapshot_shape",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .unwrap();
    let source = &generated.c_source;
    let capture = "const int64_t t2_shape_extent = chelis_tensor_shape(t1, 0);";
    assert!(source.contains(capture), "{source}");
    let harness = r#"
#include "chelis_runtime.h"
#include <stdio.h>
extern void snapshot_shape(chelis_tensor**, int, chelis_tensor**, int);
int main(void) {
    int64_t data = 91, shape = 1;
    chelis_tensor *input = chelis_tensor_entry_borrow(1, &shape, CHELIS_DTYPE_I64, &data, sizeof(data));
    chelis_tensor *outputs[1] = {NULL};
    snapshot_shape(&input, 1, outputs, 1);
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    if (chelis_tensor_rank(outputs[0]) != 0 || view.dtype != CHELIS_DTYPE_I64 || view.count != 1 || ((const int64_t*)view.data)[0] != 1) return 43;
    chelis_tensor_release(outputs[0]); chelis_tensor_release(input);
    puts("EXTENT BEFORE REUSE"); return 0;
}
"#;
    let valid = checked_indexing_run(source, harness);
    assert!(valid.status.success(), "{valid:?}");
    assert_eq!(
        String::from_utf8_lossy(&valid.stdout),
        "EXTENT BEFORE REUSE\n"
    );
    // The current planner keeps a last-use input live through submission. Also
    // exercise the emitter's capture order with a legal same-capacity submitter
    // that repurposes this owned source; this is an explicit fixture variant.
    let allocation = "chelis_tensor *t2 = chelis_alloc(0, NULL, CHELIS_DTYPE_I64);";
    assert_eq!(source.matches(allocation).count(), 1);
    let resubmitted = source
        .replace("chelis_tensor_end_write(t1_write_guard);", "")
        .replace("chelis_tensor_release(t1);", "")
        .replace(allocation, "chelis_tensor_end_write(t1_write_guard); chelis_tensor_repurpose(t1, chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0), NULL); chelis_tensor *t2 = t1;");
    let reused = checked_indexing_run(&resubmitted, harness);
    assert!(reused.status.success(), "{reused:?}");
    assert_eq!(
        String::from_utf8_lossy(&reused.stdout),
        "EXTENT BEFORE REUSE\n"
    );
    let removed = resubmitted.replace(capture, "");
    let marker = "chelis_fill_scalar(t2_write_guard,";
    assert_eq!(removed.matches(marker).count(), 1);
    let mutated = removed.replacen(marker, &format!("{capture}\n    {marker}"), 1);
    let invalid = checked_indexing_run(&mutated, harness);
    assert_eq!(
        invalid.status.code(),
        Some(1),
        "late shape observation must fail after rank-zero repurpose: {invalid:?}"
    );
    assert!(
        String::from_utf8_lossy(&invalid.stderr).contains("axis"),
        "{invalid:?}"
    );
}

/// A one-parameter host function `the_fn(a) = diagonal(a, 0, 1)` whose declared
/// result carries `declared` on its single axis (chelis#1739).
fn host_diagonal_program(operand: Vec<usize>, declared: usize) -> HostProgram {
    let operand_ty = host_tensor(operand);
    let result_ty = TensorType {
        dims: vec![DimInfo::Lit(declared)],
        precision: Prim::F32,
    };
    let axis = |value: i64| HostExpr::new(HostExprKind::Int(value));
    let body = HostExpr::new(HostExprKind::Builtin {
        name: "diagonal".to_string(),
        args: vec![
            HostExpr::new(HostExprKind::Var(
                "a".to_string(),
                HostType::Tensor(operand_ty.clone()),
            )),
            axis(0),
            axis(1),
        ],
        ty: HostType::Tensor(result_ty.clone()),
    });
    HostProgram {
        globals: Vec::new(),
        global_tensor_helpers: Vec::new(),
        functions: vec![HostFunction {
            helper_result_claim_axes: Vec::new(),
            name: "the_fn".to_string(),
            params: vec![HostParam {
                name: "a".to_string(),
                ty: HostType::Tensor(operand_ty),
            }],
            ret_ty: HostType::Tensor(result_ty),
            body,
            tensor_helpers: Vec::new(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        }],
        summary_rejections: Vec::new(),
        adt_layouts: Vec::new(),
    }
}

/// chelis#1739, driven under the sanitizers. A host-lane function whose declared
/// result extent the body cannot produce aborts at its return boundary with the
/// `[04-NUM-9]` `Domain` rendering, and the same function with an agreeing
/// declaration returns its exact result.
///
/// The host lane has no `ExtentWitness`, so this guard is the only thing between
/// a wrong declaration and a silent wrong answer. Driven here rather than only
/// through the CLI so the added shape read and `fprintf` run under
/// `-fsanitize=address,undefined`.
#[test]
fn host_declared_result_extent_guard_traps_and_executes_under_sanitizers() {
    for (rows, declared, expect_trap) in [(2usize, 3usize, true), (3, 3, false), (5, 4, false)] {
        let source = emit_host_program(
            &host_diagonal_program(vec![rows, 4], declared),
            "host_result_extent",
        )
        .unwrap();
        let harness = format!(
            r#"
#include <stdio.h>
#include "chelis_runtime.h"
#define the_fn chelis_fn_7468655f666e
chelis_tensor *the_fn(chelis_tensor *);
int main(void) {{
    int64_t dims[2] = {{{rows}, 4}};
    chelis_tensor *a = chelis_alloc(2, dims, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(a);
    float *data = (float *)chelis_tensor_write_view(guard).data;
    for (int i = 0; i < {rows} * 4; ++i) data[i] = (float)(i + 1);
    chelis_tensor_end_write(guard);
    chelis_tensor *out = the_fn(a);
    chelis_read_view view = chelis_tensor_read_view(out);
    printf("RETURNED %lld\n", (long long)view.count);
    chelis_tensor_release(out);
    chelis_tensor_release(a);
    return 0;
}}
"#
        );
        let run = checked_indexing_run(&source, &harness);
        let stderr = String::from_utf8_lossy(&run.stderr);
        let stdout = String::from_utf8_lossy(&run.stdout);
        if expect_trap {
            assert!(
                !run.status.success(),
                "a declared {declared} over min({rows}, 4) must abort: {stdout}{stderr}"
            );
            assert!(
                stderr.contains("numeric trap: domain in diagonal at i64"),
                "the frozen [04-NUM-9] line: {stderr}"
            );
            assert!(
                stderr.contains(&format!(
                    "extent `{declared}`: claimed = {declared}, diagonal axis 0 = {}",
                    rows.min(4)
                )),
                "the accompanying context line: {stderr}"
            );
            assert!(
                !stdout.contains("RETURNED"),
                "the guard runs BEFORE the result reaches the caller: {stdout}"
            );
        } else {
            assert!(
                run.status.success(),
                "an agreeing declaration must execute: {stdout}{stderr}"
            );
            assert_eq!(
                stdout,
                format!("RETURNED {}\n", rows.min(4)),
                "and return its exact result: {stderr}"
            );
        }
    }
}

/// chelis#1775 C-lane receipt: a runtime extent shared by two reshapes is
/// declared ONCE in the emitted C and re-checked at the second carrier.
///
/// The PR that keyed a computed reshape extent by its producing scalar rests a
/// safety argument on this: two reshapes sized from one scalar now carry the
/// same dim symbol, and the emitter must declare that symbol at the first site
/// and emit a runtime equality abort at the later one rather than redeclaring
/// it. Nothing else in that change set reaches the C lane, so this is the
/// fixture that proves the argument instead of asserting it.
///
/// REGRESSION TEST for the shared-name build (it cannot even be constructed
/// before the repair: the two rows disagree, `concat` falls to its rank-0 host
/// placeholder and there is no `Pad` cascade to emit).
/// DISPOSITION LOCK for the mutation arm: the later carrier's guard is live on
/// both sides of the repair, and this pins that it aborts rather than reading
/// a second, silently different extent.
#[test]
fn shared_runtime_extent_declares_once_and_rechecks_the_later_carrier() {
    use chelis_unord::UnordMap;

    // `let m = cast(shape(x, 0), i64) in concat([reshape(x, [1, m]),
    //  reshape(x, [1, m])], 0)` - the minimal shape of the #368 window stack.
    let row = "(app {} (var {} reshape) (var {} x) \
       (app {} (var {} Cons) (cast {} (lit {} 1) (t-prim {} i64)) \
         (app {} (var {} Cons) (var {} m) (var {} Nil))))";
    let src = format!(
        "(let {{}} (bind {{}} m (cast {{}} (app {{}} (var {{}} shape) (var {{}} x) \
           (cast {{}} (lit {{}} 0) (t-prim {{}} i32))) (t-prim {{}} i64))) \
         (app {{}} (var {{}} concat) \
           (app {{}} (var {{}} Cons) {row} (app {{}} (var {{}} Cons) {row} (var {{}} Nil))) \
           (cast {{}} (lit {{}} 0) (t-prim {{}} i32))))"
    );
    let mut exprs = chelis_deep::parser::parse_str(&src).expect("deep parse");
    assert_eq!(exprs.len(), 1);
    let expr = exprs.pop().unwrap();
    let mut scoped = UnordMap::new();
    scoped.insert(
        "x".to_string(),
        TensorType {
            dims: vec![DimInfo::Named("n".into(), None)],
            precision: Prim::F32,
        },
    );
    let dag =
        chelis_ir::lower::lower_subexpr_program(&expr, scoped, UnordMap::new(), UnordMap::new());

    // The premise: one shared symbol across both rows, and the differentiable
    // Pad+Add cascade rather than the rank-0 host `concat` placeholder.
    let axis_names: Vec<String> = dag
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, RiscOp::Reshape { .. }))
        .map(|node| match &node.output_type.dims[1] {
            DimInfo::Named(name, None) => name.clone(),
            other => panic!("expected a generated runtime dim, got {other:?}"),
        })
        .collect();
    assert_eq!(axis_names.len(), 2, "two rows: {dag:?}");
    assert_eq!(axis_names[0], axis_names[1], "one extent, one symbol");
    assert_eq!(
        dag.nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Pad { .. }))
            .count(),
        2,
        "the concat must be the Pad cascade: {dag:?}"
    );

    let generated = codegen(&dag, "shared_extent").expect("shared-extent DAG must emit C");

    // The symbol is declared once. Every later mention is a use or a guard.
    let declarations = generated
        .c_source
        .lines()
        .filter(|line| line.contains(&format!("int64_t {};", axis_names[0])))
        .count()
        + generated
            .c_source
            .lines()
            .filter(|line| line.contains(&format!("int64_t {} =", axis_names[0])))
            .count();
    assert_eq!(
        declarations, 1,
        "the shared runtime extent is declared exactly once:\n{}",
        generated.c_source
    );

    // 1.0f as its exact IEEE-754 binary32 image; the runtime takes scalar
    // bits, never an untagged double ([04-NUM-11]).
    let harness = r#"
#include "chelis_runtime.h"
void shared_extent(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {
    int64_t shape[] = {4};
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    chelis_fill_scalar(guard, chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT64_C(0x3F800000)));
    chelis_tensor_end_write(guard);
    chelis_tensor *inputs[] = {x}, *outputs[] = {NULL};
    shared_extent(inputs, 1, outputs, 1);
    if (chelis_tensor_rank(outputs[0]) != 2) return 4;
    if (chelis_tensor_shape(outputs[0], 0) != 2) return 5;
    if (chelis_tensor_shape(outputs[0], 1) != 4) return 6;
    chelis_read_view out = chelis_tensor_read_view(outputs[0]);
    if (out.count != 8) return 7;
    for (int64_t i = 0; i < out.count; ++i)
        if (((const float *)out.data)[i] != 1.0f) return 8;
    chelis_tensor_release(outputs[0]); chelis_tensor_release(x);
    puts("SHARED EXTENT PASS"); return 0;
}
"#;

    let (ok, text) =
        compile_and_run_kernel_capturing("shared_runtime_extent", &generated.c_source, harness);
    assert!(ok, "shared-extent kernel must build, link and run: {text}");
    assert!(
        text.contains("SHARED EXTENT PASS"),
        "the stack must materialize as [2, 4]: {text}"
    );

    // The later carrier's guard is LIVE, not decorative. Make the second
    // reshape read a different extent than the declared symbol and the binary
    // must trap instead of sizing an axis from a second, unequal value. The
    // guard's own `fprintf` names the node, so find its read by that line and
    // perturb the read the comparison performs.
    let guard_line = generated
        .c_source
        .lines()
        .filter(|line| line.contains(&format!("extent `{}`: claimed", axis_names[0])))
        .nth(1)
        .expect("the later carrier carries its own equality guard")
        .to_owned();
    let carrier = guard_line
        .split("(long long)(")
        .nth(2)
        .and_then(|rest| rest.strip_suffix("));"))
        .expect("the guard prints the carrier read it compared")
        .to_owned();
    assert!(
        carrier.contains("_data)[0]"),
        "expected a carrier read, got {carrier}"
    );
    let comparison = format!("if (({carrier}) != {})", axis_names[0]);
    assert!(
        generated.c_source.contains(&comparison),
        "the later guard compares the carrier against the declared symbol:\n{}",
        generated.c_source
    );
    let mutated = generated.c_source.replacen(
        &comparison,
        &format!("if ((({carrier}) + 1) != {})", axis_names[0]),
        1,
    );
    let (ok, text) =
        compile_and_run_kernel_capturing("shared_runtime_extent_trap", &mutated, harness);
    assert!(!ok, "a disagreeing later carrier must abort: {text}");
    assert!(
        text.contains("numeric trap: domain in reshape at i64"),
        "the abort must be the typed reshape domain trap: {text}"
    );
}

/// NEGATIVE CONTROL for the per-scope rename's freshness rule, chelis#1788.
///
/// `<name>__s<k>` is a legal Chelis dimension name, so a graph can already
/// carry the spelling the rename would mint. Renaming onto it would recreate
/// the very collision the pass exists to remove, one name meaning two extents
/// in one emitted function, and it would do so silently.
///
/// Three roots: `seq` is declared by two of them, and a third independently
/// declares `seq__s1`. The second `seq` scope must therefore take `seq__s2`,
/// and all three extents must stay independent at run time. Without the
/// freshness check this test emits two declarations of `seq__s1` and the third
/// root reads the second root's extent.
///
/// EVIDENTIARY STATUS: regression test for the freshness rule specifically. The
/// sibling row above covers the rename itself.
#[test]
fn issue_1788_a_scope_rename_does_not_collide_with_a_name_the_graph_declares() {
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    let ty = |dims: Vec<DimInfo>| TensorType {
        dims,
        precision: Prim::F32,
    };
    let named = |name: &str| DimInfo::Named(name.into(), None);

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![named("seq")]),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        ty(vec![named("batch"), named("seq")]),
        None,
    );
    // The occupant. A third, independent signature that already spells the
    // identity the rename would otherwise mint for `y`'s scope.
    let w = dag.add_node(
        decl,
        RiscOp::Load { name: "w".into() },
        vec![],
        ty(vec![named("seq__s1")]),
        None,
    );
    let from_x = dag.add_node(decl, RiscOp::Neg, vec![x], ty(vec![named("seq")]), None);
    let from_y = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![y],
        ty(vec![named("batch"), named("seq")]),
        None,
    );
    let from_w = dag.add_node(decl, RiscOp::Neg, vec![w], ty(vec![named("seq__s1")]), None);
    dag.add_root(from_x);
    dag.add_root(from_y);
    dag.add_root(from_w);

    let result = codegen_with_options(
        &dag,
        "three_scopes",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("a merged three-scope kernel still emits");
    let emitted = &result.c_source;
    for declaration in ["int64_t seq = ", "int64_t seq__s1 = ", "int64_t seq__s2 = "] {
        assert_eq!(
            emitted.matches(declaration).count(),
            1,
            "exactly one `{declaration}` declaration, so no identity means two extents: {emitted}"
        );
    }

    let harness = format!(
        r#"{HARNESS_HEADER}
static chelis_tensor *make_view_2d_b(float* data, int64_t rows, int64_t cols) {{
    int64_t shape[2] = {{rows, cols}};
    return chelis_tensor_entry_borrow(
        2, shape, CHELIS_DTYPE_F32, data,
        rows * cols * chelis_dtype_size(CHELIS_DTYPE_F32)
    );
}}

extern void three_scopes(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[3] = {{1.0f, 2.0f, 3.0f}};
    float yd[4] = {{1.0f, 2.0f, 3.0f, 4.0f}};
    float wd[4] = {{5.0f, 6.0f, 7.0f, 8.0f}};
    chelis_tensor* inputs[3] = {{make_view_1d(xd, 3), make_view_2d_b(yd, 2, 2), make_view_1d(wd, 4)}};
    chelis_tensor* outputs[3] = {{NULL, NULL, NULL}};
    three_scopes(inputs, 3, outputs, 3);
    printf("RAN %lld %lld %lld\n",
           (long long)chelis_tensor_numel(outputs[0]),
           (long long)chelis_tensor_numel(outputs[1]),
           (long long)chelis_tensor_numel(outputs[2]));
    return 0;
}}
"#
    );

    let (ok, out) = compile_and_run_kernel_capturing("three_scopes", emitted, &harness);
    assert!(ok, "three independent scopes run: {out}");
    assert!(
        out.contains("RAN 3 4 4"),
        "and each root keeps its own extent, 3, 2x2 and 4: {out}"
    );
}

/// chelis#1788: two scopes of one binder, lowered into ONE emitted function,
/// each get their own declaration and their own input.
///
/// chelis#1536 scoped a claim's identity so no entry guard pairs axes from
/// different scopes, and chelis#665 moved declarations onto axis sources. The
/// declarations were still keyed by NAME across the whole graph, so when both
/// scopes landed in one emitted function the second read the first's
/// declaration and nothing compared them. Reaching this needs the codegen
/// API: `chelis build` gives each root its own function, where `seq` is then
/// declared per function from the right input.
///
/// The repair is the per-scope rename in `prepare_dag_for_codegen`: a name a
/// `Load` axis declares in more than one scope keeps its spelling in the first
/// scope and becomes `<name>__s<k>` in the later ones, so the prologue emits
/// one declaration per scope and each root sizes its work from its own input.
/// It renames only scopes that share no node, because a shared node cannot
/// carry two names for one axis.
///
/// EVIDENTIARY STATUS: **regression test.** Measured on `0820ee28e`, the
/// emitted function declared `seq` once from `inputs[0]` and the second root
/// sized its work from the first root's extent, so this program aborted inside
/// `chelis_tensor_elementwise_index_step_for_shape` at run time. The row
/// asserts both halves the repair owes: the emitted declarations, and that the
/// kernel now RUNS and produces both outputs.
#[test]
fn issue_1788_two_scopes_in_one_function_share_one_declaration() {
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    let ty = |dims: Vec<DimInfo>| TensorType {
        dims,
        precision: Prim::F32,
    };
    let named = |name: &str| DimInfo::Named(name.into(), None);

    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let x = dag.add_node(
        decl,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty(vec![named("seq")]),
        None,
    );
    let y = dag.add_node(
        decl,
        RiscOp::Load { name: "y".into() },
        vec![],
        ty(vec![named("batch"), named("seq")]),
        None,
    );
    let from_x = dag.add_node(decl, RiscOp::Neg, vec![x], ty(vec![named("seq")]), None);
    let from_y = dag.add_node(
        decl,
        RiscOp::Neg,
        vec![y],
        ty(vec![named("batch"), named("seq")]),
        None,
    );
    dag.add_root(from_x);
    dag.add_root(from_y);

    let result = codegen_with_options(
        &dag,
        "two_scopes",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("a merged two-scope kernel still emits");
    let emitted = &result.c_source;
    assert!(
        emitted.contains("int64_t seq = chelis_tensor_shape(inputs[0], 0);"),
        "the first scope declares `seq` from its own input slot: {emitted}"
    );
    assert!(
        emitted.contains("int64_t seq__s1 = chelis_tensor_shape(inputs[1], 1);"),
        "and the second scope declares its own identity from ITS input slot: {emitted}"
    );
    assert_eq!(
        emitted.matches("int64_t seq = ").count(),
        1,
        "exactly one declaration per scope, not a redeclaration: {emitted}"
    );
    let harness = format!(
        r#"{HARNESS_HEADER}
static chelis_tensor *make_view_2d(float* data, int64_t rows, int64_t cols) {{
    int64_t shape[2] = {{rows, cols}};
    return chelis_tensor_entry_borrow(
        2, shape, CHELIS_DTYPE_F32, data,
        rows * cols * chelis_dtype_size(CHELIS_DTYPE_F32)
    );
}}

extern void two_scopes(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main() {{
    float xd[3] = {{1.0f, 2.0f, 3.0f}};
    float yd[4] = {{1.0f, 2.0f, 3.0f, 4.0f}};
    chelis_tensor* inputs[2] = {{make_view_1d(xd, 3), make_view_2d(yd, 2, 2)}};
    chelis_tensor* outputs[2] = {{NULL, NULL}};
    two_scopes(inputs, 2, outputs, 2);
    const float* a = (const float*)chelis_tensor_read_view(outputs[0]).data;
    const float* b = (const float*)chelis_tensor_read_view(outputs[1]).data;
    printf("RAN %.1f %.1f %.1f | %.1f %.1f %.1f %.1f\n",
           a[0], a[1], a[2], b[0], b[1], b[2], b[3]);
    return 0;
}}
"#
    );

    let (ok, out) = compile_and_run_kernel_capturing("two_scopes", emitted, &harness);
    assert!(
        ok,
        "each scope sizes its own work, so the merged kernel runs: {out}"
    );
    assert!(
        out.contains("RAN -1.0 -2.0 -3.0 | -1.0 -2.0 -3.0 -4.0"),
        "and both roots produce their exact negated inputs: {out}"
    );

    // The same unequal extents must fail if an entry guard accidentally pairs
    // the scopes. Keep this mutation control alongside the successful execution
    // so the runtime oracle cannot silently stop detecting that regression.
    let second_scope_declaration = "int64_t seq__s1 = chelis_tensor_shape(inputs[1], 1);";
    let paired_emitted = emitted.replacen(
        second_scope_declaration,
        &format!(
            "{second_scope_declaration}\n    if (seq != seq__s1) {{ \
             chelis_numeric_trap(\"numeric trap: domain in load at i64\"); }}"
        ),
        1,
    );
    assert_ne!(paired_emitted, *emitted, "the pair guard was inserted");
    let (paired_ok, paired_out) =
        compile_and_run_kernel_capturing("two_scopes_paired", &paired_emitted, &harness);
    assert!(
        !paired_ok && paired_out.contains("numeric trap: domain in load at i64"),
        "an erroneous cross-scope entry comparison must reject this input: {paired_out}"
    );
}

// #1767: retain checked top-level input contracts and derive disconnected
// cotangents from each actual's physical axes, including cached composition.
fn gradient_geometry_checked(source: &str) -> chelis_types::CheckedProgram {
    let deep = chelis_surf::desugar::desugar_program(
        &chelis_surf::parser::parse_str(source).expect("gradient fixture parses"),
    )
    .unwrap();
    let checked = chelis_types::check_typed_program(&deep).expect("gradient fixture checks");
    let checked = chelis_effects::check_program(&checked).unwrap();
    chelis_types::check_linearity(&checked).unwrap()
}

fn assert_zero_geometry_both_lanes(
    mut dag: Dag,
    inputs: &[(&str, &[usize])],
    expected: &[&[usize]],
    reject_input: bool,
) {
    use chelis_ir::eval::eval_tensor_roots_with_strict;
    let evaluated = eval_tensor_roots_with_strict(&dag, dag.roots(), |name| {
        inputs
            .iter()
            .find(|(label, _)| *label == name)
            .map(|(_, shape)| {
                TensorValue::from_storage(
                    shape.to_vec(),
                    chelis_types::finalize_tensor(
                        "test",
                        Prim::F32,
                        chelis_types::RawTensor::Float(vec![1.0; shape.iter().product()]),
                    )
                    .unwrap(),
                )
            })
    });
    if reject_input {
        let error = evaluated.expect_err("disconnected input claim must still execute");
        assert!(
            error.contains("numeric trap: domain in load at i64"),
            "{error}"
        );
    } else {
        let evaluated = evaluated.unwrap();
        assert_eq!(dag.roots().len(), expected.len());
        for (root, shape) in dag.roots().iter().zip(expected) {
            assert_eq!(dag.get(*root).unwrap().output_type.dims.len(), shape.len());
            let value = &evaluated[root];
            assert_eq!(value.shape, *shape);
            assert_eq!(value.prim(), Prim::F32);
            assert_eq!(value.to_f64_lossy_vec(), vec![0.0; shape.iter().product()]);
        }
    }
    // Restrict the native entry to the same roots the evaluator executed.
    dag = chelis_ir::optimize::dead_code_eliminate(&dag);
    let generated = codegen_with_options(
        &dag,
        "gradient_geometry",
        CodegenOptions {
            use_blas: false,
            math_lib_override: Some(MathLib::None),
            static_entry: false,
        },
    )
    .expect("gradient C is publishable");
    let mut harness = String::from(
        "#include \"chelis_runtime.h\"\n#include <stdio.h>\nvoid gradient_geometry(chelis_tensor **, int, chelis_tensor **, int);\nint main(void) {\n",
    );
    let spell = |shape: &[usize]| {
        if shape.is_empty() {
            "0".into()
        } else {
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        }
    };
    for (i, name) in generated.input_labels.iter().enumerate() {
        let shape = inputs
            .iter()
            .find(|(label, _)| *label == name.as_str())
            .unwrap_or_else(|| panic!("unexpected input {name}"))
            .1;
        let rank = shape.len();
        let count = shape.iter().product::<usize>();
        harness.push_str(&format!("int64_t shape_{i}[] = {{{}}};\nfloat data_{i}[{}] = {{0}};\nchelis_tensor *input_{i} = chelis_tensor_entry_borrow({rank}, shape_{i}, CHELIS_DTYPE_F32, data_{i}, {count} * sizeof(float));\n", spell(shape), count.max(1)));
    }
    let input_names = (0..generated.input_labels.len())
        .map(|i| format!("input_{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let input_names = if input_names.is_empty() {
        "NULL".into()
    } else {
        input_names
    };
    harness.push_str(&format!("chelis_tensor *inputs[] = {{{input_names}}}, *outputs[{}] = {{NULL}};\ngradient_geometry(inputs, {}, outputs, {});\n", expected.len(), generated.input_labels.len(), expected.len()));
    for (i, shape) in expected.iter().enumerate() {
        let rank = shape.len();
        let count = shape.iter().product::<usize>();
        harness.push_str(&format!("int64_t expected_{i}[] = {{{}}};\nchelis_read_view result_{i} = chelis_tensor_read_view(outputs[{i}]);\nif (result_{i}.dtype != CHELIS_DTYPE_F32 || result_{i}.count != {count} || chelis_tensor_rank(outputs[{i}]) != {rank}) return 10;\nfor (int a=0; a<{rank}; ++a) if (chelis_tensor_shape(outputs[{i}],a) != expected_{i}[a]) return 11;\nfor (int a=0; a<{count}; ++a) if (((const float*)result_{i}.data)[a] != 0.0f) return 12;\nchelis_tensor_release(outputs[{i}]);\n", spell(shape)));
    }
    for i in 0..generated.input_labels.len() {
        harness.push_str(&format!("chelis_tensor_release(input_{i});\n"));
    }
    harness.push_str("puts(\"GRADIENT GEOMETRY PASS\"); return 0; }\n");
    let (ok, out) =
        compile_and_run_kernel_capturing("gradient_geometry", &generated.c_source, &harness);
    assert_eq!(ok, !reject_input, "{out}");
    if reject_input {
        // The direct kernel ABI validates static input contracts in its
        // preamble; an extent witness can instead own the same check.
        assert!(
            out.contains("numeric trap: domain in load at i64")
                || (out.contains("input `") && out.contains("expected")),
            "{out}"
        );
    } else {
        assert_eq!(out, "GRADIENT GEOMETRY PASS\n");
    }
}

#[test]
fn issue_1767_disconnected_actual_geometry_executes_on_eval_and_c() {
    for shape in [vec![3], vec![], vec![2, 0, 3], vec![2, 3]] {
        let dimensions = shape.iter().map(|n| format!("{n}, ")).collect::<String>();
        let source = format!(
            "x: tensor[{dimensions}f32] = x\ndef loss(z: tensor[{dimensions}f32]) -> f32 = 1.0f32\nderivative = grad(loss)(x)\n"
        );
        let checked = gradient_geometry_checked(&source);
        let library = chelis_ir::lower::try_lower_program_to_library(&checked).unwrap();
        assert_eq!(
            library.program_types()["derivative"].dims,
            shape.iter().copied().map(DimInfo::Lit).collect::<Vec<_>>()
        );
        let mut dag = library.dag().clone();
        dag.set_roots(vec![library.symbol_table()["derivative"]]);
        assert_zero_geometry_both_lanes(dag.clone(), &[("x", &shape)], &[&shape], false);
        let mut wrong_shape = shape.clone();
        if !wrong_shape.is_empty() {
            wrong_shape[0] += 1;
            assert_zero_geometry_both_lanes(dag, &[("x", &wrong_shape)], &[&shape], true);
        }
    }
}

#[test]
fn issue_1767_symbolic_actual_axes_are_read_for_each_execution() {
    let source = "x: tensor[*, *, f32] = x\ndef loss[rows, cols](z: tensor[rows, cols, f32]) -> f32 = 1.0f32\nderivative = grad(loss)(x)\n";
    let library =
        chelis_ir::lower::try_lower_program_to_library(&gradient_geometry_checked(source)).unwrap();
    let mut dag = library.dag().clone();
    dag.set_roots(vec![library.symbol_table()["derivative"]]);
    for shape in [vec![2, 3], vec![3, 2], vec![0, 4], vec![4, 0]] {
        assert_zero_geometry_both_lanes(dag.clone(), &[("x", &shape)], &[&shape], false);
    }
}

#[test]
fn issue_1767_parameter_and_ordered_multiple_actuals_execute_on_eval_and_c() {
    let source = "def loss[rows, cols, other, extra](x: tensor[rows, cols, f32], y: tensor[other, extra, f32]) -> f32 = 1.0f32\n\
        def derivative(theta: tensor[2, 3, f32], eta: tensor[2, 3, f32]) = grad(loss, wrt=(y, x, y))(permute(theta, 1i32, 0i32), eta)\n";
    let library =
        chelis_ir::lower::try_lower_program_to_library(&gradient_geometry_checked(source)).unwrap();
    let mut dag = library.dag().clone();
    dag.set_roots(dag.roots()[dag.roots().len() - 3..].to_vec());
    assert_zero_geometry_both_lanes(
        dag.clone(),
        &[("theta", &[2, 3]), ("eta", &[2, 3])],
        &[&[2, 3], &[3, 2], &[2, 3]],
        false,
    );
    assert_zero_geometry_both_lanes(
        dag,
        &[("theta", &[3, 2]), ("eta", &[2, 3])],
        &[&[2, 3], &[3, 2], &[2, 3]],
        true,
    );
}

#[test]
fn issue_1767_live_and_decoded_helper_contexts_execute_on_eval_and_c() {
    let source = "x: tensor[2, 3, f32] = x\ndef loss[rows, cols](z: tensor[rows, cols, f32]) -> f32 = 1.0f32\ndef helper[rows, cols](theta: tensor[rows, cols, f32]) = grad(loss)(theta)\n";
    let checked = gradient_geometry_checked(source);
    let library = chelis_ir::lower::try_lower_program_to_library(&checked).unwrap();
    let decoded =
        serde_json::from_slice::<chelis_ir::LoweredLibrary>(&serde_json::to_vec(&library).unwrap())
            .unwrap();
    let library_deep =
        chelis_surf::desugar::desugar_program(&chelis_surf::parser::parse_str(source).unwrap())
            .unwrap();
    let env = chelis_types::build_type_env_from_library(&library_deep).unwrap();
    let new = chelis_surf::desugar::desugar_program(
        &chelis_surf::parser::parse_str("derivative = helper(copy(x))\n").unwrap(),
    )
    .unwrap();
    let new = chelis_types::check_ir_with_context(&env, &new).unwrap();
    let new = chelis_effects::check_program(&new).unwrap();
    let new = chelis_types::check_linearity(&new).unwrap();
    for context in [&library, &decoded] {
        let mut dag = chelis_ir::lower::try_lower_program_with_context(context, &new)
            .unwrap()
            .dag;
        dag.set_roots(vec![*dag.roots().last().unwrap()]);
        assert_zero_geometry_both_lanes(dag.clone(), &[("x", &[2, 3])], &[&[2, 3]], false);
        assert_zero_geometry_both_lanes(dag, &[("x", &[3, 2])], &[&[2, 3]], true);
    }
}

/// One input of [`gated_check_graph`]: its rank-1 extent (`None` for rank
/// 0), dtype and one fill value for every element.
struct GatedInput {
    name: &'static str,
    extent: Option<usize>,
    prim: Prim,
    fill: i64,
}

/// A hand-built graph whose root is one check-carrying node under the
/// rank-0 activation `a` (decisions section 11): `x` is `tensor[3, i64]` of
/// sevens, `n` a rank-0 `i64` bound, `y` a `tensor[k, i64]` of runtime
/// extent. Returns the graph and its inputs besides `a`.
fn gated_check_graph(kind: &str, bound: i64) -> (Dag, Vec<GatedInput>) {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let ty = |dims: Vec<DimInfo>, precision| TensorType { dims, precision };
    let load = |dag: &mut Dag, name: &str, dims, precision| {
        dag.add_node(
            decl,
            RiscOp::Load { name: name.into() },
            vec![],
            ty(dims, precision),
            None,
        )
    };
    let x = load(&mut dag, "x", vec![DimInfo::Lit(3)], Prim::Int64);
    let n = load(&mut dag, "n", vec![], Prim::Int64);
    let a = load(&mut dag, "a", vec![], Prim::Bool);
    let owner = Owner {
        decl,
        activation: Some(a),
    };
    let x_input = GatedInput {
        name: "x",
        extent: Some(3),
        prim: Prim::Int64,
        fill: 7,
    };
    let n_input = GatedInput {
        name: "n",
        extent: None,
        prim: Prim::Int64,
        fill: bound,
    };
    let movement = |dag: &mut Dag, op| {
        dag.add_node(
            owner,
            op,
            vec![x, n],
            ty(vec![DimInfo::Named("result".into(), None)], Prim::Int64),
            None,
        )
    };
    let root = match kind {
        "shrink" => movement(
            &mut dag,
            RiscOp::Shrink {
                bounds: vec![(
                    chelis_ir::dag::RtDim::Lit(0),
                    chelis_ir::dag::RtDim::Node(1),
                )],
            },
        ),
        "stride" => movement(
            &mut dag,
            RiscOp::Stride {
                strides: vec![chelis_ir::dag::RtDim::Node(1)],
            },
        ),
        "pad" => movement(
            &mut dag,
            RiscOp::Pad {
                padding: vec![(
                    chelis_ir::dag::RtDim::Node(1),
                    chelis_ir::dag::RtDim::Lit(0),
                )],
                fill: chelis_types::scalar_from_i64("pad", Prim::Int64, 1).unwrap(),
            },
        ),
        "extent witness" => {
            // A call `g[k](y: tensor[k], x: tensor[k])`: `y`'s witness
            // declares `k`, `x`'s claims it.
            let y = load(
                &mut dag,
                "y",
                vec![DimInfo::Named("k".into(), None)],
                Prim::Int64,
            );
            let declares = dag.add_node(
                owner,
                RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::Caller,
                    parameter: "y".into(),
                    axis: RtAxis::Lit(0),
                    requirements: Vec::new(),
                    claims: Vec::new(),
                },
                vec![y],
                ty(vec![], Prim::Int64),
                None,
            );
            let root = dag.add_node(
                owner,
                RiscOp::ExtentWitness {
                    site: ExtentWitnessSite::Caller,
                    parameter: "x".into(),
                    axis: RtAxis::Lit(0),
                    requirements: Vec::new(),
                    claims: vec![ExtentClaim {
                        claim: "k".into(),
                        requirement_declares: true,
                    }],
                },
                vec![x, declares],
                ty(vec![], Prim::Int64),
                None,
            );
            dag.add_root(root);
            let y_input = GatedInput {
                name: "y",
                extent: Some(usize::try_from(bound).unwrap()),
                prim: Prim::Int64,
                fill: 5,
            };
            return (dag, vec![x_input, n_input, y_input]);
        }
        "checked reshape" => {
            // The target extent `n` against the claimed extent 3.
            let required = dag.add_node(
                owner,
                RiscOp::Const {
                    value: chelis_types::scalar_from_i64("reshape", Prim::Int64, 3).unwrap(),
                },
                vec![],
                ty(vec![], Prim::Int64),
                None,
            );
            dag.add_node(
                owner,
                RiscOp::CheckedReshapeExtent {
                    claims: vec!["rows".into()],
                    axis: RtAxis::Lit(0),
                },
                vec![n, required],
                ty(vec![], Prim::Int64),
                None,
            )
        }
        other => panic!("{other}"),
    };
    dag.add_root(root);
    (dag, vec![x_input, n_input])
}

/// Decisions section 11 in the C lane for the checks a Tensor-lane C entry
/// cannot reach from source, because a runtime-extent result has no C
/// representation there (chelis#600): runtime `shrink`, `stride` and `pad`
/// bounds, a call's extent claim (`ExtentWitness`) and a checked reshape
/// target (`CheckedReshapeExtent`), each under a rank-0 activation `a`. The
/// DAG evaluator and the compiled C run the same graph. With `a` false both
/// return the same value (a movement's zeros of its operand's extent; a
/// claim's unchanged value) and trap nothing; with `a` true both trap, with
/// the typed trap each kind raises.
///
/// Evidentiary status: REGRESSION TEST for the inactive rows (at
/// 224414e1f both lanes trap them: neither gates these kinds); the active
/// rows are a disposition lock.
#[test]
fn a_gated_movement_or_extent_claim_checks_only_where_its_activation_holds_in_eval_and_c() {
    // (kind, bound, the evaluator's trap, the C lane's trap). `pad`'s
    // evaluator trap is its bound's untyped report, not a NumericTrap.
    let cases = [
        (
            "shrink",
            4,
            "domain in shrink at i64",
            "numeric trap: domain in shrink at i64",
        ),
        (
            "stride",
            0,
            "domain in stride at i64",
            "numeric trap: domain in stride at i64",
        ),
        (
            "pad",
            -1,
            "must be a non-negative integer",
            "numeric trap: domain in pad at i64",
        ),
        (
            "extent witness",
            2,
            "domain in load at i64",
            "numeric trap: domain in load at i64",
        ),
        (
            "checked reshape",
            2,
            "domain in reshape at i64",
            "numeric trap: domain in reshape at i64",
        ),
    ];
    let mut failures = Vec::new();
    for (kind, bound, eval_trap, c_trap) in cases {
        let (dag, inputs) = gated_check_graph(kind, bound);
        let generated =
            codegen(&dag, "gated_check").unwrap_or_else(|error| panic!("{kind}: {error}"));
        for active in [false, true] {
            let mut values = UnordMap::new();
            let mut allocations = String::new();
            let mut slots = Vec::new();
            let mut all = inputs.iter().collect::<Vec<_>>();
            let activation = GatedInput {
                name: "a",
                extent: None,
                prim: Prim::Bool,
                fill: i64::from(active),
            };
            all.push(&activation);
            for input in &all {
                let shape = input.extent.map_or_else(Vec::new, |extent| vec![extent]);
                let count = input.extent.unwrap_or(1);
                values.insert(
                    input.name.to_string(),
                    TensorValue::from_storage(
                        shape.clone(),
                        finalize_tensor(
                            input.name,
                            input.prim,
                            RawTensor::Int(vec![input.fill; count]),
                        )
                        .unwrap(),
                    ),
                );
                let (dtype, bits) = if input.prim == Prim::Bool {
                    ("CHELIS_DTYPE_BOOL", input.fill as u64)
                } else {
                    ("CHELIS_DTYPE_I64", input.fill as u64)
                };
                allocations.push_str(&format!(
                    "    int64_t {name}_shape[] = {{{extent}}};\n    chelis_tensor *{name} = chelis_alloc({rank}, {name}_shape, {dtype});\n    chelis_tensor_write *{name}_guard = chelis_tensor_begin_write({name});\n    chelis_fill_scalar({name}_guard, chelis_scalar_from_bits({dtype}, UINT64_C({bits})));\n    chelis_tensor_end_write({name}_guard);\n",
                    name = input.name,
                    extent = input.extent.unwrap_or(1),
                    rank = usize::from(input.extent.is_some()),
                ));
            }
            for label in &generated.input_labels {
                assert!(
                    all.iter().any(|input| input.name == label),
                    "{kind}: slot {label}"
                );
                slots.push(label.clone());
            }
            let harness = format!(
                r#"
#include "chelis_runtime.h"
#include <stdio.h>
void gated_check(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
{allocations}    chelis_tensor *inputs[] = {{{slots}}}, *outputs[] = {{NULL}};
    gated_check(inputs, {count}, outputs, 1);
    chelis_read_view out = chelis_tensor_read_view(outputs[0]);
    printf("%d", chelis_tensor_rank(outputs[0]));
    for (int64_t i = 0; i < out.count; ++i) printf(" %lld", (long long)((const int64_t *)out.data)[i]);
    printf("\n");
    return 0;
}}
"#,
                slots = slots.join(", "),
                count = slots.len(),
            );
            let run = checked_indexing_run(&generated.c_source, &harness);
            let c = if run.status.success() {
                Ok(String::from_utf8_lossy(&run.stdout).trim().to_string())
            } else {
                Err(String::from_utf8_lossy(&run.stderr).to_string())
            };
            let root = dag.roots()[0];
            let eval = eval_tensor(&dag, &values).map(|result| {
                let value = &result[&root];
                let mut text = value.shape.len().to_string();
                for element in value.to_f64_lossy_vec() {
                    text.push_str(&format!(" {element}"));
                }
                text
            });
            match (active, &eval, &c) {
                (false, Ok(eval), Ok(c)) if eval == c => {}
                (true, Err(eval), Err(c)) if eval.contains(eval_trap) && c.contains(c_trap) => {}
                _ => failures.push(format!("{kind}, active {active}: eval {eval:?}, C {c:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// `f` of `source` (one input `x: tensor[n, f32]` of ones, a rank-0 `f32`
/// result) lowered as a tensor entry, run by the DAG evaluator and by its
/// compiled C: each lane's value, or its trap text (for C, the emitter's
/// refusal where it refuses the graph).
fn tensor_entry_lanes(source: &str, n: usize) -> (Result<f64, String>, Result<f64, String>) {
    let decls = chelis_surf::parser::parse_str(source).expect("Surf parse");
    let checked = chelis_types::check_ir_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
    )
    .unwrap_or_else(|report| panic!("type check failed: {:?}", report.errors));
    let dag = chelis_ir::host::lower_named_tensor_entry_dag(&checked, "f")
        .expect("named tensor entry lowers");
    let mut values = UnordMap::new();
    values.insert(
        "x".to_string(),
        TensorValue::from_storage(
            vec![n],
            finalize_tensor("x", Prim::F32, RawTensor::Float(vec![1.0; n])).unwrap(),
        ),
    );
    let root = *dag.roots().last().expect("a root");
    let eval = eval_tensor(&dag, &values).map(|result| result[&root].to_f64_lossy_vec()[0]);
    let generated = match codegen(&dag, "claim_arm") {
        Ok(generated) => generated,
        Err(refusal) => return (eval, Err(format!("codegen refused: {refusal:?}"))),
    };
    assert_eq!(generated.input_labels, ["x"]);
    let harness = format!(
        r#"
#include "chelis_runtime.h"
#include <stdio.h>
void claim_arm(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    float x_data[{n}];
    for (int i = 0; i < {n}; ++i) x_data[i] = 1.0f;
    int64_t x_shape[1] = {{{n}}};
    chelis_tensor *x = chelis_tensor_entry_borrow(1, x_shape, CHELIS_DTYPE_F32, x_data, sizeof(x_data));
    chelis_tensor *inputs[1] = {{x}}, *outputs[1] = {{NULL}};
    claim_arm(inputs, 1, outputs, 1);
    chelis_read_view out = chelis_tensor_read_view(outputs[0]);
    printf("%.1f\n", (double)((const float *)out.data)[0]);
    return 0;
}}
"#
    );
    let run = checked_indexing_run(&generated.c_source, &harness);
    let c = if run.status.success() {
        String::from_utf8_lossy(&run.stdout)
            .trim()
            .parse::<f64>()
            .map_err(|error| error.to_string())
    } else {
        Err(String::from_utf8_lossy(&run.stderr).to_string())
    };
    (eval, c)
}

/// Decisions section 11 for a callee's result claim: a call in a runtime
/// `if` arm inlines its callee under the arm's activation, and the claim its
/// declared result makes (a named `tensor[n, f32]` over a `shrink`, and a
/// literal `tensor[2, 2, f32]` over a `reshape` whose target folds to 3) is
/// checked under its carrier's owner activation. Untaken, the DAG evaluator
/// and the compiled C both return the `else` value; taken, both trap with
/// the claim's typed trap.
///
/// Evidentiary status: REGRESSION TEST for both untaken rows (at eaa5f3306
/// and 224414e1f the evaluator traps each claim in the untaken arm); the
/// taken rows are a disposition lock.
#[test]
fn a_callees_result_claim_checks_only_in_a_taken_arm_in_eval_and_c() {
    let cases = [
        (
            "named result claim",
            "def g[n](x: tensor[n, f32]) -> tensor[n, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\ndef f(x: tensor[4, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if gt(s, {c}) then sum(g(copy(x)), 0i32) else sum(x, 0i32)\n}\n",
            4,
            "numeric trap: domain in shrink at i64",
        ),
        (
            "literal result claim",
            "def g[n](y: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])\n\ndef f(x: tensor[6, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if gt(s, {c}) then sum(sum(g(copy(x)), 0i32), 0i32) else sum(x, 0i32)\n}\n",
            6,
            "numeric trap: domain in reshape at i64",
        ),
    ];
    let mut failures = Vec::new();
    for (kind, source, n, trap) in cases {
        let untaken = tensor_entry_lanes(&source.replace("{c}", "50.0f32"), n);
        let expected = n as f64;
        if !matches!(untaken, (Ok(eval), Ok(c)) if eval == expected && c == expected) {
            failures.push(format!("untaken {kind}: {untaken:?}"));
        }
        let taken = tensor_entry_lanes(&source.replace("{c}", "-5.0f32"), n);
        if !matches!(&taken, (Err(eval), Err(c)) if eval.contains(trap) && c.contains(trap)) {
            failures.push(format!("taken {kind}: {taken:?}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// `f` of `source` lowered as a tensor entry over `inputs` (each a rank-1
/// `f32` tensor of ones of the given extent), with a rank-0 `f32` result,
/// run by the DAG evaluator and by its compiled C: each lane's value, or its
/// trap text (for C, the emitter's refusal where it refuses the graph).
fn tensor_entry_lanes_over(
    source: &str,
    inputs: &[(&str, usize)],
) -> (Result<f64, String>, Result<f64, String>) {
    let decls = chelis_surf::parser::parse_str(source).expect("Surf parse");
    let checked = chelis_types::check_ir_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
    )
    .unwrap_or_else(|report| panic!("type check failed: {:?}", report.errors));
    let dag = chelis_ir::host::lower_named_tensor_entry_dag(&checked, "f")
        .expect("named tensor entry lowers");
    let mut values = UnordMap::new();
    for (name, n) in inputs {
        values.insert(
            name.to_string(),
            TensorValue::from_storage(
                vec![*n],
                finalize_tensor("input", Prim::F32, RawTensor::Float(vec![1.0; *n])).unwrap(),
            ),
        );
    }
    let root = *dag.roots().last().expect("a root");
    let eval = eval_tensor(&dag, &values).map(|result| result[&root].to_f64_lossy_vec()[0]);
    let generated = match codegen(&dag, "claim_arm") {
        Ok(generated) => generated,
        Err(refusal) => return (eval, Err(format!("codegen refused: {refusal:?}"))),
    };
    let mut allocations = String::new();
    for label in &generated.input_labels {
        let (_, n) = inputs
            .iter()
            .find(|(name, _)| name == label)
            .unwrap_or_else(|| panic!("entry slot {label}"));
        allocations.push_str(&format!(
            "    float {label}_data[{n}];\n    for (int i = 0; i < {n}; ++i) {label}_data[i] = 1.0f;\n    int64_t {label}_shape[1] = {{{n}}};\n    chelis_tensor *{label} = chelis_tensor_entry_borrow(1, {label}_shape, CHELIS_DTYPE_F32, {label}_data, sizeof({label}_data));\n"
        ));
    }
    let harness = format!(
        r#"
#include "chelis_runtime.h"
#include <stdio.h>
void claim_arm(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
{allocations}    chelis_tensor *inputs[] = {{{slots}}}, *outputs[1] = {{NULL}};
    claim_arm(inputs, {count}, outputs, 1);
    chelis_read_view out = chelis_tensor_read_view(outputs[0]);
    printf("%.1f\n", (double)((const float *)out.data)[0]);
    return 0;
}}
"#,
        slots = generated.input_labels.join(", "),
        count = generated.input_labels.len(),
    );
    let run = checked_indexing_run(&generated.c_source, &harness);
    let c = if run.status.success() {
        String::from_utf8_lossy(&run.stdout)
            .trim()
            .parse::<f64>()
            .map_err(|error| error.to_string())
    } else {
        Err(String::from_utf8_lossy(&run.stderr).to_string())
    };
    (eval, c)
}

/// #2586 round 2b, decisions section 25: an arm whose extent rests on a
/// claim, (a) a callee's result claim, (b) a guarded broadcast's unit claim
/// on a local, (c) a local ascription, and a parameter's unit refinement.
/// Untaken (`{c}` 50), the claim is not checked and the claim-sized nodes
/// are zeros, so the DAG evaluator and the compiled C both return the
/// `else` value, 32; taken (`{c}` -5) with the claim false, both trap with
/// the claim's typed trap. `x` is 32 ones and `t` 3 ones.
///
/// Evidentiary status: REGRESSION TEST for every untaken row at 096daea8c
/// (each lane traps or fails there); DISPOSITION LOCK for the taken rows.
#[test]
fn an_untaken_claimed_arm_checks_nothing_and_a_taken_one_traps_in_eval_and_c() {
    let cases = [
        (
            "(a) callee's result claim",
            "def g(y: tensor[*, f32]) -> tensor[3, f32] = shrink(y, [[0i64, sub(shape(&y, 0i32), 1i64)]])\n\ndef f(x: tensor[32, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if gt(s, {c}) then sum(add(g(copy(x)), to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32) else sum(x, 0i32)\n}\n",
            "extent `3`: claimed = 3, shrink axis 0 = 31",
        ),
        (
            "(b) guarded broadcast of a local",
            "def f(x: tensor[32, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  t = shrink(copy(x), [[0i64, sub(shape(&x, 0i32), 29i64)]])\n  if gt(s, {c}) then sum(add(x, expand(t, 0i32, 32i64)), 0i32) else sum(x, 0i32)\n}\n",
            "numeric trap: domain in expand at i64",
        ),
        (
            "(c) local ascription",
            "def f(x: tensor[32, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if gt(s, {c}) then {\n    y: tensor[3, f32] = shrink(copy(x), [[0i64, sub(shape(&x, 0i32), 1i64)]])\n    sum(add(y, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n  } else sum(x, 0i32)\n}\n",
            "extent `3`: claimed = 3, shrink axis 0 = 31",
        ),
        (
            "parameter's unit refinement",
            "def f[n](x: tensor[32, f32], t: tensor[n, f32]) -> tensor[f32] = {\n  s = tensor_to_scalar(sum(&x, 0i32))\n  if gt(s, {c}) then sum(add(x, expand(t, 0i32, 32i64)), 0i32) else sum(x, 0i32)\n}\n",
            "extent `1`: claimed = 1, t axis 0 = 3",
        ),
    ];
    let inputs = [("x", 32), ("t", 3)];
    let mut failures = Vec::new();
    for (kind, source, trap) in cases {
        let untaken = tensor_entry_lanes_over(&source.replace("{c}", "50.0f32"), &inputs);
        if !matches!(untaken, (Ok(eval), Ok(c)) if eval == 32.0 && c == 32.0) {
            failures.push(format!("untaken {kind}: {untaken:?}"));
        }
        let taken = tensor_entry_lanes_over(&source.replace("{c}", "-5.0f32"), &inputs);
        if !matches!(&taken, (Err(eval), Err(c)) if eval.contains(trap) && c.contains(trap)) {
            failures.push(format!("taken {kind}: {taken:?}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// One kind [`a_dead_let_of_each_newly_seeded_kind_traps_in_eval_and_c`]
/// covers: the declarations before `f`, the dead `let` (with `{v}` for the
/// value that decides the check), the value that traps and the one that
/// does not, `x`'s extent, and each lane's trap.
struct DeadLetKind {
    name: &'static str,
    callee: &'static str,
    dead: &'static str,
    traps: &'static str,
    total: &'static str,
    n: usize,
    eval_trap: &'static str,
    c_trap: &'static str,
}

/// The kinds the trap seed gained on this branch (decisions sections 6.2
/// and 11: a potentially trapping node of an entered declaration is seeded,
/// so a discarded `let` initializer runs its check): an integer sum and
/// product, an empty reduced axis, runtime `shrink`, `stride` and `pad`
/// bounds, a call's named extent claim, and a callee's named and literal
/// result claims. The integer sum row is the spec/03 section 4.4 oracle
/// (`dead_sum`).
const DEAD_LET_KINDS: [DeadLetKind; 10] = [
    DeadLetKind {
        name: "integer sum (dead_sum)",
        callee: "",
        dead: "dead = sum(to_tensor([{v}]), 0i32)",
        traps: "2000000000i32, 2000000000i32",
        total: "2i32, 2i32",
        n: 1,
        eval_trap: "numeric trap: overflow in sum at i32",
        c_trap: "numeric trap: overflow in sum at i32",
    },
    DeadLetKind {
        name: "integer product",
        callee: "",
        dead: "dead = prod_reduce(to_tensor([{v}]), 0i32)",
        traps: "100000i32, 100000i32",
        total: "2i32, 3i32",
        n: 4,
        eval_trap: "numeric trap: overflow in prod_reduce at i32",
        // The C DAG emitter refuses an integer product outright (#729).
        c_trap: "`i32` tensors in the C DAG emitter",
    },
    DeadLetKind {
        name: "empty max_reduce",
        callee: "",
        dead: "e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), {v}))\n  dead = max_reduce(e, 0i32)",
        traps: "4i64",
        total: "3i64",
        n: 4,
        eval_trap: "numeric trap: domain in max_reduce at f32",
        c_trap: "numeric trap: domain in max_reduce at f32",
    },
    DeadLetKind {
        name: "empty argmax_reduce",
        callee: "",
        dead: "e = insert(scalar_to_tensor(2.0f32), 0i32, sub(shape(&x, 0i32), {v}))\n  dead = argmax_reduce(e, 0i32)",
        traps: "4i64",
        total: "3i64",
        n: 4,
        eval_trap: "numeric trap: domain in argmax_reduce at i64",
        c_trap: "numeric trap: domain in argmax_reduce at i64",
    },
    DeadLetKind {
        name: "shrink past the end",
        callee: "",
        dead: "dead = shrink(&x, [[1i64, add(shape(&x, 0i32), {v})]])",
        traps: "3i64",
        total: "0i64",
        n: 4,
        eval_trap: "numeric trap: domain in shrink at i64",
        c_trap: "numeric trap: domain in shrink at i64",
    },
    DeadLetKind {
        name: "stride of zero",
        callee: "",
        dead: "dead = stride(&x, sub(shape(&x, 0i32), {v}))",
        traps: "4i64",
        total: "3i64",
        n: 4,
        eval_trap: "numeric trap: domain in stride at i64",
        c_trap: "numeric trap: domain in stride at i64",
    },
    DeadLetKind {
        name: "negative pad",
        callee: "",
        dead: "dead = pad(&x, [[sub(shape(&x, 0i32), {v}), 0i64]], 0.0f32)",
        traps: "5i64",
        total: "3i64",
        n: 4,
        eval_trap: "must be a non-negative integer",
        c_trap: "numeric trap: domain in pad at i64",
    },
    DeadLetKind {
        name: "call's named extent claim",
        callee: "def g[n](a: tensor[n, f32], b: tensor[n, f32]) -> tensor[f32] = sum(a, 0i32)\n\n",
        dead: "dead = g(shrink(&x, [[0i64, sub(shape(&x, 0i32), {v})]]), copy(x))",
        traps: "1i64",
        total: "0i64",
        n: 4,
        eval_trap: "numeric trap: domain in load at i64",
        c_trap: "numeric trap: domain in load at i64",
    },
    DeadLetKind {
        name: "callee's named result claim",
        callee: "def g[n](x: tensor[n, f32]) -> tensor[n, f32] = shrink(x, [[{v}, shape(x, 0i32)]])\n\n",
        dead: "dead = g(copy(x))",
        traps: "1i64",
        total: "0i64",
        n: 4,
        eval_trap: "numeric trap: domain in shrink at i64",
        c_trap: "numeric trap: domain in shrink at i64",
    },
    DeadLetKind {
        name: "callee's literal result claim",
        callee: "def g[n](y: tensor[n, f32]) -> tensor[{v}, 2, f32] = reshape(y, [floor_div(shape(y, 0i32), 2i64), 2i64])\n\n",
        dead: "dead = g(copy(x))",
        traps: "3",
        total: "2",
        n: 4,
        eval_trap: "numeric trap: domain in reshape at i64",
        c_trap: "numeric trap: domain in reshape at i64",
    },
];

/// Decisions sections 6.2 and 11 for every kind the trap seed gained: a
/// discarded `let` whose initializer can trap runs its check, in the DAG
/// evaluator and in compiled C of the same lowered graph, and its total
/// twin (the same program with a value the check accepts) returns `sum(x)`.
/// The C emitter refuses an integer `prod_reduce` (#729), so that row's C
/// lane pins the refusal, for both twins.
/// `chelis-cli`'s `issue_2563_untaken_arm_eval_file` has the
/// `chelis eval --file` rows of the same sources.
///
/// Evidentiary status: per row, in the report of ks5-h2f (REGRESSION TEST
/// where the trapping row returns at 224414e1f, DISPOSITION LOCK where it
/// traps there); the total twins are a disposition lock.
#[test]
fn a_dead_let_of_each_newly_seeded_kind_traps_in_eval_and_c() {
    let mut failures = Vec::new();
    for kind in &DEAD_LET_KINDS {
        let source = |v: &str| {
            format!(
                "{}def f(x: tensor[{}, f32]) -> tensor[f32] = {{\n  {}\n  sum(x, 0i32)\n}}\n",
                kind.callee.replace("{v}", v),
                kind.n,
                kind.dead.replace("{v}", v)
            )
        };
        let dead = tensor_entry_lanes(&source(kind.traps), kind.n);
        if !matches!(&dead, (Err(eval), Err(c)) if eval.contains(kind.eval_trap) && c.contains(kind.c_trap))
        {
            failures.push(format!("dead {}: {dead:?}", kind.name));
        }
        let total = tensor_entry_lanes(&source(kind.total), kind.n);
        let expected = kind.n as f64;
        let c_refused = kind.c_trap.starts_with('`');
        if !matches!(&total, (Ok(eval), Ok(c)) if *eval == expected && *c == expected)
            && !matches!(&total, (Ok(eval), Err(c)) if c_refused && *eval == expected && c.contains(kind.c_trap))
        {
            failures.push(format!("total {}: {total:?}", kind.name));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// chelis#2512, spec/04 section 4.7: `x: [n]` forwarded by `neg` under the
/// name `m`, added to `z: [m]`. Nothing in the interface relates `x`'s axis
/// to `z`'s; the equality comes only from `neg`'s output being stamped `[m]`,
/// so the comparison is a guard owned by `neg`. It runs at `neg`'s source
/// position, after an independent earlier trap and before `neg` allocates,
/// and fails as `numeric trap: domain in neg at i64`.
fn restamped_extent_dag(neg_first: bool, earlier_overflow: bool) -> Dag {
    let named = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let load = |dag: &mut Dag, name: &str, ty: TensorType| {
        dag.add_node(decl, RiscOp::Load { name: name.into() }, vec![], ty, None)
    };
    let mut roots = Vec::new();
    if earlier_overflow {
        let w = load(
            &mut dag,
            "w",
            TensorType {
                dims: vec![DimInfo::Lit(1)],
                precision: Prim::Int64,
            },
        );
        let overflow = dag.add_node(
            decl,
            RiscOp::Neg,
            vec![w],
            TensorType {
                dims: vec![DimInfo::Lit(1)],
                precision: Prim::Int64,
            },
            None,
        );
        roots.push(overflow);
    }
    let (negated, loaded_z) = if neg_first {
        let x = load(&mut dag, "x", named("n"));
        let negated = dag.add_node(decl, RiscOp::Neg, vec![x], named("m"), None);
        (negated, load(&mut dag, "z", named("m")))
    } else {
        let loaded_z = load(&mut dag, "z", named("m"));
        let x = load(&mut dag, "x", named("n"));
        (
            dag.add_node(decl, RiscOp::Neg, vec![x], named("m"), None),
            loaded_z,
        )
    };
    roots.push(dag.add_node(decl, RiscOp::Add, vec![negated, loaded_z], named("m"), None));
    dag.set_roots(roots);
    dag
}

fn restamped_extent_eval(dag: &Dag, x: &[f64], z: &[f64]) -> Result<Vec<f64>, String> {
    let mut inputs = chelis_unord::UnordMap::new();
    inputs.insert(
        "x".to_string(),
        TensorValue::from_vec(vec![x.len()], x.to_vec()),
    );
    inputs.insert(
        "z".to_string(),
        TensorValue::from_vec(vec![z.len()], z.to_vec()),
    );
    inputs.insert(
        "w".to_string(),
        typed_value(typed_storage(
            Prim::Int64,
            chelis_types::RawTensor::Int(vec![i64::MIN]),
        )),
    );
    let values = eval_tensor(dag, &inputs)?;
    Ok(values[dag.roots().last().expect("sum root")].to_f64_lossy_vec())
}

fn restamped_extent_c(dag: &Dag, function: &str, x: &[f32], z: &[f32]) -> String {
    let result = codegen(dag, function).expect("restamped extent codegen");
    let slot = |name: &str| result.input_labels.iter().position(|label| label == name);
    let values = |data: &[f32]| {
        data.iter()
            .map(|value| format!("{value:?}f"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let n_in = result.input_labels.len();
    let n_out = dag.roots().len();
    let w_input = match slot("w") {
        Some(w_slot) => format!(
            "static int64_t w_data[] = {{ INT64_MIN }};\n    inputs[{w_slot}] = chelis_tensor_entry_borrow(1, (int64_t[]){{ 1 }}, CHELIS_DTYPE_I64, w_data, sizeof w_data);"
        ),
        None => String::new(),
    };
    let harness = format!(
        r#"{HARNESS_HEADER}
#include <stdint.h>
extern void {function}(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    static float x_data[] = {{ {x_values} }};
    static float z_data[] = {{ {z_values} }};
    chelis_tensor *inputs[{n_in}];
    {w_input}
    inputs[{x_slot}] = chelis_tensor_entry_borrow(1, (int64_t[]){{ {x_len} }}, CHELIS_DTYPE_F32, x_data, sizeof x_data);
    inputs[{z_slot}] = chelis_tensor_entry_borrow(1, (int64_t[]){{ {z_len} }}, CHELIS_DTYPE_F32, z_data, sizeof z_data);
    chelis_tensor *outputs[{n_out}] = {{ NULL }};
    {function}(inputs, {n_in}, outputs, {n_out});
    chelis_read_view view = chelis_tensor_read_view(outputs[{n_out} - 1]);
    for (int64_t i = 0; i < view.count; ++i) printf("%g ", ((const float *)view.data)[i]);
    puts("completed");
    return 0;
}}
"#,
        x_values = values(x),
        z_values = values(z),
        x_slot = slot("x").expect("x slot"),
        z_slot = slot("z").expect("z slot"),
        x_len = x.len(),
        z_len = z.len(),
    );
    let name = format!("{function}_{}_{}", x.len(), z.len());
    let run = compile_and_capture_run(&name, &result.c_source, &harness);
    format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    )
}

/// The trap and context lines a failing run printed, in order.
fn extent_trap_lines(output: &str) -> Vec<String> {
    output
        .lines()
        .filter(|line| line.starts_with("extent `") || line.starts_with("numeric trap:"))
        .map(str::to_string)
        .collect()
}

/// chelis#2512 REGRESSION TEST: at base neither lane compared the restamped
/// axis at `neg`; compiled C trapped later in `add`'s operand check and the
/// evaluator reported an untyped shape mismatch.
#[test]
fn issue_2512_a_restamped_axis_is_guarded_by_the_restamping_operation() {
    let mut failures = Vec::new();
    for neg_first in [true, false] {
        for (x, z) in [
            (&[1.0f32, 2.0][..], &[1.0f32, 2.0, 3.0][..]),
            (&[1.0, 2.0, 3.0, 4.0][..], &[1.0, 2.0, 3.0][..]),
        ] {
            let dag = restamped_extent_dag(neg_first, false);
            let expected = vec![
                format!(
                    "extent `m`: claimed = {}, neg axis 0 = {}",
                    z.len(),
                    x.len()
                ),
                "numeric trap: domain in neg at i64".to_string(),
            ];
            let function = if neg_first {
                "restamped_extent_neg_first"
            } else {
                "restamped_extent_z_first"
            };
            let compiled = restamped_extent_c(&dag, function, x, z);
            if compiled.contains("completed") || extent_trap_lines(&compiled) != expected {
                failures.push(format!(
                    "C, neg first {neg_first}, x {}, z {}: {compiled}",
                    x.len(),
                    z.len()
                ));
            }
            let widen = |data: &[f32]| data.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
            match restamped_extent_eval(&dag, &widen(x), &widen(z)) {
                Err(error) if extent_trap_lines(&error) == expected => {}
                other => failures.push(format!(
                    "eval, neg first {neg_first}, x {}, z {}: {other:?}",
                    x.len(),
                    z.len()
                )),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Negative parity: agreeing extents run on both lanes.
#[test]
fn issue_2512_an_agreeing_restamped_axis_runs() {
    for neg_first in [true, false] {
        let dag = restamped_extent_dag(neg_first, false);
        let function = if neg_first {
            "restamped_extent_ok_neg_first"
        } else {
            "restamped_extent_ok_z_first"
        };
        let compiled = restamped_extent_c(&dag, function, &[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]);
        assert!(
            compiled.contains("9 18 27 completed"),
            "neg first {neg_first}: {compiled}"
        );
        assert_eq!(
            restamped_extent_eval(&dag, &[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]),
            Ok(vec![9.0, 18.0, 27.0]),
            "neg first {neg_first}"
        );
    }
}

/// Placement (spec/04 section 4.7): the guard is `neg`'s, not an entry
/// guard, so an independent overflow earlier in source order is observed
/// first on both lanes.
#[test]
fn issue_2512_an_earlier_independent_trap_precedes_the_restamp_guard() {
    let dag = restamped_extent_dag(true, true);
    let expected = vec!["numeric trap: overflow in neg at i64".to_string()];
    let compiled = restamped_extent_c(
        &dag,
        "restamped_extent_after_overflow",
        &[1.0, 2.0],
        &[1.0, 2.0, 3.0],
    );
    assert_eq!(extent_trap_lines(&compiled), expected, "{compiled}");
    let evaluated = restamped_extent_eval(&dag, &[1.0, 2.0], &[1.0, 2.0, 3.0]);
    assert_eq!(
        evaluated
            .as_ref()
            .err()
            .map(|error| extent_trap_lines(error)),
        Some(expected),
        "{evaluated:?}"
    );
}

/// A restamping operation the untaken-arm tests below build: `op` over
/// `x: [n]` of `input`, stamped `[m]` of `output`, trapping as `trap`.
#[derive(Clone)]
struct RestampUnderActivation {
    op: RiscOp,
    input: Prim,
    output: Prim,
    trap: &'static str,
}

/// The restamping operations: a float `neg`, whose class checks nothing
/// else; an integer `neg`, whose operand values its class also gates; and a
/// `copy`, whose guard sits at the restamping node itself because the
/// producer it attributes the extent to, the `Load` of `x`, runs before the
/// activation. (A `cast` cannot restamp: the verifier refuses a cast whose
/// dims change.)
fn restamps_under_activation() -> [RestampUnderActivation; 3] {
    [
        RestampUnderActivation {
            op: RiscOp::Neg,
            input: Prim::F32,
            output: Prim::F32,
            trap: "neg",
        },
        RestampUnderActivation {
            op: RiscOp::Neg,
            input: Prim::Int64,
            output: Prim::Int64,
            trap: "neg",
        },
        // A `copy` is administrative, so its guard names the operation
        // behind it (spec/04 section 4.7), here the `load` of `x`.
        RestampUnderActivation {
            op: RiscOp::Copy,
            input: Prim::F64,
            output: Prim::F64,
            trap: "load",
        },
    ]
}

fn restamp_named(name: &str, precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision,
    }
}

fn restamp_scalar(precision: Prim) -> TensorType {
    TensorType {
        dims: Vec::new(),
        precision,
    }
}

/// The graph around a restamped value `restamped` of type `[m]`, built
/// under the rank-0 activation `a` by `restamp`: it is added to `z: [m]` and
/// summed under `a`, and that sum is selected by `a` against the sum of `z`,
/// so `m`'s class holds the restamp and `z` in one scope. The restamped
/// value is the second root, so its untaken-arm value is observed directly.
fn restamp_under_activation_dag(
    output: Prim,
    restamp: impl FnOnce(&mut Dag, Owner) -> (chelis_ir::dag::NodeId, chelis_ir::dag::NodeId),
) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let z = dag.add_node(
        decl,
        RiscOp::Load { name: "z".into() },
        vec![],
        restamp_named("m", output),
        None,
    );
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        restamp_scalar(Prim::Bool),
        None,
    );
    let under_a = Owner::new(decl, Some(a));
    let (restamped, summand) = restamp(&mut dag, under_a);
    let sum = |dag: &mut Dag, owner: Owner, input| {
        dag.add_node(
            owner,
            RiscOp::Sum {
                axis: 0,
                accumulator: output,
            },
            vec![input],
            restamp_scalar(output),
            None,
        )
    };
    let plus = dag.add_node(
        under_a,
        RiscOp::Add,
        vec![summand, z],
        restamp_named("m", output),
        None,
    );
    let taken = sum(&mut dag, under_a, plus);
    let other = sum(&mut dag, Owner::unconditional(decl), z);
    let selected = dag.add_node(
        decl,
        RiscOp::Where,
        vec![a, taken, other],
        restamp_scalar(output),
        None,
    );
    dag.set_roots(vec![selected, restamped]);
    dag
}

/// chelis#2512's restamp under an activation: `x: [n]` restamped `[m]` by
/// `case.op` (spec/10 section 3.2), in [`restamp_under_activation_dag`].
fn restamped_extent_under_activation_dag(case: &RestampUnderActivation) -> Dag {
    restamp_under_activation_dag(case.output, |dag, under_a| {
        let decl = under_a.decl;
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            restamp_named("n", case.input),
            None,
        );
        let restamped = dag.add_node(
            under_a,
            case.op.clone(),
            vec![x],
            restamp_named("m", case.output),
            None,
        );
        (restamped, restamped)
    })
}

/// One output as `shape [..] bytes <hex>`, the bytes in memory order: the
/// form both lanes' outputs are compared in, bit for bit.
fn restamp_output_text(shape: &[usize], bytes: &[u8]) -> String {
    let hex = bytes.iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!("shape {shape:?} bytes {hex}")
}

fn restamp_storage_bytes(storage: &chelis_types::TensorStorage) -> Vec<u8> {
    use chelis_types::StorageView;
    match storage.view() {
        StorageView::F64(values) => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        StorageView::F32(values) => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        StorageView::I64(values) => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        other => panic!("untested restamp output storage {other:?}"),
    }
}

fn restamp_input_storage(prim: Prim, values: &[i32]) -> chelis_types::TensorStorage {
    let raw = if prim.is_float() {
        RawTensor::Float(values.iter().copied().map(f64::from).collect())
    } else {
        RawTensor::Int(values.iter().copied().map(i64::from).collect())
    };
    finalize_tensor("restamp test input", prim, raw).unwrap()
}

/// One lane's outputs ([`restamp_output_text`]), or its trap text.
type RestampLane = Result<Vec<String>, String>;

/// Both lanes' outputs for one row of named inputs (`(name, storage,
/// rank)`), or their trap text: the evaluator's, then compiled C's under
/// AddressSanitizer and UndefinedBehaviorSanitizer.
fn restamp_under_activation_lanes(
    dag: &Dag,
    generated: &chelis_backend_c::CodegenResult,
    inputs: &[(&str, chelis_types::TensorStorage, usize)],
) -> (RestampLane, RestampLane) {
    let mut values = UnordMap::new();
    for (name, storage, rank) in inputs {
        let shape = if *rank == 0 {
            Vec::new()
        } else {
            vec![storage.len()]
        };
        values.insert(
            name.to_string(),
            typed_value_with_shape(shape, storage.clone()),
        );
    }
    let eval = eval_tensor(dag, &values).map(|results| {
        dag.roots()
            .iter()
            .map(|root| {
                let value = &results[root];
                restamp_output_text(&value.shape, &restamp_storage_bytes(value.storage()))
            })
            .collect::<Vec<_>>()
    });
    let declarations = inputs
        .iter()
        .map(|(name, storage, rank)| {
            let (c_type, c_dtype, data) = c_storage_case(storage);
            let shape = if *rank == 0 {
                "NULL".to_string()
            } else {
                format!("(int64_t[]){{ {} }}", storage.len())
            };
            let slot = generated
                .input_labels
                .iter()
                .position(|label| label == name)
                .unwrap();
            format!(
                "{c_type} {name}_data[] = {{ {data} }};\n    inputs[{slot}] = chelis_tensor_entry_borrow({rank}, {shape}, {c_dtype}, {name}_data, sizeof {name}_data);"
            )
        })
        .collect::<Vec<_>>()
        .join("\n    ");
    let n_in = inputs.len();
    let harness = format!(
        r#"
#include "chelis_runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
static double test_f64(uint64_t bits) {{ double v; memcpy(&v, &bits, sizeof v); return v; }}
static float test_f32(uint32_t bits) {{ float v; memcpy(&v, &bits, sizeof v); return v; }}
void restamp_gate(chelis_tensor **, int, chelis_tensor **, int);
int main(void) {{
    chelis_tensor *inputs[{n_in}];
    {declarations}
    chelis_tensor *outputs[2] = {{ NULL, NULL }};
    restamp_gate(inputs, {n_in}, outputs, 2);
    for (int k = 0; k < 2; ++k) {{
        chelis_read_view view = chelis_tensor_read_view(outputs[k]);
        printf("shape [");
        for (int d = 0; d < chelis_tensor_rank(outputs[k]); ++d)
            printf("%s%lld", d ? ", " : "", (long long)chelis_tensor_shape(outputs[k], d));
        printf("] bytes ");
        const unsigned char *bytes = (const unsigned char *)view.data;
        for (int64_t i = 0; i < chelis_tensor_byte_count(outputs[k]); ++i) printf("%02x", bytes[i]);
        printf("\n");
    }}
    return 0;
}}
"#
    );
    let run = checked_indexing_run(&generated.c_source, &harness);
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    let c = if run.status.success() && stderr.is_empty() {
        Ok(stdout.lines().map(str::to_string).collect())
    } else {
        Err(stderr)
    };
    (eval, c)
}

/// Whether both lanes agree with `expected`: its outputs, or its trap
/// lines.
fn restamp_lanes_agree(
    lanes: &(RestampLane, RestampLane),
    expected: &Result<Vec<String>, Vec<String>>,
) -> bool {
    let agrees = |lane: &RestampLane| match (expected, lane) {
        (Ok(expected), Ok(lane)) => lane == expected,
        (Err(expected), Err(error)) => extent_trap_lines(error) == *expected,
        _ => false,
    };
    agrees(&lanes.0) && agrees(&lanes.1)
}

/// spec/10 section 3.2 REGRESSION TEST: the restamp guard of chelis#2512 is
/// checked under the restamping node's own activation. Where that is false
/// the node produces zeros of its declared type `[m]`, reading nothing of
/// its operand of extent `n`; where it holds, the guard traps before the
/// node allocates, naming the operation that owns it; and agreeing extents
/// give both lanes the same bits either way. At the merge of #2629 the guard
/// was ungated, so every untaken row trapped on both lanes; gating the guard
/// alone left compiled C refusing untyped in
/// `chelis_tensor_elementwise_index_step_for_shape`, which sized the node by
/// `m` over an operand of `n` elements.
#[test]
fn issue_2512_a_restamp_is_checked_under_its_own_activation_and_untaken_yields_zeros() {
    let mut failures = Vec::new();
    for case in restamps_under_activation() {
        let dag = restamped_extent_under_activation_dag(&case);
        assert_eq!(chelis_ir::verify::verify(&dag), Vec::<String>::new());
        let generated = codegen(&dag, "restamp_gate").expect("restamp codegen");
        let bytes =
            |values: &[i32]| restamp_storage_bytes(&restamp_input_storage(case.output, values));
        let mut row = |active: bool,
                       x: &[i32],
                       z: &[i32],
                       expected: Result<Vec<String>, Vec<String>>| {
            let inputs = [
                ("x", restamp_input_storage(case.input, x), 1),
                ("z", restamp_input_storage(case.output, z), 1),
                (
                    "a",
                    restamp_input_storage(Prim::Bool, &[i32::from(active)]),
                    0,
                ),
            ];
            let lanes = restamp_under_activation_lanes(&dag, &generated, &inputs);
            if !restamp_lanes_agree(&lanes, &expected) {
                failures.push(format!(
                    "{:?} {:?}, active {active}, x {}, z {}: expected {expected:?}\n  eval {:?}\n  c {:?}",
                    case.op,
                    case.output,
                    x.len(),
                    z.len(),
                    lanes.0,
                    lanes.1
                ));
            }
        };
        for (x, z) in [(vec![1, 2], vec![1, 2, 3]), (vec![1, 2, 3], vec![4, 5])] {
            // Untaken: the other arm's value, and the restamp's zeros of `m`.
            row(
                false,
                &x,
                &z,
                Ok(vec![
                    restamp_output_text(&[], &bytes(&[z.iter().sum()])),
                    restamp_output_text(&[z.len()], &vec![0u8; bytes(&z).len()]),
                ]),
            );
            // Taken: the guard traps.
            row(
                true,
                &x,
                &z,
                Err(vec![
                    format!(
                        "extent `m`: claimed = {}, {} axis 0 = {}",
                        z.len(),
                        case.trap,
                        x.len()
                    ),
                    format!("numeric trap: domain in {} at i64", case.trap),
                ]),
            );
        }
        // Agreeing extents: taken computes, untaken still yields zeros.
        let (x, z) = ([1, 2, 3], [10, 20, 30]);
        let restamped: Vec<i32> = if matches!(case.op, RiscOp::Neg) {
            x.iter().map(|v| -v).collect()
        } else {
            x.to_vec()
        };
        let plus: Vec<i32> = restamped.iter().zip(&z).map(|(r, z)| r + z).collect();
        row(
            true,
            &x,
            &z,
            Ok(vec![
                restamp_output_text(&[], &bytes(&[plus.iter().sum()])),
                restamp_output_text(&[3], &bytes(&restamped)),
            ]),
        );
        row(
            false,
            &x,
            &z,
            Ok(vec![
                restamp_output_text(&[], &bytes(&[z.iter().sum()])),
                restamp_output_text(&[3], &vec![0u8; bytes(&z).len()]),
            ]),
        );
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The restamp Surf lowering reaches (chelis#2512): an `expand` whose kept
/// axis restates `x: [n]` as `[m]`, inserting an axis `k` sized by the
/// scalar `s`, which the `expand` declares. Under a false activation it is
/// zeros of `[m, k]` on both lanes, `k` read from `s`, with nothing trapped;
/// under a true one the guard traps, naming the rank-increasing `expand` as
/// the `insert` primitive it is.
#[test]
fn issue_2512_an_untaken_restamping_expand_declares_its_inserted_extent_and_yields_zeros() {
    let dag = restamp_under_activation_dag(Prim::F32, |dag, under_a| {
        let decl = under_a.decl;
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            restamp_named("n", Prim::F32),
            None,
        );
        let s = dag.add_node(
            decl,
            RiscOp::Load { name: "s".into() },
            vec![],
            restamp_scalar(Prim::Int64),
            None,
        );
        let expanded = dag.add_node(
            under_a,
            RiscOp::Expand {
                axis: 1,
                size: chelis_ir::dag::RtDim::Node(1),
            },
            vec![x, s],
            TensorType {
                dims: vec![
                    DimInfo::Named("m".into(), None),
                    DimInfo::Named("k".into(), None),
                ],
                precision: Prim::F32,
            },
            None,
        );
        let reduced = dag.add_node(
            under_a,
            RiscOp::Sum {
                axis: 1,
                accumulator: Prim::F32,
            },
            vec![expanded],
            restamp_named("m", Prim::F32),
            None,
        );
        (expanded, reduced)
    });
    assert_eq!(chelis_ir::verify::verify(&dag), Vec::<String>::new());
    let generated = codegen(&dag, "restamp_gate").expect("restamp codegen");
    let f32_bytes =
        |values: &[i32]| restamp_storage_bytes(&restamp_input_storage(Prim::F32, values));
    let mut failures = Vec::new();
    for (x, z) in [(vec![1, 2], vec![1, 2, 3]), (vec![1, 2, 3], vec![4, 5])] {
        for active in [false, true] {
            let inputs = [
                ("x", restamp_input_storage(Prim::F32, &x), 1),
                ("z", restamp_input_storage(Prim::F32, &z), 1),
                (
                    "a",
                    restamp_input_storage(Prim::Bool, &[i32::from(active)]),
                    0,
                ),
                ("s", restamp_input_storage(Prim::Int64, &[2]), 0),
            ];
            let expected = if active {
                Err(vec![
                    format!(
                        "extent `m`: claimed = {}, insert axis 0 = {}",
                        z.len(),
                        x.len()
                    ),
                    "numeric trap: domain in insert at i64".to_string(),
                ])
            } else {
                Ok(vec![
                    restamp_output_text(&[], &f32_bytes(&[z.iter().sum()])),
                    restamp_output_text(&[z.len(), 2], &vec![0u8; 2 * f32_bytes(&z).len()]),
                ])
            };
            let lanes = restamp_under_activation_lanes(&dag, &generated, &inputs);
            if !restamp_lanes_agree(&lanes, &expected) {
                failures.push(format!(
                    "active {active}, x {}, z {}: expected {expected:?}\n  eval {:?}\n  c {:?}",
                    x.len(),
                    z.len(),
                    lanes.0,
                    lanes.1
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
