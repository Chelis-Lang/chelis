//! spec/08 §2, spec/04 §4.7, and span-survival §2.4: private kernel
//! implementation sharing preserves each site's checks and attribution.
mod support;

use chelis_dag_core::{Dag, DimInfo, RiscOp, TensorType};
use chelis_ir::ConcreteHostType as Ty;
use chelis_ir::host::{
    ConcreteHostBinding as Binding, ConcreteHostExpr as Expr, ConcreteHostExprKind as Kind,
    ConcreteHostFunction as Function, ConcreteHostParam as Param, ConcreteHostProgram as Program,
    HostFunctionOrigin, HostTensorHelper, HostTensorInput,
};
use chelis_numeric_core::prim::Prim;
use std::{
    fs,
    process::{Command, Output},
};

fn tensor(prim: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision: prim,
    }
}

fn helper(site: usize, prim: Prim, op: RiscOp) -> HostTensorHelper {
    let ty = tensor(prim);
    let mut dag = Dag::new();
    let owner = dag.declare(format!("site_{site}"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        Some(format!("site.{site}.load")),
    );
    let result = dag.add_node(
        owner,
        op,
        vec![x],
        ty.clone(),
        Some(format!("site.{site}.op")),
    );
    dag.add_root(result);
    HostTensorHelper {
        name: format!("site_{site}"),
        dag,
        inputs: vec![HostTensorInput {
            name: "x".into(),
            ty: ty.clone(),
        }],
        output: ty,
        specialization: None,
        summary_rejection: None,
    }
}

fn program(count: usize) -> Program {
    program_for(Prim::F32, count)
}

fn program_for(prim: Prim, count: usize) -> Program {
    let ty = Ty::Tensor(tensor(prim));
    let mut previous = "x".to_string();
    let mut bindings = Vec::new();
    for site in 0..count {
        let name = format!("value_{site}");
        bindings.push(Binding {
            name: name.clone(),
            display_name: Some(name.clone()),
            display_roots: vec![],
            ty: ty.clone(),
            value: Expr::new(Kind::TensorCall {
                helper: site,
                args: vec![Expr::new(Kind::Var(previous, ty.clone()))],
                ty: ty.clone(),
            }),
        });
        previous = name;
    }
    Program {
        functions: vec![Function {
            name: "forward".into(),
            params: vec![Param {
                name: "x".into(),
                ty: ty.clone(),
            }],
            ret_ty: ty.clone(),
            body: Expr::new(Kind::Let {
                bindings,
                body: Box::new(Expr::new(Kind::Var(previous, ty.clone()))),
                ty,
            }),
            tensor_helpers: (0..count)
                .map(|site| helper(site, prim, RiscOp::Neg))
                .collect(),
            helper_result_claim_axes: vec![],
            result_claim: None,
            entry_claims: vec![],
            entry_contract: Default::default(),
            origin: HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: vec![],
        }],
        ..Program::default()
    }
}

fn implementations(source: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            line.starts_with("static __attribute__((noinline)) void __chelis_tensor_impl_")
        })
        .count()
}

#[test]
fn increasing_extraction_sites_share_one_implementation() {
    for count in [1, 8, 32] {
        let source = support::emit_host_program(&program(count), "sharing").unwrap();
        assert_eq!(implementations(&source), 1, "{count} extraction sites");
        for site in 0..count {
            assert!(source.contains(&format!("// span: site.{site}.op")));
            assert!(source.contains(&format!("chelis_fn_666f7277617264__tensor_{site}__private")));
        }
    }
}

#[test]
fn different_operations_keep_distinct_implementations() {
    let mut program = program(2);
    program.functions[0].tensor_helpers[1] = helper(1, Prim::F32, RiscOp::Abs);
    let source = support::emit_host_program(&program, "sharing").unwrap();
    assert_eq!(implementations(&source), 2);
}

#[test]
fn different_constants_dtypes_and_input_labels_are_not_interned() {
    let mut program = program(2);
    for (site, value) in [1.0, 2.0].into_iter().enumerate() {
        let helper = &mut program.functions[0].tensor_helpers[site];
        let mut dag = Dag::new();
        let owner = dag.declare("offset");
        let ty = tensor(Prim::F32);
        let x = dag.add_node(
            owner,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty.clone(),
            None,
        );
        let constant = dag.add_node(
            owner,
            RiscOp::synth_const(Prim::F32, value),
            vec![],
            ty.clone(),
            None,
        );
        let out = dag.add_node(owner, RiscOp::Add, vec![x, constant], ty, None);
        dag.add_root(out);
        helper.dag = dag;
    }
    let source = support::emit_host_program(&program, "sharing").unwrap();
    assert_eq!(implementations(&source), 2, "distinct constant bits");

    let mut f64_function = program_for(Prim::F64, 1).functions.remove(0);
    f64_function.name = "double".into();
    program.functions.push(f64_function);
    let source = support::emit_host_program(&program, "sharing").unwrap();
    assert_eq!(implementations(&source), 3, "distinct storage dtypes");

    let mut other = program_for(Prim::F32, 1).functions.remove(0);
    other.name = "other".into();
    let helper = &mut other.tensor_helpers[0];
    helper.inputs[0].name = "other_input".into();
    let mut dag = Dag::new();
    let owner = dag.declare("other");
    let ty = tensor(Prim::F32);
    let x = dag.add_node(
        owner,
        RiscOp::Load {
            name: "other_input".into(),
        },
        vec![],
        ty.clone(),
        None,
    );
    let out = dag.add_node(owner, RiscOp::Neg, vec![x], ty, None);
    dag.add_root(out);
    helper.dag = dag;
    program.functions.push(other);
    let source = support::emit_host_program(&program, "sharing").unwrap();
    assert_eq!(
        implementations(&source),
        4,
        "distinct diagnostic input labels"
    );
}

#[test]
fn sharing_is_byte_deterministic() {
    let program = program(8);
    let first = support::emit_host_program(&program, "sharing").unwrap();
    assert_eq!(
        first,
        support::emit_host_program(&program, "sharing").unwrap()
    );
}

fn compile(source: &str, driver: &str) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
    fs::write(dir.path().join("kernel.c"), source).unwrap();
    fs::write(
        dir.path().join("main.c"),
        format!("#include \"kernel.c\"\n{driver}"),
    )
    .unwrap();
    let toolchain = chelis_backend_c::toolchain::test_toolchain(Default::default());
    let binary = dir.path().join("probe");
    let output = Command::new(toolchain.compiler)
        .current_dir(dir.path())
        .args([
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            "main.c",
        ])
        .args(toolchain.compile_flags)
        .arg(staged.archive)
        .args(toolchain.link_flags)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    (dir, binary)
}

fn run(binary: &std::path::Path, args: &[&str]) -> Output {
    Command::new(binary)
        .args(args)
        .env("ASAN_OPTIONS", "detect_leaks=0")
        .output()
        .unwrap()
}

#[test]
fn compiled_shared_sites_match_eval() {
    use chelis_ir::eval::{TensorValue, eval_tensor};
    let program = program(8);
    let mut value = TensorValue::from_vec(vec![4], vec![1.0, -2.0, 3.0, -4.0]);
    for helper in &program.functions[0].tensor_helpers {
        let mut inputs = chelis_unord::UnordMap::new();
        inputs.insert("x".to_string(), value);
        let outputs = eval_tensor(&helper.dag, &inputs).unwrap();
        value = outputs[&helper.dag.roots()[0]].clone();
    }
    let expected = value
        .to_f64_lossy_vec()
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let source = support::emit_host_program(&program, "sharing").unwrap();
    let (_dir, binary) = compile(
        &source,
        &format!(
            r#"
int main(void) {{
    int64_t shape[] = {{4}};
    chelis_tensor *input = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *write = chelis_tensor_begin_write(input);
    chelis_write_view view = chelis_tensor_write_view(write);
    float values[] = {{1, -2, 3, -4}};
    memcpy(view.data, values, sizeof(values));
    chelis_tensor_end_write(write);
    chelis_tensor *output = chelis_fn_666f7277617264(input);
    chelis_read_view result = chelis_tensor_read_view(output);
    float expected[] = {{{expected}}};
    assert(result.dtype == CHELIS_DTYPE_F32);
    assert(chelis_tensor_rank(output) == 1 && chelis_tensor_shape(output, 0) == 4);
    for (int i = 0; i < 4; ++i) assert(((const float *)result.data)[i] == expected[i]);
    chelis_tensor_release(output);
    chelis_tensor_release(input);
    return 0;
}}
"#
        ),
    );
    let output = run(&binary, &[]);
    assert!(output.status.success(), "{output:?}");
}

#[test]
fn site_diagnostics_and_result_claims_survive_sharing() {
    let mut program = program(2);
    // Uncalled helpers retain their own entry checks. All four helpers have
    // the same emitted entry schedule and must share.
    program.functions[0].tensor_helpers.extend([
        helper(2, Prim::F32, RiscOp::Neg),
        helper(3, Prim::F32, RiscOp::Neg),
    ]);
    let source = support::emit_host_program(&program, "sharing").unwrap();
    assert_eq!(implementations(&source), 1);
    let (_dir, binary) = compile(
        &source,
        r#"
int main(int argc, char **argv) {
    assert(argc == 3);
    int mode = atoi(argv[1]);
    int site = atoi(argv[2]);
    int64_t shape[] = {mode == 8 ? 3 : 4};
    chelis_tensor *input = chelis_alloc(mode == 7 ? 0 : 1, shape, mode == 6 ? CHELIS_DTYPE_I32 : CHELIS_DTYPE_F32);
    chelis_tensor *inputs[] = {mode == 5 ? NULL : input};
    chelis_tensor *outputs[] = {NULL};
    __chelis_host_result_axis axis = {0, mode == 9 ? 5 : 4, NULL, NULL, 0};
    __chelis_host_result_claim frame = {0};
    frame.rank = 1; frame.count = 1; frame.axes = &axis;
    const __chelis_host_result_claim *claims[] = {&frame};
    void (*helper)(chelis_tensor **, int, chelis_tensor **, int, const __chelis_host_result_claim *const *) =
        site == 2 ? chelis_fn_666f7277617264__tensor_2__private : chelis_fn_666f7277617264__tensor_3__private;
    helper(mode == 2 ? NULL : inputs, mode == 1 ? 0 : 1, mode == 4 ? NULL : outputs, mode == 3 ? 0 : 1, claims);
    assert(mode == 0);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(input);
    return 0;
}
"#,
    );
    for site in ["2", "3"] {
        assert!(run(&binary, &["0", site]).status.success());
        for (mode, diagnostic) in [
            ("1", "expected 1 inputs, got 0"),
            ("2", "inputs array is NULL"),
            ("3", "expected 1 outputs, got 0"),
            ("4", "outputs array is NULL"),
            ("5", "input `x` at slot 0 is NULL"),
            ("6", "expected dtype f32, got i32"),
            ("7", "expected rank 1, got 0"),
            ("8", "axis 0 expected 4, got 3"),
            ("9", "numeric trap: domain in neg at i64"),
        ] {
            let output = run(&binary, &[mode, site]);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success(),
                "site {site}, mode {mode}: {output:?}"
            );
            assert!(
                stderr.contains(diagnostic),
                "site {site}, mode {mode}: {stderr}"
            );
            if mode != "9" {
                assert!(
                    stderr.contains(&format!(
                        "chelis_fn_666f7277617264__tensor_{site}__private:"
                    )),
                    "{stderr}"
                );
            }
        }
    }
}
