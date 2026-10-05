//! Evaluator-agreement tests: compare canonical eval renderings with canonical
//! C codegen+compile+run renderings through chelis#732's one Phase 3
//! comparator.
//!
//! WS-1 (dtype + Metal cleanup cycle) extension: bf16 / f16 cases are
//! added below. The HIP cross-validation lives behind the
//! `hip-local-gpu` feature flag (`cargo test -p chelis-e2e --features
//! hip-local-gpu`); the default-feature Linux CI runner skips them
//! because hipcc / libhipblas are not present.
//!
//! ## Exact-string oracle (chelis#687, chelis#732 Phase 3)
//!
//! The old `eval_last -> f64`, `parse_c_output -> f64`, and ad-hoc
//! `assert_close` path is gone. Eval values are rendered at the result
//! dtype through `format_element`; generated C is rendered through the
//! same Phase 2 helper used by shipped roots. The byte-equal-or-table-
//! bounded decision then comes only from `chelis_types::agreement`.
//!
//! This file deliberately gains no i64-above-2^53 row: at this DAG level
//! `RiscOp::Const { value: f64 }` cannot express it (chelis#684), so that
//! row would test the wrong layer. The exact-integer cross-lane oracle is:
//! `crates/chelis-cli/tests/precision_matrix.rs` (`eval_lane_str` /
//! `c_lane_str`, verbatim strings) and
//! `crates/chelis-cli/tests/issue_680_int_exactness.rs` (`eval_int` /
//! `parse_out_binding`, exact `i64` parses). Do not add integer rows to
//! THIS file; add them there.
//!
//! [04-NUM-8] eligibility is explicit. chelis#897 records that the current
//! evaluator computes float operations in f64, so any byte difference in
//! this harness is ineligible for the tolerance table and must remain a
//! named issue-linked failure. Byte equality is still meaningful. When
//! #897 lands, the status below changes in the same change set as its
//! arithmetic-width oracle.

use chelis_unord::UnordMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use chelis_ir::dag::{Dag, DimInfo, NodeId, RiscOp, TensorType};
use chelis_ir::eval::eval_tensor;
use chelis_types::agreement::{
    AgreementError, AgreementOp, AgreementOutcome, ArithmeticWidthStatus,
    compare_exact_observations, compare_rendered_elements,
};
use chelis_types::observation::{ElementRef, format_element};
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

fn c_render_result(prim: Prim) -> &'static str {
    match prim {
        Prim::F32
        | Prim::F64
        | Prim::F16
        | Prim::Bf16
        | Prim::Int8
        | Prim::Int16
        | Prim::Int32
        | Prim::Int64
        | Prim::Bool => {
            "chelis_scalar scalar = chelis_tensor_to_scalar(outputs[0]); \
             chelis_string text = chelis_string_from_scalar(scalar); \
             chelis_print_string(text); chelis_string_release(text);"
        }
        Prim::F8e4m3 | Prim::String | Prim::Key => {
            panic!("eval agreement has no C renderer for {}", prim.name())
        }
    }
}

/// Build a DAG, generate C, compile, run, and return the canonical rendered
/// result element. The exhaustive Rust match above admits every active scalar
/// dtype, then the exact tagged runtime carrier owns dtype-correct extraction
/// and canonical rendering instead of an ad-hoc decimal `printf`.
fn compile_and_run(dag: &Dag, func_name: &str) -> String {
    let selected = chelis_backend_c::prepare_dag_for_codegen(
        dag.clone(),
        chelis_backend_c::CodegenOptions::default(),
    );
    let verified = chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(selected).unwrap(),
    )
    .unwrap();
    let result = chelis_backend_c::codegen(verified, func_name).unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(tmp.path())
        .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    write_temp_file(tmp.path(), "model.c", &result.c_source);

    let prim = dag
        .nodes()
        .last()
        .expect("agreement DAG has a result")
        .output_type
        .precision;
    let render_result = c_render_result(prim);
    let main_c = format!(
        r#"
#include "chelis_runtime.h"
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    {render_result}
    printf("\n");
    chelis_tensor_release(outputs[0]);
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
    cmd.arg(&staged.archive);
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

fn eval_last_rendered(dag: &Dag) -> (Prim, String) {
    let inputs = UnordMap::new();
    let vals = eval_tensor(dag, &inputs).unwrap();
    let last_id = NodeId(dag.len() - 1);
    let declared = dag
        .nodes()
        .last()
        .expect("agreement DAG has a result")
        .output_type
        .precision;
    let value = &vals[&last_id];
    // chelis#729 Phase 1/2 storage is dtype-tagged, so the observation
    // element is read at its own width. Widening through f64 here would
    // reintroduce exactly the lossy channel this oracle exists to detect.
    let prim = value.prim();
    assert_eq!(
        prim,
        declared,
        "eval stored {} for a node the DAG declares {}; the C lane renders \
         the declared dtype, so a substitution here would be compared against \
         the wrong width",
        prim.name(),
        declared.name()
    );
    (prim, format_element(prim, value.storage().element_ref(0)))
}

fn arithmetic_width_status(prim: Prim) -> ArithmeticWidthStatus {
    if prim.is_float() {
        ArithmeticWidthStatus::Nonconforming { issue: 897 }
    } else {
        ArithmeticWidthStatus::StoredAtArithmeticWidth
    }
}

/// Map the operation that actually produces the observed result into the
/// comparator's closed identity. This match is deliberately exhaustive: a
/// future RISC variant must decide whether it is exact or has numbered-spec
/// authority for a tolerance before this harness compiles again.
fn agreement_op_for_risc(op: &RiscOp) -> AgreementOp {
    match op {
        // [05-OP-46] makes the transcendentals and `sqrt` correctly rounded,
        // so the lanes agree exactly.
        RiscOp::Atan
        | RiscOp::Cos
        | RiscOp::Exp
        | RiscOp::Log
        | RiscOp::Sin
        | RiscOp::Sqrt
        | RiscOp::Tan
        | RiscOp::Tanh
        // [05-OP-48] retains softmax's exact primitive graph after AD.
        | RiscOp::Softmax { .. }
        | RiscOp::Erf
        | RiscOp::Erfc
        | RiscOp::Add
        | RiscOp::Sub
        | RiscOp::Mul
        | RiscOp::Div
        | RiscOp::FloorDiv
        | RiscOp::TruncDiv
        | RiscOp::Mod
        | RiscOp::Bitwise(_)
        | RiscOp::Compare(_)
        | RiscOp::Logical(_)
        | RiscOp::Where
        // chelis#1464 / [05-OP-68]: the guard carries the fallback's stored
        // bits unchanged when it does not fire, so it is exact by
        // construction; when it does fire there is no value to compare.
        | RiscOp::GuardedFail { .. }
        | RiscOp::MaxElem
        | RiscOp::MinElem
        | RiscOp::ExtremaAdjoint { .. }
        | RiscOp::Relu
        | RiscOp::ReluAdjoint
        | RiscOp::Neg
        | RiscOp::Abs
        | RiscOp::Floor
        | RiscOp::Ceil
        | RiscOp::Round
        | RiscOp::Recip
        // [05-RNG-1] makes every random result bit-identical across lanes,
        // a draw key is a word no lane observes as a result, and [05-RNG-2]
        // defines every key derivation bit for bit; a branch's join selects
        // one of two such keys.
        | RiscOp::UniformLike
        | RiscOp::Dropout
        | RiscOp::DropoutReplay
        | RiscOp::UniformBoundAdjoint { .. }
        | RiscOp::KeyFromSeed
        | RiscOp::Split { .. }
        | RiscOp::FoldIn
        | RiscOp::SplitN { .. }
        | RiscOp::KeySelect
        | RiscOp::Sum { .. }
        | RiscOp::Count { .. }
        // [05-OP-54]: runtime range materializes exact signed i64 elements.
        | RiscOp::Iota
        | RiscOp::ListMapCapture { .. }
        | RiscOp::OrderedAdjointSum { .. }
        | RiscOp::MaxReduce { .. }
        | RiscOp::MinReduce { .. }
        | RiscOp::ProdReduce { .. }
        | RiscOp::ReduceWindow { .. }
        | RiscOp::ReduceWindowGrad { .. }
        | RiscOp::Argmax { .. }
        | RiscOp::Argmin { .. }
        | RiscOp::Reshape { .. }
        | RiscOp::Permute { .. }
        | RiscOp::Expand { .. }
        | RiscOp::OneHot { .. }
        | RiscOp::Pad { .. }
        | RiscOp::Shrink { .. }
        | RiscOp::Stride { .. }
        | RiscOp::Shape { .. }
        | RiscOp::ExtentWitness { .. }
        | RiscOp::CheckedReshapeExtent { .. }
        | RiscOp::CheckedUnitAxis { .. }
        | RiscOp::Const { .. }
        | RiscOp::ConstTensor { .. }
        | RiscOp::Load { .. }
        | RiscOp::Store { .. }
        | RiscOp::Copy
        | RiscOp::Drop
        | RiscOp::Realize
        | RiscOp::Cast { .. }
        | RiscOp::NamedCast { .. }
        | RiscOp::FusedElem { .. }
        | RiscOp::BlasMatmul { .. }
        | RiscOp::Gather { .. }
        | RiscOp::ScatterAdd { .. }
        | RiscOp::Scatter { .. }
        | RiscOp::ScatterElements { .. } => AgreementOp::Exact,
    }
}

fn record_phase3_receipt(case: &str, detail: &str) {
    let Ok(path) = std::env::var("CHELIS_PHASE3_RECEIPT_PATH") else {
        return;
    };
    let mut receipt = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open Phase 3 receipt file");
    // One buffer, one append. `writeln!` can emit the case, the separator and
    // the detail as separate `write` calls, and nextest runs each of these
    // tests in its own process, so a split write interleaves two receipts into
    // one corrupt line.
    let line = format!("{case}\t{detail}\n");
    receipt
        .write_all(line.as_bytes())
        .expect("append Phase 3 receipt");
}

fn compare_lanes_with<F>(
    dag: &Dag,
    func_name: &str,
    label: &str,
    compiled_transform: F,
) -> Result<(String, AgreementOutcome), AgreementError>
where
    F: FnOnce(String) -> String,
{
    let (prim, eval) = eval_last_rendered(dag);
    let compiled = compiled_transform(compile_and_run(dag, func_name));
    let result_op =
        agreement_op_for_risc(&dag.nodes().last().expect("agreement DAG has a result").op);
    let outcome = compare_rendered_elements(
        result_op,
        prim,
        arithmetic_width_status(prim),
        &eval,
        &compiled,
    )?;
    record_phase3_receipt(
        label,
        &format!(
            "op={}\tprim={}\teval={}\tcompiled={}",
            result_op.name(),
            prim.name(),
            eval,
            compiled
        ),
    );
    Ok((eval, outcome))
}

fn assert_agrees(dag: &Dag, func_name: &str, label: &str) -> String {
    compare_lanes_with(dag, func_name, label, |compiled| compiled)
        .map(|(eval, _)| eval)
        .unwrap_or_else(|error| panic!("{label}: {error}"))
}

fn assert_expected(label: &str, actual: &str, expected: &str) {
    compare_exact_observations(label, expected, actual)
        .unwrap_or_else(|error| panic!("{label}: {error}"));
}

#[test]
fn agreement_operation_identity_is_derived_from_ir() {
    let cases = [
        (RiscOp::Add, AgreementOp::Exact),
        // [05-OP-54]: i64 range elements agree bit for bit across lanes.
        (RiscOp::Iota, AgreementOp::Exact),
        (RiscOp::Exp, AgreementOp::Exact),
        (RiscOp::Log, AgreementOp::Exact),
        (RiscOp::Sin, AgreementOp::Exact),
        (RiscOp::Sqrt, AgreementOp::Exact),
        (RiscOp::Cos, AgreementOp::Exact),
        (RiscOp::Tan, AgreementOp::Exact),
        (RiscOp::Atan, AgreementOp::Exact),
        (RiscOp::Tanh, AgreementOp::Exact),
        (RiscOp::Softmax { axis: 0 }, AgreementOp::Exact),
        (RiscOp::Erf, AgreementOp::Exact),
        (RiscOp::Erfc, AgreementOp::Exact),
    ];
    for (risc, expected) in cases {
        assert_eq!(agreement_op_for_risc(&risc), expected);
    }
    record_phase3_receipt("operation-identity-canary", "closed exhaustive IR mapping");
}

#[test]
fn agreement_compiled_observation_reaches_comparator() {
    if !gcc_available() {
        panic!("Phase 3 compiled-observation canary requires a host C compiler");
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    dag.add_node(
        decl,
        RiscOp::synth_const(Prim::Int32, 7.0),
        vec![],
        scalar_ty(Prim::Int32),
        None,
    );

    let error = compare_lanes_with(
        &dag,
        "test_compiled_observation_canary",
        "compiled-observation-canary-unexpected-pass",
        |_| "8".to_string(),
    )
    .expect_err("the perturbed compiled lane must reach the exact comparator");
    assert!(matches!(
        error,
        AgreementError::ExactMismatch {
            reference,
            candidate,
            ..
        } if reference == "7" && candidate == "8"
    ));
    record_phase3_receipt(
        "compiled-observation-canary",
        "eval=7\tcompiled=8\terror=ExactMismatch",
    );
}

/// chelis#1104: the verbatim expected-value leg's usage canary, and the
/// counterpart of the compiled-observation canary above.
///
/// Every value row states its result as a verbatim string through the shared
/// `assert_expected`. Because that helper is shared, a body that stopped
/// consulting `compare_exact_observations` would delete the expected-value leg
/// from all of them at once without editing one digest-frozen test definition.
/// Source-level guards cannot see that: they check that the comparator is
/// *named* in this file, not that it is *invoked*. So this canary drives the
/// shipped helper rather than the comparator directly, and pins which argument
/// takes the reference role: a known-wrong expected value must still reject,
/// with the exact comparator's own `ExactMismatch` rendering.
#[test]
fn agreement_expected_value_reaches_comparator() {
    let rejection =
        std::panic::catch_unwind(|| assert_expected("expected-value-canary", "8.0", "7.0"))
            .expect_err("a wrong expected value must reach the exact comparator");
    let message = rejection
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| rejection.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    let expected_rejection = AgreementError::ExactMismatch {
        context: "expected-value-canary".to_string(),
        reference: "7.0".to_string(),
        candidate: "8.0".to_string(),
    };
    assert_eq!(
        message,
        format!("expected-value-canary: {expected_rejection}")
    );

    assert_expected("expected-value-canary", "7.0", "7.0");

    record_phase3_receipt(
        "expected-value-canary",
        "expected=7.0\tactual=8.0\terror=ExactMismatch",
    );
}

#[test]
fn agreement_width_nonconformance_is_behavioral() {
    let eval = format_element(Prim::F32, ElementRef::F32(1.0));
    let compiled = format_element(
        Prim::F32,
        ElementRef::F32(f32::from_bits(1.0_f32.to_bits() + 1)),
    );
    let error = compare_rendered_elements(
        AgreementOp::Exact,
        Prim::F32,
        arithmetic_width_status(Prim::F32),
        &eval,
        &compiled,
    )
    .expect_err("a known arithmetic-width violation cannot pass as a value difference");
    assert!(matches!(
        error,
        AgreementError::ArithmeticWidthNonconforming { issue: 897, .. }
    ));
    record_phase3_receipt(
        "width-nonconformance-canary",
        "op=exact\terror=ArithmeticWidthNonconforming\tissue=897",
    );
}

#[test]
fn agreement_add() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 3.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 4.0),
        vec![],
        scalar_f32(),
        None,
    );
    dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);

    let result = assert_agrees(&dag, "test_add", "add(3,4)");
    assert_expected("add(3,4) expected", &result, "7.0");
}

#[test]
fn agreement_mul() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 5.0),
        vec![],
        scalar_f32(),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 6.0),
        vec![],
        scalar_f32(),
        None,
    );
    dag.add_node(decl, RiscOp::Mul, vec![a, b], scalar_f32(), None);

    let result = assert_agrees(&dag, "test_mul", "mul(5,6)");
    assert_expected("mul(5,6) expected", &result, "30.0");
}

#[test]
fn agreement_neg() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 7.0),
        vec![],
        scalar_f32(),
        None,
    );
    dag.add_node(decl, RiscOp::Neg, vec![a], scalar_f32(), None);

    let result = assert_agrees(&dag, "test_neg", "neg(7)");
    assert_expected("neg(7) expected", &result, "-7.0");
}

#[test]
fn agreement_relu() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }

    // Dedicated ReLU identity with negative input.
    {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Relu, vec![x], scalar_f32(), None);

        let result = assert_agrees(&dag, "test_relu_neg", "relu(-2)");
        assert_expected("relu(-2) expected", &result, "0.0");
    }

    // relu with positive input
    {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Relu, vec![x], scalar_f32(), None);

        let result = assert_agrees(&dag, "test_relu_pos", "relu(3)");
        assert_expected("relu(3) expected", &result, "3.0");
    }
}

#[test]
fn agreement_exp() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, 0.0),
        vec![],
        scalar_f32(),
        None,
    );
    dag.add_node(decl, RiscOp::Exp, vec![a], scalar_f32(), None);

    let result = assert_agrees(&dag, "test_exp", "exp(0)");
    assert_expected("exp(0) expected", &result, "1.0");
}

fn assert_unary_transcendental(op: RiscOp, op_name: &str, input: f64, expected: &str) {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let argument = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_f32().precision, input),
        vec![],
        scalar_f32(),
        None,
    );
    dag.add_node(decl, op, vec![argument], scalar_f32(), None);
    let func_name = format!("test_{op_name}");
    let label = format!("{op_name}({input})");
    let result = assert_agrees(&dag, &func_name, &label);
    assert_expected(&format!("{label} expected"), &result, expected);
}

#[test]
fn agreement_log() {
    assert_unary_transcendental(RiscOp::Log, "log", 1.0, "0.0");
}

#[test]
fn agreement_sin() {
    assert_unary_transcendental(RiscOp::Sin, "sin", 0.0, "0.0");
}

#[test]
fn agreement_sqrt_is_exact() {
    assert_unary_transcendental(RiscOp::Sqrt, "sqrt", 4.0, "2.0");
}

#[test]
fn agreement_cos() {
    assert_unary_transcendental(RiscOp::Cos, "cos", 0.0, "1.0");
}

#[test]
fn agreement_tan() {
    assert_unary_transcendental(RiscOp::Tan, "tan", 0.0, "0.0");
}

#[test]
fn agreement_atan() {
    assert_unary_transcendental(RiscOp::Atan, "atan", 0.0, "0.0");
}

/// WS-1: bf16 add agrees exactly with the evaluator. Both operands and the
/// result are representable, and `add` has no [05-OBS-3] tolerance row.
#[test]
fn agreement_bf16_add() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_ty(Prim::Bf16).precision, 1.5),
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_ty(Prim::Bf16).precision, 2.5),
        vec![],
        scalar_ty(Prim::Bf16),
        None,
    );
    dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_ty(Prim::Bf16), None);
    let result = assert_agrees(&dag, "test_bf16_add", "bf16 add(1.5, 2.5)");
    assert_expected("bf16 add expected", &result, "4.0");
}

#[test]
fn agreement_f16_add() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_ty(Prim::F16).precision, 1.5),
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::synth_const(scalar_ty(Prim::F16).precision, 2.5),
        vec![],
        scalar_ty(Prim::F16),
        None,
    );
    dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_ty(Prim::F16), None);
    let result = assert_agrees(&dag, "test_f16_add", "f16 add(1.5, 2.5)");
    assert_expected("f16 add expected", &result, "4.0");
}

#[test]
fn agreement_bf16_reduce_sum_matches_eval_exactly() {
    if !gcc_available() {
        eprintln!("skipping: gcc not available");
        return;
    }
    // Build a fused Const + reduce_sum so the comparison stays
    // self-contained (no Load to thread inputs through the runtime).
    let n = 8;
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let c = dag.add_node(
        decl,
        RiscOp::synth_const(vec_ty(n, Prim::Bf16).precision, 0.25),
        vec![],
        vec_ty(n, Prim::Bf16),
        None,
    );
    let sum = RiscOp::sum_default(0, Prim::Bf16).expect("sum constructs");
    dag.add_node(decl, sum, vec![c], scalar_ty(Prim::F32), None);
    let result = assert_agrees(&dag, "test_bf16_sum_const", "bf16 reduce_sum(0.25 x 8)");
    assert_expected("bf16 reduce_sum expected", &result, "2.0");
}
