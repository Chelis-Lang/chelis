//! chelis#309: multi-target gradients preserve result order and ABI arity.
//! spec/04 §6 tuple-get and spec/06 §§2.1–2.2 permit static projection;
//! a projected result need not materialize a host tuple. Execute projections
//! and retain the two-output host-carrier check on an unprojected gradient.

use std::{collections::BTreeMap, fs, process::Command};

use chelis_backend_c::GeneratedHeader;
use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{
    CompileRequest, CompileTarget, EvalRequest, ExecutionValue, SourceKind,
};
use chelis_types::types::Prim;

const LOSS: &str = r#"
def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(mul(x, w), 0i32))
"#;

fn source(body: &str, result: &str) -> String {
    format!("{LOSS}\ndef dloss(x: tensor[2, f32], w: tensor[2, f32]) -> {result} = {body}\n")
}

fn compile_c(source: &str) -> (String, String) {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("repro".into()),
    })
    .unwrap_or_else(|err| panic!("compile failed: {err:?}"));
    let file = |path: &str| {
        result
            .files
            .iter()
            .find(|file| file.path == path)
            .unwrap_or_else(|| panic!("missing {path}"))
            .contents
            .clone()
    };
    (file("repro.c"), file("repro.h"))
}

/// Compile the published function and use its generated declaration metadata.
/// Native assertions check exact values, dtype, rank, extent, and element count.
fn run_c(source: &str, driver: impl FnOnce(&str) -> String) -> String {
    let (c, header) = compile_c(source);
    let declarations = GeneratedHeader::parse(&header).unwrap();
    declarations.validate_source(&c).unwrap();
    let symbol = declarations.declaration("dloss").unwrap().symbol();
    let dir = tempfile::tempdir().unwrap();
    let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
    fs::write(dir.path().join("repro.h"), &header).unwrap();
    let program = format!(
        "#define main unused_generated_main\n{c}\n#undef main\n{DRIVER_SUPPORT}\n{}",
        driver(symbol)
    );
    let c_path = dir.path().join("probe.c");
    fs::write(&c_path, program).unwrap();
    let binary = dir.path().join("probe");
    let mut cc = Command::new("cc");
    cc.args(["-std=c11", "-O0", "-Werror=incompatible-pointer-types"])
        .arg(c_path)
        .arg("-I")
        .arg(dir.path())
        .arg(staged.archive)
        .args(["-lm", "-lpthread"]);
    if cfg!(target_os = "macos") {
        cc.args([
            "-framework",
            "Accelerate",
            "-framework",
            "Security",
            "-framework",
            "CoreFoundation",
        ]);
    } else {
        cc.arg("-ldl");
    }
    let output = cc.arg("-o").arg(&binary).output().expect("compile C");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(binary).output().expect("execute C");
    assert!(
        output.status.success(),
        "{}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    c
}

fn eval_result(source: &str) -> ExecutionValue {
    let mut result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: format!("{source}\nx = to_tensor([2.0f32, 3.0f32])\nw = to_tensor([5.0f32, 7.0f32])\nout = dloss(x, w)\n"),
        bindings: BTreeMap::new(),
    }).unwrap();
    std::mem::take(&mut result.roots)
        .into_iter()
        .find(|root| root.name.as_deref() == Some("out"))
        .expect("out root")
        .value
}

fn assert_tensor(value: &ExecutionValue, expected: &[f64]) {
    let ExecutionValue::Tensor { value } = value else {
        panic!("expected tensor: {value:?}")
    };
    assert_eq!(value.shape, vec![2]);
    assert_eq!(value.data.prim(), Prim::F32);
    assert_eq!(value.data.to_f64_lossy_vec(), expected);
}

fn assert_gradient(body: &str, expected: [f64; 2]) {
    let source = source(body, "tensor[2, f32]");
    assert_tensor(&eval_result(&source), &expected);
    run_c(&source, |symbol| {
        format!(
            r#"
int main(void) {{
    chelis_tensor *x = input(2.0f, 3.0f), *w = input(5.0f, 7.0f);
    chelis_tensor *result = {symbol}(x, w);
    tensor_values(result, {:?}f, {:?}f);
    chelis_tensor_release(result);
    chelis_tensor_release(x);
    chelis_tensor_release(w);
    return 0;
}}
"#,
            expected[0], expected[1]
        )
    });
}

#[test]
fn issue309_multi_wrt_grad_projections_match_native_and_eval() {
    // Unequal derivatives catch swapped slots and accidental projection 0.
    assert_gradient("(grad(loss)(x, w)).0", [5.0, 7.0]);
    assert_gradient("(grad(loss)(x, w)).1", [2.0, 3.0]);
}

#[test]
fn issue309_single_wrt_grad_call_stays_single_tensor() {
    assert_gradient("grad(loss, wrt=x)(x, w)", [5.0, 7.0]);
    assert_gradient("grad(loss, wrt=w)(x, w)", [2.0, 3.0]);
}

#[test]
fn issue309_multi_wrt_grad_projection_reads_from_real_tuple() {
    let source = source("grad(loss)(x, w)", "(tensor[2, f32], tensor[2, f32])");
    let c = run_c(&source, |symbol| {
        format!(
            r#"
int main(void) {{
    chelis_tensor *x = input(2.0f, 3.0f), *w = input(5.0f, 7.0f);
    chelis_tuple *result = {symbol}(x, w);
    chelis_tensor *dx = chelis_tensor_take_value(chelis_tuple_get(result, 0));
    chelis_tensor *dw = chelis_tensor_take_value(chelis_tuple_get(result, 1));
    tensor_values(dx, 5.0f, 7.0f);
    tensor_values(dw, 2.0f, 3.0f);
    chelis_tensor_release(dx);
    chelis_tensor_release(dw);
    chelis_tuple_release(result);
    chelis_tensor_release(x);
    chelis_tensor_release(w);
    return 0;
}}
"#
        )
    });
    // This fixture must actually exercise #309's multi-output host carrier.
    assert!(c.contains("chelis_tuple_from_values"), "{c}");
    assert!(c.contains("chelis_value_take_tensor"), "{c}");
}

#[test]
fn issue309_multi_wrt_grad_call_sizes_two_output_slots() {
    let source = source("grad(loss)(x, w)", "(tensor[2, f32], tensor[2, f32])");
    let (c, _) = compile_c(&source);
    // Keep the original host-helper arity check beside the executed receiver
    // check above; a statically projected gradient legitimately has one root.
    assert!(c.contains("if (n_out != 2)"), "{c}");
    assert!(
        c.lines()
            .any(|line| line.contains("__outputs") && line.contains("[2]")),
        "{c}"
    );
    assert!(
        c.lines().any(|line| line.contains("__tensor_")
            && line
                .rsplit(',')
                .nth(1)
                .is_some_and(|count| count.trim() == "2")),
        "{c}"
    );
}

#[test]
fn issue309_invalid_gradient_projections_are_rejected() {
    for body in ["(grad(loss)(x, w)).2", "(grad(loss, wrt=x)(x, w)).0"] {
        let result = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source(body, "tensor[2, f32]"),
            target: CompileTarget::C,
            entry_name: Some("repro".into()),
        });
        assert!(result.is_err(), "invalid projection accepted: {body}");
    }
}

const DRIVER_SUPPORT: &str = r#"
#include <assert.h>
static chelis_tensor *input(float a, float b) {
    int64_t n = 2;
    chelis_tensor *t = chelis_alloc(1, &n, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(t);
    float *data = chelis_tensor_write_view(guard).data;
    data[0] = a;
    data[1] = b;
    chelis_tensor_end_write(guard);
    return t;
}
static void tensor_values(const chelis_tensor *t, float a, float b) {
    assert(chelis_tensor_rank(t) == 1);
    assert(chelis_tensor_shape(t, 0) == 2);
    chelis_read_view view = chelis_tensor_read_view(t);
    assert(view.dtype == CHELIS_DTYPE_F32 && view.count == 2);
    const float *data = view.data;
    assert(data[0] == a && data[1] == b);
}
"#;
