//! Source/API/native acceptance for [05-OP-37] and [05-RNG-1] through a host helper.
#[path = "fixed_control_host_c/context.rs"]
mod context;
#[allow(dead_code)]
mod ownership_support;

use chelis_compiler_api::compiler::compile_for_execution;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

#[test]
fn signed_int64_host_boundaries_compile_without_literal_overflow() {
    // [04-NUM-11], [05-RNG-1], chelis#1859. Expected masks are fixed
    // ordinal-zero words from the specified seed-bit map, not emitted C.
    for (seed, literal, expected) in [
        (i64::MIN, "INT64_MIN", "-6.0f,-2.0f,0.0f,6.0f"),
        (i64::MIN + 1, "(INT64_MIN + 1)", "0.0f,-2.0f,0.0f,6.0f"),
        (-1, "-1", "-6.0f,-2.0f,2.0f,0.0f"),
        (0, "0", "0.0f,0.0f,2.0f,0.0f"),
        (i64::MAX, "INT64_MAX", "0.0f,-2.0f,0.0f,0.0f"),
    ] {
        let source = format!(
            "def boundary() -> int64 = {seed}i64\n\
             def sample(x: tensor[4,f32]) -> tensor[4,f32] = with seed({seed}i64) {{ dropout(x,0.5f32) }}"
        );
        let generated = ownership_support::emit(&source, "signed-host-boundary");
        // Make the actual bad decimal token fail compilation on both native
        // compiler families, without suppressing diagnostics elsewhere.
        let strict = format!(
            "#if defined(__clang__)\n#pragma clang diagnostic error \"-Wimplicitly-unsigned-literal\"\n\
             #elif defined(__GNUC__)\n#pragma GCC diagnostic error \"-Woverflow\"\n#endif\n\
             #define main unused_generated_main\n{generated}\n#undef main\n"
        );
        let driver = format!(
            r#"
int main(void) {{
    assert(boundary() == {literal});
    chelis_tensor *x = input(4);
    const float expected[] = {{{expected}}};
    for (int repeat = 0; repeat < 3; ++repeat) {{
        chelis_tensor *result = sample(x);
        chelis_read_view view = chelis_tensor_read_view(result);
        assert(view.dtype == CHELIS_DTYPE_F32 && view.count == 4);
        assert(memcmp(view.data, expected, sizeof(expected)) == 0);
        chelis_tensor_release(result);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#
        );
        ownership_support::balanced(&ownership_support::run(&strict, &driver));
    }
}

#[test]
fn generic_static_rate_reaches_native_host_helper() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            r#"
def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))
result = with seed(42i64) {{
  keep(to_tensor([cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype})]))
}}
"#
        );
        let c = ownership_support::emit(&source, dtype);
        let (summary, stdout) = ownership_support::run_program(&c);
        ownership_support::balanced(&summary);
        assert_eq!(
            stdout, "result = tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0])\n",
            "generic helper specialization must preserve {dtype} values"
        );
    }
}

#[test]
fn concrete_wrapper_reaches_generic_static_rate_host_helper() {
    let c = ownership_support::emit_selected(
        r#"
def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))
def run(x: tensor[4, f32]) -> tensor[4, f32] = keep(x)
result = with seed(42i64) {
  run(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
}
"#,
        "generic-wrapper",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        "result = tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0])\n"
    );
}

#[test]
fn computed_and_record_wrappers_preserve_generic_fixed_control_composition() {
    let computed = ownership_support::emit(
        r#"
def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))
result = with seed(42i64) {
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  keep(add(x, x))
}
"#,
        "generic-computed",
    );
    let (summary, stdout) = ownership_support::run_program(&computed);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        "result = tensor(shape=[4], data=[0.0, 4.0, 0.0, 0.0])\n"
    );

    let record = ownership_support::emit(
        r#"
type Inputs = | Inputs { q: tensor[4, f32] }
def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))
def run(inp: Inputs) -> tensor[4, f32] = keep(add(inp.q, inp.q))
result = with seed(42i64) {
  inp = Inputs { q: to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]) }
  a = run(inp)
  b = run(inp)
  [a, b]
}
"#,
        "generic-record-wrapper",
    );
    let (summary, stdout) = ownership_support::run_program(&record);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        "result = [tensor(shape=[4], data=[0.0, 4.0, 0.0, 0.0]), tensor(shape=[4], data=[4.0, 0.0, 0.0, 0.0])]\n"
    );
}

#[test]
fn concrete_wrappers_do_not_admit_runtime_controls_or_gpu_dropout() {
    let cases = [
        (
            CompileTarget::C,
            "def keep[p: Float](x: tensor[4,p], rate: p) -> tensor[4,p] = dropout(x, rate)\n\
             def run(x: tensor[4,f32], rate: f32) -> tensor[4,f32] = with seed(42i64) { keep(x, rate) }",
        ),
        (
            CompileTarget::C,
            "def keep(x: tensor[4,f32], seed: int64) -> tensor[4,f32] = with seed(seed) { dropout(x, 0.5f32) }\n\
             def run(x: tensor[4,f32], seed: int64) -> tensor[4,f32] = keep(x, seed)",
        ),
        (
            CompileTarget::Hip,
            "def keep[p: Float](x: tensor[4,p]) -> tensor[4,p] = with seed(42i64) { dropout(x, cast(0.5, p)) }\n\
             def run(x: tensor[4,f32]) -> tensor[4,f32] = keep(x)",
        ),
    ];
    for (target, source) in cases {
        let error = compile_for_execution(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            target,
            entry_name: Some("run".into()),
        })
        .expect_err("unsupported wrapper control must remain loud");
        let message = format!("{error:?}");
        assert!(
            message.contains("dropout")
                || message.contains("Dropout")
                || message.contains("fixed-control")
                || message.contains("requires a signed int64 literal seed"),
            "{message}"
        );
    }
}

#[test]
fn ordinary_uniform_helpers_do_not_poison_planned_dropout_admission() {
    for used in [false, true] {
        let intervening = if used {
            "u = sample(x)\n  _ = drop(u)"
        } else {
            ""
        };
        let source = format!(
            r#"
def keep[p: Float](x: tensor[4,p]) -> tensor[4,p] = dropout(x,cast(0.5,p))
def sample(x: tensor[4,f32]) -> tensor[4,f32] = uniform_like(x,0.0f32,1.0f32)
result = with seed(42i64) {{
  x = to_tensor([1.0f32,1.0f32,1.0f32,1.0f32])
  a = keep(x)
  {intervening}
  b = keep(x)
  [a,b]
}}
"#
        );
        let c = ownership_support::emit(&source, "uniform-cohabitation");
        let (summary, stdout) = ownership_support::run_program(&c);
        ownership_support::balanced(&summary);
        // [05-RNG-1] seed42: the second Dropout uses ordinal1 without
        // the intervening draw, ordinal2 with it. No claim about Uniform's
        // legacy numerical formula is made or encoded in this oracle.
        let second = if used {
            "2.0, 2.0, 0.0, 0.0"
        } else {
            "2.0, 0.0, 0.0, 0.0"
        };
        assert_eq!(
            stdout,
            format!(
                "result = [tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0]), tensor(shape=[4], data=[{second}])]\n"
            )
        );
    }
}

#[test]
fn direct_source_entry_restarts_and_preserves_borrowed_input_ownership() {
    let c = ownership_support::emit_selected(
        r#"
def main(x: tensor[4, f32]) -> tensor[4, f32] = with seed(42i64) {
  dropout(x, 0.5f32)
}
"#,
        "main",
    );
    let driver = r#"
int main(void) {
    chelis_tensor *x = input(4);
    const float input_expected[] = {-3.0f, -1.0f, 1.0f, 3.0f};
    tensor_bits(x, 4, input_expected);
    chelis_tensor *inputs[] = {x};
    chelis_tensor *first[] = {NULL};
    chelis_tensor *second[] = {NULL};
    chelis_main(inputs, 1, first, 1);
    chelis_main(inputs, 1, second, 1);
    const float expected[] = {0.0f, -2.0f, 0.0f, 0.0f};
    tensor_bits(first[0], 4, expected);
    tensor_bits(second[0], 4, expected);
    tensor_bits(x, 4, input_expected);
    chelis_tensor_release(first[0]);
    chelis_tensor_release(second[0]);
    chelis_tensor_release(x);
    return 0;
}
"#;
    ownership_support::balanced(&ownership_support::run(&c, driver));
}

#[test]
fn host_empty_and_zero_rate_draws_advance_before_the_next_helper() {
    let c = ownership_support::emit(
        r#"
result = with seed(42i64) {
  empty: tensor[0, f32] = to_tensor([])
  dead_empty = dropout(empty, 0.5f32)
  _ = drop(dead_empty)
  ones = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  dead_zero = dropout(ones, 0.0f32)
  _ = drop(dead_zero)
  dropout(ones, 0.5f32)
}
"#,
        "empty-zero-next",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        "result = tensor(shape=[4], data=[2.0, 2.0, 0.0, 0.0])\n"
    );
}

#[test]
fn host_gradient_replay_and_nested_seed_restore_share_one_rng_frame() {
    let c = ownership_support::emit(
        r#"
def loss(x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(dropout(x, 0.5f32), 0))
result = with seed(42i64) {
  ones = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  g = grad(loss)(ones)
  equal = with seed(42i64) { dropout(ones, 0.5f32) }
  different = with seed(99i64) { dropout(ones, 0.5f32) }
  next = dropout(ones, 0.5f32)
  [g, equal, different, next]
}
"#,
        "grad-nested-next",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        "result = [tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0]), tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0]), tensor(shape=[4], data=[2.0, 2.0, 2.0, 2.0]), tensor(shape=[4], data=[2.0, 0.0, 0.0, 0.0])]\n",
        "each replay, nested-scope, and restored-stream position must remain observable"
    );
}

#[test]
fn cpu_resource_gradient_compiles_and_executes_without_a_runtime_resource_action() {
    let c = ownership_support::emit_selected(
        r#"
def loss(x: tensor[4, f32]) -> f32 = with device("cpu:author-device") {
  with seed(42i64) { tensor_to_scalar(sum(dropout(x, 0.5f32), 0)) }
}
def derivative(x: tensor[4, f32]) -> tensor[4, f32] = grad(loss)(x)
"#,
        "derivative",
    );
    let driver = r#"
int main(void) {
    chelis_tensor *x = input(4);
    chelis_tensor *outputs[] = {NULL};
    outputs[0] = derivative(x);
    const float expected[] = {0.0f, 2.0f, 0.0f, 0.0f};
    tensor_bits(outputs[0], 4, expected);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(x);
    return 0;
}
"#;
    ownership_support::balanced(&ownership_support::run(&c, driver));
}

#[test]
fn tensor_loss_gradient_tuple_does_not_feed_the_next_dropout_helper() {
    let c = ownership_support::emit(
        r#"
def loss(x: tensor[4, f32]) -> tensor[f32] = sum(dropout(x, 0.5f32), 0)
result = with seed(42i64) {
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  gradient = grad(loss)(x)
  (gradient, dropout(x, 0.5f32))
}
"#,
        "tensor-loss-tuple-next",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        "result.0 = tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0])\n\
result.1 = tensor(shape=[4], data=[2.0, 0.0, 0.0, 0.0])\n"
    );
}
