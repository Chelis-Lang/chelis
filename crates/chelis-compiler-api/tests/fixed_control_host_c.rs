//! Source/API/native acceptance for keyed [05-OP-37] draws through a host
//! helper (chelis#2413). Every expected value comes from `key_reference`, the
//! transcription of `key_ref.py`, never from emitted C.
#[path = "fixed_control_host_c/context.rs"]
mod context;
mod key_reference;
mod ownership_support;

use chelis_compiler_api::compiler::compile_for_execution;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
use key_reference::{THREE_KEYS, TWO_KEYS, half_dropout, three_keys, two_keys};

/// `key_from_seed(7i64)`, whose first four elements are mixed.
fn key7() -> u64 {
    key_reference::key_from_seed(7)
}

/// The rate-0.5 dropout of four ones under `key`, at f64's exact width when
/// `exact`.
fn mask4(key: u64, exact: bool) -> Vec<f64> {
    key_reference::mask_at_width(key, 4, exact)
}

/// The generated program's rendering of a rank-1 tensor.
fn shown(values: &[f64]) -> String {
    format!(
        "tensor(shape=[{}], data=[{}])",
        values.len(),
        values
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// The C spelling of `values` as a `float` initializer list.
fn c_floats(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:?}f"))
        .collect::<Vec<_>>()
        .join(",")
}

/// The driver's `input(4)`: `2i - 3` for `i` in `0..4`.
const INPUT4: [f64; 4] = [-3.0, -1.0, 1.0, 3.0];

#[test]
fn signed_int64_host_boundaries_compile_without_literal_overflow() {
    // [04-NUM-11], [05-OP-69], chelis#1859. Each seed is `key_from_seed`'s
    // literal operand, emitted as a C integer literal.
    for (seed, literal) in [
        (i64::MIN, "INT64_MIN"),
        (i64::MIN + 1, "(INT64_MIN + 1)"),
        (-1, "-1"),
        (0, "0"),
        (i64::MAX, "INT64_MAX"),
    ] {
        let expected = c_floats(&half_dropout(
            key_reference::key_from_seed(seed),
            &INPUT4,
            false,
        ));
        let source = format!(
            "def boundary() -> i64 = {seed}i64\n\
             def sample(x: tensor[4,f32]) -> tensor[4,f32] = dropout(key_from_seed({seed}i64), x, 0.5f32)"
        );
        let generated = ownership_support::emit(&source, "signed-host-boundary");
        // Make the actual bad decimal token fail compilation on both native
        // compiler families, without suppressing diagnostics elsewhere.
        let strict = format!(
            "#if defined(__clang__)\n#pragma clang diagnostic error \"-Wimplicitly-unsigned-literal\"\n\
             #elif defined(__GNUC__)\n#pragma GCC diagnostic error \"-Woverflow\"\n#endif\n\
             #define main unused_generated_main\n{generated}\n#undef main\n"
        );
        let strict = generated.with_source(strict);
        let boundary = generated.symbol("boundary");
        let sample = generated.symbol("sample");
        let driver = format!(
            r#"
int main(void) {{
    assert({boundary}() == {literal});
    chelis_tensor *x = input(4);
    const float expected[] = {{{expected}}};
    for (int repeat = 0; repeat < 3; ++repeat) {{
        chelis_tensor *result = {sample}(x);
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
def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))
result = keep(key_from_seed(7i64), to_tensor([cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype})]))
"#
        );
        let c = ownership_support::emit(&source, dtype);
        let (summary, stdout) = ownership_support::run_program(&c);
        ownership_support::balanced(&summary);
        assert_eq!(
            stdout,
            format!("result = {}\n", shown(&mask4(key7(), dtype == "f64"))),
            "generic helper specialization must preserve {dtype} values"
        );
    }
}

#[test]
fn generic_static_rate_gradient_and_next_draw_reach_native_host_helpers() {
    // The generic source AD replays the forward mask of `k2`, beside the
    // draws keyed by `k1` and `k3`.
    let (k1, k2, k3) = three_keys();
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))\n\
             def loss(k: key, x: tensor[4, {dtype}]) -> {dtype} = tensor_to_scalar(sum(keep(k, x), 0i32))\n\
             result = {{\n {THREE_KEYS}\n\
               x = to_tensor([cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype})])\n\
               first = keep(k1, copy(x))\n backward = grad(loss, wrt=x)(k2, copy(x))\n next = keep(k3, copy(x))\n\
               [first, backward, next, x]\n }}"
        );
        let c = ownership_support::emit(&source, dtype);
        let (summary, stdout) = ownership_support::run_program(&c);
        ownership_support::balanced(&summary);
        let exact = dtype == "f64";
        assert_eq!(
            stdout,
            format!(
                "result = [{}, {}, {}, {}]\n",
                shown(&mask4(k1, exact)),
                shown(&mask4(k2, exact)),
                shown(&mask4(k3, exact)),
                shown(&[1.0; 4])
            ),
            "{dtype}"
        );
    }
}

#[test]
fn concrete_wrapper_reaches_generic_static_rate_host_helper() {
    let c = ownership_support::emit_selected(
        r#"
def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))
def run(k: key, x: tensor[4, f32]) -> tensor[4, f32] = keep(k, x)
result = run(key_from_seed(7i64), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
"#,
        "generic-wrapper",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        format!("result = {}\n", shown(&mask4(key7(), false)))
    );
}

#[test]
fn computed_and_record_wrappers_preserve_generic_fixed_control_composition() {
    let computed = ownership_support::emit(
        r#"
def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))
result = {
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  keep(key_from_seed(7i64), add(x, x))
}
"#,
        "generic-computed",
    );
    let (summary, stdout) = ownership_support::run_program(&computed);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        format!(
            "result = {}\n",
            shown(&half_dropout(key7(), &[2.0; 4], false))
        )
    );

    let record = ownership_support::emit(
        &format!(
            r#"
type Inputs = | Inputs {{ q: tensor[4, f32] }}
def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))
def run(k: key, inp: Inputs) -> tensor[4, f32] = keep(k, add(inp.q, inp.q))
result = {{
  inp = Inputs {{ q: to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]) }}
  {TWO_KEYS}
  a = run(k1, inp)
  b = run(k2, inp)
  [a, b]
}}
"#
        ),
        "generic-record-wrapper",
    );
    let (summary, stdout) = ownership_support::run_program(&record);
    ownership_support::balanced(&summary);
    let (k1, k2) = two_keys();
    assert_eq!(
        stdout,
        format!(
            "result = [{}, {}]\n",
            shown(&half_dropout(k1, &[2.0; 4], false)),
            shown(&half_dropout(k2, &[2.0; 4], false))
        )
    );
}

/// [05-OP-37]: a runtime rate is an ordinary operand the C draw validates at
/// execution (chelis#2411). A runtime seed is an ordinary `key_from_seed`
/// operand ([05-OP-69]); the counter stream refused one because its handler
/// needed a literal. HIP still has no compiled dropout (chelis#1192).
#[test]
fn concrete_wrappers_admit_runtime_rates_and_seeds_but_not_gpu_dropout() {
    for source in [
        "def keep[p: Float](k: key, x: tensor[4,p], rate: p) -> tensor[4,p] = dropout(k, x, rate)\n\
         def run(x: tensor[4,f32], rate: f32) -> tensor[4,f32] = keep(key_from_seed(42i64), x, rate)",
        "def keep(x: tensor[4,f32], seed: i64) -> tensor[4,f32] = dropout(key_from_seed(seed), x, 0.5f32)\n\
         def run(x: tensor[4,f32], seed: i64) -> tensor[4,f32] = keep(x, seed)",
    ] {
        compile_for_execution(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            target: CompileTarget::C,
            entry_name: Some("run".into()),
        })
        .unwrap_or_else(|error| panic!("a runtime control compiles for C: {source}\n{error:?}"));
    }
    let error = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: "def keep[p: Float](k: key, x: tensor[4,p]) -> tensor[4,p] = dropout(k, x, cast(0.5, p))\n\
                 def run(x: tensor[4,f32]) -> tensor[4,f32] = keep(key_from_seed(42i64), x)"
            .into(),
        target: CompileTarget::Hip,
        entry_name: Some("run".into()),
    })
    .expect_err("GPU dropout must remain loud");
    let message = format!("{error:?}");
    assert!(
        message.contains("compiled `dropout` kernels are not implemented"),
        "{message}"
    );
}

#[test]
fn ordinary_uniform_helpers_do_not_poison_planned_dropout_admission() {
    let (k1, _, k3) = three_keys();
    for used in [false, true] {
        let intervening = if used {
            "u = sample(k2, x)\n  _ = drop(u)"
        } else {
            ""
        };
        let source = format!(
            r#"
def keep[p: Float](k: key, x: tensor[4,p]) -> tensor[4,p] = dropout(k,x,cast(0.5,p))
def sample(k: key, x: tensor[4,f32]) -> tensor[4,f32] = uniform_like(k,x,0.0f32,1.0f32)
result = {{
  {THREE_KEYS}
  x = to_tensor([1.0f32,1.0f32,1.0f32,1.0f32])
  a = keep(k1, x)
  {intervening}
  b = keep(k3, x)
  [a,b]
}}
"#
        );
        let c = ownership_support::emit(&source, "uniform-cohabitation");
        let (summary, stdout) = ownership_support::run_program(&c);
        ownership_support::balanced(&summary);
        // Each dropout draws its own key's mask with or without the
        // intervening uniform draw. No claim about Uniform's values is made
        // or encoded in this oracle.
        assert_eq!(
            stdout,
            format!(
                "result = [{}, {}]\n",
                shown(&mask4(k1, false)),
                shown(&mask4(k3, false))
            ),
            "used={used}"
        );
    }
}

#[test]
fn direct_source_entry_restarts_and_preserves_borrowed_input_ownership() {
    let c = ownership_support::emit_selected(
        r#"
def main(x: tensor[4, f32]) -> tensor[4, f32] = dropout(key_from_seed(7i64), x, 0.5f32)
"#,
        "main",
    );
    let expected = c_floats(&half_dropout(key7(), &INPUT4, false));
    let driver = format!(
        r#"
int main(void) {{
    chelis_tensor *x = input(4);
    const float input_expected[] = {{-3.0f, -1.0f, 1.0f, 3.0f}};
    tensor_bits(x, 4, input_expected);
    chelis_tensor *inputs[] = {{x}};
    chelis_tensor *first[] = {{NULL}};
    chelis_tensor *second[] = {{NULL}};
    chelis_main(inputs, 1, first, 1);
    chelis_main(inputs, 1, second, 1);
    const float expected[] = {{{expected}}};
    tensor_bits(first[0], 4, expected);
    tensor_bits(second[0], 4, expected);
    tensor_bits(x, 4, input_expected);
    chelis_tensor_release(first[0]);
    chelis_tensor_release(second[0]);
    chelis_tensor_release(x);
    return 0;
}}
"#
    );
    ownership_support::balanced(&ownership_support::run(&c, &driver));
}

/// [05-OP-37]: an accepted draw consumes its key even for an empty tensor
/// or rate zero. Both run natively, and the draw after them is its own
/// key's (the counter stream's version asserted that they advanced it).
#[test]
fn host_empty_and_zero_rate_draws_consume_their_own_keys() {
    let c = ownership_support::emit(
        &format!(
            r#"
result = {{
  {THREE_KEYS}
  empty: tensor[0, f32] = to_tensor([])
  dead_empty = dropout(k1, empty, 0.5f32)
  _ = drop(dead_empty)
  ones = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  dead_zero = dropout(k2, ones, 0.0f32)
  _ = drop(dead_zero)
  dropout(k3, ones, 0.5f32)
}}
"#
        ),
        "empty-zero-next",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    assert_eq!(
        stdout,
        format!("result = {}\n", shown(&mask4(three_keys().2, false)))
    );
}

/// The gradient replays the forward mask of its key, a second draw with an
/// equal key draws that same mask, and other keys draw their own (the
/// counter stream's version checked nested seed handlers and the restored
/// outer stream).
#[test]
fn host_gradient_replay_and_an_equal_key_draw_the_same_mask() {
    let c = ownership_support::emit(
        r#"
def loss(k: key, x: tensor[4, f32]) -> f32 = tensor_to_scalar(sum(dropout(k, x, 0.5f32), 0))
result = {
  ones = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  g = grad(loss, wrt=x)(key_from_seed(7i64), ones)
  equal = dropout(key_from_seed(7i64), ones, 0.5f32)
  different = dropout(key_from_seed(99i64), ones, 0.5f32)
  next = dropout(fold_in(key_from_seed(7i64), 1i64), ones, 0.5f32)
  [g, equal, different, next]
}
"#,
        "grad-equal-next",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    let seven = mask4(key7(), false);
    let different = mask4(key_reference::key_from_seed(99), false);
    let next = mask4(key_reference::fold_in(key7(), 1), false);
    assert!(seven != different && seven != next && different != next);
    assert_eq!(
        stdout,
        format!(
            "result = [{}, {}, {}, {}]\n",
            shown(&seven),
            shown(&seven),
            shown(&different),
            shown(&next)
        ),
        "each replay, equal-key and distinct-key mask must remain observable"
    );
}

#[test]
fn exact_cpu_resource_gradient_compiles_and_executes_without_a_runtime_resource_action() {
    let c = ownership_support::emit_selected(
        r#"
def loss(k: key, x: tensor[4, f32]) -> f32 = with device("cpu") {
  tensor_to_scalar(sum(dropout(k, x, 0.5f32), 0))
}
def derivative(x: tensor[4, f32]) -> tensor[4, f32] = grad(loss, wrt=x)(key_from_seed(7i64), x)
"#,
        "derivative",
    );
    let derivative = c.symbol("derivative");
    let expected = c_floats(&mask4(key7(), false));
    let driver = format!(
        r#"
int main(void) {{
    chelis_tensor *x = input(4);
    chelis_tensor *outputs[] = {{NULL}};
    outputs[0] = {derivative}(x);
    const float expected[] = {{{expected}}};
    tensor_bits(outputs[0], 4, expected);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(x);
    return 0;
}}
"#
    );
    ownership_support::balanced(&ownership_support::run(&c, &driver));
}

#[test]
fn tensor_loss_gradient_tuple_does_not_feed_the_next_dropout_helper() {
    let c = ownership_support::emit(
        &format!(
            r#"
def loss(k: key, x: tensor[4, f32]) -> tensor[f32] = sum(dropout(k, x, 0.5f32), 0)
result = {{
  x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])
  {TWO_KEYS}
  gradient = grad(loss, wrt=x)(k1, x)
  (gradient, dropout(k2, x, 0.5f32))
}}
"#
        ),
        "tensor-loss-tuple-next",
    );
    let (summary, stdout) = ownership_support::run_program(&c);
    ownership_support::balanced(&summary);
    let (k1, k2) = two_keys();
    assert_eq!(
        stdout,
        format!(
            "result.0 = {}\nresult.1 = {}\n",
            shown(&mask4(k1, false)),
            shown(&mask4(k2, false))
        )
    );
}
