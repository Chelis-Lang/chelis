//! Source/API/native acceptance for [05-OP-37] and [05-RNG-1] through a host helper.
#[allow(dead_code)]
mod ownership_support;

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
