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
