//! Source/API/native acceptance for [05-OP-37] and [05-RNG-1] through a host helper.
#[allow(dead_code)]
mod ownership_support;

#[test]
fn generic_static_rate_reaches_native_host_helper() {
    let source = r#"
def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))
def main() = with seed(42i64) {
  keep(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
}
"#;
    let c = ownership_support::emit(source, "main");
    let driver = r#"
int main(void) {
    chelis_tensor *fixture__main(void);
    chelis_tensor *value = fixture__main();
    const float expected[] = {0.0f, 2.0f, 2.0f, 2.0f};
    tensor_bits(value, 4, expected);
    chelis_tensor_release(value);
    return 0;
}
"#;
    ownership_support::balanced(&ownership_support::run(&c, driver));
}
