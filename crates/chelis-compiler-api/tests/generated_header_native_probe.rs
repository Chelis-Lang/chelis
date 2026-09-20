//! chelis#2107: compiler-API native probes consume generated declaration metadata.

mod ownership_support;

use ownership_support::{balanced, emit, run};

#[test]
fn native_probe_uses_the_generated_header_symbol() {
    let program = emit("def entry(x: f32) -> f32 = add(x, 1.0f32)", "entry");
    let entry = program.symbol("entry").to_string();
    assert_eq!(
        program.declaration("entry"),
        format!("float {entry}(float x);")
    );
    let driver = format!(
        "int main(void) {{\n\
         \x20   assert({entry}(2.0f) == 3.0f);\n\
         \x20   chelis_tensor *receipt = input(1);\n\
         \x20   chelis_tensor_release(receipt);\n\
         \x20   return 0;\n\
         }}\n"
    );
    balanced(&run(&program, &driver));
}

#[test]
fn missing_stale_and_disagreeing_compiler_api_headers_fail_closed() {
    let program = emit("def entry(x: f32) -> f32 = add(x, 1.0f32)", "entry");
    let declaration = program.declaration("entry");
    let symbol = program.symbol("entry");

    assert!(
        program.with_header(String::new()).is_err(),
        "missing generated declarations must not fall back to source-name reconstruction"
    );

    let stale_declaration = declaration.replace(symbol, "stale_entry");
    let stale_header = program
        .header()
        .replacen(declaration, &stale_declaration, 1);
    let stale = program
        .with_header(stale_header)
        .expect("stale header remains syntactic");
    assert!(
        stale.validate().is_err(),
        "a stale generated symbol must disagree with the emitted source"
    );

    let disagreeing_declaration = declaration.replace("(float x)", "(double x)");
    let disagreeing_header = program
        .header()
        .replacen(declaration, &disagreeing_declaration, 1);
    let disagreeing = program
        .with_header(disagreeing_header)
        .expect("disagreeing header remains syntactic");
    assert!(
        disagreeing.validate().is_err(),
        "a generated declaration type mismatch must fail before native execution"
    );
}
