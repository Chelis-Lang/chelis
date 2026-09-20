//! [04-LIN-3,4,7,8]: unused internal owners terminate without consuming caller borrows.
mod ownership_support;
use ownership_support::{balanced, emit, emit_selected, run};

fn constant_gradient(n: usize) {
    let source = format!(
        "\
def loss(x: tensor[{n}, f32]) -> f32 = 3.0f32
def derivative(x: tensor[{n}, f32]) -> tensor[{n}, f32] = grad(loss)(x)
"
    );
    let c = emit_selected(&source, "derivative");
    let loss = c.symbol("loss").to_string();
    let derivative = c.symbol("derivative").to_string();
    assert!(c.contains(&format!("float {loss}(")));
    assert!(c.contains(&format!("chelis_tensor* {derivative}(")));
    assert!(!c.contains("float loss("));
    assert!(!c.contains("chelis_tensor* derivative("));
    let driver = format!(
        r#"
int main(void) {{
    chelis_tensor *x = input({n});
    const float original[2] = {{-3, -1}}, zero[2] = {{0, 0}};
    for (int i = 0; i < 16; ++i) {{
        assert({loss}(x) == 3.0f);
        chelis_tensor *dx = {derivative}(x);
        tensor_bits(dx, {n}, zero);
        chelis_tensor_release(dx);
        tensor_bits(x, {n}, original);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#
    );
    balanced(&run(&c, &driver));
}

#[test]
fn unused_owned_tensor_entry_and_constant_gradient_balance() {
    constant_gradient(2);
}

#[test]
fn unused_empty_tensor_entry_and_constant_gradient_balance() {
    constant_gradient(0);
}

#[test]
fn unused_mixed_and_shared_entry_arguments_balance() {
    let c = emit(
        r#"
def entry(x: tensor[2, f32], y: tensor[2, f32], z: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(y, 0))
"#,
        "entry",
    );
    let entry = c.symbol("entry").to_string();
    assert!(c.contains(&format!("float {entry}(")));
    assert!(!c.contains("float entry("));
    let driver = format!(
        r#"
int main(void) {{
    chelis_tensor *x = input(2), *y = input(2);
    const float original[2] = {{-3, -1}};
    for (int i = 0; i < 16; ++i) {{
        assert({entry}(x, y, x) == -4.0f);
        assert({entry}(x, x, x) == -4.0f);
        tensor_bits(x, 2, original);
        tensor_bits(y, 2, original);
    }}
    chelis_tensor_release(x);
    chelis_tensor_release(y);
    return 0;
}}
"#
    );
    balanced(&run(&c, &driver));
}

#[test]
fn unused_string_entry_keeps_caller_alive_and_balances() {
    let c = emit("def entry(text: string) -> f32 = 3.0f32", "entry");
    let entry = c.symbol("entry").to_string();
    assert!(c.contains(&format!("float {entry}(")));
    assert!(!c.contains("float entry("));
    let driver = format!(
        r#"
int main(void) {{
    chelis_string text = chelis_string_from_cstr("caller survives");
    for (int i = 0; i < 16; ++i) {{
        assert({entry}(text) == 3.0f);
        assert(strcmp(chelis_string_data(text), "caller survives") == 0);
    }}
    chelis_string_release(text);
    return 0;
}}
"#
    );
    balanced(&run(&c, &driver));
}

#[test]
fn unused_borrow_is_not_released_and_missing_owned_drop_is_detected() {
    let borrowed = emit("def entry(x: &tensor[2, f32]) -> f32 = 3.0f32", "entry");
    let entry = borrowed.symbol("entry").to_string();
    let driver = format!(
        r#"
int main(void) {{
    chelis_tensor *x = input(2);
    const float original[2] = {{-3, -1}};
    for (int i = 0; i < 16; ++i) {{
        assert({entry}(x) == 3.0f);
        tensor_bits(x, 2, original);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#
    );
    assert!(borrowed.contains(&format!("float {entry}(")));
    assert!(!borrowed.contains("float entry("));
    balanced(&run(&borrowed, &driver));
    let owned = emit("def entry(x: tensor[2, f32]) -> f32 = 3.0f32", "entry");
    assert!(owned.contains(&format!("float {entry}(")));
    assert!(!owned.contains("float entry("));
    balanced(&run(&owned, &driver));
    let release = "    chelis_tensor_release(x);";
    assert_eq!(
        owned.matches(release).count(),
        1,
        "one terminal, not both projections"
    );
    let leaking = owned.with_source(owned.replacen(release, "", 1));
    let summary = run(&leaking, &driver);
    assert_eq!(
        summary["live_owners"], 17,
        "16 retained references plus storage: {summary}"
    );
    assert_eq!(summary["live_bytes"], 8, "{summary}");
}

#[test]
fn unused_monomorphized_parameter_balances_without_an_abi_clone() {
    let c = emit(
        r#"
def ignore[a](x: a) -> f32 = 3.0f32
def entry(text: string) -> f32 = ignore(text)
"#,
        "entry",
    );
    let entry = c.symbol("entry").to_string();
    assert!(c.contains(&format!("float {entry}(")));
    assert!(!c.contains("float entry("));
    let driver = format!(
        r#"
int main(void) {{
    chelis_string text = chelis_string_from_cstr("specialized parameter");
    for (int i = 0; i < 16; ++i) {{
        assert({entry}(text) == 3.0f);
        assert(strcmp(chelis_string_data(text), "specialized parameter") == 0);
    }}
    chelis_string_release(text);
    return 0;
}}
"#
    );
    balanced(&run(&c, &driver));
}
