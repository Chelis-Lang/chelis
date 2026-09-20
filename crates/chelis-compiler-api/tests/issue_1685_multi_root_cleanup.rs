//! [04-LIN-4,7]: helper result owners transfer into the retaining tuple once.
mod ownership_support;
use ownership_support::{authored_c_symbol, balanced, emit, emit_selected, run};

fn assert_selected_symbol_contract(c: &str, authored_name: &str) {
    let symbol = authored_c_symbol(authored_name);
    assert!(
        c.contains(&format!("{symbol}(")),
        "selected host execution must expose the injective authored symbol"
    );
    assert!(
        !c.lines()
            .any(|line| line.contains(&format!(" {authored_name}("))),
        "strict selected execution must not expose the raw authored name"
    );
}

fn assert_legacy_symbol_contract(c: &str, authored_names: &[&str]) {
    for name in authored_names {
        let symbol = authored_c_symbol(name);
        assert!(
            c.contains(&format!("{symbol}(")),
            "legacy whole-program C must expose the injective authored symbol for `{name}`"
        );
        assert!(
            !c.lines().any(|line| line.contains(&format!(" {name}("))),
            "legacy whole-program C must not expose the raw authored name `{name}`"
        );
    }
}

fn product_gradient(n: usize, m: usize) {
    let source = format!("\
def loss(x: tensor[{n}, f32], y: tensor[{m}, f32]) -> f32 = tensor_to_scalar(sum(x, 0)) * tensor_to_scalar(sum(y, 0))
def derivative(x: tensor[{n}, f32], y: tensor[{m}, f32]) -> (tensor[{n}, f32], tensor[{m}, f32]) = grad(loss)(x, y)
");
    let c = emit_selected(&source, "derivative");
    assert_selected_symbol_contract(&c, "derivative");
    let derivative = authored_c_symbol("derivative");
    let driver = format!(
        r#"
int main(void) {{
    chelis_tensor *x = input({n}), *y = input({m});
    const float original[3] = {{-3, -1, 1}};
    float dx[3], dy[3];
    float sx = 0.0f, sy = 0.0f;
    for (int i = 0; i < {n}; ++i) sx += original[i];
    for (int i = 0; i < {m}; ++i) sy += original[i];
    for (int i = 0; i < {n}; ++i) dx[i] = sy;
    for (int i = 0; i < {m}; ++i) dy[i] = sx;
    for (int i = 0; i < 16; ++i) {{
        chelis_tuple *result = {derivative}(x, y);
        assert(chelis_tuple_len(result) == 2);
        chelis_value a = chelis_tuple_get(result, 0);
        chelis_value b = chelis_tuple_get(result, 1);
        assert(a.tag == CHELIS_VALUE_TENSOR && b.tag == CHELIS_VALUE_TENSOR);
        // Acquired fields must survive release of the tuple that supplied them.
        chelis_tuple_release(result);
        tensor_bits(chelis_tensor_borrow_value(a), {n}, dx);
        tensor_bits(chelis_tensor_borrow_value(b), {m}, dy);
        chelis_value_release(a);
        chelis_value_release(b);
        tensor_bits(x, {n}, original);
        tensor_bits(y, {m}, original);
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
fn unequal_gradient_shapes_balance() {
    product_gradient(2, 3);
}

#[test]
fn empty_gradient_shapes_balance() {
    for (n, m) in [(0, 3), (2, 0), (0, 0)] {
        product_gradient(n, m);
    }
}

#[test]
fn shared_gradient_roots_and_aliased_inputs_balance() {
    let c = emit_selected(
        r#"
def loss(x: tensor[2, f32], y: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(add(x, y), 0))
def derivative(x: tensor[2, f32], y: tensor[2, f32]) -> (tensor[2, f32], tensor[2, f32]) = grad(loss)(x, y)
"#,
        "derivative",
    );
    assert_selected_symbol_contract(&c, "derivative");
    let driver = r#"
int main(void) {
    chelis_tensor *x = input(2);
    const float original[2] = {-3, -1}, ones[2] = {1, 1};
    for (int i = 0; i < 16; ++i) {
        chelis_tuple *result = DERIVATIVE_ENTRY(x, x);
        assert(chelis_tuple_len(result) == 2);
        chelis_value a = chelis_tuple_get(result, 0);
        chelis_value b = chelis_tuple_get(result, 1);
        tensor_bits(chelis_tensor_borrow_value(a), 2, ones);
        chelis_value_release(a);
        chelis_tuple_release(result);
        tensor_bits(chelis_tensor_borrow_value(b), 2, ones);
        chelis_value_release(b);
        tensor_bits(x, 2, original);
    }
    chelis_tensor_release(x);
    return 0;
}
"#
    .replace("DERIVATIVE_ENTRY", &authored_c_symbol("derivative"));
    balanced(&run(&c, &driver));
}

#[test]
fn multi_root_leak_mutation_preserves_values_but_fails_accounting() {
    let c = emit_selected(
        r#"
def loss(x: tensor[2, f32], y: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, y), 0))
def derivative(x: tensor[2, f32], y: tensor[2, f32]) -> (tensor[2, f32], tensor[2, f32]) = grad(loss)(x, y)
"#,
        "derivative",
    );
    assert_selected_symbol_contract(&c, "derivative");
    let driver = r#"
int main(void) {
    chelis_tensor *x = input(2), *y = input(2);
    const float xv[2] = {2, 3}, yv[2] = {5, 7};
    chelis_tensor_write *gx = chelis_tensor_begin_write(x), *gy = chelis_tensor_begin_write(y);
    memcpy(chelis_tensor_write_view(gx).data, xv, sizeof(xv));
    memcpy(chelis_tensor_write_view(gy).data, yv, sizeof(yv));
    chelis_tensor_end_write(gx); chelis_tensor_end_write(gy);
    for (int i = 0; i < 16; ++i) {
        chelis_tuple *result = DERIVATIVE_ENTRY(x, y);
        assert(chelis_tuple_len(result) == 2);
        chelis_value a = chelis_tuple_get(result, 0), b = chelis_tuple_get(result, 1);
        tensor_bits(chelis_tensor_borrow_value(a), 2, yv);
        tensor_bits(chelis_tensor_borrow_value(b), 2, xv);
        chelis_value_release(a); chelis_value_release(b);
        chelis_tuple_release(result);
        tensor_bits(x, 2, xv); tensor_bits(y, 2, yv);
    }
    chelis_tensor_release(x); chelis_tensor_release(y);
    return 0;
}
"#
    .replace("DERIVATIVE_ENTRY", &authored_c_symbol("derivative"));
    balanced(&run(&c, &driver));
    let mut removed = 0;
    let mutated = c
        .lines()
        .filter(|line| {
            let skip = line
                .trim_start()
                .starts_with("chelis_value_release(__tuple_values_");
            if skip {
                removed += 1;
            }
            !skip
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(removed, 2, "only the two helper output temporaries");
    let summary = run(&mutated, &driver);
    assert_eq!(summary["live_owners"], 64, "{summary}");
    assert_eq!(summary["live_bytes"], 256, "{summary}");
}

#[test]
fn single_output_and_ordinary_tuple_paths_remain_balanced() {
    let c = emit(
        r#"
def loss(x: tensor[2, f32], y: tensor[2, f32]) -> f32 = tensor_to_scalar(sum(mul(x, y), 0))
def derivative(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = grad(loss, wrt=x)(x, y)
def pair(x: tensor[2, f32]) -> (tensor[2, f32], tensor[2, f32]) = (x, x)
"#,
        "derivative",
    );
    assert_legacy_symbol_contract(&c, &["derivative", "pair"]);
    let driver = r#"
int main(void) {
    chelis_tensor *x = input(2);
    const float original[2] = {-3, -1};
    for (int i = 0; i < 16; ++i) {
        chelis_tensor *dx = DERIVATIVE_ENTRY(x, x);
        tensor_bits(dx, 2, original);
        chelis_tensor_release(dx);
        chelis_tuple *result = PAIR_ENTRY(x);
        assert(chelis_tuple_len(result) == 2);
        for (int j = 0; j < 2; ++j) {
            chelis_value value = chelis_tuple_get(result, j);
            tensor_bits(chelis_tensor_borrow_value(value), 2, original);
            chelis_value_release(value);
        }
        chelis_tuple_release(result);
        tensor_bits(x, 2, original);
    }
    chelis_tensor_release(x);
    return 0;
}
"#
    .replace("DERIVATIVE_ENTRY", &authored_c_symbol("derivative"))
    .replace("PAIR_ENTRY", &authored_c_symbol("pair"));
    balanced(&run(&c, &driver));
}
