use chelis_deep::printer::print_canonical;
use chelis_macros::{ExpansionError, ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;

fn expand_surf(source: &str) -> String {
    let decls = parse_str(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let expanded = expand_program(&deep, &ExpansionOptions::default()).expect("macro expansion");
    print_canonical(expanded.exprs())
}

#[test]
fn simple_macro_expands_to_base_tags_with_source_metadata() {
    let text = expand_surf(
        r#"
macro relu_ref(x) = max_elem(x, 0.0)
def f(x: tensor[4, f32]): tensor[4, f32] = relu_ref(x)
"#,
    );

    assert!(!text.contains("defmacro"));
    assert!(!text.contains("macro-invoke"));
    assert!(text.contains("max_elem"));
    assert!(text.contains("source: (relu_ref"));
}

#[test]
fn lexical_binding_blocks_prelude_macro_expansion() {
    let text = expand_surf(
        r#"
def f(residual, x: tensor[4, f32]): tensor[4, f32] = residual(x)
"#,
    );

    assert!(text.contains("(var {} residual)"));
    assert!(!text.contains("source: (residual"));
}

#[test]
fn hygiene_renames_macro_introduced_binders_only() {
    let decls = parse_str(
        r#"
macro capture(y) = { let x = 1.0; add(x, y) }
def f(x: f32): f32 = capture(x)
"#,
    )
    .expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let expanded = expand_program(&deep, &ExpansionOptions::default()).expect("macro expansion");
    let text = print_canonical(expanded.exprs());

    assert!(text.contains("x_macro_"));
    assert!(text.contains("(var {} x)"));
}

#[test]
fn free_references_survive_hygiene() {
    let text = expand_surf(
        r#"
def f(batch: int32, x: tensor[batch, hidden, f32], w: tensor[hidden, out_dim, f32], b: tensor[out_dim, f32]): tensor[batch, out_dim, f32] =
  linear_layer(x, w, b)
"#,
    );

    assert!(text.contains(" batch)"));
    assert!(!text.contains("batch_macro_"));
}

#[test]
fn recursive_macro_hits_expansion_limit() {
    let decls = parse_str(
        r#"
macro loop(x) = loop(x)
def f(x: f32): f32 = loop(x)
"#,
    )
    .expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let err = expand_program(
        &deep,
        &ExpansionOptions {
            max_iterations: 4,
            load_std_prelude: true,
        },
    )
    .expect_err("recursive macro should fail");

    assert!(matches!(err, ExpansionError::ExpansionLimitExceeded { .. }));
}
