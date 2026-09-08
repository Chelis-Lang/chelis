//! Match-arm binders are lexical locals, not eager top-level references.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;

fn diagnostics(source: &str) -> Vec<String> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| error.message.clone())
            .collect(),
    }
}

#[test]
fn arm_binder_does_not_create_a_top_level_cycle() {
    let errors = diagnostics(
        r#"
type Box =
  | Wrap(string)
def unwrap(b: Box) -> string = match b with {
  | Wrap(text) => text
}
text = unwrap(Wrap("hi"))
"#,
    );
    assert!(errors.is_empty(), "lexical arm binder leaked: {errors:#?}");
}

#[test]
fn nested_pattern_binders_scope_guard_and_body() {
    let errors = diagnostics(
        r#"
type PairBox =
  | PairBox(string, string)
def choose(b: PairBox) -> string = match b with {
  | PairBox(left, right) if string_contains(left, right) => left
  | PairBox(left, right) => string_concat(left, right)
}
left = choose(PairBox("a", "b"))
right = "outside"
"#,
    );
    assert!(errors.is_empty(), "nested arm binders leaked: {errors:#?}");
}

#[test]
fn genuine_top_level_cycle_still_rejects() {
    let source = r#"
a = (b : tensor[4, f32])
b = (a : tensor[4, f32])
"#;
    let decls = parse_surf(source).expect("Surf cycle fixture must parse");
    let deep = desugar_program(&decls);
    let errors: Vec<_> = chelis_types::infer_ir_program(&deep)
        .errors
        .into_iter()
        .map(|error| error.message)
        .collect();
    assert!(
        errors.iter().any(|error| error.contains("binding cycle")),
        "a genuine cycle must still reject: {errors:#?}"
    );
}
