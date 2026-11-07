//! Ownership acceptance and diagnostic order agree across both checker entries.
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::{CheckedProgram, check_ir_program, check_linearity, check_typed_program};

#[derive(Clone, Copy)]
enum Entry {
    Ir,
    Typed,
}

fn checked_program(source: &str, entry: Entry) -> CheckedProgram {
    let declarations =
        parse_str(source, &chelis_syntax_host::NativeStack).expect("Surf fixture parses");
    let program = desugar_program(&declarations, &chelis_syntax_host::NativeStack)
        .expect("Surf fixture must desugar");
    match entry {
        Entry::Ir => check_ir_program(&program).expect("IR entry type-checks the fixture"),
        Entry::Typed => check_typed_program(&program).expect("typed entry type-checks the fixture"),
    }
}

fn ordered_linearity_diagnostics(program: &CheckedProgram) -> Vec<String> {
    match check_linearity(program) {
        Ok(_) => Vec::new(),
        Err(errors) => errors
            .into_iter()
            .map(|error| format!("[{:?}] {}", error.kind, error.message))
            .collect(),
    }
}

#[test]
fn nested_destructure_alias_single_consume_matches_across_entries() {
    let source = r#"
def ok(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (left, right) = pair
    alias: tensor[4, f32] = left
    consumed_left: tensor[4, f32] = realize(alias)
    consumed_right: tensor[4, f32] = realize(right)
    add(consumed_left, consumed_right)
  }
"#;

    let typed = ordered_linearity_diagnostics(&checked_program(source, Entry::Typed));
    let ir = ordered_linearity_diagnostics(&checked_program(source, Entry::Ir));
    assert!(typed.is_empty(), "positive control: {typed:?}");
    assert_eq!(typed, ir, "checker entry choice must not change acceptance");
}

#[test]
fn nested_destructure_alias_reuse_preserves_diagnostic_order_across_entries() {
    let source = r#"
def bad(pair: (tensor[4, f32], tensor[4, f32])) -> tensor[4, f32] =
  {
    (left, right) = pair
    alias: tensor[4, f32] = left
    consumed_left: tensor[4, f32] = realize(alias)
    consumed_right: tensor[4, f32] = realize(right)
    add(left, consumed_left)
  }
"#;

    let typed = ordered_linearity_diagnostics(&checked_program(source, Entry::Typed));
    let ir = ordered_linearity_diagnostics(&checked_program(source, Entry::Ir));
    assert_eq!(
        typed, ir,
        "binding identity, diagnostic ownership, and order must match"
    );
    assert!(
        typed.iter().any(|message| {
            message.contains("[UseAfterConsume]")
                && message.contains("variable `left`")
                && message.contains("consumed by realize")
        }),
        "negative control must retain the component/alias consume diagnostic: {typed:?}"
    );
}
