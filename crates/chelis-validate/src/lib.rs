use chelis_deep::printer::print_canonical;
use pest::Parser;
use pest::iterators::Pair;
use thiserror::Error;

mod deep {
    use pest_derive::Parser;

    #[derive(Parser)]
    #[grammar = "deep.pest"]
    pub struct Grammar;
}

mod surf {
    use pest_derive::Parser;

    #[derive(Parser)]
    #[grammar = "surf.pest"]
    pub struct Grammar;
}

const VALID_TAGS: &[&str] = &[
    "module",
    "import",
    "import-all",
    "export",
    "def",
    "defsig",
    "deftype",
    "typealias",
    "variant",
    "field",
    "defdim",
    "fn",
    "app",
    "let",
    "match",
    "arm",
    "if",
    "var",
    "lit",
    "record",
    "access",
    "pipe",
    "block",
    "tuple",
    "tuple-get",
    "par",
    "borrow",
    "handle-effect",
    "pat-var",
    "pat-lit",
    "pat-ctor",
    "pat-tuple",
    "pat-record",
    "pat-wild",
    "pat-as",
    "record-update",
    "t-prim",
    "t-fn",
    "t-tensor",
    "t-adt",
    "t-var",
    "t-unit",
    "t-tuple",
    "d-name",
    "d-var",
    "d-lit",
    "grad",
    "vmap",
    "jit",
    "realize",
    "cast",
    "copy",
    "quote",
    "unquote",
    "splice",
    "params",
    "bind",
    "kv",
    "effects",
    "resource",
];

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("validation failed: {0}")]
    Failed(String),
}

pub fn validate_surf(source: &str) -> Result<(), ValidationError> {
    match surf::Grammar::parse(surf::Rule::program, source) {
        Ok(_) => Ok(()),
        Err(pest_err) => chelis_surf::parser::parse_str(source)
            .map(|_| ())
            .map_err(|parse_err| {
                ValidationError::Failed(format!("{pest_err}\ncompiler parse failed: {parse_err}"))
            }),
    }
}

pub fn validate_deep(source: &str) -> Result<(), ValidationError> {
    let mut parsed = deep::Grammar::parse(deep::Rule::program, source)
        .map_err(|err| ValidationError::Failed(err.to_string()))?;
    let Some(program) = parsed.next() else {
        return Err(ValidationError::Failed("empty Deep program".to_string()));
    };
    for pair in program.into_inner() {
        if pair.as_rule() == deep::Rule::node {
            validate_deep_node(pair)?;
        }
    }
    Ok(())
}

pub fn validate_desugared(source: &str) -> Result<(), ValidationError> {
    let decls = chelis_surf::parser::parse_str(source)
        .map_err(|err| ValidationError::Failed(format!("compiler parse failed: {err}")))?;
    let deep = chelis_surf::desugar::desugar_program(&decls);
    let deep = chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .map_err(|err| ValidationError::Failed(format!("macro expansion failed: {err}")))?
        .into_exprs();
    let canonical = print_canonical(&deep);
    validate_deep(&canonical)
}

fn validate_deep_node(pair: Pair<'_, deep::Rule>) -> Result<(), ValidationError> {
    let span = pair.as_span();
    let mut inner = pair.into_inner();
    let tag = inner.next().expect("node tag").as_str().to_string();
    let meta = inner.next().expect("node meta");
    let children: Vec<_> = inner
        .filter_map(|pair| match pair.as_rule() {
            deep::Rule::child => pair.into_inner().next(),
            other => Some(pair).filter(|_| other != deep::Rule::EOI),
        })
        .collect();

    if meta.as_rule() != deep::Rule::meta {
        return Err(ValidationError::Failed(format!(
            "Deep node `{tag}` at byte {} is missing metadata",
            span.start()
        )));
    }

    if !VALID_TAGS.contains(&tag.as_str()) {
        return Err(ValidationError::Failed(format!(
            "unknown Deep tag `{tag}` at byte {}",
            span.start()
        )));
    }

    for child in &children {
        match child.as_rule() {
            deep::Rule::node => validate_deep_node(child.clone())?,
            deep::Rule::typed_helper => validate_typed_helper(child.clone())?,
            deep::Rule::unit_list | deep::Rule::literal | deep::Rule::bare_name => {}
            other => {
                return Err(ValidationError::Failed(format!(
                    "unexpected Deep child rule {:?} under `{tag}` at byte {}",
                    other,
                    child.as_span().start()
                )));
            }
        }
    }

    validate_tag_shape(&tag, &children, span.start())
}

fn validate_typed_helper(pair: Pair<'_, deep::Rule>) -> Result<(), ValidationError> {
    let mut inner = pair.clone().into_inner();
    let name = inner
        .next()
        .expect("typed helper name")
        .as_str()
        .to_string();
    let meta = inner.next().expect("typed helper meta");
    if meta.as_rule() != deep::Rule::typed_meta {
        return Err(ValidationError::Failed(format!(
            "typed helper `{name}` at byte {} is missing metadata",
            pair.as_span().start()
        )));
    }
    Ok(())
}

fn validate_tag_shape(
    tag: &str,
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    let child_count = children.len();

    let wrong_arity = |expected: &str| {
        ValidationError::Failed(format!(
            "Deep tag `{tag}` has invalid arity at byte {offset}: expected {expected}, found {child_count}"
        ))
    };

    match tag {
        "if" | "arm" if child_count != 3 => Err(wrong_arity("exactly 3 children")),
        "handle-effect" if child_count != 2 => Err(wrong_arity("exactly 2 children")),
        "fn" => {
            if child_count != 2 {
                return Err(wrong_arity("exactly 2 children"));
            }
            expect_node_tag(&children[0], "params", offset)
        }
        "let" => {
            if child_count != 2 {
                return Err(wrong_arity("exactly 2 children"));
            }
            expect_node_tag(&children[0], "bind", offset)
        }
        "app" if child_count < 1 => Err(wrong_arity("at least 1 child")),
        "params" => validate_params_children(children, offset),
        "bind" => validate_bind_children(children, offset),
        "effects" => validate_effects_children(children, offset),
        "resource" if child_count != 1 => Err(wrong_arity("exactly 1 child")),
        _ => Ok(()),
    }
}

fn expect_node_tag(
    child: &Pair<'_, deep::Rule>,
    expected_tag: &str,
    offset: usize,
) -> Result<(), ValidationError> {
    if child.as_rule() != deep::Rule::node {
        return Err(ValidationError::Failed(format!(
            "Deep tag `{expected_tag}` expected nested node at byte {offset}"
        )));
    }
    let mut inner = child.clone().into_inner();
    let found = inner.next().expect("nested tag").as_str().to_string();
    if found != expected_tag {
        return Err(ValidationError::Failed(format!(
            "expected `{expected_tag}` helper at byte {offset}, found `{found}`"
        )));
    }
    Ok(())
}

fn validate_params_children(
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    for child in children {
        match child.as_rule() {
            deep::Rule::bare_name | deep::Rule::typed_helper => {}
            _ => {
                return Err(ValidationError::Failed(format!(
                    "`params` must contain bare names or typed helpers at byte {offset}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_bind_children(
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    if !children.len().is_multiple_of(2) {
        return Err(ValidationError::Failed(format!(
            "`bind` must contain name/expression pairs at byte {offset}"
        )));
    }
    for (index, child) in children.iter().enumerate() {
        if index % 2 == 0 && child.as_rule() != deep::Rule::bare_name {
            return Err(ValidationError::Failed(format!(
                "`bind` name position {} must be a bare name at byte {offset}",
                index / 2
            )));
        }
    }
    Ok(())
}

fn validate_effects_children(
    children: &[Pair<'_, deep::Rule>],
    offset: usize,
) -> Result<(), ValidationError> {
    for child in children {
        match child.as_rule() {
            deep::Rule::bare_name => {}
            deep::Rule::node
                if child
                    .clone()
                    .into_inner()
                    .next()
                    .is_some_and(|tag| tag.as_str() == "resource") => {}
            _ => {
                return Err(ValidationError::Failed(format!(
                    "`effects` must contain bare names or `(resource {{}} ...)` entries at byte {offset}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_deep, validate_desugared, validate_surf};

    #[test]
    fn surf_accepts_top_level_binding_program() {
        let source = "x = (x : tensor[32, 784, f32])\ny = relu(x)\n";
        validate_surf(source).expect("validator should accept executable script-style bindings");
    }

    #[test]
    fn surf_accepts_semicolon_block_and_axis_identifier() {
        let source = "def f(axis) = { y = axis; y }\ndef g() = par { a; b }\n";
        validate_surf(source)
            .expect("validator should accept semicolon block/par and axis identifiers");
    }

    #[test]
    fn surf_accepts_arrow_return_types() {
        let source = "def f(x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n";
        validate_surf(source).expect("validator should accept arrow return types");
    }

    #[test]
    fn surf_accepts_short_block_bindings() {
        let source = "def f(x) = {\n  y = relu(x)\n  y\n}\n";
        validate_surf(source).expect("validator should accept short block bindings");
    }

    #[test]
    fn deep_rejects_unknown_tag() {
        let source = "(mystery {} x)";
        let error = validate_deep(source).expect_err("unknown tag should fail");
        assert!(error.to_string().contains("unknown Deep tag"));
    }

    #[test]
    fn deep_accepts_snake_case_metadata_keys() {
        let source = "(def {c_earchin_role: \"property_witness\"} req_PRC_001 (fn {} (params {}) (lit {type: (t-prim {} bool)} true)))";
        validate_deep(source).expect("producer metadata keys may be snake_case");
    }

    #[test]
    fn deep_rejects_invalid_effects_children() {
        let source = "(effects {} 1)";
        let error = validate_deep(source).expect_err("non-symbol effects child should fail");
        assert!(
            error
                .to_string()
                .contains("`effects` must contain bare names")
        );
    }

    #[test]
    fn deep_rejects_invalid_resource_arity() {
        let source = "(resource {} x y)";
        let error = validate_deep(source).expect_err("resource arity should fail");
        assert!(error.to_string().contains("expected exactly 1 child"));
    }

    #[test]
    fn desugared_accepts_dotted_module_paths() {
        let source = "module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n";
        validate_desugared(source).expect("desugared dotted module/import paths should validate");
    }

    #[test]
    fn desugared_accepts_executable_example_shape() {
        let source = "module HelloTensor\n\ndef main() -> tensor[f32] = 1\n";
        validate_desugared(source).expect("desugared Deep should validate");
    }

    // ---- issue #167: leading `;` comment lines ----------------------------
    //
    // The pest grammar must treat `;` as a line comment everywhere Deep
    // does, including before the first node and inside `{...}` metadata.
    // Before the `comment = @{...}` atomic fix, pest auto-inserted
    // implicit WHITESPACE that swallowed the newline terminating the
    // comment, so any of these leading-comment shapes failed to parse.
    //
    // Each positive case asserts that the commented form validates AND
    // that it accepts exactly what the comment-free form does, so the two
    // surfaces stay in parity with `chelis_deep::parser::parse_str_strict`.

    /// The base program these comment cases wrap, comment-free.
    const BASE: &str = "(module {} hello)\n";

    fn assert_validates(source: &str, label: &str) {
        // Both the commented and the bare form must validate, and the
        // strict hand-rolled parser must also accept the source, so the
        // two Deep surfaces agree (the core complaint of #167).
        validate_deep(source).unwrap_or_else(|err| panic!("{label} should validate: {err}"));
        validate_deep(BASE).expect("base program should validate");
        chelis_deep::parser::parse_str_strict(source)
            .unwrap_or_else(|err| panic!("{label} should parse strictly: {err}"));
    }

    #[test]
    fn deep_accepts_single_leading_comment() {
        assert_validates("; a leading comment\n(module {} hello)\n", "single leading comment");
    }

    #[test]
    fn deep_accepts_directive_leading_comment() {
        // The `; chelis-lint:` directive form is the shape that forced
        // every CLI callsite to pre-strip; it must now validate directly.
        assert_validates(
            "; chelis-lint: disable=foo\n(module {} hello)\n",
            "leading chelis-lint directive",
        );
    }

    #[test]
    fn deep_accepts_multiple_leading_comments() {
        assert_validates(
            "; first\n; second\n(module {} hello)\n",
            "two consecutive leading comments",
        );
    }

    #[test]
    fn deep_accepts_leading_comment_with_blank_lines() {
        assert_validates(
            "; a leading comment\n\n; another\n\n(module {} hello)\n",
            "leading comments separated by blank lines",
        );
    }

    #[test]
    fn deep_accepts_trailing_comment() {
        assert_validates("(module {} hello)\n; trailing\n", "trailing comment");
    }

    #[test]
    fn deep_accepts_mid_program_comment() {
        assert_validates(
            "(module {} hello)\n; mid\n(module {} world)\n",
            "mid-program comment between nodes",
        );
    }

    #[test]
    fn deep_accepts_meta_internal_comment() {
        // The surprising case from the issue table: a `;` comment inside a
        // `{...}` metadata block. `spacing` (which includes `comment`)
        // appears throughout `meta`, so the atomic fix covers it too.
        assert_validates(
            "(module {\n; inside meta\n} hello)\n",
            "comment inside metadata block",
        );
    }

    #[test]
    fn deep_accepts_leading_comment_without_trailing_newline() {
        // No newline after the final node; the leading comment is still
        // terminated by its own newline before the node.
        assert_validates("; leading\n(module {} hello)", "leading comment, no trailing newline");
    }

    #[test]
    fn deep_leading_comment_equals_uncommented_program() {
        // The commented program validates to the same acceptance decision
        // as the program with the comments removed, which is the issue's
        // core "equals the same program without the comments" requirement.
        let commented = "; chelis-lint: disable=foo\n; plain comment\n(module {} hello)\n";
        validate_deep(commented).expect("commented program should validate");
        validate_deep(BASE).expect("uncommented program should validate");
    }

    // ---- negative parity: the grammar was not loosened -------------------

    #[test]
    fn deep_still_rejects_unterminated_node() {
        // A genuinely malformed Deep program (missing closing paren) must
        // still fail to parse. Confirms the atomic-comment fix did not
        // relax the grammar into accepting broken structure.
        let source = "; a comment\n(module {} hello";
        validate_deep(source).expect_err("unterminated node must still fail to parse");
    }

    #[test]
    fn deep_still_rejects_missing_metadata() {
        // A node without its mandatory `{}` metadata block must still fail,
        // even when preceded by a leading comment.
        let source = "; a comment\n(module hello)\n";
        validate_deep(source).expect_err("node without metadata must still fail");
    }

    #[test]
    fn deep_still_rejects_garbage_after_comment() {
        // Non-node, non-comment garbage following a leading comment must
        // not be swallowed into a comment and silently accepted.
        let source = "; a comment\n@@@ not a node\n";
        validate_deep(source).expect_err("garbage after a comment must still fail to parse");
    }
}
