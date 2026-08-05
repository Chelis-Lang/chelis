//! Structural Deep-to-Surf resugaring.
//!
//! Deep and Surf are two representations of the same public language. This
//! module therefore owns no handwritten Surf dialect: it constructs the
//! shared Surf AST through [`crate::resugar`] and renders that AST through the
//! one canonical formatter in [`crate::format`].

use chelis_deep::role::is_declaration_tag;
use chelis_deep::{DeepTag, Expr};

use crate::ast::Decl;
use crate::format::format_program;
use crate::resugar::{ResugarError, resugar_expression, resugar_program};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecompileOptions {
    /// Append stable, comment-only canonical Deep debug output after the
    /// canonical Surf program. This never selects a second Surf grammar.
    pub emit_deep_debug_comments: bool,
}

impl DecompileOptions {
    pub const fn idiomatic() -> Self {
        Self {
            emit_deep_debug_comments: false,
        }
    }

    /// Compatibility constructor for the CLI's historical `--verbose` flag.
    /// The verbose Surf dialect is gone; verbose mode is canonical Surf plus
    /// comments containing the canonical Deep source.
    pub const fn verbose() -> Self {
        Self {
            emit_deep_debug_comments: true,
        }
    }
}

/// Display-oriented compatibility wrapper.
///
/// Boundaries that can return an error must use [`try_decompile_program`].
/// This wrapper keeps malformed Deep visible as a stable comment rather than
/// panicking or inventing placeholder Surf.
pub fn decompile_program(exprs: &[Expr]) -> String {
    decompile_program_with_context(exprs, &DecompileOptions::idiomatic(), None)
}

pub fn decompile_program_with_options(exprs: &[Expr], options: &DecompileOptions) -> String {
    decompile_program_with_context(exprs, options, None)
}

pub fn decompile_program_with_context(
    exprs: &[Expr],
    options: &DecompileOptions,
    synthetic_name: Option<&str>,
) -> String {
    match try_decompile_program_with_context(exprs, options, synthetic_name) {
        Ok(source) => source,
        Err(error) => format!("-- Deep resugaring error: {error}\n"),
    }
}

/// Resugar a Deep program through the shared Surf AST and canonical printer.
pub fn try_decompile_program(exprs: &[Expr]) -> Result<String, ResugarError> {
    try_decompile_program_with_context(exprs, &DecompileOptions::idiomatic(), None)
}

/// Resugar a Deep program with an optional name for a single expression-only
/// input (useful for `.dp` inspection files that contain no declaration).
pub fn try_decompile_program_with_context(
    exprs: &[Expr],
    options: &DecompileOptions,
    synthetic_name: Option<&str>,
) -> Result<String, ResugarError> {
    let declarations = match resugar_program(exprs) {
        Ok(declarations) => declarations,
        Err(_) if exprs.len() == 1 && !is_top_level_declaration(&exprs[0]) => {
            let value = resugar_expression(&exprs[0])?;
            vec![Decl::LetDef {
                name: synthetic_name.unwrap_or("result").to_string(),
                ty: None,
                value,
                span: exprs[0].span(),
            }]
        }
        Err(error) => return Err(error),
    };

    let mut source = format_program(&declarations);
    if options.emit_deep_debug_comments {
        append_deep_debug_comments(&mut source, exprs);
    }
    crate::parser::parse_str(&source).map_err(|error| ResugarError::InvalidSurfaceProgram {
        reason: error.to_string(),
    })?;
    Ok(source)
}

fn is_top_level_declaration(expr: &Expr) -> bool {
    expr.tag()
        .is_some_and(|tag| tag == DeepTag::Module || is_declaration_tag(tag))
}

fn append_deep_debug_comments(source: &mut String, exprs: &[Expr]) {
    for line in chelis_deep::printer::print_canonical(exprs).lines() {
        source.push_str("-- deep-debug: ");
        source.push_str(line);
        source.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desugar::desugar_program;
    use crate::parser::parse_str;

    #[test]
    fn verbose_mode_is_canonical_surf_plus_comments() {
        let surf = parse_str("def f(x: f32) -> f32 = x").expect("Surf parses");
        let deep = desugar_program(&surf);

        let rendered =
            try_decompile_program_with_context(&deep, &DecompileOptions::verbose(), None)
                .expect("Deep resugars");

        assert!(rendered.starts_with("def f(x: f32) -> f32 = x\n"));
        assert!(rendered.contains("-- deep-debug: (defsig"));
        parse_str(&rendered).expect("debug comments do not introduce a dialect");
    }

    #[test]
    fn malformed_deep_is_an_error_at_fallible_boundaries() {
        use chelis_deep::ast::{Atom, MetaMap, UnknownFormData};
        use chelis_deep::{Expr, Span};

        let span = Span::new(0, 0);
        let deep = vec![Expr::UnknownForm(Box::new(UnknownFormData {
            head: "future-form".to_string(),
            meta: MetaMap::default(),
            children: vec![Expr::Atom(Atom::Name("value".to_string()), span)],
            span,
        }))];

        let error = try_decompile_program(&deep).expect_err("unknown Deep form is rejected");

        assert!(error.to_string().contains("unknown form `future-form`"));
    }

    #[test]
    fn integer_spelled_float_preserves_suffix_and_exact_payload() {
        let surf = parse_str(
            "def exact() -> f32 = 18014399583223809f32\n\
             def contextual() -> tensor[2, bf16] = [18084767253659649, 3]\n",
        )
        .expect("Surf parses");
        let deep = desugar_program(&surf);

        let rendered = try_decompile_program(&deep).expect("typed literals resugar");

        assert!(
            rendered.contains("18014399583223809f32"),
            "explicit suffix or exact payload was lost:\n{rendered}"
        );
        assert!(
            rendered.contains("18084767253659649bf16"),
            "context-selected suffix or exact payload was lost:\n{rendered}"
        );
        parse_str(&rendered).expect("resugared Surf reparses");
    }
}
