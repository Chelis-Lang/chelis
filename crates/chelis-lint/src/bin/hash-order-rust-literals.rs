//! Internal Rust-token literal census for the hash-order determinism oracle.

use proc_macro2::{TokenStream, TokenTree};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};
use std::process::ExitCode;
use std::str::FromStr;

#[derive(Deserialize)]
struct SourceInput {
    path: String,
    source: String,
}

#[derive(Serialize)]
struct SourceOutput {
    path: String,
    literals: Vec<String>,
}

fn collect_utf8_strings(tokens: TokenStream, literals: &mut Vec<String>) -> Result<(), String> {
    for token in tokens {
        match token {
            TokenTree::Group(group) => collect_utf8_strings(group.stream(), literals)?,
            TokenTree::Literal(literal) => {
                let spelling = literal.to_string();
                match syn::parse_str::<syn::Lit>(&spelling) {
                    Ok(syn::Lit::Str(value)) => literals.push(value.value()),
                    Ok(_) => {}
                    Err(error) => {
                        return Err(format!(
                            "could not classify Rust literal {spelling:?}: {error}"
                        ));
                    }
                }
            }
            TokenTree::Ident(_) | TokenTree::Punct(_) => {}
        }
    }
    Ok(())
}

fn source_literals(path: &str, source: &str) -> Result<Vec<String>, String> {
    let tokens = TokenStream::from_str(source)
        .map_err(|error| format!("could not tokenize Rust source {path}: {error}"))?;
    let mut literals = Vec::new();
    collect_utf8_strings(tokens, &mut literals)
        .map_err(|error| format!("Rust literal census failed for {path}: {error}"))?;
    Ok(literals)
}

fn run() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("could not read literal-census input: {error}"))?;
    let sources: Vec<SourceInput> = serde_json::from_str(&input)
        .map_err(|error| format!("invalid literal-census input: {error}"))?;
    let outputs = sources
        .into_iter()
        .map(|source| {
            Ok(SourceOutput {
                literals: source_literals(&source.path, &source.source)?,
                path: source.path,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    serde_json::to_writer(io::stdout(), &outputs)
        .map_err(|error| format!("could not write literal-census output: {error}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hash-order Rust literal census: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::source_literals;

    #[test]
    fn lifetime_tokens_do_not_hide_the_intervening_string() {
        let literals = source_literals(
            "lifetime.rs",
            "load!(#, 'before, \"generated/escape.rs\", 'after);",
        )
        .expect("tokenize lifetime fixture");
        assert_eq!(literals, ["generated/escape.rs"]);
    }

    #[test]
    fn all_doc_forms_desugar_and_non_doc_comments_disappear() {
        let literals = source_literals(
            "docs.rs",
            "///outer-line.rs\n//!inner-line.rs\n/**outer-block.rs*/\n/*!inner-block.rs*/\n\
             ////not-doc-line.rs\n/***not-doc-block.rs*/\n",
        )
        .expect("tokenize doc-comment fixture");
        for expected in [
            "outer-line.rs",
            "inner-line.rs",
            "outer-block.rs",
            "inner-block.rs",
        ] {
            assert!(
                literals.iter().any(|literal| literal == expected),
                "missing {expected:?} from {literals:?}"
            );
        }
        assert!(!literals.iter().any(|literal| literal.contains("not-doc")));
    }

    #[test]
    fn utf8_strings_decode_and_non_utf8_string_kinds_stay_excluded() {
        let literals = source_literals(
            "strings.rs",
            "emit!(\"ordinary\\u{2f}path.rs\", r#\"raw/path.rs\"#, \
             b\"bytes.rs\", br\"raw-bytes.rs\", c\"c-string.rs\");",
        )
        .expect("tokenize string-kind fixture");
        assert_eq!(literals, ["ordinary/path.rs", "raw/path.rs"]);
    }

    #[test]
    fn nested_macro_groups_preserve_source_order() {
        let literals = source_literals(
            "groups.rs",
            "outer!({ inner!([\"first.rs\", (r\"second.rs\")]) });",
        )
        .expect("tokenize nested token groups");
        assert_eq!(literals, ["first.rs", "second.rs"]);
    }
}
