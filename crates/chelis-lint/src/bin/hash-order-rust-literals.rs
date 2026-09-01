//! Internal Rust-token literal census for the hash-order determinism oracle.

use proc_macro2::{TokenStream, TokenTree};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::{self, Read};
use std::process::ExitCode;
use std::str::FromStr;
use syn::{Expr, Item, Lit, Meta};

#[derive(Deserialize)]
struct SourceInput {
    path: String,
    source: String,
}

#[derive(Serialize)]
struct SourceOutput {
    path: String,
    literals: Vec<String>,
    external_modules: Vec<String>,
    inline_module_paths: Vec<Vec<String>>,
    path_modules: Vec<PathModule>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct PathModule {
    inline_modules: Vec<String>,
    target: String,
}

#[derive(Debug, Eq, PartialEq)]
struct SourceFacts {
    literals: Vec<String>,
    external_modules: Vec<String>,
    inline_module_paths: Vec<Vec<String>>,
    path_modules: Vec<PathModule>,
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

fn rust_ident(ident: &proc_macro2::Ident) -> String {
    let spelling = ident.to_string();
    spelling.strip_prefix("r#").unwrap_or(&spelling).to_owned()
}

fn collect_external_modules(
    tokens: TokenStream,
    external_modules: &mut BTreeSet<String>,
) -> Result<(), String> {
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    for token in &tokens {
        if let TokenTree::Group(group) = token {
            collect_external_modules(group.stream(), external_modules)?;
        }
    }
    for window in tokens.windows(3) {
        let TokenTree::Ident(keyword) = &window[0] else {
            continue;
        };
        if keyword != "mod" {
            continue;
        }
        match (&window[1], &window[2]) {
            (TokenTree::Ident(name), TokenTree::Punct(end)) if end.as_char() == ';' => {
                external_modules.insert(rust_ident(name));
            }
            (TokenTree::Punct(dollar), _) if dollar.as_char() == '$' => {
                return Err(
                    "dynamic external module name after `mod` cannot be inventoried".to_owned(),
                );
            }
            _ => {}
        }
    }
    Ok(())
}

fn path_attribute(item: &syn::ItemMod) -> Result<Option<String>, String> {
    let mut target = None;
    for attribute in &item.attrs {
        if !attribute.path().is_ident("path") {
            continue;
        }
        if target.is_some() {
            return Err(format!(
                "module {} has more than one path attribute",
                item.ident
            ));
        }
        let Meta::NameValue(name_value) = &attribute.meta else {
            return Err(format!(
                "module {} has a non-literal path attribute",
                item.ident
            ));
        };
        let Expr::Lit(expression) = &name_value.value else {
            return Err(format!(
                "module {} has a non-literal path attribute",
                item.ident
            ));
        };
        let Lit::Str(value) = &expression.lit else {
            return Err(format!(
                "module {} has a non-string path attribute",
                item.ident
            ));
        };
        target = Some(value.value());
    }
    Ok(target)
}

fn collect_ast_modules(
    items: &[Item],
    inline_modules: &[String],
    inline_module_paths: &mut BTreeSet<Vec<String>>,
    path_modules: &mut BTreeSet<PathModule>,
) -> Result<(), String> {
    for item in items {
        let Item::Mod(module) = item else {
            continue;
        };
        if let Some((_, nested_items)) = &module.content {
            let mut nested_modules = inline_modules.to_vec();
            nested_modules.push(rust_ident(&module.ident));
            inline_module_paths.insert(nested_modules.clone());
            collect_ast_modules(
                nested_items,
                &nested_modules,
                inline_module_paths,
                path_modules,
            )?;
        } else if let Some(target) = path_attribute(module)? {
            path_modules.insert(PathModule {
                inline_modules: inline_modules.to_vec(),
                target,
            });
        }
    }
    Ok(())
}

fn source_facts(path: &str, source: &str) -> Result<SourceFacts, String> {
    let tokens = TokenStream::from_str(source)
        .map_err(|error| format!("could not tokenize Rust source {path}: {error}"))?;
    let mut literals = Vec::new();
    collect_utf8_strings(tokens.clone(), &mut literals)
        .map_err(|error| format!("Rust literal census failed for {path}: {error}"))?;
    let mut external_modules = BTreeSet::new();
    collect_external_modules(tokens, &mut external_modules)
        .map_err(|error| format!("Rust module census failed for {path}: {error}"))?;
    let file = syn::parse_file(source)
        .map_err(|error| format!("could not parse Rust source {path}: {error}"))?;
    let mut inline_module_paths = BTreeSet::new();
    let mut path_modules = BTreeSet::new();
    collect_ast_modules(
        &file.items,
        &[],
        &mut inline_module_paths,
        &mut path_modules,
    )
    .map_err(|error| format!("Rust module census failed for {path}: {error}"))?;
    Ok(SourceFacts {
        literals,
        external_modules: external_modules.into_iter().collect(),
        inline_module_paths: inline_module_paths.into_iter().collect(),
        path_modules: path_modules.into_iter().collect(),
    })
}

fn run() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("could not read source-census input: {error}"))?;
    let sources: Vec<SourceInput> = serde_json::from_str(&input)
        .map_err(|error| format!("invalid source-census input: {error}"))?;
    let outputs = sources
        .into_iter()
        .map(|source| {
            let facts = source_facts(&source.path, &source.source)?;
            Ok(SourceOutput {
                literals: facts.literals,
                external_modules: facts.external_modules,
                inline_module_paths: facts.inline_module_paths,
                path_modules: facts.path_modules,
                path: source.path,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    serde_json::to_writer(io::stdout(), &outputs)
        .map_err(|error| format!("could not write source-census output: {error}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hash-order Rust source census: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PathModule, source_facts};

    #[test]
    fn lifetime_tokens_do_not_hide_the_intervening_string() {
        let literals = source_facts(
            "lifetime.rs",
            "load!(#, 'before, \"generated/escape.rs\", 'after);",
        )
        .expect("tokenize lifetime fixture")
        .literals;
        assert_eq!(literals, ["generated/escape.rs"]);
    }

    #[test]
    fn all_doc_forms_desugar_and_non_doc_comments_disappear() {
        let literals = source_facts(
            "docs.rs",
            "//!inner-line.rs\n/*!inner-block.rs*/\n\
             ///outer-line.rs\n/**outer-block.rs*/\nfn documented() {}\n\
             ////not-doc-line.rs\n/***not-doc-block.rs*/\nfn not_documented() {}\n",
        )
        .expect("tokenize doc-comment fixture")
        .literals;
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
        let literals = source_facts(
            "strings.rs",
            "emit!(\"ordinary\\u{2f}path.rs\", r#\"raw/path.rs\"#, \
             b\"bytes.rs\", br\"raw-bytes.rs\", c\"c-string.rs\");",
        )
        .expect("tokenize string-kind fixture")
        .literals;
        assert_eq!(literals, ["ordinary/path.rs", "raw/path.rs"]);
    }

    #[test]
    fn nested_macro_groups_preserve_source_order() {
        let literals = source_facts(
            "groups.rs",
            "outer!({ inner!([\"first.rs\", (r\"second.rs\")]) });",
        )
        .expect("tokenize nested token groups")
        .literals;
        assert_eq!(literals, ["first.rs", "second.rs"]);
    }

    #[test]
    fn module_facts_cover_external_inline_and_path_ingress() {
        let facts = source_facts(
            "modules.rs",
            "mod direct;\n\
             macro_rules! load { () => { mod generated; }; }\n\
             mod outer {\n\
                 #[path = \"generated/escape.rs\"] mod escaped;\n\
                 mod nested { mod leaf; }\n\
             }\n",
        )
        .expect("parse module fixture");
        assert_eq!(
            facts.external_modules,
            ["direct", "escaped", "generated", "leaf"]
        );
        assert_eq!(
            facts.inline_module_paths,
            [
                vec!["outer".to_owned()],
                vec!["outer".to_owned(), "nested".to_owned()]
            ]
        );
        assert_eq!(
            facts.path_modules,
            [PathModule {
                inline_modules: vec!["outer".to_owned()],
                target: "generated/escape.rs".to_owned(),
            }]
        );
    }

    #[test]
    fn dynamic_external_module_names_fail_closed() {
        let error = source_facts(
            "dynamic.rs",
            "macro_rules! load { ($name:ident) => { mod $name; }; }",
        )
        .expect_err("dynamic module name must fail closed");
        assert!(error.contains("dynamic external module name"), "{error}");
    }
}
