//! C6 structural observations for declarations rustdoc does not publish.
//! The Python driver derives the accepted syntax contexts from rustc's actual
//! macro DefIds. We inspect each item; an outer generated scope is no exemption.
use proc_macro2::Span;
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use syn::parse::Parser;
use syn::visit::{self, Visit};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    source: String,
    derive_contexts: BTreeMap<String, String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GeneratedBy {
    Serde,
    Schema,
}

struct Inspector<'a> {
    request: &'a Request,
    block_depth: usize,
    module_nominals: usize,
    generated_local_items: usize,
    findings: Vec<String>,
    schema_marker_blocks: Vec<BTreeSet<(String, String)>>,
}

impl Inspector<'_> {
    fn context(&self, span: Span) -> Option<&str> {
        // rustc emits a hygiene comment immediately after each identifier. Its
        // ordinary source comments are absent; strings remain AST literals and
        // cannot manufacture the comment at this parsed identifier's location.
        let after = self.request.source.get(span.byte_range().end..)?;
        let comment = after.trim_start().strip_prefix("/*")?;
        let (comment, _) = comment.split_once("*/")?;
        let (symbol, context) = comment.trim().split_once('#')?;
        (!symbol.is_empty()
            && symbol.bytes().all(|byte| byte.is_ascii_digit())
            && !context.is_empty()
            && context.bytes().all(|byte| byte.is_ascii_digit()))
        .then_some(context)
    }

    fn generated_by(&self, span: Span) -> Option<GeneratedBy> {
        let definition = self.request.derive_contexts.get(self.context(span)?)?;
        if definition.ends_with("::Serialize") || definition.ends_with("::Deserialize") {
            Some(GeneratedBy::Serde)
        } else if definition.ends_with("::JsonSchema") {
            Some(GeneratedBy::Schema)
        } else {
            None
        }
    }

    fn marker_key(&self, ident: &syn::Ident) -> Option<(String, String)> {
        Some((ident.to_string(), self.context(ident.span())?.into()))
    }

    fn schema_marker(&self, ty: &syn::Type) -> bool {
        let syn::Type::Path(path) = ty else {
            return false;
        };
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return false;
        }
        let Some(key) = self.marker_key(&path.path.segments[0].ident) else {
            return false;
        };
        self.schema_marker_blocks
            .last()
            .is_some_and(|markers| markers.contains(&key))
    }

    fn declaration(&mut self, kind: &str, ident: &syn::Ident, unit_marker: bool) {
        if self.block_depth == 0 {
            if matches!(kind, "struct" | "enum" | "union") {
                self.module_nominals += 1;
            }
        } else if self.generated_by(ident.span()) == Some(GeneratedBy::Serde)
            || (unit_marker && self.generated_by(ident.span()) == Some(GeneratedBy::Schema))
        {
            self.generated_local_items += 1;
        } else {
            self.findings.push(format!(
                "unsupported block-local {kind} {ident}; move its declaration to module scope"
            ));
        }
    }
}

impl<'ast> Visit<'ast> for Inspector<'_> {
    fn visit_block(&mut self, block: &'ast syn::Block) {
        let mut markers = BTreeSet::new();
        for statement in &block.stmts {
            if let syn::Stmt::Item(syn::Item::Struct(item)) = statement
                && matches!(item.fields, syn::Fields::Unit)
                && item.generics.params.is_empty()
                && self.generated_by(item.ident.span()) == Some(GeneratedBy::Schema)
                && let Some(key) = self.marker_key(&item.ident)
            {
                markers.insert(key);
            }
        }
        self.schema_marker_blocks.push(markers);
        self.block_depth += 1;
        visit::visit_block(self, block);
        self.block_depth -= 1;
        self.schema_marker_blocks.pop();
    }

    fn visit_item(&mut self, item: &'ast syn::Item) {
        match item {
            syn::Item::Struct(item) => self.declaration(
                "struct",
                &item.ident,
                matches!(item.fields, syn::Fields::Unit) && item.generics.params.is_empty(),
            ),
            syn::Item::Enum(item) => self.declaration("enum", &item.ident, false),
            syn::Item::Union(item) => self.declaration("union", &item.ident, false),
            syn::Item::Type(item) => self.declaration("type alias", &item.ident, false),
            syn::Item::Mod(item) => self.declaration("module", &item.ident, false),
            syn::Item::Trait(item) => self.declaration("trait", &item.ident, false),
            syn::Item::TraitAlias(item) => self.declaration("trait alias", &item.ident, false),
            syn::Item::Impl(item) if self.block_depth > 0 => {
                let origin = item.attrs.iter().find_map(|attr| {
                    attr.path()
                        .is_ident("automatically_derived")
                        .then(|| self.generated_by(attr.path().segments[0].ident.span()))
                        .flatten()
                });
                let schema_trait = item.trait_.as_ref().is_some_and(|(_, path, _)| {
                    path.segments.len() == 2
                        && path.segments[0].ident == "schemars"
                        && path.segments[1].ident == "JsonSchema"
                });
                let generated = origin == Some(GeneratedBy::Serde)
                    || (schema_trait
                        && (origin == Some(GeneratedBy::Schema)
                            || self.schema_marker(&item.self_ty)));
                if generated {
                    self.generated_local_items += 1;
                } else {
                    self.findings.push(
                        "unsupported block-local impl; move its implementation to module scope"
                            .into(),
                    );
                }
            }
            syn::Item::Macro(item) if item.ident.is_some() => {
                // A macro definition is inert. Every active invocation is
                // expanded by rustc before this visitor receives the library.
                return;
            }
            syn::Item::Verbatim(_) => {
                self.findings.push("unsupported expanded Rust item".into());
                return;
            }
            _ => {}
        }
        visit::visit_item(self, item);
    }

    fn visit_expr(&mut self, expression: &'ast syn::Expr) {
        match expression {
            // syn represents an empty statement (`;`) as an empty verbatim
            // expression. It contains no subtree to visit.
            syn::Expr::Verbatim(tokens) if tokens.is_empty() => {}
            syn::Expr::Verbatim(tokens) => self
                .findings
                .push(format!("unsupported expanded Rust expression: {tokens}")),
            _ => visit::visit_expr(self, expression),
        }
    }

    fn visit_type(&mut self, ty: &'ast syn::Type) {
        if matches!(ty, syn::Type::Verbatim(_)) {
            self.findings.push("unsupported expanded Rust type".into());
        } else {
            visit::visit_type(self, ty);
        }
    }

    fn visit_pat(&mut self, pattern: &'ast syn::Pat) {
        if matches!(pattern, syn::Pat::Verbatim(_)) {
            self.findings
                .push("unsupported expanded Rust pattern".into());
        } else {
            visit::visit_pat(self, pattern);
        }
    }

    fn visit_impl_item(&mut self, item: &'ast syn::ImplItem) {
        if matches!(item, syn::ImplItem::Verbatim(_)) {
            self.findings.push("unsupported expanded impl item".into());
        } else {
            visit::visit_impl_item(self, item);
        }
    }

    fn visit_trait_item(&mut self, item: &'ast syn::TraitItem) {
        if matches!(item, syn::TraitItem::Verbatim(_)) {
            self.findings.push("unsupported expanded trait item".into());
        } else {
            visit::visit_trait_item(self, item);
        }
    }

    fn visit_foreign_item(&mut self, item: &'ast syn::ForeignItem) {
        if matches!(item, syn::ForeignItem::Verbatim(_)) {
            self.findings
                .push("unsupported expanded foreign item".into());
        } else {
            visit::visit_foreign_item(self, item);
        }
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        // rustc prints its built-in FormatArgs AST as format_args!(...). Parse
        // every argument expression: a local type in an interpolation is still
        // inside the publication boundary. All other opaque invocations fail.
        if mac.path.is_ident("format_args") {
            let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
            match parser.parse2(mac.tokens.clone()) {
                Ok(arguments) if !arguments.is_empty() => {
                    for argument in &arguments {
                        self.visit_expr(argument);
                    }
                }
                _ => self
                    .findings
                    .push("unsupported expanded format arguments".into()),
            }
        } else {
            self.findings
                .push("unsupported unexpanded Rust macro invocation".into());
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: Request = serde_json::from_reader(io::stdin().lock())?;
    let file = syn::parse_file(&request.source)?;
    let mut inspector = Inspector {
        request: &request,
        block_depth: 0,
        module_nominals: 0,
        generated_local_items: 0,
        findings: Vec::new(),
        schema_marker_blocks: Vec::new(),
    };
    inspector.visit_file(&file);
    println!(
        "{}",
        json!({"module_nominals": inspector.module_nominals,
            "generated_local_items": inspector.generated_local_items,
            "findings": inspector.findings})
    );
    Ok(())
}
