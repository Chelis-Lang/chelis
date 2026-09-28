//! Structural required-call adoption; runtime checks own execution evidence.

use serde::Deserialize;
use std::collections::BTreeSet;
use syn::parse::Parser;
use syn::visit::{self, Visit};

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredTest {
    pub name: String,
    pub calls: Vec<String>,
    pub run_parity: Option<bool>,
    pub result: Option<RequiredResult>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequiredResult {
    pub call: String,
    pub kind: ResultUse,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResultUse {
    Success,
    AssertOk,
    AssertErr,
    ExpectErr,
    AssertEq,
}

pub fn validate(source: &str, required: &[RequiredTest]) -> Result<(), String> {
    let file = syn::parse_file(source).map_err(|error| format!("invalid Rust source: {error}"))?;
    validate_file(&file, required)
}

pub fn validate_file(file: &syn::File, required: &[RequiredTest]) -> Result<(), String> {
    let mut errors = Vec::new();
    let mut names = BTreeSet::new();
    for row in required {
        if row.calls.is_empty() || !names.insert(&row.name) {
            errors.push(format!(
                "{}: empty or duplicate required call contract",
                row.name
            ));
            continue;
        }
        if row
            .result
            .as_ref()
            .is_some_and(|result| !row.calls.contains(&result.call))
        {
            errors.push(format!("{}: checked result has no required call", row.name));
            continue;
        }
        let definitions: Vec<_> = file
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Fn(function) if function.sig.ident == row.name => Some(function),
                _ => None,
            })
            .collect();
        if definitions.len() != 1 {
            errors.push(format!(
                "{}: expected one required top-level definition",
                row.name
            ));
            continue;
        }
        let function = definitions[0];
        if !function
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("test"))
        {
            errors.push(format!("{}: missing test attribute", row.name));
        }
        let mut audit = BodyAudit {
            row,
            owners: row
                .calls
                .iter()
                .map(|call| call.split("::").next().unwrap())
                .collect(),
            found: BTreeSet::new(),
            checked_result: false,
            errors: Vec::new(),
            collect: true,
        };
        audit.visit_item_fn(function);
        for call in &row.calls {
            if !audit.found.contains(call) {
                audit.errors.push(format!("missing required call {call}"));
            }
        }
        if let Some(result) = &row.result
            && !audit.checked_result
        {
            audit.errors.push(format!(
                "missing checked {:?} result for {}",
                result.kind, result.call
            ));
        }
        errors.extend(
            audit
                .errors
                .into_iter()
                .map(|error| format!("{}: {error}", row.name)),
        );
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

fn ungroup(expression: &syn::Expr) -> &syn::Expr {
    match expression {
        syn::Expr::Paren(value) => ungroup(&value.expr),
        syn::Expr::Group(value) => ungroup(&value.expr),
        _ => expression,
    }
}

fn call_identity(expression: &syn::Expr) -> Option<String> {
    let syn::Expr::Call(call) = ungroup(expression) else {
        return None;
    };
    let syn::Expr::Path(path) = ungroup(&call.func) else {
        return None;
    };
    if path.qself.is_some() || path.path.leading_colon.is_some() {
        return None;
    }
    Some(
        path.path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
    )
}

fn direct_panic(expression: &syn::Expr) -> bool {
    match ungroup(expression) {
        syn::Expr::Macro(value) => value.mac.path.is_ident("panic"),
        syn::Expr::Block(value) if value.block.stmts.len() == 1 => match &value.block.stmts[0] {
            syn::Stmt::Expr(value, _) => direct_panic(value),
            syn::Stmt::Macro(value) => value.mac.path.is_ident("panic"),
            _ => false,
        },
        _ => false,
    }
}

struct BodyAudit<'a> {
    row: &'a RequiredTest,
    owners: BTreeSet<&'a str>,
    found: BTreeSet<String>,
    checked_result: bool,
    errors: Vec<String>,
    collect: bool,
}

impl BodyAudit<'_> {
    fn checked(&mut self, expression: &syn::Expr, kind: ResultUse) {
        if self.collect
            && let Some(required) = &self.row.result
            && required.kind == kind
            && call_identity(expression).as_ref() == Some(&required.call)
        {
            self.checked_result = true;
        }
    }
}

impl<'ast> Visit<'ast> for BodyAudit<'_> {
    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        if attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr") {
            self.errors
                .push("conditional configuration is outside the required body contract".into());
        }
    }

    fn visit_item(&mut self, _item: &'ast syn::Item) {
        self.errors.push(
            "local items are outside the required body contract; declare helpers at module scope"
                .into(),
        );
    }

    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        if self.owners.contains(pattern.ident.to_string().as_str()) {
            self.errors.push(format!(
                "local pattern shadows required owner {}",
                pattern.ident
            ));
        }
        visit::visit_pat_ident(self, pattern);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if self.collect
            && let syn::Expr::Path(path) = ungroup(&call.func)
            && path.qself.is_none()
            && path.path.leading_colon.is_none()
        {
            let identity = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            if self.row.calls.contains(&identity) {
                self.found.insert(identity.clone());
                if identity == "drive_parity"
                    && let Some(expected) = self.row.run_parity
                {
                    let matches = call.args.len() == 2
                        && matches!(ungroup(&call.args[1]), syn::Expr::Lit(value) if matches!(&value.lit, syn::Lit::Bool(value) if value.value == expected));
                    if !matches {
                        self.errors.push(format!(
                            "drive_parity must use reviewed literal execution mode {expected}"
                        ));
                    }
                }
            }
        }
        visit::visit_expr_call(self, call);
    }

    fn visit_expr_method_call(&mut self, method: &'ast syn::ExprMethodCall) {
        let name = method.method.to_string();
        if (name == "unwrap" && method.args.is_empty())
            || (name == "expect" && method.args.len() == 1)
            || (name == "unwrap_or_else"
                && method.args.len() == 1
                && matches!(ungroup(&method.args[0]), syn::Expr::Closure(closure) if direct_panic(&closure.body)))
        {
            self.checked(&method.receiver, ResultUse::Success);
        }
        if (name == "unwrap_err" && method.args.is_empty())
            || (name == "expect_err" && method.args.len() == 1)
        {
            self.checked(&method.receiver, ResultUse::ExpectErr);
        }
        visit::visit_expr_method_call(self, method);
    }

    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        // Only assertion conditions are adopted. Formatting arguments execute
        // on failure; arbitrary macro tokens and debug-only assertions do not
        // establish a required comparison. This does not expand Rust macros.
        let count = if mac.path.is_ident("assert") {
            1
        } else if mac.path.is_ident("assert_eq") || mac.path.is_ident("assert_ne") {
            2
        } else {
            return;
        };
        let parser = syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated;
        match parser.parse2(mac.tokens.clone()) {
            Ok(arguments) if arguments.len() >= count => {
                if mac.path.is_ident("assert")
                    && let syn::Expr::MethodCall(method) = ungroup(&arguments[0])
                    && method.args.is_empty()
                {
                    if method.method == "is_ok" {
                        self.checked(&method.receiver, ResultUse::AssertOk);
                    }
                    if method.method == "is_err" {
                        self.checked(&method.receiver, ResultUse::AssertErr);
                    }
                }
                for argument in arguments.iter().take(count) {
                    if mac.path.is_ident("assert_eq") {
                        self.checked(argument, ResultUse::AssertEq);
                    }
                    self.visit_expr(argument);
                }
            }
            _ => self.errors.push("unreadable assertion condition".into()),
        }
    }

    fn visit_expr_closure(&mut self, expression: &'ast syn::ExprClosure) {
        let previous = self.collect;
        self.collect = false;
        visit::visit_expr_closure(self, expression);
        self.collect = previous;
    }

    fn visit_expr_async(&mut self, expression: &'ast syn::ExprAsync) {
        let previous = self.collect;
        self.collect = false;
        visit::visit_expr_async(self, expression);
        self.collect = previous;
    }

    fn visit_expr_const(&mut self, expression: &'ast syn::ExprConst) {
        let previous = self.collect;
        self.collect = false;
        visit::visit_expr_const(self, expression);
        self.collect = previous;
    }
}
