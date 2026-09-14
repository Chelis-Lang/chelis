//! Declared test inputs are membership evidence, not proof of comparator execution.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use syn::visit::Visit;
use syn::{Expr, Item, Pat, Stmt};

pub fn discover(root: &Path) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    for entry in fs::read_dir(root).map_err(|error| format!("read examples: {error}"))? {
        let path = entry
            .map_err(|error| format!("read example entry: {error}"))?
            .path();
        if path.extension() != Some(OsStr::new("ch")) {
            continue;
        }
        if !path
            .metadata()
            .map_err(|error| format!("read {}: {error}", path.display()))?
            .is_file()
        {
            continue;
        }
        let name = path
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or_else(|| format!("example filename is not UTF-8: {}", path.display()))?;
        names.insert(name.to_owned());
    }
    if names.is_empty() {
        return Err("executable example corpus is empty".into());
    }
    Ok(names)
}

// Parentheses do not change ownership; references do. Keep the two grammars separate.
fn parenthesized(expression: &Expr) -> &Expr {
    match expression {
        Expr::Paren(value) => parenthesized(&value.expr),
        Expr::Group(value) => parenthesized(&value.expr),
        _ => expression,
    }
}

fn identifier(expression: &Expr) -> Option<String> {
    let Expr::Path(path) = parenthesized(expression) else {
        return None;
    };
    path.path.get_ident().map(ToString::to_string)
}

fn filename(expression: &Expr) -> Result<String, String> {
    let Expr::Lit(literal) = parenthesized(expression) else {
        return Err("example input must have a literal filename".into());
    };
    let syn::Lit::Str(value) = &literal.lit else {
        return Err("example input must have a string filename".into());
    };
    let name = value.value();
    if name.contains(['/', '\\']) || !name.ends_with(".ch") || name.len() == 3 {
        return Err(format!("example input must name a root .ch file: {name:?}"));
    }
    Ok(name)
}

#[derive(Clone)]
struct OwnedExamplePath(String);

// This closed grammar constructs an owned PathBuf, never a reference-valued alias.
fn owned_example_path(
    expression: &Expr,
    locals: &BTreeMap<String, OwnedExamplePath>,
) -> Result<Option<OwnedExamplePath>, String> {
    if let Some(name) = identifier(expression) {
        return Ok(locals.get(&name).cloned());
    }
    let Expr::MethodCall(join) = parenthesized(expression) else {
        return Ok(None);
    };
    if join.method != "join" {
        return Ok(None);
    }
    let Expr::Call(root) = parenthesized(&join.receiver) else {
        return Ok(None);
    };
    if identifier(&root.func).as_deref() != Some("examples_root") || !root.args.is_empty() {
        return Ok(None);
    }
    if join.args.len() != 1 || join.turbofish.is_some() {
        return Err("example path must use examples_root().join(literal)".into());
    }
    filename(&join.args[0]).map(|name| Some(OwnedExamplePath(name)))
}

fn input_argument(
    expression: &Expr,
    locals: &BTreeMap<String, OwnedExamplePath>,
) -> Result<Option<OwnedExamplePath>, String> {
    match parenthesized(expression) {
        Expr::Reference(reference) if reference.mutability.is_none() => {
            input_argument(&reference.expr, locals)
        }
        Expr::Reference(_) => Ok(None),
        expression => owned_example_path(expression, locals),
    }
}

fn input_helper(expression: &Expr) -> Option<(String, usize)> {
    let helper = identifier(expression)?;
    let arity = match helper.as_str() {
        "drive_parity" | "check_dropout_eval_and_c_rejection" => 2,
        "assert_check_clean" => 1,
        _ => return None,
    };
    Some((helper, arity))
}

fn binding(pattern: &Pat) -> Option<&syn::PatIdent> {
    match pattern {
        Pat::Ident(name) if name.subpat.is_none() => Some(name),
        Pat::Type(typed) => binding(&typed.pat),
        _ => None,
    }
}

#[derive(Default)]
struct InputSyntax {
    conditional: bool,
    local_item: bool,
    bindings: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for InputSyntax {
    fn visit_item(&mut self, item: &'ast Item) {
        self.local_item = true;
        syn::visit::visit_item(self, item);
    }

    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        if attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr") {
            self.conditional = true;
        }
        syn::visit::visit_attribute(self, attribute);
    }

    fn visit_pat_ident(&mut self, pattern: &'ast syn::PatIdent) {
        self.bindings.insert(pattern.ident.to_string());
        syn::visit::visit_pat_ident(self, pattern);
    }
}

pub fn declared_inputs(source: &str) -> Result<BTreeSet<String>, String> {
    let file = syn::parse_file(source).map_err(|error| format!("parse parity tests: {error}"))?;
    if file
        .attrs
        .iter()
        .any(|attribute| attribute.path().is_ident("cfg") || attribute.path().is_ident("cfg_attr"))
    {
        return Err("conditional parity source cannot supply test inputs".into());
    }
    let mut inputs = BTreeSet::new();
    let mut tests = BTreeSet::new();
    for item in file.items {
        let Item::Fn(test) = item else { continue };
        if !test
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("test"))
        {
            continue;
        }
        let name = test.sig.ident.to_string();
        if !tests.insert(name.clone()) {
            return Err(format!("duplicate test: {name}"));
        }
        for attribute in &test.attrs {
            // The authoritative Phase 3 ledger separately validates ignore identity/reason.
            if !attribute.path().is_ident("test")
                && !attribute.path().is_ident("ignore")
                && !attribute.path().is_ident("doc")
            {
                return Err(format!("{name}: conditional or unsupported test attribute"));
            }
        }
        let mut syntax = InputSyntax::default();
        syntax.visit_block(&test.block);
        if syntax.conditional {
            return Err(format!(
                "{name}: conditional test body cannot supply inputs"
            ));
        }
        let has_inputs = test.block.stmts.iter().any(|statement| {
            matches!(statement, Stmt::Expr(Expr::Call(call), _)
                if input_helper(&call.func).is_some())
        });
        if has_inputs && syntax.local_item {
            return Err(format!(
                "{name}: local items are outside the input grammar; declare helpers at module scope"
            ));
        }
        let mut locals = BTreeMap::new();
        let mut shadowed = BTreeSet::new();
        for statement in &test.block.stmts {
            match statement {
                Stmt::Local(local) => {
                    // All bindings shadow, including destructuring and `name @ pattern`.
                    // Resolve a simple immutable initializer in the preceding scope.
                    let input = if let Some(pattern) = binding(&local.pat)
                        && pattern.mutability.is_none()
                        && pattern.by_ref.is_none()
                        && let Some(init) = &local.init
                    {
                        owned_example_path(&init.expr, &locals)?
                    } else {
                        None
                    };
                    let mut syntax = InputSyntax::default();
                    syntax.visit_pat(&local.pat);
                    for key in syntax.bindings {
                        if key == "examples_root" {
                            return Err(format!("{name}: shadowed examples_root input owner"));
                        }
                        locals.remove(&key);
                        shadowed.insert(key);
                    }
                    if let Some(pattern) = binding(&local.pat) {
                        let key = pattern.ident.to_string();
                        if let Some(input) = input {
                            locals.insert(key, input);
                        }
                    }
                }
                Stmt::Expr(Expr::Assign(assignment), _) => {
                    if let Some(key) = identifier(&assignment.left) {
                        locals.remove(&key);
                    }
                }
                Stmt::Expr(Expr::Call(call), _) => {
                    let Some((helper, arity)) = input_helper(&call.func) else {
                        continue;
                    };
                    if shadowed.contains(&helper) || call.args.len() != arity {
                        return Err(format!("{name}: ambiguous {helper} input"));
                    }
                    let input = if helper == "check_dropout_eval_and_c_rejection" {
                        OwnedExamplePath(filename(&call.args[0])?)
                    } else {
                        input_argument(&call.args[0], &locals)?
                            .ok_or_else(|| format!("{name}: unresolved {helper} example input"))?
                    };
                    inputs.insert(input.0);
                }
                _ => {}
            }
        }
    }
    Ok(inputs)
}

pub fn validate(root: &Path, source: &str) -> Result<(), String> {
    let actual = discover(root)?;
    let declared = declared_inputs(source)?;
    let missing: Vec<_> = declared.difference(&actual).collect();
    let uncovered: Vec<_> = actual.difference(&declared).collect();
    if !missing.is_empty() || !uncovered.is_empty() {
        return Err(format!(
            "parity inputs differ from examples/: missing files {missing:?}; files without test inputs {uncovered:?}"
        ));
    }
    Ok(())
}
