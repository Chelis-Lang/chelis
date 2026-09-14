//! Declared test inputs are membership evidence, not proof of comparator execution.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::path::Path;
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

fn unwrapped(expression: &Expr) -> &Expr {
    match expression {
        Expr::Reference(value) => unwrapped(&value.expr),
        Expr::Paren(value) => unwrapped(&value.expr),
        Expr::Group(value) => unwrapped(&value.expr),
        _ => expression,
    }
}

fn identifier(expression: &Expr) -> Option<String> {
    let Expr::Path(path) = unwrapped(expression) else {
        return None;
    };
    path.path.get_ident().map(ToString::to_string)
}

fn filename(expression: &Expr) -> Result<String, String> {
    let Expr::Lit(literal) = unwrapped(expression) else {
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

fn example_path(
    expression: &Expr,
    locals: &BTreeMap<String, String>,
) -> Result<Option<String>, String> {
    if let Some(name) = identifier(expression) {
        return Ok(locals.get(&name).cloned());
    }
    let Expr::MethodCall(join) = unwrapped(expression) else {
        return Ok(None);
    };
    if join.method != "join" {
        return Ok(None);
    }
    let Expr::Call(root) = unwrapped(&join.receiver) else {
        return Ok(None);
    };
    if identifier(&root.func).as_deref() != Some("examples_root") || !root.args.is_empty() {
        return Ok(None);
    }
    if join.args.len() != 1 || join.turbofish.is_some() {
        return Err("example path must use examples_root().join(literal)".into());
    }
    filename(&join.args[0]).map(Some)
}

fn binding(pattern: &Pat) -> Option<&syn::PatIdent> {
    match pattern {
        Pat::Ident(name) => Some(name),
        Pat::Type(typed) => binding(&typed.pat),
        _ => None,
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
        let mut locals = BTreeMap::new();
        // Rust item declarations are in scope before their textual position.
        let mut shadowed: BTreeSet<_> = test
            .block
            .stmts
            .iter()
            .filter_map(|statement| {
                if let Stmt::Item(Item::Fn(function)) = statement {
                    Some(function.sig.ident.to_string())
                } else {
                    None
                }
            })
            .collect();
        if shadowed.contains("examples_root") {
            return Err(format!("{name}: shadowed examples_root input owner"));
        }
        for statement in &test.block.stmts {
            match statement {
                Stmt::Local(local) => {
                    if let Some(pattern) = binding(&local.pat) {
                        let key = pattern.ident.to_string();
                        if key == "examples_root" {
                            return Err(format!("{name}: shadowed examples_root input owner"));
                        }
                        locals.remove(&key);
                        shadowed.insert(key.clone());
                        if pattern.mutability.is_none()
                            && pattern.by_ref.is_none()
                            && let Some(init) = &local.init
                            && let Some(input) = example_path(&init.expr, &locals)?
                        {
                            locals.insert(key, input);
                        }
                    }
                }
                Stmt::Item(Item::Fn(function)) => {
                    shadowed.insert(function.sig.ident.to_string());
                }
                Stmt::Item(Item::Use(_)) => {
                    return Err(format!(
                        "{name}: local imports make harness input ownership ambiguous"
                    ));
                }
                Stmt::Expr(Expr::Assign(assignment), _) => {
                    if let Some(key) = identifier(&assignment.left) {
                        locals.remove(&key);
                    }
                }
                Stmt::Expr(Expr::Call(call), _) => {
                    let Some(helper) = identifier(&call.func) else {
                        continue;
                    };
                    let arity = match helper.as_str() {
                        "drive_parity" | "check_dropout_eval_and_c_rejection" => 2,
                        "assert_check_clean" => 1,
                        _ => continue,
                    };
                    if shadowed.contains(&helper) || call.args.len() != arity {
                        return Err(format!("{name}: ambiguous {helper} input"));
                    }
                    let input = if helper == "check_dropout_eval_and_c_rejection" {
                        filename(&call.args[0])?
                    } else {
                        example_path(&call.args[0], &locals)?
                            .ok_or_else(|| format!("{name}: unresolved {helper} example input"))?
                    };
                    inputs.insert(input);
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
