//! Exact source registration provenance, before PyO3 removes its attributes.
//!
//! The supported registrar is deliberately direct. Conditional attributes,
//! macro-produced registrations and arbitrary helper calls require an explicit
//! implementation of their provenance before they can enter this census.

use quote::ToTokens;
use serde::Serialize;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, ImplItem, Item, Meta, Token};

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Registration {
    pub owner: Option<String>,
    pub python_name: String,
    pub rust_name: String,
    pub kind: String,
    pub line: usize,
    pub column: usize,
}

fn path_is(path: &syn::Path, names: &[&str]) -> bool {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .eq(names.iter().map(|s| s.to_string()))
}

fn direct_pyo3(path: &syn::Path, name: &str) -> bool {
    path.leading_colon.is_some() && path_is(path, &["pyo3", name])
}

fn options(attr: &Attribute) -> Result<Vec<Meta>, String> {
    match &attr.meta {
        Meta::Path(_) => Ok(vec![]),
        Meta::List(list) => {
            // Python argument syntax such as (*args, **kwargs) is not a
            // Rust expression. PyO3 validates that syntax when compiling the
            // actual registrar. Capacity comes from every Rust parameter;
            // this parser only recognizes the signature option's boundary.
            let mut input = list.tokens.clone().into_iter().peekable();
            let mut normalized = proc_macro2::TokenStream::new();
            while let Some(token) = input.next() {
                let signature =
                    matches!(&token, proc_macro2::TokenTree::Ident(name) if name == "signature");
                normalized.extend([token]);
                if signature
                    && matches!(input.peek(), Some(proc_macro2::TokenTree::Punct(punctuation)) if punctuation.as_char() == '=')
                {
                    normalized.extend([input.next().unwrap()]);
                    match input.next() {
                        Some(proc_macro2::TokenTree::Group(group))
                            if group.delimiter() == proc_macro2::Delimiter::Parenthesis =>
                        {
                            normalized.extend([proc_macro2::TokenTree::Group(
                                proc_macro2::Group::new(
                                    proc_macro2::Delimiter::Parenthesis,
                                    proc_macro2::TokenStream::new(),
                                ),
                            )]);
                        }
                        _ => {
                            return Err(
                                "Python signature requires a parenthesized argument list".into()
                            );
                        }
                    }
                }
            }
            Punctuated::<Meta, Token![,]>::parse_terminated
                .parse2(normalized)
                .map(|v| v.into_iter().collect())
                .map_err(|error| error.to_string())
        }
        _ => Err("unsupported PyO3 attribute form".into()),
    }
}

fn renamed(meta: &Meta) -> Result<String, String> {
    if let Meta::NameValue(value) = meta
        && let Expr::Lit(literal) = &value.value
        && let syn::Lit::Str(name) = &literal.lit
    {
        return Ok(name.value());
    }
    Err("PyO3 name must be an explicit string literal".into())
}

fn attributes(
    attrs: &[Attribute],
    primary: Option<&str>,
    rust_name: &str,
) -> Result<(String, String), String> {
    let mut name = rust_name.to_string();
    let mut kind = if primary == Some("pyfunction") {
        "function"
    } else {
        "method"
    };
    let mut primary_count = 0;
    let mut kind_count = 0;
    let mut named = false;
    for attr in attrs {
        if attr.path().is_ident("doc") || attr.path().is_ident("allow") {
            continue;
        }
        if primary.is_some_and(|p| direct_pyo3(attr.path(), p)) {
            primary_count += 1;
        } else if attr.path().is_ident("pyo3") {
            if primary == Some("pymethods") || primary == Some("pyclass") {
                return Err("unsupported class/impl attribute provenance".into());
            }
        } else if primary.is_none()
            && ["getter", "setter", "new", "classmethod", "staticmethod"]
                .iter()
                .any(|key| attr.path().is_ident(key))
        {
            kind_count += 1;
            let key = attr.path().get_ident().unwrap().to_string();
            kind = match key.as_str() {
                "new" => {
                    name = "__new__".into();
                    "constructor"
                }
                "getter" => "getter",
                "setter" => "setter",
                "classmethod" => "classmethod",
                _ => "staticmethod",
            };
            if matches!(kind, "getter" | "setter") {
                name = match &attr.meta {
                    Meta::Path(_) => rust_name
                        .strip_prefix(if kind == "getter" { "get_" } else { "set_" })
                        .unwrap_or(rust_name)
                        .into(),
                    Meta::List(list) => {
                        if let Ok(id) = syn::parse2::<syn::Ident>(list.tokens.clone()) {
                            id.to_string()
                        } else if let Ok(text) = syn::parse2::<syn::LitStr>(list.tokens.clone()) {
                            text.value()
                        } else {
                            return Err("unsupported property name provenance".into());
                        }
                    }
                    _ => return Err("unsupported property provenance".into()),
                };
            } else if !matches!(attr.meta, Meta::Path(_)) {
                return Err("unsupported callable kind attribute arguments".into());
            }
            continue;
        } else {
            return Err(format!(
                "unsupported registration attribute provenance: {}",
                attr.path().to_token_stream()
            ));
        }
        for option in options(attr)? {
            if option.path().is_ident("name") {
                if named {
                    return Err("duplicate PyO3 name provenance".into());
                }
                name = renamed(&option)?;
                named = true;
            } else if option.path().is_ident("signature")
                && !matches!(primary, Some("pymethods" | "pyclass"))
            {
                if !matches!(option, Meta::List(_) | Meta::NameValue(_)) {
                    return Err("unsupported Python signature attribute".into());
                }
            } else if option.path().is_ident("unsendable") && primary == Some("pyclass") {
                if !matches!(option, Meta::Path(_)) {
                    return Err("unsupported pyclass option".into());
                }
            } else {
                return Err(format!(
                    "unsupported PyO3 option provenance: {}",
                    option.to_token_stream()
                ));
            }
        }
    }
    if let Some(primary) = primary
        && primary_count != 1
    {
        return Err(format!(
            "missing or duplicate explicit ::pyo3::{} provenance",
            primary
        ));
    }
    if kind_count > 1 || (named && matches!(kind, "getter" | "setter" | "constructor")) {
        return Err("ambiguous callable kind/name provenance".into());
    }
    Ok((name, kind.into()))
}

fn ident_type(ty: &syn::Type) -> Result<String, String> {
    if let syn::Type::Path(path) = ty
        && path.qself.is_none()
        && let Some(id) = path.path.get_ident()
    {
        return Ok(id.to_string());
    }
    Err("registration requires a direct local Rust type, not an alias/path expression".into())
}

fn registrar(file: &syn::File) -> Result<(Vec<String>, Vec<String>), String> {
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function) if function.sig.ident == "register_module" => Some(function),
            _ => None,
        })
        .collect();
    if functions.len() != 1 {
        return Err("missing or duplicate register_module provenance".into());
    }
    let function = functions[0];
    if function
        .attrs
        .iter()
        .any(|attr| !attr.path().is_ident("doc"))
    {
        return Err("unsupported registrar attribute provenance".into());
    }
    let mut classes = vec![];
    let mut exports = vec![];
    for statement in &function.block.stmts {
        let syn::Stmt::Expr(expr, semicolon) = statement else {
            return Err("unsupported registrar statement provenance".into());
        };
        if semicolon.is_none() && expr.to_token_stream().to_string() == "Ok (())" {
            continue;
        }
        let Expr::Try(expr) = expr else {
            return Err("unsupported registrar expression provenance".into());
        };
        let Expr::MethodCall(call) = &*expr.expr else {
            return Err("unsupported registrar helper provenance".into());
        };
        if !matches!(&*call.receiver, Expr::Path(path) if path.path.is_ident("module")) {
            return Err("registration receiver must be the actual module parameter".into());
        }
        match call.method.to_string().as_str() {
            "add_class" if call.args.is_empty() => {
                let args = &call
                    .turbofish
                    .as_ref()
                    .ok_or("missing registered class type")?
                    .args;
                if args.len() != 1 {
                    return Err("ambiguous registered class type".into());
                }
                let syn::GenericArgument::Type(ty) = &args[0] else {
                    return Err("unsupported registered class argument".into());
                };
                classes.push(ident_type(ty)?);
            }
            "add_function" if call.turbofish.is_none() && call.args.len() == 1 => {
                let Expr::Try(wrapped) = &call.args[0] else {
                    return Err("function registration requires direct wrap_pyfunction".into());
                };
                let Expr::Macro(mac) = &*wrapped.expr else {
                    return Err("function registration requires direct wrap_pyfunction".into());
                };
                if !direct_pyo3(&mac.mac.path, "wrap_pyfunction") {
                    return Err("unsupported wrapper alias provenance".into());
                }
                let args = Punctuated::<syn::Ident, Token![,]>::parse_terminated
                    .parse2(mac.mac.tokens.clone())
                    .map_err(|error| error.to_string())?;
                if args.len() != 2 || args[1] != "module" {
                    return Err("unsupported function registration provenance".into());
                }
                exports.push(args[0].to_string());
            }
            "add"
                if call.to_token_stream().to_string()
                    == "module . add (\"ChelisError\" , module . py () . get_type :: < ChelisError > ())" =>
                {}
            _ => return Err("unsupported dynamic module registration provenance".into()),
        }
    }
    if classes
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != classes.len()
        || exports
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != exports.len()
    {
        return Err("duplicate registrar provenance".into());
    }
    Ok((classes, exports))
}

pub fn registrations(source: &str) -> Result<Vec<Registration>, String> {
    let file = syn::parse_file(source).map_err(|error| error.to_string())?;
    let (classes, functions) = registrar(&file)?;
    let mut rows = vec![];
    for name in functions {
        let candidates: Vec<_> = file
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(function) if function.sig.ident == name => Some(function),
                _ => None,
            })
            .collect();
        if candidates.len() != 1 {
            return Err("missing or duplicate registered function provenance".into());
        }
        let function = candidates[0];
        let (python_name, kind) = attributes(&function.attrs, Some("pyfunction"), &name)?;
        let position = function.sig.ident.span().start();
        rows.push(Registration {
            owner: None,
            python_name,
            rust_name: name,
            kind,
            line: position.line,
            column: position.column + 1,
        });
    }
    for owner in classes {
        let candidates: Vec<_> = file
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Struct(class) if class.ident == owner => Some(class),
                _ => None,
            })
            .collect();
        if candidates.len() != 1 {
            return Err("missing or duplicate registered pyclass provenance".into());
        }
        let class = candidates[0];
        attributes(&class.attrs, Some("pyclass"), &owner)?;
        for item in &file.items {
            let Item::Impl(implementation) = item else {
                continue;
            };
            if ident_type(&implementation.self_ty).as_ref() != Ok(&owner) {
                continue;
            }
            if !implementation
                .attrs
                .iter()
                .any(|attr| direct_pyo3(attr.path(), "pymethods"))
            {
                if !implementation.attrs.is_empty() && implementation.trait_.is_none() {
                    return Err("unsupported possible pymethods attribute provenance".into());
                }
                continue;
            }
            attributes(&implementation.attrs, Some("pymethods"), &owner)?;
            if implementation.trait_.is_some() || !implementation.generics.params.is_empty() {
                return Err("unsupported generic/trait pymethods provenance".into());
            }
            for method in &implementation.items {
                let ImplItem::Fn(function) = method else {
                    return Err("unsupported macro/non-callable pymethods provenance".into());
                };
                let rust_name = function.sig.ident.to_string();
                let (python_name, kind) = attributes(&function.attrs, None, &rust_name)?;
                let position = function.sig.ident.span().start();
                rows.push(Registration {
                    owner: Some(owner.clone()),
                    python_name,
                    rust_name,
                    kind,
                    line: position.line,
                    column: position.column + 1,
                });
            }
        }
    }
    let mut seen = std::collections::BTreeMap::<_, Vec<&str>>::new();
    for row in &rows {
        let kinds = seen.entry((&row.owner, &row.python_name)).or_default();
        if kinds.contains(&row.kind.as_str())
            || (!kinds.is_empty()
                && !["getter", "setter"]
                    .iter()
                    .all(|kind| kinds.contains(kind) || row.kind == *kind))
        {
            return Err("ambiguous registered callable provenance".into());
        }
        kinds.push(&row.kind);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(items: &str, exports: &str) -> String {
        format!("{items}\npub fn register_module(module: &Bound<PyModule>) -> PyResult<()> {{ {exports} Ok(()) }}")
            .replace("#[pyfunction", "#[::pyo3::pyfunction")
            .replace("#[pymethods", "#[::pyo3::pymethods")
            .replace("#[pyclass", "#[::pyo3::pyclass")
            .replace("wrap_pyfunction!", "::pyo3::wrap_pyfunction!")
    }

    #[test]
    fn registered_rust_identities_follow_all_pyo3_kinds_and_renames() {
        let source = module(
            r#"
            fn public_fn() -> bool { true }
            #[pyfunction(name = "public_fn")]
            fn numeric_function(value: f64) -> f64 { value }
            #[pyclass(name = "Model")]
            struct Handle;
            impl Handle {
                fn dtype(&self) -> &str { "f32" }
                fn new() -> Self { Self }
                fn run(&self) -> bool { true }
            }
            #[pymethods]
            impl Handle {
                #[new]
                fn create(dtype: i32) -> Self { Self }
                #[getter(dtype)]
                fn dtype_code(&self) -> i32 { 7 }
                #[setter(dtype)]
                fn assign_dtype(&mut self, dtype: i32) {}
                #[pyo3(name = "run")]
                fn compute(&self, value: f64) -> f64 { value }
                #[classmethod]
                fn from_source(cls: &Bound<PyType>, source: &str) -> Self { Self }
                #[staticmethod]
                fn supported() -> bool { true }
                fn __call__(&self) -> bool { true }
            }
        "#,
            r#"
            module.add_class::<Handle>()?;
            module.add_function(wrap_pyfunction!(numeric_function, module)?)?;
        "#,
        );
        let rows = registrations(&source).unwrap();
        let actual: Vec<_> = rows
            .iter()
            .map(|row| {
                (
                    row.owner.as_deref(),
                    row.python_name.as_str(),
                    row.rust_name.as_str(),
                    row.kind.as_str(),
                )
            })
            .collect();
        assert_eq!(
            actual,
            vec![
                (None, "public_fn", "numeric_function", "function"),
                (Some("Handle"), "__new__", "create", "constructor"),
                (Some("Handle"), "dtype", "dtype_code", "getter"),
                (Some("Handle"), "dtype", "assign_dtype", "setter"),
                (Some("Handle"), "run", "compute", "method"),
                (Some("Handle"), "from_source", "from_source", "classmethod"),
                (Some("Handle"), "supported", "supported", "staticmethod"),
                (Some("Handle"), "__call__", "__call__", "method"),
            ]
        );
        assert!(rows.iter().all(|row| row.line > 0 && row.column > 0));
    }

    #[test]
    fn unregistered_helpers_are_not_provenance() {
        let source = module(
            "fn helper() -> bool { true }",
            "module.add_function(wrap_pyfunction!(helper, module)?)?;",
        );
        assert!(registrations(&source).unwrap_err().contains("pyfunction"));
        let source = module(
            "struct Handle; impl Handle { fn dtype(&self) -> bool { true } }",
            "module.add_class::<Handle>()?;",
        );
        assert!(registrations(&source).unwrap_err().contains("pyclass"));
    }

    #[test]
    fn unsupported_or_ambiguous_registration_fails_closed() {
        for items in [
            "#[cfg(any())] #[pyfunction] fn f() {}",
            "#[cfg_attr(any(), pyfunction)] fn f() {}",
            "#[other_macro] #[pyfunction] fn f() {}",
            "#[pyfunction] fn f() {} #[pyfunction] fn f() {}",
            "#[pyfunction] fn f() {}",
        ] {
            let exports = if items == "#[pyfunction] fn f() {}" {
                "helper(module)?;"
            } else {
                "module.add_function(wrap_pyfunction!(f, module)?)?;"
            };
            assert!(registrations(&module(items, exports)).is_err(), "{items}");
        }
        for methods in [
            "#[cfg(any())] fn f(&self) {}",
            "#[other_macro] fn f(&self) {}",
            "#[classattr] fn f() -> i32 { 7 }",
            "make_methods!();",
            "#[getter(value)] fn a(&self) -> i32 { 1 } #[getter(value)] fn b(&self) -> i32 { 2 }",
        ] {
            let items =
                format!("#[pyclass] struct Handle; #[pymethods] impl Handle {{ {methods} }}");
            assert!(
                registrations(&module(&items, "module.add_class::<Handle>()?;")).is_err(),
                "{methods}"
            );
        }
    }

    #[test]
    fn property_prefixes_and_separate_getter_setter_roots_are_explicit() {
        let items = "#[pyclass] struct Handle; #[pymethods] impl Handle { #[getter] fn get_value(&self) -> bool { true } #[setter] fn set_value(&mut self, value: i64) {} }";
        let rows = registrations(&module(items, "module.add_class::<Handle>()?;")).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].python_name, "value");
        assert_eq!(rows[1].python_name, "value");
        assert_ne!(rows[0].kind, rows[1].kind);
    }

    #[test]
    fn python_argument_markers_do_not_hide_rust_capacity() {
        let items = "#[pyclass] struct Handle; #[pymethods] impl Handle { #[pyo3(signature = (*args, **kwargs))] fn __call__(&self, args: &PyTuple, kwargs: Option<&PyDict>) {} }";
        let rows = registrations(&module(items, "module.add_class::<Handle>()?;")).unwrap();
        assert_eq!(rows[0].rust_name, "__call__");
        assert_eq!(rows[0].kind, "method");
        let unsupported = items.replace("(*args, **kwargs)", "choose_signature()");
        assert!(registrations(&module(&unsupported, "module.add_class::<Handle>()?;")).is_err());
    }
}
