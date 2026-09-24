use std::collections::{BTreeMap, BTreeSet};
use syn::visit::{self, Visit};

const DIRECT: &[&str] = &["object", "serde", "serde_json", "sha2", "thiserror"];
const CLOSURE: &[&str] = &[
    "chelis-runtime-identity",
    "object",
    "serde",
    "serde_core",
    "serde_derive",
    "serde_json",
    "sha2",
    "digest",
    "block-buffer",
    "crypto-common",
    "generic-array",
    "typenum",
    "cfg-if",
    "cpufeatures",
    "libc",
    "version_check",
    "itoa",
    "memchr",
    "zmij",
    "proc-macro2",
    "quote",
    "syn",
    "unicode-ident",
    "thiserror",
    "thiserror-impl",
];

// Like pipeline_core_dependency_guard, inspect every declared production table,
// including aliases, optional edges, build dependencies and target-specific edges.
pub fn declared(text: &str) -> Result<(), String> {
    let value: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
    fn tables(value: &toml::Value) -> Result<(), String> {
        for kind in ["dependencies", "build-dependencies"] {
            if let Some(table) = value.get(kind) {
                for (alias, dependency) in table.as_table().ok_or("invalid dependency table")? {
                    let name = dependency
                        .get("package")
                        .and_then(toml::Value::as_str)
                        .unwrap_or(alias);
                    if !DIRECT.contains(&name) {
                        return Err(format!("unreviewed declared production dependency: {name}"));
                    }
                }
            }
        }
        Ok(())
    }
    tables(&value)?;
    if let Some(targets) = value.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            tables(target)?;
        }
    }
    Ok(())
}

pub fn resolved(metadata: &serde_json::Value) -> Result<(), String> {
    let packages = metadata["packages"].as_array().ok_or("missing packages")?;
    let names: BTreeMap<_, _> = packages
        .iter()
        .map(|p| {
            Ok((
                p["id"].as_str().ok_or("missing package id")?,
                p["name"].as_str().ok_or("missing name")?,
            ))
        })
        .collect::<Result<_, String>>()?;
    let nodes: BTreeMap<_, _> = metadata["resolve"]["nodes"]
        .as_array()
        .ok_or("missing nodes")?
        .iter()
        .map(|n| Ok((n["id"].as_str().ok_or("missing node id")?, n)))
        .collect::<Result<_, String>>()?;
    let root = names
        .iter()
        .find_map(|(id, name)| (*name == "chelis-runtime-identity").then_some(*id))
        .ok_or("missing core")?;
    let mut pending = vec![(root, vec!["chelis-runtime-identity"])];
    let mut visited = BTreeSet::new();
    while let Some((id, path)) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        let name = *names.get(id).ok_or("unknown package")?;
        if !CLOSURE.contains(&name) {
            return Err(format!("unreviewed production path: {}", path.join(" -> ")));
        }
        let node = nodes.get(id).ok_or("missing graph node")?;
        let reviewed: Option<&[&str]> = match name {
            "object" => Some(&[
                "read",
                "read_core",
                "std",
                "archive",
                "coff",
                "elf",
                "macho",
                "pe",
                "unaligned",
                "xcoff",
            ]),
            "sha2" => Some(&["default", "std"]),
            "serde" => Some(&["default", "std", "derive", "serde_derive", "rc", "alloc"]),
            "serde_json" => Some(&["default", "std", "raw_value", "alloc"]),
            "thiserror" => Some(&["default", "std"]),
            _ => None,
        };
        if let Some(reviewed) = reviewed {
            for feature in node["features"].as_array().ok_or("missing features")? {
                let feature = feature.as_str().ok_or("invalid feature")?;
                if !reviewed.contains(&feature) {
                    return Err(format!("unreviewed {name} production feature: {feature}"));
                }
            }
        }
        for dependency in node["deps"].as_array().ok_or("missing deps")? {
            let kinds = dependency["dep_kinds"]
                .as_array()
                .ok_or("missing dependency kinds")?;
            if kinds.is_empty() {
                return Err("empty dependency kinds".into());
            }
            if kinds.iter().any(|kind| kind["kind"] != "dev") {
                let id = dependency["pkg"].as_str().ok_or("invalid dependency id")?;
                let mut path = path.clone();
                path.push(*names.get(id).ok_or("unknown dependency")?);
                pending.push((id, path));
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct Effects {
    aliases: BTreeMap<String, Vec<String>>,
    violations: Vec<String>,
}

fn test_only(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| attribute.path().is_ident("cfg")
        && matches!(attribute.parse_args::<syn::Meta>(), Ok(syn::Meta::Path(path)) if path.is_ident("test")))
}

impl Effects {
    fn import(&mut self, prefix: Vec<String>, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Path(path) => {
                let mut prefix = prefix;
                prefix.push(path.ident.to_string());
                self.import(prefix, &path.tree);
            }
            syn::UseTree::Name(name) => {
                let mut path = prefix;
                path.push(name.ident.to_string());
                self.check(&path);
                self.aliases.insert(name.ident.to_string(), path);
            }
            syn::UseTree::Rename(rename) => {
                let mut path = prefix;
                path.push(rename.ident.to_string());
                self.check(&path);
                self.aliases.insert(rename.rename.to_string(), path);
            }
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.import(prefix.clone(), item);
                }
            }
            syn::UseTree::Glob(_) => {
                self.check(&prefix);
                // An unknown imported name must not hide an effect provider.
                if prefix
                    .first()
                    .is_some_and(|name| ["std", "core", "alloc"].contains(&name.as_str()))
                {
                    self.violations
                        .push("standard-library glob import hides capabilities".into());
                }
            }
        }
    }

    fn check(&mut self, parts: &[String]) {
        let mut path = parts.to_vec();
        let mut seen = BTreeSet::new();
        while let Some(root) = path.first().cloned() {
            if !seen.insert(root.clone()) {
                break;
            }
            let Some(alias) = self.aliases.get(&root) else {
                break;
            };
            let mut expanded = alias.clone();
            expanded.extend_from_slice(&path[1..]);
            path = expanded;
        }
        let Some(root) = path.first().map(String::as_str) else {
            return;
        };
        let module = path.get(1).map(String::as_str).unwrap_or("");
        if (["std", "core", "alloc"].contains(&root)
            && ![
                "",
                "collections",
                "cmp",
                "fmt",
                "hash",
                "str",
                "string",
                "vec",
                "option",
                "result",
                "marker",
                "mem",
                "convert",
                "iter",
                "array",
                "slice",
                "num",
                "borrow",
                "ops",
                "error",
                "default",
                "primitive",
                "any",
                "boxed",
                "ascii",
                "char",
            ]
            .contains(&module))
            || [
                "getrandom",
                "rand",
                "tracing",
                "log",
                "reqwest",
                "tokio",
                "libc",
                "chelis_runtime_identity_build",
            ]
            .contains(&root)
            || path.iter().any(|name| {
                [
                    "Fn",
                    "FnMut",
                    "FnOnce",
                    "UnsafeCell",
                    "Cell",
                    "RefCell",
                    "HashMap",
                    "HashSet",
                    "RandomState",
                ]
                .contains(&name.as_str())
            })
        {
            self.violations
                .push(format!("effect-capable path: {}", path.join("::")));
        }
    }
}

impl<'ast> Visit<'ast> for Effects {
    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if !test_only(&item.attrs) {
            visit::visit_item_mod(self, item);
        }
    }
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        if !test_only(&item.attrs) {
            visit::visit_item_fn(self, item);
        }
    }
    fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
        if !test_only(&item.attrs) {
            self.import(Vec::new(), &item.tree);
        }
    }
    fn visit_item_extern_crate(&mut self, item: &'ast syn::ItemExternCrate) {
        let name = item.ident.to_string();
        let alias = item
            .rename
            .as_ref()
            .map(|(_, alias)| alias.to_string())
            .unwrap_or_else(|| name.clone());
        self.aliases.insert(alias, vec![name]);
    }
    fn visit_expr_unsafe(&mut self, _: &'ast syn::ExprUnsafe) {
        self.violations
            .push("unsafe effects require explicit boundary review".into());
    }
    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.check(
            &path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>(),
        );
        visit::visit_path(self, path);
    }
    fn visit_item_foreign_mod(&mut self, _: &'ast syn::ItemForeignMod) {
        self.violations
            .push("foreign effects are not data ports".into());
    }
    fn visit_item_static(&mut self, _: &'ast syn::ItemStatic) {
        self.violations
            .push("global state is not captured input".into());
    }
    fn visit_type_bare_fn(&mut self, _: &'ast syn::TypeBareFn) {
        self.violations
            .push("callback port may perform effects".into());
    }
    fn visit_macro(&mut self, item: &'ast syn::Macro) {
        let name = item.path.segments.last().unwrap().ident.to_string();
        if [
            "env",
            "option_env",
            "include",
            "include_bytes",
            "include_str",
            "print",
            "println",
            "eprint",
            "eprintln",
            "dbg",
            "thread_local",
        ]
        .contains(&name.as_str())
        {
            self.violations.push(format!("ambient macro: {name}"));
        }
        // Inspect arguments as expressions, not token strings. Macro expansion is
        // otherwise a blind spot in syn::Visit, including nested helper calls.
        use syn::parse::Parser;
        let parsed = if name == "vec" {
            let tokens = &item.tokens;
            syn::parse2::<syn::Expr>(quote::quote!([#tokens])).map(|expression| vec![expression])
        } else if name == "matches" {
            (|input: syn::parse::ParseStream<'_>| {
                let expression = input.parse::<syn::Expr>()?;
                input.parse::<syn::Token![,]>()?;
                let _pattern = syn::Pat::parse_multi_with_leading_vert(input)?;
                let mut expressions = vec![expression];
                if input.peek(syn::Token![if]) {
                    input.parse::<syn::Token![if]>()?;
                    expressions.push(input.parse::<syn::Expr>()?);
                }
                if input.peek(syn::Token![,]) {
                    input.parse::<syn::Token![,]>()?;
                }
                Ok(expressions)
            })
            .parse2(item.tokens.clone())
        } else if [
            "format",
            "format_args",
            "write",
            "writeln",
            "panic",
            "unreachable",
            "assert",
            "assert_eq",
            "assert_ne",
            "debug_assert",
            "debug_assert_eq",
            "debug_assert_ne",
            "todo",
            "unimplemented",
        ]
        .contains(&name.as_str())
        {
            syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated
                .parse2(item.tokens.clone())
                .map(|expressions| expressions.into_iter().collect())
        } else {
            self.violations
                .push(format!("unreviewed macro expansion: {name}"));
            return;
        };
        match parsed {
            Ok(expressions) => {
                for expression in &expressions {
                    self.visit_expr(expression);
                }
            }
            Err(error) => self
                .violations
                .push(format!("unparsed macro arguments: {name}: {error}")),
        }
        visit::visit_macro(self, item);
    }
}

pub fn source(text: &str) -> Result<(), String> {
    let syntax = syn::parse_file(text).map_err(|error| error.to_string())?;
    let mut effects = Effects::default();
    effects.visit_file(&syntax);
    if effects.violations.is_empty() {
        Ok(())
    } else {
        Err(effects.violations.join("\n"))
    }
}
