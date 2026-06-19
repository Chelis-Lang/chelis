//! Rule `opaque-domain-construction` — domain types marked
//! `opaque: true` may only be materialized by code in their defining
//! module. This is the lint half of the proven-constructor discipline used by
//! downstream domain shells: admitted code must call the constructors whose
//! postconditions are proved, not write the representation directly.

use crate::{Context, Rule, Surface, Violation};
use chelis_deep::Span;
use chelis_deep::ast as deep;
use chelis_surf::ast as surf;
use std::collections::HashSet;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OpaqueType {
    name: String,
    module: Option<String>,
}

/// The lint's opaque catalog, keyed by (type leaf, defining module)
/// rather than bare leaf (CR-9).
///
/// `opaque` lists the opaque types and their defining modules.
/// `declared_leaves` is the set of `(module, leaf)` for EVERY type
/// declaration (opaque, non-opaque, alias) -- it records that a module
/// declares a local type of that leaf. A construction site in module M
/// of leaf L is a cross-module forge only when M does NOT declare a
/// local L (so the bare `L` resolves to an imported type): if M
/// declares its own L, the construction is local and must not be
/// flagged, even if an UNRELATED module defines an opaque same-leaf
/// type. The checker keys opacity by (type, defining module); this is
/// the advisory lint approximating that with per-file/per-corpus
/// declaration data (no symbol table).
///
/// CR2-7: the corpus walk records every module-LESS file's declarations
/// under the same `module = None` key, so they would all share one
/// shadow bucket -- a local type in one module-less file would falsely
/// suppress the forge lint for a DIFFERENT module-less file. A
/// module-less file's local types shadow only its OWN constructions, so
/// the `None` (module-less) shadow lookup uses
/// `current_file_module_less_leaves` -- the leaves the CURRENTLY-CHECKED
/// file declares at top level -- never the shared corpus `None` bucket.
/// Named-module shadow keeps using `declared_leaves` so a module split
/// across files still shadows correctly.
#[derive(Debug, Default)]
struct Catalog {
    opaque: Vec<OpaqueType>,
    declared_leaves: HashSet<(Option<String>, String)>,
    current_file_module_less_leaves: HashSet<String>,
}

impl Catalog {
    fn is_empty(&self) -> bool {
        self.opaque.is_empty()
    }

    fn extend(&mut self, other: Catalog) {
        self.opaque.extend(other.opaque);
        self.declared_leaves.extend(other.declared_leaves);
        // `current_file_module_less_leaves` is per-checked-file state,
        // set after the corpus is built; it is not merged across files.
    }

    /// Record the leaves declared at top level (module-less) by the
    /// file currently being checked, for the file-scoped `None` shadow
    /// (CR2-7).
    fn set_current_file_module_less_leaves(&mut self, leaves: HashSet<String>) {
        self.current_file_module_less_leaves = leaves;
    }
}

pub struct OpaqueDomainConstruction;

impl Rule for OpaqueDomainConstruction {
    fn id(&self) -> &str {
        "opaque-domain-construction"
    }

    fn spec_ref(&self) -> &str {
        "§12.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource, Surface::DeepSource]
    }

    fn summary(&self) -> &str {
        "types marked opaque may not be directly constructed, record-updated, or cast into outside their defining module"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        match ctx.surface {
            Surface::SurfSource => check_surf(ctx, source),
            Surface::DeepSource => check_deep(ctx, source),
            _ => Vec::new(),
        }
    }
}

fn check_surf(ctx: &Context<'_>, source: &str) -> Vec<Violation> {
    let Ok(decls) = chelis_surf::parser::parse_str(source) else {
        return Vec::new();
    };
    let mut catalog = collect_surf_catalog(ctx.root);
    if catalog.is_empty() {
        collect_surf_decls_catalog(&decls, None, &mut catalog);
    }
    if catalog.is_empty() {
        return Vec::new();
    }
    // CR2-7: the module-less shadow for THIS file's constructions is
    // this file's own top-level type declarations, never the shared
    // corpus `None` bucket.
    catalog.set_current_file_module_less_leaves(surf_module_less_leaves(&decls));
    let mut out = Vec::new();
    check_surf_decls(ctx, source, &decls, &catalog, None, &mut out);
    out
}

/// Leaves of the type declarations the file declares at TOP LEVEL
/// (outside any `module` wrapper), for the file-scoped module-less
/// shadow (CR2-7).
fn surf_module_less_leaves(decls: &[surf::Decl]) -> HashSet<String> {
    let mut out = HashSet::new();
    for decl in decls {
        match decl {
            surf::Decl::TypeDef { name, .. } | surf::Decl::TypeAlias { name, .. } => {
                out.insert(type_leaf(name).to_string());
            }
            _ => {}
        }
    }
    out
}

fn collect_surf_catalog(root: &Path) -> Catalog {
    if root.is_file() {
        return std::fs::read_to_string(root)
            .ok()
            .and_then(|source| chelis_surf::parser::parse_str(&source).ok())
            .map(|decls| {
                let mut catalog = Catalog::default();
                collect_surf_decls_catalog(&decls, None, &mut catalog);
                catalog
            })
            .unwrap_or_default();
    }
    let mut out = Catalog::default();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("ch") {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let Ok(decls) = chelis_surf::parser::parse_str(&source) else {
            continue;
        };
        let mut catalog = Catalog::default();
        collect_surf_decls_catalog(&decls, None, &mut catalog);
        out.extend(catalog);
    }
    out
}

fn collect_surf_decls_catalog(decls: &[surf::Decl], module: Option<String>, out: &mut Catalog) {
    for decl in decls {
        match decl {
            surf::Decl::Module { name, decls, .. } => {
                collect_surf_decls_catalog(decls, Some(name.clone()), out);
            }
            surf::Decl::TypeDef { name, opaque, .. } => {
                out.declared_leaves
                    .insert((module.clone(), type_leaf(name).to_string()));
                // CR3: `@opaque` requires a NAMED enclosing module (the
                // checker rejects a module-less @opaque as a declaration
                // error). A module-less opaque type keys to the shared
                // `None` module, which collapses distinct module-less
                // files and yields false positives against an invalid
                // declaration. Catalog opaque types only from named
                // modules; defer the invalid module-less declaration to
                // the checker.
                if *opaque && module.is_some() {
                    out.opaque.push(OpaqueType {
                        name: name.clone(),
                        module: module.clone(),
                    });
                }
            }
            surf::Decl::TypeAlias { name, .. } => {
                out.declared_leaves
                    .insert((module.clone(), type_leaf(name).to_string()));
            }
            _ => {}
        }
    }
}

fn check_surf_decls(
    ctx: &Context<'_>,
    source: &str,
    decls: &[surf::Decl],
    catalog: &Catalog,
    module: Option<&str>,
    out: &mut Vec<Violation>,
) {
    for decl in decls {
        match decl {
            surf::Decl::Module { name, decls, .. } => {
                check_surf_decls(ctx, source, decls, catalog, Some(name), out);
            }
            surf::Decl::FunDef { body, .. } => {
                check_surf_expr(ctx, source, body, catalog, module, out);
            }
            surf::Decl::LetDef { value, .. } => {
                check_surf_expr(ctx, source, value, catalog, module, out);
            }
            surf::Decl::Property {
                preconditions,
                body,
                options,
                ..
            } => {
                for expr in preconditions {
                    check_surf_expr(ctx, source, expr, catalog, module, out);
                }
                check_surf_expr(ctx, source, body, catalog, module, out);
                for option in options {
                    let expr = match option {
                        surf::PropertyOption::Tolerance(expr, _)
                        | surf::PropertyOption::Seed(expr, _)
                        | surf::PropertyOption::Samples(expr, _) => expr,
                        surf::PropertyOption::Contract(_, _) => continue,
                    };
                    check_surf_expr(ctx, source, expr, catalog, module, out);
                }
            }
            surf::Decl::MacroDef { body, .. } => {
                check_surf_expr(ctx, source, body, catalog, module, out);
            }
            _ => {}
        }
    }
}

fn check_surf_expr(
    ctx: &Context<'_>,
    source: &str,
    expr: &surf::Expr,
    catalog: &Catalog,
    module: Option<&str>,
    out: &mut Vec<Violation>,
) {
    match expr {
        surf::Expr::Record(name, fields, span) => {
            if is_outside_opaque_module(name, module, catalog) {
                push_violation(
                    ctx,
                    source,
                    *span,
                    out,
                    format!(
                        "opaque domain type `{name}` must be materialized through its proved constructor; direct record construction is only allowed inside the defining module"
                    ),
                );
            }
            for (_, value) in fields {
                check_surf_expr(ctx, source, value, catalog, module, out);
            }
        }
        surf::Expr::Cast(inner, target, span) => {
            if is_outside_opaque_module(target, module, catalog) {
                push_violation(
                    ctx,
                    source,
                    *span,
                    out,
                    format!(
                        "opaque domain type `{target}` cannot be materialized by `cast`; call its proved constructor"
                    ),
                );
            }
            check_surf_expr(ctx, source, inner, catalog, module, out);
        }
        surf::Expr::Apply(func, args, _span) => {
            check_surf_expr(ctx, source, func, catalog, module, out);
            for arg in args {
                check_surf_expr(ctx, source, arg, catalog, module, out);
            }
        }
        surf::Expr::List(items, _) | surf::Expr::Tuple(items, _) | surf::Expr::Par(items, _) => {
            for item in items {
                check_surf_expr(ctx, source, item, catalog, module, out);
            }
        }
        surf::Expr::Access(target, _, _)
        | surf::Expr::TupleGet(target, _, _)
        | surf::Expr::Unary(_, target, _)
        | surf::Expr::Grad(target, _, _)
        | surf::Expr::Vmap(target, _, _)
        | surf::Expr::Jit(target, _)
        | surf::Expr::Realize(target, _)
        | surf::Expr::Copy(target, _)
        | surf::Expr::Borrow(target, _)
        | surf::Expr::Annotate(target, _, _) => {
            check_surf_expr(ctx, source, target, catalog, module, out);
        }
        surf::Expr::Binary(_, lhs, rhs, _)
        | surf::Expr::WithSeed(lhs, rhs, _)
        | surf::Expr::WithDevice(lhs, rhs, _) => {
            check_surf_expr(ctx, source, lhs, catalog, module, out);
            check_surf_expr(ctx, source, rhs, catalog, module, out);
        }
        surf::Expr::Pipe(head, stages, _) => {
            check_surf_expr(ctx, source, head, catalog, module, out);
            for stage in stages {
                check_surf_expr(ctx, source, stage, catalog, module, out);
            }
        }
        surf::Expr::If(cond, then_expr, else_expr, _) => {
            check_surf_expr(ctx, source, cond, catalog, module, out);
            check_surf_expr(ctx, source, then_expr, catalog, module, out);
            check_surf_expr(ctx, source, else_expr, catalog, module, out);
        }
        surf::Expr::Match(scrutinee, arms, _) => {
            check_surf_expr(ctx, source, scrutinee, catalog, module, out);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    check_surf_expr(ctx, source, guard, catalog, module, out);
                }
                check_surf_expr(ctx, source, &arm.body, catalog, module, out);
            }
        }
        surf::Expr::Lambda(_params, body, _) => {
            check_surf_expr(ctx, source, body, catalog, module, out);
        }
        surf::Expr::Block(bindings, body, _) => {
            for binding in bindings {
                check_surf_expr(ctx, source, &binding.value, catalog, module, out);
            }
            check_surf_expr(ctx, source, body, catalog, module, out);
        }
        surf::Expr::Lit(_, _) | surf::Expr::Var(_, _) | surf::Expr::Constructor(_, _) => {}
    }
}

fn check_deep(ctx: &Context<'_>, source: &str) -> Vec<Violation> {
    let Ok(exprs) = chelis_deep::parser::parse_str_strict(source) else {
        return Vec::new();
    };
    let mut catalog = collect_deep_catalog(&exprs);
    if catalog.is_empty() {
        return Vec::new();
    }
    // A `.dp` is one check unit (one source), so its module-less
    // declarations belong to one anonymous module; the file-scoped
    // module-less shadow is this source's own top-level type leaves.
    catalog.set_current_file_module_less_leaves(deep_module_less_leaves(&exprs));
    let mut out = Vec::new();
    for expr in &exprs {
        check_deep_expr(ctx, source, expr, &catalog, None, &mut out);
    }
    out
}

/// Leaves of the top-level (module-less) `deftype`/`typealias` nodes in
/// a Deep source, for the file-scoped module-less shadow (CR2-7).
fn deep_module_less_leaves(exprs: &[deep::Expr]) -> HashSet<String> {
    let mut out = HashSet::new();
    for expr in exprs {
        let Some(list) = as_list(expr) else {
            continue;
        };
        if matches!(tag(list), Some("deftype") | Some("typealias"))
            && let Some(name) = children(list).first().and_then(sym_str)
        {
            out.insert(type_leaf(name).to_string());
        }
    }
    out
}

fn collect_deep_catalog(exprs: &[deep::Expr]) -> Catalog {
    let mut out = Catalog::default();
    for expr in exprs {
        collect_deep_decls_catalog(expr, None, &mut out);
    }
    out
}

fn collect_deep_decls_catalog(expr: &deep::Expr, module: Option<String>, out: &mut Catalog) {
    let Some(list) = as_list(expr) else {
        return;
    };
    match tag(list) {
        Some("module") => {
            let module_name = children(list).first().and_then(sym_str).map(str::to_string);
            for child in children(list).iter().skip(1) {
                collect_deep_decls_catalog(child, module_name.clone(), out);
            }
        }
        // Every `deftype`/`typealias` records that its module declares a
        // local type of that leaf (CR-9); opaque deftypes also enter the
        // opaque list.
        Some("deftype") => {
            if let Some(name) = children(list).first().and_then(sym_str) {
                out.declared_leaves
                    .insert((module.clone(), type_leaf(name).to_string()));
                // CR3: only catalog opaque types from a named module --
                // a module-less @opaque is a checker declaration error
                // and would collapse distinct module-less files under
                // the shared `None` key. See the Surf collector.
                if meta_bool(list, "opaque") && module.is_some() {
                    out.opaque.push(OpaqueType {
                        name: name.to_string(),
                        module,
                    });
                }
            }
        }
        Some("typealias") => {
            if let Some(name) = children(list).first().and_then(sym_str) {
                out.declared_leaves
                    .insert((module, type_leaf(name).to_string()));
            }
        }
        _ => {
            for child in children(list) {
                collect_deep_decls_catalog(child, module.clone(), out);
            }
        }
    }
}

fn check_deep_expr(
    ctx: &Context<'_>,
    source: &str,
    expr: &deep::Expr,
    catalog: &Catalog,
    module: Option<&str>,
    out: &mut Vec<Violation>,
) {
    let Some(list) = as_list(expr) else {
        return;
    };
    match tag(list) {
        Some("module") => {
            let module_name = children(list).first().and_then(sym_str);
            for child in children(list).iter().skip(1) {
                check_deep_expr(ctx, source, child, catalog, module_name, out);
            }
            return;
        }
        Some("record") => {
            if let Some(name) = children(list).first().and_then(sym_str)
                && is_outside_opaque_module(name, module, catalog)
            {
                push_violation(
                    ctx,
                    source,
                    expr.span(),
                    out,
                    format!(
                        "opaque domain type `{name}` must be materialized through its proved constructor; direct record construction is only allowed inside the defining module"
                    ),
                );
            }
        }
        Some("cast") => {
            if let Some(target) = children(list).get(1).and_then(type_name_from_type_expr)
                && is_outside_opaque_module(target, module, catalog)
            {
                push_violation(
                    ctx,
                    source,
                    expr.span(),
                    out,
                    format!(
                        "opaque domain type `{target}` cannot be materialized by `cast`; call its proved constructor"
                    ),
                );
            }
        }
        Some("record-update") => {
            let typed_target = type_name_from_meta(list)
                .or_else(|| children(list).first().and_then(type_name_from_meta_expr));
            if let Some(target) = typed_target
                && is_outside_opaque_module(target, module, catalog)
            {
                push_violation(
                    ctx,
                    source,
                    expr.span(),
                    out,
                    format!(
                        "opaque domain type `{target}` cannot be materialized by `record-update`; call its proved constructor"
                    ),
                );
            } else if typed_target.is_none() && is_outside_all_opaque_modules(module, catalog) {
                push_violation(
                    ctx,
                    source,
                    expr.span(),
                    out,
                    "untyped Deep `record-update` cannot be verified against opaque domain types; add type metadata or call the proved constructor".to_string(),
                );
            }
        }
        _ => {}
    }
    for child in children(list) {
        check_deep_expr(ctx, source, child, catalog, module, out);
    }
}

/// Whether constructing `type_name` in `current_module` materializes
/// an opaque type defined in a DIFFERENT module (CR-9).
///
/// The construction is a cross-module forge only when the current
/// module does not declare its own type of that leaf: a bare `L` in a
/// module that declares a local `L` resolves to the local type (which
/// may be a non-opaque same-leaf type in an unrelated module, or the
/// module's own opaque type), so it is never flagged. Otherwise (no
/// local shadow), it is flagged iff some opaque type of that leaf is
/// defined in another module.
fn is_outside_opaque_module(
    type_name: &str,
    current_module: Option<&str>,
    catalog: &Catalog,
) -> bool {
    let leaf = type_leaf(type_name);
    // Local declaration shadows: the bare name resolves to this
    // module's own type, not an imported opaque one. For a NAMED
    // module the shadow is corpus-wide (a module split across files
    // still shadows). For a MODULE-LESS site the shadow is file-scoped
    // (CR2-7): distinct module-less files collapse into the corpus
    // `None` bucket, so only THIS file's own top-level types may
    // shadow it.
    let shadowed = match current_module {
        Some(module) => catalog
            .declared_leaves
            .contains(&(Some(module.to_string()), leaf.to_string())),
        None => catalog.current_file_module_less_leaves.contains(leaf),
    };
    if shadowed {
        return false;
    }
    // No local shadow: flag iff an opaque same-leaf type is defined in
    // another module (the genuine out-of-module forge).
    catalog
        .opaque
        .iter()
        .any(|opaque| type_leaf(&opaque.name) == leaf && opaque.module.as_deref() != current_module)
}

/// Whether `current_module` is outside the defining module of EVERY
/// opaque type (used for the fail-closed untyped Deep `record-update`
/// arm). True when the current module declares no opaque type itself.
fn is_outside_all_opaque_modules(current_module: Option<&str>, catalog: &Catalog) -> bool {
    !catalog
        .opaque
        .iter()
        .any(|opaque| opaque.module.as_deref() == current_module)
}

fn type_leaf(name: &str) -> &str {
    name.rsplit(['.', ':', '/']).next().unwrap_or(name)
}

fn push_violation(
    ctx: &Context<'_>,
    source: &str,
    span: Span,
    out: &mut Vec<Violation>,
    message: String,
) {
    let (line, col) = line_col(source, span.offset);
    out.push(Violation {
        rule_id: OpaqueDomainConstruction.id().to_string(),
        spec_ref: OpaqueDomainConstruction.spec_ref().to_string(),
        path: ctx.path.to_path_buf(),
        line: Some(line),
        col: Some(col),
        message,
    });
}

fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (idx, ch) in source.char_indices() {
        if idx >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn as_list(expr: &deep::Expr) -> Option<&deep::List> {
    match expr {
        deep::Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn tag(list: &deep::List) -> Option<&str> {
    match list.elements.first()? {
        deep::Expr::Atom(deep::Atom::Symbol(tag), _) => Some(tag.as_str()),
        _ => None,
    }
}

fn meta(list: &deep::List) -> Option<&deep::MetaMap> {
    match list.elements.get(1)? {
        deep::Expr::Map(map, _) => Some(map),
        _ => None,
    }
}

fn children(list: &deep::List) -> &[deep::Expr] {
    if list.elements.len() <= 2 {
        &[]
    } else {
        &list.elements[2..]
    }
}

fn sym_str(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Symbol(value), _) => Some(value.as_str()),
        _ => None,
    }
}

fn meta_bool(list: &deep::List, key: &str) -> bool {
    meta(list).is_some_and(|map| {
        map.entries.iter().any(|(entry_key, value)| {
            entry_key == key && matches!(value, deep::Expr::Atom(deep::Atom::Bool(true), _))
        })
    })
}

fn type_name_from_meta(list: &deep::List) -> Option<&str> {
    let value = meta(list)?
        .entries
        .iter()
        .find_map(|(key, value)| (key == "type").then_some(value))?;
    type_name_from_type_expr(value)
}

fn type_name_from_meta_expr(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::List(list, _) => type_name_from_meta(list),
        deep::Expr::MetaExpr(meta, _) => meta
            .entries
            .iter()
            .find_map(|(key, value)| (key == "type").then_some(value))
            .and_then(type_name_from_type_expr),
        _ => None,
    }
}

fn type_name_from_type_expr(expr: &deep::Expr) -> Option<&str> {
    let list = as_list(expr)?;
    match tag(list) {
        Some("t-prim") => children(list).first().and_then(sym_str),
        Some("t-adt") => children(list)
            .iter()
            .find_map(sym_str)
            .or_else(|| children(list).first().and_then(sym_str)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::tempdir;

    fn run_surf(src: &str) -> Vec<Violation> {
        let path = Path::new("strategy.ch");
        let ctx = Context {
            root: path,
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        OpaqueDomainConstruction.check(&ctx)
    }

    fn run_surf_in_package(agent_src: &str, whale_src: &str) -> Vec<Violation> {
        let temp = tempdir().expect("tempdir");
        let whale = temp.path().join("whale.ch");
        let agent = temp.path().join("agent.ch");
        std::fs::write(&whale, whale_src).expect("write whale source");
        std::fs::write(&agent, agent_src).expect("write agent source");
        let ctx = Context {
            root: temp.path(),
            path: &agent,
            source: Some(agent_src),
            surface: Surface::SurfSource,
        };
        OpaqueDomainConstruction.check(&ctx)
    }

    fn run_deep(src: &str) -> Vec<Violation> {
        let path = Path::new("strategy.dp");
        let ctx = Context {
            root: path,
            path,
            source: Some(src),
            surface: Surface::DeepSource,
        };
        OpaqueDomainConstruction.check(&ctx)
    }

    #[test]
    fn allows_constructor_inside_defining_module() {
        let src = r#"
module Whale.Types
@opaque
type Probability = | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
"#;
        assert!(run_surf(src).is_empty());
    }

    #[test]
    fn rejects_constructor_outside_defining_module() {
        let whale = r#"
module Whale.Types
@opaque
type Probability = | Probability { value: f32 }
"#;
        let agent = r#"
module Agent.Strategy
def bad(x: f32) -> Probability = Probability { value: x }
"#;
        let violations = run_surf_in_package(agent, whale);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].rule_id, "opaque-domain-construction");
        assert!(violations[0].message.contains("direct record construction"));
    }

    #[test]
    fn rejects_deep_record_update_when_type_metadata_names_opaque_type() {
        let src = r#"
(module {} whale.types
  (deftype {opaque: true} Probability () (variant {} Probability (field {} value (t-prim {} f32)))))
(module {} agent.strategy
  (record-update {type: (t-adt {} Probability)} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 2.0))))
"#;
        let violations = run_deep(src);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("record-update"));
    }

    #[test]
    fn rejects_untyped_deep_record_update_outside_opaque_defining_module() {
        let src = r#"
(module {} whale.types
  (deftype {opaque: true} Probability () (variant {} Probability (field {} value (t-prim {} f32)))))
(module {} agent.strategy
  (record-update {} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 2.0))))
"#;
        let violations = run_deep(src);
        assert_eq!(violations.len(), 1);
        assert!(
            violations[0]
                .message
                .contains("untyped Deep `record-update`")
        );
    }

    #[test]
    fn rejects_deep_cast_into_opaque_type() {
        let src = r#"
(module {} whale.types
  (deftype {opaque: true} Probability () (variant {} Probability (field {} value (t-prim {} f32)))))
(module {} agent.strategy
  (cast {} (var {} x) (t-adt {} Probability)))
"#;
        let violations = run_deep(src);
        assert_eq!(violations.len(), 1);
        assert!(
            violations[0]
                .message
                .contains("cannot be materialized by `cast`")
        );
    }

    // ── CR-9: catalog must key by (type, defining module), not leaf ──

    #[test]
    fn does_not_flag_non_opaque_same_leaf_type_in_unrelated_module() {
        // CR-9 false positive: `Whale.Types` defines an @opaque
        // `Probability`; `Other.Domain` defines an UNRELATED non-opaque
        // `Probability` and constructs its OWN type. The lint must NOT
        // flag the unrelated type just because it shares the leaf name.
        let whale = r#"
module Whale.Types
@opaque
type Probability = | Probability { value: f32 }
"#;
        let other = r#"
module Other.Domain
type Probability = | Probability { value: f32 }
def make(x: f32) -> Probability = Probability { value: x }
"#;
        let violations = run_surf_in_package(other, whale);
        assert!(
            violations.is_empty(),
            "a non-opaque same-leaf type in an unrelated module must not be flagged; got {violations:?}"
        );
    }

    #[test]
    fn deep_does_not_flag_non_opaque_same_leaf_type_in_unrelated_module() {
        // CR-9 false positive, Deep surface: same leaf, one opaque
        // (whale.types) and one non-opaque (other.domain) that
        // constructs its own. Only the genuine forge should ever flag.
        let src = r#"
(module {} whale.types
  (deftype {opaque: true} Probability () (variant {} Probability (field {} value (t-prim {} f32)))))
(module {} other.domain
  (deftype {} Probability () (variant {} Probability (field {} value (t-prim {} f32))))
  (record {} Probability (kv {} value (lit {type: (t-prim {} f32)} 2.0))))
"#;
        let violations = run_deep(src);
        assert!(
            violations.is_empty(),
            "non-opaque same-leaf Deep construction in an unrelated module must not flag; got {violations:?}"
        );
    }

    #[test]
    fn still_flags_genuine_out_of_module_forge_with_no_local_shadow() {
        // CR-9 negative parity: the genuine forge -- a module that does
        // NOT declare a local same-leaf type but constructs the opaque
        // type from another module -- must still be flagged.
        let whale = r#"
module Whale.Types
@opaque
type Probability = | Probability { value: f32 }
"#;
        let agent = r#"
module Agent.Strategy
def bad(x: f32) -> Probability = Probability { value: x }
"#;
        let violations = run_surf_in_package(agent, whale);
        assert_eq!(
            violations.len(),
            1,
            "the genuine forge must still flag; got {violations:?}"
        );
        assert!(violations[0].message.contains("direct record construction"));
    }

    #[test]
    fn defining_module_constructing_its_own_opaque_type_is_still_allowed() {
        // CR-9 negative parity: the opaque type's OWN defining module
        // constructing it stays allowed even with the (type, module)
        // keying.
        let src = r#"
module Whale.Types
@opaque
type Probability = | Probability { value: f32 }
def probability(x: f32) -> Probability = Probability { value: x }
"#;
        assert!(run_surf(src).is_empty());
    }

    // ── CR2-7: module-less files must not share one None shadow bucket ──

    /// Write a corpus of named `.ch` files into a temp dir and lint
    /// `checked_rel`. Used for the multi-module-less-file cases that
    /// `run_surf_in_package` (two fixed files) cannot express.
    fn run_corpus(files: &[(&str, &str)], checked_rel: &str) -> Vec<Violation> {
        let temp = tempdir().expect("tempdir");
        for (rel, src) in files {
            std::fs::write(temp.path().join(rel), src).expect("write source");
        }
        let checked = temp.path().join(checked_rel);
        let source = std::fs::read_to_string(&checked).expect("read checked");
        let ctx = Context {
            root: temp.path(),
            path: &checked,
            source: Some(&source),
            surface: Surface::SurfSource,
        };
        OpaqueDomainConstruction.check(&ctx)
    }

    #[test]
    fn module_less_local_type_does_not_suppress_another_module_less_files_forge() {
        // CR2-7 false suppression: the opaque type lives in a NAMED
        // module (`@opaque` requires one). `filea.ch` is module-less
        // and declares its OWN non-opaque `Secret`; `fileb.ch` is a
        // DIFFERENT module-less file that constructs the opaque
        // `Secret`. file A's local type keys to `module = None`, the
        // same bucket as file B, so before the fix it falsely
        // suppressed file B's forge. File B has no local `Secret`, so
        // its forge must be flagged.
        let victim = r#"
module Victim.Types
@opaque
type Secret = | Secret { value: f32 }
"#;
        let filea = r#"
type Secret = | Secret { value: f32 }
def make_local(x: f32) -> Secret = Secret { value: x }
"#;
        let fileb = r#"
def forge(x: f32) -> Secret = Secret { value: x }
"#;
        let violations = run_corpus(
            &[
                ("victim.ch", victim),
                ("filea.ch", filea),
                ("fileb.ch", fileb),
            ],
            "fileb.ch",
        );
        assert_eq!(
            violations.len(),
            1,
            "a module-less file's local type must not suppress a DIFFERENT module-less \
             file's forge; got {violations:?}"
        );
        assert!(violations[0].message.contains("direct record construction"));
    }

    #[test]
    fn module_less_file_constructing_its_own_local_type_is_still_allowed() {
        // CR2-7 negative parity: a module-less file's OWN local type
        // shadows its OWN constructions (file A here is the checked
        // file). It must stay unflagged even though a named module
        // defines an opaque same-leaf type.
        let victim = r#"
module Victim.Types
@opaque
type Secret = | Secret { value: f32 }
"#;
        let filea = r#"
type Secret = | Secret { value: f32 }
def make_local(x: f32) -> Secret = Secret { value: x }
"#;
        let violations = run_corpus(&[("victim.ch", victim), ("filea.ch", filea)], "filea.ch");
        assert!(
            violations.is_empty(),
            "a module-less file constructing its OWN local type must not be flagged; got {violations:?}"
        );
    }

    // ── CR3: a module-less @opaque is invalid (checker rejects it) ──
    //   `@opaque` requires a named enclosing module -- the checker
    //   rejects a module-less @opaque as a declaration error. So the
    //   lint must NOT catalog a module-less opaque type: doing so keys
    //   it to the shared `None` module, collapsing distinct module-less
    //   files and producing a false positive on a same-leaf
    //   construction in an unrelated module. The lint defers the
    //   invalid declaration to the checker.

    #[test]
    fn does_not_flag_construction_against_a_module_less_opaque_definer() {
        // `filea.ch` declares an @opaque type at top level (module-less,
        // which the checker rejects). `fileb.ch` is a NAMED module that
        // constructs a same-leaf type. Before the fix the lint cataloged
        // the module-less opaque under `None` and flagged fileb -- a
        // false positive against an invalid opaque declaration.
        let filea = r#"
@opaque
type Secret = | Secret { value: f32 }
"#;
        let fileb = r#"
module Other.Domain
def f(x: f32) -> Secret = Secret { value: x }
"#;
        let violations = run_corpus(&[("filea.ch", filea), ("fileb.ch", fileb)], "fileb.ch");
        assert!(
            violations.is_empty(),
            "a construction must not be flagged against a module-less (invalid) @opaque \
             definer; the checker rejects the @opaque declaration. got {violations:?}"
        );
    }

    #[test]
    fn deep_does_not_flag_construction_against_a_module_less_opaque_definer() {
        // Deep surface: a top-level `opaque: true` deftype (module-less)
        // must not be cataloged, so a same-leaf construction inside a
        // named module is not flagged against it.
        let src = r#"
(deftype {opaque: true} Secret () (variant {} Secret (field {} value (t-prim {} f32))))
(module {} other.domain
  (record {} Secret (kv {} value (lit {type: (t-prim {} f32)} 2.0))))
"#;
        let violations = run_deep(src);
        assert!(
            violations.is_empty(),
            "a module-less Deep opaque deftype must not be cataloged; got {violations:?}"
        );
    }

    #[test]
    fn named_module_opaque_still_flags_out_of_module_forge_cr3_parity() {
        // CR3 negative parity: a properly NAMED-module @opaque still
        // catalogs and still flags the genuine out-of-module forge.
        let whale = r#"
module Whale.Types
@opaque
type Secret = | Secret { value: f32 }
"#;
        let agent = r#"
module Agent.Strategy
def bad(x: f32) -> Secret = Secret { value: x }
"#;
        let violations = run_surf_in_package(agent, whale);
        assert_eq!(
            violations.len(),
            1,
            "a named-module @opaque must still catalog and flag the forge; got {violations:?}"
        );
        assert!(violations[0].message.contains("direct record construction"));
    }
}
