//! Rule `opaque-domain-construction` — domain types marked
//! `opaque: true` may only be materialized by code in their defining
//! module. This is the lint half of the proven-constructor discipline used by
//! downstream domain shells: admitted code must call the constructors whose
//! postconditions are proved, not write the representation directly.

use crate::{Context, LintError, PreparedRuleState, Rule, Surface, Violation};
use chelis_deep::DeepTag;
use chelis_deep::Span;
use chelis_deep::ast as deep;
use chelis_surf::ast as surf;
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::path::Path;

/// The lint's opaque catalog, keyed by type leaf and defining module
/// (CR-9).
///
/// `opaque_modules_by_leaf` maps each opaque type leaf to its defining
/// modules, making each construction-site lookup independent of the total
/// corpus size. `opaque_defining_modules` supports the untyped Deep
/// `record-update` check without scanning all opaque declarations.
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
/// under the same `module = None` key, so they cannot be used as a
/// corpus-wide shadow bucket. Named-module shadow remains corpus-wide.
#[derive(Debug, Default)]
struct Catalog {
    opaque_modules_by_leaf: HashMap<String, HashSet<String>>,
    opaque_defining_modules: HashSet<String>,
    declared_leaves: HashSet<(Option<String>, String)>,
}

impl Catalog {
    fn is_empty(&self) -> bool {
        self.opaque_modules_by_leaf.is_empty()
    }

    fn insert_opaque(&mut self, name: &str, module: &str) {
        self.opaque_modules_by_leaf
            .entry(type_leaf(name).to_string())
            .or_default()
            .insert(module.to_string());
        self.opaque_defining_modules.insert(module.to_string());
    }
}

/// Immutable corpus data paired with the current file's module-less
/// declarations. The latter must never leak to another checked file (CR2-7).
struct CatalogContext<'a> {
    corpus: &'a Catalog,
    current_file_module_less_leaves: &'a HashSet<String>,
}

impl Deref for CatalogContext<'_> {
    type Target = Catalog;

    fn deref(&self) -> &Self::Target {
        self.corpus
    }
}

#[cfg(test)]
thread_local! {
    static CATALOG_PARSE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn reset_catalog_parse_count() {
    CATALOG_PARSE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
fn catalog_parse_count() -> usize {
    CATALOG_PARSE_COUNT.with(std::cell::Cell::get)
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

    fn prepare_run(
        &self,
        _root: &Path,
        entries: &[crate::walker::Entry],
        _policy: &crate::policy::TraversalPolicy,
    ) -> Result<PreparedRuleState, LintError> {
        Ok(Box::new(collect_surf_catalog_from_entries(entries)))
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

    fn check_prepared(
        &self,
        ctx: &Context<'_>,
        prepared: &(dyn std::any::Any + Send + Sync),
    ) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        match ctx.surface {
            Surface::SurfSource => {
                if let Some(catalog) = prepared.downcast_ref::<Catalog>() {
                    check_surf_prepared(ctx, source, catalog)
                } else {
                    debug_assert!(false, "opaque rule received another rule's prepared state");
                    check_surf(ctx, source)
                }
            }
            Surface::DeepSource => check_deep(ctx, source),
            _ => Vec::new(),
        }
    }
}

fn check_surf(ctx: &Context<'_>, source: &str) -> Vec<Violation> {
    let Ok(decls) = chelis_surf::parser::parse_str(source) else {
        return Vec::new();
    };
    let mut catalog = match collect_surf_catalog(ctx.root) {
        Ok(catalog) => catalog,
        Err(error) => {
            return vec![Violation {
                rule_id: "opaque-domain-construction".to_string(),
                spec_ref: "§12.1".to_string(),
                path: ctx.path.to_path_buf(),
                line: None,
                col: None,
                message: format!("could not prepare opaque catalog: {error}"),
            }];
        }
    };
    if catalog.is_empty() {
        collect_surf_decls_catalog(&decls, None, &mut catalog);
    }
    check_surf_decls_with_catalog(ctx, source, &decls, &catalog)
}

fn check_surf_prepared(ctx: &Context<'_>, source: &str, catalog: &Catalog) -> Vec<Violation> {
    let Ok(decls) = chelis_surf::parser::parse_str(source) else {
        return Vec::new();
    };
    check_surf_decls_with_catalog(ctx, source, &decls, catalog)
}

fn check_surf_decls_with_catalog(
    ctx: &Context<'_>,
    source: &str,
    decls: &[surf::Decl],
    catalog: &Catalog,
) -> Vec<Violation> {
    if catalog.is_empty() {
        return Vec::new();
    }
    // CR2-7: the module-less shadow for THIS file's constructions is
    // this file's own top-level type declarations, never the shared
    // corpus `None` bucket.
    let current_file_module_less_leaves = surf_module_less_leaves(decls);
    let catalog = CatalogContext {
        corpus: catalog,
        current_file_module_less_leaves: &current_file_module_less_leaves,
    };
    let mut out = Vec::new();
    check_surf_decls(ctx, source, decls, &catalog, None, &mut out);
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

fn collect_surf_catalog(root: &Path) -> Result<Catalog, LintError> {
    // Direct `Rule::check` callers may supply in-memory source with a
    // not-yet-written file path. Validate any ancestor policy, then let the
    // caller seed the catalog from that current source instead of treating
    // the absent synthetic root as a filesystem walk failure.
    if !root.exists() {
        crate::policy::TraversalPolicy::load_for(root)?;
        return Ok(Catalog::default());
    }
    let entries: Vec<crate::walker::Entry> = crate::walker::walk(root)?
        .into_iter()
        .collect::<Result<_, _>>()?;
    Ok(collect_surf_catalog_from_entries(&entries))
}

fn collect_surf_catalog_from_entries(entries: &[crate::walker::Entry]) -> Catalog {
    let mut out = Catalog::default();
    for entry in entries {
        if entry.surface != Some(Surface::SurfSource) {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(&entry.path) else {
            continue;
        };
        #[cfg(test)]
        CATALOG_PARSE_COUNT.with(|count| count.set(count.get() + 1));
        let Ok(decls) = chelis_surf::parser::parse_str(&source) else {
            continue;
        };
        collect_surf_decls_catalog(&decls, None, &mut out);
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
                if *opaque && let Some(module) = module.as_deref() {
                    out.insert_opaque(name, module);
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
    catalog: &CatalogContext<'_>,
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
    catalog: &CatalogContext<'_>,
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
    let catalog = collect_deep_catalog(&exprs);
    if catalog.is_empty() {
        return Vec::new();
    }
    // A `.dp` is one check unit (one source), so its module-less
    // declarations belong to one anonymous module; the file-scoped
    // module-less shadow is this source's own top-level type leaves.
    let current_file_module_less_leaves = deep_module_less_leaves(&exprs);
    let catalog = CatalogContext {
        corpus: &catalog,
        current_file_module_less_leaves: &current_file_module_less_leaves,
    };
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
        let Some((tag, _, children)) = deep_node_parts(expr) else {
            continue;
        };
        if matches!(tag, DeepTag::Deftype | DeepTag::Typealias)
            && let Some(name) = children.first().and_then(sym_str)
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
    let Some((tag, _, children)) = deep_node_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::Module => {
            let module_name = children.first().and_then(sym_str).map(str::to_string);
            for child in children.iter().skip(1) {
                collect_deep_decls_catalog(child, module_name.clone(), out);
            }
        }
        // Every `deftype`/`typealias` records that its module declares a
        // local type of that leaf (CR-9); opaque deftypes also enter the
        // leaf/module indices.
        DeepTag::Deftype => {
            if let Some(name) = children.first().and_then(sym_str) {
                out.declared_leaves
                    .insert((module.clone(), type_leaf(name).to_string()));
                // CR3: only catalog opaque types from a named module --
                // a module-less @opaque is a checker declaration error
                // and would collapse distinct module-less files under
                // the shared `None` key. See the Surf collector.
                if meta_bool(expr, "opaque")
                    && let Some(module) = module.as_deref()
                {
                    out.insert_opaque(name, module);
                }
            }
        }
        DeepTag::Typealias => {
            if let Some(name) = children.first().and_then(sym_str) {
                out.declared_leaves
                    .insert((module, type_leaf(name).to_string()));
            }
        }
        _ => {
            for child in children {
                collect_deep_decls_catalog(child, module.clone(), out);
            }
        }
    }
}

fn check_deep_expr(
    ctx: &Context<'_>,
    source: &str,
    expr: &deep::Expr,
    catalog: &CatalogContext<'_>,
    module: Option<&str>,
    out: &mut Vec<Violation>,
) {
    let Some((tag, _, children)) = deep_node_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::Module => {
            let module_name = children.first().and_then(sym_str);
            for child in children.iter().skip(1) {
                check_deep_expr(ctx, source, child, catalog, module_name, out);
            }
            return;
        }
        DeepTag::Record => {
            if let Some(name) = children.first().and_then(sym_str)
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
        DeepTag::Cast => {
            if let Some(target) = children.get(1).and_then(type_name_from_type_expr)
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
        DeepTag::RecordUpdate => {
            let typed_target = type_name_from_meta(expr)
                .or_else(|| children.first().and_then(type_name_from_meta_expr));
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
    for child in children {
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
    catalog: &CatalogContext<'_>,
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
    // No local shadow: use the leaf index to determine whether at least one
    // opaque definition belongs to another module. This remains constant-time
    // with respect to unrelated opaque declarations.
    let Some(defining_modules) = catalog.opaque_modules_by_leaf.get(leaf) else {
        return false;
    };
    match current_module {
        Some(module) => defining_modules.len() > usize::from(defining_modules.contains(module)),
        None => !defining_modules.is_empty(),
    }
}

/// Whether `current_module` is outside the defining module of EVERY
/// opaque type (used for the fail-closed untyped Deep `record-update`
/// arm). True when the current module declares no opaque type itself.
fn is_outside_all_opaque_modules(
    current_module: Option<&str>,
    catalog: &CatalogContext<'_>,
) -> bool {
    current_module.is_none_or(|module| !catalog.opaque_defining_modules.contains(module))
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

fn deep_node_parts(expr: &deep::Expr) -> Option<(DeepTag, &deep::MetaMap, &[deep::Expr])> {
    match expr {
        deep::Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        deep::Expr::List(list, _) => {
            let tag = list.tag()?;
            let meta = match list.elements.get(1)? {
                deep::Expr::Map(meta, _) => meta,
                _ => return None,
            };
            Some((tag, meta, list.elements.get(2..)?))
        }
        _ => None,
    }
}

fn sym_str(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Name(value), _) => Some(value.as_str()),
        _ => None,
    }
}

fn meta_bool(expr: &deep::Expr, key: &str) -> bool {
    deep_node_parts(expr).is_some_and(|(_, map, _)| {
        map.entries.iter().any(|(entry_key, value)| {
            entry_key == key && matches!(value, deep::Expr::Atom(deep::Atom::Bool(true), _))
        })
    })
}

fn type_name_from_meta(expr: &deep::Expr) -> Option<&str> {
    let (_, meta, _) = deep_node_parts(expr)?;
    let value = meta
        .entries
        .iter()
        .find_map(|(key, value)| (key == "type").then_some(value))?;
    type_name_from_type_expr(value)
}

fn type_name_from_meta_expr(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Node(..) | deep::Expr::List(..) => type_name_from_meta(expr),
        deep::Expr::MetaExpr(meta, _) => meta
            .entries
            .iter()
            .find_map(|(key, value)| (key == "type").then_some(value))
            .and_then(type_name_from_type_expr),
        _ => None,
    }
}

fn type_name_from_type_expr(expr: &deep::Expr) -> Option<&str> {
    let (tag, _, children) = deep_node_parts(expr)?;
    match tag {
        DeepTag::TPrim => children.first().and_then(sym_str),
        DeepTag::TAdt => children
            .iter()
            .find_map(sym_str)
            .or_else(|| children.first().and_then(sym_str)),
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
  (def {} update
    (record-update {type: (t-adt {} Probability)} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 2.0)))))
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
  (def {} update
    (record-update {} (var {} p) (kv {} value (lit {type: (t-prim {} f32)} 2.0)))))
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
  (def {} coerce (cast {} (var {} x) (t-adt {} Probability))))
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

    #[test]
    fn catalog_indexes_opaque_definitions_by_leaf_and_module() {
        let mut catalog = Catalog::default();
        for index in 0..256 {
            let source = format!(
                "module Domain{index}\n@opaque\ntype Type{index} = | Type{index} {{ value: f32 }}\n"
            );
            let decls = chelis_surf::parser::parse_str(&source).expect("parse indexed type");
            collect_surf_decls_catalog(&decls, None, &mut catalog);
        }
        for module in ["First.Domain", "Second.Domain"] {
            let source =
                format!("module {module}\n@opaque\ntype Shared = | Shared {{ value: f32 }}\n");
            let decls = chelis_surf::parser::parse_str(&source).expect("parse shared type");
            collect_surf_decls_catalog(&decls, None, &mut catalog);
        }

        assert_eq!(catalog.opaque_modules_by_leaf.len(), 257);
        assert_eq!(catalog.opaque_defining_modules.len(), 258);
        assert_eq!(
            catalog
                .opaque_modules_by_leaf
                .get("Type173")
                .expect("leaf index")
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["Domain173"]
        );
        assert_eq!(
            catalog
                .opaque_modules_by_leaf
                .get("Shared")
                .expect("shared leaf index")
                .len(),
            2,
            "same-leaf opaque definitions retain each defining module without a corpus scan"
        );
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

    fn lint_root(root: &Path) -> Vec<Violation> {
        let rules: Vec<Box<dyn Rule>> = vec![Box::new(OpaqueDomainConstruction)];
        crate::lint(root, &rules).expect("lint root")
    }

    #[test]
    fn catalog_ignores_surf_sources_in_every_canonical_skipped_tree() {
        let temp = tempdir().expect("tempdir");
        std::fs::create_dir_all(temp.path().join("spec")).expect("create policy spec dir");
        std::fs::write(
            temp.path().join("spec/lint.md"),
            "# Test lint policy\n\n### 12.2 Traversal exclusions\n",
        )
        .expect("write policy spec");
        std::fs::write(
            temp.path().join("chelis-lint.toml"),
            "version = 1\nspec = \"spec/lint.md\"\n\n[[exclude]]\npattern = \"tests/corpus/opaque_invariants/programs/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
        )
        .expect("write repository traversal policy");
        let skipped = [
            ("target", "TargetSecret", "target_secret"),
            (".git", "GitSecret", "git_secret"),
            ("node_modules", "NodeSecret", "node_secret"),
            ("__pycache__", "PySecret", "py_secret"),
            (".venv-issue-603", "VenvSecret", "venv_secret"),
            (
                ".claude/worktrees/generated",
                "WorktreeSecret",
                "worktree_secret",
            ),
            (
                "tests/corpus/opaque_invariants/programs",
                "CorpusSecret",
                "corpus_secret",
            ),
        ];
        let mut agent = String::from("module Agent.Strategy\n");
        for (index, (directory, type_name, function_name)) in skipped.iter().enumerate() {
            let directory = temp.path().join(directory);
            std::fs::create_dir_all(&directory).expect("create skipped directory");
            std::fs::write(
                directory.join(format!("opaque_{index}.ch")),
                format!(
                    "module Hidden.Types\n@opaque\ntype {type_name} = | {type_name} {{ value: f32 }}\n"
                ),
            )
            .expect("write skipped opaque declaration");
            agent.push_str(&format!(
                "def {function_name}(x: f32) -> {type_name} = {type_name} {{ value: x }}\n"
            ));
        }
        std::fs::write(temp.path().join("agent.ch"), agent).expect("write admitted source");

        let violations = lint_root(temp.path());
        assert!(
            violations.is_empty(),
            "skipped Surf declarations must not influence the opaque catalog: {violations:?}"
        );
    }

    #[test]
    fn admitted_surf_declaration_still_contributes_to_catalog() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("whale.ch"),
            "module Whale.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
        )
        .expect("write opaque declaration");
        std::fs::write(
            temp.path().join("agent.ch"),
            "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
        )
        .expect("write forge");

        let violations = lint_root(temp.path());
        assert_eq!(violations.len(), 1, "admitted declaration must catalog");
        assert!(violations[0].message.contains("direct record construction"));
    }

    #[test]
    fn catalog_parses_each_admitted_surf_candidate_once_per_invocation() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(temp.path().join("first.ch"), "def first() = 1\n")
            .expect("write first source");
        std::fs::write(temp.path().join("second.ch"), "def second() = 2\n")
            .expect("write second source");
        std::fs::write(temp.path().join("third.ch"), "def third() = 3\n")
            .expect("write third source");

        reset_catalog_parse_count();
        let violations = lint_root(temp.path());
        assert!(violations.is_empty());
        assert_eq!(
            catalog_parse_count(),
            3,
            "three admitted Surf candidates must produce three catalog parses, not one corpus parse per checked file"
        );
    }

    #[test]
    fn repeated_lint_invocations_rebuild_opaque_catalog_state() {
        let temp = tempdir().expect("tempdir");
        let agent = temp.path().join("agent.ch");
        let whale = temp.path().join("whale.ch");
        std::fs::write(
            &agent,
            "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
        )
        .expect("write forge");
        let rules: Vec<Box<dyn Rule>> = vec![Box::new(OpaqueDomainConstruction)];

        assert!(
            crate::lint(temp.path(), &rules)
                .expect("initial lint")
                .is_empty()
        );
        std::fs::write(
            &whale,
            "module Whale.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
        )
        .expect("add opaque declaration");
        assert_eq!(
            crate::lint(temp.path(), &rules)
                .expect("lint after addition")
                .len(),
            1,
            "later invocation must observe a newly added opaque declaration"
        );
        std::fs::write(
            &whale,
            "module Whale.Types\n@opaque\ntype RenamedSecret = | RenamedSecret { value: f32 }\n",
        )
        .expect("edit opaque declaration");
        assert!(
            crate::lint(temp.path(), &rules)
                .expect("lint after edit")
                .is_empty(),
            "later invocation must observe an edited opaque declaration"
        );
        std::fs::write(
            &whale,
            "module Whale.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
        )
        .expect("restore opaque declaration");
        assert_eq!(
            crate::lint(temp.path(), &rules)
                .expect("lint after restoring edit")
                .len(),
            1,
            "later invocation must observe a second edit with the same rule objects"
        );
        std::fs::remove_file(&whale).expect("remove opaque declaration");
        assert!(
            crate::lint(temp.path(), &rules)
                .expect("lint after removal")
                .is_empty(),
            "later invocation must not retain a removed opaque declaration"
        );
    }

    #[test]
    fn invocation_path_preserves_single_file_module_less_and_deep_semantics() {
        let temp = tempdir().expect("tempdir");
        let single = temp.path().join("single.ch");
        std::fs::write(
            &single,
            "module Victim.Types\n@opaque\ntype Secret = | Secret { value: f32 }\ndef make_secret(x: f32) -> Secret = Secret { value: x }\n",
        )
        .expect("write single-file corpus");
        assert!(
            lint_root(&single).is_empty(),
            "single-file root must catalog itself while allowing defining-module construction"
        );
        std::fs::write(
            temp.path().join("victim.ch"),
            "module Victim.Types\n@opaque\ntype Local = | Local { value: f32 }\n",
        )
        .expect("write module-less parity declaration");
        std::fs::write(
            temp.path().join("local.ch"),
            "type Local = | Local { value: f32 }\ndef make_local(x: f32) -> Local = Local { value: x }\n",
        )
        .expect("write module-less local source");
        std::fs::write(
            temp.path().join("module_less_forge.ch"),
            "def forge_local(x: f32) -> Local = Local { value: x }\n",
        )
        .expect("write separate module-less forge");
        std::fs::write(temp.path().join("malformed.ch"), "module ???\n")
            .expect("write malformed candidate");
        let root_violations = lint_root(temp.path());
        assert_eq!(
            root_violations.len(),
            1,
            "one module-less file's local type must not suppress a different module-less file's forge, while defining-module construction, own-file shadow, and malformed candidates remain accepted: {root_violations:?}"
        );
        assert!(
            root_violations[0]
                .message
                .contains("direct record construction")
        );
        assert_eq!(
            root_violations[0]
                .path
                .file_name()
                .and_then(|name| name.to_str()),
            Some("module_less_forge.ch"),
            "the own-file module-less shadow must be allowed while only the other-file forge is rejected"
        );

        let deep = temp.path().join("opaque.dp");
        std::fs::write(
            &deep,
            "(module {} victim.types\n  (deftype {opaque: true} Token () (variant {} Token (field {} value (t-prim {} f32)))))\n(module {} agent.strategy\n  (def {} forge (record {} Token (kv {} value (lit {type: (t-prim {} f32)} 1.0)))))\n",
        )
        .expect("write Deep source");
        assert_eq!(
            lint_root(&deep).len(),
            1,
            "Deep checking remains source-local"
        );
    }

    #[test]
    fn empty_catalog_and_malformed_candidates_are_fail_soft_for_multiple_surf_files() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(temp.path().join("first.ch"), "def first() = 1\n")
            .expect("write first source");
        std::fs::write(temp.path().join("second.ch"), "def second() = 2\n")
            .expect("write second source");
        std::fs::write(temp.path().join("malformed.ch"), "module ???\n")
            .expect("write malformed source");

        reset_catalog_parse_count();
        let violations = lint_root(temp.path());
        assert!(
            violations.is_empty(),
            "an empty opaque catalog and malformed candidate must remain fail-soft: {violations:?}"
        );
        assert_eq!(
            catalog_parse_count(),
            3,
            "every admitted candidate is attempted once even when the catalog remains empty"
        );
    }

    #[test]
    fn malformed_candidate_does_not_hide_another_files_valid_opaque_declaration() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(temp.path().join("malformed.ch"), "module ???\n")
            .expect("write malformed source");
        std::fs::write(
            temp.path().join("victim.ch"),
            "module Victim.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
        )
        .expect("write valid opaque declaration");
        let forge = temp.path().join("forge.ch");
        std::fs::write(
            &forge,
            "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
        )
        .expect("write forge");

        let violations = lint_root(temp.path());
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].path, forge);
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_surf_candidate_is_fail_soft_without_hiding_valid_catalog_entries() {
        use std::os::unix::fs::symlink;

        let temp = tempdir().expect("tempdir");
        symlink(
            temp.path().join("missing-source"),
            temp.path().join("unreadable.ch"),
        )
        .expect("create broken Surf symlink");
        std::fs::write(
            temp.path().join("victim.ch"),
            "module Victim.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
        )
        .expect("write valid opaque declaration");
        let forge = temp.path().join("forge.ch");
        std::fs::write(
            &forge,
            "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
        )
        .expect("write forge");

        let violations = lint_root(temp.path());
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].path, forge);
    }

    #[test]
    fn deep_checks_ignore_a_prepared_surf_catalog_in_the_same_directory_lint() {
        let temp = tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("surf_opaque.ch"),
            "module Victim.Types\n@opaque\ntype Token = | Token { value: f32 }\n",
        )
        .expect("write Surf opaque declaration");
        std::fs::write(
            temp.path().join("deep_record.dp"),
            "(module {} agent.strategy\n  (record {} Token (kv {} value (lit {type: (t-prim {} f32)} 1.0))))\n",
        )
        .expect("write Deep record without a Deep opaque declaration");

        let violations = lint_root(temp.path());
        assert!(
            violations.is_empty(),
            "Deep checks must not consume the prepared repository-wide Surf catalog: {violations:?}"
        );
    }
}
