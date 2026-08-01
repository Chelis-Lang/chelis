//! Reachable-defs pruning of a Deep program to a single named entry.
//!
//! The build path (`chelis build`) already prunes a reef-linked program to
//! the defs reachable from its entry so an unused transitive dependency
//! cannot drag an unlowerable function into the lowering target. The WI-3
//! graph-extraction producer needs the same thing for a SINGLE-FILE source:
//! extract one entry (e.g. a pricer function) from a module that also defines
//! unrelated functions which reference unresolved imports, without those
//! unrelated functions blocking the target's lowering.
//!
//! ## Two program shapes
//!
//! The reef linker flattens a linked program into top-level `def`/`defsig`
//! siblings, so the build-path pruner ([`prune_top_level_to_reachable_defs`])
//! operates on a flat `Vec<DeepExpr>`. A single source file that has not been
//! reef-linked is instead expanded into ONE top-level `(module {} name
//! (import ...) (export ...) (defsig ...) (def ...) ...)` node, with the
//! `def`/`defsig` pairs as the module's children. [`prune_to_entry`] handles
//! both: it descends into a `module` node and prunes its children, and falls
//! back to the flat-sibling form otherwise.
//!
//! ## What pruning does and does not hide
//!
//! Pruning drops only GENUINELY-UNREACHABLE named decls. It seeds the
//! reachable set from the entry name and follows every `var` reference whose
//! name is a local `def`. A reference to a name that is not a local `def`
//! (e.g. an unresolved import, or a built-in primitive) is never added to the
//! reachable set, so it cannot keep an unrelated def alive. The corollary is
//! the safety property: if the ENTRY's own reachable closure references an
//! unresolved symbol, that reference is inside a kept def and still surfaces
//! at the type checker. Pruning never hides a real error in the target's own
//! dependencies; it only removes defs the target does not transitively use.

use chelis_deep::DeepTag;
use chelis_deep::{Atom as DeepAtom, Expr as DeepExpr};
use std::collections::{HashMap, HashSet, VecDeque};

/// The `def`/`defsig` children of a `module` node start after the tag,
/// metadata map, and module-name atom.
const MODULE_DECL_OFFSET: usize = 3;

/// Prune `exprs` to the defs reachable from `entry`, handling both the
/// flat-top-level-`def`-siblings form and the single `(module ...)` wrapper
/// form.
///
/// If `exprs` is a single `module` node, its `def`/`defsig` children are
/// pruned in place and the module wrapper (with its `import`/`export`
/// metadata) is preserved. Otherwise the top-level siblings are pruned
/// directly. Non-decl elements (the module name, `import`, `export`, bare
/// metadata) are always kept.
///
/// Returns the program unchanged if `entry` is not a local `def` (so an
/// unknown entry surfaces downstream as the same "unbound"/unknown-output
/// path it would without pruning, rather than silently emptying the program).
pub fn prune_to_entry(exprs: Vec<DeepExpr>, entry: &str) -> Vec<DeepExpr> {
    if let [DeepExpr::List(list, _)] = exprs.as_slice()
        && list_tag(list) == Some(DeepTag::Module)
    {
        // Descend into the single module node: prune its children, keep the
        // wrapper and its non-decl elements (name, import, export).
        let DeepExpr::List(module, meta) = exprs.into_iter().next().expect("len-1 slice") else {
            unreachable!("matched List above");
        };
        let chelis_deep::List { mut elements } = module;
        // Keep the fixed head (tag, metadata map, module-name atom) verbatim
        // and prune the tail. `import` / `export` live in the tail (after the
        // name atom) but are non-decl elements, so the tail pruner keeps them
        // via its keep-non-decl filter. The split is clamped to the element
        // count so a header-only module (no decls) is a no-op.
        let split = MODULE_DECL_OFFSET.min(elements.len());
        let decls = elements.split_off(split);
        let pruned_decls = prune_top_level_to_reachable_defs(decls, entry);
        elements.extend(pruned_decls);
        return vec![DeepExpr::List(chelis_deep::List { elements }, meta)];
    }
    prune_top_level_to_reachable_defs(exprs, entry)
}

/// Prune a flat list of top-level Deep declarations to those reachable from
/// `entry`, by BFS over `var` references that resolve to a local `def`. Both
/// the `def` body and its sibling `defsig` are dropped for an unreachable
/// name. Non-decl elements are kept.
///
/// If `entry` is not a local `def`, the program is returned UNCHANGED (rather
/// than emptied), so an unknown entry fails downstream the same way it would
/// without pruning. This is the WI-3 single-entry contract; the build path
/// uses [`prune_to_reachable_seeds`] directly with its multi-name seed set.
pub fn prune_top_level_to_reachable_defs(exprs: Vec<DeepExpr>, entry: &str) -> Vec<DeepExpr> {
    let is_local_def = exprs.iter().any(|expr| deep_def_name(expr) == Some(entry));
    if !is_local_def {
        return exprs;
    }
    prune_to_reachable_seeds(exprs, std::iter::once(entry.to_string()))
}

/// Prune a flat list of top-level Deep declarations to those reachable from
/// the given seed names, by BFS over `var` references that resolve to a local
/// `def`. Both the `def` body and its sibling `defsig` are dropped for an
/// unreachable name; non-decl elements are kept.
///
/// This is the shared core both the WI-3 single-entry pruner and the build
/// path use, so the reachability definition lives in one place. Unlike
/// [`prune_top_level_to_reachable_defs`], an EMPTY seed set drops every named
/// decl (reachable stays empty) — the build path relies on that to strip a
/// linked program down to nothing when the entry program contributes no
/// top-level defs.
pub fn prune_to_reachable_seeds(
    exprs: Vec<DeepExpr>,
    seeds: impl IntoIterator<Item = String>,
) -> Vec<DeepExpr> {
    let def_map = exprs
        .iter()
        .filter_map(|expr| deep_def_name(expr).map(|name| (name.to_string(), expr)))
        .collect::<HashMap<_, _>>();

    let mut reachable = HashSet::<String>::new();
    let mut queue = VecDeque::from_iter(seeds);
    while let Some(name) = queue.pop_front() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(expr) = def_map.get(&name) {
            for reference in deep_referenced_vars(expr) {
                if def_map.contains_key(reference) && !reachable.contains(reference) {
                    queue.push_back(reference.to_string());
                }
            }
        }
    }

    exprs
        .into_iter()
        .filter(|expr| {
            deep_named_decl_name(expr)
                .map(|name| reachable.contains(name))
                .unwrap_or(true)
        })
        .collect()
}

/// The name of a `def` declaration (not `defsig`), or `None`. This is the
/// reachability seed key: only a `def` provides a body to follow references
/// through.
pub fn deep_def_name(expr: &DeepExpr) -> Option<&str> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match (list.tag(), list.elements.get(2)) {
        (Some(DeepTag::Def), Some(DeepExpr::Atom(DeepAtom::Name(name), _))) => Some(name.as_str()),
        _ => None,
    }
}

/// The name of a `def` or `defsig` declaration, or `None`. Used to decide
/// which elements the reachable filter applies to (both a function's `def`
/// and its `defsig` are dropped together when unreachable).
pub fn deep_named_decl_name(expr: &DeepExpr) -> Option<&str> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match (list.tag(), list.elements.get(2)) {
        (Some(DeepTag::Def | DeepTag::Defsig), Some(DeepExpr::Atom(DeepAtom::Name(name), _))) => {
            Some(name.as_str())
        }
        _ => None,
    }
}

/// Every `var` reference name in `expr`, in pre-order. A `var` node is the
/// 3-tuple `(var {} name)`.
pub fn deep_referenced_vars(expr: &DeepExpr) -> Vec<&str> {
    let mut out = Vec::new();
    collect_deep_referenced_vars(expr, &mut out);
    out
}

fn collect_deep_referenced_vars<'a>(expr: &'a DeepExpr, out: &mut Vec<&'a str>) {
    match expr {
        DeepExpr::Atom(_, _) => {}
        DeepExpr::MetaExpr(meta, _) => collect_deep_referenced_vars(&meta.expr, out),
        DeepExpr::Map(map, _) => {
            for (_, value) in &map.entries {
                collect_deep_referenced_vars(value, out);
            }
        }
        DeepExpr::List(list, _) => {
            if let (Some(DeepTag::Var), Some(DeepExpr::Atom(DeepAtom::Name(name), _))) =
                (list.tag(), list.elements.get(2))
            {
                out.push(name.as_str());
            }
            for child in &list.elements {
                collect_deep_referenced_vars(child, out);
            }
        }
        // Transitional arms for new Expr variants (#908)
        DeepExpr::Node(node, _) => {
            for child in node.expr_children() {
                collect_deep_referenced_vars(child, out);
            }
        }
        DeepExpr::BareList(elems, _) => {
            for elem in elems {
                collect_deep_referenced_vars(elem, out);
            }
        }
        DeepExpr::UnknownForm(data) => {
            for child in &data.children {
                collect_deep_referenced_vars(child, out);
            }
        }
    }
}

fn list_tag(list: &chelis_deep::List) -> Option<DeepTag> {
    list.tag()
}

#[cfg(test)]
mod tests;
