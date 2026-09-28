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
use chelis_unord::{UnordMap, UnordSet};
use std::collections::VecDeque;

/// A module node carries the tag and metadata map outside its child vector,
/// so its declarations start after the module-name atom alone.
const MODULE_DECL_OFFSET: usize = 1;

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
    prune_entry(exprs, entry, ReferenceScope::Syntactic)
}

/// Resource admission concerns executed definitions, not same-spelled locals
/// or references in type metadata. Reuse the checker's capture/binding analysis
/// for this scope, leaving the broader authoring/build pruner unchanged.
pub(crate) fn prune_checked_runtime_to_entry(exprs: Vec<DeepExpr>, entry: &str) -> Vec<DeepExpr> {
    prune_entry(exprs, entry, ReferenceScope::LexicalRuntime)
}

#[derive(Clone, Copy)]
enum ReferenceScope {
    Syntactic,
    LexicalRuntime,
}

fn prune_entry(exprs: Vec<DeepExpr>, entry: &str, scope: ReferenceScope) -> Vec<DeepExpr> {
    // chelis#1125 PP7 / spec/04-type-system.md §10 [04-TOT-5]: a module
    // wrapper has the single spelling `Expr::Node` whether it came from Surf
    // expansion or from a `.dp` file through `parse_and_stamp_file`. Before
    // the legacy list spelling was deleted, a reader of only one spelling
    // recognized no module on the other, fell through to the flat-sibling
    // branch, found no `def` there either, and returned the program UNPRUNED.
    match exprs.as_slice() {
        [DeepExpr::Node(node, _)] if node.tag() == DeepTag::Module => {
            let DeepExpr::Node(mut module, span) = exprs.into_iter().next().expect("len-1 slice")
            else {
                unreachable!("matched Node above");
            };
            // The module-name atom is child 0 and the declarations follow.
            // `import` / `export` live after the name atom but are non-decl
            // elements, so the tail pruner keeps them via its keep-non-decl
            // filter. The split is clamped to the child count so a
            // header-only module (no decls) is a no-op.
            let mut children = module.children_slice().to_vec();
            let split = MODULE_DECL_OFFSET.min(children.len());
            let decls = children.split_off(split);
            children.extend(prune_top_level_entry(decls, entry, scope));
            module
                .try_replace_children(children)
                .expect("pruning a module drops whole declarations, never its name binder");
            vec![DeepExpr::Node(module, span)]
        }
        _ => prune_top_level_entry(exprs, entry, scope),
    }
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
    prune_top_level_entry(exprs, entry, ReferenceScope::Syntactic)
}

fn prune_top_level_entry(
    exprs: Vec<DeepExpr>,
    entry: &str,
    scope: ReferenceScope,
) -> Vec<DeepExpr> {
    let is_local_def = exprs.iter().any(|expr| deep_def_name(expr) == Some(entry));
    if !is_local_def {
        return exprs;
    }
    prune_reachable(exprs, std::iter::once(entry.to_string()), scope)
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
    prune_reachable(exprs, seeds, ReferenceScope::Syntactic)
}

fn prune_reachable(
    exprs: Vec<DeepExpr>,
    seeds: impl IntoIterator<Item = String>,
    scope: ReferenceScope,
) -> Vec<DeepExpr> {
    let def_map = exprs
        .iter()
        .filter_map(|expr| deep_def_name(expr).map(|name| (name.to_string(), expr)))
        .collect::<UnordMap<_, _>>();

    let mut reachable = UnordSet::<String>::new();
    let mut queue = VecDeque::from_iter(seeds);
    while let Some(name) = queue.pop_front() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(expr) = def_map.get(&name) {
            let lexical_references;
            let references = match scope {
                ReferenceScope::Syntactic => deep_referenced_vars(expr),
                ReferenceScope::LexicalRuntime => {
                    lexical_references = chelis_types::linearity::free_runtime_variables(expr);
                    lexical_references.iter().map(String::as_str).collect()
                }
            };
            for reference in references {
                if def_map.contains_key(reference) && !reachable.contains(reference) {
                    queue.push_back(reference.to_string());
                }
            }
        }
    }

    // `def_map` borrows `exprs`; the wrapper's canonical `Drop` makes that
    // borrow explicit through scope end unless we retire it before moving the
    // declarations into the filtered result.
    drop(def_map);

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
    match decl_head(expr)? {
        (DeepTag::Def, name) => Some(name),
        _ => None,
    }
}

/// The name of a `def` or `defsig` declaration, or `None`. Used to decide
/// which elements the reachable filter applies to (both a function's `def`
/// and its `defsig` are dropped together when unreachable).
pub fn deep_named_decl_name(expr: &DeepExpr) -> Option<&str> {
    match decl_head(expr)? {
        (DeepTag::Def | DeepTag::Defsig, name) => Some(name),
        _ => None,
    }
}

/// The decoded tag and leading name atom of a declaration node, or `None` for
/// anything else.
///
/// chelis#1125 PP7 / [04-TOT-5]: the two name readers above once read only
/// the deleted list spelling, so on a stamped node every declaration read as
/// nameless. `prune_top_level_to_reachable_defs` then saw no local `def`,
/// took its unknown-entry escape hatch, and returned the program unpruned;
/// `prune_to_reachable_seeds`'s keep-filter would likewise have kept every
/// declaration. Reading the carrier is the whole repair: the reachability
/// rule below is unchanged.
fn decl_head(expr: &DeepExpr) -> Option<(DeepTag, &str)> {
    match expr {
        DeepExpr::Node(node, _) => match node.children_slice().first() {
            Some(DeepExpr::Atom(DeepAtom::Name(name), _)) => Some((node.tag(), name.as_str())),
            _ => None,
        },
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

fn collect_annotation_references<'a>(metadata: &'a chelis_deep::Metadata, out: &mut Vec<&'a str>) {
    if let Some(targets) = metadata.wrt() {
        out.extend(targets.variables().map(|var| var.name().value().as_str()));
    }
    metadata.visit_expressions(&mut |value, _| collect_deep_referenced_vars(value, out));
}

fn collect_deep_referenced_vars<'a>(expr: &'a DeepExpr, out: &mut Vec<&'a str>) {
    match expr {
        DeepExpr::Atom(_, _) => {}
        DeepExpr::MetaExpr(meta, _) => {
            collect_annotation_references(&meta.metadata, out);
            collect_deep_referenced_vars(&meta.expr, out);
        }
        DeepExpr::Map(map, _) => {
            collect_annotation_references(map, out);
        }
        DeepExpr::Node(node, _) => {
            collect_annotation_references(node.meta(), out);
            use chelis_deep::node::ChildRef;
            if node.tag() == DeepTag::Var {
                for child_ref in node.children_iter() {
                    if let ChildRef::Syntax(DeepExpr::Atom(DeepAtom::Name(name), _)) = child_ref {
                        out.push(name.as_str());
                    }
                }
            }
            for child_ref in node.children_iter() {
                let child_expr = match child_ref {
                    ChildRef::Expr(e)
                    | ChildRef::Syntax(e)
                    | ChildRef::Type(e)
                    | ChildRef::EffectHandler(e)
                    | ChildRef::Bypass(e) => Some(e),
                    ChildRef::Binder(_) | ChildRef::Selector(_) => None,
                };
                if let Some(child) = child_expr {
                    collect_deep_referenced_vars(child, out);
                }
            }
        }
        DeepExpr::BareList(elems, _) => {
            for elem in elems {
                collect_deep_referenced_vars(elem, out);
            }
        }
        DeepExpr::UnknownForm(data) => {
            collect_annotation_references(&data.meta, out);
            for child in &data.children {
                collect_deep_referenced_vars(child, out);
            }
        }
    }
}

#[cfg(test)]
mod tests;
