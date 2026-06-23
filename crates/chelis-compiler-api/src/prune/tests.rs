//! Unit tests for reachable-defs pruning, paired positive/negative.
//!
//! These pin the reachability + module-descent behavior on real expanded
//! Deep programs (parsed Surf -> desugar -> expand), not hand-built AST, so a
//! change in the expanded module shape is caught here.

use super::*;
use chelis_macros::ExpansionOptions;

/// Parse + desugar + expand Surf into the Deep program form `prune_to_entry`
/// consumes (one `module` node).
fn expand(source: &str) -> Vec<DeepExpr> {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parses");
    chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
        &ExpansionOptions::default(),
    )
    .expect("expansion succeeds")
    .into_exprs()
}

/// The sorted `def` names remaining in a pruned program (descending into a
/// module wrapper if present).
fn def_names(exprs: &[DeepExpr]) -> Vec<String> {
    let mut names = Vec::new();
    for expr in exprs {
        if let DeepExpr::List(list, _) = expr
            && list_tag(list) == Some("module")
        {
            for child in &list.elements {
                if let Some(name) = deep_def_name(child) {
                    names.push(name.to_string());
                }
            }
        } else if let Some(name) = deep_def_name(expr) {
            names.push(name.to_string());
        }
    }
    names.sort();
    names
}

const MULTI_DEF_MODULE: &str = "module Demo.Pricer\n\
    import Other.Pkg (missing_sym)\n\
    def scaled[n](v: tensor[n, f32], k: tensor[n, f32]) -> tensor[n, f32] = mul(v, k)\n\
    def priced[n](v: tensor[n, f32], k: tensor[n, f32], b: tensor[n, f32]) -> tensor[n, f32] = add(scaled(v, k), b)\n\
    def unrelated[n](v: tensor[n, f32]) -> tensor[n, f32] = missing_sym(v)\n";

// ===========================================================================
// Positive: pruning to an entry keeps its closure, drops the rest.
// ===========================================================================

#[test]
fn prune_keeps_entry_closure_and_drops_unrelated_defs() {
    let pruned = prune_to_entry(expand(MULTI_DEF_MODULE), "priced");
    // `priced` uses `scaled`; both kept. `unrelated` (and its `missing_sym`
    // reference) is genuinely unreachable from `priced`, so it is dropped.
    assert_eq!(
        def_names(&pruned),
        vec!["priced".to_string(), "scaled".to_string()],
        "pruning to `priced` keeps {{priced, scaled}} and drops `unrelated`"
    );
}

#[test]
fn prune_descends_module_wrapper_not_just_flat_siblings() {
    // The expanded program is a single (module ...) node, not flat def
    // siblings; this asserts pruning actually descended into it (the input is
    // length-1, the output is still length-1, but its def children shrank).
    let expanded = expand(MULTI_DEF_MODULE);
    assert_eq!(expanded.len(), 1, "expanded Surf is one module node");
    assert!(
        matches!(&expanded[0], DeepExpr::List(list, _) if list_tag(list) == Some("module")),
        "the single top-level node is a module"
    );
    let pruned = prune_to_entry(expanded, "scaled");
    assert_eq!(pruned.len(), 1, "the module wrapper is preserved");
    // `scaled` has no local-def references, so only `scaled` remains.
    assert_eq!(def_names(&pruned), vec!["scaled".to_string()]);
}

#[test]
fn prune_preserves_module_head_and_import() {
    // The module tag, metadata map, module-name atom, and the `import`
    // element (a non-decl child) are kept so the pruned module is still a
    // well-formed module node carrying its import context.
    let pruned = prune_to_entry(expand(MULTI_DEF_MODULE), "priced");
    let DeepExpr::List(list, _) = &pruned[0] else {
        panic!("expected a module list");
    };
    assert_eq!(list_tag(list), Some("module"));
    assert!(
        matches!(list.elements.first(), Some(DeepExpr::Atom(DeepAtom::Symbol(t), _)) if t == "module"),
        "the module tag is first"
    );
    assert!(
        matches!(list.elements.get(2), Some(DeepExpr::Atom(DeepAtom::Symbol(n), _)) if n == "demo.pricer"),
        "the module-name atom is retained at index 2"
    );
    let has_import = list
        .elements
        .iter()
        .any(|e| matches!(e, DeepExpr::List(l, _) if list_tag(l) == Some("import")));
    assert!(
        has_import,
        "the import element (a non-decl child) is preserved through pruning"
    );
}

#[test]
fn flat_top_level_form_prunes_without_a_module_wrapper() {
    // The build-path shape: flat def siblings (no module wrapper). Pruning
    // operates directly on the siblings.
    let exprs = vec![
        expand("module M\ndef a[n](v: tensor[n, f32]) -> tensor[n, f32] = mul(v, v)\n")
            .into_iter()
            .next()
            .and_then(|m| match m {
                DeepExpr::List(list, _) => list
                    .elements
                    .into_iter()
                    .find(|e| deep_def_name(e) == Some("a")),
                _ => None,
            })
            .expect("a def `a`"),
    ];
    // `a` is the only def and is the entry, so it survives.
    let pruned = prune_top_level_to_reachable_defs(exprs, "a");
    assert_eq!(def_names(&pruned), vec!["a".to_string()]);
}

// ===========================================================================
// Negative twins.
// ===========================================================================

#[test]
fn prune_does_not_drop_a_def_in_the_entrys_own_closure() {
    // The safety property: if the entry transitively uses a def, that def must
    // NOT be dropped. Pruning to `priced` must keep `scaled` (negative twin of
    // "drops unrelated": it must not over-prune a real dependency).
    let pruned = prune_to_entry(expand(MULTI_DEF_MODULE), "priced");
    assert!(
        def_names(&pruned).contains(&"scaled".to_string()),
        "a def reachable from the entry must survive pruning"
    );
}

#[test]
fn unknown_entry_leaves_the_program_unchanged() {
    // An entry that is not a local def must NOT silently empty the program;
    // it is returned unchanged so the unknown name fails downstream the same
    // way it would without pruning.
    let original = expand(MULTI_DEF_MODULE);
    let original_names = def_names(&original);
    let pruned = prune_to_entry(original, "does_not_exist");
    assert_eq!(
        def_names(&pruned),
        original_names,
        "an unknown entry leaves every def in place"
    );
}
