//! Unit tests for reachable-defs pruning, paired positive/negative.
//!
//! These pin the reachability + module-descent behavior on real expanded
//! Deep programs (parsed Surf -> desugar -> expand), not hand-built AST, so a
//! change in the expanded module shape is caught here.

use super::*;
use chelis_deep::DeepTag;
use chelis_macros::ExpansionOptions;

/// Parse + desugar + expand Surf into the Deep program form `prune_to_entry`
/// consumes (one `module` node).
fn expand(source: &str) -> Vec<DeepExpr> {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parses");
    chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
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
        if let DeepExpr::Node(node, _) = expr
            && node.tag() == DeepTag::Module
        {
            for child in node.children_slice() {
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
        matches!(&expanded[0], DeepExpr::Node(node, _) if node.tag() == DeepTag::Module),
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
    let DeepExpr::Node(node, _) = &pruned[0] else {
        panic!("expected a module node");
    };
    assert_eq!(node.tag(), DeepTag::Module, "the module tag is kept");
    assert!(
        matches!(node.children_slice().first(), Some(DeepExpr::Atom(DeepAtom::Name(n), _)) if n == "demo.pricer"),
        "the module-name atom is retained as the first child"
    );
    let has_import = node
        .children_slice()
        .iter()
        .any(|e| e.tag() == Some(DeepTag::Import));
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
                DeepExpr::Node(node, _) => node
                    .into_parts()
                    .2
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

// ===========================================================================
// chelis#1125 PP7 slice E5c: ingress parity ([04-TOT-5]).
//
// Spec authority: spec/04-type-system.md §10 [04-TOT-5] -- a Deep program's
// verdict does not depend on which ingress produced it. Design:
// spec/design/checker_totality.md §"PP7. Stamped-ingress reader parity".
//
// The tests above all drive the Surf macro-expansion ingress. A `.dp` file
// reaches this pruner through `parse_and_stamp_file`. PP7 measured the
// consequence of a pruner that read only one ingress's spelling through
// tide's `/lower`: the identical two-def program lowered cleanly from Surf and
// failed the check stage from Deep with the UNRELATED def's `unbound
// variable`, because `prune_to_entry` recognized no `module` and no `def` in
// the stamped tree and returned the program unpruned. Both ingresses now
// produce the single node spelling; these tests keep the Deep ingress pinned.
// ===========================================================================

/// The Deep-ingress tree of exactly the program `expand` produces: print the
/// expanded Deep and parse it back through the stamping ingress. Deriving it
/// mechanically, rather than hand-authoring a second `.dp` text, is what makes
/// "the same program through two ingresses" true by construction.
fn stamp(exprs: &[DeepExpr]) -> Vec<DeepExpr> {
    let text = chelis_deep::printer::print_canonical(exprs);
    chelis_deep::parse_and_stamp_file(&text).unwrap_or_else(|e| {
        panic!("canonical Deep must re-parse through the stamping ingress: {e}")
    })
}

/// `(tag, first-child-name)` for a declaration node. Written against the node
/// directly and NOT through `deep_def_name`, so a reader defect under test
/// cannot make an assertion vacuous.
fn decl_head(expr: &DeepExpr) -> Option<(DeepTag, &str)> {
    match expr {
        DeepExpr::Node(node, _) => match node.children_slice().first() {
            Some(DeepExpr::Atom(DeepAtom::Name(name), _)) => Some((node.tag(), name.as_str())),
            _ => None,
        },
        _ => None,
    }
}

/// Children of a decoded node (tag and metadata dropped).
fn decl_children(expr: &DeepExpr) -> &[DeepExpr] {
    match expr {
        DeepExpr::Node(node, _) => node.children_slice(),
        _ => &[],
    }
}

/// The sorted `def` names in a program from either ingress, descending into a
/// `module` wrapper when there is one.
fn def_names_any_carrier(exprs: &[DeepExpr]) -> Vec<String> {
    let mut names = Vec::new();
    for expr in exprs {
        let items: &[DeepExpr] = match decl_head(expr) {
            Some((DeepTag::Module, _)) => decl_children(expr),
            _ => std::slice::from_ref(expr),
        };
        for item in items {
            if let Some((DeepTag::Def, name)) = decl_head(item) {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    names
}

/// PP7's tide `/lower` entry-pruning row. REGRESSION TEST (red before the
/// `prune.rs` repair, green after): the same program pruned to the same entry
/// must keep the same defs whichever ingress produced it. Before the repair
/// the Deep-ingress tree came back UNPRUNED, so `unrelated` survived and
/// dragged its `missing_sym` reference into the check stage that the Surf
/// ingress never reached.
#[test]
fn prune_to_entry_agrees_across_the_surf_and_deep_ingresses() {
    let expanded = expand(MULTI_DEF_MODULE);
    let stamped = stamp(&expanded);
    assert!(
        matches!(stamped.as_slice(), [DeepExpr::Node(node, _)] if node.tag() == DeepTag::Module),
        "the stamping ingress produces one `Expr::Node` module wrapper"
    );
    let from_surf = def_names_any_carrier(&prune_to_entry(expanded, "priced"));
    let from_deep = def_names_any_carrier(&prune_to_entry(stamped, "priced"));
    assert_eq!(
        from_surf,
        vec!["priced".to_string(), "scaled".to_string()],
        "the Surf ingress prunes to the entry's closure"
    );
    assert_eq!(
        from_deep, from_surf,
        "pruning to `priced` must keep the same defs on the Deep ingress as on \
         the Surf ingress (chelis#1125 [04-TOT-5]); the Deep ingress kept \
         `unrelated` and its unbound `missing_sym` reference"
    );
}

/// The Deep-ingress twin of `prune_preserves_module_head_and_import`.
/// DISPOSITION LOCK (green before and after, for opposite reasons: before the
/// repair the stamped program is returned untouched, so the wrapper trivially
/// survives). Its job is to constrain the repair: the descent must rebuild
/// the module node, keeping the name atom first and the `import` element
/// present. A repair that dropped the non-decl children while pruning fails
/// here.
#[test]
fn stamped_module_head_and_import_survive_pruning() {
    let stamped = stamp(&expand(MULTI_DEF_MODULE));
    let pruned = prune_to_entry(stamped, "priced");
    assert_eq!(pruned.len(), 1, "the module wrapper is preserved");
    let DeepExpr::Node(node, _) = &pruned[0] else {
        panic!("expected the stamped module node, got {:?}", pruned[0]);
    };
    assert_eq!(node.tag(), DeepTag::Module);
    assert!(
        matches!(node.children_slice().first(), Some(DeepExpr::Atom(DeepAtom::Name(n), _)) if n == "demo.pricer"),
        "the module-name atom stays the first child"
    );
    assert!(
        node.children_slice()
            .iter()
            .any(|e| decl_head(e).map(|(tag, _)| tag) == Some(DeepTag::Import)),
        "the import element (a non-decl child) is preserved through pruning"
    );
}

/// The over-rejection twin, DISPOSITION LOCK (green before and after): an
/// entry that is not a local `def` must leave the program UNCHANGED on the
/// stamped carrier too. Without it, "return the input unpruned" would satisfy
/// nothing above but "prune everything" would, and this pins the other edge.
#[test]
fn unknown_entry_leaves_the_stamped_program_unchanged() {
    let stamped = stamp(&expand(MULTI_DEF_MODULE));
    let before = def_names_any_carrier(&stamped);
    let pruned = prune_to_entry(stamped, "does_not_exist");
    assert_eq!(
        def_names_any_carrier(&pruned),
        before,
        "an unknown entry leaves every def in place on the stamped carrier"
    );
    assert_eq!(
        before,
        vec![
            "priced".to_string(),
            "scaled".to_string(),
            "unrelated".to_string()
        ],
        "the unpruned stamped program carries all three defs"
    );
}

/// The reachability twin on the stamped carrier, DISPOSITION LOCK (green
/// before and after the repair, for opposite reasons: unpruned before, pruned
/// correctly after). A def the entry transitively uses must survive.
#[test]
fn stamped_pruning_does_not_drop_a_def_in_the_entrys_closure() {
    let stamped = stamp(&expand(MULTI_DEF_MODULE));
    let pruned = prune_to_entry(stamped, "priced");
    assert!(
        def_names_any_carrier(&pruned).contains(&"scaled".to_string()),
        "a def reachable from the entry must survive pruning on the stamped carrier"
    );
}
