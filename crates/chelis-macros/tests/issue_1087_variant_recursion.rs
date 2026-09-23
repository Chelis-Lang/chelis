//! chelis#1087: macro expansion recurses into `Expr::BareList` and
//! `Expr::UnknownForm`, and their symbols are visible to hygiene.
//!
//! Before the fix, five expansion walks and `collect_symbols` gave both
//! transitional variants a silent pass-through: a macro invocation nested
//! inside either was never expanded, names inside them were invisible to the
//! fresh-placeholder collision check, and hygiene renames never reached
//! references inside them. Each test here drives one of those holes through
//! the public `expand_program` boundary over parsed `.dp` text.
//!
//! Assertions use the ` name)` suffix idiom of the sibling raw-form boundary
//! test: the `source` provenance metadata legitimately spells the invocation
//! as `(name args...)`, so a bare `contains("name")` would false-positive on
//! provenance rather than on a leftover invocation.
//!
//! The existing raw-form boundary test
//! (`expansion.rs::parsed_deep_internal_macro_expands_from_raw_form_boundary`)
//! must stay green UNMODIFIED beside these.

use chelis_deep::ast::{Atom, Expr, Metadata, UnknownFormData};
use chelis_deep::{DeepTag, Span};
use chelis_macros::{ExpansionOptions, expand_program};

fn expand_deep(source: &str) -> String {
    let deep = chelis_deep::parser::parse_str(source).expect("deep fixture must stamp");
    let expanded = expand_program(
        &deep,
        &ExpansionOptions {
            max_iterations: 100,
            load_std_prelude: false,
        },
    )
    .expect("expansion must succeed");
    chelis_deep::printer::print_canonical(expanded.exprs())
}

const BUMP_MACRO: &str =
    "(defmacro {} bump (params {} x) (app {} (var {} add) (var {} x) (lit {} 1.0)))";

#[test]
fn macro_replacements_preserve_owners_and_reject_conflicting_data() {
    let source = "(defmacro {} keep (params {} x) (var {template: 1} x)) (def {} f (app {call_data: 3} (var {} keep) (lit {argument: 2} 7)))";
    let parsed = chelis_deep::parser::parse_str(source).unwrap();
    let options = ExpansionOptions {
        max_iterations: 100,
        load_std_prelude: false,
    };
    let expanded = expand_program(&parsed, &options).unwrap();
    let body = match &expanded.exprs()[0] {
        Expr::Node(node, _) => node.expr_child(1),
        other => panic!("definition: {other:?}"),
    };
    let metadata = match body {
        Expr::Node(node, _) => node.meta(),
        Expr::MetaExpr(meta, _) => &meta.metadata,
        other => panic!("replacement: {other:?}"),
    };
    for key in ["template", "argument", "call_data"] {
        assert!(
            metadata.extensions().get(key).is_some(),
            "{key}: {metadata:?}"
        );
    }
    let conflict = source.replace("argument: 2", "template: 2");
    let parsed = chelis_deep::parser::parse_str(&conflict).unwrap();
    let before = parsed.clone();
    assert!(
        expand_program(&parsed, &options)
            .unwrap_err()
            .to_string()
            .contains("template")
    );
    assert_eq!(parsed, before);
}

#[test]
fn macro_invocation_inside_bare_list_expands() {
    // `(future-form {} ...)` at the lenient top level stamps as a BareList
    // (unknown head at a bare/syntax position). The invocation nested inside
    // it must still expand.
    let text = expand_deep(&format!(
        "{BUMP_MACRO}\n(future-form {{}} (app {{}} (var {{}} bump) (lit {{}} 2.0)))"
    ));
    assert!(
        !text.contains(" bump)"),
        "invocation inside a BareList must expand: {text}"
    );
    assert!(text.contains(" add)"), "expansion output present: {text}");
}

#[test]
fn macro_invocation_expands_in_children_and_is_preserved_in_extension_data() {
    // A def body with an unknown head stamps as an UnknownForm at the
    // RuntimeExpr slot. Its live children expand; recorded producer data stays intact.
    let text = expand_deep(&format!(
        "{BUMP_MACRO}\n(def {{}} f (mystery {{note: (app {{}} (var {{}} bump) (lit {{}} 1.0))}} \
         (app {{}} (var {{}} bump) (lit {{}} 2.0))))"
    ));
    assert!(
        text.contains(" bump)"),
        "the extension invocation remains opaque: {text}"
    );
    assert!(text.contains(" add)"), "expansion output present: {text}");
}

#[test]
fn symbols_inside_bare_list_visible_to_hygiene() {
    // Fresh-placeholder collision negative: the macro body spells the first
    // placeholder candidate (`__chelis_macro_arg_1_0_0` for expansion 1,
    // parameter 0, attempt 0) inside a BareList. `collect_symbols` must see
    // it there, forcing `fresh_placeholder` onto attempt 1 — otherwise the
    // colliding var is captured by the macro argument and the user's
    // reference is silently rewritten out of the output.
    let text = expand_deep(
        "(defmacro {} keep (params {} v) \
           (holder {} (var {} __chelis_macro_arg_1_0_0) (var {} v)))\n\
         (def {} f (app {} (var {} keep) (lit {} 9.0)))",
    );
    assert!(
        text.contains("__chelis_macro_arg_1_0_0"),
        "the user's colliding name must survive uncaptured: {text}"
    );
    assert!(
        !text.contains("__chelis_macro_arg_1_0_1"),
        "the shifted fresh placeholder is fully consumed by substitution: {text}"
    );
    assert!(text.contains("9"), "the argument arrives: {text}");
}

// ── Programmatic stamped-node fixture helpers ────────────────────────
//
// These fixtures build the stamped nodes directly so each test isolates
// exactly the UnknownForm-recursion disposition, with the compiler-internal
// `defmacro` form carried as the `UnknownForm` the stamper gives it.

fn sp() -> Span {
    Span::new(0, 0)
}

fn atom_name(name: &str) -> Expr {
    Expr::Atom(Atom::Name(name.to_string()), sp())
}

fn tag_list(tag: DeepTag, children: Vec<Expr>) -> Expr {
    Expr::node(tag, Metadata::default(), children, sp())
}

fn defmacro_form(name: &str, params: Vec<Expr>, body: Expr) -> Expr {
    Expr::UnknownForm(Box::new(UnknownFormData {
        head: "defmacro".to_string(),
        meta: Metadata::default(),
        children: vec![atom_name(name), tag_list(DeepTag::Params, params), body],
        span: sp(),
    }))
}

fn var_ref(name: &str) -> Expr {
    tag_list(DeepTag::Var, vec![atom_name(name)])
}

#[test]
fn hygienize_renames_binders_inside_unknown_form() {
    // The macro body binds `tmp` and references it from inside an
    // UnknownForm. Hygiene renames the binder; the reference inside the
    // UnknownForm must follow, or the expansion dangles.
    let body = tag_list(
        DeepTag::Let,
        vec![
            tag_list(DeepTag::Bind, vec![atom_name("tmp"), var_ref("v")]),
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "mystery".to_string(),
                meta: Metadata::default(),
                children: vec![var_ref("tmp")],
                span: sp(),
            })),
        ],
    );
    let defmacro = defmacro_form("wrapt", vec![atom_name("v")], body);
    let call = tag_list(
        DeepTag::Def,
        vec![
            atom_name("f"),
            tag_list(
                DeepTag::App,
                vec![
                    var_ref("wrapt"),
                    tag_list(DeepTag::Lit, vec![Expr::Atom(Atom::Int(3), sp())]),
                ],
            ),
        ],
    );
    let expanded = expand_program(
        &[defmacro, call],
        &ExpansionOptions {
            max_iterations: 100,
            load_std_prelude: false,
        },
    )
    .expect("expansion must succeed");
    let text = chelis_deep::printer::print_canonical(expanded.exprs());
    assert!(
        !text.contains(" tmp)"),
        "the reference inside the UnknownForm must follow the hygiene rename: {text}"
    );
    assert!(
        text.contains("tmp_macro_"),
        "the hygienic rename is visible: {text}"
    );
}

#[test]
fn substitute_reaches_params_inside_bare_list() {
    // A macro parameter referenced from inside a BareList body must receive
    // its argument.
    let text = expand_deep(
        "(defmacro {} inject (params {} v) (carrier {} (var {} v)))\n\
         (def {} f (app {} (var {} inject) (lit {} 7.0)))",
    );
    assert!(
        !text.contains(" v)"),
        "the parameter reference inside the BareList must substitute: {text}"
    );
    assert!(text.contains("7"), "the argument arrives: {text}");
}

// ── Metadata-value recursion (PR #1319 review) ───────────────────────
//
// Registered expression fields participate in macros. Producer extensions
// and historical source are opaque data under [03-META-2/3].

#[test]
fn macro_invocation_inside_extension_data_is_preserved() {
    // The review probe, verbatim: a map nested inside an UnknownForm
    // metadata value hid the invocation from the one-level metadata walk.
    let text = expand_deep(&format!(
        "{BUMP_MACRO}\n(def {{}} f (mystery {{outer: {{inner: (app {{}} (var {{}} bump) \
         (lit {{}} 2.0))}}}} (lit {{}} 0.0)))"
    ));
    assert!(
        text.contains(" bump)"),
        "an invocation in producer data remains data: {text}"
    );
    assert!(
        !text.contains(" add)"),
        "producer data is not expanded: {text}"
    );
}

#[test]
fn macro_invocation_inside_vocabulary_node_metadata_expands() {
    // Same class at a structural position: a vocabulary node's metadata map
    // is walked with the node, so an invocation in one of its values must
    // expand too.
    let text = expand_deep(&format!(
        "{BUMP_MACRO}\n(def {{property_seed: (app {{}} (var {{}} bump) (lit {{}} 1.0))}} f (lit {{}} 0.0))"
    ));
    assert!(
        !text.contains(" bump)"),
        "an invocation in a vocabulary node's metadata value must expand: {text}"
    );
    assert!(text.contains(" add)"), "expansion output present: {text}");
}

#[test]
fn substitution_preserves_extension_data_and_rewrites_live_children() {
    // A macro parameter referenced from a map nested inside a metadata
    // value must receive its argument.
    let text = expand_deep(
        "(defmacro {} inject_meta (params {} v) \
           (carrier {outer: {inner: (var {} v)}} (var {} v)))\n\
         (def {} f (app {} (var {} inject_meta) (lit {} 7.0)))",
    );
    assert!(
        text.contains(" v)"),
        "producer data retains its original reference: {text}"
    );
    assert!(text.contains("7"), "the argument arrives: {text}");
}

#[test]
fn hygiene_preserves_extension_references_and_renames_live_children() {
    // The macro body binds `tmp`; a reference from a map nested inside an
    // UnknownForm metadata value must follow the hygiene rename.
    let body = tag_list(
        DeepTag::Let,
        vec![
            tag_list(DeepTag::Bind, vec![atom_name("tmp"), var_ref("v")]),
            Expr::UnknownForm(Box::new(UnknownFormData {
                head: "mystery".to_string(),
                meta: {
                    let mut outer = Metadata::default();
                    outer
                        .extensions_mut()
                        .insert(
                            "outer".into(),
                            chelis_deep::ExtensionData::parse("{inner: (var {} tmp)}").unwrap(),
                        )
                        .unwrap();
                    outer
                },
                children: vec![var_ref("tmp")],
                span: sp(),
            })),
        ],
    );
    let defmacro = defmacro_form("wrapm", vec![atom_name("v")], body);
    let call = tag_list(
        DeepTag::Def,
        vec![
            atom_name("f"),
            tag_list(
                DeepTag::App,
                vec![
                    var_ref("wrapm"),
                    tag_list(DeepTag::Lit, vec![Expr::Atom(Atom::Int(3), sp())]),
                ],
            ),
        ],
    );
    let expanded = expand_program(
        &[defmacro, call],
        &ExpansionOptions {
            max_iterations: 100,
            load_std_prelude: false,
        },
    )
    .expect("expansion must succeed");
    let text = chelis_deep::printer::print_canonical(expanded.exprs());
    assert!(
        text.contains(" tmp)"),
        "the producer reference must remain unchanged: {text}"
    );
    assert!(
        text.contains("tmp_macro_"),
        "the hygienic rename is visible: {text}"
    );
}

#[test]
fn metadata_without_macro_syntax_round_trips_byte_identically() {
    // Control: the metadata walk must not rewrite metadata containing no
    // macro syntax.
    let deep =
        chelis_deep::parser::parse_str("(def {note: {inner: (var {} plain)}} f (lit {} 0.0))")
            .expect("control fixture must stamp");
    let before = chelis_deep::printer::print_canonical(&deep);
    let expanded = expand_program(
        &deep,
        &ExpansionOptions {
            max_iterations: 100,
            load_std_prelude: false,
        },
    )
    .expect("expansion must succeed");
    let after = chelis_deep::printer::print_canonical(expanded.exprs());
    assert_eq!(
        before, after,
        "metadata with no macro syntax must round-trip byte-identically"
    );
}

#[test]
fn provenance_source_metadata_is_not_reexpanded() {
    // Negative: macro-shaped text under the `source` provenance key is a
    // verbatim historical record and must survive expansion unchanged.
    let text = expand_deep(&format!(
        "{BUMP_MACRO}\n(def {{source: (app {{}} (var {{}} bump) (lit {{}} 1.0))}} f (lit {{}} 0.0))"
    ));
    assert!(
        text.contains(" bump)"),
        "the source record must keep its original spelling: {text}"
    );
}
