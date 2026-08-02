//! Resolve + splice + round-trip tests over real Chelis modules.
//!
//! The fixtures below are the canonical Deep rendered by
//! `chelis deep <file.ch>` on real sources in sibling repos
//! (`economoist/src/growth.ch` and `economoist/properties/growth.ch`),
//! embedded verbatim and re-parsed with `parse_str`. They exercise
//! by-name function resolution, body-only splice, and the metadata-only
//! invariant against producer-annotated property defs that carry
//! `chelis_role`.

use chelis_deep::DeepTag;
use chelis_deep::parser::parse_str;
use chelis_deep::path::{
    DeepPath, PathSegment, ResolveError, function_body, resolve_function, splice_function_body,
};
use chelis_deep::printer::print_canonical;
use chelis_deep::{Atom, Expr};

/// Canonical Deep of `economoist/src/growth.ch`: a single function def
/// `gordon_pv` in module `economoist.growth`, preceded by an `export` and
/// a `defsig`.
const GROWTH: &str = r#"(module {}
  economoist.growth
  (export {} gordon_pv)
  (defsig {}
    gordon_pv
    (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
  (def {}
    gordon_pv
    (fn {}
      (params {}
        (d {type: (t-prim {} f32)})
        (r {type: (t-prim {} f32)})
        (g {type: (t-prim {} f32)}))
      (app {span: "surf:1852..1862"}
        (var {} div)
        (var {span: "surf:1852..1853"} d)
        (app {span: "surf:1857..1862"}
          (var {} sub)
          (var {span: "surf:1857..1858"} r)
          (var {span: "surf:1861..1862"} g))))))
"#;

/// Canonical Deep of the first def block of
/// `economoist/properties/growth.ch`: the property def `gordon_positive`
/// in module `economoist.properties.growth`. The def metadata carries
/// `chelis_role: "property"` plus `property_preconditions`,
/// `property_quantifiers`, and `property_source_kind`, all of which a
/// body-only splice must preserve verbatim.
const PROPERTY: &str = r#"(module {}
  economoist.properties.growth
  (defsig {}
    gordon_positive
    (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32) (t-prim {} bool)))
  (def {chelis_role: "property",
         property_preconditions: (tuple {}
                                   (app {span: "surf:835..842"}
                                     (var {} cmplt)
                                     (lit {span: "surf:839..842", type: (t-prim {} f32)} 0.0)
                                     (var {span: "surf:835..836"} d))
                                   (app {span: "surf:846..851"}
                                     (var {} cmplt)
                                     (var {span: "surf:850..851"} g)
                                     (var {span: "surf:846..847"} r))),
         property_quantifiers: (params {}
                                 (d {type: (t-prim {} f32)})
                                 (r {type: (t-prim {} f32)})
                                 (g {type: (t-prim {} f32)})),
         property_source_kind: "user"
       }
    gordon_positive
    (fn {}
      (params {}
        (d {type: (t-prim {} f32)})
        (r {type: (t-prim {} f32)})
        (g {type: (t-prim {} f32)}))
      (app {span: "surf:858..884"}
        (var {} and)
        (app {span: "surf:858..865"}
          (var {} cmplt)
          (lit {span: "surf:862..865", type: (t-prim {} f32)} 0.0)
          (var {span: "surf:858..859"} d))
        (app {span: "surf:872..884"}
          (var {} cmplt)
          (lit {span: "surf:881..884", type: (t-prim {} f32)} 0.0)
          (app {span: "surf:872..877"}
            (var {} sub)
            (var {span: "surf:872..873"} r)
            (var {span: "surf:876..877"} g)))))))
"#;

/// A value binding rendered as `(def {} <name> <expr>)` with no `(fn
/// ...)` child. Real Surf sources lower zero-arg bindings into zero-arg
/// functions, so this bare value-node shape is constructed directly from
/// Deep (a real Deep AST the parser accepts) to exercise the
/// not-a-function path that an external Deep producer can still emit.
const VALUE_BINDING: &str =
    "(module {} econ.consts (def {} pi (lit {type: (t-prim {} f32)} 3.14159)))\n";

fn parse(src: &str) -> Vec<Expr> {
    parse_str(src).expect("fixture parse failed")
}

/// The embedded fixtures are the canonical Deep produced by `chelis deep`
/// on real sources. Lock that they are already canonical: parsing and
/// canonical-printing each one reproduces it byte-for-byte. A drift here
/// means the embedded text is no longer the real renderer's output.
#[test]
fn embedded_fixtures_are_canonical() {
    for fixture in [GROWTH, PROPERTY, VALUE_BINDING] {
        assert_eq!(
            print_canonical(&parse(fixture)),
            fixture,
            "embedded fixture is not canonical Deep"
        );
    }
}

// ── Resolve + splice + round-trip ────────────────────────────────────

#[test]
fn resolve_splice_roundtrip_function_def() {
    let module = parse(GROWTH);

    // The new body is a real parsed Deep expression, not a hand-built
    // node: a constant float literal replacing the div/sub expression.
    let new_body = parse("(lit {type: (t-prim {} f32)} 0.0)")
        .into_iter()
        .next()
        .expect("new body parsed");

    let spliced = splice_function_body(&module, "economoist.growth.gordon_pv", new_body)
        .expect("splice succeeded");

    // Round-trip invariant: print, re-parse, re-print is identical.
    let printed = print_canonical(&spliced);
    let reparsed = parse_str(&printed).expect("spliced output re-parsed");
    let reprinted = print_canonical(&reparsed);
    assert_eq!(
        printed, reprinted,
        "spliced module did not round-trip through parse/print"
    );

    // The new body is present and the old div/sub body is gone.
    let resolved_def = resolved_def(&spliced, "economoist.growth.gordon_pv");
    let body = function_body(resolved_def).expect("resolved body present");
    assert_eq!(
        print_canonical(std::slice::from_ref(body)),
        "(lit {type: (t-prim {} f32)} 0.0)\n"
    );
}

#[test]
fn splice_preserves_signature_and_params_verbatim() {
    let module = parse(GROWTH);
    let original_printed = print_canonical(&module);

    let new_body = parse("(var {} d)").into_iter().next().unwrap();
    let spliced = splice_function_body(&module, "economoist.growth.gordon_pv", new_body).unwrap();
    let spliced_printed = print_canonical(&spliced);

    // The export, defsig, def metadata, name, fn metadata, and params are
    // all unchanged; only the body line differs.
    assert!(spliced_printed.contains("(export {} gordon_pv)"));
    assert!(
        spliced_printed
            .contains("(t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32) (t-prim {} f32))")
    );
    assert!(spliced_printed.contains("(d {type: (t-prim {} f32)})"));
    assert!(spliced_printed.contains("(r {type: (t-prim {} f32)})"));
    assert!(spliced_printed.contains("(g {type: (t-prim {} f32)})"));
    // The original div/sub body is gone.
    assert!(original_printed.contains("(var {} div)"));
    assert!(!spliced_printed.contains("(var {} div)"));
}

// ── Metadata-only invariant ──────────────────────────────────────────

/// After a body-only splice, the canonical Deep of the changed def is
/// identical to the original everywhere except the body slot
/// (`def.elements[3].elements[3]`). This is checked structurally: every
/// element of the def except the fn body is `==` to the original, and the
/// fn metadata and params are byte-identical under canonical printing.
#[test]
fn splice_changes_only_the_body_slot() {
    let module = parse(GROWTH);
    let original_def = resolved_def(&module, "economoist.growth.gordon_pv").clone();

    let new_body = parse("(var {} d)").into_iter().next().unwrap();
    let spliced = splice_function_body(&module, "economoist.growth.gordon_pv", new_body).unwrap();
    let spliced_def = resolved_def(&spliced, "economoist.growth.gordon_pv");

    let (orig_elems, spliced_elems) = (def_elements(&original_def), def_elements(spliced_def));
    assert_eq!(orig_elems.len(), spliced_elems.len());

    // def tag, def metadata (elements[1]), and bare name (elements[2]) are
    // structurally identical.
    assert_eq!(orig_elems[0], spliced_elems[0], "def tag changed");
    assert_eq!(orig_elems[1], spliced_elems[1], "def metadata changed");
    assert_eq!(orig_elems[2], spliced_elems[2], "def name changed");

    // The fn node differs only at the body slot.
    assert_fn_differs_only_in_body(&orig_elems[3], &spliced_elems[3]);
}

/// The same metadata-only invariant against a property def carrying
/// `chelis_role` and other producer metadata. The rich def metadata at
/// `elements[1]` must be preserved byte-identically by a body-only
/// splice.
#[test]
fn splice_preserves_property_chelis_role_metadata() {
    let module = parse(PROPERTY);
    let name = "economoist.properties.growth.gordon_positive";
    let original_def = resolved_def(&module, name).clone();
    assert!(
        print_canonical(std::slice::from_ref(&original_def)).contains("chelis_role: \"property\""),
        "fixture precondition: property def carries chelis_role"
    );

    let new_body = parse("(lit {type: (t-prim {} bool)} true)")
        .into_iter()
        .next()
        .unwrap();
    let spliced = splice_function_body(&module, name, new_body).unwrap();
    let spliced_def = resolved_def(&spliced, name);

    let (orig_elems, spliced_elems) = (def_elements(&original_def), def_elements(spliced_def));

    // def metadata (chelis_role, property_preconditions,
    // property_quantifiers, property_source_kind) survives verbatim.
    assert_eq!(
        orig_elems[1], spliced_elems[1],
        "property def metadata (chelis_role et al.) changed"
    );
    assert_eq!(orig_elems[2], spliced_elems[2], "property def name changed");
    assert_fn_differs_only_in_body(&orig_elems[3], &spliced_elems[3]);

    // The new body landed; the old `and`-of-comparisons body is gone.
    let body = function_body(spliced_def).expect("body present");
    assert_eq!(
        print_canonical(std::slice::from_ref(body)),
        "(lit {type: (t-prim {} bool)} true)\n"
    );
    assert!(!print_canonical(std::slice::from_ref(spliced_def)).contains("(var {} and)"));

    // The spliced property module still round-trips.
    let printed = print_canonical(&spliced);
    let reparsed = parse_str(&printed).expect("spliced property re-parsed");
    assert_eq!(printed, print_canonical(&reparsed));
}

// ── Module-prefix normalization ──────────────────────────────────────

#[test]
fn lowercase_and_pascalcase_prefixes_resolve_to_same_def() {
    let module = parse(GROWTH);

    let lower = resolve_function(&module, "economoist.growth.gordon_pv").unwrap();
    let pascal = resolve_function(&module, "Economoist.Growth.gordon_pv").unwrap();
    let bare = resolve_function(&module, "gordon_pv").unwrap();

    assert_eq!(lower.decl_index, pascal.decl_index);
    assert_eq!(lower.decl_index, bare.decl_index);
    assert_eq!(lower.qualified_name, "economoist.growth.gordon_pv");
    assert_eq!(pascal.qualified_name, lower.qualified_name);
    assert_eq!(bare.qualified_name, lower.qualified_name);
    assert_eq!(lower.body_path, DeepPath::body());
}

// ── DeepPath resolution ──────────────────────────────────────────────

#[test]
fn deeppath_body_resolves_function_body() {
    let module = parse(GROWTH);
    let def = resolved_def(&module, "economoist.growth.gordon_pv");

    let body = DeepPath::body().resolve(def).expect("body resolved");
    // The gordon_pv body is the div application.
    assert!(print_canonical(std::slice::from_ref(body)).starts_with("(app {span: \"surf:1852"));

    // Body then Child(0) addresses the `div` var.
    let div = DeepPath::body()
        .child(0)
        .resolve(def)
        .expect("body child 0 resolved");
    assert_eq!(print_canonical(std::slice::from_ref(div)), "(var {} div)\n");
}

#[test]
fn deeppath_resolve_mut_rewrites_in_place() {
    let mut module = parse(GROWTH);
    let decl_index = resolve_function(&module, "economoist.growth.gordon_pv")
        .unwrap()
        .decl_index;
    let def = def_node_mut(&mut module, decl_index);

    let body_slot = DeepPath::body().resolve_mut(def).expect("body slot");
    *body_slot = parse("(lit {type: (t-prim {} f32)} 1.0)")
        .into_iter()
        .next()
        .unwrap();

    let body = function_body(def_node(&module, decl_index)).unwrap();
    assert_eq!(
        print_canonical(std::slice::from_ref(body)),
        "(lit {type: (t-prim {} f32)} 1.0)\n"
    );
}

#[test]
fn deeppath_body_on_value_binding_errors() {
    let module = parse(VALUE_BINDING);
    let def = def_node(&module, 0);
    // `pi` is a value binding `(def {} pi (lit ...))` with no `(fn ...)`,
    // so `Body` has no body slot to step into.
    let err = DeepPath::body().resolve(def).unwrap_err();
    assert!(
        matches!(
            err,
            chelis_deep::PathError::BodyNeedsFnAddressing { depth: 0 }
        ),
        "expected BodyNeedsFnAddressing, got {err:?}"
    );
}

#[test]
fn child_segment_indexes_past_tag_and_meta() {
    // PathSegment::Child(0) addresses elements[2], skipping tag + meta.
    let exprs = parse("(app {} (var {} f) (var {} x))");
    let app = &exprs[0];
    let path = DeepPath::new().then(PathSegment::Child(0));
    let first = path.resolve(app).unwrap();
    assert_eq!(print_canonical(std::slice::from_ref(first)), "(var {} f)\n");
}

// ── Resolution failures (structured errors, not panics) ──────────────

#[test]
fn resolve_nonexistent_function_returns_structured_error() {
    let module = parse(GROWTH);
    let err = resolve_function(&module, "economoist.growth.no_such_fn").unwrap_err();
    match err {
        ResolveError::FunctionNotFound { searched, name } => {
            assert_eq!(searched, "economoist.growth.no_such_fn");
            assert_eq!(name, "no_such_fn");
        }
        other => panic!("expected FunctionNotFound, got {other:?}"),
    }
}

#[test]
fn resolve_wrong_module_prefix_returns_module_not_found() {
    let module = parse(GROWTH);
    let err = resolve_function(&module, "wrong.module.gordon_pv").unwrap_err();
    match err {
        ResolveError::ModuleNotFound {
            searched,
            requested,
            actual,
        } => {
            assert_eq!(searched, "wrong.module.gordon_pv");
            assert_eq!(requested, "wrong.module");
            assert_eq!(actual, "economoist.growth");
        }
        other => panic!("expected ModuleNotFound, got {other:?}"),
    }
}

#[test]
fn resolve_value_binding_as_function_returns_not_a_function() {
    let module = parse(VALUE_BINDING);
    let err = resolve_function(&module, "econ.consts.pi").unwrap_err();
    match err {
        ResolveError::NotAFunction { searched, name } => {
            assert_eq!(searched, "econ.consts.pi");
            assert_eq!(name, "pi");
        }
        other => panic!("expected NotAFunction, got {other:?}"),
    }
}

#[test]
fn splice_value_binding_as_function_returns_not_a_function() {
    let module = parse(VALUE_BINDING);
    let new_body = parse("(lit {type: (t-prim {} f32)} 0.0)")
        .into_iter()
        .next()
        .unwrap();
    let err = splice_function_body(&module, "econ.consts.pi", new_body).unwrap_err();
    assert!(
        matches!(err, ResolveError::NotAFunction { .. }),
        "expected NotAFunction, got {err:?}"
    );
}

#[test]
fn resolve_with_no_module_returns_no_module() {
    let exprs = parse("(def {} f (fn {} (params {}) (var {} f)))");
    let err = resolve_function(&exprs, "f").unwrap_err();
    assert!(
        matches!(err, ResolveError::NoModule { .. }),
        "expected NoModule, got {err:?}"
    );
}

// ── Local test helpers over the public AST ───────────────────────────

/// Declarations begin at `module.elements[3]` (after the `module` tag,
/// its metadata map, and the module name), so a 0-based declaration index
/// addresses `elements[3 + decl_index]`.
const MODULE_DECLS_START: usize = 3;

/// Resolve a function by qualified name and return its `(def ...)` node.
fn resolved_def<'a>(module: &'a [Expr], qualified_name: &str) -> &'a Expr {
    let decl_index = resolve_function(module, qualified_name)
        .expect("function resolves")
        .decl_index;
    def_node(module, decl_index)
}

fn def_node(module: &[Expr], decl_index: usize) -> &Expr {
    match &module[0] {
        Expr::List(module_list, _) => {
            let node = &module_list.elements[MODULE_DECLS_START + decl_index];
            assert_is_def(node);
            node
        }
        Expr::Node(node, _) if node.tag() == DeepTag::Module => {
            // Node children: [0]=name, [1..]=decls; decl_index is relative to decls
            let node = &node.children_slice()[1 + decl_index];
            assert_is_def(node);
            node
        }
        _ => panic!("expected module list or Node"),
    }
}

fn def_node_mut(module: &mut [Expr], decl_index: usize) -> &mut Expr {
    match &mut module[0] {
        Expr::List(module_list, _) => &mut module_list.elements[MODULE_DECLS_START + decl_index],
        Expr::Node(node, _) if node.tag() == DeepTag::Module => {
            &mut node.children_slice_mut()[1 + decl_index]
        }
        _ => panic!("expected module list or Node"),
    }
}

fn assert_is_def(node: &Expr) {
    match node {
        Expr::List(list, _) => {
            assert!(
                matches!(list.elements.first(), Some(Expr::Atom(Atom::Tag(t), _)) if *t == DeepTag::Def),
                "expected a def node"
            );
        }
        Expr::Node(n, _) => {
            assert_eq!(n.tag(), DeepTag::Def, "expected a def node");
        }
        _ => panic!("expected def list or Node"),
    }
}

fn def_elements(def: &Expr) -> Vec<Expr> {
    match def {
        Expr::List(list, _) => list.elements.clone(),
        Expr::Node(node, span) => {
            // Reconstruct the canonical List form: [tag, meta, children...]
            node.to_list(*span).elements
        }
        _ => panic!("expected def list or Node"),
    }
}

/// Assert that two `(fn {} (params {} ...) BODY)` nodes are identical
/// everywhere except the body slot.
fn assert_fn_differs_only_in_body(orig_fn: &Expr, spliced_fn: &Expr) {
    match (orig_fn, spliced_fn) {
        (Expr::List(orig, _), Expr::List(spliced, _)) => {
            assert_eq!(orig.elements.len(), spliced.elements.len());
            assert_eq!(orig.elements[0], spliced.elements[0], "fn tag changed");
            assert_eq!(orig.elements[1], spliced.elements[1], "fn metadata changed");
            assert_eq!(orig.elements[2], spliced.elements[2], "params changed");
            assert_ne!(
                orig.elements[3], spliced.elements[3],
                "body slot did not change"
            );
        }
        (Expr::Node(orig, _), Expr::Node(spliced, _)) => {
            assert_eq!(orig.tag(), spliced.tag(), "fn tag changed");
            assert_eq!(orig.meta(), spliced.meta(), "fn metadata changed");
            assert_eq!(orig.child_count(), spliced.child_count());
            // children[0] = params, children[1] = body
            assert_eq!(
                orig.children_slice()[0],
                spliced.children_slice()[0],
                "params changed"
            );
            assert_ne!(
                orig.children_slice()[1],
                spliced.children_slice()[1],
                "body slot did not change"
            );
        }
        _ => panic!("expected matching fn node types"),
    }
}
