//! Issue #181 (review LOW #9): verify `pat-as` wrapping `pat-record`
//! against a generic ADT scrutinee correctly stamps both the outer
//! `as`-binding and the inner field bindings with their substituted
//! types in the annotated Deep.
//!
//! Surf syntax does not currently expose `pat-as`, so this test
//! constructs the pattern at the Deep level directly. Without this
//! coverage, a regression where `stamp_pattern_binding_types` stops
//! recursing into pat-as's inner sub-pattern (or stops stamping the
//! outer name) would only surface when actual users write `pat-as`
//! via custom Deep tooling.
//!
//! Expected behavior post-#181:
//! - The outer `(pat-as {} whole ...)` node's metadata gets a
//!   `:type` entry equal to `(t-adt {} FooState (t-tensor ...))`,
//!   matching the scrutinee's instantiation.
//! - The inner `(pat-var {} x)` node's metadata gets a `:type`
//!   entry equal to `(t-tensor {} ...)`, matching the substituted
//!   field type.

use chelis_deep::ast::Expr;
use chelis_deep::parser::parse_str;
use chelis_types::check_typed_program;

fn deep(src: &str) -> Vec<Expr> {
    parse_str(src).expect("deep parse")
}

/// Walk an Expr tree and return the first node whose tag matches.
/// Used to locate the inner pat-var/pat-as we want to inspect.
fn find_tagged<'a>(expr: &'a Expr, tag: &str) -> Option<&'a Expr> {
    if let Expr::List(list, _) = expr {
        if let Some(Expr::Atom(chelis_deep::ast::Atom::Symbol(t), _)) = list.elements.first()
            && t == tag
        {
            return Some(expr);
        }
        for child in &list.elements {
            if let Some(found) = find_tagged(child, tag) {
                return Some(found);
            }
        }
    }
    None
}

/// Extract the `type` entry from an Expr's metadata map. Returns None
/// if the expr isn't a list, has no metadata map, or has no `type` key.
fn type_metadata(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let meta = list.elements.get(1)?;
    let Expr::Map(map, _) = meta else {
        return None;
    };
    map.entries
        .iter()
        .find_map(|(k, v)| (k == "type").then_some(v))
}

/// Render an Expr to canonical Deep text for substring matching.
fn render(expr: &Expr) -> String {
    chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
}

#[test]
fn pat_as_wrapping_pat_record_stamps_both_outer_and_inner_types() {
    // type FooState[a] = | FooState { x: a, y: a }
    // def use_foo[n](state: FooState[tensor[n, f32]]) -> int32 = {
    //   match state with {
    //     | (pat-as {} whole (pat-record {} FooState (kv {} x (pat-var {} x))
    //                                                (kv {} y (pat-var {} y)))) => 0
    //   }
    // }
    //
    // Constructed at Deep level because Surf does not yet expose `pat-as`.
    let src = r"
(module {} Issue181PatAs
  (deftype {} FooState (a)
    (variant {} FooState (field {} x (t-var {} a)) (field {} y (t-var {} a))))
  (defsig {} use_foo
    (t-fn {}
      (t-adt {} FooState (t-tensor {} (d-var {} n) (t-prim {} f32)))
      (t-prim {} int32)))
  (def {} use_foo
    (fn {}
      (params {} (state {type: (t-adt {} FooState (t-tensor {} (d-var {} n) (t-prim {} f32)))}))
      (match {}
        (var {} state)
        (arm {}
          (pat-as {}
            whole
            (pat-record {} FooState
              (kv {} x (pat-var {} x))
              (kv {} y (pat-var {} y))))
          ()
          (lit {type: (t-prim {} int32)} 0))))))
";
    let exprs = deep(src);
    let checked = check_typed_program(&exprs).expect("type check should succeed");

    // The annotated tree should contain a `pat-as` node whose meta
    // carries `:type (t-adt FooState (t-tensor ...))`.
    let annotated = checked.exprs();
    let pat_as = annotated
        .iter()
        .find_map(|e| find_tagged(e, "pat-as"))
        .expect("annotated tree must contain the pat-as node");

    let pat_as_type = type_metadata(pat_as)
        .expect("stamper must attach :type to pat-as so linearity can resolve the outer binding");
    let pat_as_rendered = render(pat_as_type);
    assert!(
        pat_as_rendered.contains("FooState") && pat_as_rendered.contains("t-tensor"),
        "pat-as type metadata must reflect scrutinee's instantiation \
         FooState[tensor[..]]; got: {pat_as_rendered}"
    );

    // The inner pat-var `x` should carry a tensor type from the
    // substituted field, not an abstract `a`.
    let inner_pat_var = annotated
        .iter()
        .find_map(|e| find_tagged(e, "pat-var"))
        .expect("annotated tree must contain at least one pat-var");

    let pat_var_type = type_metadata(inner_pat_var)
        .expect("stamper must attach :type to pat-var so linearity can resolve the field binding");
    let pat_var_rendered = render(pat_var_type);
    assert!(
        pat_var_rendered.contains("t-tensor"),
        "destructured field pat-var must be stamped with the substituted tensor type, \
         not an abstract type variable; got: {pat_var_rendered}"
    );
    assert!(
        !pat_var_rendered.contains("(t-var {} a)"),
        "pat-var metadata must not retain the un-instantiated type parameter `a`; \
         got: {pat_var_rendered}"
    );
}
