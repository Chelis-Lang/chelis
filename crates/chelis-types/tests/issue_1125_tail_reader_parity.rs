//! chelis#1125 PP7/E5e: structural function-tail readers preserve ingress parity.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).expect("fixture must stamp")
}

fn messages(errors: Vec<CheckError>) -> Vec<String> {
    errors
        .into_iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect()
}

fn ingress_messages(source: &str) -> Vec<String> {
    let program = stamped(source);
    let typed = check_typed_program(&program)
        .err()
        .map(|report| messages(report.errors))
        .unwrap_or_default();
    let ir = check_ir_program(&program)
        .err()
        .map(|report| messages(report.errors))
        .unwrap_or_default();
    assert_eq!(typed, ir, "checker ingresses must report the same defects");
    typed
}

const CHOICE: &str = "
    (deftype {} Choice () (variant {} Left) (variant {} Right))";

const T3: &str = "(t-tensor {} (d-lit {} 3) (t-prim {} f32))";

/// An ADT whose payload is itself a borrow. Without that, a `pat-ctor`
/// binder under a borrowed scrutinee is typed *owned*, the def-body unify
/// succeeds outright, and the Shape A relaxation is never consulted -- so a
/// projection fixture built on an owned payload cannot reach the code it
/// claims to pin.
const HOLDER: &str = "
    (deftype {} Holder () (variant {} Hold (t-ref {} (t-tensor {} (d-lit {} 3)
      (t-prim {} f32)))))";

/// Same shape with a named field, for the `pat-record` form.
const FIELD_HOLDER: &str = "
    (deftype {} Boxed () (variant {} Boxed (field {} r (t-ref {} (t-tensor {}
      (d-lit {} 3) (t-prim {} f32))))))";

/// The guarded `Left` arm is followed by an unguarded one because a guarded
/// arm covers nothing (spec/04-type-system.md section 2.4, [04-PAT-2]).
#[test]
fn guarded_match_tail_accepts_the_same_owned_return_on_both_ingresses() {
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose (n)
           (t-fn {{}} (t-adt {{}} Choice)
             (t-ref {{}} (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32)))
             (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32))))
         (def {{}} choose
           (fn {{}} (params {{}} choice value)
             (match {{}} (var {{}} choice)
               (arm {{}} (pat-ctor {{}} Left)
                 (app {{}} (var {{}} eq)
                   (lit {{type: (t-prim {{}} i32)}} 1)
                   (lit {{type: (t-prim {{}} i32)}} 1))
                 (var {{}} value))
               (arm {{}} (pat-ctor {{}} Left) () (var {{}} value))
               (arm {{}} (pat-ctor {{}} Right) () (var {{}} value)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn guarded_match_tail_rejects_a_different_return_on_both_ingresses() {
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose (n)
           (t-fn {{}} (t-adt {{}} Choice)
             (t-ref {{}} (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32)))
             (t-ref {{}} (t-tensor {{}} (d-var {{}} n) (t-prim {{}} i32)))
             (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32))))
         (def {{}} choose
           (fn {{}} (params {{}} choice expected wrong)
             (match {{}} (var {{}} choice)
               (arm {{}} (pat-ctor {{}} Left)
                 (app {{}} (var {{}} eq)
                   (lit {{type: (t-prim {{}} i32)}} 1)
                   (lit {{type: (t-prim {{}} i32)}} 1))
                 (var {{}} expected))
               (arm {{}} (pat-ctor {{}} Left) () (var {{}} expected))
               (arm {{}} (pat-ctor {{}} Right) () (var {{}} wrong)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

/// The guarded `Left` arm is followed by an unguarded one because a guarded
/// arm covers nothing (spec/04-type-system.md section 2.4, [04-PAT-2]).
#[test]
fn same_local_alias_spelling_does_not_merge_distinct_borrowed_parameters() {
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose
           (t-fn {{}}
             (t-adt {{}} Choice)
             (t-ref {{}} (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} f32)))
             (t-ref {{}} (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} f32)))
             (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} f32))))
         (def {{}} choose
           (fn {{}} (params {{}} choice left right)
             (match {{}} (var {{}} choice)
               (arm {{}} (pat-ctor {{}} Left)
                 (lit {{type: (t-prim {{}} bool)}} true)
                 (let {{}} (bind {{}} result (var {{}} left)) (var {{}} result)))
               (arm {{}} (pat-ctor {{}} Left) ()
                 (let {{}} (bind {{}} result (var {{}} left)) (var {{}} result)))
               (arm {{}} (pat-ctor {{}} Right) ()
                 (let {{}} (bind {{}} result (var {{}} right)) (var {{}} result))))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn locally_constructed_borrow_is_not_relaxed_into_an_owned_return() {
    let diagnostics = ingress_messages(
        "(defsig {} local
           (t-fn {}
             (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))
             (t-tensor {} (d-lit {} 3) (t-prim {} f32))))
         (def {} local
           (fn {} (params {} input)
             (let {}
               (bind {}
                 owned (copy {} (var {} input))
                 borrowed (borrow {} (var {} owned)))
               (var {} borrowed))))",
    );
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

// ── Arm binders that denote the whole scrutinee ──────────────────────────
//
// The seven-row `pat-*` table in `spec/03-deep-syntax.md` splits the pattern
// forms in two. `pat-var` ("Bind to name") and `pat-as` ("Bind name, then
// match") bind the whole scrutinee, so a binder in either position denotes
// the value the scrutinee denotes and may inherit its provenance. The
// children of `pat-ctor`, `pat-tuple` and `pat-record` name projections out
// of that value, and a projection is a different value: letting one inherit
// would relax a def into returning a borrow the caller never supplied.
//
// Whole-scrutinee-ness is a property of where a sub-pattern sits, not of
// which tag it carries, so the `pat-var` inside a `pat-ctor` is covered by
// the negative half below rather than the positive half here.

#[test]
fn arm_binder_traces_the_borrowed_scrutinee_to_its_parameter() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} x)
           (match {{}} (var {{}} x) (arm {{}} (pat-var {{}} v) () (var {{}} v)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn pat_as_outer_name_traces_when_the_inner_pattern_destructures() {
    let diagnostics = ingress_messages(&format!(
        "{HOLDER}
         (defsig {{}} keep (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder))
           (t-adt {{}} Holder)))
         (def {{}} keep (fn {{}} (params {{}} h)
           (match {{}} (var {{}} h)
             (arm {{}} (pat-as {{}} whole (pat-ctor {{}} Hold (pat-var {{}} r))) ()
               (var {{}} whole)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn borrowed_record_and_tuple_patterns_preserve_the_whole_scrutinee_binder() {
    let record = ingress_messages(&format!(
        "{FIELD_HOLDER}
         (defsig {{}} keep_record (t-fn {{}} (t-ref {{}} (t-adt {{}} Boxed))
           (t-adt {{}} Boxed)))
         (def {{}} keep_record (fn {{}} (params {{}} b)
           (match {{}} (var {{}} b)
             (arm {{}} (pat-as {{}} whole
               (pat-record {{}} Boxed (kv {{}} r (pat-var {{}} inner)))) ()
               (var {{}} whole)))))"
    ));
    assert!(record.is_empty(), "borrowed record pattern: {record:?}");

    let tuple = ingress_messages(&format!(
        "(defsig {{}} keep_tuple
           (t-fn {{}} (t-ref {{}} (t-tuple {{}} {T3} (t-prim {{}} i32)))
             (t-tuple {{}} {T3} (t-prim {{}} i32))))
         (def {{}} keep_tuple (fn {{}} (params {{}} pair)
           (match {{}} (var {{}} pair)
             (arm {{}} (pat-as {{}} whole
               (pat-tuple {{}} (pat-var {{}} a) (pat-var {{}} n))) ()
               (var {{}} whole)))))"
    ));
    assert!(tuple.is_empty(), "borrowed tuple pattern: {tuple:?}");
}

#[test]
fn borrowed_scrutinee_still_rejects_wrong_pattern_shape() {
    let foreign_ctor = ingress_messages(&format!(
        "{HOLDER}
         (deftype {{}} Other () (variant {{}} Wrong))
         (defsig {{}} f (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder))
           (t-prim {{}} i32)))
         (def {{}} f (fn {{}} (params {{}} h)
           (match {{}} (var {{}} h)
             (arm {{}} (pat-ctor {{}} Wrong) ()
               (lit {{type: (t-prim {{}} i32)}} 1))
             (arm {{}} (pat-wild {{}}) ()
               (lit {{type: (t-prim {{}} i32)}} 0)))))"
    ));
    assert!(
        foreign_ctor
            .iter()
            .any(|message| message.contains("Other vs Holder")),
        "foreign constructor on borrowed Holder: {foreign_ctor:?}"
    );

    let foreign_record = ingress_messages(&format!(
        "{FIELD_HOLDER}
         (deftype {{}} Other () (variant {{}} Wrong (field {{}} r {T3})))
         (defsig {{}} f (t-fn {{}} (t-ref {{}} (t-adt {{}} Boxed))
           (t-prim {{}} i32)))
         (def {{}} f (fn {{}} (params {{}} b)
           (match {{}} (var {{}} b)
             (arm {{}} (pat-record {{}} Wrong (kv {{}} r (pat-var {{}} v))) ()
               (lit {{type: (t-prim {{}} i32)}} 1))
             (arm {{}} (pat-wild {{}}) ()
               (lit {{type: (t-prim {{}} i32)}} 0)))))"
    ));
    assert!(
        foreign_record
            .iter()
            .any(|message| message.contains("Other vs Boxed")),
        "foreign record on borrowed Boxed: {foreign_record:?}"
    );

    let non_tuple = ingress_messages(&format!(
        "{HOLDER}
         (defsig {{}} f (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder))
           (t-prim {{}} i32)))
         (def {{}} f (fn {{}} (params {{}} h)
           (match {{}} (var {{}} h)
             (arm {{}} (pat-tuple {{}} (pat-var {{}} a) (pat-var {{}} b)) ()
               (lit {{type: (t-prim {{}} i32)}} 1))
             (arm {{}} (pat-wild {{}}) ()
               (lit {{type: (t-prim {{}} i32)}} 0)))))"
    ));
    assert!(
        non_tuple
            .iter()
            .any(|message| message.contains("Holder vs (")),
        "tuple pattern on borrowed Holder: {non_tuple:?}"
    );

    let wrong_tuple = ingress_messages(&format!(
        "(defsig {{}} f
           (t-fn {{}} (t-ref {{}} (t-tuple {{}} {T3} (t-prim {{}} i32)))
             (t-prim {{}} i32)))
         (def {{}} f (fn {{}} (params {{}} pair)
           (match {{}} (var {{}} pair)
             (arm {{}} (pat-tuple {{}} (pat-var {{}} a)) ()
               (lit {{type: (t-prim {{}} i32)}} 1))
             (arm {{}} (pat-wild {{}}) ()
               (lit {{type: (t-prim {{}} i32)}} 0)))))"
    ));
    assert!(
        wrong_tuple
            .iter()
            .any(|message| message.contains("ArityMismatch")),
        "wrong tuple arity under borrow: {wrong_tuple:?}"
    );
}

#[test]
fn pat_as_inner_bare_binder_denotes_the_whole_scrutinee_too() {
    // `(pat-as {} whole (pat-var {} v))` matches the inner pattern against
    // the same value, so `v` denotes the scrutinee exactly as `whole` does.
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} x)
           (match {{}} (var {{}} x)
             (arm {{}} (pat-as {{}} whole (pat-var {{}} v)) () (var {{}} v)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn one_binder_spelling_in_two_arms_agrees_on_one_parameter() {
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose (t-fn {{}} (t-adt {{}} Choice) (t-ref {{}} {T3})
           (t-ref {{}} {T3}) {T3}))
         (def {{}} choose (fn {{}} (params {{}} c p q)
           (match {{}} (var {{}} c)
             (arm {{}} (pat-ctor {{}} Left) ()
               (match {{}} (var {{}} p) (arm {{}} (pat-var {{}} v) () (var {{}} v))))
             (arm {{}} (pat-ctor {{}} Right) ()
               (match {{}} (var {{}} p) (arm {{}} (pat-var {{}} v) () (var {{}} v)))))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn arm_binder_provenance_survives_a_nested_match() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} x)
           (match {{}} (var {{}} x)
             (arm {{}} (pat-var {{}} v) ()
               (match {{}} (var {{}} v)
                 (arm {{}} (pat-var {{}} w) () (var {{}} w)))))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn arm_binder_shadowing_another_parameter_takes_the_scrutinee_root() {
    // `other` is both a parameter and this arm's binder. Inside the arm it
    // denotes `x`, so the other branch returning `x` agrees with it.
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-prim {{}} bool) (t-ref {{}} {T3})
           (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} c x other)
           (if {{}} (var {{}} c)
             (match {{}} (var {{}} x) (arm {{}} (pat-var {{}} other) () (var {{}} other)))
             (var {{}} x))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn a_let_aliased_scrutinee_still_feeds_its_arm_binders() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} x)
           (let {{}} (bind {{}} y (var {{}} x))
             (match {{}} (var {{}} y)
               (arm {{}} (pat-var {{}} v) () (var {{}} v))))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// The repair also broadens the accepted set, and this is the shape that
/// moved: branch agreement through two different alias spellings of one
/// parameter. Name equality rejected it; provenance accepts it.
#[test]
fn two_alias_spellings_of_one_parameter_agree_across_branches() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} f (t-fn {{}} (t-prim {{}} bool) (t-ref {{}} {T3}) {T3}))
         (def {{}} f (fn {{}} (params {{}} c x)
           (if {{}} (var {{}} c)
             (let {{}} (bind {{}} y (var {{}} x)) (var {{}} y))
             (let {{}} (bind {{}} z (var {{}} x)) (var {{}} z)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

// ── Projections must not inherit: the unsound direction ──────────────────

#[test]
fn ctor_payload_binder_does_not_inherit_the_scrutinee_provenance() {
    let diagnostics = ingress_messages(&format!(
        "{HOLDER}
         (defsig {{}} peek (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder)) {T3}))
         (def {{}} peek (fn {{}} (params {{}} h)
           (match {{}} (var {{}} h)
             (arm {{}} (pat-ctor {{}} Hold (pat-var {{}} r)) () (var {{}} r)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn pat_as_inner_ctor_binder_does_not_inherit_only_the_outer_name() {
    // One pattern carrying both kinds of binder at once: `whole` denotes the
    // scrutinee and may inherit, `r` names the projection and must not.
    let diagnostics = ingress_messages(&format!(
        "{HOLDER}
         (defsig {{}} peek (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder)) {T3}))
         (def {{}} peek (fn {{}} (params {{}} h)
           (match {{}} (var {{}} h)
             (arm {{}} (pat-as {{}} whole (pat-ctor {{}} Hold (pat-var {{}} r))) ()
               (var {{}} r)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

/// The shared-spelling case, and the control directly above it is the same
/// program with the two binders spelled apart. One spelling is both the
/// `pat-as` outer name, which denotes the whole scrutinee, and the payload
/// binder, which denotes a projection. Being a whole-scrutinee binder
/// somewhere in the pattern must not launder the projection: the name has to
/// refuse.
#[test]
fn a_spelling_that_is_both_whole_and_projection_refuses() {
    let diagnostics = ingress_messages(&format!(
        "{HOLDER}
         (defsig {{}} peek (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder)) {T3}))
         (def {{}} peek (fn {{}} (params {{}} h)
           (match {{}} (var {{}} h)
             (arm {{}} (pat-as {{}} h (pat-ctor {{}} Hold (pat-var {{}} h))) () (var {{}} h)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

/// The same defect without the parameter in the collision, so the fixture
/// above is not read as being about shadowing a parameter. Only the two
/// binders share a spelling here; the parameter is spelled apart from both.
#[test]
fn two_binders_sharing_a_spelling_refuse_without_shadowing_the_parameter() {
    let diagnostics = ingress_messages(&format!(
        "{HOLDER}
         (defsig {{}} peek (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder)) {T3}))
         (def {{}} peek (fn {{}} (params {{}} boxed)
           (match {{}} (var {{}} boxed)
             (arm {{}} (pat-as {{}} w (pat-ctor {{}} Hold (pat-var {{}} w))) () (var {{}} w)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn record_field_binder_does_not_inherit_the_scrutinee_provenance() {
    let diagnostics = ingress_messages(&format!(
        "{FIELD_HOLDER}
         (defsig {{}} peek (t-fn {{}} (t-ref {{}} (t-adt {{}} Boxed)) {T3}))
         (def {{}} peek (fn {{}} (params {{}} b)
           (match {{}} (var {{}} b)
             (arm {{}} (pat-record {{}} Boxed (kv {{}} r (pat-var {{}} inner))) ()
               (var {{}} inner)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn one_binder_spelling_in_two_arms_still_rejects_two_parameters() {
    // Both arms bind `v`. Name equality would call that agreement; the two
    // binders root in different parameters and must not agree.
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose (t-fn {{}} (t-adt {{}} Choice) (t-ref {{}} {T3})
           (t-ref {{}} {T3}) {T3}))
         (def {{}} choose (fn {{}} (params {{}} c p q)
           (match {{}} (var {{}} c)
             (arm {{}} (pat-ctor {{}} Left) ()
               (match {{}} (var {{}} p) (arm {{}} (pat-var {{}} v) () (var {{}} v))))
             (arm {{}} (pat-ctor {{}} Right) ()
               (match {{}} (var {{}} q) (arm {{}} (pat-var {{}} v) () (var {{}} v)))))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn arm_binder_shadowing_another_parameter_does_not_keep_its_own_root() {
    // The mirror of the accepting case above. Inside the arm `other` denotes
    // `x`; the other branch returns the parameter `other`. Those are two
    // parameters, so the branches disagree. Carrying the binder's own prior
    // entry instead of the scrutinee's root would call this agreement.
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-prim {{}} bool) (t-ref {{}} {T3})
           (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} c x other)
           (if {{}} (var {{}} c)
             (match {{}} (var {{}} x) (arm {{}} (pat-var {{}} other) () (var {{}} other)))
             (var {{}} other))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn arm_binder_over_a_locally_constructed_borrow_is_not_relaxed() {
    // Restoring arm tracing must not leak a borrow of a function-local owner
    // into the relaxation: the scrutinee resolves to no parameter, so the
    // binder inherits nothing.
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} local (t-fn {{}} (t-ref {{}} {T3}) {T3}))
         (def {{}} local (fn {{}} (params {{}} input)
           (let {{}} (bind {{}} owned (copy {{}} (var {{}} input))
                      borrowed (borrow {{}} (var {{}} owned)))
             (match {{}} (var {{}} borrowed)
               (arm {{}} (pat-var {{}} v) () (var {{}} v))))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

/// The scrutinee's own provenance is resolved with the same descent used
/// everywhere else, so a scrutinee that is not a bare name still feeds its
/// arm binders when every path through it lands on one parameter.
#[test]
fn a_branching_scrutinee_that_resolves_to_one_parameter_feeds_its_binders() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-prim {{}} bool) (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} c x)
           (match {{}} (if {{}} (var {{}} c) (var {{}} x) (var {{}} x))
             (arm {{}} (pat-var {{}} v) () (var {{}} v)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// A `let` right-hand side is read with the same descent as a tail, so a
/// binding whose right-hand side branches onto one parameter keeps that
/// parameter's identity. `chelis-cli`'s `sr_let_with_if_rhs_lowers_cleanly`
/// covers the same shape end to end, but `.config/ci-test-targets.toml`
/// routes this source path to `chelis-types` alone, so the required lane
/// only sees it here.
#[test]
fn a_let_bound_branch_that_resolves_to_one_parameter_still_traces() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} f (t-fn {{}} (t-prim {{}} bool) (t-ref {{}} {T3}) {T3}))
         (def {{}} f (fn {{}} (params {{}} c x)
           (let {{}} (bind {{}} y (if {{}} (var {{}} c) (var {{}} x) (var {{}} x)))
             (var {{}} y))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// The mirror: a `let` right-hand side that branches onto two different
/// parameters resolves to neither, so the binding acquires no provenance.
#[test]
fn a_let_bound_branch_across_two_parameters_acquires_no_provenance() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} f (t-fn {{}} (t-prim {{}} bool) (t-ref {{}} {T3})
           (t-ref {{}} {T3}) {T3}))
         (def {{}} f (fn {{}} (params {{}} c p q)
           (let {{}} (bind {{}} y (if {{}} (var {{}} c) (var {{}} p) (var {{}} q)))
             (var {{}} y))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

/// Whole-scrutinee-ness is path-recursive. Every edge on the path from the
/// pattern root to `c` preserves the whole value, so all three names denote
/// the scrutinee and any of them may trace.
#[test]
fn a_nested_pat_as_chain_traces_every_name() {
    let diagnostics = ingress_messages(&format!(
        "(defsig {{}} pick (t-fn {{}} (t-ref {{}} {T3}) {T3}))
         (def {{}} pick (fn {{}} (params {{}} x)
           (match {{}} (var {{}} x)
             (arm {{}} (pat-as {{}} a (pat-as {{}} b (pat-var {{}} c))) () (var {{}} c)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// The converse, and the shape that fails unsound if the predicate is keyed on
/// the tag rather than the path: `b` and `c` are projections wearing a
/// `pat-as`. The `pat-ctor` edge breaks the path, so only `a` traces.
#[test]
fn a_pat_as_inside_a_ctor_does_not_launder_the_projection() {
    let diagnostics = ingress_messages(&format!(
        "{HOLDER}
         (defsig {{}} peek (t-fn {{}} (t-ref {{}} (t-adt {{}} Holder)) {T3}))
         (def {{}} peek (fn {{}} (params {{}} h)
           (match {{}} (var {{}} h)
             (arm {{}} (pat-as {{}} a (pat-ctor {{}} Hold (pat-as {{}} b (pat-var {{}} c)))) ()
               (var {{}} c)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}
