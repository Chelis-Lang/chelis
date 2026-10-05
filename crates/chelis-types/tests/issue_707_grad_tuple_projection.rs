//! Issue Chelis-Lang/chelis#707 — nominal ADT type checking was lost for
//! any value derived from a tuple **projection** (`t.0`), the reported
//! symptom being a multi-arg `grad(...)` tuple whose transposed downstream
//! use type-checked (score 1) and then died at runtime with a
//! `non-exhaustive runtime match`.
//!
//! Root cause (pinned by a wrong-ascription probe ladder): `infer_tuple_get`
//! read the projection index ONLY from a bare `deep::Atom::Int`, but the
//! Surf `.N` desugar emits the index as a `lit` node `(lit {i32} N)`
//! (`desugar.rs`, `Expr::TupleGet`). The index never matched, so every
//! Surf-level `.N` projection fell through to `Type::Error`. `Type::Error`
//! unifies with anything, so the projected element (and every value derived
//! from it) silently lost its nominal type at every downstream boundary —
//! sibling arguments in the same call skipped checking too, and the
//! `Type::Error` return crossed function-return boundaries. This is NOT
//! grad-specific: a bare tuple literal `(a, b).0` lost its type the same
//! way (see `bare_tuple_literal_projection_carries_element_type`).
//!
//! Fix: `infer_tuple_get` reads the index from both the bare-atom and the
//! `lit`-node shapes, so a concrete tuple projection now carries its real
//! element type. A projection whose target is still an unresolved type var
//! (e.g. an unannotated `fold` accumulator) defers to a fresh element type
//! rather than erroring — this type system has no open/row-polymorphic
//! tuple, so the arity cannot be pinned from one projection.
//!
//! These assertions traverse from Surf source through desugar + macro
//! expansion into `check_ir_program`, and pin the exact nominal-naming
//! diagnostic (a swallowed reject with an empty/generic message would be a
//! laundered pass).
//!
//! Spec authority: spec/04-type-system.md (tuple projection, nominal ADT
//! equality).

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::{InferResult, check_ir_program};

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
}

fn messages(rep: &InferResult) -> Vec<String> {
    rep.errors.iter().map(|e| e.message.clone()).collect()
}

/// Reject-path assertion: `check_ir_program` must `Err`, AND at least one
/// diagnostic must contain every `needle`. The needle set pins the
/// nominal-naming so a generic propagated error (or an empty message
/// vector) cannot masquerade as the intended reject.
fn assert_rejects_naming(source: &str, needles: &[&str], label: &str) {
    let deep = surf_to_deep(source);
    let rep = check_ir_program(&deep)
        .err()
        .unwrap_or_else(|| panic!("{label}: expected a check rejection, but it passed"));
    let msgs = messages(&rep);
    assert!(
        msgs.iter()
            .any(|m| needles.iter().all(|needle| m.contains(needle))),
        "{label}: expected a diagnostic containing all of {needles:?}; got {msgs:?}",
    );
}

/// Accept-path assertion: `check_ir_program` must succeed. A false positive
/// from the projection fix (over-rejecting a correct program) fails here.
fn assert_checks(source: &str, label: &str) {
    let deep = surf_to_deep(source);
    assert!(
        check_ir_program(&deep).is_ok(),
        "{label}: expected a clean check, got {:?}",
        check_ir_program(&surf_to_deep(source))
            .err()
            .map(|r| messages(&r)),
    );
}

/// Three same-shaped, distinct nominal record types plus the loss whose
/// gradient the reproducer projects. `g_of` is the honest `P -> G` adapter
/// the issue routes the projected cotangent through.
const PRELUDE: &str = "\
type P =
  | P { v: tensor[2, f32] }
type G =
  | G { v: tensor[2, f32] }
type V =
  | V { v: tensor[2, f32] }
def e_loss(p: P, s: tensor[2, f32]) -> f32 = match p with {
  | P { v } => sum(add(v, s), cast(0, i32)) |> tensor_to_scalar
}
def g_of(x: P) -> G = match x with {
  | P { v } => G { v }
}
def e_step(params: P, grads: G, vel: V) -> f32 = match params with {
  | P { v: pv } => match grads with {
    | G { v: gv } => match vel with {
    | V { v: vv } => sum(add(add(pv, gv), vv), cast(0, i32)) |> tensor_to_scalar
  } } }
";

fn src(body: &str) -> String {
    format!("{PRELUDE}{body}")
}

// ─── must-fail-check: the erasure is closed ──────────────────────────────

#[test]
fn reproducer_transposed_grad_tuple_slots_reject() {
    // The issue reproducer: `grad(e_loss)(p0, s).0` -> `g_of(...)` -> a
    // transposed `e_step(p, vel, grads)` (slots 2 and 3 swapped). Before
    // the fix this scored 1 and died at runtime; now the nominal mismatch
    // is caught at check.
    assert_rejects_naming(
        &src("\
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  gt = grad(e_loss)(p0, s)
  grads = g_of(gt.0)
  vel = V { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  p = P { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  e_step(p, vel, grads)
}
"),
        &["mismatch", "G", "V"],
        "reproducer transposed grad-tuple",
    );
}

#[test]
fn bare_tuple_literal_projection_carries_element_type() {
    // The root cause is NOT grad-specific: a bare tuple literal `(p0, s).0`
    // is `P`; ascribing it `V` must reject. This is the tightest witness
    // that the projection index now parses.
    let deep = surf_to_deep(&src("\
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  a: V = (p0, s).0
  cast(0.0, f32)
}
"));
    let report = check_ir_program(&deep).expect_err("a projected P cannot be assigned to V");
    assert!(
        report.errors.iter().any(|error| {
            error.expected.as_deref() == Some("V") && error.got.as_deref() == Some("P")
        }),
        "the projection must retain its P type and reject the V ascription: {:?}",
        report.errors
    );
}

#[test]
fn grad_tuple_projection_into_wrong_param_rejects() {
    // `grad(e_loss)(p0, s).0` is `P`; feeding it into a `V` parameter must
    // reject — with no `let` ascription in the way, isolating the
    // projection's carried type at a call boundary.
    assert_rejects_naming(
        &src("\
def take_v(x: V) -> f32 = cast(0.0, f32)
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  gt = grad(e_loss)(p0, s)
  take_v(gt.0)
}
"),
        &["mismatch", "V", "P"],
        "grad projection into V param",
    );
}

#[test]
fn cross_return_helper_seal_does_not_launder() {
    // rlronan's "partial containment": sealing grad + projection + re-wrap
    // behind a helper with an honest `-> G` return does NOT restore
    // laundering. `mk_grads(...)` is `G`; into a `V` slot it must reject.
    // (The erasure was provenance-based and crossed returns *because* the
    // projected value was `Type::Error`; now it carries `G`.)
    assert_rejects_naming(
        &src("\
def mk_grads(p0: P, s: tensor[2, f32]) -> G = {
  gt = grad(e_loss)(p0, s)
  match gt.0 with { | P { v } => G { v } }
}
def take_v(x: V) -> f32 = cast(0.0, f32)
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  take_v(mk_grads(p0, s))
}
"),
        &["mismatch", "V", "G"],
        "cross-return helper seal",
    );
}

#[test]
fn wrong_nominal_match_two_returns_downstream_rejects() {
    // rlronan's better reproducer: a grad-derived value flows through two
    // declared `-> G` returns into a function with NO `grad` in it, then is
    // `match`ed at the wrong nominal type (`V`). Now caught statically as a
    // non-exhaustive match missing the real `G` variant — the exact runtime
    // failure, lifted to compile time.
    assert_rejects_naming(
        &src("\
def hop1(p0: P, s: tensor[2, f32]) -> G = {
  gt = grad(e_loss)(p0, s)
  g_of(gt.0)
}
def hop2(p0: P, s: tensor[2, f32]) -> G = hop1(p0, s)
def accum(grads: G) -> f32 =
  match grads with { | V { v: vv } => sum(vv, cast(0, i32)) |> tensor_to_scalar }
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  accum(hop2(p0, s))
}
"),
        &["non-exhaustive", "G"],
        "wrong nominal match two returns downstream",
    );
}

#[test]
fn transposed_call_inside_generic_fn_rejects() {
    // A non-trigger in the issue's map that must remain caught: the same
    // transposition inside a generic function.
    assert_rejects_naming(
        &src("\
def generic_driver[a](witness: tensor[a, f32]) -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  gt = grad(e_loss)(p0, s)
  grads = g_of(gt.0)
  vel = V { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  p = P { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  e_step(p, vel, grads)
}
"),
        &["mismatch", "G", "V"],
        "transposed inside generic fn",
    );
}

#[test]
fn transposed_call_inside_match_body_rejects() {
    // Another non-trigger that must remain caught: the transposition inside
    // a `match` arm (params destructured and rebuilt first).
    assert_rejects_naming(
        &src("\
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  gt = grad(e_loss)(p0, s)
  grads = g_of(gt.0)
  vel = V { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  p = P { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  match p with {
    | P { v: pv } => e_step(P { v: pv }, vel, grads)
  }
}
"),
        &["mismatch", "G", "V"],
        "transposed inside match body",
    );
}

#[test]
fn direct_construction_control_still_rejects() {
    // The issue's one-line control: replace the grad-derived binding with a
    // directly constructed `G`. This was always caught; it must stay caught
    // (guards against the fix over-suppressing).
    assert_rejects_naming(
        &src("\
def driver() -> f32 = {
  gt = G { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  vel = V { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  p = P { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  e_step(p, vel, gt)
}
"),
        &["mismatch", "G", "V"],
        "direct-construction control",
    );
}

// ─── must-stay-green: correct programs still check ───────────────────────

#[test]
fn reproducer_correctly_ordered_slots_check() {
    // The reproducer with the slots in the RIGHT order must type-check —
    // the fix must not manufacture a spurious mismatch on a correct call.
    assert_checks(
        &src("\
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  gt = grad(e_loss)(p0, s)
  grads = g_of(gt.0)
  vel = V { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  p = P { v: to_tensor([cast(1.0, f32), cast(1.0, f32)]) }
  e_step(p, grads, vel)
}
"),
        "reproducer correctly ordered",
    );
}

#[test]
fn projection_ascribed_correct_element_type_checks() {
    assert_checks(
        &src("\
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  gt = grad(e_loss)(p0, s)
  a: P = gt.0
  cast(0.0, f32)
}
"),
        "projection ascribed correct element",
    );
}

#[test]
fn two_grad_instantiations_of_one_loss_check() {
    assert_checks(
        &src("\
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  g1 = grad(e_loss)(p0, s)
  g2 = grad(e_loss)(p0, s)
  a: P = g1.0
  b: P = g2.0
  cast(0.0, f32)
}
"),
        "two grad instantiations",
    );
}

#[test]
fn cross_return_helper_into_matching_slot_checks() {
    assert_checks(
        &src("\
def mk_grads(p0: P, s: tensor[2, f32]) -> G = {
  gt = grad(e_loss)(p0, s)
  match gt.0 with { | P { v } => G { v } }
}
def take_g(x: G) -> f32 = cast(0.0, f32)
def driver() -> f32 = {
  p0 = P { v: to_tensor([cast(1.0, f32), cast(2.0, f32)]) }
  s = to_tensor([cast(0.5, f32), cast(0.5, f32)])
  take_g(mk_grads(p0, s))
}
"),
        "cross-return helper into matching slot",
    );
}

#[test]
fn polymorphic_tuple_fold_accumulator_projection_checks() {
    // Guards the `Type::Var` deferral arm: an unannotated `fold`
    // accumulator is projected (`state.0`, `state.1`) before its tuple
    // type is known. The projection must defer (fresh element type), not
    // reject with "expected tuple type".
    assert_checks(
        "\
def f[n](xs: tensor[n, f32]) -> (tensor[n, f32], i64) = {
  idxs = range(cast(0, i64), numel(copy(xs)))
  state0 = (to_tensor(map(fn (x: f32) -> cast(0.0, f32), to_list(copy(xs)))), cast(0, i64))
  step = fn (state, i) -> {
    acc = state.0
    total = state.1
    (acc, add(total, i))
  }
  fold(step, state0, idxs)
}
",
        "polymorphic tuple fold accumulator projection",
    );
}
