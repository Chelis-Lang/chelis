//! chelis#1489, gates 3 and 4: the shape-computing routes must defer an
//! operand that is merely not resolved yet.
//!
//! `spec/04-type-system.md` §3.2 "Operand decisions on an unresolved operand"
//! states the rule for every operation: where the decision depends on the
//! operand's type and that type is still an unresolved variable, the decision
//! is deferred and settled at the binding, and an operand that never resolves
//! is rejected with the same diagnostic as a concretely inadmissible one.
//!
//! Four routes are converted here: `gather`, `scatter` and `scatter_replace`
//! (through `infer_gather_result_type`, which the issue names as one of its two
//! STRONGER defects -- it takes no `subst`, so it cannot tell "not a tensor"
//! from "not yet resolved" even in principle) and `trace`.
//!
//! `concat` and `diagonal` are deliberately NOT converted. Both compute a new
//! extent arithmetically from the operand's dims -- a sum and a minimum -- and
//! both helpers encode a dim they cannot compute with as `Dim::Wildcard`, which
//! the declared signature then narrows to ANY extent. Deferring them therefore
//! reopens a false-signature defect whenever a dim they read is still a variable
//! at discharge: three consecutive review rounds each found it through a
//! different door. `gather`/`scatter` only COPY dims into the result, so an
//! unresolved dim passes through and resolves later, and they are unaffected.
//! `the_arithmetic_routes_are_not_converted` pins the exclusion.
//!
//! These routes are not a relabelling of the `copy`/`cast` conversion. Each
//! computes its result FROM the operand's shape, and each arm does work
//! AROUND its helper that a deferral must reproduce:
//!
//!   - the axis is normalized against the operand's rank BEFORE the helper
//!     runs, and `resolve_builtin_axis` silently yields 0 for a negative axis
//!     when the operand is not a concrete tensor. A deferral that replays only
//!     the helper therefore freezes a WRONG axis;
//!   - `scatter` and `scatter_replace` additionally unify the updates operand
//!     against the helper's result. A deferral that replays only the helper
//!     drops that check.
//!
//! `an_axis_is_normalized_against_the_settled_rank` and
//! `a_deferred_scatter_still_checks_its_updates_operand` are the pins for
//! exactly those two hazards.

use std::process::Command;

fn chelis_bin() -> &'static str {
    env!("CARGO_BIN_EXE_chelis")
}

fn check_json(source: &str) -> serde_json::Value {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("input.ch");
    std::fs::write(&path, source).expect("write source");
    let out = Command::new(chelis_bin())
        .arg("check")
        .arg(&path)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .output()
        .expect("run chelis check");
    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("bad JSON: {e}\n{text}"))
}

fn messages(report: &serde_json::Value) -> Vec<String> {
    report["errors"]
        .as_array()
        .map(|es| {
            es.iter()
                .filter_map(|e| e["message"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// A lambda parameter is a genuine `Type::Var` while the body is inferred and
/// is bound only by the outer application. This is the shape chelis#1489
/// reports as `got ?N`; it does not depend on `expand`, which chelis#1277 S2b
/// made concrete.
fn deferred(call: &str, ret: &str) -> serde_json::Value {
    check_json(&format!(
        "module Issue1489Shape\n\
         def apply_it(f: (tensor[4, 3, f32]) -> {ret}, t: tensor[4, 3, f32]) -> {ret} = f(t)\n\
         def probe(t: tensor[4, 3, f32]) -> {ret} = apply_it(fn (v) -> {call}, t)\n"
    ))
}

fn eager(call: &str, ret: &str) -> serde_json::Value {
    check_json(&format!(
        "module Issue1489ShapeEager\n\
         def probe(v: tensor[4, 3, f32]) -> {ret} = {call}\n"
    ))
}

/// (call, declared result) for each of the four converted routes, at a shape where the
/// eager form checks clean.
const ROUTES: &[(&str, &str)] = &[
    ("gather(v, to_tensor([0i32]), 0i32)", "tensor[1, 3, f32]"),
    (
        "scatter(v, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32, 3.0f32]]), 0i32, \"add\")",
        "tensor[4, 3, f32]",
    ),
    (
        "scatter_replace(v, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32, 3.0f32]]), 0i32)",
        "tensor[4, 3, f32]",
    ),
    ("trace(v, 0i32, 1i32)", "tensor[f32]"),
];

/// The defect itself: an operand settled by a LATER unification must not be
/// rejected at the instant the route runs.
#[test]
fn a_shape_route_accepts_an_operand_resolved_after_the_gate() {
    let mut still_rejecting = Vec::new();
    for (call, ret) in ROUTES {
        let found = messages(&deferred(call, ret));
        // A row whose fixture does not even parse would otherwise satisfy every
        // assertion below by reporting nothing this test looks for. That is how
        // a mis-spelled `trace` row let the whole `trace` deferral be deleted
        // with the suite still green.
        assert!(
            !found.iter().any(|m| m.contains("expected type, found")
                || m.contains("Unbound")
                || m.contains("arity mismatch")),
            "{call}: the fixture is malformed, so this row asserts nothing; \
             got {found:?}"
        );
        if found.iter().any(|m| m.contains('?')) {
            still_rejecting.push(format!("  {call} -> {found:?}"));
        }
    }
    assert!(
        still_rejecting.is_empty(),
        "these routes still reject an operand that is merely unresolved, naming \
         an inference identity the user never wrote:\n{}",
        still_rejecting.join("\n")
    );
}

/// Parity: every route must reach the same verdict deferred as eagerly.
#[test]
fn every_shape_route_agrees_deferred_and_eager() {
    let mut divergences = Vec::new();
    for (call, ret) in ROUTES {
        let (e, d) = (messages(&eager(call, ret)), messages(&deferred(call, ret)));
        if e.is_empty() != d.is_empty() {
            divergences.push(format!(
                "  {call}: eager {} / deferred {}\n    eager: {e:?}\n    deferred: {d:?}",
                if e.is_empty() { "accepted" } else { "REJECTED" },
                if d.is_empty() { "accepted" } else { "REJECTED" },
            ));
        }
    }
    assert!(
        divergences.is_empty(),
        "deferring changed the verdict:\n{}",
        divergences.join("\n")
    );
}

/// HAZARD 1. `resolve_builtin_axis` normalizes a negative axis against the
/// operand's rank, and yields 0 when the operand is not a concrete tensor.
///
/// A deferral that captured that eagerly-computed axis would freeze 0 where
/// the settled rank makes `-1` mean the last axis. `gather(v, i, -1i32)` on
/// rank 2 must mean axis 1, deferred exactly as eagerly.
#[test]
fn an_axis_is_normalized_against_the_settled_rank() {
    // axis -1 on rank 2 is axis 1: result is tensor[4, 1, f32], NOT the
    // tensor[1, 3, f32] that freezing axis 0 would produce.
    let call = "gather(v, to_tensor([0i32]), -1i32)";
    let e = messages(&eager(call, "tensor[4, 1, f32]"));
    let d = messages(&deferred(call, "tensor[4, 1, f32]"));
    assert!(
        e.is_empty(),
        "the eager form must accept a negative axis; got {e:?}"
    );
    assert!(
        d.is_empty(),
        "a deferred negative axis must normalize against the SETTLED rank, not \
         freeze to 0 while the operand was unresolved; got {d:?}"
    );
    // And the wrong shape must still be rejected, so the test above cannot
    // pass by the route having stopped constraining anything.
    let wrong = messages(&deferred(call, "tensor[1, 3, f32]"));
    assert!(
        !wrong.is_empty(),
        "axis -1 on rank 2 is axis 1, so tensor[1, 3, f32] is the WRONG result \
         and must be rejected; got a clean report"
    );
}

/// HAZARD 2. `scatter` and `scatter_replace` unify the updates operand against
/// the helper's result. A deferral that replays only the helper drops that
/// check and admits a program whose updates shape cannot be right.
#[test]
fn a_deferred_scatter_still_checks_its_updates_operand() {
    for call in [
        // updates is tensor[1, 3] for axis 0 on rank 2; tensor[1, 2] is wrong.
        "scatter(v, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32]]), 0i32, \"add\")",
        "scatter_replace(v, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32]]), 0i32)",
    ] {
        let e = messages(&eager(call, "tensor[4, 3, f32]"));
        let d = messages(&deferred(call, "tensor[4, 3, f32]"));
        assert!(
            !e.is_empty(),
            "{call}: the eager form rejects this updates shape; got a clean report"
        );
        assert!(
            !d.is_empty(),
            "{call}: deferring must not drop the updates-shape check; got a \
             clean report"
        );
    }
}

/// Negative parity: deferring must not become accepting. An operand that
/// settles to something concretely wrong is still rejected.
#[test]
fn an_operand_that_settles_to_a_non_tensor_is_still_rejected() {
    for (call, _) in ROUTES {
        let report = check_json(&format!(
            "module Issue1489ShapeNeg\n\
             def apply_it(f: (unit) -> unit, t: unit) -> unit = f(t)\n\
             def probe(t: unit) -> unit = {{\n\
            \x20 g = apply_it(fn (v) -> {{ h = {call}\n\
            \x20   () }}, t)\n\
            \x20 g\n\
             }}\n"
        ));
        let found = messages(&report);
        assert!(
            !found.is_empty(),
            "{call}: an operand that settles to `unit` is concretely wrong and \
             must still be rejected; got a clean report"
        );
    }
}

/// Negative parity: a variable that is NEVER bound must still be rejected, or
/// the route has been made tolerant rather than deferred.
#[test]
fn an_operand_that_never_resolves_is_still_rejected() {
    for (call, _) in ROUTES {
        let report = check_json(&format!(
            "module Issue1489ShapeNever\n\
             def go[t](x: t) -> int32 = {{\n\
            \x20 g = {call2}\n\
            \x20 1i32\n\
             }}\n",
            call2 = call.replace('v', "x")
        ));
        assert!(
            !messages(&report).is_empty(),
            "{call}: a never-resolved operand must stay rejected; got a clean report"
        );
    }
}

/// P1 from review round 1. `scatter`'s mode string is validated by the call,
/// not by the arm, so deferring cannot skip it.
///
/// The eager arm checked the mode after resolving the axis; the deferred path
/// returned before reaching it and the gate did not carry `mode`, so an invalid
/// mode became a RUNTIME abort in both the C and eval lanes instead of a type
/// error. `spec/04-type-system.md` §3.2 forbids exactly that: "the verdict never
/// depends on the order in which inference reaches the operand".
#[test]
fn a_deferred_scatter_still_validates_its_mode() {
    let call =
        "scatter(v, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32, 3.0f32]]), 0i32, \"bogus\")";
    for (label, found) in [
        ("eager", messages(&eager(call, "tensor[4, 3, f32]"))),
        ("deferred", messages(&deferred(call, "tensor[4, 3, f32]"))),
    ] {
        assert!(
            found.iter().any(|m| m.contains("mode must be")),
            "{label}: an invalid scatter mode is a type error, not a runtime \
             abort; got {found:?}"
        );
    }
}

/// Review round 1 found that a borrowed operand rejected deferred while the eager
/// arm accepted it, because the eager arms strip one `Type::Ref` through
/// `type_for_readonly_check` and discharge did not. The strip now lives inside
/// the shared decision, so both paths get it.
#[test]
fn a_borrowed_operand_is_accepted_deferred_as_it_is_eagerly() {
    for (call, ret) in [
        ("gather(v, to_tensor([0i32]), 0i32)", "tensor[1, 3, f32]"),
        ("trace(v, 0i32, 1i32)", "tensor[f32]"),
    ] {
        let report = check_json(&format!(
            "module Issue1489ShapeBorrow\n\
             def apply_it(f: (&tensor[4, 3, f32]) -> {ret}, t: &tensor[4, 3, f32]) -> {ret} = f(t)\n\
             def probe(t: &tensor[4, 3, f32]) -> {ret} = apply_it(fn (v) -> {call}, t)\n"
        ));
        assert!(
            messages(&report).is_empty(),
            "{call}: the eager arm accepts a borrowed operand, so the deferred \
             form must too; got {:?}",
            messages(&report)
        );
    }
}

/// The published diagnostic KIND must not change with inference order.
///
/// chelis#1334 counts these by name, and `DeferredOperandGate::kind`'s own doc
/// calls a kind change smuggled in behind a timing fix a wire change. Round 3
/// found `scatter_replace`'s updates failure rendering as `DimensionMismatch`
/// eagerly and `TypeMismatch` deferred.
#[test]
fn a_deferred_rejection_keeps_the_eager_diagnostic_kind() {
    let call = "scatter_replace(v, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32]]), 0i32)";
    let kinds = |v: &serde_json::Value| -> Vec<String> {
        v["errors"]
            .as_array()
            .map(|es| {
                es.iter()
                    .filter_map(|e| e["kind"].as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let e = eager(call, "tensor[4, 3, f32]");
    let d = deferred(call, "tensor[4, 3, f32]");
    assert!(!kinds(&e).is_empty(), "the eager form must reject this");
    assert_eq!(
        kinds(&e),
        kinds(&d),
        "deferring changed the published diagnostic kind: eager {:?} / deferred {:?}",
        messages(&e),
        messages(&d)
    );
}

/// The borrow strip belongs to the routes whose arms strip, and to no others.
///
/// `gather` and `trace` read their operand through `type_for_readonly_check`,
/// which strips one `Type::Ref`; `scatter` and `scatter_replace` do not, and
/// reject a borrowed operand. Review round 2 found the shared decision
/// stripping unconditionally, which widened the non-stripping routes in the
/// ACCEPTING direction -- and, while `concat` still went through the shared
/// decision, reopened round 1's false-signature defect: a borrowed `concat`
/// passed the element guard with EMPTY per-element dims, making the concat
/// axis a wildcard, so every declared extent was admitted and the program
/// built and ran with an exported signature that was a lie.
///
/// `concat` is no longer converted, so its rows below now pin `main`'s own
/// behaviour, which this change must not disturb. The companion
/// `a_borrowed_operand_is_accepted_deferred_as_it_is_eagerly` covers the
/// routes that DO strip.
#[test]
fn the_routes_that_reject_a_borrowed_operand_still_reject_it() {
    // `concat` first: it is the route that produced a false signature, and is
    // unconverted, so this pins `main`'s own rejection.
    for n in ["8", "9", "100"] {
        let report = check_json(&format!(
            "module Issue1489ShapeBorrowNeg\n\
             def probe(a: tensor[4, 3, f32], c: tensor[4, 3, f32]) -> tensor[{n}, 3, f32] =\n\
            \x20 concat([&a, &c], 0i32)\n"
        ));
        let found = messages(&report);
        assert!(
            found
                .iter()
                .any(|m| m.contains("concat expects List[tensor[...]]")),
            "concat rejects a borrowed element; admitting it makes the concat \
             axis a wildcard and every declared extent pass. n={n}, got {found:?}"
        );
    }
    for call in [
        "scatter(&t, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32, 3.0f32]]), 0i32, \"add\")",
        "scatter_replace(&t, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32, 3.0f32]]), 0i32)",
    ] {
        let report = check_json(&format!(
            "module Issue1489ShapeBorrowNegS\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[4, 3, f32] = {call}\n"
        ));
        let found = messages(&report);
        assert!(
            found.iter().any(|m| m.contains("expects tensor input")),
            "{call}: a shared borrow is not a mutable base and must be \
             rejected; got {found:?}"
        );
    }
}

/// A call wrong in two ways reports the same one it reported before.
///
/// `scatter`'s arm resolved the axis (bounds included) before checking the
/// mode string. Validating the mode first in the shared decision silently
/// swapped which error a doubly-wrong call reports.
#[test]
fn axis_bounds_are_reported_before_the_mode_string() {
    let report = check_json(
        "module Issue1489ShapeOrder\n\
         def probe(v: tensor[4, 3, f32]) -> tensor[4, 3, f32] =\n\
        \x20 scatter(v, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32, 3.0f32]]), 9i32, \"bogus\")\n",
    );
    let found = messages(&report);
    assert!(
        found.iter().any(|m| m.contains("axis 9 out of bounds")),
        "the axis-bounds error comes first, as it did before the deferral; \
         got {found:?}"
    );

    // A non-tensor operand does NOT come first: the eager resolvers let the
    // route's helper reject it, after the mode check. Review round 3 found the
    // shared decision reporting the operand error instead, across 30 cells of
    // an 800-cell eager matrix against `main`.
    for operand in ["1i32", "&t"] {
        let report = check_json(&format!(
            "module Issue1489ShapeOrderOperand\n\
             def probe(t: tensor[4, 3, f32]) -> tensor[4, 3, f32] =\n\
            \x20 scatter({operand}, to_tensor([0i32]), to_tensor([[1.0f32, 2.0f32, 3.0f32]]), 0i32, \"bogus\")\n"
        ));
        let found = messages(&report);
        assert!(
            found.iter().any(|m| m.contains("mode must be")),
            "operand {operand}: `main` reports the mode error before the operand \
             error; got {found:?}"
        );
    }
}

/// `concat` and `diagonal` are deliberately NOT converted, and this pins why.
///
/// Both compute a NEW extent arithmetically from the operand's dims -- `concat`
/// a sum, `diagonal` a minimum -- and both helpers encode a dim they cannot
/// compute with as `Dim::Wildcard`. Deferral discharges when the operand's
/// TYPE variable binds, and at that moment its dims can still be variables:
/// here `apply_n`'s dim-polymorphic `n` is not fixed until the argument is
/// processed. The helper then yields a wildcard, the declared signature
/// narrows it to whatever extent it claims, and a function declared
/// `tensor[100, 3, f32]` that really returns `tensor[8, 3, f32]` checked,
/// built, and ran. Three consecutive review rounds found that one defect class
/// through three different doors, which is why the routes were withdrawn from
/// this change rather than patched a fourth time.
///
/// Converting either route again must first stop its helper encoding "unknown"
/// as permissive -- a constrained fresh dim, or a rejection -- and this test is
/// the first thing that will say so.
#[test]
fn the_arithmetic_routes_are_not_converted() {
    for (call, lie) in [
        ("concat([v, v], 0i32)", "tensor[100, 3, f32]"),
        ("diagonal(v, 0i32, 1i32)", "tensor[100, f32]"),
    ] {
        let report = check_json(&format!(
            "module Issue1489ShapeArith\n\
             def apply_n[b](f: tensor[n, 3, f32] -> b, x: tensor[n, 3, f32]) -> b = f(x)\n\
             def probe(t: tensor[4, 3, f32]) -> {lie} = apply_n(fn (v) -> {call}, t)\n"
        ));
        let found = messages(&report);
        assert!(
            !found.is_empty(),
            "{call}: declared {lie} is FALSE -- the real extent is smaller -- so \
             this must be rejected. A clean report means the route was converted \
             again and the wildcard false-signature defect is back"
        );
        // And it is rejected as `main` rejects it: as an unresolved operand the
        // route does not defer, not as a shape mismatch it somehow computed.
        assert!(
            found.iter().any(|m| m.contains('?')),
            "{call}: the route is unconverted, so it must still reject the \
             unresolved operand itself; got {found:?}"
        );
    }
}
