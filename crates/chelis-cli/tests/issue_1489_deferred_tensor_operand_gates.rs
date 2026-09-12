//! chelis#1489: gates that decided on an unresolved operand too early.
//!
//! `copy`, `cast`, `gather` and `concat` (and, per the issue's per-builtin
//! census, `diagonal`, `trace`, `scatter`, `scatter_replace`, `round_to` and
//! the csv routes) rejected an operand that was still an unresolved
//! `Type::Var`, where ~71 sibling gates treat one as "not yet known" and carry
//! on. Deciding at the instant the node was inferred made them sensitive to
//! WHEN inference resolved a variable rather than to whether the program was
//! well typed: 0.18.6 changed that timing and they fired ~50x more often on an
//! unchanged corpus, with no measurable change in outcomes.
//!
//! This change converts the routes that can be converted safely, and the split
//! is not arbitrary. `copy`, `cast`, and the ten csv host-lane slots are
//! converted. By the issue's own measured breakdown that is ~97% of the 7042
//! occurrences: `cast` 4534 and `copy` 2314 carry nearly all of it, while the
//! csv slots contribute ZERO measured occurrences and gain only a
//! message-quality improvement -- they still reject, and the test below asserts
//! it. `round_to` is NOT converted: the deferring seam carries one expected
//! type and `round_to` accepts more than one. The six tensor routes
//! (`gather`, `scatter`, `scatter_replace`, `diagonal`, `trace`, `concat`)
//! compute a result FROM the operand's shape; a deferral there must reproduce
//! every constraint the eager arm imposed, and measurement showed that is more
//! than the one helper call the arm appears to make. They were unconverted when
//! this file was written and two tests here pinned that. They now defer, by
//! carrying that other evidence on the gate:
//! `issue_1489_shape_route_deferral.rs` owns them, and the two pins are retired
//! rather than inverted, because that file asserts the same routes from the
//! other side.
//!
//! `cast` and `copy` defer by re-entering their OWN eager decision with the
//! settled type, rather than by a re-derivation of it. An earlier revision
//! wrote a separate recipe and silently dropped constraints; a later one
//! called the shared function with a DIFFERENT argument (peeling `Type::Ref`
//! on one path only), which is the same divergence wearing the shape of a
//! fix. The tests below pin eager and deferred agreement directly.
//!
//! The decision is deferred, never dropped. The issue's suggested tolerant arm
//! would remove the rejection, and that is unsafe here: measured, a tolerant
//! `cast` lets `def go[t](x: t) -> int32 = cast(x, int32)` check at score 1.0
//! AND build, with the backend selecting a dtype for the never-resolved `t`.
//! A variable that is never bound is still rejected by the per-def pass.

use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

fn check_json(source: &str) -> serde_json::Value {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("probe.ch");
    fs::write(&path, source).expect("write fixture");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .arg(&path)
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("stdout is not JSON: {e}\n{stdout}"))
}

fn messages(report: &serde_json::Value) -> Vec<String> {
    report["errors"]
        .as_array()
        .map(|errors| {
            errors
                .iter()
                .filter_map(|e| e["message"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The measured regression, for `copy`: an operand settled by a LATER
/// unification must not be rejected at the moment the gate runs.
///
/// The operand is a lambda parameter, which is an unresolved `Type::Var` while
/// the lambda body is inferred and is bound only when the argument meets
/// `apply_it`'s declared parameter type. An earlier revision built the
/// unresolved operand out of a positional `expand` result instead; chelis#1277
/// S2b gave `expand` exactly one result shape, so that operand now resolves
/// immediately and the fixture asserted nothing. This shape is the one
/// chelis#1489 actually reports ("got ?N") and does not depend on `expand`.
#[test]
fn an_operand_resolved_after_the_gate_is_accepted() {
    let report = check_json(
        "module Issue1489Late\n\
         def apply_it(f: (tensor[3, f32]) -> tensor[3, f32], t: tensor[3, f32]) \
         -> tensor[3, f32] = f(t)\n\
         def probe(t: tensor[3, f32]) -> tensor[3, f32] =\n\
        \x20 apply_it(fn (v) -> copy(v), t)\n",
    );
    assert!(
        messages(&report).is_empty(),
        "an operand resolved later must not be rejected; got {:?}",
        messages(&report)
    );
}

/// Negative parity: deferring must not become accepting. An operand that
/// resolves to something concretely wrong is still rejected.
#[test]
fn an_operand_resolved_to_a_non_tensor_is_still_rejected() {
    let report = check_json(
        "module Issue1489Concrete\n\
         def probe() -> int32 = {\n\
        \x20 g = copy(())\n\
        \x20 1i32\n\
         }\n",
    );
    let found = messages(&report);
    assert!(
        found.iter().any(|m| m.contains("copy requires tensor")),
        "copy of a unit value must still be rejected by name; got {found:?}"
    );
}

/// The property the issue's suggested fix would have removed. `t` is never
/// pinned; a tolerant arm accepts this at 1.0 and lets the backend invent a
/// dtype for it.
#[test]
fn an_operand_that_never_resolves_is_still_rejected() {
    let report = check_json(
        "module Issue1489Never\n\
         def go[t](x: t) -> int32 = {\n\
        \x20 g = copy(x)\n\
        \x20 1i32\n\
         }\n",
    );
    assert!(
        !messages(&report).is_empty(),
        "a never-resolved operand must stay rejected, or the backend will \
         choose a dtype nobody wrote; got a clean report"
    );
}

/// [04-FIT-9] via chelis#260 Site 2: the rejection names the declared type
/// parameter rather than an internal identity, and names the DEFERRING def's
/// parameter — the ledger is per def, so a second signature must not supply
/// the name.
#[test]
fn a_never_resolved_declared_parameter_is_named_not_numbered() {
    let report = check_json(
        "module Issue1489Named\n\
         def alpha[t](x: t) -> int32 = {\n\
        \x20 g = copy(x)\n\
        \x20 1i32\n\
         }\n\
         def beta[q](y: q) -> int32 = 1i32\n",
    );
    let found = messages(&report);
    assert!(
        found.iter().any(|m| m.contains("`t`")),
        "the diagnostic must name alpha's parameter; got {found:?}"
    );
    assert!(
        !found.iter().any(|m| m.contains("`q`") || m.contains('?')),
        "neither beta's parameter nor an internal identity may appear; \
         got {found:?}"
    );
}

/// Whether a report contains a gate's OWN rejection, as opposed to any other
/// diagnostic the fixture may also raise (a declared-return mismatch, say).
///
/// The differential test below compares this rather than raw acceptance,
/// because the eager and deferred fixtures are different programs and can
/// differ in unrelated ways. What must never differ is whether the GATE
/// objected to the operand.
fn gate_rejected(report: &serde_json::Value) -> bool {
    messages(report).iter().any(|m| {
        m.contains("copy requires tensor input") || m.contains("cast requires tensor or prim type")
    })
}

/// A deferred call and an eager call must reach the same verdict about the
/// same operand type. Mechanically, over a grid — not one hand-picked pair.
///
/// This is the pin the previous four rounds lacked. Each of them repaired the
/// exact instance named and added a pin for that exact instance, and the next
/// round found the same class in a new place: a re-derived decision that
/// dropped constraints, then a shared function called with a peeled argument,
/// then a pass that re-derived `copy` inline. Point-pins caught none of the
/// successors. This asserts the invariant instead of its instances.
///
/// The deferred half MUST use the lambda-argument form. It is a verified
/// deferral probe: the operand's type is a `Type::Var` when the gate runs and
/// is settled only by the outer application. The `expand`-based form is NOT:
/// chelis#1277 S2b gave `expand` exactly one result shape, so an `expand`
/// result is concrete by the time a gate looks at it and never reaches any
/// deferral. Earlier revisions of these tests used that form and passed with
/// the whole deferral deleted.
#[test]
fn eager_and_deferred_gates_agree_on_the_same_operand() {
    // (operand type, call, declared result)
    let grid = [
        ("tensor[3, f32]", "copy(v)", "tensor[3, f32]"),
        ("&tensor[3, f32]", "copy(v)", "tensor[3, f32]"),
        ("&tensor[3, f32]", "copy(v)", "&tensor[3, f32]"),
        ("unit", "copy(v)", "unit"),
        ("f32", "copy(v)", "f32"),
        ("(int32, int32)", "copy(v)", "(int32, int32)"),
        ("tensor[3, f32]", "cast(v, f64)", "tensor[3, f64]"),
        ("&tensor[3, f32]", "cast(v, f64)", "tensor[3, f64]"),
        ("f32", "cast(v, f64)", "f64"),
        ("unit", "cast(v, f64)", "f64"),
        ("(int32, int32)", "cast(v, f64)", "f64"),
        ("tensor[3, f32]", "cast_trunc(v, int32)", "tensor[3, int32]"),
        (
            "&tensor[3, f32]",
            "cast_trunc(v, int32)",
            "tensor[3, int32]",
        ),
        ("f64", "cast_trunc(v, int32)", "int32"),
    ];
    let mut divergences = Vec::new();
    for (ty, call, ret) in grid {
        let eager = check_json(&format!(
            "module Issue1489DiffEager\n\
             def probe(v: {ty}) -> {ret} = {call}\n"
        ));
        let deferred = check_json(&format!(
            "module Issue1489DiffDeferred\n\
             def apply_it(f: ({ty}) -> {ret}, t: {ty}) -> {ret} = f(t)\n\
             def probe(t: {ty}) -> {ret} = apply_it(fn (v) -> {call}, t)\n"
        ));
        let (e, d) = (gate_rejected(&eager), gate_rejected(&deferred));
        if e != d {
            divergences.push(format!(
                "  {call} on {ty} -> {ret}: eager {} / deferred {}\n    eager: {:?}\n    deferred: {:?}",
                if e { "REJECTED" } else { "accepted" },
                if d { "REJECTED" } else { "accepted" },
                messages(&eager),
                messages(&deferred)
            ));
        }
    }
    assert!(
        divergences.is_empty(),
        "the deferred path must reach the same gate verdict as the eager one; \
         a divergence means some path sees a different type, which is every \
         defect this issue has had:\n{}",
        divergences.join("\n")
    );
}

/// A constraint whose operand is settled only by ANOTHER suspended
/// constraint's result must still reach the eager verdict.
///
/// This is the case the end-of-def drain could not get right, and the reason
/// the drain is gone. A pass over a ledger has an order; `copy(copy(t))` in
/// deferred position puts the settler after the settled, so the outer
/// constraint was decided against an unbound variable and rejected a program
/// the eager form accepts. Discharging at the binding has no order to get
/// wrong -- settling the inner constraint binds the variable the outer one is
/// waiting on, which discharges the outer one in turn.
///
/// Nesting is what makes these load-bearing: a single `apply_it` only proves a
/// constraint survives one hop.
#[test]
fn a_chain_of_deferred_gates_agrees_with_the_eager_form() {
    let chains = [
        ("copy(copy(t))", "copy(v)", "copy(w)", "tensor[3, f32]"),
        (
            "copy(cast_trunc(t, int32))",
            "copy(v)",
            "cast_trunc(w, int32)",
            "tensor[3, int32]",
        ),
        (
            "cast_trunc(copy(t), int32)",
            "cast_trunc(v, int32)",
            "copy(w)",
            "tensor[3, int32]",
        ),
    ];
    let mut divergences = Vec::new();
    for (eager_expr, outer, inner, ret) in chains {
        let eager = check_json(&format!(
            "module Issue1489ChainEager\n\
             def probe(t: tensor[3, f32]) -> {ret} = {eager_expr}\n"
        ));
        let deferred = check_json(&format!(
            "module Issue1489ChainDeferred\n\
             def apply_it[a, b](f: (a) -> b, t: a) -> b = f(t)\n\
             def probe(t: tensor[3, f32]) -> {ret} =\n\
             \x20 apply_it(fn (v) -> {outer}, apply_it(fn (w) -> {inner}, t))\n"
        ));
        if messages(&eager).is_empty() != messages(&deferred).is_empty() {
            divergences.push(format!(
                "  {eager_expr}: eager {:?} / deferred {:?}",
                messages(&eager),
                messages(&deferred)
            ));
        }
    }
    assert!(
        divergences.is_empty(),
        "a gate settled by another gate's result must reach the eager verdict; \
         a divergence here means something is deciding on a schedule:\n{}",
        divergences.join("\n")
    );
}

/// `cast` deferred and `cast` eager must decide the same way.
///
/// This is the pin for the defect that killed two earlier revisions: the
/// deferred path is only sound if it imposes exactly the eager path's
/// constraints. A borrowed source is the sharpest probe, because `cast`
/// rejects `&tensor` — so if the two paths disagree at all, they disagree
/// here. They did: one revision peeled `Type::Ref` on the deferred path and not in the
/// eager arm, so a borrowed source was rejected eagerly and accepted deferred,
/// and the accepted program built.
#[test]
fn a_deferred_cast_decides_as_the_eager_one_does() {
    let eager = check_json(
        "module Issue1489CastEager\n\
         def probe(t: &tensor[3, f32]) -> tensor[3, int32] = cast_trunc(t, int32)\n",
    );
    let deferred = check_json(
        "module Issue1489CastDeferred\n\
         def apply_it(f: (&tensor[3, f32]) -> tensor[3, int32], t: &tensor[3, f32]) -> tensor[3, int32] = f(t)\n\
         def probe(t: tensor[3, f32]) -> tensor[3, int32] = apply_it(fn (v) -> cast_trunc(v, int32), &t)\n",
    );
    assert!(
        !messages(&eager).is_empty(),
        "eager cast of a borrowed source is rejected; got a clean report"
    );
    assert!(
        !messages(&deferred).is_empty(),
        "a deferred cast must reject the same borrowed source the eager one \
         rejects; accepting it means the two paths see different types"
    );
}

/// The measured regression shape for `cast`, which is 64% of the issue's
/// occurrences: an operand settled by a LATER ascription must be accepted.
/// The operand form here is deliberate and must not be "simplified" to the
/// `expand` form earlier revisions used. chelis#1277 S2b gave `expand` exactly
/// one result shape, so an `expand` result is already resolved by the time
/// `cast` looks at it and never reaches the deferral at all. This test passed
/// with the entire deferral deleted while it used that form. The
/// lambda-argument form is a verified probe: `v` is genuinely a `Type::Var`
/// when the gate runs.
///
/// Round 5 claimed this test "cannot detect a deleted deferral by
/// construction". Measured, it does: delete `cast`'s suspension so a `Type::Var`
/// source falls to the reject arm, and this test fails. The round-5 claim
/// reasoned from a mutation that returned an unconstrained fresh variable,
/// which is a state the checker never had -- before the suspension existed, an
/// unresolved source was REJECTED. Reasoning about a model of the bug instead
/// of the bug got the conclusion exactly backwards, and it named
/// `a_deferred_cast_constrains_its_result` as the half that catches deletion
/// when that one passes under the mutation (its `!messages.is_empty()` is
/// satisfied by the eager rejection).
#[test]
fn a_cast_operand_resolved_after_the_gate_is_accepted() {
    let report = check_json(
        "module Issue1489CastLate\n\
         def apply_it(f: (tensor[3, f32]) -> tensor[3, f64], t: tensor[3, f32]) -> tensor[3, f64] = f(t)\n\
         def probe(t: tensor[3, f32]) -> tensor[3, f64] = apply_it(fn (v) -> cast(v, f64), t)\n",
    );
    assert!(
        messages(&report).is_empty(),
        "a cast whose operand resolves later must not be rejected; got {:?}",
        messages(&report)
    );
}

/// A suspended `cast` must CONSTRAIN its result, or an ill-typed program
/// reaches codegen with a dtype nobody wrote.
///
/// Uses the lambda-argument form for the reason given above: the `expand` form
/// never reaches `cast`'s suspension.
///
/// This test does NOT detect a deleted suspension -- its assertion is only
/// that the program is rejected, and an eager rejection satisfies that just as
/// well. `a_cast_operand_resolved_after_the_gate_is_accepted` is the half that
/// catches deletion.
#[test]
fn a_deferred_cast_constrains_its_result() {
    for declared in ["f64", "tensor[99, 7, f64]", "tensor[3, f32]"] {
        let report = check_json(&format!(
            "module Issue1489CastResult\n\
             def apply_it(f: (tensor[3, f32]) -> {declared}, t: tensor[3, f32]) -> {declared} = f(t)\n\
             def probe(t: tensor[3, f32]) -> {declared} = apply_it(fn (v) -> cast(v, f64), t)\n"
        ));
        assert!(
            !messages(&report).is_empty(),
            "declared {declared} is not what this cast produces and must be \
             rejected; got a clean report"
        );
    }
}

/// The deferral must not settle the very operand it deferred. An earlier
/// revision returned the source's own variable as the result, so a declared
/// return bound the operand and the deferred path saw a settled type.
#[test]
fn a_deferred_cast_does_not_resolve_its_own_operand() {
    let report = check_json(
        "module Issue1489CastSelf\n\
         def go[t](x: t) -> int32 = cast(x, int32)\n",
    );
    let found = messages(&report);
    assert!(
        !found.is_empty(),
        "a direct-body cast must not let the declared return settle the operand \
         it deferred; got a clean report"
    );
    assert!(
        found.iter().any(|m| m.contains("`t`")),
        "[04-FIT-9]: the rejection names the declared parameter; got {found:?}"
    );
}

/// `copy` of a borrow yields the UNWRAPPED tensor, deferred exactly as eager.
///
/// A revision gave the deferred path the operand's own variable, so a deferred
/// `copy(&x)` typed as `&tensor` where an eager one is `tensor` — making the
/// expression's type depend on WHEN the operand resolved, which is the
/// inference-order sensitivity this issue exists to delete.
#[test]
fn a_deferred_copy_unwraps_a_borrow_as_the_eager_one_does() {
    let eager = check_json(
        "module Issue1489CopyEager\n\
         def probe(x: tensor[3, f32]) -> tensor[3, f32] = copy(&x)\n",
    );
    assert!(
        messages(&eager).is_empty(),
        "eager copy(&x) is tensor, not &tensor; got {:?}",
        messages(&eager)
    );
    let deferred_tensor = check_json(
        "module Issue1489CopyDeferredT\n\
         def apply_it(f: (&tensor[3, f32]) -> tensor[3, f32], t: &tensor[3, f32]) -> tensor[3, f32] = f(t)\n\
         def probe(x: tensor[3, f32]) -> tensor[3, f32] = apply_it(fn (v) -> copy(v), &x)\n",
    );
    assert!(
        messages(&deferred_tensor).is_empty(),
        "a deferred copy of a borrow must also be `tensor`; got {:?}",
        messages(&deferred_tensor)
    );
    let deferred_ref = check_json(
        "module Issue1489CopyDeferredR\n\
         def apply_it(f: (&tensor[3, f32]) -> &tensor[3, f32], t: &tensor[3, f32]) -> &tensor[3, f32] = f(t)\n\
         def probe(x: tensor[3, f32]) -> &tensor[3, f32] = apply_it(fn (v) -> copy(v), &x)\n",
    );
    assert!(
        !messages(&deferred_ref).is_empty(),
        "a deferred copy must NOT type as `&tensor`, which the eager arm rejects; \
         got a clean report"
    );
}

/// The ten csv host-lane slots convert at one seam, `unify_host_slot`. They
/// admit no tensor at all, so deferring does not make them accept one — it
/// makes the rejection name the RESOLVED type instead of an inference
/// identity. That is the whole of their contribution here: the behavior change
/// this issue is about belongs to `copy` and `cast`, which newly ACCEPT
/// programs that were rejected before.
///
/// The operand is a lambda parameter, so it really is unresolved when the slot
/// runs. An earlier revision built it from a positional `expand` result, which
/// chelis#1277 S2b made concrete — the seam then decided eagerly and this test
/// passed without ever reaching the deferral.
#[test]
fn a_host_slot_rejection_names_the_resolved_type_not_an_identity() {
    // (call, the builtin's own result type, the slot wording it rejects with)
    // `round_to` is absent deliberately: it is on the eager path, because its
    // accepted set is {f64, f32} and the deferring gate carries one expected
    // type. The ten csv slots are the converted family.
    for (call, ret, gate) in [
        (
            "to_csv(v)",
            "string",
            "to_csv expects List[Dict[string,string]] as the first argument",
        ),
        (
            "csv_nrows(v)",
            "int64",
            "csv_nrows expects List[Dict[string,string]] as the first argument",
        ),
    ] {
        // The lambda's declared return is the builtin's own result type, so the
        // ONLY objection left is the operand. Declaring anything else adds a
        // second, unrelated return-type error and the assertions below stop
        // being about the gate.
        let report = check_json(&format!(
            "module Issue1489Host\n\
             def apply_it(f: (tensor[3, f32]) -> {ret}, t: tensor[3, f32]) -> {ret} = f(t)\n\
             def probe(t: tensor[3, f32]) -> {ret} =\n\
            \x20 apply_it(fn (v) -> {call}, t)\n"
        ));
        let found = messages(&report);
        assert!(
            found.iter().any(|m| m.contains(gate)),
            "{call}: a tensor operand is still wrong for a host slot and must \
             be rejected with the slot's own wording; got {found:?}"
        );
        assert!(
            found
                .iter()
                .any(|m| m.contains(gate) && m.contains("tensor[3, f32]")),
            "{call}: the rejection must name the RESOLVED type; got {found:?}"
        );
        assert!(
            !found.iter().any(|m| m.contains('?')),
            "{call}: no message may name an inference identity; got {found:?}"
        );
    }
}

/// Identifying two variables must carry the suspended constraint across.
///
/// `bind_tvar` can bind a variable to ANOTHER variable rather than to a type.
/// `discharge_operand_gates` re-suspends the obligation on the target when that
/// happens; dropping it there would silently un-defer the constraint and let an
/// operand that is never a tensor through. Nested `apply_it` produces exactly
/// that var-to-var binding.
///
/// This is a regression pin for a branch the rest of the suite does not reach:
/// deleting `realias_operand_gate`'s call site leaves every other test in this
/// file passing while these two programs check clean at score 1.
#[test]
fn a_constraint_survives_its_variable_being_aliased_to_another() {
    for (call, ret, gate) in [
        ("copy(w)", "unit", "copy requires tensor input, got ()"),
        (
            "cast(w, f64)",
            "f64",
            "cast requires tensor or prim type, got ()",
        ),
    ] {
        let report = check_json(&format!(
            "module Issue1489Realias\n\
             def apply_it[a, b](f: (a) -> b, t: a) -> b = f(t)\n\
             def probe(x: unit) -> {ret} =\n\
            \x20 apply_it(fn (v) -> apply_it(fn (w) -> {call}, v), x)\n"
        ));
        let found = messages(&report);
        assert!(
            found.iter().any(|m| m == gate),
            "{call}: a constraint re-suspended onto an aliased variable must \
             still reject a non-tensor operand; got {found:?}"
        );
    }
}

/// An operand a host slot accepts must not draw that SLOT'S OWN rejection
/// because the decision was deferred.
///
/// The negative host-slot test above only ever feeds a tensor, which no slot
/// accepts in any form, so it cannot see a false rejection. That gap let a real
/// one ship: the deferring seam carries a single expected type while `round_to`
/// accepts more than one, so deferring it against `f64` drew
/// `round_to expects an f64 or f32 first argument, got f32` on an operand the
/// eager arm accepts. `round_to` is on the eager path for that reason, and this
/// pins the family against a re-deferral reintroducing it.
///
/// What this asserts, exactly: no message carries the slot's own
/// "<name> expects" wording. It deliberately does NOT assert a clean report.
/// The `f32` row still reports `precision mismatch: expected f32, got f64`,
/// because the eager arm hard-codes an `f64` result while `[05-OP-1]` makes the
/// result follow the operand; that divergence is chelis#1295's, predates this
/// change, and is unaffected by it. Asserting emptiness here would pin
/// chelis#1295's bug instead of this one's.
#[test]
fn an_accepted_host_slot_operand_draws_no_slot_rejection_when_deferred() {
    // (operand type, call, the builtin's own result type)
    for (ty, call, ret) in [
        ("f64", "round_to(v, 2i32)", "f64"),
        ("f32", "round_to(v, 2i32)", "f32"),
        ("List[Dict[string,string]]", "csv_nrows(v)", "int64"),
        ("List[Dict[string,string]]", "to_csv(v)", "string"),
    ] {
        let deferred = check_json(&format!(
            "module Issue1489HostPos\n\
             def apply_it(f: ({ty}) -> {ret}, t: {ty}) -> {ret} = f(t)\n\
             def probe(t: {ty}) -> {ret} = apply_it(fn (v) -> {call}, t)\n"
        ));
        let eager = check_json(&format!(
            "module Issue1489HostPosEager\n\
             def probe(v: {ty}) -> {ret} = {call}\n"
        ));
        assert!(
            messages(&eager).is_empty(),
            "{call} on {ty}: the eager form must accept this operand; got {:?}",
            messages(&eager)
        );
        assert!(
            !messages(&deferred)
                .iter()
                .any(|m| m.contains(&format!("{} expects", call.split('(').next().unwrap()))),
            "{call} on {ty}: the slot accepts this operand eagerly, so the \
             deferred form must not draw the slot's own rejection; got {:?}",
            messages(&deferred)
        );
    }
}
