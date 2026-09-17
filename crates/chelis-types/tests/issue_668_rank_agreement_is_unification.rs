//! chelis#668: rank agreement between elementwise tensor operands is decided
//! by unification, and by nothing else.
//!
//! # The subject
//!
//! `spec/04-type-system.md` section 4.7.2 gives each operation exactly one
//! result shape, and `spec/05-risc-primitives.md` section 1.2 makes identical
//! dimension lists a hard rule for elementwise operands with no broadcasting
//! exception. One mechanism implements that rule: `unify`'s tensor arm
//! resolves both dimension rows and rejects unequal lengths. It runs inside
//! `infer_app`, so it runs for `check_ir_program` and for
//! `check_typed_program` alike, which is why every row below is stated at both
//! ingresses.
//!
//! Until chelis#668 PR A a second mechanism ran beside it. The post-inference
//! shape validator kept a private `ShapeTypeFact::RankOnly` carrier, derived a
//! rank for `stride`, `expand`, and `insert` from the operand rank plus a
//! per-callee delta, and compared two such ranks at every
//! `ShapeClass::Identity` call, on the `check_ir_program` ingress only. The
//! completion design (`spec/design/checker_totality.md` section PP5, D5 and
//! D6 (d)) retires it: no numbered-spec sentence authorizes a second rank
//! model, and the one that existed disagreed with the checker's own stamped
//! types.
//!
//! # Evidentiary status, per assertion
//!
//! `assert_ingress_agreement`, inside `agreed_diagnostics`, is a REGRESSION
//! assertion. On the pre-deletion tree the normalizing ingress reported a
//! second, validator-authored `DimensionMismatch` for a call unification had
//! already rejected, and the stamped ingress reported one. Same program, two
//! diagnostic sets: the chelis#1107 class. Six rows here are red on
//! `12c04c66a`, the base this branch sits on, for exactly that reason.
//!
//! Every verdict assertion, rejections and acceptances alike, is a
//! DISPOSITION LOCK: measured on both trees, the deletion changes no
//! accept/reject verdict and no unification diagnostic in this file. Their job
//! is to hold unification to what it does today, so a later change that
//! weakens the tensor arm cannot pass by re-introducing a side channel that
//! happens to reject the same programs. The pull request's mutation receipt
//! establishes that they measure unification: with `unify`'s ground-row length
//! check disabled, every `tensor rank mismatch` row goes red at BOTH
//! ingresses. The scalar-beside-tensor rows do not, and are not claimed to:
//! they are rejected by the type constructor, not by the rank comparison.
//!
//! # What is not claimed
//!
//! Nothing here is a claim about elementwise operations in general. The
//! covered forms are exactly the ones spelled below: `add`, `mul`, `eq`,
//! `max_elem`, `where`, and `floor_div`, over the operand shapes each fixture
//! writes. No enumerator drives this file, so an operation absent from it is
//! unclaimed rather than excluded.
//!
//! The programs built from `expand` are positive controls, not negatives.
//! Under the one-shape rule `expand` is the same-rank unit-extent broadcast,
//! so the chelis#668 reproducer is rank 1 beside rank 1 and well typed; its
//! loud outcome when the operand extent is not 1 is section 2.4.1's `Domain`
//! trap, which chelis#1277 S2b landed and which
//! `chelis-cli/tests/issue_668_elementwise_rank_honesty.rs` runs on both
//! lanes. The six escapes the PP5 section records (`cast`, `realize`,
//! `normalize`, a user `def`, and the `sum`/`mean` reductions) are runtime
//! outcomes owned by chelis#597 and chelis#1512; this file asserts no verdict
//! on them. The seven comparison identities still carry the chelis#1506
//! scalar rewrite, which is PR B's subject; nothing here asserts on them
//! either.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn deep(source: &str) -> Vec<chelis_deep::Expr> {
    desugar_program(&parse_surf(source).unwrap_or_else(|error| {
        panic!("fixture must parse as Surf: {error:?}\n{source}");
    }))
}

fn messages(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect();
    out.sort();
    out
}

fn ir_diagnostics(exprs: &[chelis_deep::Expr]) -> Vec<String> {
    match check_ir_program(exprs) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

fn typed_diagnostics(exprs: &[chelis_deep::Expr]) -> Vec<String> {
    match check_typed_program(exprs) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

/// REGRESSION assertion (red before the chelis#668 deletion, green after).
///
/// The two ingresses must return the same diagnostic set for the same program.
/// The retired identity-rank validator ran only inside `validate_ir_program`,
/// which `check_typed_program` never calls, so every rank-divergent program
/// below produced two diagnostics on one ingress and one on the other.
///
/// Returns the agreed set so each caller can lock the verdict on top of it.
fn agreed_diagnostics(source: &str, label: &str) -> Vec<String> {
    let exprs = deep(source);
    let ir = ir_diagnostics(&exprs);
    let typed = typed_diagnostics(&exprs);
    assert_eq!(
        ir,
        typed,
        "{label}: rank agreement is unification's, and unification runs on \
         both ingresses, so `check_ir_program` and `check_typed_program` must \
         return the same diagnostic set (chelis#668, chelis#1107).\n\
         ir-only: {:?}\ntyped-only: {:?}",
        ir.iter().filter(|m| !typed.contains(m)).collect::<Vec<_>>(),
        typed.iter().filter(|m| !ir.contains(m)).collect::<Vec<_>>(),
    );
    ir
}

/// DISPOSITION LOCK on unification's rejection, at both ingresses.
///
/// `needle` is the exact `unify` text, so a rejection that started arriving
/// from somewhere else with a different message fails here.
fn assert_rejects_with(source: &str, needle: &str, label: &str) {
    let diagnostics = agreed_diagnostics(source, label);
    assert!(
        diagnostics.iter().any(|m| m.contains(needle)),
        "{label}: both ingresses must reject with a diagnostic containing \
         {needle:?}; got {diagnostics:?}"
    );
}

/// DISPOSITION LOCK on acceptance, at both ingresses. Negative parity for
/// every rejection above: the deletion must not become "stop checking".
fn assert_accepts(source: &str, label: &str) {
    let diagnostics = agreed_diagnostics(source, label);
    assert!(
        diagnostics.is_empty(),
        "{label}: this program is well typed and must check clean on BOTH \
         ingresses; got {diagnostics:?}"
    );
}

// ── Fixtures ──────────────────────────────────────────────────────────────
//
// `s` is rank 1 with wildcard extents (`stride`'s runtime-step axes), `e` is
// rank 2 (`insert` adds an axis). The pair is the chelis#668 shape after
// chelis#1277 S2a/S2b moved the rank-raising meaning to `insert`.

fn insert_built(rhs: &str) -> String {
    format!(
        "module Repro.RankAgreement\n\
         sig f[n, u]: tensor[n, f32] -> tensor[u, f32]\n\
         def f(x) = {{\n\
           s = stride(x, 2i64)\n\
           e = insert(x, 0i32, 2i64)\n\
           {rhs}\n\
         }}\n"
    )
}

/// The same program with the rank-preserving `expand`. Under section 4.7.2's
/// one-shape rule both operands are rank 1, so these are positive controls.
fn expand_built(rhs: &str) -> String {
    format!(
        "module Repro.ExpandControl\n\
         sig f[n, u]: tensor[n, f32] -> tensor[u, f32]\n\
         def f(x) = {{\n\
           s = stride(x, 2i64)\n\
           e = expand(x, 0i32, 2i64)\n\
           {rhs}\n\
         }}\n"
    )
}

// ── Rank negatives: unification rejects, both operand orders ───────────────

/// The chelis#668 reproducer as it now reads: `insert` is the operation that
/// raises the rank, and unification alone refuses the pair.
///
/// Every row here is a DISPOSITION LOCK on the rejection plus the REGRESSION
/// assertion inside `agreed_diagnostics`.
#[test]
fn a_rank_divergent_insert_operand_is_refused_in_both_orders() {
    assert_rejects_with(
        &insert_built("add(s, e)"),
        "tensor rank mismatch: 1 dims vs 2 dims",
        "add(s, insert(...)) as let-bound operands",
    );
    assert_rejects_with(
        &insert_built("add(e, s)"),
        "tensor rank mismatch: 2 dims vs 1 dims",
        "add(insert(...), s) as let-bound operands",
    );
}

/// The same pair spelled inline, which is the form the retired validator
/// needed a structural derivation to see at all.
#[test]
fn an_inline_insert_operand_is_refused_in_both_orders() {
    assert_rejects_with(
        &insert_built("add(s, insert(x, 0i32, 2i64))"),
        "tensor rank mismatch: 1 dims vs 2 dims",
        "add(s, insert(x, 0i32, 2i64)) inline",
    );
    assert_rejects_with(
        &insert_built("add(insert(x, 0i32, 2i64), s)"),
        "tensor rank mismatch: 2 dims vs 1 dims",
        "add(insert(x, 0i32, 2i64), s) inline",
    );
}

/// Rank evidence survives an intervening rank-preserving call, in both
/// argument positions, whether the intermediate is let-bound or inline.
/// Unification carries it because the operand and the result share one
/// dimension row; nothing derives or forwards a rank fact.
#[test]
fn an_intervening_identity_call_does_not_erase_the_rank_disagreement() {
    for (rhs, needle, label) in [
        (
            "sn = neg(s)\n  add(sn, e)",
            "tensor rank mismatch: 1 dims vs 2 dims",
            "let-bound neg",
        ),
        (
            "ss = add(s, s)\n  add(ss, e)",
            "tensor rank mismatch: 1 dims vs 2 dims",
            "let-bound add",
        ),
        (
            "add(relu(neg(s)), e)",
            "tensor rank mismatch: 1 dims vs 2 dims",
            "nested inline unary chain",
        ),
        (
            "add(mul(s, s), e)",
            "tensor rank mismatch: 1 dims vs 2 dims",
            "inline binary",
        ),
        (
            "mul(relu(s), e)",
            "tensor rank mismatch: 1 dims vs 2 dims",
            "inline unary, divergent operand second",
        ),
        (
            "mul(e, relu(s))",
            "tensor rank mismatch: 2 dims vs 1 dims",
            "inline unary, divergent operand first",
        ),
        (
            "add(floor_div(s, s), e)",
            "tensor rank mismatch: 1 dims vs 2 dims",
            "inline floor_div, divergent operand second",
        ),
        (
            "add(e, floor_div(s, s))",
            "tensor rank mismatch: 2 dims vs 1 dims",
            "inline floor_div, divergent operand first",
        ),
        (
            "fd = floor_div(s, s)\n  add(fd, e)",
            "tensor rank mismatch: 1 dims vs 2 dims",
            "let-bound floor_div",
        ),
    ] {
        assert_rejects_with(&insert_built(rhs), needle, label);
    }
}

/// A comparison whose result is bound and then discarded still has its operand
/// ranks unified, because `cmplt_sig` gives both operands one dimension row
/// irrespective of what the result is used for.
#[test]
fn a_discarded_comparison_still_unifies_its_operand_ranks() {
    assert_rejects_with(
        &insert_built("ignored = eq(e, neg(s))\n  s"),
        "tensor rank mismatch: 2 dims vs 1 dims",
        "eq bound to a discarded name",
    );
}

/// A rank raised inside a user `def` with a declared rank-2 signature. This is
/// D1 row 1 of the completion design: the side channel never derived a rank
/// through a user def, and unification always did.
#[test]
fn a_declared_rank_two_user_def_result_is_refused_in_both_orders() {
    let program = |rhs: &str| {
        format!(
            "module Repro.UserDef\n\
             def g[a, b](y: tensor[a, b, f32]) -> tensor[a, b, f32] = y\n\
             sig f[n, u]: tensor[n, f32] -> tensor[u, f32]\n\
             def f(x) = {{\n\
               s = stride(x, 2i64)\n\
               e = insert(x, 0i32, 2i64)\n\
               {rhs}\n\
             }}\n"
        )
    };
    assert_rejects_with(
        &program("add(s, g(e))"),
        "tensor rank mismatch: 1 dims vs 2 dims",
        "add(s, g(e)) through a declared rank-2 def",
    );
    assert_rejects_with(
        &program("add(g(e), s)"),
        "tensor rank mismatch: 2 dims vs 1 dims",
        "add(g(e), s) through a declared rank-2 def",
    );
}

/// D1 row 2: a rank-0 reduction result beside its own rank-1 input. The
/// retired validator skipped rank-0 operands by construction; unification does
/// not, and reports the disagreement as `0 dims vs 1 dims`.
#[test]
fn a_rank_zero_reduction_result_is_refused_beside_its_rank_one_input() {
    assert_rejects_with(
        "module Repro.ReduceA\ndef f[n](x: tensor[n, f32]) = add(sum(x, 0i32), x)\n",
        "tensor rank mismatch: 0 dims vs 1 dims",
        "add(sum(x, 0), x)",
    );
    assert_rejects_with(
        "module Repro.ReduceB\ndef f[n](x: tensor[n, f32]) = add(x, sum(x, 0i32))\n",
        "tensor rank mismatch: 1 dims vs 0 dims",
        "add(x, sum(x, 0))",
    );
}

/// D1 row 3: a scalar beside a tensor is refused by `add` and by `max_elem`,
/// in both orders.
///
/// The rejection here is the type constructor's, not the rank comparison's,
/// which is why the pull request's mutation of `unify`'s length check leaves
/// these four rows green. They are locked because `spec/05-risc-primitives.md`
/// section 2.1 calls the scalar form of `max_elem`/`min_elem` "the rank-zero
/// instance of the tensor rule, not scalar/tensor broadcasting", and the
/// deletion must not be read as creating a broadcast permission. The seven
/// comparison identities are deliberately absent: they still carry the
/// chelis#1506 scalar rewrite, which is PR B's subject, not this file's.
#[test]
fn a_scalar_beside_a_tensor_is_refused_by_add_and_max_elem() {
    for (callee, first, second, needle) in [
        (
            "add",
            "1.5f32",
            "to_tensor([1.0f32, 2.0f32, 3.0f32])",
            "does not admit a scalar beside a tensor",
        ),
        (
            "add",
            "to_tensor([1.0f32, 2.0f32, 3.0f32])",
            "1.5f32",
            "does not admit a scalar beside a tensor",
        ),
        (
            "max_elem",
            "1.5f32",
            "to_tensor([1.0f32, 2.0f32, 3.0f32])",
            "does not admit a scalar beside a tensor",
        ),
        (
            "max_elem",
            "to_tensor([1.0f32, 2.0f32, 3.0f32])",
            "1.5f32",
            "does not admit a scalar beside a tensor",
        ),
    ] {
        assert_rejects_with(
            &format!("module Repro.Scalar\ndef f() = {callee}({first}, {second})\n"),
            needle,
            &format!("{callee}({first}, {second})"),
        );
    }
}

/// D1 row 4 and the second half of the deletion's effect: `where`'s own
/// procedural rule reports the rank-2 branch, and reports it once.
///
/// The count assertion is a REGRESSION assertion. On the pre-deletion tree the
/// normalizing ingress reported this call twice, once from `where`'s rule and
/// once from the identity-rank validator, which D5 records as a cascade.
#[test]
fn a_rank_two_where_branch_is_reported_once_by_wheres_own_rule() {
    // The condition is a bool-tensor parameter rather than `gt(x, 0.0f32)`.
    // That comparison was incidental scaffolding, and chelis#1506 makes a
    // scalar beside a tensor a type error, which would add a second diagnostic
    // and destroy the count this row asserts. The subject is unchanged: one
    // mismatched `where` call, one diagnostic.
    let source = "module Repro.Where\n\
                  def f[n, a, b](c: tensor[n, bool], x: tensor[n, f32], y: tensor[a, b, f32]) = \
                  where(c, x, y)\n";
    let diagnostics = agreed_diagnostics(source, "where with a rank-2 alternative");
    assert_eq!(
        diagnostics.len(),
        1,
        "a single mismatched `where` call must produce exactly one diagnostic; \
         a second one is the retired identity-rank validator duplicating a \
         verdict `where`'s own rule already reported (chelis#668 D5): \
         {diagnostics:?}"
    );
    assert!(
        diagnostics[0].contains("where expects cond/both branches to have matching tensor shapes"),
        "the surviving diagnostic must be `where`'s own: {diagnostics:?}"
    );
}

/// The rebinding negative: clearing a stale shape fact must not become "stop
/// checking rebound names". The rebound `a` is rank 1 and `b` is rank 2.
#[test]
fn a_rebinding_to_a_divergent_rank_is_still_refused() {
    assert_rejects_with(
        "module Repro.RebindNegative\n\
         def f[n](x: tensor[n, f32]) = {\n\
           a = insert(x, 0i32, 2i64)\n\
           a = stride(x, 2i64)\n\
           b = insert(x, 0i32, 2i64)\n\
           add(a, b)\n\
         }\n",
        "tensor rank mismatch: 1 dims vs 2 dims",
        "rebinding to a rank-1 value beside a rank-2 one",
    );
}

// ── Positive controls: unification accepts, both ingresses ─────────────────

/// Matching ranks, let-bound and inline, must stay accepted. Negative parity
/// for every rejection above.
#[test]
fn matching_ranks_are_accepted() {
    for (rhs, label) in [
        ("add(s, s)", "add(s, s)"),
        ("ss = add(s, s)\n  neg(ss)", "let-bound identity chain"),
        ("add(relu(neg(s)), mul(s, s))", "inline identity chain"),
    ] {
        assert_accepts(&insert_built(rhs), label);
    }
}

/// The chelis#668 reproducer built from `expand` is WELL TYPED.
///
/// `spec/04-type-system.md` section 4.7.2 gives `expand` one result shape and
/// `spec/05-risc-primitives.md` section 2.4 makes it the same-rank unit-extent
/// broadcast, so both operands are rank 1 and there is nothing to reject. The
/// loud outcome when the operand's extent at the axis is not 1 is section
/// 2.4.1's `Domain` trap, asserted by `runtime_extent_slice_b` on the C and
/// evaluator lanes and by `issue_668_elementwise_rank_honesty` on the driven
/// CLI path.
///
/// DISPOSITION LOCK. chelis#1277 S2b established this verdict by giving the
/// validator rank delta 0 for `expand`; PR A must not disturb it while
/// deleting the derivation.
#[test]
fn the_expand_built_reproducer_is_accepted_in_both_orders() {
    for (rhs, label) in [
        ("add(s, e)", "add(s, expand(...))"),
        ("add(e, s)", "add(expand(...), s)"),
        ("add(relu(neg(s)), e)", "expand through an identity chain"),
        (
            "ignored = eq(e, neg(s))\n  s",
            "expand under a discarded comparison",
        ),
    ] {
        assert_accepts(&expand_built(rhs), label);
    }
}

/// The stamped types behind the row above, which is what makes it a control
/// rather than a silence: `expand` publishes rank 1, and so does the `add`.
///
/// DISPOSITION LOCK on the stamp at both ingresses.
#[test]
fn the_expand_reproducer_stamps_rank_one_at_both_ingresses() {
    let exprs = deep(&expand_built("add(s, e)"));
    for (ingress, checked) in [
        ("check_ir_program", check_ir_program(&exprs)),
        ("check_typed_program", check_typed_program(&exprs)),
    ] {
        let checked = checked.unwrap_or_else(|result| {
            panic!(
                "{ingress}: the expand-built reproducer must check clean: {:?}",
                messages(&result.errors)
            )
        });
        let rendered = print_canonical(checked.annotated_exprs());
        assert!(
            rendered.contains("(t-tensor {} (d-lit {} 2) (t-prim {} f32))"),
            "{ingress}: `expand(x, 0i32, 2i64)` must stamp the same-rank \
             `tensor[2, f32]`, not an inserted axis:\n{rendered}"
        );
        assert!(
            !rendered.contains("(d-lit {} 2) (d-var"),
            "{ingress}: no node may stamp the rank-2 shape that belongs to \
             `insert`:\n{rendered}"
        );
    }
}

/// A user-defined parameter named `floor_div` shadows the builtin, so the call
/// takes the parameter's declared rank-2 result and agrees with `e`.
///
/// DISPOSITION LOCK. It guarded the retired validator's registry lookup
/// against lexical shadowing; it now guards ordinary inference's, which is the
/// only lookup left.
#[test]
fn a_lexically_shadowed_builtin_name_keeps_its_parameter_type() {
    assert_accepts(
        "module Repro.ShadowedIdentity\n\
         def lift[n](x: tensor[n, f32]) -> tensor[2, n, f32] = insert(x, 0i32, 2i64)\n\
         def apply(\n\
           floor_div: (tensor[n, f32] -> tensor[2, n, f32]),\n\
           x: tensor[n, f32],\n\
         ) -> tensor[2, n, f32] = {\n\
           s = stride(x, 2i64)\n\
           e = insert(x, 0i32, 2i64)\n\
           add(floor_div(s), e)\n\
         }\n",
        "a parameter shadowing the `floor_div` builtin",
    );
}

/// Rebinding a name to a value whose shape the validator cannot derive must
/// clear the name's previous fact rather than inherit it (chelis#668 round-6
/// F1). The rebound `a` is rank 2 like `b`.
///
/// DISPOSITION LOCK. The stale-fact bug it was written for was reachable only
/// through the rank comparison this pull request deletes, so the row can no
/// longer go red the way it originally did. It stays because the environment
/// it exercises still feeds the exact-shape `conv` validators, where a stale
/// entry would be the same defect with a different consumer.
#[test]
fn a_rebinding_does_not_inherit_the_previous_bindings_shape() {
    assert_accepts(
        "module Repro.Rebind\n\
         def f[n](x: tensor[n, f32]) = {\n\
           a = stride(x, 2i64)\n\
           a = relu(insert(x, 0i32, 2i64))\n\
           b = insert(x, 0i32, 2i64)\n\
           add(a, b)\n\
         }\n",
        "a rebinding to a non-derivable value",
    );
}

/// An authored dimension name that spells the retired carrier's old internal
/// prefix is just a name.
///
/// DISPOSITION LOCK, and its subject changed with the deletion. It used to
/// prove that a private `RankOnly` fact could not be forged from Deep syntax.
/// With no private carrier left there is nothing to forge, and what it now
/// locks is the weaker, still-worth-holding property that an unusual
/// `d-name` does not perturb the exact-shape `conv` derivation.
#[test]
fn an_authored_rank_only_prefix_dimension_is_an_ordinary_name() {
    assert_accepts(
        "module Repro.AuthoredRankName\n\
         def convolve(\n\
           x: tensor[__chelis_rank_only_axis_0, 3, 8, 8, f32],\n\
           k: tensor[8, 3, 3, 3, f32],\n\
         ) -> tensor[__chelis_rank_only_axis_0, 8, 6, 6, f32] = conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n",
        "an authored `__chelis_rank_only_axis_0` dimension",
    );
}
