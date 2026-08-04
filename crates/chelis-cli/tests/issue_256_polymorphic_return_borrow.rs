//! Issue #256: polymorphic-return borrow type-inference — dim variables
//! not concretized at `&` borrow site.
//!
//! Pre-fix: when a let-bound name's RHS was a polymorphic-return call
//! (e.g. `relu(prev_out)`) whose dim variables had not been pinned at
//! the point of inference, a subsequent `&` borrow of that name failed
//! with either:
//!
//!   - inference: `borrow requires tensor or tensor-carrying input, got ?N`
//!   - linearity: `borrowed arguments must be tensor or tensor-carrying values`
//!
//! depending on which classifier reached the unresolved `Type::Var(_)`
//! first. The school-style reproducer was a `pool → relu → &result →
//! conv2d` chain in a forward body whose intermediate had no declared
//! type and whose RHS sub-expressions (`reshape(...)`, `mean(...)`)
//! left their output type as a free variable until the surrounding
//! call's `&tensor[..]` parameter pinned it via unification.
//!
//! Fix in three parts:
//!
//! 1. `crates/chelis-types/src/infer.rs::borrow` arm: when the borrow
//!    inner resolves to `Type::Var(_)`, wrap as `Type::Ref(Type::Var(_))`
//!    instead of erroring AND record the variable in the substitution's
//!    deferred-borrow ledger. The surrounding call's expected parameter
//!    type then unifies the variable through `infer_app`.
//! 2. `crates/chelis-types/src/linearity.rs::expr_is_owned_or_borrow_linear`:
//!    accept a stamped `(t-var ...)` (and `(t-ref (t-var ...))`) as a
//!    borrow target. The inference layer (which sees the post-pinning
//!    subst) is the ultimate gate; a borrow of a genuinely non-tensor
//!    value still fails there.
//! 3. `crates/chelis-types/src/infer.rs::validate_deferred_borrow_vars`
//!    (round 2): after a def body's inference completes, re-check each
//!    recorded deferred-borrow variable against the *final* substitution.
//!    The deferral in part 1 is sound only when the variable is
//!    eventually pinned to a tensor or tensor carrier. A fully-polymorphic
//!    consumer (`consume_any[a](t: a)`) never pins it; this pass rejects
//!    the never-pinned and pinned-to-scalar cases that parts 1 and 2 would
//!    otherwise let through. See the round-2 soundness-lock tests below.
//! 4. `validate_deferred_borrow_vars` (round 3): the round-2 pass
//!    blanket-accepted any `Type::Adt`/`Type::Tuple` and deferred the
//!    tensor-carry decision to linearity. But part 2 had loosened
//!    linearity to accept a stale `(t-var ..)` stamp, and a deferred
//!    borrow that resolved to a *non*-carrying ADT/tuple kept that stamp
//!    — so both gates waved it through (a non-tensor `Config` record
//!    reached the borrow-erased backend). Round 3 classifies the
//!    aggregate against a registry-backed carrier set inside the gate
//!    itself: a tensor-carrying ADT/tuple is accepted, a non-carrying one
//!    rejected. See the round-3 deferred-aggregate tests below.
//!
//! See also:
//!   - chelis#154 — closed in 0.7.11. Different bug: ADT FIELD tensor
//!     classification. The `tensor_carrying_adts` fixed-point fix
//!     remains untouched here.
//!   - chelis#181 — closed in 0.7.x. Different bug: destructured-field
//!     type substitution in `match` arm. The `stamp_pattern_binding_types`
//!     fix remains untouched here.
//!
//! Negative test parity: a `&` borrow on a genuinely non-tensor binding
//! (a stack `int32`) must still be rejected. The error surfaces from
//! the inference-layer `borrow` arm before linearity runs.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// Run `chelis check <path>` and return the parsed JSON stdout.
fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn error_kinds(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect()
}

fn fmt_inplace(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

/// Issue #256 reproducer (post-fix expected behavior). The body's
/// pool helper has no return-type annotation; its body uses
/// `reshape(...)` whose output type can't be derived from runtime
/// list args, so the helper's return ends up as an unresolved
/// `Type::Var`. The chain `pool(x) → relu → &borrow → consume_t` then
/// trips the polymorphic-borrow gap pre-fix. The fix accepts the
/// borrow because the surrounding `consume_t` call site pins the
/// variable via its `&tensor[..]` parameter.
#[test]
fn polymorphic_return_borrow_chain_is_accepted_post_fix() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("poly_return_borrow_chain.ch");
    write_file(
        &fixture,
        "module Issue256PolyReturnChain\n\
         def pool_no_sig(x) = {\n\
           six = reshape(x, [cast(2, int64), cast(4, int64), cast(2, int64), cast(2, int64), cast(2, int64), cast(2, int64)])\n\
           perm = permute(six, cast(0, int32), cast(1, int32), cast(2, int32), cast(4, int32), cast(3, int32), cast(5, int32))\n\
           five = reshape(perm, [cast(2, int64), cast(4, int64), cast(2, int64), cast(2, int64), cast(4, int64)])\n\
           mean(five, cast(4, int32))\n\
         }\n\
         def consume_t(t: &tensor[a, c, h, w, f32]) -> bool = true\n\
         def forward(x: tensor[2, 4, 4, 4, f32]) -> bool = {\n\
           p = pool_no_sig(x)\n\
           r = relu(p)\n\
           consume_t(&r)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "polymorphic-return borrow chain must produce no errors post-fix; got {kinds:?}"
    );
}

/// The `id4` round-trip workaround the issue body documents. This is
/// the pre-fix coping pattern (a monomorphic identity helper that
/// re-binds the dim variables through its declared signature). After
/// the fix, `id4` is redundant but still a valid program. This test
/// pins both: the workaround keeps working, AND it produces no extra
/// errors. School's `svc_id4` / `svc_id2` retire on this contract.
#[test]
fn id4_roundtrip_workaround_still_works() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("id4_roundtrip_workaround.ch");
    write_file(
        &fixture,
        "module Issue256Id4Workaround\n\
         def pool_no_sig(x) = {\n\
           six = reshape(x, [cast(2, int64), cast(4, int64), cast(2, int64), cast(2, int64), cast(2, int64), cast(2, int64)])\n\
           perm = permute(six, cast(0, int32), cast(1, int32), cast(2, int32), cast(4, int32), cast(3, int32), cast(5, int32))\n\
           five = reshape(perm, [cast(2, int64), cast(4, int64), cast(2, int64), cast(2, int64), cast(4, int64)])\n\
           mean(five, cast(4, int32))\n\
         }\n\
         def consume_t(t: &tensor[a, c, h, w, f32]) -> bool = true\n\
         def id4[a, c, h, w](x: tensor[a, c, h, w, f32]) -> tensor[a, c, h, w, f32] = x\n\
         def forward(x: tensor[2, 4, 4, 4, f32]) -> bool = {\n\
           p = pool_no_sig(x)\n\
           r = relu(p)\n\
           r_bound = id4(r)\n\
           consume_t(&r_bound)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "id4 round-trip workaround must produce no errors; got {kinds:?}"
    );
}

/// Negative parity (CLAUDE.md "Negative Test Parity"): borrowing a
/// genuinely non-tensor binding must STILL be rejected with a
/// diagnostic naming the offending type. The fix does not silently
/// widen the borrow surface; the inference-layer `borrow` arm's
/// `_ => TypeMismatch` catch-all still fires for `Type::Prim`,
/// `Type::Unit`, `Type::Fn`, etc. (anything that is not
/// `Type::Var | Type::Tensor | Type::Adt | Type::Tuple | Type::Ref |
/// Type::Error`).
#[test]
fn borrow_of_non_tensor_is_still_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("non_tensor_borrow_rejected.ch");
    write_file(
        &fixture,
        "module Issue256NonTensor\n\
         def consume_t(t: &tensor[a, c, h, w, f32]) -> bool = true\n\
         def forward(x: int32) -> bool = consume_t(&x)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        !kinds.is_empty(),
        "borrow of int32 against a &tensor parameter must be rejected; got clean score {json}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| k == "TypeMismatch" || k == "InvalidBorrow"),
        "borrow of int32 must surface a TypeMismatch or InvalidBorrow; got {kinds:?}"
    );
    let messages: Vec<String> = json["errors"]
        .as_array()
        .expect("errors")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        messages.iter().any(|m| m.contains("int32")),
        "rejection diagnostic should mention `int32`; got {messages:?}"
    );
}

/// Regression guard: the simple `relu(x) → consume_t(&result)` chain
/// (without an intervening polymorphic-return helper) keeps working.
/// Pre-fix this already passed because `relu`'s output type is
/// concretely the same tensor as its input; this test exists to make
/// sure the fix doesn't regress that direct path.
#[test]
fn direct_relu_borrow_chain_keeps_working() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("direct_relu_borrow_chain.ch");
    write_file(
        &fixture,
        "module Issue256DirectRelu\n\
         def consume_t(t: &tensor[a, c, h, w, f32]) -> bool = true\n\
         def forward[a, c, h, w](x: tensor[a, c, h, w, f32]) -> bool = {\n\
           r = relu(x)\n\
           consume_t(&r)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "direct relu → borrow chain must keep working; got {kinds:?}"
    );
}

/// Regression guard for chelis#154 (closed in 0.7.11): borrowing a
/// tensor-carrying record ADT continues to work. This locks the
/// adjacent borrow-classification surface against the issue #256
/// changes.
#[test]
fn issue_154_tensor_carrying_record_adt_still_borrows() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("issue_154_regression.ch");
    write_file(
        &fixture,
        "module Issue256Issue154Regression\n\
         type BatchNormParams[n] =\n\
           | BatchNormParams { gamma: tensor[n, f32], beta: tensor[n, f32] }\n\
         sig borrow_params: &BatchNormParams[n] -> bool\n\
         def borrow_params(p) = true\n\
         def consume_params[n](p: BatchNormParams[n]) -> bool = borrow_params(&p)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "chelis#154 tensor-carrying record ADT borrow must keep working; got {kinds:?}"
    );
}

/// Extended negative parity: a non-tensor-carrying record ADT bound
/// against a fully polymorphic consumer must still be rejected. The
/// inference-layer `borrow` arm accepts `Type::Adt(_, _)` unconditionally
/// (the tensor-carry check lives at linearity), and a `consume_any[a](t: a)`
/// consumer instantiates `a` to `&Record`, so neither the inference
/// `borrow` arm nor the consumer's call site catches this — the
/// linearity layer's `expr_is_owned_or_borrow_linear` is the gate.
/// This test pins that the loosened linearity classifier (which now
/// also accepts a `(t-var ...)` to support issue #256) did not break
/// the `(t-adt ...)` rejection path for non-tensor records.
#[test]
fn borrow_of_non_tensor_record_with_polymorphic_consumer_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("non_tensor_record_poly_consumer.ch");
    write_file(
        &fixture,
        "module Issue256NonTensorRecordPolyConsumer\n\
         type Config = | Config { lr: f32, bs: int32 }\n\
         def consume_any[a](t: a) -> bool = true\n\
         def forward() -> bool = {\n\
           c = Config { lr: 0.1, bs: cast(32, int32) }\n\
           consume_any(&c)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        !kinds.is_empty(),
        "borrow of non-tensor record against a polymorphic consumer must be rejected; got clean score {json}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| k == "InvalidBorrow" || k == "TypeMismatch"),
        "borrow of non-tensor record must surface an InvalidBorrow or TypeMismatch; got {kinds:?}"
    );
}

/// Extended negative parity: a tuple of scalars borrowed against a
/// `&tensor[..]` parameter must still be rejected. The inference
/// borrow arm accepts `Type::Tuple(_)` unconditionally; the rejection
/// surfaces from the call-site type mismatch (`(int32, int32)` vs
/// `tensor[..]`). This locks the adjacent path the linearity loosening
/// does not affect, so a future refactor that moves the tuple-of-scalars
/// classification to linearity does not silently regress.
#[test]
fn borrow_of_scalar_tuple_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("scalar_tuple_borrow_rejected.ch");
    write_file(
        &fixture,
        "module Issue256ScalarTupleBorrowRejected\n\
         def consume_t(t: &tensor[a, c, h, w, f32]) -> bool = true\n\
         def forward() -> bool = {\n\
           pair = (cast(1, int32), cast(2, int32))\n\
           consume_t(&pair)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        !kinds.is_empty(),
        "borrow of (int32, int32) against a &tensor parameter must be rejected; got clean score {json}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| k == "TypeMismatch" || k == "InvalidBorrow"),
        "borrow of scalar tuple must surface a TypeMismatch or InvalidBorrow; got {kinds:?}"
    );
}

/// Negative parity for the inference-layer catch-all: a `&unit` borrow
/// (or any `Type::Unit` inner) must still hit the inference `borrow` arm
/// `_ => TypeMismatch` and never reach linearity. This locks the
/// boundary between "deferred classification" (accepted Type::Var) and
/// "concretely-non-tensor" (Type::Prim, Type::Unit, Type::Fn).
#[test]
fn borrow_of_unit_is_rejected_at_inference() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("unit_borrow_rejected.ch");
    write_file(
        &fixture,
        "module Issue256UnitBorrowRejected\n\
         def consume_t(t: &tensor[a, c, h, w, f32]) -> bool = true\n\
         def forward() -> bool = {\n\
           u = ()\n\
           consume_t(&u)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        !kinds.is_empty(),
        "borrow of `()` against a &tensor parameter must be rejected; got clean score {json}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| k == "TypeMismatch" || k == "InvalidBorrow"),
        "borrow of `()` must surface a TypeMismatch or InvalidBorrow; got {kinds:?}"
    );
}

/// Round-2 soundness lock (issue #256): a genuinely *free* type variable
/// borrowed against a fully-polymorphic consumer must be REJECTED. This
/// is the exploit the first review pass argued was "only reachable via
/// non-terminating programs"; it is not. `use_it[a](seed: a)` is a
/// terminating, non-recursive def, yet `seed` is a genuinely-unconstrained
/// `Type::Var` at the `&v` borrow site. The consumer `consume_any[a](t: a)`
/// unifies its own parameter to `&a` WITHOUT ever pinning `a` to a tensor,
/// so the deferral the borrow arm relies on never resolves to a tensor.
///
/// Pre-round-2 this passed with a perfect score (a non-tensor value
/// reaching the linearity-erased borrow surface). The
/// `validate_deferred_borrow_vars` pass drains the borrow arm's deferral
/// ledger after the def body's inference completes and rejects any
/// recorded variable that did not become a tensor or tensor carrier.
///
/// The diagnostic is the same `borrow requires tensor or tensor-carrying
/// input, got ?N` the borrow arm emits for a concretely-non-tensor inner;
/// here `?N` is the unresolved variable.
#[test]
fn borrow_of_free_var_against_polymorphic_consumer_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("free_var_poly_consumer_rejected.ch");
    write_file(
        &fixture,
        "module Issue256FreeVarPolyConsumer\n\
         def consume_any[a](t: a) -> bool = true\n\
         def use_it[a](seed: a) -> bool = {\n\
           v = seed\n\
           consume_any(&v)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        !kinds.is_empty(),
        "borrow of a genuinely-free type variable against a polymorphic \
         consumer must be rejected; got clean score {json}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| k == "TypeMismatch" || k == "InvalidBorrow"),
        "free-var borrow must surface a TypeMismatch or InvalidBorrow; got {kinds:?}"
    );
}

/// Round-2 soundness lock, end-to-end variant: the same free-var borrow
/// reached through a CONCRETE caller that instantiates the polymorphic
/// parameter at a non-tensor type (`int32`). This is the fully-terminating
/// program a user could actually write: no recursion, a concrete `main`,
/// and a non-tensor value flowing into a `&` borrow. It must be rejected
/// so a non-tensor never reaches the (borrow-type-erased) backend.
#[test]
fn borrow_of_free_var_instantiated_at_int32_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("free_var_int32_instantiation_rejected.ch");
    write_file(
        &fixture,
        "module Issue256FreeVarInt32\n\
         def consume_any[a](t: a) -> bool = true\n\
         def use_it[a](seed: a) -> bool = {\n\
           v = seed\n\
           consume_any(&v)\n\
         }\n\
         def main() -> bool = use_it(cast(42, int32))\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        !kinds.is_empty(),
        "a terminating program borrowing a non-tensor via a polymorphic \
         consumer must be rejected; got clean score {json}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| k == "TypeMismatch" || k == "InvalidBorrow"),
        "free-var-at-int32 borrow must surface a TypeMismatch or InvalidBorrow; got {kinds:?}"
    );
}

/// Round-2 positive lock: the sound deferral must STILL be accepted. A
/// polymorphic parameter borrowed against a `&tensor[..]` consumer pins
/// the variable to a tensor through unification, so the deferral resolves
/// soundly. This is the counterpart to the rejection tests above and
/// guards `validate_deferred_borrow_vars` against over-rejecting the
/// legitimate issue #256 pattern (the variable DOES become a tensor).
#[test]
fn borrow_of_poly_param_pinned_to_tensor_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("poly_param_pinned_tensor_accepted.ch");
    write_file(
        &fixture,
        "module Issue256PolyParamPinned\n\
         def consume_t(t: &tensor[a, c, h, w, f32]) -> bool = true\n\
         def use_it[a, c, h, w](seed: tensor[a, c, h, w, f32]) -> bool = {\n\
           v = relu(seed)\n\
           consume_t(&v)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "a deferred borrow that resolves to a tensor must be accepted; got {kinds:?}"
    );
}

/// Round-3 soundness lock (issue #256): a borrow whose inner is a free
/// `Type::Var` at the borrow site (so it takes the *deferred* path) but
/// that later resolves to a *non-tensor-carrying* record ADT must be
/// REJECTED. This is the gap round 2 left open: the gate accepted any
/// `Type::Adt` and delegated the carry decision to linearity, but part 2
/// had loosened linearity to accept the stale `(t-var ..)` stamp this
/// shape carries — so both gates waved a non-tensor `Config` through to
/// the borrow-erased backend.
///
/// Distinct from `borrow_of_non_tensor_record_with_polymorphic_consumer_is_rejected`:
/// there the record is *concrete* at the borrow site (`c = Config {..}`),
/// so the borrow arm sees `Type::Adt` directly and linearity rejects the
/// `(t-adt ..)` stamp. Here the borrow inner is a genuine `Type::Var`
/// (`v = seed`), recorded in the deferred-borrow ledger; only the
/// consumer's `&Config` parameter pins it — after the borrow check ran.
/// The fix classifies the resolved aggregate against the registry carrier
/// set inside `validate_deferred_borrow_vars` itself.
#[test]
fn borrow_of_deferred_var_resolving_to_non_carrying_adt_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("deferred_non_carrying_adt_rejected.ch");
    write_file(
        &fixture,
        "module Issue256DeferredNonCarryingAdt\n\
         type Config = | Config { lr: f32, bs: int32 }\n\
         sig consume_config: &Config -> bool\n\
         def consume_config(c) = true\n\
         def use_it[a](seed: a) -> bool = {\n\
           v = seed\n\
           consume_config(&v)\n\
         }\n\
         def main() -> bool = use_it(Config { lr: 0.1, bs: cast(32, int32) })\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        !kinds.is_empty(),
        "a deferred borrow that resolves to a non-tensor-carrying ADT must \
         be rejected; got clean score {json}"
    );
    assert!(
        kinds
            .iter()
            .any(|k| k == "TypeMismatch" || k == "InvalidBorrow"),
        "deferred non-carrying ADT borrow must surface a TypeMismatch or \
         InvalidBorrow; got {kinds:?}"
    );
}

/// Round-3 positive lock (issue #256): the counterpart to the rejection
/// test above. A deferred borrow that resolves to a *tensor-carrying*
/// record ADT (`BatchNormParams { gamma: tensor[..], beta: tensor[..] }`,
/// the chelis#154 carrier shape) must STILL be accepted — the round-3
/// carrier-set classification must not over-reject a legitimate carrier
/// reached through the deferred path. Guards against the fix degrading
/// into "reject every deferred aggregate."
#[test]
fn borrow_of_deferred_var_resolving_to_carrier_adt_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("deferred_carrier_adt_accepted.ch");
    write_file(
        &fixture,
        "module Issue256DeferredCarrierAdt\n\
         type BatchNormParams[n] =\n\
           | BatchNormParams { gamma: tensor[n, f32], beta: tensor[n, f32] }\n\
         sig consume_bnp: &BatchNormParams[n] -> bool\n\
         def consume_bnp(p) = true\n\
         def use_it[a](seed: a) -> bool = {\n\
           v = seed\n\
           consume_bnp(&v)\n\
         }\n\
         def main[n](p: BatchNormParams[n]) -> bool = use_it(p)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "a deferred borrow that resolves to a tensor-carrying ADT must be \
         accepted; got {kinds:?}"
    );
}

/// Round-3 transitive-carrier lock: the carrier-set classification must
/// follow ADT field chains. `Outer { inner: Inner }` where `Inner { w:
/// tensor[..] }` carries a tensor only transitively. A deferred borrow
/// resolving to `Outer` must be accepted, proving the gate runs the same
/// fixed-point closure linearity does rather than a one-level field check.
#[test]
fn borrow_of_deferred_var_resolving_to_transitive_carrier_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("deferred_transitive_carrier_accepted.ch");
    write_file(
        &fixture,
        "module Issue256DeferredTransitiveCarrier\n\
         type Inner[n] = | Inner { w: tensor[n, f32] }\n\
         type Outer[n] = | Outer { inner: Inner[n] }\n\
         sig consume_outer: &Outer[n] -> bool\n\
         def consume_outer(o) = true\n\
         def use_it[a](seed: a) -> bool = {\n\
           v = seed\n\
           consume_outer(&v)\n\
         }\n\
         def main[n](o: Outer[n]) -> bool = use_it(o)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "a deferred borrow that resolves to a transitively-tensor-carrying \
         ADT must be accepted; got {kinds:?}"
    );
}
