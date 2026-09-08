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

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
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
/// pool helper has no return-type annotation; its input annotation keeps this
/// #256 fixture outside [04-INF-1]'s unannotated-shape-lambda class. Its body uses
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
         def pool_no_sig(x: tensor[2, 4, 4, 4, f32]) = {\n\
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
         def pool_no_sig(x: tensor[2, 4, 4, 4, f32]) = {\n\
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
/// input` the borrow arm emits for a concretely-non-tensor inner. Since
/// chelis#260 Site 2 it renders the declared spelling of the variable, so
/// this fixture reports ``got `a` `` rather than the `got ?N` it emitted
/// when this test was written. The assertion is on the error kind, so the
/// wording is documented here rather than pinned.
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
    // [04-INF-6] now also reports the `[a]` header, and that error is a
    // `TypeMismatch` too, so a kind-only assertion can no longer tell this
    // row's subject from an unrelated rejection. Assert the subject's own
    // diagnostic text, and record which kind carries it.
    let messages = error_messages(&json);
    assert!(
        messages.iter().any(
            |m| m.contains("borrow requires tensor or tensor-carrying input")
                && m.contains("Config")
        ),
        "the carrier-set classification must reject `Config` by name, not \
         merely leave some rejection behind; got {messages:?}"
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
///
/// **DISPOSITION LOCK for the concrete-carrier case, not a regression test for
/// the deferred path.** The header was `def use_it[a](seed: a) -> bool`, a
/// polymorphic binder the body then pinned through the callee's signature.
/// [04-INF-6] rejects that correctly, so the header now names the concrete
/// carrier the callee already pins.
///
/// That relabel is not cosmetic. A borrow reaches
/// `validate_deferred_borrow_vars` only when its inner type is an unresolved
/// `Type::Var` at the borrow arm; a concrete header defers nothing, so that
/// validator returns at its first line. This row locks that a borrow of a
/// tensor-carrying ADT is accepted, and nothing more. The deferred
/// classification's positive side is carried by the chelis#1589 rows at the
/// end of this file. Those four rows are
/// `borrow_of_inferred_param_resolving_to_carrier_is_accepted`,
/// `borrow_of_unannotated_def_resolving_to_carrier_is_accepted`,
/// `borrow_of_lambda_param_resolving_to_carrier_is_accepted` and
/// `borrow_of_let_alias_of_inferred_param_is_accepted`, each of which reaches
/// the validator's `sound=true` branch. The negative twin below reaches the
/// reject branch.
#[test]
fn borrow_of_concrete_carrier_adt_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("deferred_carrier_adt_accepted.ch");
    write_file(
        &fixture,
        "module Issue256DeferredCarrierAdt\n\
         type BatchNormParams[n] =\n\
           | BatchNormParams { gamma: tensor[n, f32], beta: tensor[n, f32] }\n\
         sig consume_bnp: &BatchNormParams[n] -> bool\n\
         def consume_bnp(p) = true\n\
         def use_it[n](seed: BatchNormParams[n]) -> bool = {\n\
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
///
/// **DISPOSITION LOCK for the concrete-carrier case, not a regression test for
/// the deferred path.** The header was `def use_it[a](seed: a) -> bool`, a
/// polymorphic binder the body then pinned through the callee's signature.
/// [04-INF-6] rejects that correctly, so the header now names the concrete
/// carrier the callee already pins.
///
/// That relabel is not cosmetic. A borrow reaches
/// `validate_deferred_borrow_vars` only when its inner type is an unresolved
/// `Type::Var` at the borrow arm; a concrete header defers nothing, so that
/// validator returns at its first line. This row locks that a borrow of a
/// tensor-carrying ADT is accepted, and nothing more. The deferred
/// classification's positive side is carried by the chelis#1589 rows at the
/// end of this file. Those four rows are
/// `borrow_of_inferred_param_resolving_to_carrier_is_accepted`,
/// `borrow_of_unannotated_def_resolving_to_carrier_is_accepted`,
/// `borrow_of_lambda_param_resolving_to_carrier_is_accepted` and
/// `borrow_of_let_alias_of_inferred_param_is_accepted`, each of which reaches
/// the validator's `sound=true` branch. The negative twin below reaches the
/// reject branch.
#[test]
fn borrow_of_concrete_transitive_carrier_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("deferred_transitive_carrier_accepted.ch");
    write_file(
        &fixture,
        "module Issue256DeferredTransitiveCarrier\n\
         type Inner[n] = | Inner { w: tensor[n, f32] }\n\
         type Outer[n] = | Outer { inner: Inner[n] }\n\
         sig consume_outer: &Outer[n] -> bool\n\
         def consume_outer(o) = true\n\
         def use_it[n](seed: Outer[n]) -> bool = {\n\
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

// ---------------------------------------------------------------------------
// chelis#1589: the deferred classification's accept branch, reached honestly.
//
// Rows C, E, F and G below are the four headers that reach
// `validate_deferred_borrow_vars` on its `sound=true` branch. Each was
// rejected before this change with score 0.80 and a single `InvalidBorrow`,
// "borrowed arguments must be tensor or tensor-carrying values", raised by
// linearity's `check_borrow_arg` after inference had already resolved the
// inner to `BatchNormParams[n]`. `spec/04-type-system.md` §8.2 says the inner
// "must be — or must ultimately resolve to" a carrier, so the rejection was a
// spec-compliance defect: linearity failed closed on absent type information
// rather than reading the resolved `&T` the annotate pass stamps on the
// `borrow` node.
//
// These four rows carry the positive side of the classification that
// `borrow_of_concrete_carrier_adt_is_accepted` and
// `borrow_of_concrete_transitive_carrier_is_accepted` cannot: those two use a
// concrete header, which defers nothing.
// ---------------------------------------------------------------------------

/// The shared prelude for the chelis#1589 rows: a tensor-carrying record ADT
/// and a consumer whose parameter is `&BatchNormParams[n]`.
const BNP_PRELUDE: &str = "type BatchNormParams[n] =\n  \
     | BatchNormParams { gamma: tensor[n, f32], beta: tensor[n, f32] }\n\
     sig consume_bnp: &BatchNormParams[n] -> bool\n\
     def consume_bnp(p) = true\n";

/// Write a `.ch` fixture with the `BNP_PRELUDE`, canonicalize it, and check it.
fn check_bnp_program(dir: &Path, module: &str, body: &str) -> Value {
    let fixture = dir.join(format!("{module}.ch"));
    write_file(&fixture, &format!("module {module}\n{BNP_PRELUDE}{body}"));
    fmt_inplace(&fixture);
    run_check(&fixture)
}

/// Lower a `.ch` fixture to canonical Deep and check the `.dp` instead, so a
/// row's verdict can be compared across both checker ingresses.
fn check_via_deep_ingress(source: &Path) -> Value {
    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", source.to_str().unwrap()])
        .output()
        .expect("run chelis deep");
    assert!(
        deep.status.success(),
        "chelis deep must succeed: {}",
        String::from_utf8_lossy(&deep.stderr)
    );
    let lowered = source.with_extension("dp");
    fs::write(&lowered, &deep.stdout).expect("write .dp");
    run_check(&lowered)
}

/// chelis#1589 row C: **regression test**. An inferred parameter with a
/// declared return type. `seed` has no annotation, so the borrow arm sees
/// `Type::Var`, defers, and the call to `consume_bnp` pins it to
/// `BatchNormParams[n]`; `validate_deferred_borrow_vars` then resolves it
/// `sound=true`.
///
/// Red before this change: score 0.80, one `InvalidBorrow`, "borrowed
/// arguments must be tensor or tensor-carrying values". Proven by reverting
/// `crates/chelis-types/src/linearity.rs` to the parent commit and rerunning
/// this test file.
#[test]
fn borrow_of_inferred_param_resolving_to_carrier_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let json = check_bnp_program(
        dir.path(),
        "Issue1589InferredParam",
        "def use_it(seed) -> bool = consume_bnp(&seed)\n",
    );
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "a borrow of an inferred parameter that inference resolves to a \
         tensor-carrying ADT must be accepted (spec/04 §8.2); got {kinds:?}"
    );
}

/// chelis#1589 row C at the Deep ingress: **regression test**. The verdict
/// must not depend on the entry point, so this asserts equality of score and
/// error list between `chelis check f.ch` and `chelis check f.dp`, not merely
/// that each is clean. Row C is the row whose verdict this change moves.
///
/// Red before this change at both ingresses (0.80 `InvalidBorrow` each), so
/// the equality assertion held while the shared verdict was wrong; the score
/// assertion is what turns red.
#[test]
fn borrow_of_inferred_param_agrees_across_both_ingresses() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("Issue1589InferredParamDeep.ch");
    write_file(
        &fixture,
        &format!(
            "module Issue1589InferredParamDeep\n{BNP_PRELUDE}\
             def use_it(seed) -> bool = consume_bnp(&seed)\n"
        ),
    );
    fmt_inplace(&fixture);

    let surf = run_check(&fixture);
    let deep = check_via_deep_ingress(&fixture);
    assert_eq!(
        surf["score"], deep["score"],
        "Surf and Deep ingresses must agree on score: surf={surf}, deep={deep}"
    );
    assert_eq!(
        error_messages(&surf),
        error_messages(&deep),
        "Surf and Deep ingresses must agree on the error list"
    );
    assert_eq!(surf["score"], 1, "perfect-score contract: {surf}");
    assert!(
        error_kinds(&surf).is_empty(),
        "both ingresses must accept the inferred-parameter borrow; got {:?}",
        error_kinds(&surf)
    );
}

/// chelis#1589 row E: **regression test**. The same borrow with no annotation
/// anywhere on `use_it` — neither a parameter type nor a return type. The
/// return type is not what makes the inner visible, so removing it must not
/// change the verdict.
///
/// Red before this change: score 0.80, one `InvalidBorrow`.
#[test]
fn borrow_of_unannotated_def_resolving_to_carrier_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let json = check_bnp_program(
        dir.path(),
        "Issue1589NoAnnotation",
        "def use_it(seed) = consume_bnp(&seed)\n",
    );
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "a borrow inside a def with no annotation at all must be accepted \
         when the callee pins the inner to a carrier; got {kinds:?}"
    );
}

/// chelis#1589 row F: **regression test**. The lambda-parameter form. `v` is a
/// lambda parameter, which carries no declared type in the linear scope for
/// the same reason an inference-hole `def` parameter does, so the borrow inner
/// is equally invisible to `expr_type`.
///
/// Red before this change: score 0.80, one `InvalidBorrow`.
#[test]
fn borrow_of_lambda_param_resolving_to_carrier_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let json = check_bnp_program(
        dir.path(),
        "Issue1589LambdaParam",
        "def use_it[n](p: BatchNormParams[n]) -> bool = (fn (v) -> consume_bnp(&v))(p)\n",
    );
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "a borrow of a lambda parameter that resolves to a carrier must be \
         accepted; got {kinds:?}"
    );
}

/// chelis#1589 row G: **regression test**. The `let`-alias form. `v` takes its
/// type from an inferred parameter, so the alias inherits the invisibility.
/// This is the shape the two concrete-carrier locks above use, with the
/// concrete header replaced by an inferred one.
///
/// Red before this change: score 0.80, one `InvalidBorrow`.
#[test]
fn borrow_of_let_alias_of_inferred_param_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let json = check_bnp_program(
        dir.path(),
        "Issue1589LetAlias",
        "def use_it(seed) -> bool = {\n  v = seed\n  consume_bnp(&v)\n}\n",
    );
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "a borrow of a let alias of an inferred parameter must be accepted \
         when the alias resolves to a carrier; got {kinds:?}"
    );
}

/// chelis#1589 row J: **disposition lock**. The never-pinned program with an
/// inferred parameter and no authored binder anywhere. Its job is to hold
/// `validate_deferred_borrow_vars`'s reject branch on the one header shape the
/// suite did not cover: the three existing reject rows all carry an authored
/// `[a]` binder somewhere, so none of them proves the validator still fires
/// when the deferral originates from a synthesized inference hole.
///
/// This is the row that shows the chelis#1589 repair does not widen the
/// deferred path. It reaches linearity's new fallback only if the validator
/// stops rejecting first, and it must not: a value never pinned to a tensor
/// reaching the borrow-erased backend is the chelis#256 round-2 unsoundness.
///
/// Green in both states. The subject in the message is an internal type-variable
/// identity, so this asserts the kind and the message prefix, never the number.
#[test]
fn borrow_of_never_pinned_inferred_param_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("Issue1589NeverPinned.ch");
    write_file(
        &fixture,
        "module Issue1589NeverPinned\n\
         def consume_any[a](t: a) -> bool = true\n\
         def use_it(seed) -> bool = {\n\
           v = seed\n\
           consume_any(&v)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "TypeMismatch"),
        "a never-pinned deferred borrow must be rejected by the deferred \
         validator, not accepted; got {kinds:?} in {json}"
    );
    assert!(
        error_messages(&json)
            .iter()
            .any(|m| m.starts_with("borrow requires tensor or tensor-carrying input, got ")),
        "the rejection must name the unresolved subject; got {:?}",
        error_messages(&json)
    );
}

/// chelis#1589 negative parity: **disposition lock**. The failure twin of rows
/// C/E/F/G — the same inferred-parameter route, with the consumer's parameter
/// changed to a record that carries no tensor. It must stay rejected.
///
/// Which gate rejects it is measured, not assumed, and it is not linearity:
/// `validate_deferred_borrow_vars` resolves the deferred variable to `Config`,
/// classifies it `sound=false`, and reports `TypeMismatch`, "borrow requires
/// tensor or tensor-carrying input, got Config". Type analysis therefore fails
/// and `check_linearity` never runs, so this row cannot exercise the carrier
/// test in linearity's new fallback; the validator's round-3 comment says as
/// much ("linearity's loosened classifier can no longer be relied on to catch
/// it"). Recorded because the chelis#1589 design expected this row to be a
/// regression test for the fallback, and a mutation receipt showed it is not.
///
/// Green in both states. Its job is to hold the boundary of the acceptance the
/// repair adds: an inferred parameter is accepted because it resolves to a
/// carrier, not because it is inferred.
#[test]
fn borrow_of_inferred_param_resolving_to_non_carrier_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("Issue1589InferredNonCarrier.ch");
    write_file(
        &fixture,
        "module Issue1589InferredNonCarrier\n\
         type Config = | Config { lr: f32 }\n\
         sig consume_cfg: &Config -> bool\n\
         def consume_cfg(c) = true\n\
         def use_it(seed) -> bool = consume_cfg(&seed)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    assert!(
        error_kinds(&json).iter().any(|k| k == "TypeMismatch"),
        "a borrow of an inferred parameter that resolves to a non-carrying \
         record must stay rejected; got clean {json}"
    );
    assert!(
        error_messages(&json)
            .iter()
            .any(|m| m == "borrow requires tensor or tensor-carrying input, got Config"),
        "the rejection must come from the deferred validator and name \
         `Config`; got {:?}",
        error_messages(&json)
    );
}

/// chelis#1589 row L: **disposition lock** on `spec/04-type-system.md` §8.2's
/// own worked example, written out. `v = relu(seed)` where `seed`'s dimension
/// variables are bound by the enclosing header, borrowed against a
/// `&tensor[a, c, h, w, f32]` parameter.
///
/// §8.2 used to offer this shape as its illustration of the deferral. It is
/// not one: a tensor whose dimension variables are unresolved is still
/// `Type::Tensor`, which the borrow arm matches and classifies immediately;
/// only an unknown outer type constructor takes the defer branch. This change
/// corrects that paragraph, and this row locks the verdict the corrected
/// sentence predicts — accepted, with the classification decided at the borrow
/// site rather than at the drain.
///
/// Green in both states, and the verdict alone does not distinguish the two
/// mechanisms: a deferral that resolved soundly would also be accepted. That
/// `validator-drain count=0` for this program was established by probe
/// instrumentation during the chelis#1589 measurement, not by this assertion.
#[test]
fn borrow_of_relu_result_with_unresolved_dims_is_accepted() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("Issue1589UnresolvedDims.ch");
    write_file(
        &fixture,
        "module Issue1589UnresolvedDims\n\
         sig consume_t: &tensor[a, c, h, w, f32] -> bool\n\
         def consume_t(t) = true\n\
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
        "a borrow of a tensor with unresolved dimension variables must be \
         accepted; got {kinds:?}"
    );
}
