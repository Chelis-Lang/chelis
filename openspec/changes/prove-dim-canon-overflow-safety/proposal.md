# Proposal: prove-dim-canon-overflow-safety

## Why

Tracked as chelis#888, filed unverified: the collision below is hand-traced, not executed, and reachability from a source program is not established. Group 1 of the tasks runs the reproduction, and group 2 settles reachability. If the reproduction passes, this change closes with chelis#888.

Three functions traverse the same `DimExpr` tree in `crates/chelis-ir/src/dag.rs`, and each one handles integer overflow differently.

| Function | Line | Multiplication | Behavior on overflow |
| --- | --- | --- | --- |
| `DimExpr::evaluate` | `dag.rs:178` | `lhs * rhs` | panics in debug, wraps in release |
| `DimExpr::as_concrete` | `dag.rs:203` | `lhs * rhs` | panics in debug, wraps in release |
| canonicalizer helpers | `dag.rs:320`, `372`, `390`, `392` | `saturating_mul` | clamps to `usize::MAX` |

The third row is the one that matters, because saturation is not injective. Two products with different true values that both exceed `usize::MAX` clamp to the same `usize::MAX`, and the canonicalizer then emits the **same key** for expressions denoting different sizes.

That is not merely an overflow. It is a false entry in the equality relation that `capacity_fits` consults:

```rust
// crates/chelis-backend-c/src/memory.rs:315-324
match (slot_concrete, req_concrete) {
    (Some(slot_elems), Some(req_elems)) => slot_elems >= req_elems,
    _ => slot_key == req_key,
}
```

The concrete branch is unaffected, so the exposure is the mixed case: an expression containing a symbol makes `as_concrete` return `None`, control falls to `slot_key == req_key`, and the key's concrete factor is the one that saturated. `crates/chelis-backend-hip/src/memory.rs:360-362` repeats the pattern.

The rustdoc at `dag.rs:147-161` states the contract as unconditional — the key "is mathematically equivalent for positive-integer-valued dim expressions" — and records no bound on those integers. So the codebase relies on an assumption it does not state and does not check.

**What is not yet known is whether a Chelis source program can reach it.** Dimension values originate in tensor shapes, and a shape large enough to saturate a `usize` product is not allocatable. But memory planning runs at compile time, long before allocation, so unallocatable is not the same as unreachable. Establishing which of the two applies is part of this change, and it determines the severity rather than whether the fix is warranted: an equality relation that is unsound in a corner is worth correcting even when the corner is hard to reach, because the correction is small and the reasoning about it is not.

`add-dim-canon-property-tests` deliberately excludes this regime. Its generator is bounded below the overflow threshold because its own oracle, `evaluate`, panics there. Property testing therefore cannot find this defect by construction, which is precisely why it needs its own change.

## What Changes

- `crates/chelis-ir/src/dag.rs` — the canonicalizer's concrete-factor multiplication SHALL stop saturating. On overflow it SHALL leave the product unfolded and structural rather than collapsing it to a clamped constant. Distinct expressions then keep distinct keys, and the equality relation stays sound. This is the conservative direction: the key becomes less complete in a regime where it is currently unsound, and completeness was never promised.
- The key output SHALL be unchanged for every input that does not overflow. This is a hard constraint, not an aspiration: the existing example tests and the property suite are the evidence.
- The three overflow policies SHALL be reconciled and documented in the rustdoc, so a later reader does not have to derive them from three call sites.
- A regression test constructing the colliding keys directly, added to `crates/chelis-ir/tests/dim_canon_adversarial.rs` in the style already used there.
- **A Verus spike, with an explicit kill criterion.** The candidate proof obligations are narrow and are the only parts of the span expressible in Verus's subset: that `usize_gcd` (`dag.rs:298-306`) terminates and returns the true greatest common divisor, and that the replacement multiplication cannot produce a folded constant that differs from the true product. If the spike does not discharge both within its declared budget, the change ships the tested fix and records the negative result. The fix does not depend on the spike succeeding.
- New `docs/dim_arithmetic.md` — the overflow policy, why the key is deliberately incomplete above the threshold, and the outcome of the Verus decision either way.

### Non-Goals

- **No general Verus adoption.** This change evaluates Verus against two named obligations on one leaf function. It proposes no verified-code programme, no second verified module, and no policy that future code be verified.
- **No change to the pinned toolchain for other jobs.** `rust-toolchain.toml` governs every invocation in this repository. Verus requires its own toolchain, so any adoption SHALL confine it to a dedicated lane and SHALL NOT alter what the gate jobs install. If that confinement is not achievable, the spike fails its criterion.
- **No behavior change outside the overflow regime.** For every input whose products fit in `usize`, `normalized_key` SHALL return exactly what it returns today.
- **No change to `evaluate` or `as_concrete` semantics.** Their unchecked multiplication is documented by this change, not altered. Changing an evaluator's panic behavior affects the IR interpreter that serves as the bit-parity oracle for the C backend (chelis#770), and that belongs in a change reviewed against that oracle.
- **No `spec/**` change.** No normative specification is modified. The requirements below make explicit a contract the `DimExprKey` rustdoc already asserts unconditionally.

### Behavior change statement

Unlike the two sibling changes, this one **does change compiler behavior**. Memory planning in `chelis-backend-c` and `chelis-backend-hip` will reach a different slot-reuse decision for programs whose dimension products overflow `usize`. In that regime the current decision is unsound, so the change replaces a wrong answer with a conservative one: two expressions that today compare equal by accident will compare unequal, and a buffer slot that today is reused will not be. Runtime, CLI, package, and generated-code behavior are otherwise untouched, and no program outside that regime is affected.

## Capabilities

### New Capabilities

- `dim-arith-overflow-policy`: the overflow policy for dimension arithmetic, the requirement that key equality remain a sound equivalence relation, the requirement that non-overflowing behavior be preserved exactly, and the conditions under which mechanically verified arithmetic is adopted or declined.

### Modified Capabilities

None. No existing requirement changes.

## Impact

- `crates/chelis-ir/src/dag.rs` — canonicalizer multiplication and the `DimExprKey` rustdoc. The only source change in the non-spike path.
- `crates/chelis-ir/tests/dim_canon_adversarial.rs` — regression test for the collision.
- `crates/chelis-ir/tests/dim_canonicalization.rs`, `crates/chelis-ir/src/dag.rs:2374-2467` — unchanged, and used as the evidence that non-overflowing behavior is preserved.
- `crates/chelis-backend-c/src/memory.rs`, `crates/chelis-backend-hip/src/memory.rs` — not modified. They consume the corrected key.
- `crates/chelis-dim-arith/` — a new leaf crate, **only if** the Verus spike passes its criterion. Otherwise not created.
- `devenv.nix` — a Verus toolchain entry, only if adopted, and confined per the Non-Goals.
- `docs/dim_arithmetic.md`, `AGENTS.md`, `CHANGELOG.md`.
- Relationship to `add-dim-canon-property-tests`: complementary and non-blocking. That change covers the domain where `evaluate` is a trustworthy oracle; this one covers the domain where it is not. Its D7 records the boundary from the other side.
- Relationship to `add-mutation-baseline-dim-canon`: a mutation run over the corrected multiplication is a cheap check that the new regression test pins it. Not a dependency in either direction.
