# Design: prove-dim-canon-overflow-safety

## Context

`DimExprKey` (`crates/chelis-ir/src/dag.rs:163-168`) derives `Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord`. Those derives are load-bearing: `assemble_product` sorts atom vectors (`dag.rs:344`), `normalize_quotient_parts` sorts both numerator and denominator atoms and cancels them by a merge walk (`dag.rs:470-490`), and the backends store a `capacity_key: DimExprKey` per slot (`crates/chelis-backend-c/src/memory.rs:47`). Any change to the type must preserve `Eq` as an equivalence relation and `Ord` as a total order, or the sorting and cancellation logic silently misbehaves.

The canonicalizer folds concrete factors with `saturating_mul` at four sites: `push_product_atoms` (`dag.rs:320`) and `take_factor` inside `normalize_dim_product` (`dag.rs:372`, and the `Div` arm at `dag.rs:390` and `dag.rs:392`).

`usize_gcd` (`dag.rs:298-306`) is the exact-cancellation primitive for the concrete parts of a quotient, called once at `dag.rs:494`. It is eight lines of pure `usize` arithmetic with no allocation:

```rust
fn usize_gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
```

Its correctness underwrites the claim in `normalize_quotient_parts`' rustdoc that "constants are GCD-reduced exactly". Today that claim is supported by a handful of example tests.

## Goals / Non-Goals

**Goals:**

- Remove the false-equality hazard from dimension-key comparison.
- Preserve today's key output exactly for every non-overflowing input, with evidence rather than assertion.
- Determine whether the hazard is reachable from a Chelis source program, and record the answer.
- Evaluate Verus against two narrow obligations under a declared budget, and record the decision either way.

**Non-Goals:**

- No general verified-code programme. See the proposal's Non-Goals.
- No change to `evaluate` or `as_concrete`.
- No frontend bound on dimension sizes. See D3.

## Decisions

### D1: On overflow, do not fold — keep the product structural

When the concrete factors of a product exceed `usize`, the canonicalizer stops folding them into a single `Concrete` and leaves the factors in place as separate atoms of the `Mul`.

Two distinct expressions then retain distinct factor multisets, so their keys stay distinct and no false equality is created. The key becomes **less complete** above the threshold — two expressions that are genuinely equal may now receive different keys — and that is the correct direction to fail. Incompleteness makes memory planning refuse a reuse it could have allowed. Unsoundness makes it perform a reuse it must not. The rustdoc already declines to promise completeness, and `(n * 3) / 2` staying structural is the precedent.

**A poison or sentinel key variant was rejected outright.** The obvious alternative — an `Overflow` variant that compares unequal to everything — breaks `Eq`'s reflexivity, because such a value would not equal itself. `DimExprKey` is sorted in two places and hashed, and a non-reflexive `Eq` with a derived `Ord` produces a comparator that violates the total-order contract. The consequence would be corrupted cancellation in `normalize_quotient_parts`, which is a worse defect than the one being fixed and a much harder one to observe.

### D2: `checked_mul` at the fold sites, not a panic

The four fold sites move from `saturating_mul` to `checked_mul`, with `None` routed to the structural path from D1.

Panicking on overflow was rejected. Memory planning runs inside the compiler over shapes the user wrote. A shape whose product exceeds `usize` is not a compiler-internal invariant violation; it is an input. Turning it into a panic converts a planning corner into a crash with no source span attached.

### D3: Bounding dimensions at the frontend is deferred, not dismissed

The alternative fix is to reject an overflowing shape at type-check time, so no such `DimExpr` ever reaches the canonicalizer. That is arguably the better long-term answer, and it is out of scope here for two reasons.

First, it changes user-visible diagnostics and belongs in a change reviewed against the frontend and its error surface, not inside a canonicalizer fix. Second, it would not discharge the obligation: `normalized_key` is a library function over a public type, and its soundness should not rest on a precondition established three layers away and nowhere checked. D1 makes the function sound on its own terms, which is what allows a frontend bound to be added later as an optimization rather than as a load-bearing assumption.

### D4: Non-overflowing behavior is preserved, and the evidence is differential

"The key is unchanged for non-overflowing inputs" is the constraint most likely to be violated silently, because the fold sites are on the hot path for every ordinary expression.

The evidence is a direct differential, not an argument. The pre-change implementation is retained under a test-only name for the duration of the change, and a test asserts that old and new agree exactly over a bounded generator plus the whole existing example corpus. When `add-dim-canon-property-tests` has landed, its generator is the natural source of inputs, since it is already bounded below the overflow threshold by its own D7. If it has not landed, a local bounded generator serves; the changes are not ordered with respect to each other.

The retained old implementation is deleted before the change closes. Leaving it in would be a second definition of the canonicalizer with no caller.

### D5: The Verus spike is bounded, and only two obligations are in scope

Verus verifies Rust written inside the `verus!{}` macro, in a restricted subset. `normalized_key` is not expressible in it: the type carries `String`, `Vec<DimExprKey>`, and `Box`, and the algorithm is a recursion over a heap-allocated enum. Any proposal to verify the canonicalizer wholesale would be a proposal to rewrite it, and this change does not make one.

Two leaf obligations are expressible, and they are the ones with real content:

1. **`usize_gcd` terminates and is correct.** The loop needs a decreasing measure, and the Euclid invariant gives the correctness argument. This underwrites the exact-cancellation claim that `normalize_quotient_parts` relies on and that tests currently only sample.
2. **The replacement fold never yields a folded constant differing from the true product.** This is the D1/D2 property stated as a proof obligation rather than as a test.

The spike carries a declared effort budget and three conditions, all of which must hold:

- **(a)** The Verus toolchain installs reproducibly through `devenv.nix` without altering what the gate jobs install from `rust-toolchain.toml`.
- **(b)** Both obligations are discharged.
- **(c)** The verified code is callable from `chelis-ir` on the pinned stable toolchain, and building or testing this repository does not require Verus to be installed.

**Condition (c) is the one most likely to fail, and it is the crux.** `verus!{}` expands to ordinary Rust, so callability is plausible in principle; the difficulty is arranging that the verification step is a separate, optional lane while the same source compiles normally for everyone else. If that arrangement is not clean, the correct outcome is to decline.

Failing any condition means the spike is recorded as declined with its evidence, and the change ships the D1 fix. **Declining is a successful outcome of this change, not a failure of it.** The fix is sound and tested without any proof; Verus would add a machine-checked argument for one helper and a precondition written down rather than assumed. That is genuine and modest, and it does not justify unbounded effort.

### D6: Reachability is investigated, and does not gate the fix

Whether lowering can produce a `DimExpr` whose concrete factors overflow is determined by inspection of the paths that construct `DimExpr::Mul` — `logical_elements` (`crates/chelis-backend-c/src/memory.rs:327-337`) folds a tensor's dims into a product, and `dim_size` (`memory.rs:339-346`) maps each `DimInfo` to a factor — and the answer is recorded.

It is deliberately not a precondition for the fix. If reachable, the change is a defect fix and the finding should be reported. If unreachable today, the change removes an unstated assumption that a future lowering path could violate without anyone noticing, which is the more common way this class of defect ships. The cost of the fix is four call sites either way.

### D7: Activation ordering

1. A failing test constructing the colliding keys directly. No reachability argument needed: it is a fact about `normalized_key` at the unit level.
2. The reachability investigation, recorded.
3. The D1/D2 fix, with the D4 differential as its guard.
4. Rustdoc reconciling the three overflow policies, and `docs/dim_arithmetic.md`.
5. The Verus spike against its budget and conditions; the decision recorded either way.
6. Deletion of the retained pre-change implementation.

**Rollback:** revert the four fold sites and the rustdoc. The regression test then fails, which is the correct signal that the hazard is back rather than a reason to delete it. If the Verus spike was adopted, the leaf crate and its devenv entry are removed separately; nothing in `chelis-ir` depends on the proof for its behavior.

## Risks / Trade-offs

**[The fix perturbs ordinary key output]** → The primary risk, since the fold sites run for every expression. D4's differential against the retained implementation is the specific mitigation, and it is exhaustive over the existing corpus rather than a sample.

**[Memory reuse gets worse]** → Only above the overflow threshold, where the current reuse decision is wrong. A refused reuse in that regime replaces an incorrect acceptance. There is no non-overflowing input for which reuse quality changes, and D4 is what establishes that.

**[The Verus spike consumes effort and delivers nothing]** → Bounded by the declared budget and the three conditions in D5, and made survivable by ordering the fix ahead of the spike. The worst case is a recorded negative result, which is itself useful: the question "should this repository adopt Verus" is currently open and unevidenced, and a bounded spike against two real obligations answers it better than speculation.

**[The retained old implementation is left behind]** → Task 6 in the ordering deletes it, and the validation group checks for it explicitly. A second canonicalizer with no caller is exactly the kind of residue that outlives the change that created it.

**[The collision is unreachable and the change is unnecessary]** → Accepted. The fix is four call sites, the regression test is a dozen lines, and the result is that a documented contract becomes true unconditionally instead of true under an unstated bound. D6 records the reachability answer so a reader can judge the severity for themselves rather than inheriting this proposal's estimate.

## Open Questions

- Whether `evaluate`'s unchecked multiplication (`dag.rs:178`) should become checked. It is out of scope here: `eval.rs` and its evaluator are the bit-parity oracle for the C backend (chelis#770), and a change to panic behavior in that path needs review against that oracle. The reconciled rustdoc from group 4 is what makes the question visible rather than answering it.
- Whether a frontend bound on dimension products should follow, per D3.
- Whether the two Verus obligations, if discharged, should be re-checked in CI or verified once and recorded. Settled during the spike; it depends entirely on how condition (c) resolves.
