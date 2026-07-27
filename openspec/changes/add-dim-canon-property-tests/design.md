# Design: add-dim-canon-property-tests

## Context

`DimExpr` is a four-constructor tree: `Concrete(usize)`, `Sym(String)`, `Mul`, `Div` (`crates/chelis-ir/src/dag.rs:139-145`). Three functions traverse it, and all three matter here.

- `evaluate(&self, bindings) -> Result<usize, String>` (`dag.rs:171-192`). Rejects division by zero and inexact division. This is the reference oracle.
- `as_concrete(&self) -> Option<usize>` (`dag.rs:199-210`). Used by `capacity_fits` to decide whether the concrete comparison applies.
- `normalized_key(&self) -> DimExprKey` (`dag.rs:283-294`), with helpers `usize_gcd` (`dag.rs:298`), `flatten_product` (`dag.rs:311`), `push_product_atoms` (`dag.rs:318`), `assemble_product` (`dag.rs:338`), `normalize_dim_product` (`dag.rs:353`), `normalize_dim_quotient` (`dag.rs:419`), and `normalize_quotient_parts` (`dag.rs:447`). This is the subject.

The consumer that gives the subject its weight is `capacity_fits` (`crates/chelis-backend-c/src/memory.rs:315-324`):

```rust
match (slot_concrete, req_concrete) {
    (Some(slot_elems), Some(req_elems)) => slot_elems >= req_elems,
    _ => slot_key == req_key,
}
```

When either side is symbolic, buffer-slot reuse is decided by key equality alone. The HIP backend repeats the pattern at `crates/chelis-backend-hip/src/memory.rs:360-362`.

Existing coverage is example-based: `crates/chelis-ir/tests/dim_canonicalization.rs` (~340 lines) and `crates/chelis-ir/tests/dim_canon_adversarial.rs` (~200 lines), plus a unit module in `dag.rs:2374-2467`. The adversarial file already tests cancellation, alpha-renaming, zero denominators, and constant-factor division. It is good work, and it is finite.

`proptest` is already a dev-dependency of `chelis-deep` and `chelis-surf`, so the workspace has precedent for generative testing but none in `chelis-ir`.

## Goals / Non-Goals

**Goals:**

- State the soundness contract of `normalized_key` as a machine-checked property rather than a rustdoc sentence.
- Generate within the domain the contract actually covers, and prove that the generator does so productively rather than vacuously.
- Turn any counterexample into a committed, deterministic regression test.
- Keep the gate deterministic and its runtime bounded.

**Non-Goals:**

- No completeness requirement outside a closed, named rewrite family. See D3.
- No change to `normalized_key`. See the proposal's Non-Goals.
- No new workflow, no new gate stage.
- No exploration of the integer-overflow regime. See D7, and `prove-dim-canon-overflow-safety`.

## Decisions

### D1: hegel for the generator, and the properties stay portable

The properties are predicates over `DimExpr` values. They depend on the library for generation and shrinking, not for their statement. Written as plain functions taking already-constructed inputs, they are portable to `proptest` or to a hand-rolled loop with a one-line adapter change.

That portability is what makes the library choice low-risk, and it is the reason to record the alternative plainly: **`proptest` is already in the workspace and would cost nothing new.** The case for `hegel` is its Hypothesis-derived internal shrinking model, which reduces the underlying choice sequence rather than applying per-type shrinkers. Recursive tree generators are where that difference shows: in `proptest`, `prop_recursive` requires explicit depth and size budgets, and shrinking a recursive structure depends on the shrinker written for it. The counterexamples that matter here are small trees — `(n * 3) / 2`, `(a * c) / c` — and the difference between a minimal counterexample and a 12-node random tree is the difference between a finding a reader can act on and one they must first decode.

**The trade-off is real.** A second property-testing library in the workspace is a maintenance cost, and `hegel` is young. It is accepted because the blast radius is one dev-dependency on one crate, the properties survive a library swap, and the alternative is available at any time. If the integration proves awkward during group 1, the fallback is `proptest` with the same properties, and the change still delivers.

`hegel` enters through `crates/chelis-ir/Cargo.toml` under `[dev-dependencies]`, the same route `proptest` takes in `chelis-deep` and `chelis-surf`. It is a library, not a CLI tool, so it resolves through Cargo and needs no `devenv.nix` entry.

### D2: Divisibility is constructed, not filtered

A naive generator that builds `Div(a, b)` from two independent subtrees produces an inexactly-divisible expression almost every time, and `evaluate` returns `Err`. Filtering on that outcome would discard nearly every case and exercise the `Div` path — the most intricate part of the canonicalizer — almost never.

The generator therefore builds divisible quotients by construction. It draws a factor multiset, assembles the numerator as a product over that multiset, draws a sub-multiset for the denominator, and emits `Div(numerator, denominator)`. Exact divisibility then holds for every binding, by construction rather than by luck.

Rejection is still permitted for the residual cases the constructive path does not cover, which is why D4 exists.

### D3: Completeness is specified only over a closed rewrite family

General completeness — key inequality implies value inequality — is not a property of `normalized_key` and must not be asserted. `(n * 3) / 2` stays structural on purpose, because divisibility of `n` is unknown. Asserting general completeness would fail on correct code.

But a restricted completeness is both real and already tested by hand. `dim_expr_normalized_key_canonicalizes_mul_order_and_associativity` (`dag.rs:2374`) asserts that reassociating and reordering a product preserves the key. That generalizes to a closed family of value-preserving rewrites over which the key is required to be invariant:

1. commute the operands of a `Mul`;
2. reassociate nested `Mul` nodes;
3. multiply by `Concrete(1)`;
4. wrap a subterm `x` as `Div(Mul(x, k), k)` for a concrete `k >= 1`.

Applying any sequence from this family SHALL leave the key unchanged. Anything outside the family is governed by the sound direction only. The line is drawn at a named, closed list precisely so that a future reader can tell which failures indicate a defect and which indicate an intentional limit.

### D4: A vacuous pass is a failure, not a pass

The soundness property is conditioned: it applies when both expressions evaluate to `Ok`. A generator defect, a too-aggressive bound, or a future change to `evaluate`'s error conditions could drive the usable fraction toward zero. Every remaining case would then satisfy the property trivially and the suite would report green while checking nothing.

The suite therefore counts usable cases and fails when the fraction falls below a declared floor, naming the observed fraction and the floor. This mirrors the fail-closed correction made in `add-coverage-baseline-chelis-ir` D4, where an empty measurement was reclassified from "0.0% coverage" to a loud failure, and the same correction in `harden-lint-traversal-edges`. A test suite that cannot distinguish "checked many cases and found nothing" from "checked nothing" is not evidence.

The floor is a property of the generator, not of the subject, so it is recorded in the test file next to the generator it governs.

### D5: Fixed seed in the gate, random seeds by hand

The suite runs inside the existing `integration` stage, which is a required check. A property suite that draws fresh randomness on every CI run converts a latent defect into an intermittent red build on an unrelated pull request, and the repository has no quarantine lane for that.

The default seed is therefore fixed and committed. An environment variable overrides it, the effective seed is printed on failure, and `docs/property_testing.md` documents the random-seed invocation for local and maintainer use.

**The cost is stated plainly:** a fixed seed explores the same cases on every run, so steady-state CI value decays to that of an example test once the seed has passed. The exploration happens when a maintainer runs random seeds, and its results are preserved by D6 rather than by re-drawing. A scheduled random-seed lane would recover continuous exploration; it is deliberately deferred, because this change adds no workflow and the gate-flake argument would have to be settled first.

### D6: Counterexamples are promoted to committed example tests

A counterexample found by a random draw is not reproducible from the suite alone once the seed changes. Every counterexample is therefore added to `crates/chelis-ir/tests/dim_canon_adversarial.rs` as an ordinary named test, in the style already used there.

This keeps the regression pinned deterministically, keeps the finding legible to a reader who never runs the property suite, and means the property suite's long-term contribution is the corpus it produced, not the run itself.

### D7: The generator stays below the overflow regime, deliberately

`evaluate` multiplies with unchecked `*` (`dag.rs:178`), as does `as_concrete` (`dag.rs:203`). In a debug build — which is how tests run — an overflowing product panics. The canonicalizer, by contrast, uses `saturating_mul` (`dag.rs:320`, `dag.rs:372`, `dag.rs:390`, `dag.rs:392`).

So the oracle and the subject disagree about what overflow means, and the oracle is the one that panics. A generator that reached the overflow regime would report panics originating in `evaluate`, describing a defect this change is not scoped to fix and cannot express a property about.

Concrete values and constructed products are therefore bounded well below `usize::MAX`, and the bound is documented in the test file as a deliberate blind spot rather than an incidental limit.

**This blind spot is exactly the subject of `prove-dim-canon-overflow-safety`.** The two changes are complementary by construction: property testing covers the domain where the oracle is trustworthy, and the overflow change covers the domain where it is not. Recording the boundary in both places is what keeps the gap from being forgotten.

### D8: Activation ordering

1. Generator and properties, with the health check, against the current `normalized_key`. No source file changes.
2. Run at scale by hand with random seeds. Record findings.
3. Promote any counterexample to `dim_canon_adversarial.rs`.
4. Documentation.

**Rollback:** delete `crates/chelis-ir/tests/dim_canon_properties.rs`, the `[dev-dependencies]` line, and the doc. Committed counterexamples in `dim_canon_adversarial.rs` are ordinary tests with no dependency on the property library and are kept. No source file, no spec, and no gate command is touched.

## Risks / Trade-offs

**[The suite passes and proves nothing]** → This is the primary risk, and D4 is the specific mitigation: a vacuous pass is converted into a failure. The residual risk — a generator that is productive but explores a narrow shape — is mitigated by D2 constructing `Div` cases deliberately, and is measurable: `add-mutation-baseline-dim-canon` exists to answer whether these properties pin the implementation, and this suite is one of its inputs.

**[A second property-testing library is a maintenance burden]** → Accepted, and bounded by D1. One dev-dependency, one crate, properties portable to `proptest` by adapter.

**[Gate flake]** → Mitigated by the fixed default seed in D5, at the documented cost of decayed steady-state exploration.

**[The integration stage gets slower]** → The case count is bounded so the test binary stays within a stated wall-clock budget, and `scripts/test_timing_baseline.json` already makes suite timing visible if the budget is missed.

**[The change finds a real soundness hole and the proposal forbids fixing it here]** → Deliberate. A fix to dimension equality changes buffer-slot reuse in two backends, and that belongs in a change whose review is aimed at generated code. The finding is recorded and routed. The cost is a window in which a known defect is documented and unfixed, which is preferable to an unreviewed semantics change landing inside a test-only proposal.

## Open Questions

- Whether `hegel`'s current API is stable enough for a pinned dev-dependency, or whether the `proptest` fallback in D1 should be taken immediately. Settled in group 1 by building the generator; it does not affect the requirements, which name no library.
- Whether the closed rewrite family in D3 should later grow a fifth member for `Div` reassociation. Deferred: the four listed are the ones the existing example tests already assert.
- Whether a scheduled random-seed lane is worth a workflow. Deferred per D5.
