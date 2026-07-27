# Tasks: prove-dim-canon-overflow-safety

**Authoritative completion oracle:** `cargo nextest run -p chelis-ir -p chelis-backend-c -p chelis-backend-hip`. The two backend crates are in the oracle because they are the consumers whose slot-reuse decisions this change alters. Group 7 is not started until that oracle is green on groups 1 through 6. The Verus spike in group 5 is entered only after the fix in group 3 is green, so no proof effort is on the critical path for the correction.

## 1. Demonstrate the hazard

- [ ] 1.1 Failing test in `crates/chelis-ir/tests/dim_canon_adversarial.rs` constructing two expressions whose concrete factors overflow `usize` to different true products, and asserting their keys differ. It fails against the current implementation. Written in the style of the existing `unsound_a` / `unsound_b` cases in `dim_canonicalization.rs:277-279`.
- [ ] 1.2 Failing test asserting that an overflowing product does not present itself as a single folded constant (spec: an overflowing product stays structural).
- [ ] 1.3 Test asserting the mixed symbolic case specifically: an expression containing a symbol makes `as_concrete` return `None`, so `capacity_fits` reaches the key-equality branch. This is the shape that carries the consequence, and a purely concrete test would not exercise it.
- [ ] 1.4 Backend-level test in `chelis-backend-c` confirming that two slots whose capacities differ only above the overflow threshold are not treated as interchangeable by `capacity_fits`. Extend the existing `capacity_fits_concrete_larger_and_rejects_distinct_symbolic_keys` (`crates/chelis-backend-c/src/memory.rs:584`) rather than starting a new module: it already asserts that distinct symbolic keys are rejected, and the overflow case is that same claim at a boundary it does not reach. This is what makes the defect legible as a memory-planning fault rather than an arithmetic curiosity.
- [ ] 1.5 Confirm all tests in this group fail red before group 2 begins. If 1.1 passes instead, the hand-trace in chelis#888 was wrong: stop, close chelis#888, and abandon this change rather than looking for a different way to make it fail.

## 2. Reachability investigation

- [ ] 2.1 Inspect every construction path for `DimExpr::Mul` and record whether a source program can produce concrete factors whose product overflows. Start from `logical_elements` (`crates/chelis-backend-c/src/memory.rs:327-337`) and `dim_size` (`memory.rs:339-346`), and follow the lowering paths that populate `DimInfo::Lit`.
- [ ] 2.2 Record the answer in the validation notes with the evidence, whichever way it falls. A negative result is recorded as explicitly as a positive one.
- [ ] 2.3 If reachable, construct a Chelis source program that reaches it and add it as a test. If not reachable, record which invariant prevents it and where that invariant is enforced, so a later reader can tell whether it still holds.
- [ ] 2.4 Do not gate group 3 on the outcome. The fix ships either way per design D6.

## 3. The fix

- [ ] 3.1 Retain the pre-change `normalized_key` and its helpers under a test-only name, for the differential in 3.5. This is temporary and is removed in group 6.
- [ ] 3.2 Replace `saturating_mul` with `checked_mul` at the four fold sites: `dag.rs:320`, `dag.rs:372`, `dag.rs:390`, `dag.rs:392`.
- [ ] 3.3 Route the `None` case to the structural path: leave the concrete factors unfolded as separate atoms rather than collapsing them (spec: concrete-factor folding never creates a false key equality). Confirm no path panics (spec: overflow does not panic).
- [ ] 3.4 Confirm group 1 turns green.
- [ ] 3.5 Differential test: old and new implementations agree exactly over the whole existing example corpus and over a bounded generator whose products fit in `usize` (spec: key output is unchanged for every non-overflowing input). If `add-dim-canon-property-tests` has landed, reuse its generator, which is already bounded below the threshold by its own design; otherwise use a local bounded generator.
- [ ] 3.6 Confirm the existing example tests pass with no change to their expected values — `dim_canonicalization.rs`, `dim_canon_adversarial.rs`, and the unit module at `dag.rs:2374-2467` (spec: existing corpus is unchanged).
- [ ] 3.7 Equivalence-relation tests: every key equals itself including overflow-derived keys; a vector of mixed keys sorts to a consistent total order; quotient cancellation over overflow-derived atoms behaves as for ordinary keys (spec: key equality remains an equivalence relation and a total order).
- [ ] 3.8 Confirm by inspection that no sentinel or poison variant was introduced (spec: no poison variant exists).

## 4. Documentation

- [ ] 4.1 Rewrite the `DimExprKey` rustdoc at `dag.rs:147-161`: state the canonicalizer's overflow policy, state that the key is incomplete above the threshold, and name the bound within which the equivalence claim holds (spec: the contract no longer claims an unbounded guarantee). Cite `chelis#888` inline, matching the provenance convention already used for chelis#770, #616, and #392.
- [ ] 4.2 In the same rustdoc, record that `evaluate` (`dag.rs:178`) and `as_concrete` (`dag.rs:203`) multiply without checking and therefore do not share the canonicalizer's policy (spec: the divergence between traversals is visible).
- [ ] 4.3 `docs/dim_arithmetic.md` — the policy, the reachability finding from group 2, and the Verus decision from group 5.
- [ ] 4.4 Pointer from `AGENTS.md`, and a `CHANGELOG.md` entry. The entry states the behavior change plainly: slot reuse differs for programs whose dimension products overflow.

## 5. Verus spike

- [ ] 5.1 Declare the effort budget in writing in the validation record before starting, so the kill criterion is falsifiable rather than retrospective.
- [ ] 5.2 Condition (a): install the Verus toolchain through `devenv.nix` and confirm it does not alter what gate jobs install from `rust-toolchain.toml`. Record the result. Per the coverage change's D8, `devenv`'s `languages.rust.components` is ignored when `toolchainFile` is set, so expect this to need a route that does not go through `languages.rust`.
- [ ] 5.3 Condition (c) first, before writing any proof: confirm a `verus!{}` module can live in a leaf crate that `chelis-ir` depends on, that it compiles on the pinned stable toolchain, and that `cargo nextest run --workspace` succeeds with Verus absent from the machine (spec: an ordinary build never requires the verification tool). This is the condition most likely to fail, so it is tested before effort is spent on proofs.
- [ ] 5.4 Obligation 1: prove `usize_gcd` terminates and returns the greatest common divisor. Termination needs a decreasing measure on the loop; correctness follows the Euclid invariant.
- [ ] 5.5 Obligation 2: prove the replacement fold never yields a folded constant differing from the true product.
- [ ] 5.6 If every condition holds, adopt: move the primitives into `crates/chelis-dim-arith/`, have `dag.rs` call them, and record the decision with its evidence (spec: adoption proceeds when every condition holds).
- [ ] 5.7 If any condition fails, record the decline and the failing condition, and remove every partial artifact of the attempt (spec: a failed condition results in a recorded decline). Confirm the group 3 fix and its tests remain green and unmodified (spec: the fix does not depend on the proof).

## 6. Remove the scaffolding

- [ ] 6.1 Delete the retained pre-change implementation from 3.1 and the differential test that consumed it (spec: no second canonicalizer is left behind).
- [ ] 6.2 Confirm by search that no second definition of the canonicalizer or its helpers remains in the tree.
- [ ] 6.3 Confirm the oracle is still green after the deletion.

## 7. Adversarial validation

- [ ] 7.1 Fresh subagent red team against the fix: search for any remaining input for which two distinct dimension expressions receive the same key, including expressions mixing symbols with overflowing concrete factors, nested quotients whose numerator and denominator both overflow, and products that overflow only after quotient cancellation has run.
- [ ] 7.2 Confirm the fix did not move the hazard rather than remove it: check that the structural path from 3.3 cannot itself produce two identical atom multisets from expressions with different true values.
- [ ] 7.3 Attack the comparator contract directly: build vectors of overflow-derived keys, sort them, and confirm cancellation in `normalize_quotient_parts` produces the same result under different input orderings.
- [ ] 7.4 Confirm the differential in 3.5 cannot pass vacuously — instrument it to count the expressions compared and record the figure, since a generator producing nothing would agree with anything.
- [ ] 7.5 Confirm no `saturating_mul` remains in the canonicalizer span.

## 8. Validation and acceptance

- [ ] 8.1 Authoritative oracle green: `cargo nextest run -p chelis-ir -p chelis-backend-c -p chelis-backend-hip`.
- [ ] 8.2 Full workspace green: `cargo nextest run --workspace`.
- [ ] 8.3 `openspec validate --all --strict` passes.
- [ ] 8.4 `python3 scripts/gate.py --local` green.
- [ ] 8.5 Confirm every scenario in `specs/dim-arith-overflow-policy/spec.md` maps to at least one test from groups 1, 3, 5, or 6.
- [ ] 8.6 Record in the validation notes: the reachability finding, the differential comparison count, the Verus decision and its evidence, and the behavior change as it will read in `CHANGELOG.md`.

### Validation record

_Populated during implementation._
