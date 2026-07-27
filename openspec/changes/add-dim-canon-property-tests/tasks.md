# Tasks: add-dim-canon-property-tests

**Authoritative completion oracle:** `cargo nextest run -p chelis-ir`, which runs the new `dim_canon_properties` binary alongside the existing `dim_canonicalization` and `dim_canon_adversarial` binaries. Group 6 is not started until that oracle is green on groups 1 through 5. No property assertion is enabled in a committed test until its generator satisfies the group 2 health check, so the suite cannot report a vacuous pass at any point in the sequence.

## 1. Generator and failing properties

- [ ] 1.1 Add `hegel` to `crates/chelis-ir/Cargo.toml` under `[dev-dependencies]`. Confirm it resolves and that `cargo nextest run -p chelis-ir` still passes unchanged, so a dependency failure is separated from a test failure.
- [ ] 1.2 `crates/chelis-ir/tests/dim_canon_properties.rs` — expression generator over the four `DimExpr` constructors, with an explicit depth bound and a small symbol alphabet. Symbols drawn from a fixed small set, so collisions between the two generated expressions are common rather than rare; a generator drawing fresh unique names would make key equality almost unreachable and the soundness property near-vacuous.
- [ ] 1.3 Binding generator drawing every symbol to at least 1 (spec: bindings are positive). Assert the invariant on the generator itself, not only on its consumers.
- [ ] 1.4 Constructive divisible-quotient path per design D2: draw a factor multiset, assemble the numerator, draw a sub-multiset for the denominator, emit `Div`. Assert directly that both sides evaluate `Ok` (spec: quotients are exact by construction).
- [ ] 1.5 Document and enforce the concrete-value and product ceiling per D7, with a comment naming the overflow regime as a deliberate exclusion and pointing at `prove-dim-canon-overflow-safety` (spec: the overflow regime is not entered).
- [ ] 1.6 Soundness property: equal keys imply equal evaluated values, with failure output naming both expressions, the binding, the shared key, and both values (spec: equal keys agree on value; colliding keys fail loudly).
- [ ] 1.7 Assert the negative direction is absent: a value-equal pair with differing keys passes (spec: unequal keys are not a failure). Written as a test over a hand-constructed pair such as `(n * 3) / 2` against its structural equal, so the intentional incompleteness is pinned and a later reader cannot mistake it for a gap.
- [ ] 1.8 Rewrite-family invariance per D3, one property per member: `Mul` commutation, `Mul` reassociation, multiply-by-one, and the `Div(Mul(x, k), k)` round trip. Enumerate the family in one place in the source (spec: the closed rewrite family).
- [ ] 1.9 Usable-case counter and the declared floor, with the floor declared next to the generator (spec: a vacuous run fails).
- [ ] 1.10 Confirm the properties fail red against a deliberately broken key function before group 2 begins — a local stub that ignores `Div` entirely is sufficient, and it must be reverted, not committed.

## 2. Health check and seeding

- [ ] 2.1 Wire the usable-case floor into a failure that names the observed fraction and the floor (spec: too many discarded cases fails the run).
- [ ] 2.2 Verify the health check does not mask a real counterexample: with the floor satisfied and a broken key stub in place, the run must fail on the counterexample and not on the floor (spec: a productive run with a real counterexample still fails).
- [ ] 2.3 Fixed committed default seed; environment-variable override; effective seed and reproduction invocation printed on failure (spec: gate runs are deterministic).
- [ ] 2.4 Confirm two consecutive default runs exercise the same cases on the same commit and toolchain (spec: default invocation is deterministic).
- [ ] 2.5 Bound the case count so the test binary stays within a stated wall-clock budget on the recording host. Record the measured figure in the validation notes; `scripts/test_timing_baseline.json` makes a later regression visible.

## 3. Exploration and counterexample promotion

- [ ] 3.1 Run the suite by hand at high case counts across many random seeds. Record the number of cases attempted, the usable fraction actually observed, and the wall-clock cost, in the validation record below.
- [ ] 3.2 For every soundness counterexample found, add a named deterministic test to `crates/chelis-ir/tests/dim_canon_adversarial.rs` constructing the offending expressions directly, in the style already used in that file (spec: counterexamples become committed example tests).
- [ ] 3.3 Confirm each promoted test compiles and fails for the same reason with the property library removed from the build (spec: promoted tests survive removal of the property suite).
- [ ] 3.4 If no counterexample is found, record that outcome explicitly with the case count and seed range, so a future reader knows the strength of the negative result rather than inferring it.
- [ ] 3.5 Route any finding to a separate change rather than editing `crates/chelis-ir/src/dag.rs` here. Record the routing decision in the validation notes. This change modifies no source file.

## 4. Documentation

- [ ] 4.1 `docs/property_testing.md` — running the suite, overriding the seed, reproducing a failure, and promoting a counterexample into `dim_canon_adversarial.rs`.
- [ ] 4.2 In the same doc, state the two deliberate limits plainly: completeness is asserted only over the closed rewrite family, and the overflow regime is excluded because the oracle itself panics there.
- [ ] 4.3 Pointer to the runbook from the `AGENTS.md` testing section.
- [ ] 4.4 `CHANGELOG.md` entry under contributor tooling.

## 5. Strict validation

- [ ] 5.1 `openspec validate --all --strict` passes.
- [ ] 5.2 Confirm every scenario in `specs/dim-canon-properties/spec.md` maps to at least one assertion from group 1, 2, or 3.
- [ ] 5.3 `python3 scripts/gate.py --local` green, confirming the new test binary does not disturb an existing stage.
- [ ] 5.4 Confirm no file under `crates/*/src/` is modified by this change.

## 6. Adversarial validation

- [ ] 6.1 Fresh subagent red team against vacuity: force the generator to emit only inexact quotients and confirm the health check fails loudly rather than reporting green; set the floor to zero and confirm that requires an explicit, visible source edit rather than being reachable by configuration.
- [ ] 6.2 Confirm the soundness property cannot pass by never generating equal keys. Instrument a run to count the cases in which the two keys were in fact equal, and record that count. A property that never observed a collision has not tested the implication, and the count is the only evidence that distinguishes the two.
- [ ] 6.3 Attempt to reach a panic inside `evaluate` by pushing the generator bound upward, and confirm the documented ceiling is what prevents it rather than luck.
- [ ] 6.4 Confirm the rewrite-family properties fail when a member is applied incorrectly — for example a `Div(Mul(x, k), k)` round trip with `k = 0` must not be generated, and must be shown to be excluded by construction rather than by the key happening to match.

## 7. Acceptance

- [ ] 7.1 Authoritative oracle green: `cargo nextest run -p chelis-ir`.
- [ ] 7.2 Full workspace green: `cargo nextest run --workspace`.
- [ ] 7.3 Record in the validation notes the final case count, the observed usable fraction, the observed key-collision count from 6.2, the wall-clock cost, and every counterexample promoted.

### Validation record

_Populated during implementation._
