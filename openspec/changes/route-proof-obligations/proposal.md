# Proposal: route-proof-obligations

## Why

Two mechanized-verification tools arrived in this repository through two changes written independently of each other:

- `prove-dim-canon-overflow-safety` spikes **Verus** against `usize_gcd`'s termination and the canonicalizer's multiplication.
- `harden-vocabulary-kernel` spikes **Aeneas** against `RuntimeDType`'s tag round-trip, injectivity, and exhaustive rejection.

Neither cited the other. Left alone, the repository acquires two prover toolchains, two CI lanes, two proof idioms, and two maintenance tails — without anyone having decided to.

**The decision is one lane: Aeneas.** This change records that, records the criterion a second lane must meet before it is added, and records the conditions any lane must satisfy to exist at all. It adds no proof and verifies nothing.

### Why one lane, and why that one

Adopting both was the initial instinct, on the strength of a real finding: `prove-dim-canon-overflow-safety`'s design D5 shows Verus provably cannot express `normalized_key`, because the type carries `String`, `Vec`, and `Box` and recurses over a heap-allocated enum. That shape is Aeneas' central case. The tools genuinely differ.

But "the tools differ" is an argument for choosing correctly, not for running both. The deciding question is what Verus would actually buy on the obligations currently in flight, and the answer is thin:

- **`usize_gcd` terminates and returns the true GCD.** Euclid's algorithm, six lines, the most-studied algorithm in existence. A property test finds any bug in it.
- **Multiplication cannot fold a wrong constant.** Genuine, but the accompanying fix achieves it *by construction* — it stops calling `saturating_mul`. Proving the absence of a deleted call is a weak result.

Neither obligation is weak because Verus is weak. They are weak because they are the obligations that happened to be nearby. A prover lane should be opened by an obligation that demands it, not by proximity.

Against that, Aeneas has three properties Verus does not:

- **The toolchain already exists here.** Lean 4.29.0 is installed and green, because LaCaDiLE's ~86k-line mechanization runs on it. Verus would be new.
- **It composes outward.** An Aeneas-extracted model lands in the same logic as LaCaDiLE and `spec/design/verification_stack_sketch.md`'s L5 trust ladder, so an implementation proof and a metatheory proof can eventually meet. A Verus contract composes with nothing outside Verus.
- **It owns the shape Verus cannot express**, per D5, while Verus owns no shape Aeneas cannot reach — only shapes Aeneas reaches less conveniently.

At four theorems total across the repository, one lane is the right ratio. The cost of deferring Verus is that adopting it later is slightly more expensive than adopting it now; that is a good trade.

### What this change deliberately preserves

The routing analysis is not discarded. It becomes the **entry criterion** for a second lane rather than a rota between two. Recording it now is the point: without it, the next contributor with an obligation and a favorite prover reopens the whole argument from zero.

## What Changes

- **One prover lane, Aeneas**, recorded as the repository's verification lane.
- **An entry criterion for any second lane**: a named, wanted obligation that the existing lane cannot express. Inconvenience is not inexpressibility, and a preference is not a criterion.
- **Adoption conditions applying to any lane**, generalized from `prove-dim-canon-overflow-safety`'s D5 conditions (a), (b), (c). The crux — that the repository builds and tests without the prover installed — governs whether a lane may exist at all.
- **A double-counting prohibition**, retained against the day a second lane exists.
- **A requirement that cheaper oracles be considered first.** Where an exhaustive check, a differential test, or a property test establishes the same guarantee at lower cost, a proof is not the default answer and the comparison is recorded. This is what the `harden-vocabulary-kernel` pilot found about its own headline property.
- Documentation recording the rule and its rationale.

### Non-Goals

- **No mandate to verify anything.** This change routes obligations someone has already decided to discharge. Absence of a proof is not a defect.
- **Verus is not rejected on the merits.** It is deferred for want of an obligation that needs it. The entry criterion is how it returns.
- **No third prover.** Kani and the rest face the same entry criterion.
- **Not a gate.** Whether a discharged proof is re-checked in CI or verified once and recorded stays open, per `prove-dim-canon-overflow-safety`'s open question 3.
- **No change to either spike's kill criterion.** Both remain declinable; declining is a successful outcome per that change's D5.
- **No `spec/**` change**, and no change to `rust-toolchain.toml` or to what gate jobs install.

## Capabilities

### New Capabilities

- `proof-obligation-routing`: which obligations are eligible for mechanized proof at all, the requirement that cheaper oracles be compared first, the conditions a prover lane must satisfy to exist, the criterion a second lane must meet, the prohibition on reporting overlapping proofs as compounded assurance, and the requirement that a declined or absent proof be recorded rather than implied.

### Modified Capabilities

None. Neither spike change has landed, so there is no shipped requirement to modify.

## Impact

- `docs/verification_lanes.md` (new) — the lane, the entry criterion, the conditions, the cheaper-oracle rule.
- `devenv.nix` — the Aeneas/Charon/Lean lane enters through it, per the existing convention. It may not alter what `rust-toolchain.toml` installs for gate jobs.
- `scripts/test_gate.py` — `NON_GATE_WORKFLOWS` gains any prover workflow, or `test_all_workflow_files_are_scope_classified` fails.
- `AGENTS.md` — a pointer, so the next contributor does not re-derive the analysis.
- `prove-dim-canon-overflow-safety` — its Verus spike is **deferred, not cancelled**. Its D1 fix, reachability investigation, rustdoc reconciliation, and regression test are unaffected and land independently; D5 notes that declining the spike is already a successful outcome of that change. Its D5 analysis is the source of this change's entry criterion.
- `harden-vocabulary-kernel` — operates in the Aeneas lane; supplies the cheaper-oracle finding that motivates that requirement here.
- Relationship to LaCaDiLE: no dependency added. That its Lean toolchain already exists here is part of why the lane is Aeneas.
