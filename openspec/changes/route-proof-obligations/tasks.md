# Tasks: route-proof-obligations

This change records a decision. It writes no proof and verifies no code. Its tasks are
complete when the lane is written down, the lane satisfies the isolation condition, the
entry criterion for a second lane is stated, and the cheaper-oracle rule has a worked
example.

## Phase 1 — Write the rule

- [ ] **1.1** Create `docs/verification_lanes.md` recording one lane, Aeneas, with the three reasons: the Lean toolchain already exists here (LaCaDiLE runs green on `leanprover/lean4:v4.29.0`); an extracted model composes into the same logic as `spec/design/verification_stack_sketch.md`'s L5 ladder; and it owns the `Box`-recursive shape Verus provably cannot express, per `prove-dim-canon-overflow-safety` D5.
- [ ] **1.2** Record the entry criterion for a second lane: a named, wanted obligation the existing lane cannot express. State explicitly that inconvenience is not inexpressibility and that a tool's general merits are not a criterion.
- [ ] **1.3** Record that Verus is **deferred, not rejected**, and why — its two nearby obligations are Euclid's algorithm and the absence of a deleted `saturating_mul` call. Name the criterion that would bring it back.
- [ ] **1.4** Record the cheaper-oracle rule, with the `harden-vocabulary-kernel` finding as its worked example: a property quantified over `i32` is exhaustively checkable in seconds, so a proof of it must be justified on other grounds.
- [ ] **1.5** State that this rule creates no obligation to verify anything, and that an unverified component is not thereby a defect.
- [ ] **1.6** Add a pointer from `AGENTS.md` so the next contributor does not re-derive the analysis.

## Phase 2 — The lane satisfies the isolation condition

The crux, generalized from `prove-dim-canon-overflow-safety` D5(c). A lane failing this is
declined, not adopted with an exception.

- [ ] **2.1** Confirm Charon, Aeneas, and Lean install through `devenv.nix` without altering what gate jobs install from `rust-toolchain.toml`.
- [ ] **2.2** Confirm the extracted crate compiles and tests on pinned stable with none of them installed. Extraction runs over unmodified source, so this should be easier than the in-place-annotation case — verify it rather than assuming it.
- [ ] **2.3** Negative test: with the prover absent, `python3 scripts/gate.py` succeeds.
- [ ] **2.4** Negative test: with the prover absent, the proof lane reports "not run", never "passed".
- [ ] **2.5** Add any prover workflow file to `NON_GATE_WORKFLOWS` in `scripts/test_gate.py`, or `test_all_workflow_files_are_scope_classified` fails.
- [ ] **2.6** **If 2.1 or 2.2 fails, decline the lane** and record the evidence. The repository then has no prover lane, which is a permitted outcome; the cheaper oracles still apply.

## Phase 3 — Measure the lane once

- [ ] **3.1** Take the first obligation discharged in the lane — `harden-vocabulary-kernel`'s tag properties — as the cost measurement.
- [ ] **3.2** Record effort spent, toolchain friction, proof-artifact length, and whether the extracted model is reusable by a later Lean theorem about the same code.
- [ ] **3.3** Publish it as a **cost figure**. It SHALL NOT describe the measured obligation as more strongly established than any other.
- [ ] **3.4** Record it as the sizing basis for any subsequent proof work, replacing estimation.

## Phase 4 — Honesty

- [ ] **4.1** Confirm no document presents a count of provers, lanes, or proof artifacts as an assurance figure.
- [ ] **4.2** Confirm every declined lane and every declined obligation is recorded with its reason.
- [ ] **4.3** Confirm the routing doc states that absence of a proof is not a defect, so the rule cannot be read as a verification mandate.
- [ ] **4.4** Confirm no proof in the repository is justified as establishing a property that a cheaper oracle also establishes, unless the justification names lane-establishment as the reason.
- [ ] **4.5** Negative test: a planted document sentence claiming assurance from multiple independent provers fails the anti-overclaim check that `harden-vocabulary-kernel` builds.
- [ ] **4.6** Name the acceptance oracle for this change.

## Deferred, deliberately

- **Verus.** Deferred for want of a qualifying obligation, not rejected. `prove-dim-canon-overflow-safety`'s D1 fix, reachability investigation, rustdoc reconciliation, and regression test are unaffected and land without it; its own D5 already records that declining the spike is a successful outcome.
- **Kani and any other prover.** Same entry criterion.
- **Gating on the lane.** Whether a discharged proof is re-checked in CI or verified once and recorded stays open, per `prove-dim-canon-overflow-safety`'s open question 3.
- **A verified-code programme.** This routes obligations someone already chose to discharge; choosing what to verify next is separate work.
- **Retrofitting the rule onto existing unverified code.** No audit of what "should" be proved follows from this change.
