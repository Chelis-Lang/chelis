# Implementation Plan — chelis_plan.md: 7 Work Items with Red-Team Milestones

> Historical July 2026 execution plan. Its task status and version targets are
> not current; use [`chelis_project_plan.md`](../../spec/design/chelis_project_plan.md)
> and the linked issues for active planning.

**Filed:** 2026-07-22
**Status:** In progress

## Problem Statement

Seven unblocked-consumer items need to ship: transcendental eval perf, timeout
verdicts, capabilities release, quantile primitive, induction tier, eval-side
import probe, and tracker hygiene. Each item unblocks a consumer that is
currently stuck. Verification must happen against the published v0.16.1 binary
and a fresh-context red-team agent validates each milestone.

## Requirements

- All fixes verified against a published binary (v0.16.1 baseline), not just a local build
- Red team at milestones = spawn fresh-context subagent per the `redteam-exec` skill
- Parallelize independent items; serialize dependent chains
- Transcendental perf must have before/after measurement on the same model
- Any new primitive or tier must ship with honest documentation of what the prover can/cannot establish

## Dependency Graph

```
Stream 1 (serialize): Task 1 → Task 2 → Task 5 (release)
Stream 2 (independent): Task 3, Task 4 (can run alongside Stream 1)
Stream 3 (after release): Task 6, Task 7

         ┌─ Task 1 (perf) ──┐
         │                   │
         │   Task 2 (verdicts)│
         │                   │
         │   Task 3 (probe)  │
         │                   ├──► Milestone 1 Red Team ──► Task 5 (release)
         │   Task 4 (tracker)│                                    │
         └───────────────────┘                                    │
                                                                  ▼
                                              Task 6 (quantile) + Task 7 (induction)
                                                                  │
                                                                  ▼
                                                     Milestone 2 Red Team
                                                                  │
                                                                  ▼
                                                   Task 8 (final verification)
```

## Task Breakdown

### Task 1: Transcendental fuzz evaluation — add `normal_cdf` to concrete_eval

**Objective:** Make the Tier C fuzzer able to evaluate properties involving
`normal_cdf` without starving. Currently `apply_intrinsic` in
`crates/chelis-prove/src/concrete_eval.rs` returns `NaN` for any function not in
its whitelist (exp/log/sqrt/sin/cos/abs/min/max), which causes every sample
involving `normal_cdf` to be rejected, producing the >200s single-sample
behavior in chelis#659.

**Implementation:**
- Add `"normal_cdf"` and `"erf"` to `apply_intrinsic`
- Use Abramowitz & Stegun 7.1.26 rational approximation for erf (sufficient for f64 fuzz)
- Record before/after timing on the BS positivity property

**Tests:**
- `normal_cdf(0) ≈ 0.5`, `normal_cdf(5) ≈ 1.0`, `normal_cdf(-5) ≈ 0.0`
- `erf(0) = 0`, `erf(3) ≈ 1`
- Integration: property with `normal_cdf` completes in < 5s

**Acceptance:** `chelis prove --tier fuzz` on a normal_cdf property < 5s.

---

### Task 2: Timeout verdicts — audit and fix beacon subprocess death paths

**Objective:** Every prove exit path produces a response. Dropped connections
produce honest `unknown` verdicts with reason.

**Implementation:**
- Audit `beacon_shim.rs`: child exits non-zero, pipe breaks, OOM kill
- Audit `worker.rs`: panicking worker thread
- Fix gaps → `TierBResult::Error(reason)` or `TierBResult::Timeout`
- Tests via `mock_chelis_beacon`

**Tests:**
- Beacon crash → "beacon crashed (signal N)"
- Partial output + die → "incomplete beacon response"
- No response within timeout → "budget exhausted"

**Acceptance:** `cargo test -p chelis-prove` green including new death-path tests.

---

### Task 3: Eval-side package import resolution — probe against v0.16.1

**Objective:** Re-probe the shell-to-shell import blocker against v0.16.1.
Report what is true before building.

**Implementation:**
- Construct minimal repro: two shells, shell B imports from shell A
- Run `chelis eval` against v0.16.1 binary
- Document verdict: blocker holds or resolved

**Acceptance:** Documented report with exact commands and output.

---

### Task 4: Tracker hygiene — create upstream issue entry

**Objective:** Give the BS deferral gate an entry in the canonical tracker.

**Implementation:**
- The former `docs/issue_drafts/fuzz_sampler_transcendental_cost.md` draft was
  superseded by [chelis#659](https://github.com/Chelis-Lang/chelis/issues/659)
  and removed after that issue closed.
- Status, gated invariants, cross-references

**Acceptance:** Entry exists with current status and gated capabilities.

---

### 🔴 Milestone 1 Red Team Gate

Fresh-context subagent validates Tasks 1–4.

---

### Task 5: Cut capabilities-honesty release

**Objective:** Release carrying capabilities fix + perf + timeout fixes.

**Implementation:**
- Verify `prove_capabilities()` accuracy
- CHANGELOG update, version bump, build, tag
- Verify released binary's `--capabilities` output

**Acceptance:** Tagged release binary with correct capabilities JSON.

---

### Task 6: Quantile primitive

**Objective:** Build `quantile` for VaR and quantile-coherence properties.

**Implementation:**
- `quantile(xs, q)` with linear interpolation semantics
- Prover contracts: monotonicity, range boundedness
- Honest limits documented

**Tests:**
- Correctness: standard quantile values
- Prove: monotonicity, boundedness
- Negative: false property correctly disproved

**Acceptance:** `chelis eval` computes quantiles; `chelis prove` establishes contracts.

---

### Task 7: Induction tier for structural recursion

**Objective:** Turn fixed-size proofs into general-size results for CRR/term-structure.

**Implementation:**
- New tier module: structural induction over lattice steps and periods
- Base case + step case → general statement
- Integrate into dispatch

**Tests:**
- Regression: CRR 2-step still works
- New: CRR n-step gets induction proof
- Negative: non-structural recursion falls through
- Adversarial: bad step case rejected

**Acceptance:** `chelis prove` produces `ProofTier::Induction` artifact.

---

### 🔴 Milestone 2 Red Team Gate

Fresh-context subagent validates Tasks 5–7.

---

### Task 8: Final verification against published binary

**Objective:** Full verification against the release binary.

**Acceptance:** Verification report with evidence for all seven items.
