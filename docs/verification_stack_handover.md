# Verification Stack: Handover State

**Date:** 2026-07-06
**Chelis version:** 0.14.0 (`origin/main` at `3a7e1932` after #622 for this handover polish)
**Beacon version:** 0.1.6 was the last verified cross-repo witness; this
polish did not requalify Beacon.

> **Update 2026-07:** A three-wave compiler/prover **Soundness hardening** campaign closed since
> this doc was written — see §9. It resolved #496 and #463 (both dropped from §8 below).

This document is the single-source handover artifact for the verification
orchestrator. A newcomer reads this to understand what the stack proves today,
what falls to fuzz and why, and where the frontier is.

---

## 1. What the stack proves today

The verification orchestrator dispatches proof goals through a heterogeneous
engine portfolio, recombines under a qualifier-set lattice, and renders an
honest composite verdict. The engines:

| Engine | Goal shape | Guarantee | Status |
|--------|-----------|-----------|--------|
| **cvc5** | `GoalShape::Smt` (polynomial/logical NRA) | Exact | Live, release binary |
| **Z3** | `GoalShape::Smt` (polynomial NRA fallback) | Exact | Live, try-until-discharge after cvc5 |
| **Clarabel** | `GoalShape::Smt` (univariate poly ≥ 0 on interval) | CertificateBearing | Registered, no standalone integration test |
| **Carcara** | Post-discharge audit of cvc5 Alethe proofs | Audit evidence (not a verdict) | Live, 15 tests (requires `--features carcara`) |
| **Beacon** | `GoalShape::BoxRange` (interval bounds) | SoundOverApproximation | Live via subprocess shim; ignored live e2e gate |

### What actually discharged in the last measured corpus (12 properties):

- **Proved (25%):** Polynomial structural claims — Shoals put-call parity,
  call ≤ spot, delta ∈ [0,1]. Discharged by cvc5 over the reals, with fuzz-
  validated contracts for the normal-CDF abstraction.
- **Disproved (8%):** Deliberately corrupted properties (test coverage).
- **Falls to fuzz (67%):** Behavioral properties with opaque-type invariants.

### The fuzz bucket, differentiated:

| Cause | Count | What blocks |
|-------|-------|-------------|
| Shape-blocked | 1 | Opaque type not supported in the L2 v1 property-lowering surface |
| Generator starvation | 2 | Exact-equality predicates starve the fuzz sampler by design |
| No deductive path | 5 | Properties pass fuzz but have no SMT/Beacon route |

---

## 2. Architecture in one paragraph

A `@property` declaration is discovered in Surf/Deep source, lowered to a
`Goal` (either `GoalShape::Smt` for logical/polynomial claims or
`GoalShape::BoxRange` for interval bounds), dispatched through the
`DischargeRegistry` to the first engine whose `fitness(goal)` returns true,
and the resulting `Discharge` (carrying `Soundness`, `QualifierSet`, and
`TierBResult`) is composed into the final `CompositeVerdict`. The transformation
layer can rewrite goals before dispatch (abstract-subterm, goal-splitting).

---

## 3. The integrity core (honesty layer)

These are the non-negotiable invariants. Each has a named test:

1. **No laundering:** `recombine_split_discharges` rolls up at minimum
   soundness. A Beacon `SoundApproximate` never reads as `Exact`.
   Test: `goal_split::row2_mixed_soundness_returns_min_soundness`

2. **Non-vacuity mandatory:** A green verdict carries a non-vacuity record.
   Precondition-unsatisfiable poisons the composite.
   Test: `obligation_engine` non-vacuity enforcement paths

3. **Float-vs-real hedge:** SMT proofs over reals carry `RealArith` qualifier,
   rendering as `proven_modulo_real_arithmetic` (not plain `proven`). Disproved
   carries the hedge too.
   Test: `cross_engine_oracle` + Shoals composites output

4. **Per-assumption provenance:** Each assumption in the artifact carries
   `discharge_tier` (which engine, what guarantee). The Discharge is built
   only through `Discharge::new` (the integrity gate).

---

## 4. The seam (chelis ↔ Beacon)

Five frozen surfaces, documented in `docs/design/phase2_seam_contract.md`:

1. **Goal** — `GoalShape::BoxRange { inputs: IntervalBox, output: OutputRange }`
2. **IrHandle** — sha256 of serialized WireDag bytes + root_index (no live IR)
3. **WireDag** — JSON, schema_version 3 (v2 adds FloorDiv/TruncDiv, v3 adds runtime Reshape extents; Beacon accepts 1+2+3)
4. **BeaconShim request** — schema_version 1, inline base64 of exact bytes, expected_dag_sha256
5. **CheckReport → Discharge** — mapping table in `docs/design/beacon_subprocess_shim.md` §5

**Current state:** chelis emits schema v3; Beacon accepts v1, v2, and v3 and
fails closed above. The default CI oracle for
`BeaconShim` remains the mock suite, while `beacon_e2e` is the ignored live
cross-repo gate: with `CHELIS_BEACON_BIN` pointing at an Arb-enabled
`chelis-beacon`, it exercises the real shim, real Beacon binary, Chelis-produced
WI-3 bytes, Beacon-side exact-byte evidence, and negative non-proof cases. Build
Beacon with `--features arb-oracle` for verified dispatch.

---

## 5. The transformation layer

| Component | Status | Tests |
|-----------|--------|-------|
| `Transformation` trait + `TransformationPipeline` | Shipped | 25 tests |
| Soundness harness | Shipped | Catches laundering, vacuous-empty |
| Abstract-subterm (erf envelope → polynomial) | Shipped | 10 tests, range-sound |
| Goal splitting + lattice recombination | Shipped | 11 tests, exhaustive matrix |

**Key design facts:**
- The harness validates before any transformation ships (spec §5 discipline)
- Abstract-subterm evaluates the envelope over the argument's RANGE (not endpoints)
- Goal splitting recombines at min-soundness; Disproved carries its qualifier
- Vacuity is precondition-level (checked once), not per-conjunct

---

## 6. Build and run

```sh
# Default (solver-free) build
cargo build --workspace

# SMT build (cvc5 linked)
cargo build --workspace --features smt

# Full engine features (Z3 requires libz3-dev / z3-devel system package)
cargo build --workspace --features "smt z3 clarabel carcara"

# Run prove with real engines
cargo run -p chelis-cli --features smt -- prove --json path/to/file.ch

# Beacon e2e (requires Arb-enabled beacon binary)
CHELIS_BEACON_BIN=/path/to/chelis-beacon cargo test -p chelis-prove --test beacon_e2e -- --ignored --nocapture
```

**Toolchain:** `stable` (pinned in `rust-toolchain.toml`, no version lock).
**Python:** `.venv/bin/python` (uv-managed, 3.11+). Required for PyO3 link step.
**Prerequisites:** See `README.md` §Prerequisites.

---

## 7. What's next (the frontier, not broken)

### Reach expansion (what would make more goals discharge):
- **L2 v1 type-support expansion:** Opaque types in the property-lowering path
  (the shape-blocked goals need this)
- **Beacon routing for behavioral properties:** Wire the opaque-invariant
  properties through BoxRange extraction → Beacon
- **More transformations:** Controlled inlining, compute/partial-eval,
  construct-elimination (each harness-gated)
- **Driver-policy fitting:** Mechanical search over transformation sequences
  on the corpus (deprioritized — marginal until more goals discharge)

### Infrastructure:
- **Clarabel integration test:** Registered but unexercised end-to-end
- **WI-17 MetiTarski:** Optional shell engine, low-reach
- **WI-18/19/20:** Build features, vendor config, deployment surface

### Cross-repo:
- **School cascade:** Unblocked — the #527 len/index borrow fix shipped in 0.12.0; the downstream
  cascade into School is the remaining work.
- **Shoals BS seam:** #506 (scalar WireDag root, reopened) — settled as **Option B**: a Shoals
  *tensor* entry is the seam; scalar host entries stay non-root, no chelis lowering change.
- **Discharge attribution:** #496 — **resolved** (uniform top-level `engine` key, #564); see §9.

### Trust-stack track (business-prioritized):
- **Phase B:** Effect taxonomy expansion (`Network` + `Filesystem`)
- **Phase C:** Capability enforcement (`chelis run --refuse`)

---

## 8. Open issues touching the prove surface

(#496 and #463 were **resolved** in the 2026-07 soundness campaign — see §9 — and removed from this list.)

| Issue | Current state | Blocks |
|-------|-------|--------|
| #506 | WI-3 scalar-returning entry has no WireDag root (reopened) | Shoals scalar pricer dispatch (Option B: tensor entry is the seam) |
| #507 | Deep (.dp) properties have direct Tier-B lowering only for the supported scalar SMT subset; broader property shapes remain follow-up | Deep-format proofs |
| #434 | Transcendental-discharge capability (Black-Scholes positivity via log/exp/sqrt) — rescoped; internal-message leak fixed + honest-Unsupported locked (#564); exp/log/sqrt envelope is the remaining reach | Direct transcendentals in goals |
| #423 | eval does not resolve package imports for standalone files | eval/prove parity |

---

## 9. Soundness hardening (closed 2026-07)

A three-wave campaign hardened the compiler→IR→goal→discharge trust chain: the prover's
"proven, not tested" claim is only as sound as the lowering beneath it, so every silent-miscompile and
check-vs-eval-vs-backend gap in the backlog was closed with a corrected result or a fail-loud reject,
each behind a fresh-context red-team gate. **Structure:** Wave 1 (parallel honest-reject + eval/type
fixes, PRs #562–#567), Wave 2 (sequential AD/symbolic-dim grad machinery, #590), Wave 3 (Form-3 runtime
expand-size, #596). Each red team (RT-1/2/3) EXECUTED adversarial tests and *found a real issue*, all
resolved fail-closed; a phase-wide RT-final returned **PASS** (acceptance oracle green except the
pre-existing `chelisup` `/tmp/reef.toml` env leak; no swallowed reject, no silent-wrong; complete
positive+negative coverage). Documented in the `[0.13.0]` CHANGELOG entry.

**Re-verified on v0.14.0 (2026-07-06): no regression.** After #615/#612/#611 subsequently touched the
hardened files, all phase oracle suites pass (issue_549 14/14, issue_551 7/7, issue_513 3/3 +
`symbolic_axis_adjoints` 19/19, rank_poly_tier3 98/98, chelis-ir 828, chelis-types 777, chelis-backend-c
374, chelis-prove 323, issue_522 10/10) and every fail-closed spot-check still rejects loud.

**Governing discipline for whoever continues this line:** a reject that is swallowed and falls through
to a green result (default-0, hardcoded extent, empty goal, fuzz-green) is the same laundering wearing a
different mask. Every reject introduced here fails LOUD, and the red team specifically exercises each
reject path. Keep that bar.

### Resolved (merged to main)

| Issue | PR | What |
|-------|----|------|
| #549 | #562 | grad by-position named-axis recovery re-validated against a permute; fails loud when the axis is ambiguous |
| #530 | #563 | inline `expand` size (tuple-get / cast / arith / if / match) routes through the Form-3 gate → rejects loud |
| #524 | #563 | `vmap` present-but-non-constant axis rejects loud (was a silent default-to-0) |
| #517 | #565 | C-backend `emit_cmplt` reads each operand's real dtype (was a `float*` reinterpret) |
| #340 | #565 | named-axis max/min/prod/argmax/argmin_reduce in a `..r` body builds + runs (was host-lane unsupported) |
| #522 | #567 | host evaluator accepts negative axes uniformly (closed a check↔eval gap) |
| #550 | #567 | chelis-ir eval traps integer floor_div/trunc_div by zero (fail-closed, matches the C backend) |
| #458 | #566 | mixed (f32,i32) numeric op rejects consistently at check — resolved-by-design (no implicit promotion) |
| #463 | #564 | `&&`/`||` goal-site lowering verified sound + locked (residual: `and`/`or` keyword → #589) |
| #496 | #564 | uniform top-level `engine` discharge-attribution key across all engines |
| #551 | #590 | grad over a symbolic-axis concat builds under `--target c` (was a dag.rs ICE) |
| #513 (gap 1) | #590 | grad through a shape()-derived reshape target (gaps 2/3 → rlronan/#611) |
| #523 | #590 | `eval::shrink` loud bounds asserts + post-bind re-verify |
| #383 | #590 | verify-closed (fixed by #549) + regression-locked |
| #469 | #596 | Form-3 runtime expand-size: shape / let-bound / static-arith resolve correct + byte-deterministic; arith-over-shape rejects loud |

### Open follow-ups (all fail-loud today — none silently wrong)

| Issue | State | Note |
|-------|-------|------|
| **#609** | open, unassigned | **Top item.** Checker accepts a wrong-rank ascription on a Form-3 `expand` result → eval silently returns a contradicting rank. Live silent-wrong in the Form-3 area; overlaps rlronan's active Form-3 track (#611/#469) → hand off there. |
| #593 | open, unassigned | Deeper fix for the concat symbolic-wrapper Pad-output mis-sizing (memory-safety). Interim is a loud fail-closed abort; the real fix sizes the wrapper output correctly or emits a graceful error. |
| #592 | open, unassigned | `vmap(grad(f))(y)` C-build ICE (eval FD-correct); #513-family. |
| #587 | open, unassigned | Deeper #549: track the named-axis anchor *through* a permute (recover, don't reject the square case). |
| #572 / #573 | open, unassigned | Sibling silent-default-0 sites: expand axis-slot (#572), tuple-get index (#573). Same #364/#524/#530 class. |
| #597 | open, unassigned | let-bound-static expand size errors in eval though C build resolves it (checker annotation gap; fails loud). |
| #594 | open, unassigned | concat marks its concat-axis extent symbolic `*` instead of the concrete sum of operands (type-inference). |
| #589 | open, unassigned | chelis-surf `and`/`or` keyword diagnostic (#463 residual; fails safe, usability). |
| #434 | open, jeff | Transcendental-discharge capability (exp/log/sqrt envelope); fail-safe locked (never falsely proven). |
| #513 gaps 2/3 | open, **rlronan (#611)** | Runtime symbolic-value grad adjoints — in progress; do not duplicate. |
| #516 | open, unassigned (parked) | vmap capture-broadcast; overlaps rlronan's reopened #377 (his vmap half) — coordinate before touching. |
| #469 Case 2 | **closed by rlronan** | scalar-param expand size; confirm the capability shipped. |

**Post-phase surface drift:** #615 (error-localization + ConstTensor), #612 (match/ADT grad), and #611
(#513 gap 3) landed after the phase and touch the hardened files — the re-verify above confirms no
regression, but a colleague extending any of these files should re-run the phase oracle suites.
