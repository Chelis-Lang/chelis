# Verification Stack: Handover State

**Date:** 2026-06-29
**Chelis version:** 0.11.1 (main has unreleased work toward 0.12.0)
**Beacon version:** 0.1.6 (accepts WireDag schema v1 and v2)

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
| **Beacon** | `GoalShape::BoxRange` (interval bounds) | SoundOverApproximation | Live via subprocess shim, 4 e2e tests |

### What actually discharges (measured corpus, 12 properties):

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
3. **WireDag** — JSON, schema_version 2 (v2 adds FloorDiv/TruncDiv; Beacon accepts 1+2)
4. **BeaconShim request** — schema_version 1, inline base64 of exact bytes, expected_dag_sha256
5. **CheckReport → Discharge** — mapping table in `docs/design/beacon_subprocess_shim.md` §5

**Current state:** Beacon accepts schema v2. The default CI oracle for
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
- **School cascade:** Blocked on chelis 0.12.0 release (#527 len/index borrow fix)
- **Shoals BS seam:** #506 (scalar WireDag root) blocks scalar-returning pricer entries
- **Discharge attribution:** #496 (canonical evidence key across engines)

### Trust-stack track (business-prioritized):
- **Phase B:** Effect taxonomy expansion (`Network` + `Filesystem`)
- **Phase C:** Capability enforcement (`chelis run --refuse`)

---

## 8. Open issues touching the prove surface

| Issue | Title | Blocks |
|-------|-------|--------|
| #496 | Canonical discharge-attribution evidence key | Evidence schema consistency |
| #506 | WI-3 scalar-returning entry has no WireDag root | Shoals scalar pricer dispatch |
| #507 | Deep (.dp) properties now have direct Tier-B lowering for the supported scalar SMT subset; broader property shapes remain follow-up | Deep-format proofs |
| #434 | SMT cannot lower transcendental finance properties | Direct erf/log in goals |
| #463 | Boolean connective goal-site lowering | `and`/`or` keyword forms |
| #423 | eval does not resolve package imports for standalone files | eval/prove parity |
