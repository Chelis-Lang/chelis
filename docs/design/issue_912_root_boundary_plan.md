# Implementation Plan — GH Issue #912: Authoring What a Top-Level Root IS

## Problem Statement

Nothing authoritatively defines what a top-level root IS at the observation
boundary. Five faces of one missing decision: silent root disappearance (#820,
#947), naming inconsistency, artifact/lane/precision routing via fragile
syntactic predicate, and dead-library-node symbolic-dim errors (#848).

## Verified Behavior (executed against `origin/main`)

| Program | `chelis eval` | `chelis build --target c` |
|---|---|---|
| f64 root, no print | ✓ (f64, tensor eval) | hard error ("only supports f32/bool/bf16/f16") |
| f64 root + print | ✓ (f64) | ✓ (f64, host lane) |
| f32 root, no print | ✓ (f64 intermediate) | "Compile object" — no main, root invisible |
| f32 root + print | ✓ (f64 intermediate) | ✓ (C `float`) — `0.010000001` |

Cross-lane numeric difference (eval `0.010000000000000002` vs C `0.010000001`
for f32-declared tensors) is the eval-f64-intermediate vs C-f32-float gap —
[05-OBS-3]'s domain, not this plan's.

## Design Decisions

1. **Builtin `Realizability`**: `Universal`, `HostOnly`, `TensorAtTensorType`.
   Required struct field on `BuiltinDecl`. `trybuild` compile-fail test. No
   `Default` on the type.

2. **Precision-capability at def level, from backend.** Each backend declares
   `TENSOR_CAPABLE_PRIMS`. Eval is a target with its own set (includes f64).
   The def-level check compares declared return type + intermediates against the
   target's capability. Precision is NOT a builtin variant — it is a fact about
   the type and the backend.

3. **`chelis eval --target <T>` is a deliverable.** Bare `chelis eval` uses
   eval's own target (full capability). `chelis eval --target c` manifests under
   C's capability set. #763's cross-lane comparison runs eval manifested for the
   target being compared.

4. **`Target` enum in `chelis-types`. Capability mapping in `compiler-api`.**
   The mapping is a wildcard-free enum match — adding a `Target` variant without
   a capability arm fails to compile. Lives in compiler-api because that is where
   the pipeline is driven and where library consumers (C Proof notebook, etc.)
   obtain manifests.

5. **`ManifestedProgram` carries `Target`.**
   ```rust
   pub struct ManifestedProgram {
       pub checked: CheckedProgram,
       pub manifest: RootManifest,
       pub target: Target,
   }
   ```

6. **`requires_main()` is a method**: `!self.entries.is_empty()`. Empty = object.
   All-[05-UNS-1] = runnable artifact printing summary.

7. **`[05-UNS-1]` is a realizability-defect alarm.** On a healthy corpus it
   never fires. Suggested-fix on the diagnostic: "file a bug against the
   realizability inference." Backend rejection paths (e.g. "only supports
   f32/bool/bf16/f16") are wired to emit `[05-UNS-1]` with
   `PrecisionExceedsCapability`, so declaration-only backends (HIP, Metal)
   self-check in the field.

8. **Byte-identity per target.** For a given target T, eval-manifested-for-T and
   T's artifact agree. Golden gate until #732 Phase 2 delivers
   `chelis_format_shortest`.

9. **Sequencing: three fixes gated on structural work.** Requires tracker
   ratification. CI binary upload lands as a separate prerequisite PR.

10. **Lane-migration diff is per-target.** f64-returning defs changing under C
    but not eval is expected, classified separately from real reclassifications.

11. **Backend capability tests scoped to CI-runnable.** C and eval tested via
    compile+run. HIP and Metal: declared from hardware spec, verified in the
    field via the [05-UNS-1] wiring (Decision 7). Integer capability verified by
    execution before fixing the C constant.

12. **vrk210 gate with fallback.** If confirmation has not arrived by red-team
    milestone 1, author [05-OBS-7] for the verified case (hard error → success),
    note what it does not cover, and file the residue as its own issue.

13. **`RootEntry.reasons`** records why the def routes Host (routing rationale).
    The [05-UNS-1] diagnostic carries its own failure reason separately (why
    realization failed). Different facts, different fields.

14. **Effects and realizability share one call-graph build** so they cannot
    disagree about it.

## Coordination

- **#733** (spec provenance): manifest machinery binds to atoms.
- **#883** (typed diagnostics): [05-UNS-1] carries typed fields.
- **#729** (dtype): precision upstream; #729 Phase 4 downstream.
- **#732 Phase 2**: delivers `chelis_format_shortest`. This plan delivers the
  format specification and manifest ordering it targets.

## Empirical Measurement Results (Task 1, executed against origin/main)

### Within-lane numeric stability (C lane, host-C vs DAG-C)

Tested: `sum(a, 0)` over 1000 × 0.1f32 values.
- Host-C path (forced via `print` dep): `100.00024` (shortest round-trip of f32)
- DAG-C path (object + driver link): `100.00023651123047` (full f64 repr of same bits)
- **Result: BIT-IDENTICAL f32 values.** The apparent difference is rendering only.

Conclusion: within the C lane, routing between host-C and DAG-C does NOT change
numerics at declared precision. Both paths produce the same f32 bits for the same
computation. Invariant 4 is confirmed.

### Integer capability (C DAG path)

Tested: `add(a, b)` over int32 and int64 tensors.
- Both compile and produce correct C code (`int32_t`, `int64_t` operations).
- The C backend's rejection message ("only supports f32/bool/bf16/f16 tensors,
  plus int32/int64 when consumed as sparse indices") is **outdated** — the actual
  behavior supports int32/int64 for general tensor ops.
- C backend `TENSOR_CAPABLE_PRIMS` = `{F32, Bool, Bf16, F16, Int32, Int64}`.
  Only F64 is actually rejected.


## Task Breakdown

### Task 1: Branch, repros, within-lane measurement, tracker reconciliation

- Branch `fix/issue-912-root-boundary`.
- `crates/chelis-cli/tests/issue_912_root_boundary.rs`: three failing repros
  (#848, #947, #820) + completeness stubs + precision-determinism +
  cohabitation-independence tests (all red initially).
- **Within-lane measurement** (against origin/main): f32 reduction-heavy program
  compiled via DAG path (object). Write a C driver that links the object, calls
  the exported global, prints the result. Compare against the same program
  through host-C path. Also measure eval-manifested-under-C-target vs
  eval-manifested-under-eval-target for f64. Document bit-level or tolerance.
- **Integer capability verification**: compile a boundary program with int32 and
  int64 tensors through the C DAG path. Confirm whether they lower or reject.
  This determines the C capability constant in Task 2.
- Tracker: comment on #912 confirming repro shape with vrk210, requesting
  sequencing ratification.
- Exit: repro tests red, measurements documented, tracker comment posted.

### Task 2: Backend-declared capability sets + eval target + `Target` enum

- `Target` enum in `chelis-types/src/types.rs`: `Eval`, `C`, `Hip`, `Metal`.
- Capability constants per backend crate:
  - `chelis-backend-c`: determined by Task 1's integer verification.
  - `chelis-backend-hip`: `&[F32, F64, Bool, Bf16, F16, Int32, Int64]` (from
    hardware spec + local HIP environment).
  - `chelis-backend-metal`: declared from Metal spec (no f64).
  - `chelis-ir` (eval): `&[F32, F64, Bool, Bf16, F16, Int32, Int64]`.
- Target-to-capability mapping in `chelis-compiler-api`: wildcard-free match over
  `Target`. Adding a variant without mapping → compile error.
- Tests: C and eval constants match actual rejection behavior (compile+run).
  HIP/Metal verified in the field via [05-UNS-1] wiring (Task 10).

### Task 3: `chelis eval --target` flag (argument plumbing)

- Add `--target <T>` to `chelis eval`. Default: `Target::Eval`.
- Lands as argument plumbing only — parsed and stored. Wired to the manifest
  pipeline at Task 7 when realizability and manifest machinery exist.
- Exit: flag parses, help text updated, stored on the eval request struct.

### Task 4: Consolidate builtins into `BuiltinDecl` (standalone reviewable diff)

- `BuiltinDecl { name: &'static str, realizability: Realizability, shape_class: ShapeClass }`.
  Required fields. No `#[derive(Default)]`.
- `enum Realizability { Universal, HostOnly, TensorAtTensorType }`.
- `pub const BUILTINS: &[BuiltinDecl]` replaces `BUILTIN_NAMES` + `shape_class()`.
- `trybuild` compile-fail test: BuiltinDecl literal missing `realizability` →
  expected compile error.
- Shape-class lookup backed by `HashMap` for hot-path performance.
- Exit: single source of truth, compile-fail test green.

### Task 5: Const tag table `KNOWN_TAGS` in `chelis-types` (standalone reviewable diff)

- `TagDecl { tag: &'static str, lane_contribution: LaneContribution }`.
- `enum LaneContribution { ForcesHost, Propagates }`.
- Both realizability walker and lowering walker look up from `KNOWN_TAGS`.
- Unknown tag → `Host` + diagnostic ("unrecognized form routes host; add to
  KNOWN_TAGS if intentional").
- Drift test: every tag the lowering walker handles is in `KNOWN_TAGS`.
- Exit: one table, both walkers, drift test green.

### Task 6: Realizability inference

- `Lane` enum in `chelis-types/src/types.rs`: `Tensor`, `Host`.
- `chelis-effects/src/realizability.rs`:
  ```rust
  pub struct RealizabilityResult {
      pub lane_by_def: BTreeMap<String, Lane>,
      pub required_inputs_by_def: BTreeMap<String, BTreeSet<String>>,
      pub reasons_by_def: BTreeMap<String, Vec<HostReason>>,
  }
  pub enum HostReason {
      HostOnlyBuiltin { name: String },
      ScalarTypedOp { builtin: String },
      PrecisionExceedsCapability { prim: Prim },
      StructuralForm { tag: String },
      TransitiveCaller { callee: String },
      UnrecognizedTag { tag: String },
  }
  pub fn infer_realizability(
      program: &CheckedProgram,
      target_prims: &[Prim],
  ) -> RealizabilityResult
  ```
- Def-level precision check: return type + intermediates vs `target_prims`.
- Builtin resolution via `BUILTINS` lookup. `TensorAtTensorType` + scalar
  resolved type → Host with `ScalarTypedOp`.
- Tag resolution via `KNOWN_TAGS`. Unknown → Host + diagnostic +
  `UnrecognizedTag`.
- Reference closure for required-inputs (free tensor-typed variables,
  accumulated transitively).
- Effects and realizability share one call-graph build.
- Test: f64 def with Universal ops → Host for C, Tensor for eval. Full matrix.
- Exit: correct per-target classification.

### Task 7: `RootManifest` and `ManifestedProgram`

- ```rust
  pub struct RootManifest { pub entries: Vec<RootEntry> }
  impl RootManifest {
      pub fn requires_main(&self) -> bool { !self.entries.is_empty() }
  }
  pub struct RootEntry {
      pub name: String,
      pub def_name: String,
      pub ty: Expr,
      pub lane: Lane,
      pub required_inputs: BTreeSet<String>,
      pub reasons: Vec<HostReason>,
  }
  pub struct ManifestedProgram {
      pub checked: CheckedProgram,
      pub manifest: RootManifest,
      pub target: Target,
  }
  ```
- Expanded entries (tuple/ADT dotted names) inherit lane/inputs/reasons from
  `realizability.lane_by_def[def_name]` keyed on originating def.
- Pipeline: types → effects → linearity → realizability(target) → manifest →
  ManifestedProgram.
- Wire `chelis eval --target` (Task 3) to pass target through to this pipeline.
- Exit: target-carrying manifest, phase-enforced.

### Task 8: Delete syntactic predicate + per-target lane-migration diff

- Delete `expr_requires_host_runtime_with_ctx`, `def_is_lowered`,
  `def_body_requires_host_runtime`. All call sites found by compiler error.
- `requires_main()` method replaces three entry-point mechanisms.
- **Per-target migration diff** (`scripts/lane_migration_diff.py`): compare old
  predicate (snapshotted before deletion) vs new inference for each target across
  corpus. Classify:
  - f64 defs → Host under C, Tensor under eval = expected.
  - Integer defs reclassified = per Task 1's verification.
  - Other changes = bug-fixed or regression (investigate before proceeding).
- Exit: functions gone, migration classified, no unintended regressions.

### Task 9: Fix #848 — manifest inputs filter + observe-only equality log

- `infer_symbolic_bindings_from_inputs` takes `required: &BTreeSet<String>` and
  skips bindings whose canonical Load label is not in the set.
- Equality check against post-lowering/pre-optimization DAG Loads: **observe-only
  first**. Log disagreements across corpus. Classify (materialized constants,
  static branches, actual closure bugs). Tighten to assertion where equality
  holds; document exceptions.
- Required-inputs stored as `BTreeSet<String>` (normalized, deduplicated).
- Exit: #848 repro green. Disagreement log classified.

### Task 10: Fix #947/#946 — manifest-driven surfacing, both lanes

- Both lanes iterate manifest entries. For each: render value or emit [05-UNS-1].
- C lane: [05-UNS-1] emitted at build time on stderr as structured diagnostic
  (per #883: `{ root_name, lane, reason, suggested_fix }`).
- All-unsupported artifact's `main` prints summary to stdout.
- Completeness test: `(build stderr ∪ artifact stdout) ⊇ manifest entries`.
- `lookup_runtime_value_for_root` extended for arrow-form (nullary thunk) defs.
- **Backend rejection → [05-UNS-1] wiring**: the C backend's existing "only
  supports f32/bool/bf16/f16" rejection is wired to emit `[05-UNS-1]` with
  `PrecisionExceedsCapability`. This makes declaration-only backends (HIP, Metal)
  self-check in the field — any mismatch between declared capability and actual
  backend behavior surfaces as a structured diagnostic rather than a raw error.
- **Fault-injected [05-UNS-1] case** (in-tree test only): a test uses a
  test-only realizability override that forces a lane mismatch so the backend
  rejection fires. Exercises the diagnostic shape and the unsupported path.
- Exit: #947 green. Fault-injected diagnostic fires. Backend rejection unified.

### Task 11: Fix #820 — verification (no new code)

- `concat` is `HostOnly` in its `BuiltinDecl`. Transitive inference propagates.
  Manifest classifies as `Lane::Host`. Pipeline realizes through host runtime.
- Exit: #820 repro green.

### Task 12: Red-team milestone 1

Fresh-context adversarial review. Probes:

- (a) `BuiltinDecl` missing field → compile error (`trybuild`).
- (b) `Target` variant without capability mapping → compile error.
- (c) Unknown tag → Host + diagnostic.
- (d) Tag in lowering not in `KNOWN_TAGS` → drift test failure.
- (e) f64-returning def with Universal-only ops → Host for C target, Tensor for
  eval target.
- (f) `chelis eval --target c` on f64 program → Host lane → numeric agreement
  with `chelis build --target c` (scoped to numeric values, not rendering
  byte-identity which is #732 Phase 2's golden gate).
- (g) Three bug repros green.
- (h) Both-lane completeness on corpus (per target).
- (i) Cohabitation independence: same root, different cohabiting defs, same value.
- (j) Lane-migration diff reviewed: all per-target changes classified.
- (k) Input-closure observe-only log: disagreements classified.
- (l) Within-lane measurement: documented result (bit-level or tolerance).
- (m) Fault-injected [05-UNS-1]: fires correctly, diagnostic shape correct.
- (n) Empty manifest → no main. All-UNS-1 → runnable main.
- (o) `ManifestedProgram` carries target, prevents pre-manifest access.
- (p) Backend capability constants match actual rejection for CI-runnable targets.
- (q) Backend rejection path emits [05-UNS-1] (not raw error).

**vrk210 gate fallback**: if confirmation has not arrived by this milestone,
author [05-OBS-7] for the verified case, note what it does not cover, file
residue as own issue.

### Task 13: [05-OBS-6] always-labelled + format specification

- Delete `if result.roots.len() == 1` bare branch. All roots: `name = value` in
  manifest entry order.
- Format specification extending §8.1: manifest order, `name = ` label, bare
  scalar ([05-OBS-4]), 32-element truncation ([05-OBS-5]), Rust `{:?}` number
  grammar.
- **Expectation sweep is a separate commit**, reviewable independently. Each
  changed assertion verified to not absorb a real bug.
- Byte-identity: golden gate until #732 Phase 2.

### Task 14: Spec authoring

- [05-OBS-6]: "Every root SHALL render with a `name = value` label at every exit
  in both lanes."
- [05-OBS-7]: per vrk210 gate result. Verified case: "A root's declared type's
  precision determines what both lanes target. A lane that cannot target the
  declared precision SHALL NOT realize the root."
- §8.2 "Root Manifest": completeness per lane per target.
  `realized ∪ [05-UNS-1] = manifest`. `requires_main()` = non-empty.
  All-UNS-1 = runnable. Empty = object.
- [05-UNS-1] for this class: realizability-defect alarm. Suggested-fix: "file a
  bug." Distinct from `RootEntry.reasons` (routing rationale).
- Cross-lane intermediate precision: [05-OBS-3]'s domain, stated honestly.
- Within-lane routing stability: per Task 1's empirical result.
- Coordination: #733, #883, #729, #732 Phase 2.

### Task 15: Red-team milestone 2 — against CI-uploaded artifact

- Prerequisite: separate CI-upload PR has landed.
- Run manifest-completeness against uploaded release artifact on full corpus.
- Per-target: eval-target completeness + C-target completeness.
- Precision determinism: f64 roots route Host under C, Tensor under eval.
- Golden gate: observation_roundtrip_harness at full coverage.
- Absence: deleted predicates gone.
- Empty-manifest edge case against release binary.
- (Fault-injected and all-UNS-1 cases stay in-tree only — test-only hook not in
  release binary.)

### Task 16: PR, self-review, merge

- `gh pr create` with structured description: summary, per-target byte-identity
  honesty, golden-gate scope, coordination handoffs, sequencing ratification
  reference, within-lane measurement result.
- **Self-review checklist** (promote to PR template for root-semantics area):
  - No routing-context name matches beyond `BuiltinDecl`/`KNOWN_TAGS`.
  - `chelis-types` doesn't import `chelis-ir`.
  - Expectation sweep is a separate diff.
  - Migration diff classified per target, no unintended regressions.
  - Target-to-capability match is wildcard-free.
  - `ManifestedProgram` carries target.
  - [05-UNS-1] path tested (fault injection).
  - Backend rejection emits [05-UNS-1].
  - Effects and realizability share one call-graph build.
  - Integer capability verified by execution.
- Fix issues. Push. Merge on green gate.
