# Proposal: harden-bounded-monomorphization

## Why

PR #1202 is a parallel, independently built implementation of chelis#1158 (bounded
memoized monomorphization for recursive generic host calls). Its adversarial review
pass confirmed several defect classes in that architecture. Our implementation
(`add-bounded-monomorphization`) shared the applicable code shapes without
guards or tests:

1. **Probe-driven nondeterminism and probe pollution.** The speculative
   summary-rejection probe (`top_level_fn_helper_summary_rejects`) fully lowers
   non-polymorphic callee bodies. When such a body contains a recursive generic
   call, the probe interns and emits specializations into `MONO_SPECIALIZATIONS`
   with no snapshot/restore: probe iteration order (a `HashSet`) can leak into
   emission order (nondeterministic `.c`, the chelis#1002 class), probe-only
   specializations are emitted unconditionally, and a probe-only specialization
   that fails to lower aborts the whole build. Our determinism test compares
   symbol *sets* across two builds and is blind to all three.
2. **Entry-selection displacement.** `preferred_tensor_entry_name` and
   `function_has_tensor_signature` do not exclude `__mono_` specializations, so a
   tensor-signature specialization can displace the authored preferred entry and
   silently flip the program's compiled ABI.
3. **Published-header leakage.** Specializations join `host.functions` before
   emission, so hash-named compiler internals (`<def>__mono_<hash>`) are declared
   in the user-facing `.h`. Their names change whenever the instantiation set
   does; nothing outside the translation unit may call them.
4. **Symbolic-dimension coverage gap.** `tensor[n, f32]` is a resolved host
   type term. A specialization takes its parameter types from the checked call
   site, so no separate dimension signature exists. The positive recursive ADT
   case had no native parity test.
5. **Canonical-identity ambiguity.** The definition resolver accepted the first
   terminal-name match before it searched for an exact package identity. Two
   package definitions named `depth` silently shared one specialization.
6. **Provenance and linkage gaps.** A name suffix acted as function provenance,
   although authored snake_case permits that suffix. Object-mode specializations
   also kept external linkage and collided across generated object files.

This change adopts those findings — the good parts of PR #1202 — into our
architecture, which keeps the check-time uniform-recursive-instantiation rule
([04-INF-2]/[04-INF-3]) as the acceptance boundary.

## What Changes

- **Probe isolation:** speculative lowering becomes side-effect-free on the
  specialization state (snapshot/restore guard around every probe) and probe
  iteration becomes deterministic (sorted). Repeated builds of one program emit
  byte-identical C, asserted on the probe-trigger shape (a probing caller
  preceding its wrappers in source) across repeated builds, with a
  mutation-verified test (guard disabled ⇒ the test fails).
- **Surface hygiene:** each host function carries explicit authored or
  monomorphized provenance through both type projections. The header and entry
  selectors use that provenance, not symbol text. Each specialization receives
  translation-unit-local C linkage in binary mode and object mode. Internal C
  prototypes remain available for forward references.
- **Symbolic-dimension contract:** specialization parameter and result types
  come from the checked call site. Symbolic and literal dimensions use the same
  path. A recursive ADT with a `tensor[n, f32]` payload now has native and eval
  parity coverage.
- **Canonical identity and collision safety:** the interning key uses a
  purpose-built canonical signature format. Definition lookup prefers an exact
  package identity and accepts a terminal-name fallback only when it is unique.
  A reverse symbol map detects FNV-1a collisions and fails loudly. Qualified and
  short references to one definition share one specialization.

Non-goals:

- **No specialization or signature-size caps.** PR #1202 bounds polymorphic
  recursion at lowering with numeric caps; our acceptance boundary stays the
  check-time [04-INF-2]/[04-INF-3] rule (`add-bounded-monomorphization` D2), and
  lowering continues to assume boundedness rather than re-litigate it. A
  checker/lowering disagreement must surface as the defect it is, not be masked
  by fuel.
- **No worklist restructure.** PR #1202 drains a flat FIFO after the main pass;
  our re-entrant lowering is behavior-equivalent under the checker rule and
  switching is churn. Revisit only if native-stack depth ever shows up.
- **No checker or eval changes.** The uniform-recursion rule, its diagnostics,
  and eval-lane behavior are untouched.
- **No numeric-surface change.** Removing internal symbols from the published
  header shrinks exported C surface; nothing new is exported.

## Capabilities

### Modified Capabilities

- `generic-monomorphization`: add the determinism, compiler-internal-symbol,
  symbolic-dimension, and canonical-identity requirements to the capability
  introduced by `add-bounded-monomorphization`. **Depends on that change's
  deltas landing first**; this delta layers on its requirement set.

## Impact

- `crates/chelis-ir/src/host.rs` — probe isolation, explicit function
  provenance, exact identity resolution, a canonical key, and collision checks.
- `crates/chelis-backend-c/src/host_abi.rs` — preservation of function
  provenance through C-ABI projection.
- `crates/chelis-backend-c/src/host_emit.rs` — provenance-based header filters
  and translation-unit-local linkage for specializations.
- `crates/chelis-cli/tests/recursive_generic_monomorphization.rs` — the owning
  oracle adds determinism, surface, linkage, symbolic-dimension, and package
  identity scenarios.
- `spec/design/loud_unsupported.md` and `CHANGELOG.md` — delivered behavior and
  PR #1202 review credit.
- Coordination: PR #1202 edits the same `host.rs` call-site region; the
  disposition of that PR (merge, close in favor, or split) is a maintainer
  decision recorded on chelis#1158, outside this change.
