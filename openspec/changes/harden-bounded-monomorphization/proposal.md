# Proposal: harden-bounded-monomorphization

## Why

PR #1202 is a parallel, independently built implementation of chelis#1158 (bounded
memoized monomorphization for recursive generic host calls). Its adversarial review
pass confirmed four defect classes in that architecture — and our implementation
(`add-bounded-monomorphization`) shares the code shapes that produce every one of
them, unguarded and untested:

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
4. **Symbolic-dimension blind spot.** Concreteness is decided by
   `HostTypeTerm::is_unresolved`, a question about type *terms*: `tensor[n, f32]`
   is a resolved term however its dims are spelled, dims are not type-term slots,
   and nothing substitutes them. A dim disagreement between the interned signature
   and the caller survives to C ABI projection as an internal error instead of a
   diagnosis — and the passing case (a recursive generic instantiated at a
   symbolic-dim tensor payload, the exact coral#26 `Hamt` shape) has no coverage
   at all.

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
- **Surface hygiene:** monomorphized specializations are recognized by their
  symbol form and excluded from (a) the published host header and (b) preferred
  tensor-entry selection and tensor-signature classification. In-`.c` prototypes
  remain emitted over the unfiltered function list so forward references still
  resolve.
- **Symbolic-dimension guard:** before interning, the call site compares the
  instantiated parameter and result types against the types the call actually
  supplies; a tensor-shape disagreement rejects with the branded `[05-UNS-1]`
  diagnostic naming both types instead of dying at C ABI projection. Agreement —
  including a symbolic-dim payload such as `tensor[n, f32]` — monomorphizes and
  runs, with eval/build value parity asserted on a coral-shaped recursive-trie
  fixture.
- **Canonical identity and collision safety:** the interning key becomes a
  purpose-built canonical rendering of the checked signature (replacing Rust
  `Debug` formatting); a reverse map from emitted symbol to canonical key makes
  an FNV-1a hash collision a loud internal error rather than two instantiations
  silently collapsing into one C definition; and the callee identity is
  normalized to the package-internal name so reef'd qualified-vs-short spellings
  of one def intern one specialization per instantiation
  (`pkg__<pkg>__<Module>__<def>__mono_<hash>`).

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

- `crates/chelis-ir/src/host.rs` — probe snapshot/restore guard; sorted probe
  iteration; entry-selection and tensor-signature exclusion; symbolic-dim
  call-site comparison; canonical key + collision map; callee-identity
  normalization.
- `crates/chelis-backend-c/src/host_emit.rs` — published-header filtering
  (specializations omitted from the `.h`; `.c` prototypes unfiltered).
- `crates/chelis-cli/tests/recursive_generic_monomorphization.rs` — the owning
  oracle suite grows byte-determinism, entry-selection minimal-pair,
  header-exclusion, symbolic-dim negative/positive, and reef one-symbol-per-type
  scenarios; the weak set-equality determinism test is superseded.
- `spec/design/loud_unsupported.md` — the delivered-behavior record gains the
  symbolic-dim residue description.
- `CHANGELOG.md` — behavior notes (header no longer declares `__mono_` symbols;
  symbolic-dim recursion now rejects cleanly instead of ICEing). Credit the
  PR #1202 review findings.
- Coordination: PR #1202 edits the same `host.rs` call-site region; the
  disposition of that PR (merge, close in favor, or split) is a maintainer
  decision recorded on chelis#1158, outside this change.
