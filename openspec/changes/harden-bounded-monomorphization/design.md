# Design: harden-bounded-monomorphization

## Context

`add-bounded-monomorphization` compiles recursive generic host calls through a
memoized specialization state (`MONO_SPECIALIZATIONS` in
`crates/chelis-ir/src/host.rs`): call sites derive a canonical checked type
application, intern `(def, application) → symbol`, and lowered definitions
accrete into `state.functions`, which `lower_host_program` appends to
`host.functions` before the refinement fixpoint.

PR #1202 solved the same issue independently and ran an adversarial review over
the same architecture shape. Its applicable findings started this change. A
fresh local red team then found three more defects in this implementation:

- **Probe reachability.** `top_level_fn_helper_summary_rejects` (host.rs)
  early-returns for a *type-polymorphic* callee, but fully lowers every
  non-polymorphic callee body — and such a body may contain a recursive generic
  call, which interns and pushes a specialized `HostFunction` during the probe.
  The probe's result is discarded; the interning is not. Probes are driven from
  `expr_calls_summary_rejecting_top_level_fn` over a `HashSet` of names, so
  probe order varies per process. Consequences: emission order (completion order
  of `state.functions`) depends on probe order; a probe-only specialization is
  emitted though no surviving call site references it; a probe-only
  specialization that fails to lower propagates `Err` and aborts the build; and
  on that failure path the memo entry survives while the definition was never
  pushed, so a later real call site would receive a symbol with no definition.
- **Determinism test blindness.** The existing
  `specialized_symbol_set_is_deterministic_across_builds` compares
  `BTreeSet`-collected symbol *sets* across two builds. It cannot observe
  emission-*order* divergence or probe-only extra definitions.
- **Entry selection.** `preferred_tensor_entry_name` and
  `function_has_tensor_signature` scan `host.functions` with no exclusion for
  specializations. A specialization whose parameters are tensors is a candidate
  entry; PR #1202 measured the resulting ABI flip (authored 1-in/1-out entry
  displaced by a 1-in/4-out specialization) while the build reported success.
- **Published header.** `emit_host_header` declares every function in
  `host.functions`, so `<def>__mono_<hash>` internals land in the user-facing
  `.h`.
- **Symbolic dims.** `tensor[n, f32]` is a resolved term. The specialization
  receives its types from the checked call site, so no separate dimension
  signature exists. The positive coral-shaped payload case was untested.
- **Key and identity hygiene.** The key used Rust `Debug` text and the call-site
  name. The symbol used a 16-hex FNV-1a hash without collision detection.
  Definition lookup also returned the first terminal-name match before an exact
  package match. Two package definitions named `depth` then shared one symbol.
- **Surface provenance.** A suffix test treated valid authored names as
  specializations. Such an authored function disappeared from its header and
  from entry selection.
- **C linkage.** Object-mode specializations had external linkage. Two generated
  objects with one specialization name failed to link because of a duplicate
  symbol.

## Goals / Non-Goals

**Goals:**

- Byte-identical emitted C across repeated builds of one program, including
  under the probe-trigger shape; speculative lowering leaves no trace on
  specialization state.
- Specialized symbols are compiler-internal. They stay absent from the header,
  entry selection, and the external object symbol table.
- Valid authored names remain authored surface even when their text matches the
  specialization mangling grammar.
- Symbolic-dimension payloads compile and match eval because call-site checked
  types are the specialization types.
- One specialization exists per canonical callee identity and instantiation.
  Exact package identities win before unique terminal-name fallback.
- Hash collisions fail loudly. The canonical key does not use `Debug` text.

**Non-Goals:**

- No caps, no worklist restructure, no checker/eval changes, no change to the
  non-recursive inlining path (all per the proposal).

## Decisions

### D1: Probe isolation by snapshot/restore, not by skipping

A RAII guard snapshots the whole `MonoSpecializationState` (memo, functions,
in-progress stack) at probe entry and restores it on drop, and the probe
name-set is iterated in sorted order. After a probe the state is byte-for-byte
what it was before, so every surviving specialization was interned by the real
lowering pass in that pass's deterministic traversal order — which also fixes
the memo-without-definition failure residue for free. **Alternative
considered:** making probes refuse to enter the mono path (return a placebo
symbol without emitting). Rejected: it changes the probe's answer — a body
whose helper summary depends on the lowered call shape would be probed against
different lowering than the real pass performs. Restore-on-exit keeps probe
fidelity exact. The guard must restore, not clear: a probe can run while a
specialization body is itself being lowered, and that outer in-progress frame
must survive.

### D2: Explicit provenance and local C linkage

`HostFunctionOrigin` records `Authored` or `Monomorphized`. Both host type
projections preserve this value. Header filters and entry selectors inspect the
value instead of the function name. Thus, an authored name can match the private
mangling grammar without loss of public surface.

The C emitter gives each specialization `static inline` linkage in binary mode
and object mode. Authored object-mode functions keep external linkage. The
internal prototype pass includes specializations, so forward references still
compile.

**Alternative rejected:** infer provenance from the symbol suffix. Surf permits
that suffix in authored snake_case identifiers, so text cannot represent the
provenance state.

### D3: Call-site checked types own symbolic dimensions

The specializer derives each parameter type from the checked argument expression.
It derives the result from the checked application. No second instantiated
signature exists at this boundary. Thus, a dimension disagreement is not a
representable specialization state. Symbolic and literal dimensions follow the
same path.

The checker remains the boundary for dimension mismatches that source can
express. A symbolic `tensor[n, f32]` payload specializes and runs with eval
parity. No redundant comparison guard exists in host lowering.

### D4: Canonical key, canonical identity, loud collisions

The interning key uses a purpose-built canonical representation of
`(callee identity, param types, ret type)`. Definition lookup first searches
for the exact package identity. It accepts terminal-name fallback only when one
definition matches. This order keeps two package definitions named `depth`
distinct and lets short and qualified references to one definition share a
specialization.

A reverse map from symbol to canonical key detects a hash collision. A second
key for one symbol produces an internal error before C emission. The exact map
keeps the short FNV-1a symbol format without silent aliasing.

### D5: Oracle stays singular

All user-visible scenarios join the existing owning suite:
`cargo nextest run -p chelis-cli --test recursive_generic_monomorphization
--no-fail-fast`. This suite remains the one acceptance oracle. A supporting
`chelis-ir` seam test forces a lowering error inside `MonoProbeGuard`. It locks
`Ok(false)`, complete specialization-state restoration, and later attribution
of the genuine error. The byte-determinism test supersedes the weaker
set-equality test. A development mutation disables D1's guard and makes the
test fail on the trigger shape.

## Risks / Trade-offs

- [Determinism test passes coincidentally] → construct the fixture from the
  measured trigger (probing caller preceding its wrapper defs in source);
  verify by mutation (guard off ⇒ red) before trusting green; compare whole
  emitted `.c` bytes across ≥10 builds, not symbol sets.
- [Snapshot cost] → the state is small (a map, a vec, a stack) and probes are
  per-callee-name, already memoized upstream; clone cost is noise against the
  lowering it wraps.
- [Header filtering breaks a consumer] → explicit provenance filters only
  generated specializations. An authored mangling-shaped name remains public.
- [Local linkage breaks an internal call] → the unfiltered C prototype pass
  uses the same local linkage as each specialized definition. The object-link
  scenario also checks two generated translation units.
- [Identity normalization changes emitted symbol names] → internal symbol names
  have no stability contract. Package parity tests lock semantic identity.
- [Two active changes delta one capability] → this change declares the
  dependency on `add-bounded-monomorphization` and must archive after it;
  `openspec validate --all --strict` is run with both active.

## Migration Plan

1. Add red tests for determinism, surface hygiene, symbolic dimensions,
   canonical package identity, collision detection, and object linkage.
2. Add D1 probe isolation and run its mutation control.
3. Add D2 provenance filters and local specialization linkage.
4. Lock D3 with symbolic-dimension native and eval parity.
5. Add D4 exact identity resolution, canonical keys, and collision detection.
6. Update the docs, run the gate, and run a fresh local red team.

Rollback: each decision is independently revertible; reverting D1–D4 restores
current behavior with the known defects, never a new silent path.

## Resolved Questions

- Reef lowering supplies the exact package-internal callee identity at the
  interning site. Exact identity lookup must precede terminal fallback.
- The probe guard uses a repository-local snapshot and restore implementation.
  PR #1202 retains credit for the adversarial finding.
