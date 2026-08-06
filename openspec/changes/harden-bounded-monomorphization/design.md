# Design: harden-bounded-monomorphization

## Context

`add-bounded-monomorphization` compiles recursive generic host calls through a
memoized specialization state (`MONO_SPECIALIZATIONS` in
`crates/chelis-ir/src/host.rs`): call sites derive a canonical checked type
application, intern `(def, application) → symbol`, and lowered definitions
accrete into `state.functions`, which `lower_host_program` appends to
`host.functions` before the refinement fixpoint.

PR #1202 solved the same issue independently and ran an adversarial review over
the same architecture shape. Each confirmed finding was checked against our
tree; all four apply:

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
- **Symbolic dims.** Concreteness gating is `HostTypeTerm::is_unresolved`;
  `tensor[n, f32]` is a resolved *term*, dims are not term slots, and nothing
  substitutes them, so a dim disagreement reaches C ABI projection as an
  internal error. The agreeing symbolic-dim payload case (coral's `Hamt` at
  `tensor[n, f32]`) is unexercised.
- **Key and identity hygiene.** The interning key is
  `format!("{name}\u{1}{param_tys:?}\u{1}{ret_ty:?}")` — Rust `Debug` output,
  stable only by accident of derive shape. The symbol is a 16-hex FNV-1a of that
  key with no collision detection. The key's `name` is the program's spelling,
  so a reef'd qualified spelling and a short spelling of one def could mint
  duplicate specializations.

## Goals / Non-Goals

**Goals:**

- Byte-identical emitted C across repeated builds of one program, including
  under the probe-trigger shape; speculative lowering leaves no trace on
  specialization state.
- Specialized symbols are compiler-internal: absent from the published header,
  never eligible as the preferred tensor entry.
- Symbolic-dim disagreement rejects loudly at the call site under the existing
  `[05-UNS-1]` brand; agreeing symbolic-dim payloads compile and match eval.
- One specialization per (canonical callee identity, canonical instantiation);
  hash collisions are loud; the canonical key is purpose-built, not `Debug`.

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

### D2: Recognition by symbol form at the existing chokepoints

A predicate `is_monomorphized_specialization(name)` matches the exact minted
form (`__mono_` followed by 16 lowercase hex digits, suffix-anchored) — strict
enough that user-authored snake_case Surf identifiers cannot collide with it.
It is applied at exactly the surfaces where a specialization must be invisible:
`preferred_tensor_entry_name`, `function_has_tensor_signature`, and the
published-header emission in `host_emit.rs`. The `.c`-internal prototype pass
stays unfiltered so a specialization may call a function declared later.
**Alternative considered:** a structured flag on `HostFunction` instead of name
recognition. Cleaner in principle, but the header emitter consumes the
ABI-projected program where provenance is already erased; threading a flag
through projection is a wider change for the same observable behavior. The
mangling scheme and the predicate are locked to each other by a unit test.

### D3: Call-site shape comparison for symbolic dims

Before interning, compare each instantiated parameter type and the result type
against the types the call actually supplies, tensor shapes included. The first
disagreement rejects with the branded `[05-UNS-1]` diagnostic naming the
instantiated type and the supplied type (the chelis#730 unresolved-term shape's
tensor-shaped sibling: the terms resolved, so the diagnostic names two types
rather than a free variable). Dim *equality* — literal or symbolic — proceeds,
so a `tensor[n, f32]` payload monomorphizes; the specialization simply carries
the symbolic dim the way every non-generic host function already does.
**Alternative considered:** substituting dims through specialization (making
dims term slots). That is real machinery for a case the comparison handles by
construction, and it would fork the dim story from the rest of host lowering.

### D4: Canonical key, canonical identity, loud collisions

The interning key becomes a purpose-built canonical rendering of
`(callee identity, param types, ret type)` with a stability test, replacing
`{:?}`. The callee identity component is the package-internal name (the
`pkg__<pkg>__<Module>__<def>` form reef lowering already uses), so qualified
and short spellings intern the same specialization and the minted symbol is
`<package-internal-def>__mono_<hash>`. A reverse map `symbol → canonical key`
is consulted at minting: the same symbol arriving for a different key is an
internal error (an FNV-1a collision would otherwise collapse two
instantiations into one C definition, invisible to the emitter's
duplicate-name check). **Alternative considered:** widening the hash or using
a cryptographic hash. The collision map is exact, free at this scale, and
keeps symbols short.

### D5: Oracle stays singular

All new scenarios join the existing owning suite
(`cargo nextest run -p chelis-cli --test recursive_generic_monomorphization
--no-fail-fast`), which remains the one acceptance oracle. The byte-determinism
test supersedes the set-equality test (the latter is strictly weaker) and must
be mutation-verified once during development: with D1's guard disabled, the
test fails on the trigger shape.

## Risks / Trade-offs

- [Determinism test passes coincidentally] → construct the fixture from the
  measured trigger (probing caller preceding its wrapper defs in source);
  verify by mutation (guard off ⇒ red) before trusting green; compare whole
  emitted `.c` bytes across ≥10 builds, not symbol sets.
- [Snapshot cost] → the state is small (a map, a vec, a stack) and probes are
  per-callee-name, already memoized upstream; clone cost is noise against the
  lowering it wraps.
- [Header filtering breaks a consumer] → nothing outside the translation unit
  can legitimately name a hash-suffixed internal; the filter is
  behavior-preserving for every documented consumer. The link-clean scenario
  guards the `.c`-internal forward-reference path.
- [Identity normalization changes emitted symbol names] → symbol names for
  reef'd programs change once (they were per-spelling before); CHANGELOG notes
  it; no stability promise existed for internal symbols.
- [Two active changes delta one capability] → this change declares the
  dependency on `add-bounded-monomorphization` and must archive after it;
  `openspec validate --all --strict` is run with both active.

## Migration Plan

1. Failing tests first: byte-determinism (trigger shape), entry-selection
   minimal pair, header exclusion, symbolic-dim negative + positive, reef
   one-symbol-per-type, key/collision unit tests.
2. D1 probe isolation → determinism scenarios green.
3. D2 surface hygiene → entry/header scenarios green.
4. D3 symbolic-dim guard → both dim scenarios green.
5. D4 key/identity/collision → reef and unit scenarios green.
6. Docs (`loud_unsupported.md`, CHANGELOG), gate, red team per protocol.

Rollback: each decision is independently revertible; reverting D1–D4 restores
current behavior with the known defects, never a new silent path.

## Open Questions

- Whether the reef package-internal identity is already available at the
  interning site or must be resolved through the existing reef lowering tables
  (task verifies; expected: available, since the call site lowers a resolved
  program).
- Whether PR #1202's `MonoProbeGuard` implementation can be adapted directly
  (same-repository code, review credit recorded) or is re-derived here; either
  satisfies D1.
