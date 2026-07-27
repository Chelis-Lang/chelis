## Context

`spec/10-serialization.md` records the settled text forms (`.ch`/`.dp` UTF-8), the deliberately
under-specified `.chb` binary Shell metadata artifact, the additive JSON wire-schema
compatibility policy, and the normative decode-boundary invariant-revalidation rule (with a V1
reality note that no production ADT-value codec exists yet). This change records that content as
a `serialization` capability.

## Goals / Non-Goals

**Goals:**
- Capture the text forms, `.chb` role, wire-compatibility policy, and the decode-boundary
  invariant-revalidation contract as SHALL requirements with positive and failure scenarios.

**Non-Goals:**
- Specifying the `.chb` low-level wire layout, which the source explicitly keeps
  implementation-owned and unfrozen.
- Restating the full invariant well-formedness grammar (owned by `type-system` §2.5.1); the
  decode requirement references it.

## Decisions

- Capture the decode-boundary rule with three scenarios (valid, violating-not-repaired,
  structural-mismatch-distinct) to lock the source's explicit "distinct failure classes" and
  "reject, never repair" invariants.

## Risks / Trade-offs

- [No production caller in V1] → The decode chokepoint ships with only the conformance suite as
  its caller and is documented experimental. The requirement is normative for the first codec
  that lands; this V1 reality is recorded in Open Questions.

## Open Questions

- No production codec materializes a typed ADT value from an external payload as of this release
  (`EvalRequest.bindings` is tensors-only); the decode rule is normative for the first codec that
  lands. Verified against code: `decode_adt_value` (`crates/chelis-compiler-api/src/decode.rs`)
  is referenced only by the conformance tests (`tests/invariant_decode.rs`,
  `observation_roundtrip_harness.rs`) and the `lib.rs` re-export — no production caller —
  confirming the V1-reality note. This is an implementation-timing note, not a spec ambiguity.
- Binary program artifacts, DAG serialization, and tensor interop formats (DLPack) are deferred
  to the phases that implement them, per the source; they are out of scope here.
