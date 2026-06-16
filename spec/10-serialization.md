# Serialization

**Status:** Partial.
Text formats are settled enough to reference.
Binary formats remain intentionally under-specified until there is an implemented owner.

## 1. Text Forms

- `.ch` is Surf source text in UTF-8
- `.dp` is Deep source text in UTF-8

Deep canonical printing is defined by `spec/03-deep-syntax.md`.

## 2. Binary Shell Form

`.chb` is the binary Shell metadata artifact used by the Reef package system.
Its role is stable, but the exact wire format is still owned by the implementation.

Current `.chb` expectations:

- public package metadata
- exported symbol metadata
- compiler compatibility metadata
- room for future cached products without freezing the public wire layout

The project should not publish fake low-level `.chb` layout guarantees while the
implementation is still expected to evolve.

## 3. Compiler API Wire Compatibility

The compiler API wire models in `crates/chelis-compiler-api/src/schema.rs` are
the current machine-facing JSON surface. During this pre-release period, adding
new tagged variants such as `WireRiscOp::Gather`,
`WireRiscOp::ScatterAdd`, or internal `WireRiscOp::OneHot` is an additive
schema change. Producers may emit the new variant after the owning compiler
behavior lands. `OneHot` is only a transient IR/specialization marker; backends
must not receive it after specialization.

Consumers should tolerate unknown additive variants where possible and report a
clear unsupported-variant diagnostic rather than failing only because the enum
grew. Consumers that intentionally pattern-match exhaustively must treat the
wire schema as version-coupled to the compiler crate they were built with.

## 4. Invariant Revalidation At Decode Boundaries

An opaque type may carry one declared invariant — a boolean property of
its representation (`spec/01-nomenclature.md` §12.1,
`spec/design/opaque_invariants_rfc.md` D-DECODE). The everyday type
checker never evaluates the invariant, and `chelis prove` discharges it
on the *producer* side. The remaining channel is the *decode* side: any
path that materializes a value of an invariant-carrying opaque type from
an external payload (bytes, JSON, a wire message) bypasses both the
defining module's constructors and the producer obligations. Decode is
therefore a soundness boundary, not a convenience.

### 4.1 Normative rule

Any codec, deserializer, or external-payload ingestion path that
materializes a value of an invariant-carrying opaque type **must**
revalidate the declared invariant at materialization time, by calling the
decode chokepoint `chelis_compiler_api::decode_adt_value` (or an
equivalent that performs the same two passes):

1. **Structural validation.** The payload's constructor must be declared,
   and its fields must match the declared representation in arity, order,
   and field type (scalar primitive, fixed-shape numeric tensor, or nested
   single-variant record). A structural mismatch is a decode failure that
   is *distinct* from an invariant violation — it means the payload is not
   shaped like the type at all.
2. **Invariant revalidation.** The declared predicate is evaluated on the
   materialized value, with any in-module zero-argument constant defs the
   predicate references (permitted by the well-formedness grammar; see
   `spec/01-nomenclature.md` §12.1) resolved to their values. Before
   predicate evaluation, the representation sanity pre-check rejects any
   payload containing a NaN or non-finite value in a numeric
   representation field. This pre-check is mandatory for fail-closed
   behavior: an in-grammar predicate such as `not (p.value > 1.0)` is
   *true* on NaN, so relying on the comparison to reject NaN is unsound.
   A predicate that evaluates to `false`, fails to evaluate (partiality
   — division by zero, a domain error in `log`/`sqrt`, or an unresolved
   reference), or does not evaluate to a boolean is a decode failure.

**Decode of a violating payload is a failure, never a repair.** A codec
must not clamp, normalize, saturate, or otherwise coerce a violating
payload into the admissible set; it must reject it. A rejected decode
yields no value of the type.

The chokepoint evaluates the predicate through the evaluator's own
interpreter; `chelis-compiler-api` does **not** depend on `chelis-prove`,
so the decode boundary carries no solver and no prove-tier machinery.

### 4.2 V1 reality (no external ADT-value codec exists yet)

As of this release **no production codec materializes a typed ADT value
from an external payload.** The machine-facing wire surface
(`crates/chelis-compiler-api/src/schema.rs`) confirms this directly:
`EvalRequest.bindings` is `BTreeMap<String, TensorValue>` — tensors only;
`ExecutionValue::Adt` is an output-only shape the evaluator produces from
program text and is never accepted as an input; and the cache envelope,
the reef prepared-graph, and the `.chb` shell metadata serialize compiler
state, source text, and symbol metadata — not domain values. The
evaluator materializes ADT values from program text, which is the
construction-discipline surface (checker plus lint), not a codec.

Consequently the decode chokepoint ships with the conformance suite
(`crates/chelis-compiler-api/tests/invariant_decode.rs`) as its only
caller in V1, and `decode_adt_value` is documented experimental until a
real codec consumes it. This is the honest current state: the rule above
is normative for the first codec that lands, and the chokepoint is the
contract point it must call, but nothing exercises that boundary in
production yet.

The pre-check deliberately narrows out representations whose invariants
would legitimately admit infinities (for example a log-probability field
wanting `-Inf`): such representations are outside V1's decodable class.
Any future relaxation must re-derive fail-closed semantics for the
admitted non-finite values rather than silently reopening the NaN hole.

## 5. Related Serialization Work

Future serialization work may include:

- binary program artifacts for Shell distribution
- DAG serialization
- tensor interop formats such as DLPack

These belong to the phases that implement them rather than to speculative prose here.
