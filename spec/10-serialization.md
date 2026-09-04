# Serialization

## 1. Text Forms

- `.ch` is Surf source text in UTF-8
- `.dp` is Deep source text in UTF-8

Deep canonical printing is defined by `spec/03-deep-syntax.md`.

## 2. Binary Shell Form

`.chb` is the binary Shell metadata artifact used by the Reef package system.
It carries:

- public package metadata
- exported symbol metadata
- compiler compatibility metadata
- optional cached compiler products

The Chelis language does not define a byte-level `.chb` portability contract.
A `.chb` consumer must validate the embedded compiler-compatibility metadata
before consuming any package or symbol metadata.

## 3. Compiler API Wire Contract

WireDag JSON is an exact-version contract. Schema version 7 is explicitly
present in every payload and is the only accepted version. A missing version,
versions 1 through 6, and every future version are decode errors before any IR
node is consumed. There is no versionless default, legacy migration, additive-
variant tolerance, or best-effort compatibility path.

Version 7 preserves version 6's `WireRiscOp::Count { axes }`; `axes` is the complete
non-empty vector of unique normalized original-axis positions in strictly
descending order under [05-OP-29]. An encoder rejects any empty, duplicate,
increasing, source-order, or out-of-range vector rather than rewriting it,
and the decoder rejects an empty,
duplicate, increasing, or out-of-range vector before IR construction.

Version 7 also preserves version 6's padding representation:
`WireRiscOp::Pad { fill: ScalarValue, ... }`. The scalar tag and payload must
be the exact active tensor element dtype required by the padded tensor and
preserve its stored bits; a raw JSON number, an untagged payload, a string-mode
fill, or a mismatched dtype is a decode error before IR construction. No v5
numeric-fill migration or inferred fill dtype exists.

Version 7 includes the distinct `WireRiscOp::Relu` and
`WireRiscOp::ReluAdjoint` identities required by [05-OP-43]. `Relu` has
exactly one input; `ReluAdjoint` has exactly two, ordered as the forward input
and incoming cotangent. Every input has the output's exact float dtype and
dimensions. An unknown identity, a non-float dtype, wrong cardinality,
unresolved input, or shape/dtype mismatch is a decode error before IR
construction; the decoder does not replace either identity with an extrema
operation.

Version 7 represents every runtime movement bound and reshape target with the
tagged `WireRtDim` carrier defined by [05-MOV-1]. In particular,
`WireRiscOp::Expand.size` is a `WireRtDim`, never a display string.
`InputAxis { tensor, axis }` names an absolute nonzero input slot of the owning
node and a normalized literal int32 axis of that input tensor. `Node { input }`
names an absolute nonzero input slot whose source is an earlier rank-zero exact
int64 node. The decoder enforces the owner matrix from
`spec/05-risc-primitives.md` §2.4.1, the source rank and dtype, the normalized
axis range, and the exact input cardinality before IR construction.

Every tagged variant must be known to the version 7 decoder. `OneHot` remains only a transient
IR/specialization marker and backends must not receive it after specialization.

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

An invariant representation whose valid domain contains a non-finite numeric
field is outside this decode contract. A codec for such a representation
requires its own normative fail-closed rule that distinguishes every admitted
non-finite value from NaN and from evaluation failure; it cannot bypass the
pre-check by convention.
