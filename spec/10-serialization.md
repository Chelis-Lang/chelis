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

WireDag JSON is an exact-version contract. Schema version 9 is explicitly
present in every payload and is the only accepted version. A missing version,
versions 1 through 8, and every future version are decode errors before any IR
node is consumed. There is no versionless default, legacy migration, additive-
variant tolerance, or best-effort compatibility path.

`WireRiscOp::ExtentWitness { parameter, axis, requirements }` preserves a
call's shape observation and its literal requirements separately. It has one
tensor input and a rank-zero `int64` output. The normalized `int32` axis must
be within that input's rank. Every requirement uses the exact
`NonnegativeExtent` adapter over a nonnegative `int64`; an absent vector,
non-integer or negative requirement,
invalid axis, arity, or output type is an encoding and decoding error.
`parameter` is diagnostic text, not dimension identity. The node's source
provenance and invocation dependencies survive transport as ordinary node
fields and edges. Requirement order and duplicates are preserved; an empty
requirements vector is valid. `WireDagNode.shape_deps` contains exact u64 node
references to strictly earlier nodes. It does not carry shape numbers.
`shape_deps`, `span_id` (explicitly null when absent), and `merged_spans` are
mandatory fields, including when their lists are empty.

`WireRiscOp::Count { axes }` carries the complete
non-empty vector of unique normalized original-axis positions in strictly
descending order under [05-OP-29]. An encoder rejects any empty, duplicate,
increasing, source-order, or out-of-range vector rather than rewriting it,
and the decoder rejects an empty,
duplicate, increasing, or out-of-range vector before IR construction.

`WireRiscOp::Pad { fill: ScalarValue, ... }` carries a scalar whose tag and payload must
be the exact active tensor element dtype required by the padded tensor and
preserve its stored bits; a raw JSON number, an untagged payload, a string-mode
fill, or a mismatched dtype is a decode error before IR construction. No
numeric-fill migration or inferred fill dtype exists.

The wire carries the distinct `WireRiscOp::Relu` and
`WireRiscOp::ReluAdjoint` identities required by [05-OP-43]. `Relu` has
exactly one input; `ReluAdjoint` has exactly two, ordered as the forward input
and incoming cotangent. Every input has the output's exact float dtype and
dimensions. An unknown identity, a non-float dtype, wrong cardinality,
unresolved input, or shape/dtype mismatch is a decode error before IR
construction; the decoder does not replace either identity with an extrema
operation.

Every runtime movement bound and reshape target uses the
tagged `WireRtDim` carrier defined by [05-MOV-1]. In particular,
`WireRiscOp::Expand.size` is a `WireRtDim`, never a display string.
`InputAxis { tensor, axis }` names an absolute nonzero input slot of the owning
node and a normalized literal int32 axis of that input tensor. `Node { input }`
names an absolute nonzero input slot whose source is an earlier rank-zero exact
int64 node. The decoder enforces the owner matrix from
`spec/05-risc-primitives.md` §2.4.1, the source rank and dtype, the normalized
axis range, and the exact input cardinality before IR construction.

Every tagged variant must be known to the version 9 decoder. `OneHot` remains only a transient
IR/specialization marker and backends must not receive it after specialization.

Execution-value envelopes carry the independently required exact
`schema_version: 3`. Missing, older, and future execution versions are rejected
before decoding values. An execution version never substitutes for WireDag
version validation, or conversely. Tensor bindings in requests use the same
execution-value carrier grammar. A cache or compiled-context worker handoff
containing these objects validates its enclosing format and compiler-build
compatibility identity before decoding them; it cannot provide an alternate
legacy decoder for a usable value. The worker handoff preserves the complete
checked context and rejects an incompatible or corrupted payload before use.

(These wire requirements are not fully implemented; see chelis#1288.)

### 3.1 Numeric Values And Structural Fields

A compiler API field SHALL retain the semantic domain defined by its owning
language contract. Its integer or floating-point representation, field name,
or enclosing variant tag does not establish a different domain. Each field of
a container retains its own contract; a structural field does not confer its
meaning on a numeric sibling or descendant.

A field carrying a Chelis numeric value SHALL preserve its declared dtype and
full value set under [04-NUM-11]. An application-level variant tag SHALL NOT
substitute for the payload's dtype contract or turn arbitrary numeric data
into source-location or reference metadata. A tagged `f64` payload therefore
does not acquire source-location semantics merely by being tagged; a numeric
carrier remains subject to its exact declared dtype contract.

Source coordinates and measured source extents MAY be represented as structural
location fields rather than Chelis scalar values. Their domain is a position or
measured extent in a source context, governed by [04-FIT-16/17] for diagnostics
and `spec/03-deep-syntax.md` §1.1.1 for external-source identities. The source
context may be supplied by the enclosing document or request. This permission
does not extend to arbitrary counts or numeric payloads merely named as source
metadata; the owning source-location contract must govern the field.

A scoped input reference, such as `WireRtDim::InputAxis.tensor`, denotes a slot
in the owning node's inputs and is resolved and validated under §3's reference
contract. Its magnitude is not the selected tensor extent. The associated axis
literal and the selected extent retain their separate int32 and int64 domains.

Structural transport does not waive the governing requirements on a consumer.
In particular, runtime rank, extents, strides, element counts, and byte capacities
retain [04-NUM-11]'s descriptor domains and validation requirements; they cannot
inherit source-location or input-reference semantics through an outer tag.

### 3.2 Exact Numeric Value Codec

The scalar wire carrier is exactly one of these shapes, with `dtype` drawn
from the closed primitive vocabulary of spec/04 §1.1:

| dtype family | scalar object | storage object |
|---|---|---|
| `f64`, `f32`, `f16`, `bf16` | `{"dtype":p,"bits":h}` | `{"dtype":p,"bits":[h,...]}` |
| `int64`, `int32`, `int16`, `int8` | `{"dtype":p,"value":n}` | `{"dtype":p,"values":[n,...]}` |
| `bool` | `{"dtype":"bool","value":b}` | `{"dtype":"bool","values":[b,...]}` |

Here `p` is a JSON string naming the exact dtype, `b` is a JSON boolean,
and `n` is an exact signed JSON integer within that dtype's range, never a
float or numeric string. `h` is a JSON string of lowercase hexadecimal digits
containing the stored IEEE bits, most significant digit first, with no sign,
prefix, separators, or whitespace. Its exact digit counts are
f64: 16; f32: 8; f16: 4; bf16: 4. Leading zeroes are required to reach that
width. The encoding is independent of machine endianness.

All bit patterns are representable, including finite values, subnormals,
positive and negative zero, infinities, and quiet or signaling NaNs with their
payloads and signs. No codec normalizes a NaN payload or a signed zero.
Transport is a bit move, not numeric finalization; arithmetic and conversion
still obey [04-NUM-2]. A numeric JSON float, `null`, a wrong-width or uppercase
bit string, a mismatched payload member, a reserved dtype, or an extra payload
member is rejected. There is one encoding per stored value, not alternative
number and bit encodings. Numeric integers never pass through binary64.

Numeric scalar execution values use `{"type":"scalar","value":s}`, where
`s` is the scalar carrier above. Boolean execution values retain
`{"type":"bool","value":b}`. Tensor execution values use
`{"type":"tensor","value":{"shape":[d,...],"data":t}}`, where `t` is
the storage carrier and each `d` is an exact nonnegative int64 JSON integer.
Non-scalar aggregate execution variants retain their recursive element order
and constructor identity; their numeric descendants use these carriers.
The dimension vector has dynamic int32 rank, and the exact product of its
extents equals the payload's element count, including scalar-shaped and
empty tensors. Checked count and target-capacity admission precedes allocation
or access under [04-NUM-11].

`Const.value`, `Pad.fill`, and `ConstTensor.data` use the same scalar/storage
grammar, not private alternate encodings. `UniformLike.low`, `UniformLike.high`,
and `Dropout.rate` use scalar carriers of the exact active float dtype of the
template/input. Their value-domain checks remain [05-OP-8/37], before Random
consumption; the codec neither inserts casts nor implements an adjoint.
`UniformLike.inputs` contains its template followed by at most one rank-zero
Bool path activation; `Dropout.inputs` contains exactly its data input.
Both operations preserve the first input's exact shape and dtype. The optional
activation is an earlier-node reference under §3.4, not another template.
Their `seed` fields are exact uint64 JSON integers holding [05-RNG-1]'s
two's-complement image of a signed int64 seed. Every seed bit is significant;
zero is a seed, not absence, and there is no default seed.

### 3.3 Source Syntax And Locations

`WireDeepAtom.Int` and `WireLiteral.Int`/`TypedInt` preserve exact signed i64
source values. Their Float/TypedFloat counterparts preserve finite binary64
lexical values as JSON numbers, including negative zero; source syntax admits
no infinity or NaN. A typed suffix retains its source spelling and is admitted
against spec/02 P10 and spec/03 §6.4's closed suffix and literal-origin rules.
The lexical carrier does not replace the selected literal dtype or perform
its target-width rounding. Integer-written float literals keep the exact
integer until the governing source admission performs that rounding.

A raw source DTO is not an admitted executable AST. Parsing a DTO, preserving
macro history, or transporting an extension annotation does not establish
well-formed tags, defined-key placement, uniqueness, type correctness, or
permission to execute. Every path that materializes executable syntax runs
the owning source admission, including live annotation expressions. Unknown
extension data has no measured-source or numeric-operation authority.

`WireSurfExpr.TupleGet.index` and `Vmap.axis` retain signed i64 source spelling
as JSON integers until normal source admission. Tuple selection requires the
owning tuple index contract; a vmap axis must satisfy the int32 axis contract.
Absence of `Vmap.axis` denotes the specified axis-zero syntax; a supplied
out-of-domain integer is not replaced by zero.

`Span.offset`/`len` and `DiagnosticSpan.Point.offset`/`Range.offset`/`Range.len`
are exact nonnegative u64 JSON integers counting UTF-8 bytes in the enclosing
document or request's source context. A measured range is half-open, and
its endpoint is the exact sum of offset and length. Transport does not require
possession of a foreign source buffer; local slicing checks endpoint arithmetic,
target representability, bounds and UTF-8 boundaries before access. A point
has no extent. An absent location, a point, and a measured empty range remain
distinct under [04-FIT-16/17]. Source identity is independently optional and
opaque; neither its spelling nor a coordinate can manufacture the other.

### 3.4 References, Dimensions And Operation Parameters

A reference is resolved only in its declared owner and namespace.
Reference fields are exact nonnegative u64 JSON integers unless the table
specifies u32. A decoder never narrows a reference to fit a host index.

| fields | scope and admission |
|---|---|
| `WireDagNode.id`, `WireDagNode.inputs`, `WireDagNode.shape_deps`, `WireDag.roots` | Node IDs are unique zero-based positions in the owning ordered DAG; inputs and shape dependencies refer to earlier nodes and roots to existing nodes. Zero is an ordinary node ID. Shape dependencies identify nodes, not extents. |
| `LowerResult.named_roots`, `GradResult.output_node`, `GradResult.grad_nodes_by_name`, `GradResult.forward_nodes_by_name` | IDs select nodes in that result's DAG; a name never changes the owning DAG. |
| `EvaluatedRoot.node_id` | Identity in the evaluated graph associated with that result; without that graph it remains an opaque result identity and cannot be dereferenced. |
| `WireFusedInput.External.index` | Zero-based slot in the owning fused node's external inputs. |
| `WireFusedInput.PreviousStep.index` | Zero-based strictly earlier step in the owning ordered fused program. |
| `WireRtDim.Node.input`, `WireRtDim.InputAxis.tensor` | Absolute nonzero slot in the owning operation's input vector, subject to §3's owner/source/axis checks. |
| `WireInferredType.Var.id`, `WireInferredPrecision.Var.id` | Exact u32 identity in the owning inference product's type-variable namespace. |
| `WireInferredDim.Var.id`, `WireInferredDim.Rank.id` | Exact u32 identities in that inference product's distinct dimension-variable and rank-variable namespaces. |
| `WireInferredParameter.index` | Zero-based u64 position in the owning inferred function signature's ordered parameter list. Every parameter's index equals its position in that list; names do not change this scope. |
| `WireDag.schema_version`, `EvalResult.schema_version` | Exact u32 version discriminants governed by §3, with no missing-value default. |

`WireDimInfo.Lit.size`, `WireDimInfo.Named.size` when present,
`WireInferredDim.Lit.size`, `ExecutionDim.size` when present,
`WireDimExpr.Concrete.value`, and
`WireRtDim.Lit.value` and every `WireRiscOp::ExtentWitness.requirements` item
are nonnegative exact int64 JSON integers. An absent
named size is unresolved, not zero. Observed extents use a fixed-int64
JSON-number adapter over the canonical sealed numeric carrier; construction
and decode reject a different dtype or a negative value. These are numeric
quantities, not structural reference identities, and their bounds alone do
not confer transport authority. Multiplication and division expression
structure is preserved; exact intermediate products and partial division
validity are not replaced by saturated machine integers. A transported
expression or observed runtime equality does not prove capacity equality.
Symbol resolution, program scope and representation equality remain required
at the consumer that claims reuse.

Every positional `WireRiscOp` axis, including reduction/gather/scatter/shape
axes and permutation entries, is a normalized nonnegative int32 JSON integer
under [05-DIM-3]. Its owning operation checks rank, uniqueness and order as
applicable. Window shapes and strides are positive exact int64 JSON integers
under [05-OP-39]. `OneHot.vocab` is a positive exact int64 extent and does not
authorize the transient operation to reach a backend. `WireRtDim.ToEnd` and
`Sym` remain distinct variants with the owner restrictions of [05-MOV-1];
numeric sentinel values cannot stand for them. Literal bounds and extents
retain their operation-specific zero/positivity/range rules.

The shared IR `WireRiscOp::Expand` represents both singleton broadcasting
(`expand`) and axis insertion (`insert`). Equal input/output ranks select
broadcasting, whose axis is less than the input rank. An output rank exactly
one greater selects insertion, whose axis may equal the input rank, including
axis zero on a rank-zero input. Every other rank relationship is invalid.
These wire layouts preserve the distinct source operations of [05-MOV-1];
they do not discharge the operations' extent claims or runtime guards.

### 3.5 Fixed-Dtype Report Numbers

Compiler report quantities are numeric values, not structural metadata.
Bounds alone never establish transport authority. Their recognized wire form
is a fixed-dtype JSON-number adapter over the canonical sealed numeric carrier:
the owning field contract fixes `f64` or `int64`, and both construction and
decode validate that exact dtype and the field's domain. A wrapper or numeric
range annotation without this carrier/codec contract is insufficient.

`score`, all four fitness components, and diagnostic `severity` have the
finite f64 unit-interval domains of spec/04 §6.4. Encoding emits the exact
shortest round-trippable binary64 JSON number and preserves signed zero;
decoding performs the JSON-to-f64 interpretation once and validates the
result. It does not subsequently round to another dtype, clamp, or supply a
missing value. This codec cannot encode NaN or infinity as `null`.

`typed_nodes`, `untyped_nodes`, `total_nodes`, `rewritten_calls`, and
`renamed_references` are nonnegative exact int64 JSON integers. The latter
two count the call sites and reference occurrences actually rewritten by
that result, not attempted edits. `peak_device_bytes_estimate`, when present,
is a nonnegative exact int64 estimate of peak simultaneously live device
allocation bytes for the compiled entry; absence means no estimate is
available, while zero is an actual zero-byte estimate. It is not an
allocation authorization or a proof of target capacity. Producer overflow
is an error, never a wrapped count or a fabricated estimate. Report integers
are decoded as integers and never via f64. `CheckResult`/`WireCheckResult`
and `Diagnostic`/`WireDiagnostic` obey the same numeric contract despite
their distinct producer and consumer types.

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
