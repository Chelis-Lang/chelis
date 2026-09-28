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

WireDag JSON is an exact-version contract. Schema version 22 is explicitly
present in every payload and is the only accepted version. A missing version,
versions 1 through 21, and every future version are decode errors before any IR
node is consumed. There is no versionless default, legacy migration, additive-
variant tolerance, or best-effort compatibility path.

`WireRiscOp::Load.name` distinguishes a graph input from a resolved top-level
value read. An ordinary name obeys `[A-Za-z_][A-Za-z0-9_.-]*`. A resolved read
uses `@chelis_global_` followed by the lowercase hexadecimal UTF-8 bytes of
its nonempty validated declaration name. This encoding is injective and cannot
be authored as an ordinary identifier, including a linked qualified name.
`WireRiscOp::Store.name` is always ordinary. An empty, malformed, noncanonical,
or wrong-kind name is an encoding and decoding error. Version 21 has no
resolved-origin label contract and is rejected before node decoding.

`WireRiscOp::ExtentWitness { site, parameter, axis, requirements, claims }`
preserves a
call's shape observation, its literal requirements and its named requirements
separately. An observing witness has one
tensor input followed by one earlier `ExtentWitness` input per named claim, and a
rank-zero `int64` output. The normalized `int32` axis must
be within the tensor input's rank. Every requirement uses the exact
`NonnegativeExtent` adapter over a nonnegative `int64`; an absent vector,
non-integer or negative requirement,
invalid axis, arity, or output type is an encoding and decoding error.
`site` is mandatory. Its `caller` and `local_expand` forms preserve
whether failure belongs to call entry (`load`, with parameter context) or a
local broadcast (`expand`, with the observed input node context); missing or
unknown sites are decoding errors. Its `result_claim` form is an object with
mandatory `claim` and `axis` fields: a nonempty diagnostic label and a
normalized nonnegative `WireRtAxis`. This witness observes the declaring
tensor axis and has exactly one shape dependency on the earlier `caller`
witness observing that identical tensor and axis. Its own `requirements` and
`claims` vectors are empty. A producing operation's shape dependency on this
token retains an equality against its scalar value at the token's result axis.
The producer must support observing that axis before allocation; the token's
node identity, not the label, identifies the required extent. Missing or
inconsistent declaring dependencies, entry obligations on a result token,
and unsupported producing axes are encoding and decoding errors.

Its distinct `literal_result_claim` role retains one declaration's literal
requirement independently of every other declaration. It has no value inputs,
no shape dependencies of its own, empty `parameter` and `claims`, exactly one
`NonnegativeExtent` in `requirements`, and a rank-zero `int64` output containing
that requirement. Its normalized `RtAxis` identifies the returned axis of the
producing operation that retains the token through `shape_deps`. The token's
node identity identifies the declaration; equal literal values do not identify
or discharge another declaration's obligation.

Its distinct `local_ascription_claim` role is an object with mandatory
`ascription_id`, `binding`, `claim`, and `axis` fields. The identifier preserves
the checker-owned authored-ascription identity as an opaque artifact-local
`u64`; it is not an extent or another numeric value. `binding` and `claim` are
nonempty diagnostic provenance; `axis` is a normalized nonnegative
`WireRtAxis` equal to the enclosing witness's axis. A literal local claim has
no value inputs or shape dependencies, empty `parameter` and `claims`, and
exactly one `NonnegativeExtent` requirement. A named local claim has one tensor
input, no requirements or entry claims, a nonempty declaring parameter, and
exactly one shape dependency on an earlier `caller` witness observing the
identical tensor and axis. Exactly one producing operation owns either form
through `shape_deps`, and that operation must support the claimed output axis.
Missing, empty, hybrid, unknown, multiply owned, unowned, or inconsistent
representations are encoding and decoding errors; no absent role decodes as an
empty obligation set.

The retaining operation preserves the primitive that produced the declared
result. A cast retains its own primitive attribution and takes its input's
placement under section 4.7 of the type system. The requirement must be
available at that placement; an obligation imposed on an already-produced
value executes at the declaring invocation boundary. A transformation that
replaces a producing operation preserves its guard's primitive attribution and
placement, or retains the guard-bearing operation. Declaration execution order
is carried by the producer's ordered obligation dependencies; token allocation
order and node numbers do not determine first failure. Missing or extra literal
requirements, observing inputs, entry claims, invalid axes, unsupported producer
relationships, and invalid dependency ordering are encoding and decoding errors.
The observing `result_claim` role retains its exact declaring-witness contract.
`parameter` is diagnostic text, not dimension identity. The node's source
provenance and invocation dependencies survive transport as ordinary node
fields and edges. Requirement order and duplicates are preserved; an empty
requirements vector is valid. `claims` is mandatory and its entries correspond
one-to-one, in order, with the inputs after the tensor. Each entry carries a
nonempty dimension binder and a boolean saying whether its requirement input is
the binder's declaring observation; that role is not recoverable from the edges,
because either observation can be the later one. An absent `claims` vector, an
entry with an empty binder, a missing or extra requirement input, and a
requirement input that is not an earlier rank-zero `int64` `ExtentWitness` are
each encoding and decoding errors. `WireDagNode.shape_deps` contains exact u64 node
references to strictly earlier nodes. It does not carry shape numbers.
`shape_deps`, `span_id` (explicitly null when absent), `merged_spans` and
`declaration` are mandatory fields, including when their lists are empty, and
so is `WireDag.declarations`. `WireDag.declarations` is the graph's declaration
table: one row per declaration, holding its name. `WireDagNode.declaration` is
the row of the declaration the node belongs to (§3.4). Two rows may hold one
name, so a name never identifies a declaration. Every row is the declaration of
at least one node; a row that no node names is an encoding and decoding error.
`WireDagNode.activation` is mandatory too, explicitly null when absent. It is
the node's activation: an earlier node of `bool` dtype that holds exactly where
the node's source position is entered, the conjunction of the runtime branch
predicates enclosing it, or null where every execution of its declaration
enters it. An activation is rank zero, or under a batched branch shaped like
the node's leading axes, one element per row. A node and its activation form
its owner, with the declaration. A node whose activation is false is still
computed, since a `Where` may read its value, but checks nothing: no numeric
trap, draw validation, count or extent check fires for it, and it computes its
value from operands its checks accept. A draw's, a draw replay's and a key
operation's activation is its node's activation; no input carries it. A
`KeySelect`'s two activation inputs are its keys' consumption activations,
its own activation conjoined with its branch's condition and with that
condition's negation (§3.2). An activation that is not an earlier `bool` node
is an encoding and decoding error.

`WireRiscOp::Mod` preserves the exact signed-remainder identity of [05-OP-64].
It has exactly two earlier input nodes, each with its output's integer dtype
and dimension list. Other arities, dtypes or shapes are encoding and decoding
errors.

`WireRiscOp::CheckedReshapeExtent { claims, axis }` has an earlier rank-zero
`int64` input for the independently computed target extent, followed by one
earlier scalar `int64` input per requirement. The nonempty `claims` list contains
diagnostic labels in the same order as those requirement inputs. Its rank-zero
`int64` output carries the computed extent after all equality checks, in list
order. Labels are diagnostic text; input edges identify requirements, including
each declaring signature's shape witness. The normalized
`int32` result-axis position is nonnegative. Missing fields, invalid references,
wrong arity, or non-scalar/non-`int64` inputs or output are encoding and decoding
errors. A resolved result-type dimension does not substitute for either input.

`WireRiscOp::CheckedUnitAxis { axis }` has two earlier inputs: a tensor and an
`ExtentWitness` reading exactly that tensor at the same normalized axis, with
an explicit requirement of one. Its output preserves the input's dtype, rank
and all other dimensions, refining only that axis to literal one. A different
tensor, axis, witness operation, missing unit requirement, or unrelated type
refinement is an encoding and decoding error. These checked operations preserve
their source provenance and invocation dependencies through the mandatory node
fields and edges above; discarded data results do not erase their checks.

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

Version 20 adds the closed `Bitwise { bitwise }` operation with the five signed
integer identities in [05-OP-47]. Every tagged variant must be known to the
version 22 decoder. Version 21 adds `Iota` for the two exact scalar i64
endpoints of `range` [05-OP-54], plus `ListMapCapture` and
`OrderedAdjointSum` for the executed callback and cotangent order in
[05-OP-55] and spec/06. Ordered contribution group counts use a tagged,
nonnegative i64 carrier; zero groups, zero widths, overflow, or a count total
different from the input arity are decode errors. `OneHot` remains only a transient
IR/specialization marker and backends must not receive it after specialization.

Execution-value envelopes carry the independently required exact
`schema_version: 4`. Missing, older, and future execution versions are rejected
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
from the closed interchange vocabulary below. Chelis source and canonical
Deep use the language spelling `i8`/`i16`/`i32`/`i64`, while this existing
versioned codec retains the interchange spelling
`int8`/`int16`/`int32`/`int64`. Encoders and decoders must perform that explicit
mapping; neither vocabulary is accepted as an alias at the other's ingress.

| dtype family | scalar object | storage object |
|---|---|---|
| `f64`, `f32`, `f16`, `bf16` | `{"dtype":p,"bits":h}` | `{"dtype":p,"bits":[h,...]}` |
| `int64`, `int32`, `int16`, `int8` | `{"dtype":p,"value":n}` | `{"dtype":p,"values":[n,...]}` |
| `bool` | `{"dtype":"bool","value":b}` | `{"dtype":"bool","values":[b,...]}` |
| `key` | none | `{"dtype":"key","bits":[h,...]}` |

Here `p` is a JSON string naming the exact dtype, `b` is a JSON boolean,
and `n` is an exact signed JSON integer within that dtype's range, never a
float or numeric string. `h` is a JSON string of lowercase hexadecimal digits
containing the stored IEEE bits, or a key's 64 bits ([05-RNG-2]), most
significant digit first, with no sign, prefix, separators, or whitespace. Its
exact digit counts are f64: 16; f32: 8; f16: 4; bf16: 4; key: 16. Leading
zeroes are required to reach that width. The encoding is independent of
machine endianness. A key is not a number: it has no scalar object and is
never an integer carrier, and its storage object appears only in a tensor
execution value.

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
`{"type":"bool","value":b}`. A scalar key execution value is
`{"type":"key","bits":h}`, with `h` the key's 16 digits. Tensor execution values use
`{"type":"tensor","value":{"shape":[d,...],"data":t}}`, where `t` is
the storage carrier and each `d` is an exact nonnegative int64 JSON integer.
Non-scalar aggregate execution variants retain their recursive element order
and constructor identity; their numeric descendants use these carriers.
The dimension vector has dynamic int32 rank, and the exact product of its
extents equals the payload's element count, including scalar-shaped and
empty tensors. Checked count and target-capacity admission precedes allocation
or access under [04-NUM-11].

`Const.value`, `Pad.fill`, and `ConstTensor.data` use the same scalar/storage
grammar, not private alternate encodings. The random operations carry no
numeric fields: their controls and their key are operand nodes.
`UniformLike.inputs` is exactly its template, its `low` and `high` bounds and
its key. `Dropout.inputs` is exactly its data input, its rate and its key. The
bounds and the rate are operands of the template's or input's exact active
float dtype.
(Not fully implemented; chelis#1295.) A draw's key has any rank `r`, and its
shape equals the leading `r` axes of the draw's data: the first input of
`UniformLike`, `Dropout` and `DropoutReplay`, and the cotangent, input 1, of
`UniformBoundAdjoint`. Row `b` of the draw, the data elements whose leading
`r` indices are the key's row-major index `b`, draws with `key[b]`, with its
elements' row-major indices within the row as [05-RNG-2]'s element indices.
Each control and the node's activation has the shape of the key's leading
`c` axes for some `c <= r`, and row `b` reads the element that `b`'s leading
`c` indices name, so under a rank-zero key every control and the activation
are rank zero.
Both operations preserve the first input's exact shape and dtype. Their
value-domain checks remain [05-OP-8/37], before any element is drawn. Each row
is one draw and checks the control elements it reads only when it is active,
so a row whose activation is false, or a batch with no rows, checks nothing,
as the stack of the rows' draws would. The codec neither inserts casts nor
implements an adjoint. `DropoutReplay.inputs` is exactly its cotangent, its
rate and its forward draw's key, and its result has its cotangent's exact
shape and dtype. `UniformBoundAdjoint.inputs` is exactly its template, its
cotangent and its forward draw's key. Its cotangent has its template's exact
shape and dtype, and its result has the template's dtype and the shape of the
key's leading `c` axes for some `c <= r`, and each element is one canonical
tree over the contributions, in row-major order, of the rows whose leading
`c` indices name it. Each reads the key without consuming it, and its
activation is its forward draw's.

`key` is a structural precision with no literal or storage carrier in a
graph, at any rank: no `Const`, `ConstTensor` or `Pad.fill` holds a key. Every
key is produced by `KeyFromSeed`, `Split`, `FoldIn`, `SplitN` or `KeySelect`,
or enters as a key-precision `Load`; a key-precision `Store`, which names a
root, is the key it stores, so rooting the `Store` is that key's place among
the roots. A `Drop` ([05-OP-67]) consumes the key it closes, and its output,
typed as its input, is not a key. Every node carries its declaration's row, and a
parameter is its declaration's row and its name: every `Load` of one name in one
row is one key, and `Load`s of one name in two rows read two parameters, even
when the two rows hold one name. A node reads a node of another row only when
no node of that row is potentially trapping (spec/06 §5.2): a reference to a
value declaration whose initializer may trap is that initializer's own nodes in
the referencing row, under the reference's activation. A
key
has at most one use: one `UniformLike`, `Dropout`, `FoldIn`, `SplitN` or
`Drop`, one key input of one `KeySelect`, one place among the roots, or at most one
`Split` of each branch. It is otherwise read only by its draw's replays. Two
uses of one key, by draws, key operations and `KeySelect` inputs alike, other
than one `Split` of each branch, may both consume it only when each consumes
it under an activation and one activation's `And` conjuncts include a node `X`
and the other's include `Not(X)`, or either's include the `bool` constant
`false`, whose consumer never runs. A draw or key operation consumes its key
under its node's activation. `KeySelect` is a branch's join. Its inputs are
two `key` tensors of its output's exact shape, then two Bool activations,
each shaped like the key's leading `c` axes for some `c`; it consumes input 0
under input 2 and input 1 under input 3, and each element of its output is
input 0's key where input 2 holds for that element's row, and input 1's
elsewhere. Its two activations are the two arms of one branch under its own
activation `S`: the `And` conjuncts of one are `S`'s and a node `X`, and those
of the other are `S`'s and `Not(X)`, or `X` and `Not(X)` are the `bool`
constants `true` and `false`, a constant `true` conjunct being set aside; with
no activation, `S` has no conjuncts. So wherever `S` holds exactly one arm
does, and where it does not, neither does. Two key
operations sharing a key under exclusive activations may derive equal keys,
as two `Split`s of one branch do, so a key that a key operation derives from a
key it shares this way, and every key derived from that key in turn, is
consumed only under that operation's activation or by the join of its branch:
each of its uses consumes it under an activation whose `And` conjuncts include
that activation or the constant `false`, and it is never a root. A join's
output is a key under `S`: each of its uses consumes it under an activation
whose `And` conjuncts include `S`'s, and each such requirement of its inputs
that both arms include, or the constant `false`, and while it has any such
requirement it is never a root. A key reaching any other operation or input
is a decode error, except that any operation may read a key tensor at an
input it reads only for its extent (an `ExtentWitness`'s or `Shape`'s tensor
input, or an input a bound reads by `input_axis`) and any node may name a key
among its shape dependencies: an extent is not key material, so neither is a
use and the key stays live.

`KeyFromSeed.inputs` is one `int64` tensor, and its output is the `key` tensor
of that shape holding [05-OP-69]'s key of each element. `Split` carries its
`branch`, `left` or `right`, and takes one `key` tensor; its output has that
shape and holds [05-RNG-2]'s `derive(k, 0)` or `derive(k, 1)` of each element
([05-OP-70]). `FoldIn.inputs` is a `key` tensor and an `int64` tensor of
exactly equal shape, and its output is [05-OP-72]'s key of each pair.
`SplitN.count` is a `WireRtDim` under the same carrier rules as
`Expand.size`, except that only `lit` and a `node` at input slot 1 are
admitted; its output appends that extent to its `key` input's shape, and row
`j` of each key is [05-OP-71]'s. `Split`, `FoldIn` and `SplitN` take no
input besides these operands. Their node's activation is shaped like the key's
leading `c` axes for some `c`, and it changes no key that the operation
derives. Where a `SplitN`'s activation is false in every element, the split
reads no count and checks nothing: its count axis has the extent its output
type declares when a literal or an earlier node fixes that extent, and zero
when the split alone would bind it.

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
| `WireDagNode.declaration` | Zero-based row of the owning DAG's `declarations` table. The row, not the name it holds, is the declaration's identity; every row is some node's declaration. |
| `WireDagNode.activation` | Null, or the ID of an earlier node of the owning DAG whose dtype is `bool`. |
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
