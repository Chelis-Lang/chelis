# Generic tensor actualization across rank and dtype

Parent tracker: [#729](https://github.com/Chelis-Lang/chelis/issues/729).
Related reports: [#1760](https://github.com/Chelis-Lang/chelis/issues/1760)
and [#1761](https://github.com/Chelis-Lang/chelis/issues/1761),
[#3997](https://github.com/Chelis-Lang/chelis/issues/3997) and
[#3998](https://github.com/Chelis-Lang/chelis/issues/3998).

`spec/04-type-system.md` §4.5.3 owns rank-spread instantiation and the checked
result type. `spec/05-risc-primitives.md` [05-OP-36] and [05-OP-53] own
comparison and `where`: a Bool mask selects same-shape, same-dtype branches.
[05-OP-40] owns direct extrema selection and its shape/dtype identity;
[05-OP-20..22] own floating classification of those selected values.
This document selects an implementation of those rules. It does not change
the operations' accepted language or the numeric semantics in
[`dtype_semantics.md`](dtype_semantics.md).

## The shared failure boundary

At a generic call boundary, checked rank and precision remain one instantiated
contract. A comparison such as `gt(xs, xs)` retains the input's concrete axes
while producing Bool; its C iteration domain follows those axes. In a nested
expression such as `gt(where(gt(xs, xs), xs, xs), xs)`, the mask and final
comparison produce Bool while the selected numeric values retain the bound
float dtype. Recovering one node's type from an adjacent consumer can lose
either the iteration rank or the numeric precision. The acceptance matrix
below checks these relationships independently.

This bounded design owns concrete generic-call actualization through an elementwise DAG
and its C iteration domain. It does not own a runtime extent equality claim:
an axis's observed size and any independent declaration remain governed by
[`runtime_extents.md`](runtime_extents.md) C2.3–C2.5. An extent guard cannot
repair an incorrectly chosen rank or element dtype.

## Selected representation

At each concrete activation, instantiate the checked tensor type of every
parameter, intermediate and result before lowering its body. A generic
definition may keep symbolic rank until that activation supplies its actuals.
Keep two independent facts
on each tensor-producing node:

1. its ordered concrete axis-source vector, including the resolved rank-spread
   positions and the operation's output-axis mapping;
2. its checked element dtype after precision-binder substitution.

The operation's typed input edges identify the Bool condition and the two
selected values of `where`. A shared tensor value is not globally assigned
one role merely because it feeds a particular consumer.

The result of a comparison inherits the operand shape, not the operand dtype.
`where` inherits the branches' shape and dtype, not the condition's dtype.
These relationships are derived from the checked operation and instantiated
inputs once and are carried into the DAG. The C iteration domain is derived
from the result's concrete axis sources; it cannot fall back to a scalar rank
when the result metadata contains an unresolved spread or a Bool dtype. A
remaining spread, ambiguous precision binder, missing source, or inconsistent
input role at a boundary that requires a concrete type is a typed lowering
failure before C emission. The ownership
verifier remains a consumer of this contract and continues to reject mismatched
precisions; it is not relaxed to make either reproducer pass.

Host result inference derives type-preserving operand positions from the
builtin scheme and its declared shape-identity contract. When a scheme returns
the same type variable as an operand, that operand supplies the complete host
type term, including scalar precision, ordered dimensions, and unresolved rank
or precision binders. The admitted declaration cache is shared across inference
calls; an operation-name mirror cannot omit a producer. Shape-changing operations and
precision-changing predicates retain their own checked result relations. A
Boolean consumer's result annotation never supplies a missing float producer
type.

Tuple parameters remain typed aggregate inputs when a declaration returns a
tensor. The shared host staging boundary projects an already evaluated tuple
once and bridges each numeric leaf with its actual scalar or tensor type.
Partition validation requires exact agreement between the typed producer and
its graph input, including dimensions and dtype. Local ascriptions restore
that retained physical source type and keep their authored claims on separate
witness tokens. The checked load survives partitioning so those obligations
execute after projection and before consumers. The whole authored entry is
checked in signature order before a staged body or capture provider executes,
including unused nested fields; Eval uses the same cached claim patterns and
entry plan as ordinary host calls. Nested aggregates keep their structure until
projected; a result annotation never supplies their contents.

The shared IR interface observation roster identifies each fixed tensor leaf
by formal position, ordered tuple path and complete declared tensor type. One
canonical traversal supplies the C entry plan and witness materialization
from both typed host inputs and authored native tuple-call signatures. The C
plan retains its direct-tensor authored-contract selection and its fixed
tuple declaration order. Each graph materializes this roster before lowering
the body, including fields the body never reads, and binds each observed axis to
an activation-owned `ExtentWitness`. Signature order followed by tuple/axis
order chooses the first observation of a binder. Fixed tensor projections
reuse the roster's node by host value identity and tuple path, including through
aggregate aliases; reading a field again does not manufacture a second input
origin. Interface-only result comparisons therefore keep their entry placement,
while local claims retain their initializer position. Local and result claim
tokens refer to that witness by graph identity across partition boundaries; they do
not acquire a declaration from the value being checked. Entry checks remain
owned by the host interface and precede staged materialization. Native tuple
call preparation transports local/result witnesses without taking over the
aggregate entry-guard migration tracked by #4018. Numeric leaves retain their
physical type independently of these logical claims.

Guard-only validation cannot supply a later claim's expected extent, and
body-projection-only transport omits unread declaring fields. Both therefore
leave the interface contract incomplete. The observation roster supplies the
missing declaration evidence without turning result-only/computed binders
into input binders or relating same-spelled binders of different activations.
Dynamic container claims retain their canonical claim-pattern owner; the fixed
roster does not invent a path through a value-selected container.

The executable class oracle pairs independent interface witness and returned
wildcard leaves under declared results and local annotations. It covers equal
and unequal extents (zero included), unread and nested declaring fields,
multiple axes, parameter/path order, aliases and repeated activations with
different extents. Positive cases require exact Eval/native-C agreement;
negative cases require the owning extent rejection, never a successful value
or an unresolved-witness lowering error. Existing entry-before-body,
physical-carrier, and result-only movement-extent controls remain enabled.
Whole-body helper result transfer uses the same declaration roster: an axis
owned by the staged helper is excluded from the C result frame. A host-control
body retains its result-frame owner. Transport does not duplicate a claim.

The C host-control result frame also consumes the fixed roster. Each named
result requirement captures the declaring tensor axis before entry drops can
release parameters, following the roster's tuple path with balanced retained
value handles. A selected body result consumes that frame; reading a witness
at entry does not move the result guard ahead of the body. Direct tensor and
List witnesses retain their existing frame owners. Value-selected container
claims remain outside this fixed-interface slice and keep the #3456 owner.
This lets tuple-carried `i64` extents feed movement operations under
`spec/04-type-system.md` §4.7.2 while entry, local and result extent claims
retain their separate witnesses and ordering.

The two extrema identities use one constructor selected by `ExtremaKind`.
It takes the lowered operands and derives the complete output `TensorType`
from the left operand, preserving its ordered physical axes and dtype. It
accepts no output-type annotation: metadata with an unresolved rank spread
cannot stand in for a scalar, and a consumer's Bool annotation cannot choose
the numeric producer's dtype. Minimum and maximum remain distinct direct
RISC nodes with their original operand order and source span; neither is
implemented through negation or surrogate arithmetic.

The extrema verifier requires two numeric operands and a numeric result of
one active signed-integer or float dtype. All three have equal rank,
including rank zero, and compatible extents at every axis. Known extents
must agree; named or wildcard extents whose equality needs execution retain
the existing runtime agreement guards. Labels need not be byte-identical:
physical axis sources and independently authored logical extent claims keep
their existing separate carriers. An authored claim never overwrites the
constructor's physical operand shape. Helper actualization and C emission
consume this result contract rather than repair an extrema constructor's
speculative rank or dtype.

Nominal dimension arguments remain exact logical terms through field
instantiation. They belong to the activation's dimension evidence, separately
from type and precision substitution: an inferred caller result cannot replace
a retained callee's own named extent witness. The private C projection compares
nominal arguments by value layout when assigning values, because a dimension
has no C value slot. Logical equality and accumulator reuse certificates still
compare the exact dimension terms; entry and result claims enforce their extents.
An empty collection of a known nominal type needs no numeric dimension witness
for its parametric value layout. Dimension metadata stays distinct from an
unresolved stored type or precision, and nonempty tensor fields supply the
activation's actual extent witnesses.

The call-site instantiation already used for authored parameter/result claims
in `chelis-ir/src/lower.rs` is the starting point. The implementation should
make its checked rank/precision mapping available to the whole lowered body
rather than add a special `gt` or `where` substitution path. Graph import,
inlining, specialization, and cache transport must remap axis references
without changing the instantiated dtype or typed operand roles. No numeric value
or raw dtype selector is introduced by this metadata; any changed numeric
operation remains subject to `dtype_semantics.md` §C6 and [05-OP-N]
registration.

## Acceptance and order

Write the exact positive and negative tests before changing lowering. First,
cover each reported expression with explicitly declared rank binders and a
generic passthrough control. Inspect the
remaining matrix for a failing consumer; close any gap by making the contract
mandatory at the call/body boundary, migrating the affected elementwise and C
iteration-domain consumers, and removing a reconstruction path that infers a
result rank or dtype from an adjacent operand's role.

| Issue | Required positive receipt | Required negative/control receipt |
|---|---|---|
| #1760 | The reported Bool result agrees on Eval and linked native C at concrete ranks 0, 1 and 2, with independent rank-spread and precision instantiations. | Incompatible shapes or a residual unresolved spread fail before C emission; a numeric-output generic control retains its original shape and dtype. |
| #1761 | The reported nested comparison/`where` agrees at f32 and f64; the mask and final result remain Bool while selected values retain the bound float dtype. | Mixed-family branch values and incompatible shapes reject at the owning checker or lowering stage; the simpler generic `where` passthrough continues to execute. |
| #3997 | Both extrema preserve direct numeric results at all eight active numeric dtypes, scalar surface and tensor ranks 0/1/2, including nonsquare `[2,3]` and empty `[2,0]`, named dimensions and independent bounded-precision activations. | Incompatible shapes, precisions and surfaces reject; malformed DAG result rank, extent or dtype and Bool operands reject. Constructor tests retain one direct node, operand order and source span. |
| #3998 | Minimum through `is_nan`, `is_finite` and `is_infinite` agrees at every float width, including nonfinite inputs and local intermediates. An imported/exported f32 `[2,3]` tuple wrapper retains its minimum and finite mask. | Nonfloat classifiers and false static output claims reject; agreeing dynamic claims execute and contradictory authored/input extents trap without weakening runtime guards. |

Every successful C build is compiled, linked and run. Compare exact result
shape, dtype, iteration rank, and values with Eval, not just exit status.
Replay an inlined call, a named helper call, and two independent instantiations
in one program so a cached specialization cannot reuse the first call's rank
or precision. Use a comparison result both as a `where` condition and as a
returned value, and vary legal operand order to ensure roles belong to use
edges. A checker
rejection is acceptable only for a program the controlling specs reject;
neither reported legal program may be fenced as an implementation shortcut.

This matrix defines acceptance for these issues, alongside #729's numeric
operation oracle. It does not certify all rank-polymorphic callables or close
#1277's extent-witness corpus.
