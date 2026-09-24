# Values, transformations, runtime, and backend obligations

This chapter accounts for numeric computation, dynamic shapes, differentiation,
graph rewriting, scheduling, storage, and foreign execution at source snapshot
`38ab515f5e0d070d8d31ecfc9d114bbbb7b91ce0`, inspected on 2026-09-09. It is part of
the [soundness account](../soundness_obligations.md), not a new specification or a
release verdict. The [inventory](inventory.md) reconciles the atom definitions;
the [language](language.md) and [integration](integration.md) chapters cover the
static and entry/reporting sides of the same boundaries.

Every implementation and test statement below is **source-inspection evidence**.
No compiler, generated program, numerical test, sanitizer, GPU workload, or oracle
was executed for this chapter. Commands identify existing executable evidence
surfaces, not successful runs. A source-visible discrepancy is identified as such;
where caller validation or reachability has not been traced, it is not presented
as a demonstrated `.ch` failure. No issue list is used as the obligation universe.

## 1. Authority, domain, and what a carrier establishes

The controlling sources are [spec/04](../../../spec/04-type-system.md),
[spec/05](../../../spec/05-risc-primitives.md) and its incorporated registries,
[spec/06](../../../spec/06-transformations.md),
[spec/07](../../../spec/07-concurrency.md),
[spec/08](../../../spec/08-backends.md), and
[spec/11](../../../spec/11-ffi.md). No chapter-to-capability transfer is in force
at this snapshot. Implementation plans such as
[dtype semantics](../../../spec/design/dtype_semantics.md),
[runtime extents](../../../spec/design/runtime_extents.md), and
[compiled-value ownership](../../../spec/design/compiled_value_ownership.md)
organize delivery and tests; they cannot weaken those rules.

The relevant domain is not merely an operation enum. It includes each callable
identity, scalar/tensor/container surface, admitted dtype and rank, dynamic
metadata, normal and exceptional values, AD context, transformed graph, execution
lane, backend option, external representation, ownership state, and observation.
The same operation reached through source lowering, a public IR API, a decoded
graph, a host helper, or a foreign binding still owes its contract.

Four claims must remain separate:

| Claim | Example | What remains to establish |
|---|---|---|
| A rule forbids a class of behavior | Integer addition cannot silently lose an int64 low bit. | Nothing about existing code follows from the prohibition alone. |
| A representation blocks a class of construction | `ScalarValue` has private tagged bits; `ShapeMetadata` couples checked shape and counts. | Constructor arithmetic, every method, unsafe ingress, and consumer interpretation. |
| A consumer requires a validated artifact | C emission takes verified ownership programs; reuse takes a private token. | The verifier's algorithm, proof-to-payload identity, all entry paths, and emitted execution. |
| An oracle detects specified violations | A receipt-backed suite executes a frozen set of positive/negative cases. | The comparator, domain completeness, mutation adequacy, configurations, and fresh execution. |

The existing structures establish useful local boundaries. None alone establishes
the universal statement that every accepted program computes its specified result.

## 2. Numeric representation and operation semantics

### 2.1 Cross-cutting numeric rules

The table accounts explicitly for `[04-NUM-1]` through `[04-NUM-16]`. The type-side
admission of `[04-DTYPE-1]` and `[04-DTYPE-2]` is shared with the language chapter.

| Authorities | Obligation, including negative domain | Inspected mechanism and residual |
|---|---|---|
| `[04-DTYPE-1]`, `[04-DTYPE-2]`, `[04-NUM-4]` | The active numeric family is f16/bf16/f32/f64/int8/int16/int32/int64; tensors additionally admit bool. String is not a numeric dtype, and reserved F8 is not an accepted dtype. Bool has only 0/1 representations and is not arithmetic. Families are constraints, not implicit promotion. | Closed `Prim`, runtime dtype decoding, typed scalar/storage constructors, and exhaustive census classification distinguish these cases. Recognizing reserved F8 in a census does not authorize execution. Each checker/backend/foreign route still needs admission. |
| `[04-NUM-1]`, `[04-NUM-2]`, `[04-NUM-5]`, `[04-NUM-6]`, `[04-NUM-8]` | Finalize at every primitive boundary, then compare or consume the stored result. Use RNE, specified subnormal/infinity/signed-zero behavior and arithmetic NaN canonicalization. f64 arithmetic is f64, f32 is f32, and f16/bf16 arithmetic is f32 followed by own-storage finalization. No implicit wider arithmetic or reduced-precision mode; explicitly specified accumulators are separate. | `ScalarValue`, `TensorStorage`, `float_binop`, and finalizers in `dtype_semantics.rs` implement typed paths. A finalizer cannot repair computation previously done at the wrong width/order. Legacy evaluator and optimizer wide-image paths remain visible below. |
| `[04-NUM-3]`, `[04-NUM-7]`, `[04-NUM-13]` | Signed arithmetic is exact or traps at the declared width. Only named wrap operations are modular. Bitwise shifts operate on fixed-width two's-complement bits; oversized counts and negative-count rejection have specified behavior, not C undefined behavior. | Closed `IntBinOp`/`IntUnOp`, `int_binop`, and `int_unop` centralize checked operations. Generated C, device intrinsics, folds, and host helpers must use equivalent algorithms; a shared operation name is insufficient. |
| `[04-NUM-9]`, `[04-NUM-10]`, `[04-NUM-12]`, `[04-NUM-15]` | Trap kinds are closed, diagnostics identify the lowered primitive/dtype, a trap is a value inside a lane and a failure at its boundary, and the first offending tensor element is the lowest row-major index. Occurrence is deterministic within a lane. NUM-12 allows a narrow trap-versus-exact difference when an operation actually permits different lane accumulation orders. | `NumericTrap`, its `Display`, and `IndexedTrapCandidate` provide typed vocabulary and minimum-index selection. Callers must propagate rather than panic/default/exit prematurely. Runtime process-exit helpers and noncanonical metadata messages need boundary-specific scrutiny. A tree fixed by an operation atom is not exempted by NUM-12. |
| `[04-NUM-11]` | Storage and transport preserve exact bits and dtype. Tensor descriptors have dynamic rank with int32 rank and int64 extents, strides, element counts, and byte capacity. Checked address-domain projection does not replace semantic metadata. | Runtime descriptors are substantially coupled and private. Legacy GPU/Python descriptor paths and foreign assumptions remain separate obligations. A `Vec<f64>` image is not exact transport for all int64 values or NaN payloads. |
| `[04-NUM-14]`, `[04-NUM-16]` | Explicit checked casts obey the complete source/target product. Float-to-int requires finite integral input and distinguishes domain from range failures. Truncating, saturating, and wrapping casts are distinct names, never recovery defaults. Float-to-float AD follows the specified inverse cast. | `CheckedCastPlan`, typed cast dispatch, and finalizers implement a central path. Every decoder, list/tensor conversion, fold, Python import, and backend emitter must preserve that distinction. |

Sources: [typed numeric kernels](../../../crates/chelis-types/src/dtype_semantics.rs),
[runtime element representations](../../../crates/chelis-runtime/src/element.rs),
[runtime metadata](../../../crates/chelis-runtime/src/metadata.rs), and
[runtime ABI implementation](../../../crates/chelis-runtime/src/lib.rs).
`ElementStorage` is sealed, associates each representation with its arithmetic
type, and has representation assertions. This is stronger than choosing a byte
width in each emitter. It does not certify a transcendental implementation or an
accumulation tree.

### 2.2 Complete primitive-family account

These are obligation groups, not claims that all member implementations are
complete. All `[05-OP-1]` through `[05-OP-66]` appear explicitly. Atom identity is
only the outer unit: the incorporated callable registries and per-dtype/domain
rules remain part of the obligation.

| Exact atoms | Required semantics and adversarial domain | Enforcement disposition |
|---|---|---|
| `[05-OP-17]`, `[05-OP-18]`, `[05-OP-19]`, `[05-OP-41]`, `[05-OP-46]`, `[05-OP-47]`, `[05-OP-64]` | Wrap add/sub/mul are signed-integer-only; ordinary sub is direct, not negation-plus-addition with an extra minimum-integer trap. Unary arithmetic, floor/truncation, integer bit operations, and arithmetic distinguish their exact dtype domains, exceptional cases, derivatives, and primitive boundaries. | Typed numeric kernels and operation-specific dispatch are the main implementation boundary. Direct sub was also traced into exact constant folding. Other folds still use f64 images. OP-64's zero-divisor rule conflicts with NUM-9; see §8. |
| `[05-OP-6]`, `[05-OP-23]`, `[05-OP-24]`, `[05-OP-63]` | Checked, truncating, saturating, and modular conversions are different operations; preserve scalar/tensor shape, named failure, exact own-width source interpretation, and specified AD/rejection. | `CheckedCastPlan` and scalar/tensor cast kernels are centralized. Their coverage must include foreign ingestion and optimization, not just the direct builtin. |
| `[05-OP-20]`, `[05-OP-21]`, `[05-OP-22]`, `[05-OP-26]`, `[05-OP-27]`, `[05-OP-28]`, `[05-OP-36]` | Float predicates inspect stored values and yield zero cotangent. Bool logic is eager, ordered, and structurally non-differentiable. Recursive equality uses exact type/dtype-aware structure; comparison identity must survive lowering so AD does not mistake a comparison for forbidden logical operations. | Builtin identity registration and typed comparisons distinguish operation classes. Preserving identity through desugaring, host container recursion, and AD is an additional obligation; registration alone does not prove it. |
| `[05-OP-11]`, `[05-OP-12]`, `[05-OP-13]`, `[05-OP-14]`, `[05-OP-15]`, `[05-OP-16]`, `[05-OP-29]`, `[05-OP-30]` | Mean is the specified nonempty sum/division graph. Sum/count/product use prescribed leaf order and adjacent-pair trees, accumulator/result types, and empty identities. Multi-axis operations follow original-axis order. Extrema select first NaN/first equal representative, preserve selected bits, split tie adjoints as specified, and arg extrema return exact int64 indices with empty-domain rejection. Product AD reverses its actual tree, including zeros. Count is dedicated Bool-to-int64 reduction, not a cast-and-sum rewrite. | `reduce_group`, `checked_adjacent_pair_fold`, count kernels, `arg_reduce_tensor_groups`, and runtime reduction metadata are concrete owners. Sum/count have balanced paths; product remains a left fold in the inspected helper. Empty arg groups produce `-1` there; caller preguards were not exhaustively traced. This helper cannot independently establish the full atom. |
| `[05-OP-39]`, `[05-RWIN-1]`, `[05-RWIN-2]` | Window size/stride/padding lists have matching nonempty arity and checked domains; runtime metadata and all stated execution modes are legal. Sum/mean trees, max/min NaN/tie handling, overlap accumulation, saved forward choices, and higher adjoints are specified. No fixed-rank/static-only narrowing follows from an incomplete emitter. | IR reduction/gradient kernels and checked runtime metadata provide mechanisms. Window sum/mean currently call a sequential `reduce_sum_group`; window extrema have a separate NaN policy in that helper. Runtime-window tests explicitly contain compiled rejection expectations, which expose a capability gap rather than proving full support. |
| `[05-OP-40]`, `[05-OP-42]`, `[05-OP-43]`, `[05-OP-48]` | Direct extrema select operands, including first NaN/tie payloads, with their own adjoints. `stop_gradient` evaluates once, preserves effects, and stops recursive rejection/adjoint traversal. ReLU remains a distinct operation: its zero/NaN derivative is not a max tie derivative. Activations and softmax preserve the exact authored own-width graph. | Direct extrema and their constant folds use typed kernels; AD has distinct operation cases. Recognition/fusion must preserve these semantic identities, not merely a similar algebraic expression. Complete higher-order and all-lane closure was not established. |
| `[05-OP-7]`, `[05-OP-49]`, `[05-OP-50]`, `[05-OP-65]` | Shape reads return int64, rank int32, and scalar/tensor conversion is explicit. Movement preserves element bits and validates shape/axis/bounds. Expand changes an existing unit axis; insert adds one. Alias identities sharing IR/wire representation retain separate source contracts. | Runtime extent carriers, witness dependencies, shape metadata, and movement checks are substantial enforcement; §3 traces their limits. Shared IR representation is not authority to collapse source signatures or axis rules. |
| `[05-OP-51]` | Matmul follows explicit batch alignment, elementwise multiplication, canonical sum, and result cast. Einsum has its own exact contraction grammar/tree. Convolution has arbitrary spatial rank and checked stride/padding/output arithmetic; layer norm has explicit same-dtype epsilon and the stated graph, including all parameter adjoints. | Recognizers, generic IR lowering, evaluator helpers, and backend library specializations are distinct implementations. Inspected evaluator matmul widens payloads to f64 and accumulates sequentially. Vendor GEMM is legal only if it reproduces specified bits/traps; no current blanket opt-in authorizes a different tree. |
| `[05-OP-52]`, `[05-OP-66]`, `[05-SPARSE-1]`, `[05-SPARSE-2]` | Gather is selection, not one-hot arithmetic that touches unselected NaNs. Indices have the admitted signed dtype and bounds. Replace scatter has specified row-major winner; additive scatter has specified contributions/tree. Internal `OneHot` must be consumed by its specialization or rejected, not silently emitted. Every advertised sparse mode retains its support obligation. | Sparse IR dispatch and runtime helpers exist. Evaluator scatter-add's f64 image and sequential accumulation are source-visible fidelity risks. No complete source-to-all-device sparse-path execution is claimed. |
| `[05-OP-53]`, `[05-OP-62]` | Where selects exact bits; cumsum uses its specified prefix arithmetic; sort is stable with prescribed NaN/signed-zero/index handling; diagonal/trace have exact axes and reduction semantics; clamp validates bounds and follows exact selection branches; split/concat preserve dynamic metadata and element dtype. Tensor concat and list concat are distinct overload identities. | Runtime tensor callables and metadata planners own checks. Operation-specific algorithms and AD, not a generic tensor-carrier test, decide conformance. Full identity coverage is supplied by the registry; full implementation verification is not. |
| `[05-OP-8]`, `[05-OP-37]`, `[05-RNG-1]` | Same-dtype finite ordered uniform bounds, exact width/FMA mapping, validated dropout rate, deterministic seed/call/element stream, validation before ordinal consumption, and one ordinal even for accepted empty/rate-zero operations. Reverse execution reuses forward randomness and has exact pathwise parameter derivatives. | Evaluator carries random progress and typed helper entry points, but `uniform_sample` still accepts f32 bounds for every dtype and uses a different seed/index mixer; `uniform_like` narrows bounds before calling it. Dropout's helper uses f64 images and rate clamping. These are algorithm discrepancies, not missing tags. |
| `[05-OP-1]` | Decimal `round_to` uses the exact binary rational, decimal half-even decision, full signed places domain, one finalization, and specified signed-zero/nonfinite treatment; AD structurally rejects it as `PiecewiseConstant`. | Host/builtin identity and numeric registration give a route to the owner. A general float `round` test or formatting test cannot establish this algorithm. No complete algorithm proof or fresh execution was obtained here. |
| `[05-OP-2]`, `[05-OP-3]`, `[05-OP-4]`, `[05-OP-5]`, `[05-OP-61]` | JSON preserves int64, f64, and out-of-int64 integer token distinctions; accessors/constructors do not coerce them. Canonical serialization is exact. CSV remains text, with explicit typed projections and defined CSV grammar/row handling. | ADT signatures, exact exported definitions, source census, and host runtime paths establish distinguishable surfaces. Parsing/serialization/domain algorithms need separate validation, including nested and malformed inputs; no all-input proof follows from a tagged `JsonInt` constructor. |
| `[05-OP-9]`, `[05-OP-10]`, `[05-OP-54]`, `[05-OP-55]`, `[05-OP-57]` | Lists retain generic element structure, exact int64 counts/index domains, copy/drop semantics, and ordered callback effects. Nested tensor conversion is rectangular and dtype-preserving, including empty dimensions; padding has checked width/truncation and gradients to source/pad. Map/fold/scan/filter/partition AD preserves the executed trajectory and saved selection masks. | Host lowering, runtime containers, and the standard-library/index adjoint tests cover real paths. Tensor element storage does not prove nested-list shape reconstruction or callback multiplicity. Arbitrary nesting, runtime control, and compiled recursive AD remain separate coverage obligations. |
| `[05-OP-25]`, `[05-OP-56]`, `[05-OP-58]`, `[05-OP-59]` | `to_string` has its exact admitted domain, not arbitrary recursive objects. Dict keys have exact allowed static types and canonical observation order. Strings use Unicode scalar indexing without normalization. Named string-to-int/float parsers consume their full specified grammar; JSON and C parse grammars are not interchangeable. | Container/boundary identity registration gives each overload a disposition, including rejection-only variants. Runtime implementation and exact-output tests are required per domain. A registry row for a rejected `to_string` category does not make it accepted. |
| `[05-OP-38]`, `[05-OP-60]`, `[05-HOST-1]`, `[05-HOST-2]`, `[05-HOST-3]`, `[05-HOST-4]` | Host calls preserve complete types, shapes, dtypes, effects, and failure behavior. Tensor scan has exact recurrence/callback count, even for empty element shapes; zero steps call nothing. Assertions use own-width comparisons and specified NaN/inf/tolerance handling. IO/process/file/mmap operations have exact return types and effects; directory ordering and complete writes are observable. | Host-capability declarations, lowering, verified host emission, and runtime helpers are owners. Opaque containers do not prove IO completion, process argument handling, recurrence AD, or generic-result preservation. Entry/observation reporting is shared with integration. |
| `[05-OP-31]`, `[05-OP-32]`, `[05-OP-33]`, `[05-OP-34]`, `[05-OP-35]`, `[05-OP-44]`, `[05-OP-45]` | Exact C/binding/stdlib identities and their full declared domains are normative, including arithmetic metadata, aggregate fields, allocation, indexing, and lifetime calls. | The incorporated registries are expanded by structural domain in §2.3; runtime/lifetime and Python boundaries are traced in §§5–6. |

The main source dispatch points are
[numeric kernels](../../../crates/chelis-types/src/dtype_semantics.rs),
[IR evaluation](../../../crates/chelis-ir/src/eval.rs),
[builtin declarations](../../../crates/chelis-types/src/builtins.rs),
[runtime implementation](../../../crates/chelis-runtime/src/lib.rs), and the
[normative builtin identity registry](../../../spec/registry/builtin_semantic_identities.md).
The registry separates numeric, container, and boundary identities and associates
each with its exact atom. It does not prove the selected atom actually governs a
new callable's semantics; that remains a substantive review obligation.

The unnumbered construction/composition rules are obligations too. `const` and
`load` are pure tensor constructors, distinct from effectful random construction;
their source contributions have zero cotangent. Root-scoped evaluation must not
invent inputs for unrelated dead declarations, but must retain a load used only
as a live shape witness. Borrow-typed read-only primitives/queries may erase source
borrow markers only after recording ownership disposition; `index` retains its
returned element while leaving its container live. Consuming `realize` and `drop`
are not read-only queries, and materialization barriers constrain fusion.

Spec/05 §4 also defines complete library compositions beyond individual atoms:
cross-entropy selects the per-batch label via gather plus diagonal, not arithmetic
masking; embedding is exact gather across independent index/entry rank spreads;
multi-head attention has explicit batch/head axes, mask shape and same-dtype scale,
then the prescribed transpose/matmul/selection/softmax/matmul graph. A fully masked
row follows that nonfinite graph rather than receiving an invented zero default.
Softmax, layer norm, and convolution similarly retain their written graphs and
domain checks. A recognizer may replace them only while preserving values, bounds
failures, adjoints, and primitive finalization. Their complete execution across
all shapes/dtypes/lanes was not established here. Reference pseudocode in §6 is a
useful implementation description, not permission to ignore the stricter typed
and operation-specific rules elsewhere in the chapter.

### 2.3 Incorporated registries: more than atom membership

All eight incorporated registry files were inspected. Their structural domains
are accounted for here rather than treating a row count as semantic coverage.

| Registry and authority | Distinct obligations carried by its rows | Current enforcement / unverified remainder |
|---|---|---|
| [C scalar carrier](../../../spec/registry/c_scalar_carrier.md), `[05-OP-31]` | Tagged scalar creation/read/write, exact stored-width payloads, canonical reserved bytes, float bit handling, and descriptor/view transport. No raw dtype id or untagged numeric channel may substitute. | Typed scalar storage plus ABI validators; raw FFI memory validity and all callers remain assumptions/check obligations. |
| [C container boundary](../../../spec/registry/c_container_boundary.md), `[05-OP-32]` | Scalar extraction, list length/index/slice/append, tuple/ADT field access, dict key semantics/order, strings measured in Unicode scalars, byte values as int64 in 0..255, and observable writes. | Runtime value/container APIs are owners. Bounds, key equality, Unicode and write-completion algorithms are not established by the opaque handle alone. |
| [C tensor runtime](../../../spec/registry/c_tensor_runtime.md), `[05-OP-33]` | Allocation; rank/extents/strides/count/bytes; shape-only versus tensor-based indexing; reshape/copy; typed ingress/egress/padding; concat/split/sparse/selection/prefix/sort/diagonal/trace/clamp/contraction; coordinate encode/decode; movement target checks; and reduction-plan construction, index mapping, target/scratch checking, release. | `ShapeMetadata`, `IterationSpace`, `ReductionMetadata`, tagged arguments, and full target validation provide shared arithmetic checks. Metadata-only calls must not read payloads; all validation must precede writes. Kernel fidelity and unsafe caller memory remain additional obligations. |
| [C heap lifetime](../../../spec/registry/c_heap_lifetime.md), `[05-OP-44]` | Retain/release for opaque heap kinds; tensor read views, entry borrows, exclusive write guards, end-write validation, and storage repurposing. | Private heap header/storage provenance, refcounts, state checks, and unique-owner tests are substantial mechanisms; §5 specifies what they do and do not prove. |
| [Python metadata](../../../spec/registry/python_tensor_metadata.md), `[05-OP-45]` | `NativeTensor.shape` preserves full dynamic metadata using the specified int64 carrier and Python exact integers. | Binding currently spells `Vec<usize>`, not the registered `Vec<i64>` transport. Positive values on a 64-bit host may coincide; that does not satisfy the exact carrier contract or prove all GPU/target paths. |
| [Stdlib ADT identities](../../../spec/registry/stdlib_adt_identities.md), `[05-OP-34]` | All five nominal identities preserve exact declared numeric fields recursively, constructor identity, and the specified field-wise differentiable/discrete cotangent structure. A language `t-prim f64` field is already tagged but its constructor is still a numeric operation. | Checker/nominal representation and exact registration provide admission and identity; numeric meaning and AD cannot be inferred solely from their presence. |
| [Stdlib numeric manifest](../../../spec/registry/stdlib_numeric_manifest.md), `[05-OP-35]` | Every exported numeric definition has an exact signature/effect identity. The groups below include all manifest module families. | Census/registration closes discoverable identity accounting; implementation algorithms for all definitions were not individually proved in this investigation. |
| [Builtin semantic identities](../../../spec/registry/builtin_semantic_identities.md), `[05-OP-1]`–`[05-OP-66]` as mapped | Numeric table identities, host boundaries, and generic container overloads retain exact identities, signatures, domains, effects, adjoint/rejection, and target disposition through elaboration and lowering. | Closed builtin declarations and atom-closure tests prevent an unregistered identity from being silently classified. They cannot make an incorrect registered kernel correct. |

The complete stdlib-manifest family domain is:

- `contracts`: normal CDF's prescribed graph and stronger inner-exp rounding,
  fixed sample/seed/tolerance helpers; a numerical approximation is not free to
  choose another formula merely because it lies within a broad tolerance.
- `decimal`: exact rational arithmetic and comparisons, canonical coefficient/
  scale, parsing/rendering and explicit rounding modes; intermediate host widths
  cannot invent overflow before the specified final int64 projection.
- `index`: list index/take/drop domains and shape-preserving source gradients.
- `init/kaiming`, `init/random`, `init/xavierext`: normal/truncated-normal and
  initializer graphs, exact bound/scale validation, random-consumption order,
  pathwise derivatives, and stronger internal log/cos rounding where specified.
- `io/json`: all projections, parse/load/serialize/write and `try_*` variants;
  full grammar and arbitrary nesting, exact integer distinctions, and validation
  before a write truncates its destination. `io`: mmap size and byte prefixes.
- `process`: argv execution without a shell and the explicit executable
  capability for `run_chelis`; exit code/stdout/stderr are not interchangeable
  with successful compiler execution.
- `scalar`, `sort`: scalar arithmetic/selection and stable value/index results.
- `tensor/construct`, `tensor/mask`: checked exact arange, rational linspace
  weights/finalization, squeeze/unsqueeze/stack shape rules and mask indices.
- `test`: exact equality, shape assertions, and own-width closeness semantics,
  including exceptional values and tolerance domains. The test primitive is
  itself part of the language, not an exemption from numeric obligations.
- `time`: proleptic Gregorian dates over the specified full integer domain,
  exact ordinals/durations/arithmetic, parse/format and comparison semantics.
- `tokenizer`: validated vocabulary bijection and merge ranks, specified
  leftmost tie resolution, encode/decode/batch padding and error behavior.

These groups require both accepted and rejected inputs, exact effects, numeric
edges, and AD/rejection where applicable. Counting the exported definitions does
not establish any of those algorithms.

### 2.4 Concrete algorithm limits behind typed carriers

Several inspected functions show why the carrier and algorithm questions cannot
be merged:

1. `reduce_group` uses `checked_adjacent_pair_fold` for sum but initializes product
   to one and iterates a left fold. Window sum/mean call `reduce_sum_group`, also a
   left fold. The controlling atoms prescribe trees. Their source discrepancy
   persists even though each individual multiply/add uses a typed kernel.
2. `uniform_sample(prim, low: f32, high: f32, seed, index)` cannot itself preserve
   same-dtype f64 bounds. Its mixer also differs from RNG-1's seed/call/element
   construction. Evaluator `uniform_like` visibly casts bounds to f32. This is
   stronger evidence than the mere absence of a random test, but no end-to-end
   random reproduction was run here.
3. Evaluator `matmul`/batched matmul and `scatter_add` use f64 payload images and
   sequential accumulation before finalization. Own-width arithmetic and exact
   int64 transport, where admitted, are not consequences of the final result's
   tag. Every public route using these helpers requires separate disposition.
4. `arg_reduce_tensor_groups` returns `-1` for an empty group. Whether each source
   caller rejects the empty domain first was not established. The safe conclusion
   is that this helper is not an independent enforcement boundary for OP-15/16.
5. `eval_tensor_with` and `eval_tensor_roots_with` select non-strict input handling;
   explicit `*_strict` entry points select strict handling. A production claim
   about missing-input rejection must prove it chooses the strict route. The
   existence of a strict API does not remove the other public route.

These are not an exhaustive bug list. They are explicit dispositions of shared
enforcement mechanisms that affect many obligations. Other operation algorithms
remain unverified, rather than presumed correct because no discrepancy was found.

## 3. Dynamic shapes, capacity identity, and access safety

This section covers `[04-SHAPE-1]`, `[05-AXIS-1]`, `[05-DIM-1]`, `[05-DIM-2]`,
`[05-DIM-3]`, `[05-MOV-1]`, `[05-SHAPE-1]`, the shape portions of `[05-OP-7]`,
`[05-OP-33]`, `[05-OP-45]`, `[05-OP-49]`, `[05-OP-50]`, `[05-OP-65]`, and the
unnumbered runtime-shape rules in spec/04 §4.7 and spec/06 §3.7.

Five different facts must not be substituted for one another:

| Fact | Required meaning | Inspected carrier/check |
|---|---|---|
| Static shape/type claim | Axis identity, rank, dtype, literal/list arity, and symbolic equality constraints; no implicit elementwise broadcasting. | Checked tensor types and lowering; language chapter owns the static judgment. |
| Runtime extent provenance | The particular input-axis or computed int64 value that witnesses a claim, including source context and invocation scope. | `RtDim`, `RtAxis`, `ExtentWitness`, `shape_deps`, and `axis_sources`. |
| Semantic capacity equality | Equality in the specified exact mathematical domain, including quotient validity and source identity. Runtime coincidence or a machine-size hash is not proof. | Private `CapacityKey` and explicit `prove_equal`, conservatively returning `NotProvenEqual`. |
| Valid runtime metadata | All extents/strides/counts/bytes have checked signed semantic domains, even when total element count is zero. | Private `ShapeMetadata`, `ElementCount`, `ByteCount`, `IterationSpace`, `ReductionMetadata`. |
| Safe physical access | Target address-space conversion, byte capacity, alignment, initialization, index bounds, provenance, and live ownership all hold now. | `AllocationBytes`, runtime descriptor/storage validation, checked index plans, and read/write ownership state. |

Relevant sources are [DAG carriers](../../../crates/chelis-ir/src/dag.rs),
[axis sources](../../../crates/chelis-ir/src/axis_sources.rs),
[lowering](../../../crates/chelis-ir/src/lower.rs),
[capacity keys](../../../crates/chelis-ir/src/capacity_key.rs), and
[checked metadata](../../../crates/chelis-runtime/src/metadata.rs).

The language permits arbitrary admitted int64 computations to produce extents;
it does not restrict them to a syntactic literal or special `shape` expression.
Computed `shape` axes are int32 and normalized against the unbatched rank as
specified. Other operations have their explicitly static/named axis domains.
Confusing these domains can either reject a supported program or read the wrong
axis. A shape list may have statically known arity and dynamic elements.

Interface extent guards run once, in signature/ABI slot order, before computation;
local guards follow producer/source order. Their diagnostics preserve the owning
int64 operation plus separate source/axis/value context. A fallback extent of one,
a comparison of two copies of the same claim, or evaluating a witness only after
allocation cannot enforce this rule. Even a discarded local result can carry an
obligatory invocation witness.

`prepare_parameter_witnesses`, `preserve_literal_result`, and
`retain_invocation_witnesses` make those dependencies explicit in lowering.
`ExtentWitness` and `shape_deps` give optimizer/rebuilder/transport paths something
to preserve. The tests in
[runtime_extent_literal_transport.rs](../../../crates/chelis-ir/tests/runtime_extent_literal_transport.rs)
address witness survival, unrelated roots, and rebuild/transform cases. The wire
admission side of the same invariant belongs in the integration chapter. A
transported literal value without its distinct requirement witness is not enough.

`ShapeMetadata` validates all negative extents before exploiting a zero product,
computes checked suffix strides separately, and couples rank, shape, dtype,
element count and byte count. `IterationSpace` represents traversal cases that
need no unnecessary storage-stride product. `ReductionMetadata` validates axis
ordering, result shape and leaf/output index domains. Consequently a zero-sized
tensor cannot automatically hide invalid later metadata, but a metadata result
does not prove payload initialization or lifetime.

The current `CapacityKey` deliberately lacks public `Eq`/`Hash`; compile-fail
coverage prevents consumers from replacing `prove_equal` with structural equality
or hashing. Its private canonical representation uses arbitrary-precision
products and validity-aware quotient handling. Unsupported expressions yield
`NotProvenEqual`, an allowed conservative refusal. Literal byte projection is
checked separately. This establishes a materially stronger equality boundary than
symbol spelling, `u64` multiplication, or observed equal extents. Correctness of
the canonicalization and completeness of every builder/source-identity path
remain obligations; an opaque key cannot prove its own constructor correct.

Residual paths include arbitrary runtime-bound lowering, transformation of
computed axes, graph mutation/rebuild, symbolic operation results, empty/rank-zero
shapes, legacy device descriptors, and every C helper that forms an offset.
Source inspection found real guards and tests, not a complete path-closure proof.
The runtime-extent oracle reinforces this limit: only slice `a` and `b` are
registered at this snapshot; requested `c` or `final` cannot truthfully certify the
whole contract merely because those names appear in its CLI choices.

## 4. Transformations and scheduling

### 4.1 Differentiation is its own semantic relation

[Spec/06](../../../spec/06-transformations.md) does not merely require a gradient
with the right shape. It requires the specified derivative/adjoint of the actual
executed graph and its numeric choices.

| Subject in spec/06 | Obligation | Inspected implementation and residual |
|---|---|---|
| `grad` signature, `wrt`, recursive cotangents | Scalar output restriction; requested targets retain order and full arity, including uninfluential zero gradients and repeated equal roots. Lists/tuples/ADTs preserve recursive primal shape/constructor; discrete fields receive unit. | `grad_dag_checked` checks tensor-DAG admission; host/pytree lowering must preserve the broader recursive contract. A raw DAG `filter_map` of available gradients is not proof that the user-facing multi-target result fills every zero slot. |
| Reverse graph and contribution order | Shared forward nodes are differentiated as a graph, not duplicated syntax. Contributions are ordered by original forward consumer ordinal and input slot, then accumulated with the specified positive-zero/adjacent-pair scheme. | `grad_dag_result` queues and sorts `(consumer_position, input_slot, gradient)` and uses `balanced_adjoint_sum`. `Dag::topological_order` is currently append/NodeId order, so this use does not itself introduce an alternate topological numbering. Mutation preserving that premise remains important. |
| Operation-specific AD disposition | An operation has an exact adjoint, zero cotangent, or structural rejection. Comparison/shape/index paths, bool logic, discrete casts/reductions, stop-gradient, direct extrema/ReLU, and saved random masks cannot be merged into one generic differentiability rule. | Checked AD traverses live nodes and distinguishes shape-bound/index inputs; individual operation cases exist. Every host overload and higher-order reconstructed graph still owes the same disposition. |
| Reductions, sparse operations, movement | Reverse the prescribed forward tree and selected values, including NaNs, signed zero, ties, duplicate indices, zeros, overlap, and dynamic shape reconstruction. | Numeric adjoint helpers and gather tests provide concrete paths. Correct forward tags or an approximate finite-difference result do not prove these discrete choices or exact tree semantics. |
| Runtime `if`, `match`, loops, folds, recursion | Differentiate the actually executed finite trajectory; conditions/discrete branches carry zero cotangent. No language-level fixed differentiation-depth cap. Untaken branches are not evaluated and contribute no cotangent; this does not waive §2.7's structural rejection rules. | Tensor-DAG AD alone cannot establish host dynamic control-flow and recursive AD. Some host/ADT tests exist, but complete runtime-condition/compiled/higher-order coverage was not demonstrated. |
| Effects and randomness | AD is not an effect handler. Preserve forward effect multiplicity/order; seeded forward/reverse execution uses the prescribed stream and saved random source choices. `with device` selects a checked resource boundary, not another DAG transform. | Random progress, checked effects, and operation adjoints are separate mechanisms. The RNG algorithm discrepancies in §2 preclude deriving this guarantee from stream bookkeeping alone. |
| Checkpointing | Recompute the same required primal values without changing effects, random choices, traps, aliasing, or derivative semantics. | The specification supplies the obligation; a complete checkpoint implementation/path census and execution proof were not established here. |

Sources: [grad.rs](../../../crates/chelis-ir/src/grad.rs),
[grad gather contract tests](../../../crates/chelis-ir/tests/grad_gather_contract.rs),
[low-precision gradient tests](../../../crates/chelis-ir/tests/bf16_f16_grad.rs), and
[stdlib adjoint integration tests](../../../crates/chelis-cli/tests/issue_1293_stdlib_adjoints.rs).
The last suite exercises linked stdlib/index gradients through evaluation and
compiled execution in its test bodies; it was not run here. Some low-precision
gradient tests assert type/shape only. They establish neither derivative values
nor bit-exact arithmetic. Finite differences are useful independent evidence for
smooth interior cases, but need explicit step/tolerance/dtype choices and cannot
replace exact adjoint tests at discontinuities, NaNs, tied selections, or integers.

The formal AD section additionally requires an acyclic, type-consistent combined
graph and complete treatment of requested targets. The named error domains are
also distinct: `non_differentiable` for a requested type with no continuous leaf;
`no_differentiable_path` is a warning with a valid zero gradient, not rejection;
`non_scalar_grad_output` for an unseeded nonscalar result; and vmap's
`axis_out_of_bounds`/`batch_varying_extent`. A JIT shape cache miss requests another
compilation, not a semantic shape-mismatch rejection. Existing checked AD entry
points provide some of these distinctions; all public surfaces and exact
diagnostic behavior remain unverified here. Higher-order validity depends on every
reached operation's higher adjoint, not merely the syntax `grad(grad(f))` parsing.

### 4.2 vmap, JIT, composition, and optimization

`vmap` must implement the specified batch-independent mapping while shifting axes
correctly. Non-tensor/shared values stay shared. A scalar subgraph computing a
shared extent runs once outside the batch; an element-varying bound that would
produce a ragged result is rejected. Computed axes must be normalized in the
unbatched domain before shifting. Arbitrary nonzero batch axes are handled by
the specified permutation relation, not by pretending every request is axis zero.

[vmap.rs](../../../crates/chelis-ir/src/vmap.rs) contains axis-zero rewriting,
`shared_bound_nodes`, bound-slot analysis, and explicit movement/reduction shifts.
Its `shift_input_axis` increments a literal `RtAxis::Lit` but clones other cases,
including `RtAxis::Node`, unchanged. The surrounding complete computed-axis route
was not proved here; that helper alone does not enforce the computed-axis rule.
Its shared-bound fixed point also has an operation whitelist. A whitelist can
legitimately reject a capability gap, but cannot redefine the language's admitted
arbitrary scalar extent computations.

`jit` preserves the function's meaning and requires the specified specialization
identity, including function, names, sizes, and precisions. Optimization cannot
cross its boundary in a way the contract forbids. The source tests in
[jit_par_runtime_gap.rs](../../../crates/chelis-cli/tests/jit_par_runtime_gap.rs)
exercise scalar/tensor eval and compiled results. Identity/pass-through examples
do not establish the specialization cache key, boundary preservation, or all
`grad`/`vmap`/`jit` compositions. Cache admission is shared with integration.

Optimization preserves values **and** stored bits, traps/order, dynamic guard
dependencies, selected branches, effects, RNG ordinals, roots, and ownership
facts. Spec/06 orders AD before the relevant recognition/fusion, then DCE and
in-place work. Fusing arithmetic must preserve primitive finalization; replacing
an expression by a mathematical identity is not generally legal under IEEE
exceptions or checked integer overflow. Reassociation needs the operation's
permission, not just a performance rationale.

The inspected [optimizer](../../../crates/chelis-ir/src/optimize.rs) exposes three
important boundaries:

- Constant folding routes direct subtraction/extrema through typed kernels, but
  legacy add/mul/comparison and some unary folds use `wide_image: f64`. Declining
  inexact integer *inputs* and unfinalizable results does not prove arithmetic on
  exact inputs has the prescribed intermediate/result semantics.
- CSE keys use debug spelling of the op, remapped inputs, and shape dependencies;
  they do not include the output type. The pass skips `ExtentWitness`, but that
  exception is not a general proof of effect/trap-safe equivalence. Whether every
  caller guarantees identical types and legal CSE eligibility was not established.
- DCE roots stores and observable graph roots and follows shape/reuse dependencies.
  This is useful dependency preservation, but not an all-operation proof that a
  discarded node has no required trap/effect/guard. A dedicated witness policy
  cannot substitute for the complete semantic liveness condition.

Rebuilding a valid DAG or successfully verifying ownership after a transform is
valuable, but it proves neither the original-to-transformed numeric relation nor
that the new graph retained every observable obligation.

Spec/06 §9 also owes termination of local cleanup loops and the stated standard
pipeline, including post-AD closed-list no-op cleanup, recognition,
cross-function specialization, settled DCE liveness, and in-place fusion.
`SPECIALIZATION_PIPELINE_ORDER` is the named source tripwire. An iteration cap can
bound a cleanup loop; non-increasing node count alone does not prove convergence
when rewrites can change a graph without changing its size. No termination or
fixed-point execution was measured here. The general early/post-transform
pipeline in §6.3 must respect these more specific ordering constraints.

### 4.3 Concurrency and backend scheduling

[Spec/07](../../../spec/07-concurrency.md) permits independent pure DAG nodes to
be scheduled in parallel. `par` may execute sequentially or concurrently only
with the specified result agreement. This is not permission to reorder IO,
random consumption, traps, guard evaluation, or dependent storage reuse. Streams,
actor systems, shared mutable-state concurrency, and scatter/gather-focused
parallel runtime features are explicitly non-goals of this concurrency model.
The tensor gather/scatter operations are not evidence of such a parallel runtime.

Scheduling correctness therefore depends on complete data/shape/effect/ownership
dependencies and synchronized foreign completion. Chelis#2388 currently fences
`par` in the checker because compiled host lowering can erase non-final effects;
therefore the retained sequential lowerer supplies no runtime acceptance
evidence. Atomic reference counts
protect a lifetime protocol; they do not independently make all payload access,
device queues or callbacks race-free. No concurrent execution was
performed for this investigation.

## 5. Ownership, lifetime, and the physical runtime

This section covers `[04-LIN-1]` through `[04-LIN-8]`, `[05-OP-44]`, the lifetime
parts of `[05-OP-31]`–`[05-OP-33]`, and spec/08/spec/11 execution lifetimes. Static
binding/borrow admission is shared with the language chapter.

### 5.1 From logical owners to verified emission

| Authority | Runtime/lowering obligation | Inspected boundary |
|---|---|---|
| `[04-LIN-1]`, `[04-LIN-2]` | Track binding identity, not spelling; closure capture acquires its ownership obligation at creation. Aggregate tensor-containing values carry transitive ownership. | Ownership lowering uses explicit identities/actions, with tests for capture/copy and call edges. Checker admission alone cannot establish later closure lifetime. |
| `[04-LIN-3]`, `[04-LIN-4]` | Exactly one logical owner and one terminal disposition along each executed path. Owned and borrowed internal parameters are different call edges; required copies are explicit. | `lower_host_ownership`, `lower_dag_ownership`, ownership sites/actions and verifier distinguish these relations. |
| `[04-LIN-5]` | Branch joins and loop/fold block parameters transfer ownership without duplication or disappearance. | Host verification and last-use scheduling have join/loop cases and source tests; complete dynamic-path execution remains unverified here. |
| `[04-LIN-6]`, `[04-LIN-7]` | Root manifests consume in order, copying aliases as necessary. External entry arguments are borrowed for the whole call; internal owned calls copy rather than consume them. Returned roots own independent live values. | Root materialization, manifest-sink verification, verified host emission, and entry-borrow runtime storage. Root enumeration itself is also an integration obligation. |
| `[04-LIN-8]` | Reclaim after terminal use before the next tail/back edge; reuse requires unique writable program-owned storage, never borrowed entry/view provenance disguised by a physical copy. | Last-use actions, live-set bounds, and private reuse tokens; runtime write guard/repurpose checks provide a second boundary. |

In [ownership/mod.rs](../../../crates/chelis-ir/src/ownership/mod.rs),
`OwnershipProgram` couples an emission payload and ownership information;
`VerifiedOwnershipProgram` is privately constructed by `verify_ownership`.
Verification checks the host/DAG case, manifest sinks, sites, and nested tensor
DAGs. Backends receive immutable emission views rather than arbitrary mutable
payloads under a detached proof. This reduces stale-proof and post-verification
mutation paths, subject to the correctness and completeness of the verifier.

[C codegen](../../../crates/chelis-backend-c/src/lib.rs) consumes verified DAG/host
programs. [Storage planning](../../../crates/chelis-ir/src/ownership/storage.rs)
uses `ExactStorageCapacity`, representation identity and capacity proof;
`ReusableOwnedStorage` has private minting from the required candidate facts and
consumer-specific single-use extraction. Placements distinguish entry borrow,
input mirror, owned slot, shared view, drop, and materialized store. Physical HIP
mirroring does not convert external semantic provenance into reusable ownership.
Metal has an explicit never-reuse plan rather than an invented reuse proof.

The source tests in
[ownership_lowering.rs](../../../crates/chelis-ir/tests/ownership_lowering.rs)
cover ordered aliased roots, explicit copy, borrowed-to-owned calls, capture,
joins, and drops before tail transitions. Many assert lowered structure. Runtime
allocation, temporal use, actual peak live bytes and device execution need the
separate ownership oracle and ledger, not just these structural assertions.

### 5.2 Runtime storage and temporal contracts

The [runtime implementation](../../../crates/chelis-runtime/src/lib.rs) has a
closed heap-kind header and atomic reference count; tensor descriptors own
private storage with `RuntimeOwned` versus `EntryBorrowed` provenance. Retain uses
checked relaxed count updates; release uses release ordering and an acquire fence
for final destruction. The closed finalizer releases children once and frees
tensor backing memory only for runtime-owned storage. Option has its prescribed
single heap-node representation; this is not permission to introduce cycles or
arbitrary mutable graphs into the immutable acyclic ownership contract.

`validate_data_contract` checks capacity/address conversion, nonempty null data
and alignment. `validate_tensor` checks kind/write state and canonical bool
payloads where required. Entry borrowing validates metadata and data contract but
never promotes, mutates, or frees the caller's backing memory.

`chelis_tensor_begin_write` requires a unique descriptor, unique storage,
runtime-owned provenance, and no active write state. The embedded guard belongs
to that tensor; its write view is checked against the live guard.
`chelis_tensor_end_write` validates bool payloads before returning to idle.
Repurposing requires the same uniqueness/provenance conditions plus validated
metadata and exact byte capacity. These checks materially block mutation through
shared owners and entry borrows.

They do not make raw C views statically safe. The caller must not retain a read
view across invalidating mutation/release, use a stale pointer, provide unreadable
memory, falsify initialized length, or concurrently violate exclusivity. The spec
does not promise diagnosis of every invalid stale handle. A runtime check cannot
safely dereference an arbitrary hostile pointer merely to discover it is invalid.
These are explicit FFI preconditions/trust assumptions, not hidden guarantees.

The distinction also matters for failures. `runtime_fail!` prints and calls
`std::process::exit(1)`, and `metadata_or_fail` formats Domain/Overflow context
directly. Process failure can be the final compiled-program boundary, but an
embedded caller may be inside a larger process; NUM-10 requires an internal lane
error value rather than an arbitrary deep exit. Each exposed call and wrapper
needs a boundary argument. This chapter did not establish one for all helpers or
execute a trap through the Python embedding path.

Tests in
[checked_metadata.rs](../../../crates/chelis-runtime/tests/checked_metadata.rs)
exercise actual checked constructors with normal/negative/overflow/zero-size and
reduction-index domains. Tests in
[tensor_write_guard.rs](../../../crates/chelis-runtime/tests/tensor_write_guard.rs)
include child-process negative cases for shared/entry/active-guard states and
positive writes, but are feature-gated on `ownership-ledger`. Selecting the file
without its feature can execute none of those tests. No stale-pointer test should
claim to validate arbitrary undefined foreign behavior.

## 6. Backends, configurations, and foreign execution

### 6.1 Backend capability is a product, not a backend name

`[04-TGT-1]` and its surrounding text require rejection of every f64 value in a
program using `--target metal`, before kernel emission and without a software
substitution. However, `[05-OBS-10]` requires roots outside the selected target's
Tensor capability to route through Host without changing dtype. These leave an
unresolved authority boundary between target-wide rejection and kernel-lane
rejection; this account does not silently narrow the first rule to reconcile them
(see §8.1). Neither rule authorizes f64 Metal kernels. The target tables also
distinguish other active tensor dtypes and conditional device support.
Unsupported implementation cells must use
`[05-UNS-1]`–`[05-UNS-6]`'s earliest competent typed failure and independent
backend backstop; they cannot emit a default result or redefine the language.

[Target capability declarations](../../../crates/chelis-compiler-api/src/target_capability.rs)
give closed Tensor-lane target/dtype sets and explicitly describe Host routing
for definitions exceeding those sets. That routing does not resolve the normative
Metal f64 conflict. The inspected Metal set also omits int8/int16 while
the normative target table admits them. This is a source-visible support
discrepancy, not evidence of silent wrong execution. Per-operation, dynamic-shape,
host-result, device-generation and compiler-option support still require separate
cells. [Typed unsupported errors](../../../crates/chelis-types/src/unsupported.rs)
and [closed C expressions](../../../crates/chelis-backend-c/src/emitted_expr.rs)
help prohibit unclassified fallback emission; they do not prove every caller
reaches those checks.

| Lane/configuration | Obligation and assumptions | Evidence boundary |
|---|---|---|
| Evaluator | Execute the admitted graph with correct stored values, dynamic guards, effects, roots, traps and AD. | It is an implementation, not mathematical authority; shared typed kernels can make two lanes agree on the same error. Strict versus convenience evaluation routes matter. |
| C reference/CPU | C integer arithmetic must avoid UB; floats must use authorized widths/finalizations; allocation/indexing/lifetimes and process results must match. Optimization level, contraction/fast-math, libm, SIMD, OpenMP and BLAS selections can change behavior and must be included in claims. | Generated source inspection or C compilation is not execution. Eval-versus-C comparison also needs an independent expected result where implementations share algorithms. |
| HIP | Host control, generated kernels, hipRTC/runtime/compiler versions, transfers, synchronization, error propagation and allocations must preserve the same semantics. hipBLAS specialization has no exemption from exact matmul trees. | Actual AMD hardware execution is distinct from Rust unit tests, emitted HIP text, C++ compilation, and a zero-match ignored suite. |
| Metal | MSL/runtime/device capability must agree with admitted dtypes; f64 cannot enter kernel emission. Target-wide rejection versus Host routing remains the authority conflict above. Objective-C++ ARC, MTL buffers, command completion and MPS wrapper scopes must keep objects alive for execution and reclaim them afterward. Library/tiled specializations owe the same numeric result. | Source contains family/version checks, autorelease handling and completion waits. This does not verify every device family, toolchain, numeric mode, or driver. Hardware-generation descriptions were not independently validated here. |
| External libraries/compiler | Correct code generation, ABI compatibility, IEEE behavior under selected flags, libm/device-library accuracy, thread/runtime synchronization and allocator behavior are assumptions or separately tested contracts. | A library name or vendor guarantee cannot silently override the narrower language operation rule. No external toolchain validation was performed. |

Spec/08's StableHLO and FX integrations, and its later target discussion including
Triton, are future/additive backend scope, not evidence that C/HIP/Metal already
validate those ecosystems. Interactive Tide/eval execution is not another backend:
the stated route begins with the IR evaluator, with later latency-driven cache,
persistent-helper and JIT options. No Cranelift backend is inferred. Spec/11's
embedding rule applies the same source/value/diagnostic/admission contract to Rust
library entries and external integrations; a CLI-only guard cannot satisfy it.

Spec/05 §4.1 explicitly requires matmul's multiplication/finalization and OP-30
balanced reduction, and says that a future vendor-kernel opt-in would need its own
named semantic authority. Spec/08's optimization/library implementation discussion
must be read subject to that numeric contract. “Uses BLAS/MPS” is not a numerical
soundness argument. Nor does NUM-12 grant tree freedom where these operations pin
one tree.

`[05-OBS-1]`–`[05-OBS-11]` are primarily covered in integration, but OBS-3 is a
numeric enforcement condition here: only its named transcendental operations get
their specified ULP allowance; sqrt and absent operations require exact agreement.
For f16/bf16, the f32 arithmetic result must be checked **before** narrow storage
rounding where the rule requires it. Two final half values landing in the same
rounding bin cannot establish a prefinal ULP bound. NaN/signed-zero rules are not
ordinary real-valued tolerances.

The inspected [Metal GPU suite](../../../crates/chelis-backend-metal/tests/gpu_correctness.rs)
has ignored execution cases and historical absolute/relative tolerances such as
`1e-4` and an exp-specific `1e-3`. Such tests can detect gross backend errors but
cannot certify the current exact/ULP contract for their whole operation set.
The [HIP GPU suite](../../../crates/chelis-backend-hip/tests/gpu_correctness.rs)
likewise requires explicit hardware selection/execution and per-case comparator
inspection. Merely including a hardware row in an oracle manifest is not a run.

Cargo features are another explicit boundary. The workspace-manifest scan found
the following relevant feature families; no combination was built here:

| Feature/configuration | Disposition in this account |
|---|---|
| C backend `sleef` | Alternative numerical-library path requiring its own accuracy/configuration evidence. `chelis-backend-c/build.rs` also enables the cfg when `pkg-config` finds Sleef, so the Cargo command alone does not fully identify this environment-dependent selection. |
| Runtime `ownership-ledger` | Instrumented lifetime/allocation evidence and feature-gated tests; default runtime execution and instrumented execution are distinct claims. |
| IR `lowering-trace`; types `checkpoint-compile-probe` and `generalize-sweep-oracle`; compiler API `emission-observer` | Diagnostic/probe/observation configurations. Their intended observational role does not prove noninterference; configuration-specific behavior and selected tests remain unexecuted here. Static-checker/probe semantics are shared with language and integration. |
| E2E `hip-local-gpu` | Explicit opt-in hardware cross-validation; default E2E green cannot establish it. |
| Python `extension-module` versus default library/test build | Different embedding/linking configuration; ownership, exception, GIL and ABI behavior require evidence for the actual published module. |
| CLI default `chelis-prove` and `smt`; Tide `smt`; prove `smt`, `z3`, `clarabel`, `carcara`, `arb` and implicit optional-dependency features | Proof-engine/library and report configurations, primarily owned by integration. Exact-rational proof arithmetic or interval validation is not automatically a theorem about runtime floating-point execution. Their foreign library/build assumptions remain explicit. |
| HIP and Metal backend manifests | No local Cargo feature table was found in these manifests. Hardware, generated-language flags, library selection, OS and toolchain still create configuration distinctions. |

This accounts for the feature names discovered in the workspace manifests, not
every possible transitive dependency-feature combination. Dependency-feature
unification, target cfgs, debug/release behavior, sanitizer builds, native build
scripts and environment-selected libraries need their own exact-build record.

### 6.2 Python, DLPack, and artifact invocation

[Spec/11](../../../spec/11-ffi.md) requires exact dtype/shape/stride/capacity
admission, appropriate Python error translation, ownership across calls/results,
and the specified GIL/execution behavior. Artifact ABI version admission precedes
source/library interpretation. Function selection is exact; selecting a callable
is not executing the whole source as an approximation. Compiled callable and
evaluator execution owe agreement for corresponding operations in their shared
admitted domain, not identical admission or execution scope. Spec/11 §1.4 explicitly
rejects scalar-signature entries at the callable tensor interface while evaluation
admits scalar results. Linked compilation scopes to the selected entry; evaluation
retains whole-program semantics. `compile_and_load` can discover Reef context from
an importing Surf source, whereas raw-text evaluation requires an explicit
`project_root` for Reef dependencies. These differences must be preserved rather
than erased by a blanket parity claim. Artifact decoding and identity are shared
with integration.

The [Python binding](../../../crates/chelis-python/src/lib.rs) has `TensorOwner`
and CPU/GPU handle destructors; CPU tensors use opaque runtime handles.
`create_dlpack_capsule` holds a cloned owner with shape/stride storage and deletion
callbacks, providing an explicit lifetime mechanism. Dtype reporting uses the
actual runtime dtype rather than a fixed float32 string.

However, `NativeTensor.__dlpack__` explicitly discards `stream`, `max_version`,
`dl_device`, and `copy` before creating a capsule. Spec/11 requires these protocol
requests to be honored or rejected according to their semantics, not ignored.
Capsule ownership does not establish stream synchronization, device negotiation,
copy requests, or versioned flags. This is a direct source discrepancy; no Python
consumer was executed here.

The same file's `ChelisGpuTensor` retains fixed-size shape/stride arrays and i32
size fields, distinct from the dynamic int64 runtime contract. Some conversions
fail loudly when values do not fit, which is safer than silent narrowing but
still does not implement the declared full descriptor domain. `NativeTensor.shape`
returns `Vec<usize>`, conflicting with OP-45's exact carrier registration.
Metadata conversion and physical device descriptors therefore need a complete
route census; the corrected CPU runtime cannot certify the legacy GPU path.

These foreign boundaries assume a cooperating caller where the API requires one.
Effect annotations and opaque handles are not a sandbox against arbitrary native
code, invalid foreign pointers, malicious dynamically loaded libraries, or an
untrusted driver. No broader isolation guarantee is inferred.

## 7. Tests, censuses, and executable closure

The commands below are existing invocation surfaces identified from source.
They were **not run** for this chapter. A later execution must record the exact
head, selected cases, feature/configuration/target, per-test outcomes, comparator,
and required hardware rather than copy the command as evidence.

| Surface / invocation | What its inspected design can establish | What a green marker cannot establish |
|---|---|---|
| [Capacity census tripwire](../../../crates/chelis-cli/tests/capacity_census_tripwire.rs) and its [authority classifier](../../../tests/support/capacity_census_authority.rs); selected by dtype capacity/oracle runners | Enumerated public Rust/C/stdlib identities have exactly one final nonnumeric, exact tagged-carrier/transport, or exact registered numeric-operation authority. Header closure and conservative numeric classification prevent raw arithmetic surfaces from hiding as plumbing. | That an enumerator covers a new surface kind, that an atom semantically governs the callable, or that a registered algorithm implements it. |
| [Wire census](../../../crates/chelis-compiler-api/tests/capacity_census_wire.rs), Python binding census, representation inventories | Additional wire/PyO3 surfaces have their required structural admission and execution/mutation legs; exact identities cannot be blessed by an editable disposition alone. | Primary-census success cannot substitute for these separate entry gates. The binding legacy cohort is not authorization for new/changed untagged numeric rows. |
| `.venv/bin/python scripts/dtype_capacity_oracle.py` | Source/identity-bound targets and real per-test execution receipts, including negative selection/receipt controls. | Arbitrary numeric correctness or omitted family/configuration coverage. |
| `.venv/bin/python scripts/dtype_builtin_atom_closure_oracle.py` | Live builtin identities reconcile with atom/registration rows and selected executed tests; missing/skipped outcomes are rejected. | Semantic correctness from a citation string or proof of every domain edge. |
| `.venv/bin/python scripts/dtype_direct_arithmetic_oracle.py` | The named direct-arithmetic source contracts and executable legs, including exact operation identity preservation. | All other arithmetic/folds/AD/backend combinations. |
| `.venv/bin/python scripts/dtype_phase3_oracle.py` | The current flattened manifest of phase-0-through-3 nextest and non-test legs, with preflight and actual outcome evidence. | A universal theorem over all subsequently authored OP atoms or all hardware. |
| `.venv/bin/python scripts/runtime_representation_oracle.py --phase 0` | Frozen representation inventory/mutations plus checked metadata/movement/reduction, representation, capacity-key and planner evidence in its selected scope. | The entire runtime-representation migration, every GPU descriptor, or final ownership closure. Phase zero names a particular oracle scope. |
| `.venv/bin/python scripts/runtime_extent_oracle.py --phase a` or `--phase b` | Registered slice corpus, exact status classifications, source/target binding and execution receipts. Slice A allows explicitly tracked deferrals; B has its own targets. | Unregistered slice C/final, all modes executing rather than typed-unsupported, or hardware execution from test presence alone. |
| `.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 2` | Frozen fixture/polarity census, expected stage-specific outcomes, actual execution/ledger receipts and anti-false-green controls. | Final ownership completion: earlier phases may retain exact expected failures and omit HIP by their contract. |
| Same ownership runner with `--phase launch` | Only its frozen launch subset, with every selected fixture required to pass. | All ownership fixtures or HIP-inclusive completion. |
| Same ownership runner with `--phase complete --require-hip` | Every manifest fixture is `MustPass`, hardware rows included, with required identity/receipt/ledger controls. Phases 3 and 4 also require `--require-hip`. | Cases/configurations outside the manifest, an incorrect expected result, arbitrary unsafe foreign callers, or a stale/different-head run. |
| `cargo nextest run -p chelis-runtime --test checked_metadata` | Actual checked metadata positive/negative constructor and index cases. | Generated backend offset arithmetic or payload lifetime by itself. |
| `cargo nextest run -p chelis-runtime --features ownership-ledger --test tensor_write_guard` | Feature-enabled positive/negative guard state and provenance cases. | General absence of C UB, data races, leaks, or stale foreign pointers. |
| `.venv/bin/python scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1` | Actual selected HIP GPU executions when the required environment exists and the wrapper/test receipt confirms them. | Metal, a different device/compiler configuration, or stronger exactness than each comparator checks. |

The runner sources are
[dtype capacity](../../../scripts/dtype_capacity_oracle.py),
[builtin atom closure](../../../scripts/dtype_builtin_atom_closure_oracle.py),
[direct arithmetic](../../../scripts/dtype_direct_arithmetic_oracle.py),
[dtype phase 3](../../../scripts/dtype_phase3_oracle.py),
[runtime representation](../../../scripts/runtime_representation_oracle.py),
[runtime extents](../../../scripts/runtime_extent_oracle.py), and
[compiled ownership](../../../scripts/compiled_value_ownership_oracle.py).
They are materially stronger than a test-name list: expected selections, actual
framework callbacks/JUnit/libtest outcomes, immutable identities, and negative
controls make skipped/ignored/zero-match/self-certified green harder to report.
Their stated corpus still bounds what they prove.

Additional inspected source surfaces include
[observation roundtrips](../../../crates/chelis-cli/tests/observation_roundtrip_harness.rs),
[dtype operation matrices](../../../crates/chelis-e2e/tests/dtype_op_matrix.rs),
[runtime-window matrix](../../../crates/chelis-cli/tests/reduce_window_nonliteral_matrix.rs),
and [typed-kernel boundary checks](../../../crates/chelis-ir/tests/dtype_kernel_boundary.rs).
These have different evidence strength: executing linked binaries and comparing
exact values is not the same as asserting emitted text or searching source for a
typed dispatch function. Optional-tool skips need the enclosing oracle's actual
outcome checks to avoid apparent success without execution.

The inspected [CI workflow](../../../.github/workflows/ci.yml) wires dtype phase 3,
runtime representation phase 0, and ownership phase 2 plus the launch subset into
named jobs/aggregation. This records workflow intent, not current-head hosted
success; no live run was fetched here. A job called ownership is not automatically
the HIP-inclusive complete oracle, and documentation-only routing is not fresh
runtime validation.

## 8. Authority conflicts, future scope, and remaining closure

### 8.1 Conflicting active statements

Three concrete conflicts require correction at the controlling tier, not an
implementation guess:

- `[04-TGT-1]` and spec/04's accompanying `--target metal` rule require rejection
  when a program uses f64; `[05-OBS-10]` requires roots outside the selected target's
  Tensor capability to use Host without narrowing. The target-capability source
  excludes f64 from Metal's Tensor set and describes Host routing. Whether the
  prohibition is target-wide or only a kernel-lane boundary remains unresolved
  between these active rules. Host routing is not evidence that f64 Metal kernels
  execute, and this account claims no such execution.
- `[05-OP-64]` says integer zero divisors trap `Domain`, while `[04-NUM-9]` assigns
  integer divide-by-zero to `DivZero`; OP-64 also points back to NUM-9. An exact
  diagnostic oracle cannot satisfy both for the same operation/input.
- Spec/06 §3.5 applies the scalar-output restriction to `grad(vmap(f))` unless the
  batched output is explicitly reduced; its later composition discussion §6.2
  describes implicit summation. Those are different language meanings, not two
  implementations of one rule.

Older status/implementation descriptions in spec/08 and spec/12 cannot establish
that a lane currently conforms or license deviations from the explicit numeric
rules. Some primitive descriptions and registry counts also have historical
wording; exact incorporated identities and the explicit semantic definitions must
be reconciled rather than silently replaced by remembered phase claims. This
investigation does not edit normative authority.

### 8.2 The differentiable-language horizon

[Spec/12's committed differentiable-language scope](../../../spec/12-roadmap.md)
includes runtime control flow, user-defined data, effects, implicit constructs,
differentiability typing, and integration/on-ramp work. It records D1–D6 sequencing
and points to a design. This is a real committed project horizon, not a claim that
every design-only implicit `fix`/`argmin`/`solve` mechanism or proposed type-level
marker is already an active implemented language capability.

Conversely, already decided runtime control-flow/ADT/pathwise differentiation
rules in spec/06 do not disappear because a roadmap table still calls a delivery
slice planned. The account therefore distinguishes a current semantic obligation
with incomplete support from future design deliverables needing their own final
semantic decisions and implementation. Neither blanket “all differentiable
language is implemented” nor “everything in that chapter is merely future work”
is an adequate disposition.

### 8.3 What is established, and what is still unknown

The account covers every assigned numeric/shape/operation atom and the unnumbered
transformation, scheduling, backend, and foreign-lifecycle subjects. Incorporated
registry domains, negative cases, target/configuration distinctions and relevant
compositions are explicitly represented. That is **obligation-accounting
coverage**, not an all-kernel implementation audit or executed proof.

The strongest inspected structural boundaries are tagged scalar/storage kernels,
checked dynamic metadata, explicit extent dependencies, exact conservative
capacity equality, verified ownership payloads, private single-use reuse tokens,
runtime provenance/write-state checks, and receipt-backed censuses/oracles.
They already prohibit broad classes of undiscovered construction, metadata,
ownership and false-green failures within their actual reach.

The remaining completeness limits are specific:

1. Algorithm fidelity is not closed: reduction/random/wide-image paths visibly
   diverge from relevant contracts, and the other operation/stdlib algorithms
   have not been individually proved across their complete domains.
2. Route closure is not established across every raw IR API, decoder, graph
   mutation, host callback, nested aggregate, recursive AD path, emitter, and
   foreign protocol. Known strict/legacy and CPU/GPU representation splits make
   this an actual boundary question, not merely hypothetical caution.
3. Transform preservation needs semantic relations and complete effect/guard/
   trap/identity tracking, beyond valid output types and ownership verification.
4. Backend exactness remains configuration- and hardware-specific. A generic
   relative tolerance, source-only test or library specialization cannot prove
   the active exact/ULP/tree contract.
5. Unsafe foreign memory, external libraries, compilers and devices require their
   stated assumptions or independent evidence. No universal hostile-FFI safety,
   sandbox, termination, constant-time or global memory bound is invented here.
6. No execution freshness was established by this chapter. Located commands and
   inspected tests are not a passed acceptance oracle; the known active authority
   conflicts also prevent an unqualified universal conformance claim.

An undiscovered counterexample in any of these domains is already a violation of
its rule, not an operation waiting for an issue number to become specified.
Establishing that it cannot occur requires a correct enforcing algorithm, a closed
set of relevant transitions/entry paths, valid external assumptions, and evidence
whose actual execution covers the stated claim. Those are separate deliverables;
this account makes their present boundaries visible without declaring them done.
