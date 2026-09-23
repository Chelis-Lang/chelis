# Semantic core architecture

## Ownership of stages

```text
Expanded Deep / original-equivalent AST
    → PreparedProgram
    → type analysis
    → effects and linearity
    → CheckedCompilation / CheckedLibrary
    → production lowering and canonical roots
    → DAG or host/execution products
    → transforms, reverse AD, evaluation, ownership lowering and verification
```

`chelis-pipeline-core` owns the stage transitions. Source parsing/desugaring/expansion
policy and dynamic compiler-facade goals are not part of this workspace. A caller of
`PreparedProgram::from_expanded_deep` supplies an already expanded AST; a Deep parse by
itself does not promise expansion or semantic acceptance.

- `analyze_prepared` returns a rejected fitness report or one accepted analysis.
- `complete_checks` creates `CheckedCompilation` only after semantic checks accept.
- Contextual analysis retains the exact checked library against which it was performed.
  Composition cannot substitute a separate library or environment.
- `lower_library` binds a lowered payload to the checked library identity. Contextual
  lowering rejects another identity, including a same-shape foreign context.
- `NamedRoots` contains declared tensor outputs. `ForwardNodeIndex` additionally exposes
  internal Load aliases. These are different lookup contracts.
- Host-only and selected-host outputs retain their explicit root-binding policy; a
  failed collecting lowerer is not permission to silently select another execution path.

The Rust checked/verified carriers are runtime API seals backed by the implemented
checkers. Their names do not mean that a Lean theorem has been proved about those checkers.

## Production dependency boundary

The workspace contains only the eight crates listed in README. `chelis-ir` depends on
real type, numerical, effect, Deep and deterministic-collection implementations; they
cannot be replaced by a convenient small kernel without changing the verification target.
The dependency closure also retains host CFGs, symbolic/result-shape checks, additional
RISC operations, optimization and storage planning. Those are production code, not
automatically covered by the smaller Lean tensor calculus.

External crates and system services remain explicit dependencies: typed half-precision
arithmetic, big-integer capacity calculations, serialization, hashing and stack management.
Native Cargo and Charon intentionally use different pinned compiler toolchains.
Historical specification citations and generated registries inside the retained sources
refer to the upstream snapshot; the product specification/generator tree is not bundled.

## Reference correspondence map

The pinned reference is `LaCaDiLE/lean/LaCaDiLECore`. The following are obligations, not
completed proofs:

| Production implementation | Reference contract |
| --- | --- |
| `dag.rs`, strict entrypoints in `eval.rs`, numerical kernels in `chelis-types` | `Program`, `Tensor`, graph decoding/correspondence |
| `grad_dag_checked`, `compute_adjoints`, balanced contribution accumulation | `Pullback`, `CompileAdjoint`, `CompileReverse`, `Contributions`, `Accumulation` |
| `ownership::lower_dag_ownership` and independent `verify_ownership` | `Ownership`, `OwnedCode`, `OwnedMachine`, ownership normalization |
| `EvaluationPlan`, source execution spine, Random frames and replay | `Protocol`, `EffectProgram`, `EffectAD` |
| Actual target checking composed with preserved Resource requirements | Target-parametrized Resource protocol admission |

The seven reference tensor constructors are const, copy, add, mul, sum, insert and expand.
Rust represents both insert and expand with `RiscOp::Expand`; their rank/axis conditions
must distinguish the two interpretations. Loads correspond to entry variables rather
than constants. Rust append-index node IDs must be related to Lean context positions and
owners; raw integer equality is not that relation. Operand order and duplicate ports
must survive lowering, pruning and scheduling.

## Conditions that cannot be omitted

1. **Shape and representation.** Successful admission, backward references, valid roots,
   complete inputs, resolved runtime dimensions, matching flat row-major storage, and
   nonoverflowing shape products/indices are required. Rust rank-zero broadcasting is
   broader than the equal-shape reference add/mul operations.
2. **Floating point.** Finite IEEE values alone do not satisfy the reference semiring or
   real-number laws. Rounding, signed zero, overflow, NaN, casts and typed accumulation
   need an IEEE/error model or a proved exact-operation domain. The ordinary evaluator
   cannot simply be assigned real arithmetic semantics.
3. **AD output and generated operations.** Rust omits disconnected requested gradients;
   the reference returns total entry cotangents with zero filling. Rust also permits
   differentiation with respect to internal nodes. A sparse/total interface relation or
   a stated representability condition is necessary. A same-rank Expand adjoint emits
   Reshape after Sum; relating that reshape to singleton insertion is another lemma.
   Mixed accumulator precisions can introduce Cast.
4. **Ownership.** Borrowed roots are cloned before transfer. Repeated owned roots are
   rejected by the Rust lowering path, while reference normalization copies earlier
   occurrences and moves the last. Logical ownership does not establish physical buffer
   reuse or allocator safety.
5. **Random.** Rust uses wrapping `u64` draw ordinals versus unbounded reference naturals.
   Direct state equality needs a no-overflow condition or a modular simulation. Nested
   controls, inherited keys and replay without another draw require state relations.
   For non-F64 dropout, Rust rounds the sampled unit value before thresholding; rate
   conversion, denominator rounding and division also differ from exact rational
   thresholding and inverse multiplication. Matching masks/scales cannot be assumed.
6. **Resource.** Rust's C target check accepts exactly `"cpu"`; the reference C predicate
   accepts any device string not starting with `"gpu"`. Unknown Rust target strings also
   have no direct reference constructor. Device/target domains or a one-way refinement
   must be explicit. Trace presence alone is not runtime protocol acceptance.
7. **Runtime and tooling.** Thread-local state, atomics, dynamic downcasts, panic recovery,
   stack growth and foreign/runtime dependencies are real reachable code. Standard
   Aeneas translation does not evaluate Rust drops by default; Rust-side destructor and
   guard effects need explicit treatment rather than being conflated with Chelis's
   logical ownership instructions. No unsupported production path may be silently
   removed to manufacture a proof.

## Precise bridge contract

The initial theorem target is **successful checked graphs in the seven-operation
fragment**, not arbitrary Deep programs or all retained host operations. Extraction of
the entire production closure and correctness of this fragment are distinct obligations.
No theorem about the parser, type inference, lowering, or unsupported operations follows
merely from validating the resulting tensor graph.

Fix an ordered entry context `Γ`, a scalar loss, resolved shapes, and a homogeneous
floating dtype whose reduction accumulator is that same dtype. Require strict input
admission, valid backward references, and nonoverflowing dimensions and indices.
Let `decode` read the actual graph's initializers, ports and ordered roots; it must not
replace them with the reference compiler's output. Its coefficient domain is `ℚ`, using
the exact binary rational value of each finite Rust constant, not its printed decimal.

The bridge must establish these relations:

1. **Bindings and values.** Each admitted Rust value maps to a shape-indexed Lean
   `Var`. Loads map to entries; every other retained initializer extends the context.
   At any context, `owner + offset + 1 = context.length`; Rust `NodeId` is neither
   automatically the Lean owner nor its reverse-context offset. `PartialRenaming.Related`
   requires equality at every mapped value. Flat storage maps to tensor coordinates
   in row-major order. Ports retain order and multiplicity, including `mul(x, x)`.
   Removed bindings may be absent only when no retained port or root needs them.
2. **Operations.** Equal-shape add/mul are admitted without implicit broadcasting.
   Rank-increasing Expand maps to insert; same-rank Expand requires source extent one
   and unchanged other dimensions. Generated singleton-axis Reshape requires a proved
   value-preserving translation to insert; it is not silently erased. Cast and other
   additional operations require their own semantics or an explicitly narrower theorem.
3. **Numerical observation.** `realize` maps finite tensor elements to their exact real
   values, identifying the two signed zeros. For equality with real evaluation, every
   executed primal and adjoint arithmetic/reduction step must be exact and nontrapping.
   This is a per-execution obligation, not a consequence of finiteness. Under it,
   `realize(eval_Rust G inputs) = (decode G).eval (realize inputs)`.
   This observation theorem is not bitwise equivalence or differentiability of the
   discrete IEEE implementation. The derivative is that of the reference real function.
4. **Gradient interface.** For the reference all-entry gradient, request the ordered
   entry nodes, not arbitrary internal nodes. Let `D` be the entries reachable from the
   loss through differentiable value ports; shape witnesses are not such ports.
   Prove coverage of every requested member of `D`. Define `totalize` explicitly:
   use the returned gradient for a present entry, and a correctly shaped zero only for
   a proved disconnected entry. A missing reachable gradient is an error, never a
   default. The desired observation is `realize(totalize(grad_Rust G)) =
   (decode G).compileGrad.eval (realize inputs)`, with the separately returned primal
   related to the source loss. This does not change Rust's sparse-map API.
5. **Ownership.** Decode the actual final plan, expanding RootClone into explicit copy
   and yield, with borrowed entries and fresh local owners. Require no repeated owned
   root transfers unless a separately proved normalization is applied. The independent
   ownership checker must accept the original plan; value-only copy elimination cannot
   excuse invalid ownership. Physical allocation, buffer reuse and cross-plan capacity
   identities are separate claims.

For a proof-producing acceptance path, supply independently decoded `source`,
`postAD` and `final` to `GraphCorrespondence.checkGradient`. Its concrete types are
`Program ℚ Γ []`, `Circuit ℚ Γ Γ`, and `OwnedCode ℚ Γ Γ`. Both witnesses name actual
final-context bindings; they cannot assert an arithmetic law or invent a missing port.
An accepted check provides the reference real-derivative and logical-ownership theorem.
Connecting that acceptance to the extracted Rust return value, the sparse-interface
relation, generated Reshape, and exact numerical execution remain proof obligations.
Compilation of a generated Lean file alone establishes none of those relations.

Effectful extensions additionally relate the actual validated source spine, target
checks, frame stack, cached keys and replay transitions. Require fixed control/rates/
shapes with respect to differentiation, no `u64` counter overflow for the direct Nat
relation, and coordinate-wise agreement of dropout threshold and scaling operations.
For C, production acceptance implies the weaker reference device predicate; equivalence
requires restricting device names. The pure `checkGradient` theorem does not discharge
these effect obligations.

## Runtime state and unwind boundary

Full-core extraction must model a thread-indexed TLS store, shared cancellation flags,
process-global state, and exit outcomes separately from graph values. At minimum:

- `CancelTokenGuard` restores the previous per-thread token on normal return and
  unwinding. Linked-program and opacity guards likewise carry scoped state. A TLS read
  cannot be replaced by a fresh unrelated global value.
- Cancellation uses shared `Arc<AtomicBool>` identity and relaxed atomic accesses;
  a concurrent model must account for permitted observations, not assume an immediately
  synchronized ordinary Boolean.
- Lowering's suppression guards clear their flags to false; they do not generally
  restore an outer Boolean. `raise_fatal_lowering_diagnostic` explicitly re-enables
  suppression after an inner catch. A uniform restore-previous model would be wrong.
- `catch_lowering` catches unwind payloads, including typed diagnostics and
  `UnrepresentableDag`, after guard destruction and stack growth. Successful values,
  returned diagnostics, unwinding and non-unwinding aborts are distinct outcomes.
  Panic-hook state and any claimed diagnostic-output behavior also need a relation.

Charon currently rejects `mir::Rvalue::ThreadLocalRef`. Aeneas's default
`drop_as_no_op` and symbolic panic handling do not supply TLS restoration,
`catch_unwind`, panic payload downcasting, or `stacker` semantics. Passing opaque TLS
values to otherwise translated code would not solve this boundary. An explicit-state
implementation would also need a proved wrapper relation preserving reentrancy,
thread isolation and unwind cleanup; excluding those wrappers is not full-core extraction.

The native correspondence smoke observed `(1e16 + -1e16) + 1 = 1` versus
`1e16 + (-1e16 + 1) = 0` in the strict f64 evaluator, a sparse missing disconnected
gradient alongside `d(x*x)/dx = 4` at `x = 2`, and cancellation-token restoration after
a caught unwind with isolation on a second thread. These are concrete boundary checks,
not universal proofs.

## Extraction status

`verification/extract.py` is a reproducible model-generation attempt, not a proof checker.
It uses the standard Aeneas preset with borrow checking enabled and no exclusion/opacity
rules for the retained crates. The local Charon patches preserve branching Box
initialization, closure trait clauses and nonlocal continuations. The Aeneas patches
make unknown externals fail in strict-warning mode and normalize function-returning
loop continuations without joining uninitialized return slots. Independent branches
remain joined. Exact revisions and patches are recorded in `verification/toolchain.json`.

The diagnostic AD slice produces LLBC and completes its loop prepasses. An equivalent
ordered search in `shape_source_for_axis` removes the recursive closure-dictionary
cycle. Extraction now stops at the opaque borrowed type `core::slice::iter::Windows`.
The actual AD closure also contains unmodeled `Chunks`, B-tree iterators, and mutable
B-tree entry types; isolated extractions reproduce these failures. Their ownership
structure and operations need explicit library models, not relaxed region checks.
The full core still fails at Charon's TLS boundary.

Generated extractor regression models compile and execute in Lean, including returned
mutable borrows and independent branches before returning loops. No production Lean
model, bridge theorem or acceptance result is claimed.

Backend library assumptions require their own audit. In the pinned Aeneas
`Aeneas/Std/Core/Fmt.lean`, `core.fmt.Formatter` is an axiom and formatting arguments
are modeled as `Unit`. Strict unknown-external diagnostics do not detect such known
builtin abstractions. Lean compilation alone therefore cannot establish faithful
diagnostics or axiom-free correspondence; acceptance must audit the used assumptions
against the stated observation relation.

The native integration smoke exercised real preparation, type/effect/linearity checking,
lowering, strict evaluation, reverse AD and independent ownership verification on a
squared-norm program: input `[2, 3]`, primal `13`, gradient `[4, 6]`. That execution checks
the stripped library integration; it is not evidence of universal correspondence.
