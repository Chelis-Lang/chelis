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
`PreparedProgram::from_expanded_deep` supplies an already expanded AST and explicit
`ProgramOrigin`. A Deep parse by itself promises neither expansion nor link provenance
nor semantic acceptance.

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
  unwinding. Checker opacity and recursive scopes restore state owned by their
  session. Link provenance instead arrives as immutable input, with no ambient
  state to restore. A TLS read cannot be replaced by a fresh unrelated global value.
- Cancellation uses shared `Arc<AtomicBool>` identity and relaxed atomic accesses;
  a concurrent model must account for permitted observations, not assume an immediately
  synchronized ordinary Boolean.
- `LoweringRuntime` owns the native thread's suppression flags. Its guards clear
  them to false; they do not generally restore an outer Boolean.
  `raise_fatal_lowering_diagnostic` explicitly re-enables output suppression after
  an inner catch. A uniform restore-previous model would be wrong.
- `LoweringRuntime::catch_lowering` catches unwind payloads after the closure's
  finalizers run on the grown stack. Its output-suppression guard remains live
  through payload conversion and destruction, then clears before returning.
  Successful values, returned diagnostics, unwinding and non-unwinding aborts
  are distinct outcomes. Panic-hook observations also need a relation.

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

### Explicit-state cutover

Host lowering now owns its invocation state in `HostLoweringSession`, separate from
the program-bound fact caches. The existing session reference carried by `LowerCtx`
also carries specialization state through staged DAG lowering. Checked-program
identity is unchanged; the native service is now a separate borrowed input.

`HostInvocationGuard` starts each whole-program call with empty state and restores
its caller on return or unwind. Inlining and pointer-keyed preflight facts have
session-bound scope guards. Type-substitution guards move the previous map into the
guard and restore it on exit: nested bindings replace rather than extend the outer
map, and type decoding no longer clones that map. `MonoProbeGuard` still snapshots
and restores the complete specialization state, including emission order and
in-progress recursive entries.

Native regressions cover nested substitutions, duplicate inlining scopes, preflight
pointer isolation between programs, fatal-probe retries, and retained recursive
calls and definition order after speculative or failed lowering. This is a production
Rust ownership cutover, not host-lowering Lean extraction or a wrapper correspondence
proof. It does not establish extracted semantics for `RefCell` accesses or guard
destruction. Panic/cancellation services, global inference IDs, persistent thread
counters, stack growth, and the native catch boundary remain unchanged.

Checker opacity now lives on the authoritative `DiagnosticSink`, not in
`OPACITY_CONTEXT` TLS. `OpacityScope` borrows that same sink and restores the previous
policy on return or unwind without replacing its diagnostics or witness table.
Module/declaration updates and deferred opacity checks use the explicit carrier in
both inference drivers and recursive sweeps. Annotation-only sessions start with no
policy; another active checker session cannot enable or disable their enforcement.
The persisted `TypeEnv` opacity metadata and its serialization layout are unchanged.

Native scope regressions exercise two live sessions on one thread, nested return
and unwind, and retention of authoritative violations. The native API smoke covers
plain/IR diagnostic parity, serialized and layered library contexts, exported
producer access, cancellation followed by reuse, and four concurrent checks.
This removes one checker TLS family; it does not establish extracted checker
semantics or a relation for scope destruction and the native runtime shell.

Recursive inference now owns `RecursiveState` on the same `DiagnosticSink`, replacing
`GROUP_CTX` TLS. Each whole-program inference driver enters a fresh frame before
collecting declarations. `RecursiveInferenceScope` moves aside the enclosing frame
and restores it on return or unwind, without discarding authoritative diagnostics.
`RecursiveCallerScope` restores caller attribution within a group while retaining
the group's recorded occurrences. An aborted invocation drops its unfinished
evidence; completing a group takes its evidence and releases pins before validation.

`Env::generalize`, collection-contract generalization, and rejected-signature recovery
now require an explicit borrowed pin set. Both level-based generalization and its
sweep oracle consume that input; neither reads ambient recursive state. Standalone
callers supply an empty set. The frame is plain owned data, not another `RefCell` or
shared-state handle, and is not part of persisted `TypeEnv` metadata.

Native regressions cover two live sessions, nested checks through both drivers,
return/unwind restoration, caller attribution and diagnostic order, and mid-SCC
cancellation followed by independent instantiation. The public-API smoke checks seven
recursive fixtures through both drivers, ordered diagnostic parity, serialized and
layered generic libraries used at `i32` and `bool`, cancellation/reuse, and four
concurrent invocations.

Link provenance is now an immutable `ProgramOrigin::{Unlinked, Linked}` input,
replacing `LINKED_PROGRAM` TLS and its install guard. Every public type-inference
and fitness entry requires the origin of its current unit. `PreparedProgram` pairs
that origin with its owned expanded AST; monolithic, library, contextual, and layered
semantic transitions pass it to the same production checker.

The two consumers take the input directly: reserved-name rejection and the narrowly
scoped orphan-signature exemption. `Linked` is a trusted native caller assertion
after source validation and linker rewriting, not a checker-produced link proof.
An internal-looking name, source metadata, or previously checked library cannot
grant that authority to new code. Ordinary orphan names still reject for linked
input. Origin has no default or serialization; neither `DiagnosticSink` nor
persisted `TypeEnv` stores it, and the checked-program/library seals are unchanged.

Regressions cover all four inference ingresses, structured ordered rejections,
linked opacity enforcement, forged metadata, and a serialized linked context used
to check a new unlinked unit. Pipeline regressions cover each semantic transition.
The native pipeline smoke takes linked and unlinked inputs through checking,
lowering, and evaluation; observes `7` and a contextual `7 * 3 = 21`; and exercises
layered composition, cancellation/reuse, and four concurrent invocations. These
checks exercise the retained core, not the removed native linker implementation.

Lowering now receives an explicit `&LoweringRuntime` from
`chelis_ir::lowering_runtime::with_native_lowering_runtime`. The native adapter
borrows one persistent per-thread service; nested acquisition neither resets nor
copies it. `HostLoweringSession::new(program, runtime)` and `LowerCtx` carry that
reference through DAG, host, staged, evaluation, and trace paths. Sessions holding
the service are thread-bound. Checked programs, lowered libraries, and immutable
subexpression contexts contain no runtime handle and keep their existing layouts.

Canonical lowering APIs require the runtime as their final argument. Semantic
completion and contextual composition also need it while computing DAG root
classifications: those predicates retain suppression-dependent malformed-binder
rejections. Pure type analysis, cached-library validation, and immutable context
construction remain runtime-free. No second classifier or error policy was added.

Only the native adapter and process-global panic hook select thread-local
suppression state. The hook observes it before unwinding. `catch_lowering`
retains the 512 MiB stack boundary and existing payload conversion; fatal
re-raise, speculative markers, and clear-to-false scope destruction are unchanged.
This isolates a native dependency rather than replacing control panics with
`Result` or proving the native shell pure.

The process-isolated native regression observes returning and unwinding nested
catches, speculative payload selection, fatal re-raise, hook/finalizer order,
payload-destruction coverage, nested acquisition, and cross-thread isolation.
The production API smoke exercises checking, strict and sealed contextual lowering,
evaluation of `7 * 3 = 21`, live-input fatal AD rejection through DAG and host paths,
reuse, and four concurrent threads.

Ownership lowering takes an explicit `&mut CapacityScopeSupply` as the final argument
of `lower_dag_ownership`, `lower_host_ownership`, `lower_host_execution_ownership`,
and `EvaluationPlan::verify_ownership`. Its only primitive, `fresh_id(&mut self) -> u64`,
performs the process-global relaxed checked increment; ownership lowering wraps the
returned id in a `CapacityScope`. `CapacityScopeSupply::native()` is shell-only and
is never called from semantic code. Every native supply draws from that one
sequence, so scopes never reset per invocation and keep their allocation order: one
per successfully lowered DAG plan, global helpers before function helpers. Payload
specialization is selected through a sealed borrowed enum instead of `dyn Any`
downcasts, and terminator successors use a concrete iterator in the same order.

These are steps toward FCIS, not evidence of a pure functional core. Explicit ownership
removes hidden dependencies in these subsystems; it does not remove cancellation,
global ID allocation, counters, stack growth, or panic effects from the remaining
native boundary. Local mutation of invocation-owned data is not that boundary.

Keep one production implementation: no `_for_lean` checker, global-to-constant
substitution, or opaque successful fallback.

The extraction boundary should separate an explicit invocation state from a native
runtime shell. Reuse `HostLoweringSession`, `LowerCtx`, and the authoritative
`DiagnosticSink` rather than introducing a second cache or diagnostic authority.
Checked-program/library provenance, local tensor obligations, and root manifests
remain inputs with their existing seals; an invocation state must not mint them.

| State family | Owner or remaining target | Required behavior |
| --- | --- | --- |
| Former `OPACITY_CONTEXT` | `DiagnosticSink::opacity` and `OpacityScope` (implemented) | Restore the enclosing policy on return/unwind; keep diagnostics and witnesses in the same sink. Independent sessions never inherit another session's policy. |
| Former `GROUP_CTX` | `DiagnosticSink::recursive`, `RecursiveInferenceScope`, and `RecursiveCallerScope` (implemented) | Generalization borrows explicit pins. Restore enclosing invocation/caller state on return or unwind; retain completed diagnostics and recorded caller evidence. Abort discards unfinished group evidence and releases pins. |
| Former `LINKED_PROGRAM` | Immutable `ProgramOrigin` input, retained by `PreparedProgram` (implemented) | Only the trusted native linker asserts `Linked`. Each unit supplies its own origin; names, metadata, cached contexts, and other active invocations grant no authority. No mutable scope or restoration guard remains. |
| Former `INLINING_STACK`, tensor-helper preflight stack, `ACTIVE_TYPE_SUBST` | `HostLoweringSession::invocation` and scoped guards (implemented) | Preserve scope order and the current specialization's bindings. Nested sessions do not share mutable frames. |
| Former `MONO_SPECIALIZATIONS` and `MonoProbeGuard` | Host invocation state plus scoped probe snapshot (implemented) | Preserve entry initialization, completion order, and rollback of speculative definitions and in-progress entries. Rollback restores the outer snapshot; it does not clear it. |
| Former lowering-module panic-suppression flags | Explicit borrowed `LoweringRuntime`; per-thread storage and hook bridge confined to `lowering_runtime` (implemented) | Preserve shared thread state, clear-to-false behavior, and explicit fatal re-enable. Native hook/unwind semantics still require a wrapper relation. |
| Cancellation token identity and relaxed loads | Native runtime services | Keep shared token identity and the original polling points. A frozen entry Boolean is not equivalent to cancellation arriving during a check. |
| Atomic host-inference IDs, persistent thread counters, panic hook | Native runtime services | Preserve allocation/observation order and lifetime across invocations, including native ID wrapping or an explicit no-wrap bound. Resetting a global ID supply per run can invalidate identity comparisons. |
| Stack growth, depth guards, first stack-bail record | Native resource boundary | Preserve the first reported site/span and normal-versus-abort behavior. Arbitrary Lean fuel is not a replacement for native stack availability. |

Remaining runtime migration and verification:

1. Apply the same ownership cutover to the remaining stateful subsystems, retaining
   the public native shell and removing each subsystem's TLS reads. Mutable state
   belongs to the invocation; immutable checked-program facts can remain borrowed.
   Do not clone the entire program or runtime state at every recursive call.
2. Make scope entry and exit explicit, with separate restore, clear, and rollback
   operations matching the table. First preserve the existing panic boundary.
   Converting a control panic into `Result` is a separate behavioral change:
   hooks and destructors can distinguish normal return from unwinding.
3. Relate native runtime events to the stateful core: cancellation loads at the same
   sites, ID-allocation events, stack checks, diagnostics, and exits. Concurrent atomic
   observations must satisfy the native memory model, not an arbitrary Boolean
   stream or an assumed immediately synchronized flag.
4. Keep successful return, semantic rejection, unsupported DAG lowering, unwinding,
   and non-unwinding abort distinct. The panic hook observes state before stack
   unwinding; finalizers then run in order; `catch_lowering` maps the payload only
   after unwinding. Preserve `panic_payload_to_lower_diagnostic` at the wrapper
   boundary. Unknown panics must not become a successful result or host fallback.
5. Extract and relate the stateful inner fragment. A separate wrapper relation is
   still required for the public entry point; retaining a native shell is not
   full-core extraction.

Acceptance for the complete runtime cutover additionally requires identical ordered
roots, values and diagnostics on success/rejection; explicit per-unit link authority;
nested cancellation/opacity/recursive restoration; suppression clear/re-enable behavior
across nested catches; speculative monomorphization rollback
without leaked definitions; cancellation during a run and isolation across threads;
unchanged panic-hook/finalizer order and first stack-bail receipt. Existing native
checks are the baseline, not permission to erase runtime observations.

## Extraction status

`verification/extract.py` is a reproducible model-generation attempt, not a proof checker.
It uses the standard Aeneas preset with borrow checking enabled and no exclusion/opacity
rules for the retained crates. The local Charon patches preserve branching Box
initialization, closure trait clauses and nonlocal continuations. The Aeneas patches
make unknown externals fail in strict-warning mode and normalize function-returning
loop continuations without joining uninitialized return slots. Independent branches
remain joined. Exact revisions and patches are recorded in `verification/toolchain.json`.

`verification/check_production_ad.py` is the separate production AD acceptance gate.
It builds the supplied pinned patched Lean library, invokes the same extraction
implementation for the fixed `grad_dag_checked` root, and compiles the fresh,
unmodified generated module. Build/translation/compilation failures are fatal;
old generated artifacts are never reused. Dependency `sorry` and linter warnings
remain visible. This gate checks extraction and compilation, not correspondence
proofs or the absence of backend axioms.

The AD closure now has explicit models for slice windows/chunks, ordered B-tree
iteration/entries, recursive trait dictionaries, formatting and floating operations.
The ordered `shape_source_for_axis` search avoids the former recursive
closure-dictionary cycle. Collection admission is narrower than Rust's generic API:
unsupported hints and consumer specializations are rejected before model generation.

The packaged regression matrix generates, compiles and executes 25 native/Lean
models, checks 16 deliberately rejected collector roots, and compares 1,099 floating
observations. Production AD generation and Lean acceptance are recorded in
[README.md](README.md#verification-status-and-remaining-boundaries). Neither those
checks nor compilation supply a Rust/Lean/LaCaDiLE correspondence theorem.

The full core remains blocked before LLBC generation. After the runtime-input
cutover, the first observed failure is Charon/Hax's external closure-region handling
for `chelis_ir::lower::try_lower_program_to_library::{closure#0}`: rustc exports no
closure region requirements for zero erased signature regions and two erased
capture regions. The former first failure was the lowering module's suppression
TLS. Native TLS still exists inside `lowering_runtime`; its hook, unwind, and stack
semantics remain separate obligations even after the closure-region issue is fixed.
An accepted AD slice is not whole-core extraction.

The patched formatter has explicit arguments and callback state, including partial
output on writer failure; it no longer uses the upstream erased formatter model.
Backend assumptions still need a separate audit. Allocation/capacity, destructor and
pointer observations, callback scheduling restrictions, runtime resources and
numerical relations cannot be discharged by strict unknown-external checks or Lean
compilation alone.

The native integration smoke exercised real preparation, type/effect/linearity checking,
lowering, strict evaluation, reverse AD and independent ownership verification on a
squared-norm program: input `[2, 3]`, primal `13`, gradient `[4, 6]`. That execution checks
the stripped library integration; it is not evidence of universal correspondence.

## Production AD patch audit and source proposal

The audit target is the real `chelis_ir::grad::grad_dag_checked` dependency closure,
not a rewritten AD example. The inventory is fourteen packaged patches: six Charon,
one Charon-ML, six Aeneas compiler, and one Lean stdlib patch. Evidence combines the
fresh production LLBC/generated Lean with executable model regressions. No
leave-one-patch-out ablation was performed: reachability of a feature does not prove
that every hunk supporting it is indispensable.

| Patch group | Production evidence and decision |
| --- | --- |
| Charon box initialization | Boxed dimension expressions, cloning and payload formatting remain in AD. Keep initializer reconstruction; removing a few collectors does not remove these representations. Necessity of every hunk for this root is unmeasured. |
| Charon closure clauses, local trait bounds, closure regions, hidden traits; Charon-ML binders | AD still uses concrete iterator closures, associated types, borrowed state and nested trait dictionaries. Keep the metadata/identity fixes rather than replace ordinary Rust traits with a parallel implementation. |
| Charon nonlocal continuations; Aeneas loop continuations | Production verification, graph traversal and remapping contain loops and early errors. Preserve their continuation and cleanup semantics; the model regressions exercise mixed exits and returned borrows. |
| Aeneas trait models and shared borrows | DAG access, mutable collection entries, borrowed iteration and recursively formatted/compared payloads remain reachable. Source cleanup does not justify weakening projection or region checks. |
| Aeneas floating scalars | AD has bit-exact floating constants and floating operations. Retain the numerical models and their explicit float relation; native/Lean samples are not a universal numerical proof. |
| Lean stdlib | Actual AD uses windows/chunks, vector operations, ordered tree iteration/entries and extensive formatting. Model these operations rather than erase diagnostics or mark their implementations opaque. |
| Strict warnings and collection contract | These are acceptance guards, not inconveniences to refactor away. Unknown externals and unsupported collector dictionaries must prevent acceptance. |

Keeping these patch families is a conservative dependency decision, not a claim that
the patch set is minimal. In particular, the old generated AD model referenced
`FromIteratorBTreeMap` through the generic `UnordMap` collector; this was not evidence
that arbitrary iterator callbacks matched native collection.
The audited production LLBC after the cutover has no `chelis_unord` `from_iter`
method. The unrelated
`dag::map_node_symbolic_bindings` is not reachable from this AD root and was left
unchanged. These are reachability observations, not executed-call counts.

The fresh AD LLBC contains 1,387 function declarations. Its generated Lean has 283
distinct explicit `Aeneas.Std` references, including 48 `FromIteratorVec` occurrences,
one `FromIteratorBTreeSet`, no `FromIteratorBTreeMap`, 285 `alloc.fmt.format`
occurrences and 26 `F64.ofBits` occurrences. Windows/chunks and B-tree
iteration/entry operations remain. These textual counts explain why the collector
cutover does not eliminate the stdlib, formatting or numerical model dependencies.

### Applied concrete source cutover

The first strict collector attempts rejected actual production Chain consumers,
inclusive-range extension, and the verifier's generic B-tree collector. The following
changes address those source boundaries without changing the public generic
`UnordMap::FromIterator` implementation:

| Production location | Change and preserved observation |
| --- | --- |
| `grad.rs` | Five concrete map-building sites use ordered insertion: topology positions, requested gradients, execution sites, pruned gradient nodes and remapping. Small private helpers isolate the first three loops from the traversal's large borrowed state. Keys are `NodeId`/`usize`, so equal keys have identical representations; source order and last-value behavior are retained. |
| `evaluation.rs`, `ExecutionMetadata::remap` | Build a temporary map and commit only after successful remapping. Keep the first missing-node diagnostic and the original ordering relative to spine remapping. |
| `axis_sources.rs`, `output_axis_sources` | Replace three Chain collectors with preallocated vectors and supported extension/push operations. Keep OneHot, batched matmul and Gather axis provenance, order and malformed-rank fallback. |
| `verify.rs` | Build expected matmul dimensions without Chain, reserving the full size. The four bound-validation callers pass concrete owned input-slot vectors to `check_exact_bound_inputs`; paired bounds use a concrete traversal helper. Per-node checks are separate from graph traversal, preserving node/diagnostic order and the local-ascription early exit. |
| `execution_spine.rs`, `Spine::complete` | Replace inclusive ranges with equivalent half-open ranges. The existing `node.0 < dag.len()` guard makes the endpoint increment safe; scheduling order and control/drop weaving remain unchanged. |

The intermediate generic verifier loop failed extraction with an unbound loop type
parameter. Its unnecessary generic boundary was removed, not hidden behind an opaque
model. The resulting concrete helpers extract and compile in Lean; an executable
smoke preserves mixed bound variants, repeated slot indices, input order and empty
input. The AD helper split keeps independent construction loops out of the
large traversal context. The verifier likewise calls a private per-node function
instead of carrying all operation-specific branches through the graph loop's symbolic
join. Neither change disables strict joins or invariant checks.

Native regressions support value/diagnostic preservation under the stated runtime
relation; this is not an allocator-trace equivalence or a universal proof.
Ordered insertion can cost more than bulk tree
construction. A generic collect-to-insert rewrite is not justified: arbitrary equal
keys can have different representations, and custom iterator hints, comparator
callbacks, drops and native specialization can be observable.

### Further source simplification

Keep idiomatic supported iterators and the existing ordered `shape_source_for_axis`
search. Do not replace every adapter with a loop. The AD model's large formatting
closure suggests a possible structured-diagnostic core with a separate renderer,
but that would change the extraction boundary: retaining native diagnostics requires
an explicit renderer relation, including partial writes. It is a proposal, not an
implemented shortcut or a reason to drop formatting from the accepted root.

Runtime-state extraction is the separate cutover described above. Compiler metadata
repairs, strict admission, numerical semantics and ownership checks remain required
obligations regardless of how small the extracted source becomes.
