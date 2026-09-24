# Chelis semantic core

This workspace extracts the existing production semantic core from
[Chelis e0250b290b0fb297220378aeb7f1d0dce64aa5d7](https://github.com/Chelis-Lang/chelis/tree/e0250b290b0fb297220378aeb7f1d0dce64aa5d7).
It is a library workspace, not the Chelis product distribution and not a rewritten miniature interpreter.

## Retained code

`chelis-pipeline-core` is the canonical semantic owner. Its complete local production dependency closure is retained:

| Crate | Responsibility |
| --- | --- |
| `chelis-pipeline-core` | Prepared inputs, checked products, library identity, semantic stages, canonical roots and lowering |
| `chelis-ir` | Production DAG/host IR, transforms, reverse AD, evaluation, Random execution and ownership verification |
| `chelis-types` | Type/shape inference, linearity, manifests, primitive and numerical semantics |
| `chelis-effects` | Effect and target-admissibility checks |
| `chelis-deep` | Canonical Deep syntax, AST, metadata, validation and printing |
| `chelis-pred` | Predicate representation and checking |
| `chelis-vocab` | Closed primitive, dtype and effect vocabularies |
| `chelis-unord` | Deterministic collection contracts |

Frontend preparation, macro expansion, the compiler facade, CLI/LSP, target emitters,
foreign-language bindings, runtime packaging, proof engines, packages, examples and
product tooling have been removed. No frontend crate is required by the retained tests.
Retired source migration commands are also removed; the core continues to reject retired primitive names.

Tests use captured Deep or lossless JSON AST fixtures from the original parser/expander.
Legacy carriers, spans and signature metadata are retained where observable; canonical
Deep printing alone is not an exact AST round trip. Dynamic matrices use actual AST
builders or fixture templates. Source-scanning and wording-only tests were removed rather
than made into empty passing tests.

## Build and check

Run from this workspace. Nix provides the pinned native compiler, formatter, linter,
Charon, Aeneas, Elan and Python; an ambient Rust installation is unnecessary.

```sh
nix develop --command cargo test --workspace --locked --all-features --jobs 2
nix develop --command cargo clippy --workspace --locked --all-targets --all-features --jobs 2 -- -D warnings
nix develop --command cargo fmt --all --check
```

`cargo test --workspace` exercises the default production configuration. The trace feature
adds the retained lowering/execution observation tests. The optional
`chelis-types/generalize-sweep-oracle` feature additionally compares inference against a
reference environment sweep; it is not enabled in production by default. The sweep now
retains active declaration binders as monomorphic roots, including their substituted
variables. The original scalar-surface and explicit-binder rejection tests, the new
substituted-binder regression, and the full all-feature workspace suite pass without
weakening the production checker or the oracle assertions.

Context-checked extensions require their bound library or explicit composition before
isolated lowering. DAG, evaluation, library, C, and trace entry points reject detached
extensions; cloning or inspecting a lowered product does not remove that provenance.
Contextual lowering invalidates shadowed declaration/signature classifications and
dependent library callables while retaining already-materialized library values.
Lowered-library caches preserve the original signatures and dtype binder metadata;
older serialized caches missing that metadata must be regenerated.

See [ARCHITECTURE.md](ARCHITECTURE.md) for API boundaries and the formal correspondence obligations.

## Rust to Lean extraction

[verification/toolchain.json](verification/toolchain.json) records the source and reference
revisions. [flake.lock](flake.lock) pins the tool builds:

- Native Rust: `1.98.0`.
- Aeneas: `ee9bb3935375aacfa8ea3682fc21b051b79fc931`.
- Matching Charon: `ed22146b1cd4d0b578006a58b3299d41ecbe0fd4`, with its own `nightly-2026-02-07` compiler.
- Lean: `leanprover/lean4:v4.29.0`.
- [LaCaDiLE reference](https://github.com/Chelis-Lang/LaCaDiLE/tree/11531c8b0e3e6fb6650ddbc404937e5c527b0785/lean): `11531c8b0e3e6fb6650ddbc404937e5c527b0785`.

The local Charon patches are applied by the flake, with the matching Aeneas package
built against that Charon:

- `verification/charon-box-init.patch` reconstructs acyclic branching Box initialization
  at the original pointee assignment, preserving initializer evaluation order and
  user cleanup. It relies on the existing infallible builtin-Box allocation abstraction;
  allocator observations and allocation-failure timing are not covered.
- `verification/charon-closure-clauses.patch` preserves generic clauses on closure
  types and their instantiations. Erasing them left Fn implementation metadata with
  unbound trait references.
- `verification/charon-nonlocal-continuations.patch` preserves jumps to outer
  continuations instead of incorrectly treating them as local fallthrough. This
  prevents a loop's early error return from being overwritten by a later success.
- The local-trait, closure-region, hidden-trait, and Charon-ML binder patches retain
  runtime dictionaries, external closure metadata, late-bound regions, and
  specialized implementation identities. Compiler-only destruction constraints
  do not replace explicit runtime witnesses.

The local Aeneas patches are also applied by the flake:

- `verification/aeneas-strict-warnings.patch` makes unknown external definitions fail
  with `-warnings-as-errors`, rather than logging a warning and returning success.
- `verification/aeneas-loop-continuations.patch` normalizes function-returning loop
  exits without joining uninitialized return slots or scoped locals. Normal
  continuations run only on their original exits; independent conditionals stay
  joined instead of expanding exponentially. Continuations that resume an outer
  iteration and nonlocal continues remain explicitly unsupported.
  Last-use move recovery follows indirect borrowers and borrows retained by mutable
  callees; a dead direct variable does not imply that its shared readers are dead.
  Fixed-point loop wrappers reconstruct tuple, unit, and ignored input patterns
  from bound arguments rather than requiring every input to remain a named variable.
  Reference identities are inlined only after checking their complete Rust bodies;
  effectful overrides and dynamic calls retain their original dispatch.
  Registered function and global models replace their Rust implementations before
  symbolic execution. Reachability preserves local roots, required interfaces and
  initializers, but removes dependencies used only inside replaced bodies.
- `verification/aeneas-trait-models.patch` preserves associated types and inherited
  method dispatch, and adapts modeled state only when its borrowed signature is
  compatible with the extracted Rust specialization.
  Owned constructors retain explicit recovery lenses, so consuming adapters and
  delegates preserve inputs needed by later borrowed givebacks. Structured
  specializations retain runtime and marker dictionaries. Default-only modeled
  methods require every retained Rust implementation to reuse the default;
  native overrides are rejected rather than silently bypassed.
  Recursive function/implementation groups emit recursive functions before
  reconstructing their dictionaries, including generic `Debug`, `Display`, and
  `PartialEq` implementations.
  Scoped mixed-control joins lower through tagged exit payloads: fallthrough shares its
  continuation, while `return`, `break`, `continue`, and panic retain their targets.
- `verification/aeneas-shared-borrows.patch` preserves nested projection ownership,
  parent dependencies, and retained payloads across calls, joins, and loop exits.
  Global region coalescence preserves neighboring projection masks, and concrete
  shared-borrow fragments retain their selected projection owner's region.
  Retained payloads do not absorb independently borrowed lifetimes; merges retarget
  children of both original owners. Shared identity reduction is replayed after
  branch projection. Saved lookup environments retain only concrete loan-bearing
  entries, not continuation history; borrow-free values retain no environment.
  Ended ignored loans preserve lookup snapshots and live child ownership when
  normalized for joins.
  Selected mutable let-binders retain their owned referents, including unchanged
  optional payloads returned through early loop exits.
  When reconciliation transfers a nested lender into a fresh abstraction, input
  references still resolve its source-branch payload; existing lenders retain their
  already-reconciled projections.
  Snapshots capture names directly, and read-only place access does not rebuild
  the environment. Live-ownership checks skip types and saved continuation metadata
  without disabling the separate typing checks.
  Operational borrow updates, projection-mask isolation, and region substitution
  preserve shared continuation histories instead of expanding them into trees.
  Historical symbolic expressions are validated once per abstraction owner, while type
  classification and region erasure use domain-local weak caches. Shared-loan
  simplification indexes borrowers once per immutable context.
  Region coalescence reuses unchanged type annotations within each substitution.
  Identity-preserving abstraction updates stop at their target; renames still
  repair parent references throughout the current frame.
  Binder transformations memoize shared expressions at their actual scope and
  skip subgraphs without relevant dependencies; identity alone never merges
  distinct binding depths.
  Continuation composition keeps local binders closed when substituting free
  variables, validates scope and pattern arity, and preserves returned ownership
  annotations instead of repeatedly reopening whole expression suffixes.
  Identifier collection visits shared type and continuation nodes once per query.
  Live ownership queries distinguish ownership levels when reusing expression
  visits; value-only predicates skip irrelevant type annotations.
  Type queries and substitutions also share separately allocated equal annotations,
  avoiding collision chains from identity-only caches.
  Full region erasure visits shared type nodes once; local and ADT-field
  annotations reuse the same erasure cache.
  Switches share repeated match-arm bodies, then join fallthrough states and
  loop exits with the same destination without redirecting other outcomes.
  Calls returning `!` terminate their path rather than falling through.
- `verification/aeneas-floating-scalars.patch` retains floating operations and
  literals in the pure IR. Empty matches retain their contextual result type
  through lowering and extract as Lean `nomatch`.
  Callable tuple and variant constructors use a separate generated name from
  their pure type constructors, preserving higher-order constructor calls.
  Monadic constructors are qualified so Rust methods such as `Result::ok` cannot
  shadow generated successful returns.
- `verification/aeneas-collection-contract.patch` enforces the supported collection
  domain before translation. Modeled `Vec`/B-tree collection and vector extension
  require reviewed standard iterator dictionaries, recursively through supported
  adapters and `Result` collectors. Custom iterators, unresolved generic dictionaries,
  and unreviewed consumers fail with `E_UNSUPPORTED_COLLECTION`; they do not produce
  a guessed successful result or a replacement Lean panic.
  Structured specializations retain their real borrowed-state ABI; admission follows
  their original Rust implementation identity and instantiated dictionary arguments.
- `verification/aeneas-stdlib-models.patch` supplies the checked Lean models and
  generated builtin metadata: slice windows/chunks and borrowed accessors;
  shared and mutable vector iteration; owned array iteration; double-ended,
  copied/cloned, filtered, chained, zipped, skipped, and flat-map adapters;
  inclusive ranges and single-item sources; ordered maps and mutable entries;
  `Result` iterator collection with retained errors and resumable source state;
  UTF-8 string ownership, comparisons and joining; checked integer conversions,
  optional cloning and borrow-preserving owned helpers; fallible control flow,
  optional equality with its runtime dictionary, boxed and vector inequality,
  length-checked slice-array comparisons, nonzero primitive constructors, and
  kernel-reducible floating scalars. Vector extension and stable sorting retain
  iterator and callback state. Formatting produces observable text through real
  display/debug callbacks, including escaping, padding, precision and floating
  conversion; callback and writer failures stop formatting.
  Recursive callbacks and loop bodies carry checked monotonicity proofs.
  The custom `do` elaborator processes statement-scoped holes iteratively, retaining
  lexical scope and deferred type inference without a growing continuation stack.
  Nonrecursive loop bodies register directly for simplification by unfolding,
  avoiding redundant equation synthesis over large generated definitions.
  Imported model names are root-qualified. Closed array literals use kernel-checked
  length proofs; symbolic lengths retain reflexivity and simplification fallbacks.
  Large string bounds likewise avoid recursive simplification.

The production `shape_source_for_axis` search uses an ordered, short-circuiting
loop instead of recursive `find_map` closures. This preserves the first successful
input's provenance while avoiding unsupported recursive closure dictionaries.

The patched Charon, Charon-ML, and Aeneas packages build through Nix. The packaged
compiler passes all 25 native/Lean model fixtures, 16 per-root collector rejection
checks, and 1,099 native/Lean floating observations. The loop fixture checks 43 outcomes,
including mixed exits, nested returns, panic propagation, nested shared iteration,
constructor payloads after shared wildcard arms, and preservation of optional
owners on first-iteration and later errors.
Interpreter checks, strict joins, and warnings-as-errors remain enabled.
This is exercised agreement, not a correspondence theorem.

The model runner covers loop error precedence, returned mutable borrows, nested
projection state, borrowed metadata, iterator state transitions, inherited dispatch,
uninhabited variants, never-returning calls both inside and outside loops, and
post-move cleanup. Shared-projection checks include selecting a mutable iterator
without invalidating its shared slice, short-circuiting optional observations, and
conditionally lending a borrowed vector before consuming its retained references.
Recursive shared or-pattern checks retain both borrowed children and preserve
the order and existing contents of the mutable output.
Filtered-iterator checks retain callback writes on rejected and terminal elements.
Post-move checks also copy lenders while direct readers and references stored in
mutable containers remain live, exercising both paths through generated Lean.
Constructor-item checks retain tuple payloads and enum argument order while
compiling both the callable definitions and their generated types.
Comparison checks preserve vector and boxed `ne` overrides and verify that
slice-array length mismatches do not invoke element comparators.
String checks exercise UTF-8 bytes, embedded NULs, empty values, ownership and
separator placement. Sequence checks cover removal bounds, non-fused explicit append,
admitted extension with captured-state giveback, stable duplicate ordering and callback failures.
Formatting checks compile the extracted formatting calls and compare their
rendered text with native Rust cases. Bounded-writer checks preserve each Unicode
padding write, integral sign/prefix/digit boundaries, floating Copy/Zero parts,
and partial output when a callback fails. They also compile a large literal template
under a constrained elaboration recursion depth. The formatter callback models
`write_str` and default UTF-8 `write_char`; custom Rust `fmt::Write` implementations
and `write_char` overrides have no extraction mapping.
Stable sorting uses a merge-sort schedule, not the pinned implementation's exact
comparison/key invocation order or count. Correspondence through this model
requires schedule-independent comparison and key callbacks.
`Result` collection checks cover early collectors, polling after an error,
specialized `try_fold` dispatch, source and callback restoration, and effectful
`by_ref` overrides. For admitted calls, collection delegates to the supplied
`FromIterator` rather than precollecting values or forcing fused iterator behavior.
Use the built Lean library from the patched `aeneas-source` package, not the
unmodified upstream library:

```sh
nix develop --command python verification/check_models.py \
  --aeneas-lean-dir .work/aeneas-verification/aeneas/backends/lean
```

The runner executes each native fixture, extracts fresh LLBC and Lean with the
pinned tools, compiles the generated module, and executes its Lean checks.
`--model projection_iterator_state` selects one regression; `--model` is repeatable.
Temporary models are removed after each run.

`--model collection_contracts` exercises explicit rejection as well as native
counterexamples. Generic map/set and associated-iterator delegates that exceed the
new contract retain their native tests and have per-root rejection checks; they are
not counted as successful generated Lean observations.

The admitted collector inputs are standard slice, array, vector and B-tree
iterators, chunks/windows, once, chars, and `usize`/`i32` ranges. Mutable references,
map, copied/cloned, enumerate, filter and filter-map recursively require an admitted
source. This restriction applies to modeled collection/extension, not to every use
of an iterator. Other consumers such as Zip, Take, Chain and inclusive ranges remain
unreviewed at this boundary. Even a custom iterator inheriting default `size_hint`
is rejected: restricted Charon roots do not establish absence of unsafe specialization
markers. Capacity, allocation-failure timing, destructor effects and pointer identity
are still outside the value-only collection model; comparator/key callbacks retain
the separate schedule-independence obligation above.

The extraction command invokes the real production code, includes reachable bodies from
all retained crates, treats translation warnings as errors, forces a fresh root-package
compilation, translates functions sequentially to limit live symbolic contexts, and
refuses a missing/empty model:

```sh
nix develop --command python verification/extract.py
```

A diagnostic slice is explicit, never reported as the full core:

```sh
nix develop --command python verification/extract.py --entry chelis_ir::grad::grad_dag_checked
```

Outputs are isolated per invocation under `.work/aeneas-verification/extracted/`.
The command performs Charon/Aeneas model generation only. It is not a correspondence
proof or Lean acceptance gate.

### Production AD acceptance

Use a writable Lean library from the pinned patched source. If one has not already
been prepared, this one-time setup copies the source, preserves its pinned Lake
manifest/toolchain, and builds it. The destination must not already exist:

```sh
nix develop --command python verification/prepare_aeneas_lean.py \
  --source "$(nix build --no-link --print-out-paths .#aeneas-source)" \
  --destination .work/aeneas-verification/aeneas/backends/lean
```

Run the production gate with that library:

```sh
nix develop --command python verification/check_production_ad.py \
  --aeneas-lean-dir .work/aeneas-verification/aeneas/backends/lean
```

The gate builds the library using Lake's existing cache, freshly extracts the fixed
`chelis_ir::grad::grad_dag_checked` root with promoted MIR, and compiles the unmodified
`ChelisIr.lean` to `ChelisIr.olean`. It shares the strict extraction implementation
with `extract.py`; there is no alternate root, old-model reuse, or checks-disabled mode.
Library-build, Charon, Aeneas and Lean failures return a nonzero status, including
unsupported collector dictionaries. Missing or empty AD artifacts also fail.

Only after successful compilation does the command print `Production AD accepted`
and the compiled artifact path. Each invocation retains its own LLBC, generated Lean
and compiled model under `.work/aeneas-verification/extracted/`. This acceptance is
not full-core extraction, a backend axiom audit, or a correspondence theorem.

### Verification status and remaining boundaries

The packaged compiler passes the 25-model native/Lean matrix and all 16 unsupported
collector rejection checks described above.
The former AD blockers for slice windows/chunks, B-tree iteration and mutable
entries, generic iterator dispatch, and recursive formatting/comparison
dictionaries are covered by the patched models and extraction regressions.

The production AD slice `chelis_ir::grad::grad_dag_checked` passes both extraction
and Lean compilation. Its promoted-MIR LLBC translates with interpreter checks,
strict joins, abort-on-error, and warnings-as-errors enabled. The unmodified
generated `ChelisIr.lean` compiles to `ChelisIr.olean` against the patched Lean
library, without extra command-line stack overrides. The generated module reports
only a naming-style warning for `__build_unimplemented_rejection`. The library build
also reports existing proof-admission (`sorry`) and linter warnings; the gate leaves
these visible rather than treating compilation as proof completion.
This acceptance includes the fail-closed collector gate and the concrete production
AD/verifier refactors; the public generic `UnordMap::FromIterator` is unchanged.

The [patch dependency audit and source proposal](ARCHITECTURE.md#production-ad-patch-audit-and-source-proposal)
record remaining dependencies and the limits of that audit. The
[explicit-state runtime cutover](ARCHITECTURE.md#explicit-state-cutover) now gives
`HostLoweringSession` ownership of specialization, inlining, preflight, and scoped
type-substitution state. `DiagnosticSink` owns scoped checker opacity policy and
recursive inference frames, without ambient opacity or recursive-group TLS.
Generalization now takes explicit borrowed recursive pins in both its level-based
and sweep algorithms. Native regressions cover independent sessions, nested return
and unwind, caller attribution, ordered diagnostics, and mid-SCC cancellation.
The API smoke also covers serialized and layered generic library contexts, reuse,
and concurrent checks. Persisted `TypeEnv` metadata is unchanged.

Link provenance is also explicit: type-inference and fitness APIs take
`ProgramOrigin::{Unlinked, Linked}`, and `PreparedProgram::from_expanded_deep` requires
it alongside the expanded AST. There is no linked-program TLS, install guard, or
implicit default. `Linked` is a trusted native linker assertion, not a checked proof;
names, metadata, and cached libraries cannot authorize a new source unit. Origin is
not serialized. Native checks cover rejection boundaries and linked/contextual
checking through lowering and evaluation, including cancellation/reuse and concurrency.

Lowering's native dependency is explicit too: acquire `&LoweringRuntime` with
`chelis_ir::lowering_runtime::with_native_lowering_runtime` and pass it through
canonical lowering and semantic completion APIs. Completion needs it when computing
suppression-dependent DAG root classifications; pure type analysis and cached-library
validation do not. `HostLoweringSession::new(program, runtime)` borrows the same
thread-bound service. Checked products and persisted caches contain no runtime handle.

The native adapter and panic hook retain the per-thread storage. Nested calls share
its state; guards clear flags to false, and fatal re-raise explicitly restores quiet
output after an inner catch. Process-isolated regression coverage observes payloads,
hook/finalizer order, payload destruction, nested scopes, and thread isolation.
The API smoke covers strict/contextual evaluation (`21`), fatal AD rejection through
DAG and host paths, nested reuse, and concurrent invocations.

Explicit inputs and ownership are progress toward FCIS, not purity: cancellation,
global IDs, counters, stack growth, and panic handling remain native boundaries.
These changes are not checker/host-lowering extraction or wrapper correspondence
proofs.

Whole `chelis_pipeline_core` extraction is still blocked before LLBC generation.
The current first failure is Charon/Hax's external closure-region handling for
`chelis_ir::lower::try_lower_program_to_library::{closure#0}` with two erased capture
regions. Native suppression TLS now lives behind `lowering_runtime`, and lowering
rejection returns an explicit `LowerResult`; neither supplies the remaining native
hook/unwind correspondence. The AD slice is not the full core.

There is no Rust/Lean/LaCaDiLE correspondence theorem or full-core proof-acceptance
claim. Unsupported runtime paths have not been replaced with opaque shims,
no-ops, feature-gated substitutes, or axioms. Correspondence still requires
checked proofs with the numerical and runtime relations stated. Strict
external-definition checks and Lean compilation alone do not audit the backend
library's axioms or abstractions; those are separate acceptance obligations.
