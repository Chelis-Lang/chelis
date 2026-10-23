# Architecture

Chelis has two source syntaxes and one checked compiler pipeline. This page
maps the implemented workspace; the
[canonical project reference](spec/design/chelis_canonical_reference.md)
owns project-level decisions, and the [numbered specs](spec/00-context.md)
own language semantics.

## Source to checked program

```text
Surf (.ch) → parse → desugar ─┐
                              ├→ expanded Deep → type analysis → effects and
Deep (.dp) → parse ────────────┘                  linearity checks → checked program
                                                                 ↓
                                                    lower → RISC DAG
```

Surf is the readable syntax. Deep is the canonical machine-facing syntax.
Surf macros expand before semantic analysis; both source paths enter the
pipeline as expanded Deep. `chelis-surf` and `chelis-deep` own their respective
parsers, `chelis-types` owns type analysis, and `chelis-effects` owns effect
checking. Linearity checking follows type analysis. For file inputs, the CLI
runs its formatter and blocking lint checks before the compiler pipeline.

`chelis-pipeline-core` owns the typed transitions from a prepared Deep
program through semantic checks and lowering. Successful checks produce a
`CheckedCompilation`; lowering produces a `LoweredCompilation` with the DAG
and root metadata. A rejected program cannot yield either success artifact.
`chelis-compiler-api::pipeline` is the public facade. It prepares source and
offers three goals: type analysis, full semantic checking, and lowering.
For CLI file checks and builds, Reef links Surf declarations; the CLI passes
them through the compiler API for desugaring and macro expansion. For
`chelis reef build`, Reef expands linked declarations and uses the semantic
core directly.

`chelis-dag-core` owns graph models, structural verification, and graph-only transforms.
`chelis-ir` retains source lowering, evaluation, host state, and target admission.
Backend preparation verifies the DAG and applies target-specific checks before emission.

## Execution and build paths

| CLI path | Result |
|---|---|
| `chelis eval` | Checks and lowers the program, then executes it with the local IR evaluator. `--target eval`, `c`, `hip`, or `metal` selects the target's capability and root manifest; it does not run generated native or GPU code. |
| `chelis build --target c` | Emits C source and a header, stages the C runtime artifacts, and invokes the native compiler to produce an executable or static library. C is the default build target. |
| `chelis build --target hip` | Emits C++ host source and a header with embedded HIP kernels, stages the HIP and C runtime artifacts, and invokes `hipcc`. The HIP runtime compiles kernels with `hiprtc` when the resulting program runs. |
| `chelis build --target metal` | Emits Objective-C++ host source (`.mm`) and a header with embedded Metal Shading Language kernels, stages the Metal and C runtime artifacts, and invokes `clang++`. The resulting program compiles kernels through Metal when it runs. |

Build retains generated sources and runtime artifacts alongside the native
product. `--emit-c` stops after source emission without invoking a native
compiler. The C backend handles CPU execution and runtime integration; the HIP
and Metal backends generate GPU helpers alongside host code where the
program requires them. A target may reject a program it cannot lower or
emit. The [build guide](docs/book/src/backends.md) covers user-facing C
builds; [GPU backends](docs/gpu_backends.md) covers the HIP and Metal targets.

## Workspace map

| Area | Primary crates |
|---|---|
| Source syntax and predicates | `chelis-surf`, `chelis-deep`, `chelis-pred`, `chelis-macros` |
| Numeric values and policy | `chelis-numeric-core`, `chelis-crmath`, `chelis-vocab` |
| Type models and unification | `chelis-type-core` |
| Semantic pipeline | `chelis-types`, `chelis-effects`, `chelis-pipeline-core`, `chelis-compiler-api` |
| DAG models and graph transforms | `chelis-dag-core` |
| Source lowering and evaluation | `chelis-ir` |
| Native emission and runtime | `chelis-backend-c`, `chelis-backend-hip`, `chelis-backend-metal`, `chelis-runtime` |
| User entrypoints and packages | `chelis-cli`, `chelis-reef`, `chelisup` |

For implementation details, see the
[compiler pipeline inventory](docs/investigations/compiler_pipeline_inventory.md).
For a language change, start with the owning numbered spec and follow
[spec sync](agent-skills/spec-sync/SKILL.md).

## Portable version predicates

`chelis-version` uses unconditional `no_std` without allocation or package dependencies.
Its public predicates retain their installer and conformance rules.
Strict versions contain three nonempty ASCII digit components.
Leading zeroes and arbitrarily long components are valid.
Path components accept channels and Unicode text but reject separators and control characters.
Empty names and single-dot or double-dot names are invalid.

Run `.venv/bin/python scripts/version_core_oracle.py` for target compilation,
an independent host consumer, allocation checks, and installer resolution checks.
The [acceptance gate](docs/manual_gates.md#version-core-portability) lists prerequisites and CI owners.
Target compilation does not prove device execution.

## Portable syntax and predicates

`chelis-deep`, `chelis-pred`, `chelis-surf`, and `chelis-macros` use unconditional `no_std` with `alloc`.
They share the existing Deep representation, grammar, macro rules, and Surf formatter.
`chelis-unord` retains its order-free collection interface with allocation-backed portable storage.
These libraries need an allocator and a caller-defined stack policy.

`chelis-stack-services` defines the allocation-free `StackServices` interface.
Parser, formatter, desugarer, and decompiler entry points take a final `&impl StackServices` argument.
Native callers pass `&chelis_syntax_host::NativeStack`.
Embedded callers supply their platform implementation.
The portable crates contain no implicit host implementation or `std` feature.

`chelis-syntax-host` owns native stack growth and path-based module identity checks.
The module identity import is `chelis_syntax_host::module_identity`, not `chelis_surf::module_identity`.
Resugar operations use request-local state instead of thread-local state.
Compiler orchestration, source discovery, diagnostics presentation, and filesystem access remain outside the portable graph.

Run `.venv/bin/python scripts/syntax_predicate_core_oracle.py` for isolated target builds and fixed-revision native parity.
The [acceptance gate](docs/manual_gates.md#syntax-core-portability) lists prerequisites, evidence, and CI owners.
The command checks parser, wire, formatter, macro, predicate, and host behavior.
Allocation checks compare four source cases with the accepted source revision.
Target compilation proves neither device execution nor device memory capacity.

## Portable type models and unification

`chelis-type-core` owns `Type<E>`, `NominalArg<E>`, `Scheme<E>`, substitutions, and pure scheme and solver operations.
It uses unconditional `no_std`, explicit `alloc`, empty defaults, and optional portable `serde`.
Consumers must specify their evidence type.
There is no default evidence parameter, conversion API, old model alias, or host re-export.

`chelis-numeric-core` retains the sole `Prim` identity.
The core uses current `chelis-unord` collections and private spin mutexes.
The normal graph disables dependency defaults.
The serde feature enables portable core model transport.

Portable Deep supplies the canonical cast-mode data and retains its own serde dependency.
Vocabulary remains unchanged below Deep.
The core has no DAG or hosted inference dependency.

The solver retains current nominal arguments, dtype restrictions, inference levels, collection contracts, and shared type children.
Logical aliases remain separate from transient resolution caches.
Scheme substitution stops at quantified aliases after ordinary resolution.
Protected-operation interruption rejects later store access.
Mutable substitution clones remain independent.

Normal concurrent queries wait for active store operations.
Identifier reservation visits each shared allocation once across scheme bodies, collection contracts, and result equations.

`chelis-types` uses `Type<ErrorWitness>`, `Scheme<ErrorWitness>`, and `Subst<ErrorWitness>` directly.
It retains witness construction, diagnostic reports, private sinks and checkpoints, source resolution, recursion policy, and the successful-product veto.
Pure decisions return portable failures and optional source labels.
The host attaches source diagnostics through its existing report boundary.
Trusted cache decoding does not establish authority for external payloads.

The solver needs an allocator and target atomics.
A reentrant interrupt must not enter a solver whose lock it preempts.
Run `uv run --managed-python --python 3.11 --no-project python scripts/type_unification_core_oracle.py` from the repository root.
The [acceptance gate](docs/manual_gates.md#type-unification-core-portability) separates immutable-source parity, host concurrency, and target compilation.
No numbered language chapter transfers authority through this extraction.

## Portable Reef bootstrap graph policy

`chelis-reef-bootstrap-core` uses unconditional `no_std` with `alloc` and no package dependencies.
It checks version conflicts, exact edges, and cycles before it returns dependency-first input indices.
Each input occurrence remains separate, and dependency lookup selects the last matching occurrence.
Traversal uses explicit heap-backed frames instead of recursive graph descent.

For explicit bootstrap inputs, Reef matches typed peer requirements before graph projection.
It retains canonical release checks, runtime checks, manifest access, artifact verification, and each occurrence's original release spec.
The default bootstrap resolver remains in Reef.
After default resolution, Reef sends selected package identities to the same checked core.
This extraction does not make Reef or the compiler pipeline portable.

Run `.venv/bin/python scripts/reef_bootstrap_core_oracle.py` for isolated target consumers and real host bootstrap checks.
The [acceptance gate](docs/manual_gates.md#reef-bootstrap-core-portability) lists prerequisites, evidence, and CI owners.
Target compilation proves neither device execution nor allocator capacity.

## Portable CHB metadata and complete validation

`chelis-shell-core` owns CHB 8 metadata models and complete metadata validation.
It uses unconditional `no_std` with `alloc`, portable Deep, explicit stack services, Serde, and SemVer.
It checks type representations, canonical variables, type-variable restrictions, and collection contracts, as well as envelope fields and order.

Metadata imports now use `chelis_shell_core`.
There are no compatibility re-exports from `chelis_shell`.
`validate_shell`, `validate_collection_obligation_ledger`, and `validate_collection_obligations_for_type_repr` take a final `&impl StackServices` argument.
They return typed errors without a binary-codec dependency.

`chelis-shell` retains bincode, `SHELL_MAGIC`, byte encoding, canonical decoding, and filesystem adapters.
Its codec validates complete metadata through the core and maps failures to the existing host error form.
Reef retains archive-digest and requested-package identity checks before installation.
Valid metadata alone does not prove archive integrity or publisher identity.

Run `.venv/bin/python scripts/shell_metadata_core_oracle.py` for isolated target consumers, fixed-revision CHB parity, and real Reef and CLI checks.
The [acceptance gate](docs/manual_gates.md#shell-metadata-core-portability) lists prerequisites, evidence, and CI owners.
Target compilation proves neither device execution nor allocator capacity.

## Portable certificate arithmetic

`chelis-certificate-core` owns exact rational polynomials, matrices, PSD checks, certificate arithmetic, and affine projection.
It uses unconditional `no_std` with `alloc` and arbitrary-precision rational dependencies without `std` defaults.
Consumers import its `poly`, `exact`, and `linalg` modules directly.
The former `chelis_prove::clarabel_sos` arithmetic modules do not exist.

Arithmetic verification checks Gram symmetry, positive semidefiniteness, and the certificate's own polynomial identity.
It does not establish the requested goal, interval validity, or multiplier nonnegativity.
`RatSymMatrix::is_psd` needs a symmetric input.
`SosCertificate::verify` checks symmetry before PSD.

The host retains goal extraction, canonical interval layouts, Clarabel, native BLAS, floating-point proposals, evidence, and proof admission.
Admission needs exact goal and endpoint agreement, arithmetic verification, and the canonical block count, matrix dimensions, and ordered interval multipliers.
Layout construction and admission share canonical block definitions.
Only host admission can produce `CertificateBearing@Exact`.

Run `.venv/bin/python scripts/certificate_core_oracle.py` for independent target consumers, fixed-revision arithmetic comparisons, and native host admission.
The [acceptance gate](docs/manual_gates.md#certificate-arithmetic-core-portability) defines prerequisites, evidence, and CI owners.
Target compilation does not prove device execution, allocator capacity, or whole-pipeline portability.

## Portable numeric values and operations

`chelis-numeric-core` owns primitive identities, sealed scalar and tensor values, numeric operations, activation graphs, observation, and agreement.
It uses unconditional `no_std`.
Default `alloc` adds tensor buffers and owned text.
Optional `serde` preserves exact stored-value transport.
Scalar transport does not need `alloc`.
Consumers import the canonical modules directly.

`chelis-crmath` retains the shared CORE-MATH kernels.
Its `std` feature adds host profile and C-source services.
Bare targets use Clang, `llvm-ar`, 128-bit C integer support, and allocation-free IEEE helpers.
Host crates retain schema projections, I/O, native allocation, and process effects.
The runtime retains its C ABI and calls the shared IEEE conversions through private adapters.

Run `.venv/bin/python scripts/numeric_core_oracle.py` for target consumers, immutable-base parity, allocation checks, and numeric authority checks.
The [acceptance gate](docs/manual_gates.md#numeric-core-portability) states prerequisites and evidence.
Target compilation does not prove device execution, allocator capacity, or whole-pipeline portability.

## Portable graph models and transforms

`chelis-dag-core` uses unconditional `no_std` with `alloc`.
It owns graph models, checked names, structural verification, axis-source derivation, Tier-2 decomposition, reverse AD, fusion, pure optimization, spans, and graph-only ownership.
Graph consumers import these modules directly from `chelis_dag_core`.
The old IR modules and root graph exports do not exist.
Declaration owners, shape-only edges, roots, remaps, reuse metadata, numeric payloads, and valid wire data retain their current contracts.

The embedding supplies an allocator and a stack policy.
Public binding and remap maps retain `chelis_unord`.
Hash-based graph lookups use fixed portable hashes.
Adjoint storage is sparse.
Ordered outputs use node order, root order, or explicit sorted exits.

Numeric carriers and operations come directly from `chelis-numeric-core`.
Its `bitwise` and `key_admission` modules also own the shared pure operation identities.
The existing f64-image folds use `dtype_semantics::{fold_wide_unary, fold_wide_binary}` with closed operation types, sealed operands, and an explicit destination dtype.
The numeric owner retains their exact-integer image guard, refusal behavior, NaN payloads, and finalization.

Host lowering converts source linearity into graph reuse metadata before ownership normalization.
`chelis_ir::axis_admission` creates host diagnostics with the selected stage.
`chelis_ir::device_admission` retains target fences and Metal precision admission.
`chelis_ir::ordered_layout` applies existing ABI layout checks to portable contraction candidates.
Evaluation, source conversion, stack growth, host state, backend selection, and production pass order stay outside the portable crate.

Run `uv run --managed-python --python 3.11 --no-project python scripts/dag_core_oracle.py` for the complete acceptance command.
The [acceptance gate](docs/manual_gates.md#dag-core-portability) states prerequisites and evidence.
The oracle uses immutable source revision `3ca99faa7a46158eb6a0d90c2c812a7e3a5f44c1`.
Target compilation proves neither device execution nor allocator capacity.
The numbered language, transform, span, and serialization specifications retain authority.






