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

`chelis-ir` owns the compact RISC DAG, transforms, verification, and the local
evaluator. Backend preparation verifies the DAG and applies target-specific
checks before emission.

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
| Semantic pipeline | `chelis-types`, `chelis-effects`, `chelis-pipeline-core`, `chelis-compiler-api` |
| DAG, transforms, evaluation | `chelis-ir` |
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


