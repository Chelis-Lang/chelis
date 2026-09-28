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
| `chelis build --target c` | Emits C source and a header, stages the C runtime artifacts, and prints the native compile command. C is the default build target. |
| `chelis build --target hip` | Emits C++ host source and a header with embedded HIP kernels, stages the HIP and C runtime artifacts, and prints an `hipcc` command. The HIP runtime compiles kernels with `hiprtc` when the resulting program runs. |
| `chelis build --target metal` | Emits Objective-C++ host source (`.mm`) and a header with embedded Metal Shading Language kernels, stages the Metal and C runtime artifacts, and prints a `clang++` command. The resulting program compiles kernels through Metal when it runs. |

Build emits files and compile instructions. A native compiler is a separate
step. The C backend handles CPU execution and runtime integration; the HIP
and Metal backends generate GPU helpers alongside host code where the
program requires them. A target may reject a program it cannot lower or
emit. The [backend guide](docs/book/src/backends.md) covers user-facing
commands and target limits.

## Workspace map

| Area | Primary crates |
|---|---|
| Source syntax and macros | `chelis-surf`, `chelis-deep`, `chelis-macros` |
| Semantic pipeline | `chelis-types`, `chelis-effects`, `chelis-pipeline-core`, `chelis-compiler-api` |
| DAG, transforms, evaluation | `chelis-ir` |
| Native emission and runtime | `chelis-backend-c`, `chelis-backend-hip`, `chelis-backend-metal`, `chelis-runtime` |
| User entrypoints and packages | `chelis-cli`, `chelis-reef`, `chelisup` |

For implementation details, see the
[compiler pipeline inventory](docs/investigations/compiler_pipeline_inventory.md).
For a language change, start with the owning numbered spec and follow
[spec sync](agent-skills/spec-sync/SKILL.md).
