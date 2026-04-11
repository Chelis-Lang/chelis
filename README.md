# Chelis

Chelis is a functional programming language for AI research.
It is designed for a workflow where a coding agent is the primary author and a human is
the supervisor.
Surf is the readable syntax for humans.
Deep is the canonical s-expression syntax for machines and the compiler.

**Status:** Phase 0 is complete.
Phases 0a-0i are complete.
Phases 1a-1f are implemented.
Phase 1 is structurally complete for the shipped fixed-workload deliverable, with known
backend limitations carried forward explicitly in the design docs.
Phase 2a has shipped its first bounded subset: effect syntax in Surf/Deep, annotated
checked Deep as the downstream contract, `Random` via `dropout` + `with seed(...)` in
the evaluator path, and `Resource(Device)` validation on the build boundary.

## Prerequisites

**Rust toolchain** (stable, with rustfmt and clippy):
```sh
rustup default stable
rustup component add rustfmt clippy
```

**C toolchain** (for the C backend — Phase 0f+):
```sh
# Fedora / RHEL
sudo dnf install gcc openblas-devel valgrind

# Ubuntu / Debian
sudo apt-get install gcc libopenblas-dev valgrind

# macOS
brew install gcc openblas
```

Required:
- **GCC** (or clang) — compiles generated C code
- **OpenBLAS** — BLAS matmul path (`cblas_sgemm`)
- **OpenMP** — ships with GCC (`-fopenmp`)

Optional:
- **Valgrind** — memory leak tests on generated C

## Build

```sh
cargo build --workspace
cargo test --workspace
```

`cargo test --workspace` covers the compiler, evaluator, backend, and spec regressions.
It does not run the full release-mode real-MNIST milestone by default.

## Current Focus

Phase 0 is now complete through:

- Surf -> Deep -> typecheck -> lower -> grad -> eval execution
- C backend codegen with BLAS and OpenMP validation
- executable spec-suite coverage by language behavior
- end-to-end MNIST training on CPU

Phase 0i shipped:

- `chelis deep`
- `chelis surf`
- `chelis fmt`
- `chelis eval`
- `chelis check`
- `chelis build`
- `chelis tide`
- `chelis tide serve`
- `chelis tide mcp`
- `chelis tide lsp`
- `chelis cove`

Deep formatting now defaults to canonical pretty-printed `.dp` output:

- `chelis deep app.ch` prints width-aware canonical Deep
- `chelis deep --flat app.ch` preserves flat per-form rendering for explicit machine
  pipelines while keeping top-level forms separated
- `chelis fmt file.dp --check` verifies canonical `.dp` formatting without rewriting the
  file

The current shipped surface includes the Phase 1 HIP backend work,
`chelis validate` for executable grammar conformance, and the first Phase 2a effect
surface:

- optional effect annotations on `sig` and `def`
- `with seed(...) { ... }` and `with device(...) { ... }` Surf handler syntax
- checked programs now flow downstream as annotated Deep, with type metadata written
  onto the returned tree
- `dropout(x, rate)` as the concrete `Random` source in evaluation/lowering
- build-target validation for resource regions such as `with device("gpu:0") { ... }`
- the Phase 2e Tide machine-facing API surface (`chelis tide serve` and
  `chelis tide mcp`)
- the Phase 2f Tide LSP/editor package surface in `editors/vscode/`
- the Phase 2g `chelis cove` terminal UI plus checked-in tree-sitter grammars for Surf
  and Deep
The shipped Phase 1 benchmark models compile and run on both backends.
Known carried-forward limitations are:

- HIP does not yet implement `pad` / `shrink`; no current Phase 1 model uses them
- symbolic dimensions now ship on both backends through the stable tensor ABI:
  generated functions bind symbolic names from input tensor metadata at runtime, so
  batch/sequence changes do not require recompilation on the supported Phase 1 surface
- `layer_norm` still requires a concrete normalized-axis extent; symbolic leading dims
  are supported, but a symbolic hidden size remains a follow-up
- compiler-emitted dotted Deep module/import paths still do not fully round-trip through
  the compiler-side Deep parser
- `chelis build` does not yet codegen `dropout` for either `c` or `hip`; seeded dropout
  currently ships on the evaluator path only

These limitations are real debt, but they do not block the Phase 2 language work.
AI assistance planning is split cleanly:

- Phase 2: `SKILL.md` + Tide MCP/HTTP API for frontier models
- Phase 3: ecosystem foundations, starting with the Surf style pass (pipe-first,
  short-form block bindings, width-aware multiline pipes), then package/FFI work and
  the `SKILL.md` v2 refresh
- Phase 4: a local coding model that ships with the toolchain

Phase 2g currently ships as:

- `chelis cove --file examples/mnist.ch`
- live Surf editing with checked-in tree-sitter highlighting
- read-only Deep view derived from the current Surf buffer
- live diagnostics and numeric fitness status
- `Ctrl-S` save, `Ctrl-R` compile, `Ctrl-E` eval, `Ctrl-Q` quit

Phase 3 now ships:

- `chelis reef init`
- `chelis reef build`
- `chelis reef publish`
- local-first Reef package resolution for `chelis check` and `chelis build`
- bundled `packages/chelis-std` dogfooding the package system through `.chb` shell
  metadata plus source archives

## Phase 0h Validation

Release-mode MNIST validation is checked in as:

```sh
cargo run --release -p chelis-e2e --bin train_mnist -- --epochs 5 --min-acc 0.90
```

Measured on the checked-in path:
- 5 epochs on real MNIST
- final test accuracy: `0.9272`

This release runner is the authoritative 0h gate.
The ignored MNIST tests mirror it for manual test-harness use, but the normal workspace test run does not attempt the full long real-data training job.

## Project Structure

```text
crates/
  chelis-shell/      .chb Shell metadata
  chelis-reef/       Reef manifests, lockfiles, local registry, package linker
  chelis-deep/       Deep parser and canonical printer
  chelis-surf/       Surf parser, desugaring, decompilation
  chelis-types/      Type checker, dimensions, precision, fitness
  chelis-effects/    Effect inference/checking over annotated Deep
  chelis-ir/         RISC DAG, lowering, transforms, evaluator
  chelis-runtime/    Rust runtime library and C ABI header
  chelis-backend-c/  C backend code emitter
  chelis-tide/       Tide HTTP/JSON API and MCP server
  chelis-lsp/        Tide Language Server Protocol support
  chelis-cove/       Cove terminal coding environment
  chelis-cli/        CLI binary
  spec/                Numbered language specs and design docs
editors/vscode/      VS Code-compatible extension and TextMate grammars
grammars/            Tree-sitter grammars for Surf and Deep
packages/chelis-std/ Reef-packaged standard-library dogfood set
examples/            Executable Phase 0 example programs
examples/illustrative/  Non-executable syntax/design examples
```

## Key Ideas

- **Dual syntax:** Surf (`.ch`) for supervision, Deep (`.dp`) for canonical machine-facing
  structure
- **Deep is canonical:** every Deep node has the form `(tag {} children...)`
- **No implicit surprises:** no silent precision promotion, broadcasting, or hidden
  partial application
- **Compiler as training signal:** fitness scores, structured errors, and repair
  suggestions
- **Small computational core:** tensor programs lower to a compact RISC DAG
- **First-class transforms:** `grad`, `vmap`, and `jit` are compiler-level rewrites

## Documentation

- [Agent Contract](AGENTS.md)
- [LLM Skill File](SKILL.md)
- [Canonical Project Reference](spec/design/chelis_canonical_reference.md)
- [Ecosystem Context](spec/design/chelis_ecosystem_context.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Context](spec/00-context.md)
- [Nomenclature](spec/01-nomenclature.md)
- [Phase 2 Plan](spec/design/chelis_phase2_plan.md)
- [Project Plan](spec/design/chelis_project_plan.md)

## Agent Tooling

This repo keeps shared agent guidance in:

- `AGENTS.md`: canonical coding-agent instructions
- `CLAUDE.md`: should resolve to `AGENTS.md`
- `SKILL.md`: compact Chelis code-generation teaching document
- `agent-skills/`: project-local reusable workflows for red teaming, phase gating, spec sync,
  backend numerics, CLI surface validation, and example policy

Tool-specific skill entry points should resolve to the same local skill library:

- `.claude/skills` -> `agent-skills/`
- `.codex/skills` -> `agent-skills/`

External Phase 2e validation can be run with:

```sh
python scripts/redteam_tide_phase2e.py
```

## License

MIT
