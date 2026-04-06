# Chelis

Chelis is a functional programming language for AI research.
It is designed for a workflow where a coding agent is the primary author and a human is
the supervisor.
Surf is the readable syntax for humans.
Deep is the canonical s-expression syntax for machines and the compiler.

**Status:** Phase 0h complete.
Phases 0a-0h complete.
Phase 0i is next.

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

Next up is Phase 0i: the first Tide workflow and evaluator/backend agreement tooling.
AI assistance planning is split cleanly:

- Phase 2: `SKILL.md` + Tide MCP for frontier models
- Phase 3: a local coding model that ships with the toolchain

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
  chelis-deep/       Deep parser and canonical printer
  chelis-surf/       Surf parser, desugaring, decompilation
  chelis-types/      Type checker, dimensions, precision, fitness
  chelis-ir/         RISC DAG, lowering, transforms, evaluator
  chelis-backend-c/  C backend and runtime
  chelis-cli/        CLI binary
spec/                Numbered language specs and design docs
examples/            Example Chelis programs
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

- [LLM Skill File](SKILL.md)
- [Canonical Project Reference](spec/design/chelis_canonical_reference.md)
- [Ecosystem Context](spec/design/chelis_ecosystem_context.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Context](spec/00-context.md)
- [Nomenclature](spec/01-nomenclature.md)
- [Project Plan](spec/design/chelis_project_plan.md)

## License

MIT
