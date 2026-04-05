# Chelis

Chelis is a functional programming language for AI research.
It is designed for a workflow where a coding agent is the primary author and a human is
the supervisor.
Surf is the readable syntax for humans.
Deep is the canonical s-expression syntax for machines and the compiler.

**Status:** Phase 0f (C backend codegen) in progress.
Phases 0a-0e complete.
All core spec documents written and reviewed.

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

## Current Focus

The active implementation work is the C backend:

- RISC DAG to C emission
- BLAS integration
- OpenMP-parallel elementwise and reduction loops
- runtime support and numerical verification

Interactive execution is planned around the IR evaluator, not a JIT backend.

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

- [Canonical Project Reference](spec/design/chelis_canonical_reference.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Context](spec/00-context.md)
- [Nomenclature](spec/01-nomenclature.md)
- [Project Plan](spec/design/chelis_project_plan.md)

## License

MIT
