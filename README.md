# Chelis

Chelis is a functional programming language for AI research. It is designed for a
workflow where a coding agent is the primary author and a human is the supervisor.
Surf is the readable syntax for humans. Deep is the canonical s-expression syntax for
machines and the compiler.

<p align="center">
  <img src="assets/mascot/chev.svg" alt="Chev Chelis, the project mascot — a turtle on a mountain bike climbing a hill" width="320"/>
</p>

## Prerequisites

**Rust toolchain** (stable, with rustfmt and clippy):
```sh
rustup default stable
rustup component add rustfmt clippy
```

**C toolchain** (for the C backend):
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

`cargo test --workspace` covers the compiler, evaluator, backend, and spec
regressions.

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
  chelis-backend-hip/ HIP backend code emitter
  chelis-tide/       Tide HTTP/JSON API and MCP server
  chelis-lsp/        Tide Language Server Protocol support
  chelis-cove/       Cove terminal coding environment
  chelis-cli/        CLI binary
editors/vscode/      VS Code-compatible extension and TextMate grammars
grammars/            Tree-sitter grammars for Surf and Deep
packages/chelis-std/ Reef-packaged standard library
examples/            Executable example programs
spec/                Numbered language specs and design docs
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
- [Roadmap](spec/12-roadmap.md)
- [Project Plan](spec/design/chelis_project_plan.md)

## License

MIT
