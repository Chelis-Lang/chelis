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
xcode-select --install
```

Required:
- **Linux:** GCC + OpenBLAS for BLAS-backed matmul and OpenMP loop parallelism
- **macOS Apple Silicon:** Apple clang + Accelerate (`vecLib`) are supported out of the box
- **Homebrew GCC on macOS:** optional performance path when you want OpenMP-enabled CPU loops

Optional on macOS:
```sh
brew install gcc
```

Optional:
- **Valgrind** — memory leak tests on generated C

**Python via [uv](https://docs.astral.sh/uv/)** (for `scripts/gate.py`, the
chelis-tools CLI in `py/`, and the `chelis-python` PyO3 link step):

```sh
# Install uv once
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the project-local venv (from the repo root)
uv venv --python 3.11
```

The project pins Python 3.11 (`py/pyproject.toml` requires `>=3.11`).
Always use the uv-managed interpreter at `.venv/bin/python` — `.cargo/config.toml`
sets `PYO3_PYTHON` to it so `cargo build -p chelis-python` links against the
right `libpython` on every developer's machine. Using the system Python is
**not supported**; on macOS, Apple's bundled `python3` reports a stale
`sysconfig.LIBDIR` that breaks the PyO3 link step, and on Linux the
system Python may not match the chelis-tools version constraint.

Install Python dependencies into the uv venv as needed:

```sh
uv pip install -e py            # chelis-tools (gate, loc-report, skill-eval, ...)
uv pip install -e bindings/python # chelis Python bindings (optional)
```

## Build

```sh
cargo build --workspace
cargo test --workspace
```

`cargo test --workspace` covers the compiler, evaluator, backend, and spec
regressions.

`chelis build`, `chelis check`, `chelis validate`, and `chelis eval --file`
each enforce a built-in **style gate** (`chelis fmt --check` plus the
blocking `chelis lint` rule set) on the input file before the
front-end runs. Style failures fail the command; advisory lint warnings
such as `redundant-linearity-call` do not. Run `chelis fmt --inplace
path/to/file.ch` to canonicalize, or pass `--allow-style-violations`
to bypass only the style gate for emergency builds (CI must not). See
[`docs/book/src/cli.md`](docs/book/src/cli.md) for the full contract.

For real downstream proof against Nautilus without going through release
artifacts or GitHub Actions, build a local compiler binary and run:

```sh
python3 scripts/nautilus_local_gate.py baseline
python3 scripts/nautilus_local_gate.py tensor-grad
python3 scripts/nautilus_local_gate.py tensor-fold
python3 scripts/nautilus_local_gate.py eval-imports
```

See [scripts/README.md](scripts/README.md) for the local downstream gate
workflow.

## Project Structure

```text
crates/
  chelis-shell/        .chb Shell metadata
  chelis-reef/         Reef manifests, lockfiles, local registry, package linker
  chelis-std-bundle/   Compile-time-embedded chelis-std runtime artifacts
  chelis-deep/         Deep parser and canonical printer
  chelis-surf/         Surf parser, desugaring, decompilation
  chelis-types/        Type checker, dimensions, precision, fitness
  chelis-effects/      Effect inference/checking over annotated Deep
  chelis-ir/           RISC DAG, lowering, transforms, evaluator
  chelis-runtime/      Rust runtime library and C ABI header
  chelis-backend-c/    C backend code emitter
  chelis-backend-hip/  HIP backend code emitter
  chelis-tide/         Tide HTTP/JSON API and MCP server
  chelis-lsp/          Tide Language Server Protocol support
  chelis-cove/         Cove terminal coding environment
  chelis-cli/          CLI binary
editors/vscode/        VS Code-compatible extension and TextMate grammars
grammars/              Tree-sitter grammars for Surf and Deep
packages/chelis-std/   Source for the chelis-std language runtime (compiler-bundled,
                       not installed via reef; bytes embedded by chelis-std-bundle)
docs/book/             mdBook source for developer-facing docs
examples/              Executable example programs
spec/                  Numbered language specs and design docs
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
- [chelis-std SKILL File](packages/chelis-std/SKILL.md)
- [Developer Book](docs/book/src/README.md)
- [Canonical Project Reference](spec/design/chelis_canonical_reference.md)
- [Ecosystem Context](spec/design/chelis_ecosystem_context.md)
- [Context](spec/00-context.md)
- [Nomenclature](spec/01-nomenclature.md)
- [Roadmap](spec/12-roadmap.md)
- [Project Plan](spec/design/chelis_project_plan.md)

## License

MIT
