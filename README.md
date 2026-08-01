# Chelis

Chelis is a functional programming language for AI research. It is designed for a
workflow where a coding agent is the primary author and a human is the supervisor.
Surf is the readable syntax for humans. Deep is the canonical s-expression syntax for
machines and the compiler.

<p align="center">
  <img src="assets/mascot/chev.svg" alt="Chev Chelis, the project mascot — a turtle on a mountain bike climbing a hill" width="320"/>
</p>

## Install

There are two paths: **use** Chelis (install the toolchain and provision a
project), or **build the compiler from source** (the Prerequisites and Build
sections below). Most users want the first.

Chelis ships a rustup-style toolchain manager, **`chelisup`**, and a
one-command project orchestrator, **`chelis reef setup`**.

```sh
# 1. Bootstrap chelisup (drops ~/.chelis/bin/chelisup and prints the PATH line).
#    Private-repo pre-launch (needs an authenticated `gh`): fetch + run the script.
gh release download --repo Chelis-Lang/chelis --pattern chelisup.sh --output - | sh
#    (Or from a checkout: sh crates/chelisup/bootstrap/chelisup.sh)
#    Once releases are public:  curl -fsSL https://<host>/chelisup.sh | sh
export PATH="$HOME/.chelis/bin:$PATH"     # add to your shell rc

# 2. Install a toolchain (side-by-side under ~/.chelis/toolchains/<ver>).
chelisup install 0.12.1

# 3. Provision a freshly-cloned project in one command.
cd my-shell && chelis reef setup
```

`chelis reef setup` reads the project's `reef.toml` pin and brings every
dependency class to it: the toolchain (auto-installed via `chelisup`), source
packages and binary artifacts (`reef install --from-lockfile`), and chelis
source crates (`reef src sync`), then prints a `chelis reef doctor` summary.

**Version management.** A small `chelis` shim resolves the active toolchain at
each call, first match wins: a leading `+<ver>` (`chelis +0.13.0 build main.ch`)
→ `CHELIS_TOOLCHAIN` → a `chelis-toolchain` file → the nearest `reef.toml`
`compiler =` pin → the recorded default (`chelisup default <ver>`). A
resolved-but-not-installed version is a loud error, never a silent fallback.
`chelisup list-installed`, `chelisup show`, and `chelis reef doctor` report the
state.

Full guide: **[Install](docs/book/src/install.md)** and **[Reef and
Packages](docs/book/src/reef.md)**.

## Prerequisites

The rest of this README builds the Chelis compiler from a checkout.

**Rust toolchain.** Install [rustup](https://rustup.rs) (the Rust toolchain
installer) if you do not already have it:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

The repo pins its toolchain in `rust-toolchain.toml` (stable, with rustfmt
and clippy components); rustup installs all of it automatically on the first
`cargo` invocation, so no `rustup default` or `rustup component add` step is
needed.

**cargo-nextest** (the test runner CI and `scripts/gate.py` use; plain
`cargo` does not include it):

```sh
cargo install cargo-nextest --locked
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
uv pip install -e py            # chelis-tools (loc-report, skill-eval, ...)
uv pip install -e bindings/python # chelis Python bindings (optional)
```

`scripts/gate.py` itself is stdlib-only: it needs the 3.11+ interpreter
but no pip installs.

## Build

**The uv venv is a hard prerequisite for `cargo build` on every platform**:
`chelis-python` links against `libpython`, and the PyO3 link step fails
with an obscure linker error if `.venv/` does not exist. Run
`uv venv --python 3.11` (see Prerequisites) before your first build.

```sh
cargo build --workspace
cargo test --workspace
```

`cargo test --workspace` covers the compiler, evaluator, backend, and spec
regressions.

The `chelis` binary referenced throughout the docs is built from this
workspace, not installed separately:

```sh
cargo build -p chelis-cli
target/debug/chelis --help
# or, without a separate build step:
cargo run -p chelis-cli --bin chelis -- --help
```

Before pushing, run the local pre-push gate (chelis#360). `scripts/gate.py`
is the single source of truth for the per-PR gate; CI runs the same
commands:

```sh
.venv/bin/python scripts/gate.py --list   # print the canonical command list
.venv/bin/python scripts/gate.py --local  # developer pre-push subset
```

`--local` runs workspace clippy, `cargo fmt --check`,
`chelis lint --check .`, and per-crate nextest for the crates changed vs
`origin/main`. The full workspace nextest stage is CI-owned: open a
draft PR early and let CI (macOS Smoke is the authoritative workspace
oracle) run the full suite; see
[`docs/local_macos_environment.md`](docs/local_macos_environment.md)
for why that suite does not belong in the local loop on macOS. The gate
needs Python 3.11+ (it uses `tomllib`), which is why the examples use
`.venv/bin/python`: the stock macOS `python3` is 3.9 and fails with
`ModuleNotFoundError: No module named 'tomllib'`.

Documentation-only changes are exempt from the local gate: push and
require green CI instead (the lint stage and the Docs job cover
everything a docs-only diff can break).

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
.venv/bin/python scripts/nautilus_local_gate.py baseline
.venv/bin/python scripts/nautilus_local_gate.py tensor-grad
.venv/bin/python scripts/nautilus_local_gate.py tensor-fold
.venv/bin/python scripts/nautilus_local_gate.py eval-imports
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
- [Install and toolchain management (chelisup)](docs/book/src/install.md)
- [Reef, `reef setup`, and packages](docs/book/src/reef.md)
- [Packaging & install design](spec/design/chelis_packaging_and_install.md)
- [Canonical Project Reference](spec/design/chelis_canonical_reference.md)
- [Ecosystem Context](spec/design/chelis_ecosystem_context.md)
- [Context](spec/00-context.md)
- [Nomenclature](spec/01-nomenclature.md)
- [Project Plan](spec/design/chelis_project_plan.md)

## License

MIT
