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

### Devenv development shell

Devenv is optional for local work. Native Nix CI requires Devenv. The shell
supplies pinned Rust, rust-analyzer, Python, C, and contributor tools.

The tracked environment supports `x86_64-linux` and Apple silicon macOS
(`aarch64-darwin`). Other systems must use the manual setup below.

These installation commands come from the
[Devenv getting-started guide](https://devenv.sh/getting-started/).

#### Install Nix on macOS

The macOS environment requires Apple silicon.

1. Install Nix with the official Nix installer:

   ```sh
   curl -sSfL https://artifacts.nixos.org/nix-installer | sh -s -- install
   ```

2. When the installation is complete, open a new terminal.

#### Install Nix on Linux

The Linux environment requires an x86_64 system.

1. Install Nix in multi-user mode:

   ```sh
   sh <(curl -L https://nixos.org/nix/install) --daemon
   ```

2. When the installation is complete, open a new login shell.

#### Install Devenv

After Nix is available, run these steps.

1. Install the Devenv release that this repository pins:

   ```sh
   nix --extra-experimental-features 'nix-command flakes' profile install github:cachix/devenv/v2.2
   ```

2. Make sure that Devenv reports version `2.2.0`:

   ```sh
   devenv --version
   ```

If Devenv reports a Bash evaluation error on macOS, install a current Bash
version:

```sh
nix-env --install --attr bashInteractive -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable
```

Do not run `devenv init` because this repository already contains the required
Devenv files.

#### Use the shell

1. From the repository root, run the environment smoke test:

   ```sh
   devenv test
   ```

2. Enter the interactive shell:

   ```sh
   devenv shell
   ```

3. Inside the shell, build the workspace:

   ```sh
   cargo build --workspace
   ```

4. Run the Chelis command from the workspace:

   ```sh
   cargo run -p chelis-cli --bin chelis -- --help
   ```

5. Before a code push, run the local gate:

   ```sh
   chelis-gate --local
   ```

Use `devenv shell --` to run one command without an interactive shell:

```sh
# Run the authoritative C-backend acceptance oracle.
devenv shell -- cargo nextest run -p chelis-backend-c

# List the repository gate commands.
devenv shell -- chelis-gate --list

# List orphaned Chelis build processes.
devenv shell -- chelis-reap-orphans

# Validate the active OpenSpec tree.
devenv shell -- openspec validate --all --strict --no-interactive
```

#### Shell behavior

The repository pins the Devenv modules to release `v2.2`. The version of the
local Devenv CLI must match this module version.

`devenv.nix` imports six local configuration modules. `devenv.yaml` defines
the inputs and CLI options.

`devenv.yaml` pins the shared `nixpkgs` and `rust-overlay` inputs to exact
revisions. Therefore, `devenv update` cannot change them.
`scripts/check_nix_lock_parity.py` keeps these revisions aligned with
`flake.lock`.

On macOS, the `gcc` and `g++` shims invoke the Nixpkgs clang wrapper from
`pkgs.stdenv.cc`. They do not invoke host Apple clang.

On Linux, the shell supplies GCC, OpenBLAS, and Valgrind from Nixpkgs.

Devenv creates and activates Python 3.11 at `.devenv/state/venv`. It sets
`PYO3_PYTHON` to that interpreter.

Devenv supplies OpenSpec 1.6.0 for local structural validation.

Devenv does not modify the repository-root `.venv`. The manual setup path below
owns that environment outside Devenv.

`devenv test` initializes the managed files and Python. It then runs separate
smoke tasks for the toolchain, Python, C, and C++.

Devenv exposes the canonical root flake packages as build outputs. Build one
output with its full attribute name:

```sh
devenv build outputs.chelis
devenv build outputs.chelis-runtime
devenv build outputs.chelisup
```

Run `devenv build` without an attribute to build all outputs, including the
`default` alias. Each output uses the same derivation as its `nix build`
counterpart.

These builds use the local Git input. They include changes to tracked files.
They exclude untracked files and ignored directories.

The shell also supplies these platform commands:

- `chelis-exec-preflight` on macOS
- `chelis-z3-test` on Linux
- `chelis-hip-test` on Linux

Each command forwards its arguments to the applicable Python file under
`scripts/`.

Devenv installs the `no-ai-authorship` hook at the `commit-msg` stage. The hook
runs `scripts/check_commit_message.py`.

The manual setup retains cargo-husky as a fallback. `cargo test` installs a
POSIX wrapper that invokes the same checker with `.venv/bin/python`.

Both hooks reject AI tool authorship markers before Git creates a commit. All
listed format and lint hooks remain disabled.

#### Native Nix CI

Both jobs for native Nix packages use the reviewed portable Devenv action from
`Chelis-Lang/ci`. The jobs run their tasks through the portable shell.

The action uses exact Nix and Devenv inputs. Its public Devenv cache is
read-only. The cache does not contain the custom cvc5 derivation.

Each job stores the prebuilt cvc5 toolchain closure in the GitHub Actions cache.
The derivation name identifies the cache entry.

The Linux job runs for each code pull request and each push to `main`. The
shared detector skips documentation-only pull requests and reports success.
The job first removes unused preinstalled toolchains. It limits Nix to two
concurrent builds.

The macOS job is a manual gate. Default CI does not run this job.

Run this command to dispatch the macOS job:

```sh
gh workflow run "Nix Packages" --ref <branch>
```

When the `Nix Packages (aarch64-darwin)` job completes with all checks green,
the gate passes.

Devenv is not a product requirement.

If you do not use Devenv locally, use the manual setup below.

### Nix source packages

The root flake provides locked source packages for these native systems:

- `x86_64-linux`
- `aarch64-darwin`

Build a package from the repository root:

```sh
nix build .                 # default package, identical to .#chelis
nix build .#default         # explicit default alias
nix build .#chelis          # compiler, runtime library, and five public headers
nix build .#chelis-runtime  # runtime static library and five public headers
nix build .#chelisup        # installer command and its internal Nix payload
```

The equivalent Devenv outputs reuse these package derivations. Use the
`devenv build outputs.<name>` commands in the Devenv section.

Run an application from the repository root:

```sh
nix run . -- --version              # default application, identical to .#chelis
nix run .#default -- --version       # explicit default application alias
nix run .#chelis -- --version        # packaged compiler
nix run .#chelisup -- --help         # packaged installer
nix run .#chelisup -- install 0.17.1 # install one release toolchain
```

Before an install, the Nix `chelisup` wrapper creates `$CHELIS_HOME/nix-gcroots/chelisup.next`. After success, it promotes `$CHELIS_HOME/nix-gcroots/chelisup`.

The stable root keeps the copied installer dependencies available after Nix garbage collection. A failed install preserves the prior stable root.

If a failed install copied a new binary, `$CHELIS_HOME/nix-gcroots/chelisup.partial` protects that binary.

After each successful install, the Nix wrapper restores itself at `$CHELIS_HOME/bin/chelisup`. The generic installer contains no Nix root logic.

Through the installed Nix wrapper, `chelisup self uninstall` removes all three roots after executable cleanup.

Run the complete check set for the native system:

```sh
.venv/bin/python scripts/test_nix_flake_contract.py
nix flake check --print-build-logs
```

Nix is an additive source-build channel. It does not create the version store that release toolchains use.

`chelisup` remains the release installer and version router. Use `chelisup` when a project needs release pins or side-by-side toolchains.

The flake does not export internal crates, the Python extension, `chelis-std`, or documentation as separate packages.

The Rust packages use an automatic graph from crate2nix 0.15.0. The graph gives each Rust crate a separate Nix derivation.

Nix generates the graph from the Cargo workspace through import from derivation. Nix fetchers prepare dependencies before Cargo runs in offline mode.

The repository does not track `Cargo.nix`.

A Cargo input change gives the generator a new derivation identity. No graph refresh command is necessary.

The first evaluation builds the pinned generator before Nix schedules crate builds. Later evaluations can reuse the generated graph from the Nix store.

Each native CI job evaluates its matching graph. A foreign-system package evaluation requires a compatible remote builder.

To bump the shared Nix pins, pick a `cachix/devenv-nixpkgs` revision and read its locked inner `NixOS/nixpkgs` revision. Set that inner revision in `flake.nix` and the outer revision in `devenv.yaml`. Keep the `rust-overlay` revision identical in both files.

Then refresh both lock files and verify the result:

```sh
nix flake lock
devenv update
.venv/bin/python scripts/check_nix_lock_parity.py
.venv/bin/python scripts/test_devenv_version.py
```

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
Outside Devenv, always use the uv-managed interpreter at `.venv/bin/python`.
`.cargo/config.toml` sets `PYO3_PYTHON` to that manual environment.

Inside Devenv, use the activated environment at `.devenv/state/venv`.
Devenv overrides `PYO3_PYTHON` with its managed interpreter.

Using the system Python is **not supported**. Apple's bundled Python reports a stale library path, and Linux Python can violate the version contract.

Install Python dependencies into the uv venv as needed:

```sh
uv pip install -e py            # chelis-tools (loc-report, skill-eval, ...)
uv pip install -e bindings/python # chelis Python bindings (optional)
```

`scripts/gate.py` itself is stdlib-only: it needs the 3.11+ interpreter
but no pip installs.

## Build

Outside Devenv, the root uv environment is a hard prerequisite for `cargo build` on every platform.

Inside Devenv, the activated `.devenv/state/venv` environment satisfies the same PyO3 requirement.

`chelis-python` links against `libpython`. Run `uv venv --python 3.11` before a manual build if `.venv/` does not exist.

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
.venv/bin/python scripts/gate.py --list   # manual environment
.venv/bin/python scripts/gate.py --local  # manual environment
chelis-gate --local                       # active Devenv shell
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
- [Roadmap](spec/12-roadmap.md)
- [Project Plan](spec/design/chelis_project_plan.md)

## License

MIT
