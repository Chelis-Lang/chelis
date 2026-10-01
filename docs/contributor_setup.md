# Contributor setup

This guide is for building and changing the Chelis compiler from a checkout.
For a release toolchain or a downstream shell, use the [install guide](book/src/install.md).
Read [CONTRIBUTING.md](../CONTRIBUTING.md) and [AGENTS.md](../AGENTS.md)
before preparing a change.

## Native toolchain

Install [rustup](https://rustup.rs), then let `rust-toolchain.toml` select
the repository's Rust toolchain and its rustfmt and clippy components.
Install the test runner used by the gate and CI:

```sh
cargo install cargo-nextest --locked
```

The C backend needs a C compiler and a BLAS provider. On Linux, install
GCC and OpenBLAS; Valgrind is useful for the optional leak tests:

```sh
# Fedora / RHEL
sudo dnf install gcc openblas-devel valgrind
# Ubuntu / Debian
sudo apt-get install gcc libopenblas-dev valgrind
```

On Apple Silicon, install the command-line tools with
`xcode-select --install`. Apple clang and Accelerate are the default C
path. Homebrew GCC (`brew install gcc`) is optional for OpenMP-enabled
CPU loops. Backend-specific hardware setup is in the
[HIP](local_hip_environment.md) and [macOS](local_macos_environment.md)
runbooks.

## Python 3.11

Every checkout and dedicated worktree needs its own uv-managed Python 3.11
environment. The compiler workspace includes PyO3, and the scripts use this
interpreter. A system Python or another checkout's environment is not a
substitute.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
uv venv --python 3.11
```

Run scripts with `.venv/bin/python scripts/<name>.py`. The gate is the
exception: launch it with `python3 scripts/gate.py --fast`; it selects the
checkout's interpreter and passes it to its children. Direct Cargo commands
use `.venv/bin/python` through `.cargo/config.toml`. An explicit
`PYO3_PYTHON` wins if a different managed interpreter is needed; for example,
`PYO3_PYTHON="$(uv python find 3.11)" cargo check -p chelis-cli`. The
[local gate guide](local_gate.md) owns the full resolution rules.

Install Python packages into this checkout's environment only when needed:

```sh
uv pip install -e py
uv pip install -e bindings/python
```

The second command makes an editable Python binding for development. Standard
wheels embed their runtime; see the [install guide](book/src/install.md#build-the-python-distribution-wheel)
and [bindings README](../bindings/python/README.md) for wheel checks.

## Build and check

```sh
cargo build -p chelis-cli
target/debug/chelis --help
python3 scripts/gate.py --list
python3 scripts/gate.py --fast
```

Use focused `cargo check -p <crate> --tests` and
`cargo nextest run -p <crate> --test <file>` while editing. The fast gate
is required before a push that changes code; hosted CI owns routine broader
validation. See the [local gate](local_gate.md) and
[CI validation](ci_validation.md) guides for coverage and manual dispatches.

Development compiler and Python builds stage the runtime they were built
with. After changing declared runtime sources or `Cargo.lock`, rebuild
before running `chelis build`; the freshness check names changed files.

The tracked `.githooks/commit-msg` hook runs
`scripts/check_commit_message.py` to reject AI authorship markers.
`cargo-husky` installs a wrapper for it when tests run, and Devenv also
installs it on shell entry. Do not set `core.hooksPath`: that setting is
shared across worktrees while the tracked hook belongs to a branch. The
hook finds this checkout's managed Python or uses uv.

## Optional Devenv shell

The native path above works without Devenv. The optional shell supplies
pinned Rust, Python, C and contributor tools on `x86_64-linux` and
`aarch64-darwin`. It creates `.devenv/state/venv` for Python 3.11 and does not
replace the root `.venv`.

Install Nix using its [official instructions](https://nixos.org/download/):
the macOS installer uses
`curl -sSfL https://artifacts.nixos.org/nix-installer | sh -s -- install`,
and the Linux multi-user installer uses
`sh <(curl -L https://nixos.org/nix/install) --daemon`.
Open a new shell after installation. The repository pins the Devenv
release with:

```sh
nix --extra-experimental-features 'nix-command flakes' profile install github:cachix/devenv/360b5eb1397291383d10845a63a0247981bd5598
devenv --version
```

The pinned CLI reports `2.2.3`. Do not run `devenv init`; this checkout
already has its configuration. If Devenv reports a Bash evaluation error
on macOS, install a current Bash with
`nix-env --install --attr bashInteractive -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable`.
From the repository root:

```sh
devenv test
devenv shell
```

Inside a persistent shell, use `cargo build -p chelis-cli` and
`chelis-gate --fast`. For one command outside it, use
`devenv shell -- <command>`. The shell includes mdBook; run
`mdbook build docs/book` to build the user book. The
[Nix source packages](book/src/install.md#nix-source-packages) are a separate
source-build channel.

For optional local direnv activation, install direnv and enable its hook in
your shell after installing the pinned Devenv 2.2.3. Create an ignored `.envrc`
in each worktree with:

```sh
eval "$(devenv direnvrc)"
use devenv
```

Review the file before running `direnv allow`. For example,
`direnv exec . chelis-gate --list` uses that worktree's environment.
`/.envrc` and `/.direnv/` are ignored; hosted CI and downstream conformance
audits do not depend on direnv. The native toolchain above remains available.

Devenv also provides `chelis-reap-orphans` for stale build processes,
`chelis-exec-preflight` for the macOS executable preflight,
`chelis-z3-test` for the Linux solver gate, and `chelis-hip-test` for the
Linux HIP manual gate.

The tracked `.kache.toml` configures Kache, Devenv's compiler cache, as a local
cache: it names no remote store, and its `ignore_env` setting stops environment
variables from adding one. The [manual gates](manual_gates.md) document the
Devenv and Kache smoke tests.
