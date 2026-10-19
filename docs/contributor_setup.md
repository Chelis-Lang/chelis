# Contributor setup

This guide is for building and changing the Chelis compiler from a checkout.
For a release toolchain or a downstream shell, use the [install guide](book/src/install.md).
Read the [maintainer guide](maintainer_guide.md) and [AGENTS.md](../AGENTS.md)
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
wheels embed their runtime; see
[Build the Python distribution wheel](#build-the-python-distribution-wheel)
and the [bindings README](../bindings/python/README.md) for wheel checks.

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

Two more tracked hooks, `.githooks/pre-commit` and `.githooks/pre-push`, are
installed the same two ways and run `scripts/regen_hooks.py` with the same
uv-managed interpreter, never a `python3` from `PATH`. Neither writes or
stages anything; they report derived artifacts that disagree with their
sources and print the command that fixes them.

- **pre-commit** runs the legs in the table at the top of the hook when a
  staged path, renamed or deleted ones included, is one of their inputs or
  outputs, such as `AGENTS.md`, `docs/CHELIS_SURFACE.md`,
  `packages/chelis-std/SKILL.md`, `agent-skills/`, or the embedded
  conformance assets themselves. It runs the generator
  on a private, always-removed copy of the staged files, so partial staging,
  `git commit <paths>`, and `git commit -a` are judged by what they commit.
  On disagreement it names both files, prints the write command, and asks
  you to `git add` the result. One leg checks instead of generating: a
  staged file under `packages/chelis-std/dist/` or
  `crates/chelis-std-bundle/dist/`, or a staged `packages/chelis-std/reef.lock`
  that records the bundled runtime, fails the commit, names the files, and
  prints the command that untracks them. It never runs cargo. git runs it for
  `commit`, `commit --amend`, `cherry-pick --continue`, and the commit that
  concludes a conflicted merge, and a refused commit leaves the operation
  resumable; `rebase --continue` and conflict-free merges, cherry-picks, and
  rebases do not run it.
- **pre-push** runs the `--check` form of each pure-Python tier-0 leg of
  `scripts/regen_all.py` whose inputs or outputs the push changes. On
  staleness it fails the push and prints the write command. It skips a leg
  with a one-line notice when the pushed commit is not this worktree's
  `HEAD` or the leg's paths have uncommitted changes. It runs only checks
  that never write into the worktree, and no hook ever runs cargo: the
  rejection registry is CI's to check, and the chelis-std bundle is built
  with the compiler and refused when committed.

On a branch that predates `scripts/regen_hooks.py`, both hooks do nothing.
Without a managed venv or `uv`, pre-commit fails, naming the fix, only when
a commit stages a leg's input or output, and pre-push prints a notice and lets the push
through. `git commit --no-verify` and `git push --no-verify` are the only
bypass.

## Build the Python distribution wheel

With the prerequisites above installed, run this from the checkout root:

```sh
uv build --wheel --out-dir target/python-wheel/wheels bindings/python
```

This builds a wheel from the checkout. The GitHub toolchain release does not
include a Python wheel. For editable bindings during development, use the
`uv pip install -e bindings/python` step under [Python 3.11](#python-311).

## Distribution builds

A distributed compiler builds `chelis-cli` with its `distribution` feature:

```sh
cargo build --release -p chelis-cli --features distribution
```

The GitHub release and the Nix `chelis` package build this feature, and a
downstream package uses it too. It enables `smt`, `sealed-runtime`, and
`clarabel-provider`. The `smt` feature links cvc5 through `cvc5-sys`, which
builds the cvc5 version that it binds unless `CVC5_DIR` names a prebuilt cvc5
tree. The [SMT build guide](smt_build_setup.md) has the prerequisites.

## Nix source packages

From the checkout root, Nix provides source builds of `chelis`,
`chelis-unwrapped`, `chelis-runtime`, and `chelisup` on `x86_64-linux`,
`aarch64-linux`, and `aarch64-darwin`. Each system has a native CI job that runs the complete check
set, and `checks.<system>.lane-check` exists on both Linux systems.

`x86_64-darwin` (Intel macOS) is not supported. nixpkgs removes that system for
26.11 (NixOS/nixpkgs#535508), and Devenv publishes no build for it.

Build and run the compiler:

```sh
nix build .#chelis
nix run .#chelis -- --version
```

In the `chelis` package, `bin/chelis` is a wrapper, and `libexec/chelis` is the
compiler. `libexec/chelis` is a symlink to the binary of the `chelis-cli` crate
build, so the store holds one copy of the compiler. The wrapper puts the C
compiler and the binutils of the Nix `stdenv` first on `PATH`, so
`chelis build` needs no host toolchain. The `chelisWrapperArgs` list in
`nix/packages.nix` holds the wrapper arguments.
The `wrappedBuild` check builds an executable and a static library with an
empty host `PATH`, and runs the executable. It also builds and runs the BLAS
fixture program `examples/recurrence_extent_scope.ch`, whose generated C
includes `chelis_blas.h`. On Linux, the build links the OpenBLAS that the
wrapper declares. The check has two negative controls:

- The empty-`PATH` control builds `hello_tensor.ch` with `libexec/chelis` and
  an empty `PATH`. The build must fail, and its error must contain
  `was not found`.
- The OpenBLAS control runs on Linux. It builds the BLAS fixture with
  `libexec/chelis` and the `PATH` of the wrapper, but without
  `CHELIS_C_INCLUDE_PATH` and `CHELIS_C_LIBRARY_PATH`. The build must fail, and
  its error must name `cblas.h` or `-lopenblas`.

`chelis-unwrapped` has the layout of a release toolchain: `bin/chelis` is the
compiler itself, beside the runtime archive and the six headers. Its
`bin/chelis` and `libexec/chelis` of `chelis` are links to one binary of the
compiler crate, and the `sharedCompiler` check compares them.

The `unwrappedClosure` check fails when the closure of `chelis-unwrapped` holds
the `stdenv` C compiler, its wrapper, an output of its binutils, or an output of
OpenBLAS. The library outputs of the C compiler, such as `libstdc++` and
`libgcc_s` on Linux, can stay. The check prints the closure size of both
compiler packages and the closure of `chelis-unwrapped`. As a control, the
closure of `chelis` must hold the `stdenv` C compiler.

The `chelis`, `chelis-unwrapped`, and `chelis-runtime` packages verify their
copied archive and all six public headers against the sealed compiler's export.
`nix flake check` rechecks the three outputs against the compiler of the
combined package. Then it links and runs a native C consumer separately with
the include directory and archive path of each output. Rebuild the outputs from
the same compiler revision. Never mix an archive or header from another derivation. A failed
Nix-wrapper installation leaves its previous stable GC root and selected
toolchain untouched.

The packages refer to no full copy of the flake source. A build script that
they run enters `packageScripts` in `nix/packages.nix`, a store path with only
those scripts. Thus an edit of the documentation keeps every package
derivation, and `scripts/test_nix_flake_contract.py` checks this.

On Linux, `nix flake check` also checks that `libexec/chelis` carries a
GNU build id. The compiler identifies its own build by that id. Without it,
each `chelis` process computes SHA-256 over the whole executable.

The flake also has `overlays.default`, which adds the four packages for the
system of the consumer, and `formatter.<system>`. `nix fmt` formats each
tracked `.nix` file except `Cargo.nix`, and `nix fmt -- --check` only checks
them. The `nixFormat` check uses the same nixfmt and the same files.

A run of `nix-packages.yml` on `main` or on a release tag pushes to the
`chelis-lang` Cachix cache. It pushes the `chelis`, `chelis-runtime`,
`chelis-unwrapped`, and `chelisup` outputs, the cvc5 tree, the Rust crate builds, and the Devenv
shell closure. A daily schedule
runs the workflow on `main`. That run skips its native builds when the cache
already holds the packages of the `main` revision.

`flake.nix` names that cache and its public signing key. When Nix trusts the
cache, a build of a pushed revision downloads those outputs. A build of a later
revision downloads the unchanged cvc5 tree and crates. It compiles only the
crates whose sources or dependencies changed.

Nix uses the cache only for a trusted user. The
[install guide](book/src/install.md#use-the-binary-cache) gives the settings
for other users, for NixOS, and for `/etc/nix/nix.conf`.

Each pull request runs the `Nix flake evaluation` CI job. It builds nothing.
It runs these two commands, which you can also run before a push:

```sh
nix flake check --no-build --all-systems --option allow-import-from-derivation false
nix fmt -- --check
```

The first command evaluates the packages and checks of every supported system.
The second command checks the format of each tracked Nix file except the
generated `Cargo.nix`. Run `nix fmt` to format the files.

The flake imports the committed `Cargo.nix`, so Nix evaluation needs no import
from derivation. A change to `Cargo.lock`, a workspace manifest, or a crate
target (for example, a file in `src/bin/`) must also regenerate that file:

```sh
nix run --inputs-from . nixpkgs#crate2nix -- generate --no-default-features --features chelis-cli/distribution
```

The `Cargo.nix freshness` CI job runs the same crate2nix 0.15.0 command and
fails on any byte difference. Resolve a `Cargo.nix` merge conflict with this
command, not by hand.

Each Nix crate build gets only its own directory, without `tests/`,
`benches/`, and `examples/`. A crate build that reads another file, outside
its directory or under one of these directories, names that path in
`crateBuildReads` in `nix/packages.nix`. An `include_str!` argument and a
build script input are examples. Without that entry, the Nix build of the
crate fails because the file is absent. No crate build reads the other test
files, so an edit to them changes no crate derivation. Nix builds the crates
with the minimal profile (rustc, cargo, and rust-std) of the Rust version in
`rust-toolchain.toml`.

The workspace crates build with the codegen units of the Cargo release
profile, 16 when `Cargo.toml` sets no value, as `cargo build --release` does.
The third-party crates keep the one codegen unit of `buildRustCrate`. An edit
of workspace source changes no third-party crate derivation, so the cache keeps
their outputs.

## Optional Devenv shell

The native path above works without Devenv. The optional shell supplies
pinned Rust, Python, C and contributor tools on `x86_64-linux`,
`aarch64-linux`, and `aarch64-darwin`. It creates `.devenv/state/venv` for Python 3.11 and does not
replace the root `.venv`.

Install Nix using its [official instructions](https://nixos.org/download/):
the macOS installer uses
`curl -sSfL https://artifacts.nixos.org/nix-installer | sh -s -- install`,
and the Linux multi-user installer uses
`sh <(curl -L https://nixos.org/nix/install) --daemon`.
Open a new shell after installation. The repository pins the Devenv
release with:

```sh
nix --extra-experimental-features 'nix-command flakes' profile install github:cachix/devenv/v2.4.0
devenv --version
```

The pinned CLI reports `2.4.0`. Do not run `devenv init`. This checkout
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
[Nix source packages](#nix-source-packages) are a separate source-build
channel.

For optional local direnv activation, install direnv and enable its hook in
your shell after installing the pinned Devenv 2.4.0. Create an ignored `.envrc`
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

Devenv's shared Kache cache requires access to the trusted Tunnet mesh. When
connected, supply the gateway's public placeholders:

```sh
export KACHE_S3_ACCESS_KEY=tunnet-anonymous
export KACHE_S3_SECRET_KEY=tunnet-anonymous
```

These are not AWS credentials. Without mesh access, remote cache requests
fall back to local compilation. For an offline shell, set `KACHE_CONFIG` to
a separate config with `[cache] local_only = true` after activation. The
[manual gates](manual_gates.md) document the Devenv and Kache smoke tests.

## OpenSpec planning

Planning documents live in the separate `Chelis-Lang/openspec` repository.
Register its checkout once before validating a plan; the local `openspec/`
directory only declares and pins that store:

```sh
git clone git@github.com:Chelis-Lang/openspec.git /absolute/path/to/openspec
openspec store register /absolute/path/to/openspec --id chelis-plans --json
openspec doctor --json
```

`openspec doctor` must identify `chelis-plans` as a declared store. Author
planning changes in a dedicated worktree of that store, and validate there
with `openspec validate chelis-<change> --strict --no-interactive`. The
consumer's `openspec/store.lock.yaml` pins the revision used by CI; registration
alone does not enforce it. The [agent contract](../AGENTS.md#openspec) owns
the review order and document-only acceptance boundary.
