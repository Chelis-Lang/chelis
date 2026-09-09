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

The rest of this README builds the Chelis compiler from a checkout. There are
two ways to get the toolchain: the primary path below (rustup, a C toolchain,
and uv-managed Python) and an optional Devenv shell, described at the end of
this section.

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
- **Valgrind:** memory leak tests on generated C

### Python and the gate

Chelis needs a uv-managed Python 3.11 for `scripts/gate.py`, every other script
under `scripts/`, the chelis-tools CLI in `py/`, and the `chelis-python` PyO3
link step. The project pins Python 3.11 (`py/pyproject.toml` requires
`>=3.11`). Using the system Python is **not supported**: Apple's bundled Python
reports a stale library path, and a Linux distribution Python can violate the
version contract.

Install [uv](https://docs.astral.sh/uv/) once, provision the interpreter, and
create a `.venv` at the root of every checkout and every dedicated worktree.
Never copy or symlink another checkout's `.venv`:

```sh
# Install uv once
curl -LsSf https://astral.sh/uv/install.sh | sh

# Start a new shell if the installer changed PATH, then verify the install
uv --version

# Install the project Python, then create this checkout's local venv
uv python install 3.11
uv venv --python 3.11
```

Three invocation forms follow from that setup, and the repository uses them
consistently:

- **Scripts:** `.venv/bin/python scripts/<name>.py`, for example
  `.venv/bin/python scripts/regen_all.py --check`. Inside an activated Devenv
  shell the equivalent is `python scripts/<name>.py`.
- **The gate:** `python3 scripts/gate.py --fast`, `python3 scripts/gate.py
  --local`, or `python3 scripts/gate.py --list`, in every environment.
  `scripts/gate.py` is stdlib-only; when `python3` is not already a uv- or
  Devenv-managed runtime it re-executes itself as
  `uv run --managed-python --python 3.11 --no-project python scripts/gate.py`
  and then exports its selected interpreter as `PYO3_PYTHON` to every child
  command. It never requires a checkout-local `.venv`, and its `--fast` and
  `--local` preflight warns when one is missing (create it with
  `uv venv --python 3.11`), because direct cargo and nextest invocations
  outside the gate fall back to it. Do not invoke the gate through
  `.venv/bin/python`; the `uv run` form above is the gate's own fallback, not a
  routine invocation.
- **Direct cargo commands:** `.cargo/config.toml` defaults `PYO3_PYTHON` to
  `.venv/bin/python`, so a checkout with the environment above builds as is. A
  worktree without its own `.venv` points PyO3 at uv's managed interpreter
  instead:

  ```sh
  PYO3_PYTHON="$(uv python find 3.11)" cargo build --workspace
  PYO3_PYTHON="$(uv python find 3.11)" cargo nextest run -p chelis-compiler-api
  ```

  An explicit `PYO3_PYTHON` is authoritative. A missing configured path fails
  with setup guidance instead of silently falling back.

Install Python dependencies into the venv as needed:

```sh
uv pip install -e py            # chelis-tools (loc-report, skill-eval, ...)
uv pip install -e bindings/python # chelis Python bindings (optional)
```

**Commit-message hook.** `.githooks/commit-msg` is the tracked commit-msg hook.
It runs `scripts/check_commit_message.py` through Devenv, `.venv`, or a managed
uv interpreter, in that order, and resolves its repository at run time, so one
installed copy is correct from every worktree of a clone and on every branch.
On the primary path, cargo-husky installs it: `cargo test` installs a POSIX
wrapper that invokes the same checker, using the same uv fallback when a
worktree has neither Devenv nor `.venv`. Both hooks reject AI tool authorship
markers before Git creates a commit. All listed format and lint hooks remain
disabled.

### Devenv development shell (optional)

Devenv is optional. It supplies pinned Rust, Python, C, and contributor tools in
one shell, and native Nix CI requires it, but every command pays shell
evaluation and activation unless it runs inside one persistent shell, which in
practice makes it slower per command than the primary path above; an agent
fleet should use the primary path. Nothing in this subsection is needed to
build, test, or gate Chelis.

The tracked environment supports `x86_64-linux` and Apple silicon macOS
(`aarch64-darwin`). Other systems use the primary path.

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
   nix --extra-experimental-features 'nix-command flakes' profile install github:cachix/devenv/v2.2.2
   ```

2. Make sure that Devenv reports version `2.2.2`:

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

5. Run the gates through `chelis-gate`, which forwards every argument to
   `python3 scripts/gate.py`: `--fast` before every push. CI validates the pushed
   candidate; `--local` is available for optional troubleshooting:

   ```sh
   chelis-gate --fast
   chelis-gate --local  # optional
   ```

Use `devenv shell --` to run one command without an interactive shell:

```sh
# Run the authoritative C-backend acceptance oracle.
devenv shell -- cargo nextest run -p chelis-backend-c

# List the repository gate commands.
devenv shell -- chelis-gate --list

# List orphaned Chelis build processes.
devenv shell -- chelis-reap-orphans
```

For repeated commands, enter one persistent `devenv shell` and run every
command inside it. The required entry checks run before the session starts,
then the exact activated toolchain is reused without paying shell evaluation
and activation for every command. After one successful realization and while
the Devenv files are unchanged, a single noninteractive command may use
`devenv shell --no-reload -- <command>`.

Measure the ordinary, no-reload, task, activation, and persistent-session costs
with:

```sh
.devenv/state/venv/bin/python scripts/devenv_startup_benchmark.py --samples 5
```

#### Shell behavior

The repository pins the Devenv modules and update target to release `v2.2.2`.
Use Devenv 2.2.2 locally when possible. The closed, reviewed CLI range is
2.2.0 through 2.2.2 so the immutable shared CI action remains supported while
the repository-owned atomic load-export task removes the concurrency race in
those versions. CLIs outside that range are rejected before Nix evaluation.

The upstream `v2.2.2` tag builds a 2.2.2 CLI but its module metadata still
advertises 2.2.1. Chelis overrides that stale metadata to 2.2.2 so every
accepted CLI reports the right update target.

`devenv.nix` imports six local configuration modules. `devenv.yaml` defines
the inputs and CLI options.

`devenv.yaml` pins the shared `nixpkgs` and `rust-overlay` inputs to exact
revisions. Therefore, `devenv update` cannot change them.
`scripts/check_nix_lock_parity.py` keeps these revisions aligned with
`flake.lock`.

On macOS, the `gcc` and `g++` shims invoke the Nixpkgs clang wrapper from
`pkgs.stdenv.cc`. They do not invoke host Apple clang.
Target-specific tree-sitter build scripts use the pinned, SDK-aware Nixpkgs
compiler wrapper while `CRATE_CC_NO_DEFAULTS=1` prevents cc-rs from injecting
the redundant `arm64-apple-macosx` native-target alias. This keeps the wrapper's
SDK and C++ headers without the false cross-target warning.

On Linux, the shell supplies GCC, OpenBLAS, and Valgrind from Nixpkgs.

Devenv creates and activates Python 3.11 at `.devenv/state/venv`. It sets
`PYO3_PYTHON` to that interpreter. The shell also supplies mdBook 0.5.2 and
Pyright; `pyrightconfig.json` owns the editor and CLI analysis roots and keeps
generated environment/build trees out of the source graph.

The shell builds and pins Kache 0.16.0, sets it as the exact `RUSTC_WRAPPER`,
and reads the repository-owned `.kache.toml`. That policy caches test and CLI
executables, ignores file-backed machine `KACHE_*` overrides, and retains a
no-Kache correctness control. Ordinary Cargo commands outside Devenv may still
honor the contributor's Cargo configuration.

Chelis's relocatable macOS executable/dSYM representation uses Kache cache-key
schema 28. Existing schema-27 entries deliberately take one cold miss instead
of restoring pre-sanitization paths. The executable-cache regression first
proves that upgrade miss with the managed schema-27 fixture, then proves that
schema-28 clean checkouts restore exact path-clean executable/dSYM pairs.

Devenv does not modify the repository-root `.venv`; the primary path above owns
that environment outside Devenv.

`devenv test` initializes the managed files and Python. Explicit task-graph
edges keep Python-backed checks behind the virtual environment and compiler
checks behind the generated probes. It then runs separate smoke tasks for the
toolchain, Python, C, C++, Kache, Pyright, docs, and the Darwin tree-sitter
compiler/parser contract.

Shell entry publishes `.devenv/load-exports` by atomic rename. The committed
concurrency regression is:

```sh
.devenv/state/venv/bin/python scripts/devenv_entry_regression.py --workers 8
```

Every entry must either complete its entry graph and run the payload, or fail
without running the payload.

The shell also supplies these platform commands:

- `chelis-exec-preflight` on macOS
- `chelis-z3-test` on Linux
- `chelis-hip-test` on Linux

Each command forwards its arguments to the applicable Python file under
`scripts/`.

Devenv copies the tracked `.githooks/commit-msg` hook into the shared hooks
directory on shell entry; see the commit-message hook paragraph in the primary
path above for what the hook runs.

#### Native Nix CI

Both jobs for native Nix packages use the reviewed portable Devenv action from
`Chelis-Lang/ci`. The jobs run their tasks through the portable shell.

The action uses exact Nix and Devenv inputs. Its public Devenv cache is
read-only. The cache does not contain the custom cvc5 derivation.

Each job stores the prebuilt cvc5 toolchain closure in the GitHub Actions cache.
The derivation name identifies the cache entry.

Routine pull-request and push CI does not run either native Nix job. A manual
dispatch or a published GitHub release runs both the Linux and macOS jobs. The
Linux job first removes unused preinstalled toolchains and limits Nix to two
concurrent builds.

Run this command to dispatch both jobs for a branch:

```sh
gh workflow run "Nix Packages" --ref <branch>
```

When both native Nix jobs complete with all checks green, the gate passes.

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

## Build

`chelis-python` links against `libpython`, so a managed Python is a prerequisite
for `cargo build` on every platform: see Python and the gate above for the
`.venv` and `PYO3_PYTHON` forms. Inside Devenv the activated `.devenv/state/venv`
environment satisfies the same requirement.

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

`--fast` is the pre-push gate: fix-in-place, run before every push. `--local`
(chelis#360) is optional for troubleshooting or additional local validation. Applicable
CI checks must pass on the pushed candidate before marking the draft ready for review;
no per-PR `--local` run is required. `scripts/gate.py` defines the commands shared with
the CI gate stages:

```sh
python3 scripts/gate.py --list  # re-executes through uv when needed
python3 scripts/gate.py --fast  # before every push; fixes fmt and tier-0 regeneration in place
python3 scripts/gate.py --local # optional troubleshooting and local validation
```

`--fast` regenerates the tier-0 artifacts (`scripts/regen_all.py --tier 0`) and
runs `cargo fmt --all` in write mode, then `chelis lint --check .`,
`cargo clippy -p <crate> --tests` for each changed crate, and one nextest run
over the drift tripwires; it prints the files it changed and exits non-zero
only when a check fails. `--local` runs two workspace clippy configurations,
`cargo fmt --check`, `chelis lint --check .`, the regeneration and compile-fail
guards, both oracles, and per-crate nextest for the crates changed vs
`origin/main`; it takes an advisory workstation-wide lease on `gate.lock` under
`$CHELIS_GATE_LEASE_DIR`, else `$XDG_CACHE_HOME/chelis`, else `~/.cache/chelis`,
so two full gates in different worktrees do not run at once (`--no-wait`,
`--lease-timeout SECONDS`, and `--no-lease` change that). Every run other than
`--list` writes a JSON summary under `target/gate-reports/` and prints one
summary line. Push before the review round; the round reviews the pushed
head while CI runs, and CI is watched by one background waiter, never a
polling loop. The full workspace nextest stage is CI-owned: open a
draft PR early and let CI (macOS Smoke is the authoritative workspace
oracle) run the full suite; see
[`docs/local_macos_environment.md`](docs/local_macos_environment.md)
for why that suite does not belong in the local loop on macOS.

Child stdout and stderr stream live. On failure, the gate prints the stage and
command index, duration, exit code or signal, host and toolchain environment,
the exact rerun command, and the final 200 output lines. The complete combined
transcript is retained under `target/gate-failures/`; successful-command
transcripts are deleted.

Gate Cargo artifacts are forced into the current worktree. An inherited
`CARGO_TARGET_DIR` that resolves outside it is rejected, and cargo-husky's
build-time hook installer is disabled for gate children so sibling worktrees
cannot overwrite shared Git hook state. Every nextest profile runs without
fail-fast, allowing one run to report all failing tests. Parallel worktrees may
run more slowly from CPU contention, but they do not share writable build
artifacts.

Prose-only documentation changes are exempt from the local gate: run their focused
documentation checks, push, and require green CI. Markdown consumed structurally by
tools is not inert prose; shared `agent-skills/*/SKILL.md` and command-wrapper changes
still require skill-schema validation, the conformance-asset regeneration check,
live/embedded and Claude/Codex byte comparisons, and the focused
`chelis-conformance` asset/uniformity tests documented in `AGENTS.md`. The hosted
docs-only classification is routing evidence, not proof that every Markdown control
artifact has an owning validator in the always-run Docs job.

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

## Submitting an OpenSpec document change

**Just push the branch.** A push that touches `openspec/**` on any branch
other than `main` starts `openspec-autoland`: it classifies the pushed
commit, opens an internal pull request when every changed path is an
OpenSpec document, waits for the required checks on that exact commit, and
merges it. Nothing local is required and no human approval is involved.

```sh
git switch -c openspec/add-thing
# edit openspec/** only
git commit -am "docs(openspec): add the thing"
git push -u origin HEAD
```

A push that touches anything outside the OpenSpec document set is
classified `review`, nothing is written, and the change follows the
ordinary path. Normative `openspec/specs/**` text is inside the document
set, by explicit maintainer authorization.

`openspec-submit` remains available inside Devenv as an optional local
helper -- it validates before pushing and reports the outcome in your
terminal -- but it is no longer how a change lands:

```sh
openspec-submit --dry-run       # print the plan; write nothing
openspec-submit                 # submit and wait for accepted or blocked
```

### One-time activation

Autoland is inert until two things are true. Both are maintainer actions
outside any automated session.

**1. The workflow files must be on `main`.** `workflow_run` and
`pull_request_target` only take effect from the default branch, so nothing
runs until this change set lands there.

**2. The existing GitHub App installation must cover this repository.**

A pull request opened with the built-in `GITHUB_TOKEN` raises no
`pull_request` event, so the workflows publishing the required status
checks never start and the pull request could never go green. Measured
against this repository: `conformance.yml` runs only on `push: [main]` and
`pull_request`, and `changelog.yml` has only `pull_request`, so
`Hull Conformance Gate (Linux)` and `Changelog` can never appear.

The controller therefore opens the pull request with a short-lived
installation token, minted per run from **`chelis-openspec`**, a GitHub App
dedicated to this mechanism -- `vars.OPENSPEC_APP_ID` and
`secrets.OPENSPEC_APP_PRIVATE_KEY`. No token is stored:
`actions/create-github-app-token` revokes it when the job ends.

It is deliberately **not** the shared `CI_APP_*` App that
`conformance-nightly.yml` and `ecosystem-drift.yml` use. Minting from that
one returned HTTP 422, `The permissions requested are not granted to this
installation`: its installation grants no pull-request write, and it exists
for cross-repo reads. Widening it would give pull-request write to every
workflow that already holds its key in order to fix one call in this one.
The dedicated App's installation grants `contents: read`, `metadata: read`,
and `pull requests: write`, and only the controller holds its key.

The token is requested as narrowly as the action allows:

| Scope | Value |
|---|---|
| `owner` | `${{ github.repository_owner }}` |
| `repositories` | `${{ github.event.repository.name }}` (this repository only) |
| Permission | `permission-pull-requests: write` and `permission-contents: read`, and nothing else |
| Revocation | automatic at job end (`skip-token-revoke` deliberately unset) |

`POST /repos/{owner}/{repo}/pulls` needs `Pull requests: write` to perform
the write and `Contents: read` to resolve the head and base refs. A token
without the second cannot see the branch and the call answers
`422 Validation Failed` on `head` -- measured, on the third hosted run. The
endpoint is called directly rather than through `gh pr create`, which would
additionally read repository and branch metadata. The **Workflows**
permission is *not* needed: it governs writing repository content, and this
credential never pushes.

Both `vars.OPENSPEC_APP_ID` and `secrets.OPENSPEC_APP_PRIVATE_KEY` are
configured, and the App's installation grants `Pull requests: write` on
this repository. If that grant is ever removed, the mint step fails with
HTTP 422 and the controller writes nothing -- it never falls back to
`GITHUB_TOKEN`.

There is deliberately **no fallback**. A missing or insufficient credential
is reported, not worked around.

**Other current blockers.** A branch whose
`.github`, `scripts`, Devenv, Nix, or toolchain content differs from
`origin/main` is refused; rebase first.

Outside Devenv, run `python3 scripts/openspec_submit.py` with a managed
Python (see [Python and the gate](#python-and-the-gate)).

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
