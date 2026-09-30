# Install

There are two ways in: **use** Chelis (install the `chelis` toolchain with
`chelisup` and provision a project with `chelis reef setup`), or **build the
compiler from source** (a checkout, for compiler developers). Most users want
the first; the second is covered further down.

## Install the toolchain with chelisup

`chelisup` is Chelis's toolchain installer and version manager (the rustup
analogue). It installs side-by-side `chelis` toolchains under `~/.chelis` and a
small `chelis` **shim** that resolves the right one at every invocation. It
also owns the recorded default and uninstall.

### 1. Bootstrap chelisup

`Chelis-Lang/chelis` is private, so a public release URL does not serve its
assets. You need repository access and an authenticated
[`gh`](https://cli.github.com). This command fetches the bootstrap script
from the latest GitHub release; the script uses `gh` to install the matching
`chelisup` binary for your platform in `~/.chelis/bin`:

```sh
gh auth login    # once, if you have not already
gh release download --repo Chelis-Lang/chelis --pattern chelisup.sh --output - | sh
```

Already have a checkout of the compiler repo? Run the vendored script directly
instead:

```sh
sh crates/chelisup/bootstrap/chelisup.sh   # installs ~/.chelis/bin/chelisup
```

If the bootstrap reports that `~/.chelis/bin` is not on your PATH, add it
(the bootstrap prints the exact line):

```sh
export PATH="$HOME/.chelis/bin:$PATH"   # add to ~/.profile, ~/.bashrc, or ~/.zshrc
```

Working from a checkout instead of bootstrapping? `cargo build -p chelisup`
produces `target/debug/chelisup`.

### 2. Install a toolchain

```sh
release_tag="$(gh release view --repo Chelis-Lang/chelis --json tagName --jq .tagName)"
version="${release_tag#v}"
chelisup install "$version"
chelis --version
```

The GitHub tag starts with `v`; `chelisup install` accepts only bare `X.Y.Z`.
The first install also records that version as the default and installs the
`chelis` shim, so `chelis --version` works from anywhere. Before placing a
toolchain, chelisup checks that the sealed compiler's exported archive and all
six public runtime headers match the files shipped under `lib/` and `include/`;
a missing or crossed file refuses installation before store placement. The
receipt comes from the unpacked compiler's live export, not from the tarball.
`chelisup` also refuses a malformed or ambiguous receipt (including duplicate
JSON keys in nested objects) before placing the new toolchain. Older releases
that predate `chelis runtime export` install with a warning that their runtime
files are unchecked. If chelisup refuses a newer release because it cannot
read its format, re-run the bootstrap to update chelisup and try again.

Rebuild the compiler and its carried runtime together, then publish and install
their **matching** archive and headers as one versioned toolchain. Do not
replace just `lib/libchelis_runtime.a` or an `include/` header in an installed
version. If installing a new release fails validation, the previously installed
toolchain remains available. To roll back, select a previously installed
complete version with `chelisup default <previous>` outside a project; an
explicit `+<version>`, `CHELIS_TOOLCHAIN`, `chelis-toolchain` file or `reef.toml`
compiler pin takes precedence and must be switched separately.

### 3. Provision a project in one command

Inside a freshly-cloned project (anything with a `reef.toml`):

```sh
chelis reef setup
```

`reef setup` is the orchestrator. It reads the `reef.toml` `compiler =` pin
and, in order:

1. ensures the pinned toolchain is installed, auto-installing it through
   `chelisup` when it is missing;
2. installs source packages and binary artifacts from `reef.lock`
   (`reef install --from-lockfile`), when a lockfile is present;
3. syncs chelis source crates when the manifest has a `[chelis-src]` section
   (`reef src sync`);
4. prints a `reef doctor` health summary.

That is the whole "clone and build" story: bootstrap `chelisup` once, then
`chelis reef setup` per clone.

## Managing versions

The `chelis` shim resolves which installed toolchain answers each call, **first
match wins**:

1. a leading `+<version>` argument, such as `chelis +X.Y.Z build main.ch`;
2. the `CHELIS_TOOLCHAIN` environment variable (CI and scripting);
3. a `chelis-toolchain` file found by walking up from the cwd (one bare version
   on its first non-comment line): a deliberate directory override that ranks
   above the package pin;
4. the nearest `reef.toml` `compiler =` pin, walking up: a clone builds with
   its declared toolchain and no extra files;
5. the recorded default (`chelisup default <ver>`), used outside any package.

A resolved-but-not-installed version is a loud error naming
`chelisup install <ver>`. The shim never silently downloads a toolchain on
`cd` and never falls back to a different installed version.

chelisup's management verbs:

```sh
chelisup list-installed        # installed toolchains (marks the default)
chelisup default "$version"    # set the default used outside a package
chelisup show                  # store layout + what resolves in the cwd
chelisup which                 # print the toolchain binary chelis resolves to
chelisup self uninstall        # remove the shim + installer (keeps toolchains)
```

Use `chelisup uninstall "$version"` to remove that installed toolchain.

### Cross-version commands

Run a concrete `+<ver>` when the `reef.toml` pin would route you to a toolchain
that lacks the verb you want, for example running a newer `reef src` from a
project pinned to an older compiler:

```sh
chelis "+$version" reef src sync
```

`+latest` is deliberately unsupported (it is ambiguous between newest-installed
and newest-available); `chelisup list-installed` shows what you have. If you
hit an `unrecognized subcommand`, chelis names its own version, the pin that
routed you there, and the `+<ver>` override to try.

### Health check

```sh
chelis reef doctor --root ~    # every shell under ~, across all classes
```

`reef doctor` reports, per shell: whether the pinned toolchain is installed in
the chelisup store (or the `chelisup install` fix), source-crate drift, and
binary-artifact status. It is read-only and never installs. See
[Reef and Packages](reef.md) for the full orchestration story.

## Build from source (compiler developers)

The rest of this page builds the Chelis compiler itself from a checkout.

### Nix source packages

The root flake supports `x86_64-linux` and `aarch64-darwin`. It provides these packages:

- `chelis`
- `chelis-runtime`
- `chelisup`

Run these commands from the repository root:

```sh
nix build .#chelis
nix build .#chelis-runtime
nix build .#chelisup
nix run .#chelis -- --version
nix run .#chelisup -- --help
.venv/bin/python scripts/test_nix_flake_contract.py
nix flake check --print-build-logs
```

Nix is a source-build channel. It does not replace the release store or the version router.

The `chelis` and `chelis-runtime` derivations verify the copied archive and all
six public headers against the sealed compiler's export. `nix flake check`
rechecks both outputs against the combined package's own compiler, then links
and runs the C runtime consumer separately against each output's include
directory and exact archive path. It does not substitute for an executed check
on each hosted platform.

Rebuild both Nix outputs from the same compiler revision when changing the
carried runtime; do not copy a `chelis-runtime` archive/header set from a
different compiler into `chelis`. For a release-store rollback, reinstall or
select a previously verified complete toolchain, not an archive from an older
derivation. A failed Nix-wrapper installation leaves its previous stable GC
root and selected installed toolchain untouched.

Before an install, the Nix wrapper creates `$CHELIS_HOME/nix-gcroots/chelisup.next`. After success, it promotes `$CHELIS_HOME/nix-gcroots/chelisup`.

The stable root preserves the copied installer dependencies. A failed install preserves the prior stable root.

If a failed install copied a new binary, `$CHELIS_HOME/nix-gcroots/chelisup.partial` protects that binary.

After each successful install, the Nix wrapper restores itself at `$CHELIS_HOME/bin/chelisup`. The generic installer contains no Nix root logic.

Through the installed Nix wrapper, `chelisup self uninstall` removes all three roots after executable cleanup.

The `chelisup` release workflow owns release pins and side-by-side toolchains.

## Build the CLI from a checkout

Set up Rust, the C toolchain, cargo-nextest, and a uv-managed Python 3.11
environment using the [contributor setup guide](../../contributor_setup.md).
Use focused tests and the [local gate](../../local_gate.md) when changing the
compiler.

```sh
cargo build -p chelis-cli
target/debug/chelis --help
```

This runs the checkout build; the release toolchain and `chelisup` store are
separate.

## Build the Python distribution wheel

From the repository root, a standard wheel build seals the runtime into the
extension. Editable installs remain development builds:

```sh
uv pip install -e bindings/python
```

They use the checkout's runtime and reject changed declared runtime sources
until the extension is rebuilt. The wheel-only `sealed-runtime` feature does
not apply to editable installs or `maturin develop`.

Build a wheel directly with:

```sh
CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR="$PWD/target/python-wheel" \
uv build --wheel --out-dir target/python-wheel/wheels bindings/python
```

For end-to-end source-free acceptance, run the smoke instead of first building
the wheel. It performs one standard wheel build from a disposable source copy,
removes the copy before installed execution, and tests persisted reload in a
second Python process:

```sh
CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0 \
CARGO_TARGET_DIR="$PWD/target/python-wheel-smoke/cargo" \
.venv/bin/python bindings/python/tests/python_wheel_smoke.py \
  --receipt target/python-wheel-smoke/cargo/wheel-smoke.json
```

To avoid rebuilding an already-produced wheel, pass it with `--wheel <path>`.
That mode proves installed consumer behavior but not source-free wheel
production. `--crossed-bundle` builds a synthetic second sealed wheel with a
changed exported key-seed operation. A native C program links each staged archive
and requires seed 7 to yield key bits 7 in A and 8 in B; the first wheel must
reject B's compiled artifact. The receipt includes archive and linked-library
digests, exact results, separate compile/reload process evidence, and negative
controls. Failure logs and receipts are retained under the Cargo target
directory. This is a bounded Python distribution check, not the aggregate
runtime-artifact oracle.

On macOS the installed-wheel processes run in a filesystem sandbox denying
reads from the original checkout, and their receipts record a denied read
of its `Cargo.toml`. On Linux the copy is removed and the developer target
withheld, but the original checkout is still readable; a receipt with
`checkout_read_denied: false` is not checkout-denied evidence.
`source_free_build_proven` is true only when the smoke built the wheel and
every installed consumer verified that checkout read denial.

## Build the Book

The pinned Devenv shell includes mdBook 0.5.2, the same version used by the
hosted Docs job. From one active `devenv shell` session, run:

```sh
mdbook build docs/book
```

From outside an active session, the equivalent one-shot command is:

```sh
devenv shell -- mdbook build docs/book
```

## Validate Docs Examples

```sh
cargo test -p chelis-e2e --test skill_suite
```

That test validates the checked `chelis-surf` and `chelis-deep` fences used as teaching
examples. Fragment fences are intentionally illustrative only.
