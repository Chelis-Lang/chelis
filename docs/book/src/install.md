# Install

There are two ways in: **use** Chelis (install the `chelis` toolchain with
`chelisup` and provision a project with `chelis reef setup`), or **build the
compiler from source** (a checkout, for compiler developers). Most users want
the first; the second is covered further down.

## Install the toolchain with chelisup

`chelisup` is Chelis's toolchain installer and version manager (the rustup
analogue). It installs side-by-side `chelis` toolchains under `~/.chelis` and a
small `chelis` **shim** that resolves the right one at every invocation. It
also owns upgrades, the recorded default, and uninstall.

### 1. Bootstrap chelisup

Once chelis releases are public, one line drops the prebuilt `chelisup` at
`~/.chelis/bin/chelisup` and prints the PATH line to add:

```sh
curl -fsSL https://<host>/chelisup.sh | sh
```

During the private-repo pre-launch era the public release URL does not serve
asset bytes, so the bootstrap needs an authenticated
[`gh`](https://cli.github.com). Run the vendored bootstrap script, which uses
`gh release download`:

```sh
gh auth login                              # once, if you have not already
sh crates/chelisup/bootstrap/chelisup.sh   # installs ~/.chelis/bin/chelisup
```

Then put `~/.chelis/bin` on your PATH (the bootstrap prints the exact line):

```sh
export PATH="$HOME/.chelis/bin:$PATH"   # add to ~/.profile, ~/.bashrc, or ~/.zshrc
```

Working from a checkout instead of bootstrapping? `cargo build -p chelisup`
produces `target/debug/chelisup`.

### 2. Install a toolchain

```sh
chelisup install 0.12.1        # into ~/.chelis/toolchains/0.12.1/
```

The first install also records `0.12.1` as the default and installs the
`chelis` shim, so `chelis --version` works from anywhere.

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

1. a leading `+<ver>` argument: `chelis +0.13.0 build main.ch`;
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
chelisup default 0.12.1        # set the default used outside a package
chelisup show                  # store layout + what resolves in the cwd
chelisup which                 # print the toolchain binary chelis resolves to
chelisup uninstall 0.11.0      # remove a toolchain
chelisup self uninstall        # remove the shim + installer (keeps toolchains)
```

### Cross-version commands

Run a concrete `+<ver>` when the `reef.toml` pin would route you to a toolchain
that lacks the verb you want, for example running a newer `reef src` from a
project pinned to an older compiler:

```sh
chelis +0.13.0 reef src sync
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

## Rust Toolchain

Chelis is built with stable Rust:

```sh
rustup default stable
rustup component add rustfmt clippy
```

## C Toolchain

The C backend expects a native compiler and a BLAS provider for matmul fast paths.

Fedora / RHEL:

```sh
sudo dnf install gcc openblas-devel valgrind
```

Ubuntu / Debian:

```sh
sudo apt-get install gcc libopenblas-dev valgrind
```

macOS:

```sh
xcode-select --install
```

Supported macOS paths:

- Default: Apple clang + Accelerate. This is the supported correctness path on Apple Silicon and does not require Homebrew OpenBLAS.
- Optional: Homebrew GCC for OpenMP-enabled CPU loops.

```sh
brew install gcc
```

## Build and Test the Repo

From a Chelis checkout:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

During normal downstream work, you rarely need the whole gate above. Use it when you are
checking a local compiler build or preparing a change to Chelis itself.

## Build the CLI

```sh
cargo build -p chelis-cli
export PATH="$PWD/target/debug:$PATH"
chelis --help
```

If you install a release build instead, make sure `chelis --help` works before starting
the first program loop.

## Build the Book

```sh
mdbook build docs/book
```

## Validate Docs Examples

```sh
cargo test -p chelis-e2e --test skill_suite
```

That test validates the checked `chelis-surf` and `chelis-deep` fences used as teaching
examples. Fragment fences are intentionally illustrative only.
