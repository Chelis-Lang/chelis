# Install

Install a release toolchain with `chelisup`, then use `chelis` to work in a
project. `chelisup` keeps installed versions side by side and provides the
`chelis` command that selects a version for each invocation. [Reef and
Packages](reef.md) covers package creation and dependencies.

## Install a release toolchain

The Chelis GitHub releases are private. You need access to
`Chelis-Lang/chelis` and an authenticated [GitHub CLI](https://cli.github.com).
Prebuilt release assets are available for macOS arm64 and Linux x86-64. For
source-build workflows, see [Contributor setup](https://github.com/Chelis-Lang/chelis/blob/main/docs/contributor_setup.md).

Run these commands in a terminal outside a Chelis project:

```sh
gh auth login
gh release download --repo Chelis-Lang/chelis --pattern chelisup.sh --output - | sh
export PATH="$HOME/.chelis/bin:$PATH"
release_tag="$(gh release view --repo Chelis-Lang/chelis --json tagName --jq .tagName)"
chelisup install "${release_tag#v}"
chelis --version
```

If you are already signed in with `gh`, you can skip `gh auth login`. The
bootstrap script installs **only `chelisup`**. The separate `chelisup install`
command downloads the toolchain and installs the `chelis` version-selecting
shim. The script prints the PATH line for your install location; the line above
uses the default `~/.chelis/bin`. `chelisup install` takes a bare `X.Y.Z`, while
GitHub release tags begin with `v`.

The first installed toolchain becomes the default outside a project. Check the
installed versions with `chelisup list-installed`, inspect the active selection
with `chelisup show`, or set another installed default with
`chelisup default X.Y.Z`.

## Use a project

A Reef project's `reef.toml` declares its toolchain under `[package]`, for
example `compiler = "=0.18.11"`. Install that version **before** invoking
`chelis` inside the project:

```sh
cd path/to/project
chelisup install 0.18.11
chelis reef setup
chelis reef build
```

Replace `0.18.11` with the `X.Y.Z` in your project's compiler pin. The shim
checks the pin before starting `chelis reef setup`, so setup cannot install a
missing pinned toolchain when invoked this way. Setup installs dependencies
recorded in `reef.lock`, if the file exists; it also syncs declared Chelis
source crates and reports toolchain, source-crate, and binary-artifact status.
`chelis reef build` is a separate command. See [Reef and Packages](reef.md) for
its outputs and dependency workflow.

The shim chooses an installed toolchain in this order: a leading `+X.Y.Z`
argument, `CHELIS_TOOLCHAIN`, the nearest `chelis-toolchain` file, the nearest
`reef.toml` compiler pin, then the recorded default. For example,
`chelis +0.18.11 --version` selects that installed version for one command.
A missing selected version produces an error naming `chelisup install X.Y.Z`;
there is no automatic fallback. `+latest` is unsupported.

`chelis reef doctor` reports the installed toolchain, source-crate sync, and
binary artifacts for the current project without installing anything. With
`--root DIR`, it scans `DIR` and its immediate subdirectories for projects.
It does not check whether source-package dependencies are installed.

## Build from a checkout

[Contributor setup](https://github.com/Chelis-Lang/chelis/blob/main/docs/contributor_setup.md) covers the Rust, C, and Python
tools needed to build Chelis from source. A checkout build is separate from a
release installed by `chelisup`.

### Nix source packages

From the compiler checkout root, Nix provides source builds of `chelis`,
`chelis-runtime`, and `chelisup` on `x86_64-linux` and `aarch64-darwin`:

```sh
nix build .#chelis
nix run .#chelis -- --version
```

### Build the Python distribution wheel

With the [contributor prerequisites](https://github.com/Chelis-Lang/chelis/blob/main/docs/contributor_setup.md) installed, run
this from the compiler checkout root:

```sh
uv build --wheel --out-dir target/python-wheel/wheels bindings/python
```

This builds a wheel from the checkout. The Chelis GitHub toolchain release does
not include a Python wheel. For editable Python bindings during development,
see [Contributor setup](https://github.com/Chelis-Lang/chelis/blob/main/docs/contributor_setup.md#python-311).
