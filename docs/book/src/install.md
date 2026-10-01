# Install

Install a release toolchain with `chelisup`, then use `chelis` to work in a
project. `chelisup` keeps installed versions side by side and provides the
`chelis` command that selects a version for each invocation. [Reef and
Packages](reef.md) covers package creation and dependencies.

## Install a release toolchain

You need an authenticated [GitHub CLI](https://cli.github.com). `chelisup`
downloads release assets through the authenticated GitHub REST API, so it needs a
GitHub token even though the releases are public: it reads `GITHUB_TOKEN`, or
`gh auth token` when that is unset. Prebuilt release assets are available for macOS arm64 and Linux x86-64. For
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

Before placing a toolchain, `chelisup` reads the unpacked compiler's live
`chelis runtime export` receipt and checks its sealed archive and six public
headers against the shipped `lib/` and `include/` files. Missing or crossed
files and malformed or ambiguous receipts (including duplicate JSON keys in
nested objects) refuse installation before store placement. A toolchain whose
compiler has no runtime export installs with a warning that its runtime files
are unchecked; if a newer receipt format is unreadable, update `chelisup` with
the bootstrap.

Publish the compiler, archive, and headers together as one versioned toolchain;
never replace only an installed archive or header. A rejected install leaves
the previous toolchain available. To roll back, select a previously installed
complete version with `chelisup default X.Y.Z` outside a project; explicit
version overrides and project pins must be changed separately.

## Use a project

A Reef project's `reef.toml` declares its toolchain under `[package]`, for
example `compiler = "=0.18.12"`. Install that version **before** invoking
`chelis` inside the project:

```sh
cd path/to/project
chelisup install 0.18.12
chelis reef setup
chelis reef build
```

Replace `0.18.12` with the `X.Y.Z` in your project's compiler pin. The shim
checks the pin before starting `chelis reef setup`, so setup cannot install a
missing pinned toolchain when invoked this way. Setup installs dependencies
recorded in `reef.lock`, if the file exists; it also syncs declared Chelis
source crates and reports toolchain, source-crate, and binary-artifact status.
`chelis reef build` is a separate command. See [Reef and Packages](reef.md) for
its outputs and dependency workflow.

The shim chooses an installed toolchain in this order: a leading `+X.Y.Z`
argument, `CHELIS_TOOLCHAIN`, the nearest `chelis-toolchain` file, the nearest
`reef.toml` compiler pin, then the recorded default. For example,
`chelis +0.18.12 --version` selects that installed version for one command.
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

Both `chelis` and `chelis-runtime` verify their copied archive and all six
public headers against the sealed compiler's export. `nix flake check`
rechecks both outputs against the combined package's own compiler, then links
and runs a native C consumer separately with each output's exact include
directory and archive path. Rebuild both outputs from the same compiler
revision; never mix an archive or header from another derivation. A failed
Nix-wrapper installation leaves its previous stable GC root and selected
toolchain untouched.

### Build the Python distribution wheel

With the [contributor prerequisites](https://github.com/Chelis-Lang/chelis/blob/main/docs/contributor_setup.md) installed, run
this from the compiler checkout root:

```sh
uv build --wheel --out-dir target/python-wheel/wheels bindings/python
```

This builds a wheel from the checkout. The Chelis GitHub toolchain release does
not include a Python wheel. For editable Python bindings during development,
see [Contributor setup](https://github.com/Chelis-Lang/chelis/blob/main/docs/contributor_setup.md#python-311).
