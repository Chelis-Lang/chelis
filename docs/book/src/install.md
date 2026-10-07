# Install

Chelis ships as prebuilt binaries for Linux x86_64 and macOS arm64. Each
release on [GitHub](https://github.com/Chelis-Lang/chelis/releases) carries the
compiler archives, a `chelisup` binary for each platform, and `chelisup.sh`, a
short script that detects your platform and installs `chelisup`. `chelisup` is
the toolchain manager: it downloads compilers, keeps several versions side by
side, and installs the `chelis` command that picks the right version for each
invocation.

## Install a release toolchain

The commands below use the [GitHub CLI](https://cli.github.com) (`gh`). Run
them outside a Chelis project:

```sh
# Download the installer script from the latest release and run it
gh release download -R Chelis-Lang/chelis -p chelisup.sh
sh chelisup.sh
# Put ~/.chelis/bin on your PATH (add this line to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.chelis/bin:$PATH"
# Install the latest compiler and confirm it runs
release_tag="$(gh release view -R Chelis-Lang/chelis --json tagName --jq .tagName)"
chelisup install "${release_tag#v}"
chelis --version
```

The script installs only `chelisup`; `chelisup install` downloads the compiler.
Release tags start with `v`, while `chelisup install` takes a bare `X.Y.Z`
version, which is why the last lines strip the `v`.

On Linux, `chelisup` and the compiler it installs are static executables with
no system library dependencies, so they start on any x86_64 distribution. A
build linked against glibc 2.31 is also attached to each release. If you
prefer to unpack a compiler yourself, each release archive
(`chelis-vX.Y.Z-darwin-arm64.tar.gz`, `chelis-vX.Y.Z-linux-x86_64-static.tar.gz`)
holds `bin/chelis`, the runtime archive in `lib/`, and its headers in
`include/`. Keep those three together.

The runtime archive is built against glibc. On a musl distribution such as
Alpine, `chelis check`, `eval`, `test` and `prove` work, but `chelis build`
refuses to link with the system C compiler.

## Manage versions

The first installed toolchain becomes the default outside a project.

| Command | What it does |
|---|---|
| `chelisup install X.Y.Z` | Downloads and installs that version beside the others. |
| `chelisup list-installed` | Lists installed versions. |
| `chelisup show` | Prints the default, the version the current directory selects, and where toolchains are stored. |
| `chelisup which` | Prints the compiler binary `chelis` runs in the current directory. |
| `chelisup uninstall X.Y.Z` | Removes an installed version. |
| `chelisup default X.Y.Z` | Sets the default for directories outside a project. |

When you run `chelis`, it selects an installed version in this order, and the
first match wins:

1. a leading `+X.Y.Z` argument, as in `chelis +X.Y.Z --version`;
2. the `CHELIS_TOOLCHAIN` environment variable;
3. the nearest `chelis-toolchain` file in the current or a parent directory;
4. the nearest `reef.toml`'s `compiler` pin;
5. the default set by `chelisup default`.

A selected version that is not installed is an error naming the
`chelisup install` command to run; `chelis` never falls back to another
version. Because the first three outrank the project pin, a stray
`CHELIS_TOOLCHAIN` or `chelis-toolchain` file overrides `reef.toml`.

## Use a project

A Reef project's `reef.toml` pins its compiler under `[package]`, for example
`compiler = "=X.Y.Z"`. In a fresh clone, one command installs that version if
it is missing, installs the dependencies recorded in `reef.lock`, and prints a
status summary:

```sh
chelis reef setup
chelis reef build
```

The [Reef and packages](reef.md) guide covers creating a project
and managing dependencies.

## Build requirements

`chelis eval` and `chelis check` need nothing beyond the compiler.
`chelis build` also needs a native C compiler, and a definitions-only module,
which builds to a static library, needs the `ar` archiver:

- On macOS, install Apple's Command Line Tools with `xcode-select --install`.
- On Debian or Ubuntu, install `build-essential`.
- On Fedora, install `gcc` and `binutils`.

Generated C carries its own matrix-multiplication loops and correctly rounded
math kernels, so an ordinary build links only the Chelis runtime archive and
`-lm`; GCC builds add `-fopenmp` for parallel loops. The build prints the
compiler it used, and `--emit-c` prints the exact compile command.
`chelis build app.ch --emit-c` writes the C sources and runtime files without
calling a compiler, so it needs none of these tools. See
[Build programs](backends.md) for compiler overrides and linking a
generated static library.
