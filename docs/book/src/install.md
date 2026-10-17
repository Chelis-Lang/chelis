# Install

Chelis ships as prebuilt binaries for Linux x86_64 and macOS arm64. Each
release on [GitHub](https://github.com/Chelis-Lang/chelis/releases) carries the
compiler archives, a `chelisup` binary for each platform, and `chelisup.sh`, a
short script that detects your platform and installs `chelisup`. `chelisup` is
the toolchain manager: it downloads compilers, keeps several versions side by
side, and installs the `chelis` command that picks the right version for each
invocation.

## Install a release toolchain

The commands below use the [GitHub CLI](https://cli.github.com) (`gh`),
signed in with `gh auth login`; without it, download `chelisup.sh` from the
[latest release](https://github.com/Chelis-Lang/chelis/releases/latest) page
instead. Run them outside a Chelis project:

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
The compiler also bundles `chelis-std`. Reef checks a shell's declared runtime
requirement against that bundled version; the runtime cannot be installed or
replaced independently.

Release tags start with `v`, while `chelisup install` takes a bare `X.Y.Z`
version, which is why the last lines strip the `v`.

On Linux, `chelisup` and the compiler it installs are static executables with
no system library dependencies, so they start on any x86_64 distribution. A
build linked against glibc 2.31 is also attached to each release. If you
prefer to unpack a compiler yourself, each release archive
(`chelis-vX.Y.Z-darwin-arm64.tar.gz`, `chelis-vX.Y.Z-linux-x86_64-static.tar.gz`)
holds `bin/chelis`, the runtime archive in `lib/`, and its headers in
`include/`. Keep those three together.

`chelis build` links your program with the runtime archive and with the C
library of your system. So the archive must match that C library. The static
and glibc 2.31 builds carry a glibc archive. On a musl distribution such as
Alpine, `chelisup` installs the musl build instead
(`chelis-vX.Y.Z-linux-x86_64-musl.tar.gz`), which carries a musl archive. Each
build refuses to link with a compiler for the other C library. `chelis check`,
`eval`, `test` and `prove` work with every build.

## Manage versions

The first installed toolchain becomes the default outside a project.

| Command | What it does |
|---|---|
| `chelisup install X.Y.Z` | Downloads and installs that version beside the others. |
| `chelisup link DIR` | Registers a toolchain directory that contains `bin/chelis`, under the version that its `bin/chelis --version` reports. |
| `chelisup list-installed` | Lists installed versions. |
| `chelisup show` | Prints the default, the version the current directory selects, and where toolchains are stored. |
| `chelisup which` | Prints the compiler binary `chelis` runs in the current directory. |
| `chelisup uninstall X.Y.Z` | Removes an installed version. For a linked version, it removes only the link. |
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

## Install with Nix

The Chelis flake builds the compiler from source for Linux x86_64 and macOS
arm64. It is a second channel beside the release binaries. `chelisup` stays
the installer of release toolchains and selects the compiler version for each
project. Run a release compiler, or install it in your Nix profile. Replace
`X.Y.Z` with a release version, as in the commands above:

```sh
nix run github:Chelis-Lang/chelis/vX.Y.Z -- --version
nix profile install github:Chelis-Lang/chelis/vX.Y.Z
```

The overlay and the C toolchain in the `chelis` package, which the text below
describes, start with the first release after 0.20.0.

The flake has three packages:

| Package | Contents |
|---|---|
| `chelis` (the default) | The compiler, the runtime archive in `lib/`, and its headers in `include/`. |
| `chelis-runtime` | Only the runtime archive and its headers, for C programs that link generated code. |
| `chelisup` | The toolchain manager. |

The `chelis` package carries a C compiler and binutils from Nix. Its
`bin/chelis` puts them first on `PATH`, so `chelis build` needs no host C
compiler or `ar`. `CHELIS_CC` and `CHELIS_AR` still select another
toolchain. See [Build programs](backends.md) for these overrides.

To add the packages to your own Nix package set, use the overlay of the flake.
It adds `chelis`, `chelis-runtime`, and `chelisup` for a system that the flake
supports:

```nix
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.chelis.url = "github:Chelis-Lang/chelis/vX.Y.Z";

  outputs = { nixpkgs, chelis, ... }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        overlays = [ chelis.overlays.default ];
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell { packages = [ pkgs.chelis ]; };
    };
}
```

### Use the binary cache

Without a binary cache, Nix compiles the compiler, its Rust dependencies, and
the cvc5 solver. That takes a long time. The `chelis-lang` cache holds these
builds for each release:

- substituter: `https://chelis-lang.cachix.org`
- public key: `chelis-lang.cachix.org-1:F4W65FQ7sunLoF2q6rzkWNTIBd3P/tJ/NFcv12OS9v8=`

The flake names this cache, but Nix uses the cache settings of a flake only
for a trusted user. A trusted user is `root`, a user in `trusted-users`, or
the user of a single-user install. A trusted user can accept the settings
with `--accept-flake-config`, or add the cache with `cachix use chelis-lang`.
For each other user, the Nix daemon prints `ignoring untrusted substituter`
and compiles the packages.

On NixOS, add the cache to the system configuration and rebuild the system:

```nix
nix.settings = {
  extra-substituters = [ "https://chelis-lang.cachix.org" ];
  extra-trusted-public-keys = [
    "chelis-lang.cachix.org-1:F4W65FQ7sunLoF2q6rzkWNTIBd3P/tJ/NFcv12OS9v8="
  ];
};
```

On another Linux system or on macOS, an administrator adds these lines to
`/etc/nix/nix.conf`:

```text
extra-substituters = https://chelis-lang.cachix.org
extra-trusted-public-keys = chelis-lang.cachix.org-1:F4W65FQ7sunLoF2q6rzkWNTIBd3P/tJ/NFcv12OS9v8=
```

Then the administrator restarts the Nix daemon. On Linux with systemd, use
`sudo systemctl restart nix-daemon`. On macOS, use
`sudo launchctl kickstart -k system/org.nixos.nix-daemon`.
