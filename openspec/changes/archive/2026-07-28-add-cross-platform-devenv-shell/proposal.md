## Why

PR [#894](https://github.com/Chelis-Lang/chelis/pull/894) includes a contributor environment change that is independent of its runtime work.

Many C tests invoke `gcc` or `g++` by exact name. The Nix Darwin shell provides the compiler as `cc` and `c++`.

Host aliases vary by macOS installation and are not Devenv inputs. As a result, local runs can skip or fail the C test tier.

## What Changes

- Add a tracked Devenv shell for the Rust, Python, C, and test tools that Chelis uses.
- Add `gcc` and `g++` shims on macOS. The shims invoke the Nixpkgs Darwin compiler wrappers.
- Keep real GCC, OpenBLAS, and Valgrind in the Linux shell.
- Create the required Python 3.11 `.venv` when a contributor enters the shell.
- Add shell checks for the tool versions and C and C++ compilation.
- Ignore Devenv state and generated flake files in Git and the repository lint corpus.
- Document the optional Devenv path for contributors.
- Extract only the Devenv files from PR #894. Exclude all runtime, ABI, backend, vocabulary, and nextest-profile changes.

## Capabilities

### New Capabilities

- `cross-platform-devenv`: Defines the tracked development shell, platform compiler commands, Python environment, and executable shell checks.

### Modified Capabilities

None.

## Impact

- `devenv.nix`, `devenv.yaml`, and `devenv.lock` become tracked environment inputs.
- `.gitignore` and `chelis-lint.toml` exclude `.devenv/` and `.devenv.flake.nix` from generated-state checks.
- The contributor documentation gains the Devenv activation and test commands.
- macOS uses the pinned Nixpkgs clang wrapper behind the `gcc` and `g++` command names inside the shell.
- Linux continues to use the GCC package from Nixpkgs.
- The change does not modify the shipped compiler, runtime, C ABI, generated code, or CI gate.
