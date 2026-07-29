# Acceptance evidence

## Pre-shim macOS probe

The probe ran on macOS with Devenv 2.1.2.

`devenv test --no-tui` exited with a nonzero status before the macOS shims existed.

The initial generated test executable reported `missing required shell command: gcc`.

The final managed-compiler check reports `unmanaged shell command: gcc (/usr/bin/gcc)` when the macOS shims are absent.

The check rejects `/usr/bin/gcc` and `/usr/bin/g++` as host fallbacks. The shell must supply reproducible compiler commands from the Nix store.

## Implemented macOS smoke check

`devenv test --no-tui` exited with status 0 after both shims existed.

The successful check ran the required tool versions, Python 3.11 assertion, and valid C and C++17 compile probes.

The successful check also proved that the deliberate C warning failed under `-Werror`.

`nixfmt-rs --check devenv.nix` exited with status 0.

`git check-ignore` matched both `.devenv` and `.devenv.flake.nix` against the new root rules.

## Compiler research decision

The [locked Darwin stdenv source](https://github.com/NixOS/nixpkgs/blob/f205b5574fd0cb7da5b702a2da51507b7f4fdd1b/pkgs/stdenv/darwin/default.nix) selects `llvmPackages_21.clang` as `stdenv.cc`.

The [Nixpkgs platform notes](https://github.com/NixOS/nixpkgs/blob/f205b5574fd0cb7da5b702a2da51507b7f4fdd1b/doc/stdenv/platform-notes.chapter.md) specify Clang as the default Darwin compiler.

The platform notes also specify the Nix compiler wrapper and default SDK for hard-coded compiler commands.

Inside the shell, `NIX_CC` names the clang 21.1.8 wrapper. `SDKROOT` names the Nixpkgs Apple SDK 14.4.

Both shim version commands report clang 21.1.8 from `/nix/store/qsl14qgcaf1y526f6an28kp1s8pal3h1-clang-21.1.8/bin`.

Outside the shell, `/usr/bin/clang` reports Apple clang 21.0.0. This host compiler is not pinned by `devenv.lock`.

The [Nix compiler wrapper](https://github.com/NixOS/nixpkgs/blob/f205b5574fd0cb7da5b702a2da51507b7f4fdd1b/pkgs/build-support/cc-wrapper/cc-wrapper.sh) omits linker flags for `-c`.

A valid `cc -Werror -c` probe exited with status 0 without the shim suppression.

The wrapper does not classify `-fsyntax-only` as a no-link mode. A valid probe then failed on injected `-L` arguments.

The same `-fsyntax-only` probe exited with status 0 through the shim. The deliberate source warning still failed under `-Werror`.

The decision keeps `${pkgs.stdenv.cc}`. The specification and documentation identify it as the pinned Nixpkgs clang wrapper, not host Apple clang.

## macOS acceptance gates

Devenv 2.1.2 ran on `aarch64-darwin`.

`devenv test --no-tui` exited with status 0 after the research corrections.

The authoritative oracle ran with an absolute isolated target directory and its matching `CHELIS_RUNTIME_DIR`.

`devenv shell -- cargo nextest run -p chelis-backend-c` exited with status 0.

The oracle completed 384 tests in 146.730 seconds. All 384 tests passed, and one excluded test remained skipped.

No compiler-dependent test reported a missing compiler.

## Repository lint integration

The first local gate scanned `.devenv/*.sh` because Chelis lint does not use Git ignore rules.

The gate reported 469 `no-shell-scripts` errors from generated Devenv files.

The allowed file list now includes `chelis-lint.toml`. This correction keeps generated state outside the editable lint corpus under §12.2.

The focused repository lint exited with status 0 after the policy correction. It reported only pre-existing advisory warnings.

The local gate ran inside Devenv with the project Python and Cargo toolchain. It exited with status 0.

Strict OpenSpec validation exited with status 0 after the final implementation edits.

The final implementation paths match the allowed file list. The diff contains no runtime, ABI, backend, vocabulary, or nextest-profile edit.

## Linux manual gate

Devenv 2.1.2 generated the `x86_64-linux` shell and `devenv.config.test` program from the tracked inputs.

The macOS process cannot execute Linux binaries. Therefore, a Nix derivation ran the generated test inside the realized Linux Devenv profile.

The Linux test reported GCC 15.2.0 for both `gcc` and `g++`.

The generated test completed every tool check, Python 3.11 assertion, positive compile probe, and negative warning probe.

The Linux derivation exited with status 0. Its result file contains `linux devenv test passed`.

Default CI did not run this Linux gate during the initial acceptance. The later GitHub Actions integration now runs it.

## Fresh local red-team result

A fresh local Pi agent ran the red-team pass in a temporary worktree. It did not edit the primary checkout.

The agent removed both macOS shims. `devenv test --no-tui` failed and rejected `/usr/bin/gcc`.

The agent removed `-Wno-unused-command-line-argument`. The valid C `-fsyntax-only` probe failed under `-Werror`.

The agent replaced the flag with `-w`. The negative warning probe detected the hidden source warning and failed the test.

A clean temporary checkout created Python 3.11.15 from the Nix store.

A second checkout contained an existing environment marker. Devenv preserved the marker and the Python file metadata.

The agent removed the lint exclusions. Repository lint then failed with 26 generated Devenv shell-script errors.

After all mutations were reverted, the acceptance oracle passed all 384 tests in 201.801 seconds.

The red team found no high-severity or medium-severity error. It found one low-severity error in the recorded pre-shim diagnostic.

The accepted correction records both the initial diagnostic and the final managed-compiler diagnostic.

The final post-correction oracle passed all 384 tests in 172.932 seconds. One excluded test remained skipped.

## Devenv v2.2 upgrade

The release tag `v2.2` resolves to commit `ffce215a42d09c6375c3d60dd9c4110438fc4d87`.

The local Nix profile reports `devenv 2.2.0+ffce215 (aarch64-darwin)`.

`devenv.yaml` pins `github:cachix/devenv/v2.2?dir=src/modules`. `devenv.lock` records the same tag, module directory, and commit.

The static release contract passed five tests. Its negative tests reject an old revision, a wrong locked directory, and a misplaced input key.

`devenv test --no-tui` passed in 3.16 seconds with Devenv v2.2.

The full Python script suite, strict OpenSpec validation, native flake check, and local repository gate passed.

The independent reviewer service failed three attempts. The user then requested a direct review instead.

The direct review found missing coverage for YAML section boundaries and the locked module directory. Both corrections now have negative tests.

## Named Devenv test tasks

The shell smoke check now runs four tasks after `devenv:enterShell` and before `devenv:enterTest`:

- `chelis:toolchain-test`
- `chelis:python-test`
- `chelis:c-compiler-test`
- `chelis:cpp-compiler-test`

`devenv test --no-tui` passed all four tasks. The empty `devenv:enterTest` lifecycle step then completed.

The task graph defines no Devenv service or long-running process.

The Devenv static contract passed eight positive and negative tests. The full Python script suite also passed.

Strict OpenSpec validation, the Nix format check, and the local repository gate passed.

The C-backend acceptance oracle passed all 384 tests. One test remained skipped by its existing configuration.

The independent final review returned `PASS`.

All 15 pull request checks passed for commit `33c2899a`. This result includes both native Nix package jobs.

## Official Devenv GitHub Actions integration

Both native Nix package jobs now configure `devenv.cachix.org` through `cachix/cachix-action@v16`. The `skipPush` option prevents uploads.

Each job installs the CLI from revision `ffce215a42d09c6375c3d60dd9c4110438fc4d87`. A temporary profile reported `devenv 2.2.0+ffce215`.

Each job runs `devenv test --no-tui` before the complete native flake check.

A clean temporary configuration passed all four tasks. The explicit lifecycle edges prevented a race with `.venv` creation.

The eight native workflow tests, `actionlint`, the full Python script suite, strict OpenSpec validation, and the local repository gate passed.

The public Devenv cache does not contain the custom non-GPL cvc5 derivation. Native package jobs can still build cvc5 from source.

The reviewer requested a cache cleanup input. The v16 action defines no cleanup input, and GitHub-hosted runners discard job state.

The workflow instead sets the supported `skipPush` input to `true`. This value makes the public cache explicitly read-only.
