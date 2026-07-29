**Authoritative completion oracle:**

```sh
devenv shell -- cargo nextest run -p chelis-backend-c
```

Run this oracle on macOS. The Linux manual gate is `devenv test`, and default CI does not run it.

**Allowed implementation file list:**

- `.gitignore`
- `CHANGELOG.md`
- `README.md`
- `chelis-lint.toml`
- `devenv.lock`
- `devenv.nix`
- `devenv.yaml`
- `openspec/changes/add-cross-platform-devenv-shell/.openspec.yaml`
- `openspec/changes/add-cross-platform-devenv-shell/acceptance.md`
- `openspec/changes/add-cross-platform-devenv-shell/design.md`
- `openspec/changes/add-cross-platform-devenv-shell/proposal.md`
- `openspec/changes/add-cross-platform-devenv-shell/specs/cross-platform-devenv/spec.md`
- `openspec/changes/add-cross-platform-devenv-shell/tasks.md`
- `scripts/test_devenv_version.py`

The later OpenSpec sync and archive workflow can change the corresponding main spec and archive paths.

## 1. Isolate the extraction and write the shell checks

- [x] 1.1 Create the implementation branch from current `origin/main`, not from PR #894.
- [x] 1.2 Bring this OpenSpec change into the implementation branch.
- [x] 1.3 Record an allowed file list for the extraction before any implementation edit.
- [x] 1.4 Add `enterTest` assertions for every required common tool and Python 3.11.
- [x] 1.5 Add a valid C `-fsyntax-only` probe and a valid C++17 `-c` probe with `-Werror`.
- [x] 1.6 Add a negative compile probe with a deliberate warning and require compiler failure.
- [x] 1.7 On macOS, run the probes before the shims and record the expected unmanaged-command failure.
- [x] 1.8 Do not cherry-pick `cbe955cd` or `6dbf8605`. Those commits contain unrelated files.

## 2. Add the tracked Devenv inputs

- [x] 2.1 Add `devenv.yaml` with the Nixpkgs and Rust overlay inputs.
- [x] 2.2 Add `devenv.lock` and verify that it pins each resolved input revision.
- [x] 2.3 Add `devenv.nix` with Rust from `rust-toolchain.toml`.
- [x] 2.4 Add Python 3.11 and uv to `devenv.nix`.
- [x] 2.5 Add `cargo-llvm-cov`, `cargo-nextest`, CMake, Git, and pkg-config to the common package list.
- [x] 2.6 Add an `enterShell` hook that creates `.venv` only when `.venv/bin/python` is absent.
- [x] 2.7 Add GCC, OpenBLAS, and Valgrind to the Linux package list.
- [x] 2.8 Keep OpenBLAS and Valgrind out of the macOS package list.
- [x] 2.9 Add `.devenv/` and `.devenv.flake.nix` to `.gitignore`.
- [x] 2.10 Run `git check-ignore` for both generated paths and verify successful matches.
- [x] 2.11 Exclude both generated paths from the editable Chelis lint corpus.

## 3. Add the macOS compiler shims

- [x] 3.1 Add a macOS-only `gcc` shim that invokes `${pkgs.stdenv.cc}/bin/cc`.
- [x] 3.2 Add a macOS-only `g++` shim that invokes `${pkgs.stdenv.cc}/bin/c++`.
- [x] 3.3 Pass `-Wno-unused-command-line-argument` before each shim argument list.
- [x] 3.4 Run `devenv test` on macOS and verify successful C and C++ probes.
- [x] 3.5 Verify that the deliberate warning still fails under `-Werror`.
- [x] 3.6 Run `gcc --version` and `g++ --version` inside the shell and record Nixpkgs clang.
- [x] 3.7 Run `nixfmt-rs --check devenv.nix`.

## 4. Document the contributor path

- [x] 4.1 Add the optional `devenv shell` command to the source-build prerequisites.
- [x] 4.2 Add the `devenv test` command and its expected success result.
- [x] 4.3 Add the authoritative C-backend oracle command.
- [x] 4.4 State that the macOS `gcc` and `g++` shims invoke the Nixpkgs clang wrapper.
- [x] 4.5 Keep the existing manual rustup, uv, and platform compiler instructions.
- [x] 4.6 Add a contributor-tooling entry to `CHANGELOG.md`.

## 5. Run platform and repository gates

- [x] 5.1 Run `devenv test` on macOS and record exit status 0.
- [x] 5.2 Run `devenv test` on Linux and record exit status 0.
- [x] 5.3 State that the Linux command is a manual gate and is absent from default CI.
- [x] 5.4 Run the authoritative completion oracle on macOS and record exit status 0.
- [x] 5.5 Verify that no compiler-dependent `chelis-backend-c` test reports a missing compiler.
- [x] 5.6 Run `python3 scripts/gate.py --local` and record exit status 0.
- [x] 5.7 Run `openspec validate add-cross-platform-devenv-shell --strict` and record exit status 0.
- [x] 5.8 Compare the final diff with the allowed file list from task 1.3.
- [x] 5.9 Verify that the diff contains no runtime, ABI, backend, vocabulary, or nextest-profile edit.

## 6. Run adversarial validation

- [x] 6.1 Spawn a fresh local red-team subagent after all implementation gates pass.
- [x] 6.2 Remove both macOS shims in a temporary worktree and verify that `devenv test` fails.
- [x] 6.3 Remove the wrapper-warning flag in a temporary worktree and verify the valid compile probe fails.
- [x] 6.4 Replace the warning flag with `-w` and verify that the negative compile probe detects it.
- [x] 6.5 Create a clean temporary checkout without `.venv` and verify that Devenv creates Python 3.11.
- [x] 6.6 Add a marker to an existing temporary `.venv` and verify that Devenv preserves it.
- [x] 6.7 Re-run the authoritative completion oracle after valid red-team fixes.

## 7. Record acceptance evidence

- [x] 7.1 Record the macOS and Linux Devenv versions.
- [x] 7.2 Record the macOS compiler version output from both shims.
- [x] 7.3 Record the result and duration of the authoritative completion oracle.
- [x] 7.4 Record the fresh red-team result and each accepted correction.

## 8. Upgrade Devenv to v2.2

- [x] 8.1 Add positive and negative tests for the exact release pin.
- [x] 8.2 Pin the Devenv module input to release `v2.2`.
- [x] 8.3 Update `devenv.lock` to commit `ffce215a42d09c6375c3d60dd9c4110438fc4d87`.
- [x] 8.4 Upgrade the local Devenv CLI to `2.2.0+ffce215`.
- [x] 8.5 Run the Devenv smoke check and the repository gates.
- [x] 8.6 Review the final release pin and parser boundary cases directly.
