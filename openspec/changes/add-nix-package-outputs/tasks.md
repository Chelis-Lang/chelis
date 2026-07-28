**Authoritative completion oracle:**

Both native CI jobs must pass:

- `Nix Packages (x86_64-linux)`
- `Nix Packages (aarch64-darwin)`

Each job runs the complete flake check set on its named system.

## 1. Add contract tests before package code

- [x] 1.1 Add positive lock-parity test fixtures for Nixpkgs and the Rust overlay.
- [x] 1.2 Add a negative fixture with a mismatched Nixpkgs revision.
- [x] 1.3 Add a negative fixture with a mismatched Rust overlay revision.
- [x] 1.4 Add flake contract tests for package names, app names, default aliases, and supported systems.
- [x] 1.5 Add package-shape check stubs for every required and forbidden artifact path.
- [x] 1.6 Add compiler behavior check stubs for version, help, the release fixture, and SMT activation.
- [x] 1.7 Add runtime consumer check stubs for Linux OpenBLAS and macOS Accelerate.
- [x] 1.8 Record the exact five-header list from the release workflow.

## 2. Add locked flake inputs

- [x] 2.1 Add `flake.nix` with outputs for `x86_64-linux` and `aarch64-darwin`.
- [x] 2.2 Add focused package definitions under `nix/`.
- [x] 2.3 Pin Nixpkgs and the Rust overlay in `flake.lock`.
- [x] 2.4 Make the Rust overlay follow the flake Nixpkgs input.
- [x] 2.5 Select the Rust compiler from `rust-toolchain.toml`.
- [x] 2.6 Use `Cargo.lock` for the vendored Cargo dependency source.
- [x] 2.7 Filter local build products, generated Devenv state, and unrelated worktree files from package sources.

## 3. Implement lock parity

- [x] 3.1 Implement `scripts/check_nix_lock_parity.py` with parsed lock-node types.
- [x] 3.2 Compare flake Nixpkgs with the Devenv `nixpkgs-src` node.
- [x] 3.3 Compare the Rust overlay node in both lock files.
- [x] 3.4 Emit both node names and revisions for each mismatch.
- [x] 3.5 Run the positive and negative Python tests with `.venv/bin/python`.
- [x] 3.6 Add the parity checker to the flake check set.

## 4. Build the product packages

- [x] 4.1 Add a fixed-source cvc5 derivation that matches `cvc5-sys` in `Cargo.lock`.
- [x] 4.2 Verify that the cvc5 derivation supplies the complete `CVC5_DIR` tree.
- [x] 4.3 Add the private SMT-enabled `chelis-cli` derivation.
- [x] 4.4 Make the compiler build use the prebuilt cvc5 tree without network access.
- [x] 4.5 Add `packages.chelis-runtime` with the static library and five public headers.
- [x] 4.6 Add `packages.chelisup` with only `bin/chelisup`.
- [x] 4.7 Compose `packages.chelis` from the compiler and runtime outputs.
- [x] 4.8 Set `packages.default` to the exact `packages.chelis` derivation.
- [x] 4.9 Read every package version from the workspace manifest.

## 5. Add applications and package checks

- [x] 5.1 Add `apps.chelis`, `apps.chelisup`, and the default app alias.
- [x] 5.2 Verify that every app program path belongs to its package output.
- [x] 5.3 Verify every required file and reject undeclared product executables.
- [x] 5.4 Run `chelis --version` and compare it with the workspace version.
- [x] 5.5 Run `chelis --help` and the release pipe-stage fixture.
- [x] 5.6 Run `.github/scripts/verify_release_smt.py` against the packaged compiler.
- [x] 5.7 Compile and link a minimal C consumer against `packages.chelis-runtime`.
- [x] 5.8 Run `chelisup --help` from `packages.chelisup`.
- [x] 5.9 Combine all contracts into one native check target per supported system.

## 6. Add CI and documentation

- [x] 6.1 Add native Nix package jobs for `x86_64-linux` and `aarch64-darwin`.
- [x] 6.2 Make each job run its complete native flake check set.
- [x] 6.3 Keep the Nix jobs separate from the canonical Cargo gate.
- [x] 6.4 Update CI guard tests for the new workflow classification.
- [x] 6.5 Document every `nix build` and `nix run` output in `README.md`.
- [x] 6.6 Document the supported systems and native check command.
- [x] 6.7 State that `chelisup` remains the release installer and version router.
- [x] 6.8 Add a Nix package entry to `CHANGELOG.md`.

## 7. Run platform and repository gates

- [x] 7.1 Run Nix formatting checks for every tracked Nix file.
- [x] 7.2 Run `nix flake show` and verify the exact public output inventory.
- [ ] 7.3 Run the complete native flake check set on `x86_64-linux`.
- [x] 7.4 Run the complete native flake check set on `aarch64-darwin`.
- [x] 7.5 Verify that both native builds run without network access in their build sandboxes.
- [x] 7.6 Compare the toolchain and runtime paths with `.github/workflows/release.yml`.
- [x] 7.7 Run `.venv/bin/python scripts/test_check_nix_lock_parity.py`.
- [x] 7.8 Run `python3 scripts/gate.py --local` inside Devenv.
- [x] 7.9 Run `openspec validate add-nix-package-outputs --strict`.

## 8. Run adversarial validation

- [ ] 8.1 Spawn a fresh local red-team agent after all implementation gates pass.
- [ ] 8.2 Change only one Nixpkgs pin and verify a parity failure.
- [ ] 8.3 Change only one Rust overlay pin and verify a parity failure.
- [ ] 8.4 Remove one runtime header and verify a package-shape failure.
- [ ] 8.5 Add `bin/chelis` to the chelisup output and verify a collision failure.
- [ ] 8.6 Point one app at a host command and verify an app-contract failure.
- [ ] 8.7 Disable the SMT feature and verify an SMT-check failure.
- [ ] 8.8 Re-run both authoritative native CI jobs after accepted corrections.

## 9. Record acceptance evidence

- [ ] 9.1 Record all flake input revisions and package versions.
- [ ] 9.2 Record the closure paths and package contents for both supported systems.
- [ ] 9.3 Record both native CI job results and durations.
- [ ] 9.4 Record the fresh red-team result and each accepted correction.
