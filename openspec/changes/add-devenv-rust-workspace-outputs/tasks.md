## 1. Add executable contracts

- [ ] 1.1 Add positive tests for the pinned Devenv crate2nix input and lock parity.
- [ ] 1.2 Add negative tests for floating crate2nix references and revision drift.
- [ ] 1.3 Add composition tests for the workspace import module and its exact argument set.
- [ ] 1.4 Add negative tests for `rootCrate`, `builtins.getFlake`, Rust module replacement, and duplicate graphs.
- [ ] 1.5 Add native fixtures for all output layouts, the default alias, executable behavior, and SMT support.
- [ ] 1.6 Add CI contract tests for all four Devenv outputs and native package checks.

## 2. Extract shared package rules

- [ ] 2.1 Extract the crate2nix workspace graph helper from `nix/packages.nix`.
- [ ] 2.2 Extract the cvc5 and workspace source overrides into one shared helper.
- [ ] 2.3 Extract compiler, runtime, and chelisup artifact assembly into one shared helper.
- [ ] 2.4 Adapt the root flake package path to the shared helpers without output changes.
- [ ] 2.5 Run the root flake contract tests and native package checks after the refactor.

## 3. Add the Devenv workspace path

- [ ] 3.1 Pin crate2nix 0.15.0 in `devenv.yaml` and `devenv.lock`.
- [ ] 3.2 Extend the lock parity checker for crate2nix.
- [ ] 3.3 Add `devenv/rust-workspace.nix` with the lazy `config.chelis.rust.importWorkspace` function.
- [ ] 3.4 Use `config.languages.rust.toolchainPackage` and the Devenv crate2nix input in the graph helper.
- [ ] 3.5 Import the workspace module from the root Devenv composition.
- [ ] 3.6 Replace self-flake package outputs with one `workspaceMembers` graph.
- [ ] 3.7 Map the selected crate derivations through the shared artifact assembly helper.
- [ ] 3.8 Format all changed Nix files and confirm that shell evaluation stays lazy.

## 4. Update CI and documentation

- [ ] 4.1 Update both native jobs to build and check all four Devenv outputs.
- [ ] 4.2 Run the SMT fixture against the Devenv compiler in each native job.
- [ ] 4.3 Update README and installation documentation for separate graph ownership and shared contracts.
- [ ] 4.4 Update the changelog and active package specifications.
- [ ] 4.5 Keep the root flake package and application documentation unchanged for downstream users.

## 5. Run local validation

- [ ] 5.1 Run the focused Python tests for Devenv composition, pins, workflows, and Nix contracts.
- [ ] 5.2 Run `openspec validate --all --strict --no-interactive`.
- [ ] 5.3 Run `devenv test --no-tui` and confirm that no product graph builds.
- [ ] 5.4 Run `devenv build --no-tui` and confirm all four output names.
- [ ] 5.5 Check Devenv package layouts, executables, launcher lint, and SMT discharge.
- [ ] 5.6 Run `git diff --check` and the repository pre-commit command.

## 6. Run adversarial validation

- [ ] 6.1 Run a fresh local red-team agent against the specifications, code, tests, and CLI behavior.
- [ ] 6.2 Execute mutations for each forbidden fallback and confirm the required failure reason.
- [ ] 6.3 Audit empty member sets, absent overrides, default values, and hidden root-flake references.
- [ ] 6.4 Resolve each valid finding and rerun the affected acceptance surface.

## 7. Complete the hosted oracle

- [ ] 7.1 Obtain a successful native `x86_64-linux` package job for the final revision.
- [ ] 7.2 Dispatch and obtain a successful native `aarch64-darwin` package job for the same revision.
- [ ] 7.3 Record both native job results as the authoritative completion oracle.
