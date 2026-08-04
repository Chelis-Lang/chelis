## 1. Add executable contracts

- [x] 1.1 Add positive tests for the pinned Devenv crate2nix input and lock parity.
- [x] 1.2 Add negative tests for floating crate2nix references and revision drift.
- [x] 1.3 Add composition tests for the workspace import module and its exact argument set.
- [x] 1.4 Add negative tests for `rootCrate`, `builtins.getFlake`, Rust module replacement, and duplicate graphs.
- [x] 1.5 Add native fixtures for all output layouts, the default alias, executable behavior, and SMT support.
- [x] 1.6 Add CI contract tests for all four Devenv outputs and native package checks.
- [x] 1.7 Add positive and negative contracts for the tracked graph, digest, and conditional regeneration.

## 2. Extract shared package rules

- [x] 2.1 Extract the crate2nix workspace graph helper from `nix/packages.nix`.
- [x] 2.2 Extract the cvc5 and workspace source overrides into one shared helper.
- [x] 2.3 Extract compiler, runtime, and chelisup artifact assembly into one shared helper.
- [x] 2.4 Adapt the root flake package path to the shared helpers without output changes.
- [x] 2.5 Run the root flake contract tests and native package checks after the refactor.
- [x] 2.6 Replace package-evaluation IFD with one checked-in graph for both interfaces.
- [x] 2.7 Add the fast digest check and exact offline regeneration derivation.

## 3. Add the Devenv workspace path

- [x] 3.1 Pin crate2nix 0.15.0 in `devenv.yaml` and `devenv.lock`.
- [x] 3.2 Extend the lock parity checker for crate2nix.
- [x] 3.3 Add `devenv/rust-workspace.nix` with the lazy `config.chelis.rust.importWorkspace` function.
- [x] 3.4 Use `config.languages.rust.toolchainPackage` and the Devenv crate2nix input in the graph helper.
- [x] 3.5 Import the workspace module from the root Devenv composition.
- [x] 3.6 Replace self-flake package outputs with one `workspaceMembers` graph.
- [x] 3.7 Map the selected crate derivations through the shared artifact assembly helper.
- [x] 3.8 Format all changed Nix files and confirm that shell evaluation stays lazy.

## 4. Update CI and documentation

- [x] 4.1 Update both native jobs to build and check all four Devenv outputs.
- [x] 4.2 Run the SMT fixture against the Devenv compiler in each native job.
- [x] 4.3 Update README and installation documentation for one graph and separate package instantiation.
- [x] 4.4 Update the changelog and active package specifications.
- [x] 4.5 Keep the root flake package and application documentation unchanged for downstream users.
- [x] 4.6 Select exact graph regeneration only in the Linux job for graph changes and dispatches.
- [x] 4.7 Remove graph regeneration from the macOS job and update graph documentation.

## 5. Run local validation

- [x] 5.1 Run the focused Python tests for Devenv composition, pins, workflows, and Nix contracts.
- [x] 5.2 Run `openspec validate --all --strict --no-interactive`.
- [x] 5.3 Run `devenv test --no-tui` and confirm that no product graph builds.
- [x] 5.4 Run `devenv build --no-tui` and confirm all four output names.
- [x] 5.5 Check Devenv package layouts, executables, launcher lint, and SMT discharge.
- [x] 5.6 Run `git diff --check` and the repository pre-commit command.

## 6. Run adversarial validation

- [x] 6.1 Run a fresh local red-team agent against the specifications, code, tests, and CLI behavior.
- [x] 6.2 Execute mutations for each forbidden fallback and confirm the required failure reason.
- [x] 6.3 Audit empty member sets, absent overrides, default values, and hidden root-flake references.
- [x] 6.4 Resolve each valid finding and rerun the affected acceptance surface.

## 7. Complete the hosted oracle

- [ ] 7.1 Obtain a successful native `x86_64-linux` package job for the final revision.
- [ ] 7.2 Dispatch and obtain a successful native `aarch64-darwin` package job for the same revision.
- [ ] 7.3 Record both native job results as the authoritative completion oracle.
