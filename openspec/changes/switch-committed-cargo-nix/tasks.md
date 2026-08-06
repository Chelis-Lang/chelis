## 1. Commit the graph and rewire the package layer

- [x] 1.1 Generate `Cargo.nix` with crate2nix 0.15.0 `generate --no-default-features --features chelis-cli/smt`.
- [x] 1.2 Mark `Cargo.nix` `linguist-generated` in `.gitattributes`.
- [x] 1.3 Replace `generatedCargoNix` with `import (root + "/Cargo.nix")` in `nix/packages.nix`.
- [x] 1.4 Drop `nixConfig.allow-import-from-derivation` and the crate2nix flake input.
- [x] 1.5 Flip the native check to `crate2nixCommittedGraph` (file, banner, members).

## 2. Add the shared ci consumer and compose it

- [x] 2.1 Add `devenv/shared/pins.nix` and the profile-free `devenv/consumer/devenv.nix` to ci.
- [x] 2.2 Point ci `devenv/shared/packages.nix` at the shared pins (no drift).
- [x] 2.3 Add the ci `tools/test_consumer_module.py` static guard.
- [x] 2.4 Compose `ci/devenv/consumer` in chelis `devenv.yaml`.

## 3. Regeneration, freshness, and OpenSpec through Devenv

- [x] 3.1 Add `scripts/regenerate_cargo_nix.py` (`--check` drift mode) and its unit test.
- [x] 3.2 Add the `regenerate-crate2nix` Devenv command using `config.outputs.crate2nix`.
- [x] 3.3 Add the `chelis:cargo-nix-fresh` Devenv smoke task.
- [x] 3.4 Replace `openspecPinned` with `config.outputs.openspec`.
- [x] 3.5 Remove `nix/regenerate.nix`, the flake app, and the CI `nix run` freshness step.

## 4. Update contracts and specs

- [x] 4.1 Rewrite the flake-contract suite for committed-graph, positive and negative.
- [x] 4.2 Update the gate workflow test (freshness now runs in `devenv test`).
- [x] 4.3 Update the devenv-composition test (composed tools, remote import, no local yaml import).
- [x] 4.4 Modify the `nix-package-outputs` and `cross-platform-devenv` capability specs.

## 5. Local validation

- [x] 5.1 `nix build .#chelis` from the committed graph.
- [x] 5.2 `nix eval` the default package with `allow-import-from-derivation false` (succeeds).
- [x] 5.3 `devenv build outputs.crate2nix outputs.openspec` from the composed ci module.
- [x] 5.4 Run the regenerate, flake-contract, gate, and devenv Python suites and the nixFormat check.

## 6. Acceptance oracle

- [ ] 6.1 Land the ci consumer module on `Chelis-Lang/ci` main and pin `github:Chelis-Lang/ci/<rev>` in `devenv.yaml`.
- [ ] 6.2 Green native `x86_64-linux` Nix Packages job on the final revision.
- [ ] 6.3 Green dispatched `aarch64-darwin` Nix Packages job on the same revision.
