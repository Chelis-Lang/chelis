## 1. Add contract tests first

- [x] 1.1 Add a static inspector for the release helper, Devenv module, and supported platform contract.
- [x] 1.2 Add positive tests for the committed graph, exact inventory, version probes, architecture checks, runtime checks, and checksum checks.
- [x] 1.3 Add negative mutations for each required check and for forbidden Nix-package reuse.
- [x] 1.4 Add release workflow tests that require the matrix output build and reject a direct `chelisup` Cargo build.
- [x] 1.5 Run the focused tests and record the expected failures before implementation.

## 2. Implement the Devenv release output

- [x] 2.1 Add the focused Nix helper that selects `workspaceMembers."chelisup"` from committed `Cargo.nix`.
- [x] 2.2 Use the configured Rust toolchain and the musl target with the musl cross package set on Linux.
- [x] 2.3 Rewrite Darwin `libiconv` to the Apple system path before executable verification.
- [x] 2.4 Add exact inventory, version, help, architecture, runtime dependency, and SHA-256 checks.
- [x] 2.5 Add `devenv/release-outputs.nix` and import it from `devenv.nix`.
- [x] 2.6 Make the Rust host detector and bootstrap script reject unsupported release platforms.

## 3. Migrate release workflow ownership

- [x] 3.1 Add the manual `build-chelisup-release` matrix for `linux-x86_64` and `darwin-arm64`.
- [x] 3.2 Set up Devenv and private ci authentication before the matrix build.
- [x] 3.3 Upload only the executable and sidecar from `outputs.release-chelisup`.
- [x] 3.4 Remove direct Cargo builds and stage steps for `chelisup` from the full toolchain jobs.
- [x] 3.5 Keep `chelisup.sh` publication, cold cvc5 builds, release triggers, and tag publication policy unchanged.
- [x] 3.6 Add the release-output matrix as a publication dependency.

## 4. Update project guidance

- [x] 4.1 Document `devenv build --no-tui outputs.release-chelisup` and distinguish it from `outputs.chelisup`.
- [x] 4.2 Add an Unreleased changelog entry for the portable release output.

## 5. Run acceptance evidence

- [x] 5.1 Run the focused release-output and workflow tests.
- [x] 5.2 Run the complete Python script discovery suite.
- [x] 5.3 Run `devenv build --no-tui outputs.release-chelisup` on the current native host.
- [x] 5.4 Run `devenv test --no-tui`, the Nix format check, actionlint, pre-commit, and `git diff --check`.
- [ ] 5.5 Run the manual `Build portable chelisup` release workflow matrix. Both platform legs must pass as the authoritative completion oracle.
