## Why

Chelis ships the Linux toolchain from two build systems. The Nix flake builds a fully pinned, sandboxed, checked toolchain. The release workflow builds a second toolchain with Cargo inside a `debian:11` container. The container lane exists only to cap the glibc floor at 2.31 for distribution reach. Chelis has zero users today, so that reach requirement does not exist.

The container lane is the loosest-pinned build in the repository. It uses a tag-only container image, a floating stable Rust toolchain, apt packages of the day, and a network fetch of cvc5 at build time. It duplicates the cvc5 recipe that `nix/cvc5.nix` already pins exactly. The duplication is the drift class this repository's contracts exist to prevent.

An investigation (2026-08) confirmed the technical blocker is absent. `libchelis_runtime.a` is a pure-Rust archive with no dynamic section. Its glibc symbol binding happens at the consumer's link, so the archive built on Nix glibc links under an ordinary glibc `gcc` and under `hipcc`/ROCm regardless of the build host's glibc. Only the compiler binary needs a portability rewrite, and `nix/release-chelisup.nix` already proves that pattern in this repository.

## What Changes

- Add a Devenv output `outputs.release-chelis` on `x86_64-linux`. It builds the compiler, runtime static library, and headers from the committed `Cargo.nix` graph and the pinned cvc5 tree, then stages the exact Linux release tarball.
- Make the staged compiler binary portable off the Nix store: rewrite the ELF interpreter to `/lib64/ld-linux-x86-64.so.2`, link `libstdc++` and `libgcc` statically, remove store rpaths, and assert that no `/nix/store` reference remains.
- Record the exact glibc floor of the portable binary as a committed contract value. A nixpkgs bump that moves the floor becomes a reviewed diff, not a surprise.
- Keep the runtime static library on the glibc Rust target. Add a negative control that fails if the archive ever carries bundled libc objects (the musl failure mode).
- Add an off-Nix consumption job to the release workflow: a clean Linux environment without Nix runs the portable binary, compiles emitted C against the shipped archive, and executes the result.
- Remove the `build-linux-x86_64` and `build-linux-x86_64-glibc231` Cargo release jobs and the `smt-build-glibc231` CI lane. Move the `chelisup.sh` asset staging into a remaining job.
- Defer the public glibc-floor support policy until users exist. The design records the retrofit paths.

## Capabilities

### New Capabilities

- `portable-chelis-release`: Defines the portable Linux toolchain release artifacts built from the Devenv output, their off-store portability contract, the recorded glibc floor, the runtime-archive ABI controls, and the release workflow publication rules.

### Modified Capabilities

- `portable-chelisup-release`: Rescopes the release-workflow requirement. The Darwin full toolchain job keeps its current Cargo evidence. Linux toolchain publication moves to the `portable-chelis-release` capability.

## Impact

The change affects `devenv/release-outputs.nix`, a new `nix/release-chelis.nix`, `nix/contracts.nix`, `.github/workflows/release.yml`, `.github/workflows/ci.yml`, release verification scripts under `scripts/` and `.github/scripts/`, release workflow contract tests, `docs/smt_build_setup.md`, `docs/book/src/install.md`, and the changelog.

No compiler, runtime, CLI, backend, or generated-code behavior changes. The shipped Linux artifacts change provenance and glibc floor: the tarball asset name that `chelisup` downloads stays `chelis-v<version>-linux-x86_64.tar.gz`, the `-glibc2.31` asset disappears, and the recorded floor replaces the 2.31 cap. The Darwin release lane and the flake packages are unchanged.

The authoritative completion oracle is one green manual dispatch of the release workflow at a branch, with the Linux tarball produced by `outputs.release-chelis` and the off-Nix consumption job passing. The required `Nix Packages (x86_64-linux)` check provides the per-PR supporting evidence.
