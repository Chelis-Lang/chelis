## Why

After `switch-linux-release-to-nix`, the Darwin tarball is the last release artifact built outside Nix: a Cargo job on a floating stable toolchain with a cold cvc5 source build. The 2026-08-08 spike proved the Nix path on `aarch64-darwin` with a small delta from the Linux derivation: the Nix clang already links the system `/usr/lib/libc++.1.dylib`, so the only store load paths are `libzstd` (removed by the same static-archive link trick the Linux leg uses) and `libiconv` (rewritten to `/usr/lib` exactly as `release-chelisup.nix` already does). The rewritten binary runs its behavior probes inside the build sandbox.

One Nix release derivation for both platforms completes the single-source goal: every shipped artifact comes from the committed `Cargo.nix` graph and the pinned cvc5 tree.

## What Changes

- Generalize `nix/release-chelis.nix` to both supported systems. Darwin stages `chelis-v<version>-darwin-arm64.tar.gz` with the same tree; its portability leg statically links `libzstd`, rewrites the `libiconv` load path to `/usr/lib`, asserts every load command is an Apple system path, and runs the behavior probes on the rewritten binary.
- Expose `outputs.release-chelis` on both systems in `devenv/release-outputs.nix`.
- Extend `scripts/verify_release_chelis.py` with the platform dimension and Mach-O checks.
- Rewire `release.yml`: `build-chelis-release` becomes a two-platform matrix, a Darwin consumption job runs the Accelerate smoke on the exact staged tarball, and the Cargo `build-darwin-arm64` job retires.
- The cold cargo-from-source cvc5 recipe proof moves entirely to `smt-full-prove.yml`'s cold path; the docs say so.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `portable-chelis-release`: The release output, inventory, portability, probe, and workflow requirements gain the Darwin platform; the Cargo Darwin release job requirement is removed. (This capability is introduced by the active `switch-linux-release-to-nix` change; this change sequences after it.)
- `portable-chelisup-release`: The release-workflow requirement drops its reference to a Cargo-built Darwin full toolchain job.

## Impact

The change affects `nix/release-chelis.nix`, `devenv/release-outputs.nix`, `scripts/verify_release_chelis.py` and its tests, `scripts/test_release_chelis_output.py`, `scripts/test_release_runtime_header_manifest.py`, `scripts/test_ci_cvc5_build.py`, `scripts/ci_cvc5_build.py` (retires with its last caller), `.github/workflows/release.yml`, `docs/smt_build_setup.md`, and the changelog.

No compiler, runtime, CLI, backend, or generated-code behavior changes. The Darwin asset keeps the exact name `chelisup` requests: `chelis-v<version>-darwin-arm64.tar.gz`.

The authoritative completion oracle is one green manual dispatch of the release workflow at a branch with both platform tarballs produced by the Devenv output and both consumption jobs green.
