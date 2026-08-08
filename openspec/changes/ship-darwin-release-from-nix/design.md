## Context

The Linux release tarball ships from `nix/release-chelis.nix` (`switch-linux-release-to-nix`). The Darwin tarball still ships from a Cargo job: floating stable rustc, brew cmake, a cold cvc5 source build through `scripts/ci_cvc5_build.py`, and the Accelerate smoke.

Spike (2026-08-08, this workstation, `.work/spike-release-darwin.nix`): the flake's `aarch64-darwin` compiler links `/usr/lib/libc++.1.dylib` (the Nix clang uses the system C++ runtime on Darwin), Apple frameworks, and exactly two store paths - `libzstd.1.5.7.dylib` and `libiconv.2.dylib`. With the static-archive `-Lnative` directory holding only `libzstd.a`, zstd links statically. `install_name_tool -change` moves iconv to `/usr/lib` (the `release-chelisup.nix` precedent), the tool re-signs ad-hoc, and the rewritten binary passes `--version`/`--help` inside the sandbox.

## Goals / Non-Goals

**Goals:**

- Both release tarballs come from one Nix derivation over the committed graph and the pinned cvc5 tree.
- The Darwin portability contract is executable: Apple-only load commands, no `/nix/store` path, probes on the rewritten binary.
- The exact staged Darwin tarball is consumed off-Nix on a stock macOS runner, including the Accelerate smoke.

**Non-Goals:**

- No change to the flake packages, the chelisup release output, or the Linux release leg's contracts.
- No macOS version floor contract in this change. The binary records `minos` from the pinned SDK; a recorded-floor ratchet like the Linux glibc one can follow if it ever moves.
- No change to the smt lanes (owned by `converge-smt-lanes-on-nix-cvc5`).

## Decisions

### One derivation, a platform-conditional portability leg

`nix/release-chelis.nix` keeps one staging and inventory pipeline and switches only the portability leg:

- **Linux:** static `libstdc++` + `libzstd`, strip, patchelf interpreter rewrite, ELF postconditions, recorded glibc floor, explicit-loader run (unchanged).
- **Darwin:** static `libzstd`, `strip -x`, `install_name_tool` iconv rewrite, Mach-O postconditions (arm64; every load command under `/usr/lib/` or `/System/Library/Frameworks/`; no `/nix/store`), and the behavior probes re-run on the REWRITTEN binary - the Darwin sandbox can execute it, so Darwin gets a stronger in-derivation proof than Linux.

The pre-rewrite probes (version, help, release fixture, cvc5 discharge) stay common to both platforms.

### The Darwin consumption job is natively off-Nix

`macos-latest` runners carry no Nix, so the consumption job needs no container: download the staged tarball, unpack, run `bin/chelis --version`, `verify_release_smt.py --tarball`, and `.github/scripts/smoke_macos_accelerate.py` against the unpacked binary (clang + Accelerate come with the image). This also keeps the Accelerate smoke coverage the retiring Cargo job carried.

### The Cargo Darwin job retires, and the cold-recipe proof moves

`build-darwin-arm64` in `release.yml` was the last cargo-from-source cvc5 build in the repository, and the docs name it the macOS cold-recipe proof. After it retires, the standing cargo-from-source proof is `smt-full-prove.yml`'s cold path (Linux, nightly cadence). `scripts/ci_cvc5_build.py` and `scripts/test_ci_cvc5_build.py` retire with their last caller. Accepted trade-off, recorded in `docs/smt_build_setup.md`: macOS contributors on the plain Cargo path lose CI coverage of the cold cvc5 build; the recipe remains documented, and Devenv is the recommended environment.

### Workflow shape

`build-chelis-release` becomes a matrix over `linux-x86_64` (ubuntu-latest) and `darwin-arm64` (macos-latest), mirroring `build-chelisup-release`. The Linux consumption job keeps its pinned container; the Darwin consumption job runs on the bare runner. `publish-release` depends on the matrix and both consumption jobs. Manual `v*`-tag dispatch and publication policy are unchanged.

## Risks / Trade-offs

- **The system-libc++ linkage is toolchain-dependent** → The otool postcondition fails the derivation the moment a nixpkgs bump links a store libc++; the fallback is a static `libc++.a` link directory, same mechanism as zstd.
- **`install_name_tool` signature invalidation** → The spike shows the ad-hoc re-sign happens implicitly and the binary runs in-sandbox; the in-derivation probe run is the executable guard.
- **Losing the macOS cold cargo cvc5 proof** → Accepted and recorded; revisit if a macOS cargo-path regression ever bites.
- **Matrix conversion breaks the Linux contract tests** → The Linux job markers move into matrix form; `scripts/test_release_chelis_output.py` is updated in the same change set with mutations.

## Migration Plan

1. Generalize `nix/release-chelis.nix`; build both platform outputs (Darwin locally, Linux on the builder).
2. Expose the output on both systems; update the release-output contract tests.
3. Extend `scripts/verify_release_chelis.py` with `--platform` and Mach-O checks; update its unit tests.
4. Rewire `release.yml` (matrix, Darwin consumption, retire the Cargo job and `ci_cvc5_build.py`); update the workflow contract tests and docs.
5. Run the scripts suite and the local gate; dispatch the release workflow at a branch as the completion oracle.

Rollback restores the Cargo Darwin job from git history; the Linux leg is untouched.

## Open Questions

- None blocking. The `minos`/SDK recorded-floor ratchet is deferred until the value matters.
