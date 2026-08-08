## 1. Run the confirming spike on x86_64-linux

- [x] 1.1 Build `.#chelis` and `.#chelis-runtime` and record the compiler binary's `NEEDED` set, interpreter, and maximum `GLIBC_*` verneed. (Recorded in design.md §Spike Results: floor 2.39; baseline `NEEDED` includes `libstdc++.so.6` and `libzstd.so.1`.)
- [x] 1.2 Prove the static C++/zstd link through the crate2nix release build. (`-static-libstdc++` alone is insufficient against cvc5-sys's explicit `-lstdc++`; the static-archive `-Lnative` directory through `crateOverrides.extraRustcOpts` works. Spike expression: `.work/spike-release-portability.nix`.)
- [x] 1.3 Rewrite one binary with `patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 --remove-rpath` and confirm zero `/nix/store` references remain. (Confirmed; the rewritten binary runs through the explicit glibc loader and prints `chelis 0.18.4`.)
- [ ] 1.4 In a clean container without Nix, run the rewritten binary, compile emitted C with `gcc` against the archive, and execute the program. (Approximated in-sandbox with the explicit glibc loader; the full container run lands with the release workflow's consumption job.)
- [x] 1.5 Record the measured glibc floor and select a digest-pinned consumption image with a glibc at or above it. (Floor 2.39; image candidate: digest-pinned Ubuntu 24.04, glibc 2.39.)
- [ ] 1.6 Check the HIP workstation's glibc against the measured floor and record the result.
- [ ] 1.7 If task 1.4 fails, stop and re-open the design against the musl-binary fallback. (Not triggered; in-sandbox evidence is positive.)

## 2. Add failing contract tests first

- [x] 2.1 Add static release-output tests for `outputs.release-chelis`: committed-graph use, no flake evaluation, no import-from-derivation, no host Cargo, no musl runtime target. (`scripts/test_release_chelis_output.py`.)
- [x] 2.2 Add release workflow tests: no `cargo build` for published Linux artifacts, no distribution container build, no `-glibc2.31` asset, `chelisup.sh` staged from the Linux job, publish dependency on the off-Nix job. (`ReleaseChelisWorkflowContractTests` with per-marker mutations.)
- [x] 2.3 Add the recorded-floor consistency test: the committed floor does not exceed the consumption image's glibc. (`RecordedFloorContractTests`; both values live in `nix/contracts.nix`.)
- [x] 2.4 Add negative mutations for each release-output and floor test and confirm that each fails for its intended reason. (Workflow mutations land with task 2.2.)

## 3. Land the release derivation

- [x] 3.1 Add `nix/release-chelis.nix` from the `release-chelisup.nix` template: crate builds, staging layout, tarball, sidecar. (Full derivation built green on `x86_64-linux`; exact output inventory verified.)
- [x] 3.2 Add the pre-rewrite behavior probes: `--version`, `--help`, the release fixture, and `verify_release_smt.py`. (All passed in-derivation, including the cvc5 discharge record.)
- [x] 3.3 Add the portability rewrite and the static ELF postconditions: interpreter, no rpath/runpath, no `/nix/store`, `NEEDED` allowlist (glibc members plus `libgcc_s.so.1` per the spec), no `libstdc++.so.6`/`libzstd.so.1`, and the explicit-loader run proof.
- [x] 3.4 Add the recorded glibc floor to `nix/contracts.nix` (`2.39`) and the in-derivation comparison that fails on drift in either direction.
- [x] 3.5 Add the archive negative control: no defined `malloc`, `free`, `calloc`, or `realloc` in `libchelis_runtime.a`.
- [x] 3.6 Expose `outputs.release-chelis` in `devenv/release-outputs.nix` for `x86_64-linux`. (Darwin evaluation verified: output absent there, existing outputs unchanged.)
- [x] 3.7 Run `nixfmt` on every touched Nix file.

## 4. Rewire the release workflow

- [x] 4.1 Add the `build-chelis-release` Linux job: shared Devenv setup, private-ci authentication, `devenv build --no-tui outputs.release-chelis`, tag-parity verification (`scripts/verify_release_chelis.py` + `scripts/test_verify_release_chelis.py`), artifact upload.
- [x] 4.2 Add the off-Nix consumption job (`consume-chelis-release`) on the staged tarball: digest-pinned `ubuntu:24.04` (glibc 2.39), `verify_consumption_glibc.py` contract tie, `verify_release_smt.py --tarball`, and `smoke_linux_openblas.py` (emitted C + `gcc` + OpenBLAS).
- [x] 4.3 Move `chelisup.sh` staging into the new Linux job.
- [x] 4.4 Delete the `build-linux-x86_64` and `build-linux-x86_64-glibc231` jobs.
- [x] 4.5 Update `publish-release` dependencies and asset globs; manual `v*`-tag dispatch and publication conditions unchanged.
- [x] 4.6 Remove the `smt-build-glibc231` lane from `ci.yml`; the plain `smt-build` lane stays. (Also retired the lane's supply chain: the `build-linux-glibc231` job in `build-cvc5.yml` and the `linux-glibc231` namespace in `scripts/ci_cvc5_cache.py`, with test updates.)

## 5. Update documentation

- [x] 5.1 Update `docs/smt_build_setup.md`: glibc231 lanes removed, Nix release path described, publish dependency wording updated.
- [x] 5.2 Update `docs/book/src/install.md` with the recorded Linux glibc floor (2.39) and its contract location.
- [x] 5.3 Record the deferred floor-policy decision and the retrofit paths. (design.md §Deferred low-floor retrofit paths; CHANGELOG Removed entry; superseded note in `spec/design/phase3j_pre_release.md`.)
- [x] 5.4 Update the changelog.

## 6. Validate and accept

- [x] 6.1 Run the workflow and release-output contract tests and confirm every mutation still fails correctly. (All release, verify, cvc5-cache, and gate suites green; mutations are self-verifying subTests.)
- [ ] 6.2 Run `devenv test --no-tui` and the Nix package check set on `x86_64-linux`.
- [x] 6.3 Run `python3 scripts/gate.py --local`. (Success, 2026-08-08: clippy, fmt, lint, doctests, checkpoint fixture, and the changed-crate nextest stage all green.)
- [ ] 6.4 Dispatch the release workflow at a branch and require every job green, including the off-Nix consumption job. This dispatch is the authoritative completion oracle.
- [ ] 6.5 Run the HIP manual gate against the shipped archive from the dispatched tarball and record the result.
- [ ] 6.6 Run `openspec validate --all --strict --no-interactive` and resolve findings.
- [ ] 6.7 Spawn a fresh local red-team subagent per the repository protocol before closing the phase.
