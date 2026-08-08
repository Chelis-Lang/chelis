## 1. Generalize the release derivation

- [x] 1.1 Make `nix/release-chelis.nix` platform-conditional: shared staging/inventory pipeline; Linux keeps its ELF leg; Darwin adds static `libzstd`, the `install_name_tool` iconv rewrite, Apple-only load-command postconditions, and behavior probes on the rewritten binary. (The crate output is already stripped by the Nix fixup phase, so no separate `strip -x` step is needed.)
- [x] 1.2 Expose `outputs.release-chelis` on both systems in `devenv/release-outputs.nix`.
- [ ] 1.3 Build the Darwin output locally and the Linux output on the builder; confirm both pass their in-derivation checks. (Darwin: green locally, plus a full off-store consumption run -- unpacked tarball, cvc5 discharge, Accelerate smoke. Linux rebuild: in flight.)
- [x] 1.4 Run `nixfmt` on every touched Nix file.

## 2. Update the verification and contract surfaces

- [x] 2.1 Extend `scripts/verify_release_chelis.py` with `--platform` (asset naming, inventory, tag parity per slug) and update `scripts/test_verify_release_chelis.py`.
- [x] 2.2 Update `scripts/test_release_chelis_output.py`: module markers for both systems, Darwin helper markers (`install_name_tool`, `/usr/lib/libiconv.2.dylib`, `otool -L`), matrix workflow markers, the Darwin consumption job markers, and mutations.
- [x] 2.3 Update `scripts/test_release_runtime_header_manifest.py`: zero workflow staging blocks; the manifest comes from `nix/contracts.nix`.

## 3. Rewire the release workflow

- [x] 3.1 Convert `build-chelis-release` to a two-platform matrix mirroring `build-chelisup-release`; the Linux leg keeps `chelisup.sh` staging and the tag-parity check runs per leg.
- [x] 3.2 Add the Darwin consumption job on the stock macOS runner: unpack, `--version`/`--help`, `verify_release_smt.py --tarball`, and `smoke_macos_accelerate.py` against the unpacked binary.
- [x] 3.3 Delete the Cargo `build-darwin-arm64` job; retire `scripts/ci_cvc5_build.py` and `scripts/test_ci_cvc5_build.py` with their last caller. (Also retired `scripts/test_release_workflow_pyo3_isolation.py`: release.yml runs no host Cargo, so its invariant lost its subject.)
- [x] 3.4 Update `publish-release` dependencies; manual `v*`-tag dispatch and publication conditions unchanged.
- [x] 3.5 Update the workflow contract tests (`test_release_chelis_output.py`, `test_gate.py` macOS job census) with mutations.

## 4. Update documentation

- [x] 4.1 Update `docs/smt_build_setup.md`: both release artifacts ship from the Nix output; the cold cargo-from-source cvc5 proof lives in `smt-full-prove.yml`'s cold path; the chelis#1004 fetch-retry section retired with its wrapper.
- [x] 4.2 Update the changelog. (`docs/manual_gates.md` needed no change: its macOS release gate rows reference the workflow dispatch, not the retired job.)

## 5. Validate and accept

- [ ] 5.1 Run the scripts test suite; confirm every mutation fails correctly.
- [ ] 5.2 Run `python3 scripts/gate.py --local`.
- [ ] 5.3 Dispatch the release workflow at a branch; every job green, including both consumption jobs, is the authoritative completion oracle.
- [ ] 5.4 Run `openspec validate --all --strict --no-interactive` and resolve findings.
