## 1. Add failing contract tests first

- [x] 1.1 Add workflow tests for the two smoke lanes: Devenv setup + auth order, `devenv shell` cargo commands, `CVC5_DIR` from the flake's `cvc5-dir`, closure cache restore/save, and forbidden host-toolchain markers. (`scripts/test_smt_lane_workflows.py`.)
- [x] 1.2 Add retirement tests: no `build-cvc5.yml`, no publish script, no workflow reference to either.
- [x] 1.3 Add the toolchain-mixing test: `smt-full-prove.yml` has no Nix store `CVC5_DIR` while its cargo runs on the host toolchain.
- [x] 1.4 Add negative mutations per marker and confirm that each fails for its intended reason.

## 2. Rewrite the smoke lanes

- [x] 2.1 Rewrite `smt-build`: pinned setup-devenv, private-ci authentication, the `cvc5-dir` closure cache, `outputs.cvc5-dir`, and the `smt` profile. Keep the job name, docs-only gate, `Swatinem/rust-cache` shared key, and the 75-minute cold-closure timeout.
- [x] 2.2 Rewrite `smt-build-darwin-arm64` the same way for `aarch64-darwin`; manual dispatch condition kept.
- [x] 2.3 Remove the durable-asset `fetch` step from `smt-full-prove.yml`; the Actions-cache harvest cycle stays, with an updated lane comment.

## 3. Retire the producer

- [x] 3.1 Delete `.github/workflows/build-cvc5.yml`.
- [x] 3.2 Delete `scripts/ci_publish_cvc5_release.py` and `scripts/test_ci_publish_cvc5_release.py`.
- [x] 3.3 Trim retired references from `scripts/ci_cvc5_cache.py` docs text; the module records that `fetch`/`pack`/`plan` leave with the full-prove follow-up. (`scripts/ci_cvc5_build.py` docstring updated too.)
- [x] 3.4 Update `scripts/test_gate.py`: producer removed from the workflow scope list and the macOS job census; both smoke lanes added to `DEVENV_WORKFLOW_JOBS`.

## 4. Update documentation

- [x] 4.1 Rewrite the "Durable prebuilt cvc5" section of `docs/smt_build_setup.md` as "The pinned cvc5 supply"; CI Configuration and companion-lane paragraphs updated; the retired cvc5 Darwin asset gate removed from `docs/manual_gates.md`.
- [x] 4.2 Update the changelog.

## 5. Validate and accept

- [x] 5.1 Run the scripts test suite and confirm every mutation fails correctly.
- [x] 5.2 Run `python3 scripts/gate.py --local`.
- [ ] 5.3 Land through a pull request; the required `SMT Feature Build (Linux)` run with a Nix store `CVC5_DIR` in its log is the authoritative completion oracle.
- [ ] 5.4 Dispatch `smt-build-darwin-arm64` once and record the result.
- [ ] 5.5 File the follow-up issue for the `smt-full-prove` Devenv conversion (z3, Gappa, m4, Arb wiring) and the `ci_cvc5_cache.py` trim.
