## 1. Wire the solver stack into Devenv

- [x] 1.1 Add `z3`, `gappa`, `gmp`, `gnum4`, and `gnumake` to the Devenv package set; add `Z3_SYS_Z3_HEADER`, `Z3_LIBRARY_PATH_OVERRIDE`, and the GMP `CPATH`/`LIBRARY_PATH` wiring.
- [x] 1.2 Validate locally inside the shell: smt build with `CVC5_DIR` (1m28s, no CMake), `--features z3` tests (390 passed), `--features carcara` build, `--features clarabel` build, `--features arb` build (vendored FLINT/Arb), and `python scripts/generate_erf_proof.py --check-only` with the shell's `gappa`. All green, 2026-08-08.
- [x] 1.3 Run `nixfmt` on every touched Nix file.

## 2. Convert the lane

- [x] 2.1 Rewrite `full-smt-prove` on the smoke-lane shape: Devenv setup, auth, `cvc5-dir` closure cache, `CVC5_DIR` export, `devenv shell` for every command; name, schedule, timeout, `smt-smt-build` shared key, and reporter kept.
- [x] 2.2 Export `LD_LIBRARY_PATH="$Z3_LIBRARY_PATH_OVERRIDE"` inside the z3 and cross-engine steps.
- [x] 2.3 Delete `scripts/ci_cvc5_cache.py` and `scripts/test_ci_cvc5_cache.py`.

## 3. Update the contract surfaces

- [x] 3.1 Extend `scripts/test_smt_lane_workflows.py`: `full-smt-prove` joins the supply/toolchain contract; the retirement checks cover the cache script; mutations updated.
- [x] 3.2 Add `smt-full-prove.yml`/`full-smt-prove` to `DEVENV_WORKFLOW_JOBS` in `scripts/test_gate.py`; update the carcara canonical-command oracle to the devenv-prefixed form.

## 4. Documentation

- [x] 4.1 Update `docs/smt_build_setup.md`: all three lanes on the pinned supply; the harvest cycle and its script retired; the cargo-from-source recipe documented but no longer CI-executed; the carcara brew recipe superseded inside Devenv. (`scripts/ci_cache_prune.py`'s protected-prefix comment updated too.)
- [x] 4.2 Update the changelog.

## 5. Validate and accept

- [ ] 5.1 Run the scripts suite; confirm every mutation fails correctly.
- [ ] 5.2 Run `python3 scripts/gate.py --local`.
- [ ] 5.3 Dispatch `SMT Full Prove (Linux)` once; a green run with the Nix store `CVC5_DIR` in its log is the authoritative completion oracle.
