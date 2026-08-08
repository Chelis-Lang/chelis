## Why

`smt-full-prove.yml` is the last CI lane on a host toolchain: floating stable rustc, apt solver packages, and a self-harvested cvc5 through `scripts/ci_cvc5_cache.py`. `converge-smt-lanes-on-nix-cvc5` deferred it because a host cargo link against a Nix-built `libcvc5.a` risks a C++ ABI mismatch and the z3/Gappa/Arb system wiring was unverified.

Both blockers are now resolved by local spikes (2026-08-08, inside the Devenv shell): the smt build links the flake's `cvc5-dir` through `CVC5_DIR` with the Devenv libclang, and the z3 feature builds and passes its test suite against the Nix `z3` through `Z3_SYS_Z3_HEADER` and `Z3_LIBRARY_PATH_OVERRIDE`. Converting the lane removes the last floating toolchain, retires `ci_cvc5_cache.py` entirely, and gives every developer the solver stack (z3, Gappa, GMP, m4) in the Devenv shell instead of a per-machine recipe.

## What Changes

- Add the solver stack to the Devenv shell: `z3`, `gappa`, `gmp`, `gnum4`, and `gnumake`, plus the `Z3_SYS_Z3_HEADER`, `Z3_LIBRARY_PATH_OVERRIDE`, and GMP `CPATH`/`LIBRARY_PATH` environment wiring that the z3 and carcara features need.
- Rewrite `smt-full-prove.yml` on the converted smoke-lane shape: pinned Devenv setup, private-ci authentication, the shared `cvc5-dir` closure cache, `CVC5_DIR` from the flake, and every command through `devenv shell`. The apt, floating-rustc, host-uv, and cvc5 harvest-cycle steps leave.
- Retire `scripts/ci_cvc5_cache.py` and `scripts/test_ci_cvc5_cache.py`: the harvest cycle loses its last caller.
- The cargo-from-source cvc5 recipe loses its last CI execution. The recipe stays documented for non-Devenv contributors; the pinned Nix build is the proven supply everywhere. The docs say this deliberately.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `smt-lane-cvc5-supply`: The full-prove lane joins the pinned-supply and Devenv-toolchain requirements; the toolchain-mixing prohibition and the harvest-cycle allowance are replaced by full convergence. (This capability is introduced by the active `converge-smt-lanes-on-nix-cvc5` change; this change sequences after it.)

## Impact

The change affects `devenv/toolchains.nix`, `.github/workflows/smt-full-prove.yml`, `scripts/ci_cvc5_cache.py` and `scripts/test_ci_cvc5_cache.py` (deleted), `scripts/test_smt_lane_workflows.py`, `scripts/test_gate.py`, `docs/smt_build_setup.md`, `docs/local_z3_environment.md` pointers, and the changelog.

No compiler, runtime, CLI, backend, package-output, or generated-code behavior changes. The lane keeps its name, nightly schedule, manual dispatch, timeout, and tracking-issue reporter.

The authoritative completion oracle is one green manual dispatch of `SMT Full Prove (Linux)` with its log showing the Devenv toolchain and the Nix store `CVC5_DIR`.
