## Context

`converge-smt-lanes-on-nix-cvc5` moved the two SMT smoke lanes onto the Devenv toolchain and the flake's pinned `cvc5-dir`, and deferred `smt-full-prove.yml` for two reasons: a host cargo link against a Nix-gcc `libcvc5.a` risks unresolved `GLIBCXX`/`CXXABI` versions, and the lane's solver features need system libraries (z3, GMP, Gappa, m4/make for the vendored Arb/GMP builds) whose Devenv wiring was unverified.

Local spikes (2026-08-08, this workstation, inside `devenv shell`) resolved both:

- `cargo build -p chelis-cli --features smt` with `CVC5_DIR` pointing at the flake's `cvc5-dir` completes in ~90 seconds. bindgen finds libclang from the Devenv Rust module. The mixing hazard disappears when cargo itself runs inside the Devenv shell.
- `cargo build`/`cargo test -p chelis-prove --features z3` (390 tests) passes against the Nix `z3` with `Z3_SYS_Z3_HEADER=${z3.dev}/include/z3.h` and `Z3_LIBRARY_PATH_OVERRIDE=${z3.lib}/lib`. On Darwin the Nix `libz3.dylib` carries an absolute install name, so the tests also pass with no loader variable.
- The full corpus mirror on this workstation: the smt suite, the serialized carcara suite, z3, the cross-engine oracle, clarabel, the combined `smt clarabel` config, and the Gappa `--check-only` re-validation (69 proofs) all pass inside `devenv shell`. The vendored GMP/MPFR/FLINT C stack compiles; the `arb` feature itself is Darwin-incapable upstream (see Decisions), so the two Arb steps are Linux-only evidence.

## Goals / Non-Goals

**Goals:**

- Every CI lane that builds Chelis code uses the pinned Devenv toolchain.
- All three SMT lanes link one cvc5: the flake's pinned tree.
- The harvest-cycle cache machinery retires completely.
- Developers get the solver stack in the Devenv shell (replacing the per-machine apt/brew recipes for z3, Gappa, and carcara's GMP).

**Non-Goals:**

- No change to the lane's corpus, schedule, timeout, or reporter.
- No change to `docs/local_z3_environment.md`'s non-Devenv workstation recipe beyond a pointer; `scripts/z3_test.py` keeps honoring an explicit `Z3_LIBRARY_PATH_OVERRIDE`, which Devenv now supplies.
- The cargo-from-source cvc5 recipe keeps its documentation but loses CI execution (see Decisions).

## Decisions

### The solver stack lives in the Devenv shell

`devenv/toolchains.nix` adds `z3`, `gappa`, `gmp`, `gnum4`, and `gnumake` to the common package set, plus:

- `Z3_SYS_Z3_HEADER` and `Z3_LIBRARY_PATH_OVERRIDE` for z3-sys's prebuilt-link branch (the same variables `scripts/z3_test.py` honors first).
- `CPATH`/`LIBRARY_PATH` entries for GMP, so carcara's `gmp-mpfr-sys/use-system-libs` probe finds the headers without a per-machine export (the macOS brew recipe in `docs/smt_build_setup.md` becomes unnecessary inside Devenv).

Runtime resolution: on Darwin the Nix `libz3.dylib` carries an absolute install name, so no loader variable is needed. On Linux the z3 test steps export `LD_LIBRARY_PATH="$Z3_LIBRARY_PATH_OVERRIDE"` inside the `devenv shell` command; a global `LD_LIBRARY_PATH` in the shell would shadow system libraries for every process and is deliberately avoided.

### The lane converts to the smoke-lane shape

`full-smt-prove` adopts the exact converted block: pinned `setup-devenv`, private-ci authentication, the `cvc5-dir` closure cache shared with `nix-packages.yml` and the smoke lanes, `nix build` + `CVC5_DIR` export, `Swatinem/rust-cache` with the existing `smt-smt-build` shared key, and every build/test/certify command through `devenv shell`. The Gappa re-validation runs the Devenv Python (`devenv shell -- python scripts/generate_erf_proof.py --check-only`) with the `gappa` binary on the shell PATH.

### The arb feature is Linux-only, upstream

`arb-sys` 0.3.6 invokes the vendored Arb `configure` with only `--with-gmp` and `--with-flint`. The configure script's MPFR check then tests `${MPFR_DIR}/lib` with an empty `MPFR_DIR`: on Linux `-d /lib` accidentally passes and MPFR resolves from the gmp-mpfr-sys output directory through the GMP flags, while macOS has no `/lib`, so the check always fails (`Invalid MPFR directory`). This predates the conversion - the arb lane has only ever run on Linux CI - and the converted Linux lane keeps the same accidental-pass shape with MPFR supplied by the vendored gmp-mpfr-sys build. A pinned upstream fix (passing `--with-mpfr`) is the clean remedy if Darwin arb ever matters.

### The cargo-from-source cvc5 proof retires deliberately

After this change no CI lane compiles cvc5 through `cvc5-sys`. The recipe remains documented in `docs/smt_build_setup.md` for contributors outside Devenv, and `cvc5-sys` itself is unchanged, but CI proves only the pinned Nix supply. Accepted on the zero-users footing: the from-source path is a contributor convenience, not a shipped artifact, and a regression in it is discoverable locally. Recorded in the docs; revisit if a non-Devenv contributor path becomes supported.

### The cache machinery retires completely

`scripts/ci_cvc5_cache.py` and its test suite leave with their last caller. The `cvc5-prebuilt-` protected prefix in `cache-prune.yml` stays: `adopt-shared-ci-actions` pins it as data, and protecting a now-absent key family is harmless.

## Risks / Trade-offs

- **The Devenv closure grows for every developer** (~z3 + gappa + gmp) → Accepted; these are the project's own solver dependencies, and the per-machine recipes they replace were a recurring setup cost.
- **The lane's cold path now depends on the cvc5 closure cache** → A cold run pays one ~30m Nix cvc5 build inside the 75m ceiling; the cache is shared with three other jobs, so cold runs are rare.
- **`generate_erf_proof.py --check-only` under the Devenv Python** → Validated locally before landing; its imports are stdlib plus the repository.
- **carcara's GMP probe under the Nix cc** → Validated locally via the `CPATH`/`LIBRARY_PATH` wiring before landing; the serial-run SIGSEGV containment (`--test-threads=1`) is unchanged.
- **The Arb configure on the converted Linux lane** → The accidental-pass shape (empty `MPFR_DIR` against an existing `/lib`, MPFR from the gmp-mpfr-sys output) is unchanged from the host lane; the hosted dispatch is the executable proof, and a failure is loud at the configure step.

## Migration Plan

1. Add the solver stack and environment wiring to `devenv/toolchains.nix`; validate carcara, clarabel, arb, and the Gappa check locally inside the shell.
2. Rewrite `smt-full-prove.yml`; delete the cache machinery and its tests.
3. Update `scripts/test_smt_lane_workflows.py` (full-prove joins the smoke-lane contract; retirement extends to the cache script) and `scripts/test_gate.py` (`DEVENV_WORKFLOW_JOBS` gains the lane).
4. Update `docs/smt_build_setup.md` and the changelog.
5. Run the scripts suite and the local gate; dispatch `SMT Full Prove (Linux)` once as the completion oracle.

Rollback restores the lane and the cache script from git history.

## Open Questions

- None blocking. The Linux `LD_LIBRARY_PATH` step wiring is exercised only by the hosted dispatch; the failure mode is loud (link or dlopen error).
