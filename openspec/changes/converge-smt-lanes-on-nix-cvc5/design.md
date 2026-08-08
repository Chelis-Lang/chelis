## Context

Three CI lanes build `chelis-cli --features smt`: the required `smt-build` (Linux), the manual `smt-build-darwin-arm64`, and the nightly/manual `smt-full-prove.yml`. All three obtain cvc5 through a two-store prebuilt system: a durable Release asset published by `build-cvc5.yml`, with an Actions-cache fallback, driven by `scripts/ci_cvc5_cache.py`.

Since `switch-linux-release-to-nix`, the shipped Linux compiler links the Nix-pinned cvc5 tree from `nix/cvc5.nix`. Its `dir` output is byte-shaped exactly like the `CVC5_DIR` tree `cvc5-sys` consumes (that is what `nix/packages.nix` and `nix/release-chelis.nix` pass to the crate). The `nix-packages` CI job already computes the `cvc5-dir` derivation key and caches its closure in the GitHub Actions cache.

## Goals / Non-Goals

**Goals:**

- The required SMT smoke lanes build with the same pinned Rust toolchain and C toolchain as the gate and the converted CI jobs.
- The per-PR SMT proof links the same cvc5 tree the shipped Linux binary links.
- The harvested-prebuilt producer and its publish path retire.
- Required context names, triggers, and docs-only gating stay unchanged.

**Non-Goals:**

- The change does not convert `smt-full-prove.yml` to the Devenv toolchain. Its z3, Gappa, m4, and Arb system dependencies need their own verification pass, and a host cargo link against a Nix-gcc-compiled `libcvc5.a` is an ABI risk (see Decisions). A follow-up change owns that conversion.
- The change does not alter the local developer cargo path. `cvc5-sys` still builds cvc5 from source when `CVC5_DIR` is absent.
- The change does not add cvc5 to the Devenv shell environment. Exporting `CVC5_DIR` at shell entry would make first shell entry build cvc5; a follow-up can add an opt-in command.
- The change does not touch the Darwin release job (owned by the Darwin release follow-up).

## Decisions

### The lanes take cvc5 from the flake, not from a harvested artifact

`smt-build` and `smt-build-darwin-arm64` run, per system:

1. the pinned `setup-devenv` and private-ci authentication (the same block every converted job uses),
2. the `cvc5-dir` closure cache restore/save steps copied from `nix-packages.yml`, keyed on the `cvc5-dir` derivation path,
3. `nix build .#legacyPackages.<system>.cvc5-dir` with a local out-link, exporting `CVC5_DIR` into `$GITHUB_ENV`,
4. `devenv shell --no-tui -- cargo build -p chelis-cli --features smt`, the cvc5 discharge verifier, and the engine smoke tests, all inside `devenv shell`.

With `CVC5_DIR` set, `cvc5-sys` links the prebuilt archives and never runs CMake, so the apt C-dependency step disappears. bindgen's libclang comes from the Devenv shell.

### Toolchain mixing is the constraint that splits the scope

Linking is consistent only when the compiler that builds the crate and the toolchain that compiled `libcvc5.a` agree on the C++ runtime. Inside `devenv shell`, cargo links with the Nix cc against the Nix libstdc++ - the same toolchain that built the flake's cvc5. A HOST cargo (ubuntu's gcc and libstdc++) linking the Nix-gcc-compiled archive can hit unresolved `GLIBCXX`/`CXXABI` versions. `smt-full-prove` therefore keeps its host toolchain AND its host-built cvc5: it retains the Actions-cache harvest cycle (`key`/`restore`/`activate`/`harvest`/`save`), which is self-sufficient without the durable asset. Only the durable-asset `fetch` step leaves.

### The producer retires; the library stays

`build-cvc5.yml` and `scripts/ci_publish_cvc5_release.py` (plus its test) are deleted: their only consumers were the durable-asset fetch paths. `scripts/ci_cvc5_cache.py` stays for the full-prove harvest cycle; its `fetch`, `pack`, and `plan` subcommands lose their callers and leave with the full-prove follow-up. The cache-prune protected prefix `cvc5-prebuilt-` stays: the full-prove Actions cache still uses it, and `adopt-shared-ci-actions` pins it as data.

### The cold cargo-from-source recipe keeps one proof

After this change, no per-PR lane builds cvc5 from source. The cold recipe stays proven by `smt-full-prove`'s cold-cache path (nightly cadence) and by the Darwin release job's cargo build (until the Darwin release moves to Nix). `docs/smt_build_setup.md` says this explicitly.

### Job identity is frozen

`SMT Feature Build (Linux)` is a required branch-protection context. The job name, `needs: [changes]` docs-only gate, timeout, and `Swatinem/rust-cache` shared key survive the rewrite. The Darwin job keeps its manual-dispatch condition.

## Risks / Trade-offs

- **The devenv-shell cargo build cannot find libclang for bindgen** → The Devenv Rust module exports the libclang path; if a hosted run disproves this, add the explicit env to the job in review. The failure is loud (bindgen abort), never silent.
- **The cvc5-dir closure cache is cold on the first run** → The job pays one ~30m Nix cvc5 build, then the Actions cache and the nix-packages job share the key. Timeout raised for the cold case with a comment.
- **Swatinem cache invalidation from the toolchain switch** → The first converted run recompiles the workspace under the pinned toolchain; subsequent runs are warm. Accepted one-time cost.
- **Retiring the producer breaks a hidden consumer** → Contract tests assert no workflow references the deleted script or workflow; the repository grep in review is the backstop.
- **The full-prove lane rots on its host toolchain** → Recorded as the explicit follow-up; the lane is manual/nightly and non-required.

## Migration Plan

1. Add failing workflow contract tests for the two converted lanes and the retirement.
2. Rewrite `smt-build` and `smt-build-darwin-arm64` in `ci.yml`.
3. Remove the durable-asset fetch step from `smt-full-prove.yml`.
4. Delete `build-cvc5.yml`, `scripts/ci_publish_cvc5_release.py`, `scripts/test_ci_publish_cvc5_release.py`; trim the retired references from docs and tests.
5. Run the scripts suite and the local gate.
6. Land through a PR; the required `SMT Feature Build (Linux)` run on that PR is the completion oracle. Dispatch the Darwin lane once for the manual evidence.

Rollback: restore the deleted workflow and scripts from git history and revert the two job blocks. The cvc5 Release assets remain published under their prerelease tag during any rollback window.

## Open Questions

- Whether the Devenv shell exports a usable libclang path for bindgen on both systems is confirmed by the first hosted run (the local Nix package build proves it for the crate-override path, not the shell path).
