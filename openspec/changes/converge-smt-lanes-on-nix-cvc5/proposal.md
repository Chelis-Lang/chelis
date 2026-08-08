## Why

The per-PR SMT smoke lanes build with a floating `dtolnay/rust-toolchain@stable`, apt packages of the day, and a harvested prebuilt cvc5 from a two-store cache system (`build-cvc5.yml` plus `scripts/ci_cvc5_cache.py`). The gate and every converted CI job build with the repository's pinned toolchain instead, so the required SMT context tests a compiler and a cvc5 supply that nothing else uses.

After `switch-linux-release-to-nix`, the shipped Linux binary links the Nix-pinned cvc5 1.3.1 tree (`nix/cvc5.nix`). The per-PR SMT proof should link the same tree: the flake's `cvc5-dir` already has the exact `CVC5_DIR` shape that `cvc5-sys` consumes, and CI already caches its closure for the Nix package job.

The harvested-prebuilt producer then loses its consumers. Retiring it removes a scheduled workflow, a publish script, and the durable-asset half of the cache machinery.

## What Changes

- Rebuild `smt-build` and `smt-build-darwin-arm64` on the Devenv toolchain. Use pinned `setup-devenv`, private-ci authentication, and the `smt` profile for each Cargo command. Remove apt, `dtolnay/rust-toolchain`, and host uv.
- Build Devenv `outputs.cvc5-dir` in both jobs. Restore the same derivation through the GitHub Actions closure cache.
- Keep required context names, triggers, docs-only gating, timeouts, and the `Swatinem/rust-cache` cargo cache.
- Keep `smt-full-prove.yml` on its host toolchain: mixing a host `cargo` link with a Nix-gcc-compiled `libcvc5.a` risks a libstdc++ ABI mismatch. The lane keeps the self-sufficient Actions-cache harvest cycle and loses only the durable-asset fetch. Its cold path remains the cargo-from-source cvc5 recipe proof.
- Retire the producer: delete `build-cvc5.yml` and `scripts/ci_publish_cvc5_release.py` with its tests. `scripts/ci_cvc5_cache.py` stays for the full-prove lane's local harvest cycle.
- Update `docs/smt_build_setup.md` and the changelog.

## Capabilities

### New Capabilities

- `smt-lane-cvc5-supply`: Defines where CI SMT lanes obtain cvc5, which toolchain the required SMT smoke lanes build with, and the retirement of the harvested-prebuilt producer.

### Modified Capabilities

None. The flake already exports `cvc5-dir` under `legacyPackages`, and no package output changes.

## Impact

The change affects `.github/workflows/ci.yml`, `.github/workflows/smt-full-prove.yml`, `.github/workflows/build-cvc5.yml` (deleted), `scripts/ci_publish_cvc5_release.py` and `scripts/test_ci_publish_cvc5_release.py` (deleted), `scripts/ci_cvc5_cache.py` (durable-asset path trimmed), contract tests, `docs/smt_build_setup.md`, and the changelog.

No compiler, runtime, CLI, backend, package-output, or generated-code behavior changes. The required context `SMT Feature Build (Linux)` keeps its name and trigger conditions.

The authoritative completion oracle is a green required `SMT Feature Build (Linux)` check on the consumer pull request, with its log showing the pinned toolchain and the Nix store `CVC5_DIR`. A manual dispatch of `smt-build-darwin-arm64` provides the Darwin evidence.
