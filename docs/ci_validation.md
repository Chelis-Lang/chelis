# CI validation cadence

Ordinary PRs and main pushes use Linux. Passing required PR checks is **not a phase acceptance result** for a full or feature-specific oracle that runs nightly.

| Owner | Cadence | Coverage |
|---|---|---|
| `ci.yml` `ci-fast` | PR and main push, with the existing docs-only skip | Every default-feature library/binary unit target and the 17 package/target identities in `.config/ci-test-targets.toml`; 20-minute limit |
| `ci.yml` retained workers | PR and main push | Rust policy and doctests, Python/script units, focused SMT plus its existing Deep-obligation integration target, Linux glibc compatibility, Docs, backend sanitizer units and explicit backend doctests; change-triggered diagnostic mutation and rejection liveness |
| `conformance.yml` | PR and main push | Existing frozen Hull conformance gate |
| `heavy-e2e.yml` | Daily 03:17 UTC and manual dispatch | Unsharded full non-ignored default workspace, exhaustive generalization feature partitions, dtype Phases 0–3, faithful observation Phase 2, ownership Phase 2 and launch, runtime representation Phase 0, frontend/domain support, and full backend sanitizer integration coverage |
| `macos-nightly.yml` | Daily 04:17 UTC and manual dispatch | Both Mac workspace partitions, both Clippy configurations, architecture/ABI/Metal smokes, and Darwin SMT |
| `build-cvc5.yml` Darwin producer | Daily 01:17 UTC and manual dispatch | Missing Darwin prebuilt assets; relevant main pushes may produce Linux assets only |
| `smt-full-prove.yml` | Existing nightly/manual cadence | Full SMT proof validation |
| `release.yml` and release-triggered Nix checks | Release/manual cadence | Existing shipping-artifact validation, including Mac |

The full Linux nightly passes `--ignore-default-filter` and deliberately includes the PR selection. A newly added integration target is automatically covered there; entering the PR lane requires adding its exact package and target to the reviewed manifest. The fast driver rejects absent, duplicate, ambiguous, and feature-gated integration identities before building. Cargo `--lib --bins --test NAME` selectors bound compilation; nextest filters alone would still compile the entire integration surface.

`scripts/ci_test_targets.py` first builds workspace product libraries and binaries so cross-package `cargo_bin` users and generated-C tests have their CLI and runtime static library. It then compiles/lists only the selected test targets, checks the complete binary receipt (including units and zero-test unit binaries), and executes the same selection without default filters. Missing unit binaries, unexpected integration binaries, empty selected integrations, or filtered non-ignored tests fail the worker. Ignored tests remain ignored; this pass neither deletes tests nor changes language semantics.

`Integration Tests (Linux)` fails closed on the fast worker. Its JUnit and `ci-fast-receipts` artifacts retain the exact selection, compiled test list, build/list/run times, and gate report for 14 days. Nightly JUnit reports stay in their producing workflow. The Linux nightly report inspects every execution worker and opens a failure tracker on non-success; a manual branch run cannot close a main-nightly tracker.

Use `gh workflow run heavy-e2e.yml --ref BRANCH` or `gh workflow run macos-nightly.yml --ref BRANCH` for candidate validation. Full Linux execution and generalization shards have 60-minute timeouts; other extended Linux workers have 45 minutes, and Darwin SMT has 60. Existing ignored/manual gates still require their documented prerequisite and explicit invocation. The stdlib self-test corpus remains explicitly invoked nightly. Existing nightly failures must be recorded against a baseline, never treated as passing evidence.

The developer's `gate.py --fast`, `--local`, `integration`, and full/manual commands retain their previous selections. The separate `gate.py ci-fast` stage owns only the new hosted selection. Run the owning phase oracle on the candidate when claiming phase completion.
