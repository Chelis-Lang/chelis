# CI selection sidecar feasibility, 2026-09-29

## Verdict

A statically proven test-only Rust source could route to required Fast lib units without selecting package integrations. The current planner and TOML cannot safely express that route. No tracked code or config changed. A local two-file patch would lack an all-target Rust reachability proof and a Fast receipt for exact lib-test execution.

## Exact state and effect

Assessment worktree `/Users/robertronan/chelis-worktrees/ci-selection-sidecar-20260929`, branch `research/ci-selection-sidecar-20260929`, HEAD `e16b2f795045e26a366cfbba570b73a0c6278ea3` (fresh `origin/main`), clean tracked status, own uv Python 3.11 `.venv`. The older `ci-selection-assessment` worktree was left intact; only this report was written under its ignored `target/`.

Supplied #2774 plan: PR head `d699109991d5c9f32d6df60d0c0cf2fdacff09da`, synthetic candidate `54592ee74808a0560ea78548ce9d5ceea6b26d58`, base `e16b2f795045e26a366cfbba570b73a0c6278ea3`. Its sole change is `crates/chelis-compiler-api/src/source_arch.rs`; `workspace_package` selects 130 compiler-API expansion targets, including `capacity_census_wire`, with zero change-owned targets. Of 130, 28 require features and 26 are standing integrations. Duration weights total 3,560,960 ms; `capacity_census_wire` weighs 1,236,852 ms. Weights are not observed time.

If that sole path qualified, predicted expansion is **130 → 0**, with the already required Fast default-feature compiler-API lib unit; no new integration selection. This was not implemented. In the latest 80 first-parent commits on `origin/main` (2026-09-25–29), 52 changed crate `src/*.rs`; 8 touched one of seven files now declared by direct `#[cfg(test)] mod NAME;` parents (10 edits). No GitHub run inventory was queried.

## Required invariants

`static_path_classification` gives package membership priority over path rules; `make_plan` then expands eligible package integrations. A `[[path_rule]]` naming `ci-fast` cannot override this. An exact-source disposition needs a digested plan claim, validated on base and candidate for rename/delete. Any co-changed production source, manifest, integration target, or required-package rule keeps package selection. Targeted rebase keeps required package/dependent closure because `.github/workflows/ci.yml` skips Fast there.

Prove reachability from **all Cargo target roots**, across claimed cfg/feature configurations, to be solely through a `cfg(test)` Fast lib module. Check regular tracked-file identity, module resolution and alternate source/data inputs. Reject or resolve a second parent, `cfg(any(test, feature = "x"))`, `cfg_attr(path)`, `#[path]`, `include!`, `include_str!`/`include_bytes!`, macro module wiring, symlinks and ambiguous build-script reads. Unknown forms fail closed. `crates/chelis-compiler-api/tests/source_wire.rs` actually imports `../src/source_wire.rs` via `#[path]`. The existing `syn` production scanner in `crates/chelis-repr-inventory/src/lib.rs` handles some cfg/module/path/include cases, but is not an all-target proof.

Fast `scripts/ci_test_targets.py` lists/runs `--workspace --lib --bins`, validates default-feature lib suites, and isolates nondefault-feature integrations. An integration compiles the library without its `cfg(test)` module, so the 130 integrations do not directly run `source_arch.rs` tests. Fast JUnit contains lib cases, but `coverage.json` binds selected/executed **integration** tests only. The route needs a lib-test listing and JUnit execution receipt bound to candidate SHA, config digest and plan, with rejection for missing/failed/filtered/ignored-only or feature-mismatched claims. Nondefault-feature unit tests need an exact Fast feature run before claiming coverage; package expansion alone leaves that direct-test gap. Preserve feature isolation.

Negative controls: production `src/tests.rs`; parent cfg removed/weakened; second bin/integration/example parent; literal/computed include or path, production `include_str!`/`include_bytes!`; moved/deleted/symlinked file; co-changed production/manifest; new integration target; gated or ignored-only unit tests; targeted rebase without Fast; absent/stale Fast lib JUnit, receipt or SHA.

## Commands and limits

- `git fetch origin main`, new worktree, `uv venv --python 3.11`: passed; no other checkout switched or built.
- Read-only `git show` of #2774 sources, supplied plan, planner/TOML/workflow/runner, `cargo metadata --no-deps --format-version 1 --locked`, bounded first-parent path census: passed.
- Three focused `scripts.test_ci_change_owned` and three `scripts.test_ci_test_targets` tests: passed. The latter initially lacked PyYAML; `uv pip install --python .venv/bin/python PyYAML` installed 6.0.3, then the rerun passed.
- No cargo build/full gate, push, PR, new routing plan or hosted run.
