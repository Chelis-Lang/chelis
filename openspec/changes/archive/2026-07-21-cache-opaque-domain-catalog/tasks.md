## 1. Write Regression Tests First

- [x] 1.1 Add a lint-driver test rule that proves preparation runs exactly once while checks run for multiple admitted files, plus negative parity proving a second `lint` invocation prepares fresh state.
- [x] 1.2 Add opaque-domain tests proving declarations under every canonical skipped-tree class cannot influence the catalog, while an equivalent admitted declaration still causes an out-of-module forge violation.
- [x] 1.3 Add invocation-level parity tests for defining-module allowance, out-of-module rejection, single-file roots, file-scoped module-less shadowing, unreadable or unparsable candidates, and corpus-independent Deep checking.
- [x] 1.4 Add an index-shape regression covering many distinct opaque leaves plus multiple defining modules for one shared leaf.

## 2. Add Lint-Run Preparation

- [x] 2.1 Extend `Rule` with default invocation-preparation and prepared-check hooks that preserve existing rule implementations and direct `check` callers.
- [x] 2.2 Update `chelis_lint::lint` to resolve the canonical walker vector once, prepare each rule once from those admitted entries, and keep prepared state paired with its rule through dispatch.
- [x] 2.3 Verify the driver lifecycle tests from task 1.1 pass without changing diagnostics, ordering, exceptions, or severity behavior.

## 3. Refactor the Opaque-Domain Catalog

- [x] 3.1 Remove the rule-local `WalkDir` discovery and build `Catalog` only from admitted `Surface::SurfSource` entries supplied during lint-run preparation.
- [x] 3.2 Split immutable corpus-wide catalog data from `current_file_module_less_leaves`, passing the latter as per-file state so CR2-7 remains file-scoped without cloning or mutating the shared catalog.
- [x] 3.3 Route prepared Surf checks through the shared catalog, retain one-off direct-check and single-file behavior, and leave Deep checks on their existing local-expression path.
- [x] 3.4 Run the focused lifecycle, skip-filter, and semantic parity tests until every positive and negative case from section 1 is green.
- [x] 3.5 Replace per-site opaque declaration scans with type-leaf and defining-module hash indices, including the fail-closed untyped Deep update query.

## 4. Documentation and Acceptance

- [x] 4.1 Add a `CHANGELOG.md` entry linking #603 and describing the filtered, once-per-invocation catalog preparation; update active lint documentation only if implementation changes its stated behavior.
- [x] 4.2 From the worktree root, inspect orphaned Rust processes with `python3 scripts/reap_orphans.py`, build the current-worktree CLI with `CARGO_TARGET_DIR=target/agents/issue-603 cargo build -p chelis-cli --bin chelis`, then run the authoritative oracle with `CARGO_BIN_EXE_chelis="$PWD/target/agents/issue-603/debug/chelis" CARGO_TARGET_DIR=target/agents/issue-603 cargo nextest run -p chelis-lint`.
- [x] 4.3 Run `cargo fmt --all -- --check` and `python3 scripts/gate.py --local`, keeping cargo work in the isolated issue-603 target directory where the command permits it.
- [x] 4.4 Run a fresh adversarial validation pass against #603 that executes both skipped-tree and multi-file regression cases, then record any honest residual limitations before claiming resolution.

### Adversarial Validation Record

Fresh local red-team validation used `target/agents/redteam-603`, added default-hook/state-pairing, empty-catalog, malformed/unreadable-candidate, Deep-independence, and exact CR2-7 path probes, and passed its then-current crate and CLI probes. A later PR review found one remaining P2 complexity gap: each construction site still scanned the opaque declaration vector. The follow-up replaced that scan with leaf/module hash indices, added an executable index-shape regression, and passed 337/337 `chelis-lint` tests plus 14/14 standalone traversal-policy CLI tests. Residual limitation: there is no wall-clock benchmark against a genuinely massive `target/`; linearity is locked mechanically by preparation counts, catalog-parse counts, and indexed construction-site lookups. The unreadable-candidate probe uses a Unix-only broken symlink.
