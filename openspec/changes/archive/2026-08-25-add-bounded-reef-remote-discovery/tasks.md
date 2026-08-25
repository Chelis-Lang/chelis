## 1. Verify prerequisites and add test stubs

- [x] 1.1 Record accepted revisions for `version-reef-manifest-schema` and `adopt-semver-for-reef`.
- [x] 1.2 Add mode fixtures for locked, local, refresh, inspect, and network-disabled operations.
- [x] 1.3 Add update fixtures where a compatible local graph hides a newer remote graph.
- [x] 1.4 Add targeted-update fixtures for fixed unrelated packages and required transitive changes.
- [x] 1.5 Add outdated human and JSON snapshots for current, compatible, incompatible, and blocked versions.
- [x] 1.6 Add provider fixtures for valid, malformed, unsupported, and inconsistent source locators.
- [x] 1.7 Add one positive and one negative fixture for every production budget dimension.
- [x] 1.8 Add tar fixtures for one manifest, duplicates, links, devices, unsafe paths, malformed records, and size limits.
- [x] 1.9 Add publication fixtures for identical races, byte conflicts, index failure, and lock replacement failure.
- [x] 1.10 Add activation fixtures for failed and successful prerequisite oracle states.

## 2. Add explicit resolver modes

- [x] 2.1 Add a closed `DiscoveryMode` type at the CLI-to-resolver boundary.
- [x] 2.2 Route valid-lock commands through locked mode without version listing.
- [x] 2.3 Route unlocked normal commands through local-first resolve mode.
- [x] 2.4 Route full and targeted updates through refresh mode.
- [x] 2.5 Route outdated through inspect mode.
- [x] 2.6 Prevent HTTP client construction when network access is disabled.
- [x] 2.7 Add typed errors for unsupported mode and unavailable remote discovery.

## 3. Add a provider-neutral boundary

- [x] 3.1 Add typed `SourceLocator`, exact origin, and provider-result models.
- [x] 3.2 Add the private `CandidateProvider` interface.
- [x] 3.3 Move GitHub release-list logic behind `GitHubReleaseProvider`.
- [x] 3.4 Keep direct exact-tag installation on its current request path.
- [x] 3.5 Parse provider identities before candidates reach the resolver.
- [x] 3.6 Keep descriptive package URLs outside all provider selection paths.
- [x] 3.7 Reject unsupported source schemes without fallback.

## 4. Enforce the production resolution budget

- [x] 4.1 Add `ResolutionBudget` and checked counters for every specified dimension.
- [x] 4.2 Set the exact production limits from the capability specification.
- [x] 4.3 Add a crate-private constructor for smaller test budgets.
- [x] 4.4 Charge release pages, requests, tags, manifests, and bytes before each remote operation.
- [x] 4.5 Count accepted remote candidates against the local candidate limit from `reef-package-versioning`.
- [x] 4.6 Inherit local package, dependency, depth, and resolver-state counters without a duplicate owner.
- [x] 4.7 Enforce compressed, decompressed, manifest, total-byte, and request-time limits during I/O.
- [x] 4.8 Return typed budget errors with dimension, limit, observed value, package, and operation.
- [x] 4.9 Treat checked-counter overflow as a budget failure.
- [x] 4.10 Stop pagination when active requirements have enough candidates.

## 5. Inspect candidate manifests safely

- [x] 5.1 Stream candidate archives into command-owned temporary storage with byte accounting.
- [x] 5.2 Scan tar entries without unpacking the archive.
- [x] 5.3 Accept exactly one root regular-file `reef.toml` entry.
- [x] 5.4 Reject duplicates, links, devices, unsafe paths, malformed records, and excessive bytes.
- [x] 5.5 Parse the manifest schema and strict DTO before candidate insertion.
- [x] 5.6 Fetch `.chb` only for candidates in the selected complete graph.
- [x] 5.7 Route selected pairs through complete artifact verification.
- [x] 5.8 Remove all command-owned temporary bytes after success or failure.

## 6. Keep remote resolution deterministic

- [x] 6.1 Sort package names and candidate precedence through the local resolver rules.
- [x] 6.2 Cache each provider query once per command.
- [x] 6.3 Memoize failed resolver states and charge each new state.
- [x] 6.4 Remove API order, response timing, and metadata text from all tie-breaks.
- [x] 6.5 Preserve complete requester, requirement, candidate, exclusion, and budget context.
- [x] 6.6 Prove identical resolution across provider-order and response-order permutations.

## 7. Add update and outdated commands

- [x] 7.1 Add the full refresh library API without lock preferences.
- [x] 7.2 Add targeted refresh with fixed unrelated locks.
- [x] 7.3 Permit only required transitive changes during a targeted update.
- [x] 7.4 Add `chelis reef update [<package>]` dispatch and exact change output.
- [x] 7.5 Add `chelis reef outdated [<package>] [--json]` dispatch.
- [x] 7.6 Report current, newest compatible, newest incompatible, and blocked states.
- [x] 7.7 Keep exact requirements exact in both commands.
- [x] 7.8 Prove that outdated writes no final registry, index, or lock state.

## 8. Publish verified cache entries before the lock

- [x] 8.1 Stage and verify every selected artifact outside final registry paths.
- [x] 8.2 Acquire the package-root project lock before the Reef registry lock.
- [x] 8.3 Hold both locks through package, index, and project `reef.lock` replacement.
- [x] 8.4 Recheck selected identities under both locks.
- [x] 8.5 Publish each absent complete package directory through a same-parent rename.
- [x] 8.6 Accept an existing byte-identical package and reject a byte conflict.
- [x] 8.7 Update `index.json` only after all selected package directories are complete.
- [x] 8.8 Replace `reef.lock` last through atomic sibling replacement.
- [x] 8.9 Preserve the prior lock after every failure.
- [x] 8.10 Permit only complete verified unreferenced cache entries after a late failure.
- [x] 8.11 Prove that no new lock references an incomplete registry entry.
- [x] 8.12 Prove that concurrent project writers never remove another writer's sibling.

## 9. Activate manifest schema 2 and resolver 2

- [x] 9.1 Extend `reef upgrade` with the schema-1 to schema-2 manifest migration.
- [x] 9.2 Rewrite exact registry and path versions through `=X.Y.Z`.
- [x] 9.3 Preserve path-only dependencies, comments, and key order.
- [x] 9.4 Add the manifest schema-2 JSON Schema and drift test.
- [x] 9.5 Make `reef init` write schema 2 and resolver 2 only after activation.
- [x] 9.6 Keep controlled manifests on resolver 1 until all prerequisite oracles pass.
- [x] 9.7 Migrate controlled manifests after the combined activation gate passes.
- [x] 9.8 Record manifest schema 3 as the package-metadata allocation.

## 10. Update documentation and release plans

- [x] 10.1 Update `docs/book/src/reef.md` with modes, budgets, update, outdated, and failure recovery.
- [x] 10.2 Amend `spec/design/reef_distribution.md` with bounded provider discovery and lock-last publication.
- [x] 10.3 Amend `spec/design/chelis_packaging_and_install.md` with resolver activation and cache semantics.
- [x] 10.4 Update examples and `CHANGELOG.md` with schema-2 migration instructions.
- [x] 10.5 Document GitHub as the only provider and keep mirrors deferred.
- [x] 10.6 Document complete unused cache entries and defer garbage collection.
- [x] 10.7 Route downstream migrations through normal compiler-pin reviews after release.

## 11. Run strict validation

- [x] 11.1 Verify OpenSpec 1.6.0, then run `openspec validate --all --strict --no-interactive`.
- [x] 11.2 Run `cargo fmt --all -- --check` and correct all format failures.
- [x] 11.3 Run targeted Clippy with warnings denied for `chelis-reef` and `chelis-cli`.
- [x] 11.4 Run `cargo nextest run -p chelis-cli --test reef_document_schema --no-fail-fast` on the activation revision.
- [x] 11.5 Run `cargo nextest run -p chelis-reef --test reef_package_versioning --no-fail-fast` on the activation revision.
- [x] 11.6 Run the authoritative oracle: `cargo nextest run -p chelis-cli --test reef_remote_discovery --no-fail-fast`.
- [x] 11.7 Verify that every requirement has positive and negative executable evidence.
- [x] 11.8 Keep schema-2 and resolver-2 output inactive until every oracle passes.

## 12. Run adversarial validation

- [x] 12.1 Start a fresh local red-team agent after the authoritative oracle passes.
- [x] 12.2 Mutate pagination, counters, response order, graph shape, archives, sources, and request time.
- [x] 12.3 Stop every publication step and prove lock-last recovery.
- [x] 12.4 Verify zero HTTP requests under network-disabled modes.
- [x] 12.5 Correct all material findings and rerun all prerequisite and authoritative oracles.
- [x] 12.6 Record residual risks and prove that OpenSpec validation remains structural evidence only.

## 13. Obtain hosted acceptance

- [x] 13.1 Open the implementation review only after local strict and adversarial validation passes.
- [x] 13.2 Require green Linux integration, documentation, and applicable release checks on the final revision.
- [x] 13.3 Verify that hosted fixtures expose no private repository credentials or local file content.
- [x] 13.4 Record hosted results as supporting evidence without replacing the authoritative oracle.
