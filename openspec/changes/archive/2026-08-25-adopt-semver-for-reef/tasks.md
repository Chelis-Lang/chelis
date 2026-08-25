## 1. Verify prerequisites and add test stubs

- [x] 1.1 Record the accepted `version-reef-manifest-schema` revision and oracle result.
- [x] 1.2 Select one implementation order for the Reef SNAFU phase and this change.
- [x] 1.3 Add package-name tests for canonical, uppercase, underscore, separator, hyphen, length, and reserved inputs.
- [x] 1.4 Add package-version tests for stable, prerelease, build metadata, partial, leading-zero, and overflow inputs.
- [x] 1.5 Add schema-1 exact and inactive schema-2 resolver fixtures.
- [x] 1.6 Add dependency tests for string, inline, caret, exact, intersection, zero-major, prerelease, and invalid forms.
- [x] 1.7 Add path tests for path-only, matching requirements, version mismatch, name mismatch, and publication rejection.
- [x] 1.8 Add compiler-pin tests for exact releases, ranges, prereleases, metadata, and active-version mismatch.
- [x] 1.9 Add local-index fixtures for numeric order, invalid identities, candidate limits, and duplicate sources.
- [x] 1.10 Add resolver fixtures for one-version graphs, transitive constraints, dead-end recovery, disjoint conflicts, depth, and state limits.
- [x] 1.11 Add lock fixtures for valid reuse, stale requirements, root mismatch, malformed identities, hash failure, and unavailable origins.
- [x] 1.12 Add format snapshots for schema-1 `reef.toml`, `reef.lock`, `index.json`, `.chb`, and artifact names.
- [x] 1.13 Add activation fixtures that keep generated and migrated manifests on resolver 1.

## 2. Add typed identity and manifest boundaries

- [x] 2.1 Add workspace `semver` version 1 with Serde support and add the Reef workspace dependency.
- [x] 2.2 Add `PackageName`, `PackageVersion`, `ResolverVersion`, `PackageRequirement`, `ExactCompilerVersion`, and `ResolvedPackageId`.
- [x] 2.3 Reject package-version build metadata at every identity boundary.
- [x] 2.4 Add a typed resolver-2 input model without schema dispatch.
- [x] 2.5 Convert schema-1 dependencies into exact typed requirements.
- [x] 2.6 Convert schema-2 string and inline forms into Cargo-style typed requirements.
- [x] 2.7 Convert path dependencies into one typed variant with an optional requirement.
- [x] 2.8 Parse every Reef manifest reader into a typed identity model before downstream use.
- [x] 2.9 Parse lock, index, release, and shell identities before their consumers receive them.
- [x] 2.10 Preserve current external string serializers and prove each format snapshot.
- [x] 2.11 Bump the prepared-graph cache version and verify fail-closed rebuilds.

## 3. Add versioned local dependency semantics

- [x] 3.1 Keep schema-1 dependency versions complete and exact.
- [x] 3.2 Define the required resolver-2 field contract for the later schema-2 DTO.
- [x] 3.3 Parse schema-2 dependencies through Cargo-style `VersionReq` rules.
- [x] 3.4 Support schema-2 string shorthand with inline-version semantics.
- [x] 3.5 Permit path plus version and verify the loaded package name and version.
- [x] 3.6 Keep path-only local dependencies valid and keep publication rejection unchanged.
- [x] 3.7 Keep compiler pins and the bundled `chelis-std` candidate exact.
- [x] 3.8 Keep schema-2 generation and upgrade registration inactive.

## 4. Implement bounded deterministic local resolution

- [x] 4.1 Add candidate models for lock, local registry, bundled runtime, and path sources.
- [x] 4.2 Sort local registry versions by SemVer precedence and reject corrupt identities.
- [x] 4.3 Enforce the exact local limits from the `reef-package-versioning` capability specification.
- [x] 4.4 Use checked counters and return typed limit errors before excess operations.
- [x] 4.5 Collect every requester and requirement for each package name.
- [x] 4.6 Implement deterministic depth-first search with bytewise names and descending precedence.
- [x] 4.7 Memoize failed resolver states and retain complete conflict context.
- [x] 4.8 Reject conflicting same-version non-path sources and enforce explicit path-override priority.
- [x] 4.9 Prove deterministic results across candidate and graph-order permutations.

## 5. Integrate exact lock preference

- [x] 5.1 Verify the root identity, requirements, source kinds, and canonical path declarations before lock reuse.
- [x] 5.2 Route build, check, eval, schema, and prepared-graph paths through a valid lock first.
- [x] 5.3 Preserve hard failure for locked hash mismatches and unavailable origins.
- [x] 5.4 Run bounded local resolution when current requirements or source declarations invalidate a lock preference.
- [x] 5.5 Keep `install --from-lockfile` exact and outside requirement matching.
- [x] 5.6 Sort lock dependencies by canonical package name before serialization.
- [x] 5.7 Return the unresolved package and active requirements when no local graph completes.
- [x] 5.8 Preserve the current lock until local resolution and verification succeed.

## 6. Add typed errors and documentation

- [x] 6.1 Add typed errors for identity, requirement, compiler, resolver, path, limit, conflict, and lock failures.
- [x] 6.2 Include requesters, requirements, candidates, sources, and reached limits in conflict errors.
- [x] 6.3 Preserve legacy text API boundaries while the CLI owns error presentation.
- [x] 6.4 Update `docs/book/src/reef.md` with local resolver semantics and inactive schema 2.
- [x] 6.5 Amend `spec/design/reef_distribution.md` with local candidate and lock behavior.
- [x] 6.6 Amend `spec/design/chelis_packaging_and_install.md` with exact lock preference.
- [x] 6.7 Update examples and `CHANGELOG.md` without announcing resolver-2 activation.
- [x] 6.8 Record `add-bounded-reef-remote-discovery` as the activation owner.

## 7. Run strict validation

- [x] 7.1 Verify OpenSpec 1.6.0, then run `openspec validate --all --strict --no-interactive`.
- [x] 7.2 Run `cargo fmt --all -- --check` and correct all format failures.
- [x] 7.3 Run targeted Clippy with warnings denied for `chelis-reef` and `chelis-cli`.
- [x] 7.4 Run the authoritative oracle: `cargo nextest run -p chelis-reef --test reef_package_versioning --no-fail-fast`.
- [x] 7.5 Verify that every requirement has positive and negative executable evidence.
- [x] 7.6 Keep schema-2 generation, migration, and downstream adoption inactive.

## 8. Run adversarial validation

- [x] 8.1 Start a fresh local red-team agent after the authoritative oracle passes.
- [x] 8.2 Mutate names, versions, requirements, indexes, locks, graph order, depth, states, and sources.
- [x] 8.3 Verify byte-identical lock ordering across repeated local resolver runs.
- [x] 8.4 Verify that build metadata never reaches package identity or artifact paths.
- [x] 8.5 Correct all material findings and rerun the authoritative oracle.
- [x] 8.6 Record residual risks and prove that OpenSpec validation remains structural evidence only.

## 9. Record the hosted acceptance handoff

- [x] 9.1 Record that implementation review starts only after local strict and adversarial validation passes.
- [x] 9.2 Record the required Linux integration, documentation, and applicable release checks.
- [x] 9.3 Verify that hosted fixtures use private canonical repositories without remote enumeration.
- [x] 9.4 Record that hosted results support but do not replace the authoritative oracle.
