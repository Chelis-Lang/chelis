## Context

`adopt-semver-for-reef` adds compatible requirements and deterministic local resolution. Remote ranges need candidate lists and candidate manifests.

The current exact GitHub path fetches one named release. It does not enumerate versions or bound recursive dependency search.

A normal build and an explicit update have different intent. A local compatible graph is sufficient for a normal build, but not for an update.

The registry and project lock usually live under different roots. Their final writes cannot form one filesystem transaction.

`version-reef-manifest-schema` owns document versions and the general upgrade command. Accepted revision: `d6479aaa2ad5fea22ac7ac773e88491e22ffbbd0`.

`adopt-semver-for-reef` owns typed local resolution. Accepted revision: `1299452528e3735bf42edd584f6b90878b028906`.

This change activates manifest schema 2 and resolver 2 after bounded discovery passes.

## Goals / Non-Goals

**Goals:**

- Make remote discovery explicit, deterministic, and bounded.
- Preserve lock-first behavior for normal commands.
- Make update and outdated commands inspect remote candidates.
- Inspect untrusted candidate archives without unrestricted extraction.
- Keep source selection independent from descriptive package metadata.
- Publish only complete verified cache entries.
- Preserve the old lock until every selected cache entry is complete.
- Activate resolver 2 only after the complete remote path passes.

**Non-Goals:**

- Add a public registry, mirrors, multiple publishers, or arbitrary source commands.
- Permit two package versions with one name in one graph.
- Add toolchain version ranges.
- Change artifact formats, compiler semantics, runtime behavior, or backend output.
- Promise an unchanged registry after a crash during final publication.
- Transfer specification authority from `spec/**` to OpenSpec.

## Decisions

### 1. Use four explicit resolution modes

The resolver receives one closed mode:

- `Locked`: reuse one valid exact lock and perform no version listing.
- `Resolve`: use local candidates first, then remote discovery only if no local graph completes.
- `Refresh`: enumerate remote candidates even when a local graph completes.
- `Inspect`: run refresh discovery but write no final registry, index, or lock state.

Normal build, check, eval, schema, and prepared-graph paths use `Locked` when the current lock is valid.

A missing or requirement-stale lock uses `Resolve`. An exact locked-origin fetch is not version discovery and remains allowed under lock policy.

`chelis reef update` uses `Refresh`. `chelis reef outdated` uses `Inspect`.

When network access is disabled, every mode excludes remote providers. No mode creates an HTTP client in that state.

Alternative provider fallback based only on local failure makes an explicit update miss newer remote versions.

### 2. Keep the provider interface source-neutral

Add a private `CandidateProvider` interface. It lists typed versions, fetches one bounded manifest, and fetches one selected artifact pair.

Provider results contain parsed package identities and a typed `SourceLocator`. An exact selected source becomes the existing scheme-tagged `remote_origin` in `reef.lock`.

The first implementation is `GitHubReleaseProvider`. Direct `--from-github` keeps its current exact request path.

A provider never reads current or future descriptive manifest URLs as source configuration.

Mirrors and additional providers require later reviewed changes. They can implement the same interface without changing resolver semantics.

Alternative direct GitHub calls inside the resolver make source policy difficult to test and extend.

### 3. Enforce one production resolution budget

Add a typed `ResolutionBudget`. The `reef-remote-discovery` capability specification owns each exact remote limit.

The remote budget inherits the local candidate, package, dependency, depth, and resolver-state limits from `reef-package-versioning`.

Every accepted remote candidate counts against the shared candidate limit for its package name. Local and remote candidates together cannot exceed 256.

The production CLI cannot disable a limit. Tests can inject smaller budgets through a crate-private constructor.

A limit failure returns a typed error with the dimension, limit, observed value, package, and provider operation.

Budget accounting uses checked integers. A counter overflow is a limit failure.

Pagination stops when GitHub returns no next page, all active requirements have enough candidates, or one limit is reached.

Alternative unbounded retries or environment overrides weaken the safety boundary.

### 4. Inspect candidate manifests without full extraction

Download each candidate archive to command-owned temporary storage. Enforce the compressed-byte limit during streaming.

Read the compressed tar stream entry by entry. Count all decompressed header and payload bytes against the scan limit.

Accept exactly one root regular-file entry named `reef.toml`. Reject duplicate entries, links, devices, absolute paths, parent paths, and malformed tar records.

Read at most 1 MiB from that entry. Parse its schema and strict manifest DTO before the candidate enters the resolver.

Do not unpack another archive entry during candidate inspection. Fetch the `.chb` only after the resolver selects that candidate.

Selected artifact verification continues through `verify_artifact_pair` before final placement.

Alternative full temporary extraction enlarges the path, link, and decompression attack surface.

### 5. Keep discovery and resolution deterministic

Sort package names by byte value. Sort candidates by descending SemVer precedence.

Reject release versions with build metadata through the package-version rule. Do not use API order, response time, or metadata text as a tie-break.

Cache provider results for one command. The same package and provider query consumes one budget entry and returns one stable typed list.

Memoize failed resolver states. Stop before a state enters the memo table when the state budget is exhausted.

A conflict reports each requester, requirement, considered version, excluded provider result, and any reached budget limit.

### 6. Give update and outdated different commit policies

Add these commands:

```text
chelis reef update [<package>]
chelis reef outdated [<package>] [--json]
```

A full update removes all lock preferences and queries remote providers for every package with a remote source policy.

A targeted update removes the named preference. Other locked packages remain fixed unless the selected target requires a transitive change.

Outdated computes the same candidate view. It reports current, newest compatible, newest incompatible, and blocked reasons.

Outdated writes no final registry, index, or lock file. Temporary bytes disappear when the command ends.

An exact requirement remains exact in both commands.

### 7. Treat the registry as an append-only verified cache

Stage selected artifacts outside final registry paths. Verify the complete pair, manifest identity, compiler pin, hashes, and selected origin.

Acquire the package-root project lock first. Then acquire the Reef registry lock and hold both through cache, index, and project `reef.lock` replacement.

This total lock order prevents two update commands from deleting or replacing each other's project state.

Recheck the selected identity and final path. Publish each absent package directory through a same-parent rename.

If a final name and version already exist, verify exact bytes. Accept identical bytes and reject different bytes as a source conflict.

Update `index.json` through its atomic writer after every selected package directory is complete.

Replace the project `reef.lock` last through an atomic sibling rename. The old lock remains authoritative until this replacement succeeds.

A failure before cache publication leaves final state unchanged. A failure after cache publication can leave unused complete verified entries.

No failure can leave a new lock that refers to an incomplete package directory. A later garbage-collection change can remove unreferenced entries.

Alternative rollback of complete cache entries can delete data that another concurrent project uses.

### 8. Preserve integrity and exact locked origins

A valid lock does not list releases. If one locked package is absent locally, Reef fetches only its exact recorded origin.

A hash mismatch, unavailable locked release, or source identity mismatch remains a hard error. Reef does not start discovery after that error.

Normal discovery can use canonical GitHub package sources only when no valid lock controls that package.

Drafts, unrelated tags, and stable tags marked as GitHub prereleases remain excluded. SemVer prereleases follow Cargo requirement rules.

### 9. Activate schema 2 and resolver 2 after bounded discovery

This change depends on manifest schema 1 and the local SemVer resolver.

Add the registered manifest schema-1 to schema-2 migration. It rewrites each legacy exact dependency `X.Y.Z` as `=X.Y.Z` and adds `resolver = "2"`.

The migration preserves path-only dependencies and rewrites path versions through the same exact rule.

`chelis reef init` starts schema-2 resolver-2 output only after the authoritative oracle passes.

No controlled or downstream manifest migrates before the local resolver and remote discovery oracles pass on the same revision.

Package metadata allocates manifest schema 3 after this activation.

### 10. Use one executable completion oracle

The authoritative completion oracle is:

```sh
cargo nextest run -p chelis-cli --test reef_remote_discovery --no-fail-fast
```

Wiremock fixtures cover mode selection, pagination, every budget, archive inspection, cache publication, lock-last commits, and zero-request offline behavior.

A fresh local red-team agent mutates response order, archive sizes, tar entries, graph depth, resolver states, publication failures, and lock replacement.

OpenSpec validation and documentation builds are supporting evidence. They do not replace the oracle.

## Validation Record

The activation revision uses OpenSpec 1.6.0. Strict OpenSpec validation passed for all 49 active items.

The document, package-versioning, and remote-discovery oracles passed 76 tests on one revision. The authoritative remote oracle contributed 24 tests.

The complete `chelis-reef` suite passed 144 tests, with 2 skipped. The Reef CLI test set passed 124 tests, with 10 skipped.

Two fresh adversarial audits examined modes, budgets, archives, provider order, graph branches, publication failures, and schema migration.

The implementation corrected findings for inline tables, trailing comments, compiler pins, budget scopes, retry closure, source selection, and project-state races.

The implementation also corrected findings for path escapes, memo charges, typed conflict context, request time, artifact drift, and byte-conflict evidence.

OpenSpec validation proves the artifact structure only. It does not replace the executable oracles or the adversarial audits.

Residual limits remain explicit. Refresh fails when more than 256 accepted tags are necessary. A hard source conflict stops resolution for that package version.

Complete unused cache entries can remain after a late failure. Garbage collection remains deferred.

## Hosted Acceptance Handoff

An implementation review can start only after the local oracles, strict validation, formatting, Clippy, and documentation build pass.

The hosted gate requires green Linux integration, documentation, and applicable release checks on the final revision.

Hosted fixtures use canonical private-repository shapes and fixed test tokens. They contain no private credentials or local file content.

Hosted results provide supporting evidence. They do not replace the authoritative local oracle.

## Risks / Trade-offs

- **[Risk] Production limits reject a legitimate large ecosystem.** → Report the exact limit and change it only through a reviewed contract update.
- **[Risk] Explicit update uses many GitHub requests.** → Cache command-local provider results and stop after active requirements have enough candidates.
- **[Risk] A failed lock write leaves extra cache entries.** → Keep entries complete and unreferenced, then remove them through later garbage collection.
- **[Risk] Two sources publish one name and version with different bytes.** → Reject the source conflict and preserve the existing verified entry.
- **[Risk] SemVer activates before bounded discovery.** → Keep generated and migrated resolver-2 manifests inactive until both oracles pass.
- **[Risk] Provider abstraction implies unsupported sources.** → Document GitHub as the only implementation and reject unknown locator schemes.

## Migration Plan

1. Land `version-reef-manifest-schema`.
2. Land the inactive local resolver from `adopt-semver-for-reef`.
3. Add failing mode, budget, archive, update, and commit fixtures.
4. Add the provider interface and GitHub provider.
5. Add bounded candidate inspection and resolver accounting.
6. Add update, outdated, and lock-last publication.
7. Add the schema-1 to schema-2 migration.
8. Activate resolver-2 output and migrate controlled manifests.
9. Run both prerequisite oracles and this change's oracle on one revision.
10. Run fresh adversarial validation and hosted acceptance.
11. Publish the compatible compiler before downstream migration.

Rollback first stops new schema-2 output. Then it reverts controlled manifests to schema 1 and exact resolver-1 dependencies.

Existing exact locks remain valid. Unused verified cache entries can remain until garbage collection.

## Open Questions

None.
