# Maintenance schedule

Scheduled maintenance items with hard external deadlines. Items get
removed once the corresponding work has shipped.

## GitHub Actions Node 20 -> Node 24 migration

GitHub is deprecating Node 20 as the runtime for JavaScript actions.
Until Node-24-compatible pins ship for each action we depend on, this
repo's workflows continue to run on Node 20 implicitly. Two hard dates
matter:

- **2026-06-02**: GitHub flips the default runtime to Node 24. Any
  workflow still pinned to an action version whose JS bundle targets
  Node 20 will emit a deprecation warning and (per the announcement)
  may be force-migrated. All four chelis-ecosystem repos must have
  Node-24-compatible action pins in place before this date.
- **2026-09-16**: Node 20 runtime support is removed entirely.
  Workflows still on Node-20-bundled actions stop running.

Deprecation announcement:
<https://github.blog/changelog/2025-09-15-actions-deprecating-node-20-support/>

### chelis-specific migration list

This repo's workflows that need attention:

| Workflow | Action | Current pin | Target pin |
|---|---|---|---|
| `ci.yml` | `actions/checkout` | `@v4` | `@v4` (latest @v4 release ships Node-24 bundle; verify before deadline) |
| `release.yml` | `actions/checkout` | `@v4` | `@v4` (same) |
| `release.yml` | `softprops/action-gh-release` | `@v2` | `@v2` (verify Node-24 bundle ships) |
| `heavy-e2e.yml` | `actions/checkout` | `@v4` | `@v4` (same) |

No `docker/setup-buildx-action` or `docker/build-push-action` usage in
this repo; those are hello-chelis-only.

The pin bumps themselves are deferred until upstream actions publish
Node-24-compatible releases. This file exists so the deadline is not
forgotten; the corresponding tracker issue is the work item.

### Out of scope here

Non-JS-runtime actions (`dtolnay/rust-toolchain`, `Swatinem/rust-cache`,
`taiki-e/install-action`, `astral-sh/setup-uv`, `actions/upload-artifact`,
`actions/download-artifact`) are not part of the Node 20 deprecation
matrix but should be re-verified opportunistically when the JS actions
are bumped.
