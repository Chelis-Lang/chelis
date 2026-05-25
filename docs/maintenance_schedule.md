# Maintenance schedule

Scheduled maintenance items with hard external deadlines. Items get
removed once the corresponding work has shipped.

## GitHub Actions Node 20 -> Node 24 migration

**Completed 2026-05-25** (chelis#188, PR landed before 2026-06-02
default-runtime cutover). The four issue-matrix actions were bumped
to Node-24-compatible releases:

| Workflow | Action | Old pin | New pin | Evidence |
|---|---|---|---|---|
| `ci.yml` | `actions/checkout` | `@v4` | `@v6` | v6 ships Node 24 (<https://github.com/actions/checkout/releases/tag/v6.0.0>) |
| `release.yml` | `actions/checkout` | `@v4` | `@v6` | same |
| `release.yml` | `softprops/action-gh-release` | `@v2` | `@v3` | v3.0.0 "moves the action runtime from Node 20 to Node 24" (<https://github.com/softprops/action-gh-release/releases/tag/v3.0.0>) |
| `heavy-e2e.yml` | `actions/checkout` | `@v4` | `@v6` | same |

Background: GitHub deprecated Node 20 for JavaScript actions with two
hard dates — **2026-06-02** (default runtime flips to Node 24) and
**2026-09-16** (Node 20 removed entirely). Announcement:
<https://github.blog/changelog/2025-09-15-actions-deprecating-node-20-support/>.

No `docker/setup-buildx-action` or `docker/build-push-action` usage in
this repo; those are hello-chelis-only.

### Adjacent JS-runtime pins (verified Node-24 at time of bump)

- `Swatinem/rust-cache@v2` → resolves to v2.9.1, `runs.using: node24`
- `astral-sh/setup-uv@v8.1.0` → `runs.using: node24`

### Adjacent JS-runtime pins still on Node 20 (outside this work item)

- `actions/upload-artifact@v4` — Node 20; v6 ships Node 24
- `actions/download-artifact@v4` — Node 20; v8 ships Node 24
- `taiki-e/install-action@nextest`/`@v2` — composite action; no JS runtime

These were explicitly excluded from chelis#188's matrix. They do not
emit deprecation warnings yet because they are still on supported
floating majors; tracker for the next bump cycle if/when upstream
deprecates the v4 lines.
