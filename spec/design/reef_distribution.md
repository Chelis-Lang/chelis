# Reef Distribution

**Status:** Current Phase A distribution contract, with historical implementation-plan
notes preserved for context. Phase A is complete in `spec/12-roadmap.md`: the shipped
surface includes `chelis reef install --from-github`, `--bootstrap`, default-on
auto-fetch during `chelis reef build`, lockfile `remote_origin`, and
`chelis reef install --from-lockfile`. The public registry server remains deferred.

Companion to `chelis_trust_stack.md` and `effect_taxonomy_expansion.md` (which adds
install-time effect manifests on top of the install path described here).

---

## Context

Reef is the Chelis package manager. It has a content-addressed local registry,
validated install paths, lockfiles with `(name, version, source-kind, compiler-pin,
archive_sha256, shell_sha256)` tuples, SHA256 verification on install, and remote
fetch through canonical GitHub release assets.

This document records the contract that closed the pre-launch distribution gap:
canonical shell artifacts are fetched from the canonical hosting org's release tags,
installed through the same validation path as local artifacts, and recorded with source
provenance so fresh checkouts can reproduce dependency state. The eventual public
registry server is post-launch and is recorded here for forward-compatibility but is
not designed in this round.

---

## Background

### Shipped surface

- `chelis reef init` scaffolds a new package.
- `chelis reef build` resolves dependencies from the local registry, and auto-fetches
  missing dependencies from recorded or canonical GitHub release origins unless
  `--no-auto-fetch` is passed. It type-checks, lowers, and emits
  `dist/<name>-<version>.{chb,tar.zst}` plus a `reef.lock` recording the resolved
  dependency tuples.
- `chelis check <package-source>` and `chelis build <package-source>` use the same
  resolved package graph and MUST repair a missing or malformed `reef.lock`, including
  when the prepared graph itself came from a warm cache (chelis#971).
- `chelis reef publish` runs build, then copies artifacts into
  `$CHELIS_REEF_HOME/packages/<name>/<version>/` and updates
  `$CHELIS_REEF_HOME/index.json`.
- `chelis reef install --from-monorepo <path>` discovers packages under
  `<path>/packages/`, validates prebuilt artifacts (archive SHA256 match,
  shell readability, name/version agreement, embedded archive hash check),
  copies into the local registry, and updates the index.
- `chelis reef install --from-github <org>/<repo>@<tag>` fetches release assets from
  the GitHub API, validates the same artifact pair, records `remote_origin`, and
  updates the index.
- `chelis reef install --bootstrap [<org>/<repo>@<tag>...]` installs canonical shells
  in topological dependency order, rejecting `chelis-std` because it is the bundled
  runtime rather than a bootstrap target.
- `chelis reef install --from-lockfile` re-fetches lockfile dependencies from their
  recorded `remote_origin` and verifies bytes against the lockfile hashes.

The validation step at install time is the same regardless of source. It
runs entirely on bytes already on disk; no code from the artifact executes.

### Reproducible package artifact contract

For identical package inputs, compiler version, and `SOURCE_DATE_EPOCH`,
repeated `chelis reef build` invocations produce byte-identical
`<name>-<version>.tar.zst` and `<name>-<version>.chb` artifacts. Source archive
members are emitted in bytewise lexical order by their UTF-8 package-relative
paths. Every regular-file tar header has mode `0644`, uid `0`, gid `0`, and an
mtime equal to `SOURCE_DATE_EPOCH`; when the variable is absent, Reef uses the
fixed Unix epoch (`0`). A present value must be a non-negative integer number
of seconds or the build fails. Filesystem mtimes, ownership, permissions, and
directory enumeration order never enter the artifact.

The CHB continues to embed the SHA-256 of the resulting canonical source
archive. Changing `SOURCE_DATE_EPOCH` can therefore intentionally change both
artifacts; keeping it fixed (or absent) makes repeated builds reproducible.
The executable contract oracle is:

```sh
cargo test -p chelis-cli --test reef_build_reproducible
```

### Historical gap this doc closed

Before Phase A, Reef only installed from monorepo-built artifacts. New developers had
to clone shell repos and publish them into the local registry manually in dependency
order, builds could not recover from an empty registry, and lockfiles did not record
where installed bytes came from. Items 6-9 below are the shipped answer to that gap.

### Pre-launch backend choice

The pre-launch artifact backend is GitHub Releases on the canonical hosting
org's repositories. Each shell repository maintains its own release tags
(`v0.5.0` for Nautilus, `v0.5.0` for Coral, `v0.2.0` for Shoals,
`v0.4.0` for Octant — each shell's own semver trajectory; the version is
not synchronized across shells; the live pins are tracked in
`DEFAULT_BOOTSTRAP_LIST` in `crates/chelis-reef/src/lib.rs`). Each release
attaches the prebuilt `<name>-<version>.chb` and
`<name>-<version>.tar.zst` files (filenames omit the leading `v` even
though the tag carries it).

The dev team has authenticated access to the canonical org. Authentication is via
standard GitHub PATs supplied through the `GITHUB_TOKEN` environment variable; there is
no unauthenticated fallback because the canonical repos are private, so
unauthenticated fetch would simply 404.

The post-launch endgame is a registry server (Item 10). That is recorded
here as future work but is not designed in this round.

---

## Design

### Item 6 — `chelis reef install --from-github`

**Surface.**

```
chelis reef install --from-github <org>/<repo>@<tag>
```

`<tag>` is the shell's release tag, of the form `v<shell-version>` (for example,
`v0.5.0` for the current Nautilus release). This is the **shell's own version**, not
the chelis-compiler version; the two are independent. The shell's
`compiler = "=X.Y.Z"` pin in its `reef.toml` is the constraint that links
it to a chelis version.

Multiple `--from-github` flags are independent installs.

**Behavior.**

- Parse `<org>/<repo>@<tag>` into components.
- Resolve auth: read `GITHUB_TOKEN` from the environment; if unset, shell
  out to `gh auth token` to obtain one. Hard-fail with a clear error if
  neither yields a token, suggesting `export GITHUB_TOKEN=$(gh auth token)`
  as the fix. Authentication is mandatory because the canonical repos are
  private.
- Fetch the two release assets via the GitHub REST API in two steps:
  1. `GET https://api.github.com/repos/<org>/<repo>/releases/tags/<tag>`
     with `Accept: application/vnd.github+json` to look up the
     release's asset list. Find the entries whose `name` matches
     `<repo>-<ver>.tar.zst` and `<repo>-<ver>.chb`; capture each
     asset's numeric `id`.
  2. `GET https://api.github.com/repos/<org>/<repo>/releases/assets/<asset_id>`
     with `Accept: application/octet-stream` and the auth header to
     stream the bytes to a temp directory.

  GitHub's public-facing `/releases/download/<tag>/<asset>` URL form
  does not serve private-repo asset bytes even with a valid
  `Authorization: token …` header — it returns 404. The canonical
  chelis-lang shells are private during the pre-launch era; the API
  path is required, not optional. Test fixtures inject a localhost
  base URL via `CHELIS_REEF_GITHUB_BASE_API` (default
  `https://api.github.com`).
- Call the existing validation and placement logic, refactored out of
  `install_from_monorepo` into a shared
  `install_validated_artifact_pair(archive_path, shell_path, name, version,
  registry_root)` helper. Same SHA256 verification, same
  name/version-agreement check, same registry placement.
- Lockfile records the fetched artifacts' content hashes. First fetch is
  trust-on-first-use against the bytes the canonical repo serves;
  subsequent fetches verify against the pinned hashes from the lockfile
  (same semantics as `--from-monorepo`).
- Clear error categories: auth failure (`AuthMissing` for no token,
  `AuthRejected` for 401/403 on either API endpoint), metadata-step 404
  (`ReleaseTagNotFoundOrUnauthorized` naming the tag URL; this 404 is
  ambiguous because GitHub returns 404 both for a missing tag and for a
  private repo the token cannot read, so the message names both
  possibilities and points at `gh release view <tag> --repo
  <org>/<repo>` for verification, per issue #147), asset-step 404
  (`ReleaseAssetNotFound`, which fires only after metadata fetch
  succeeded, so the repo is reachable and the tag exists; it names the
  expected asset name plus the assets actually present on the release),
  rate-limit (`RateLimited` carrying `Retry-After`), 5xx
  (`ServerError`), network/DNS (`Network`), hash mismatch on subsequent
  fetch (`Validation` from the shared helper; do not auto-overwrite),
  I/O error (`Io`).

**Acceptance oracle.**

- `chelis reef install --from-github <org>/nautilus@v0.5.0` with
  `GITHUB_TOKEN` set in env successfully fetches and installs Nautilus.
  The local registry afterward has `packages/nautilus/0.5.0/` populated
  identically to the `--from-monorepo` outcome.
- Without the token, fails with a clear error pointing at the env var.
- With a tampered local cache (alter the archive's hash), subsequent
  install fails with hash-mismatch error.
- The MCP and HTTP surfaces of Tide are unaffected.

**Scope.** Approximately 180 lines of new code across two files:
- `crates/chelis-reef/src/lib.rs`: `install_validated_artifact_pair`
  is the shared validation+placement helper (currently around line
  892); it is invoked by both `install_from_monorepo` (line ~1000) and
  `install_from_github` (line ~1528). Line numbers may drift; the
  helper's name is the stable reference.
- `crates/chelis-cli/src/main.rs`: `--from-github` handler that parses
  the repo+tag and calls `install_from_github`. The CLI handler stays
  thin — the HTTP layer lives in `chelis-reef` per the locked layering
  decision in this document. ~100 lines.
- `crates/chelis-reef/Cargo.toml` adds
  `reqwest = { version = "0.11", default-features = false, features = ["blocking", "rustls-tls"] }`.

---

### Item 7 — Bootstrap: `chelis reef install --bootstrap`

**Surface.**

```
chelis reef install --bootstrap <list>
```

Where `<list>` is a sequence of `org/repo@tag` entries (or a config-file
reference, or the built-in default list of canonical shells). Installs
them in topological dependency order, using Item 6's per-shell install
path under the hood.

**Behavior.**

- Parse the entries into a graph keyed by `(name, version)` with edges
  derived from the shell's published `reef.toml` dependencies. The
  dependency edges are read from each shell's `<name>-<version>.tar.zst`
  archive after fetch but before placement.
- Topologically sort. If a cycle is detected, fail with a clear error
  naming the cycle.
- Install each shell in order via Item 6's `install_validated_artifact_pair`
  + lockfile update.
- The built-in default list covers the canonical shells: `nautilus`,
  `coral`, `shoals`, `octant`. This list is hard-coded for the
  pre-launch dev team; multi-publisher generalization is post-launch.

**chelis-std is the language runtime, not a shell.** The runtime is
distributed bundled with the compiler — it version-marches with the
toolchain and cannot be substituted independently. Programs depend on
it the same way Rust programs depend on `core`/`std`. Concretely:

- `chelis-std` is **never** in the bootstrap input list. An explicit
  `chelis-std` entry in `--bootstrap` arguments is rejected with a
  typed `BootstrapError::RuntimeNotABootstrapTarget` error naming
  both the requested and the bundled version.
- A shell's `reef.toml` may declare `chelis-std = { version = "X" }`.
  The bootstrap installer soft-verifies `X` against the compiler's
  bundled runtime version: on match the dep is filtered from the
  bootstrap graph (it is implicit, not an edge in the install loop);
  on mismatch the bootstrap aborts with a typed validation error
  naming both versions.
- Lockfile entries for chelis-std use `LockSource::Bundled
  { compiler_version }`, recording the version of the compiler that
  supplied the bytes for auditability. There is no archive to fetch
  and integrity comes from the compiler binary itself.

**Bundling and lockfile synthesis (Phase A correction).** The runtime
bytes — both the source archive (`.tar.zst`) and the shell
(`.chb`) — are baked into the chelis binary at compile time via
`include_bytes!()` in the `chelis-std-bundle` crate. The reef loader
checks for chelis-std specifically and serves bytes from the embedded
bundle; everything else falls through to the local-registry path.
Two consequences:

1. **Project-driven blanket synthesis.** Every reef.toml has a
   `compiler =` pin, and that pin IS the runtime declaration.
   `build_lockfile` therefore unconditionally records a
   `LockSource::Bundled` chelis-std entry for every project,
   regardless of whether the project listed chelis-std in
   `[dependencies]`. The synthesis is idempotent with the explicit-
   listing path: when chelis-std is in the dep graph, the same
   `Bundled` entry is produced; when absent, the lockfile-build step
   appends it after iterating the graph. The `archive_sha256` and
   `shell_sha256` fields come from the embedded bundle bytes for
   both paths so the recorded entry is byte-identical.
2. **Compile-time embedding, not two-stage build.** Bundle artifacts
   (`crates/chelis-std-bundle/dist/chelis-std-<version>.{tar.zst,chb}`)
   are committed to the repo. The pipeline is "regenerate artifacts ->
   commit -> build"; `scripts/regenerate_chelis_std_bundle.py` is the
   canonical regen entry. The bundle crate's build.rs verifies the
   dist files exist and emits `cargo:rerun-if-changed=` so cargo
   invalidates the bundle when the bytes change.
3. **No registry seeding required.** `chelis reef build` against a
   project that depends on chelis-std (implicitly or explicitly)
   succeeds against an empty `$CHELIS_REEF_HOME`. The bundled bytes
   are reachable from the chelis binary, not from the filesystem.
   Manual gates that previously pre-installed chelis-std via
   `--from-monorepo` no longer need that step.

**Acceptance oracle.**

- A clean dev environment with `GITHUB_TOKEN` set runs `chelis reef
  install --bootstrap` (no arguments — uses the default list) and ends
  with all four canonical shells (`nautilus`, `coral`, `shoals`,
  `octant`) installed in the local registry, in the correct order so
  each shell's dependencies were already present when it was
  installed.
- Explicit list form `chelis reef install --bootstrap
  <org>/nautilus@v0.5.0 <org>/coral@v0.5.0` installs only the two
  named shells in topological order.

**Scope.** ~50 lines on top of Item 6. The graph build is small (each
shell has a handful of dependencies); the topological sort is a textbook
post-order traversal; the install step is a loop over Item 6's helper.

---

### Item 8 — Auto-fetch during `chelis reef build`

**Surface.** Default-on; opt out with `--no-auto-fetch`.

**Behavior.**

- The missing-from-registry error site at
  `crates/chelis-reef/src/lib.rs:1243-1245` currently bails with
  "missing from local registry index — run `chelis reef build` first to
  populate the cache". The message is a workaround for the absence of
  remote fetch.
- When auto-fetch is enabled (the default), insert a fetch attempt
  before bailing. The fetch source is derived from the lockfile's
  `remote_origin` field (Item 9) if present; otherwise from the
  canonical hosting-org default for the named package (e.g.
  `<canonical-org>/<name>@v<version>`).
- On fetch success, retry the registry lookup. On fetch failure, bail
  with a clearer error than the current one: it now names the URL that
  was tried and the failure category (auth, 404, hash mismatch).
- `--no-auto-fetch` disables this fallback for users who want explicit
  control of when network access happens during build.

**Acceptance oracle.**

- A clean dev environment with `GITHUB_TOKEN` set runs `chelis reef
  build` on a project that depends on Nautilus (no manual install
  step), and the build succeeds with an auto-fetch in the middle.
- Same project under `--no-auto-fetch` fails with the existing error
  shape but improved wording.
- A project whose lockfile pins Nautilus to a specific
  `remote_origin` and whose registry is empty fetches from that
  origin under auto-fetch.

**Scope.** Roughly one day of work. The fetch path itself is Item 6's
helper; the work is wiring it into the build error site, propagating
the `--no-auto-fetch` flag through the build command, and refining the
error catalog.

**Canary/dev escape hatch — `CHELIS_REEF_ALLOW_DEP_COMPILER_DRIFT`.** When
this env is set to a non-empty, non-`0` value, `validate_manifest` waives
*only* the `package.compiler` equality assertion — every other manifest
check stays enforced — and emits a loud per-package warning naming the
package and both pins. Its sole purpose is to let the ecosystem drift
canary (and local dev) build a shell's *code* against chelis HEAD while its
released dependencies still pin the compiler they were cut against: the
canary asks "does this code still compile against HEAD", whereas a
dependency's pin freshness is that shell's own release-cadence concern, not
a code-drift signal. It is **not** for a shell's own CI gate, where the pin
equality is the whole point of the check; leave it unset there.

---

### Item 9 — Lockfile records remote origin

**Surface.** Lockfile gains an optional field on each entry:

```toml
[[dependencies]]
name = "nautilus"
version = "0.5.0"
compiler = "=<compiler-version>"
archive_sha256 = "..."
shell_sha256 = "..."

[dependencies.source]
kind = "local_registry"
remote_origin = "github://<canonical-org>/nautilus@v0.5.0"
```

The `remote_origin` field is optional for backward compatibility;
older lockfiles without it continue to deserialize. New lockfiles
populate it whenever the install came from a remote source.

**New command:** `chelis reef install --from-lockfile`. Reads the
project's lockfile, re-fetches every dependency from its recorded
remote origin, and validates against the lockfile's pinned hashes.
This gives a one-command "match the lockfile" path for fresh
checkouts.

**Behavior.**

- Lockfile serialization gains the optional field. Deserialization is
  forward-compatible (old lockfiles deserialize fine; new ones round-
  trip through cargo-style serde with the field present or absent).
- `chelis reef install --from-lockfile` walks the lockfile, calls
  Item 6's helper for each entry that has a `remote_origin`, and
  errors clearly on entries that lack one (suggesting a manual
  `--bootstrap` to populate them).

**Acceptance oracle.**

- One developer fetches a set of dependencies via `--from-github`
  and commits the resulting `reef.lock`.
- A second developer clones the repo and runs `chelis reef install
  --from-lockfile`; ends with the same local registry state, all
  hashes verifying against the lockfile pins.
- The same operation from a stale registry (different bytes already
  installed under the same name/version) fails with hash-mismatch.

**Scope.** Roughly one day. Field addition is small; the install-from-
lockfile command is a thin wrapper over Item 6's helper.

---

### Item 10 — Public registry server (deferred)

Post-launch endgame. A dedicated registry service replaces GitHub
Releases as the artifact backend. Adds version search, semver
resolution, multiple publishers, discoverability. Substantial product
work; do not start without a specific driver pulling for it. Recorded
here for forward-compatibility so the design of Items 6-9 stays
compatible with a future migration (the lockfile's `remote_origin`
already records source provenance in a string form that can route to
either a GitHub release or a registry URL).

---

### Item 11 — First-class binary distribution (chelis#468)

**Status:** planned. Tracked by chelis#468; framed as Layer 1 of
[`chelis_packaging_and_install.md`](chelis_packaging_and_install.md).

Items 6-9 deliver and verify *source* packages (a `.tar.zst` archive plus a
`.chb` shell). They cannot carry a **compiled binary**. Every binary-bearing
component in the ecosystem — the toolchain, nautilus, shoals, and octant's
translator — therefore distributes its binary out of band through a GitHub
release tarball, and every consumer hand-rolls download-by-tag plus SHA
verification (octant's `docs/src/consuming.md`; C Note's verify-in-Dockerfile
step). reef does the source half; an informal channel does the binary half, and
the integrity work reef would own is reimplemented per consumer. Item 11 folds
that verb into the resolver and the lockfile.

**Manifest.** A dedicated `[artifacts]` section — binaries are neither reef
packages nor Cargo crates, so they get their own artifact-class section, the
same shape chelis#571 established with `[chelis-src]`:

```toml
[artifacts.octant-translator]
repo = "Chelis-Lang/octant"          # publisher; sensible default
tag  = "v0.4.2"
platforms.linux-x86_64 = { asset = "octant-translator-linux-x86_64.tar.gz", sha256 = "…" }
platforms.darwin-arm64 = { asset = "octant-translator-darwin-arm64.tar.gz",  sha256 = "…" }
```

`linux-x86_64` is the minimum; additional platforms are additive. The
publisher's release SHA is the source of truth — the consumer pins and verifies
against it rather than transcribing a copy.

**Install flow.** `chelis reef install` (and `reef setup`) resolves the **host
platform** entry, downloads the asset through the existing GitHub REST path
(Item 6's `install_from_github`), **SHA-256-verifies fail-closed** — a mismatch
aborts before any placement — extracts, and places a runnable binary at
`~/.chelis/bin/<name>`. A new `chelis reef which <artifact>` prints the resolved
path so consumers point at it with no out-of-band knowledge.

**Lockfile.** A new `LockSource::Binary { remote_origin, platform, asset,
sha256 }` variant on `LockedDependency`. Per the lockfile-ownership rule
([`chelis_packaging_and_install.md`](chelis_packaging_and_install.md) §3),
binaries *belong* in `reef.lock`: reef fetches them, their integrity primitive
is a content hash, and nothing else records them. Threads through
`build_lockfile` and `install_from_lockfile`, so `--from-lockfile` re-fetches
and re-verifies a binary exactly as it does a source package.

**Reuse.** `install_validated_artifact_pair` (validation + placement); the
shipped SHA-256 helpers; `install_from_github` (REST asset fetch);
`try_github_token` (private-repo auth, added in chelis#571); the Item 9
`remote_origin` lockfile pattern; the `cmd_reef_*` dispatch. The one genuinely
new mechanism is host-platform / target-triple selection.

**Acceptance:**

- `reef.toml` expresses a per-platform, SHA-pinned binary artifact.
- `reef install` on a supported platform downloads, SHA-verifies fail-closed,
  and installs a runnable binary, with no consumer-side hand-rolled download or
  SHA literal.
- The binary dependency and its SHA are recorded in the lockfile.
- A hybrid package (octant) declares both its source shell and its binary, and a
  consumer obtains both through one dependency set.
- octant's `consuming.md` workaround and C Note's download-and-verify Dockerfile
  step retire in favor of the reef path.

**Toolchain note.** The chelis toolchain itself is binary-bearing, but its
installer of record is `chelisup` (chelis#164), not `reef install` — see
[`chelis_packaging_and_install.md`](chelis_packaging_and_install.md) §5.6. Item
11 may *describe* the toolchain as an `[artifacts]` entry for reproducibility,
but does not install it.

---

## Acceptance summary

Per item:

- Item 6: install single shell from canonical org's release; lockfile
  records hashes; subsequent fetches verify.
- Item 7: install all canonical shells in dependency order from one
  command.
- Item 8: a fresh `chelis reef build` succeeds end-to-end without
  manual install steps.
- Item 9: a developer can re-create another developer's local
  registry state from the committed lockfile alone.
- Item 11 (planned): a per-platform SHA-pinned binary is declared once,
  downloaded + verified fail-closed, placed runnable, and lock-recorded;
  a consumer obtains a hybrid package's source shell and binary through one
  dependency set, retiring the out-of-band download-and-verify channel.

With Items 6-9 shipped, a fresh dev environment's onboarding is
`git clone <project> && export GITHUB_TOKEN=$(gh auth token) &&
chelis reef build`. That is the experience. Item 11 extends it to binary
dependencies; the full cross-class onboarding becomes `chelis reef setup`
([`chelis_packaging_and_install.md`](chelis_packaging_and_install.md) §7).

---

## Historical Sequencing

Item 6 had to land first; Items 7, 8, 9 each depended on Item 6's helper.
This section is retained as implementation history, not a remaining schedule.

Item 10 is demand-driven and not part of this design's
implementation scope.

The original Item 6-9 estimate was 3-4 days of focused work. It is retained only to
explain the scope of the shipped Phase A change set.

---

## Reused machinery

- `install_validated_artifact_pair` is the shared validation+placement
  helper (currently in `crates/chelis-reef/src/lib.rs` around line
  892; line numbers drift, the function name is the stable
  reference). Item 6 extracted it from `install_from_monorepo` so
  both `--from-monorepo` and `--from-github` call into it. Both
  produce byte-identical local registry state for the same
  `(name, version)` — locked as a contract invariant by the byte-
  equality sub-case of the named oracle.
- SHA256 verification logic already shipped (computes
  `archive_sha256` and `shell_sha256` per
  `LockedDependency` at `lib.rs:64-71`).
- Lockfile schema (cargo-style toml serde). Item 9 added an optional field that
  round-trips via existing serde paths.
- `crates/chelis-cli/src/main.rs::cmd_reef_*` handlers as the
  command-dispatch entry points.
- The error site at `lib.rs:1243-1245` for Item 8's insertion
  point.

---

## Out of scope

- **Semver / version-range resolution.** Today's reef is exact-pin
  only. Version ranges are a real feature for a public ecosystem
  but not needed for a small private team using exact pins. Defer
  to public-launch alongside Item 10.
- **Mirror configuration / private registries.** Users may want to
  host their own mirrors. Post-launch concern.
- **Multi-publisher support.** Today's hardcoded canonical-org
  default works for the team. Multi-publisher comes with Item 10.
- **Authenticity beyond content hashing.** Cryptographic signing
  of artifacts (publisher key, signature verification on install)
  is tracked separately under the trust-stack expansion's Item 5
  and is demand-driven; not in this round.
- **Bit-reproducible artifact comparison across machines.** The
  install path verifies bytes received against bytes pinned in the
  lockfile, but does not verify that two independent rebuilds from
  source would produce bit-identical bytes. That is a separate
  workstream contingent on auditing the C emitter for non-
  determinism.

---

## Cross-references

- `effect_taxonomy_expansion.md` Item 4 (install-time effect
  manifests) is the natural composition point: the install path
  designed here gains a `--print-effects` and `--refuse` flag once
  the effect aggregation is available. The two designs share the
  install boundary.
- `chelis_trust_stack.md` references this doc as the Phase A distribution surface.
- `spec/12-roadmap.md` Phase A (distribution unblock) covers
  Items 6-9 of this doc.
