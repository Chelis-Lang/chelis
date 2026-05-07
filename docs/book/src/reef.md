# Reef and Packages

Reef is Chelis's package and shell distribution layer. This chapter
covers the day-to-day workflow: what's bundled with the compiler, how
to install shells, how `chelis reef build` works, and what the
lockfile records.

## Runtime vs Shells

Chelis distinguishes a single **runtime** from a family of substitutable
**shells**:

- `chelis-std` is the language **runtime**. It is compiler-bundled —
  every Chelis program depends on it implicitly the same way a Rust
  program depends on `core`/`std`. It version-marches with the
  toolchain and is **never** installed via reef.
- The four canonical shells are `nautilus`, `coral`, `shoals`, and
  `octant`. Each is a distributable library that builds on the
  runtime; users install them via reef.

The runtime bytes (the `chelis-std` source archive and `.chb` shell)
are embedded into the `chelis` binary at compile time via the
`chelis-std-bundle` crate. The reef loader serves them directly when
a project depends on chelis-std; the local registry is not
consulted for the runtime.

## Package Manifest

Every reef package has a `reef.toml`:

```toml
[package]
name = "demo"
version = "0.1.0"
compiler = "=0.5.0"
module_prefix = "Demo"

[dependencies]
nautilus = { version = "0.5.0" }
```

Package source normally lives under `src/`. Module declarations should line up with the
manifest `module_prefix`; for example, `src/nn/linear.ch` in the manifest above would
declare a module under `Demo.Nn.Linear`.

The `compiler =` pin is exact (no version ranges in this round) and
implicitly declares the chelis-std runtime version: a project that
pins `compiler = "=0.5.0"` automatically gets chelis-std at the
runtime version that compiler ships. You can also list `chelis-std`
explicitly in `[dependencies]` (the installer soft-verifies that the
declared version matches the bundled version, mismatches surface a
typed error), but it is never required.

## Install Paths

`chelis reef install` accepts four sources, used independently:

```sh
# Discover packages under a chelis monorepo checkout (offline).
chelis reef install --from-monorepo <path>

# Fetch a single shell from the canonical hosting org's GitHub
# release. Authentication is required (canonical repos are private
# pre-launch); GITHUB_TOKEN must be set or `gh auth token` must work.
chelis reef install --from-github <org>/<repo>@<tag>

# Install a topo-ordered set of shells. With no arguments, uses the
# built-in default list (nautilus, coral, shoals, octant).
chelis reef install --bootstrap [<org>/<repo>@<tag> ...]

# Re-fetch every dependency from its lockfile-recorded remote origin
# and verify against the lockfile's pinned hashes.
chelis reef install --from-lockfile
```

All four paths use the same validation+placement helper
(`install_validated_artifact_pair`): SHA256-verify the archive and
shell, check name/version agreement, place under
`$CHELIS_REEF_HOME/packages/<name>/<version>/`. Local registry state
is byte-identical regardless of which install path produced it.

### chelis-std is not installable

`chelis reef install --bootstrap chelis-std@<v>` is rejected with a
typed `BootstrapError::RuntimeNotABootstrapTarget`, naming both the
requested version and the compiler's bundled version. The runtime
ships with the compiler; there is no separate install step.

### Asset naming

Each shell's GitHub release attaches two assets per published version:

- `<name>-<version>.tar.zst` — the source archive
- `<name>-<version>.chb` — the prebuilt shell artifact

Note the convention: the **tag** is `v<version>` (with the leading
`v`), but the **asset filenames** are `<name>-<version>.{tar.zst,chb}`
without the leading `v`. Both forms are accepted by the
`<org>/<repo>@<tag>` parser (`@v0.5.0` and `@0.5.0` are equivalent).

## Auto-fetch During Build

`chelis reef build` is auto-fetch-by-default: if a dependency is
missing from the local registry, the build attempts to fetch it from
the canonical hosting org (or from the `remote_origin` recorded in
the lockfile, when present) before falling back to the
"missing-from-registry" error. Pass `--no-auto-fetch` to disable this
fallback.

The result of all four items above is that a fresh dev environment's
onboarding is just:

```sh
git clone <project>
export GITHUB_TOKEN=$(gh auth token)
chelis reef build
```

Auto-fetch handles the rest. `GITHUB_TOKEN` is mandatory because the
canonical-org shell repositories are private during the pre-launch era;
the install path hard-fails with a clear error if no token is
available.

## Lockfile

`reef.lock` records every resolved dependency as a tuple of
`(name, version, source-kind, compiler-pin, archive_sha256,
shell_sha256)`. The `source-kind` discriminator (`LockSource`) has
three variants:

- `Path` — a path-source dependency (rare; mostly for local development)
- `LocalRegistry { remote_origin }` — installed from the local
  registry. The optional `remote_origin` records where the bytes were
  originally fetched from, in scheme-tagged URI form (currently
  `github://<org>/<repo>@<tag>`); pre-Item-9 lockfiles omit the field
  and round-trip cleanly without it.
- `Bundled { compiler_version }` — the chelis-std runtime, served
  from the compiler binary's embedded bundle. Recorded for
  auditability — a lockfile reader can see which compiler version
  supplied the runtime bytes.

**Project-driven blanket synthesis:** every reef.toml's `compiler =`
pin is itself the runtime declaration. `build_lockfile` therefore
records a `LockSource::Bundled` chelis-std entry on every project,
regardless of whether the project listed chelis-std in
`[dependencies]`. Lockfiles produced by older compilers that recorded
chelis-std as `LocalRegistry` are auto-migrated to `Bundled` on read
and rewritten on the next `chelis reef build`.

## Import Syntax

```chelis-surf-fragment
import Std.Nn.Linear(..)
import Nautilus.LinAlg(matmul_wrap, transpose)
```

`Std.*` resolves to the bundled chelis-std runtime; `Nautilus.*`,
`Coral.*`, `Shoals.*`, and `Octant.*` resolve to whichever version of
each shell the lockfile pins.

## Typical Workflow

```sh
chelis reef init demo --module-prefix Demo --output demo
cd demo
# Edit reef.toml to pin compiler version and any shells you need.
chelis reef build
```

## Environment Variables

- `CHELIS_REEF_HOME` — local registry root. Default is
  `~/.chelis/reef`.
- `GITHUB_TOKEN` — required for any remote fetch
  (`--from-github`, `--bootstrap`, auto-fetch during build). Falls
  back to `gh auth token` if unset.
- `CHELIS_REEF_GITHUB_BASE_API`, `CHELIS_REEF_GITHUB_BASE` —
  test-fixture overrides for the GitHub API and download base URLs.
  Production use defaults to `https://api.github.com` and
  `https://github.com` respectively.

## Concurrent Builds

`chelis reef build` is safe to run concurrently across independent
working trees. The local registry is protected by file locks
(`flock`); two builds racing for the same package serialize cleanly
without corrupting the registry.

## See Also

- `spec/design/reef_distribution.md` — full distribution design,
  including error categories and acceptance oracles.
- `spec/design/chelis_canonical_reference.md` §5 — runtime-vs-shell
  distinction and the canonical shell roster.
- `spec/design/phase3j_pre_release.md` — release contract and
  downstream pinning policy.
