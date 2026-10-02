# Reef and Packages

Reef builds and installs Chelis packages, also called shells. A package has a
`reef.toml` manifest and Chelis source files. `chelis-std` ships inside the
compiler: a Reef package can import `Std.*` modules without installing them
through Reef. A standalone file outside a package cannot import `Std` modules.
[Install](install.md) covers the toolchain needed to run Reef commands.

## Create and build a package

With a release toolchain installed, run:

```sh
chelis reef init demo --module-prefix Demo --output demo
cd demo
chelis reef build
```

`reef init` creates a valid, dependency-free package for the running compiler.
The generated `src/main.ch` declares `module Demo.Main`. `reef build` checks the
package and produces a source archive and a `.chb` package artifact; it does
not create an executable. The commands produce these files:

| File | Purpose |
|---|---|
| `reef.toml` | Package name, version, exact compiler pin, and dependencies. |
| `src/main.ch` | The initial Chelis module. |
| `.gitignore` | Added when needed to ignore Reef's project write lock and temporary files. |
| `.reef-write.lock` | The ignored file used to coordinate project writes. |
| `reef.lock` | The selected dependency versions, sources, and hashes. |
| `dist/demo-0.1.0.tar.zst` | The package source archive. |
| `dist/demo-0.1.0.chb` | The built package artifact. |

The generated manifest has package version `0.1.0` and an exact
`compiler = "=X.Y.Z"` pin for the toolchain that ran `reef init`. Package
source normally lives under `src/`. Module names follow
`module_prefix` and the file path: `src/nn/linear.ch` in this example must
declare `module Demo.Nn.Linear`.

## Add dependencies

Add a package under `[dependencies]` in `reef.toml`. Resolver 2 accepts
Semantic Version requirements, including ranges such as:

```toml
[dependencies]
nautilus = "^0.7"
```

If the manifest already has a `[dependencies]` table, add only the entry.
Use `=X.Y.Z` for an exact dependency version; the compiler pin is always
exact. Choose a version compatible with your compiler.

`chelis reef build` prefers a valid `reef.lock`. If dependencies need to be
resolved and the local registry cannot complete the graph, Reef searches
GitHub Releases. Use `chelis reef build --no-auto-fetch` when the build must
avoid network access. A package with no external dependencies, such as the
one above, builds without a remote fetch.

To inspect available versions or deliberately refresh the lockfile from a
resolver-2 package directory, run:

```sh
chelis reef outdated
chelis reef outdated --json
chelis reef update
```

Both commands accept an optional package name; `update` without one refreshes
the dependency graph. Add `--offline` to use local candidates only.
`outdated` reports versions without changing the lockfile or installing
packages.

## Install dependencies for a project

In a cloned project, first install the exact `X.Y.Z` from its `reef.toml`
`[package] compiler = "=X.Y.Z"` pin with `chelisup install X.Y.Z`. The
`chelis` shim needs that toolchain before it can start `reef setup` inside
the project. Then run:

```sh
chelis reef setup
chelis reef build
```

Setup reads `reef.lock` when present and attempts to install its remotely
sourced packages and host binaries. Path dependencies and the compiler-bundled
`chelis-std` have no separate Reef download; a binary for another platform is
skipped. A package installed only from a local checkout cannot be fetched from
the lockfile, and setup reports an error. Without `reef.lock`, setup skips the
install step. Setup also syncs Chelis source crates if `[chelis-src]` is
declared, then prints the `reef doctor` summary. Build is a separate command.
See [Install](install.md#use-a-project) for the full toolchain sequence.

To repeat the lockfile install step explicitly, run
`chelis reef install --from-lockfile` in the package directory. Reef compares
fetched files with the lockfile hashes and reports mismatches.

## Other install sources

- `chelis reef install --from-github ORG/REPO@vX.Y.Z` fetches a package from
  a GitHub release and records its origin in the local registry.
- `chelis reef install --from-monorepo PATH` installs package artifacts already
  built under a Chelis source checkout's `packages/` directory; the checkout
  commits none, so run `chelis reef build` in each package first. Such local
  installs have no GitHub origin for later lockfile fetching. The bundled
  `chelis-std` needs no install: every compiler embeds it.
- `chelis reef install --bootstrap` uses the toolchain's fixed list of
  shell release tags. You can supply explicit `ORG/REPO@TAG` entries
  instead.

Remote GitHub fetches go through the authenticated GitHub REST API, so they
need a token even for a public repository: Reef uses `GITHUB_TOKEN` if set,
then `gh auth token`. A private repository also needs read access. Source
packages install in the local Reef registry, normally `~/.chelis/reef`.
`CHELIS_REEF_HOME` selects a different registry; setting `CHELIS_HOME` alone
does not move it.

A package with declared binary artifacts can place a host binary under
`$CHELIS_HOME/bin` (default `~/.chelis/bin`). `chelis reef which NAME` prints
its installed path. `chelis reef publish` builds a package and copies its
archive and `.chb` into the **local** registry; it does not upload a GitHub
release.

## Check package state

`chelis reef doctor` reports the pinned toolchain, declared Chelis source
crates, and binary artifacts for the current package. It is read-only and does
not check installed source-package dependencies. Use
`chelis reef doctor --root DIR` to scan `DIR` and its immediate
subdirectories for packages.

For reproducible artifacts, keep package inputs, compiler version, and
`SOURCE_DATE_EPOCH` fixed. If `SOURCE_DATE_EPOCH` is unset, Reef uses timestamp
zero in the source archive; if set, it must be a non-negative integer number
of seconds.

For program imports and the bundled library, see [Runtime and Standard
Library](stdlib.md). For compiler command workflows, see [CLI Workflow](cli.md).
