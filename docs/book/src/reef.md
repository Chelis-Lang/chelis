# Reef and Packages

Reef is Chelis's package and shell distribution layer. This chapter
covers the day-to-day workflow: what's bundled with the compiler, how
to install shells, how `chelis reef build` works, and what the
lockfile records.

## Runtime vs Shells

Chelis distinguishes a single **runtime** from a family of substitutable
**shells**:

- `chelis-std` is the language **runtime**. It is compiler-bundled.
  Every Chelis program depends on it implicitly the same way a Rust
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

Every Reef package has a `reef.toml`:

```toml
schema = "3"

[package]
name = "demo"
version = "<version>"
compiler = "=<version>"
module_prefix = "Demo"
resolver = "2"
description = "Example Chelis package"
license = "MIT"
repository = "https://github.com/example/demo"
readme = "README.md"

[dependencies]
nautilus = "^0.7"
```

Package source normally lives under `src/`. A file's module declaration is fixed by the
manifest `module_prefix` and the file's path beneath its source root: `src/nn/linear.ch`
in the manifest above must declare `module Demo.Nn.Linear`. This is a load-time
requirement, not a convention — reef rejects a package whose module identities disagree
with its layout, and `chelis lint --check` reports the same mismatch as
`reef-module-identity`. The rule, including how additional source roots contribute a
component, is `spec/01-nomenclature.md` §6.5.

A conformant shell can declare owned domain skills through `[conform] local_skills`. Reef validates and preserves this table.

The `compiler =` pin is exact (no version ranges in this round) and
implicitly declares the chelis-std runtime version: a project that
pins an exact compiler version automatically gets chelis-std at the
runtime version that compiler ships. You can also list `chelis-std`
explicitly in `[dependencies]` (the installer soft-verifies that the
declared version matches the bundled version, mismatches surface a
typed error), but it is never required.

### Shell guidance and skills

`chelis reef conform sync` refreshes the toolchain-managed regions in a shell's
`AGENTS.md` and `docs/CHELIS_SURFACE.md` and synchronizes `agent-skills/`.
The `AGENTS.md` region contains the pinned toolchain's complete root Chelis
agent contract. Shell-owned text outside the managed document regions remains
in place. Sync also points `.claude/skills` and `.codex/skills` at the one
materialized `agent-skills/` tree with `../agent-skills` symlinks so both agents
discover the same set without duplicated copies.
Including the complete contract is the default. Each shell should review which
parts apply, exclude irrelevant inherited sections, and keep its own guidance
when it remains relevant and current. The shell can adjust the synchronized
skill set in `reef.toml`:

```toml
[conform]
local_skills = ["my-shell-domain"]
excluded_skills = ["backend-numerics", "cli-surface"]
```

`local_skills` preserves shell-owned additions. `excluded_skills` removes exact
embedded shared-skill names during sync; removing a name restores the current
toolchain copy. Sync and audit reject unknown excluded names so misspellings and
upstream renames are visible.

To omit an irrelevant section of the inherited `AGENTS.md`, put an exact
heading selector span anywhere outside its managed block:

```markdown
<!-- shell-local:exclude:begin -->
<!-- ### Numeric Surface Discipline -->
<!-- shell-local:exclude:end -->
```

Each selector removes the heading and its section through the next heading of
equal or shallower depth. Removing the selector restores the current upstream
section. Selecting `# Chelis Agent Contract` removes the whole inherited body.
Shell-owned prose outside the managed block is never replaced. The begin and
end control markers must be standalone Markdown comments; markers shown inside
a code fence or an enclosing HTML block are rejected rather than activated.

To keep a shared skill while removing irrelevant inherited sections, add exact
heading selectors inside its trailing shell-local block:

```markdown
<!-- shell-local:begin -->
<!-- shell-local:exclude:begin -->
<!-- ## Backend-only workflow -->
<!-- ## Device validation -->
<!-- shell-local:exclude:end -->

## Shell-specific guidance

Local additions remain here.
<!-- shell-local:end -->
```

Each comment-wrapped selector removes that heading and its section. Removing the
selector restores the current upstream section. Sync and audit fail if a
selector is missing, duplicated, malformed, or overlaps another selected
section. The same standalone-comment rule applies to these control markers.
Run `chelis reef conform sync --help` for the same configuration summary at the
command line.

### Package metadata

Manifest schema 3 accepts optional package metadata. Schema 2 rejects these fields as unknown keys.

The available fields are:

- `description`
- `license`
- `license-file`
- `repository`
- `documentation`
- `homepage`
- `readme`

A description contains 1 through 512 UTF-8 bytes after trim. It cannot contain line breaks, control characters, or Unicode format characters.

The `license` field contains an SPDX expression of at most 1024 UTF-8 bytes. The `license-file` field records a custom license.

A manifest cannot declare both license fields.

Raw and normalized package URLs contain at most 2048 UTF-8 bytes. They use HTTPS, contain a host, and contain no credentials.

Package URLs do not select package sources.

A declared file path uses Unicode NFC and `/` separators. It rejects unsafe components, format characters, platform-sensitive names, and Reef document names.

Each declared file contains at most 4 MiB. Reef opens each path without symbolic-link traversal on Linux and macOS.

Reef reads each file twice from one handle. A content mismatch stops archive replacement.

Metadata is not part of the logical package name/version or dependency
selection. Its fields are not serialized into `reef.lock`, `index.json`, or
`.chb`. Changing the manifest or a declared file does change the source archive
hash; the `.chb` embeds that hash, and lock and registry records of artifact
hashes must match the published bytes.

### Package identities and resolution

Package names use lowercase ASCII letters, digits, and internal hyphens. Names start with a lowercase letter and contain at most 64 bytes.

Package versions use complete Semantic Versions. Prerelease versions are valid. Package build metadata is not valid.

Manifest schema 1 keeps exact dependency semantics. A schema-1 dependency version such as `0.7.2` selects only `0.7.2`.

Manifest schema 2 requires `package.resolver = "2"`. Resolver 2 uses Cargo-style requirements and accepts string or inline dependency forms.

Use `=0.7.2` for an exact resolver-2 requirement. A path dependency can include the same exact requirement.

Resolver 2 sorts package names by byte value. It sorts unlocked candidates by descending Semantic Version precedence.

A compatible lock candidate sorts before other candidates during normal resolution. An explicit refresh removes that preference for the selected package.

Reef selects one version for each package name. Shared limits cover candidates, packages, dependencies, depth, and resolver states.

### Discovery modes

Every resolver operation uses one mode:

- `Locked` reuses a valid exact lock and does not list releases.
- `Resolve` uses a complete local graph before it uses a remote provider.
- `Refresh` lists remote candidates when a complete local graph exists.
- `Inspect` uses refresh discovery without final registry, index, or lock writes.

GitHub Releases is the only remote provider. Reef does not use descriptive package URLs for source selection.

When network access is disabled, Reef does not create a provider client. Local candidates and exact locked origins remain available.

A locked hash failure or unavailable origin is a hard error. `reef update` and
`reef outdated` check registry index entries against existing lock hashes even
when cached package directories are missing, including packages outside a
targeted refresh. If an index entry is absent, they compare fetched archives
and shells against the matching existing lock before reporting versions or
publishing packages. Reef does not replace damaged locked bytes with another
version.

### Remote limits

Reef applies these production limits:

| Resource | Limit |
|---|---:|
| Release pages for each package | 10 |
| HTTP requests for each command | 2048 |
| Accepted release tags for each package | 256 |
| Candidate manifests for each package | 64 |
| Compressed candidate archive | 64 MiB |
| Scanned candidate bytes | 256 MiB |
| Parsed `reef.toml` | 1 MiB |
| Total candidate download bytes | 1 GiB |
| Time for each HTTP request | 60 seconds |

A limit error names the resource, limit, observed value, package, and provider operation.

When an existing registry lock pin has no verified local package or index entry,
Reef reserves one candidate-manifest slot for that exact release before scanning
newer candidates. If the pin is absent from the bounded release listing, cannot
be fetched, or disagrees with the locked archive or shell hash, refresh and
inspection fail without selecting another version, even for an unrelated package
in a targeted refresh.

Reef scans candidate archives without archive extraction. It accepts one regular root `reef.toml` and rejects unsafe archive entries.

### Update and outdated commands

Run a full refresh:

```sh
chelis reef update
```

Run a targeted refresh:

```sh
chelis reef update <package>
```

Inspect compatible and incompatible versions without final writes:

```sh
chelis reef outdated [<package>]
chelis reef outdated [<package>] --json
```

Add `--offline` to either command for local candidates only.

A targeted refresh keeps unrelated locked packages fixed, including their hashes.
It changes a transitive package only when the selected target requires that change.

Reef verifies all selected remote pairs in temporary storage. Then Reef gets the project lock before the registry lock.

Reef publishes complete package directories and updates `index.json`. Reef replaces the project `reef.lock` last.

If lock replacement fails, the prior lock remains authoritative. Complete unused cache entries can remain for a later garbage-collection command.

## Document Schemas and Upgrades

Each `reef.toml` and `reef.lock` declares an independent top-level schema. New manifests use schema 3. New locks use schema 1.

Reef reads a document without a schema as legacy schema 0. It reports one warning that names the upgrade command.

Schema 1, schema 2, and schema 3 reject unknown keys in fixed tables. Reef selects the schema before complete document parsing.

Use one mode for each upgrade command:

```sh
# Report each required step without a write.
chelis reef upgrade --check

# Apply each required step in the package rooted at the current directory.
chelis reef upgrade --inplace

# Select a package and explicit supported targets.
chelis reef upgrade --inplace --path <dir> --manifest-to 3 --lock-to 1
```

The schema-1 to schema-2 step adds resolver 2. It rewrites each exact dependency from `X.Y.Z` to `=X.Y.Z`.

The schema-2 to schema-3 step enables typed package metadata. It does not invent metadata values.

Each migration preserves path-only dependencies, accepted comments, and key order. It also updates a known Taplo manifest directive.

The command examines both documents before a write. It replaces `reef.toml` first and `reef.lock` second.

If a process stops between replacements, run the command again. Both intermediate documents use readable schemas.

Project file writers serialize through `.reef-write.lock`. Reef adds its lock and sibling patterns to the package ignore file.

Versioned editor schemas are in `docs/schemas/reef/`. These schemas check syntax and table shape only.

Editor validation does not check paths, network sources, compiler compatibility, or artifact bytes. Reef remains the validation authority.

## Install Paths

`chelis reef install` accepts four sources, used independently:

```sh
# Discover packages under a chelis monorepo checkout (offline).
chelis reef install --from-monorepo <path>

# Fetch a single shell from the canonical hosting org's GitHub
# release. Authentication is required (the canonical repos are
# private); GITHUB_TOKEN must be set or `gh auth token` must work.
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

- `<name>-<version>.tar.zst`: the source archive
- `<name>-<version>.chb`: the prebuilt shell artifact

Note the convention: the **tag** is `v<version>` (with the leading
`v`), but the **asset filenames** are `<name>-<version>.{tar.zst,chb}`
without the leading `v`. Both forms are accepted by the
`<org>/<repo>@<tag>` parser (`@v<version>` and `@<version>` are equivalent).

### Reproducible builds

Unchanged package inputs produce byte-identical archive and CHB files across
repeated `chelis reef build` runs. Reef sorts archive members by their UTF-8
package-relative paths and normalizes regular-file headers to mode `0644`,
uid/gid `0`, and mtime `0`.

The source archive includes each declared README and license file. Reef does not discover undeclared metadata files.

Two declarations for one path produce one member. A declared metadata snapshot supplies bytes when a source root selects the same path.

Set `SOURCE_DATE_EPOCH` to a non-negative integer number of seconds when a
release requires a different canonical timestamp:

```sh
SOURCE_DATE_EPOCH=1700000000 chelis reef build
```

The same inputs and epoch produce the same bytes. A malformed
`SOURCE_DATE_EPOCH` stops the build instead of silently producing artifacts
under a different timestamp. The CHB embeds the SHA-256 of the canonical
source archive, so changing the epoch intentionally changes both artifact
identities.

CHB format 5 has an explicit `CHELCHB\0` magic and version envelope. Each
function symbol records any quantified type-variable domain restrictions by
the same alpha-canonical variable identity used in its printed type. The
machine-readable output of `chelis reef schema` reports `format_version: 3`
and the identical `type_variable_restrictions` ledger; for example, a shared
active-float precision appears as
`[{"variable":"t0","domain":"active_float"}]`. Older unversioned CHB bytes
and unknown versions are rejected explicitly.

CHB format 5 adds a `collection_obligations` ledger to each function symbol.
An already-checked `len`, `index`, `append`, `concat`, `key_from_seed`, `split_key`,
`split_keys`, or `fold_in` value retains its
operand/result relation, written in canonical Deep form under the printed
type's alpha-canonical variables. Hidden relation variables are rejected;
newly authored wrappers must state a sufficient public type contract. Two
exports that differ only in their checked relations have different identities.

## Auto-fetch During Build

`chelis reef build` uses exact locked origins when a valid lock needs missing bytes.

Without a valid resolver-2 lock, the command uses local candidates first. It starts bounded remote discovery only when no local graph completes.

Schema-1 exact dependencies keep their direct exact-tag fetch path. Pass `--no-auto-fetch` to disable network access during the build.

The result of all four items above is that a fresh dev environment's
onboarding is just:

```sh
git clone <project>
export GITHUB_TOKEN=$(gh auth token)
chelis reef build
```

Auto-fetch handles the rest. `GITHUB_TOKEN` is mandatory because the
canonical-org shell repositories are private; the install path
hard-fails with a clear error if no token is available.

## One-command provisioning: `chelis reef setup`

`chelis reef build` fetches source packages, but a freshly-cloned shell has
three more dependency classes to satisfy first: the pinned **toolchain**,
**binary artifacts**, and (for crate-linking shells) chelis **source crates**.
`chelis reef setup` is the orchestrator that brings all of them to their pins
in one verb:

```sh
chelis reef setup            # provisions the project rooted at .
chelis reef setup --path <dir>
```

It reads the `reef.toml` `compiler =` pin and runs, in order:

1. **toolchain**: if the pinned toolchain is not installed in the chelisup
   store, it auto-installs it by delegating to the `chelisup` binary
   (`chelisup install <ver>`). If `chelisup` itself is missing it stops with a
   loud, actionable error. This is *explicit, user-invoked* provisioning, so it
   is exempt from the shim's "never auto-install on `cd`" rule; the toolchain
   step deliberately subprocesses the real `chelisup` rather than installing
   in-process, so it never overwrites the shim with the compiler.
2. **source packages + binaries**: `reef install --from-lockfile`, when a
   `reef.lock` is present.
3. **source crates**: `reef src sync`, when the manifest declares a
   `[chelis-src]` section.
4. **summary**: prints the `reef doctor` report (below).

`reef setup` runs from a *current* chelis (it may itself install the pinned
one), so a clone-and-`setup` does the right thing without reaching for
`+<ver>`. Combined with the `chelisup` bootstrap, the full onboarding is two
commands: install `chelisup` once, then `chelis reef setup` per clone. See
[Install](install.md) for the chelisup side.

## Health: `chelis reef doctor`

`chelis reef doctor` is the read-only counterpart of `setup`. It never
installs.

```sh
chelis reef doctor              # the shell rooted at .
chelis reef doctor --root ~     # every shell one level under ~
```

It prints a machine-wide header (the chelis home, the shim, the recorded
default), then for each discovered shell reports every dependency class:

- **toolchain**: installed in the chelisup store
  (`~/.chelis/toolchains/<ver>`) or `MISSING` with the `chelisup install <ver>`
  fix;
- **source crates**: `ok`, drift with the `reef src sync` fix, or `n/a` for a
  pure-Chelis shell;
- **binary artifacts** (from `[artifacts]`): `ok` with the resolved path, or
  `MISSING` with the `reef install --from-lockfile` fix.

## Lockfile

Each new `reef.lock` starts with `schema = "1"`. Its schema evolves independently from the manifest schema.

`reef.lock` records every resolved dependency as a tuple of
`(name, version, source-kind, compiler-pin, archive_sha256,
shell_sha256)`. The `source-kind` discriminator (`LockSource`) has
four variants:

- `Path`: a path-source dependency (rare; mostly for local development)
- `LocalRegistry { remote_origin }`: installed from the local
  registry. The optional `remote_origin` records where the bytes were
  originally fetched from, in scheme-tagged URI form (currently
  `github://<org>/<repo>@<tag>`); lockfiles that omit the field
  round-trip cleanly without it.
- `Bundled { compiler_version }`: the chelis-std runtime, served
  from the compiler binary's embedded bundle. Recorded for
  auditability, so a lockfile reader can see which compiler version
  supplied the runtime bytes.
- `Binary { remote_origin, platform, asset, sha256 }`: a host-specific
  executable artifact; it has no `.chb` shell or source modules.

**Project-driven blanket synthesis:** every reef.toml's `compiler =`
pin is itself the runtime declaration. `build_lockfile` therefore
records a `LockSource::Bundled` chelis-std entry on every project,
regardless of whether the project listed chelis-std in
`[dependencies]`. Lockfiles produced by older compilers that recorded
chelis-std as `LocalRegistry` are auto-migrated to `Bundled` on read
and rewritten on the next `chelis reef build`.

If a compiler is rebuilt with different bundled runtime bytes but the same
version, an existing lock's runtime hash mismatch is a hard error. For a
resolver-2 manifest, explicitly run `chelis reef update --offline` to refresh
the bundled hashes. For a legacy or schema-1 manifest, first run
`chelis reef upgrade --inplace` or remove the old lock and then run
`chelis reef build`. `reef update` requires a schema-2-or-newer resolver-2
manifest.

### Prepared package cache

Commands that repeatedly consume the same locked package graph reuse a
prepared-graph cache under `$CHELIS_REEF_HOME/.cache/prepared-graphs/` (or the
normal XDG/Home cache fallback). Entries are tied to the compiler version and
canonical project root. The determinant inventories the path and exact bytes
of every `.ch` file under every declared source root, so source additions,
deletions, and renames invalidate it alongside content changes, manifests,
`reef.lock`, and published archive/shell identities. Graph construction is
bracketed by identical pre/post inventory snapshots; concurrent edits cause a
retry instead of storing declarations parsed from one version under another
version's hash. The cache file has a versioned, checksummed envelope; corrupt
or version-skewed entries emit a stderr diagnostic and are rebuilt rather than
trusted. Declared source-root symlinks may not escape their package root.

For `chelis prove`, the package graph is prepared once per invocation. The
post-verdict type check remains fail-closed for every declaration in the
selected module and for all transitively referenced dependency declarations;
unreachable declarations in an otherwise large installed shell are not
rechecked.

## Import Syntax

```chelis-surf-fragment
import Std.Init.Kaiming(..)
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

- `CHELIS_HOME`: the consolidated chelis home. Default is `~/.chelis`. It roots
  the chelisup toolchain store (`toolchains/<ver>`), the `chelis`/`chelisup`
  binaries (`bin/`), the reef registry (`reef/`), and the source-crate store
  (`src/`). `reef setup` and `reef doctor` resolve toolchains and binary
  artifacts under it.
- `CHELIS_TOOLCHAIN`: pin the toolchain the `chelis` shim resolves to, above
  the `reef.toml` pin and the recorded default (see
  [Managing versions](install.md#managing-versions)).
- `CHELIS_REEF_HOME`: local registry root. Default is
  `~/.chelis/reef`.
- `GITHUB_TOKEN`: required for any remote fetch
  (`--from-github`, `--bootstrap`, auto-fetch during build). Falls
  back to `gh auth token` if unset.
- `CHELIS_REEF_GITHUB_BASE_API`, `CHELIS_REEF_GITHUB_BASE`:
  test-fixture overrides for the GitHub API and download base URLs.
  Production use defaults to `https://api.github.com` and
  `https://github.com` respectively.

## Concurrent Builds

`chelis reef build` is safe to run concurrently across independent working trees.

Each package root uses `.reef-write.lock` for manifest, lock, and package archive replacement. The local registry uses its separate `.reef-lock`.

Two writers for one package serialize without deleting another writer's sibling. A failed single-file replacement restores the prior readable target.

## See Also

- `spec/design/reef_distribution.md`: full distribution design,
  including error categories and acceptance oracles.
- `spec/design/chelis_canonical_reference.md` §5: runtime-vs-shell
  distinction and the canonical shell roster.
- `spec/design/phase3j_pre_release.md`: release contract and
  downstream pinning policy.
