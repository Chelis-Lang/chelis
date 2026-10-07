---
name: packaging-install
description: Use when changing or validating Chelis toolchain install, version routing, or the reef orchestration surface — chelisup (install/default/shim), the chelisup.sh bootstrap and its private-repo `gh` auth, chelis reef setup, and reef doctor. Covers the shim resolution order, the store layout, the shim-corruption trap, and the offline test seams.
---

# Packaging & Install Orchestration

Use this skill for any change or validation pass touching how Chelis is
installed, how a toolchain version is selected, or how a clone is provisioned.

## Surface

Three layers (design: `spec/design/chelis_packaging_and_install.md`):

- **Layer 0 — `chelisup`** (`crates/chelisup`): the toolchain installer and the
  pin-resolving `chelis` **shim**. One binary, dual role by `argv[0]`: invoked
  as `chelis` it resolves + execs a toolchain; invoked as `chelisup` it runs
  the installer CLI (`install`, `default`, `show`, `list-installed`, `which`,
  `uninstall`, `update`, `self {uninstall,update}`). Bootstrap:
  `crates/chelisup/bootstrap/chelisup.sh` (the one permitted shell script).
- **Layer 1 — reef** (`crates/chelis-reef`): resolve/verify/place source
  packages, chelis-std, source crates (`reef src`), and binary artifacts
  (`[artifacts]` / `LockSource::Binary`).
- **Layer 2 — orchestration** (`crates/chelis-cli`): `chelis reef setup` and the
  unified `chelis reef doctor`.

**Bootstrap & private-repo auth.** `chelisup.sh` drops the `chelisup` prebuilt on
a bare machine; `chelisup install <ver>` then fetches toolchains. Public releases
work without a GitHub token. The script uses an authenticated `gh` when
available and otherwise fetches the public release URL with `curl`. `chelisup`
resolves `GITHUB_TOKEN` first, then `gh auth token`; without either it sends
anonymous GitHub REST requests. Private releases require a token with read
access. Docs to keep in sync: `docs/book/src/install.md` §1, `README.md`, and the
design doc §5.5.

## Store Layout

`$CHELIS_HOME` (default `~/.chelis`), resolved identically by
`chelisup::paths::resolve_home` and `chelis_reef::chelis_home`:

```
~/.chelis/
  bin/{chelis, chelisup}     the shim + the installer
  toolchains/<ver>/          side-by-side toolchains (<ver>/bin/chelis is real)
  reef/                       the reef registry
  src/                        the source-crate store
  default                     the recorded default version
```

In tests, the single `CHELIS_HOME` seam isolates the toolchain store, the
shim/default, and the binary-artifact dir at once.

## Nix `chelisup` Closure

The Nix package uses `bin/chelisup` as a wrapper around `libexec/chelisup`.
Before `install`, the wrapper creates `$CHELIS_HOME/nix-gcroots/chelisup.next`.
After success, it promotes `$CHELIS_HOME/nix-gcroots/chelisup` and removes the
staging root. A failed install preserves the prior root. If the install copied
a new binary, the wrapper promotes `$CHELIS_HOME/nix-gcroots/chelisup.partial`.
A new attempt recovers a stale staging root before it changes that root.
After success, the Nix wrapper restores itself at `$CHELIS_HOME/bin/chelisup`.
The generic Rust installer contains no Nix root path or cleanup logic. The
installed Nix wrapper removes all three roots after the real `self uninstall`
command succeeds.

The wrapper, package inventory, self-uninstall behavior, Nix contract test,
and native Nix workflows form one contract. A direct Nix-built executable
copy is invalid because external copies do not become Nix GC roots.

## The Embedded chelis-std Runtime

Each binary embeds the chelis-std archive and shell. `crates/chelis-std-bundle`'s
build script stages `packages/chelis-std` (its `reef.toml`, the `.ch` files
under its source roots, and its declared metadata files) and packs it with
`chelis_reef::pack_runtime_package` at archive mtime 0 whatever
`SOURCE_DATE_EPOCH` says. `chelis reef build` runs the same packing step but
archives every file under `src/` and honours `SOURCE_DATE_EPOCH`, so it writes
the embedded pair only from a copy holding just those inputs, with the epoch
unset. Editing a std `.ch` file and rebuilding is the whole workflow; nothing
generated is committed.

- `chelis-reef` never depends on the bundle (the bundle build-depends on reef).
  Every reef and `chelis-compiler-api` graph entry point takes the runtime as
  `&'static EmbeddedRuntime`; binaries pass
  `&chelis_std_bundle::EMBEDDED_RUNTIME`. Only binaries and test harnesses
  depend on the bundle, so do not add a global provider or a library edge to it.
- Never commit `packages/chelis-std/dist/`, `crates/chelis-std-bundle/dist/`, or
  a `reef.lock` recording the bundled runtime: every lock names the running
  binary's runtime hashes. `scripts/check_std_bundle_untracked.py` refuses them.
- `bundled_chelis_std_loader`'s fixed-point test requires `chelis reef build` of
  such a copy to reproduce the embedded pair and its locks to name those
  bytes; `scripts/check_std_bundle_reproducible.py` requires two builds to agree.

## Shim Resolution Order (first match wins)

1. leading `+<ver>` arg (`chelis +0.13.0 build main.ch`)
2. `CHELIS_TOOLCHAIN` env
3. nearest `chelis-toolchain` file walking up (ranks above the package pin)
4. nearest `reef.toml` `compiler =` pin walking up
5. recorded default (`chelisup default <ver>`)

Invariants to preserve:

- **No auto-install, no silent fallback in the shim.** A resolved-but-not-
  installed version is a loud error naming `chelisup install <ver>`. The shim
  never downloads on `cd` and never routes to a different installed version.
- **Concrete pins only.** `+latest` is not supported (ambiguous); pins are
  exact `X.Y.Z`. `is_safe_path_component` gates any version joined into a path.

## `reef setup` And The Shim-Corruption Trap

`chelis reef setup [--path]` (`cmd_reef_setup`) runs, in order: ensure the
pinned toolchain → `reef install --from-lockfile` (if `reef.lock`) → `reef src
sync` (if `[chelis-src]`) → `reef doctor` summary.

- **The toolchain step auto-installs by subprocessing the real `chelisup`
  binary.** This is *explicit, user-invoked* provisioning, exempt from the
  shim's "no auto-install" rule (which governs only the implicit per-invocation
  shim). It is not exempt from good taste: it skips the install when the pin is
  already present, and errors loudly when `chelisup` itself is absent.
- **NEVER call `chelisup::install::install(...)` in-process from `chelis-cli`.**
  That helper calls `ensure_shim_installed`, which copies `current_exe()` into
  `<home>/bin/{chelis,chelisup}`. From the `chelis` compiler binary that would
  overwrite the shim and the installer with the compiler.
- **The guard is compile-time.** chelisup's `install`, `ensure_shim_installed`,
  `uninstall`, and `self_uninstall` are `pub(crate)` (only chelisup's own
  `cli.rs` calls them; the only external uses are the pure asset-naming helpers
  `detect_slug`, `host_is_musl`, `release_build` and `asset_name`, from
  `reef_setup.rs`, and `nss::use_builtin_services`, which `chelis`'s `main`
  calls first).
  Any in-process reference from another crate is an `E0603` build error caught
  by the normal clippy/build/test stages. Do NOT widen that visibility to
  `pub`; keep the call-site comment and the test asserting the shim stays
  chelisup after an auto-install.

## `reef doctor`

Read-only, never installs. Prints a machine header (chelis home, shim,
default), then per shell: toolchain (chelisup store, `chelisup install` fix),
source-crate drift (`reef src sync` fix), and `[artifacts]` binaries
(`reef install --from-lockfile` fix). The toolchain check reads the chelisup
store, **not** the retired `~/.local/share/chelis` / `install_chelis_toolchain.py`.

## §5.4 Unknown-Subcommand Hint

`main()` uses `Cli::try_parse()`; only `clap::error::ErrorKind::InvalidSubcommand`
is augmented (chelis version + pin source via `chelisup::resolve` + `+<ver>`
guidance). Help, version, unknown flags, and extra positionals defer to clap
byte-for-byte. When adding a subcommand-bearing node, keep it positional-free
so a bogus token stays `InvalidSubcommand` and the hint still fires.

## Validation Rules (positive + negative parity)

- `reef setup`: toolchain present → proceeds; missing + chelisup available →
  auto-installs and the shim stays chelisup; missing + chelisup absent → loud
  error naming `chelisup install <ver>`, never `install_chelis_toolchain.py`;
  bad/missing `reef.toml` → clear error.
- `reef doctor`: toolchain installed → ok; absent → MISSING + `chelisup install`;
  artifact present/absent → ok/MISSING.
- hint: unknown subcommand → hint fires; `--help`/`--version`/unknown-flag → no
  hint, clap-native exit codes.
- chelisup: exercise the resolution table (each of the five levels) and the
  not-installed loud error.

## Offline Test Seams

- `CHELIS_HOME` — isolate the whole store to a tempdir.
- `CHELISUP_RELEASE_BASE` — read the toolchain tarball
  (`chelis-vX.Y.Z-<build>.tar.gz`, gzip; name it with
  `chelisup::install::asset_name` and `release_build`, which on Linux names the
  preferred build: `linux-x86_64-musl` on a musl host (`host_is_musl`: `/bin/sh`'s
  program interpreter is an `ld-musl-*` loader), else `linux-x86_64-static`;
  `install` falls back to `linux-x86_64-static`, then `linux-x86_64-glibc2.31`,
  for a release without the preferred one) from a local dir instead of GitHub.
- `CHELISUP_GITHUB_BASE_API`, `CHELISUP_REPO` — wiremock the GitHub REST path.
- `CHELISUP_BIN` — point `reef setup` at a specific `chelisup` binary.
- `CHELIS_REEF_GITHUB_BASE_API`, `CHELIS_SRC_REMOTE`, `CHELIS_SRC_HOME` — the
  reef-side seams reused by setup's install/src steps.

Mirror `crates/chelis-cli/tests/reef_setup.rs`,
`reef_doctor_unified.rs`, and `unknown_subcommand_hint.rs`.

## Documentation Sync

Behavior changes here must update, in the same change set:
`spec/design/chelis_packaging_and_install.md`, `docs/book/src/install.md`,
`docs/book/src/reef.md`, and the `chelis reef` rustdoc / help text. Keep the
`AGENTS.md` (= `CLAUDE.md`) "Toolchain and packaging" pointer
honest.

## Authoritative References

- Design: `spec/design/chelis_packaging_and_install.md`
- User guide: `docs/book/src/install.md`, `docs/book/src/reef.md`
- Shim + store: `crates/chelisup/src/{resolve,paths,install,shim,cli}.rs`
- Orchestrator + doctor + hint: `crates/chelis-cli/src/main.rs`
  (`cmd_reef_setup`, `ensure_pinned_toolchain`, `cmd_reef_doctor`,
  `handle_parse_error`, `eprint_pin_hint`)
- Binary artifacts: `crates/chelis-reef/src/lib.rs` (`LockSource::Binary`,
  `install_binary_artifact`, `which_artifact`)
