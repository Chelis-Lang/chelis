---
name: packaging-install
description: Use when changing or validating Chelis toolchain install, version routing, or the reef orchestration surface — chelisup (install/default/shim), the chelisup.sh bootstrap, chelis reef setup, and reef doctor. Covers the shim resolution order, the store layout, the shim-corruption trap, and the offline test seams.
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
  `cli.rs` calls them; the sole external use is the pure `detect_slug` helper).
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
  (`chelis-vX.Y.Z-<slug>.tar.gz`, gzip) from a local dir instead of GitHub.
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
`AGENTS.md` (= `CLAUDE.md`) "Toolchain And Packaging Orchestration" section
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
