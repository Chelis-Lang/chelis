# Chelis packaging & install: the unified end-state

**Status:** North-star design + roadmap (scoped 2026-06-30). The toolchain
launcher behavior in §3 is contract-normative today
([`shell_repo_contract.md`](shell_repo_contract.md) §2); the source-crate layer
is implemented (chelis#571). Binary distribution (chelis#468, WS-A) and the
first-party installer `chelisup` (chelis#164, WS-B) are **implemented**, and the
orchestration layer (WS-C) — `chelis reef setup`, the `reef doctor`
unification, and §5.4's unknown-subcommand hint — is **implemented** and
described in §7 / §5.4. This doc is the umbrella the rest of the
install/packaging work hangs off; it does not supersede
[`reef_distribution.md`](reef_distribution.md) (reef's package-delivery design)
but frames it as one layer of a larger whole.

## 1. Why this doc

reef resolves and installs source packages well, and chelis#571 added
version-keyed sourcing for the compiler's own Rust crates. But "use chelis"
still does not, by itself, deliver everything a consumer runs:

- the **toolchain binary** is installed by a per-shell Python script and a
  hand-rolled PATH launcher, duplicated across every shell repo (chelis#164);
- **binary artifacts** (translators, shell binaries, the toolchain tarball)
  are distributed out of band through GitHub releases with hand-transcribed
  SHA verification, because reef cannot carry a binary (chelis#468).

Each gap has its own issue. This doc places them in one model so they compose
into a coherent **install tool + package manager** rather than three
disconnected efforts.

## 2. The three layers

```
Layer 0  BOOTSTRAP + ROUTING       chelisup (chelis#164)   — the only non-reef layer
         Get the first `chelis` onto a bare machine, and route which installed
         version answers `chelis` per repo. reef cannot do this: you need a
         `chelis` on PATH before any `chelis reef ...` runs (chicken-and-egg).
                              │  (once a chelis exists)
                              ▼
Layer 1  RESOLUTION + DELIVERY      reef                    — the package manager proper
         Resolve / verify / lock / place every artifact class:
           • source packages   reef_distribution.md Items 6-9   (shipped)
           • chelis-std         compiler-bundled                 (shipped)
           • source crates      `reef src`  (chelis#571)         (shipped)
           • binary artifacts   `reef install` Item 11 (#468)    (planned)
         plus reef.lock + `--from-lockfile` reproduction and `reef doctor`.
                              │
                              ▼
Layer 2  ORCHESTRATION              `chelis reef setup` + unified `reef doctor`
         One verb brings a freshly-cloned shell to a fully-pinned, buildable
         state across all classes; doctor reports the whole machine.
```

### Ownership

| Concern | Owner | Lockfile | Status |
|---|---|---|---|
| Cold-start first `chelis` | chelisup | — | #164 |
| Per-repo toolchain routing (shim) | chelisup | — | #164 |
| Source reef packages (shells/libs) | `reef install` | `reef.lock` | shipped |
| chelis-std | compiler bundle | `reef.lock` (Bundled) | shipped |
| Source crates (Cargo path deps) | `reef src` | **Cargo.lock** + `[chelis-src]` pin | #571 |
| Binary artifacts (translators, toolchain) | `reef install` | `reef.lock` (Binary) | #468 |
| One-command reproduce + health | `reef setup` / `doctor` | composes all | this doc |

## 3. The lockfile-ownership rule

`reef.lock` records **exactly the artifacts reef itself fetches and
content-verifies by hash.** Membership is decided by three questions: *who
fetches it, what is its integrity primitive, and who else already records it.*

- **Source packages and binaries belong in `reef.lock`.** reef fetches them;
  integrity is a content SHA-256 (the `archive_sha256` the lock schema is built
  around); nothing else records them.
- **Source crates do not.** Cargo fetches them; their integrity primitive is a
  *git commit*, not an archive hash, so the lock schema has no slot for them;
  and `Cargo.lock` already records them authoritatively. The pin lives in
  `reef.toml [chelis-src]` plus the workflow `CHELIS_PIN_COMMIT` surface. A
  `reef.lock` entry would be a redundant, schema-mismatched second copy.

"Not Chelis source code" is the wrong axis; *who fetches it and what verifies
it* is the right one. That is why chelis#571 deliberately stayed out of the
lock and chelis#468 must go in. See
[`chelis_source_crate_sourcing.md`](chelis_source_crate_sourcing.md) §5 (lands
with chelis#571) for the source-crate side.

## 4. Store consolidation

Chelis state is currently scattered and inconsistent:

| State | Current path | Set by |
|---|---|---|
| toolchains | `~/.local/share/chelis/<ver>/` | `install_chelis_toolchain.py` |
| launcher | `~/.local/bin/chelis` | `install_chelis_toolchain.py` |
| reef registry | `~/.chelis/reef/` | `registry_root()` |
| source-crate store | `~/.local/share/chelis-src/` | chelis#571 `default_store_root()` |

**Target:** one rustup-style home `~/.chelis/` (override `$CHELIS_HOME`):

```
~/.chelis/
  bin/{chelis, chelisup}     the shim + the installer
  toolchains/<ver>/          side-by-side toolchains
  reef/                       the reef registry (already here)
  src/                        the source-crate store (moved from ~/.local/share/chelis-src)
```

This matches where the reef registry already lives and the paths the chelisup
proposal assumes. It is cheap to adopt now: chelis#571's store is empty
(nothing has been migrated), and chelisup owns the one-time toolchain-store
migration regardless. The existing `$CHELIS_SRC_HOME` / `$CHELIS_REEF_HOME`
overrides remain for back-compat. **Decision:** the `default_store_root` change
should precede any shell adopting `reef src` (a small follow-up to chelis#571,
or folded into the chelisup work).

## 5. Layer 0 — `chelisup` (chelis#164)

A first-party single-binary, rustup-style toolchain installer. It is the
cold-start and routing layer reef structurally cannot be. New monorepo member
`crates/chelisup` (a small Rust binary).

### 5.1 Commands

- `chelisup install <ver>` — download the host-platform release tarball from
  `Chelis-Lang/chelis/releases/v<ver>` into `~/.chelis/toolchains/<ver>/`.
- `chelisup default <ver>` — set the recorded default the shim falls back to.
- `chelisup show` / `list-installed` / `which` — status.
- `chelisup update` — self-update the installer.
- `chelisup uninstall <ver>` / `chelisup self uninstall`.

### 5.2 The shim — resolution order, reasoned for chelis

The shim is a tiny `~/.chelis/bin/chelis` wrapper that resolves the active
toolchain **at each invocation** and re-execs it. The legacy
`install_chelis_toolchain.py` order (env → reef pin → default) is the right
kernel, but each level is justified and improved on here rather than inherited
wholesale. Precedence, first match wins:

1. **`+<ver>` argument** (e.g. `chelis +0.13 reef src sync`) — new versus the
   legacy launcher. A rustup-style explicit per-invocation override; highest
   because it is the most explicit, and the ergonomic answer to the
   cross-version caveat in §5.4.
2. **`CHELIS_TOOLCHAIN` env** — explicit, scriptable / CI override.
3. **`chelis-toolchain` file** (walking up from cwd) — a *deliberate* directory
   override, ranked **above** the package pin, and the way to pin a directory
   that is not a reef package at all.
4. **nearest `reef.toml` `compiler` pin** (walking up) — the package's declared
   toolchain. Honoring this is **correct and chelis-native, not blind
   inheritance**: chelis is exact-pin by design (this contract treats a stale
   pin as drift, §2), so for the compiler verbs "the version you run" and "the
   version the package is pinned to" are *meant* to coincide — and `reef.toml`
   already encodes it, so a clone-and-build works with **zero extra files**.
   That is a genuine advantage over rustup's separate `rust-toolchain.toml`.
5. **recorded default** (`chelisup default`) — used outside any package.

### 5.3 Invariants

**No auto-install, no silent fallback.** A resolved-but-not-installed version is
a loud error naming `chelisup install <ver>`, never a silent fall-through to
another version. This preserves the existing toolchain-store invariant (§2) and
deliberately avoids rustup's surprise auto-download on `cd`.

### 5.4 Cross-version management verbs — DECIDED

Auto-routing to the `reef.toml` pin is right for *compiler* verbs
(`build`/`check`/`test`/`eval`) but creates one friction for *cross-version
management* verbs (`reef src {sync,check,status}`, `reef doctor`, future `reef
setup`): run inside a `=0.8.0`-pinned shell, `chelis reef src sync` routes to
0.8.0, which has no `reef src` at all — the exact case chelis#571 handles today
via a parse-only manifest read plus an env override.

**Decision (chelis#574 review, @jeffreyksmithjr):** these verbs **stay in
chelis** — chelis is the primary tool and its surface is not split; chelisup
stays minimal (install / route-shim / default only). The friction is handled at
the UX layer, not by moving verbs, on two mechanics:

1. **A pinned-toolchain hint on unrecognized subcommands.** When the shim routes
   to a chelis that lacks the requested verb, chelis's unknown-subcommand error
   names what it knows and points at the override — e.g. *"unrecognized
   subcommand `src`. You are running chelis 0.8.0, pinned by ./reef.toml. If
   `src` is a newer command, run it with a version that has it —
   `chelis +<ver> reef src …` — or check `chelis --version`."* The hint is
   **generic, not version-specific about the introducing version**, and that is a
   deliberate limit: the version that emits the error is the one the shim routed
   *to* (the old pin), and a version only errors on verbs *newer than itself* —
   exactly the set it cannot know about. Naming *"`src` was added in 0.13"* would
   need a verb→version table updated independently of the routed toolchain, i.e.
   inside the shim, which would couple chelisup to chelis's command surface and
   break "keep chelisup small." chelis names its own version + the pin source
   (both knowable) and stops there.

2. **Concrete `+<ver>` (or `CHELIS_TOOLCHAIN=<ver>`), never `+latest`.** chelis is
   exact-pin by design (a floating pin is drift, §2; no semver ranges), so the
   cross-pin override is a *concrete* version too: `chelis +0.13 reef src sync`.
   `+latest` is a floating, ambiguous reference (newest *installed* vs newest
   *available*) and is deliberately **not** a supported pattern; `chelisup
   list-installed` tells you what you have.

This **closes the prior open question** (move these verbs to chelisup? — no).
Action items: a small chelis-cli change implements the unknown-subcommand hint
(tracked with WS-C, where §5.4 is finalized); WS-B (chelisup) keeps `+<ver>`
concrete-only and verifies its edge cases — a not-installed `+<ver>` already hits
the loud `chelisup install` error; nested `chelis` re-invocations under a
`+<ver>` route must not silently flip back to the pin.

This shim **supersedes the per-shell launcher**: §2 of the contract already
states shells satisfy the toolchain-resolution behavior via chelisup once it
ships, and retire their vendored `install_chelis_toolchain.py`.

### 5.5 Bootstrap

`curl -fsSL <host>/chelisup.sh | sh` fetches a prebuilt `chelisup` for the host
platform, drops it at `~/.chelis/bin/chelisup`, and prompts the user to add
`~/.chelis/bin` to PATH. The chelis releases page hosts the `chelisup`
prebuilts alongside the existing toolchain tarballs. **Private-repo caveat:**
until chelis releases are public the bootstrap needs a PAT or an authenticated
`gh` (the fallback the current School install script already uses).

### 5.6 Relationship to binary distribution (chelis#468)

chelis#468 defines the verb "fetch a per-platform release binary and SHA-verify
it fail-closed." chelisup needs the *same* verb for the toolchain at bootstrap,
but **cannot depend on reef being present**, so it carries a minimal standalone
copy. The overlap is resolved by ownership, not by sharing code across the
bootstrap boundary:

- **chelisup owns the toolchain lifecycle** — bootstrap, routing, upgrades,
  uninstall.
- **chelis#468 binaries are non-toolchain** — translators (octant), shell
  binaries, etc.

The toolchain may additionally be *described* as a `[artifacts]` entry for
reproducibility, but its installer of record is chelisup, not `reef install`.
This keeps two installers from fighting over the toolchain.

## 6. Layer 1 — binary distribution (chelis#468)

Specified in full as Item 11 of [`reef_distribution.md`](reef_distribution.md).
In brief: a `[artifacts]` manifest section carrying per-platform, SHA-pinned
release-asset entries; `reef install` resolves the host platform, downloads via
the existing GitHub REST path, SHA-256-verifies fail-closed, places a runnable
binary at `~/.chelis/bin/<name>`, and records it in `reef.lock` via a new
`LockSource::Binary` variant so it is as reproducible as a source package. It
reuses `install_validated_artifact_pair`, the shipped SHA-256 helpers,
`install_from_github`, and `try_github_token` (added in chelis#571).

## 7. Layer 2 — orchestration: `chelis reef setup`

**Implemented (WS-C).** `chelis reef setup [--path <PATH>]` brings a
freshly-cloned shell to its pins across every class in one verb:

1. **ensure the pinned toolchain** — check the chelisup store; when the pin is
   missing, **auto-install** by delegating to the `chelisup` binary
   (`chelisup install <ver>`), or emit a loud, actionable error naming that
   command if `chelisup` itself is absent. This is explicit, user-invoked
   provisioning, so it does *not* violate the §5.3 *shim* invariant (which
   forbids the implicit per-invocation auto-install). It subprocesses the real
   `chelisup` rather than calling its install path in-process: that path copies
   `current_exe()` into `<home>/bin/{chelis,chelisup}`, which from the `chelis`
   binary would overwrite the shim with the compiler;
2. **`reef install --from-lockfile`** — source packages + binaries (#468) from
   `reef.lock`, when one is present;
3. **`reef src sync`** — source crates (chelis#571) when `[chelis-src]` is
   present;
4. print the **`reef doctor`** summary.

This is the **one-command reproduce** that unifies the three reproduction
records (`reef.lock` + `Cargo.lock` + the `[chelis-src]` pin) behind a single
command. `reef doctor` — already multi-class from chelis#571 — is extended to
also report binary-artifact deps (#468) and the active toolchain/shim
(chelisup), as the read-only health counterpart of `setup`.

`reef setup` is the canonical *current-chelis* entry point for the cross-version
case §5.4 decided: it runs from a current chelis (it may itself install the
pinned toolchain), so a clone-and-`setup` does the right thing without the user
reaching for `+<ver>`. WS-C also lands §5.4's unknown-subcommand hint in chelis.

## 8. Roadmap & sequencing

```
WS-0  finish chelis#571 Phase 5 (shell adoption + migration)   ··· parked / optional
WS-A  chelis#468  binary distribution (reef Item 11)         ──┐
WS-B  chelis#164  chelisup (bootstrap + shim)                ──┤ parallel
                                                               ▼
WS-C  reef setup orchestrator + doctor unification          ◄── needs WS-A + WS-B
```

WS-A and WS-B are independent — WS-A extends `install_from_github` inside
`crates/chelis-reef`; WS-B is a new standalone `crates/chelisup` — and run
concurrently. WS-C composes both, so it lands last. This doc plus the
store-consolidation default-path change land first (cheap, and they unblock
both workstreams). WS-0 (chelis#571 downstream adoption) gates nothing here.

**Status:** WS-A, WS-B, and WS-C are implemented (`chelis reef setup`, the
unified `reef doctor`, and the §5.4 unknown-subcommand hint all ship in
`crates/chelis-cli`). WS-0 — downstream shells adopting `chelisup` and retiring
their vendored `install_chelis_toolchain.py` — remains parked/optional.

## 9. End-state acceptance

The north star is the "clone a shell on a bare machine, run two commands,
build" walkthrough:

1. `curl … | sh` → `chelisup` is present (WS-B).
2. `git clone <shell> && cd <shell> && chelis reef setup` →
   - the toolchain at the `reef.toml` pin is on PATH via the shim (WS-B),
   - source packages + binaries are installed and SHA-verified from `reef.lock`
     (WS-A),
   - source crates are synced and `../chelis` is wired (chelis#571),
   - `cargo build` / `chelis test` are green with everything at its pin, no
     lockfile churn.
3. `chelis reef doctor --root ~` reports every shell green across all classes.
4. octant's `consuming.md` workaround and C Note's verify-in-Dockerfile step
   retire (WS-A); shells drop their vendored `install_chelis_toolchain.py`
   (WS-B).

## 10. Cross-references

- [`reef_distribution.md`](reef_distribution.md) — reef package delivery (Items
  6-9 shipped; Item 11 is binary distribution, WS-A).
- [`chelis_source_crate_sourcing.md`](chelis_source_crate_sourcing.md) — the
  source-crate layer (chelis#571).
- [`shell_repo_contract.md`](shell_repo_contract.md) §2 — the toolchain/pin
  hygiene and source-crate MUSTs that this design implements.
- `spec/12-roadmap.md` Phase A — distribution unblock.
- `chelis_trust_stack.md` / `effect_taxonomy_expansion.md` Item 4 — artifact
  signing and install-time effect manifests: adjacent future work that composes
  at the same install boundary (cross-reference only; out of scope here).

## 11. Out of scope

Inherited from `reef_distribution.md` §Out of scope: semver / version-range
resolution, a public registry server (Item 10), cryptographic artifact signing
(trust-stack Item 5), and bit-reproducible cross-machine artifact comparison.
Additionally: garbage-collection / uninstall of the three stores — all of
`~/.chelis/{toolchains,reef,src}` currently grow unbounded — is deferred to a
later hygiene workstream.
