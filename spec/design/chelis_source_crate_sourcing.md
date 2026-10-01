# chelis source-crate sourcing for crate-linking shells

**Status:** Implemented (reef, 2026-06-29). Normative for any downstream
shell that links chelis compiler crates as Cargo path dependencies. This is
the monorepo home of the convention; per the Scaffolding Drift Rule
([`shell_repo_contract.md`](shell_repo_contract.md) §10) it lands here first
and each crate-linking shell then conforms (contract §2, conformance row 17).

> This doc supersedes the per-shell draft that previously lived in
> `hydronnx/docs/chelis_source_pinning.md`.

## 1. The dependency taxonomy

A shell consumes chelis through up to three artifact classes. Each must come
from a **pinned, version-keyed, immutable store**, resolved per the consuming
repo's `reef.toml` — never a single mutable shared working tree, never a
global default the last install repoints.

| Class | What | Store | Resolution | Status |
|---|---|---|---|---|
| (a) Toolchain binary | the `chelis` CLI | `~/.chelis/toolchains/<ver>/` | `chelisup` pin-resolving shim (chelis#164, shipped) | solved (contract §2) |
| (b) chelis-std | the std reef package | compiler-bundled, or `~/.chelis/reef/packages/.../<ver>/` | reef registry | solved |
| (c) Source crates | `chelis-ir` / `chelis-types` / … as Cargo path deps | `~/.chelis/src/<ver>/` | `../chelis` symlink → version-keyed worktree | **this doc** |

Classes (a) and (b) are location-independent and already coexist per-repo.
Class (c) is the gap this doc closes, and it applies **only** to a shell that
links chelis Rust crates as path dependencies.

## 2. Class (c): why a path dep is hard

A Cargo path dependency carries **no version constraint** — Cargo compiles
whatever working tree sits at the path. A crate-linking shell's `Cargo.toml`
reads (hydronnx shown):

```toml
chelis-ir    = { path = "../chelis/crates/chelis-ir" }
chelis-types = { path = "../chelis/crates/chelis-types" }
```

so the single `../chelis` sibling slot is forced to be simultaneously the
developer's mutable dev clone, the shell's pinned build input, and the slot
every co-located shell's `../chelis` resolves to. When several shells on
different pins share a parent directory (on the reference machine, hydronnx
`=0.8.0` and calcify `=0.7.21` both under `~`, both resolving `../chelis` to
one monorepo checkout), the crate-linking lane silently compiles against the
wrong chelis version, and `cargo build` rewrites the committed `Cargo.lock`
off the pin.

Constraints the fix must respect:

- The committed `Cargo.toml` **keeps** the relative `path = "../chelis/..."`
  (CI's sibling checkout relies on it). The fix is **local-dev-only** and must
  not edit `Cargo.toml` or CI.
- The source is the **full chelis crates tree at the pinned commit** —
  intra-repo path deps (`chelis-ir → chelis-types → chelis-pred`) mean a
  partial copy will not compile; it is a checkout / git worktree.
- The source comes from canonical `Chelis-Lang/chelis`, resolved **by
  commit** (the shell's pinned release commit), never a local tag (which can
  be stale). The repo is private → `gh` / token auth.

## 3. Design: one version-keyed store + a per-shell symlink

One store, shared across all shells (like the toolchain store); each shell's
`../chelis` slot points at the worktree matching *its own* pin:

```
~/.chelis/src/
  mirror.git/    bare mirror of Chelis-Lang/chelis (canonical refs/tags)
  0.8.0/    → git worktree @ <0.8.0 release commit>    (hydronnx)
  0.7.21/   → git worktree @ <0.7.21 release commit>   (calcify)

<shell>/../chelis → symlink → ~/.chelis/src/<that shell's pin>
~/chelis-dev/     the developer's mutable dev clone, OUT of every sibling slot
```

The store lives under the shared `~/.chelis/` home (alongside the reef
registry at `~/.chelis/reef/` and, once chelisup ships, the toolchains at
`~/.chelis/toolchains/`). Override with `$CHELIS_SRC_HOME`, or `$CHELIS_HOME`
for the whole home.

git worktrees share one object store, so each pin is a cheap checkout and a
pin bump is "add a `<ver>/` worktree, re-point the slot." Cargo resolves
`../chelis/crates/chelis-ir` through the symlink to pinned source with no
Cargo magic: committed manifest untouched, lockfile stays at the pin,
resolution identical to CI.

### 3.1 Why symlink, not a `.cargo/config.toml` override (empirical)

An alternative keeps the layout by overriding the crates with a gitignored
`<shell>/.cargo/config.toml` `paths = [...]` entry pointing at the store. A
probe (2026-06-29) characterized it against a version-mismatched base
(`../chelis` at 0.12.0, override to 0.8.0):

| Property | Result |
|---|---|
| Override honored across the version mismatch? | **Yes — the build resolves the override** (it is *not* silently ignored) |
| Recorded in `Cargo.lock`? | **No — the lock follows the base `../chelis`** |
| Works with the base slot absent? | **No** — the base path must exist and drives the lock |

So the override yields a *correct build* but leaves `Cargo.lock` tracking the
dev-clone version: if the base ≠ pin, the lockfile churns off the pin (the
"non-pin version in `git diff Cargo.lock`" canary stuck red locally, and
thrash against CI). The **symlink** keeps the base path *at* the pin, so build
and lockfile both stay pinned. The symlink is therefore the supported
mechanism; the `.cargo` override is a documented build-only escape hatch, not
recommended. Co-located crate-linking shells that cannot each own a `../chelis`
symlink relocate so each parent holds at most one (see §6).

## 4. Tooling: `chelis reef src` and `chelis reef doctor`

The store, resolution, wiring, and guard are implemented in `reef`
(`crates/chelis-reef/src/chelis_src.rs` + the CLI in `crates/chelis-cli`):

- **`chelis reef src sync [--path <shell>]`** — ensure the bare mirror
  (`git clone --mirror` / `fetch`, private-repo auth via per-invocation
  `http.extraheader`), resolve the pin commit (explicit `[chelis-src]
  pin_commit`, else `v<version>` tag → commit **from the mirror**), add the
  per-version worktree (idempotent), and point `../chelis` at it. An existing
  real directory in the slot (a dev clone) is **refused, not deleted**, with
  relocation guidance.
- **`chelis reef src check [--path <shell>]`** — the local drift guard:
  asserts the store worktree HEAD, the `../chelis` symlink, and `Cargo.lock`
  are all at the pin. Loud, actionable failure. Belongs in the shell's
  Workspace Gate, **local-only** (CI checks out the pinned sibling fresh, so
  it skips this).
- **`chelis reef src status [--path <shell>]`** — read-only report of the
  resolved pin, store worktree HEAD, and slot target.
- **`chelis reef doctor [--root <dir>]`** — machine-wide health across all
  three classes: for each shell under `<dir>`, the pinned toolchain's
  presence and (for crate-linking shells) the source-crate sync state, with
  the fix for each gap.

The `reef src` commands read the shell's manifest via `read_manifest_for_src`
— **parse-only**, skipping the same-compiler-version assertion — because they
run from whatever chelis is on PATH while operating on a shell pinned to a
*different* compiler. (`chelisup`, chelis#164, is the eventual
version-independent home.)

## 5. The `[chelis-src]` manifest section and pin surfaces

A crate-linking shell declares the class in `reef.toml`:

```toml
[chelis-src]
crates     = ["chelis-ir", "chelis-types"]   # the chelis crates it path-deps
pin_commit = "b741149b23db7c05849ebd8f5cccc5ce95ca626b"  # canonical release commit
```

`pin_commit` is the single source of truth `reef` reads; the workflow
`CHELIS_PIN_COMMIT` env mirrors it, and the shell's offline pin-consistency
guard enforces their agreement (alongside the `compiler`↔`CHELIS_TAG`/
`CHELIS_VERSION` checks). When `pin_commit` is omitted, `reef src` resolves
the `v<compiler>` tag to a commit from canonical. There is deliberately **no
`reef.lock` entry / `LockSource` variant** for source crates: they are Cargo
crates, not reef packages (`reef.lock`'s `LockedDependency` carries
archive/shell SHAs they do not have, and `build_lockfile` walks only the reef
package graph).

## 6. Migration (guide, don't auto-perform)

On a machine where `../chelis` is currently a real dev clone (and where
co-located crate-linking shells share a parent), adopting this design is a
one-time, **guided** migration — `reef src` instructs, it does not move the
developer's trees:

1. Move the dev clone out of every sibling slot (e.g. to `~/chelis-dev`). A
   clone with attached worktrees needs `git worktree repair` after the move.
2. Arrange crate-linking shells so each parent directory holds at most one
   (relocate as needed), so each can own its `../chelis` symlink.
3. `chelis reef src sync` in each crate-linking shell to provision the store
   worktree and the symlink.

## 7. Verification (per crate-linking shell)

- `chelis reef src sync` then `cargo build` → chelis crates at the pin in
  `Cargo.lock`, no churn.
- Re-run the crate-linking gate (hydronnx: the `translator_parity` fixture
  parity test) → version-drift artifacts disappear; cited divergences stay
  honest.
- `chelis reef src check` fails loudly when the slot is repointed off the pin
  or `Cargo.lock` drifts.
- `chelis reef doctor --root <home>` reports each shell's per-class state.

## 8. Applicability

| Shell | (a) toolchain | (b) std | (c) source crates |
|---|---|---|---|
| hydronnx | launcher | bundled in toolchain | **yes — `chelis-ir`, `chelis-types`** |
| calcify | launcher | reef registry | **yes — `chelis-types`** |
| school / coral / nautilus / other pure-Chelis | launcher | bundled or installed | none |

Most shells are pure-Chelis and never trigger class (c).
