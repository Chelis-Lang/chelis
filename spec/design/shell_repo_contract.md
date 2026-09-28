# Downstream Shell Repo Contract

**Status:** NORMATIVE for every shell repo in the
[`chelis_canonical_reference.md`](chelis_canonical_reference.md) §Shell
Ecosystem table (`nautilus`, `coral`, `shoals`, `octant`, `school`, `darwin`,
`hull`, `hydrostatic`, `beacon`, and any future shell). Made binding by `AGENTS.md`
§Pointers, Downstream shells. Changes to this contract land in the monorepo
first and propagate to every shell per §10.

**Key words** MUST / SHOULD / MAY are RFC-2119. Conditional requirements
state their trigger; once triggered they are MUST.

**Reference implementation:** [`Chelis-Lang/school`](https://github.com/Chelis-Lang/school)
satisfies every unconditional MUST as of 2026-06-10 and is the stamping
source for new shells (§11). Each requirement links its School exemplar.

**Why this exists.** The 2026-06 School↔chelis alignment review
(School `spec/design/generality_audit.md`) found a repeatable downstream
failure mode with three causes: (1) shell authors could not see what
chelis/chelis-std actually provide, so they designed around gaps that did
not exist or were already scheduled to close; (2) acceptance criteria shaped
as checklists with single goldens rewarded fixture-shaped implementations;
(3) "blocked on upstream" claims drifted from upstream reality — workarounds
outlived their justifications, prose-name citations dodged every audit, and
in one case a feature shipped monomorphic four days after its blocker was
fixed *in the very pin it was built on* (chelis#293). Every requirement
below is a countermeasure to an observed instance of one of those causes,
not a hypothetical.

---

## 1. Identity & inheritance (MUST)

- The shell has a repo-local `AGENTS.md`; `CLAUDE.md` is a **symlink** to it
  so Claude-style and Codex-style entry points cannot drift.
- `AGENTS.md` carries the pinned toolchain's complete root Chelis `AGENTS.md`
  inside the `agents-inheritance` managed block: a generated,
  HTML-comment-fenced region stamped `chelis@X.Y.Z (sha256:…)` and regenerated
  by `chelis reef conform sync`. The root document is the authored source; an
  embedded byte copy ships with the toolchain and a freshness guard prevents
  those copies from drifting. `chelis reef conform audit` fails when the
  block's stamp falls behind the reef pin or its body is hand-edited. Machine-local environment sections of the
  monorepo `AGENTS.md` (the HIP workstation runbook, the macOS first-exec
  notes, workstation-specific measurements) bind only where the named
  environment actually exists. Everything **outside** the managed fences —
  Repo Identity, intent, shell-specific sections — is the shell's own and is
  never touched by `sync`. Full inheritance is the default, not a requirement
  that every shell keep every upstream section. Each shell SHOULD judge which
  portions apply to its work at initialization and pin bumps, exclude irrelevant
  portions, and retain its own guidance when that guidance remains relevant and
  current. A shell may omit any inherited section by placing one
  `shell-local:exclude` selector span outside the managed block. Its exact ATX
  heading selectors use the same structural matching, range, validation, and
  restoration rules as shared-skill selectors in §8. Selecting the root
  `# Chelis Agent Contract` heading omits the complete inherited body; the
  shell's own text outside the block remains. Selector control markers are
  standalone Markdown comments; sync and audit reject markers inside a code
  fence or an enclosing HTML block rather than interpreting quoted examples as
  configuration.
- `AGENTS.md` contains at minimum these sections: **Repo Identity**,
  **Toolchain Policy**, **Pin Bump Checklist** (§7), an **Upstream Bugs**
  pointer (§4), and the **Scaffolding Drift Rule** (§10). The
  inheritance/checklist/drift sections MAY be carried as managed blocks so
  `conform sync` keeps them current; Repo Identity is always hand-authored.
- **Toolchain-mechanized conformance.** This contract ships *in the pinned
  toolchain* as `chelis reef conform` (audit / init / sync / bump /
  bump-check), the machine-readable form of §11. A new shell is stamped with
  `chelis reef conform init` (not by copying School), audited by
  `chelis reef conform audit` in CI and the local gate, and kept current with
  `chelis reef conform sync`. Because the tool is version-bound to the pin, it
  cannot copy-drift the way hand-vendored scripts did; it reaches a shell only
  once that shell pins a release carrying it, and the ecosystem-drift canary
  runs the HEAD audit advisory against every shell in the meantime.
- Repo Identity states the shell's **intent** — what users can ultimately do
  with it — and links a vision/intent doc when the surface is non-trivial
  (School: `spec/vision.md`). A deliverables checklist is not an intent
  statement: checklist-shaped specs measurably produce fixture-shaped,
  over-narrowed implementations (School audit §2, cause 2).

School exemplar: [`school/AGENTS.md`](https://github.com/Chelis-Lang/school/blob/main/AGENTS.md).

## 2. Toolchain & pin hygiene (MUST)

- **Track the latest published, validation-clean chelis release.** A stale
  pin is drift, not a choice. If the latest release fails the shell's
  validation: document the blocker (filed upstream, §4), pin the newest
  known-good release, and record the re-probe trigger. (Snapshot 2026-06:
  School is on the newest known-good `=0.7.23` with the 0.7.24 blocker
  filed as chelis#345; nautilus sits on `=0.7.8` and coral on `=0.7.10`
  with no documented blocker — that is the failure mode this rule names.)
- `reef.toml` pins exactly: `compiler = "=X.Y.Z"`. **Every** workflow that
  installs a toolchain (`ci.yml`, `release.yml`, nightlies) carries a
  matching `CHELIS_TAG`/`CHELIS_VERSION` env pair, and an **offline
  pin-consistency check runs as a blocking CI guard** over all of them.
  Rationale: with only a reef↔ci guard, School's `release.yml` silently sat
  at v0.7.20 across two pin bumps.
  School exemplar: [`scripts/audit_workarounds.py`](https://github.com/Chelis-Lang/school/blob/main/scripts/audit_workarounds.py)
  `--pins-only`, wired as a `hard-rule-guard` step in
  [`ci.yml`](https://github.com/Chelis-Lang/school/blob/main/.github/workflows/ci.yml).

For the historical Coral/Nautilus thin callers, the offline conformance audit
recognizes only the exact legacy central implementation
`Chelis-Lang/ci/.github/workflows/consumer.yml@4394706b569bdd7d557f6edc7b9818249decc330`
and its closed, profile-specific inputs and secret. An audit `Pass` is a
structural claim about those legacy profile jobs, **not** independent consumer
acceptance of that SHA, private workflow access, or evidence of hosted
execution. Coral and Nautilus must each review and accept their central
revision independently and prove their own hosted suite before migration;
Nautilus #32 remains a draft and does not authorize Coral. The newer
`ci/main` capability selectors have different semantics and are not covered
by this historical recognition rule. Audit of a new central SHA requires
reviewing its job behavior and updating the auditor, not treating any
40-character immutable reference as equivalent.
An unknown central pointer in any workflow is a failed workflow-pin row,
even if another workflow invokes the known legacy revision.
Central workflow references use GitHub's case-insensitive owner/repository
identity; the `.github/workflows/consumer.yml` path and accepted commit
revision stay exact. A job-level `uses` pointing at the same central
repository with a mutable or different revision fails the workflow-pin row
even if another job calls the known revision. Comments and run strings are
not central callers.
The offline auditor reads YAML jobs structurally, independent of indentation
width or quoted keys, and resolves aliases before deciding whether a job
calls that central workflow. Duplicate keys or malformed YAML fail closed.
Nested step fields, comments, and run blocks cannot become job-level callers.
For a legacy central CI profile, `chelis-tag` must equal `vX.Y.Z` and
`chelis-version` must equal `X.Y.Z` for the exact `reef.toml` compiler pin;
extra or missing `v` prefixes do not count. The historical Coral CI guard
also requires `package-version` to equal `[package].version` and
`nautilus-tag` to equal `v` plus `[dependencies].nautilus.version` in that
same `reef.toml`; missing or non-numeric versions cannot establish the guard.
For that exact historical revision, the audit also checks the raw `reef.toml`
shape the 439 profile's grep-based guards can read, not merely equivalent TOML
values: its compiler pin is an unindented `compiler = "=X.Y.Z"` line; Coral's
package version is the unindented `version = "X.Y.Z"` source, and its Nautilus
dependency is an unindented same-line `nautilus = { version = "X.Y.Z" }`.
The auditor requires one unambiguous readable compiler and Nautilus source,
and uses the first unindented package-version source like the historical guard;
each value must agree with parsed TOML and caller inputs. Other valid TOML
spellings, such as `[dependencies.nautilus]` with `version` on a following
line, do not certify the *historical* central guard; this does not constrain
newer central profiles or the shell's own TOML interpretation.
For each recognized historical Coral/Nautilus profile, the supplied Linux
digest (and Darwin digest when that profile supplies one) must equal the
matching platform value under the compiler version in the committed
`.github/chelis-toolchains.json` lock. That lock must be a regular non-symlink
file of at most 65536 bytes, with schema `chelis-toolchain-digests/v1`, 1–32
numeric-version entries, and exactly valid `linux-x86_64` and `darwin-arm64`
SHA-256 digests per entry, as checked by the historical workflow. Source
drift or an unsafe/missing lock fails the workflow-pin row and cannot make
the central caller certify a blocking guard, installer or suite. This
comparison is for the accepted historical revision only, not newer `ci/main`.
To certify the CI pin guard and negative or blocked suites, the thin caller
must also have explicit top-level `on` events covering both pull requests
targeting `main` and pushes to `main`.
The event policy reads the same structurally parsed YAML root as job `uses`,
so quoted `on`, `push`, and `pull_request` keys have the same meaning as their
unquoted spellings; duplicate keys and malformed YAML cannot certify a gate.
The offline audit accepts unfiltered events or literal `main` branch filters
without negations or other restricting filters; manual-only, absent, or
inert source markers do not establish a blocking change gate. This is only
minimum offline event viability: the central wrapper validator owns exact
profile trigger and concurrency equality under ci#5 RWF-015.

- Toolchain installs go through an installer that reads the reef pin. The
  first-party path (shipped; WS-B/WS-C of
  [`chelis_packaging_and_install.md`](chelis_packaging_and_install.md)) is
  `chelisup`: bootstrap once via `chelisup.sh`, then `chelisup install <ver>`
  — or let `chelis reef setup` provision the pin, the lockfile install, and
  the source-crate sync in one verb (user guide:
  [`docs/book/src/install.md`](../../docs/book/src/install.md)). New shells
  MUST use `chelisup`; a shell still carrying a checked-in per-repo installer
  (School: `scripts/install_chelis_toolchain.py`, the pre-chelisup form)
  remains conformant until its WS-0 migration, which is parked/optional.
  Never vendor or build the compiler inside a shell; never hand-symlink;
  consume the released tarball (CI auth via a `CHELIS_RELEASE_TOKEN`-style
  PAT with `contents: read` on every private dep the shell consumes).
- **Optional Nix verification job (MAY).** A shell MAY add one CI job that
  rebuilds the shell with atoll's
  [`chelis2nix`](https://github.com/Chelis-Lang/atoll/tree/main/pkgs/by-name/chelis2nix)
  and compares the result with its chelisup lane. The released-tarball rule
  above yields to this job only while the job meets every condition:
  - The chelisup lane stays and remains the gate. The job runs only on
    `push` to `main` and is never a required check.
  - The job takes the compiler only by substitution from the CProof mesh
    cache, signed by that cache, and fails rather than build it. The
    compiler is the Chelis flake's `packages.<system>.chelis` at the pinned
    release tag's commit, as atoll's reviewed toolchain table records it.
  - The `<name>-<version>.tar.zst` and `<name>-<version>.chb` it builds equal,
    byte for byte, the files that the chelisup lane's `chelis reef build`
    wrote in the same workflow run. Otherwise the job fails and names both
    hashes.
  - It reads private repositories only through short-lived tokens limited to
    `contents: read` on the repositories each step reads. No credential
    enters a Nix build or the Nix store.
  - Its workflow carries the `CHELIS_TAG`/`CHELIS_VERSION` pair above. Before
    the shell bumps its pin, atoll's toolchain table gains the new version.
  - The shell lists the job under its `AGENTS.md` Scaffolding Drift Rule
    section as a recorded per-repo divergence (§10), with a link to this
    clause. Other shells need not mirror it.

  Rationale: the Nix-built compiler is not the released binary, yet Nix
  builds of published shell releases reproduced them byte for byte, and
  atoll's tests check those rebuilds on every push. The comparison on every
  merge extends that evidence to the shell's own code.
- **Per-repo toolchain resolution; installs have no machine-global side
  effects.** Toolchains install side-by-side in a version-keyed store
  (first-party: `$CHELIS_HOME/toolchains/<ver>`, default `~/.chelis/`;
  School's pre-chelisup launcher used `~/.local/share/chelis/<ver>/`), and
  the PATH entrypoint resolves the version **at invocation time from the
  invoking repo's reef pin**. The chelisup shim's order, first match wins:
  `+<ver>` argument → `CHELIS_TOOLCHAIN` env → nearest `chelis-toolchain`
  file → nearest `reef.toml` `compiler =` pin → an explicitly recorded
  default used only outside packages. Installing a toolchain MUST NOT
  repoint the machine default — a fixed
  symlink-to-the-last-installed-version makes bare `chelis` run the wrong
  toolchain for every *other* repo on the machine wherever the reef pin
  guard doesn't reach (single-file `fmt`/`check`, `eval`, editor
  integrations, scripts). A resolved-but-not-installed version is a loud
  error naming `chelisup install <ver>`, never a silent fallback to another
  version. This requirement is the *behavior*, not the tool: first-party
  `chelisup` (chelis#164, shipped) satisfies it out of the box, and shells
  retire their per-repo launchers as they migrate (WS-0). School exemplar
  (pre-chelisup form): the pin-resolving launcher written by
  [`scripts/install_chelis_toolchain.py`](https://github.com/Chelis-Lang/school/blob/main/scripts/install_chelis_toolchain.py).
- **Python is uv-managed, never the system installation** (this surfaces the
  monorepo `AGENTS.md` §Python And Scripts for shells): repo scripts
  are stdlib-only and invoked via `python3`/`uv run --python X.Y`; any
  dependency-bearing harness (parity oracles etc.) is its **own uv project**
  (`pyproject.toml` + `uv.lock`, `uv sync --frozen` in CI), keeping heavy
  deps out of the shell's import space. Never `.sh` scripts — Python only.
- **Source crates (conditional MUST — only shells that link chelis crates
  as Cargo path deps).** A shell whose `Cargo.toml` carries
  `chelis-ir`/`chelis-types`/… as `path = "../chelis/..."` dependencies
  consumes a third dependency class beyond the toolchain binary and
  chelis-std: the chelis compiler **source crates**. A Cargo path dep
  carries no version constraint, so the single `../chelis` sibling slot
  cannot be every co-located shell's pinned build input at once — left
  unmanaged, the crate-linking lane silently compiles against whatever
  version sits at `../chelis`. Such shells MUST:
  - declare a `[chelis-src]` section in `reef.toml` (`crates = [...]` plus a
    `pin_commit` that mirrors the workflow `CHELIS_PIN_COMMIT` surface; the
    offline pin-consistency check guards their agreement);
  - source those crates from a **version-keyed, immutable store**
    (`$CHELIS_HOME/src/<ver>/`, default `~/.chelis/src/<ver>/`, a git
    worktree of canonical `Chelis-Lang/chelis` at the pinned release
    **commit** — never a local tag), and point `../chelis` at the worktree
    matching the shell's own pin
    via a **symlink**, exactly as the chelisup shim resolves the toolchain
    binary per `reef.toml`. `chelis reef src sync` provisions the store and the
    symlink; it refuses (never deletes) a real dev clone in the slot;
  - add a **local-only** drift guard (`chelis reef src check`) to the
    Workspace Gate — it asserts the store worktree, the `../chelis` symlink,
    and `Cargo.lock` are all at the pin. CI is exempt: it checks out the
    pinned chelis sibling fresh at `CHELIS_PIN_COMMIT`.

  The committed `Cargo.toml` keeps the relative `path = "../chelis/..."` (CI's
  sibling checkout relies on it); the wiring is local-dev-only and must not
  edit `Cargo.toml` or CI. `chelis reef doctor` reports this class alongside
  the toolchain and chelis-std across a machine's shells. As with the
  launcher this is the *behavior*, not a specific script; the sync verbs
  live in the pinned toolchain, and the version-independent `chelisup` shim
  (chelis#164, shipped) provides cross-version reach via
  `chelis +<ver> reef src sync` (packaging design §5.4). Full design,
  including the symlink-vs-`.cargo`-override rationale, lives in
  [`chelis_source_crate_sourcing.md`](chelis_source_crate_sourcing.md).

## 3. Capability surface doc — `docs/CHELIS_SURFACE.md` (MUST)

An inventory of what chelis + chelis-std **actually provide to this shell**,
with two-state version markers:

- the primitive/builtin families the shell's domain touches (with AD-adjoint
  status wherever the shell uses `grad`), rank-polymorphism status,
  grad-lane rules, type-system limits relevant to the shell's architecture,
  the chelis-std module surface, and a where-to-read-more table into the
  chelis numbered specs and design docs;
- every capability row marked **`@pin`** (usable today) vs **`@upstream`**
  (next bump); header carries pinned version, latest upstream version, and
  last-refreshed date; refreshed at every pin bump (§7 step 5).

The binding rule this doc carries: **read it before designing around a
suspected language gap.** The School audit's first cause was exactly this
visibility gap — contributors wrote recursive list-walks in a tensor-first
language and froze parameters out of training because nobody could see the
real surface or the upstream roadmap.

School exemplar: [`docs/CHELIS_SURFACE.md`](https://github.com/Chelis-Lang/school/blob/main/docs/CHELIS_SURFACE.md).

## 4. Upstream-issue discipline (MUST)

- **`docs/UPSTREAM_BUGS.md`** (this exact name) with sections
  §Actively blocking / §Tracking / §Parked / §Archived and a stated
  per-section re-probe cadence. Entries carry: minimal reproducer, affected
  shell surface, workaround taken, and an explicit re-probe trigger.
- **File upstream; cite by number.** Suspected chelis bugs and capability
  gaps are filed in `Chelis-Lang/chelis` — or parked as a ready-to-file
  draft under `docs/issue_drafts/` with the filing condition stated — and
  are thereafter cited as `chelis#NNN` (or the draft path), **never by a
  prose name**. A prose-name citation is invisible to every mechanical
  audit: School carried a "generic-callback-unification limit" through
  three docs and a shipped PR while the fix (chelis#293 → PR #297) was
  already in the pin the PR was built on. Before filing, search the
  upstream tracker by symptom keywords — the same incident also nearly
  produced a duplicate filing.
- **A sibling-shell blocker cites the sibling.** When the blocking artifact
  is another shell's issue or PR rather than a compiler defect, cite it as
  `<repo>#NNN` for any repo in the shell-ecosystem registry — `nautilus#43`,
  `coral#27`, `shoals#52`. The registry is the machine-readable §Shell
  Ecosystem set (`chelis_conformance::registry::REGISTRY`), and membership
  in it is exactly the liveness the cite-by-number rule buys: the reference
  resolves in the org, dedupes across sibling shells, and can be re-probed
  at the next bump. Write a sibling citation **tight**: `coral#27`, with no
  space around the `#`. Four shells are also ordinary English nouns
  (`school`, `hull`, `coral`, `whale`), so the spaced form would make prose
  such as "the school #1 priority" scan as a citation; only `chelis` keeps
  the older spaced spellings. A **bare `#NNN`** and a repo **outside** the
  registry stay rejected — neither resolves without guessing which tracker
  was meant.
  A cascade wave makes sibling blockage the common case rather than an edge
  (seven of eleven registry shells were blocked on one sibling release
  during the 0.18.5 wave), so do **not** manufacture a `docs/issue_drafts/`
  file whose only content is a pointer at a sibling PR: a draft is
  pre-filing staging for an issue that will be filed, not a citation
  costume.
- **Narrowing-citation rule.** Any narrowing in shell code or spec — a
  `fail(...)` guard on a config the reference accepts, a frozen/untrained
  parameter, a fixed shape, a per-rank verb copy, a forward-only verb —
  cites, **at the narrowing site**, either `chelis#NNN` / a registry
  sibling's `<repo>#NNN` / a parked draft, or a dated deferral slot in the
  shell's own plan. "Implementation convenience" is not a citable reason.
  An uncited narrowing is invisible to de-narrowing and will outlive its
  justification.
- **Surface loudly; never silently work around.** Hitting a suspected
  upstream bug mid-build means, in the same change set as the workaround:
  minimal reproducer → upstream-tracker dedup search → file (or park a
  draft) → cite at the site → UPSTREAM_BUGS entry.
- **Staleness audit.** A stdlib-only script scans the repo for
  `chelis#NNN` / `<sibling>#NNN` citations and flags issues that are CLOSED
  upstream but still cited from code. Resolved-upstream-but-still-worked-around
  is the default failure state, not the exception. Run at every bump (§7 step 3);
  the offline pin check from §2 lives in the same script.

School exemplars: [`docs/UPSTREAM_BUGS.md`](https://github.com/Chelis-Lang/school/blob/main/docs/UPSTREAM_BUGS.md),
[`docs/issue_drafts/`](https://github.com/Chelis-Lang/school/tree/main/docs/issue_drafts),
[`scripts/audit_workarounds.py`](https://github.com/Chelis-Lang/school/blob/main/scripts/audit_workarounds.py).

## 5. Upstream-blocker probes — `tests_blocked/` (conditional MUST)

Triggered the moment the shell has ≥1 open upstream blocker with an
expressible reproducer (every active shell qualifies today). The blocker
inventory must be **executable**, not archival:

- Each open blocker the harness can express gets a minimal reproducer as an
  **expected-to-fail** test: `tests_blocked/<area>/<name>.ch` + `.expect`
  sidecar — line 1 is the pinned diagnostic substring, lines 2+ are the
  required blocker citation (§4) plus the **on-pass de-narrowing
  instructions**.
- A runner with three verdicts: **OK** (still fails with the pinned
  diagnostic — the healthy state), **FIX-detected** (the probe passes →
  upstream fixed it → the gate fails loudly and prints the sidecar's
  de-narrowing instructions), **DRIFTED** (fails with a different
  diagnostic → the upstream failure mode moved; investigate before
  re-citing). Probes run in **isolated per-file invocations** — some pinned
  failure modes poison a shared compile unit and would contaminate sibling
  probes' diagnostics.
- Runs in **CI on every PR** (proves the probes still fail at the pin) and
  as a pin-bump checklist step (§7 step 2), where FIX-detected is the
  *wanted* outcome: execute the sidecar instructions, promote the probe to
  a real test under `tests/`, archive the UPSTREAM_BUGS entry — same
  change set.
- Pin diagnostics **from the live pinned toolchain, never transcribed from
  docs** (a School-documented failure mode turned out not to reproduce —
  the probe-authoring run is what caught it). A README maps probe↔blocker
  and lists blockers the harness *cannot* express (check-context-only,
  package-build-context-only, cross-module-boundary), which stay on the
  manual re-probe list.

School exemplars: [`tests_blocked/`](https://github.com/Chelis-Lang/school/tree/main/tests_blocked),
[`scripts/run_blocked_probes.py`](https://github.com/Chelis-Lang/school/blob/main/scripts/run_blocked_probes.py).

## 6. Negative tests — `tests_neg/` (MUST)

The monorepo's negative-test-parity rule, made structural: each negative
case is `tests_neg/<area>/<name>.ch` + `.expect` sidecar (first line = the
required diagnostic substring), driven by a runner that asserts
fail-with-substring and fails closed on config errors; wired into CI and
the local gate.

School exemplars: [`tests_neg/`](https://github.com/Chelis-Lang/school/tree/main/tests_neg),
[`scripts/run_negative_tests.py`](https://github.com/Chelis-Lang/school/blob/main/scripts/run_negative_tests.py).

## 7. Pin Bump Checklist (MUST)

A pin bump is a **de-narrowing event**, not a version edit. It is **mechanized
by `chelis reef conform bump <ver>`** (rewrite every pin in lockstep → restamp
the managed blocks + re-materialize skills → run the audit + blocked/negative
suites) and **enforced by `chelis reef conform bump-check --base <ref>`**, a CI
guard that fails any diff which changes the reef pin without a green audit — so
a raw pin edit that skips the checklist cannot land. A bump lands as a **PR**
(shell-driven `bump-pr.yml` opens it; branch protection blocks direct pushes to
`main`), never a direct-to-`main` cascade.

**The write verbs are all-or-nothing on their prerequisites** (chelis#1263).
`bump` and `sync` restamp `reef.toml`, `AGENTS.md`, and
`docs/CHELIS_SURFACE.md` **in place**, so they check all three before their first
write and refuse, with a nonzero exit naming the whole gap and pointing at
`conform init`, if any is absent. A repo that has never been conformed is not a
repo they partially bump: the older behavior ran the edit sequence until it
reached the first missing artifact, leaving the pins rewritten and the skills
materialized behind a failure, and in one measured case reporting success while
doing it. Exiting nonzero *because CI is not green* stays legitimate (CI owns
green-ness); exiting after abandoning its own edit sequence does not.

The shell's `AGENTS.md` carries the checklist (adapted to its surfaces), and
every bump runs all of it in one change set:

1. Update **every** pin location (reef.toml + each workflow's env pair);
   `conform bump` does this and the §2 offline pin check confirms it. Install
   via the §2 installer.
2. Run the §5 blocked-probe suite. FIX-detected → execute the sidecar's
   de-narrowing instructions; DRIFTED → investigate before re-citing.
3. Run the §4 staleness audit; triage every CLOSED-but-still-cited
   workaround site: retire it, or re-cite the live residue issue. No
   silent carryover.
4. Re-probe every UPSTREAM_BUGS entry whose trigger names this release,
   **per-verb/per-surface** — a changelog claim is not a verification
   (School's 0.7.23 lesson: a "fixed" issue unblocked exactly two of the
   verbs it nominally covered). Blockers the probe suite cannot express
   are re-probed manually here.
5. Refresh `docs/CHELIS_SURFACE.md`: header versions and every
   `@pin`/`@upstream` column.
6. Promote UPSTREAM_BUGS entries per the re-probe verdicts (→ §Archived,
   or back to §Tracking with the residue).
7. Run the complete local gate before pushing — fmt + lint + **the
   package build** (`chelis reef build` catches package-context borrow/
   linearity errors that per-file lint+test miss) + tests + negative
   tests + blocked probes + the parity gate where one exists (§9).

Large bumps SHOULD get a one-off migration doc
(School template: [`spec/design/migrate_0.7.24.md`](https://github.com/Chelis-Lang/school/blob/main/spec/design/migrate_0.7.24.md))
pre-staging required changes, the unlock wave, and the re-probe table.

### 7.1 Bump wave audit: what `conform bump` does not do

`chelis +<new-version> reef conform bump <new-version>` is a mechanical
starter, not a green-PR oracle. Run it from a fresh shell worktree with the new
toolchain selected explicitly, then audit every item below:

- Inspect both its exit status and `git status`. Pre-conformance repos can fail
  or exit zero after a partial edit and materialize orphaned `.claude/`,
  `.codex/`, or `agent-skills/` content; remove that half-retrofit and defer
  full adoption to a separate `conform init` change.
- Cascade every dependency surface manually: sibling-shell versions in
  `reef.toml`, sibling tag/version variables in workflows, `[chelis-src]`'s
  exact `CHELIS_PIN_COMMIT`, and every non-frozen nested project `reef.toml`.
  The standard pin guard does not prove tag-to-SHA agreement or cover sibling
  and nested pins.
- Audit the shell's own package version, CHANGELOG convention, and hard-coded
  version strings in both CI and release workflows before tagging. A workflow
  at the tagged commit cannot be repaired by merely rerunning the failed
  release.
- Treat `reef.lock` by entry authority. Regenerate `bundled` toolchain entries.
  Never commit `local_registry` entries without `remote_origin` or hashes
  produced by a private local registry. Keep published dependency entries at
  the last published release during a cascade block, and ensure `reef build`
  precedes any `chelis test` or `chelis prove` step that reads the committed
  lock.
- Re-run `conform audit` and repair every live `docs/UPSTREAM_BUGS.md` entry to
  carry its own `chelis#NNN` or `docs/issue_drafts/<file>` citation. Nested
  detail bullets must not accidentally parse as uncited independent entries.
- Verify a claimed acceptance command by opening the workflow and locating the
  exact step. If the authoritative campaign is expensive, compare a bounded
  pilot at the old and new pins on identical sources; only the delta supports
  a "no new regressions" claim.
- When a skipped version range crosses canonical Surf v0.19, run
  `chelis migrate surf --from 0.18 --inplace` over maintained `.ch` sources,
  then handle semantic migrations the tool cannot choose: explicit literal
  suffixes, i64 extents, and checked `cast` versus truncating `cast_trunc`.
- `chelis test` deliberately does not run the style gate. A shell that
  generates Chelis source must run `chelis check` or `chelis fmt --check` over
  emitted files; do not hide formatter drift behind a measured threshold in
  the generator.
- Finish with `reef build`, the shell's real CI-equivalent gates, and exact
  review of the generated diff. A successful `conform bump` alone is never
  completion evidence.

## 8. Agent surface & skills (MUST)

- The shared skill set (`redteam-exec`, `spec-sync`, `phase-gate`,
  `backend-numerics`, `example-corpus`, `cli-surface`, `packaging-install`,
  `issue-resolution`) is a **materialized pointer upstream, not a fork**. It is
  **embedded in the pinned toolchain**; `chelis reef conform sync` (and
  `reef setup`) materialize it into the shell's `agent-skills/`, and
  `conform audit` derives the expected **toolchain-owned span** of every present
  shared skill from the embedded body plus its declared section selectors and
  byte-checks that result — so the shell owns **zero shared-skill content** and
  any drift is a hard failure. Three declared,
  propagation-safe controls cover the shell's own additions, exclusions, and
  skill-specific overrides, described below. A thin
  `agent-skills/UPSTREAM.toml` records the stamp. This replaces the older
  hand-vendored copy, which drifted silently.
- `.claude/skills` and `.codex/skills` are `../agent-skills` symlinks to the one
  materialized skill tree;
  `.claude/commands/` and `.codex/commands/` wrappers stay mirrored; the
  `red-team` alias stays wired to `redteam-exec` (per monorepo `AGENTS.md`
  §Pointers, Shared skills). `conform sync` restores both symlinks and
  `conform audit` rejects a missing, non-symlinked, or misdirected surface.
- Because the set is materialized from the pinned toolchain, it is always in
  lockstep with the monorepo at the shell's pin after applying the shell's
  declared additions, whole-skill exclusions, and section selectors — a shell
  cannot *silently* fork a shared skill, and a shared-skill change propagates on
  the next `conform sync` / pin bump. The only shell-owned edits are the declared
  controls below.
- **Repo-local domain skills** (chelis#651): a shell MAY carry a skill outside
  the shared set by declaring it in `reef.toml` under
  `[conform] local_skills = ["<name>", ...]`. `conform sync` then preserves those
  dirs and `conform audit` §8 exempts them; an *undeclared* extra skill is still
  pruned, now with a warning rather than a silent delete. A `local_skills` entry
  may not shadow a shared skill. New shells SHOULD vendor School's
  downstream-authoring skill,
  [`agent-skills/chelis-std/`](https://github.com/Chelis-Lang/school/tree/main/agent-skills/chelis-std),
  declared this way.
- **Shared-skill exclusions:** a shell MAY omit irrelevant embedded skills by
  declaring exact names in
  `[conform] excluded_skills = ["<shared-name>", ...]`. `conform sync` removes
  those directories and records only the materialized shared skills in
  `agent-skills/UPSTREAM.toml`; `conform audit` accepts their absence and rejects
  their presence. Every name MUST identify a shared skill in the pinned
  toolchain. An unknown name fails both commands, so a typo or an upstream rename
  cannot become a silent exclusion. Removing a name and syncing restores the
  current embedded skill body. Use a shell-local override block instead when the
  shell needs to amend part of a shared skill while retaining the rest.
  Correspondingly, the **top-level `conform` value** in `reef.toml` carries
  exactly the declarations this contract defines. There are two:
  **`conform.local_skills`**, the repo-local domain-skill allowlist, and
  **`conform.excluded_skills`**, the embedded shared-skill omission list. Both are
  arrays of strings. Anything else under `conform` — any key, at any
  nesting depth — **fails** `conform audit` rather than being silently ignored,
  so a shell can never believe in a control the tool does not implement.
  Recognition is by **key path and value type**, not by spelling: the manifest is
  parsed as TOML and the parsed value is what is checked, so a header
  (`[conform]`), an inline table (`conform = { … }`), a dotted key
  (`conform.local_skills = …`), a quoted key (`"local_skills"`), a sub-table
  (`[conform.skills]`), and an array-of-tables all reach the same answer. A
  *table* at `local_skills` is therefore unrecognized, because the recognized
  declaration's value type is an array. By the same rule the control surface
  itself is a **table**, so a `conform` that is a string, a number, or an array
  (`conform = []`) is reported too: "anything else under `conform`" does not
  cover a `conform` with nothing under it, and a declaration in the wrong shape
  must not read as an absent one. A `conform` table that is not the top-level one
  (`package.conform`, which is what a dotted `conform.exclude` written after a
  table header actually declares) controls nothing and is reported as such. That
  test is on the value, not the name: a `conform` below the top level that
  **cannot carry keys** — a version string, a number, an array of scalars — is
  left alone, because `[dependencies] conform = "1"` is an ordinary dependency
  named `conform` and failing a shell for that would be a false alarm on a MUST
  row. A `reef.toml` that does not parse **fails this row** with the
  parse error: §8 cannot be checked against a file the tool cannot read, and
  reading an unreadable manifest as "declares nothing" would be a silent pass on
  a MUST row. The rule behind all of that is one sentence: **a checker that
  quietly normalizes or drops what it cannot read is itself the bypass.**
- **Shell-specific overrides on a shared skill** (chelis#653): a shell MAY append
  a single trailing `<!-- shell-local:begin -->…<!-- shell-local:end -->` block to
  a shared skill's `SKILL.md` to adapt toolchain guidance that does not fit the
  shell. Ordinary Markdown inside the block adds or supersedes local guidance.
  To remove inherited sections, the block MAY contain one nested
  `<!-- shell-local:exclude:begin -->…<!-- shell-local:exclude:end -->` span.
  Each nonblank line in that span is an exact ATX heading (`##` through
  `######`) wrapped as an HTML comment, for example
  `<!-- ## Device validation -->`; sync omits that heading and its section
  through the next heading of equal or shallower depth. A selector MUST match
  exactly one upstream heading, and
  selected ranges MUST NOT overlap. Missing, duplicate, malformed, or overlapping
  selectors fail sync and audit, so an upstream rename cannot silently restore
  irrelevant guidance. Removing a selector restores the current upstream section
  on the next sync.

  `conform sync` regenerates the retained toolchain-owned body above the trailing
  block and preserves the block verbatim, so upstream edits to retained sections
  still propagate downstream. Audit derives the same filtered body from the
  embedded skill and byte-checks it. When a bump changes retained upstream text
  underneath a block, `conform bump` flags that skill so the author re-checks the
  override. The outer block MUST be a well-formed file suffix (exactly one
  begin/end pair, nothing after the end marker).

## 9. Acceptance & parity (conditional MUST)

- **If the shell validates against external oracles** (torch / sklearn /
  scipy / pandas / ...): the parity harness is its own uv project (§2);
  goldens are checked in and CI only validates — regeneration is an
  explicit, reviewed operation, never automatic; CI guards keep oracle
  libraries out of the shell's own code (no oracle imports outside the
  parity dir, no oracle callables in `.ch` sources — School `ci.yml`
  hard-rule guards).
- **No single-golden acceptance** for new public surface: every new verb's
  acceptance includes ≥2 distinct shape/config cases. One golden is a smoke
  test, not an oracle — single-golden acceptance is how fixture-shaped
  implementations pass review (School audit §2, cause 2; normative
  amendment in School `spec/vision.md`).
- Phase/milestone discipline is inherited from the monorepo `AGENTS.md`
  (one named acceptance oracle; explicit deferral enumeration). The
  shell-side pattern to copy when deferrals pile up: School `AGENTS.md`
  §Phase Completion Discipline.

## 10. Scaffolding drift rule (MUST)

All shells share the same scaffolding shape by design. Any structural
change — layout, CI workflow shape, agent surface, manifest format, the
files this contract requires — is mirrored into the other shells in the
same change set, or explicitly flagged as a per-repo divergence with a
recorded reason. Changes to **this contract** land in the chelis monorepo
first; shells then conform per this rule.

## 11. Bootstrapping a new shell / auditing an existing one

The stamping tool is **`chelis reef conform init`** (shipped in the pinned
toolchain); a new shell is generated from the embedded templates, not by copying
School. This table is the normative source for that tool: it is encoded as data
in `chelis_conformance::manifest::MANIFEST`, and the `manifest_matches_contract_doc`
tripwire fails the build if the two drift — so a row added, renumbered, or
retiered here must move in lockstep with the machine form. `chelis reef conform
audit` walks these rows; the table remains the human bootstrap checklist and
self-audit.

| # | Artifact | Tier | Contract § | School exemplar |
|---|---|---|---|---|
| 1 | `AGENTS.md` (+ `CLAUDE.md` symlink) with required sections + intent | MUST | §1 | `AGENTS.md`, `spec/vision.md` |
| 2 | `reef.toml` exact pin = latest validation-clean release | MUST | §2 | `reef.toml` |
| 3 | Workflow env pins in every toolchain-installing workflow | MUST | §2 | `.github/workflows/{ci,release}.yml` |
| 4 | Offline pin-consistency CI guard | MUST | §2 | `audit_workarounds.py --pins-only` (ci.yml guard) |
| 5 | Toolchain installer + pin-resolving launcher (no global-default side effects) | MUST | §2 | first-party `chelisup` (bootstrap + shim; do not copy School's pre-chelisup `scripts/install_chelis_toolchain.py`) |
| 6 | uv-only Python (stdlib scripts; uv projects for dep-bearing harnesses) | MUST | §2 | `parity/pyproject.toml` |
| 7 | `docs/CHELIS_SURFACE.md` (domain-relevant subset, @pin/@upstream) | MUST | §3 | `docs/CHELIS_SURFACE.md` |
| 8 | `docs/UPSTREAM_BUGS.md` (sections + cadence) | MUST | §4 | `docs/UPSTREAM_BUGS.md` |
| 9 | Citation staleness audit script | MUST | §4 | `scripts/audit_workarounds.py` |
| 10 | `docs/issue_drafts/` convention for parked filings | SHOULD | §4 | `docs/issue_drafts/README.md` |
| 11 | `tests_neg/` + runner, in CI | MUST | §6 | `tests_neg/`, `scripts/run_negative_tests.py` |
| 12 | `tests_blocked/` + runner, in CI | MUST once a blocker exists | §5 | `tests_blocked/`, `scripts/run_blocked_probes.py` |
| 13 | Pin Bump Checklist in AGENTS.md | MUST | §7 | `AGENTS.md` §Pin Bump Checklist |
| 14 | Declared shared-skill subset + validated local additions/section exclusions + agent skill symlinks + mirrored commands | MUST | §8 | `reef.toml [conform]`, `agent-skills/`, `.claude/skills`, `.codex/skills` |
| 15 | Parity harness (own uv project, checked-in goldens, oracle guards) | MUST if external oracles | §9 | `parity/` |
| 16 | ≥2-config acceptance for new public surface | MUST | §9 | `spec/vision.md` amendments |
| 17 | Scaffolding Drift Rule in AGENTS.md | MUST | §10 | `AGENTS.md` §Scaffolding Drift Rule |
| 18 | `[chelis-src]` + `chelis reef src` store/symlink + local drift guard | MUST *if* the shell links chelis crates as Cargo path deps | §2 | hydronnx, calcify (the crate-linking shells; see appendix) |

Bootstrap order for a brand-new shell: stamp from School → rename
`module_prefix` + manifest + module tree → wire pins + CI guards (rows
2–5) → write the intent statement and a CHELIS_SURFACE covering the
domain-relevant subset (rows 1, 7) → land `tests_neg/`/`tests_blocked/`
wired-but-small with their runners (rows 11–12; the first real blocker
populates row 12) → vendor skills (row 14) → run the full local gate.

### Appendix: conformance snapshot (verified on-disk, 2026-06-10)

| Requirement | school | nautilus | coral |
|---|---|---|---|
| Pin freshness (§2; newest known-good = 0.7.23) | `=0.7.23` ✓ | `=0.7.8` ✗ | `=0.7.10` ✗ |
| Pin-consistency CI guard (§2) | ✓ | ✗ | ✗ |
| `docs/CHELIS_SURFACE.md` (§3) | ✓ | ✗ | ✗ |
| `docs/UPSTREAM_BUGS.md` (§4) | ✓ | ✓ | partial (`upstream-bugs.md`, non-canonical name/shape) |
| Staleness-audit script (§4) | ✓ | ✗ | ✗ |
| `tests_neg/` + runner (§6) | ✓ | ✗ | ✗ |
| `tests_blocked/` + runner (§5) | ✓ | ✗ | ✗ |
| Pin Bump Checklist (§7) | ✓ | ✗ | ✗ |
| Vendored shared skills + agent-surface symlinks (§8) | ✓ | ✓ | ✓ |
| Parity harness as uv project (§9) | ✓ | partial | partial |

nautilus and coral predate this contract; their gap rows are the standing
retrofit work list. Retrofit tracking belongs in each shell's own issue
tracker (per-shell umbrella issue mirroring this table), prioritized by the
pin-freshness row — a 13–15-release-stale pin compounds every other gap.

**Source-crate class (row 18), verified 2026-06-29.** Two shells link chelis
crates as Cargo path deps and therefore trigger §2's conditional source-crate
MUST: **hydronnx** (`=0.8.0`; `chelis-ir` + `chelis-types`) and **calcify**
(`=0.7.21`; `chelis-types`). Both currently resolve `../chelis` to the same
real monorepo working tree, so neither is yet conformant — adopting
`[chelis-src]` + `chelis reef src` is their row-18 retrofit. (The earlier
claim that hydronnx was the only crate-linking shell is superseded by calcify.)
All other shells are pure-Chelis and never trigger row 18.
