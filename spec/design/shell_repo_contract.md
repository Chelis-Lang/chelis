# Downstream Shell Repo Contract

**Status:** NORMATIVE for every shell repo in the
[`chelis_canonical_reference.md`](chelis_canonical_reference.md) §Shell
Ecosystem table (`nautilus`, `coral`, `shoals`, `octant`, `school`, `darwin`,
`hull`, `beacon`, and any future shell). Made binding by `AGENTS.md`
§Downstream Shell Contract. Changes to this contract land in the monorepo
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
- `AGENTS.md` declares: upstream of truth is `Chelis-Lang/chelis`, whose
  monorepo `AGENTS.md` applies **verbatim** unless explicitly overridden,
  and this contract applies in full. Machine-local environment sections of
  the monorepo `AGENTS.md` (the HIP workstation runbook, the macOS
  first-exec notes, workstation-specific measurements and their runbooks)
  describe the monorepo development workstation and are excepted from
  verbatim inheritance: they bind only where the named environment
  actually exists.
- `AGENTS.md` contains at minimum these sections: **Repo Identity**,
  **Toolchain Policy**, **Pin Bump Checklist** (§7), an **Upstream Bugs**
  pointer (§4), and the **Scaffolding Drift Rule** (§10).
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
- Toolchain installs go through a checked-in installer that reads the reef
  pin (School: `scripts/install_chelis_toolchain.py`). Never vendor or
  build the compiler inside a shell; never hand-symlink; consume the
  released tarball (CI auth via a `CHELIS_RELEASE_TOKEN`-style PAT with
  `contents: read` on every private dep the shell consumes).
- **Per-repo toolchain resolution; installs have no machine-global side
  effects.** Toolchains install side-by-side
  (`~/.local/share/chelis/<ver>/`), and the PATH entrypoint resolves the
  version **at invocation time from the invoking repo's reef pin** (env
  override → nearest `reef.toml` walking up from CWD → an explicitly
  recorded default used only outside packages). Installing a toolchain
  MUST NOT repoint the machine default — a fixed
  symlink-to-the-last-installed-version makes bare `chelis` run the wrong
  toolchain for every *other* repo on the machine wherever the reef pin
  guard doesn't reach (single-file `fmt`/`check`, `eval`, editor
  integrations, scripts). A resolved-but-not-installed version is a loud
  error, never a silent fallback to another version. School exemplar: the
  pin-resolving launcher written by
  [`scripts/install_chelis_toolchain.py`](https://github.com/Chelis-Lang/school/blob/main/scripts/install_chelis_toolchain.py).
  This requirement is the *behavior*, not the script: when first-party
  `chelisup` (chelis#164 — whose proposal already specifies the
  reef-pin-honoring shim) ships, shells satisfy it via `chelisup` and
  retire their per-repo launchers.
- **Python is uv-managed, never the system installation** (this surfaces the
  monorepo `AGENTS.md` §Scripting Language Policy for shells): repo scripts
  are stdlib-only and invoked via `python3`/`uv run --python X.Y`; any
  dependency-bearing harness (parity oracles etc.) is its **own uv project**
  (`pyproject.toml` + `uv.lock`, `uv sync --frozen` in CI), keeping heavy
  deps out of the shell's import space. Never `.sh` scripts — Python only.

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
- **Narrowing-citation rule.** Any narrowing in shell code or spec — a
  `fail(...)` guard on a config the reference accepts, a frozen/untrained
  parameter, a fixed shape, a per-rank verb copy, a forward-only verb —
  cites, **at the narrowing site**, either `chelis#NNN` / a parked draft,
  or a dated deferral slot in the shell's own plan. "Implementation
  convenience" is not a citable reason. An uncited narrowing is invisible
  to de-narrowing and will outlive its justification.
- **Surface loudly; never silently work around.** Hitting a suspected
  upstream bug mid-build means, in the same change set as the workaround:
  minimal reproducer → upstream-tracker dedup search → file (or park a
  draft) → cite at the site → UPSTREAM_BUGS entry.
- **Staleness audit.** A stdlib-only script scans the repo for
  `chelis#NNN` citations and flags issues that are CLOSED upstream but
  still cited from code — resolved-upstream-but-still-worked-around is the
  default failure state, not the exception. Run at every bump (§7 step 3);
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

A pin bump is a **de-narrowing event**, not a version edit. The shell's
`AGENTS.md` carries this checklist (adapted to its surfaces), and every bump
runs all of it in one change set:

1. Update **every** pin location (reef.toml + each workflow's env pair);
   verify with the §2 offline pin check. Install via the §2 installer.
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

## 8. Agent surface & skills (MUST)

- Vendor the monorepo's shared skill set from `chelis/agent-skills/`
  (currently `redteam-exec`, `spec-sync`, `phase-gate`, `backend-numerics`,
  `example-corpus`, `cli-surface`) into the shell's `agent-skills/`.
- `.claude/skills` and `.codex/skills` are **symlinks** to `agent-skills/`;
  `.claude/commands/` and `.codex/commands/` wrappers stay mirrored; the
  `red-team` alias stays wired to `redteam-exec` (per monorepo `AGENTS.md`
  §Shared Local Skills).
- Shared skills stay **behaviorally aligned with the monorepo**: a change to
  a shared skill lands in the monorepo first, then propagates to every
  shell in lockstep (§10). A shell never forks a shared skill in place.
- Shells MAY add domain-specific skills (and new shells SHOULD vendor
  School's downstream-authoring skill,
  [`agent-skills/chelis-std/`](https://github.com/Chelis-Lang/school/tree/main/agent-skills/chelis-std),
  which packages the chelis-std SKILL.md guidance for shell authors).

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

There is no stamping tool yet; shells are stamped by copying the freshest
conformant shell — **School, today** — and renaming. The table below is
both the bootstrap checklist and the conformance self-audit.

| # | Artifact | Tier | Contract § | School exemplar |
|---|---|---|---|---|
| 1 | `AGENTS.md` (+ `CLAUDE.md` symlink) with required sections + intent | MUST | §1 | `AGENTS.md`, `spec/vision.md` |
| 2 | `reef.toml` exact pin = latest validation-clean release | MUST | §2 | `reef.toml` |
| 3 | Workflow env pins in every toolchain-installing workflow | MUST | §2 | `.github/workflows/{ci,release}.yml` |
| 4 | Offline pin-consistency CI guard | MUST | §2 | `audit_workarounds.py --pins-only` (ci.yml guard) |
| 5 | Toolchain installer + pin-resolving launcher (no global-default side effects) | MUST | §2 | `scripts/install_chelis_toolchain.py` |
| 6 | uv-only Python (stdlib scripts; uv projects for dep-bearing harnesses) | MUST | §2 | `parity/pyproject.toml` |
| 7 | `docs/CHELIS_SURFACE.md` (domain-relevant subset, @pin/@upstream) | MUST | §3 | `docs/CHELIS_SURFACE.md` |
| 8 | `docs/UPSTREAM_BUGS.md` (sections + cadence) | MUST | §4 | `docs/UPSTREAM_BUGS.md` |
| 9 | Citation staleness audit script | MUST | §4 | `scripts/audit_workarounds.py` |
| 10 | `docs/issue_drafts/` convention for parked filings | SHOULD | §4 | `docs/issue_drafts/README.md` |
| 11 | `tests_neg/` + runner, in CI | MUST | §6 | `tests_neg/`, `scripts/run_negative_tests.py` |
| 12 | `tests_blocked/` + runner, in CI | MUST once a blocker exists | §5 | `tests_blocked/`, `scripts/run_blocked_probes.py` |
| 13 | Pin Bump Checklist in AGENTS.md | MUST | §7 | `AGENTS.md` §Pin Bump Checklist |
| 14 | Vendored shared skills + symlinked skill dirs + mirrored commands | MUST | §8 | `agent-skills/`, `.claude/skills` |
| 15 | Parity harness (own uv project, checked-in goldens, oracle guards) | MUST if external oracles | §9 | `parity/` |
| 16 | ≥2-config acceptance for new public surface | MUST | §9 | `spec/vision.md` amendments |
| 17 | Scaffolding Drift Rule in AGENTS.md | MUST | §10 | `AGENTS.md` §Scaffolding Drift Rule |

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
| Vendored shared skills + symlinks (§8) | ✓ | ✓ | ✓ |
| Parity harness as uv project (§9) | ✓ | partial | partial |

nautilus and coral predate this contract; their gap rows are the standing
retrofit work list. Retrofit tracking belongs in each shell's own issue
tracker (per-shell umbrella issue mirroring this table), prioritized by the
pin-freshness row — a 13–15-release-stale pin compounds every other gap.
