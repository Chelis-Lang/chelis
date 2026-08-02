# Contributing

Chelis is being built in the open with coding agents and human review.
This file defines the minimum documentation and contribution rules needed to keep the
repo coherent.

## Documentation Hierarchy

When updating docs, treat these as the owning sources:

1. `spec/design/chelis_canonical_reference.md`
   Cross-subject architecture, project boundaries, naming, and project status.
2. `openspec/specs/<capability>/spec.md`
   The controlling source for a subject whose numbered chapter has a complete transfer
   record.
3. `spec/00-11*.md`
   The controlling source for every subject not yet transferred to OpenSpec.
4. `spec/design/chelis_project_plan.md`
   Phased execution plan and remaining design work.
5. `spec/design/archive/`
   Historical rationale only.
   Archived docs must never be treated as current guidance.

If two active docs disagree, fix the disagreement instead of adding a third explanation.

Chapter transfers follow
`openspec/changes/migrate-spec-authority/specs/spec-authority-migration/spec.md`.

## Agent Guidance

Shared agent instructions live in `AGENTS.md`.
`CLAUDE.md` should resolve to the same content so Claude-style and Codex-style workflows
use one canonical rule set.

Project-local reusable agent skills live in `agent-skills/`.
`.claude/skills` and `.codex/skills` should resolve to that same directory.
`.claude/commands/` should resolve to the same canonical skill content rather than
hand-maintained copies.

## Doc Authoring Rules

- Prefer current-state descriptions over pathfinding history.
- Keep settled decisions in active docs and historical debate in archived docs.
- Do not describe rejected options as if they are still live.
- Do not over-specify unimplemented formats or protocols unless a dedicated spec owns
  them.
- When a new doc duplicates an existing active doc, merge or delete rather than keeping
  parallel canon.
- Avoid introducing em dashes in new active-doc prose. When cleaning
  one up, prefer a sentence split, colon, parentheses, comma,
  semicolon, or ASCII ` - ` where that is the clearest punctuation.

## Code and Spec Changes

- Read the relevant spec before editing code.
- Update tests with behavior changes.
- Update the owning doc when a public language or compiler behavior changes.
- Do not revert unrelated work already present in the repo.

## Branch Naming

Type prefixes follow conventional commits (`feat`, `fix`, `test`, `docs`, `style`,
`chore`, `refactor`). Work tied to a project phase uses
`{type}/{phase-id}-{item-slug-kebab}`; work with no phase dependency uses
`{type}/{item-slug-kebab}`.

Numbered phase identifiers use `phase` plus a digit and optional lowercase suffix
(`phase5`, `phase3j`, `phase1a`). Lettered tracks use `phase-a` in branch and commit
scopes and `phase_a` in filenames.

Examples:

```text
feat/phase-a-item6-from-github
fix/phase3j-runtime-shape
docs/spec-nomenclature-expansion
```

## Repo Gate (before every push)

`scripts/gate.py` is the single source of truth for the per-PR
developer-runnable gate; CI runs the same commands. Run the pre-push
subset (chelis#360) before opening or updating a PR:

```sh
.venv/bin/python scripts/gate.py --local
```

The full workspace test suite is CI-owned: open a draft PR early and
let CI (macOS Smoke is the authoritative workspace oracle) run it.
See the README Prerequisites for the toolchain the gate needs (rustup,
cargo-nextest, and the uv-managed Python venv).

## Style Gate (every build, every PR)

`chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` enforce `chelis fmt --check` and `chelis lint --check`
on the input file before the front-end pipeline runs. Blocking style
failures block the build by default. Advisory lint warnings report
valid-but-non-preferred source and do not fail `lint --check` or the
built-in gate. Before opening a PR:

1. Run `chelis fmt --inplace path/to/file.ch` (or `.dp`) on every file
   you touched, or rely on your editor formatter.
2. Run `chelis lint --check` to surface naming/style issues.
3. Re-run `chelis check` and `chelis build` on the affected entry point.

The escape hatch `--allow-style-violations` exists for local emergency
builds and migrations. It bypasses only the style gate, not parse,
type, effect, validation, evaluation, or backend errors. CI must not
pass it. The
`CHELIS_STYLE_GATE_DISABLE=1` environment variable bypasses the gate
entirely; it is reserved for the integration-test corpus and is not
appropriate for production builds.

For lint triage, distinguish **allow** from **keep**. Allow means the
form is accepted project style and should not be reported. Keep means
existing checked-in source may remain for compatibility or baseline
evidence, while new human-facing source should use the preferred form.
`redundant-linearity-call` is currently a keep-style advisory warning
for explicit `copy()` and `drop()` calls.

The style guide that the gate enforces lives in
`spec/01-nomenclature.md`. The lint rules that codify it live under
`crates/chelis-lint/src/rules/`. The user-facing CLI documentation
lives at `docs/book/src/cli.md`.
