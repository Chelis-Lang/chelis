# Contributing

Chelis is being built in the open with coding agents and human review.
This file defines the minimum documentation and contribution rules needed to keep the
repo coherent.

## Documentation Hierarchy

When updating docs, treat these as the owning sources:

1. `spec/design/chelis_canonical_reference.md`
   Project-level decisions, naming, active backend strategy, CLI surface, and current
   phase status.
2. `spec/00-12*.md`
   Language semantics and subsystem specifications.
3. `spec/design/chelis_project_plan.md`
   Phased execution plan and remaining design work.
4. `spec/design/archive/`
   Historical rationale only.
   Archived docs must never be treated as current guidance.

If two active docs disagree, fix the disagreement instead of adding a third explanation.

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

## Declarative Naming

Use declarative or informational names for branches, commits, plans, tests, files, and
other artifacts. Name the capability, behavior, invariant, or deliverable they contain;
a reader should understand the subject without first finding a roadmap that explains
an opaque sequence label.

Do not use `phase`, `tier`, `stage`, `milestone`, `step`, `item`, or a bare
letter/number as the primary identity. A sequencing label may appear as secondary
tracking context when it is genuinely useful, but it never substitutes for a semantic
name.

Branch names use a conventional-commit type prefix (`feat`, `fix`, `test`, `docs`,
`style`, `chore`, or `refactor`) followed by a descriptive kebab-case slug:

```text
feat/json-serialization
fix/runtime-shape-validation
docs/timeless-spec-contracts
```

Avoid names such as `feat/phase-a-item6`, `fix/tier-2`, or `docs/phase3j`: they record
position but not purpose. When a historical artifact must retain a phase identifier
for cross-reference compatibility, use the surrounding surface's normal case
(`phase_a` in a snake-case filename). This is a legacy compatibility rule, not
authorization for new phase-based names.

## Repo Gate (before every push)

`scripts/gate.py` is the single source of truth for the per-PR
developer-runnable gate; CI runs the same commands. `--fast` is the pre-push
gate: fix-in-place, run before every push. `--local` (chelis#360) is the
once-per-pull-request gate: run on the committed candidate immediately before
marking the draft ready for review, after it is pushed and CI has started:

```sh
python3 scripts/gate.py --fast
python3 scripts/gate.py --local
```

`scripts/gate.py` is stdlib-only and re-executes itself through uv when
`python3` is not already a uv- or Devenv-managed runtime, so that form is
correct in every environment; every other script is invoked as
`.venv/bin/python scripts/<name>.py`.

Push before requesting the red-team round; the review runs against the
pushed head while CI runs on it. The full workspace test suite is
CI-owned: open a draft PR early and
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
