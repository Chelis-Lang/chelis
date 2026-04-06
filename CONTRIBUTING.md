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

## Code and Spec Changes

- Read the relevant spec before editing code.
- Update tests with behavior changes.
- Update the owning doc when a public language or compiler behavior changes.
- Do not revert unrelated work already present in the repo.
