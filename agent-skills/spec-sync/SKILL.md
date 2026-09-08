---
name: spec-sync
description: Use when changing Chelis language behavior, CLI behavior, or compiler semantics. Keeps code, tests, examples, and active specs in sync in one change set.
---

# Spec Sync

Use this skill whenever a change affects public language/compiler behavior.

## Required Sync Surfaces

- owning implementation files
- unit/integration/e2e tests
- executable examples in `examples/`
- active specs in `spec/`
- current-state docs such as `README.md` and the canonical reference when needed

## Repository Rules

- Prefer updating the owning active doc over adding new explanation docs.
- Do not treat `spec/design/archive/` as current guidance.
- If the change invalidates an example, either rewrite it to remain executable or move it
  to `examples/illustrative/`.
- If the change affects a completion claim, update the phase oracle docs and any current
  phase-status summary in the same change set.

## Verification

Run the pre-push gate after the edits, and the full local gate once on the committed
candidate before ready-for-review:

```sh
python3 scripts/gate.py --fast
python3 scripts/gate.py --local
```
