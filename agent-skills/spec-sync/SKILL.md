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

## Verification

Run the minimum repo gate after the edits:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

