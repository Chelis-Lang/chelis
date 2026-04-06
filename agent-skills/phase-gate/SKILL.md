---
name: phase-gate
description: Use when deciding whether a Chelis phase is actually complete. Applies the repo’s completion standard, checks manual gates, examples, docs, and phase-specific acceptance criteria before any completion claim.
---

# Phase Gate

Use this skill when a phase is claimed complete or nearly complete.

## Acceptance Oracle

Before judging a phase, identify its single authoritative oracle:

- one command
- one named suite
- or one documented manual validation runner

Treat all other evidence as supporting material, not the completion decision itself.

## Default Gate

The minimum gate is:

```sh
cargo build --workspace --all-targets
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
```

## Additional Required Checks

- verify phase-specific acceptance commands or manual runners
- inspect ignored tests and confirm they are explicitly documented
- verify top-level executable examples still pass `chelis fmt` and `chelis check`
- check that docs do not overclaim behavior the repo does not ship
- verify that the named phase oracle is reflected consistently in plan docs and current-state docs

## Completion Rule

Do not call the phase complete if any of these remain:

- broken default gate
- missing or ambiguous phase oracle
- hidden manual-only acceptance criteria not documented as such
- false-perfect machine-facing reports
- examples/docs whose meaning contradicts the actual implementation
