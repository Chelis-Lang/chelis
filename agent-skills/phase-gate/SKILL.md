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

The gate is `scripts/gate.py`; do not hand-type `cargo test --workspace` in its place:

```sh
python3 scripts/gate.py --fast   # before every push
python3 scripts/gate.py --local  # once per pull request, on the committed candidate, before ready-for-review
python3 scripts/gate.py --list   # the canonical command list with local/CI ownership
```

The workspace nextest stage is CI-owned and macOS Smoke is the authoritative workspace
oracle. Completion evidence is CI green on the candidate head plus the phase's named
oracle, never a local workspace run alone.

## Additional Required Checks

- verify phase-specific acceptance commands or manual runners
- inspect ignored tests and confirm they are explicitly documented
- verify top-level executable examples still pass `chelis fmt` and `chelis check`
- check that docs do not overclaim behavior the repo does not ship
- verify that the named phase oracle is reflected consistently in plan docs and current-state docs
- On the current repo workstation, HIP manual gates should be treated as locally runnable:
  `rocminfo` exposes a `gfx1100` AMD Radeon 8060S GPU and `hipcc` is installed.
  Do not excuse ignored HIP suites on the assumption that this machine is CPU-only.

## Completion Rule

Do not call the phase complete if any of these remain:

- broken default gate
- missing or ambiguous phase oracle
- hidden manual-only acceptance criteria not documented as such
- false-perfect machine-facing reports
- examples/docs whose meaning contradicts the actual implementation
