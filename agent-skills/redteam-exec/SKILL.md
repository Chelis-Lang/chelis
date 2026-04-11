---
name: redteam-exec
description: Use when asked to red team, validate phase progress, or do adversarial review of Chelis code. Executes tests and commands, checks docs/examples/CLI behavior, and reports concrete findings instead of summaries.
---

# Red Team Execution

Use this skill when the task is to validate work, challenge a phase-completion claim, or
find issues in an implementation.

## Workflow

1. Read the owning active spec and the code under review.
2. Run the relevant test targets before making claims.
3. Check the CLI or binary behavior directly when the phase exposes user-facing commands.
4. Probe examples, fixtures, and docs for false-green situations.
5. Add and run adversarial tests when existing coverage is not enough to prove the claim.

## Fresh-Context Agent Rule

When the user asks to "spawn a red team agent" for this repository, interpret that as a
fresh local subagent in the shared workspace, not a remote deployment.

- prefer a newly spawned local subagent with fresh context when that surface is available
- if stale agents block spawning, close them first rather than silently skipping the
  request
- if the available subagent path routes to a remote deployment, is unavailable, or errors
  before work begins, do not keep retrying remote infrastructure; run the red-team pass
  locally in the main thread and say that you did so
- do not present a remote-agent failure as if the requested validation itself failed

## Chelis-Specific Checks

- Treat `cargo test --workspace` as necessary but not sufficient.
- Check whether phase claims depend on ignored tests or manual runners.
- Verify `chelis check` semantics, formatter/decompiler round-trips, and executable examples.
- Cross-check active docs against actual shipped behavior.
- Identify the authoritative phase oracle and verify it directly.
- Look for machine-facing invariant violations, not just test failures.
- On the current repo workstation, do not assume HIP gates are unavailable:
  `rocminfo` reports a local AMD Radeon 8060S / `gfx1100` GPU and `hipcc` is on PATH.
  Treat ignored HIP suites as executable evidence unless some other prerequisite is missing.

## Output Standard

- Findings first.
- Include exact file/test/command evidence.
- Treat "no findings" as suspicious unless you exercised multiple surfaces.
