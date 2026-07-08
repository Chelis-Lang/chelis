---
name: redteam-exec
description: Run a compliant Chelis red-team pass. Requires stale-agent cleanup and a fresh local subagent; anything else is blocked, not a valid red team.
---

# Red Team Exec

Use this skill when the user asks for a red team, adversarial review, or a fresh-context
validation pass.

## Repository Contract

1. Close any known stale or failed subagents from the current session first.
2. Spawn a new local subagent with fresh context for the validation pass.
3. If the built-in subagent path routes to remote infrastructure, errors, or is otherwise
   broken, retry with another fresh local subagent path.
4. Do not substitute CLI fallback or main-thread validation and call it a red team.
5. If every fresh-local subagent path is unavailable, state that red-team validation is
   blocked.

## Preferred Execution Order

1. `functions.close_agent` on stale or failed agents from the current session.
2. Fresh local subagent via the platform tool when it is actually local and working.
   In Codex sessions, that means the built-in subagent tools (`spawn_agent`,
   `send_input`, `wait_agent`, `close_agent`) rather than shelling out to `claude`,
   `codex exec`, or other external CLIs.
3. If that path is broken, close the failed handle and retry with another fresh local
   subagent.
4. Only count the review as a red team when the fresh-context subagent actually ran the
   commands and reported findings.

## Explicit Non-Goals

- Do not use external agent CLIs as a substitute for the built-in local subagent path
  unless the user explicitly asks for that toolchain.
- Do not treat a remote 404 / deployment error as a successful spawn. Close that handle
  and retry or report the red team blocked.

## Minimum Deliverable

- findings ordered by severity
- exact commands run
- coverage against the active spec and acceptance oracle
- explicit note of anything unvalidated

## Validation Discipline

- run code and commands, not just source inspection
- add adversarial probes where coverage is thin
- verify positive and negative cases
- check examples, docs, and CLI behavior against shipped behavior
