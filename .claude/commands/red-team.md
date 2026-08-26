# `/red-team`

Use the shared `redteam-exec` skill.

Repository-specific contract:

1. Inventory known stale or failed subagents, stop or interrupt them with the
   platform's available lifecycle control, and retire those handles. The platform does
   not need to support deleting them from its listing.
2. Spawn a **new local subagent with fresh context** for the red-team pass.
   In Codex, use `list_agents`, `interrupt_agent`, `spawn_agent`, `send_message` or
   `followup_task`, and `wait_agent` rather than shelling out to `claude`, `codex exec`,
   or other external agent CLIs.
3. Freshness applies to the subagent's context, not the filesystem. Reuse a clean,
   exact-head, idle worktree and its warm target artifacts when available; pass the
   subagent the path, commit, baseline status, and target path.
4. If the spawn path routes to remote infrastructure or errors, stop or interrupt and
   retire that handle, then retry until you either have a working fresh local subagent
   or can state that the red team is blocked.
5. Do **not** fall back to main-thread validation and call it a red team.
6. Restore temporary probes and mutations, then report the final worktree status unless
   the task explicitly asks to retain them.

A red team is only valid when the fresh-context local subagent actually runs the
validation work and returns findings/evidence.
