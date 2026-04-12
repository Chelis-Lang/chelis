# `/red-team`

Use the shared `redteam-exec` skill.

Repository-specific contract:

1. Close any known stale or failed subagents from the current session.
2. Spawn a **new local subagent with fresh context** for the red-team pass.
3. If the spawn path routes to remote infrastructure or errors, close that handle and
   retry until you either have a working fresh local subagent or can state that the red
   team is blocked.
4. Do **not** fall back to main-thread validation and call it a red team.

A red team is only valid when the fresh-context local subagent actually runs the
validation work and returns findings/evidence.
