# Chelis Proof — Toolchain Requirements

The Lean mechanization in WS3 (and the paper-side proof cross-checks in WS2) run on a specific local toolchain. This file documents every piece, how to install it, and how it wires together.

## Overview

```
Claude Code (orchestrator)
    │
    └─▶ vibe --agent lean -p "<task>"
              │
              ├─▶ Leanstral (labs-leanstral-2603, 119B MoE / 6.5B active)
              │     via https://api.mistral.ai/v1 with MISTRAL_API_KEY
              │
              └─▶ lean-lsp-mcp (stdio MCP server)
                    │
                    └─▶ lean --server (Lean 4.29.0 LSP)
                          │
                          └─▶ lake build / lake env lean
```

Claude Code drives Vibe in programmatic mode (`-p`). Leanstral reasons, calls `lean-lsp_*` MCP tools for goal state and diagnostics, edits files via `search_replace` / `write_file`, and runs `lake build` for final verification. Every `vibe` invocation is a closed loop — no human in it.

## Components

### 1. Lean 4 toolchain (elan)

- **Manager:** `elan` (Lean's rustup equivalent)
- **Pinned version:** `leanprover/lean4:v4.29.0` (see [`lean/lean-toolchain`](lean/lean-toolchain))
- **Install:**
  ```
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --default-toolchain stable
  ```
- **PATH:** `~/.elan/bin` must be on PATH. Persisted in `~/.bashrc`.
- **Verify:** `lean --version` → `Lean (version 4.29.0, ...)`, `lake --version` → `Lake version 5.0.0-src+...`.
- Any new `lake` project created under `lean/` should inherit `lean-toolchain` to stay pinned.

### 2. Mistral Vibe CLI

- **Package:** `mistral-vibe` (PyPI, installed as a `uv tool`)
- **Minimum version:** 2.5.0
- **Installed version:** 2.7.4
- **Install:**
  ```
  curl -LsSf https://mistral.ai/vibe/install.sh | bash
  ```
  Or directly:
  ```
  uv tool install mistral-vibe
  ```
- **Upgrade:** `uv tool upgrade mistral-vibe`
- **Entry points:** `vibe` (TUI + `-p` programmatic), `vibe-acp` (ACP mode).

### 3. Leanstral model access

Leanstral (`labs-leanstral-2603`) is served by Mistral as a Labs model. Prerequisites:

- **API key** exported as `MISTRAL_API_KEY`. Persisted in `~/.bashrc`. Key rotation is a manual step; do not commit keys.
- **Labs models enabled** at org level in the Mistral admin console: https://admin.mistral.ai/plateforme/privacy. Without this toggle, calls return `403 labs_not_enabled`. This is a one-time human action per organization and cannot be scripted.
- **Billing:** free tier (labeled "Experimental, fast moving, lower QoS"). No budget configuration needed.

### 4. Lean LSP MCP server

- **Package:** `lean-lsp-mcp` (PyPI, authored by Oliver Dressler, MIT)
- **Installed version:** 0.26.1 (checked 2026-04-13)
- **Install:** `uv tool install lean-lsp-mcp`
- **Binary:** `~/.local/bin/lean-lsp-mcp` (stdio transport by default)
- **Provides 21 tools** exposed to Leanstral as `lean-lsp_*`:
  - goal inspection: `lean_goal`, `lean_term_goal`, `lean_hover_info`, `lean_file_outline`
  - diagnostics & build: `lean_diagnostic_messages`, `lean_build`, `lean_verify`
  - lemma search: `lean_local_search`, `lean_leansearch`, `lean_loogle`, `lean_leanfinder`, `lean_state_search`, `lean_hammer_premise`
  - tactic probing: `lean_multi_attempt`, `lean_run_code`, `lean_code_actions`
  - navigation: `lean_declaration_file`, `lean_references`, `lean_completions`
  - advanced: `lean_get_widgets`, `lean_get_widget_source`, `lean_profile_proof`

### 5. Vibe config (`~/.vibe/config.toml`)

Three load-bearing settings on top of Vibe's defaults:

```toml
auto_approve = true                       # below Claude Code's own permission gate
installed_agents = ["lean"]               # activates the builtin Lean agent (equivalent to running /leanstall)
mcp_servers = [
    { name = "lean-lsp", transport = "stdio", command = "lean-lsp-mcp", startup_timeout_sec = 60.0, tool_timeout_sec = 300.0 },
]
```

The `lean` agent itself is builtin (not in this repo). It pins `active_model = "leanstral"`, `system_prompt_id = "lean"`, `thinking = "high"`, `temperature = 1.0`, 168k auto-compact threshold, and 1200s bash timeout.

### 6. Trusted folders (`~/.vibe/trusted_folders.toml`)

Vibe refuses to touch files outside the trusted list. Add the proof workdir before invoking:

```toml
trusted = [
    "/home/jeff/Documents/scratch/chelis-proof",
    # add any additional Lean project roots (e.g. /tmp/lean-smoke) as needed
]
```

## Invocation pattern

Claude Code kicks off a Leanstral proof-filling task like this:

```
cd <lake-project-root>
vibe --workdir . --agent lean \
     -p "<explicit task with absolute file paths and pass/fail criteria>" \
     --max-turns 30 --max-price 5 --output text
```

Notes:

- `--workdir` must match the Lake project root so `lean-lsp-mcp` auto-discovers `lakefile.toml`.
- `-p` puts Vibe in programmatic mode: one prompt, turn cap, auto-exit. In combination with `auto_approve = true` this makes tool calls non-interactive.
- `--max-turns` caps runaway loops. 30 is a sensible default for a single proof.
- `--max-price` caps cost if the Labs free tier ever flips to paid; 5 USD is safety margin.
- `--output text` is a known quirk: with `--agent lean` the text formatter may swallow the final assistant content; fall back to `--output json` and parse the last assistant message if you need machine-readable output.

## Verifying the local setup

Run the checker:

```
python3 proof/scripts/check_toolchain.py
```

The script is fail-fast and prints the first missing piece. Run it after any fresh install or config change.

## Setup on a fresh machine

1. Install elan + Lean 4.29.0 (see §1).
2. Install `mistral-vibe` and `lean-lsp-mcp` via `uv tool install`.
3. Export `MISTRAL_API_KEY` and enable Labs models at https://admin.mistral.ai/plateforme/privacy.
4. Edit `~/.vibe/config.toml`: set `auto_approve = true`, `installed_agents = ["lean"]`, add the `mcp_servers` entry.
5. Add this repo to `~/.vibe/trusted_folders.toml`.
6. Run `python3 proof/scripts/check_toolchain.py` — fix what it complains about.
7. Smoke test: from a scratch Lake project, run the invocation in the previous section with a trivial `sorry` to fill.
