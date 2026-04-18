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
- **PATH:** `~/.elan/bin` must be on PATH. Persist in the login shell's rc file (`~/.bashrc` on bash, `~/.zshrc` on zsh).
- **Verify:** `lean --version` → `Lean (version 4.29.0, ...)`, `lake --version` → `Lake version 5.0.0-src+...`.
- Any new `lake` project created under `lean/` should inherit `lean-toolchain` to stay pinned.

### 2. Mistral Vibe CLI

- **Package:** `mistral-vibe` (PyPI, installed as a `uv tool`)
- **Minimum version:** 2.5.0
- **Installed version:** 2.7.6 (checked 2026-04-18 on macOS arm64; 2.7.4 was confirmed on Fedora earlier)
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

- **API key** exported as `MISTRAL_API_KEY`. Persist in the login shell's rc file (`~/.bashrc` on bash, `~/.zshrc` on zsh). Key rotation is a manual step; do not commit keys.
- **Labs models enabled** at org level in the Mistral admin console: https://admin.mistral.ai/plateforme/privacy. Without this toggle, calls return `403 labs_not_enabled`. This is a one-time human action per organization and cannot be scripted.
- **Billing:** free tier (labeled "Experimental, fast moving, lower QoS"). No budget configuration needed.

### 4. Lean LSP MCP server

- **Package:** `lean-lsp-mcp` (PyPI, authored by Oliver Dressler, MIT)
- **Installed version:** 0.26.1 (checked 2026-04-13)
- **Install:** `uv tool install lean-lsp-mcp`
- **Binary:** `~/.local/bin/lean-lsp-mcp` (uv's default shim location on Linux and macOS; stdio transport)
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
    "<absolute path to this repo>",  # e.g. /Users/jeff/Documents/cproof/chelis-proof on macOS,
                                     #      /home/jeff/Documents/scratch/chelis-proof on Fedora
    # add any additional Lean project roots (e.g. /tmp/lean-smoke) as needed
]
```

Vibe may rewrite this file on first run; if it creates `trusted = []` with an `untrusted = []` sibling, keep the `untrusted` line and just fill in `trusted`.

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

1. Install elan + Lean 4.29.0 (see §1). If elan is already present but `lean --version` reports "no default toolchain", run `elan default leanprover/lean4:v4.29.0`.
2. Install `uv` if it is not already on PATH: `curl -LsSf https://astral.sh/uv/install.sh | sh`. Confirm `~/.local/bin` is on PATH in the shell rc file.
3. Install `mistral-vibe` and `lean-lsp-mcp` via `uv tool install`.
4. Export `MISTRAL_API_KEY` (persist in the shell's rc file) and enable Labs models at https://admin.mistral.ai/plateforme/privacy. Labs is an org-level toggle; once enabled on the org it persists across machines.
5. Edit `~/.vibe/config.toml`: set `auto_approve = true`, `installed_agents = ["lean"]`, add the `mcp_servers` entry.
6. Add this repo's absolute path to `~/.vibe/trusted_folders.toml`.
7. Install the LaTeX stack (see §LaTeX toolchain below).
8. Run `python3 proof/scripts/check_toolchain.py` — fix what it complains about.
9. Smoke test the API: `curl -sS -w "\nHTTP %{http_code}\n" https://api.mistral.ai/v1/chat/completions -H "Authorization: Bearer $MISTRAL_API_KEY" -H "Content-Type: application/json" -d '{"model":"labs-leanstral-2603","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'` — HTTP 200 means key + Labs toggle + billing are all wired. HTTP 403 with `labs_not_enabled` means step 4's admin toggle is still off.
10. End-to-end smoke: from a scratch Lake project, run the invocation in the previous section with a trivial `sorry` to fill.

## LaTeX toolchain (for `proof/paper/`)

The POPL paper source lives in `proof/paper/` and is built with `pdflatex` (not `latexmk` — that dep is intentionally avoided). The following are required on the build machine:

- `pdflatex` (from TeXLive)
- The `acmart.cls` class file and `ACM-Reference-Format.bst` BibTeX style — **vendored into `proof/paper/`** from the CTAN `acmart` bundle (not from the upstream `borisveytsman/acmart` GitHub default branch, which is `primary` and does not commit the generated `.cls`). If you re-vendor them, fetch from `https://mirrors.ctan.org/macros/latex/contrib/acmart.zip` and run `latex acmart.ins` to regenerate. Verify by comparing `sha256sum proof/paper/acmart.cls` against a fresh extraction.
- These TeXLive packages, which `acmart` pulls in transitively:
  - `texlive-amscls`, `texlive-amsmath`, `texlive-amsfonts`
  - `texlive-libertine`, `texlive-newtx` (fonts)
  - `texlive-microtype`, `texlive-caption`, `texlive-booktabs`, `texlive-setspace`
  - `texlive-upquote`, `texlive-oberdiek`, `texlive-totpages`, `texlive-environ`
  - `texlive-hyperxmp`, `texlive-float`, `texlive-draftwatermark`, `texlive-fancyhdr`
  - `texlive-preprint`, `texlive-comment`, `texlive-ncctools`, `texlive-trimspaces`
  - `texlive-cm-super`, `texlive-ec`
- Install options by platform (pick one; lazy paths get you everything acmart needs):
  - **macOS:** `brew install --cask mactex-no-gui` (≈4 GB; omit `-no-gui` for the full suite). After install, `eval "$(/usr/libexec/path_helper)"` in a new shell so `/Library/TeX/texbin` is picked up.
  - **Fedora / RHEL:** `sudo dnf install texlive-scheme-full` (≈2 GB), or install the `texlive-*` packages enumerated above for a surgical build.
  - **Debian / Ubuntu:** `sudo apt install texlive-full` (≈5 GB), or the equivalent `texlive-latex-extra texlive-fonts-extra texlive-publishers texlive-science texlive-xetex` subset.
  - **Other / no root:** install upstream TeX Live from https://tug.org/texlive/ into `~/texlive` and add `~/texlive/<year>/bin/<arch>` to PATH.
- A historical note from Phase 1 Wave 1 (T9): one early workstation was set up by extracting `texlive-*.rpm` contents into `~/texmf-local/` and symlinking into `~/texmf/` so kpathsea resolved them without root. That local install is **not reproducible from the repo**; treat it as a one-off workaround, not a recommended path. A fresh clone should use one of the install options above.

Build commands:

```
cd proof/paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # second pass settles references
```

The second pass is required because the first pass writes `main.aux`; the second reads it to fix cross-references. Two passes produce `main.pdf`. Warnings about missing citations are expected while the body is skeleton-only.

## Phase 1 invocation commands (Wave 1–4)

For future sessions running the Phase 1 execution plan:

- Lean project build: `cd proof/lean && lake build`
- Paper PDF build: `cd proof/paper && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex`
- Toolchain verifier: `python3 proof/scripts/check_toolchain.py`
- prove.py tests: `python3 -m unittest proof.scripts.test_prove -v`
- Leanstral smoke (requires API reachability + a Lean file with `sorry`): `python3 proof/scripts/prove.py --file <path-to-lean-file> --theorem <name> --passes N`
