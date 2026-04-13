#!/usr/bin/env python3
"""Leanstral API bridge for Chelis Proof.

Given a Lean 4 file containing a theorem with `sorry`, ask the Leanstral
model (via the Mistral chat completions API) to produce a proof body,
patch the file, and verify with `lake build`. Retries up to --passes
independent attempts.

Run:
    python3 proof/scripts/prove.py --file PATH --theorem NAME [--passes N]

Stdlib only. API key via $MISTRAL_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "labs-leanstral-2603"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 8000
DEFAULT_API_TIMEOUT = 240
DEFAULT_LAKE_TIMEOUT = 180
DEFAULT_WINDOW = 40
MAX_CONTEXT_CHARS = 4000
API_URL = "https://api.mistral.ai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a Lean 4 proof assistant. Output ONLY a tactic proof body that "
    "replaces `sorry`. Do not narrate. Do not restate the theorem. If the "
    "`sorry` follows `:=`, your output should start with `by`. Prefer short, "
    "robust tactic proofs. Return the proof inside a single ```lean fenced "
    "code block."
)


class LeanstralAPIError(RuntimeError):
    """Raised when the Mistral API call fails for any reason."""


@dataclass
class ExtractedContext:
    theorem_line: int  # 0-indexed line index of `theorem NAME` line
    sorry_line: int  # 0-indexed line index containing `sorry`
    context_snippet: str  # windowed file context
    indent: str  # leading whitespace on the sorry line
    theorem_name: str


@dataclass
class PassResult:
    success: bool
    proof: str = ""
    error: str = ""
    api_error: bool = False
    lake_output: str = ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="prove.py", description="Leanstral API bridge for Chelis Proof."
    )
    p.add_argument("--file", required=True, type=Path, help="path to .lean file")
    p.add_argument("--theorem", required=True, help="theorem name to prove")
    p.add_argument("--passes", type=_positive_int, default=1)
    p.add_argument("--hint", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--timeout", type=int, default=DEFAULT_API_TIMEOUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _positive_int(s: str) -> int:
    try:
        v = int(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not an integer: {s}") from e
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
    return v


# ---------------------------------------------------------------------------
# Project / file inspection
# ---------------------------------------------------------------------------


def find_project_root(start: Path) -> Path:
    """Walk up from `start` looking for a lakefile."""
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / "lakefile.toml").exists() or (candidate / "lakefile.lean").exists():
            return candidate
    raise FileNotFoundError(f"no lakefile.toml or lakefile.lean above {start}")


_THEOREM_RE = re.compile(r"^\s*(theorem|lemma|example)\s+(\S+)")


def extract_context(
    file_text: str, theorem_name: str, window: int = DEFAULT_WINDOW
) -> ExtractedContext:
    lines = file_text.splitlines()
    theorem_line = -1
    for i, line in enumerate(lines):
        m = _THEOREM_RE.match(line)
        if m and m.group(2).rstrip(":") == theorem_name:
            theorem_line = i
            break
    if theorem_line == -1:
        raise ValueError(f"theorem `{theorem_name}` not found")

    # Find `sorry` at or after theorem_line.
    sorry_line = -1
    for i in range(theorem_line, len(lines)):
        if "sorry" in lines[i]:
            sorry_line = i
            break
    if sorry_line == -1:
        raise ValueError(f"no `sorry` found at or after theorem `{theorem_name}`")

    indent_match = re.match(r"(\s*)", lines[sorry_line])
    indent = indent_match.group(1) if indent_match else ""

    lo = max(0, theorem_line - window)
    hi = min(len(lines), sorry_line + window + 1)
    snippet = "\n".join(lines[lo:hi])
    if len(snippet) > MAX_CONTEXT_CHARS:
        # Trim from the top half first.
        over = len(snippet) - MAX_CONTEXT_CHARS
        snippet = snippet[over:]

    return ExtractedContext(
        theorem_line=theorem_line,
        sorry_line=sorry_line,
        context_snippet=snippet,
        indent=indent,
        theorem_name=theorem_name,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_prompt(ctx: ExtractedContext, hint: str | None) -> list[dict]:
    user_parts = [
        f"Theorem to prove: `{ctx.theorem_name}`",
        "",
        "Here is the Lean 4 file context (a `sorry` appears in the theorem):",
        "",
        "```lean",
        ctx.context_snippet,
        "```",
        "",
        "Replace `sorry` with a valid Lean 4 tactic proof. "
        "Return only the replacement inside a ```lean fenced block.",
    ]
    if hint:
        user_parts.append("")
        user_parts.append(f"Hint: {hint}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Proof extraction from model response
# ---------------------------------------------------------------------------


_LEAN_FENCE = re.compile(r"```lean\b[ \t]*\n(.*?)```", re.DOTALL)
_LEAN4_FENCE = re.compile(r"```lean4\b[ \t]*\n(.*?)```", re.DOTALL)
_ANY_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def extract_proof_from_response(response_text: str) -> str:
    if not response_text or not response_text.strip():
        return ""
    for pat in (_LEAN_FENCE, _LEAN4_FENCE, _ANY_FENCE):
        m = pat.search(response_text)
        if m:
            return m.group(1).strip("\n").rstrip()
    return response_text.strip()


# ---------------------------------------------------------------------------
# File patching
# ---------------------------------------------------------------------------


def patch_file(file_path: Path, ctx: ExtractedContext, proof: str) -> str:
    """Replace the `sorry` token at ctx.sorry_line with `proof`.

    Returns original file content so the caller can restore on failure.
    """
    original = file_path.read_text()
    lines = original.splitlines(keepends=True)
    target = lines[ctx.sorry_line]

    # Preserve the trailing newline if present.
    trailing_nl = "\n" if target.endswith("\n") else ""
    body = target[: -1] if trailing_nl else target

    # Dedent the model's proof, then re-indent to match.
    proof_lines = proof.splitlines() or [""]
    first = proof_lines[0].lstrip()
    rest = proof_lines[1:]
    if rest:
        # Compute minimum indent of the non-empty rest lines for dedent.
        non_empty = [ln for ln in rest if ln.strip()]
        if non_empty:
            min_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty)
        else:
            min_indent = 0
        dedented_rest = [
            (ln[min_indent:] if len(ln) >= min_indent else ln) for ln in rest
        ]
        reindented = [ctx.indent + first] + [
            (ctx.indent + ln) if ln.strip() else ln for ln in dedented_rest
        ]
        replacement = "\n".join(reindented)
    else:
        replacement = ctx.indent + first

    # Replace the first `sorry` on that line; keep any surrounding chars.
    new_line = re.sub(r"sorry", lambda _m: first, body, count=1)
    # If the body was exactly indent+sorry, use the multi-line replacement.
    if body.strip() == "sorry":
        new_line = replacement

    lines[ctx.sorry_line] = new_line + trailing_nl
    file_path.write_text("".join(lines))
    return original


def restore_file(file_path: Path, original: str) -> None:
    file_path.write_text(original)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_leanstral(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    api_key: str | None = None,
) -> str:
    key = api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise LeanstralAPIError("MISTRAL_API_KEY not set")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise LeanstralAPIError(f"HTTP {e.code}: {body[:400]}") from e
    except urllib.error.URLError as e:
        raise LeanstralAPIError(f"URL error: {e.reason}") from e
    except TimeoutError as e:
        raise LeanstralAPIError(f"timeout after {timeout}s") from e

    if status != 200:
        raise LeanstralAPIError(f"HTTP {status}: {body[:400]}")
    try:
        parsed: Any = json.loads(body)
    except json.JSONDecodeError as e:
        raise LeanstralAPIError(f"bad JSON: {e}") from e
    try:
        return parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LeanstralAPIError(f"unexpected response shape: {parsed}") from e


# ---------------------------------------------------------------------------
# lake build
# ---------------------------------------------------------------------------


def run_lake_build(
    project_root: Path, timeout: int = DEFAULT_LAKE_TIMEOUT
) -> tuple[int, str]:
    try:
        r = subprocess.run(
            ["lake", "build"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        return 127, f"lake not found: {e}"
    except subprocess.TimeoutExpired as e:
        return 124, f"lake build timed out after {timeout}s: {e}"
    out = (r.stdout + r.stderr)
    tail = "\n".join(out.splitlines()[-40:])
    return r.returncode, tail


# ---------------------------------------------------------------------------
# Pass orchestration
# ---------------------------------------------------------------------------


def run_one_pass(
    args: argparse.Namespace,
    ctx: ExtractedContext,
    project_root: Path,
) -> PassResult:
    messages = build_prompt(ctx, args.hint)
    try:
        content = call_leanstral(
            messages,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except LeanstralAPIError as e:
        return PassResult(success=False, error=str(e), api_error=True)

    proof = extract_proof_from_response(content)
    if not proof:
        return PassResult(success=False, error="empty proof from model")

    original = patch_file(args.file, ctx, proof)
    code, tail = run_lake_build(project_root)
    if code == 0:
        return PassResult(success=True, proof=proof, lake_output=tail)
    restore_file(args.file, original)
    return PassResult(
        success=False,
        proof=proof,
        error=f"lake build exit {code}",
        lake_output=tail,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(list(sys.argv[1:] if argv is None else argv))
    except SystemExit as e:
        return int(e.code) if e.code is not None else 2

    if not args.file.exists():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        return 3

    try:
        project_root = find_project_root(args.file)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    try:
        file_text = args.file.read_text()
    except OSError as e:
        print(f"error: cannot read {args.file}: {e}", file=sys.stderr)
        return 3

    try:
        ctx = extract_context(file_text, args.theorem)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    if args.dry_run:
        messages = build_prompt(ctx, args.hint)
        print("=== DRY RUN: Leanstral prompt ===")
        print(f"project root: {project_root}")
        print(f"theorem line: {ctx.theorem_line}  sorry line: {ctx.sorry_line}")
        print(f"model: {args.model}  passes: {args.passes}")
        print("--- system ---")
        print(messages[0]["content"])
        print("--- user ---")
        print(messages[1]["content"])
        return 0

    if not os.environ.get("MISTRAL_API_KEY"):
        print("error: MISTRAL_API_KEY not set in environment", file=sys.stderr)
        return 3

    failures: list[PassResult] = []
    for i in range(1, args.passes + 1):
        print(f"[pass {i}/{args.passes}] calling Leanstral...", file=sys.stderr)
        result = run_one_pass(args, ctx, project_root)
        if result.success:
            print(f"[pass {i}] SUCCESS", file=sys.stderr)
            print("--- winning proof ---")
            print(result.proof)
            return 0
        print(f"[pass {i}] FAIL: {result.error}", file=sys.stderr)
        failures.append(result)

    all_api = all(f.api_error for f in failures) and failures
    print("\n=== all passes failed ===", file=sys.stderr)
    for i, f in enumerate(failures, 1):
        print(f"  pass {i}: {f.error}", file=sys.stderr)
    return 2 if all_api else 1


if __name__ == "__main__":
    sys.exit(main())
