#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_BASE_URL = "http://127.0.0.1:1235/v1/chat/completions"


PROMPTS = [
    ("surf", "Write a rank-0 tensor square."),
    ("surf", "Write vector addition over tensor[n, f32]."),
    ("surf", "Write relu followed by softmax(..., 0)."),
    ("surf", "Write a sigmoid transform over tensor[n, f32]."),
    ("surf", "Write a dimension-polymorphic identity function."),
    ("surf", "Write a type alias plus a passthrough function."),
    ("surf", "Write a bias-add helper using expand over tensor[batch, hidden, f32]."),
    ("surf", "Write a two-layer feed-forward helper using relu and matmul."),
    ("deep", "Rewrite one of the validated patterns as canonical Deep."),
]


BROKEN_DEEP = """(defsig {} square (t-fn {} (t-tensor {} (t-prim {} f32)) (t-tensor {} (t-prim {} f32))))

(def {}
  square
  (fn {}
    (params {} (x {type: (t-tensor {} (t-prim {} f32))}))
    (mul x x)))"""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def extract_code_block(text: str, expected_lang: str) -> str:
    text = strip_think(text)
    patterns = [
        rf"```{re.escape('chelis-' + expected_lang)}\n(.*?)```",
        rf"```{re.escape(expected_lang)}\n(.*?)```",
        r"```[a-zA-Z0-9_-]*\n(.*?)```",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.S)
        if matches:
            return matches[-1].strip()
    return text.strip()


def extract_source(text: str, expected_lang: str) -> str:
    text = extract_code_block(text, expected_lang)
    if expected_lang == "surf":
        match = re.search(
            r"(?ms)^\s*((?:module|def|sig|let|type|dim|import|export)\b.*)$",
            text,
        )
    else:
        match = re.search(r"(?ms)^\s*(\((?:.|\n)*)$", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_model(base_url: str, model: str, messages):
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "messages": messages,
    }
    request = urllib.request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def check_snippet(binary: Path, lang: str, source: str):
    proc = subprocess.run(
        [str(binary), "--lang", lang],
        input=source.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"checker failed with code {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')}"
        )
    return json.loads(proc.stdout.decode("utf-8"))


def is_pass(report) -> bool:
    return (
        report.get("parse_error") is None
        and report.get("fitness", 0.0) >= 0.9
        and not report.get("warnings")
        and not report.get("errors")
    )


def format_feedback(report) -> str:
    if report.get("parse_error"):
        return "\n".join(
            [
                f"Parse error: {report['parse_error']}",
                "Your last response was not valid Chelis source.",
                "Regenerate the full program from scratch.",
                "Do not include analysis, markdown fences, or explanatory text.",
            ]
        )
    parts = [f"Fitness: {report.get('fitness', 0.0):.4f}"]
    warnings = report.get("warnings") or []
    errors = report.get("errors") or []
    if warnings:
        parts.append("Warnings:")
        for warning in warnings:
            parts.append(f"- {warning['kind']}: {warning['message']}")
    if errors:
        parts.append("Errors:")
        for error in errors:
            parts.append(f"- {error['kind']}: {error['message']}")
    parts.append("Regenerate the full program from scratch.")
    parts.append("Do not include analysis, markdown fences, or explanatory text.")
    return "\n".join(parts)


def source_prefix(lang: str) -> str:
    return "(" if lang == "deep" else "def"


def source_rule(lang: str) -> str:
    if lang == "deep":
        return "The first non-whitespace character of your response must be `(`."
    return "Start directly with a Chelis declaration keyword such as `def` or `type`."


def initial_messages(skill_text: str, lang: str, task: str):
    return [
        {
            "role": "system",
            "content": (
                "Follow the supplied Chelis SKILL.md exactly. "
                f"Return only raw Chelis {lang} source. "
                "Do not include markdown fences. "
                "Do not include analysis. "
                "Do not include preamble text. "
                + source_rule(lang)
            ),
        },
        {
            "role": "user",
            "content": f"SKILL.md\n\n{skill_text}\n\nTask: {task}",
        },
    ]


def retry_messages(skill_text: str, lang: str, task: str, report, extra_instruction: str = ""):
    prompt = [
        f"Previous compiler result:\n{format_feedback(report)}",
        "Return only raw Chelis source.",
        "Start directly with source code.",
    ]
    if extra_instruction:
        prompt.append(extra_instruction)
    return [
        {
            "role": "system",
            "content": (
                "Follow the supplied Chelis SKILL.md exactly. "
                f"Return only raw Chelis {lang} source. "
                "Do not include markdown fences. "
                "Do not include analysis. "
                "Do not quote the previous answer. "
                "Regenerate the complete program from scratch. "
                + source_rule(lang)
            ),
        },
        {
            "role": "user",
            "content": f"SKILL.md\n\n{skill_text}\n\nTask: {task}\n\n" + "\n\n".join(prompt),
        },
    ]


def evaluate_prompt(
    base_url: str,
    model: str,
    checker: Path,
    skill_text: str,
    lang: str,
    task: str,
    extra_retry_instruction: str = "",
    pass_predicate=None,
):
    messages = initial_messages(skill_text, lang, task)
    attempts = []
    for attempt in range(1, 6):
        content = call_model(base_url, model, messages)
        code = extract_source(content, lang)
        report = check_snippet(checker, lang, code)
        attempts.append(
            {
                "attempt": attempt,
                "raw_response": content,
                "code": code,
                "report": report,
            }
        )
        passed = pass_predicate(code, report) if pass_predicate else is_pass(report)
        if passed:
            break
        messages = retry_messages(
            skill_text,
            lang,
            task,
            report,
            extra_instruction=extra_retry_instruction,
        )
    return attempts


def evaluate_repair_task(base_url: str, model: str, checker: Path, skill_text: str):
    lang = "deep"
    initial_report = check_snippet(checker, lang, BROKEN_DEEP)
    task = (
        "Repair this broken Deep program using the compiler feedback. "
        "Return only raw Chelis Deep source.\n\n"
        f"Broken program:\n{BROKEN_DEEP}\n\n"
        f"Compiler feedback:\n{format_feedback(initial_report)}"
    )
    return evaluate_prompt(
        base_url,
        model,
        checker,
        skill_text,
        lang,
        task,
        extra_retry_instruction=(
            "Do not repeat the broken body `(mul x x)`. "
            "Use the canonical Deep application form with `app` and `var`."
        ),
        pass_predicate=lambda code, report: is_pass(report)
        and "(def {}" in code
        and "(defsig {}" in code
        and "(app {} (var {} mul) (var {} x) (var {} x))" in code,
    )


def summarize(results):
    passed = 0
    for item in results:
        if item["attempts"] and is_pass(item["attempts"][-1]["report"]):
            passed += 1
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--checker", default="target/debug/check_snippet")
    parser.add_argument("--skill", default="packages/chelis-std/SKILL.md")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    skill_text = Path(args.skill).read_text()
    checker = Path(args.checker)
    if not checker.exists():
        raise SystemExit(f"checker binary not found: {checker}")

    started = time.time()
    results = []
    for index, (lang, task) in enumerate(PROMPTS, start=1):
        attempts = evaluate_prompt(args.base_url, args.model, checker, skill_text, lang, task)
        results.append(
            {
                "id": index,
                "lang": lang,
                "task": task,
                "attempts": attempts,
            }
        )
        print(
            f"[{index}/{len(PROMPTS)+1}] {lang} {task} -> "
            f"attempts={len(attempts)} fitness={attempts[-1]['report'].get('fitness', 0.0):.4f} "
            f"errors={len(attempts[-1]['report'].get('errors', []))} "
            f"warnings={len(attempts[-1]['report'].get('warnings', []))} "
            f"parse_error={attempts[-1]['report'].get('parse_error')!r}",
            file=sys.stderr,
        )

    repair_attempts = evaluate_repair_task(args.base_url, args.model, checker, skill_text)
    results.append(
        {
            "id": len(PROMPTS) + 1,
            "lang": "deep",
            "task": "Repair a broken Deep snippet after compiler feedback.",
            "attempts": repair_attempts,
        }
    )
    print(
        f"[{len(PROMPTS)+1}/{len(PROMPTS)+1}] deep repair -> "
        f"attempts={len(repair_attempts)} fitness={repair_attempts[-1]['report'].get('fitness', 0.0):.4f} "
        f"errors={len(repair_attempts[-1]['report'].get('errors', []))} "
        f"warnings={len(repair_attempts[-1]['report'].get('warnings', []))} "
        f"parse_error={repair_attempts[-1]['report'].get('parse_error')!r}",
        file=sys.stderr,
    )

    summary = summarize(results)
    output = {
        "model": args.model,
        "base_url": args.base_url,
        "skill": args.skill,
        "duration_seconds": round(time.time() - started, 3),
        "summary": summary,
        "results": results,
    }
    rendered = json.dumps(output, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
