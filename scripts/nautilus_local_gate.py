#!/usr/bin/env python3
"""Run the real Nautilus validation loop against a local Chelis binary.

This keeps GitHub Actions and release artifacts out of the debug loop:

- `baseline` delegates to Nautilus's shipped static + numeric harnesses.
- `eval-imports` exercises `chelis eval --file` against a temporary copy of the
  real Nautilus checkout with only the compiler pin rewritten.
- `tensor-fold` runs a generic tensor-accumulator fold canary on the native
  build path.
- `tensor-grad` runs a stronger LM-shaped tensor-gradient canary than the
  current upstream unit tests.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
DEFAULT_NAUTILUS = REPO.parent / "nautilus"
DEFAULT_LOCAL_CHELIS = REPO / "target" / "debug" / "chelis"

PROFILE_ORDER = ("baseline", "tensor-grad", "tensor-fold", "eval-imports")

TENSOR_GRAD_CANARY = """
def residual[n](theta: tensor[n, f32], x: f32, y: f32) -> f32 = {
  y_hat = if lt(x, cast(0.0, f32)) then tensor_to_scalar(sum(copy(theta), 0)) else add(tensor_to_scalar(sum(copy(theta), 0)), x)
  sub(y, y_hat)
}

row = grad(residual, wrt=(theta))

def jac[n, m](theta: tensor[n, f32], xs: tensor[m, f32], ys: tensor[m, f32]) -> List[tensor[n, f32]] = {
  pairs = zip(to_list(xs), to_list(ys))
  map(fn (pair: (f32, f32)) -> row(copy(theta), pair.0, pair.1), pairs)
}

out = jac(
  to_tensor([1.0, 2.0], f32),
  to_tensor([1.0, -2.0], f32),
  to_tensor([3.0, 4.0], f32),
)
"""

TENSOR_FOLD_CANARY = """
def builder[n](xs: tensor[n, f32]) -> tensor[n, f32] = {
  idxs = range(cast(0, i64), numel(copy(xs)))
  state0 = (to_tensor(map(fn (x: f32) -> cast(0.0, f32), to_list(copy(xs)))), cast(0, i64))
  step = fn (state, i) -> {
    acc = state.0
    total = state.1
    inner = fold(fn (inner_acc: tensor[n, f32], j: i64) -> add(inner_acc, xs), acc, idxs)
    (inner, add(total, i))
  }
  fold(step, state0, idxs).0
}

out = builder(to_tensor([1.0, 2.0, 3.0], f32))
"""

EVAL_IMPORT_SNIPPET = """
import Nautilus.Special (erf)

bench = erf(cast(0.5, f32))
"""


@dataclass
class RunResult:
    ok: bool
    summary: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiles",
        nargs="*",
        choices=PROFILE_ORDER + ("all",),
        help="Validation profiles to run (default: all).",
    )
    parser.add_argument(
        "--nautilus-root",
        type=Path,
        default=Path(os.environ.get("NAUTILUS_ROOT", DEFAULT_NAUTILUS)),
        help="Path to the Nautilus checkout (default: ../nautilus).",
    )
    parser.add_argument(
        "--chelis-bin",
        type=Path,
        default=None,
        help="Chelis binary to test (default: CHELIS_BIN, target/debug/chelis, then PATH).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directories for failed temp-copy profiles.",
    )
    return parser.parse_args()


def resolve_chelis_bin(explicit: Path | None) -> Path:
    candidates: list[Path | str] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_bin = os.environ.get("CHELIS_BIN")
    if env_bin:
        candidates.append(Path(env_bin).expanduser())
    candidates.append(DEFAULT_LOCAL_CHELIS)
    for candidate in candidates:
        if isinstance(candidate, Path) and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("chelis")
    if found:
        return Path(found)
    raise SystemExit("Could not resolve a Chelis binary. Build `target/debug/chelis` or pass --chelis-bin.")


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=merged_env,
        text=True,
        capture_output=True,
    )


def first_nonempty(*chunks: str) -> str:
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped:
            return stripped
    return ""


def trim_detail(text: str, limit: int = 1200) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def chelis_version(chelis_bin: Path) -> str:
    proc = run([str(chelis_bin), "--version"], cwd=REPO)
    if proc.returncode != 0:
        raise SystemExit(f"`{chelis_bin}` did not report a version:\n{first_nonempty(proc.stderr, proc.stdout)}")
    match = re.search(r"(\d+\.\d+\.\d+)", proc.stdout)
    if not match:
        raise SystemExit(f"Could not parse Chelis version from `{proc.stdout.strip()}`")
    return match.group(1)


def native_compile_cmd(binary: Path, sources: list[Path], out_dir: Path) -> list[str]:
    cc_env = os.environ.get("CC")
    if cc_env:
        prefix = [cc_env, "-O2"]
    elif sys.platform == "darwin":
        libomp = Path("/opt/homebrew/opt/libomp")
        if libomp.exists():
            prefix = [
                "clang",
                "-O2",
                "-Xpreprocessor",
                "-fopenmp",
                f"-I{libomp / 'include'}",
                f"-L{libomp / 'lib'}",
            ]
        else:
            prefix = ["clang", "-O2"]
    else:
        prefix = ["gcc", "-O2", "-fopenmp"]

    cmd = [
        *prefix,
        "-o",
        str(binary),
        *(str(source) for source in sources),
        "-I",
        str(out_dir),
        str(out_dir / "libchelis_runtime.a"),
        "-lm",
        "-lpthread",
    ]
    if sys.platform == "darwin" and Path("/opt/homebrew/opt/libomp").exists() and "clang" in prefix[0]:
        cmd.append("-lomp")
    return cmd


def write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents).strip() + "\n")


def build_and_compile_canary(chelis_bin: Path, stem: str, source: str) -> RunResult:
    with tempfile.TemporaryDirectory(prefix=f"chelis-{stem}-") as tmp:
        tmp_dir = Path(tmp)
        program = tmp_dir / f"{stem}.ch"
        out_dir = tmp_dir / "out"
        write_text(program, source)
        build = run(
            [str(chelis_bin), "build", "--emit-c", str(program), "--target", "c", "--output", str(out_dir)],
            cwd=REPO,
        )
        if build.returncode != 0:
            return RunResult(
                ok=False,
                summary=f"`chelis build` failed for {stem}",
                detail=trim_detail(first_nonempty(build.stderr, build.stdout)),
            )
        c_file = out_dir / f"{stem}.c"
        if not c_file.exists():
            return RunResult(ok=False, summary=f"expected generated C for {stem}", detail=str(c_file))
        binary = out_dir / stem
        compile_proc = run(native_compile_cmd(binary, [c_file], out_dir), cwd=out_dir)
        if compile_proc.returncode != 0:
            return RunResult(
                ok=False,
                summary=f"native compile failed for {stem}",
                detail=trim_detail(first_nonempty(compile_proc.stderr, compile_proc.stdout)),
            )
        return RunResult(ok=True, summary=f"{stem} built and compiled natively")


@contextlib.contextmanager
def temp_nautilus_copy(
    nautilus_root: Path,
    *,
    compiler_version: str,
    keep_temp: bool,
) -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix="chelis-nautilus-local-"))
    copied = tmp_root / "nautilus"
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "dist",
        "build",
        ".mypy_cache",
    )
    try:
        shutil.copytree(nautilus_root, copied, ignore=ignore)
        reef_toml = copied / "reef.toml"
        txt = reef_toml.read_text()
        patched = re.sub(r'compiler\s*=\s*"=[^"]+"', f'compiler = "={compiler_version}"', txt, count=1)
        if patched == txt:
            raise SystemExit(f"Could not rewrite compiler pin in {reef_toml}")
        reef_toml.write_text(patched)
        yield copied
    finally:
        if keep_temp:
            print(f"[keep-temp] {copied}")
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)


def profile_baseline(nautilus_root: Path, chelis_bin: Path) -> RunResult:
    env = {"CHELIS_BIN": str(chelis_bin)}
    static = run(["python3", "tests/run_static_checks.py"], cwd=nautilus_root, env=env)
    if static.returncode != 0:
        return RunResult(
            ok=False,
            summary="Nautilus static checks failed",
            detail=trim_detail(first_nonempty(static.stderr, static.stdout)),
        )
    numeric = run(["python3", "tests/run_numeric_tests.py"], cwd=nautilus_root, env=env)
    if numeric.returncode != 0:
        return RunResult(
            ok=False,
            summary="Nautilus numeric harness failed",
            detail=trim_detail(first_nonempty(numeric.stderr, numeric.stdout)),
        )
    passed_line = next(
        (line.strip() for line in reversed(numeric.stdout.splitlines()) if "numerical assertions passed" in line),
        "Nautilus numeric harness passed",
    )
    return RunResult(ok=True, summary=passed_line)


def profile_tensor_grad(chelis_bin: Path) -> RunResult:
    result = build_and_compile_canary(chelis_bin, "tensor_grad_lm_canary", TENSOR_GRAD_CANARY)
    if result.ok:
        result.detail = (
            "LM-shaped tensor-grad canary passed locally. This is stronger than the existing upstream unit tests, "
            "but it is still not the full blocked Nautilus implementation."
        )
    return result


def profile_tensor_fold(chelis_bin: Path) -> RunResult:
    result = build_and_compile_canary(chelis_bin, "tensor_fold_generic_canary", TENSOR_FOLD_CANARY)
    if result.ok:
        result.detail = "Generic tensor-accumulator fold canary built on the native path."
    return result


def profile_eval_imports(
    nautilus_root: Path,
    chelis_bin: Path,
    *,
    compiler_version: str,
    keep_temp: bool,
) -> RunResult:
    with temp_nautilus_copy(
        nautilus_root,
        compiler_version=compiler_version,
        keep_temp=keep_temp,
    ) as copied:
        snippet = copied / "tmp_eval_import.ch"
        write_text(snippet, EVAL_IMPORT_SNIPPET)
        proc = run([str(chelis_bin), "eval", "--file", str(snippet)], cwd=copied)
        if proc.returncode != 0:
            return RunResult(
                ok=False,
                summary="`chelis eval --file` failed against the Nautilus import surface",
                detail=trim_detail(first_nonempty(proc.stderr, proc.stdout)),
            )
        stdout = proc.stdout.strip()
        if not stdout:
            return RunResult(
                ok=False,
                summary="`chelis eval --file` produced no output for the Nautilus import probe",
            )
        return RunResult(ok=True, summary=f"eval import probe succeeded: {stdout}")


def main() -> int:
    args = parse_args()
    chelis_bin = resolve_chelis_bin(args.chelis_bin)
    nautilus_root = args.nautilus_root.expanduser().resolve()
    if not nautilus_root.is_dir():
        raise SystemExit(f"Nautilus root does not exist: {nautilus_root}")
    profiles = list(args.profiles) if args.profiles else ["all"]
    if "all" in profiles:
        profiles = list(PROFILE_ORDER)

    compiler_version = chelis_version(chelis_bin)

    print(f"Chelis:   {chelis_bin} ({compiler_version})")
    print(f"Nautilus: {nautilus_root}")
    print(f"Profiles: {', '.join(profiles)}")
    print()

    results: list[tuple[str, RunResult]] = []
    for profile in profiles:
        if profile == "baseline":
            result = profile_baseline(nautilus_root, chelis_bin)
        elif profile == "tensor-grad":
            result = profile_tensor_grad(chelis_bin)
        elif profile == "tensor-fold":
            result = profile_tensor_fold(chelis_bin)
        elif profile == "eval-imports":
            result = profile_eval_imports(
                nautilus_root,
                chelis_bin,
                compiler_version=compiler_version,
                keep_temp=args.keep_temp,
            )
        else:
            raise AssertionError(f"unexpected profile: {profile}")
        results.append((profile, result))
        marker = "PASS" if result.ok else "FAIL"
        print(f"[{marker}] {profile}: {result.summary}")
        if result.detail:
            print(textwrap.indent(result.detail, "  "))
        print()

    failures = [name for name, result in results if not result.ok]
    if failures:
        print(f"Local Nautilus gate failed: {', '.join(failures)}")
        return 1

    print("Local Nautilus gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
