"""Construct the Nix lane-check input from the actual locked build closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tree(path: Path, *, suffix: str | None = None) -> str:
    """Hash sorted relative UTF-8 paths, NUL, contents, NUL (no directory metadata)."""
    files = sorted(
        (p for p in path.rglob("*") if p.is_file() and (suffix is None or p.suffix == suffix)),
        key=lambda p: p.relative_to(path).as_posix().encode("utf-8"),
    )
    if not files:
        raise ValueError(f"empty source tree: {path}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def command(*argv: str) -> str:
    return subprocess.check_output(argv, text=True).strip()


def create(args: argparse.Namespace) -> None:
    lock = json.loads(args.flake_lock.read_text(encoding="utf-8"))
    nixpkgs_node = lock["nodes"][lock["nodes"]["root"]["inputs"]["nixpkgs"]]
    libc = args.libc / "lib"
    provider = args.math_provider / "lib"
    libraries = {
        "libc": str(libc / "libc.so.6"),
        "libm": str(libc / "libm.so.6"),
        "math_provider": str(provider / "libopenblas.so.0"),
    }
    for name, path in libraries.items():
        if not Path(path).exists():
            raise ValueError(f"missing {name} in pinned closure: {path}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("lane-check proof must run on native x86_64 Linux")
    chelis = args.chelis / "bin" / "chelis"
    compiler = args.compiler / "bin" / "cc"
    if not compiler.exists() or not chelis.exists():
        raise ValueError("pinned compiler or Chelis executable missing")
    keys = (
        "PATH",
        "NIX_CFLAGS_COMPILE",
        "NIX_LDFLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    )
    environment = {key: os.environ[key] for key in keys}
    if environment["OMP_NUM_THREADS"] != "1" or environment["OPENBLAS_NUM_THREADS"] != "1":
        raise ValueError("reference profile requires exactly one math thread")
    rustc_host = next(
        (
            line.removeprefix("host: ").strip()
            for line in command(str(args.rustc), "-vV").splitlines()
            if line.startswith("host: ")
        ),
        "",
    )
    if rustc_host != args.build_target:
        raise ValueError(
            f"pinned Rust build host {rustc_host!r} contradicts target {args.build_target!r}"
        )
    report = {
        "schema_version": 1,
        "source_sha256": sha256_tree(args.source),
        "compiler_source_sha256": sha256_tree(args.compiler_source),
        "corpus_sha256": sha256_tree(args.corpus, suffix=".ch"),
        "flake_lock_sha256": sha256_file(args.flake_lock),
        "nixpkgs_revision": nixpkgs_node["locked"]["rev"],
        "input_derivations": {
            "chelis": args.chelis_drv,
            "compiler": args.compiler_drv,
            "source": str(args.source),
            "compiler_source": str(args.compiler_source),
            "corpus": str(args.corpus),
            "runtime": args.runtime_drv,
            "libc": args.libc_drv,
            "math_provider": args.math_provider_drv,
        },
        "chelis": {
            "store_path": str(chelis),
            "sha256": sha256_file(chelis),
            "version": command(str(chelis), "--version"),
            "rustc_version": command(str(args.rustc), "--version"),
            "build_host": rustc_host,
            "build_target": args.build_target,
        },
        "compiler": {
            "path": str(compiler),
            "sha256": sha256_file(compiler),
            "version": command(str(compiler), "--version"),
            "target": command(str(compiler), "-dumpmachine"),
        },
        "libraries": {**libraries, "linkage": "dynamic"},
        "nix_system": "x86_64-linux",
        "execution": {"os": "linux", "cpu_isa": "x86_64"},
        "environment": environment,
    }
    if not report["compiler"]["target"].startswith("x86_64-"):
        raise ValueError(f"compiler target contradicts native Linux: {report['compiler']['target']}")
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler-source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--flake-lock", type=Path, required=True)
    parser.add_argument("--chelis", type=Path, required=True)
    parser.add_argument("--chelis-drv", required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-drv", required=True)
    parser.add_argument("--runtime-drv", required=True)
    parser.add_argument("--libc", type=Path, required=True)
    parser.add_argument("--libc-drv", required=True)
    parser.add_argument("--math-provider", type=Path, required=True)
    parser.add_argument("--math-provider-drv", required=True)
    parser.add_argument("--rustc", type=Path, required=True)
    parser.add_argument("--build-target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    create(parser.parse_args())


if __name__ == "__main__":
    main()
