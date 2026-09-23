"""Extract the retained production core; model generation is not a Lean proof."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


def run(command: list[str], root: Path, *, capture: bool = False) -> str:
    print("+ " + shlex.join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry",
        help="Diagnostic slice, e.g. chelis_ir::grad::grad_dag_checked; default: whole pipeline core",
    )
    parser.add_argument(
        "--mir",
        choices=("built", "promoted", "elaborated", "optimized"),
        default="promoted",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    metadata = json.loads(
        run(["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"], root, capture=True)
    )
    members = set(metadata["workspace_members"])
    crates = {
        package["name"].replace("-", "_"): package["name"]
        for package in metadata["packages"]
        if package["id"] in members
    }
    crate = args.entry.split("::", 1)[0] if args.entry else "chelis_pipeline_core"
    if crate not in crates:
        parser.error("entry must belong to a retained workspace crate")

    target = root / "target" / "agents" / "aeneas-charon"
    host = next(
        line.removeprefix("host: ")
        for line in run(["rustc", "-vV"], root, capture=True).splitlines()
        if line.startswith("host: ")
    )
    # Charon builds an explicit target. Clean that target's root package so a
    # cached cargo success cannot silently skip model generation.
    run(
        ["cargo", "clean", "--package", crates[crate], "--target", host, "--target-dir", str(target)],
        root,
    )
    output_root = root / ".work" / "aeneas-verification" / "extracted"
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{crate}-", dir=output_root))
    llbc = output / f"{crate}.llbc"
    command = [
        "charon", "cargo", "--preset=aeneas", f"--mir={args.mir}",
        "--monomorphize-mut=except-types", "--remove-adt-clauses",
        "--lift-associated-types=*",
        "--error-on-warnings", "--abort-on-error",
    ]
    # Transparent adapter fields recombine disjoint mutable captures using their
    # real Rust types. Opaque state models instead provide explicit mergers.
    for adapter in (
        "map::Map", "filter_map::FilterMap", "filter::Filter",
        "copied::Copied", "cloned::Cloned", "enumerate::Enumerate",
        "take::Take", "rev::Rev",
    ):
        command.extend(["--include", f"core::iter::adapters::{adapter}"])
    # Specialize mutable control-flow payloads from their real Rust bodies;
    # the generic value-only Lean models cannot restore borrowed payload state.
    for family in (
        "core::option::{core::option::Option}",
        "core::result::{core::result::Result}",
        "core::option::{core::ops::try_trait::Try}",
        "core::option::{core::ops::try_trait::FromResidual}",
        "core::result::{core::ops::try_trait::Try}",
        "core::result::{core::ops::try_trait::FromResidual}",
        "core::ops::deref::{core::ops::deref::Deref}",
        "core::ops::deref::{core::ops::deref::DerefMut}",
        "core::ops::arith::{core::ops::arith::Add}",
        "core::cmp",
        "{core::cmp::PartialEq}::ne",
        "{core::cmp::PartialOrd}",
        "{core::cmp::Ord}",
        "core::f64::consts::FRAC_PI_2",
        "core::iter::traits::iterator::Iterator::by_ref",
        "{core::iter::traits::iterator::Iterator}::by_ref",
    ):
        command.extend(["--include", family])
    for member in sorted(crates):
        command.extend(["--include", member])
    if args.entry:
        command.extend(["--start-from", args.entry])
    command.extend([
        "--dest-file", str(llbc), "--", "--package", crates[crate],
        "--lib", "--jobs", "2", "--locked", "--target-dir", str(target),
    ])
    print(f"Extraction scope: {args.entry or 'whole chelis_pipeline_core'}", flush=True)
    print(f"Artifacts: {output}", flush=True)
    run(command, root)
    if not llbc.is_file() or llbc.stat().st_size == 0:
        raise RuntimeError("Charon returned without a fresh nonempty LLBC model")
    lean = output / "lean"
    lean.mkdir()
    # Keep only one function's symbolic contexts live during translation.
    run([
        "aeneas", "-backend", "lean", "-checks", "-abort-on-error", "-warnings-as-errors",
        "-strict-joins", "-sequential", "-dest", str(lean), str(llbc),
    ], root)
    if not any(lean.rglob("*.lean")):
        raise RuntimeError("Aeneas returned without a Lean model")
    print(f"Generated model: {lean}. Correspondence and Lean acceptance remain separate obligations.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
