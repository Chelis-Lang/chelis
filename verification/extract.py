"""Extract the retained production core; model generation is not a Lean proof."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

if __package__:
    from . import native_collection_certificate as native_certificate
else:
    import native_collection_certificate as native_certificate

# Translation options shared by every extraction (single root or whole core).
CHARON_OPTIONS = (
    "--preset=aeneas", "--monomorphize-mut=except-types", "--remove-adt-clauses",
    "--lift-associated-types=*",
)
# Charon rejects its own warnings and stops at the first error when accepting.
CHARON_STRICT = ("--error-on-warnings", "--abort-on-error")
# Transparent adapter fields recombine disjoint mutable captures using their
# real Rust types. Opaque state models instead provide explicit mergers.
ITERATOR_ADAPTERS = (
    "map::Map", "filter_map::FilterMap", "filter::Filter",
    "copied::Copied", "cloned::Cloned", "enumerate::Enumerate",
    "take::Take", "rev::Rev", "skip_while::SkipWhile",
)
# Specialize mutable control-flow payloads from their real Rust bodies;
# the generic value-only Lean models cannot restore borrowed payload state.
CORE_FAMILIES = (
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
    # BTreeMap's Entry methods or_insert, or_insert_with, or_default and insert_entry,
    # translated over the modeled OccupiedEntry/VacantEntry operations (Entry::key is
    # modeled; and_modify and or_insert_with_key inline B-tree internals: rejected).
    "alloc::collections::btree::map::entry::{alloc::collections::btree::map::entry::Entry}",
)


def charon_includes(crates) -> list[str]:
    """The --include arguments: std families first, then the workspace crates."""
    command: list[str] = []
    for adapter in ITERATOR_ADAPTERS:
        command.extend(["--include", f"core::iter::adapters::{adapter}"])
    for family in CORE_FAMILIES:
        command.extend(["--include", family])
    for member in sorted(crates):
        command.extend(["--include", member])
    return command


def run(command: list[str], root: Path, *, capture: bool = False, receipt: Path | None = None) -> str:
    print("+ " + shlex.join(command), flush=True)
    if receipt is not None:
        environment = os.environ.copy()
        log = receipt.with_suffix(".log")
        lines = []
        with log.open("x") as handle:
            process = subprocess.Popen(command, cwd=root, env=environment, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                handle.write(line)
                if capture:
                    lines.append(line)
                else:
                    print(line, end="", flush=True)
            returncode = process.wait()
        native_certificate.save(receipt, {
            "command": command, "cwd": str(root.resolve()), "returncode": returncode,
            "status": "ok" if returncode == 0 else "failed",
            "log": native_certificate.identity(log),
            "environment": {name: environment.get(name) for name in native_certificate.RUST_FLAGS},
        })
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return "".join(lines) if capture else ""
    result = subprocess.run(
        command,
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def generate_model(*, entry: str | None = None, mir: str = "promoted",
                   native_profiles: Path | None = None) -> Path:
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
    crate = entry.split("::", 1)[0] if entry else "chelis_pipeline_core"
    if crate not in crates:
        raise ValueError("entry must belong to a retained workspace crate")
    selected_profile = native_certificate.select_profile(
        source_root=root, package=crates[crate], features=[], profiles=native_profiles)

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
        "charon", "cargo", *CHARON_OPTIONS, f"--mir={mir}", *CHARON_STRICT,
        *charon_includes(crates),
    ]
    if entry:
        command.extend(["--start-from", entry])
    command.extend([
        "--dest-file", str(llbc), "--", "--package", crates[crate],
        "--lib", "--jobs", "2", "--locked", "--target-dir", str(target),
    ])
    print(f"Extraction scope: {entry or 'whole chelis_pipeline_core'}", flush=True)
    print(f"Artifacts: {output}", flush=True)
    before = native_certificate.save(output / "core-before.json", native_certificate.source_manifest(root))
    extraction_command = output / "charon-command.json"
    run(command, root, receipt=extraction_command)
    after = native_certificate.save(output / "core-after.json", native_certificate.source_manifest(root))
    if not llbc.is_file() or llbc.stat().st_size == 0:
        raise RuntimeError("Charon returned without a fresh nonempty LLBC model")
    certificate_result = native_certificate.bind_certificate(
        profile_certificate=selected_profile["profile_certificate"], llbc=llbc,
        extraction_command=extraction_command, core_before=before["path"], core_after=after["path"],
        feature_chain=selected_profile["feature_chain"], output=output / "native-certificate")
    lean = output / "lean"
    lean.mkdir()
    native_certificate.save(lean / "extraction-certificate.json", certificate_result)
    # Propositions that admitted collections rely on (e.g. a crate iterator's total
    # size_hint); checked theorems must prove them.
    obligations = output / "obligations.json"
    # -sequential keeps only one function's symbolic contexts live during translation.
    run([
        "aeneas", "-backend", "lean", "-checks", "-abort-on-error", "-warnings-as-errors",
        "-strict-joins", "-sequential", "-emit-collection-obligations", str(obligations),
        "-native-collection-layouts", certificate_result["certificate"]["path"],
        "-dest", str(lean), str(llbc),
    ], root)
    if not any(lean.rglob("*.lean")):
        raise RuntimeError("Aeneas returned without a Lean model")
    data = json.loads(obligations.read_text())
    if data.get("version") != 1 or not isinstance(data.get("obligations"), list):
        raise RuntimeError(f"malformed collection obligations: {obligations}")
    print(f"Collection obligations: {obligations}", flush=True)
    print(f"Generated model: {lean}. Correspondence and Lean acceptance remain separate obligations.")
    return lean


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
    native_certificate.add_profiles_argument(parser)
    args = parser.parse_args()
    try:
        generate_model(entry=args.entry, mir=args.mir, native_profiles=args.native_profiles)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
