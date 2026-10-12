"""Accept fresh production AD extraction and Lean compilation, not a correspondence proof."""

import argparse
import json
from pathlib import Path
import subprocess

from extract import generate_model, run
from native_collection_certificate import add_profiles_argument, select_profile, verify_certificate


AD_ENTRY = "chelis_ir::grad::grad_dag_checked"


def compile_model(model_directory: Path, lean_library: Path) -> Path:
    model = model_directory / "ChelisIr.lean"
    if not model.is_file() or model.stat().st_size == 0:
        raise RuntimeError(f"production AD model is missing or empty: {model}")
    compiled = model.with_suffix(".olean")
    run([
        "lake", "env", "lean", f"--root={model_directory}",
        "-o", str(compiled), str(model),
    ], lean_library)
    if not compiled.is_file() or compiled.stat().st_size == 0:
        raise RuntimeError(f"Lean returned without a nonempty compiled AD model: {compiled}")
    return compiled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_profiles_argument(parser)
    parser.add_argument(
        "--aeneas-lean-dir",
        type=Path,
        required=True,
        help="Writable backends/lean library from the pinned patched Aeneas source",
    )
    args = parser.parse_args()
    lean_library = args.aeneas_lean_dir.expanduser().resolve()
    if not (lean_library / "Aeneas.lean").is_file():
        parser.error(f"Aeneas Lean library is missing: {lean_library}")

    # Reuse Lake's build cache, but never accept an old generated production model.
    run(["lake", "build"], lean_library)
    model_directory = generate_model(entry=AD_ENTRY, native_profiles=args.native_profiles)
    certificate_path = model_directory / "extraction-certificate.json"
    certificate_result = json.loads(certificate_path.read_text())
    verify_certificate(certificate_result, model_directory.parent / "chelis_ir.llbc")
    profile = select_profile(source_root=Path(__file__).resolve().parent.parent,
                             package="chelis-ir", features=[], profiles=args.native_profiles)
    if certificate_result["native_profile_evidence"] != profile["native_profile_evidence"]:
        raise RuntimeError("The production AD certificate uses another native profile.")
    compiled = compile_model(model_directory, lean_library)
    print(f"Production AD accepted: {AD_ENTRY}", flush=True)
    print(f"Compiled model: {compiled}", flush=True)
    print(f"Native certificate: {certificate_path}", flush=True)
    print("Acceptance covers extraction and Lean compilation only; not full-core extraction "
          "or a correspondence theorem.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
