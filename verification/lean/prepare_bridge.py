"""Seed the pinned Lean bridge dependencies without building mathlib from source.

* Every mathlib-ecosystem package in `lake-manifest.json` is cloned from the checkout
  of the already-built patched Aeneas Lean library, at exactly the manifest revision.
* LaCaDiLE is cloned from a local checkout (the upstream repository is not publicly
  clonable) and pinned to the manifest revision; its origin is reset to the canonical URL.
* `lake exe cache get` then unpacks the prebuilt mathlib oleans for exactly the Mathlib
  modules imported by Aeneas and LaCaDiLECore (offline when ~/.cache/mathlib has them).

The Aeneas library itself is required by path and is never written by this script.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
IMPORT = re.compile(r"^import\s+(Mathlib(?:\.[A-Za-z0-9_']+)+)\s*$", re.MULTILINE)


def run(command: list[str], cwd: Path, *, capture: bool = False) -> str:
    print("+ " + shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def mathlib_imports(*roots: Path) -> list[str]:
    modules: set[str] = set()
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("*.lean"))
        for path in files:
            modules.update(IMPORT.findall(path.read_text(encoding="utf-8")))
    return sorted(modules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aeneas-lean-dir", type=Path,
                        default=HERE / "../../.work/aeneas-verification/aeneas/backends/lean",
                        help="Built patched Aeneas backends/lean (required by path)")
    parser.add_argument("--lacadile-clone", type=Path, required=True,
                        help="Local LaCaDiLE git checkout containing the pinned revision")
    args = parser.parse_args()
    aeneas = args.aeneas_lean_dir.expanduser().resolve()
    lacadile = args.lacadile_clone.expanduser().resolve()
    if not (aeneas / "Aeneas.lean").is_file():
        parser.error(f"Aeneas Lean library is missing: {aeneas}")

    manifest = json.loads((HERE / "lake-manifest.json").read_text(encoding="utf-8"))
    packages = HERE / manifest["packagesDir"]
    packages.mkdir(parents=True, exist_ok=True)
    # Lake materializes locked path dependencies from the manifest or from this
    # per-checkout override file, never from `-K`; record the selected library here.
    override = {"version": manifest["version"], "packages": [{
        "type": "path", "scope": "", "name": "aeneas", "manifestFile": "lake-manifest.json",
        "inherited": False, "dir": str(aeneas), "configFile": "lakefile.lean"}]}
    (HERE / ".lake" / "package-overrides.json").write_text(
        json.dumps(override, indent=1) + "\n", encoding="utf-8")
    print(f"Aeneas library override: {aeneas}", flush=True)
    for entry in manifest["packages"]:
        if entry["type"] != "git":
            continue
        name, url, rev = entry["name"], entry["url"], entry["rev"]
        destination = packages / name
        source = lacadile if name == "LaCaDiLE" else aeneas / ".lake" / "packages" / name
        if not destination.exists():
            run(["git", "clone", "--quiet", "--no-checkout", str(source), str(destination)], HERE)
            run(["git", "remote", "set-url", "origin", url], destination)
            run(["git", "checkout", "--quiet", "--detach", rev], destination)
        head = run(["git", "rev-parse", "HEAD"], destination, capture=True)
        if head != rev:
            raise SystemExit(f"{name}: checkout is at {head}, manifest pins {rev}")

    modules = mathlib_imports(aeneas / "Aeneas.lean", aeneas / "Aeneas",
                              aeneas / "AeneasMeta.lean", aeneas / "AeneasMeta",
                              packages / "LaCaDiLE" / "lean" / "LaCaDiLECore")
    environment = os.environ.copy()
    print("+ lake exe cache get <" + str(len(modules)) + " Mathlib modules>", flush=True)
    subprocess.run(["lake", "exe", "cache", "get", *modules], cwd=HERE, env=environment, check=True)
    print("Dependencies seeded; run `lake build` in " + str(HERE))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
