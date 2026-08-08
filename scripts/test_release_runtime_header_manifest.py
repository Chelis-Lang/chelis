"""Release-tarball smoke tests for the public runtime-header closure.

Both release tarballs are staged by the Nix release derivation from the
``publicRuntimeHeaders`` list in ``nix/contracts.nix``. A public header may
include a generated sibling (currently ``chelis_runtime_dtype.h``), so a
manifest that omits a dependency produces a tarball that builds in-tree and
fails only after installation. These tests derive the local include closure
from the headers themselves, require the Nix manifest to ship it, require
that no shell staging block returns to the workflow, and preprocess the
manifest in an isolated include directory as an installed consumer would
see it.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
NIX_CONTRACTS = REPO_ROOT / "nix" / "contracts.nix"
RUNTIME_INCLUDE = REPO_ROOT / "crates" / "chelis-runtime" / "include"
STAGE_NAME = "      - name: Stage tarball contents\n"
LOCAL_INCLUDE = re.compile(r'^\s*#include\s+"(chelis_[^"]+\.h)"', re.MULTILINE)
COPY_HEADER = re.compile(
    r"cp crates/chelis-runtime/include/(chelis_[^\s/]+\.h) "
    r'"\$staging/include/"'
)


def release_header_manifests() -> list[set[str]]:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    manifests: list[set[str]] = []
    for segment in workflow.split(STAGE_NAME)[1:]:
        block = segment.split("\n      - name:", 1)[0]
        manifests.append(set(COPY_HEADER.findall(block)))
    return manifests


def nix_contract_manifest() -> set[str]:
    text = NIX_CONTRACTS.read_text(encoding="utf-8")
    match = re.search(r"publicRuntimeHeaders = \[(.*?)\];", text, re.DOTALL)
    if match is None:
        return set()
    return set(re.findall(r'"(chelis_[^"]+\.h)"', match.group(1)))


def local_dependency_closure(headers: set[str]) -> set[str]:
    closure = set(headers)
    pending = list(headers)
    while pending:
        header = pending.pop()
        source = (RUNTIME_INCLUDE / header).read_text(encoding="utf-8")
        for dependency in LOCAL_INCLUDE.findall(source):
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return closure


class ReleaseRuntimeHeaderManifestTests(unittest.TestCase):
    def test_every_release_tarball_stages_the_transitive_header_closure(self) -> None:
        workflow_manifests = release_header_manifests()
        self.assertEqual(
            len(workflow_manifests),
            0,
            "release.yml must stage no tarball in the workflow; both tarballs "
            "come from the Nix release derivation",
        )
        nix_manifest = nix_contract_manifest()
        self.assertTrue(
            nix_manifest,
            "nix/contracts.nix must record the publicRuntimeHeaders manifest",
        )
        required = local_dependency_closure(nix_manifest)
        self.assertEqual(
            nix_manifest,
            required,
            "the nix contracts release manifest omits dependent runtime headers",
        )

    def test_staged_manifest_preprocesses_as_an_installed_consumer(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("no host C compiler available")
        manifest = nix_contract_manifest()
        self.assertTrue(manifest, "nix/contracts.nix has no runtime-header manifest")
        with tempfile.TemporaryDirectory() as raw_dir:
            include_dir = Path(raw_dir) / "include"
            include_dir.mkdir()
            for header in manifest:
                shutil.copy2(RUNTIME_INCLUDE / header, include_dir / header)
            completed = subprocess.run(
                [compiler, "-fsyntax-only", "-x", "c", "-I", str(include_dir), "-"],
                input='#include "chelis_runtime.h"\nint main(void) { return 0; }\n',
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                "the staged runtime headers are not self-contained after installation:\n"
                + completed.stderr,
            )


if __name__ == "__main__":
    unittest.main()
