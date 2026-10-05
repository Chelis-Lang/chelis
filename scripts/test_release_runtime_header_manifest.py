"""Release-tarball smoke tests for the public runtime-header closure.

The release workflow stages headers one by one. A public header may include a
generated sibling (currently ``chelis_runtime_dtype.h``), so a manifest that
copies the root but omits a dependency produces a tarball that builds in-tree
and fails only after installation. These tests derive the local include
closure from the headers themselves and require every release staging block to
ship it. The first manifest is also copied into an isolated include directory
and preprocessed as an installed consumer would see it.
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
RUNTIME_INCLUDE = REPO_ROOT / "crates" / "chelis-runtime" / "include"
STAGE_NAME = "      - name: Stage tarball contents\n"
LOCAL_INCLUDE = re.compile(r'^\s*#include\s+"(chelis_[^"]+\.h)"', re.MULTILINE)
# Release tarballs ship the headers of the runtime their chelis carries, taken
# from `chelis runtime export` (spec/08-backends.md §2.1).
COPY_HEADER = re.compile(
    r'cp "\$runtime_export/(chelis_[^\s/"]+\.h)" '
    r'"\$staging/include/"'
)


def release_header_manifests() -> list[set[str]]:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    manifests: list[set[str]] = []
    for segment in workflow.split(STAGE_NAME)[1:]:
        block = segment.split("\n      - name:", 1)[0]
        manifests.append(set(COPY_HEADER.findall(block)))
    return manifests


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
        manifests = release_header_manifests()
        self.assertEqual(
            len(manifests),
            4,
            "release.yml must expose all four platform tarball manifests to this test",
        )
        for index, manifest in enumerate(manifests):
            self.assertIn(
                "chelis_runtime.h",
                manifest,
                f"release tarball staging block {index + 1} stages no runtime headers",
            )
            required = local_dependency_closure(manifest)
            self.assertEqual(
                manifest,
                required,
                f"release tarball staging block {index + 1} omits dependent runtime headers",
            )

    def test_staged_manifest_preprocesses_as_an_installed_consumer(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("no host C compiler available")
        manifests = release_header_manifests()
        self.assertTrue(manifests, "release.yml has no runtime-header manifest")
        with tempfile.TemporaryDirectory() as raw_dir:
            include_dir = Path(raw_dir) / "include"
            include_dir.mkdir()
            for header in manifests[0]:
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
