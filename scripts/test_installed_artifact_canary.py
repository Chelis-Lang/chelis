"""Fail-closed installed-artifact transport tests; no compiler build required."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import tempfile
import unittest

from scripts import installed_artifact_canary as canary


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def archive(self, entries: list[tuple[str, bytes | None]]) -> Path:
        path = self.root / "chelis-v0.18.6-darwin-arm64.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, data in entries:
                member = tarfile.TarInfo(name)
                if data is None:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "../../outside"
                    archive.addfile(member)
                else:
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        return path

    def test_checksums_bind_the_exact_filename_and_bytes(self) -> None:
        path = self.root / "asset"
        path.write_bytes(b"real bytes")
        sidecar = self.root / "asset.sha256"
        sidecar.write_text(f"{canary.digest(path)}  asset\n")
        canary.verify_sidecar(path)
        for line in (f"{canary.digest(path)}  other\n", "0" * 64 + "  asset\n",
                     f"{canary.digest(path)}  asset\nextra\n"):
            with self.subTest(line=line):
                sidecar.write_text(line)
                with self.assertRaises(ValueError):
                    canary.verify_sidecar(path)

    def test_archive_inventory_rejects_duplicates_links_and_escape(self) -> None:
        entries = [("stage/bin/chelis", b"compiler")]
        for extra in [("stage/bin/chelis", b"other"),
                      ("stage/lib/runtime", None), ("../escape", b"x"),
                      ("/absolute", b"x"), ("other/bin/chelis", b"other")]:
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    canary.archive_inventory(self.archive(entries + [extra]))

    def test_installed_inventory_cannot_substitute_runtime_or_header(self) -> None:
        entries = [("stage/" + name, name.encode()) for name in canary.REQUIRED_FILES]
        inventory = canary.archive_inventory(self.archive(entries))
        installed = self.root / "installed"
        for name in canary.REQUIRED_FILES:
            path = installed / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        canary.verify_installed(installed, inventory)
        for name in ("bin/chelis", "lib/libchelis_runtime.a", "include/chelis_runtime.h"):
            with self.subTest(name=name):
                path = installed / name
                path.write_bytes(b"swapped")
                with self.assertRaises(ValueError):
                    canary.verify_installed(installed, inventory)
                path.write_bytes(name.encode())

    def test_missing_runtime_is_not_an_installable_inventory(self) -> None:
        inventory = canary.archive_inventory(self.archive([("stage/bin/chelis", b"x")]))
        with self.assertRaises(ValueError):
            canary.require_runtime_inventory(inventory)

    def test_staged_runtime_must_be_the_shipped_sealed_runtime(self) -> None:
        entries = [("stage/" + name, name.encode()) for name in canary.REQUIRED_FILES]
        inventory = canary.archive_inventory(self.archive(entries))
        output = self.root / "generated"
        output.mkdir()
        for name in ("lib/libchelis_runtime.a", *canary.HEADERS):
            (output / Path(name).name).write_bytes(name.encode())
        receipt = output / "chelis_runtime.receipt.json"
        sealed = {"schema": "chelis-runtime-staging/1", "archive": "libchelis_runtime.a",
                  "archive_sha256": inventory["lib/libchelis_runtime.a"], "mode": "sealed",
                  "headers": {}, "chelis_version": "0.18.6"}
        receipt.write_text(json.dumps(sealed))
        canary.require_staged_runtime(output, inventory)
        for name, change in (
            ("development build", {"mode": "development"}),
            ("receipt for other bytes", {"archive_sha256": "0" * 64}),
            ("other schema", {"schema": "chelis-runtime-staging/2"}),
        ):
            with self.subTest(case=name):
                receipt.write_text(json.dumps({**sealed, **change}))
                with self.assertRaisesRegex(ValueError, "sealed runtime"):
                    canary.require_staged_runtime(output, inventory)
        for name, text in (("malformed", "{"), ("not an object", "[]")):
            with self.subTest(case=name):
                receipt.write_text(text)
                with self.assertRaises(ValueError):
                    canary.require_staged_runtime(output, inventory)
        receipt.unlink()
        with self.assertRaisesRegex(ValueError, "unreadable staging receipt"):
            canary.require_staged_runtime(output, inventory)
        receipt.write_text(json.dumps(sealed))
        (output / "libchelis_runtime.a").write_bytes(b"swapped")
        with self.assertRaisesRegex(ValueError, "different runtime archive"):
            canary.require_staged_runtime(output, inventory)

    def test_dev_root_mapping_preserves_every_original_payload(self) -> None:
        original = self.archive([("chelis-dev-abcd1234/bin/chelis", b"compiler"),
                                 ("chelis-dev-abcd1234/lib/runtime", b"runtime")])
        mapped = self.root / "mapped.tar.gz"
        canary.installer_archive(original, mapped, candidate=True,
                                 version="0.18.6", slug="darwin-arm64")
        self.assertEqual(canary.archive_inventory(mapped), canary.archive_inventory(original))
        with tarfile.open(mapped) as archive:
            self.assertEqual(archive.getnames(), [
                "chelis-v0.18.6-darwin-arm64/bin/chelis",
                "chelis-v0.18.6-darwin-arm64/lib/runtime"])
        released = self.root / "released.tar.gz"
        canary.installer_archive(original, released, candidate=False,
                                 version="0.18.6", slug="darwin-arm64")
        self.assertEqual(original.read_bytes(), released.read_bytes())

    def test_candidate_mapping_is_explicit_and_source_bound(self) -> None:
        sha = "abcd1234" + "0" * 32
        self.assertEqual(canary.archive_name("0.18.6", "dev-abcd1234", sha,
                                             "darwin-arm64"),
                         "chelis-dev-abcd1234-darwin-arm64.tar.gz")
        self.assertEqual(canary.archive_name("0.18.6", "v0.18.6", sha,
                                             "linux-x86_64"),
                         "chelis-v0.18.6-linux-x86_64.tar.gz")
        for version, label, source, slug in [
            ("0.18.6", "dev-ffffffff", sha, "darwin-arm64"),
            ("0.18.6", "v0.18.5", sha, "darwin-arm64"),
            ("../bad", "v0.18.6", sha, "darwin-arm64"),
            ("0.18.6", "v0.18.6", "unknown", "darwin-arm64"),
            ("0.18.6", "v0.18.6", sha, "other"),
        ]:
            with self.subTest(label=label, version=version, source=source, slug=slug):
                with self.assertRaises(ValueError):
                    canary.archive_name(version, label, source, slug)


class ProcessTests(unittest.TestCase):
    def test_native_controls_keep_the_original_comparer(self) -> None:
        driver = canary.fixture_module().DRIVER_C
        for target in ("last-coordinate", "input"):
            changed = canary.corrupt_driver(driver, target)
            self.assertIn("memcmp(output_view.data, expected_bits", changed)
            self.assertIn("changed a borrowed input", changed)
            self.assertEqual(changed.count("fault_guard ="), 1)
        with self.assertRaises(ValueError):
            canary.corrupt_driver("missing insertion point", "input")

    def test_real_exit_signal_and_timeout_never_count_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            records: list[dict] = []
            runner = canary.Runner(root, dict(os.environ), records)
            runner.run("success", [sys.executable, "-c", "print('ok')"])
            self.assertEqual(records[-1]["status"], "passed")
            for name, source in [
                ("exit", "raise SystemExit(7)"),
                ("signal", f"import os; os.kill(os.getpid(), {signal.SIGTERM})"),
                ("timeout", "import time; time.sleep(10)"),
            ]:
                with self.subTest(name=name):
                    with self.assertRaises(ValueError):
                        runner.run(name, [sys.executable, "-c", source], timeout=0.1)
                    self.assertEqual(records[-1]["status"], "failed")
                    self.assertTrue((root / records[-1]["stderr"]).is_file())

    def test_expected_diagnostic_does_not_excuse_wrong_exit_or_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runner = canary.Runner(Path(raw), dict(os.environ), [])
            for code in (0, 7):
                with self.subTest(code=code), self.assertRaises(ValueError):
                    runner.run(f"wrong-{code}", [sys.executable, "-c",
                        f"import sys; print('expected', file=sys.stderr); sys.exit({code})"],
                        expected_failure="expected")
            runner.run("right-negative", [sys.executable, "-c",
                "import sys; print('expected', file=sys.stderr); sys.exit(1)"],
                expected_failure="expected")


class WorkflowTests(unittest.TestCase):
    def test_release_waits_for_downloaded_two_platform_canary(self) -> None:
        text = (canary.ROOT / ".github/workflows/release.yml").read_text()
        gate = text.split("  installed-artifact-canary:\n", 1)[1].split("  publish-release:\n", 1)[0]
        publish = text.split("  publish-release:\n", 1)[1]
        self.assertIn("installed-artifact-canary]", publish)
        self.assertIn("github.event_name == 'push'", publish)
        self.assertIn("pattern: chelis-*", publish)
        for slug in ("linux-x86_64", "darwin-arm64"):
            self.assertIn(f"slug: {slug}", gate)
        self.assertEqual(gate.count("uses: actions/download-artifact@v7"), 2)
        self.assertIn("scripts/installed_artifact_canary.py", gate)
        self.assertIn("if: always()", gate)
        self.assertIn("if-no-files-found: error", gate)


if __name__ == "__main__":
    unittest.main()
