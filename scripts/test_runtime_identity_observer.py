"""Filesystem/process failures must not publish successful identity evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import runtime_identity_observer as observer
import runtime_identity_build as driver

HELPER = os.environ.get("CHELIS_IDENTITY_TEST_HELPER")


class ObservationFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {"CHELIS_IDENTITY_PROTOCOL": "1", "CHELIS_IDENTITY_STATE": str(self.root / "state"), "CHELIS_IDENTITY_BACKEND": "nix"}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        (self.root / "state").mkdir()
        if HELPER:
            os.environ["CHELIS_IDENTITY_HELPER"] = HELPER
            os.environ["CHELIS_IDENTITY_PYTHON"] = sys.executable

    def test_receipt_structural_corruption_is_rejected(self):
        path = self.root / "receipt.json"
        observer.atomic(path, {"protocol": 1, "outputs": [], "errors": []})
        value = json.loads(path.read_text())
        value["errors"] = ["an unobserved required native input"]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(observer.ObservationError, "corrupt"):
            observer.load(path)

    def test_changed_output_is_rejected_before_derivation(self):
        artifact = self.root / "libdep.rlib"
        artifact.write_bytes(b"compiler output")
        receipt = {"protocol": 1, "errors": [], "outputs": [{"path": str(artifact), "digest": observer.digest(artifact.read_bytes())}]}
        artifact.write_bytes(b"replaced compiler output")
        with self.assertRaisesRegex(observer.ObservationError, "changed compiler output"):
            observer.check_receipt(receipt)

    def test_debug_directory_cannot_supply_a_native_output_binding(self):
        artifact = self.root / "program"
        artifact.write_bytes(b"native compiler output")
        symbols = self.root / "program.dSYM"
        symbols.mkdir()
        (symbols / "symbols").write_bytes(b"debug information")
        receipt = self.root / "receipt.json"
        observer.atomic(receipt, {"protocol": 1, "errors": [], "outputs": [
            {"path": str(artifact), "digest": observer.digest(artifact.read_bytes())}]})
        observer.atomic(observer.binding(artifact), {"artifact": str(artifact), "observation": str(receipt)})
        event = {"filenames": [str(artifact), str(symbols)]}
        self.assertEqual(driver.event_receipt(event, self.root / "state"), str(receipt))
        artifact.write_bytes(b"unobserved replacement")
        self.assertIsNone(driver.event_receipt(event, self.root / "state"))
        artifact.unlink()
        with self.assertRaises(FileNotFoundError):
            driver.event_receipt(event, self.root / "state")
        self.assertIsNone(driver.event_receipt({"filenames": [str(symbols)]}, self.root / "state"))

    def test_missing_output_is_not_a_cache_hit(self):
        receipt = {"protocol": 1, "errors": [], "outputs": [{"path": str(self.root / "missing.rlib"), "digest": "0" * 64}]}
        with self.assertRaises(FileNotFoundError):
            observer.check_receipt(receipt)

    def test_binding_cannot_substitute_an_unobserved_output(self):
        receipt = self.root / "receipt.json"
        observer.atomic(receipt, {"protocol": 1, "outputs": [{"path": str(self.root / "real.rlib"), "digest": "0" * 64}], "errors": []})
        dependencies = self.root / "dependencies.json"
        dependencies.write_text(json.dumps([{"artifact": str(self.root / "other.rlib"), "observation": str(receipt)}]))
        os.environ["CHELIS_IDENTITY_DEPENDENCIES"] = str(dependencies)
        with self.assertRaisesRegex(observer.ObservationError, "does not name an observed output"):
            observer.import_dependencies()
        self.assertFalse(observer.binding(self.root / "other.rlib").exists())

    def test_failed_publication_does_not_replace_existing_receipt(self):
        parent = self.root / "not-a-directory"
        parent.write_text("preserve me")
        with self.assertRaises(OSError):
            observer.atomic(parent / "receipt.json", {"protocol": 1, "outputs": []})
        self.assertEqual(parent.read_text(), "preserve me")

    def test_missing_inventory_root_is_an_error(self):
        with self.assertRaises(OSError):
            observer.enumerate_files(self.root / "missing")

    def test_required_probe_failure_is_not_an_empty_observation(self):
        with self.assertRaisesRegex(observer.ObservationError, "probe failed"):
            observer.probe(["/bin/sh", "-c", "printf 'probe denied' >&2; exit 17"])

    def test_failed_build_script_does_not_publish_execution(self):
        script = self.root / "build-script"
        script.write_text("#!/bin/sh\nexit 27\n")
        script.chmod(0o755)
        receipt = self.root / "receipt.json"
        observer.atomic(receipt, {"protocol": 1, "outputs": [{"path": str(script), "digest": observer.digest(script.read_bytes())}], "errors": []})
        observer.atomic(observer.binding(script), {"artifact": str(script), "observation": str(receipt)})
        os.environ["OUT_DIR"] = str(self.root / "out")
        self.assertEqual(observer.build_execution(str(script), []), 27)
        self.assertFalse(observer.execution_path(os.environ["OUT_DIR"]).exists())

    def test_artifact_event_cannot_merge_different_units(self):
        paths = [str(self.root / "one.rlib"), str(self.root / "two.rlib")]
        for number, path in enumerate(paths):
            Path(path).write_bytes(bytes([number]))
            receipt = self.root / f"receipt-{number}.json"
            observer.atomic(receipt, {"outputs": [{"path": path, "digest": observer.digest(Path(path).read_bytes())}]})
            observer.atomic(observer.binding(path), {"artifact": path, "observation": str(receipt)})
        with self.assertRaisesRegex(observer.ObservationError, "different compilation receipts"):
            driver.event_receipt({"filenames": paths}, self.root / "state")

    def test_cargo_launcher_rejects_recursive_real_cargo(self):
        destination = self.root / "bin"
        with self.assertRaisesRegex(observer.ObservationError, "cannot target itself"):
            driver.install_cargo_launcher(destination, real_cargo=destination / "cargo", python="/usr/bin/python3")


    def test_build_script_uplift_binds_exact_compiler_bytes(self):
        directory = self.root / "build"
        directory.mkdir()
        original = directory / "build_script_build-hash"
        uplift = directory / "build-script-build"
        original.write_bytes(b"compiled script")
        shutil.copyfile(original, uplift)
        receipt = self.root / "build-receipt.json"
        observer.atomic(receipt, {"outputs": [{"path": str(original), "digest": observer.digest(original.read_bytes())}]})
        observer.atomic(self.root / "state/output-digests" / observer.digest(original.read_bytes()) / "binding.json", {"observation": str(receipt)})
        event = {"filenames": [str(uplift)], "target": {"kind": ["custom-build"]}}
        self.assertEqual(driver.event_receipt(event, self.root / "state"), str(receipt))
        uplift.write_bytes(b"another script")
        self.assertIsNone(driver.event_receipt(event, self.root / "state"))

    def test_builder_move_retains_only_the_exact_observed_output(self):
        original, installed = self.root / "target_name", self.root / "target-name"
        original.write_bytes(b"compiled binary")
        receipt = self.root / "binary-receipt.json"
        observer.atomic(receipt, {"protocol": 1, "outputs": [
            {"path": str(original), "digest": observer.digest(original.read_bytes())}]})
        observer.atomic(observer.binding(original), {"artifact": str(original), "observation": str(receipt)})
        original.rename(installed)
        observer.observe_output_move(original, installed)
        self.assertFalse(observer.binding(original).exists())
        self.assertEqual(observer.receipt_for(installed), receipt)
        self.assertEqual(observer.load(receipt)["outputs"], [
            {"path": str(installed), "digest": observer.digest(installed.read_bytes())}])

    def test_builder_move_refuses_replaced_compiler_bytes(self):
        original, installed = self.root / "target_name", self.root / "target-name"
        receipt = self.root / "binary-receipt.json"
        observer.atomic(receipt, {"protocol": 1, "outputs": [
            {"path": str(original), "digest": observer.digest(b"compiled binary")}]})
        observer.atomic(observer.binding(original), {"artifact": str(original), "observation": str(receipt)})
        installed.write_bytes(b"different binary")
        with self.assertRaisesRegex(observer.ObservationError, "bytes differ"):
            observer.observe_output_move(original, installed)
        self.assertFalse(observer.binding(installed).exists())
        self.assertEqual(observer.load(receipt)["outputs"][0]["path"], str(original))

    def test_installed_metadata_survives_removed_build_directory(self):
        old, new = self.root / "build-out", self.root / "installed-out"
        old.mkdir()
        (old / "data").write_bytes(b"generated dependency input")
        shutil.copytree(old, new)
        metadata = observer.relocate_build_metadata({"root": str(old), "data": str(old / "data"), "different": str(old) + "-other"}, old, new)
        shutil.rmtree(old)
        self.assertEqual(Path(metadata["data"]).read_bytes(), b"generated dependency input")
        self.assertEqual(metadata["different"], str(old) + "-other")
        (new / "data").unlink()
        with self.assertRaisesRegex(observer.ObservationError, "path is absent"):
            observer.relocate_build_metadata({"data": str(old / "data")}, old, new)

    @unittest.skipUnless(HELPER, "run through the adapter Cargo test for core-backed installation")
    def test_installation_does_not_publish_transient_compiler_probes(self):
        source = self.root / "source"
        source.mkdir()
        (source / "lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
        roots = [observer.inventory(source, "package", ["lib.rs"])]
        required, captured = observer.capture(roots)
        built = self.root / "target/lib/libfixture.rlib"
        built.parent.mkdir(parents=True)
        built.write_bytes(b"observed library output")
        installed = self.root / "installed"
        (installed / "lib").mkdir(parents=True)
        shutil.copyfile(built, installed / "lib" / built.name)
        receipt = {"protocol": 1, "errors": [], "dependencies": [], "roots": roots,
                   "required_inputs": required, "captured": captured,
                   "unit": {"package": {"name": "fixture"}}, "cwd": str(self.root),
                   "outputs": [{"path": str(built), "digest": observer.digest(built.read_bytes())}]}
        observer.atomic(self.root / "state/receipts/library.json", receipt)
        # A successful build-script compiler probe was already discarded. It is
        # neither a published artifact nor an input to the installed library.
        probe = self.root / "target/lib/fixture.out/probe/libfixture.rmeta"
        observer.atomic(self.root / "state/receipts/probe.json",
                        {**receipt, "outputs": [{"path": str(probe), "digest": "0" * 64}]})
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            observer.install_observations(["--state", str(self.root / "state"), "--lib-dir", str(installed)])
        finally:
            os.chdir(previous)
        index = observer.load(installed / "chelis-runtime-identity-observations.json")
        self.assertEqual({item["artifact"] for item in index}, {str(installed / "lib" / built.name)})
        published = observer.load(index[0]["observation"])
        self.assertEqual(Path(published["outputs"][0]["path"]).read_bytes(), built.read_bytes())

    @unittest.skipUnless(HELPER, "run through the adapter Cargo test for the native wrapper")
    def test_native_adapter_preserves_inherited_descriptors(self):
        reader, writer = os.pipe()
        try:
            os.write(writer, b"jobserver token")
            os.close(writer)
            writer = None
            result = subprocess.run([HELPER, "observe-rustc", sys.executable, "-c",
                                     f"import os,sys; sys.stdout.buffer.write(os.read({reader}, 32))"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(reader,), check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, b"jobserver token")
        finally:
            os.close(reader)
            if writer is not None:
                os.close(writer)
    @unittest.skipUnless(HELPER, "run through the adapter Cargo test for core-backed cache checks")
    def test_warm_receipt_rejects_new_production_membership_but_not_prose(self):
        source = self.root / "source"
        source.mkdir()
        (source / "lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
        roots = [observer.inventory(source, "package", ["lib.rs"])]
        required, captured = observer.capture(roots)
        receipt = {"protocol": 1, "errors": [], "outputs": [], "roots": roots, "required_inputs": required, "captured": captured, "unit": {"package": {"name": "fixture"}}}
        (source / "README.md").write_text("prose is not production source")
        observer.check_receipt(receipt)
        (source / "new.rs").write_text("pub fn added() {}\n")
        with self.assertRaisesRegex(observer.ObservationError, "stale source/input inventory"):
            observer.check_receipt(receipt)

    @unittest.skipUnless(HELPER, "run through the adapter Cargo test for core-backed cache checks")
    def test_cargo_environment_projection_is_not_a_changed_input(self):
        source = self.root / "source"
        source.mkdir()
        (source / "lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
        roots = [observer.inventory(source, "package", ["lib.rs"])]
        required, captured = observer.capture(roots)
        environment_path = self.root / "invocation-environment.json"
        original = {"RUSTFLAGS": observer.digest(b"-C debug-assertions=no")}
        observer.atomic(environment_path, original)
        os.environ.update({"CHELIS_IDENTITY_BACKEND": "cargo",
                           "CHELIS_IDENTITY_INVOCATION_ENVIRONMENT": str(environment_path)})
        # Cargo removes RUSTFLAGS from the build-script process, but rustc and
        # the driver can still receive the original value. This is one input,
        # not a stale receipt caused by comparing different process contexts.
        receipt = {"protocol": 1, "errors": [], "outputs": [], "roots": roots,
                   "required_inputs": required, "captured": captured,
                   "unit": {"package": {"name": "fixture"}},
                   "ambient_environment": [{"name": "RUSTFLAGS", "digest": original["RUSTFLAGS"]}]}
        observer.check_receipt(receipt)
        os.environ["RUSTFLAGS"] = "-C debug-assertions=no"
        observer.check_receipt(receipt)
        observer.atomic(environment_path, {"RUSTFLAGS": observer.digest(b"-C debug-assertions=yes")})
        with self.assertRaisesRegex(observer.ObservationError, "changed build input environment"):
            observer.check_receipt(receipt)

    @unittest.skipUnless(HELPER, "run through the adapter Cargo test for core-backed cache checks")
    def test_warm_receipt_rejects_content_and_declared_environment_changes(self):
        source = self.root / "source"
        source.mkdir()
        path = source / "lib.rs"
        path.write_text("pub const N: u8 = 1;\n")
        roots = [observer.inventory(source, "package", ["lib.rs"])]
        required, captured = observer.capture(roots)
        receipt = {"protocol": 1, "errors": [], "outputs": [], "roots": roots, "required_inputs": required, "captured": captured, "unit": {"package": {"name": "fixture"}}, "ambient_environment": [{"name": "FIXTURE_MODE", "digest": None}]}
        os.environ["FIXTURE_MODE"] = "changed"
        with self.assertRaisesRegex(observer.ObservationError, "changed build input environment"):
            observer.check_receipt(receipt)
        del os.environ["FIXTURE_MODE"]
        path.write_text("pub const N: u8 = 2;\n")
        with self.assertRaisesRegex(observer.ObservationError, "stale source/input inventory"):
            observer.check_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
