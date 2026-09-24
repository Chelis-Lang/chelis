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
RUSTC = shutil.which(os.environ.get("RUSTC", "rustc"))


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

    def test_exact_binding_is_not_ambiguous_with_equal_historical_outputs(self):
        receipts = []
        artifacts = [self.root / "old.rlib", self.root / "current.rlib"]
        manifest = str(self.root / "Cargo.toml")
        for number, artifact in enumerate(artifacts):
            artifact.write_bytes(b"identical compiled bytes")
            receipt = self.root / f"receipt-{number}.json"
            receipts.append(str(receipt))
            checksum = observer.digest(artifact.read_bytes())
            observer.atomic(receipt, {"manifest_path": manifest, "outputs": [
                {"path": str(artifact), "digest": checksum}]})
            observer.atomic(observer.binding(artifact), {"artifact": str(artifact), "observation": str(receipt)})
            observer.atomic(self.root / "state/output-digests" / checksum / f"{number}.json", {"observation": str(receipt)})
        event = {"filenames": [str(artifacts[1])], "manifest_path": manifest}
        self.assertEqual(driver.event_receipt(event, self.root / "state"), receipts[1])
        event["filenames"].append(str(artifacts[0]))
        with self.assertRaisesRegex(observer.ObservationError, "different compilation receipts"):
            driver.event_receipt(event, self.root / "state")

    def test_artifact_event_must_match_observed_features_and_output_bytes(self):
        artifact = self.root / "libunit.rlib"
        artifact.write_bytes(b"observed compiler output")
        receipt = {
            "unit": {"features": ["build-script-cfg", "selected"]},
            "outputs": [
                {
                    "path": str(artifact),
                    "digest": observer.digest(artifact.read_bytes()),
                }
            ],
        }
        event = {
            "features": ["selected"],
            "filenames": [str(artifact)],
        }
        driver.validate_artifact_event(receipt, event)
        event["features"] = ["different"]
        with self.assertRaisesRegex(
            observer.ObservationError, "feature is absent"
        ):
            driver.validate_artifact_event(receipt, event)
        event["features"] = ["selected"]
        receipt["outputs"].append(
            {
                "path": str(self.root / "unreported.rlib"),
                "digest": observer.digest(b"unreported compiler output"),
            }
        )
        with self.assertRaisesRegex(
            observer.ObservationError, "does not bind"
        ):
            driver.validate_artifact_event(receipt, event)

    def test_surface_producer_binds_exact_output_despite_unrelated_wrapper_inputs(self):
        original = self.root / "chelis-hash"
        uplift = self.root / "chelis"
        original.write_bytes(b"observed CLI")
        shutil.copyfile(original, uplift)
        checksum = observer.digest(original.read_bytes())
        receipt = self.root / "surface-receipt.json"
        value = {
            "protocol": 1,
            "role": "cli",
            "descriptor": {"schema_version": 1},
            "unit": {"target_name": "chelis", "features": ["default"]},
            "errors": [
                "CHELIS_IDENTITY_UNOBSERVABLE_NATIVE: native linker inputs "
                "outside the runtime closure"
            ],
            "outputs": [{"path": str(original), "digest": checksum}],
        }
        observer.atomic(receipt, value)
        observer.atomic(
            self.root / "state/output-digests" / checksum / "binding.json",
            {"observation": str(receipt)},
        )
        event = {
            "features": ["default"],
            "filenames": [str(uplift)],
            "target": {"kind": ["bin"], "name": "chelis"},
        }
        self.assertEqual(
            driver.event_receipt(event, self.root / "state"), str(receipt)
        )
        self.assertTrue(driver.surface_producer_receipt(value, "cli"))
        driver.validate_artifact_event(value, event, allow_surface_errors=True)
        with self.assertRaisesRegex(
            observer.ObservationError, "incomplete compiler observation"
        ):
            driver.validate_artifact_event(value, event)
        event["features"].append("changed")
        with self.assertRaisesRegex(observer.ObservationError, "feature is absent"):
            driver.validate_artifact_event(
                value, event, allow_surface_errors=True
            )

    def test_managed_build_resolves_one_compiler_from_the_workspace(self):
        workspace = self.root / "workspace"
        workspace.mkdir()
        pinned = self.root / "pinned" / "bin" / "rustc"
        pinned.parent.mkdir(parents=True)
        pinned.write_text("#!/bin/sh\n")
        pinned.chmod(0o755)
        proxies = self.root / "proxies"
        proxies.mkdir()
        # Like a rustup proxy, answer from the working directory's toolchain.
        proxy = proxies / "rustc"
        proxy.write_text(
            "#!/bin/sh\n"
            f'if [ "$(pwd -P)" = "{workspace.resolve()}" ]; then echo "{pinned.parent.parent}"; '
            f'else echo "{self.root / "registry-default"}"; fi\n'
        )
        proxy.chmod(0o755)
        environment = {"PATH": str(proxies)}
        self.assertEqual(
            driver.exact_compiler(environment, workspace, ["build"]), str(pinned)
        )
        environment["RUSTC"] = "/configured/rustc"
        self.assertEqual(
            driver.exact_compiler(environment, workspace, ["build"]),
            "/configured/rustc",
        )
        del environment["RUSTC"]
        proxy.write_text(f'#!/bin/sh\necho "{self.root / "missing"}"\n')
        with self.assertRaisesRegex(observer.ObservationError, "absent from its sysroot"):
            driver.exact_compiler(environment, workspace, ["build"])

    def test_cached_dependency_receipt_needs_no_synthetic_cargo_event(self):
        artifact = self.root / "libdependency.rlib"
        artifact.write_bytes(b"exact cached dependency")
        receipt = {
            "protocol": 1,
            "errors": [],
            "outputs": [
                {
                    "path": str(artifact),
                    "digest": observer.digest(artifact.read_bytes()),
                }
            ],
            "roots": [],
            "required_inputs": [],
            "captured": [],
            "ambient_environment": [],
        }
        os.environ["CHELIS_IDENTITY_BACKEND"] = "cargo"
        with (
            patch.object(observer, "capture", return_value=([], [])),
            patch.object(
                observer,
                "cargo_event",
                side_effect=AssertionError("cached dependencies have no Cargo event"),
            ),
        ):
            observer.check_receipt(receipt)

    @unittest.skipUnless(HELPER and RUSTC, "requires compiled adapter and native rustc")
    def test_mutating_explicit_sysroot_invalidates_cached_observation(self):
        subject = self.root / "subject"
        subject.mkdir()
        (subject / "Cargo.toml").write_text('[package]\nname="sysroot-probe"\nversion="0.0.0"\nedition="2021"\n[workspace]\n')
        source = subject / "lib.rs"
        source.write_text("pub fn width() -> usize { core::mem::size_of::<u64>() }\n")
        standard = Path(observer.probe([RUSTC, "--print=target-libdir"]).strip())
        sysroot = self.root / "custom-sysroot"
        libraries = sysroot / "lib/rustlib" / standard.parent.name / "lib"
        libraries.mkdir(parents=True)
        for original in standard.iterdir():
            (libraries / original.name).symlink_to(original, target_is_directory=original.is_dir())
        os.environ.update({"CARGO_MANIFEST_DIR": str(subject), "CARGO_PKG_NAME": "sysroot-probe",
                           "CARGO_PKG_VERSION": "0.0.0", "CHELIS_IDENTITY_PACKAGE_SOURCE": "path:.",
                           "CHELIS_IDENTITY_WORKSPACE": str(subject)})
        receipt = observer.collect_unit(RUSTC, [str(source), "--crate-name", "sysroot_probe",
                                               "--crate-type", "lib", "--sysroot=" + str(sysroot)])
        observer.check_receipt(receipt)
        added = libraries / "new-library.rlib"
        added.write_bytes(b"new sysroot inventory member")
        with self.assertRaisesRegex(observer.ObservationError, "stale source/input inventory"):
            observer.check_receipt(receipt)
        added.unlink()
        observer.check_receipt(receipt)
        core_libraries = list(libraries.glob("libcore-*.rlib"))
        self.assertEqual(len(core_libraries), 1)
        core_libraries[0].unlink()
        core_libraries[0].write_bytes(b"changed cached sysroot input")
        with self.assertRaisesRegex(observer.ObservationError, "stale source/input inventory"):
            observer.check_receipt(receipt)

    @unittest.skipUnless(HELPER and RUSTC, "requires compiled adapter and native rustc")
    def test_relocated_sysroot_and_nix_build_root_preserve_descriptor(self):
        standard = Path(observer.probe([RUSTC, "--print=target-libdir"]).strip())
        descriptors = []
        for label in ("original", "relocated"):
            build_root = self.root / ("build-" + label)
            subject = build_root / "source"
            subject.mkdir(parents=True)
            (subject / "Cargo.toml").write_text('[package]\nname="sysroot-probe"\nversion="0.0.0"\nedition="2021"\n[workspace]\n')
            (subject / "probe.h").write_text("#define SYSROOT_PROBE_WIDTH 8\n")
            source = subject / "lib.rs"
            source.write_text("pub fn width() -> usize { core::mem::size_of::<u64>() }\n")
            sysroot = self.root / ("toolchain-" + label)
            libraries = sysroot / "lib/rustlib" / standard.parent.name / "lib"
            libraries.mkdir(parents=True)
            for original in standard.iterdir():
                (libraries / original.name).symlink_to(original, target_is_directory=original.is_dir())
            os.environ.update({"CARGO_MANIFEST_DIR": str(subject), "CARGO_PKG_NAME": "sysroot-probe",
                               "CARGO_PKG_VERSION": "0.0.0", "CHELIS_IDENTITY_PACKAGE_SOURCE": "path:.",
                               "CHELIS_IDENTITY_WORKSPACE": str(subject), "NIX_BUILD_TOP": str(build_root),
                               "CHELIS_IDENTITY_PROVENANCE": "source-worktree"})
            receipt = observer.collect_unit(RUSTC, [
                str(source), "--crate-name", "sysroot_probe", "--crate-type", "lib",
                "--sysroot", str(sysroot), "--remap-path-prefix=" + str(build_root) + "=/",
                "--remap-path-prefix=" + str(sysroot) + "=/rustc"])
            receipt["profile"] = "debug"
            output = build_root / "record"
            output.mkdir()
            observer.emit_producer(receipt, "runtime", output)
            descriptors.append(receipt["descriptor"])
        self.assertEqual(descriptors[0], descriptors[1])

    @unittest.skipUnless(RUSTC, "requires native rustc")
    def test_metadata_only_binary_outputs_match_compiler_artifact_events(self):
        source = self.root / "main.rs"
        source.write_text("fn main() {}\n")
        arguments = [str(source), "--crate-name", "identity_check", "--crate-type", "bin",
                     "--emit=dep-info,metadata", "--out-dir", str(self.root), "-Cextra-filename=-check",
                     "--error-format=json", "--json=artifacts"]
        compiled = subprocess.run([RUSTC, *arguments], capture_output=True, text=True, check=True)
        events = [json.loads(line) for line in compiled.stderr.splitlines()]
        emitted = {event["artifact"] for event in events if event.get("emit") == "metadata"}
        self.assertEqual(set(observer.output_paths(RUSTC, arguments)), emitted)
        self.assertTrue(all(not observer.is_native_producer("cli", path) for path in emitted))

    def test_cargo_launcher_rejects_recursive_real_cargo(self):
        destination = self.root / "bin"
        with self.assertRaisesRegex(observer.ObservationError, "cannot target itself"):
            driver.install_cargo_launcher(destination, real_cargo=destination / "cargo", python="/usr/bin/python3")

    def test_cargo_launcher_imports_its_driver_under_safe_path(self):
        # Native census workers export PYTHONSAFEPATH=1 to their Cargo children.
        launcher = driver.install_cargo_launcher(self.root / "bin", real_cargo=shutil.which("true"), python=sys.executable)
        completed = subprocess.run([str(launcher), "--version"], env={"PATH": os.defpath, "PYTHONSAFEPATH": "1"},
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_driver_passes_cargo_options_through_unchanged(self):
        record = self.root / "argv"
        cargo = self.root / "real-cargo"
        cargo.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CHELIS_TEST_ARGV\"\n")
        cargo.chmod(0o755)
        os.environ.update({"CHELIS_IDENTITY_REAL_CARGO": str(cargo), "CHELIS_TEST_ARGV": str(record)})
        for arguments, forwarded in (
            (["--version"], ["--version"]),
            (["-V"], ["-V"]),
            (["--help"], ["--help"]),
            (["--cargo", str(cargo), "--", "--version"], ["--version"]),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(driver.main(arguments), 0)
                self.assertEqual(record.read_text().splitlines(), forwarded)

    def test_observed_build_script_runs_unchanged_without_managed_protocol(self):
        real = self.root / "build_script_build.identity-real"
        real.write_text('#!/bin/sh\necho "real $*"\n')
        helper = self.root / "helper"
        helper.write_text('#!/bin/sh\necho "observed $*"\n')
        launcher = self.root / "build_script_build"
        launcher.write_text(observer.build_script_launcher(helper, real))
        for path in (real, helper, launcher):
            path.chmod(0o755)
        unmanaged = subprocess.run([str(launcher), "argument"], env={"PATH": os.defpath},
                                   capture_output=True, text=True, check=True)
        self.assertEqual(unmanaged.stdout, "real argument\n")
        managed = subprocess.run([str(launcher), "argument"], env={"PATH": os.defpath, "CHELIS_IDENTITY_PROTOCOL": "1"},
                                 capture_output=True, text=True, check=True)
        self.assertEqual(managed.stdout, f"observed observe-build-script {real} argument\n")

    def test_inherited_out_dir_is_not_attributed_to_another_unit(self):
        package = self.root / "package"
        package.mkdir()
        (package / "Cargo.toml").write_text('[package]\nname = "subject"\nversion = "0.0.0"\n')
        out = self.root / "enclosing-test-out"
        os.environ.update({"CHELIS_IDENTITY_BACKEND": "cargo", "CARGO_MANIFEST_DIR": str(package), "OUT_DIR": str(out)})
        # Cargo supplies OUT_DIR only to a package with a build script.
        self.assertIsNone(observer.unit_out_dir("subject"))
        (package / "build.rs").write_text("fn main() {}\n")
        self.assertEqual(observer.unit_out_dir("subject"), str(out))
        # It never supplies a package's OUT_DIR to that package's build script.
        self.assertIsNone(observer.unit_out_dir("build_script_build"))

    def test_retained_record_is_not_newer_than_its_producer_declaration(self):
        # Cargo stamps a unit before this wrapper writes the bytes rustc then
        # includes; a newer included file would make every later build dirty.
        output = self.root / "out"
        output.mkdir()
        declaration = output / "chelis-runtime-identity-producer.json"
        declaration.write_text("{}")
        os.utime(declaration, ns=(1_000_000_000, 1_000_000_000))
        os.environ.update({"CHELIS_IDENTITY_PROVENANCE": "source-worktree", "CHELIS_IDENTITY_WORKSPACE": str(self.root)})
        derived = {"record": [1, 2], "provenance": [3], "descriptor": {}}
        with patch.object(observer, "graph_recipe", return_value=({}, [], [])), \
                patch.object(observer, "helper", return_value=derived):
            observer.emit_producer({}, "runtime", output)
        for name, content in (("chelis_runtime_identity.bin", b"\x01\x02"), ("chelis_runtime_identity_provenance.bin", b"\x03")):
            retained = output / name
            self.assertEqual(retained.read_bytes(), content)
            self.assertLessEqual(retained.stat().st_mtime_ns, declaration.stat().st_mtime_ns)

    def test_clippy_workspace_compiler_is_observed_only_for_workspace_members(self):
        member = self.root / "member"
        dependency = self.root / "dependency"
        member.mkdir()
        dependency.mkdir()
        for directory in (member, dependency):
            (directory / "Cargo.toml").write_text("[package]\nname='probe'\nversion='0.0.0'\n")
        metadata = self.root / "metadata.json"
        metadata.write_text(
            json.dumps(
                {
                    "workspace_members": ["member 0.0.0"],
                    "packages": [
                        {
                            "id": "member 0.0.0",
                            "manifest_path": str(member / "Cargo.toml"),
                        },
                        {
                            "id": "dependency 0.0.0",
                            "manifest_path": str(dependency / "Cargo.toml"),
                        },
                    ],
                }
            )
        )
        clippy = self.root / "clippy-driver"
        clippy.symlink_to(sys.executable)
        os.environ.update(
            {
                "CHELIS_IDENTITY_METADATA": str(metadata),
                "CHELIS_IDENTITY_WORKSPACE_WRAPPER": str(clippy),
                "CHELIS_IDENTITY_INNER_WRAPPER": "kache",
                "CARGO_MANIFEST_DIR": str(member),
            }
        )
        arguments = ["--crate-name", "probe"]
        with (
            patch.object(observer, "probe", return_value="clippy 0.1.98\n"),
            patch.object(
                observer, "transparent_wrapper", return_value="/tool/kache"
            ),
        ):
            command, workspace = observer.rustc_command("/tool/rustc", arguments)
            self.assertEqual(
                command,
                ["/tool/kache", str(clippy), "/tool/rustc", *arguments],
            )
            self.assertEqual(workspace["path"], str(clippy))
            self.assertTrue(
                workspace["identity"].endswith(
                    observer.digest(Path(sys.executable).read_bytes())
                )
            )
            os.environ["CARGO_MANIFEST_DIR"] = str(dependency)
            command, workspace = observer.rustc_command("/tool/rustc", arguments)
            self.assertEqual(command, ["/tool/kache", "/tool/rustc", *arguments])
            self.assertIsNone(workspace)


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
        ordinary_event = {
            "filenames": [str(uplift)],
            "target": {"kind": ["lib"], "name": "unrelated"},
        }
        self.assertIsNone(
            driver.event_receipt(ordinary_event, self.root / "state")
        )
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

    def test_local_unit_dependency_metadata_tracks_selected_inputs_and_membership(self):
        package = self.root / "package"
        for directory in ("src/nested", "tests", "docs"):
            (package / directory).mkdir(parents=True)
        for name in ("Cargo.toml", "src/lib.rs", "src/nested/mod.rs", "tests/case.rs", "docs/guide.md"):
            (package / name).write_text("")
        selected = ("Cargo.toml", "src/lib.rs", "src/nested/mod.rs")
        declarations = {"physical": str(self.root), "fixed": True, "inventory": {
            "logical_prefix": "workspace", "files": ["Cargo.toml"], "explicitly_required": ["Cargo.toml"], "class": "build"}}
        toolchain = {"physical": str(self.root / "sysroot"), "inventory": {
            "logical_prefix": "toolchain", "files": ["libstd.rlib"], "explicitly_required": ["libstd.rlib"], "class": "toolchain"}}
        generated = {"physical": str(self.root / "out"), "generated": True, "inventory": {
            "logical_prefix": "package/generated", "files": ["bindings.rs"], "explicitly_required": ["bindings.rs"], "class": "build"}}
        receipt = {"roots": [observer.inventory(package, "package", ["Cargo.toml"]), declarations, toolchain, generated],
                   "required_inputs": [{"logical_path": "package/" + name} for name in selected]
                   + [{"logical_path": path} for path in ("workspace/Cargo.toml", "toolchain/libstd.rlib", "package/generated/bindings.rs")]}
        dep_info = self.root / "unit.d"
        dep_info.write_text(f"{self.root}/libunit.rlib: src/lib.rs\n\nsrc/lib.rs:\n\n# env-dep:CARGO_PKG_NAME=unit\n")
        observer.track_inventory(receipt, dep_info)
        rule, *rest = dep_info.read_text().split("\n")
        # Cargo rebuilds when a listed file or directory changes: selected inputs,
        # the directories where one can be added or removed, and a retained
        # declaration file, never tests, toolchain or generated inputs.
        self.assertEqual(
            set(rule.split(": ", 1)[1].split()),
            {"src/lib.rs", str(self.root / "Cargo.toml"),
             *(str(package / name) for name in ("", "src", "src/nested", *selected))},
        )
        self.assertEqual(rest, ["", "src/lib.rs:", "", "# env-dep:CARGO_PKG_NAME=unit", ""])

    @unittest.skipUnless(HELPER and RUSTC, "requires compiled adapter and native rustc")
    def test_only_producers_retain_workspace_declarations(self):
        workspace = self.root / "workspace"
        package = workspace / "member"
        package.mkdir(parents=True)
        manifest = workspace / "Cargo.toml"
        manifest.write_text('[workspace]\nmembers = ["member"]\n')
        (package / "Cargo.toml").write_text('[package]\nname = "member"\nversion = "0.0.0"\nedition = "2021"\n')
        source = package / "lib.rs"
        source.write_text("pub fn value() -> u8 { 1 }\n")
        os.environ.update({"CARGO_MANIFEST_DIR": str(package), "CARGO_PKG_NAME": "member", "CARGO_PKG_VERSION": "0.0.0",
                           "CHELIS_IDENTITY_PACKAGE_SOURCE": "path:member", "CHELIS_IDENTITY_WORKSPACE": str(workspace)})
        arguments = [str(source), "--crate-name", "member", "--crate-type", "lib"]
        dependency = observer.collect_unit(RUSTC, arguments)
        os.environ["CHELIS_IDENTITY_ROLE"] = "runtime"
        producer = observer.collect_unit(RUSTC, arguments)
        # Cargo does not rebuild every unit for a workspace edit. A dependency
        # records the declarations' effect in its invocation instead, so the
        # edit cannot strand it; the producer retains and tracks the text.
        manifest.write_text('[workspace]\nmembers = ["member"]\n# edited\n')
        observer.check_receipt(dependency)
        with self.assertRaisesRegex(observer.ObservationError, "stale source/input inventory"):
            observer.check_receipt(producer)

    @unittest.skipUnless(RUSTC, "requires native rustc")
    def test_input_probe_does_not_share_compiler_incremental_state(self):
        package = self.root / "package"
        package.mkdir()
        source = package / "lib.rs"
        source.write_text("pub fn value() -> u8 { 1 }\n")
        # A cache wrapper can prune the real compilation's incremental directory
        # while the probe runs; an occupied path fails any probe that uses it.
        occupied = self.root / "incremental"
        occupied.write_text("not a directory")
        roots = [observer.inventory(package, "package", [])]
        observer.compiler_inputs(RUSTC, [str(source), "--crate-name", "subject", "--crate-type", "lib",
                                         "-C", "incremental=" + str(occupied)], roots, package, package, None)
        self.assertEqual(roots[0]["inventory"]["explicitly_required"], ["lib.rs"])

    @unittest.skipUnless(HELPER and RUSTC, "requires compiled adapter and native rustc")
    def test_build_script_inventory_is_its_compile_closure(self):
        package = self.root / "package"
        (package / "src").mkdir(parents=True)
        (package / "Cargo.toml").write_text('[package]\nname = "fixture"\nversion = "0.0.0"\nedition = "2021"\n[workspace]\n')
        (package / "src" / "lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
        script = package / "build.rs"
        script.write_text("fn main() {}\n")
        os.environ.update({"CARGO_MANIFEST_DIR": str(package), "CARGO_PKG_NAME": "fixture", "CARGO_PKG_VERSION": "0.0.0",
                           "CHELIS_IDENTITY_PACKAGE_SOURCE": "path:.", "CHELIS_IDENTITY_WORKSPACE": str(package)})
        receipt = observer.collect_unit(RUSTC, [str(script), "--crate-name", "build_script_build", "--crate-type", "bin"])
        # Cargo never recompiles a build script for its package's other sources.
        (package / "src" / "added.rs").write_text("pub fn added() {}\n")
        (package / "src" / "lib.rs").write_text("pub fn value() -> u8 { 2 }\n")
        observer.check_receipt(receipt)
        script.write_text("fn main() { println!(); }\n")
        with self.assertRaisesRegex(observer.ObservationError, "stale source/input inventory"):
            observer.check_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
