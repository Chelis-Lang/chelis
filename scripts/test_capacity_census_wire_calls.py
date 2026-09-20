"""Compiler-executed C6 serialization invocation controls (not a name census).

Fixtures compile with the pinned driver and already-built, coherent serde
dependencies. The runner must supply --extern artifacts; no project Cargo runs
are hidden in this suite. Each accepted discovery still owes a codec owner.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent


class DriverBuildControls(unittest.TestCase):
    def setUp(self):
        from capacity_census_wire_calls import DRIVER

        self.scratch = tempfile.TemporaryDirectory(
            prefix="driver-build-", dir=ROOT / "target"
        )
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.source = self.root / DRIVER
        self.source.parent.mkdir(parents=True)
        for name in (
            "Cargo.toml",
            "clippy.toml",
            "rust-toolchain.toml",
            "scripts/gate.py",
        ):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        self.source.write_text("fn main() {}\n")

    def build(self):
        from capacity_census_wire_calls import build_driver

        return build_driver(self.root, self.root / "output")

    def test_failed_cargo_reports_rendered_diagnostics_from_json_stdout(self):
        from capacity_census_wire_calls import _run

        script = (
            "import json, sys; "
            'print(json.dumps({"reason":"compiler-message",'
            '"message":{"rendered":"error: exact output is not writeable"}})); '
            'print("could not compile dependency", file=sys.stderr); sys.exit(101)'
        )
        with self.assertRaisesRegex(ValueError, "exact output is not writeable"):
            _run([sys.executable, "-c", script], cwd=self.root)

    def test_actual_dep_info_and_policy_are_bound_and_missing_dep_info_rebuilds(self):
        import hashlib

        driver = self.build()
        receipt = json.loads(driver.with_suffix(".json").read_text())
        self.assertEqual(Path(receipt["command"][0]).name, "clippy-driver")
        self.assertIn("--deny=unexpected_cfgs", receipt["command"])
        self.assertIn("warnings", receipt["command"])
        dep = Path(receipt["dep_info"]["path"])
        self.assertEqual(
            receipt["dep_info"]["sha256"], hashlib.sha256(dep.read_bytes()).hexdigest()
        )
        self.assertIn(str(self.source), dep.read_text())
        self.assertIn(str(self.source), {p["path"] for p in receipt["inputs"]})
        modified = driver.stat().st_mtime_ns
        self.assertEqual(self.build().stat().st_mtime_ns, modified)
        dep.unlink()
        self.assertEqual(self.build(), driver)
        self.assertTrue(dep.is_file())
        from capacity_census_wire_calls import _driver_dep_inputs

        self.source.write_text("fn main() { std::hint::black_box(1); }\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            _driver_dep_inputs(dep, self.root)

    def test_source_dependencies_and_policy_changes_invalidate_cached_build(self):
        helper = self.source.with_name("helper.rs")
        helper.write_text("pub fn value() -> u8 { 1 }\n")
        self.source.write_text(
            "mod helper; fn main() { std::hint::black_box(helper::value()); }\n"
        )
        driver = self.build()
        original = json.loads(driver.with_suffix(".json").read_text())
        helper.write_text("pub fn value() -> u8 { 2 }\n")
        rebuilt = self.build()
        changed = json.loads(rebuilt.with_suffix(".json").read_text())
        self.assertNotEqual(original["inputs"], changed["inputs"])
        self.assertNotEqual(original["binary_sha256"], changed["binary_sha256"])
        policy = self.root / "clippy.toml"
        policy.write_text(policy.read_text() + "\n# changed policy identity\n")
        self.assertNotEqual(self.build(), rebuilt)

    def test_actual_clippy_rejects_hash_types_and_undeclared_configuration(self):
        self.build()
        for source, reason in (
            (
                "fn main() { let _: std::collections::HashSet<u8> = Default::default(); }",
                "disallowed",
            ),
            ("#[cfg(unregistered)] fn hidden() {} fn main() {}", "unexpected"),
        ):
            self.source.write_text(source)
            with (
                self.subTest(source=source),
                patch.dict(os.environ, {"CLIPPY_ARGS": "--cap-lints=allow"}),
            ):
                with self.assertRaisesRegex(ValueError, reason):
                    self.build()


class DriverRuntimeControls(unittest.TestCase):
    def test_pinned_sysroot_library_environment_is_required_and_prefixed(self):
        from capacity_census_wire_calls import _runtime_library_environment

        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "lib"
            library.mkdir()
            self.assertEqual(
                _runtime_library_environment(
                    library, platform="linux", environ={"LD_LIBRARY_PATH": "/ambient"}
                ),
                {"LD_LIBRARY_PATH": str(library.resolve()) + os.pathsep + "/ambient"},
            )
            with self.assertRaisesRegex(ValueError, "missing pinned sysroot library"):
                _runtime_library_environment(
                    Path(directory) / "missing", platform="linux", environ={}
                )

    def test_fixture_driver_receives_the_pinned_runtime_environment(self):
        from capacity_census_wire_calls import COMPILER, analyze_fixture

        raw = {
            "format": 1,
            "compiler": COMPILER,
            "crate": {},
            "serialize_trait": {"crate": "serde_core", "path": "::ser::Serialize"},
            "serializer_trait": {"crate": "serde_core", "path": "::ser::Serializer"},
            "schema_trait": {},
            "decoder_traits": [],
            "dynamic_carriers": [],
            "bodies": [],
            "instances": 0,
            "calls": [],
            "codec_calls": [],
            "schema_calls": [],
            "dynamic_returns": [],
            "errors": [],
            "rustc_command": [],
            "inputs": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seen = {}

            def run(_command, *, cwd, env):
                seen.update(env)
                Path(env["WIRE_CALL_REPORT"]).write_text(json.dumps(raw))
                return ""

            with (
                patch(
                    "capacity_census_wire_calls._driver_runtime_environment",
                    return_value={"LD_LIBRARY_PATH": "/pinned/lib"},
                ),
                patch("capacity_census_wire_calls._run", side_effect=run),
            ):
                analyze_fixture(root / "driver", root, "pub fn output() {}", {})
            self.assertEqual(seen["LD_LIBRARY_PATH"], "/pinned/lib")


class CargoOriginControls(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(
            prefix="cargo-origin-", dir=ROOT / "target"
        )
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "Cargo.lock").write_text(
            """version = 4

[[package]]
name = "serde_core"
version = "1.0.228"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "serde_json"
version = "1.0.149"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "pyo3"
version = "0.24.2"
source = "registry+https://github.com/rust-lang/crates.io-index"
"""
        )

    def artifact(self, package, target, filename, *, features=(), directory=None):
        path = (directory or self.root) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())
        return {
            "reason": "compiler-artifact",
            "package_id": (
                "registry+https://github.com/rust-lang/crates.io-index#"
                + package
            ),
            "target": {"kind": ["lib"], "name": target},
            "features": list(features),
            "filenames": [str(path), str(path.with_suffix(".rmeta"))],
        }

    def resolve(self, name, stable_id, artifacts, artifact_ids):
        from capacity_census_wire_calls import _resolve_defining_artifact

        with patch(
            "capacity_census_wire_calls._artifact_id",
            side_effect=lambda path, _root: artifact_ids[Path(path).name],
        ):
            return _resolve_defining_artifact(
                self.root,
                name,
                [{"crate": name, "stable_crate_id": stable_id}],
                artifacts,
                require_registry_origin=True,
            )

    def resolve_construction(self, name, stable_id, artifacts, artifact_ids):
        from capacity_census_wire_calls import _construction_dependency_artifact

        invocation_target = self.root / "target/compiler-json-invocations/cargo"
        with patch(
            "capacity_census_wire_calls._artifact_id",
            side_effect=lambda path, _root: artifact_ids[Path(path).name],
        ):
            return _construction_dependency_artifact(
                self.root,
                name,
                [{"crate": name, "stable_crate_id": stable_id}],
                artifacts,
                invocation_target,
            )

    def test_split_serde_artifacts_resolve_by_locked_package_and_stable_crate_id(self):
        serde_core_old = self.artifact(
            "serde_core@1.0.228",
            "serde_core",
            "libserde_core-old.rlib",
            features=("rc", "result", "std"),
        )
        serde_core_current = self.artifact(
            "serde_core@1.0.228",
            "serde_core",
            "libserde_core-current.rlib",
            features=("std",),
        )
        serde_json_old = self.artifact(
            "serde_json@1.0.149",
            "serde_json",
            "libserde_json-old.rlib",
            features=("default", "float_roundtrip", "raw_value", "std"),
        )
        serde_json_current = self.artifact(
            "serde_json@1.0.149",
            "serde_json",
            "libserde_json-current.rlib",
            features=("default", "indexmap", "preserve_order", "std"),
        )
        artifacts = [
            serde_core_old,
            serde_core_current,
            serde_json_old,
            serde_json_current,
        ]
        artifact_ids = {
            "libserde_core-old.rlib": "1" * 16,
            "libserde_core-current.rlib": "2" * 16,
            "libserde_json-old.rlib": "1" * 16,
            "libserde_json-current.rlib": "3" * 16,
        }

        core = self.resolve("serde_core", "2" * 16, artifacts, artifact_ids)
        json_origin = self.resolve("serde_json", "3" * 16, artifacts, artifact_ids)

        self.assertEqual(core[0], serde_core_current)
        self.assertEqual(core[1].name, "libserde_core-current.rlib")
        self.assertEqual(core[2], "2" * 16)
        self.assertEqual(json_origin[0], serde_json_current)

    def test_defining_artifact_rejects_absent_ambiguous_unlocked_and_mismatched_evidence(
        self,
    ):
        locked = self.artifact(
            "serde_json@1.0.149", "serde_json", "libserde_json-locked.rlib"
        )
        duplicate = self.artifact(
            "serde_json@1.0.149", "serde_json", "libserde_json-duplicate.rlib"
        )
        unlocked = self.artifact(
            "serde_json@0.0.0", "serde_json", "libserde_json-unlocked.rlib"
        )
        ids = {
            "libserde_json-locked.rlib": "4" * 16,
            "libserde_json-duplicate.rlib": "4" * 16,
            "libserde_json-unlocked.rlib": "4" * 16,
        }
        cases = (
            ("absent", [], ids, "unresolved defining serde_json Cargo origin"),
            (
                "ambiguous",
                [locked, duplicate],
                ids,
                "unresolved defining serde_json Cargo origin",
            ),
            (
                "unlocked",
                [unlocked],
                ids,
                "unresolved defining serde_json Cargo origin",
            ),
            (
                "mismatched",
                [locked],
                {**ids, "libserde_json-locked.rlib": "5" * 16},
                "compiler/Cargo serde_json identity mismatch",
            ),
        )
        for reason, artifacts, artifact_ids, message in cases:
            with (
                self.subTest(reason=reason),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.resolve("serde_json", "4" * 16, artifacts, artifact_ids)

    def test_construction_dependency_uses_the_compiler_selected_serde_json_artifact(
        self,
    ):
        invocation_target = self.root / "target/compiler-json-invocations/cargo"
        old = self.artifact(
            "serde_json@1.0.149",
            "serde_json",
            "libserde_json-old.rlib",
            features=("default", "raw_value", "std"),
            directory=invocation_target,
        )
        current = self.artifact(
            "serde_json@1.0.149",
            "serde_json",
            "libserde_json-current.rlib",
            features=("default", "preserve_order", "std"),
            directory=invocation_target,
        )
        artifact_ids = {
            "libserde_json-old.rlib": "1" * 16,
            "libserde_json-current.rlib": "3" * 16,
        }

        selected = self.resolve_construction(
            "serde_json",
            "3" * 16,
            [old, current],
            artifact_ids,
        )
        self.assertEqual(selected, Path(current["filenames"][0]))

        for reason, artifacts, ids, message in (
            (
                "absent",
                [],
                artifact_ids,
                "unresolved defining serde_json Cargo origin",
            ),
            (
                "ambiguous",
                [current, {**current}],
                artifact_ids,
                "unresolved defining serde_json Cargo origin",
            ),
            (
                "mismatched",
                [old],
                artifact_ids,
                "compiler/Cargo serde_json identity mismatch",
            ),
            (
                "outside invocation target",
                [
                    self.artifact(
                        "serde_json@1.0.149",
                        "serde_json",
                        "libserde_json-outside.rlib",
                    )
                ],
                {"libserde_json-outside.rlib": "3" * 16},
                "missing owned construction artifact serde_json",
            ),
        ):
            with (
                self.subTest(reason=reason),
                self.assertRaisesRegex(ValueError, message),
            ):
                self.resolve_construction(
                    "serde_json",
                    "3" * 16,
                    artifacts,
                    ids,
                )

    def test_construction_dependency_without_compiler_identity_stays_exact(
        self,
    ):
        from capacity_census_wire_calls import _construction_dependency_artifact

        invocation_target = self.root / "target/compiler-json-invocations/cargo"
        pyo3 = self.artifact(
            "pyo3@0.24.2",
            "pyo3",
            "libpyo3-current.rlib",
            directory=invocation_target,
        )
        with patch(
            "capacity_census_wire_calls._artifact_id",
            return_value="8" * 16,
        ) as artifact_id:
            selected = _construction_dependency_artifact(
                self.root,
                "pyo3",
                [],
                [pyo3],
                invocation_target,
            )
        self.assertEqual(selected, Path(pyo3["filenames"][0]))
        artifact_id.assert_called_once_with(selected, self.root)

        with self.assertRaisesRegex(
            ValueError, "missing exact construction dependency pyo3"
        ):
            _construction_dependency_artifact(
                self.root,
                "pyo3",
                [],
                [pyo3, {**pyo3}],
                invocation_target,
            )

        for name, artifact in (
            (
                "pyo3",
                self.artifact(
                    "pyo3@0.0.0",
                    "pyo3",
                    "libpyo3-unlocked.rlib",
                    directory=invocation_target,
                ),
            ),
            (
                "serde_json",
                self.artifact(
                    "serde_json@0.0.0",
                    "serde_json",
                    "libserde_json-unlocked-construction.rlib",
                    directory=invocation_target,
                ),
            ),
        ):
            with (
                self.subTest(name=name),
                patch(
                    "capacity_census_wire_calls._artifact_id"
                ) as artifact_id,
                self.assertRaisesRegex(
                    ValueError, f"missing exact construction dependency {name}"
                ),
            ):
                _construction_dependency_artifact(
                    self.root,
                    name,
                    [],
                    [artifact],
                    invocation_target,
                )
            artifact_id.assert_not_called()

    def test_construction_dependency_rejects_physical_target_escapes(self):
        invocation_target = self.root / "target/compiler-json-invocations/cargo"
        traversal = self.artifact(
            "serde_json@1.0.149",
            "serde_json",
            "../libserde_json-traversal.rlib",
            directory=invocation_target,
        )
        outside = self.root / "outside"
        outside.mkdir()
        linked = invocation_target / "linked"
        linked.parent.mkdir(parents=True, exist_ok=True)
        linked.symlink_to(outside, target_is_directory=True)
        symlink = self.artifact(
            "serde_json@1.0.149",
            "serde_json",
            "libserde_json-symlink.rlib",
            directory=linked,
        )

        for reason, artifact, artifact_id in (
            ("parent traversal", traversal, "6" * 16),
            ("symlink", symlink, "7" * 16),
        ):
            with (
                self.subTest(reason=reason),
                self.assertRaisesRegex(
                    ValueError, "missing owned construction artifact serde_json"
                ),
            ):
                self.resolve_construction(
                    "serde_json",
                    artifact_id,
                    [artifact],
                    {Path(artifact["filenames"][0]).name: artifact_id},
                )

    def test_construction_dependency_rejects_target_symlink_outside_root(self):
        invocation_target = self.root / "target/compiler-json-invocations/cargo"
        outside = self.root.parent / f"{self.root.name}-outside-target"
        outside.mkdir()
        self.addCleanup(shutil.rmtree, outside)
        invocation_target.parent.mkdir(parents=True)
        invocation_target.symlink_to(outside, target_is_directory=True)
        artifact = self.artifact(
            "serde_json@1.0.149",
            "serde_json",
            "libserde_json-outside-target.rlib",
            directory=invocation_target,
        )

        with (
            patch("capacity_census_wire_calls._artifact_id") as artifact_id,
            self.assertRaisesRegex(
                ValueError, "missing owned construction artifact serde_json"
            ),
        ):
            from capacity_census_wire_calls import _construction_dependency_artifact

            _construction_dependency_artifact(
                self.root,
                "serde_json",
                [{"crate": "serde_json", "stable_crate_id": "9" * 16}],
                [artifact],
                invocation_target,
            )
        artifact_id.assert_not_called()


class InvocationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_wire_calls import build_driver, fixture_externs

        cls.scratch = tempfile.TemporaryDirectory(
            prefix="wire-calls-", dir=ROOT / "target"
        )
        cls.directory = Path(cls.scratch.name)
        target = ROOT / "target/agents/wire-codec-rustdoc"
        cls.driver = build_driver(ROOT, target / "wire-calls-tool")
        supplied = os.environ.get("WIRE_CALL_TEST_EXTERNS")
        cls.externs = (
            json.loads(Path(supplied).read_text())
            if supplied
            else fixture_externs(ROOT, target, cls.directory)
        )

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def observe(self, source, *, cfg=()):
        from capacity_census_wire_calls import analyze_fixture

        return analyze_fixture(
            self.driver, self.directory, source, self.externs, cfg=cfg
        )

    def test_aliased_macro_calls_bind_primitive_and_container_substitutions(self):
        evidence = self.observe("""
use serde_json::to_string as renamed;
macro_rules! emit { ($x:expr) => { renamed(&$x).unwrap() }; }
pub fn scalar() -> String { emit!(1.0f64) }
pub fn list() -> String { emit!(vec![1i64, 2]) }
""")
        self.assertEqual(evidence.payload_spellings(), {"f64", "std::vec::Vec<i64>"})
        self.assertFalse(evidence.errors)
        changed = self.observe(
            "pub fn scalar() -> String { serde_json::to_string(&1u64).unwrap() }"
        )
        self.assertEqual(changed.payload_spellings(), {"u64"})
        self.assertNotEqual(evidence.identity, changed.identity)

    def test_private_generic_cache_instances_are_derived_and_open_generic_fails(self):
        source = """
fn save<T: serde::Serialize>(value: &T) -> Vec<u8> { bincode::serialize(value).unwrap() }
pub fn library() -> Vec<u8> { save(&vec![1i64]) }
pub fn stdlib() -> Vec<u8> { save(&true) }
"""
        evidence = self.observe(source)
        self.assertFalse(evidence.errors)
        self.assertEqual(evidence.payload_spellings(), {"std::vec::Vec<i64>", "bool"})
        self.assertTrue(self.observe(source.replace("fn save", "pub fn save")).errors)
        changed = self.observe(source.replace("save(&true)", "save(&1.0f64)"))
        self.assertIn("f64", changed.payload_spellings())

    def test_custom_entrypoint_and_direct_serializer_do_not_need_known_names(self):
        evidence = self.observe("""
fn custom<T: serde::Serialize>(x: &T) -> String { serde_json::to_string(x).unwrap() }
pub fn output() -> String { custom(&1i64) }
pub fn direct() -> Vec<u8> {
    let mut bytes = Vec::new();
    serde::Serialize::serialize(&1.0f64, &mut serde_json::Serializer::new(&mut bytes)).unwrap();
    bytes
}
""")
        self.assertFalse(evidence.errors)
        self.assertEqual(evidence.payload_spellings(), {"i64", "f64"})
        spoof = self.observe("""
trait Serialize {} impl Serialize for f64 {}
fn custom<T: Serialize>(_: &T) -> bool { true }
pub fn output() -> bool { custom(&1.0f64) }
""")
        self.assertEqual(spoof.payload_spellings(), set())

    def test_indirect_encoder_and_erased_function_value_fail_closed(self):
        for source in (
            "pub fn output(f: fn(&f64) -> String) -> String { f(&1.0) }",
            "pub fn output() -> String { let f: fn(&f64) -> _ = serde_json::to_string; f(&1.0).unwrap() }",
            "pub fn output(f: fn(&mut Vec<u8>, &f64)) -> Vec<u8> { let mut b = Vec::new(); f(&mut b, &1.0); b }",
            "pub fn output() -> Vec<String> { [1.0f64].iter().map(serde_json::to_string).map(Result::unwrap).collect() }",
        ):
            with self.subTest(source=source):
                self.assertTrue(self.observe(source).errors)
        self.assertFalse(
            self.observe("pub fn compute(f: fn(i64) -> i64) -> i64 { f(1) }").errors
        )

    def test_dynamic_container_and_opaque_returns_are_explicit_obligations(self):
        for result in ("serde_json::Value", "impl serde::Serialize"):
            with self.subTest(result=result):
                evidence = self.observe(
                    "pub fn value() -> " + result + ' { serde_json::json!({"n": 1.0}) }'
                )
                self.assertTrue(evidence.dynamic_returns)
        evidence = self.observe(
            "pub fn value() -> Option<Vec<serde_json::Value>> { None }"
        )
        self.assertTrue(evidence.dynamic_returns)
        self.assertFalse(
            self.observe("pub fn value() -> Option<Vec<bool>> { None }").dynamic_returns
        )

    def test_configuration_changes_change_compiler_evidence(self):
        source = "#[cfg(publish)] pub fn value() -> String { serde_json::to_string(&1.0).unwrap() }"
        self.assertFalse(self.observe(source).payload_spellings())
        self.assertEqual(
            self.observe(source, cfg=("publish",)).payload_spellings(), {"f64"}
        )

    def test_codec_impl_does_not_hide_an_independent_publication(self):
        source = """
struct Payload;
impl serde::Serialize for Payload {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bool(true)
    }
}
"""
        evidence = self.observe(source)
        self.assertFalse(evidence.errors)
        self.assertTrue(evidence.raw["codec_calls"])
        self.assertFalse(evidence.raw["calls"])
        changed = self.observe(
            source.replace(
                "s.serialize_bool(true)",
                "let _ = bincode::serialize(&1.0f64); s.serialize_bool(true)",
            )
        )
        self.assertEqual(changed.payload_spellings(), {"f64"})

    def test_real_generic_derive_machinery_is_a_codec_obligation(self):
        evidence = self.observe("""
#[derive(serde::Serialize, serde::Deserialize)] struct Payload<T> { values: Vec<Option<T>> }
pub fn output() -> String { serde_json::to_string(&Payload { values: vec![Some(1i64)] }).unwrap() }
""")
        self.assertFalse(evidence.errors)
        self.assertEqual(evidence.payload_spellings(), {"Payload<i64>"})
        self.assertTrue(evidence.raw["codec_calls"])
        self.assertTrue(
            all(
                call["caller"]["implementation"] for call in evidence.raw["codec_calls"]
            )
        )
        negative = self.observe("""
fn forgotten<T: serde::Serialize>(value: &T) -> String { serde_json::to_string(value).unwrap() }
""")
        self.assertTrue(negative.errors)

    def test_trait_dispatched_and_generic_callback_encoders_fail_if_unresolved(self):
        for source in (
            "pub trait Encode { fn bytes(&self) -> Vec<u8>; } pub fn output(e: &dyn Encode) -> Vec<u8> { e.bytes() }",
            "pub fn output<F: Fn(&f64) -> String>(f: F) -> String { f(&1.0) }",
        ):
            with self.subTest(source=source):
                self.assertTrue(self.observe(source).errors)
        evidence = self.observe("""
fn invoke<F: Fn(&f64) -> String>(f: F) -> String { f(&1.0) }
pub fn output() -> String { invoke(|x| serde_json::to_string(x).unwrap()) }
""")
        self.assertFalse(evidence.errors)
        self.assertEqual(evidence.payload_spellings(), {"f64"})

    def test_text_conversion_entry_is_closed_by_an_explicit_string_parameter(self):
        source = "pub fn failure(stage: impl Into<String>) -> String { stage.into() }"
        self.assertTrue(self.observe(source).errors)
        self.assertFalse(
            self.observe(
                source.replace("impl Into<String>", "String").replace(
                    "stage.into()", "stage"
                )
            ).errors
        )

    def test_schema_metadata_role_requires_the_actual_trait_identity(self):
        evidence = self.observe("""
#[derive(schemars::JsonSchema)] struct Payload<T> { value: T }
pub fn output() -> schemars::schema::RootSchema { schemars::schema_for!(Payload<i64>) }
""")
        self.assertFalse(evidence.errors)
        self.assertTrue(evidence.raw["schema_calls"])
        self.assertTrue(evidence.dynamic_returns)
        fake = self.observe("""
pub trait JsonSchema { fn schema_name() -> String; }
pub fn output<T: JsonSchema>() -> String { T::schema_name() }
""")
        self.assertTrue(fake.errors)
        self.assertFalse(fake.raw["schema_calls"])

    def test_payload_shapes_resolve_aliases_arrays_slices_and_nominal_arguments(self):
        evidence = self.observe("""
#[derive(serde::Serialize)] struct Item { id: i64 }
type Items<'a> = Vec<(&'a str, &'a Item)>;
pub fn output(items: &Items<'_>, values: &[i64], key: &[u8; 32]) {
    let _ = bincode::serialize(items);
    let _ = bincode::serialize(values);
    let _ = bincode::serialize(key);
}
""")
        self.assertFalse(evidence.errors)
        shapes = [p["shape"] for c in evidence.raw["calls"] for p in c["payloads"]]
        self.assertIn(
            {"tag": "slice", "element": {"tag": "primitive", "name": "i64"}}, shapes
        )
        self.assertIn(
            {
                "tag": "array",
                "element": {"tag": "primitive", "name": "u8"},
                "length": 32,
            },
            shapes,
        )
        vector = next(s for s in shapes if s["tag"] == "nominal")
        self.assertEqual(vector["definition"]["path"], "::vec::Vec")
        pair = vector["arguments"][0]
        self.assertEqual(pair["tag"], "tuple")
        self.assertEqual(
            pair["elements"][0],
            {
                "tag": "reference",
                "mutable": False,
                "inner": {"tag": "primitive", "name": "str"},
            },
        )
        self.assertEqual(pair["elements"][1]["inner"]["definition"]["path"], "::Item")
        changed = self.observe(
            "pub fn output(value: &[u8; 16]) { let _ = bincode::serialize(value); }"
        )
        self.assertEqual(changed.raw["calls"][0]["payloads"][0]["shape"]["length"], 16)

    def test_inherent_method_identity_retains_self_type_and_exact_item_name(self):
        evidence = self.observe("""
#[derive(serde::Serialize)] pub struct Context { value: bool }
impl Context { pub fn encode(&self) -> Vec<u8> { bincode::serialize(self).unwrap() } }
""")
        call = next(
            c for c in evidence.raw["calls"] if c["callee"]["crate"] == "bincode"
        )
        self.assertEqual(call["caller"]["definition"]["item_name"], "encode")
        self.assertIsNone(call["caller"]["implementation"]["trait"])
        self.assertEqual(
            call["caller"]["implementation"]["self_type"]["shape"]["definition"][
                "path"
            ],
            "::Context",
        )
        other = self.observe(
            "pub fn encode(value: &bool) -> Vec<u8> { bincode::serialize(value).unwrap() }"
        )
        self.assertIsNone(other.raw["calls"][0]["caller"]["implementation"])

    def test_returned_nominal_wrapper_cannot_hide_dynamic_json(self):
        source = "pub struct Wrap { value: serde_json::Value } pub fn output() -> Wrap { Wrap { value: serde_json::Value::Null } }"
        self.assertTrue(self.observe(source).dynamic_returns)
        self.assertFalse(
            self.observe(
                source.replace("serde_json::Value", "bool").replace(
                    "bool::Null", "false"
                )
            ).dynamic_returns
        )

    def test_supertrait_bound_and_external_opaque_value_are_resolved(self):
        from capacity_census_wire_calls import _run

        source = """
trait Payload: serde::Serialize {} impl Payload for i64 {}
fn save<T: Payload>(value: &T) -> Vec<u8> { bincode::serialize(value).unwrap() }
pub fn output() -> Vec<u8> { save(&1i64) }
"""
        self.assertEqual(self.observe(source).payload_spellings(), {"i64"})
        lib = self.directory / "external.rs"
        lib.write_text(
            'pub fn value() -> impl serde::Serialize { serde_json::json!({"n": 1.0}) }'
        )
        output = self.directory / "libexternal.rlib"
        command = [
            "rustc",
            "--edition=2024",
            "--crate-type=rlib",
            "--crate-name=external",
            str(lib),
            "-o",
            str(output),
        ]
        for name, path in self.externs.items():
            path = Path(path).resolve()
            command += [
                "--extern",
                name + "=" + str(path),
                "-Ldependency=" + str(path.parent),
            ]
        _run(command, cwd=self.directory)
        original = self.externs
        try:
            self.externs = {**original, "external": str(output)}
            self.assertTrue(
                self.observe(
                    "pub fn output() -> impl serde::Serialize { external::value() }"
                ).dynamic_returns
            )
        finally:
            self.externs = original

    def test_missing_or_replaced_compiler_evidence_is_not_admission(self):
        from capacity_census_wire_calls import read_evidence

        for raw in ({}, {"format": 1}, {"format": 1, "bodies": [], "errors": []}):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    read_evidence(raw)

    def test_defining_codec_package_requires_the_exact_locked_registry_origin(self):
        import tomllib
        from capacity_census_wire_calls import _artifact_id, _locked_registry_package

        version = next(
            p["version"]
            for p in tomllib.loads((ROOT / "Cargo.lock").read_text())["package"]
            if p["name"] == "bincode"
        )
        actual = (
            f"registry+https://github.com/rust-lang/crates.io-index#bincode@{version}"
        )
        self.assertTrue(_locked_registry_package(ROOT, "bincode", actual))
        observation = self.observe(
            "pub fn output() -> Vec<u8> { bincode::serialize(&true).unwrap() }"
        )
        defining_crate = observation.raw["calls"][0]["callee"]["stable_crate_id"]
        self.assertEqual(
            _artifact_id(Path(self.externs["bincode"]).resolve(), ROOT), defining_crate
        )
        self.assertNotEqual(
            _artifact_id(Path(self.externs["serde_json"]).resolve(), ROOT),
            defining_crate,
        )
        for fake in (
            actual.replace("#bincode@", "#lookalike@"),
            actual.replace("github.com", "other.example"),
            actual.replace("@" + version, "@0.0.0"),
            "path+file:///tmp/bincode#0.0.0",
        ):
            with self.subTest(fake=fake):
                self.assertFalse(_locked_registry_package(ROOT, "bincode", fake))


if __name__ == "__main__":
    unittest.main()
