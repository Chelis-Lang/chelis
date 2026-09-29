"""C6 controls for declarations absent from the rustdoc publication graph."""

from pathlib import Path
import shutil
from dataclasses import replace
import os
import hashlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent


class LocalPublicationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_wire_local import build_probe
        from capacity_census_wire_runner import (
            WIRE_PROBE_SHA256_ENV,
            WIRE_PROBE_TARGET_ENV,
            validate_wire_probe,
        )

        cls.target = ROOT / "target/agents/wire-codec-rustdoc"
        selected_target = os.environ.get(WIRE_PROBE_TARGET_ENV)
        selected_digest = os.environ.get(WIRE_PROBE_SHA256_ENV)
        if (selected_target is None) != (selected_digest is None):
            raise ValueError("selected wire probe requires both target and digest")
        probe_target = (
            validate_wire_probe(ROOT, Path(selected_target), selected_digest)
            if selected_target is not None else cls.target
        )
        cls.probe = build_probe(ROOT, probe_target)
        if selected_target is not None:
            validate_wire_probe(ROOT, probe_target, selected_digest)

    def artifact(self, source, *, cfg=(), rustc_args=(), schema=False):
        from capacity_census_wire_local import expand_library

        with tempfile.TemporaryDirectory(
            prefix="wire-local-", dir=ROOT / "target"
        ) as tmp:
            directory = Path(tmp)
            fixture = ROOT / "scripts/fixtures/capacity_graph/serde"
            for name in ("Cargo.toml", "Cargo.lock"):
                shutil.copyfile(fixture / name, directory / name)
            if schema:
                manifest = directory / "Cargo.toml"
                manifest.write_text(manifest.read_text() + '\nschemars = "=0.8.22"\n')
            (directory / "src").mkdir()
            (directory / "src/lib.rs").write_text(source)
            if schema:
                subprocess.run(
                    [
                        "cargo",
                        "generate-lockfile",
                        "--offline",
                        "--manifest-path",
                        str(manifest),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
            return expand_library(
                ROOT,
                self.target,
                manifest=directory / "Cargo.toml",
                rustc_args=tuple(arg for name in cfg for arg in ("--cfg", name))
                + rustc_args,
            )

    def check(self, source, *, cfg=()):
        from capacity_census_wire_local import verify_expanded_library

        return verify_expanded_library(
            self.artifact(source, cfg=cfg), self.probe, root=ROOT, target=self.target
        )

    def test_module_dtos_allow_serde_internals_but_local_dtos_fail(self):
        source = """
use serde::{Serialize, Deserialize};
#[derive(Serialize, Deserialize)]
struct Payload<T> { value: Option<Vec<T>> }
pub fn publish() { let _ = Payload { value: Some(vec![1.0f64]) }; }
"""
        result = self.check(source)
        self.assertEqual(result.serde_derives, 2)
        self.assertEqual(result.module_nominals, 1)
        with self.assertRaisesRegex(ValueError, "block-local.*Payload"):
            self.check(
                source.replace("pub fn publish() {", "").replace(
                    "#[derive(Serialize, Deserialize)]",
                    "pub fn publish() { #[derive(Serialize, Deserialize)]",
                )
            )

    def test_aliases_renames_and_macros_do_not_create_a_local_exception(self):
        source = """
use serde::{Serialize as Encode, Deserialize as Decode};
macro_rules! payload { () => {
    #[derive(Encode, Decode)] struct Renamed<T> { values: Vec<Option<T>> }
}; }
payload!();
"""
        self.assertEqual(self.check(source).serde_derives, 2)
        with self.assertRaisesRegex(ValueError, "block-local.*Renamed"):
            self.check(source.replace("payload!();", "fn publish() { payload!(); }"))

    def test_active_configuration_is_expanded_and_inactive_code_is_not_a_receipt(self):
        source = """
#[derive(serde::Serialize)] struct Module { value: bool }
#[cfg(local_protocol)] fn publish() {
    #[derive(serde::Serialize)] struct Configured { value: f64 }
}
"""
        self.assertEqual(self.check(source).serde_derives, 1)
        with self.assertRaisesRegex(ValueError, "block-local.*Configured"):
            self.check(source, cfg=("local_protocol",))

    def test_block_local_alias_module_and_impl_are_rejected(self):
        baseline = "struct Module; fn work() {}"
        self.assertEqual(self.check(baseline).module_nominals, 1)
        for statement in (
            "type Hidden = Vec<f64>;",
            "mod hidden { pub struct Payload { value: f64 } }",
            "impl Module { fn encode(&self) {} }",
            "trait Hidden {}",
            "union Hidden { value: f64 }",
        ):
            with self.subTest(statement=statement):
                with self.assertRaisesRegex(ValueError, "block-local"):
                    self.check(
                        baseline.replace(
                            "fn work() {}", "fn work() {" + statement + "}"
                        )
                    )

    def test_const_and_closure_declarations_cannot_hide_from_rustdoc(self):
        self.check("struct Module; const _: () = { let _ = 1; };")
        for body in (
            "const _: () = { struct Hidden { value: f64 } };",
            "fn work() { let _ = || { struct Hidden { value: f64 } }; }",
        ):
            with self.subTest(body=body):
                with self.assertRaisesRegex(ValueError, "block-local.*Hidden"):
                    self.check("struct Module; " + body)

    def test_automatically_derived_is_not_authority_for_manual_local_helpers(self):
        source = """
struct Module;
#[automatically_derived]
impl serde::Serialize for Module {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        #[derive(serde::Serialize)] struct Hidden { value: f64 }
        Hidden { value: 1.0 }.serialize(s)
    }
}
"""
        with self.assertRaisesRegex(ValueError, "block-local.*Hidden"):
            self.check(source)

    def test_generated_serde_custom_field_helpers_match_the_actual_derive(self):
        source = """
fn encode<S: serde::Serializer>(x: &f64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_f64(*x)
}
#[derive(serde::Serialize)] struct Module {
    #[serde(serialize_with="encode")] value: f64
}
"""
        self.assertEqual(self.check(source).serde_derives, 1)
        with self.assertRaisesRegex(ValueError, "block-local.*Hidden"):
            self.check(
                source.replace(
                    "s.serialize_f64(*x)", "struct Hidden; s.serialize_f64(*x)"
                )
            )

    def test_missing_or_malformed_compiler_output_fails_closed(self):
        from capacity_census_wire_local import verify_expanded_library

        for text in ("", "not rust", "fn broken("):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    verify_expanded_library(
                        text, self.probe, root=ROOT, target=self.target
                    )

    def test_provenance_requires_exact_definition_and_complete_parent_linkage(self):
        from capacity_census_wire_local import verify_expanded_library

        artifact = self.artifact(
            "#[derive(serde::Deserialize)] struct Module { value: f64 }"
        )
        for source in (
            artifact.source.replace("macro_def_id:", "unknown_macro_def_id:"),
            artifact.source.replace(
                "syntax_context_data:", "unknown_syntax_context_data:"
            ),
            artifact.source.replace(
                "parent: crate0::{{expn0}},", "parent: crate0::{{expn999999}},"
            ),
            artifact.source.replace("serde_derive[", "other_derive["),
        ):
            with self.subTest(source=source[-30:]):
                with self.assertRaises(ValueError):
                    verify_expanded_library(
                        replace(artifact, source=source),
                        self.probe,
                        root=ROOT,
                        target=self.target,
                    )

    def test_custom_derive_named_serialize_and_source_comments_have_no_authority(self):
        from capacity_census_wire_local import verify_expanded_library

        with tempfile.TemporaryDirectory(
            prefix="wire-local-macro-", dir=ROOT / "target"
        ) as tmp:
            directory = Path(tmp)
            macro = directory / "macro.rs"
            macro.write_text(
                "extern crate proc_macro;\n"
                "#[proc_macro_derive(Serialize)]\n"
                "pub fn derive(_: proc_macro::TokenStream) -> proc_macro::TokenStream {\n"
                '"const _: () = { struct Hidden { value: f64 } };".parse().unwrap()\n}\n'
            )
            result = subprocess.run(
                [
                    "rustc",
                    "--crate-type=proc-macro",
                    "--crate-name=local_protocol_macro",
                    "--edition=2021",
                    "--out-dir",
                    str(directory),
                    str(macro),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            libraries = list(directory.glob("*local_protocol_macro*"))
            self.assertEqual(len(libraries), 1)
            artifact = self.artifact(
                "#[derive(local_protocol_macro::Serialize)] struct Module;",
                rustc_args=("--extern", f"local_protocol_macro={libraries[0]}"),
            )
            with self.assertRaisesRegex(ValueError, "block-local.*Hidden"):
                verify_expanded_library(
                    artifact, self.probe, root=ROOT, target=self.target
                )
        with self.assertRaisesRegex(ValueError, "block-local.*Hidden"):
            self.check("struct Module; fn work() { struct Hidden /* 123#7 */; }")

    def test_format_interpolation_is_visited(self):
        self.check('struct Module; fn work() { let _ = format!("{}", 1); }')
        with self.assertRaisesRegex(ValueError, "block-local.*Hidden"):
            self.check(
                'struct Module; fn work() { let _ = format!("{}", { struct Hidden; 1 }); }'
            )

    def test_schema_derive_admits_only_proven_impls_and_unit_markers(self):
        from capacity_census_wire_local import verify_expanded_library

        artifact = self.artifact(
            """
fn shape(g: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
    <String as schemars::JsonSchema>::json_schema(g)
}
#[derive(schemars::JsonSchema)] struct Module {
    #[schemars(schema_with="shape")] value: String
}
""",
            schema=True,
        )
        result = verify_expanded_library(
            artifact, self.probe, root=ROOT, target=self.target
        )
        self.assertEqual(result.schema_derives, 1)
        self.assertGreater(result.generated_local_items, 0)
        # Mutate the actual compiler output: the recognized expansion's origin
        # cannot turn a numeric marker into the observed unit helper shape.
        import re

        changed, count = re.subn(
            r"(struct _SchemarsSchemaWithFunction\s*/\*[^*]*\*/\s*);",
            r"\1 { value: f64 }",
            artifact.source,
        )
        self.assertEqual(count, 1)
        with self.assertRaisesRegex(
            ValueError, "block-local.*_SchemarsSchemaWithFunction"
        ):
            verify_expanded_library(
                replace(artifact, source=changed),
                self.probe,
                root=ROOT,
                target=self.target,
            )
        with self.assertRaisesRegex(ValueError, "block-local"):
            verify_expanded_library(
                replace(artifact, schema_crate=None),
                self.probe,
                root=ROOT,
                target=self.target,
            )
        with self.assertRaisesRegex(
            ValueError, "block-local.*_SchemarsSchemaWithFunction"
        ):
            self.check(
                "struct Module; fn work() { struct _SchemarsSchemaWithFunction; }"
            )


class LocalProbeTargetRouting(unittest.TestCase):
    def test_supervised_probe_reuses_verified_target_but_fixture_expansion_stays_separate(self):
        from capacity_census_wire_local import ExpandedLibrary

        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            target = Path(directory).resolve()
            probe = target / "debug/examples/wire_publication_probe"
            probe.parent.mkdir(parents=True)
            probe.write_bytes(b"verified probe")
            digest = hashlib.sha256(probe.read_bytes()).hexdigest()
            with (
                patch.dict(os.environ, {
                    "CHELIS_WIRE_SELECTED_PROBE_TARGET": str(target),
                    "CHELIS_WIRE_SELECTED_PROBE_SHA256": digest,
                }),
                patch("capacity_census_wire_local.build_probe", return_value=probe) as build,
                patch("capacity_census_wire_local.expand_library",
                      return_value=ExpandedLibrary("", "", ("",), "")) as expand,
            ):
                LocalPublicationControls.setUpClass()
                control = LocalPublicationControls(
                    "test_module_dtos_allow_serde_internals_but_local_dtos_fail"
                )
                control.artifact("struct Fixture;")
            build.assert_called_once_with(ROOT, target)
            self.assertEqual(
                expand.call_args.args[:2],
                (ROOT, ROOT / "target/agents/wire-codec-rustdoc"),
            )
            self.assertNotEqual(target, LocalPublicationControls.target)

    def test_standalone_control_keeps_its_original_probe_target(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("capacity_census_wire_local.build_probe") as build,
        ):
            LocalPublicationControls.setUpClass()
        build.assert_called_once_with(
            ROOT, ROOT / "target/agents/wire-codec-rustdoc"
        )


if __name__ == "__main__":
    unittest.main()
