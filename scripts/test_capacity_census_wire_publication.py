"""C6 public-root discovery controls; discovery never supplies final authority."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from capacity_census_graph import GraphError, RustdocGraph
from test_capacity_census_graph import Artifact, primitive, reference

ROOT = Path(__file__).resolve().parent.parent


class PublishedRoots(unittest.TestCase):
    def test_check_report_protocol_binds_existing_dto_producer_and_consumer(self):
        from capacity_census_wire_publication import (
            CHECK_REPORT_PROTOCOL,
            require_check_report_protocol,
        )

        definitions = {
            "chelis_compiler_api::schema::CheckResult": 0,
            "chelis_compiler_api::schema::WireCheckResult": 0,
        }
        self.assertEqual(
            require_check_report_protocol(definitions), CHECK_REPORT_PROTOCOL
        )
        self.assertEqual(
            CHECK_REPORT_PROTOCOL.producer,
            "chelis_compiler_api::schema::CheckResult::to_report_json",
        )
        self.assertEqual(
            CHECK_REPORT_PROTOCOL.consumer,
            "chelis_compiler_api::schema::WireCheckResult",
        )
        for missing in tuple(definitions):
            changed = dict(definitions)
            del changed[missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(
                GraphError, "check-report protocol"
            ):
                require_check_report_protocol(changed)

    def test_unreferenced_private_protocol_is_a_root_and_cannot_hide_a_number(self):
        artifact = Artifact()
        artifact.struct(1, "Public", [])
        artifact.struct(
            2, "LocalReport", [artifact.field("score", primitive("f64"))], public=False
        )
        publication = self.discover(artifact)
        self.assertEqual(
            [(leaf.path, leaf.primitive) for leaf in publication.graph.numeric_leaves],
            [("fixture::LocalReport.score", "f64")],
        )
        self.assertEqual(
            {row.identity for row in publication.declarations},
            {"fixture::Public", "fixture::LocalReport"},
        )
        artifact.struct(
            3,
            "Open",
            [artifact.field("value", {"generic": "T"})],
            params=("T",),
            public=False,
        )
        with self.assertRaisesRegex(
            GraphError, "uninstantiated.*serialization template"
        ):
            self.discover(artifact)

    def test_private_binary_owner_is_recorded_and_does_not_authorize_a_new_protocol(
        self,
    ):
        artifact = Artifact()
        artifact.struct(1, "Public", [])
        artifact.struct(
            2, "CacheMirror", [artifact.field("count", primitive("u64"))], public=False
        )
        publication = self.discover(
            artifact, (("fixture::CacheMirror", "context-cache"),)
        )
        self.assertFalse(publication.graph.numeric_leaves)
        self.assertEqual(
            {row.identity: row.owner for row in publication.declarations},
            {"fixture::Public": "wire", "fixture::CacheMirror": "context-cache"},
        )
        artifact.struct(
            3, "AnotherCache", [artifact.field("value", primitive("f64"))], public=False
        )
        changed = self.discover(artifact, (("fixture::CacheMirror", "context-cache"),))
        self.assertEqual(
            [(leaf.path, leaf.primitive) for leaf in changed.graph.numeric_leaves],
            [("fixture::AnotherCache.value", "f64")],
        )

    def test_private_binding_serde_definition_requires_a_shared_wire_owner(self):
        from capacity_census_wire_publication import require_shared_binding_protocols

        artifact = Artifact("chelis_python")
        artifact.struct(1, "Handle", [])
        artifact.doc["index"]["1"]["inner"]["struct"]["impls"] = []
        require_shared_binding_protocols(RustdocGraph([artifact.doc]))
        artifact.struct(
            2,
            "PrivateManifest",
            [artifact.field("extent", primitive("i64"))],
            public=False,
        )
        graph = RustdocGraph([artifact.doc])
        self.assertFalse(graph.serialization_candidates("chelis_python"))
        self.assertEqual(
            set(graph.serialization_definitions("chelis_python")),
            {"chelis_python::PrivateManifest"},
        )
        with self.assertRaisesRegex(
            GraphError, "binding-local serialized protocol.*PrivateManifest"
        ):
            require_shared_binding_protocols(graph)

    def discover(self, artifact, binary=()):
        from capacity_census_wire_publication import discover_published_graph

        return discover_published_graph(
            RustdocGraph([artifact.doc]), artifact.name, binary_owners=dict(binary)
        )

    def test_outside_schema_and_private_reachable_numeric_fields_are_discovered(self):
        artifact = Artifact()
        artifact.struct(
            1, "Private", [artifact.field("value", primitive("f64"))], public=False
        )
        artifact.struct(2, "ArtifactManifest", [artifact.field("nested", reference(1))])
        publication = self.discover(artifact)
        self.assertEqual(
            [(x.path, x.primitive) for x in publication.graph.numeric_leaves],
            [("fixture::Private.value", "f64")],
        )
        self.assertEqual(
            publication.candidates[0].identity, "fixture::ArtifactManifest"
        )
        self.assertEqual(publication.candidates[0].owner, "wire")

        del artifact.doc["index"]["1"]
        with self.assertRaisesRegex(GraphError, "missing definition"):
            self.discover(artifact)

    def test_binary_ownership_is_exact_and_does_not_spread_to_new_candidates(self):
        artifact = Artifact()
        artifact.struct(1, "Cache", [artifact.field("bytes", primitive("u64"))])
        artifact.struct(2, "Response", [artifact.field("text", primitive("bool"))])
        publication = self.discover(artifact, (("fixture::Cache", "context-cache"),))
        self.assertFalse(publication.graph.numeric_leaves)
        self.assertEqual(
            {x.identity: x.owner for x in publication.candidates},
            {"fixture::Cache": "context-cache", "fixture::Response": "wire"},
        )

        artifact.struct(3, "NewCache", [artifact.field("hidden", primitive("f64"))])
        changed = self.discover(artifact, (("fixture::Cache", "context-cache"),))
        self.assertEqual(
            [(x.path, x.primitive) for x in changed.graph.numeric_leaves],
            [("fixture::NewCache.hidden", "f64")],
        )
        self.assertNotEqual(publication.identity, changed.identity)
        with self.assertRaisesRegex(GraphError, "absent binary candidate"):
            self.discover(artifact, (("fixture::Missing", "context-cache"),))

    def test_binary_owner_cannot_hide_a_cache_reachable_from_a_wire_root(self):
        artifact = Artifact()
        artifact.struct(1, "Cache", [artifact.field("hidden", primitive("f64"))])
        artifact.struct(2, "Response", [artifact.field("cache", reference(1))])
        publication = self.discover(artifact, (("fixture::Cache", "context-cache"),))
        self.assertEqual(
            [(x.path, x.primitive) for x in publication.graph.numeric_leaves],
            [("fixture::Cache.hidden", "f64")],
        )

    def test_generic_exports_require_concrete_reachable_instantiations(self):
        artifact = Artifact()
        artifact.struct(
            1, "Response", [artifact.field("value", {"generic": "T"})], params=("T",)
        )
        with self.assertRaisesRegex(
            GraphError, "uninstantiated public/private serialization template"
        ):
            self.discover(artifact)
        artifact.struct(
            2, "Batch", [artifact.field("response", reference(1, primitive("i64")))]
        )
        publication = self.discover(artifact)
        self.assertEqual(
            [(x.path, x.primitive) for x in publication.graph.numeric_leaves],
            [("fixture::Response.value", "i64")],
        )
        self.assertEqual(publication.candidates[1].parameters, ("T",))

    def test_reexport_alias_is_accounted_without_duplicate_declared_leaves(self):
        artifact = Artifact()
        artifact.struct(1, "Response", [artifact.field("value", primitive("i64"))])
        artifact.add(
            2,
            "AlsoResponse",
            {"use": {"id": 1, "name": "AlsoResponse", "is_glob": False}},
        )
        artifact.doc["index"]["0"]["inner"]["module"]["items"].append(2)
        publication = self.discover(artifact)
        self.assertEqual(len(publication.candidates), 2)
        self.assertEqual(len(publication.graph.numeric_leaves), 1)
        self.assertEqual(len(publication.graph.roots), 1)
        before = publication.identity
        artifact.doc["index"]["2"]["inner"]["use"]["name"] = "RenamedResponse"
        self.assertNotEqual(before, self.discover(artifact).identity)


class ActualPublicationArtifacts(unittest.TestCase):
    def test_rust_source_mutations_cannot_hide_new_or_relocated_numeric_capacity(self):
        from capacity_census_wire_publication import discover_published_graph

        source = """
use serde::{Serialize, Deserialize};
mod hidden {
    #[derive(super::Serialize, super::Deserialize)]
    pub struct Payload { value: bool }
}
pub use hidden::Payload as Published;
#[derive(Serialize, Deserialize)]
pub struct Response<T> { payload: T }
pub type Snapshot = Response<Published>;
"""
        with tempfile.TemporaryDirectory(prefix="wire-publication-") as temporary:
            directory = Path(temporary)
            fixture = ROOT / "scripts/fixtures/capacity_graph/serde"
            for name in ("Cargo.toml", "Cargo.lock"):
                shutil.copyfile(fixture / name, directory / name)
            (directory / "src").mkdir()

            def build(text):
                (directory / "src/lib.rs").write_text(text)
                target = ROOT / "target/graph-fixture-target"
                result = subprocess.run(
                    [
                        "cargo",
                        "rustdoc",
                        "--locked",
                        "--manifest-path",
                        str(directory / "Cargo.toml"),
                        "--lib",
                        "--",
                        "--output-format",
                        "json",
                        "-Z",
                        "unstable-options",
                        "--document-private-items",
                    ],
                    env={
                        **os.environ,
                        "RUSTC_BOOTSTRAP": "1",
                        "CARGO_BUILD_JOBS": "1",
                        "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
                        "CARGO_TARGET_DIR": str(target),
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                artifact = json.loads(
                    (target / "doc/serde_graph_fixture.json").read_text()
                )
                engine = RustdocGraph([artifact])
                nominal = engine.serialization_definitions("serde_graph_fixture")
                if "struct PrivateManifest" in text:
                    self.assertIn("serde_graph_fixture::PrivateManifest", nominal)
                    self.assertNotIn(
                        "serde_graph_fixture::PrivateManifest",
                        engine.serialization_candidates("serde_graph_fixture"),
                    )
                return discover_published_graph(
                    engine, "serde_graph_fixture", binary_owners={}
                )

            baseline = build(source)
            self.assertFalse(baseline.graph.numeric_leaves)
            private = build(
                source
                + "#[derive(Serialize, Deserialize)] struct PrivateManifest { size: i64 }"
            )
            self.assertEqual(
                [(leaf.path, leaf.primitive) for leaf in private.graph.numeric_leaves],
                [("serde_graph_fixture::PrivateManifest.size", "i64")],
            )
            for changed, suffix, width in (
                (
                    source.replace("value: bool", "value: f64"),
                    "hidden::Payload.value",
                    "f64",
                ),
                (
                    source.replace("hidden", "relocated").replace(
                        "value: bool", "value: i64"
                    ),
                    "relocated::Payload.value",
                    "i64",
                ),
                (
                    source
                    + "#[derive(Serialize, Deserialize)] pub struct NewRoot { value: i32 }",
                    "NewRoot.value",
                    "i32",
                ),
            ):
                with self.subTest(suffix=suffix):
                    publication = build(changed)
                    self.assertEqual(
                        [
                            (x.path, x.primitive)
                            for x in publication.graph.numeric_leaves
                        ],
                        [("serde_graph_fixture::" + suffix, width)],
                    )
                    self.assertNotEqual(baseline.identity, publication.identity)
            with self.assertRaisesRegex(
                GraphError, "uninstantiated public/private serialization template"
            ):
                build(
                    source
                    + "#[derive(Serialize, Deserialize)] pub struct Unused<T> { value: T }"
                )


if __name__ == "__main__":
    unittest.main()
