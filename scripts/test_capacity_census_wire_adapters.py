"""Actual artifact and codec controls for atomic wire-census activation."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from capacity_census_graph import GraphError, RustdocGraph

ROOT = Path(__file__).resolve().parent.parent


class ConstGenericGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="wire-const-graph-")
        output = Path(cls.temporary.name)
        command = [
            "rustdoc",
            "--edition=2021",
            "--crate-type",
            "lib",
            "--crate-name",
            "graph_fixture",
            "--output-format",
            "json",
            "-Z",
            "unstable-options",
            "--document-private-items",
            str(ROOT / "scripts/fixtures/capacity_graph/graph.rs"),
            "-o",
            str(output),
        ]
        result = subprocess.run(
            command,
            env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise AssertionError(result.stderr)
        cls.document = json.loads((output / "graph_fixture.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def discover(self, document):
        graph = RustdocGraph([document])
        return graph.discover(
            "graph_fixture",
            graph.public_exports("graph_fixture")["graph_fixture::ConstRoot"],
        )

    def test_actual_const_arguments_and_arrays_preserve_numeric_width(self):
        graph = self.discover(self.document)
        self.assertEqual({leaf.primitive for leaf in graph.numeric_leaves}, {"i32"})
        root = next(
            d for d in graph.definitions if d.identity == "graph_fixture::ConstRoot"
        )
        hex_arguments = {edge.type[2] for edge in root.edges if "Hex" in edge.type[1]}
        self.assertEqual(
            hex_arguments, {(("const", "4"),), (("const", "8"),), (("const", "16"),)}
        )
        nested = next(
            d for d in graph.definitions if d.identity == "graph_fixture::SizedPayload"
        )
        self.assertEqual(
            nested.edges[0].type[:2], ("array", ("const_generic", "WIDTH"))
        )

    def test_standard_container_rejects_const_in_place_of_element_type(self):
        document = copy.deepcopy(self.document)
        field = next(i for i in document["index"].values() if i.get("name") == "tree")
        # Reuse the actual Vec path from a separate recursive field.
        vector = next(
            copy.deepcopy(i["inner"]["struct_field"])
            for i in document["index"].values()
            if i.get("inner", {})
            .get("struct_field", {})
            .get("resolved_path", {})
            .get("path")
            == "Vec"
        )
        vector["resolved_path"]["args"]["angle_bracketed"]["args"][0] = {
            "const": {"expr": "8", "value": None, "is_literal": True}
        }
        field["inner"]["struct_field"] = vector
        graph = RustdocGraph([document])
        with self.assertRaises(GraphError):
            graph.discover(
                "graph_fixture",
                graph.public_exports("graph_fixture")["graph_fixture::Root"],
            )

    def test_wrong_kind_unresolved_and_computed_const_arguments_reject(self):
        for mutation in ("unresolved", "computed", "wrong-kind", "missing"):
            with self.subTest(mutation=mutation):
                document = copy.deepcopy(self.document)
                field = next(
                    i for i in document["index"].values() if i.get("name") == "wide"
                )
                args = field["inner"]["struct_field"]["resolved_path"]["args"][
                    "angle_bracketed"
                ]["args"]
                if mutation == "unresolved":
                    args[0]["const"] = {
                        "expr": "MISSING",
                        "value": None,
                        "is_literal": False,
                    }
                elif mutation == "computed":
                    args[0]["const"] = {
                        "expr": "4 + 12",
                        "value": None,
                        "is_literal": False,
                    }
                elif mutation == "wrong-kind":
                    args[0] = {"type": {"primitive": "u64"}}
                else:
                    args.clear()
                with self.assertRaises(GraphError):
                    self.discover(document)

    def test_const_width_change_invalidates_identity_without_becoming_nonnumeric(self):
        before = self.discover(self.document)
        document = copy.deepcopy(self.document)
        field = next(i for i in document["index"].values() if i.get("name") == "wide")
        field["inner"]["struct_field"]["resolved_path"]["args"]["angle_bracketed"][
            "args"
        ][0]["const"]["expr"] = "8"
        after = self.discover(document)
        self.assertNotEqual(before.identity, after.identity)
        self.assertEqual(before.numeric_leaves, after.numeric_leaves)


class CanonicalCodecControls(unittest.TestCase):
    def test_selected_cases_include_json_binary_width_and_rejection_pairs(self):
        from capacity_census_wire_adapters import codec_cases

        vocabulary = [
            {"name": name, "width": width, "kind": kind}
            for name, width, kind in [
                ("f64", 8, "float"),
                ("f32", 4, "float"),
                ("f16", 2, "float"),
                ("bf16", 2, "float"),
                ("int64", 8, "integer"),
                ("int32", 4, "integer"),
                ("int16", 2, "integer"),
                ("int8", 1, "integer"),
                ("bool", 1, "bool"),
            ]
        ]
        cases = codec_cases(vocabulary)
        ids = [case.identity for case in cases]
        self.assertEqual(len(ids), len(set(ids)))
        for dtype in vocabulary:
            for carrier in ("scalar", "storage"):
                for codec in ("json", "binary"):
                    selected = [
                        c
                        for c in cases
                        if c.dtype == dtype["name"]
                        and c.carrier == carrier
                        and c.codec == codec
                    ]
                    self.assertTrue(any(c.expected is not None for c in selected))
                    self.assertTrue(any(c.expected is None for c in selected))
        for dtype in ("f16", "bf16"):
            exhaustive = next(
                c for c in cases if c.identity == f"storage/json/{dtype}/all-bits"
            )
            self.assertEqual(len(exhaustive.expected["elements"]), 65536)

    def test_missing_duplicate_unexecuted_and_wrong_outcomes_reject(self):
        from capacity_census_wire_adapters import CodecCase, check_observations

        case = CodecCase("one", "int8", "scalar", "json", "{}", {"value": 7})
        self.assertEqual(
            check_observations([case], [{"id": "one", "observation": {"value": 7}}])[
                0
            ].outcome,
            "passed",
        )
        rejected = CodecCase(
            "bad", "f64", "scalar", "json", "{}", None, "missing field"
        )
        with self.assertRaisesRegex(GraphError, "wrong decode rejection reason"):
            check_observations(
                [rejected],
                [
                    {
                        "id": "bad",
                        "observation": None,
                        "decode_error": "unrelated failure",
                    }
                ],
            )
        for observations in (
            [],
            [{"id": "other", "observation": {"value": 7}}],
            [{"id": "one", "observation": {"value": 8}}],
            [{"id": "one", "observation": {"value": 7}}] * 2,
            [{"id": "one", "status": "PASS"}],
        ):
            with self.subTest(observations=observations), self.assertRaises(GraphError):
                check_observations([case], observations)


class CodecTargetLease(unittest.TestCase):
    def test_canonical_codec_cleans_an_initialized_owned_target(self):
        from capacity_census_wire_adapters import _WIRE, verify_canonical_codec

        class Graph:
            orders = {_WIRE + "BinaryScalarWire": ()}

            @staticmethod
            def canonical_graph():
                return SimpleNamespace(identity="mock-codec-graph")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target/agents/wire-codec-rustdoc"
            (target / "doc").mkdir(parents=True)
            (target / "doc/chelis_types.json").write_text("{}")
            binary = target / "debug/examples/wire_codec_probe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"mock probe")
            calls = []

            def run(command, **_kwargs):
                calls.append(command)
                stdout = (
                    '{"vocabulary": []}\n'
                    if len(command) == 1 and Path(command[0]).resolve() == binary.resolve()
                    else ""
                )
                return SimpleNamespace(returncode=0, stderr="", stdout=stdout)

            with (
                patch(
                    "capacity_census_wire_adapters.source_identity", return_value="source"
                ),
                patch("capacity_census_wire_adapters.subprocess.run", side_effect=run),
                patch(
                    "capacity_census_wire_adapters._CodecShapeGraph",
                    return_value=Graph(),
                ),
                patch("capacity_census_wire_adapters.codec_cases", return_value=[]),
                patch(
                    "capacity_census_wire_adapters.check_observations", return_value=()
                ),
            ):
                verify_canonical_codec(root, target)

            self.assertEqual(calls[0][:2], ["cargo", "clean"])
            self.assertIn("--target-dir", calls[0])
            self.assertEqual(calls[0][-1], str(target.resolve()))
            self.assertTrue((target / "CACHEDIR.TAG").is_file())

    def test_owned_target_has_cargo_cache_marker_and_rejects_a_changed_marker(self):
        from capacity_census_wire_adapters import (
            CARGO_CACHE_TAG,
            CARGO_CACHE_TAG_SIGNATURE,
            _target_lease,
        )

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            with _target_lease(target):
                self.assertEqual((target / "CACHEDIR.TAG").read_text(), CARGO_CACHE_TAG)
            (target / "CACHEDIR.TAG").write_text(
                CARGO_CACHE_TAG_SIGNATURE + "Cargo data\n"
            )
            with _target_lease(target):
                self.assertEqual(
                    (target / "CACHEDIR.TAG").read_text(),
                    CARGO_CACHE_TAG_SIGNATURE + "Cargo data\n",
                )
            (target / "CACHEDIR.TAG").write_text("not a Cargo cache marker\n")
            with self.assertRaisesRegex(GraphError, "invalid CACHEDIR.TAG"):
                with _target_lease(target):
                    self.fail("invalid marker admitted")

    def test_owned_target_lease_rejects_interleaving_and_releases_after_failure(self):
        from capacity_census_wire_adapters import _target_lease

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with _target_lease(target):
                with self.assertRaisesRegex(GraphError, "already owned"):
                    with _target_lease(target):
                        self.fail("interleaved codec sequence")
            with self.assertRaisesRegex(ValueError, "probe failure"):
                with _target_lease(target):
                    raise ValueError("probe failure")
            with _target_lease(target):
                pass


class ActualCanonicalCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_wire_adapters import verify_canonical_codec

        cls.receipt = verify_canonical_codec(
            ROOT, ROOT / "target/agents/wire-codec-rustdoc"
        )
        cls.document = json.loads(cls.receipt.document)
        report = ROOT / "target/coordination/canonical-codec-execution.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(cls.receipt.execution_report(), indent=2) + "\n")

    def test_actual_private_codec_graph_and_executed_width_matrix(self):
        from capacity_census_wire_adapters import CanonicalWireGraph

        graph = CanonicalWireGraph([self.document], self.receipt)
        discovered = graph.canonical_graph()
        self.assertEqual(
            {leaf.primitive for leaf in discovered.numeric_leaves},
            {"f64", "f32", "f16", "bf16", "i64", "i32", "i16", "i8"},
        )
        self.assertEqual(len(discovered.numeric_leaves), 32)
        self.assertTrue(
            all(
                row.selected and row.executed and row.outcome == "passed"
                for row in self.receipt.outcomes
            )
        )
        identities = {d.identity for d in discovered.definitions}
        self.assertTrue(any(name.endswith("::BinaryScalarWire") for name in identities))
        self.assertTrue(any(name.endswith("::HexBits") for name in identities))

    def test_arbitrary_or_changed_artifacts_cannot_reuse_execution(self):
        from capacity_census_wire_adapters import CanonicalWireGraph, VerifiedCodec

        with self.assertRaises(TypeError):
            VerifiedCodec()
        for change in (
            "tag",
            "width",
            "private",
            "custom-source",
            "extra-variant",
            "unknown-serde",
            "relocate-tag",
        ):
            with self.subTest(change=change):
                document = copy.deepcopy(self.document)
                items = document["index"]
                mirror = next(
                    i for i in items.values() if i.get("name") == "ScalarWire"
                )
                if change == "tag":
                    mirror["attrs"] = []
                elif change == "width":
                    variant = items[str(mirror["inner"]["enum"]["variants"][0])]
                    field = items[
                        str(variant["inner"]["variant"]["kind"]["struct"]["fields"][0])
                    ]
                    field["inner"]["struct_field"]["resolved_path"]["args"][
                        "angle_bracketed"
                    ]["args"][0]["const"]["expr"] = "8"
                elif change == "private":
                    hex_bits = next(
                        i for i in items.values() if i.get("name") == "HexBits"
                    )
                    items[str(hex_bits["inner"]["struct"]["kind"]["tuple"][0])][
                        "visibility"
                    ] = "public"
                elif change == "custom-source":
                    for item in items.values():
                        if (
                            item.get("span", {})
                            and item["span"]["filename"].endswith("wire_codec.rs")
                            and "impl" in item["inner"]
                        ):
                            item["span"]["filename"] = "arbitrary.rs"
                elif change == "extra-variant":
                    mirror["inner"]["enum"]["variants"].append(
                        mirror["inner"]["enum"]["variants"][0]
                    )
                elif change == "unknown-serde":
                    mirror["attrs"].append({"other": "#[serde(default)]"})
                else:
                    variant = items[str(mirror["inner"]["enum"]["variants"][0])]
                    variant["attrs"] += mirror["attrs"]
                    mirror["attrs"] = []
                with self.assertRaises(GraphError):
                    CanonicalWireGraph([document], self.receipt).canonical_graph()
                # Re-recording an artifact hash alone is insufficient: independent
                # role structure must reject the same mutation before execution.
                from capacity_census_wire_adapters import _CodecShapeGraph

                with self.assertRaises(GraphError):
                    _CodecShapeGraph(
                        [document], json.loads(self.receipt.vocabulary)
                    ).canonical_graph()

    def test_preserved_size_and_mtime_cannot_reuse_the_old_compiled_codec(self):
        from capacity_census_wire_adapters import verify_canonical_codec

        source = ROOT / "crates/chelis-types/src/dtype_semantics/wire_codec.rs"
        original = source.read_bytes()
        stat = source.stat()
        changed = original.replace(b"text.len() != DIGITS", b"text.len() == DIGITS")
        self.assertNotEqual(changed, original)
        self.assertEqual(len(changed), len(original))
        try:
            source.write_bytes(changed)
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaisesRegex(
                GraphError, "stale against actual source bytes"
            ):
                self.receipt.validate([self.document])
            with self.assertRaisesRegex(GraphError, "wrong codec observation"):
                verify_canonical_codec(ROOT, ROOT / "target/agents/wire-codec-rustdoc")
        finally:
            source.write_bytes(original)
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.receipt.validate([self.document])

    def test_probe_is_selected_by_registered_clippy_and_has_actual_dep_info(self):
        from check_configuration_closure import (
            CLIPPY_MATRIX,
            check_owner_invokes,
            compiled_rust_sources,
        )

        metadata = json.loads(
            subprocess.run(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        package = next(p for p in metadata["packages"] if p["name"] == "chelis-types")
        target = next(t for t in package["targets"] if t["name"] == "wire_codec_probe")
        self.assertIn(package["id"], metadata["workspace_members"])
        self.assertEqual(target["kind"], ["example"])
        self.assertFalse(target.get("required-features"))
        registered = next(
            row for row in CLIPPY_MATRIX if row.label == "default-features"
        )
        self.assertIn("--workspace", registered.command)
        self.assertIn("--all-targets", registered.command)
        check_owner_invokes(registered, ROOT)
        source = "crates/chelis-types/examples/wire_codec_probe.rs"
        self.assertEqual(Path(target["src_path"]), ROOT / source)
        self.assertIn(
            source,
            compiled_rust_sources([ROOT / "target/agents/wire-codec-rustdoc"], ROOT),
        )
        self.assertNotIn(source, compiled_rust_sources([], ROOT))


class SourceIdentityControls(unittest.TestCase):
    def test_oracle_selection_and_consumer_source_changes_invalidate_evidence(self):
        from capacity_census_wire_adapters import source_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            required = (
                "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
                "crates/chelis-types/examples/wire_codec_probe.rs",
                "spec/02-surf-syntax.md", "spec/03-deep-syntax.md",
                "spec/04-type-system.md", "spec/05-risc-primitives.md",
                "spec/10-serialization.md", "spec/11-ffi.md",
            )
            controls = (
                "scripts/test_capacity_census_wire_verifier.py",
                "tests/support/capacity_census_wire_verifier.rs",
                "tests/conformance/hull/run_conformance.py",
                "bindings/python/tests/native_execution_wire.py",
            )
            for name in (*required, *controls):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("before")
            before = source_identity(root)
            for name in controls:
                with self.subTest(control=name):
                    (root / name).write_text("after")
                    self.assertNotEqual(before, source_identity(root))
                    (root / name).write_text("before")
                    self.assertEqual(before, source_identity(root))

    def test_actual_source_and_authority_bytes_ignore_git_visibility_flags(self):
        from capacity_census_wire_adapters import source_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorities = (
                "spec/02-surf-syntax.md",
                "spec/03-deep-syntax.md",
                "spec/04-type-system.md",
                "spec/05-risc-primitives.md",
                "spec/10-serialization.md",
                "spec/11-ffi.md",
            )
            (root / "crates/chelis-types/examples").mkdir(parents=True)
            for name in (
                "Cargo.toml",
                "Cargo.lock",
                "rust-toolchain.toml",
                "crates/chelis-types/examples/wire_codec_probe.rs",
                *authorities,
            ):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("before")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "update-index",
                    "--assume-unchanged",
                    "Cargo.lock",
                    *authorities,
                ],
                check=True,
            )
            before = source_identity(root)
            for name in ("Cargo.lock", *authorities):
                with self.subTest(input=name):
                    (root / name).write_text("after")
                    self.assertNotEqual(before, source_identity(root))
                    (root / name).write_text("before")
                    self.assertEqual(before, source_identity(root))
            for name in authorities:
                with self.subTest(missing=name):
                    (root / name).unlink()
                    with self.assertRaisesRegex(
                        GraphError, "missing codec source input"
                    ):
                        source_identity(root)
                    (root / name).write_text("before")


if __name__ == "__main__":
    unittest.main()
