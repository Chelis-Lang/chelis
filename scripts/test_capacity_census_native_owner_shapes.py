"""Private reachable-owner shape obligations from spec/11 §§1.2–1.3.

Field visibility is one obligation, never evidence of valid numeric transport.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from capacity_census_graph import GraphError, RustdocGraph
from test_capacity_census_native_bindings import private_owner_fixture
from test_capacity_census_graph import primitive, reference


class NativeOwnerShapeControls(unittest.TestCase):
    def test_tensor_owner_choice_requires_exact_retained_handle_payloads(self):
        from capacity_census_native_bindings import require_tensor_owner_choices

        artifact, field = private_owner_fixture()
        second = artifact.field("0", reference(31, reference(33)))
        artifact.external(31, "alloc::sync::Arc")
        artifact.external(32, "chelis_python::CpuTensorHandle")
        artifact.external(33, "chelis_python::GpuTensorHandle")
        artifact.external(34, "alloc::vec::Vec")
        artifact.doc["index"]["2"]["inner"]["module"]["items"] = []
        root = artifact.doc["index"]["0"]["inner"]["module"]
        root["items"].append(20)
        root["is_crate"] = True
        artifact.doc["paths"]["20"]["path"] = ["chelis_python", "TensorOwner"]
        item = artifact.doc["index"]["20"]
        item.update(name="TensorOwner", visibility="crate")
        item["span"]["filename"] = "crates/chelis-python/src/lib.rs"
        item["inner"] = {"enum": {"generics": {"params": []}, "variants": [21, 22],
                                  "has_stripped_variants": False}}
        for variant_id, name, field_id, payload in (
            (21, "Cpu", field, reference(31, reference(32))),
            (22, "Gpu", second, reference(31, reference(33))),
        ):
            artifact.add(variant_id, name, {"variant": {"kind": {"tuple": [field_id]},
                                                       "discriminant": None}},
                         visibility="default")
            artifact.doc["index"][str(field_id)].update(name="0", visibility="default")
            artifact.doc["index"][str(field_id)]["inner"]["struct_field"] = payload
        self.assertEqual(require_tensor_owner_choices(RustdocGraph([artifact.doc])), ("Cpu", "Gpu"))
        for mutation in ("raw-number", "tagged-number", "wrong-owner", "wrong-container",
                         "renamed", "missing", "extra"):
            changed = copy.deepcopy(artifact.doc)
            body = changed["index"]["20"]["inner"]["enum"]
            payload = changed["index"][str(field)]["inner"]
            if mutation == "raw-number":
                payload["struct_field"] = primitive("f64")
            elif mutation == "tagged-number":
                payload["struct_field"] = reference(31, primitive("f64"))
            elif mutation == "wrong-owner":
                payload["struct_field"] = reference(31, reference(33))
            elif mutation == "wrong-container":
                payload["struct_field"] = reference(34, reference(32))
            elif mutation == "renamed":
                changed["index"]["21"]["name"] = "TaggedNumber"
            elif mutation == "missing":
                body["variants"].pop()
            else:
                changed["index"]["23"] = copy.deepcopy(changed["index"]["22"])
                changed["index"]["23"].update(id=23, name="Other")
                body["variants"].append(23)
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                require_tensor_owner_choices(RustdocGraph([changed]))

    def test_tuple_owner_preserves_every_private_positional_field(self):
        from capacity_census_native_bindings import require_private_tuple_owner

        artifact, field = private_owner_fixture()
        owner = artifact.doc["index"]["20"]["inner"]["struct"]
        owner["kind"] = {"tuple": [field]}
        artifact.doc["index"][str(field)]["name"] = "0"
        identity = "chelis_python::native_tensor::ValidatedTensor"
        source = "crates/chelis-python/src/native_tensor.rs"
        self.assertEqual(
            require_private_tuple_owner(RustdocGraph([artifact.doc]), identity, source),
            (reference(30),),
        )
        for mutation in ("missing", "duplicate", "stripped", "named", "generic",
                         "public", "crate", "position", "foreign-module", "moved"):
            changed = copy.deepcopy(artifact.doc)
            item = changed["index"]["20"]
            body = item["inner"]["struct"]
            field_item = changed["index"][str(field)]
            if mutation == "missing":
                del changed["index"][str(field)]
            elif mutation == "duplicate":
                body["kind"]["tuple"].append(field)
            elif mutation == "stripped":
                body["kind"]["tuple"] = [None]
            elif mutation == "named":
                body["kind"] = {"plain": {"fields": [field], "has_stripped_fields": False}}
            elif mutation == "generic":
                body["generics"]["params"] = [{"name": "T", "kind": {"type": {}}}]
            elif mutation in {"public", "crate"}:
                field_item["visibility"] = mutation
            elif mutation == "position":
                field_item["name"] = "1"
            elif mutation == "foreign-module":
                field_item["visibility"]["restricted"]["parent"] = 0
            else:
                item["span"]["filename"] = "other.rs"
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                require_private_tuple_owner(RustdocGraph([changed]), identity, source)

    def test_private_enum_fields_inherit_only_the_exact_enclosing_owner(self):
        from capacity_census_native_bindings import require_private_enum_owner

        artifact, field = private_owner_fixture()
        item = artifact.doc["index"]["20"]
        item["visibility"] = {"restricted": {"parent": 2, "path": "::native_tensor"}}
        item["inner"] = {"enum": {"generics": {"params": []},
                                  "variants": [21, 22], "has_stripped_variants": False}}
        artifact.add(21, "Owned", {"variant": {"kind": {"tuple": [field]},
                                              "discriminant": None}}, visibility="default")
        artifact.add(22, "Empty", {"variant": {"kind": "plain", "discriminant": None}},
                     visibility="default")
        artifact.doc["index"][str(field)].update(name="0", visibility="default")
        identity = "chelis_python::native_tensor::ValidatedTensor"
        source = "crates/chelis-python/src/native_tensor.rs"
        self.assertEqual(
            require_private_enum_owner(RustdocGraph([artifact.doc]), identity, source),
            (("Owned", (reference(30),)), ("Empty", ())),
        )
        # An arbitrary numeric payload is observable, never hidden by the enum.
        numeric = copy.deepcopy(artifact.doc)
        numeric["index"][str(field)]["inner"]["struct_field"] = primitive("f64")
        self.assertEqual(
            require_private_enum_owner(RustdocGraph([numeric]), identity, source)[0],
            ("Owned", (primitive("f64"),)),
        )
        for mutation in ("public-owner", "crate-owner", "stripped", "missing-variant",
                         "duplicate-variant", "duplicate-name", "public-variant",
                         "public-field", "stripped-field", "wrong-position", "generic",
                         "detached", "explicit-discriminant"):
            changed = copy.deepcopy(artifact.doc)
            body = changed["index"]["20"]["inner"]["enum"]
            variant = changed["index"]["21"]
            if mutation.endswith("-owner"):
                changed["index"]["20"]["visibility"] = mutation.removesuffix("-owner")
            elif mutation == "stripped":
                body["has_stripped_variants"] = True
            elif mutation == "missing-variant":
                del changed["index"]["21"]
            elif mutation == "duplicate-variant":
                body["variants"].append(21)
            elif mutation == "duplicate-name":
                changed["index"]["22"]["name"] = "Owned"
            elif mutation == "public-variant":
                variant["visibility"] = "public"
            elif mutation == "public-field":
                changed["index"][str(field)]["visibility"] = "public"
            elif mutation == "stripped-field":
                variant["inner"]["variant"]["kind"]["tuple"] = [None]
            elif mutation == "wrong-position":
                changed["index"][str(field)]["name"] = "1"
            elif mutation == "generic":
                body["generics"]["params"] = [{"name": "T", "kind": {"type": {}}}]
            elif mutation == "detached":
                changed["index"]["2"]["inner"]["module"]["items"] = []
            else:
                variant["inner"]["variant"]["discriminant"] = {"expr": "1", "value": "1"}
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                require_private_enum_owner(RustdocGraph([changed]), identity, source)


class NativeOwnerRustdocControls(unittest.TestCase):
    def test_compiler_private_fields_and_numeric_enum_successors(self):
        from capacity_census_native_bindings import (
            require_private_tuple_owner,
            require_tensor_owner_choices,
        )

        root = Path(__file__).resolve().parent.parent
        source = """
use std::sync::Arc;
struct CpuTensorHandle;
struct GpuTensorHandle;
enum TensorOwner { Cpu(Arc<CpuTensorHandle>), Gpu(Arc<GpuTensorHandle>) }
mod native_tensor { struct ForeignDataPointer(*mut std::ffi::c_void); }
"""
        variants = {
            "private": source,
            "numeric-payload": source.replace("Arc<CpuTensorHandle>", "Arc<f64>"),
            "public-enum": source.replace("enum TensorOwner", "pub enum TensorOwner"),
            "public-field": source.replace("ForeignDataPointer(*mut", "ForeignDataPointer(pub *mut"),
        }
        for label, text in variants.items():
            with self.subTest(variant=label), tempfile.TemporaryDirectory(
                prefix="native-owner-rustdoc-", dir=root / "target",
            ) as scratch:
                directory = Path(scratch)
                path = directory / "crates/chelis-python/src/lib.rs"
                path.parent.mkdir(parents=True)
                path.write_text(text)
                command = [
                    "rustdoc", "--edition=2024", "--crate-name", "chelis_python",
                    "--output-format", "json", "-Z", "unstable-options",
                    "--document-private-items", "crates/chelis-python/src/lib.rs",
                    "-o", "doc",
                ]
                process = subprocess.run(command, cwd=directory,
                    env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
                    text=True, capture_output=True, check=False)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                graph = RustdocGraph([json.loads((directory / "doc/chelis_python.json").read_text())])
                if label in {"numeric-payload", "public-enum"}:
                    with self.assertRaises(GraphError):
                        require_tensor_owner_choices(graph)
                else:
                    self.assertEqual(require_tensor_owner_choices(graph), ("Cpu", "Gpu"))
                identity = "chelis_python::native_tensor::ForeignDataPointer"
                if label == "public-field":
                    with self.assertRaises(GraphError):
                        require_private_tuple_owner(graph, identity, "crates/chelis-python/src/lib.rs")
                else:
                    fields = require_private_tuple_owner(graph, identity, "crates/chelis-python/src/lib.rs")
                    self.assertEqual(len(fields), 1)
                    self.assertTrue(fields[0]["raw_pointer"]["is_mutable"])


if __name__ == "__main__":
    unittest.main()
