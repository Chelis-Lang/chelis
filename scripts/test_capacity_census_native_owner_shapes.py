"""Private reachable-owner shape obligations from spec/11 §§1.2–1.3.

Field visibility is one obligation, never evidence of valid numeric transport.
"""

import copy
import unittest

from capacity_census_graph import GraphError, RustdocGraph
from test_capacity_census_native_bindings import private_owner_fixture
from test_capacity_census_graph import primitive, reference


class NativeOwnerShapeControls(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
