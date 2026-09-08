"""Supporting graph-engine controls; these do not activate the live wire oracle."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from capacity_census_graph import (
    CarrierContract,
    FieldRole,
    GraphError,
    Role,
    RustdocGraph,
    VerifiedTransport,
    verify_transport,
)

ROOT = Path(__file__).resolve().parent.parent
SPAN = "chelis_compiler_api::schema::DiagnosticSpan"
FIXTURE_ROOT = ROOT / "scripts/fixtures/capacity_graph"
GRAPH_SOURCE = Path("graph.rs")
SERDE_SOURCE = Path("serde/src/lib.rs")


def assert_fixture_ownership(directory: Path) -> None:
    """The source exception covers only inputs the actual rustdoc tests own."""
    actual = {path.relative_to(directory) for path in directory.rglob("*.rs")}
    expected = {GRAPH_SOURCE, SERDE_SOURCE}
    if actual != expected:
        raise AssertionError(
            f"rustdoc fixture ownership differs: unowned={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )


def primitive(name):
    return {"primitive": name}


def reference(item_id, *args):
    return {
        "resolved_path": {
            "path": "display_name_is_not_identity",
            "id": item_id,
            "args": {
                "angle_bracketed": {
                    "args": [{"type": arg} for arg in args],
                    "constraints": [],
                }
            },
        }
    }


class Artifact:
    """Small rustdoc-v60 documents with the same shape as compiler output."""

    def __init__(self, name="fixture"):
        self.name = name
        self.doc = {
            "format_version": 60,
            "root": 0,
            "index": {},
            "paths": {},
            "external_crates": {},
        }
        self.add(0, name, {"module": {"items": [], "is_stripped": False}}, path=[name])
        self.next_id = 100

    def add(
        self,
        item_id,
        name,
        inner,
        *,
        path=None,
        attrs=(),
        visibility="public",
        crate_id=0,
    ):
        self.doc["index"][str(item_id)] = {
            "id": item_id,
            "name": name,
            "crate_id": crate_id,
            "visibility": visibility,
            "attrs": [
                a if a == "automatically_derived" else {"other": a} for a in attrs
            ],
            "inner": inner,
        }
        if path:
            self.doc["paths"][str(item_id)] = {
                "crate_id": crate_id,
                "path": path,
                "kind": next(iter(inner)),
            }
        return item_id

    def field(self, name, ty, *, attrs=()):
        self.next_id += 1
        return self.add(
            self.next_id, name, {"struct_field": ty}, attrs=attrs, visibility="default"
        )

    def derived(self):
        result = []
        for trait in ("Serialize", "Deserialize"):
            trait_id = 900 if trait == "Serialize" else 901
            self.external(
                trait_id,
                "serde_core::"
                + ("ser" if trait == "Serialize" else "de")
                + "::"
                + trait,
            )
            self.next_id += 1
            result.append(
                self.add(
                    self.next_id,
                    None,
                    {"impl": {"trait": {"path": trait, "id": trait_id}, "items": []}},
                    attrs=("automatically_derived",),
                )
            )
        return result

    def struct(
        self, item_id, name, fields, *, params=(), attrs=(), path=None, public=True
    ):
        self.add(
            item_id,
            name,
            {
                "struct": {
                    "kind": {"plain": {"fields": fields, "has_stripped_fields": False}},
                    "generics": {
                        "params": [
                            {"name": p, "kind": {"type": {"default": None}}}
                            for p in params
                        ],
                        "where_predicates": [],
                    },
                    "impls": self.derived(),
                }
            },
            path=path or [self.name, name],
            attrs=attrs,
        )
        if public:
            self.doc["index"]["0"]["inner"]["module"]["items"].append(item_id)
        return item_id

    def external(self, item_id, path):
        self.doc["paths"][str(item_id)] = {
            "crate_id": 1,
            "path": path.split("::"),
            "kind": "struct",
        }
        self.doc["external_crates"]["1"] = {"name": path.split("::")[0]}

    def span(self):
        variants = []
        for variant_id, name, names in [
            (10, "Point", ["offset"]),
            (11, "Range", ["offset", "len"]),
        ]:
            variants.append(
                self.add(
                    variant_id,
                    name,
                    {
                        "variant": {
                            "kind": {
                                "struct": {
                                    "fields": [
                                        self.field(n, primitive("u64")) for n in names
                                    ],
                                    "has_stripped_fields": False,
                                }
                            }
                        }
                    },
                )
            )
        self.add(
            1,
            "DiagnosticSpan",
            {
                "enum": {
                    "generics": {"params": [], "where_predicates": []},
                    "variants": variants,
                    "has_stripped_variants": False,
                    "impls": self.derived(),
                }
            },
            path=SPAN.split("::"),
            attrs=('#[serde(tag = "span", rename_all = "snake_case")]',),
        )
        self.doc["index"]["0"]["inner"]["module"]["items"].append(1)
        return self


def graph(artifact, root=1, *arguments):
    return RustdocGraph([artifact.doc]).discover(
        artifact.name, reference(root, *arguments)
    )


def span_contract(discovered):
    return CarrierContract(
        SPAN,
        discovered.identity,
        (
            FieldRole(SPAN + "::Point.offset", Role.SOURCE_OFFSET),
            FieldRole(SPAN + "::Range.offset", Role.SOURCE_OFFSET),
            FieldRole(SPAN + "::Range.len", Role.SOURCE_EXTENT),
        ),
    )


class GraphClosure(unittest.TestCase):
    def test_private_nominal_container_generic_and_nonnumeric_companion(self):
        a = Artifact()
        a.external(30, "core::option::Option")
        a.external(31, "alloc::vec::Vec")
        a.struct(
            2,
            "Private",
            [a.field("value", {"generic": "T"})],
            params=("T",),
            public=False,
        )
        a.struct(
            1,
            "Root",
            [
                a.field(
                    "hidden",
                    reference(30, reference(31, reference(2, primitive("f64")))),
                ),
                a.field("label", primitive("bool")),
            ],
        )
        found = graph(a)
        self.assertEqual(
            [(x.path, x.primitive) for x in found.numeric_leaves],
            [("fixture::Private.value", "f64")],
        )
        b = copy.deepcopy(a)
        b.doc["index"]["2"]["inner"]["struct"]["kind"]["plain"]["fields"] = []
        self.assertEqual(graph(b).numeric_leaves, ())
        self.assertNotEqual(found.identity, graph(b).identity)

    def test_missing_definition_unknown_primitive_and_open_generic_fail(self):
        for ty in (
            reference(9),
            primitive("f256"),
            {"generic": "T"},
            {"impl_trait": []},
        ):
            with self.subTest(ty=ty):
                a = Artifact()
                a.struct(1, "Root", [a.field("value", ty)])
                with self.assertRaises(GraphError):
                    graph(a)

    def test_imported_definition_requires_actual_artifact(self):
        a, b = Artifact(), Artifact("dependency")
        b.struct(1, "Imported", [b.field("value", primitive("f32"))])
        a.external(9, "dependency::Imported")
        a.struct(1, "Root", [a.field("value", reference(9))])
        with self.assertRaisesRegex(GraphError, "missing.*artifact"):
            graph(a)
        found = RustdocGraph([a.doc, b.doc]).discover("fixture", reference(1))
        self.assertEqual(
            [(x.path, x.primitive) for x in found.numeric_leaves],
            [("dependency::Imported.value", "f32")],
        )

    def test_container_lookalike_is_not_std_adapter(self):
        a = Artifact()
        a.external(9, "pretender::Vec")
        a.struct(1, "Root", [a.field("value", reference(9, primitive("f64")))])
        with self.assertRaisesRegex(GraphError, "missing.*artifact"):
            graph(a)

    def test_recursive_and_growing_generic_equations_reach_fixed_point(self):
        a = Artifact()
        a.external(9, "alloc::vec::Vec")
        a.struct(
            2,
            "Tree",
            [
                a.field("value", {"generic": "T"}),
                a.field("next", reference(2, reference(9, {"generic": "T"}))),
            ],
            params=("T",),
            public=False,
        )
        a.struct(1, "Root", [a.field("tree", reference(2, primitive("i64")))])
        found = graph(a)
        self.assertEqual(
            [(x.path, x.primitive) for x in found.numeric_leaves],
            [("fixture::Tree.value", "i64")],
        )
        self.assertLessEqual(found.iterations, 4)
        a.doc["index"][
            str(a.doc["index"]["2"]["inner"]["struct"]["kind"]["plain"]["fields"][0])
        ]["inner"]["struct_field"] = primitive("f64")
        self.assertEqual({x.primitive for x in graph(a).numeric_leaves}, {"f64"})

    def test_mutual_alias_cycles_and_generic_argument_errors_fail(self):
        a = Artifact()
        a.add(
            1,
            "A",
            {"type_alias": {"type": reference(2), "generics": {"params": []}}},
            path=[a.name, "A"],
        )
        a.add(
            2,
            "B",
            {"type_alias": {"type": reference(1), "generics": {"params": []}}},
            path=[a.name, "B"],
        )
        with self.assertRaisesRegex(GraphError, "alias cycle"):
            graph(a)
        a = Artifact()
        a.struct(1, "Generic", [a.field("value", {"generic": "T"})], params=("T",))
        with self.assertRaisesRegex(GraphError, "generic"):
            graph(a)
        self.assertEqual(
            {x.primitive for x in graph(a, 1, primitive("i32")).numeric_leaves}, {"i32"}
        )

    def test_public_reexport_discovers_private_definition(self):
        a = Artifact()
        a.struct(
            2,
            "Private",
            [a.field("value", primitive("i16"))],
            path=[a.name, "hidden", "Private"],
            public=False,
        )
        a.add(
            3,
            None,
            {
                "use": {
                    "name": "Published",
                    "id": 2,
                    "is_glob": False,
                    "source": "hidden::Private",
                }
            },
        )
        a.doc["index"]["0"]["inner"]["module"]["items"].append(3)
        engine = RustdocGraph([a.doc])
        self.assertEqual(set(engine.public_exports(a.name)), {"fixture::Published"})
        exports = engine.discover_exports(a.name)
        self.assertEqual({x.primitive for x in exports.numeric_leaves}, {"i16"})
        a.doc["index"]["3"]["inner"]["use"]["id"] = 999
        with self.assertRaises(GraphError):
            RustdocGraph([a.doc]).public_exports(a.name)

    def test_stripped_fields_unknown_format_and_unsupported_generics_reject(self):
        for mutation in ("field", "format", "const", "default", "arity"):
            a = Artifact()
            a.struct(1, "Root", [a.field("value", primitive("f64"))])
            if mutation == "field":
                a.doc["index"]["1"]["inner"]["struct"]["kind"]["plain"][
                    "has_stripped_fields"
                ] = True
            elif mutation == "format":
                a.doc["format_version"] = 999
            elif mutation in ("const", "default"):
                parameter = (
                    {"name": "T", "kind": {"const": {"type": primitive("usize")}}}
                    if mutation == "const"
                    else {"name": "T", "kind": {"type": {"default": primitive("f64")}}}
                )
                a.doc["index"]["1"]["inner"]["struct"]["generics"]["params"] = [
                    parameter
                ]
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                graph(a, 1, primitive("f64")) if mutation == "arity" else graph(a)

    def test_generic_multiple_instantiations_and_private_relocation_remain_visible(
        self,
    ):
        a = Artifact()
        a.struct(
            2,
            "Payload",
            [a.field("value", {"generic": "T"})],
            params=("T",),
            public=False,
        )
        a.struct(
            1,
            "Root",
            [
                a.field("left", reference(2, primitive("f32"))),
                a.field("right", reference(2, primitive("f64"))),
            ],
        )
        before = graph(a)
        self.assertEqual({x.primitive for x in before.numeric_leaves}, {"f32", "f64"})
        a.doc["paths"]["2"]["path"].insert(1, "elsewhere")
        after = graph(a)
        self.assertEqual({x.primitive for x in after.numeric_leaves}, {"f32", "f64"})
        self.assertNotEqual(before.identity, after.identity)
        self.assertTrue(all("::elsewhere::" in x.path for x in after.numeric_leaves))

    def test_alias_array_tuple_reference_are_traversed_and_pointer_rejected(self):
        a = Artifact()
        ty = {
            "borrowed_ref": {
                "lifetime": None,
                "is_mutable": False,
                "type": {
                    "tuple": [
                        {"array": {"type": primitive("u16"), "len": "4"}},
                        primitive("bool"),
                    ]
                },
            }
        }
        a.add(
            2,
            "Alias",
            {"type_alias": {"type": ty, "generics": {"params": []}}},
            path=[a.name, "Alias"],
        )
        a.struct(1, "Root", [a.field("v", reference(2))])
        self.assertEqual({x.primitive for x in graph(a).numeric_leaves}, {"u16"})
        a.doc["index"]["2"]["inner"]["type_alias"]["type"] = {
            "raw_pointer": {"type": primitive("f64"), "is_mutable": False}
        }
        with self.assertRaisesRegex(GraphError, "unsupported"):
            graph(a)


class CarrierRecognition(unittest.TestCase):
    def test_source_roles_require_exact_shape_and_closed_role(self):
        a = Artifact("chelis_compiler_api").span()
        found = graph(a)
        contract = span_contract(found)
        for leaf in found.numeric_leaves:
            witness = verify_transport(found, leaf, [contract])
            self.assertIsInstance(witness, VerifiedTransport)
            self.assertEqual(witness.leaf, leaf)
        with self.assertRaises(TypeError):
            VerifiedTransport()
        with self.assertRaises(AttributeError):
            witness._leaf = found.numeric_leaves[-1]
        with self.assertRaisesRegex(GraphError, "zero"):
            verify_transport(found, found.numeric_leaves[0], [])
        with self.assertRaisesRegex(GraphError, "multiple"):
            verify_transport(found, found.numeric_leaves[0], [contract, contract])

    def test_tag_width_relocation_and_new_numeric_field_invalidate_admission(self):
        original = Artifact("chelis_compiler_api").span()
        before = graph(original)
        contract = span_contract(before)
        for mutation in ("tag", "width", "relocation", "extra", "attribute_location"):
            with self.subTest(mutation=mutation):
                a = copy.deepcopy(original)
                if mutation == "tag":
                    a.doc["index"]["1"]["attrs"] = []
                elif mutation == "width":
                    a.doc["index"]["101"]["inner"]["struct_field"] = primitive("f64")
                elif mutation == "relocation":
                    a.doc["paths"]["1"]["path"].insert(1, "moved")
                elif mutation == "extra":
                    a.doc["index"]["10"]["inner"]["variant"]["kind"]["struct"][
                        "fields"
                    ].append(a.field("count", primitive("u64")))
                else:
                    a.doc["index"]["10"]["attrs"] = a.doc["index"]["1"].pop("attrs")
                after = graph(a)
                self.assertNotEqual(after.identity, before.identity)
                with self.assertRaises(GraphError):
                    verify_transport(after, after.numeric_leaves[0], [contract])
                # Rebuilding the expected identity cannot bless an invalid role.
                with self.assertRaises(GraphError):
                    verify_transport(
                        after, after.numeric_leaves[0], [span_contract(after)]
                    )

    def test_arbitrary_tagged_f64_and_integer_dtype_lookalikes_never_qualify(self):
        for dtype in ("f64", "u64"):
            a = Artifact()
            a.struct(
                1,
                "Metadata",
                [a.field("value", primitive(dtype))],
                attrs=('#[serde(tag = "span")]',),
            )
            found = graph(a)
            forged = CarrierContract(
                "fixture::Metadata",
                found.identity,
                (FieldRole("fixture::Metadata.value", Role.SOURCE_OFFSET),),
            )
            with self.assertRaises(GraphError):
                verify_transport(found, found.numeric_leaves[0], [forged])

    def test_mixed_container_does_not_authorize_numeric_sibling(self):
        a = Artifact("chelis_compiler_api").span()
        a.struct(
            2,
            "Envelope",
            [a.field("location", reference(1)), a.field("score", primitive("f64"))],
        )
        found = graph(a, 2)
        contract = span_contract(found)
        source = next(x for x in found.numeric_leaves if "Point.offset" in x.path)
        score = next(x for x in found.numeric_leaves if "score" in x.path)
        verify_transport(found, source, [contract])
        with self.assertRaisesRegex(GraphError, "zero"):
            verify_transport(found, score, [contract])

    def test_unknown_serde_and_custom_serializer_fail_closed(self):
        for attr in (
            '#[serde(with = "codec")]',
            "#[serde(skip)]",
            '#[serde(misspelled = "x")]',
        ):
            a = Artifact("chelis_compiler_api").span()
            a.doc["index"]["101"]["attrs"] = [{"other": attr}]
            with self.assertRaisesRegex(GraphError, "unsupported serde"):
                graph(a)
        a = Artifact("chelis_compiler_api").span()
        impl_id = a.doc["index"]["1"]["inner"]["enum"]["impls"][0]
        a.doc["index"][str(impl_id)]["attrs"] = []
        with self.assertRaisesRegex(GraphError, "custom.*adapter"):
            graph(a)

    def test_input_reference_keeps_axis_separate_and_rejects_role_swaps(self):
        a = Artifact("chelis_compiler_api")
        axis = "chelis_compiler_api::schema::WireRtAxis"
        dim = "chelis_compiler_api::schema::WireRtDim"
        a.add(
            10,
            "Lit",
            {
                "variant": {
                    "kind": {
                        "struct": {
                            "fields": [a.field("value", primitive("i32"))],
                            "has_stripped_fields": False,
                        }
                    }
                }
            },
        )
        a.add(
            2,
            "WireRtAxis",
            {
                "enum": {
                    "generics": {"params": []},
                    "variants": [10],
                    "impls": a.derived(),
                }
            },
            path=axis.split("::"),
            attrs=('#[serde(tag = "axis", rename_all = "snake_case")]',),
        )
        a.add(
            11,
            "InputAxis",
            {
                "variant": {
                    "kind": {
                        "struct": {
                            "fields": [
                                a.field("tensor", primitive("u64")),
                                a.field("axis", reference(2)),
                            ],
                            "has_stripped_fields": False,
                        }
                    }
                }
            },
        )
        a.add(
            1,
            "WireRtDim",
            {
                "enum": {
                    "generics": {"params": []},
                    "variants": [11],
                    "impls": a.derived(),
                }
            },
            path=dim.split("::"),
            attrs=('#[serde(tag = "bound", rename_all = "snake_case")]',),
        )
        found = graph(a)
        slot = next(x for x in found.numeric_leaves if x.path.endswith(".tensor"))
        contract = CarrierContract(
            dim, found.identity, (FieldRole(slot.path, Role.INPUT_REFERENCE),)
        )
        self.assertEqual(
            verify_transport(found, slot, [contract]).role, Role.INPUT_REFERENCE
        )
        axis_leaf = next(x for x in found.numeric_leaves if x.path.endswith(".value"))
        self.assertEqual(axis_leaf.primitive, "i32")
        for field in (
            FieldRole(axis_leaf.path, Role.INPUT_REFERENCE),
            FieldRole(slot.path, Role.SOURCE_EXTENT),
        ):
            with self.subTest(field=field):
                wrong = CarrierContract(dim, found.identity, (field,))
                with self.assertRaises(GraphError):
                    verify_transport(
                        found,
                        next(x for x in found.numeric_leaves if x.path == field.path),
                        [wrong],
                    )
        for mutation in ("width", "tag", "axis_variant"):
            b = copy.deepcopy(a)
            if mutation == "width":
                b.doc["index"]["101"]["inner"]["struct_field"] = primitive("i64")
            elif mutation == "tag":
                b.doc["index"]["1"]["attrs"] = []
            else:
                b.add(12, "Other", {"variant": {"kind": "plain"}})
                b.doc["index"]["2"]["inner"]["enum"]["variants"].append(12)
            changed = graph(b)
            rebaselined = CarrierContract(dim, changed.identity, contract.fields)
            with self.assertRaises(GraphError):
                verify_transport(changed, slot, [rebaselined])

    def test_impostor_serialize_trait_and_missing_trait_identity_do_not_prove_codec(
        self,
    ):
        for mutation in ("impostor", "missing", "renamed"):
            a = Artifact("chelis_compiler_api").span()
            if mutation == "impostor":
                a.doc["paths"]["900"]["path"] = ["pretender", "Serialize"]
            elif mutation == "missing":
                del a.doc["paths"]["900"]
            else:
                impl_id = a.doc["index"]["1"]["inner"]["enum"]["impls"][0]
                a.doc["index"][str(impl_id)]["inner"]["impl"]["trait"][
                    "path"
                ] = "RenamedSerialize"
            found = graph(a)
            if mutation == "renamed":
                verify_transport(found, found.numeric_leaves[0], [span_contract(found)])
            else:
                with self.assertRaisesRegex(GraphError, "derived"):
                    verify_transport(
                        found, found.numeric_leaves[0], [span_contract(found)]
                    )

    def test_missing_serde_proof_and_stale_leaf_are_rejected(self):
        a = Artifact("chelis_compiler_api").span()
        a.doc["index"]["1"]["inner"]["enum"]["impls"] = []
        found = graph(a)
        with self.assertRaisesRegex(GraphError, "derived"):
            verify_transport(found, found.numeric_leaves[0], [span_contract(found)])
        b = Artifact()
        b.struct(1, "Other", [b.field("value", primitive("i64"))])
        with self.assertRaisesRegex(GraphError, "leaf"):
            verify_transport(found, graph(b).numeric_leaves[0], [span_contract(found)])


class FixtureOwnership(unittest.TestCase):
    def test_only_actual_rustdoc_inputs_belong_to_fixture_gate(self):
        assert_fixture_ownership(ROOT / "scripts/fixtures/capacity_graph")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for source in ("graph.rs", "serde/src/lib.rs"):
                path = directory / source
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            assert_fixture_ownership(directory)
            unowned = directory / "unowned.rs"
            unowned.write_text("pub struct HiddenNumeric(pub f64);")
            with self.assertRaisesRegex(AssertionError, "unowned.rs"):
                assert_fixture_ownership(directory)
            unowned.unlink()
            (directory / "graph.rs").unlink()
            with self.assertRaisesRegex(AssertionError, "graph.rs"):
                assert_fixture_ownership(directory)


class ActualRustdoc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert_fixture_ownership(FIXTURE_ROOT)

    def test_actual_serde_attributes_and_custom_impl_are_distinguished(self):
        fixture = FIXTURE_ROOT / SERDE_SOURCE.parent.parent / "Cargo.toml"
        target = ROOT / "target/graph-fixture-target"
        result = subprocess.run(
            [
                "cargo",
                "rustdoc",
                "--manifest-path",
                str(fixture),
                "--locked",
                "--lib",
                "--output-format",
                "json",
                "-Z",
                "unstable-options",
                "--",
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
        document = json.loads((target / "doc/serde_graph_fixture.json").read_text())
        engine = RustdocGraph([document])
        exports = engine.public_exports("serde_graph_fixture")
        found = engine.discover(
            "serde_graph_fixture", exports["serde_graph_fixture::Wrapped"]
        )
        span = next(
            d for d in found.definitions if d.identity.endswith("::DiagnosticSpan")
        )
        self.assertEqual(span.codec, "serde-derived")
        self.assertEqual(dict(span.serde), {"tag": "span", "rename_all": "snake_case"})
        wrapper = next(d for d in found.definitions if d.identity.endswith("::Wrapped"))
        self.assertEqual(dict(wrapper.edges[0].serde), {"rename": "where"})
        with self.assertRaisesRegex(GraphError, "custom.*adapter"):
            engine.discover(
                "serde_graph_fixture", exports["serde_graph_fixture::Manual"]
            )

    def test_actual_private_reexport_generic_and_recursive_artifact(self):
        source = FIXTURE_ROOT / GRAPH_SOURCE
        with tempfile.TemporaryDirectory(prefix="capacity-graph-") as temporary:
            out = Path(temporary)
            result = subprocess.run(
                [
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
                    str(source),
                    "-o",
                    str(out),
                ],
                env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            document = json.loads((out / "graph_fixture.json").read_text())
            engine = RustdocGraph([document])
            exported = engine.public_exports("graph_fixture")
            self.assertIn("graph_fixture::Published", exported)
            found = engine.discover("graph_fixture", exported["graph_fixture::Root"])
            self.assertEqual(
                {x.primitive for x in found.numeric_leaves}, {"f64", "i64"}
            )
            changed_source = out / "changed.rs"
            changed_source.write_text(source.read_text().replace("f64", "bool"))
            changed = subprocess.run(
                [
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
                    str(changed_source),
                    "-o",
                    str(out),
                ],
                env={**os.environ, "RUSTC_BOOTSTRAP": "1"},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(changed.returncode, 0, changed.stderr)
            changed_engine = RustdocGraph(
                [json.loads((out / "graph_fixture.json").read_text())]
            )
            changed_graph = changed_engine.discover(
                "graph_fixture",
                changed_engine.public_exports("graph_fixture")["graph_fixture::Root"],
            )
            self.assertEqual(
                {x.primitive for x in changed_graph.numeric_leaves}, {"i64"}
            )
            self.assertNotEqual(changed_graph.identity, found.identity)
            # A real definition deletion reaches the resolver, not rustc.
            private = next(
                k
                for k, v in document["paths"].items()
                if v["path"] == ["graph_fixture", "hidden", "Payload"]
            )
            del document["index"][private]
            with self.assertRaises(GraphError):
                RustdocGraph([document]).discover(
                    "graph_fixture", exported["graph_fixture::Root"]
                )


if __name__ == "__main__":
    unittest.main()
