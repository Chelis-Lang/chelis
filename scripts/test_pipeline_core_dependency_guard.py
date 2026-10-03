#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_core_dependency_guard as guard


APPROVED_DEPENDENCIES = (
    "chelis-deep",
    "chelis-types",
    "chelis-effects",
    "chelis-ir",
    "chelis-unord",
)


def manifest_with(*dependencies: str) -> str:
    rows = [
        "[package]",
        'name = "chelis-pipeline-core"',
        'version = "0.18.1"',
        "publish = false",
        "",
        "[dependencies]",
    ]
    rows.extend(f'{name} = {{ path = "../{name}" }}' for name in dependencies)
    return "\n".join(rows) + "\n"


def approved_graph() -> dict[str, list[str]]:
    return {
        "chelis-pipeline-core": list(APPROVED_DEPENDENCIES),
        "chelis-deep": [],
        "chelis-types": ["chelis-deep"],
        "chelis-effects": ["chelis-deep", "chelis-types"],
        "chelis-ir": ["chelis-deep", "chelis-types"],
        "chelis-unord": [],
    }


class PipelineCoreDependencyGuardTests(unittest.TestCase):
    def test_approved_manifest_and_resolved_graph_pass(self) -> None:
        guard.validate_fixture(
            manifest_with(*APPROVED_DEPENDENCIES), approved_graph()
        )

    def test_forbidden_direct_dependency_is_named(self) -> None:
        with self.assertRaisesRegex(
            guard.DependencyBoundaryError, "chelis-reef"
        ):
            guard.validate_fixture(
                manifest_with(*APPROVED_DEPENDENCIES, "chelis-reef"),
                approved_graph(),
            )

    def test_forbidden_transitive_path_is_complete(self) -> None:
        graph = approved_graph()
        graph["chelis-types"].append("chelis-compiler-api")
        graph["chelis-compiler-api"] = []

        with self.assertRaisesRegex(
            guard.DependencyBoundaryError,
            "chelis-pipeline-core -> chelis-types -> chelis-compiler-api",
        ):
            guard.validate_fixture(
                manifest_with(*APPROVED_DEPENDENCIES), graph
            )

    def test_correctly_rounded_kernel_leaf_is_approved(self) -> None:
        graph = approved_graph()
        graph["chelis-types"].append("chelis-crmath")
        graph["chelis-crmath"] = []

        guard.validate_fixture(manifest_with(*APPROVED_DEPENDENCIES), graph)

    def test_correctly_rounded_kernels_must_stay_a_leaf(self) -> None:
        graph = approved_graph()
        graph["chelis-types"].append("chelis-crmath")
        graph["chelis-crmath"] = ["chelis-conformance"]
        graph["chelis-conformance"] = []

        with self.assertRaisesRegex(
            guard.DependencyBoundaryError,
            "chelis-pipeline-core -> chelis-types -> chelis-crmath -> chelis-conformance",
        ):
            guard.validate_fixture(manifest_with(*APPROVED_DEPENDENCIES), graph)

    def test_metadata_excludes_dev_only_edges(self) -> None:
        raw = {
            "packages": [
                {"id": "core-id", "name": "chelis-pipeline-core"},
                {"id": "effects-id", "name": "chelis-effects"},
                {"id": "macros-id", "name": "chelis-macros"},
            ],
            "workspace_members": ["core-id", "effects-id", "macros-id"],
            "resolve": {
                "nodes": [
                    {
                        "id": "core-id",
                        "deps": [
                            {
                                "pkg": "effects-id",
                                "dep_kinds": [{"kind": None, "target": None}],
                            }
                        ],
                    },
                    {
                        "id": "effects-id",
                        "deps": [
                            {
                                "pkg": "macros-id",
                                "dep_kinds": [{"kind": "dev", "target": None}],
                            }
                        ],
                    },
                    {"id": "macros-id", "deps": []},
                ]
            },
        }

        graph = guard.parse_metadata(raw)
        self.assertEqual(graph.edges["chelis-effects"], ())

    def test_metadata_keeps_normal_edges_to_denied_packages(self) -> None:
        raw = {
            "packages": [
                {"id": "core-id", "name": "chelis-pipeline-core"},
                {"id": "reef-id", "name": "chelis-reef"},
            ],
            "workspace_members": ["core-id", "reef-id"],
            "resolve": {
                "nodes": [
                    {
                        "id": "core-id",
                        "deps": [
                            {
                                "pkg": "reef-id",
                                "dep_kinds": [{"kind": None, "target": None}],
                            }
                        ],
                    },
                    {"id": "reef-id", "deps": []},
                ]
            },
        }

        with self.assertRaisesRegex(
            guard.DependencyBoundaryError,
            "chelis-pipeline-core -> chelis-reef",
        ):
            guard.validate_graph(guard.parse_metadata(raw))

    def test_unknown_build_dependency_fails_closed(self) -> None:
        manifest = manifest_with(*APPROVED_DEPENDENCIES)
        manifest += "\n[build-dependencies]\nserde = \"1\"\n"

        with self.assertRaisesRegex(
            guard.DependencyBoundaryError, "unknown dependency `serde`"
        ):
            guard.validate_fixture(manifest, approved_graph())

    def test_unknown_target_build_dependency_fails_closed(self) -> None:
        manifest = manifest_with(*APPROVED_DEPENDENCIES)
        manifest += (
            "\n[target.'cfg(unix)'.build-dependencies]\n"
            "serde = \"1\"\n"
        )

        with self.assertRaisesRegex(
            guard.DependencyBoundaryError, "unknown dependency `serde`"
        ):
            guard.validate_fixture(manifest, approved_graph())

    def test_unknown_transitive_workspace_dependency_fails_closed(self) -> None:
        graph = approved_graph()
        graph["chelis-ir"].append("chelis-new-lower-helper")
        graph["chelis-new-lower-helper"] = []

        with self.assertRaisesRegex(
            guard.DependencyBoundaryError,
            "chelis-pipeline-core -> chelis-ir -> chelis-new-lower-helper",
        ):
            guard.validate_fixture(
                manifest_with(*APPROVED_DEPENDENCIES), graph
            )

    def test_unknown_direct_dependency_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            guard.DependencyBoundaryError, "unknown dependency `serde`"
        ):
            guard.validate_fixture(
                manifest_with(*APPROVED_DEPENDENCIES, "serde"),
                approved_graph(),
            )


if __name__ == "__main__":
    unittest.main()
