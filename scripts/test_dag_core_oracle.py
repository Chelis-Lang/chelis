"""Behavior checks for real DAG dependency and import boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dag_core_oracle import (
    BASELINE_REVISION,
    CRATE,
    SOURCE_MODULES,
    inspect_source_owners,
    compare_costs,
    inspect_costs,
    snapshot_record,
    ROOT,
    assert_boundary_rejected,
    inspect_target_graph,
    isolated_graph,
    target_metadata,
)
from no_std_core_support import compile_rejected, consumer


class DagCoreOracleTests(unittest.TestCase):
    def test_real_target_graph_accepts_both_feature_configurations(self) -> None:
        for enabled in (False, True):
            with self.subTest(default_features=enabled), isolated_graph(default_features=enabled) as directory:
                packages = inspect_target_graph(target_metadata(directory))
                self.assertIn(CRATE, packages)
                self.assertIn("chelis-numeric-core", packages)

    def test_direct_host_dependency_is_rejected(self) -> None:
        dependency = f'chelis-types = {{ path = {json.dumps(str(ROOT / "crates/chelis-types"))} }}\n'
        assert_boundary_rejected(dependency, "forbidden target dependency")

    def test_transitive_host_dependency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="chelis-dag-transitive-") as temporary:
            directory = Path(temporary)
            (directory / "src").mkdir()
            (directory / "src/lib.rs").write_text("pub fn fixture() {}\n", encoding="utf-8")
            (directory / "Cargo.toml").write_text(
                '[package]\nname = "dag-transitive-control"\nversion = "0.0.0"\nedition = "2024"\n'
                '[workspace]\n[dependencies]\n'
                f'chelis-types = {{ path = {json.dumps(str(ROOT / "crates/chelis-types"))} }}\n',
                encoding="utf-8",
            )
            assert_boundary_rejected(
                f'dag-transitive-control = {{ path = {json.dumps(str(directory))} }}\n',
                "forbidden target dependency",
            )

    def test_dependency_std_feature_is_rejected(self) -> None:
        assert_boundary_rejected(
            'serde = { version = "1", default-features = false, features = ["std"] }\n',
            "target std feature",
        )

    def test_new_owner_executes_and_obsolete_imports_fail(self) -> None:
        consumer(CRATE, r'''
use chelis_dag_core::{Dag, RiscOp, TensorType, verify};
fn main() {
    let mut dag = Dag::new();
    let decl = dag.declare("fixture");
    let x = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], TensorType::scalar_f32(), None);
    dag.add_root(x);
    assert!(verify::verify(&dag).is_empty());
    assert_eq!(dag.roots(), &[x]);
}
''')
        for imported in (
            "Dag", "NodeId", "TensorType", "LoadStoreName", "dag::Dag", "dag::Owner", "dag::DeclId",
            "grad::grad_dag_checked", "fuse::fuse_with_remap", "tier2::lower_relu",
            "axis_sources::same_shape_result_agreement", "optimize::constant_fold", "analysis::analyze_copy_costs", "span_merge::append_span_to_node",
        ):
            with self.subTest(imported=imported):
                compile_rejected(
                    "chelis-ir", f"use chelis_ir::{imported};\n", expected="unresolved import",
                )
        compile_rejected(
            "chelis-ir", "use chelis_ir::verify::verify;\n", expected="unresolved import",
        )


    def test_baseline_is_the_current_immutable_source(self) -> None:
        self.assertEqual(BASELINE_REVISION, "3ca99faa7a46158eb6a0d90c2c812a7e3a5f44c1")

    def test_snapshot_requires_one_nonempty_record(self) -> None:
        from no_std_core_support import OracleFailure
        for text in ("", "DAG_SNAPSHOT {}", "DAG_SNAPSHOT []", "DAG_SNAPSHOT bad",
                     'DAG_SNAPSHOT {"value":1}\nDAG_SNAPSHOT {"value":1}'):
            with self.subTest(text=text), self.assertRaises(OracleFailure):
                snapshot_record(text, "DAG_SNAPSHOT")
        self.assertEqual(snapshot_record('DAG_SNAPSHOT {"value":1}', "DAG_SNAPSHOT"), {"value": 1})

    def test_cost_evidence_rejects_missing_scales_and_invalid_counters(self) -> None:
        from no_std_core_support import OracleFailure
        for text in ("", "DAG_COST {}", 'DAG_COST {"nodes":64}', "DAG_COST bad"):
            with self.subTest(text=text), self.assertRaises(OracleFailure):
                inspect_costs(text)

    def test_cost_comparison_rejects_each_pass_regression(self) -> None:
        from no_std_core_support import OracleFailure
        original = dict.fromkeys(("nodes", "ad_nodes", "fused_nodes", "ownership_nodes", "stack_bytes",
            "allocations", "allocated_bytes", "ad_allocations", "ad_bytes", "fusion_allocations",
            "fusion_bytes", "ownership_allocations", "ownership_bytes"), 64)
        for counter in original:
            changed = dict(original)
            changed[counter] += 1
            with self.subTest(counter=counter), self.assertRaises(OracleFailure):
                compare_costs([original], [changed])


    def test_source_owner_receipt_needs_every_destination_and_failure_lane(self) -> None:
        from no_std_core_support import OracleFailure
        complete = {field: list(SOURCE_MODULES) for field in ("canonical", "missing", "empty", "duplicate")}
        inspect_source_owners(complete)
        for field in complete:
            omitted = dict(complete)
            del omitted[field]
            with self.subTest(field=field, control="missing lane"), self.assertRaises(OracleFailure):
                inspect_source_owners(omitted)
            for module in SOURCE_MODULES:
                incomplete = dict(complete)
                incomplete[field] = [name for name in SOURCE_MODULES if name != module]
                with self.subTest(field=field, module=module), self.assertRaises(OracleFailure):
                    inspect_source_owners(incomplete)


if __name__ == "__main__":
    unittest.main()
