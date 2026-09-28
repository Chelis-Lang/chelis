"""A discovered numeric leaf must have exactly one executed final authority."""

from dataclasses import replace
import unittest

from capacity_census_graph import GraphError, Leaf


class WireAuthorityPartition(unittest.TestCase):
    def test_exact_bijection_rejects_missing_duplicate_and_surplus_authority(self):
        from capacity_census_wire_authority import LeafClassification, reconcile_leaves

        source = Leaf("schema::Span.offset", "u64")
        axis = Leaf("schema::Shape.axis", "i32")
        transport = LeafClassification(
            source, "TaggedTransport", "source-byte-coordinate"
        )
        operation = LeafClassification(axis, "NumericOperation", "[05-OP-7]")
        leaves = (source, axis)
        self.assertEqual(
            set(reconcile_leaves(leaves, (transport, operation))),
            {transport, operation},
        )
        for classifications in (
            (transport,),
            (transport, operation, transport),
            (transport, operation, replace(transport, leaf=Leaf("other", "u64"))),
        ):
            with self.assertRaisesRegex(GraphError, "authority.*bijection"):
                reconcile_leaves(leaves, classifications)

    def test_tagging_arbitrary_numeric_data_does_not_make_it_source_metadata(self):
        from capacity_census_wire_authority import LeafClassification, reconcile_leaves

        source = Leaf("schema::Span.offset", "u64")
        transport = LeafClassification(
            source, "TaggedTransport", "source-byte-coordinate"
        )
        for arbitrary in (
            Leaf("schema::Metadata::Value.value", "f64"),
            Leaf("schema::Span.numeric_value", "f64"),
            Leaf("schema::Span.offset", "f64"),
            Leaf("schema::WireRtDim::InputAxis.extent", "u64"),
        ):
            with self.subTest(leaf=arbitrary):
                with self.assertRaisesRegex(GraphError, "authority.*bijection"):
                    reconcile_leaves((source, arbitrary), (transport,))

    def test_legacy_or_nonnumeric_dispositions_cannot_admit_a_numeric_leaf(self):
        from capacity_census_wire_authority import LeafClassification, reconcile_leaves

        leaf = Leaf("schema::Metadata::Value.value", "f64")
        for disposition in (
            "Nonnumeric",
            "grandfather",
            "successor-override",
            "permanent-disposition",
            "integer-plumbing",
        ):
            candidate = LeafClassification(leaf, disposition, "reviewed")
            with self.subTest(disposition=disposition):
                with self.assertRaisesRegex(GraphError, "nonfinal.*authority"):
                    reconcile_leaves((leaf,), (candidate,))


if __name__ == "__main__":
    unittest.main()
