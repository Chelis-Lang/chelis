"""Spec/10 durable publication selection/receipt negative controls."""

import dataclasses
import copy
from pathlib import Path
import unittest
import tempfile
import shutil
import os
import stat

from capacity_census_cache_publication import (
    COMPILE_CASES,
    OWNERS,
    seal_compiled_artifacts,
    closed_payload_owners,
    RUNTIME_CASES,
    TestExecution,
    validate_runtime_receipt,
    CachePublicationError,
    CompileOutcome,
    validate_compile_outcomes,
    validate_fixture_inventory,
)

ROOT = Path(__file__).resolve().parent.parent


class CachePublicationSelection(unittest.TestCase):
    def test_compiled_artifacts_are_sealed_against_later_cargo_cache_churn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "target/debug/deps"
            cache.mkdir(parents=True)
            original = cache / "libbincode-example.rlib"
            original.write_bytes(b"compiled witness")

            sealed = seal_compiled_artifacts(
                root / "target/cache-publication",
                {"bincode": str(original)},
            )
            sealed_path = Path(sealed["bincode"])
            self.assertNotEqual(sealed_path, original)
            self.assertEqual(sealed_path.read_bytes(), b"compiled witness")

            original.write_bytes(b"later legitimate cargo build")
            self.assertEqual(sealed_path.read_bytes(), b"compiled witness")

    def test_resealing_a_read_only_cached_artifact_replaces_the_earlier_copy(self):
        # Without reflinks Kache restores Cargo outputs as hardlinks to its
        # read-only store blobs, and copy2 carries that mode to the sealed copy.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blob = root / "kache/store/blob"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"compiled witness")
            blob.chmod(0o444)
            restored = root / "target/debug/deps/libbincode-example.rlib"
            restored.parent.mkdir(parents=True)
            os.link(blob, restored)

            publication = root / "target/cache-publication"
            seal_compiled_artifacts(publication, {"bincode": str(restored)})
            sealed = seal_compiled_artifacts(publication, {"bincode": str(restored)})

            self.assertEqual(Path(sealed["bincode"]).read_bytes(), b"compiled witness")
            self.assertEqual(blob.read_bytes(), b"compiled witness")
            self.assertEqual(stat.S_IMODE(blob.stat().st_mode) & 0o222, 0)

    def test_compiled_artifact_seal_rejects_ambiguous_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "one/libsame.rlib"
            second = root / "two/libsame.rlib"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            with self.assertRaisesRegex(
                CachePublicationError, "duplicate compiled artifact filename"
            ):
                seal_compiled_artifacts(
                    root / "receipt",
                    {"one": str(first), "two": str(second)},
                )

    def test_fixture_inventory_accepts_exact_owned_sources_and_rejects_drift(self):
        source = ROOT / "crates/chelis-compiler-api/tests/fixtures"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "crates/chelis-compiler-api/tests/fixtures"
            for name in ("cache_publication", "cache_wire_v3"):
                shutil.copytree(source / name, target / name)
            validate_fixture_inventory(root)
            for name in ("cache_publication", "cache_wire_v3"):
                extra = target / name / "nested/undriven.rs"
                extra.parent.mkdir()
                extra.write_text("fn unnoticed() {}\n")
                with self.assertRaisesRegex(CachePublicationError, "fixture inventory"):
                    validate_fixture_inventory(root)
                extra.unlink()
            omitted = target / "cache_publication/library.rs"
            contents = omitted.read_bytes()
            omitted.unlink()
            with self.assertRaisesRegex(CachePublicationError, "fixture inventory"):
                validate_fixture_inventory(root)
            omitted.write_bytes(contents)
            producer = target / "cache_wire_v3/producer.rs"
            producer.write_text(producer.read_text() + "// unrecorded change\n")
            with self.assertRaisesRegex(CachePublicationError, "producer source hash"):
                validate_fixture_inventory(root)

    def test_exact_owners_have_companion_unowned_compile_rejections(self):
        self.assertEqual(
            {case.name for case in COMPILE_CASES if case.success},
            {"library", "stdlib"},
        )
        self.assertEqual(len(COMPILE_CASES), len({c.name for c in COMPILE_CASES}))
        for case in COMPILE_CASES:
            self.assertTrue((ROOT / case.fixture).is_file())
        outcomes = tuple(
            CompileOutcome(
                c.name,
                0 if c.success else 1,
                (c.error + " " + c.diagnostic) if c.error else "",
                "source",
                ("rustc",),
            )
            for c in COMPILE_CASES
        )
        validate_compile_outcomes(outcomes)
        for altered in (
            outcomes[:-1],
            outcomes + outcomes[:1],
            tuple(dataclasses.replace(x, returncode=0) for x in outcomes),
            tuple(
                dataclasses.replace(x, stderr="unrelated syntax error")
                for x in outcomes
            ),
        ):
            with self.assertRaises(CachePublicationError):
                validate_compile_outcomes(altered)

    def test_runtime_receipt_requires_exact_framework_selection(self):
        names = tuple(sorted(RUNTIME_CASES))
        receipt = TestExecution(("test_binary",), names, names, "0" * 64)
        validate_runtime_receipt(receipt)
        for changed in (
            dataclasses.replace(receipt, selected=names[:-1]),
            dataclasses.replace(receipt, executed=names[:-1]),
            dataclasses.replace(receipt, command=()),
            dataclasses.replace(receipt, output_sha256=""),
        ):
            with self.assertRaises(CachePublicationError):
                validate_runtime_receipt(changed)

    def test_actual_trait_inventory_rejects_missing_extra_or_generic_owners(self):
        document = {"format_version": 60, "index": {}, "paths": {}}
        for number, name in enumerate(("CachePayload", "Sealed")):
            base = number * 10
            document["index"][str(base)] = {
                "name": name,
                "inner": {"trait": {"implementations": [base + 1, base + 2]}},
            }
            for offset, owner in enumerate(OWNERS, 1):
                document["index"][str(base + offset)] = {
                    "inner": {
                        "impl": {
                            "generics": {"params": []},
                            "for": {"resolved_path": {"id": offset}},
                        }
                    }
                }
                document["paths"][str(offset)] = {"path": owner.split("::")}
        self.assertEqual(closed_payload_owners(document), OWNERS)
        for mutation in ("missing", "extra", "generic", "renamed"):
            changed = copy.deepcopy(document)
            if mutation == "missing":
                changed["index"]["0"]["inner"]["trait"]["implementations"].pop()
            elif mutation == "extra":
                changed["index"]["0"]["inner"]["trait"]["implementations"].append(1)
            elif mutation == "generic":
                changed["index"]["1"]["inner"]["impl"]["generics"]["params"] = [
                    {"name": "T"}
                ]
            else:
                changed["paths"]["1"]["path"][-1] = "RenamedCache"
            with self.assertRaises(CachePublicationError):
                closed_payload_owners(changed)


if __name__ == "__main__":
    unittest.main()
