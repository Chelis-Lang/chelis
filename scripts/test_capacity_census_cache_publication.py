"""Spec/10 durable publication selection/receipt negative controls."""

import dataclasses
import copy
import json
from pathlib import Path
import unittest
import tempfile
import shutil
import os
import stat

from capacity_census_cache_publication import (
    COMPILE_CASES,
    OWNERS,
    REQUIRED_DEPENDENCIES,
    seal_compiled_artifacts,
    select_compiled_artifacts,
    closed_payload_owners,
    RUNTIME_CASES,
    TestExecution,
    validate_runtime_receipt,
    CachePublicationError,
    CompileOutcome,
    validate_compile_outcomes,
    validate_fixture_inventory,
)
from capacity_census_wire_calls import RlibIdentity, parse_rlib_identity

ROOT = Path(__file__).resolve().parent.parent
DEPS = Path("/workspace/target/agents/census/debug/deps")
LINKED = sorted(REQUIRED_DEPENDENCIES - {"chelis_compiler_api"})


def identity(crate_hash, dependencies=()):
    return RlibIdentity(crate_hash, "0" * 16, tuple(dependencies))


API = identity(
    "api", [("std", "std")] + [(name, f"{name}-linked") for name in LINKED]
)
IDENTITIES = {
    "libchelis_compiler_api-a1.rlib": API,
    **{f"lib{name}-a1.rlib": identity(f"{name}-linked") for name in LINKED},
    # The copies a build script links: same names, other crate hashes.
    "libchelis_ir-b2.rlib": identity("chelis_ir-host"),
    "libserde-b2.rlib": identity("serde-host"),
    # A second copy carrying the recorded hash: a genuine collision.
    "libchelis_ir-c3.rlib": identity("chelis_ir-linked"),
}


def library_row(name, digest):
    return {
        "reason": "compiler-artifact",
        "target": {"name": name, "kind": ["lib"]},
        "filenames": [
            str(DEPS / f"lib{name}-{digest}.rlib"),
            str(DEPS / f"lib{name}-{digest}.rmeta"),
        ],
        "executable": None,
    }


def binary_row():
    return {
        "reason": "compiler-artifact",
        "target": {"name": "cache_wire_compatibility", "kind": ["test"]},
        "filenames": [str(DEPS / "cache_wire_compatibility-abc")],
        "executable": str(DEPS / "cache_wire_compatibility-abc"),
    }


def linked_rows():
    return [library_row(name, "a1") for name in sorted(REQUIRED_DEPENDENCIES)] + [
        binary_row()
    ]


def select(rows, identities=IDENTITIES):
    stream = "\n".join(json.dumps(row) for row in rows)
    return select_compiled_artifacts(stream, lambda path: identities[path.name])


# `rustc -Zls=root` output for a library, abridged to three dependencies.
LS_ROOT = """Crate info:
name chelis_compiler_api-d17a2d574173dcf9
hash fe653f5d52af1a2f4badc63d28ab0dfe stable_crate_id StableCrateId(13098500836873532018)
proc_macro false
triple aarch64-apple-darwin
=External Dependencies=
1 std-61f27fb94867ea3a hash 5977b8c3d0f76e94603d31fc33114077 host_hash None kind Unconditional public
20 chelis_ir-bfcc383cf7c624ca hash 000e0cd0d958860597a138e311b53057 host_hash None kind Unconditional public
186 pest_derive-de782af1613c0c71 hash a9a5ef2748b1251d09bdfbf13e8f1930 host_hash Some(Svh(1)) kind MacrosOnly public


"""


class CompiledArtifactSelection(unittest.TestCase):
    def test_build_script_copy_is_skipped_and_the_recorded_copy_is_linked(self):
        rows = linked_rows()
        rows.insert(0, library_row("chelis_ir", "b2"))
        rows.append(library_row("serde", "b2"))
        artifacts, binary = select(rows)
        self.assertEqual(set(artifacts), REQUIRED_DEPENDENCIES)
        for name in REQUIRED_DEPENDENCIES:
            self.assertEqual(artifacts[name], str(DEPS / f"lib{name}-a1.rlib"))
        self.assertEqual(binary, str(DEPS / "cache_wire_compatibility-abc"))

    def test_two_copies_carrying_the_recorded_hash_are_refused(self):
        rows = linked_rows() + [library_row("chelis_ir", "c3")]
        with self.assertRaisesRegex(
            CachePublicationError, "ambiguous cache dependency chelis_ir"
        ):
            select(rows)

    def test_no_copy_carrying_the_recorded_hash_is_refused(self):
        without = [r for r in linked_rows() if r["target"]["name"] != "chelis_ir"]
        for rows in (without, without + [library_row("chelis_ir", "b2")]):
            with self.assertRaisesRegex(
                CachePublicationError, "missing cache dependency chelis_ir"
            ):
                select(rows)

    def test_the_api_rlib_must_be_unique(self):
        without = [
            r for r in linked_rows() if r["target"]["name"] != "chelis_compiler_api"
        ]
        twice = linked_rows() + [library_row("chelis_compiler_api", "a1")]
        for rows in (without, twice):
            with self.assertRaisesRegex(
                CachePublicationError,
                "missing or ambiguous cache dependency chelis_compiler_api",
            ):
                select(rows)

    def test_a_dependency_the_api_records_other_than_once_is_refused(self):
        for dependencies in (
            [d for d in API.dependencies if d[0] != "chelis_ir"],
            API.dependencies + (("chelis_ir", "chelis_ir-other"),),
        ):
            identities = {
                **IDENTITIES,
                "libchelis_compiler_api-a1.rlib": identity("api", dependencies),
            }
            with self.assertRaisesRegex(
                CachePublicationError, "missing or ambiguous recorded dependency chelis_ir"
            ):
                select(linked_rows(), identities)

    def test_the_test_binary_must_be_unique(self):
        for rows, message in (
            (linked_rows() + [binary_row()], "ambiguous cache runtime binary"),
            (linked_rows()[:-1], "missing cache runtime binary"),
        ):
            with self.assertRaisesRegex(CachePublicationError, message):
                select(rows)

    def test_a_row_with_two_rlibs_is_refused(self):
        rows = linked_rows()
        serde = next(r for r in rows if r["target"]["name"] == "serde")
        serde["filenames"].append(str(DEPS / "libserde-b2.rlib"))
        with self.assertRaisesRegex(
            CachePublicationError, "missing or ambiguous serde rlib"
        ):
            select(rows)


class RlibIdentityParsing(unittest.TestCase):
    def test_reads_own_hash_and_every_recorded_dependency(self):
        parsed = parse_rlib_identity(LS_ROOT)
        self.assertEqual(parsed.crate_hash, "fe653f5d52af1a2f4badc63d28ab0dfe")
        self.assertEqual(parsed.stable_crate_id, f"{13098500836873532018:016x}")
        self.assertEqual(
            parsed.dependencies,
            (
                ("std", "5977b8c3d0f76e94603d31fc33114077"),
                ("chelis_ir", "000e0cd0d958860597a138e311b53057"),
                ("pest_derive", "a9a5ef2748b1251d09bdfbf13e8f1930"),
            ),
        )
        self.assertEqual(
            parsed.dependency_hash("chelis_ir"), "000e0cd0d958860597a138e311b53057"
        )

    def test_refuses_listings_it_cannot_read_exactly(self):
        own = LS_ROOT
        for text, message in (
            (own.replace("=External Dependencies=\n", ""), "dependency listing"),
            (own.replace(" kind Unconditional public\n", " kind Unconditional\n", 1),
             "unrecognized artifact dependency line"),
            (own + own, "stable crate identity"),
            (own.replace("stable_crate_id", "crate_id"), "stable crate identity"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_rlib_identity(text)


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

    def test_unexpected_library_compile_failure_exposes_bounded_rustc_error(self):
        outcomes = tuple(
            CompileOutcome(
                case.name,
                1 if case.name == "library" else (0 if case.success else 1),
                "error[E0433]: failed to resolve: undeclared crate `chelis_types`\n"
                + "detail\n" * 5000 if case.name == "library" else
                ((case.error + " " + case.diagnostic) if case.error else ""),
                "source",
                ("rustc",),
            )
            for case in COMPILE_CASES
        )
        with self.assertRaises(CachePublicationError) as failure:
            validate_compile_outcomes(outcomes)
        message = str(failure.exception)
        self.assertIn("wrong compile outcome: library", message)
        self.assertIn("error[E0433]", message)
        self.assertIn("undeclared crate `chelis_types`", message)
        self.assertLess(len(message), 4500)

    def test_unexpected_compile_failure_reports_first_plain_error(self):
        outcomes = tuple(
            CompileOutcome(
                case.name,
                1 if case.name == "library" else (0 if case.success else 1),
                (
                    "warning: before the errors\n"
                    "error: cannot find derive macro `DoesNotExist`\n"
                    "error[E0433]: undeclared crate `chelis_types`\n"
                ) if case.name == "library" else
                ((case.error + " " + case.diagnostic) if case.error else ""),
                "source",
                ("rustc",),
            )
            for case in COMPILE_CASES
        )
        with self.assertRaises(CachePublicationError) as failure:
            validate_compile_outcomes(outcomes)
        message = str(failure.exception)
        self.assertIn("error: cannot find derive macro `DoesNotExist`", message)
        self.assertLess(
            message.index("DoesNotExist"), message.index("chelis_types")
        )

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
