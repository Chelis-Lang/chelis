from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import runtime_bundle_oracle as oracle
import check_runtime_archive_lookups as runtime_guard


class RuntimeBundleOracleContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_prerequisite_executable_invokes_symlink_as_cargo(self) -> None:
        tools = self.root / "tools with spaces"
        tools.mkdir()
        rustup = tools / "rustup"
        rustup.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "from pathlib import Path\n"
            "print(Path(sys.argv[0]).name, *sys.argv[1:])\n",
            encoding="utf-8",
        )
        rustup.chmod(0o755)
        (tools / "cargo").symlink_to(rustup)
        (tools / "python3").symlink_to(sys.executable)

        cargo = oracle.require_executable("cargo", search_path=str(tools))
        result = subprocess.run(
            [cargo, "build", "--locked"],
            capture_output=True,
            check=True,
            text=True,
            env={**os.environ, "PATH": str(tools)},
        )
        self.assertEqual(result.stdout, "cargo build --locked\n")
        self.assertEqual(result.stderr, "")

    def test_staging_receipt_binds_archive_and_header_bytes(self) -> None:
        archive = self.root / oracle.ARCHIVE_FILE_NAME
        archive.write_bytes(b"runtime-A")
        header = self.root / "chelis_runtime.h"
        header.write_bytes(b"header-A")
        receipt = {
            "schema": oracle.STAGING_RECEIPT_SCHEMA,
            "archive": oracle.ARCHIVE_FILE_NAME,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "headers": {header.name: hashlib.sha256(header.read_bytes()).hexdigest()},
            "mode": "development",
            "chelis_version": "0.1.0",
        }
        receipt_path = self.root / "receipt.json"
        oracle.atomic_write_json(receipt_path, receipt)

        verified = oracle.read_staging_receipt(
            receipt_path,
            archive,
            expected_mode="development",
            headers_dir=self.root,
        )

        self.assertEqual(verified["archive_sha256"], hashlib.sha256(b"runtime-A").hexdigest())
        self.assertEqual(verified["headers"], receipt["headers"])

    def test_crossed_archive_and_replaced_header_are_rejected(self) -> None:
        archive = self.root / oracle.ARCHIVE_FILE_NAME
        archive.write_bytes(b"runtime-B")
        header = self.root / "chelis_runtime.h"
        header.write_bytes(b"header-B")
        receipt = {
            "schema": oracle.STAGING_RECEIPT_SCHEMA,
            "archive": oracle.ARCHIVE_FILE_NAME,
            "archive_sha256": hashlib.sha256(b"runtime-A").hexdigest(),
            "headers": {header.name: hashlib.sha256(b"header-A").hexdigest()},
            "mode": "sealed",
            "chelis_version": "0.1.0",
        }
        receipt_path = self.root / "receipt.json"
        oracle.atomic_write_json(receipt_path, receipt)

        with self.assertRaises(oracle.OracleFailure):
            oracle.read_staging_receipt(
                receipt_path,
                archive,
                expected_mode="sealed",
                headers_dir=self.root,
            )

        archive.write_bytes(b"runtime-A")
        receipt["archive_sha256"] = hashlib.sha256(b"runtime-A").hexdigest()
        oracle.atomic_write_json(receipt_path, receipt)
        with self.assertRaises(oracle.OracleFailure):
            oracle.read_staging_receipt(
                receipt_path,
                archive,
                expected_mode="sealed",
                headers_dir=self.root,
            )

    def test_python_manifest_must_match_verified_staging_bytes(self) -> None:
        archive = self.root / oracle.ARCHIVE_FILE_NAME
        archive.write_bytes(b"runtime-A")
        header = self.root / "chelis_runtime.h"
        header.write_bytes(b"header-A")
        digest_a = hashlib.sha256(archive.read_bytes()).hexdigest()
        receipt = {
            "schema": oracle.STAGING_RECEIPT_SCHEMA,
            "archive": archive.name,
            "archive_sha256": digest_a,
            "headers": {header.name: hashlib.sha256(header.read_bytes()).hexdigest()},
            "mode": "sealed",
            "chelis_version": "0.1.0",
        }
        oracle.atomic_write_json(self.root / oracle.RECEIPT_FILE_NAME, receipt)
        self.assertEqual(
            oracle.verify_python_staging(self.root, digest_a, expected_mode="sealed")[2],
            receipt,
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "differs from its persisted runtime digest"):
            oracle.verify_python_staging(
                self.root, hashlib.sha256(b"runtime-B").hexdigest(), expected_mode="sealed"
            )
        archive.write_bytes(b"runtime-B")
        with self.assertRaisesRegex(oracle.OracleFailure, "digest mismatch"):
            oracle.verify_python_staging(self.root, digest_a, expected_mode="sealed")

    def test_both_package_outputs_and_crossed_package_bind_the_export(self) -> None:
        receipt = {
            "archive_sha256": hashlib.sha256(b"runtime-A").hexdigest(),
            "headers": {"chelis_runtime.h": hashlib.sha256(b"header-A").hexdigest()},
        }
        packages = [self.root / name for name in ("chelis", "chelis-runtime")]
        for package in packages:
            (package / "lib").mkdir(parents=True)
            (package / "include").mkdir()
            (package / "lib" / oracle.ARCHIVE_FILE_NAME).write_bytes(b"runtime-A")
            (package / "include" / "chelis_runtime.h").write_bytes(b"header-A")
            oracle.verify_package_runtime(package, receipt, label=package.name)
        for package in packages:
            archive = package / "lib" / oracle.ARCHIVE_FILE_NAME
            archive.write_bytes(b"runtime-B")
            with self.assertRaisesRegex(oracle.OracleFailure, "digest mismatch"):
                oracle.verify_package_runtime(package, receipt, label=package.name)
            archive.write_bytes(b"runtime-A")

    def test_crossed_package_from_read_only_store_copy_rejects_foreign_archive(self) -> None:
        package = self.root / "store-package"
        lib_dir = package / "lib"
        lib_dir.mkdir(parents=True)
        include_dir = package / "include"
        include_dir.mkdir()
        baseline = lib_dir / oracle.ARCHIVE_FILE_NAME
        baseline.write_bytes(b"runtime-A")
        baseline.chmod(0o444)
        (include_dir / "chelis_runtime.h").write_bytes(b"header-A")
        receipt = {
            "archive_sha256": hashlib.sha256(b"runtime-A").hexdigest(),
            "headers": {"chelis_runtime.h": hashlib.sha256(b"header-A").hexdigest()},
        }
        oracle.verify_package_runtime(package, receipt, label="store baseline")
        lib_dir.chmod(0o555)
        self.addCleanup(lambda: lib_dir.chmod(0o755))

        replacement = self.root / "foreign-runtime.a"
        replacement.write_bytes(b"runtime-B")
        crossed_dir = self.root / "crossed-package"
        crossed = oracle.cross_package_runtime(package, crossed_dir, replacement)
        self.assertEqual(lib_dir.stat().st_mode & 0o777, 0o555)
        self.assertEqual(baseline.read_bytes(), b"runtime-A")
        self.assertEqual(crossed.read_bytes(), b"runtime-B")
        self.assertTrue(crossed.parent.stat().st_mode & 0o200)
        with self.assertRaisesRegex(oracle.OracleFailure, "digest mismatch"):
            oracle.verify_package_runtime(crossed_dir, receipt, label="crossed package")

    def test_printed_native_link_uses_only_the_exact_staged_archive(self) -> None:
        archive = self.root / oracle.ARCHIVE_FILE_NAME
        command = f"cc test.c {archive} -o output"
        build = {"archive": archive, "stdout": f"Compile: {command}\n".encode()}
        self.assertEqual(oracle.printed_link_argv(build, label="host-stage-hip")[-3:], [str(archive), "-o", "output"])
        build["stdout"] = f"Compile: {command} -lchelis_runtime\n".encode()
        with self.assertRaisesRegex(oracle.OracleFailure, "searches for the runtime"):
            oracle.printed_link_argv(build, label="host-stage-hip")
        build["stdout"] = b"Compile: cc test.c -o output\n"
        with self.assertRaisesRegex(oracle.OracleFailure, "one exact staged archive path"):
            oracle.printed_link_argv(build, label="host-stage-hip")

    def test_reviewed_lookup_blocks_an_otherwise_clean_guard(self) -> None:
        oracle.require_no_reviewed_lookups(())
        survivor = runtime_guard.Row(
            "tests/manual.rs",
            "linker-search",
            ('cmd.arg("-lchelis_runtime");',),
            "lookup",
            "manual gate still searches",
            "chelis#1354",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "tests/manual.rs \\[linker-search\\]"):
            oracle.require_no_reviewed_lookups((survivor,))


    def test_command_recorder_captures_real_process_result_and_streams(self) -> None:
        evidence = oracle.EvidenceRun(self.root / "run", "a" * 40)
        argv = [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('out'); sys.stderr.write('err')",
        ]

        result = evidence.run("stream-probe", argv, cwd=self.root)

        self.assertEqual(result.returncode, 0)
        record = evidence.receipt["commands"][0]
        self.assertEqual(record["argv"], argv)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(Path(evidence.root / record["stdout"]["path"]).read_bytes(), b"out")
        self.assertEqual(Path(evidence.root / record["stderr"]["path"]).read_bytes(), b"err")
        oracle.validate_receipt(evidence.receipt)

    def test_receipt_rejects_command_artifact_digest_drift(self) -> None:
        evidence = oracle.EvidenceRun(self.root / "run", "d" * 40)
        evidence.run("stream-probe", [sys.executable, "-c", "print('recorded')"], cwd=self.root)
        stdout_path = evidence.root / evidence.receipt["commands"][0]["stdout"]["path"]
        stdout_path.write_bytes(b"mutated!\n")

        with self.assertRaisesRegex(oracle.OracleFailure, "digest does not match"):
            oracle.validate_receipt(evidence.receipt)

    def test_source_mutation_restores_file_permissions(self) -> None:
        path = self.root / "source.rs"
        path.write_bytes(b"before\n")
        path.chmod(0o640)

        with oracle.mutated_source(path, b"before", b"after"):
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)

        self.assertEqual(path.stat().st_mode & 0o777, 0o640)

    def test_unexpected_command_status_is_recorded_as_failure(self) -> None:
        evidence = oracle.EvidenceRun(self.root / "run", "b" * 40)

        with self.assertRaises(oracle.OracleFailure):
            evidence.run(
                "nonzero-probe",
                [sys.executable, "-c", "raise SystemExit(7)"],
                cwd=self.root,
            )

        self.assertEqual(evidence.receipt["commands"][0]["returncode"], 7)
        self.assertEqual(evidence.receipt["commands"][0]["status"], "failed")
        self.assertEqual(evidence.receipt["overall"], "failed")

    def test_missing_mandatory_executable_is_not_a_skip(self) -> None:
        with self.assertRaisesRegex(oracle.OracleFailure, "required executable"):
            oracle.require_executable("oracle-command-that-is-not-installed", search_path=str(self.root))

    def test_exact_source_mutation_is_discriminating_and_restored(self) -> None:
        path = self.root / "source.rs"
        original = b"prefix::before::suffix\n"
        path.write_bytes(original)

        with oracle.mutated_source(path, b"before", b"after") as mutation:
            self.assertEqual(path.read_bytes(), b"prefix::after::suffix\n")
            self.assertNotEqual(mutation["original_sha256"], mutation["mutated_sha256"])

        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(oracle.OracleFailure, "exactly once"):
            oracle.replace_exact(b"before before", b"before", b"after", path=path)

    def test_incomplete_rows_cannot_claim_oracle_success(self) -> None:
        receipt = oracle.new_receipt("c" * 40, self.root)
        receipt["overall"] = "passed"
        receipt["rows"][oracle.ORACLE_ROWS[0]] = {"status": "passed", "evidence": {}}

        with self.assertRaisesRegex(oracle.OracleFailure, "mandatory rows"):
            oracle.validate_receipt(receipt)


    def test_repository_runtime_guard_is_clean_and_rejects_a_planted_lookup(self) -> None:
        clean_errors = runtime_guard.check(ROOT)
        self.assertEqual(clean_errors, [])

        with tempfile.TemporaryDirectory(prefix="runtime-bundle-guard-", dir=ROOT) as probe_dir:
            directory = Path(probe_dir)
            probe = directory / "lookup.rs"
            probe.write_text(
                f'let candidate = build_dir.join("{oracle.ARCHIVE_FILE_NAME}");\n',
                encoding="utf-8",
            )
            planted_errors = runtime_guard.check(ROOT)
            self.assertTrue(
                any(str(probe.relative_to(ROOT)) in error and "unreviewed line" in error for error in planted_errors),
                planted_errors,
            )


if __name__ == "__main__":
    unittest.main()
