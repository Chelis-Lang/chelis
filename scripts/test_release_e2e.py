"""Controls for scripts/release_e2e.py: the verdicts a release E2E run reports."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import release_e2e as e2e  # noqa: E402

SOURCE_SHA = "0" * 40
STORE_LOADER = "/nix/store/0000-glibc-2.40/lib/ld-linux-x86-64.so.2"


def quiet(argv: list[str]) -> tuple[int, str]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = e2e.main(argv)
    return code, output.getvalue()


def write_elf(
    path: Path, *, interpreter: str | None, needed: list[str], runpath: list[str]
) -> None:
    """A minimal x86-64 ELF whose one PT_LOAD maps the file at address 0."""
    strings = b"\0"
    offsets: dict[str, int] = {}
    for name in [*needed, *runpath]:
        offsets[name] = len(strings)
        strings += name.encode() + b"\0"
    phnum = 2 if interpreter is None else 3
    cursor = 64 + 56 * phnum
    interp = b"" if interpreter is None else interpreter.encode() + b"\0"
    interp_offset = cursor
    strtab_offset = interp_offset + len(interp)
    dynamic_offset = (strtab_offset + len(strings) + 7) & ~7
    entries = [(1, offsets[name]) for name in needed]
    entries += [(29, offsets[name]) for name in runpath]
    entries += [(5, strtab_offset), (0, 0)]
    dynamic = b"".join(struct.pack("<qQ", tag, value) for tag, value in entries)
    size = dynamic_offset + len(dynamic)
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", header, 0x10, 3, 0x3E, 1)
    struct.pack_into("<Q", header, 0x20, 64)
    struct.pack_into("<HHH", header, 0x34, 64, 56, phnum)

    def segment(kind: int, offset: int, length: int) -> bytes:
        return struct.pack(
            "<IIQQQQQQ", kind, 4, offset, offset, offset, length, length, 8
        )

    segments = [segment(1, 0, size)]
    if interpreter is not None:
        segments.append(segment(3, interp_offset, len(interp)))
    segments.append(segment(2, dynamic_offset, len(dynamic)))
    blob = bytes(header) + b"".join(segments) + interp + strings
    path.write_bytes(blob + b"\0" * (dynamic_offset - len(blob)) + dynamic)


class RunTests(unittest.TestCase):
    def test_missing_assets_fail_the_run_and_block_every_dependent_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            evidence = root / "evidence"
            code, _ = quiet(
                ["run", "--tag", "v0.0.1", "--source-sha", SOURCE_SHA]
                + ["--assets", str(root / "assets"), "--gates", str(root)]
                + ["--evidence", str(evidence)]
            )
            report = json.loads((evidence / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "failed")
        statuses = {step["name"]: step["status"] for step in report["steps"]}
        self.assertEqual(list(statuses), list(e2e.STEPS))
        self.assertEqual(statuses.pop("assets"), "failed")
        self.assertEqual(set(statuses.values()), {"blocked"})


class SummarizeTests(unittest.TestCase):
    @staticmethod
    def write(root: Path, directory: str, report: dict[str, object]) -> None:
        path = root / directory / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report), encoding="utf-8")

    @staticmethod
    def host(label: str) -> dict[str, object]:
        return {
            "schema": e2e.SCHEMA,
            "label": label,
            "status": "passed",
            "host": {"os": "test"},
            "steps": [{"name": "assets", "status": "passed", "detail": label}],
        }

    def test_the_canary_report_inside_a_host_evidence_is_not_a_host(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, "host", self.host("host"))
            canary = {
                "schema": "chelis-installed-callable-smoke-v1",
                "status": "failed",
            }
            self.write(root, "host/canary", canary)
            _, table = quiet(["summarize", directory])

        rows = [line for line in table.splitlines() if line.startswith("| ")][1:]
        self.assertEqual(len(rows), 1)


class WalkthroughTests(unittest.TestCase):
    def test_every_shell_command_runs_against_the_first_program(self) -> None:
        markdown = (
            "Intro.\n\n"
            "```chelis-surf\n"
            "def twice(x: i64) -> i64 = x + x\n"
            "result = twice(21)\n"
            "```\n\n"
            "```sh\n"
            "# a comment, not a command\n"
            "chelis eval --file app.ch\n"
            "\n"
            "chelis build app.ch \\\n"
            "  --output out/\n"
            "./out/app\n"
            "```\n\n"
            "```chelis-surf\n"
            "result = 0\n"
            "```\n\n"
            "```text\n"
            "chelis not-a-command\n"
            "```\n"
        )

        program, commands = e2e.walkthrough(markdown)

        self.assertEqual(
            program, "def twice(x: i64) -> i64 = x + x\nresult = twice(21)\n"
        )
        self.assertEqual(
            commands,
            [
                "chelis eval --file app.ch",
                "chelis build app.ch --output out/",
                "./out/app",
            ],
        )


class EvalProbeTests(unittest.TestCase):
    def test_only_the_softmax_of_the_relu_is_accepted(self) -> None:
        printed = (
            "result = tensor(shape=[3], data=[0.21194156, 0.21194156, 0.57611686])"
        )
        self.assertEqual(
            e2e.probe_result(printed), (0.21194156, 0.21194156, 0.57611686)
        )
        without_relu = (
            "result = tensor(shape=[3], data=[0.09003057, 0.24472847, 0.66524096])"
        )
        self.assertIsNone(e2e.probe_result(without_relu))


class LinkageTests(unittest.TestCase):
    def test_a_static_build_names_no_loader_and_no_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_elf(root / "static", interpreter=None, needed=[], runpath=[])
            linkage = e2e.elf_dynamic(root / "static")
            self.assertEqual(
                e2e.elf_verdict("chelis", linkage, static=True),
                "chelis: static, no program interpreter or shared library",
            )
            for name, interpreter in (("loader", e2e.LINUX_LOADER), ("library", None)):
                with self.subTest(name=name):
                    path = root / name
                    write_elf(
                        path, interpreter=interpreter, needed=["libc.so.6"], runpath=[]
                    )
                    with self.assertRaisesRegex(e2e.StepFailed, "chelis is not static"):
                        e2e.elf_verdict("chelis", e2e.elf_dynamic(path), static=True)

    def test_a_dynamic_build_uses_the_system_loader_without_store_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libraries = ["libc.so.6", "libm.so.6"]
            write_elf(
                root / "dynamic",
                interpreter=e2e.LINUX_LOADER,
                needed=libraries,
                runpath=[],
            )
            self.assertEqual(
                e2e.elf_verdict(
                    "chelis", e2e.elf_dynamic(root / "dynamic"), static=False
                ),
                f"chelis: {e2e.LINUX_LOADER}, needs libc.so.6, libm.so.6",
            )
            cases = (
                ("store-loader", STORE_LOADER, [], "loads through /nix/store/"),
                (
                    "store-runpath",
                    e2e.LINUX_LOADER,
                    ["/nix/store/0000-gcc/lib"],
                    "store run paths",
                ),
            )
            for name, interpreter, runpath, refusal in cases:
                with self.subTest(name=name):
                    path = root / name
                    write_elf(
                        path, interpreter=interpreter, needed=libraries, runpath=runpath
                    )
                    with self.assertRaisesRegex(e2e.StepFailed, refusal):
                        e2e.elf_verdict("chelis", e2e.elf_dynamic(path), static=False)

    def test_a_macos_build_loads_only_system_libraries(self) -> None:
        otool = (
            "/tmp/chelis:\n"
            "\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0)\n"
            "\t/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"
            " (compatibility version 1.0.0)\n"
            "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n"
        )
        self.assertEqual(
            e2e.macho_verdict("chelis", otool), "chelis: 3 system libraries"
        )
        for library in (
            "/opt/homebrew/opt/gmp/lib/libgmp.10.dylib",
            "@rpath/libcvc5.1.dylib",
        ):
            with self.subTest(library=library):
                loaded = otool + f"\t{library} (compatibility version 1.0.0)\n"
                with self.assertRaisesRegex(
                    e2e.StepFailed, f"chelis loads {re.escape(library)}"
                ):
                    e2e.macho_verdict("chelis", loaded)


if __name__ == "__main__":
    unittest.main()
