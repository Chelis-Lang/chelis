from __future__ import annotations

import importlib.util
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).with_name("hash_order_determinism_oracle.py")
SPEC = importlib.util.spec_from_file_location("hash_order_determinism_oracle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ORACLE
SPEC.loader.exec_module(ORACLE)


class HashOrderTokenTripwireTests(unittest.TestCase):
    def assert_rejected(self, path: str, text: str) -> None:
        with self.assertRaises(ORACLE.HashOrderDeterminismFailure):
            ORACLE.validate_sources(
                {path: text}, allowed=(), approved_build_scripts=()
            )

    def test_rejects_library_target(self) -> None:
        self.assert_rejected("crates/demo/src/lib.rs", "type M = HashMap<String, i32>;")

    def test_rejects_non_default_feature_source(self) -> None:
        self.assert_rejected(
            "crates/demo/src/checkpoint_compile_probe.rs",
            "use std::collections::HashSet;",
        )

    def test_rejects_build_script(self) -> None:
        self.assert_rejected("crates/demo/build.rs", "use hashbrown::HashMap;")

    def test_rejects_test_target(self) -> None:
        self.assert_rejected("crates/demo/tests/order.rs", "FxHashMap::default();")

    def test_rejects_unlisted_disallowed_type_allowance(self) -> None:
        self.assert_rejected(
            "crates/demo/src/lib.rs",
            "#[allow(clippy::disallowed_types)]\nstruct Escape;",
        )

    def test_rejects_stale_allow_list_entry(self) -> None:
        allowed = (
            ORACLE.AllowedHit(
                "crates/demo/src/lib.rs",
                "type Missing",
                "test-only stale entry",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "stale allow-list entry"
        ):
            ORACLE.validate_sources(
                {"crates/demo/src/lib.rs": "fn clean() {}"},
                allowed,
                approved_build_scripts=(),
            )

    def test_rejects_composed_generated_rust_from_a_build_script(self) -> None:
        build_path = "crates/demo/build.rs"
        generated_build = (
            "fn main() {\n"
            "    let name = [\"Hash\", \"Map\"].concat();\n"
            "    let output = std::path::PathBuf::from(std::env::var(\"OUT_DIR\").unwrap())\n"
            "        .join(\"generated.rs\");\n"
            "    std::fs::write(output, format!(\"type Escape = {name}<u8, u8>;\")).unwrap();\n"
            "}\n"
        )
        approved = (
            ORACLE.ApprovedBuildScript(
                build_path,
                hashlib.sha256(generated_build.encode("utf-8")).hexdigest(),
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unlisted token 'include'"
        ):
            ORACLE.validate_sources(
                {
                    build_path: generated_build,
                    "crates/demo/src/lib.rs": (
                        'include!(concat!(env!("OUT_DIR"), "/generated.rs"));\n'
                    ),
                },
                allowed=(),
                approved_build_scripts=approved,
            )

    def test_rejects_aliased_include_of_generated_rust(self) -> None:
        build_path = "crates/demo/build.rs"
        generated_build = (
            "fn main() {\n"
            "    let name = [\"Hash\", \"Map\"].concat();\n"
            "    let output = std::path::PathBuf::from(std::env::var(\"OUT_DIR\").unwrap())\n"
            "        .join(\"generated.rs\");\n"
            "    std::fs::write(output, format!(\"type Escape = {name}<u8, u8>;\")).unwrap();\n"
            "}\n"
        )
        approved = (
            ORACLE.ApprovedBuildScript(
                build_path,
                hashlib.sha256(generated_build.encode("utf-8")).hexdigest(),
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "unlisted token 'include'"
        ):
            ORACLE.validate_sources(
                {
                    build_path: generated_build,
                    "crates/demo/src/lib.rs": (
                        "use std::include as include_generated;\n"
                        'include_generated!(concat!(env!("OUT_DIR"), "/generated.rs"));\n'
                    ),
                },
                allowed=(),
                approved_build_scripts=approved,
            )

    def test_rejects_a_changed_approved_build_script(self) -> None:
        build_path = "crates/demo/build.rs"
        approved_source = "fn main() {}\n"
        approved = (
            ORACLE.ApprovedBuildScript(
                build_path,
                hashlib.sha256(approved_source.encode("utf-8")).hexdigest(),
                "negative-control fixture",
            ),
        )
        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "changed approved build script"
        ):
            ORACLE.validate_sources(
                {build_path: 'fn main() { std::fs::write("generated.rs", "").unwrap(); }'},
                allowed=(),
                approved_build_scripts=approved,
            )


class HashOrderOracleRunnerTests(unittest.TestCase):
    def test_scan_only_prints_the_tripwire_marker(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            ORACLE.validate_tripwire(
                source_loader=lambda: {}, allowed=(), approved_build_scripts=()
            )
        self.assertEqual(output.getvalue(), "HASH ORDER TOKEN TRIPWIRE: PASS\n")

    def test_runs_every_component_and_prints_one_pass_marker(self) -> None:
        seen: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(command, 0)

        output = io.StringIO()
        with redirect_stdout(output):
            ORACLE.validate(
                runner=runner,
                source_loader=lambda: {},
                allowed=(),
                approved_build_scripts=(),
            )
        self.assertEqual(seen, list(ORACLE.COMMANDS))
        self.assertEqual(output.getvalue(), "HASH ORDER DETERMINISM ORACLE: PASS\n")

    def test_stops_at_the_first_failed_component(self) -> None:
        commands = (("first",), ("second",), ("third",))
        seen: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen.append(tuple(command))
            return subprocess.CompletedProcess(command, 7 if command == ("second",) else 0)

        with self.assertRaisesRegex(
            ORACLE.HashOrderDeterminismFailure, "exited 7: second"
        ):
            ORACLE.validate(
                runner=runner,
                commands=commands,
                source_loader=lambda: {},
                allowed=(),
                approved_build_scripts=(),
            )
        self.assertEqual(seen, [("first",), ("second",)])


if __name__ == "__main__":
    unittest.main()
