#!/usr/bin/env python3
"""Tests for the managed Kache environment contract."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kache_toolchain_smoke import (
    ContractFailure,
    validate_environment,
    write_isolated_kache_config,
)


class KacheEnvironmentTests(unittest.TestCase):
    def test_exact_repository_wrapper_and_config_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            wrapper = root / "store" / "bin" / "kache"
            wrapper.parent.mkdir(parents=True)
            wrapper.touch()
            config = root / ".kache.toml"
            config.write_text(
                "[cache]\nignore_env = true\ncache_executables = true\n"
                "heartbeat_secs = 30\n",
                encoding="utf-8",
            )

            validated = validate_environment(
                root,
                {
                    "RUSTC_WRAPPER": str(wrapper),
                    "KACHE_CONFIG": str(config),
                    "KACHE_DISABLED": "0",
                },
                wrapper,
            )

            self.assertEqual(validated.wrapper, wrapper)
            self.assertEqual(validated.config, config)

    def test_absent_path_wrapper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            wrapper = root / "kache"
            wrapper.touch()
            (root / ".kache.toml").write_text("[cache]\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractFailure, "PATH does not expose"):
                validate_environment(
                    root,
                    {
                        "RUSTC_WRAPPER": str(wrapper),
                        "KACHE_CONFIG": str(root / ".kache.toml"),
                        "KACHE_DISABLED": "0",
                    },
                    None,
                )

    def test_different_host_wrapper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            managed = root / "managed-kache"
            host = root / "host-kache"
            managed.touch()
            host.touch()
            (root / ".kache.toml").write_text("[cache]\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractFailure, "differs"):
                validate_environment(
                    root,
                    {
                        "RUSTC_WRAPPER": str(managed),
                        "KACHE_CONFIG": str(root / ".kache.toml"),
                        "KACHE_DISABLED": "0",
                    },
                    host,
                )

    def test_disabled_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            wrapper = root / "kache"
            wrapper.touch()
            (root / ".kache.toml").write_text("[cache]\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractFailure, "KACHE_DISABLED"):
                validate_environment(
                    root,
                    {
                        "RUSTC_WRAPPER": str(wrapper),
                        "KACHE_CONFIG": str(root / ".kache.toml"),
                        "KACHE_DISABLED": "1",
                    },
                    wrapper,
                )

    def test_isolated_config_rejects_an_existing_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            with self.assertRaises(ContractFailure):
                write_isolated_kache_config(
                    '[cache]\nlocal_store = "/host/cache"\n',
                    root / "probe-kache.toml",
                    root / "cache",
                )


if __name__ == "__main__":
    unittest.main()
