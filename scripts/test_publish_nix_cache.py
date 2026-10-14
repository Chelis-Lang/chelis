"""Tests for the Cachix publisher of the Nix package closure.

Run with `.venv/bin/python scripts/test_publish_nix_cache.py`.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import re
import sys
import unittest
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import publish_nix_cache  # noqa: E402

SYSTEM = "aarch64-darwin"
CHELIS = "/nix/store/aaaa-chelis-0.19.4"
RUNTIME = "/nix/store/bbbb-chelis-runtime-0.19.4"
CHELISUP = "/nix/store/cccc-chelisup-0.19.4"
ZSTD = "/nix/store/dddd-zstd-1.5.7"
TOKEN = {publish_nix_cache.TOKEN_VARIABLE: "token"}


class FakeNix:
    """Answers the four commands the publisher runs and records each call."""

    def __init__(self, infos: object, fail: str | None = None) -> None:
        self.infos = infos
        self.fail = fail
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, argv: Sequence[str], stdin: str | None) -> str:
        argv = list(argv)
        self.calls.append((argv, stdin))
        command = argv[1]
        if command == self.fail:
            raise publish_nix_cache.PublishError(f"{command} failed")
        if command == "eval":
            return f"{SYSTEM}\n"
        if command == "build":
            return f"{CHELIS}\n{RUNTIME}\n{CHELISUP}\n"
        if command == "path-info":
            return json.dumps(self.infos)
        if command == "run":
            return ""
        raise AssertionError(f"unexpected command {argv!r}")

    def pushes(self) -> list[tuple[list[str], str | None]]:
        return [call for call in self.calls if call[0][1] == "run"]


def object_infos(local: Sequence[str], substituted: Sequence[str]) -> dict[str, dict[str, object]]:
    infos: dict[str, dict[str, object]] = {path: {"ultimate": True} for path in local}
    infos.update({path: {"ultimate": False} for path in substituted})
    return infos


def run_main(
    nix: FakeNix, argv: Sequence[str] = (), environ: dict[str, str] = TOKEN
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = publish_nix_cache.main(list(argv), run=nix, environ=environ)
    return code, stdout.getvalue(), stderr.getvalue()


class PublishNixCacheTests(unittest.TestCase):
    def test_pushes_only_the_locally_built_paths(self) -> None:
        nix = FakeNix(object_infos([CHELIS, RUNTIME, CHELISUP], [ZSTD]))
        code, stdout, _ = run_main(nix)
        self.assertEqual(code, 0)
        self.assertIn("pushed 3 aarch64-darwin path(s)", stdout)
        build = next(argv for argv, _ in nix.calls if argv[1] == "build")
        self.assertEqual(
            build[-3:],
            [f".#packages.{SYSTEM}.chelis", f".#packages.{SYSTEM}.chelis-runtime", f".#packages.{SYSTEM}.chelisup"],
        )
        [(argv, stdin)] = nix.pushes()
        self.assertEqual(
            argv,
            ["nix", "run", "--inputs-from", ".", "nixpkgs#cachix", "--", "push", publish_nix_cache.CACHE_NAME],
        )
        self.assertEqual(stdin, "".join(f"{path}\n" for path in sorted([CHELIS, RUNTIME, CHELISUP])))
        self.assertNotIn(ZSTD, stdin)

    def test_reads_the_list_shape_of_older_path_info(self) -> None:
        infos = [
            {"path": CHELIS, "ultimate": True},
            {"path": ZSTD, "ultimate": False},
        ]
        nix = FakeNix(infos)
        code, _, _ = run_main(nix)
        self.assertEqual(code, 0)
        [(_, stdin)] = nix.pushes()
        self.assertEqual(stdin, f"{CHELIS}\n")

    def test_missing_token_runs_no_command(self) -> None:
        nix = FakeNix(object_infos([CHELIS], []))
        code, _, stderr = run_main(nix, environ={})
        self.assertEqual(code, 2)
        self.assertIn(publish_nix_cache.TOKEN_VARIABLE, stderr)
        self.assertEqual(nix.calls, [])

    def test_closure_without_a_local_build_pushes_nothing(self) -> None:
        nix = FakeNix(object_infos([], [CHELIS, ZSTD]))
        code, _, stderr = run_main(nix)
        self.assertEqual(code, 2)
        self.assertIn("no locally built path", stderr)
        self.assertEqual(nix.pushes(), [])

    def test_failed_build_stops_before_the_push(self) -> None:
        nix = FakeNix(object_infos([CHELIS], []), fail="build")
        code, _, stderr = run_main(nix)
        self.assertEqual(code, 1)
        self.assertIn("build failed", stderr)
        self.assertEqual(nix.pushes(), [])

    def test_invalid_path_info_json_fails(self) -> None:
        nix = FakeNix(object_infos([CHELIS], []))
        nix.infos = "not an object"
        code, _, stderr = run_main(nix)
        self.assertEqual(code, 1)
        self.assertIn("neither an object nor a list", stderr)
        self.assertEqual(nix.pushes(), [])

    def test_dry_run_lists_the_paths_without_a_token(self) -> None:
        nix = FakeNix(object_infos([CHELIS, RUNTIME], [ZSTD]))
        code, stdout, _ = run_main(nix, argv=["--dry-run"], environ={})
        self.assertEqual(code, 0)
        self.assertEqual(stdout.splitlines(), sorted([CHELIS, RUNTIME]))
        self.assertEqual(nix.pushes(), [])


class FlakeCacheContractTests(unittest.TestCase):
    """The flake trusts the cache that the release publisher pushes to."""

    def flake_list(self, attribute: str) -> list[str]:
        flake = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        match = re.search(rf"nixConfig\.{re.escape(attribute)}\s*=\s*\[(?P<body>[^\]]*)\]", flake)
        self.assertIsNotNone(match, f"flake.nix declares no nixConfig.{attribute}")
        assert match is not None
        return re.findall(r'"([^"]*)"', match.group("body"))

    def test_flake_substitutes_from_the_published_cache(self) -> None:
        self.assertEqual(
            self.flake_list("extra-substituters"),
            [f"https://{publish_nix_cache.CACHE_NAME}.cachix.org"],
        )

    def test_flake_trusts_a_valid_key_of_the_published_cache(self) -> None:
        [key] = self.flake_list("extra-trusted-public-keys")
        name, _, encoded = key.partition(":")
        self.assertEqual(name, f"{publish_nix_cache.CACHE_NAME}.cachix.org-1")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except ValueError:
            decoded = b""
        self.assertEqual(
            len(decoded),
            32,
            "the trusted key must be the cache's Ed25519 public key, not a placeholder",
        )


if __name__ == "__main__":
    unittest.main()
