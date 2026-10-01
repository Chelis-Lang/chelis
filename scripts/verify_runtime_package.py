"""Verify a sealed distribution's runtime files against its compiler export.

The archive must survive copying byte for byte; each of the six public headers
must match both the export receipt and its copied package counterpart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


HEADERS = (
    "chelis_runtime.h",
    "chelis_runtime_views.h",
    "chelis_runtime_dtype.h",
    "chelis_blas.h",
    "chelis_simd.h",
    "chelis_math.h",
)
RECEIPT = "chelis_runtime.receipt.json"
ARCHIVE = "libchelis_runtime.a"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate export receipt field: {key}")
        result[key] = value
    return result


def _digest(root: Path, relative: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"runtime package root is not a real directory: {root}")
    path = root
    for part in relative.parts[:-1]:
        path = path / part
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"runtime package directory is missing or linked: {path}")
    path = path / relative.name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"runtime package file is missing or linked: {path}")
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def verify_runtime_package(export: Path, package: Path) -> str:
    """Require the package archive and every header to equal a sealed export.

    ``export`` is produced by this package's compiler using ``chelis runtime
    export``; ``package`` has ``lib/`` and ``include/`` directories. Returns
    the checked archive digest for receipts and diagnostics.
    """
    try:
        if export.is_symlink() or not export.is_dir():
            raise ValueError(f"runtime export is not a real directory: {export}")
        receipt_path = export / RECEIPT
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError(f"runtime export receipt is missing or linked: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"),
                             object_pairs_hook=_unique_fields)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read runtime export receipt: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != "chelis-runtime-staging/1":
        raise ValueError("runtime export has an invalid staging receipt schema")
    if receipt.get("mode") != "sealed":
        raise ValueError("runtime export is not sealed")
    if receipt.get("archive") != ARCHIVE:
        raise ValueError("runtime export has an unexpected archive name")
    headers = receipt.get("headers")
    if not isinstance(headers, dict) or set(headers) != set(HEADERS):
        raise ValueError("runtime export does not enumerate exactly the six public headers")
    files = [(Path(ARCHIVE), Path("lib") / ARCHIVE, receipt.get("archive_sha256"))]
    files.extend((Path(name), Path("include") / name, headers[name]) for name in HEADERS)
    for exported, shipped, expected in files:
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError(f"runtime export has no valid SHA-256 for {exported}")
        if _digest(export, exported) != expected:
            raise ValueError(f"runtime export {exported} differs from its receipt")
        if _digest(package, shipped) != expected:
            raise ValueError(f"runtime package {shipped} differs from compiler export")
    return receipt["archive_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="directory produced by this compiler's runtime export")
    parser.add_argument("package", type=Path, help="distribution root containing lib/ and include/")
    args = parser.parse_args()
    try:
        digest = verify_runtime_package(args.export, args.package)
    except ValueError as error:
        print(f"runtime package verification failed: {error}", file=sys.stderr)
        return 1
    print(f"runtime package matches compiler export: sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
