"""Cache the cvc5 build under a STABLE key so Cargo.lock churn (chelis#583)
does not force a ~30-minute from-source cvc5 recompile on every PR.

Background
----------
`chelis-prove`'s `smt` feature pulls `cvc5-rs -> cvc5-sys`, whose `build.rs`
clones cvc5 and builds it from source (CMake + make, ~30m cold) into the
build-script `OUT_DIR` the first time the `smt` feature is compiled. The CI
`smt-build` lane caches `target/` with `Swatinem/rust-cache`, but that cache
key invalidates on ANY `Cargo.lock` change -- so an unrelated dependency bump
blows the cache and re-pays the full cvc5 build.

`cvc5-sys` supports a prebuilt/short-circuit path (read its `build.rs`):

* If `CVC5_DIR` points at a cvc5 source tree whose `build/src/libcvc5.a`
  already exists, `ensure_cvc5_built` returns early -- NO CMake/make -- and the
  build script just links the prebuilt static libs and (cheaply) regenerates
  the bindgen bindings. This mirrors the z3-sys prebuilt-link precedent
  (`docs/local_z3_environment.md`): build the heavy solver ONCE, then LINK it.

This helper makes CI cache the harvested cvc5 build artifacts OUTSIDE `target/`
under a key that depends on the cvc5 version (proxied by the pinned `cvc5-sys`
crate version, which moves with the bundled cvc5 release) + the runner
namespace (os/arch) + the rustc version -- but NOT on `Cargo.lock`. The cache
lives outside `target/`, so it does not interact with `rust-cache`'s `target/`
domain (no two-caches-over-target ordering footgun).

Safety contract (chelis#583 is an optimization, never a correctness risk)
-------------------------------------------------------------------------
Worst case must be "no speedup", never "broken build":

* `activate` exports `CVC5_DIR` ONLY when the restored cache is BOTH present
  (`build/src/libcvc5.a`) AND marked complete (a sentinel written only after a
  fully-verified harvest). A cold/partial/incomplete cache is ignored, and the
  build falls back to the normal from-source path.
* `harvest` runs only on a cold build (when `CVC5_DIR` was not activated),
  copies a CONSERVATIVE superset of the link/bindgen inputs, verifies every
  required path exists in the destination, and writes the sentinel only then.
* The dedicated cache is saved by `actions/cache` only on a cache miss, and the
  harvest step runs after a successful build, so a failed/partial build never
  poisons the cache.

CLI
---
    python3 scripts/ci_cvc5_cache.py key --namespace <ns>
        Emit `key=` and `dir=` to $GITHUB_OUTPUT (and stdout). `<ns>` encodes
        os/arch (e.g. linux-x86_64, linux-glibc231, darwin-arm64).

    python3 scripts/ci_cvc5_cache.py activate --dir <dir>
        If <dir> holds a complete prebuilt cvc5, export `CVC5_DIR=<dir>` to
        $GITHUB_ENV (warm). Otherwise no-op (cold from-source build).

    python3 scripts/ci_cvc5_cache.py harvest --dir <dir>
        On a cold build, copy the freshly built cvc5 artifacts from
        `target/*/build/cvc5-sys-*/out/cvc5` into <dir> and mark it complete.
        No-op if CVC5_DIR is already active (warm).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_LOCK = REPO_ROOT / "Cargo.lock"

# Bump when the harvest/link contract changes in a way that makes an
# already-saved cache incompatible (e.g. a different set of static libs).
CACHE_SCHEMA = "v1"

# Sentinel written only after a fully verified harvest. `activate` requires it,
# so an incomplete or interrupted harvest can never be linked (it degrades to a
# from-source rebuild instead of breaking the build).
SENTINEL_NAME = ".chelis-cvc5-complete"

# Paths (relative to the cvc5 source/build tree) that the cvc5-sys build script
# reads on its prebuilt CVC5_DIR path. Directories are copied wholesale so the
# bindgen header set and the static-lib set are never partially captured.
#   cmake/version-base.cmake -> check_cvc5_version
#   include/                 -> bindgen source headers (cvc5/c/cvc5.h + tree)
#   build/include/           -> bindgen generated headers (cvc5_export.h, ...)
#   build/src/libcvc5.a      -> the cvc5 static library
#   build/deps/lib/          -> bundled static deps (cadical/picpoly/.../gmp)
HARVEST_FILES = ("cmake/version-base.cmake", "build/src/libcvc5.a")
HARVEST_DIRS = ("include", "build/include", "build/deps/lib")

# Subset of the harvested set whose presence in the DESTINATION is verified
# before the completeness sentinel is written. cvc5/c/cvc5.h is the bindgen
# entry header; libcvc5.a is the link anchor; build/deps/lib must exist or the
# final link cannot resolve cvc5's bundled dependencies.
REQUIRED_DEST = (
    "cmake/version-base.cmake",
    "include/cvc5/c/cvc5.h",
    "build/include",
    "build/src/libcvc5.a",
    "build/deps/lib",
)


def parse_cvc5_sys_version(lock_text: str) -> str:
    """Return the resolved `cvc5-sys` crate version from `Cargo.lock`.

    The crate version moves with the bundled cvc5 release (`cvc5-sys 0.3.1`
    bundles cvc5 1.3.1), so keying the cache on it invalidates correctly when
    the solver version changes while staying stable across unrelated
    dependency churn. Raises if `cvc5-sys` is not present (the `smt` feature
    must be in the resolved graph for this lane).
    """
    # Cargo.lock packages are TOML array-of-tables; find the cvc5-sys block and
    # read its `version = "..."`. Line-based to avoid a TOML dependency the CI
    # interpreter may not carry.
    in_block = False
    for line in lock_text.splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            in_block = False
            continue
        if stripped.startswith('name = "'):
            in_block = stripped == 'name = "cvc5-sys"'
            continue
        if in_block and stripped.startswith('version = "'):
            m = re.match(r'version = "([^"]+)"', stripped)
            if m:
                return m.group(1)
    raise RuntimeError(
        "cvc5-sys not found in Cargo.lock; the smt feature graph is required "
        "for the cvc5 cache lane"
    )


def parse_rustc_version(rustc_output: str) -> str:
    """Extract the bare `X.Y.Z` version from `rustc --version` output.

    `rustc 1.86.0 (05f9846f8 2025-03-31)` -> `1.86.0`. Falls back to a
    sanitized token if the format is unexpected (never raises: a weird version
    string must not break key computation -- worst case is a slightly coarser
    key).
    """
    m = re.search(r"\b(\d+\.\d+\.\d+)\b", rustc_output)
    if m:
        return m.group(1)
    token = re.sub(r"[^A-Za-z0-9.]+", "-", rustc_output.strip())
    return token or "unknown"


def rustc_version() -> str:
    try:
        out = subprocess.run(
            ["rustc", "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return parse_rustc_version(out)


def compute_key(namespace: str, cvc5_sys_ver: str, rustc_ver: str) -> str:
    """Stable cache key: cvc5 version (proxy) + os/arch namespace + rustc.

    Deliberately excludes `Cargo.lock`'s overall hash (the chelis#583 fix):
    unrelated dependency churn must NOT invalidate the cvc5 build.
    """
    return (
        f"cvc5-prebuilt-{namespace}-cvc5sys{cvc5_sys_ver}"
        f"-rustc{rustc_ver}-{CACHE_SCHEMA}"
    )


def cache_dir(namespace: str) -> Path:
    """Absolute, per-namespace cache directory OUTSIDE `target/`.

    Per-namespace so lanes that build a different toolchain's cvc5 (e.g. the
    glibc-2.31 container vs ubuntu-latest) never restore each other's artifacts
    even if a key ever collided.
    """
    return Path.home() / ".cache" / "chelis-cvc5" / namespace


def find_built_cvc5(target_root: Path) -> Path | None:
    """Locate the from-source cvc5 tree under `target/`.

    `cvc5-sys` clones+builds cvc5 into its build-script `OUT_DIR/cvc5`, i.e.
    `target/<profile>/build/cvc5-sys-<hash>/out/cvc5`. The `<hash>` and profile
    vary, so glob and pick the first match that actually carries the built
    static library.
    """
    matches = sorted(target_root.glob("*/build/cvc5-sys-*/out/cvc5"))
    for candidate in matches:
        if (candidate / "build" / "src" / "libcvc5.a").is_file():
            return candidate
    return None


def _github_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")
    print(f"{name}={value}")


def _github_env(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_ENV")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")
    print(f"{name}={value}")


def is_warm(dest: Path) -> bool:
    """A cache dir is usable only if the static lib AND the sentinel exist."""
    return (dest / "build" / "src" / "libcvc5.a").is_file() and (
        dest / SENTINEL_NAME
    ).is_file()


def harvest(src: Path, dest: Path) -> bool:
    """Copy the cvc5 link/bindgen inputs from `src` into `dest`.

    Returns True and writes the completeness sentinel iff every required
    destination path is present after the copy; otherwise removes any stale
    sentinel and returns False (caller treats it as "no cache, no speedup").
    """
    # Start clean so a prior partial harvest cannot leave stale files behind.
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for rel in HARVEST_DIRS:
        src_dir = src / rel
        if src_dir.is_dir():
            shutil.copytree(src_dir, dest / rel)
    for rel in HARVEST_FILES:
        src_file = src / rel
        if src_file.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dest / rel)

    sentinel = dest / SENTINEL_NAME
    missing = [rel for rel in REQUIRED_DEST if not (dest / rel).exists()]
    if missing:
        if sentinel.exists():
            sentinel.unlink()
        print(
            "cvc5 harvest INCOMPLETE; missing "
            + ", ".join(missing)
            + " -- not marking cache complete (build will stay from-source)",
            file=sys.stderr,
        )
        return False
    sentinel.write_text("complete\n", encoding="utf-8")
    print(f"cvc5 harvest complete -> {dest}")
    return True


def cmd_key(args: argparse.Namespace) -> int:
    cvc5_sys_ver = parse_cvc5_sys_version(CARGO_LOCK.read_text(encoding="utf-8"))
    rustc_ver = rustc_version()
    key = compute_key(args.namespace, cvc5_sys_ver, rustc_ver)
    _github_output("key", key)
    _github_output("dir", str(cache_dir(args.namespace)))
    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    dest = Path(args.dir)
    if is_warm(dest):
        # Link the prebuilt cvc5 (z3-style): cvc5-sys sees the existing
        # build/src/libcvc5.a and skips the from-source CMake/make.
        _github_env("CVC5_DIR", str(dest))
        print(f"cvc5 cache WARM: linking prebuilt cvc5 from {dest}")
    else:
        print(
            f"cvc5 cache COLD: {dest} has no complete prebuilt cvc5; "
            "building from source this run"
        )
    return 0


def cmd_harvest(args: argparse.Namespace) -> int:
    # `complete` drives the dedicated `actions/cache/save` step: the cache is
    # persisted under the stable key ONLY when a fresh, fully-verified harvest
    # was written this run. A warm build, a missing build tree, or an
    # incomplete harvest all emit complete=false, so a failed/partial run can
    # never poison the stable-key cache with an empty or broken payload.
    if os.environ.get("CVC5_DIR"):
        print("CVC5_DIR active (warm build): nothing to harvest")
        _github_output("complete", "false")
        return 0
    dest = Path(args.dir)
    src = find_built_cvc5(REPO_ROOT / "target")
    if src is None:
        print(
            "no from-source cvc5 build found under target/; skipping harvest "
            "(cache stays cold)",
            file=sys.stderr,
        )
        _github_output("complete", "false")
        return 0
    complete = harvest(src, dest)
    _github_output("complete", "true" if complete else "false")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_key = sub.add_parser("key", help="emit the stable cache key and dir")
    p_key.add_argument(
        "--namespace",
        required=True,
        help="os/arch namespace, e.g. linux-x86_64 / linux-glibc231 / darwin-arm64",
    )
    p_key.set_defaults(func=cmd_key)

    p_act = sub.add_parser("activate", help="export CVC5_DIR when cache is warm")
    p_act.add_argument("--dir", required=True)
    p_act.set_defaults(func=cmd_activate)

    p_harv = sub.add_parser(
        "harvest", help="copy a cold cvc5 build into the cache dir"
    )
    p_harv.add_argument("--dir", required=True)
    p_harv.set_defaults(func=cmd_harvest)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
