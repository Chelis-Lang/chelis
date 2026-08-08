"""Make the CI `smt` lanes LINK a prebuilt cvc5 instead of rebuilding it from
source (~22m CMake/make), by keeping the harvested cvc5 build in a DURABLE
GitHub Release asset (chelis#583 follow-up) plus a best-effort Actions-cache
fallback.

Background
----------
`chelis-prove`'s `smt` feature pulls `cvc5-rs -> cvc5-sys`, whose `build.rs`
clones cvc5 and builds it from source (CMake + make, ~22m cold) into the
build-script `OUT_DIR` the first time the `smt` feature is compiled.

`cvc5-sys` supports a prebuilt/short-circuit path (read its `build.rs`):

* If `CVC5_DIR` points at a cvc5 source tree whose `build/src/libcvc5.a`
  already exists, `ensure_cvc5_built` returns early -- NO CMake/make -- and the
  build script just links the prebuilt static libs and (cheaply) regenerates
  the bindgen bindings. This mirrors the z3-sys prebuilt-link precedent
  (`docs/local_z3_environment.md`): build the heavy solver ONCE, then LINK it.

Two stores hold that prebuilt tree, tried in order by the CI jobs:

1. DURABLE (primary): a GitHub Release asset published by
   `.github/workflows/build-cvc5.yml`, one per (cvc5-sys version, namespace).
   A Release asset has no 10GB Actions-cache LRU budget, no 7-day idle TTL, and
   no branch scope, so it is NOT evicted by a rustc-stable bump, a quiet
   stretch, or cache-pool pressure -- the three conditions that used to force a
   cold cvc5 rebuild onto the per-PR path. `fetch` downloads + sha256-verifies +
   extracts it.
2. FALLBACK (secondary): the Actions cache keyed by `compute_key`. Covers the
   window between a cvc5-sys bump and `build-cvc5.yml` republishing the asset,
   and any run where the Release lookup fails.

The cache/asset key depends on the cvc5 version (proxied by the pinned
`cvc5-sys` crate version, which moves with the bundled cvc5 release) + the
runner namespace (os/arch) + a cache-schema tag -- but NOT on `Cargo.lock` and
NOT on the rustc version. The harvested payload is 100% cvc5 C++/CMake output
(`bindings.rs` is regenerated per build in `OUT_DIR` and is not harvested), so
the rustc toolchain is provably irrelevant to the cached bytes; keying on it
only forced needless cold rebuilds on every ~6-week stable bump (the chelis#583
follow-up drops it). The store lives outside `target/`, so it does not interact
with `rust-cache`'s `target/` domain.

Safety contract (this is an optimization, never a correctness risk)
-------------------------------------------------------------------
Worst case must be "no speedup" (fall back to from-source), never "broken
build":

* `activate` exports `CVC5_DIR` ONLY when the store is BOTH present
  (`build/src/libcvc5.a`) AND marked complete (a sentinel written only after a
  fully-verified harvest/fetch). A cold/partial/incomplete store is ignored and
  the build falls back to the from-source path.
* `fetch` sha256-verifies the downloaded archive BEFORE extraction, rejects
  unsafe tar members, re-checks every required path after extraction, writes the
  sentinel only then, and ALWAYS exits 0. A missing asset, a network error, a
  hash mismatch, or an incomplete archive all leave no sentinel -> the Actions
  cache / from-source path runs.
* `harvest` runs only on a cold build (when `CVC5_DIR` was not activated),
  copies a CONSERVATIVE superset of the link/bindgen inputs, verifies every
  required path exists, and writes the sentinel only then.
* cvc5-sys's own `check_cvc5_version` (reads the harvested
  `cmake/version-base.cmake`) hard-fails a wrong-version link, so even a
  mislabelled asset cannot silently link the wrong solver.

CLI
---
    key --namespace <ns>
        Emit `key=` and `dir=` (the Actions-cache key + store dir) to
        $GITHUB_OUTPUT. `<ns>` is linux-x86_64 / darwin-arm64.

    asset-name --namespace <ns>
        Emit `tag=`, `asset=`, `sha256=`, `dir=` for the durable Release asset.

    fetch --namespace <ns> --dir <dir>
        Best-effort download + sha256-verify + extract of the durable Release
        asset into <dir>. Emits `warm=true|false`. Always exits 0.

    activate --dir <dir>
        If <dir> holds a complete prebuilt cvc5, export `CVC5_DIR=<dir>` to
        $GITHUB_ENV (warm). Otherwise no-op (cold from-source build).

    harvest --dir <dir>
        On a cold build, copy the freshly built cvc5 artifacts from
        `target/*/build/cvc5-sys-*/out/cvc5` into <dir> and mark it complete.
        No-op if CVC5_DIR is already active (warm).

    pack --namespace <ns> --dir <dir> --out-dir <out>
        Tar+gzip a harvested <dir> into <out>/<asset> (+ a `.sha256` sidecar)
        for `build-cvc5.yml` to upload. Emits `tag=`, `asset=`, `sha256=`.

    plan --namespace ... (repeatable)
        Query the Release for the current cvc5-sys tag and emit `tag=` plus
        `missing_<ns>=true|false` per namespace, so `build-cvc5.yml` only spends
        the ~22m from-source build on namespaces whose asset does not yet exist.
        `CVC5_FORCE_REBUILD=1` marks every namespace missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_LOCK = REPO_ROOT / "Cargo.lock"

# Bump when the harvest/link contract changes in a way that makes an
# already-saved store incompatible (e.g. a different set of static libs), or
# when the key SHAPE changes. v2 dropped the rustc component (chelis#583
# follow-up): rustc is irrelevant to the harvested cvc5 C++ bytes and only
# forced cold rebuilds on stable-toolchain bumps.
CACHE_SCHEMA = "v2"

# Sentinel written only after a fully verified harvest/fetch. `activate`
# requires it, so an incomplete or interrupted harvest/fetch can never be
# linked (it degrades to a from-source rebuild instead of breaking the build).
SENTINEL_NAME = ".chelis-cvc5-complete"

# The CI namespaces (os/arch), used by `plan` to enumerate assets. The
# former `linux-glibc231` namespace retired with the debian:11 lanes when
# the Linux release moved to the Nix build (openspec
# switch-linux-release-to-nix).
NAMESPACES = ("linux-x86_64", "darwin-arm64")

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
    bundles cvc5 1.3.1), so keying the store on it invalidates correctly when
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


def compute_key(namespace: str, cvc5_sys_ver: str) -> str:
    """Stable store key: cvc5 version (proxy) + os/arch namespace + schema.

    Deliberately excludes BOTH `Cargo.lock`'s overall hash (chelis#583) and the
    rustc version (the chelis#583 follow-up): neither changes the harvested cvc5
    C++ bytes, and both used to force needless cold rebuilds.
    """
    return f"cvc5-prebuilt-{namespace}-cvc5sys{cvc5_sys_ver}-{CACHE_SCHEMA}"


def asset_tag(cvc5_sys_ver: str) -> str:
    """Release tag holding the durable prebuilt assets for a cvc5-sys version.

    One tag per cvc5-sys version, carrying one asset per namespace. Prefixed so
    it never matches release.yml's `v*` shipped-release trigger.
    """
    return f"cvc5-prebuilt-cvc5sys{cvc5_sys_ver}"


def asset_filename(namespace: str, cvc5_sys_ver: str) -> str:
    """Per-namespace Release asset name (the store key + a `.tar.gz` suffix)."""
    return compute_key(namespace, cvc5_sys_ver) + ".tar.gz"


def cache_dir(namespace: str) -> Path:
    """Absolute, per-namespace store directory OUTSIDE `target/`.

    Per-namespace so lanes that build a different toolchain's cvc5 never
    restore each other's artifacts even if a key ever collided.
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
    """A store dir is usable only if the static lib AND the sentinel exist."""
    return (dest / "build" / "src" / "libcvc5.a").is_file() and (
        dest / SENTINEL_NAME
    ).is_file()


def finalize(dest: Path) -> bool:
    """Verify every REQUIRED_DEST path is present and write/remove the sentinel.

    Shared by `harvest` and `fetch`: the completeness gate is identical whether
    the tree was copied from a local build or extracted from a Release asset.
    Returns True and writes the sentinel iff the tree is complete; otherwise
    removes any stale sentinel and returns False.
    """
    sentinel = dest / SENTINEL_NAME
    missing = [rel for rel in REQUIRED_DEST if not (dest / rel).exists()]
    if missing:
        if sentinel.exists():
            sentinel.unlink()
        print(
            "cvc5 store INCOMPLETE; missing "
            + ", ".join(missing)
            + " -- not marking complete (build will stay from-source)",
            file=sys.stderr,
        )
        return False
    sentinel.write_text("complete\n", encoding="utf-8")
    return True


def harvest(src: Path, dest: Path) -> bool:
    """Copy the cvc5 link/bindgen inputs from `src` into `dest`.

    Returns True and writes the completeness sentinel iff every required
    destination path is present after the copy; otherwise removes any stale
    sentinel and returns False (caller treats it as "no store, no speedup").
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

    ok = finalize(dest)
    if ok:
        print(f"cvc5 harvest complete -> {dest}")
    return ok


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 hex digest of a file (bounded memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_sha256_sidecar(text: str) -> str | None:
    """Extract the hex digest from a `<hex>  <name>` sidecar line."""
    token = text.strip().split()
    if token and re.fullmatch(r"[0-9a-fA-F]{64}", token[0]):
        return token[0].lower()
    return None


def pack(src_dir: Path, out_path: Path) -> str:
    """Tar+gzip the CONTENTS of a harvested `src_dir` into `out_path`.

    Members are stored relative (`include/...`, `build/...`, `cmake/...`) so a
    later `safe_extract` reproduces the harvested layout in place. Writes an
    `<out_path>.sha256` sidecar and returns the digest. Raises if `src_dir` is
    not a complete harvested tree (never publish an incomplete asset).
    """
    if not is_warm(src_dir):
        raise RuntimeError(
            f"refusing to pack incomplete cvc5 store {src_dir} "
            "(missing libcvc5.a or completeness sentinel)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        for child in sorted(src_dir.iterdir()):
            tar.add(child, arcname=child.name)
    digest = sha256_file(out_path)
    Path(str(out_path) + ".sha256").write_text(
        f"{digest}  {out_path.name}\n", encoding="utf-8"
    )
    return digest


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract a trusted-but-verified tar.gz into `dest`, rejecting escapes.

    The archive is our own `pack` output and is sha256-verified before this is
    called, but a defensive traversal/symlink check is cheap insurance against
    a corrupted or substituted asset ever writing outside `dest`.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"unexpected link member in cvc5 asset: {member.name}")
            target = (dest / member.name).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"unsafe tar member escapes dest: {member.name}")
        tar.extractall(dest)


# --- Release lookup (stdlib urllib; injectable for tests) -------------------


def _http_get_json(url: str, token: str | None = None) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "chelis-ci",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


class _NoAutoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to auto-follow redirects so `_http_download` can re-issue the
    redirected GET WITHOUT the Authorization header. GitHub asset downloads
    302 to a signed storage URL (objects.githubusercontent.com / S3) that
    rejects the API token, and stdlib urllib on older Pythons (debian:11 ships
    3.9) forwards the Authorization header across the redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_download(url: str, dest: Path, token: str | None = None) -> None:
    """Download a release asset (works for PRIVATE repos) to `dest`.

    Uses the asset API `url` (NOT `browser_download_url`, which needs a web
    session and 404s for a token on a private repo) with
    `Accept: application/octet-stream` + Bearer auth. GitHub 302-redirects to a
    signed storage URL that must be fetched WITHOUT the Authorization header, so
    the redirect is followed manually.
    """
    headers = {"Accept": "application/octet-stream", "User-Agent": "chelis-ci"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = urllib.request.build_opener(_NoAutoRedirect)
    location = None
    resp = None
    try:
        resp = opener.open(urllib.request.Request(url, headers=headers), timeout=300)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
            location = exc.headers["Location"]
        else:
            raise
    if location is not None:
        # Signed storage URL: fetch with NO auth header (it would be rejected by
        # the storage backend and would leak the token to a third-party host).
        with urllib.request.urlopen(
            urllib.request.Request(location, headers={"User-Agent": "chelis-ci"}),
            timeout=300,
        ) as red, open(dest, "wb") as fh:
            shutil.copyfileobj(red, fh)
    else:
        with resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)


def _release_assets(tag: str) -> dict[str, dict] | None:
    """Return {asset_name: asset_json} for a Release tag, or None if absent.

    Best-effort: any lookup failure (no such release, network error, bad JSON,
    unset GITHUB_REPOSITORY) returns None so callers degrade to the fallback
    path instead of failing the job.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("GITHUB_REPOSITORY unset; cannot look up durable cvc5 asset", file=sys.stderr)
        return None
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    token = os.environ.get("GITHUB_TOKEN")
    url = f"{api}/repos/{repo}/releases/tags/{tag}"
    try:
        release = _http_get_json(url, token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
        print(f"durable cvc5 release lookup failed for {tag}: {exc}", file=sys.stderr)
        return None
    return {a["name"]: a for a in release.get("assets", []) if "name" in a}


# --- Subcommands ------------------------------------------------------------


def cmd_key(args: argparse.Namespace) -> int:
    cvc5_sys_ver = parse_cvc5_sys_version(CARGO_LOCK.read_text(encoding="utf-8"))
    _github_output("key", compute_key(args.namespace, cvc5_sys_ver))
    _github_output("dir", str(cache_dir(args.namespace)))
    return 0


def cmd_asset_name(args: argparse.Namespace) -> int:
    cvc5_sys_ver = parse_cvc5_sys_version(CARGO_LOCK.read_text(encoding="utf-8"))
    _github_output("tag", asset_tag(cvc5_sys_ver))
    _github_output("asset", asset_filename(args.namespace, cvc5_sys_ver))
    _github_output("sha256", asset_filename(args.namespace, cvc5_sys_ver) + ".sha256")
    _github_output("dir", str(cache_dir(args.namespace)))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Best-effort populate <dir> from the durable Release asset. Always 0."""
    dest = Path(args.dir)
    if is_warm(dest):
        print(f"cvc5 store already warm at {dest}; skipping durable fetch")
        _github_output("warm", "true")
        return 0

    cvc5_sys_ver = parse_cvc5_sys_version(CARGO_LOCK.read_text(encoding="utf-8"))
    tag = asset_tag(cvc5_sys_ver)
    name = asset_filename(args.namespace, cvc5_sys_ver)
    sha_name = name + ".sha256"

    assets = _release_assets(tag)
    if assets is None or name not in assets or sha_name not in assets:
        print(
            f"no durable cvc5 asset {name} on release {tag}; "
            "falling back to Actions cache / from-source",
            file=sys.stderr,
        )
        _github_output("warm", "false")
        return 0

    with TemporaryDirectory() as td:
        tar_path = Path(td) / name
        sha_path = Path(td) / sha_name
        token = os.environ.get("GITHUB_TOKEN")
        try:
            _http_download(assets[name]["url"], tar_path, token)
            _http_download(assets[sha_name]["url"], sha_path, token)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, OSError) as exc:
            print(f"durable cvc5 asset download failed: {exc}", file=sys.stderr)
            _github_output("warm", "false")
            return 0

        expected = _parse_sha256_sidecar(sha_path.read_text(encoding="utf-8"))
        actual = sha256_file(tar_path)
        if expected is None or actual != expected:
            print(
                f"durable cvc5 asset sha256 mismatch (expected {expected}, got "
                f"{actual}); NOT extracting",
                file=sys.stderr,
            )
            _github_output("warm", "false")
            return 0

        try:
            safe_extract(tar_path, dest)
        except (tarfile.TarError, RuntimeError, OSError) as exc:
            print(f"durable cvc5 asset extract failed: {exc}", file=sys.stderr)
            _github_output("warm", "false")
            return 0

    warm = finalize(dest) and is_warm(dest)
    if warm:
        print(f"durable cvc5 asset linked: {name} -> {dest}")
    _github_output("warm", "true" if warm else "false")
    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    dest = Path(args.dir)
    if is_warm(dest):
        # Link the prebuilt cvc5 (z3-style): cvc5-sys sees the existing
        # build/src/libcvc5.a and skips the from-source CMake/make.
        _github_env("CVC5_DIR", str(dest))
        print(f"cvc5 store WARM: linking prebuilt cvc5 from {dest}")
    else:
        print(
            f"cvc5 store COLD: {dest} has no complete prebuilt cvc5; "
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
            "(store stays cold)",
            file=sys.stderr,
        )
        _github_output("complete", "false")
        return 0
    complete = harvest(src, dest)
    _github_output("complete", "true" if complete else "false")
    return 0


def cmd_pack(args: argparse.Namespace) -> int:
    cvc5_sys_ver = parse_cvc5_sys_version(CARGO_LOCK.read_text(encoding="utf-8"))
    name = asset_filename(args.namespace, cvc5_sys_ver)
    out_path = Path(args.out_dir) / name
    digest = pack(Path(args.dir), out_path)
    _github_output("tag", asset_tag(cvc5_sys_ver))
    _github_output("asset", name)
    _github_output("sha256", digest)
    print(f"packed durable cvc5 asset {out_path} ({digest})")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cvc5_sys_ver = parse_cvc5_sys_version(CARGO_LOCK.read_text(encoding="utf-8"))
    tag = asset_tag(cvc5_sys_ver)
    _github_output("tag", tag)
    force = os.environ.get("CVC5_FORCE_REBUILD") == "1"
    assets = None if force else _release_assets(tag)
    present: set[str] = set() if assets is None else set(assets)
    namespaces = args.namespace or list(NAMESPACES)
    for ns in namespaces:
        have = asset_filename(ns, cvc5_sys_ver) in present
        missing = force or not have
        # Output name uses underscores (safe for `needs.*.outputs.*` refs).
        _github_output(f"missing_{ns.replace('-', '_')}", "true" if missing else "false")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_key = sub.add_parser("key", help="emit the stable Actions-cache key and dir")
    p_key.add_argument("--namespace", required=True, help="os/arch namespace")
    p_key.set_defaults(func=cmd_key)

    p_asset = sub.add_parser("asset-name", help="emit the durable Release asset name")
    p_asset.add_argument("--namespace", required=True)
    p_asset.set_defaults(func=cmd_asset_name)

    p_fetch = sub.add_parser("fetch", help="download+verify the durable Release asset")
    p_fetch.add_argument("--namespace", required=True)
    p_fetch.add_argument("--dir", required=True)
    p_fetch.set_defaults(func=cmd_fetch)

    p_act = sub.add_parser("activate", help="export CVC5_DIR when the store is warm")
    p_act.add_argument("--dir", required=True)
    p_act.set_defaults(func=cmd_activate)

    p_harv = sub.add_parser("harvest", help="copy a cold cvc5 build into the store dir")
    p_harv.add_argument("--dir", required=True)
    p_harv.set_defaults(func=cmd_harvest)

    p_pack = sub.add_parser("pack", help="tar a harvested store into a Release asset")
    p_pack.add_argument("--namespace", required=True)
    p_pack.add_argument("--dir", required=True)
    p_pack.add_argument("--out-dir", required=True)
    p_pack.set_defaults(func=cmd_pack)

    p_plan = sub.add_parser("plan", help="emit which namespace assets are missing")
    p_plan.add_argument(
        "--namespace",
        action="append",
        help="restrict to this namespace (repeatable); default all",
    )
    p_plan.set_defaults(func=cmd_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
