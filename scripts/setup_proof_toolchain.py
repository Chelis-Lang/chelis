#!/usr/bin/env python3
"""WI-13 Option B: scripted setup of the Sollya + Gappa proof toolchain.

This is the reproducible, CI-exercised installer for the offline proof
toolchain that generates the committed erf Gappa proof bundle
(``scripts/generate_erf_proof.py``). It is deliberately NOT a shell script
(repo policy): everything is here so it is testable and portable.

What it does:
  1. Verifies the base system libraries that need root are present:
     ``mpfi``, ``libxml2``, and the ``gappa`` binary. These are the only
     root-requiring pieces; if any is missing it prints the exact distro
     install command and exits non-zero (it never silently半-installs).
  2. Builds **fplll** and **Sollya** FROM SOURCE, no sudo, into a workspace
     -local prefix (``.local/``), linking the system gmp/mpfr/mpfi/libxml2.
     Two portability fixes are applied because Sollya 8.0 predates GCC 14/15's
     C23 default:
       - ``CFLAGS`` carries ``-std=gnu17`` (GCC 15 defaults to gnu23, under
         which Sollya's K&R empty-paren prototypes are hard "conflicting types"
         errors),
       - a one-line, idempotent patch to ``execute.h`` gives ``miniyyparse`` a
         prototype matching the bison-generated definition.
  3. Reports the Arb-certifier build deps (gcc/m4/make + the .cargo/config
     CFLAGS=-fPIC and isolated *_CACHE dirs are already baked; this script only
     reminds, it does not duplicate that config).

Idempotent: re-running with ``.local/bin/sollya`` already present is a no-op
unless ``--force`` is given.

Licensing note (tracked, not a blocker): Sollya is CeCILL-C and fplll is
LGPL-2.1+. Both are BUILD/OFFLINE tooling that produce the committed proof
bundle; neither links into any shipped chelis binary.

Usage:
  .venv/bin/python scripts/setup_proof_toolchain.py [--force] [--check-only]

  --check-only verifies the toolchain is present (root libs + gappa + the built
  sollya) without building anything; the CI "is the proof toolchain available"
  probe.

Exit codes:
  0  toolchain present (or successfully built)
  2  a root-requiring base library/binary is missing (prints the install cmd)
  3  a source build step failed
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PREFIX = REPO_ROOT / ".local"
BUILD_DIR = REPO_ROOT / ".local" / "build"

FPLLL_URL = "https://github.com/fplll/fplll/releases/download/5.4.5/fplll-5.4.5.tar.gz"
FPLLL_SHA256 = "76d3778f0326597ed7505bab19493a9bf6b73a5c5ca614e8fb82f42105c57d00"
SOLLYA_URL = "https://www.sollya.org/releases/sollya-8.0/sollya-8.0.tar.gz"
SOLLYA_SHA256 = "58d734f9a2fc8e6733c11f96d2df9ab25bef24d71c401230e29f0a1339a81192"

# Root-requiring base libraries, by probe and by install command. The probe is a
# header/binary path so we never silently proceed without the dep.
ROOT_DEPS = {
    "mpfi": "/usr/include/mpfi.h",
    "libxml2": "/usr/include/libxml2/libxml/parser.h",
    "gappa": None,  # binary, probed via shutil.which
}
INSTALL_HINTS = {
    "fedora": "sudo dnf install -y mpfi-devel libxml2-devel gappa",
    "debian": "sudo apt-get install -y libmpfi-dev libxml2-dev gappa",
}


def check_root_deps() -> list[str]:
    """Return the list of missing root-requiring deps."""
    missing = []
    for name, header in ROOT_DEPS.items():
        if name == "gappa":
            if shutil.which("gappa") is None:
                missing.append(name)
        elif not Path(header).exists():
            missing.append(name)
    return missing


def require_root_deps() -> None:
    missing = check_root_deps()
    if missing:
        sys.stderr.write(
            "error: missing base libraries that need root to install: "
            f"{', '.join(missing)}\n"
            "Install them with one of:\n"
            f"  Fedora:        {INSTALL_HINTS['fedora']}\n"
            f"  Debian/Ubuntu: {INSTALL_HINTS['debian']}\n"
            "(These are the ONLY root-requiring pieces; Sollya + fplll are built "
            "from source with no sudo by this script.)\n"
        )
        raise SystemExit(2)


def _download(url: str, dest: Path, sha256: str) -> None:
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha256:
        return
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        sys.stderr.write(
            f"error: checksum mismatch for {url}\n  expected {sha256}\n  got      {got}\n"
        )
        raise SystemExit(3)
    dest.write_bytes(data)


def _run(cmd: list[str], cwd: Path, env: dict, log: str) -> None:
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(
            f"error: {log} failed (rc={proc.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}\n"
        )
        raise SystemExit(3)


def _build_env(extra_cflags: str = "") -> dict:
    env = dict(os.environ)
    inc = str(LOCAL_PREFIX / "include")
    lib = str(LOCAL_PREFIX / "lib")
    env["CFLAGS"] = f"-fPIC -O2 -I{inc} {extra_cflags}".strip()
    env["CXXFLAGS"] = f"-fPIC -O2 -I{inc}".strip()
    env["LDFLAGS"] = f"-L{lib} -Wl,-rpath,{lib}"
    env["PKG_CONFIG_PATH"] = f"{lib}/pkgconfig:" + env.get("PKG_CONFIG_PATH", "")
    return env


def build_fplll() -> None:
    """Sollya 8.0 requires fplll; build it no-sudo into .local."""
    if (LOCAL_PREFIX / "include" / "fplll.h").exists():
        return
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tarball = BUILD_DIR / "fplll-5.4.5.tar.gz"
    _download(FPLLL_URL, tarball, FPLLL_SHA256)
    src = BUILD_DIR / "fplll-5.4.5"
    if src.exists():
        shutil.rmtree(src)
    _run(["tar", "xf", str(tarball)], BUILD_DIR, dict(os.environ), "fplll untar")
    env = _build_env()
    _run(
        ["./configure", f"--prefix={LOCAL_PREFIX}"],
        src,
        env,
        "fplll configure",
    )
    _run(["make", f"-j{os.cpu_count() or 4}"], src, env, "fplll make")
    _run(["make", "install"], src, env, "fplll install")


def _patch_sollya_execute_h(src: Path) -> None:
    """Idempotent: give miniyyparse a prototype matching the bison definition,
    so it is not a GCC 14/15 'conflicting types' error."""
    h = src / "execute.h"
    s = h.read_text()
    old = "extern int miniyyparse();"
    new = "extern int miniyyparse(void *myScanner);"
    if old in s:
        h.write_text(s.replace(old, new))


def build_sollya() -> None:
    """Build Sollya 8.0 no-sudo into .local, with the GCC-15 fixes."""
    if (LOCAL_PREFIX / "bin" / "sollya").exists():
        return
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tarball = BUILD_DIR / "sollya-8.0.tar.gz"
    _download(SOLLYA_URL, tarball, SOLLYA_SHA256)
    src = BUILD_DIR / "sollya-8.0"
    if src.exists():
        shutil.rmtree(src)
    _run(["tar", "xf", str(tarball)], BUILD_DIR, dict(os.environ), "sollya untar")
    _patch_sollya_execute_h(src)
    # -std=gnu17 so the K&R empty-paren prototypes stay warnings, not C23 errors.
    env = _build_env(extra_cflags="-std=gnu17")
    _run(
        [
            "./configure",
            f"--prefix={LOCAL_PREFIX}",
            f"--with-fplll-include={LOCAL_PREFIX}/include",
            f"--with-fplll-lib={LOCAL_PREFIX}/lib",
        ],
        src,
        env,
        "sollya configure",
    )
    _run(["make", f"-j{os.cpu_count() or 4}"], src, env, "sollya make")
    _run(["make", "install"], src, env, "sollya install")


def toolchain_present() -> bool:
    return (LOCAL_PREFIX / "bin" / "sollya").exists() and not check_root_deps()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify the toolchain is present without building",
    )
    args = parser.parse_args(argv)

    if args.check_only:
        missing = check_root_deps()
        sollya_ok = (LOCAL_PREFIX / "bin" / "sollya").exists()
        if missing:
            sys.stderr.write(
                f"toolchain incomplete: missing root deps {missing}\n"
                f"  {INSTALL_HINTS['fedora']}\n"
            )
            return 2
        if not sollya_ok:
            sys.stderr.write(
                "toolchain incomplete: sollya not built. Run without --check-only.\n"
            )
            return 3
        print("proof toolchain present: sollya built, gappa + mpfi + libxml2 found")
        return 0

    require_root_deps()
    if args.force:
        for p in (LOCAL_PREFIX / "bin" / "sollya", LOCAL_PREFIX / "include" / "fplll.h"):
            p.unlink(missing_ok=True)
    build_fplll()
    build_sollya()
    print(
        "proof toolchain ready: sollya at .local/bin/sollya (run with "
        "LD_LIBRARY_PATH=.local/lib). Arb certifier deps (gcc/m4/make + "
        ".cargo/config -fPIC/caches) are configured separately for the `arb` "
        "cargo feature."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
