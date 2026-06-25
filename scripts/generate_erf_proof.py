#!/usr/bin/env python3
"""WI-13 Option B: generate the erf central-arm Sollya + Gappa proof bundle.

This is the machine-checkable PROOF half of the envelope generator. It drives
Sollya (remez central polynomial + per-sub-interval certified Taylor models)
and Gappa (machine-checks |p(x) - T(x)| <= bound on each sub-interval), and
emits the committed proof bundle under
``crates/chelis-prove/data/erf_proof/``:

  - ``central_<k>.gappa`` : a self-contained, machine-checkable Gappa proof for
    sub-interval k. An auditor re-runs ``gappa central_<k>.gappa`` and gets a
    clean (proved) exit.
  - ``manifest.json``      : the central polynomial coefficients, per-interval
    proved bounds + certified Taylor remainders + local eps, the committed
    central eps, and a sha256 over the .gappa bundle.
  - ``generate_erf_proof.sollya`` : the Sollya driver (committed; this script
    prepends the run parameters to it).

Trust model: Sollya proposes the polynomial and the certified Taylor remainders;
Gappa machine-checks the polynomial bound. The committed central eps is the max
over sub-intervals of (Gappa-proved bound + certified remainder). An independent
Arb whole-box certifier (the kept Option-A path) re-validates the same eps every
CI build as a cross-check; see ``scripts/generate_erf_envelope.py`` and the
``arb``-gated CI test.

Prerequisites (installed by ``scripts/setup_proof_toolchain.py``):
  - the Sollya binary built into ``.local/bin/sollya`` (+ ``.local/lib`` on
    LD_LIBRARY_PATH),
  - the ``gappa`` binary on PATH.

Usage:
  .venv/bin/python scripts/generate_erf_proof.py [--degree N] [--subintervals N]
      [--taylor-degree N] [--bisect N] [--margin-exp E] [--check-only]

  --check-only re-runs gappa on the already-committed bundle and re-validates
  the manifest (the CI gate); it does not regenerate.

Exit codes:
  0  bundle generated (or re-validated) and every Gappa proof checks
  2  a prerequisite (sollya/gappa) is missing
  3  a Gappa proof failed to check (unsound or under-bisected bundle)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROOF_DIR = REPO_ROOT / "crates" / "chelis-prove" / "data" / "erf_proof"
DRIVER = PROOF_DIR / "generate_erf_proof.sollya"
LOCAL_PREFIX = REPO_ROOT / ".local"
SOLLYA_BIN = LOCAL_PREFIX / "bin" / "sollya"

# Defaults.
#  Central: degree-21 remez over [-3, 3], 16 sub-intervals, degree-24 local
#  Taylor models -> committed central eps ~5.7e-7.
#  Tails: saturate at +-1 out to +-300. An "active zone" [3, 6] (and mirror) is
#  finely subdivided (where |1 - erf| is largest, ~2.2e-5 at x=3) to find the
#  committed tail eps; the far zone [6, 300] is covered by wider pieces all
#  proving the bound stays under that tail eps. The far Taylor models are
#  low-degree (erf is flat there).
DEFAULTS = dict(
    central_lo=-3,
    central_hi=3,
    degree=21,
    subintervals=16,
    taylor_degree=24,
    bisect=40,
    margin_exp=7,
    sat_hi=300,
    tail_active_hi=6,
    tail_active_n=12,
    tail_far_n=14,
    tail_degree=16,
    tail_far_degree=6,
)


def _sollya_env() -> dict:
    env = dict(os.environ)
    libdir = str(LOCAL_PREFIX / "lib")
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{libdir}:{prev}" if prev else libdir
    return env


def require_tools() -> None:
    if not SOLLYA_BIN.exists():
        sys.stderr.write(
            f"error: sollya not found at {SOLLYA_BIN}.\n"
            "Build it with: .venv/bin/python scripts/setup_proof_toolchain.py\n"
        )
        raise SystemExit(2)
    if (
        subprocess.run(
            ["gappa", "--version"], capture_output=True
        ).returncode
        != 0
    ):
        sys.stderr.write(
            "error: the `gappa` binary is not on PATH. Install it (Fedora: "
            "sudo dnf install -y gappa; Debian/Ubuntu: sudo apt-get install -y "
            "gappa) or run scripts/setup_proof_toolchain.py.\n"
        )
        raise SystemExit(2)


def run_sollya_driver(params: dict) -> None:
    """Prepend the run parameters to the committed Sollya driver and run it,
    emitting the per-sub-interval .gappa files and manifest_fragment.json."""
    margin = f"2^(-{params['margin_exp']})"
    header = (
        f"CENTRAL_LO = {params['central_lo']};\n"
        f"CENTRAL_HI = {params['central_hi']};\n"
        f"CENTRAL_DEGREE = {params['degree']};\n"
        f"N_SUBINTERVALS = {params['subintervals']};\n"
        f"TAYLOR_DEGREE = {params['taylor_degree']};\n"
        f"MARGIN = {margin};\n"
        f"BISECT = {params['bisect']};\n"
        f"SAT_HI = {params['sat_hi']};\n"
        f"TAIL_ACTIVE_HI = {params['tail_active_hi']};\n"
        f"TAIL_ACTIVE_N = {params['tail_active_n']};\n"
        f"TAIL_FAR_N = {params['tail_far_n']};\n"
        f"TAIL_DEGREE = {params['tail_degree']};\n"
        f"TAIL_FAR_DEGREE = {params['tail_far_degree']};\n"
        f'OUTDIR = "{PROOF_DIR}";\n'
    )
    script = header + DRIVER.read_text()
    run_path = PROOF_DIR / ".driver_run.sollya"
    run_path.write_text(script)
    try:
        proc = subprocess.run(
            [str(SOLLYA_BIN), str(run_path)],
            capture_output=True,
            text=True,
            env=_sollya_env(),
            timeout=900,
        )
    finally:
        run_path.unlink(missing_ok=True)
    # Sollya exits 3 on a script with no explicit quit(); that is normal.
    if proc.returncode not in (0, 3):
        sys.stderr.write(f"sollya failed (rc={proc.returncode}):\n{proc.stderr}\n")
        raise SystemExit(3)


def _gappa_files() -> list[Path]:
    """All committed .gappa proof files (central + both tails), name-sorted."""
    return sorted(PROOF_DIR.glob("*.gappa"), key=lambda p: p.name)


def center_qc_lines() -> None:
    """Rewrite the `qc = ...` line of each emitted .gappa from the variable `x`
    to the centered variable `t` (Sollya prints in x; Gappa bisects on t). Only
    the qc-definition line is rewritten; the `t = x - ...` line and the goal's
    `x in [...]` stay in x."""
    for gp in _gappa_files():
        lines = gp.read_text().splitlines()
        out = []
        for line in lines:
            if line.startswith("qc = "):
                # whole-word x -> t inside the polynomial body only
                line = "qc = " + re.sub(r"\bx\b", "t", line[len("qc = ") :])
            out.append(line)
        gp.write_text("\n".join(out) + "\n")


def check_all_proofs() -> None:
    """Run gappa on every committed sub-interval proof (all arms); raise if any
    fails. Every arm -- central and both saturation tails -- is a real Gappa
    proof term, so every .gappa must machine-check."""
    files = _gappa_files()
    if not files:
        sys.stderr.write("error: no .gappa proof files were emitted.\n")
        raise SystemExit(3)
    failures = []
    for gp in files:
        proc = subprocess.run(
            ["gappa", str(gp)], capture_output=True, text=True, timeout=300
        )
        if proc.returncode != 0:
            failures.append(
                (gp.name, proc.stderr.strip().splitlines()[-1:] or ["timeout"])
            )
    if failures:
        for name, why in failures:
            sys.stderr.write(f"  {name} FAILED: {why}\n")
        sys.stderr.write(
            "error: at least one Gappa proof did not check. The bundle is not a "
            "valid proof term; do not commit it.\n"
        )
        raise SystemExit(3)


def sha256_bundle() -> str:
    """Stable sha256 over the sorted .gappa proof files + the driver."""
    h = hashlib.sha256()
    for f in sorted([DRIVER, *_gappa_files()], key=lambda p: p.name):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def assemble_manifest(params: dict) -> None:
    frag = json.loads((PROOF_DIR / "manifest_fragment.json").read_text())
    # Store the central coefficients as JSON NUMBERS (the exact doubles the
    # Sollya driver produced via roundcoefficients(..., [|D ...|])). Emitting
    # them as numbers -- the same representation the committed envelope uses --
    # means the envelope and the manifest are read back through the identical
    # serde_json number parser, so the proof-bundle consistency test compares
    # bit-identical f64s. (Sollya prints the full ~90-digit exact decimal as a
    # string; parsing that long decimal can disagree by a ULP with serde_json's
    # number parser, so we re-emit through Python's float, which round-trips the
    # double losslessly as the shortest decimal.)
    frag["coeffs"] = [float(c) for c in frag["coeffs"]]
    frag["proof_kind"] = "gappa"
    frag["generator"] = "wi13-erf-proof-2"
    frag["bundle_sha256"] = sha256_bundle()
    (PROOF_DIR / "manifest.json").write_text(
        json.dumps(frag, indent=2) + "\n"
    )
    (PROOF_DIR / "manifest_fragment.json").unlink(missing_ok=True)


def revalidate_committed() -> None:
    """The CI gate: re-check the committed bundle's proofs and its sha256."""
    manifest = json.loads((PROOF_DIR / "manifest.json").read_text())
    check_all_proofs()
    recomputed = sha256_bundle()
    if recomputed != manifest["bundle_sha256"]:
        sys.stderr.write(
            "error: committed bundle sha256 mismatch -- a .gappa file or the "
            f"driver changed without regenerating the manifest.\n"
            f"  manifest: {manifest['bundle_sha256']}\n"
            f"  computed: {recomputed}\n"
        )
        raise SystemExit(3)
    n = len(_gappa_files())
    print(
        f"re-validated {n} Gappa proofs (central + both tails); bundle sha256 "
        f"matches; central_eps={manifest['central_eps']}, "
        f"tail_pos_eps={manifest['tail_pos_eps']}, tail_neg_eps={manifest['tail_neg_eps']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=DEFAULTS["degree"])
    parser.add_argument(
        "--subintervals", type=int, default=DEFAULTS["subintervals"]
    )
    parser.add_argument(
        "--taylor-degree", type=int, default=DEFAULTS["taylor_degree"]
    )
    parser.add_argument("--bisect", type=int, default=DEFAULTS["bisect"])
    parser.add_argument("--margin-exp", type=int, default=DEFAULTS["margin_exp"])
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="re-run gappa on the committed bundle and re-validate the manifest "
        "(the CI gate); do not regenerate.",
    )
    args = parser.parse_args(argv)

    if args.check_only:
        # Only gappa is needed to re-check the committed bundle.
        if (
            subprocess.run(["gappa", "--version"], capture_output=True).returncode
            != 0
        ):
            sys.stderr.write("error: `gappa` not on PATH for --check-only.\n")
            return 2
        revalidate_committed()
        return 0

    require_tools()
    params = dict(DEFAULTS)
    params.update(
        degree=args.degree,
        subintervals=args.subintervals,
        taylor_degree=args.taylor_degree,
        bisect=args.bisect,
        margin_exp=args.margin_exp,
    )
    # Stale .gappa files from a previous run with a different interval count
    # would otherwise be globbed into the bundle; clear them first.
    for old in PROOF_DIR.glob("*.gappa"):
        old.unlink()
    run_sollya_driver(params)
    center_qc_lines()
    check_all_proofs()
    assemble_manifest(params)
    manifest = json.loads((PROOF_DIR / "manifest.json").read_text())
    print(
        f"generated + proved {len(_gappa_files())} sub-intervals "
        f"(central + both tails); central_eps={manifest['central_eps']}; "
        f"tail_pos_eps={manifest['tail_pos_eps']}; tail_neg_eps={manifest['tail_neg_eps']}; "
        f"bundle_sha256={manifest['bundle_sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
