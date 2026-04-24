"""External-oracle parity check for PseudoNautilus.Special.erf_approx.

This is the sanctioned spot for "Python-only for external-oracle parity"
work. The Chelis-side identity tests in ../tests/special.ch already prove
internal correctness (oddness, monotonicity, boundedness, erf(0) = 0). This
script's job is the complementary check: does our A-S 7.1.26 implementation
agree numerically with scipy.special.erf at a handful of sample points?

Usage:
    python3 run_parity.py              # prints a comparison table
    python3 run_parity.py --strict     # non-zero exit if any sample is off

Requires scipy and a chelis binary on PATH. If scipy is missing, prints a
diagnostic and exits 0 so the harness is non-fatal in minimal environments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SAMPLES: list[float] = [-2.0, -1.0, -0.3, 0.0, 0.3, 0.7, 1.0, 2.0, 3.0]
TOLERANCE: float = 5e-6


def load_scipy_erf():
    try:
        from scipy.special import erf  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"scipy not available ({exc}); skipping parity check", file=sys.stderr)
        return None
    return erf


def chelis_eval_erf(pkg_root: Path, x: float) -> float:
    """Invoke `chelis eval --file` on a short Chelis program that calls
    erf_approx(x) and prints the result. We run from the pseudo_nautilus
    package root so the module resolver can see src/special.ch via reef.toml.

    The probe file has to live under src/ — `chelis eval --file` rejects
    paths outside the declared source directory. We drop it as
    src/_parity_probe.ch, run eval, then unlink; chelis build output caches
    and reef.lock tolerate a new sibling just fine.

    If the chelis binary is not on PATH, or the reef environment is not
    published, this returns NaN and the caller reports the gap without
    failing the whole harness.
    """
    expr = (
        "module PseudoNautilus.Probe\n"
        "import PseudoNautilus.Special (erf_approx)\n"
        f"result = erf_approx(cast({x!r}, f32))\n"
    )
    probe_path = pkg_root / "src" / "probe.ch"
    probe_path.write_text(expr)
    try:
        completed = subprocess.run(
            ["chelis", "eval", "--file", str(probe_path)],
            cwd=str(pkg_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"chelis eval failed for x={x}: {exc}", file=sys.stderr)
        return float("nan")
    finally:
        try:
            probe_path.unlink()
        except FileNotFoundError:
            pass
    if completed.returncode != 0:
        print(
            f"chelis eval nonzero exit for x={x}: {completed.stderr}",
            file=sys.stderr,
        )
        return float("nan")
    # Extract the last floating-point number on stdout. `chelis eval` prints
    # the final binding's value; we don't rely on JSON output here so the
    # script stays portable across eval output formats.
    for token in reversed(completed.stdout.split()):
        try:
            return float(token)
        except ValueError:
            continue
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any sample exceeds the parity tolerance",
    )
    args = parser.parse_args()

    erf = load_scipy_erf()
    if erf is None:
        return 0

    pkg_root = Path(__file__).resolve().parent.parent
    print(f"pseudo_nautilus parity: {pkg_root}")
    print(f"{'x':>6}  {'chelis':>12}  {'scipy':>12}  {'|diff|':>10}")
    print("-" * 48)

    worst = 0.0
    worst_x: float | None = None
    for x in SAMPLES:
        ours = chelis_eval_erf(pkg_root, x)
        ref = float(erf(x))
        diff = abs(ours - ref) if ours == ours else float("inf")
        if diff > worst:
            worst = diff
            worst_x = x
        print(f"{x:>6.3f}  {ours:>12.8f}  {ref:>12.8f}  {diff:>10.2e}")

    print(f"max |diff| = {worst:.2e} at x = {worst_x}")
    if args.strict and worst > TOLERANCE:
        print(
            f"FAIL: worst parity error {worst:.2e} > tol {TOLERANCE:.2e}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
