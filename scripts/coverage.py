#!/usr/bin/env python3
"""Single-package region-coverage reading, via `cargo llvm-cov`.

Measures ONE workspace package and reports LLVM source-**region**
coverage for that package's own `src/` tree.

Regions, not lines, are the recorded metric. A region is finer than a
line: it splits individual `match` arms and short-circuit operands. That
distinction is the reason this tool exists. `crates/chelis-ir/src/
lower.rs` is an 11k-line dispatch over IR node shapes, and line coverage
scores a large `match` as covered the moment any single arm runs.

Region counts live only in llvm-cov's own JSON export, so this drives
`--json --summary-only` rather than `--lcov`. LCOV's `DA` records are
line hit counts; they cannot express the arm-level distinction above.

The run goes through `cargo llvm-cov nextest`, not `cargo llvm-cov
test`, so the measured suite matches the runner `scripts/gate.py` uses
for its `integration` stage.

FAIL-CLOSED: if the source filter matches no file, this exits nonzero
and names the filter. It does NOT report 0.0%. A path-filter defect, a
crate rename, and a genuine total loss of coverage all render as "0%",
but they demand opposite responses from whoever reads the output. The
same rule covers a missing `cargo-llvm-cov`, an unknown package, and a
malformed baseline.

The committed baseline is authoritative ONLY for the environment named
in its own provenance block. Region counts move with the rustc and LLVM
versions behind them, so a reading taken elsewhere is real but is not
comparable. A cross-environment comparison warns and labels its output;
it does not fail, because a developer on another host must still be able
to run this.

The baseline is hand-curated and explicitly regenerated, never written
by CI. Auto-regeneration would launder a coverage loss into the
baseline. This mirrors `scripts/test_timing_check.py`.

No percentage here is a threshold. This tool never exits 1: coverage
went down is a report, not a failure.

Usage:
    python3 scripts/coverage.py chelis-ir
    python3 scripts/coverage.py chelis-ir --baseline <path>
    python3 scripts/coverage.py chelis-ir --update-baseline
    python3 scripts/coverage.py chelis-ir --export <path>   # skip cargo

Exit codes:
    0  a summary was produced (whatever the numbers were)
    2  usage / IO / toolchain error
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
# Raw llvm-cov output goes under target/, which is already gitignored
# wholesale; the export is a build artifact, not a committed one.
RAW_OUTPUT_DIR = REPO_ROOT / "target" / "llvm-cov"
INSTALL_HINT = (
    "cargo-llvm-cov is provided by the devenv shell (devenv.nix); run "
    "`devenv shell`, or install it directly with `cargo install "
    "cargo-llvm-cov`"
)
RUNNER = "nextest"
PROVENANCE_KEYS = (
    "package",
    "rustc_version",
    "host_triple",
    "llvm_version",
    "runner",
    "recorded_at",
)


class CoverageError(Exception):
    """Raised for usage / IO / toolchain errors; mapped to exit code 2."""


def baseline_path_for(package: str, scripts_dir: Path = SCRIPTS_DIR) -> Path:
    """The committed baseline path for a package."""
    return scripts_dir / f"coverage_baseline_{package.replace('-', '_')}.json"


def validate_package_argument(package: str) -> str:
    """Reject anything that is not a single concrete package name.

    Measurement is deliberately single-package (see the module
    docstring's disk note and the change's design D6): a workspace-wide
    instrumented build compounds the runner disk pressure that
    chelis#392 already mitigates."""
    stripped = package.strip()
    if not stripped:
        raise CoverageError(
            "a package name is required; measurement is scoped to a "
            "single named package, never the whole workspace"
        )
    if any(ch in stripped for ch in "*?[]"):
        raise CoverageError(
            f"package name {package!r} looks like a pattern; measurement "
            f"is scoped to a single named package, never the whole "
            f"workspace"
        )
    return stripped


def workspace_member_packages(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Map `[package].name` -> repo-relative member directory.

    Reads the manifests directly rather than shelling out to `cargo
    metadata`, matching `scripts/gate.py`. The directory name is NOT
    assumed to equal the package name."""
    try:
        import tomllib
    except ModuleNotFoundError:
        raise CoverageError(
            "coverage.py needs Python 3.11+ (tomllib); run via "
            ".venv/bin/python per AGENTS.md"
        ) from None
    manifest = tomllib.loads((repo_root / "Cargo.toml").read_text())
    members: list[str] = manifest["workspace"]["members"]
    packages: dict[str, str] = {}
    for entry in members:
        if any(ch in entry for ch in "*?["):
            paths = sorted(p for p in repo_root.glob(entry) if p.is_dir())
        else:
            paths = [repo_root / entry]
        for path in paths:
            member_manifest = tomllib.loads((path / "Cargo.toml").read_text())
            rel = path.relative_to(repo_root).as_posix()
            packages[member_manifest["package"]["name"]] = rel
    return packages


def package_source_root(package: str, repo_root: Path = REPO_ROOT) -> Path:
    """The absolute `src/` directory of a workspace member.

    Raises if the package is not a workspace member, or is one but has
    no `src/` directory -- the latter would otherwise reach the filter
    and be indistinguishable from a coverage collapse."""
    packages = workspace_member_packages(repo_root)
    if package not in packages:
        known = ", ".join(sorted(packages))
        raise CoverageError(
            f"unknown package {package!r}: not a workspace member. "
            f"Known packages: {known}"
        )
    src = repo_root / packages[package] / "src"
    if not src.is_dir():
        raise CoverageError(
            f"package {package!r} is a workspace member but has no source "
            f"directory at {src}"
        )
    return src


def require_llvm_cov() -> None:
    """Fail loudly when `cargo-llvm-cov` is absent."""
    if shutil.which("cargo-llvm-cov") is None:
        raise CoverageError(f"cargo-llvm-cov not found on PATH. {INSTALL_HINT}")


def run_llvm_cov(
    package: str, output_path: Path, repo_root: Path = REPO_ROOT
) -> Path:
    """Run the instrumented suite and write llvm-cov's JSON export.

    `--summary-only` omits the per-line segment arrays, which are large
    and unused here; the per-file `summary.regions` block this tool
    reads is still present."""
    require_llvm_cov()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "cargo",
        "llvm-cov",
        RUNNER,
        "-p",
        package,
        "--json",
        "--summary-only",
        "--output-path",
        str(output_path),
    ]
    print(f"coverage: running {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=repo_root, check=False)
    if result.returncode != 0:
        raise CoverageError(
            f"`cargo llvm-cov` exited {result.returncode}; no coverage "
            f"export was produced"
        )
    if not output_path.is_file():
        raise CoverageError(
            f"`cargo llvm-cov` reported success but wrote no export at "
            f"{output_path}"
        )
    return output_path


def parse_export(path: Path) -> dict[str, tuple[int, int]]:
    """Parse llvm-cov's JSON export into `filename -> (covered, total)`
    region counts.

    Export shape (`"type": "llvm.coverage.json.export"`):
        {"data": [{"files": [{"filename": ..., "summary":
         {"regions": {"covered": N, "count": N, ...}}}]}]}
    """
    if not path.is_file():
        raise CoverageError(
            f"coverage export not found: {path}. Run without --export to "
            f"produce one."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CoverageError(f"malformed coverage export {path}: {exc}") from exc
    if not isinstance(data, dict) or "data" not in data:
        raise CoverageError(
            f"coverage export {path} is not an llvm-cov JSON export "
            f"(no top-level `data` key)"
        )
    entries = data["data"]
    if not isinstance(entries, list) or not entries:
        raise CoverageError(
            f"coverage export {path}: `data` is empty; the instrumented "
            f"run produced no coverage mapping"
        )
    out: dict[str, tuple[int, int]] = {}
    for block in entries:
        files = block.get("files") if isinstance(block, dict) else None
        if not isinstance(files, list):
            raise CoverageError(
                f"coverage export {path}: a `data` block has no `files` list"
            )
        for entry in files:
            try:
                filename = entry["filename"]
                regions = entry["summary"]["regions"]
                covered = int(regions["covered"])
                total = int(regions["count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CoverageError(
                    f"coverage export {path}: a file entry is missing "
                    f"`filename` or `summary.regions.covered/count`: {exc}"
                ) from exc
            out[str(filename)] = (covered, total)
    if not out:
        raise CoverageError(
            f"coverage export {path}: no file entries; the instrumented "
            f"run measured nothing"
        )
    return out


def filter_to_package(
    entries: dict[str, tuple[int, int]],
    source_root: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, tuple[int, int]]:
    """Keep only sources under the measured package's `src/`, keyed by
    repo-relative path.

    `cargo llvm-cov -p <pkg>` instruments the whole dependency graph, so
    the raw export also carries `chelis-deep`, `chelis-surf`, and the
    rest. Those are somebody else's coverage and must not enter this
    package's totals.

    An empty result is an ERROR, not 0%. See the module docstring."""
    kept: dict[str, tuple[int, int]] = {}
    for filename, counts in entries.items():
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        # Always resolve: cargo may report a path through a symlinked
        # checkout, which would not match the resolved source root.
        candidate = candidate.resolve()
        try:
            candidate.relative_to(source_root)
        except ValueError:
            continue
        try:
            key = candidate.relative_to(repo_root).as_posix()
        except ValueError:
            key = candidate.as_posix()
        kept[key] = counts
    if not kept:
        raise CoverageError(
            f"coverage filter matched no source file under {source_root}. "
            f"The export listed {len(entries)} file(s). This is a broken "
            f"filter or a moved source root, not zero coverage -- refusing "
            f"to report 0.0%."
        )
    return kept


def rustc_provenance() -> dict[str, str]:
    """Read `release:`, `host:`, and `LLVM version:` out of `rustc -vV`."""
    try:
        result = subprocess.run(
            ["rustc", "-vV"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise CoverageError(f"could not run `rustc -vV`: {exc}") from exc
    if result.returncode != 0:
        raise CoverageError(
            f"`rustc -vV` exited {result.returncode}; cannot record "
            f"provenance"
        )
    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    missing = [k for k in ("release", "host", "LLVM version") if k not in fields]
    if missing:
        raise CoverageError(
            f"`rustc -vV` output lacks {', '.join(missing)}; cannot record "
            f"provenance"
        )
    return {
        "rustc_version": fields["release"],
        "host_triple": fields["host"],
        "llvm_version": fields["LLVM version"],
    }


def build_provenance(package: str, now: str | None = None) -> dict[str, str]:
    """The provenance block written alongside the counts."""
    recorded_at = now or _datetime.datetime.now(
        _datetime.timezone.utc
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "package": package,
        "runner": RUNNER,
        "recorded_at": recorded_at,
        **rustc_provenance(),
    }


def load_baseline(path: Path) -> tuple[dict[str, str], dict[str, tuple[int, int]]]:
    """Return `(provenance, counts)` from a committed baseline file."""
    if not path.is_file():
        raise CoverageError(
            f"missing coverage baseline: {path}. Record one with "
            f"`--update-baseline`."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CoverageError(
            f"malformed coverage baseline {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CoverageError(
            f"coverage baseline {path} must be a JSON object"
        )
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        raise CoverageError(
            f"coverage baseline {path} has no `provenance` object; a "
            f"baseline without provenance cannot be compared safely"
        )
    missing = [k for k in PROVENANCE_KEYS if k not in provenance]
    if missing:
        raise CoverageError(
            f"coverage baseline {path} provenance is missing: "
            f"{', '.join(missing)}"
        )
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise CoverageError(
            f"coverage baseline {path} has no non-empty `files` object"
        )
    counts: dict[str, tuple[int, int]] = {}
    for key, value in files.items():
        try:
            counts[str(key)] = (int(value["covered"]), int(value["total"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise CoverageError(
                f"coverage baseline {path}: entry {key!r} lacks integer "
                f"`covered`/`total`: {exc}"
            ) from exc
    return {str(k): str(v) for k, v in provenance.items()}, counts


def environment_matches(
    baseline_provenance: dict[str, str], current: dict[str, str]
) -> bool:
    """Whether a baseline is comparable to the current environment."""
    return all(
        baseline_provenance.get(key) == current.get(key)
        for key in ("rustc_version", "host_triple")
    )


def percent(covered: int, total: int) -> float:
    """Region coverage percentage; a file with no regions counts as 100%."""
    if total == 0:
        return 100.0
    return 100.0 * covered / total


def write_baseline(
    path: Path, provenance: dict[str, str], counts: dict[str, tuple[int, int]]
) -> None:
    """Write a baseline file. Only ever called from `--update-baseline`."""
    payload = {
        "provenance": provenance,
        "files": {
            key: {"covered": counts[key][0], "total": counts[key][1]}
            for key in sorted(counts)
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def render_summary(
    counts: dict[str, tuple[int, int]],
    provenance: dict[str, str],
    baseline: dict[str, tuple[int, int]] | None = None,
    comparable: bool = True,
) -> str:
    """Per-file region coverage, worst gap first.

    Sorted by UNCOVERED region count descending, not by percentage: the
    question this output answers is "where is the most unreached code",
    and a 40%-covered 12-line file is not that place."""
    lines: list[str] = []
    total_covered = sum(c for c, _ in counts.values())
    total_regions = sum(t for _, t in counts.values())
    lines.append(
        f"coverage: {provenance['package']} region coverage "
        f"{percent(total_covered, total_regions):.1f}% "
        f"({total_covered}/{total_regions} regions, {len(counts)} files)"
    )
    lines.append(
        f"  environment: rustc {provenance['rustc_version']} / "
        f"{provenance['host_triple']} / LLVM "
        f"{provenance['llvm_version']} / runner {provenance['runner']}"
    )
    if baseline is not None and not comparable:
        lines.append(
            "  WARNING: baseline was recorded in a different environment; "
            "deltas below are CROSS-ENVIRONMENT and not directly comparable"
        )
    lines.append("")
    header = f"  {'uncovered':>9}  {'regions':>7}  {'pct':>6}  file"
    if baseline is not None:
        header += "  (delta)"
    lines.append(header)
    ordered = sorted(
        counts, key=lambda k: (counts[k][1] - counts[k][0], k), reverse=True
    )
    for key in ordered:
        covered, total = counts[key]
        row = (
            f"  {total - covered:>9}  {total:>7}  "
            f"{percent(covered, total):>5.1f}%  {key}"
        )
        if baseline is not None:
            if key in baseline:
                before = percent(*baseline[key])
                delta = percent(covered, total) - before
                row += f"  ({delta:+.1f} pts)"
            else:
                row += "  (new)"
        lines.append(row)
    if baseline is not None:
        dropped = sorted(set(baseline) - set(counts))
        for key in dropped:
            lines.append(f"  {'-':>9}  {'-':>7}  {'-':>6}  {key}  (gone)")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Report LLVM region coverage for one workspace package. "
            "Never gates: no percentage here is a threshold."
        ),
    )
    p.add_argument(
        "package",
        nargs="?",
        help="Workspace package to measure, e.g. chelis-ir. Required.",
    )
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=(
            "Baseline file to compare against. Default: "
            "scripts/coverage_baseline_<package>.json when it exists."
        ),
    )
    p.add_argument(
        "--export",
        type=Path,
        default=None,
        help=(
            "Read an existing llvm-cov JSON export instead of running "
            "cargo. Used by the tests and for re-reading a prior run."
        ),
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the filtered per-file counts to this path.",
    )
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "Overwrite the committed baseline from this run and exit. The "
            "baseline is hand-curated; this is the one documented "
            "regeneration command, and no workflow may invoke it."
        ),
    )
    # Recognized only so the refusal is explicit and legible. argparse
    # would otherwise reject them as unknown options with a message that
    # says nothing about why workspace-wide measurement is out of scope.
    p.add_argument("--workspace", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.workspace or args.all:
            raise CoverageError(
                "measurement is scoped to a single named package, never "
                "the whole workspace: an instrumented workspace build "
                "compounds the CI disk pressure chelis#392 mitigates"
            )
        if args.package is None:
            raise CoverageError(
                "a package name is required; measurement is scoped to a "
                "single named package, never the whole workspace"
            )
        package = validate_package_argument(args.package)
        # The module-level paths are passed explicitly, not left to
        # default arguments: a default is bound at def time, so a test
        # that points the tool at a fixture workspace could not override
        # it.
        source_root = package_source_root(package, REPO_ROOT)

        if args.export is not None:
            export_path = args.export
        else:
            export_path = RAW_OUTPUT_DIR / f"{package}.json"
            run_llvm_cov(package, export_path, REPO_ROOT)

        counts = filter_to_package(
            parse_export(export_path), source_root, REPO_ROOT
        )
        provenance = build_provenance(package)

        if args.update_baseline:
            target = args.baseline or baseline_path_for(package, SCRIPTS_DIR)
            write_baseline(target, provenance, counts)
            print(
                f"coverage baseline regenerated: {len(counts)} files "
                f"written to {target}"
            )
            return 0

        baseline_counts = None
        comparable = True
        baseline_file = args.baseline or baseline_path_for(package, SCRIPTS_DIR)
        if args.baseline is not None or baseline_file.is_file():
            baseline_provenance, baseline_counts = load_baseline(baseline_file)
            comparable = environment_matches(baseline_provenance, provenance)

        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            write_baseline(args.json_out, provenance, counts)
    except CoverageError as exc:
        print(f"coverage: error: {exc}", file=sys.stderr)
        return 2

    print(render_summary(counts, provenance, baseline_counts, comparable))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
