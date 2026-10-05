#!/usr/bin/env python3
"""Build the `call_price_wrapped.spans.json` sidecar for the wrapped
Black-Scholes Deep fixture.

The wrapped fixture inlines bodies from Octant's translation of
`references/black_scholes_call_function.tex`. This script:

1. Runs `octant translate` on the multi-equation `.tex` to produce
   the canonical sidecar in a temp dir.
2. Filters that sidecar to just the span IDs that appear in
   `call_price_wrapped.dp`.
3. Adds a single synthesized entry for the canonical
   `__synthesized_wrap__` marker (per
   `spec/03-deep-syntax.md` §1.1.1) shared by both wrapper defs.
4. Writes `call_price_wrapped.spans.json` next to the wrapped `.dp`.

The script is idempotent: running it again regenerates the sidecar
to match the current Octant output.

Usage:
    python3 scripts/build_wrapped_spans_sidecar.py --octant-repo <octant checkout>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "crates/chelis-cli/tests/fixtures/octant/black_scholes"
WRAPPED_DP = FIXTURE_DIR / "call_price_wrapped.dp"
WRAPPED_SIDECAR = FIXTURE_DIR / "call_price_wrapped.spans.json"

OCTANT_BIN = Path("target/release/octant")
OCTANT_TEX = Path("references/black_scholes_call_function.tex")

SYNTHESIZED_WRAP_MARKER = "__synthesized_wrap__"


def collect_span_ids_in_dp(dp_path: Path) -> list[str]:
    """Extract every {span: "<id>"} value from the Deep source text in
    appearance order. The Deep parser would be more rigorous; for the
    fixture this regex over canonical-printed source is sufficient."""
    text = dp_path.read_text()
    return re.findall(r'\{span:\s*"([^"]+)"', text)


def run_octant_translate(octant_repo: Path) -> dict:
    octant_bin = octant_repo / OCTANT_BIN
    octant_tex = octant_repo / OCTANT_TEX
    if not octant_bin.exists():
        raise SystemExit(
            f"octant binary not found at {octant_bin}; "
            "build with `cargo build --release` in the Octant repo first"
        )
    if not octant_tex.exists():
        raise SystemExit(f"Octant LaTeX source not found at {octant_tex}")
    with tempfile.TemporaryDirectory() as td:
        out_dp = Path(td) / "tmp.dp"
        out_spans = Path(td) / "tmp.spans.json"
        subprocess.run(
            [
                str(octant_bin),
                "translate",
                str(octant_tex),
                "--output",
                str(out_dp),
                "--spans",
                str(out_spans),
            ],
            check=True,
        )
        return json.loads(out_spans.read_text())


def synthesized_wrapper_entry() -> dict:
    """An entry for the canonical `__synthesized_wrap__` marker shared
    by multiple wrapper-shell nodes in the .dp. The sidecar consumers
    only need `deep_node_id` and `latex_text`; we still emit the rest
    of the schema with sentinel values for shape compatibility.
    `source_id: 0` is required because Octant's `SourceId` is a `u32`
    (it cannot represent a `-1` sentinel)."""
    return {
        "deep_node_id": SYNTHESIZED_WRAP_MARKER,
        "deep_path": "__synthesized_wrap__",
        "latex": {
            "source_id": 0,
            "start_byte": 0,
            "end_byte": 0,
            "start_line": 0,
            "start_column": 0,
            "end_line": 0,
            "end_column": 0,
            "label": None,
        },
        "latex_text": "__synthesized_wrap__",
    }


def build_sidecar(octant_repo: Path) -> dict:
    octant_sidecar = run_octant_translate(octant_repo)
    octant_entries = {e["deep_node_id"]: e for e in octant_sidecar["spans"]}
    wrapped_ids_in_order = collect_span_ids_in_dp(WRAPPED_DP)
    seen = set()
    spans_out = []
    for span_id in wrapped_ids_in_order:
        if span_id in seen:
            continue
        seen.add(span_id)
        if span_id == SYNTHESIZED_WRAP_MARKER:
            spans_out.append(synthesized_wrapper_entry())
        elif span_id in octant_entries:
            spans_out.append(octant_entries[span_id])
        else:
            raise SystemExit(
                f"span {span_id!r} in {WRAPPED_DP.name} is not in the Octant "
                f"sidecar and is not the canonical synthesized-wrap marker; "
                f"refusing to drop it"
            )
    return {
        "source": "references/black_scholes_call_function.tex",
        "source_hash": octant_sidecar["source_hash"],
        "spans": spans_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--octant-repo",
        type=Path,
        required=True,
        help="an Octant checkout with a release build of `octant`",
    )
    args = parser.parse_args()
    sidecar = build_sidecar(args.octant_repo.resolve())
    WRAPPED_SIDECAR.write_text(json.dumps(sidecar, indent=2) + "\n")
    n = len(sidecar["spans"])
    print(f"wrote {WRAPPED_SIDECAR.relative_to(REPO_ROOT)} ({n} spans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
