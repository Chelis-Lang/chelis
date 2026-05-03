#!/usr/bin/env python3
"""Build the `call_price_wrapped.spans.json` sidecar for the wrapped
Black-Scholes Deep fixture.

The wrapped fixture inlines bodies from Octant's translation of
`references/black_scholes_call_function.tex`. This script:

1. Runs `octant translate` on the multi-equation `.tex` to produce
   the canonical sidecar in a temp dir.
2. Filters that sidecar to just the span IDs that appear in
   `call_price_wrapped.dp`.
3. Adds wrapper-introduced span entries (`wrap_normal_cdf`,
   `wrap_call_price`) as synthesized entries with `latex_text` set
   to a sentinel (`__synthesized_wrap__`).
4. Writes `call_price_wrapped.spans.json` next to the wrapped `.dp`.

The script is idempotent: running it again regenerates the sidecar
to match the current Octant output.
"""

from __future__ import annotations

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

OCTANT_REPO = Path("/home/jeff/Documents/scratch/octant")
OCTANT_BIN = OCTANT_REPO / "target/release/octant"
OCTANT_TEX = OCTANT_REPO / "references/black_scholes_call_function.tex"

WRAPPER_SPAN_IDS = ("wrap_normal_cdf", "wrap_call_price")


def collect_span_ids_in_dp(dp_path: Path) -> list[str]:
    """Extract every {span: "<id>"} value from the Deep source text in
    appearance order. The Deep parser would be more rigorous; for the
    fixture this regex over canonical-printed source is sufficient."""
    text = dp_path.read_text()
    return re.findall(r'\{span:\s*"([^"]+)"', text)


def run_octant_translate() -> dict:
    if not OCTANT_BIN.exists():
        raise SystemExit(
            f"octant binary not found at {OCTANT_BIN}; "
            "build with `cargo build --release` in the Octant repo first"
        )
    if not OCTANT_TEX.exists():
        raise SystemExit(f"Octant LaTeX source not found at {OCTANT_TEX}")
    with tempfile.TemporaryDirectory() as td:
        out_dp = Path(td) / "tmp.dp"
        out_spans = Path(td) / "tmp.spans.json"
        subprocess.run(
            [
                str(OCTANT_BIN),
                "translate",
                str(OCTANT_TEX),
                "--output",
                str(out_dp),
                "--spans",
                str(out_spans),
            ],
            check=True,
        )
        return json.loads(out_spans.read_text())


def synthesized_wrapper_entry(span_id: str) -> dict:
    """An entry for a hand-authored wrapper span that has no LaTeX
    counterpart. The sidecar consumers only need `deep_node_id` and
    `latex_text`; we still emit the rest of the schema with sentinel
    values for shape compatibility."""
    return {
        "deep_node_id": span_id,
        "deep_path": "__synthesized_wrap__",
        "latex": {
            "source_id": -1,
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


def build_sidecar() -> dict:
    octant_sidecar = run_octant_translate()
    octant_entries = {e["deep_node_id"]: e for e in octant_sidecar["spans"]}
    wrapped_ids_in_order = collect_span_ids_in_dp(WRAPPED_DP)
    seen = set()
    spans_out = []
    for span_id in wrapped_ids_in_order:
        if span_id in seen:
            continue
        seen.add(span_id)
        if span_id in WRAPPER_SPAN_IDS:
            spans_out.append(synthesized_wrapper_entry(span_id))
        elif span_id in octant_entries:
            spans_out.append(octant_entries[span_id])
        else:
            raise SystemExit(
                f"span {span_id!r} in {WRAPPED_DP.name} is not in the Octant "
                f"sidecar and is not a wrapper-introduced ID; refusing to drop it"
            )
    return {
        "source": "references/black_scholes_call_function.tex",
        "source_hash": octant_sidecar["source_hash"],
        "spans": spans_out,
    }


def main() -> int:
    sidecar = build_sidecar()
    WRAPPED_SIDECAR.write_text(json.dumps(sidecar, indent=2) + "\n")
    n = len(sidecar["spans"])
    print(f"wrote {WRAPPED_SIDECAR.relative_to(REPO_ROOT)} ({n} spans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
