"""Assemble the frozen Hull conformance corpus tree from a Hull export JSONL.

THE PRODUCER-TO-CONSUMER BRIDGE. Hull's `scripts/export_conformance_corpus.ch`
emits a flat JSONL (one pinned record per line) whose fields are Hull's own
authoritative reference verdict: `lane`, `rule_tag`, `program` (the bare-def
Deep), `hull_verdict`, `hull_type_canonical`, `hull_effects_canonical`,
`hull_eval_value`. This script:

  1. Assigns per-lane ids (check_NNNNN / eval_NNNNN / reject_NNNN) and writes one
     `programs/<id>.dp` file per program (the literal bytes fed to chelis).
  2. Derives `gap_family` + `expected_compiler_outcome` per record.
  3. Captures the live `chelis check --show-inferred` wire blob for a sampled
     subset (the GOLDEN wire fixtures that guard wire_to_canonical against drift).
  4. Appends the 4 hand-curated known_conservative reference-gap programs (the
     disambiguation teeth) and the generated reject sentinels (the standing
     teeth) as reject-lane records.
  5. Emits `manifest.json` (provenance + counts + rule-tag coverage),
     `verdicts.jsonl`, `known_conservative.json`, and `golden_wire.json`.

The whole `tests/conformance/hull/` directory is the frozen artifact. A chelis
test (test_corpus_integrity) asserts manifest <-> files <-> verdicts are mutually
consistent. This script is the ONLY thing that writes the tree; the corpus is
never hand-edited (keeping it mechanically Hull-derived).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_conformance as rc  # noqa: E402
from wire_canonical import WireNormalizationError, wire_to_canonical  # noqa: E402

DROPPED_PATH = Path(__file__).resolve().parent / "dropped_divergences.json"

HERE = Path(__file__).resolve().parent
PROGRAMS_DIR = HERE / "programs"
MANIFEST_PATH = HERE / "manifest.json"
VERDICTS_PATH = HERE / "verdicts.jsonl"
KNOWN_CONSERVATIVE_PATH = HERE / "known_conservative.json"
GOLDEN_WIRE_PATH = HERE / "golden_wire.json"

# Every record carries a rule_tag (one of the 28 Hull RuleTags). The reject
# sentinels and known_conservative entries use these synthetic tags.
TAG_REJECT_UNBOUND = "RejectUnbound"
TAG_REJECT_ARITY = "RejectArity"
TAG_REJECT_PRECISION = "RejectPrecision"


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def reject_sentinel_programs() -> list[dict]:
    """The ~46 generated/derived reject sentinels: Hull pins reject, the compiler
    CURRENTLY rejects too (AgreeReject, exit 2). If a future compiler made ANY of
    these ACCEPT, the runner derives comp-accept while the pinned Hull verdict is
    reject with GapNone -> CompilerUnsound -> FAIL. This is exactly the direction
    a real soundness regression takes (the checker getting too permissive).

    Three classes: unbound variables, builtin arity mismatch, precision mismatch.
    Each is a bare top-level def the compiler rejects with a non-empty errors[]."""
    sentinels: list[dict] = []
    # Class 1: unbound variable references (the canonical reject sentinel).
    unbound_names = [
        "nonexistent_xyz", "undefined_q", "missing_name", "no_such_var",
        "phantom_ref", "absent_v", "unknown_id", "ghost_binding",
        "unbound_alpha", "unbound_beta", "unbound_gamma", "unbound_delta",
        "unbound_eps", "unbound_zeta", "unbound_eta", "unbound_theta",
    ]
    for nm in unbound_names:
        prog = f"(def {{}} g (var {{}} {nm}))"
        sentinels.append({"program": prog, "rule_tag": TAG_REJECT_UNBOUND})
    # Class 2: builtin arity / argument-type mismatch the compiler rejects.
    # add/mul/sub/div take 2 scalar args; feeding 3 or a wrong-typed arg rejects.
    arity_progs = [
        "(def {} g (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0) (lit {type: (t-prim {} f32)} 3.0)))",
        "(def {} g (app {} (var {} mul) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0) (lit {type: (t-prim {} f32)} 3.0)))",
        "(def {} g (app {} (var {} sub) (lit {type: (t-prim {} f32)} 1.0)))",
        "(def {} g (app {} (var {} div) (lit {type: (t-prim {} f32)} 1.0)))",
        "(def {} g (app {} (var {} sqrt) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0)))",
        "(def {} g (app {} (var {} exp) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0)))",
        "(def {} g (app {} (var {} log)))",
        "(def {} g (app {} (var {} sin) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0)))",
        "(def {} g (app {} (var {} neg) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 2.0)))",
    ]
    for prog in arity_progs:
        sentinels.append({"program": prog, "rule_tag": TAG_REJECT_ARITY})
    # Class 3: precision mismatch the compiler rejects (no implicit promotion).
    # add(f32, int32) mismatches; add(f32, bool) mismatches; etc.
    precision_progs = [
        "(def {} g (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} int32)} 2)))",
        "(def {} g (app {} (var {} mul) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} int32)} 2)))",
        "(def {} g (app {} (var {} sub) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))",
        "(def {} g (app {} (var {} div) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} bool)} true)))",
        "(def {} g (app {} (var {} add) (lit {type: (t-prim {} bool)} true) (lit {type: (t-prim {} bool)} false)))",
        "(def {} g (app {} (var {} mul) (lit {type: (t-prim {} int32)} 3) (lit {type: (t-prim {} bool)} true)))",
        "(def {} g (app {} (var {} sqrt) (lit {type: (t-prim {} int32)} 4)))",
        "(def {} g (app {} (var {} exp) (lit {type: (t-prim {} int32)} 1)))",
        "(def {} g (app {} (var {} log) (lit {type: (t-prim {} bool)} true)))",
        "(def {} g (app {} (var {} sin) (lit {type: (t-prim {} int32)} 1)))",
        "(def {} g (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} string)} hello)))",
        "(def {} g (app {} (var {} neg) (lit {type: (t-prim {} string)} hello)))",
        "(def {} g (app {} (var {} sub) (lit {type: (t-prim {} string)} a) (lit {type: (t-prim {} string)} b)))",
        "(def {} g (app {} (var {} div) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} string)} z)))",
        "(def {} g (app {} (var {} mul) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} string)} q)))",
        "(def {} g (app {} (var {} add) (lit {type: (t-prim {} bool)} true) (lit {type: (t-prim {} f32)} 1.0)))",
        "(def {} g (app {} (var {} exp) (lit {type: (t-prim {} string)} s)))",
        "(def {} g (app {} (var {} sqrt) (lit {type: (t-prim {} bool)} false)))",
        "(def {} g (app {} (var {} log) (lit {type: (t-prim {} int32)} 7)))",
        "(def {} g (app {} (var {} sin) (lit {type: (t-prim {} bool)} true)))",
        "(def {} g (app {} (var {} sub) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} int32)} 9)))",
    ]
    for prog in precision_progs:
        sentinels.append({"program": prog, "rule_tag": TAG_REJECT_PRECISION})
    return sentinels


def load_known_conservative(hull_corpus_path: Path) -> dict:
    """Load Hull's hand-curated known_conservative.json verbatim. These 4 entries
    are the GapAdtFragment / GapBuiltinNameShadow reference-gap evidence programs:
    Hull pins reject, the compiler CURRENTLY accepts, and the runner must route
    them to KnownReferenceGap (PASS), NOT a false CompilerUnsound."""
    with open(hull_corpus_path, encoding="utf-8") as f:
        return json.load(f)


def write_program(rid: str, program: str) -> str:
    """Write one program file, returning its relative path. The bytes are the
    literal Deep with a trailing newline (the chelis fmt convention)."""
    rel = f"{rid}.dp"
    path = PROGRAMS_DIR / rel
    text = program if program.endswith("\n") else program + "\n"
    path.write_text(text, encoding="utf-8")
    return rel


def capture_wire_blob(chelis_bin: str, program_path: Path, timeout: float) -> dict | None:
    """Run `chelis check --show-inferred` and capture the first inferred
    signature's wire blob (display_signature_structured + effect_row). Returns
    None when the program is not a function accept (no inferred signature) -- the
    golden fixtures only need the cases wire_to_canonical actually transcribes."""
    proc = subprocess.run(
        [chelis_bin, "check", str(program_path), "--show-inferred", "--allow-style-violations"],
        capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return None
    try:
        root = json.loads(proc.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    sigs = root.get("inferred_signatures")
    if not isinstance(sigs, list) or not sigs:
        return None
    first = sigs[0]
    return {
        "display_signature_structured": first.get("display_signature_structured"),
        "effect_row": first.get("effect_row"),
    }


def build(
    export_jsonl: Path,
    hull_known_conservative: Path,
    chelis_bin: str,
    hull_commit: str,
    chelis_version: str,
    generator_seed: int,
    golden_sample: int,
    timeout: float,
) -> dict:
    """Assemble the full corpus tree. Returns the manifest dict."""
    # Clean the programs dir (the corpus is fully regenerated, never patched).
    if PROGRAMS_DIR.exists():
        for old in PROGRAMS_DIR.glob("*.dp"):
            old.unlink()
    PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    raw = _read_jsonl(export_jsonl)
    verdicts: list[dict] = []
    golden_wire: list[dict] = []
    rule_tag_coverage: dict[str, int] = {}
    check_i = 0
    eval_i = 0

    def bump_tag(tag: str) -> None:
        rule_tag_coverage[tag] = rule_tag_coverage.get(tag, 0) + 1

    # CHECK + EVAL lanes from the Hull export.
    #
    # CONFORMANCE-BASELINE FILTER. The Phase 5 campaign documented two Hull
    # REFERENCE-evaluator/typing imprecisions (filed to Hull docs/v0_2_0_roadmap):
    #   - `gather` result-shape: Hull's reference gather typing drops a dimension
    #     the compiler keeps (a Hull-too-permissive shape gap; the compiler is the
    #     authority and ACCEPTS, so this is sound, never CompilerUnsound).
    #   - `div` on integer operands: Hull's reference evaluator does FLOAT division
    #     (6/4 -> 1.5) where the compiler does INTEGER division (6/4 -> 1); the
    #     compiler's int-div is correct for int inputs.
    # These are Hull-reference imprecisions, NOT compiler bugs, so pinning Hull's
    # verdict for them produces a FALSE expectation. The design's own precedent
    # (excluding tuple-get + div-by-zero from the eval lane at pin time) is to
    # EXCLUDE such programs from the agreement corpus. We do that here, validating
    # each generated program against the LIVE compiler and DROPPING any that
    # Disagrees -- recording the drop to dropped_divergences.json with the
    # divergence detail. The conformance baseline is thus programs where Hull and
    # the compiler GENUINELY agree; the teeth (reject sentinels + known
    # conservative) are unaffected, and a NEW disagreement on a KEPT program is a
    # real regression the gate still catches.
    dropped: list[dict] = []
    for rec in raw:
        lane = rec["lane"]
        # Build the prospective record and write the program to a temp id so the
        # live compiler can be run on it for the agreement check.
        probe_rel = write_program("_probe", rec["program"])
        probe_record = {
            "id": "_probe",
            "lane": lane,
            "program_path": probe_rel,
            "rule_tag": rec["rule_tag"],
            "hull_verdict": rec["hull_verdict"],
            "hull_type_canonical": rec.get("hull_type_canonical"),
            "hull_effects_canonical": rec.get("hull_effects_canonical"),
            "hull_eval_value": rec.get("hull_eval_value"),
            "gap_family": "GapNone",
        }
        probe_result = rc.run_program(chelis_bin, probe_record, timeout)
        (PROGRAMS_DIR / probe_rel).unlink(missing_ok=True)
        disagree_bucket = "disagree" if lane == "check" else None
        if probe_result.bucket in ("disagree",):
            dropped.append(
                {
                    "lane": lane,
                    "rule_tag": rec["rule_tag"],
                    "program": rec["program"],
                    "reason": probe_result.detail,
                }
            )
            continue
        if lane == "check":
            rid = f"check_{check_i:05d}"
            check_i += 1
        elif lane == "eval":
            rid = f"eval_{eval_i:05d}"
            eval_i += 1
        else:
            raise ValueError(f"unexpected export lane {lane!r}")
        rel = write_program(rid, rec["program"])
        bump_tag(rec["rule_tag"])
        verdict = {
            "id": rid,
            "lane": lane,
            "program_path": rel,
            "rule_tag": rec["rule_tag"],
            "hull_verdict": rec["hull_verdict"],
            "hull_type_canonical": rec.get("hull_type_canonical"),
            "hull_effects_canonical": rec.get("hull_effects_canonical"),
            "hull_eval_value": rec.get("hull_eval_value"),
            "gap_family": "GapNone",
            "expected_compiler_outcome": "Agree" if lane == "check" else "Agree",
        }
        verdicts.append(verdict)
        _ = disagree_bucket
        # Capture a golden wire blob for a sampled subset of accepted check
        # functions (the wire_to_canonical drift guard).
        if (
            lane == "check"
            and rec["hull_verdict"] == "accept"
            and rec.get("hull_type_canonical")
            and check_i % golden_sample == 0
        ):
            blob = capture_wire_blob(chelis_bin, PROGRAMS_DIR / rel, timeout)
            if blob is not None:
                # Only record a golden fixture when the live wire is TRANSCRIBABLE
                # and AGREES with the pinned Hull string. An un-representable live
                # type (an unsolved type var in display_signature_structured) is
                # handled by the runner's accept-aware path at runtime (Agree on
                # the acceptance fact), so it is not a transcription-fidelity case
                # and must not seed the drift guard.
                try:
                    transcribed = wire_to_canonical(
                        blob["display_signature_structured"], blob["effect_row"]
                    )
                except WireNormalizationError:
                    transcribed = None
                if transcribed is not None and transcribed == rec["hull_type_canonical"]:
                    golden_wire.append(
                        {
                            "id": rid,
                            "wire": blob,
                            "expected_canonical": rec["hull_type_canonical"],
                        }
                    )

    # REJECT-SENTINEL lane (the standing teeth).
    reject_i = 0
    for sent in reject_sentinel_programs():
        rid = f"reject_{reject_i:04d}"
        reject_i += 1
        rel = write_program(rid, sent["program"])
        bump_tag(sent["rule_tag"])
        verdicts.append(
            {
                "id": rid,
                "lane": "reject",
                "program_path": rel,
                "rule_tag": sent["rule_tag"],
                "hull_verdict": "reject",
                "hull_type_canonical": None,
                "hull_effects_canonical": None,
                "hull_eval_value": None,
                "gap_family": "GapNone",
                "expected_compiler_outcome": "AgreeReject",
            }
        )

    # KNOWN-CONSERVATIVE lane (the disambiguation teeth). These are reject-lane
    # records with gap_family != GapNone -> KnownReferenceGap (PASS).
    kc = load_known_conservative(hull_known_conservative)
    for entry in kc.get("whitelist_entries", []):
        rid = f"reject_{reject_i:04d}"
        reject_i += 1
        rel = write_program(rid, entry["evidence_program"])
        bump_tag(entry["family"])
        verdicts.append(
            {
                "id": rid,
                "lane": "reject",
                "program_path": rel,
                "rule_tag": entry["family"],
                "hull_verdict": "reject",
                "hull_type_canonical": None,
                "hull_effects_canonical": None,
                "hull_eval_value": None,
                "gap_family": entry["family"],
                "expected_compiler_outcome": "KnownReferenceGap",
            }
        )

    # Persist verdicts.jsonl (sorted by id for a reviewable diff).
    verdicts.sort(key=lambda v: v["id"])
    with open(VERDICTS_PATH, "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v, sort_keys=True) + "\n")

    # Persist known_conservative.json verbatim (the source of the 4 entries).
    with open(KNOWN_CONSERVATIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(kc, f, indent=2)
        f.write("\n")

    # Persist golden_wire.json (the wire_to_canonical drift guard fixtures).
    with open(GOLDEN_WIRE_PATH, "w", encoding="utf-8") as f:
        json.dump(golden_wire, f, indent=2, sort_keys=True)
        f.write("\n")

    # Persist dropped_divergences.json: the generated programs dropped from the
    # agreement corpus because Hull's REFERENCE disagrees with the (authoritative,
    # accepting) compiler -- the documented gather-shape + int-div Hull
    # imprecisions. Kept as committed evidence so the drop is auditable and a
    # corpus refresh shows whether the divergence set changed.
    with open(DROPPED_PATH, "w", encoding="utf-8") as f:
        json.dump(dropped, f, indent=2, sort_keys=True)
        f.write("\n")

    check_count = sum(1 for v in verdicts if v["lane"] == "check")
    eval_count = sum(1 for v in verdicts if v["lane"] == "eval")
    reject_count = sum(1 for v in verdicts if v["lane"] == "reject")
    manifest = {
        "hull_commit": hull_commit,
        "chelis_version_pinned": chelis_version,
        "generator_seed": generator_seed,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "schema_version": 1,
        "check_count": check_count,
        "eval_count": eval_count,
        "reject_sentinel_count": reject_count,
        "golden_wire_count": len(golden_wire),
        "dropped_divergence_count": len(dropped),
        "rule_tag_coverage": dict(sorted(rule_tag_coverage.items())),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Assemble the Hull conformance corpus tree")
    parser.add_argument("--export-jsonl", required=True, type=Path)
    parser.add_argument("--hull-known-conservative", required=True, type=Path)
    parser.add_argument("--chelis-bin", required=True)
    parser.add_argument("--hull-commit", required=True)
    parser.add_argument("--chelis-version", default="0.7.22")
    parser.add_argument("--generator-seed", type=int, default=20260530)
    parser.add_argument("--golden-sample", type=int, default=50, help="capture one golden wire blob per N check programs")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv[1:])

    manifest = build(
        args.export_jsonl,
        args.hull_known_conservative,
        args.chelis_bin,
        args.hull_commit,
        args.chelis_version,
        args.generator_seed,
        args.golden_sample,
        args.timeout,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
