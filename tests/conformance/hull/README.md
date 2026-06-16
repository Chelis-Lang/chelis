# Hull Phase 5b conformance corpus + gate

This directory is the vendored, frozen, version-stamped Hull conformance corpus
plus the chelis-side runner that gates every compiler change against it.

## What this proves

Hull is the upstream type/effect/eval REFERENCE for the in-fragment Deep program
space. For every program the corpus pins Hull's authoritative verdict (accept vs
reject, the canonical type string, the reference scalar). The runner runs the
live `chelis` binary on each program and asserts the compiler AGREES with the
pinned Hull verdict. The acceptance oracle:

- **zero CompilerUnsound** (Hull rejects, compiler accepts, no documented gap),
- **zero unexplained Disagree** (both accept, types differ),
- **zero EvalDisagree** (both evaluate, scalars differ beyond tolerance),
- **EvalAgree at or above a floor**,
- zero compiler crashes.

A `CompilerTooConservative` (sound but Hull-incomplete) is a WARNING by default,
promotable with `--strict`.

## The acceptance oracle (the named done-condition)

```
.venv/bin/python tests/conformance/hull/run_conformance.py \
    --chelis-bin "$(pwd)/target/release/chelis"
```

Exit 0 + `ACCEPTANCE ORACLE: PASS` is the gate. This is what
`.github/workflows/conformance.yml` runs per PR. A **release** binary is
required (a debug-build check pass is multi-minute per program).

## The teeth

- **Reject sentinels** (`reject_NNNN`, `gap_family: GapNone`): ~46 programs Hull
  rejects and the compiler currently rejects too (unbound vars, arity/precision
  mismatches). If a future compiler ACCEPTS any of them, the runner derives
  CompilerUnsound and FAILS. This is the direction a real soundness regression
  takes.
- **Known-conservative evidence** (`gap_family != GapNone`): the 4 hand-curated
  reference-gap programs (ADT-fragment, builtin-name-shadow) Hull rejects but the
  compiler accepts. The runner routes these to KnownReferenceGap (PASS), proving
  it correctly disambiguates a reference gap from a false CompilerUnsound AND that
  it is live-comparing (it sees the compiler accept).
- **Injected-unsound self-test**: `test_run_conformance.py::test_injected_unsound_fails`
  and the end-to-end `run_conformance.py --simulate-unsound reject_0000` both
  prove the gate FAILS on a simulated soundness regression without an actual
  unsound compiler.

## How the verdict is pinned without re-implementing Hull in Python

The design pivot (`authoritative_verdict_pinning`): Hull pins each accepted
program's type as a CANONICAL NORMAL-FORM STRING it emits via `parse.ch::unparse_type`.
`wire_canonical.py::wire_to_canonical` re-renders the live `chelis check
--show-inferred` wire type into the SAME string (a pure syntactic transcription
mirroring `che_wire_type_to_type` + `inject_outer_effects` + `unparse_type`), and
the runner compares by PLAIN STRING EQUALITY. There is NO Python type lattice and
NO re-implemented `types_equal`. `wire_to_canonical` is golden-tested against
frozen Hull wire blobs (`golden_wire.json`) so a wire-schema drift fails LOUD.

## Files

- `manifest.json` -- provenance (Hull commit, pinned chelis version, seed,
  counts, per-rule-tag coverage).
- `programs/<id>.dp` -- one bare top-level `(def {} ...)` per program; the literal
  bytes fed to `chelis check` / `chelis eval`.
- `verdicts.jsonl` -- one pinned Hull verdict per program, joined by `id`.
- `known_conservative.json` -- Hull's 4 hand-curated reference-gap evidence
  entries, verbatim.
- `golden_wire.json` -- frozen (live wire blob, expected Hull string) pairs that
  guard `wire_to_canonical` against drift.
- `dropped_divergences.json` -- generated programs dropped from the agreement
  corpus because Hull's REFERENCE (not the compiler) is imprecise: the documented
  `gather`-shape and integer-`div` Hull gaps (filed to Hull `docs/v0_2_0_roadmap`).
- `run_conformance.py` -- the gate runner.
- `wire_canonical.py` -- the wire-to-canonical transcriber.
- `build_corpus.py` -- the assembler (Hull export JSONL -> this tree).
- `test_*.py` -- pure unit tests (run with `.venv/bin/python -m unittest`).

## Refreshing the corpus

The corpus is mechanically Hull-derived, never hand-edited. To refresh after a
deliberate compiler-semantics change:

1. In the Hull repo (at the commit recorded in `manifest.json::hull_commit`), run
   the producer driver to emit the export JSONL (a documented manual gate; the
   full 1000+450 export takes a few minutes):
   ```
   chelis test scripts/export_conformance_corpus.ch \
       --filter test_export_corpus --timeout 600
   ```
   This writes `corpus/scratch/export_corpus.jsonl`.
2. Assemble the chelis tree (runs the live binary to filter Hull-reference
   divergences + capture golden wire blobs):
   ```
   .venv/bin/python tests/conformance/hull/build_corpus.py \
       --export-jsonl <hull>/corpus/scratch/export_corpus.jsonl \
       --hull-known-conservative <hull>/corpus/known_conservative.json \
       --chelis-bin "$(pwd)/target/release/chelis" \
       --hull-commit <hull-commit> --chelis-version 0.7.26
   ```
3. Re-run the gate (step "the acceptance oracle" above) and the unit tests; the
   `git diff` on this directory is the reviewable corpus refresh.

The nightly (`conformance-nightly.yml`) runs the FULL fresh >=10k campaign with a
date-derived seed against the live Hull reference, catching regressions only a
NEW program would expose and re-validating the freeze.
