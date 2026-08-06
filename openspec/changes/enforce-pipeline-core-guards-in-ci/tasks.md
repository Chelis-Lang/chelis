# Tasks: enforce-pipeline-core-guards-in-ci

## 1. Spec-first failing tests

- [x] 1.1 In `scripts/test_pipeline_core_documentation_guard.py`, add negative cases that
  fail red against the current heuristic: `false_no_std_claim("chelis-pipeline-core is
  portable to no_std targets.")` and `false_no_std_claim("The core works in a no_std
  environment.")` MUST each return `True`. (`test_environment_phrasings_are_rejected`)
- [x] 1.2 Add the false-positive regression: `false_no_std_claim("chelis-pipeline-core
  supports std only, not no_std.")` MUST return `False`; `validate_text` over an
  inventory containing that exact line MUST NOT raise.
  (`test_true_requires_std_statement_is_not_a_false_claim`,
  `test_true_statement_inside_a_full_inventory_passes`)
- [x] 1.3 Add positive anchors that MUST stay `False`: a requires-std line, a
  future-target line, and a blocker-analysis line that mentions no_std without claiming
  crate capability. (`test_blocker_analysis_mention_is_not_a_false_claim`)
- [x] 1.4 Confirm 1.1–1.3 exercise `false_no_std_claim` directly and `validate_text`
  end to end; the review's standalone run recorded that 1.1–1.2 fail red against the
  pre-fix heuristic.

## 2. Reliable false-claim detection (D1)

- [x] 2.1 Rewrite `false_no_std_claim` in `scripts/pipeline_core_documentation_guard.py`:
  negation and future-target frames are checked before any positive match; the
  present-tense capability vocabulary now includes `runs in`, `works in`, `compiles as`,
  `portable to`, and the adjective forms; a bare `not no_std` and `no_std ... not` are
  recognized as negations.
- [x] 2.2 Task 1.1–1.3 green. `test_false_current_no_std_claim_is_rejected` and
  `test_future_target_is_not_a_false_current_claim` still pass unchanged (7/7).
- [x] 2.3 `scripts/pipeline_core_documentation_guard.py` still prints `PASS` on the
  shipped `docs/investigations/pipeline_core_std_blockers.md`.

## 3. Per-PR gate enforcement (D2, D3)

- [x] 3.1 Added three commands to `STAGES["lint-and-unit"]` in `scripts/gate.py` after the
  checkpoint fixture; the two cheap guards are also in `LOCAL_STATIC_COMMANDS`, the
  out-of-workspace compile-fail build stays gate/CI-only.
- [x] 3.2 `python3 scripts/gate.py --list` shows the three commands (dep + doc = `local +
  ci`, compile-fail = `ci-owned`); CI already runs `gate.py lint-and-unit`, no workflow
  edit required.
- [x] 3.3 Extended `scripts/test_gate.py`
  (`test_pipeline_core_boundary_guards_are_in_the_lint_and_unit_stage`,
  `test_cheap_pipeline_core_guards_are_in_the_local_subset`) and updated the exact
  `--local` list lock in `scripts/test_gate_local.py`.
- [x] 3.4 `python3 scripts/test_gate.py` (66) and `python3 scripts/test_gate_local.py`
  (24) green.

## 4. Docs and spec sync

- [x] 4.1 Updated `docs/investigations/compiler_pipeline_inventory.md` Dependency
  Boundary section: the three guards run in the per-PR gate, not only the manual oracle.
- [x] 4.2 Each delta scenario maps to a test in group 1 or a gate command in group 3.

## 5. Validation and oracle

- [ ] 5.1 `python3 scripts/gate.py lint-and-unit` green in an isolated
  `CARGO_TARGET_DIR`, including the three new guard commands. (Deferred to CI /
  full-gate: the workspace build is CI-owned on macOS; the two cheap guards and the
  guard/gate unit suites were run green locally.)
- [ ] 5.2 `.venv/bin/python scripts/compiler_pipeline_oracle.py` green — the authoritative
  completion oracle; it continues to run the same three guards plus the full parity
  suite.
- [x] 5.3 `openspec validate enforce-pipeline-core-guards-in-ci --strict` passes.
- [x] 5.4 Red team: a fresh-context local `pi` subagent (herdr workspace `w43`) ran the
  guard/gate suites, the dependency and compile-fail guards, and its own adversarial
  probes. Verdict: Claims 2/3/4 sound (guards pass in the gate, lowering refactor
  preserved behavior, Option C reasoning correct); Claim 1 was a real MEDIUM robustness
  gap — 14 natural false phrasings bypassed the first-cut heuristic (hyphen `no-std`
  spelling, `has no_std support`, `is #![no_std]`). Acted on: normalized the hyphen
  spelling, added hypothetical framing, re-added the `no_std support` adjective and
  adjacent `is/are no_std`. Re-probe: 0 false negatives, 0 false positives; suite 9/9;
  inventory still PASS.

### Validation record (2026-08-04)

- Documentation guard: `test_pipeline_core_documentation_guard.py` 9/9 green (7 initial +
  2 red-team regression tests); the guard still prints `PASS` on the shipped blocker
  inventory. Direct execution confirmed the pre-fix heuristic returned the wrong value
  for `portable to no_std`, `works in a no_std environment`, and `supports std only, not
  no_std`; a fresh-context red team then found the hyphen-spelling and
  `has no_std support` bypasses, now closed (0 false negatives / 0 false positives over a
  16-false/12-honest probe).
- Gate wiring: `test_gate.py` 66/66, `test_gate_local.py` 24/24. `gate.py --list` shows
  the two cheap guards as `local + ci` and the compile-fail fixture as `ci-owned`.
  `pipeline_core_dependency_guard.py` PASS against the live `cargo metadata`;
  `test_compiler_pipeline_oracle.py` + guard unit suites 18/18 green.
- Isolated `CARGO_TARGET_DIR=target/agents/review-flaws` used for cargo commands.
