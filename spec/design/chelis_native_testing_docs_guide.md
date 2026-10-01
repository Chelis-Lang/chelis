# Docs Update Guide: Chelis-Native Testing Infrastructure

## Summary

Add `3t: Chelis-Native Testing` as a new Phase 3 sub-phase across all active design docs. This establishes a hard rule: all reef package tests are written in Chelis and run via `chelis test`, except for cross-language parity tests (scipy, pandas, QuantLib) which use Python. The deliverables are `Std.Test`, the `chelis test` CLI command, test conventions, and test migrations for Nautilus and Coral.

Full design: `spec/design/chelis_native_testing_plan.md`

## Target Files and Changes

### 1. Import the standalone plan

Create `spec/design/chelis_native_testing_plan.md` from the downloaded draft.

### 2. Edit `spec/design/chelis_project_plan.md`

**Add `3t` to Phase 3.** Find the Phase 3 sub-phase list (after the existing shell sub-phases like 3k, 3l, 3n, 3o). Insert:

> ### 3t: Chelis-Native Testing
>
> **Dependencies:** Bug 9 fix (eval hang on reef imports), fast `chelis eval` with package-aware imports.
>
> **Deliverables:**
> 1. `Std.Test` module in chelis-std — assertion functions (`assert_eq`, `assert_close`, `assert_close_tensor`, `assert_true`, `assert_false`, `fail`), either via a `Test` algebraic effect or runtime builtin.
> 2. `chelis test` CLI command — discovers `tests/*.ch` files, evaluates each via the evaluator (not build+gcc+link+run), calls every `def test_*()` function, reports pass/fail with structured output, exits 0/1.
> 3. Test file convention — `tests/*.ch` in every reef package, `def test_*()` naming, documented layout.
> 4. Nautilus test migration — mathematical identity tests, property tests, edge case tests, smoke tests move from Python to Chelis. Scipy parity tests remain in Python under `parity/`.
> 5. Coral test migration — same split: structural correctness tests move to Chelis, pandas parity stays in Python.
> 6. SKILL.md for `Std.Test` — documents assertion API for coding agents.
>
> **Hard rule:** Only code that compares Chelis output against an external oracle (scipy, pandas, QuantLib) uses Python. All other tests are written in Chelis and run via `chelis test`. This applies to all reef packages, current and future.
>
> Full design: `chelis_native_testing_plan.md`

**Update Pre-Phase 4 Investments.** The "fast `chelis eval`" entry already exists. Add a note that `chelis test` is built on top of fast eval — same dependency:

> `chelis test` is a thin layer on top of fast eval: discover test files, evaluate each, collect assertion results. The fast eval investment directly enables native testing infrastructure.

**Update Shoals (3l) section.** Add:

> Shoals ships with Chelis-native tests from day one (`tests/*.ch`, run via `chelis test`). Python parity only if comparing against QuantLib or other external pricing references.

**Update Octant section.** Add:

> Octant uses Python (sympy / latex2sympy2) as the external oracle for LaTeX parsing correctness: parse the same LaTeX in both Octant and sympy, compare expression trees. Same pattern as Nautilus vs scipy — sympy parity lives in `parity/`. Chelis-native tests cover tokenizer correctness, parser crash safety, provenance span accuracy, and parse→pretty-print round-trips.

**Update Hull section.** Add:

> Hull tests are entirely Chelis-native. Hull tests ARE Chelis programs testing Chelis — no Python involvement.

**Update future shells (School, Darwin).** Add the same convention note if these sections exist.

### 3. Edit `spec/design/chelis_canonical_reference.md`

**Add `chelis test` to the CLI commands table:**

> ```
> chelis test tests/          # discover and run Chelis-native test files
> chelis test tests/foo.ch    # run a specific test file
> chelis test tests/ --filter erf  # run only tests matching "erf"
> ```

**Add testing convention to Cross-Cutting Design Decisions:**

> **Chelis-native testing as the default.** All reef package tests are written in Chelis and run via `chelis test`, except for cross-language parity tests (comparing against scipy, pandas, or other external oracles) which use Python. This is a hard rule, not a guideline. Python test infrastructure exists only for parity verification against external libraries. `Std.Test` provides assertion functions; `chelis test` discovers and runs test files via the evaluator. The `Test` effect (or runtime builtin) tracks assertion pass/fail. Full design: `chelis_native_testing_plan.md`.

**Update the shell ecosystem table** — add `Std.Test` as part of chelis-std if the table lists chelis-std's contents.

### 4. Edit `spec/design/chelis_phase3_plan.md`

**Add `3t` as a new sub-phase.** Insert after the last existing sub-phase (before the summary/dependency table). Content mirrors the project plan entry above.

**Update the Phase 3 Dependency and Size Summary table.** Add a row:

> | 3t: Native Testing | After Bug 9 fix | chelis-std, chelis-cli | Std.Test module + chelis test command + test migrations for Nautilus and Coral | **Planned** |

**Add a note in any existing testing discussion** (if one exists) that the Python-only testing model is being superseded by native Chelis testing, with Python retained only for parity verification.

## Verification

After edits:

```bash
# Confirm 3t appears in project plan and phase3 plan
rg -n "3t.*Native Testing\|chelis test\|Std\.Test" spec/design/chelis_project_plan.md spec/design/chelis_phase3_plan.md

# Confirm CLI table updated
rg -n "chelis test" spec/design/chelis_canonical_reference.md

# Confirm hard rule stated
rg -n "hard rule" spec/design/chelis_canonical_reference.md spec/design/chelis_project_plan.md

# Confirm Shoals/Octant/Hull updated
rg -n "Chelis-native tests from day one\|no Python involvement\|No external oracle" spec/design/chelis_project_plan.md

# Confirm standalone plan exists
ls spec/design/chelis_native_testing_plan.md
```

## Assumptions

- The standalone plan doc is imported as-is
- `3t` is a Phase 3 item, not Phase 4 or 5
- Bug 9 fix and fast eval are prerequisites, not part of 3t itself
- No code changes in this round — docs only
- Existing Python test infrastructure is not removed in this docs update — the migration is recorded as planned work, not executed
