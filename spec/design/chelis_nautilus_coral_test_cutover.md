# Nautilus and Coral — Cutover to `chelis test`

**Status:** Planned. Depends on **chelis v0.2.4** (first release that ships
`Std.Test` + `chelis test`). Blocking: none. Owner: whoever lands in
`chelis-lang/nautilus` and `chelis-lang/coral` repos respectively.

## Why

Phase 3t shipped the native testing system in the Chelis monorepo. The
spec's hard rule — *tests in Chelis, Python only for external-oracle parity* —
applies to every reef package. Nautilus and Coral today run every test
through Python (golden generation + compiled-C comparison + assertion
bookkeeping), which means:

- CI needs Python + scipy + gcc + the chelis runtime library to check
  internal correctness, even for tests that only verify mathematical
  identities.
- Every assertion is a Python-side numeric compare, not a Chelis-level
  type-checked call — so a silent drift in the shell's API (a wrong-dtype
  return, a shape change) can pass if the Python test happens to round
  correctly.
- Adding a new test requires editing Python fixtures + Chelis `.ch`
  fixtures + golden JSON; the round-trip is heavy.

After cutover:

- `chelis test tests/` runs inside each shell's CI as the default test
  gate. No gcc, no linking, no scipy for the internal correctness suite.
- `python parity/run_parity.py` is the *only* Python that remains, and
  it exists *solely* to compare Chelis output against the external
  oracle (scipy for Nautilus, pandas for Coral).
- New internal-correctness tests are one-file Chelis edits.

## Prerequisites (already shipped in chelis v0.2.4)

Nothing downstream is blocked. The downstream work can start immediately
against `chelis v0.2.4`:

- `Effect::Test`, `test_assert_*` runtime builtins, propagation in
  `chelis-effects`.
- `Std.Test` in `chelis-std` with 10 exports
  (`assert_true`/`assert_false`, `assert_eq`/`_int`/`_bool`/`_string`,
  `assert_close`, `assert_close_tensor`, `assert_shape`, `fail`).
- `chelis test` CLI with `--filter`, `--json`, `--timeout`, exit 0/1/2.
- Per-file subprocess isolation and one-compile-per-file perf.
- Docs: `spec/design/chelis_native_testing_plan.md` §Worked Example,
  `packages/chelis-std/SKILL.md` §Std.Test.

## The downstream rule

> Every test in `tests/*.ch` is written in Chelis and asserts against a
> mathematical identity, a known-exact value, a structural property, or
> a shape/type constraint. Python lives only in `parity/` and exists
> solely to compare Chelis output against scipy / pandas / QuantLib /
> sympy.

If you find yourself writing a Python test that doesn't call the
external oracle, stop — that test belongs in `tests/*.ch`.

---

## Part A — Nautilus cutover

Reference: current Python test harness in `chelis-lang/nautilus/tests/`
runs every numeric test against scipy via `run_numeric_tests.py` (~1051
assertions as of v0.1.0).

### A1. Add the reef package layout

```
chelis-lang/nautilus/
├── src/                     # unchanged
├── tests/                   # NEW: Chelis-native tests
│   ├── special.ch
│   ├── distributions.ch
│   ├── linalg.ch
│   ├── stats.ch
│   ├── optim.ch
│   ├── roots.ch
│   ├── ode.ch
│   └── ...
├── parity/                  # RENAMED from tests/ — Python only
│   ├── gen_goldens.py       # scipy → golden JSON
│   ├── run_parity.py        # compare Chelis eval output vs goldens
│   └── fixtures/            # golden JSON
└── reef.toml                # bump `compiler = "=0.2.4"`
```

`tests/` and `parity/` are siblings. `chelis test tests/` never touches
`parity/`; CI runs both explicitly.

Bump `reef.toml` to `compiler = "=0.2.4"` and add `chelis-std` as a
dep so `import Std.Test` resolves:

```toml
[dependencies]
chelis-std = { version = "0.1.0" }
```

### A2. Classify every existing Python assertion

Walk `tests/run_numeric_tests.py`. For each assertion, ask: *does the
expected value come from scipy, or from math?*

**Stays in Python (`parity/`):**
- `scipy.special.erf(0.7)` as expected value → parity check.
- `scipy.linalg.lu_solve(A, b)` as the reference → parity check.
- `scipy.stats.norm.ppf(0.975)` → parity check.
- Benchmark timings vs scipy → stays in Python (not a correctness test).

**Moves to Chelis (`tests/`):**
- `erf(0) == 0`, `erf(5) ≈ 1`, `gamma(1) == 1`, `gamma(0.5) == √π` →
  known-exact values. Constants come from the mathematical definition,
  not scipy. Compliant.
- `erf(-x) == -erf(x)`, `gamma(x+1) == x*gamma(x)`,
  `beta(a,b) == beta(b,a)` → mathematical identities.
- `Q^T Q ≈ I` (QR orthogonality), `U Σ V^T ≈ A` (SVD reconstruction)
  → structural / algebraic identities.
- `lt(erf(x), 1.0) && gt(erf(x), 0.0)` → range / bound check.
- `lt(erf(0.3), erf(0.7))` → monotonicity.
- `shape(result, 0) == expected_n`, `type_of(result) == tensor[n, f32]`
  → structural constraints.
- `erfinv(erf(x)) ≈ x` within tol → round-trip.
- Smoke: every exported function is callable on legal inputs without
  crashing → `assert_true(true, "smoke")` after the call returns.

**Rule of thumb:** if you can state the expected value without writing
`scipy.`, it's a tests/ test.

Expected split: ~60-70% of the 1051 assertions move to `tests/`,
~30-40% stay in `parity/` (scipy-derived expectations + benchmarks).

### A3. Write `tests/*.ch` — one file per Nautilus module

Template (use `tests/special.ch` as the starter — already in the
chelis_native_testing_plan.md worked example):

```chelis
module Nautilus.Tests.Special

import Nautilus.Special (erf, erfinv, gamma, lgamma, beta)
import Std.Test (assert_close, assert_true, assert_eq)

-- Known-exact values (no scipy)
def test_erf_zero() -> unit ! { Test } =
  assert_close(erf(cast(0.0, f32)), cast(0.0, f32), cast(1e-10, f32), "erf(0) = 0")

def test_erf_large() -> unit ! { Test } =
  assert_close(erf(cast(5.0, f32)), cast(1.0, f32), cast(1e-6, f32), "erf(5) ~ 1")

def test_gamma_one() -> unit ! { Test } =
  assert_close(gamma(cast(1.0, f32)), cast(1.0, f32), cast(1e-7, f32), "gamma(1) = 1")

def test_gamma_half() -> unit ! { Test } =
  assert_close(gamma(cast(0.5, f32)), cast(1.7724539, f32), cast(1e-5, f32), "gamma(0.5) = sqrt(pi)")

-- Mathematical identities
def test_erf_symmetry() -> unit ! { Test } = {
  x = cast(0.7, f32)
  assert_close(erf(x), sub(cast(0.0, f32), erf(sub(cast(0.0, f32), x))), cast(1e-7, f32), "erf is odd")
}

def test_erf_erfinv_roundtrip() -> unit ! { Test } = {
  x = cast(0.3, f32)
  assert_close(erfinv(erf(x)), x, cast(1e-5, f32), "erfinv(erf(x)) = x")
}

def test_gamma_recurrence() -> unit ! { Test } = {
  x = cast(3.5, f32)
  assert_close(gamma(add(x, cast(1.0, f32))), mul(x, gamma(x)), cast(1e-4, f32), "gamma(x+1) = x*gamma(x)")
}

-- Range / property checks
def test_erf_bounded() -> unit ! { Test } = {
  _ = assert_true(lt(erf(cast(0.5, f32)), cast(1.0, f32)), "erf(x) < 1 for finite x")
  assert_true(gt(erf(cast(0.5, f32)), cast(0.0, f32)), "erf(x) > 0 for x > 0")
}

def test_gamma_positive() -> unit ! { Test } =
  assert_true(gt(gamma(cast(2.5, f32)), cast(0.0, f32)), "gamma positive for x > 0")
```

Analogous files for `distributions.ch` (PDF normalization identity,
CDF monotonicity, inverse-CDF round-trip, sample-within-support
property), `linalg.ch` (orthogonality, SVD reconstruction, cholesky
`L*L^T ≈ A`, solve `A*x = b` residual), `stats.ch`, `optim.ch`,
`roots.ch` (Newton-Raphson converges on a known root), `ode.ch` (RK4
on a known ODE with closed-form solution).

### A4. Shrink `parity/run_parity.py` to scipy-only

After A3, every assertion that doesn't need scipy moves out. What's
left in `parity/` is a short script per module:

```python
# parity/run_parity.py
"""Compare Nautilus output against scipy reference values."""
import subprocess, json, sys
from pathlib import Path
import scipy.special, scipy.linalg, scipy.stats

PKG = Path(__file__).resolve().parent.parent

def chelis_eval_scalar(snippet: str) -> float:
    # write a probe into src/, run `chelis eval --file`, parse the last float
    ...

failed = 0
for x in [-2.0, -0.3, 0.7, 2.0, 5.0]:
    ours = chelis_eval_scalar(f"result = erf(cast({x!r}, f32))")
    ref = float(scipy.special.erf(x))
    if abs(ours - ref) > 1e-5:
        print(f"FAIL erf({x}): chelis={ours} scipy={ref}")
        failed += 1
# ... repeat for gamma, lgamma, beta, svd, solve, norm.ppf, etc.

sys.exit(0 if failed == 0 else 1)
```

Pattern for every parity check: call the Chelis function via
`chelis eval --file`, call the scipy reference, compare within tol,
exit 1 on any mismatch. Support a `--strict` flag that fails loudly
when scipy is missing (don't silent-skip in CI — see the pseudo-nautilus
fixture's run_parity.py for the shape).

### A5. CI wiring

Replace the existing Python-only test step with two parallel jobs:

```yaml
jobs:
  chelis-tests:
    steps:
      - uses: actions/checkout@v4
      - run: # install chelis v0.2.4
      - run: chelis reef publish packages/chelis-std  # once
      - run: chelis test tests/                      # all Chelis-native tests
  scipy-parity:
    steps:
      - uses: actions/checkout@v4
      - run: pip install scipy numpy
      - run: # install chelis v0.2.4
      - run: chelis reef publish packages/chelis-std
      - run: python parity/run_parity.py --strict
```

Both must pass for CI green. Removing `scipy-parity` is not an option —
it's the only thing that catches Chelis drift from scipy semantics.

### A6. Deletions (after A5 green for a week)

- Delete `tests/run_numeric_tests.py` (replaced by `chelis test tests/`).
- Delete `scripts/gen_goldens.py` if unused (parity/run_parity.py calls
  scipy directly now).
- Delete checked-in golden JSON fixtures if parity/run_parity.py
  derives them at runtime.

Keep a note in `CHANGELOG.md`:
> **Breaking (CI only):** Test harness rewritten on chelis v0.2.4
> native testing. Internal-correctness tests now run via `chelis test
> tests/`; scipy parity is the only remaining Python test step. See
> `spec/design/chelis_nautilus_coral_test_cutover.md`.

### A7. Acceptance

- `chelis test tests/` exits 0 on a clean tree (expect ~30-40 Chelis
  test files, ~700+ tests — one-to-one with migrated Python assertions).
- `python parity/run_parity.py --strict` exits 0 when scipy is
  installed.
- No `.py` files under `tests/` or `src/`.
- `tests/` contains NO scipy-derived numeric literals. Every expected
  value is a mathematical identity, a known-exact value (π, e, √π,
  1/√(2π), …), or derived from the Chelis call itself.
- Red-team: grep `tests/*.ch` for numbers with ≥4 decimal digits that
  aren't documented mathematical constants. Each hit is a candidate
  hard-rule violation (smuggled scipy output).

---

## Part B — Coral cutover

Coral's test story is even more Python-heavy today because the entire
suite is golden-fixture comparisons against pandas. Split is similar
in shape but different in content.

### B1. Add the layout

```
chelis-lang/coral/
├── src/
├── tests/                   # NEW: Chelis-native tests
│   ├── frame.ch
│   ├── groupby.ch
│   ├── join.ch
│   ├── window.ch
│   ├── reshape.ch
│   ├── io.ch                # structural tests only; pandas parity lives in parity/
│   └── nan.ch
├── parity/                  # RENAMED from tests/
│   ├── gen_goldens.py       # pandas → golden Parquet/CSV
│   └── run_parity.py        # compare Coral output vs pandas
└── reef.toml                # compiler = "=0.2.4", chelis-std dep
```

### B2. Classify every assertion

**Stays in Python (`parity/`):**
- `pd.DataFrame(...).describe()` as reference → parity.
- `pd.merge(left, right, on=k)` as reference for join outputs → parity.
- `pd.rolling(window=n).mean()` as reference for `rolling_mean` → parity.
- Parquet round-trip cross-compat with pandas-written files → parity.

**Moves to Chelis (`tests/`):**
- Frame construction: `nrows`, `ncols`, column types after
  `from_pairs([...])`.
- HAMT structural sharing: `with_column` / `drop_column` / `rename`
  produce new frames; original column references equal new column
  references (via `refeq` if exposed, else via value checks).
- Filter mask correctness on known small data:
  `filter(df, gt(price, 150)) |> nrows == 2` on `[100, 200, 300]`.
- GroupBy aggregation on known data:
  `groupby([a,a,b], sum=[1,2,3]) → {a: 3, b: 3}`.
- Join key matching on known keys: inner-join on `[1,2,3]` vs `[2,3,4]`
  yields 2 rows.
- Window rolling on known sequences:
  `rolling_mean([1,2,3,4], window=2) == [NaN, 1.5, 2.5, 3.5]` (NaN
  for window-less prefix).
- NaN handling primitives: `is_nan([1, NaN, 2]) == [false, true, false]`,
  `fill_nan([1, NaN, 2], 0) == [1, 0, 2]`, `drop_nan` removes rows.
- Sort correctness on known order: `sort([3, 1, 2]) == [1, 2, 3]`.
- Reshape round-trips: pivot then melt recovers the original.
- `describe` on constructed data:
  `describe([1.0, 2.0, 3.0]).mean == 2.0`.
- AD through a filter→aggregate pipeline on known data
  (see `Coral.Frame` example in the plan).

### B3. Template for `tests/frame.ch`

```chelis
module Coral.Tests.Frame

import Coral.Frame (from_pairs, FloatCol, get_float_col, filter, nrows, ncols, with_column, columns)
import Std.Test (assert_eq, assert_close_tensor, assert_true)

def test_construction() -> unit ! { Test } = {
  prices = to_tensor([cast(100.0, f32), cast(200.0, f32), cast(300.0, f32)])
  df = from_pairs([("price", FloatCol(prices))])
  _ = assert_eq(nrows(df), cast(3, int64), "nrows = 3")
  assert_eq(ncols(df), cast(1, int64), "ncols = 1")
}

def test_filter_by_mask() -> unit ! { Test } = {
  prices = to_tensor([cast(100.0, f32), cast(200.0, f32), cast(300.0, f32)])
  df = from_pairs([("price", FloatCol(prices))])
  mask = gt(get_float_col(df, "price"), cast(150.0, f32))
  filtered = filter(df, mask)
  assert_eq(nrows(filtered), cast(2, int64), "filter keeps 2 rows")
}

def test_with_column_preserves_existing() -> unit ! { Test } = {
  prices = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  vols = to_tensor([cast(0.1, f32), cast(0.2, f32), cast(0.3, f32)])
  df = from_pairs([("price", FloatCol(prices))])
  df2 = with_column(df, "vol", FloatCol(vols))
  _ = assert_eq(ncols(df2), cast(2, int64), "added column")
  assert_close_tensor(get_float_col(df2, "price"), prices, cast(1e-10, f32), "price preserved")
}
```

Similar for `groupby.ch`, `join.ch`, `window.ch`, `reshape.ch`,
`nan.ch`. The `io.ch` file tests CSV/JSON/Parquet round-trips that
don't need pandas — e.g., write then read the same frame and verify
equality.

### B4. Shrink parity/ to pandas-only

Same shape as Nautilus: one script per module that calls
`chelis eval --file` on a probe, calls pandas, compares. Support
`--strict`. The key difference vs Nautilus: a lot of Coral's value
is in *column-typed* operations, so parity checks need to compare
whole columns, not just scalars. Use tempfile Parquet round-trip for
that.

### B5. CI wiring

Mirror Nautilus A5: two parallel jobs, one for `chelis test tests/`
and one for `python parity/run_parity.py --strict`. Pandas install is
the only Python dep.

### B6. Deletions

After a week of green CI on the new harness:
- Delete the old Python-test module (whatever Coral's equivalent of
  `run_numeric_tests.py` is).
- Delete the golden JSON/Parquet checked into the repo if parity/
  derives them at runtime.
- Delete `scripts/gen_goldens.py` if unused.

### B7. Acceptance

- `chelis test tests/` exits 0 on a clean tree (expect ~15-25 test
  files covering each Coral.* module).
- `python parity/run_parity.py --strict` exits 0 with pandas installed.
- No `.py` files under `tests/` or `src/`.
- `tests/` never calls `pd.` or `pandas.` — grep-enforced in CI.
- AD-through-dataframe tests live in `tests/`, not `parity/` (they're
  structural, not pandas-derived).

---

## Cross-cutting hard-rule grep guards

Add to both shells' CI as a cheap check that the hard rule actually
holds:

```bash
# No Python inside tests/
if find tests -name '*.py' | grep -q .; then
  echo "Hard rule violation: .py file under tests/" >&2
  exit 1
fi

# No scipy/pandas/sympy/quantlib import from a .ch file
if grep -rE 'scipy\.|pandas\.|import pandas|import scipy' src tests 2>/dev/null | grep -q .; then
  echo "Hard rule violation: Python library reference in Chelis source" >&2
  exit 1
fi
```

These guards catch the laziest kind of backslide (copying a scipy
constant into a .ch source comment, for instance). They belong in the
earliest CI step, before the chelis test run.

## Ship order

1. Open PRs in **both** `chelis-lang/nautilus` and `chelis-lang/coral`
   simultaneously. Start by bumping `compiler = "=0.2.4"` and adding
   the `chelis-std` dep — this unblocks `import Std.Test`.
2. Land `tests/` incrementally, one module per PR. Keep the Python
   harness running alongside until `tests/` coverage is equivalent.
3. Once `tests/` covers every internal-correctness assertion from
   Python, flip CI to run both in parallel.
4. After one week of green dual-harness CI, delete the superseded
   Python code.
5. Tag a downstream release (`nautilus v0.2.0`, `coral v0.2.0`)
   recording the migration.

## References

- Phase 3t plan: `spec/design/chelis_native_testing_plan.md`
- Worked example: same file, §"Worked Example" section
- Std.Test API: `packages/chelis-std/SKILL.md` §Std.Test
- Fixture pattern: `crates/chelis-cli/tests/fixtures/pseudo_nautilus/`
  — this is the canonical template for `tests/` + `parity/` split,
  including a real scipy parity script.
- Chelis release tag: `v0.2.4`.
