# Coral — Typed Dataframes for Chelis: Implementation Design Spec

**Shell name:** Coral (`chelis-lang/coral`)
**Marine rationale:** Structured colonies built from the reef — tabular data organized into typed columns.
**Depends on:** `chelis-std` (required). `nautilus` (optional — used by `Coral.Frame.describe` for summary statistics).
**Compiler pin:** `chelis v0.1.7` (all upstream bugs resolved, `libchelis_runtime.a` ships in tarball).
**Status:** Planned (Phase 3k). This is the implementation spec.

---

## 1. What Coral Is

Coral is a typed dataframe library for Chelis that competes with pandas for ML data manipulation workflows. It is a reef package — pure Chelis source with optional Rust runtime additions (Parquet I/O).

A DataFrame is a dictionary of typed columns where numeric columns are tensors on the lazy RISC DAG. This gives three structural advantages over pandas:

1. **GPU-accelerated numeric columns.** A filter→mutate→aggregate pipeline on numeric columns compiles to fused GPU kernels because the columns ARE tensors — they go through the same compilation pipeline as any other tensor operation.

2. **AD through dataframe operations.** `grad(portfolio_risk)` where `portfolio_risk` filters a dataframe and aggregates a column — the gradient flows through `gather` (filter) and `sum` (aggregate). No existing dataframe library supports this. Not pandas, not Polars, not RAPIDS.

3. **Typed columns.** Column type is statically known once retrieved. `get_float_col(df, "price")` returns `tensor[n, f32]`. Wrong column type is a compile error.

Coral does NOT compete with Polars/DuckDB query optimization (predicate pushdown, projection pushdown, join reordering) or Spark distributed processing. It is single-machine, no query planner. Numeric optimization comes from the tensor compiler's existing fusion passes, not a dataframe-specific optimizer.

---

## 2. Core Data Model

### DataFrame Type

```chelis
-- A DataFrame is a collection of named, typed columns
-- Internally: Dict[String, Column]

type Column =
  | IntCol(tensor[n, i64])
  | FloatCol(tensor[n, f32])
  | StringCol(List[String])
  | BoolCol(tensor[n, bool])
```

**Design decisions:**

- **Numeric columns are tensors.** This is the key architectural choice. `FloatCol` and `IntCol` are tensors on the lazy DAG — they get fusion, GPU dispatch, and AD for free. There is no "dataframe-specific" optimization; it's the same tensor compiler.

- **String columns are host-side lists.** Strings go through the host lane (plain C, no GPU). String operations are eager. This is intentional — string manipulation is not the performance-critical path for ML data workflows, and host-side `List[String]` composes with the existing 3d collection primitives.

- **BoolCol is a tensor.** Boolean masks from filtering are `tensor[n, bool]`, which means filter operations can run on GPU (boolean mask → `gather` on GPU).

- **Column length invariant.** All columns in a DataFrame have the same length `n`. This is enforced at construction time, not statically in the type system (column names are runtime strings, not static types).

- **Persistent column dictionary.** The internal `Dict[String, Column[n]]` backing the Frame should use a persistent data structure (hash array mapped trie / HAMT) so that `with_column`, `drop_column`, and `rename` produce new frames that share column references with the original via structural sharing. This is a performance requirement for the AD-through-dataframes story: `grad(fn_that_does_10_frame_ops)` produces intermediate frames on the backward pass, and if each intermediate copies the entire column dictionary, AD memory cost is O(num_columns * num_operations). With structural sharing, intermediate frames share unchanged columns and AD memory cost is O(num_operations). At real portfolio sizes (50-100 columns, thousands of rows), this is the difference between practical and impractical gradient computation through frame pipelines. The columns themselves (tensors) are immutable and reference-counted (`Arc`) — the persistent dict shares the column references, not the column data. Implementation: either a pure-Chelis HAMT built from existing primitives, or a Rust-side HAMT exposed through the runtime C ABI. Decide at implementation start based on which path composes with `grad`.

### Missing Data

Float columns use IEEE 754 NaN semantics. NaN handling is built into `Coral.Frame`, not a separate module — missing data is a property of float columns, not an opt-in concern.

- `NaN != NaN` (IEEE 754 standard)
- Arithmetic operations propagate NaN (tensor ops already do this correctly on CPU and GPU)
- Explicit operations: `is_nan`, `fill_nan`, `drop_nan`, `any_nan`, `count_nan`
- `fill_nan(col, value)` replaces NaN with a specified value — implemented via `where(is_nan(col), fill_value, col)` using the existing `where` RISC primitive

Integer missing values use a boolean mask column. A DataFrame can carry a `_mask_colname` column alongside an `IntCol` to track which elements are present. This is the pandas approach for nullable integers.

---

## 3. Module Architecture

### Coral.Frame — Core DataFrame Operations

The foundational module. Everything else builds on this.

**Construction:**

```chelis
-- From typed columns
def from_columns(columns: Dict[String, Column]) -> Frame

-- From a list of (name, column) pairs
def from_pairs(pairs: List[(String, Column)]) -> Frame

-- Empty frame with schema only
def empty(schema: Dict[String, ColumnType]) -> Frame
```

**Column access:**

```chelis
-- Type-safe column extraction
def get_float_col(df: Frame, name: String) -> tensor[n, f32]
def get_int_col(df: Frame, name: String) -> tensor[n, i64]
def get_string_col(df: Frame, name: String) -> List[String]
def get_bool_col(df: Frame, name: String) -> tensor[n, bool]

-- Column names and types
def columns(df: Frame) -> List[String]
def column_type(df: Frame, name: String) -> ColumnType
def nrows(df: Frame) -> i64
def ncols(df: Frame) -> i64
```

**Row selection and filtering:**

```chelis
-- Boolean mask filtering: df[mask]
-- Under the hood: gather on every column using the indices where mask is true
def filter(df: Frame, mask: tensor[n, bool]) -> Frame

-- Convenience: filter with a predicate on a float column
def filter_float(df: Frame, col_name: String, pred: f32 -> bool) -> Frame

-- Head/tail/slice
def head(df: Frame, k: i64) -> Frame
def tail(df: Frame, k: i64) -> Frame
def slice(df: Frame, start: i64, end: i64) -> Frame
```

The key operation is `filter`. Under the hood:

```
filter(df, mask) =
  indices = where(mask)           -- indices where mask is true
  for each column c in df:
    if c is tensor: gather(c, indices, 0)   -- differentiable!
    if c is List:   [c[i] for i in indices] -- host-side list indexing
```

`gather` is differentiable. This is why AD flows through filter operations — the gradient of a downstream computation can flow back through the `gather` to the original column values.

**Sorting:**

```chelis
-- Sort by a single column
def sort_by(df: Frame, col_name: String, ascending: bool) -> Frame

-- Under the hood: argsort on the key column, then gather all columns by those indices
```

Implementation: `argsort(key_col, 0)` produces a permutation, then `gather(col, perm, 0)` for every column. This is a standard dataframe sort pattern.

**Mutation (add/replace columns):**

```chelis
-- Add or replace a column
def with_column(df: Frame, name: String, col: Column) -> Frame

-- Apply a function to a float column, producing a new column
def mutate_float(df: Frame, src: String, dst: String, f: f32 -> f32) -> Frame

-- Rename a column
def rename(df: Frame, old_name: String, new_name: String) -> Frame

-- Drop a column
def drop_column(df: Frame, name: String) -> Frame
```

**NaN handling (built into Frame, not a separate module):**

```chelis
-- Per-column NaN operations on float columns
def is_nan(col: tensor[n, f32]) -> tensor[n, bool]
def fill_nan(col: tensor[n, f32], value: f32) -> tensor[n, f32]
def drop_nan(df: Frame, col_name: String) -> Frame     -- drops rows where col is NaN
def any_nan(col: tensor[n, f32]) -> bool
def count_nan(col: tensor[n, f32]) -> i64
```

**Concatenation:**

```chelis
-- Vertical concatenation (stack rows)
def concat(frames: List[Frame]) -> Frame

-- Requires all frames to have the same schema (column names and types)
-- Tensors: concat along axis 0
-- Lists: list append
```

**Summary statistics (cross-dependency on Nautilus.Stats):**

```chelis
-- Per-column summary: count, mean, std, min, 25%, 50%, 75%, max
-- Returns a new Frame with statistic names as rows
def describe(df: Frame) -> Frame

-- Uses Nautilus.Stats.mean, Nautilus.Stats.std, Nautilus.Stats.quantile
-- This is an optional dependency: if Nautilus is not imported, describe is unavailable
```

**Value counts:**

```chelis
-- Count occurrences of each value in a column
-- Returns a Frame with (value, count) columns sorted by count descending
def value_counts(df: Frame, col_name: String) -> Frame
```

### Coral.GroupBy — Aggregation

```chelis
-- Group by one column, then aggregate
def group_by(df: Frame, key_col: String) -> GroupedFrame

-- Aggregations on GroupedFrame
def agg_sum(gf: GroupedFrame, col: String) -> Frame
def agg_mean(gf: GroupedFrame, col: String) -> Frame
def agg_count(gf: GroupedFrame) -> Frame
def agg_min(gf: GroupedFrame, col: String) -> Frame
def agg_max(gf: GroupedFrame, col: String) -> Frame

-- Multiple aggregations at once
def agg(gf: GroupedFrame, specs: List[(String, AggFn)]) -> Frame
```

**Implementation strategy:**

GroupBy uses `argsort` on the key column to cluster equal keys together, then run-length detection to identify group boundaries. Aggregation uses segmented operations:

```
group_by(df, "category") =
  perm = argsort(get_col(df, "category"))
  sorted_df = gather_all_columns(df, perm)
  boundaries = detect_runs(sorted_df, "category")  -- indices where the key changes
  return GroupedFrame(sorted_df, boundaries)

agg_sum(gf, "price") =
  for each group [start, end) in boundaries:
    result[group] = sum(price[start:end])
```

The segmented sum can be implemented via `scatter(..., "add")` for GPU-friendly execution, or via a fold over group boundaries for the host path.

### Coral.Join — Table Joins

```chelis
-- Inner join on a key column
def inner_join(left: Frame, right: Frame, on: String) -> Frame

-- Left join
def left_join(left: Frame, right: Frame, on: String) -> Frame

-- Outer join
def outer_join(left: Frame, right: Frame, on: String) -> Frame
```

**Implementation:** Sort-merge join. Both sides sorted by key via `argsort` + `gather`, then a merge pass that produces index arrays for left and right, then `gather` both frames by their respective index arrays. The merge pass is host-side control flow (two pointers walking sorted arrays).

Sort-merge is simpler to implement than hash join in pure Chelis (no hash table data structure needed). Performance is O(n log n + m log m) for the sorts plus O(n + m) for the merge. Adequate for the sizes Chelis targets.

### Coral.Reshape — Pivot, Melt, Concat

```chelis
-- Pivot: long to wide
def pivot(df: Frame, index_col: String, columns_col: String, values_col: String) -> Frame

-- Melt: wide to long
def melt(df: Frame, id_cols: List[String], value_cols: List[String]) -> Frame
```

These are structural transformations that rearrange data without computation. Implementation is index manipulation + column construction.

### Coral.Window — Rolling/Window Operations

```chelis
-- Rolling window operations on float columns
def rolling_mean(col: tensor[n, f32], window: i64) -> tensor[n, f32]
def rolling_sum(col: tensor[n, f32], window: i64) -> tensor[n, f32]
def rolling_std(col: tensor[n, f32], window: i64) -> tensor[n, f32]
def rolling_min(col: tensor[n, f32], window: i64) -> tensor[n, f32]
def rolling_max(col: tensor[n, f32], window: i64) -> tensor[n, f32]

-- Exponentially weighted moving average
def ewm(col: tensor[n, f32], alpha: f32) -> tensor[n, f32]
```

**Implementation:**

`rolling_sum` can be computed efficiently via `cumsum`:

```
rolling_sum(col, w) =
  cs = cumsum(col, 0)
  cs[w:] - cs[:-w]     -- via gather + sub
```

First `w-1` elements are NaN (insufficient history). This is the standard pandas convention.

`rolling_mean` = `rolling_sum / window`. `rolling_std` requires `rolling_sum(col²)` and `rolling_sum(col)` (via the variance formula).

`ewm` is a scan: `y[0] = x[0]`, `y[i] = alpha * x[i] + (1 - alpha) * y[i-1]`. Implement via `fold` over `to_list`. This is inherently sequential (each element depends on the previous), so no GPU parallelism. Acceptable — EWM is typically applied to time series of moderate length.

### Coral.IO — Data Loading and Export

```chelis
-- CSV
def read_csv(path: String) -> Frame ! { IO }
def write_csv(df: Frame, path: String) -> unit ! { IO }

-- JSON
def read_json(path: String) -> Frame ! { IO }
def write_json(df: Frame, path: String) -> unit ! { IO }

-- Parquet (requires runtime addition)
def read_parquet(path: String) -> Frame ! { IO }
def write_parquet(df: Frame, path: String) -> unit ! { IO }
```

**CSV/JSON:** Wrap `Std.Io.Csv` and `Std.Io.Json`. Auto-detect column types: attempt numeric parse on each column, fall back to string. The wrapping layer builds a `Frame` from the parsed columns.

**Parquet:** Requires a Rust runtime addition. The `parquet2` crate reads/writes Parquet files. The integration pattern follows the same approach as `mmap` in 3g: the Rust runtime provides `chelis_read_parquet(path) -> Frame*` and `chelis_write_parquet(frame*, path)` functions, and Coral provides the Chelis-level wrappers with proper types and effect annotations.

This is the only Coral module that requires an upstream runtime change. All other modules are pure Chelis.

---

## 4. AD Through Dataframes

This is Coral's unique technical contribution. The complete chain:

```chelis
import Coral.Frame (from_columns, filter, get_float_col)
import Coral.GroupBy (group_by, agg_mean)

def portfolio_risk(prices: Frame, vol_threshold: f32) -> f32 = {
  -- 1. Filter: boolean mask → gather (differentiable)
  vol_col = get_float_col(prices, "volatility")
  mask = gt(vol_col, vol_threshold)        -- tensor[n, bool]
  high_vol = filter(prices, mask)          -- gather under the hood

  -- 2. Aggregate: mean over tensor column (differentiable)
  returns = get_float_col(high_vol, "return")
  mean(returns)
}

-- Sensitivity of risk measure to the volatility threshold
sensitivity = grad(portfolio_risk, wrt=vol_threshold)(prices, 0.3)
```

**Why this works:** `filter` is implemented via `gather`, which has an adjoint rule in the Chelis AD system (gradient flows back to the gathered positions). `mean` is `sum / n`, both differentiable. The entire filter→aggregate pipeline is a composition of differentiable tensor operations.

**What's NOT differentiable:** String column operations, join key matching (discrete), sort order (permutation is discrete). AD flows through numeric column operations only. This is the correct behavior — the gradient of "which rows pass the filter" with respect to the threshold is well-defined, but the gradient of "which row is the 3rd sorted element" is not.

---

## 5. Competitive Position vs pandas/Polars

### Where Coral Wins

| Advantage | Details |
|---|---|
| **GPU-accelerated numeric columns** | Tensor DAG fusion means a filter→mutate→aggregate pipeline on float columns can compile to GPU kernels. pandas is CPU-only (even with Arrow backend). |
| **AD through operations** | `grad(fn_that_uses_dataframe)` just works. No existing library does this. |
| **Typed columns** | `get_float_col` returns `tensor[n, f32]`. Wrong type = compile error. pandas returns untyped Series. |
| **Effect-tracked provenance** | `read_csv` carries `IO` effect. A derived computation without `IO` provably doesn't depend on external data. |
| **Fused numeric operations** | Chained numeric column operations fuse into single loops via the tensor DAG. pandas/numpy allocate intermediates. |

### Where pandas/Polars Win

| Advantage | Details |
|---|---|
| **Query optimization** | Polars has lazy query plans with predicate pushdown, projection pushdown, join reordering. Coral has none — it's eager for control flow, lazy only via tensor DAG. |
| **Ecosystem** | pandas has thousands of integrations. Coral has zero. |
| **Missing data maturity** | pandas has decades of NaN handling baked into every operation. Coral's NaN story is basic (IEEE 754 float NaN + boolean mask for integers). |
| **String operations** | pandas has rich string column methods (str.contains, str.split, regex). Coral has host-side List[String] with basic operations. |
| **Scale** | Polars handles larger-than-memory datasets via streaming. Spark handles distributed datasets. Coral is single-machine, in-memory only. |

---

## 6. Implementation Strategy

### Phase Order

All of Phase A is pure Chelis. Phase B requires one upstream runtime addition (Parquet).

**Phase A — Pure Chelis, no upstream dependencies:**

```
A1. Coral.Frame core (construction, column access, nrows/ncols/columns)
A2. Coral.Frame filtering (filter, filter_float, head/tail/slice)
A3. Coral.Frame mutation (with_column, mutate_float, rename, drop_column)
A4. Coral.Frame NaN handling (is_nan, fill_nan, drop_nan, any_nan, count_nan)
A5. Coral.Frame sorting (sort_by via argsort + gather)
A6. Coral.Frame concat (vertical concatenation)
A7. Coral.GroupBy (group_by + agg_sum/mean/count/min/max, value_counts)
A8. Coral.Window (rolling_mean/sum/std/min/max via cumsum, ewm via fold)
A9. Coral.IO CSV/JSON (wrap Std.Io.Csv and Std.Io.Json)
A10. Coral.Join (inner_join, left_join via sort-merge)
A11. Coral.Reshape (pivot, melt)
A12. Coral.Frame describe (optional Nautilus.Stats cross-dependency)
```

**Phase B — Requires upstream runtime addition:**

```
B1. Coral.IO Parquet (read_parquet, write_parquet via parquet2 Rust crate)
```

**Phase A ordering rationale:** A1-A6 form the core that everything else depends on. A7 (GroupBy) and A8 (Window) are the highest-value analytics operations after basic frame manipulation. A9 (CSV/JSON I/O) is needed for any real workflow. A10-A11 are important but less critical. A12 is a convenience that depends on Nautilus.

### What Goes in the Main chelis-lang/chelis Repo

**Nothing for Phase A.** All of Phase A is pure Chelis in the `chelis-lang/coral` repo.

**For Phase B (Parquet):** Add `parquet2` as a dependency of `chelis-runtime`, implement `chelis_read_parquet` and `chelis_write_parquet` C ABI functions, ship in a compiler release. Same pattern as `mmap_file` in 3g.

### Testing Strategy

Follow the Nautilus pattern exactly — it's proven and documented:

**Golden-file scipy/pandas parity:**

- `scripts/gen_goldens.py`: Python script calling pandas to generate JSON golden fixtures for each operation on fixed input data. Covers: construction round-trip, filter output, sort output, groupby aggregation results, rolling window values, CSV round-trip, NaN handling.
- Goldens checked in under `tests/goldens/{frame,groupby,window,io,join}/*.json`.
- Chelis-side test harnesses load goldens and assert within tolerance.

**AD tests:**

- `grad` through filter→aggregate pipeline, verified against finite differences.
- `grad` through a rolling_mean pipeline.
- Negative: `grad` through a string column operation correctly errors (not differentiable).

**Test infrastructure:**

Three binary builds mirroring Nautilus:
1. **Scalar binary** — tests pure scalar operations (column type detection, NaN checks)
2. **Tensor binary** — tests all tensor-column operations (filter via gather, sort via argsort, GroupBy via scatter, Window via cumsum). Links against `libchelis_runtime.a`.
3. **IO binary** — tests CSV/JSON read/write with actual files. Links against `libchelis_runtime.a`.

**Acceptance oracle:** the current in-repo prerequisite gate is
`cargo test -p chelis-cli --test coral_prerequisites -- --ignored --nocapture`. A
future Coral-owned downstream oracle should replace this when the full shell surface
lands.

---

## 7. Implementation Details and Chelis Patterns

### Representing a DataFrame in Chelis

Chelis doesn't have algebraic data types (ADTs) with pattern matching on variants in the traditional sense. The `Column` type needs to be representable. Options:

**Option A: Tagged union via Dict.**

```chelis
-- A Column is a Dict with a "type" key and a "data" key
-- type Column = Dict[String, Any]  -- but Chelis doesn't have Any

-- This doesn't work cleanly. Chelis's type system doesn't have a sum type
-- that holds tensors of different element types.
```

**Option B: Separate parallel structures.**

```chelis
-- A Frame stores each column type in a separate dict
type Frame = {
  float_cols: Dict[String, tensor[n, f32]],
  int_cols: Dict[String, tensor[n, i64]],
  string_cols: Dict[String, List[String]],
  bool_cols: Dict[String, tensor[n, bool]],
  col_order: List[String],     -- preserves insertion order
  nrows: i64
}
```

This is the pragmatic approach. Column access dispatches on which dict the name is found in. Type safety comes from the accessor functions: `get_float_col` looks in `float_cols` and returns `tensor[n, f32]` — if the column is actually in `int_cols`, it's a runtime error (key not found in float_cols dict).

**Option C: Chelis-level ADT if supported.**

Check whether v0.1.7 supports `type Column = IntCol(tensor[n, i64]) | FloatCol(tensor[n, f32]) | ...` with pattern matching. If yes, use it. If not, use Option B.

**Decision: verify at implementation start.** Write a 5-line probe program with a sum type and pattern match. If it compiles on v0.1.7, use Option C. Otherwise, Option B. Document the choice.

### Filter Implementation

```chelis
def filter(df: Frame, mask: tensor[n, bool]) -> Frame = {
  -- Convert boolean mask to indices
  indices = where_indices(mask)    -- tensor[k, i64] where k = count(mask == true)

  -- Gather each tensor column
  new_float_cols = map_dict(df.float_cols, fn(name, col) ->
    gather(col, indices, 0))

  new_int_cols = map_dict(df.int_cols, fn(name, col) ->
    gather(col, indices, 0))

  new_bool_cols = map_dict(df.bool_cols, fn(name, col) ->
    gather(col, indices, 0))

  -- Filter string columns host-side
  new_string_cols = map_dict(df.string_cols, fn(name, col) ->
    list_gather(col, indices))     -- host-side list indexing

  Frame(new_float_cols, new_int_cols, new_string_cols, new_bool_cols,
        df.col_order, length(indices))
}
```

**AD note:** `gather` has an adjoint rule. The gradient flows back through `gather` to the original column positions. String columns don't participate in AD (they're host-side lists, not tensors).

### GroupBy Implementation

```chelis
def group_by(df: Frame, key_col: String) -> GroupedFrame = {
  key = get_col(df, key_col)               -- could be any type
  perm = argsort(key, 0)                    -- sort indices
  sorted_df = reindex_all(df, perm)         -- gather all columns by perm
  boundaries = detect_group_boundaries(sorted_df, key_col)  -- List[i64]
  GroupedFrame(sorted_df, boundaries, key_col)
}

def agg_sum(gf: GroupedFrame, col: String) -> Frame = {
  data = get_float_col(gf.sorted_df, col)
  n_groups = length(gf.boundaries) - 1
  -- Segmented sum: for each group [boundaries[i], boundaries[i+1]):
  --   result[i] = sum(data[boundaries[i]:boundaries[i+1]])
  -- Implementable via scatter("add") with a group-index tensor,
  -- or via a fold over boundaries
  results = segmented_sum(data, gf.boundaries)
  keys = extract_group_keys(gf)
  from_pairs([("key", keys), (col, FloatCol(results))])
}
```

### Rolling Window Implementation

```chelis
def rolling_sum(col: tensor[n, f32], window: i64) -> tensor[n, f32] = {
  cs = cumsum(col, 0)
  -- result[i] = cs[i] - cs[i - window]  for i >= window
  -- result[i] = NaN                       for i < window
  shifted = gather(cs, sub(arange(0, n), window), 0)  -- cs[i - window]
  result = sub(cs, shifted)
  -- Mask first (window-1) elements as NaN
  mask = ge(arange(0, n), window)
  where(mask, result, nan_tensor(n))
}

def rolling_mean(col: tensor[n, f32], window: i64) -> tensor[n, f32] =
  div(rolling_sum(col, window), cast(window, f32))
```

This is differentiable — `cumsum`, `gather`, `sub`, `where` all have adjoint rules.

---

## 8. Repo Structure

```
chelis-lang/coral/
├── reef.toml                         -- package manifest, pin chelis =0.1.7
├── SKILL.md                          -- Deep-first agent docs (write at dev start)
├── README.md                         -- Human-readable overview
├── AGENTS.md                         -- Agent rules (same pattern as Nautilus)
├── src/
│   ├── core.ch                       -- version(), re-exports
│   ├── frame.ch                      -- Coral.Frame (construction, access, filter,
│   │                                    sort, mutation, NaN, concat, describe)
│   ├── groupby.ch                    -- Coral.GroupBy (group_by, aggregations, value_counts)
│   ├── join.ch                       -- Coral.Join (inner_join, left_join, outer_join)
│   ├── reshape.ch                    -- Coral.Reshape (pivot, melt)
│   ├── window.ch                     -- Coral.Window (rolling_*, ewm)
│   ├── io.ch                         -- Coral.IO (read_csv, write_csv, read_json, write_json,
│   │                                    read_parquet, write_parquet)
│   └── apismoke.ch                   -- type-level import gate
├── docs/
│   ├── book.toml                     -- mdBook config
│   └── src/
│       ├── SUMMARY.md
│       ├── introduction.md
│       ├── getting-started/
│       │   ├── installation.md
│       │   └── first-dataframe.md
│       ├── frame/
│       │   ├── construction.md
│       │   ├── filtering.md
│       │   ├── sorting.md
│       │   ├── mutation.md
│       │   ├── nan-handling.md
│       │   └── concatenation.md
│       ├── groupby.md
│       ├── joins.md
│       ├── window.md
│       ├── io.md
│       ├── ad-through-dataframes.md  -- the unique feature, with full examples
│       ├── performance/
│       │   └── gpu-columns.md
│       └── appendix/
│           ├── api.md
│           ├── limitations.md
│           └── pandas-comparison.md
├── tests/
│   ├── run_coral_tests.py            -- pandas-parity assertion harness
│   ├── run_static_checks.py          -- export consistency gate
│   ├── run_skill_checks.py           -- SKILL.md code block validation
│   └── goldens/
│       ├── frame/*.json              -- filter, sort, concat fixtures
│       ├── groupby/*.json            -- aggregation fixtures
│       ├── window/*.json             -- rolling window fixtures
│       ├── io/*.json                 -- CSV/JSON round-trip fixtures
│       └── join/*.json               -- join result fixtures
├── scripts/
│   ├── gen_goldens.py                -- golden generator (calls pandas)
│   ├── validate_book_examples.py     -- mdBook code block validation
│   └── bench_vs_pandas.py            -- performance benchmark (future)
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 9. Documentation (Built From Day One)

### SKILL.md (Deep-First, Agent-Facing)

Written at development start, not after shipping. The SKILL.md serves double duty: it teaches agents how to use Coral, AND it's the design document for the coding agent building Coral.

Six sections per the Nautilus template:

1. **Identity** (3 lines)
2. **Import Patterns** (Surf + Deep for each Coral module)
3. **Core Patterns** (10-15 complete programs):
   - Construct a DataFrame from columns
   - Load CSV into a DataFrame
   - Filter rows by a numeric predicate
   - Sort by a column
   - GroupBy + aggregate (sum, mean, count)
   - Rolling mean on a time series column
   - EWM on a price column
   - Inner join two DataFrames
   - AD through filter→aggregate (`grad(portfolio_risk)`)
   - NaN handling (fill_nan, drop_nan)
   - Describe (summary statistics)
   - Pivot/melt
   - Value counts
4. **Module Quick Reference** (one subsection per module)
5. **Gotchas** (NaN semantics, AD restrictions, string column limitations, no query optimizer)
6. **API Surface** (generated from `grep '^def ' src/*.ch`)

Code block convention:
- `chelis` / `deep` for complete, compilable programs (CI-validated)
- `chelis-fragment` / `deep-fragment` for partial snippets (not validated)

### mdBook (Human-Facing)

Chapter structure shown in the repo layout above. Content production follows the Nautilus guide: source material comes from the test harness, golden fixtures, and the SKILL.md patterns. The "AD through dataframes" chapter is the unique selling point — write it as a tutorial, not a reference, with a complete portfolio risk example.

CI validates all code blocks and builds the mdBook. No GitHub Pages deployment until the repo goes public.

---

## 10. Test Plan

### Assertion Targets

| Module | Expected assertions | What they verify |
|---|---|---|
| Frame construction | ~10 | Column types detected correctly, nrows/ncols correct |
| Frame filtering | ~15 | Filter output matches pandas boolean indexing |
| Frame sorting | ~10 | Sort output matches pandas sort_values |
| Frame mutation | ~10 | with_column, rename, drop produce correct results |
| Frame NaN | ~15 | is_nan, fill_nan, drop_nan match pandas behavior |
| Frame concat | ~5 | Vertical concat matches pandas.concat |
| GroupBy | ~20 | agg_sum/mean/count match pandas.groupby |
| Window | ~15 | rolling_mean/sum match pandas.rolling |
| Join | ~15 | inner/left join match pandas.merge |
| IO CSV | ~10 | Round-trip (write → read → compare) preserves data |
| IO JSON | ~5 | Round-trip preserves data |
| AD | ~10 | grad through filter+aggregate matches finite differences |
| **Total** | **~140** | |

### Negative Tests

- Wrong column name → error
- Type mismatch (get_float_col on an int column) → error
- Join key type mismatch → error
- Filter mask wrong length → error
- GroupBy on non-existent column → error
- Concat with mismatched schemas → error

### Acceptance Oracle

```bash
# Local development
python tests/run_coral_tests.py     # pandas-parity assertions
python tests/run_static_checks.py   # export consistency
python tests/run_skill_checks.py    # SKILL.md validation
chelis reef build                    # package builds
chelis check src/*.ch               # all modules score 1.0

# Current in-repo prerequisite gate
cargo test -p chelis-cli --test coral_prerequisites -- --ignored --nocapture

# Future downstream Coral-owned gate, once the full shell surface lands
# (command TBD in the Coral repo/CI)
```

---

## 11. Upstream Dependencies

| Dependency | What it provides | Status |
|---|---|---|
| `chelis v0.1.7` | Compiler, `libchelis_runtime.a`, all tensor primitives (gather, scatter, argsort, where, cumsum, concat, sort, einsum) | Shipped |
| `chelis-std` | Std.Io.Csv, Std.Io.Json, Std.Time, Dict, List operations | Shipped |
| `nautilus` (optional) | Nautilus.Stats for `describe` | Shipped (v0.1.0) |
| `parquet2` Rust crate | Parquet read/write in runtime | Not yet integrated — Phase B |

No upstream blockers for Phase A. Everything Coral needs for the core DataFrame, GroupBy, Window, Join, Reshape, and CSV/JSON I/O is already in the shipped compiler and standard library.

---

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Chelis may lack sum type / pattern matching for Column representation** | Forces the parallel-dicts approach (Option B), which is less elegant | Probe at implementation start. Either approach works. |
| **Dict operations may be slow for column access** | Every column access is a dict lookup by string key | Acceptable for the frame sizes ML targets. Profile later if needed. |
| **`gather` on boolean tensors may not work in v0.1.7** | Filter implementation depends on gathering bool columns | Test immediately. If broken, cast bool→i64→gather→i64→bool. |
| **Parquet integration requires upstream runtime change** | Phase B is blocked until chelis ships parquet2 | Phase A is complete without Parquet. CSV/JSON cover the immediate need. |
| **GroupBy segmented operations may need a primitive not in v0.1.7** | Segmented sum/mean may not compose cleanly from existing ops | Fall back to fold over group boundaries (slower but correct). |
| **String column operations are limited** | No regex, no str.contains, no str.split beyond basic List ops | Document as a known limitation. Not the performance-critical path. |
| **EWM is inherently sequential** | No GPU parallelism for exponentially weighted operations | Acceptable for time series of moderate length. Document. |

---

## 13. Open Questions (Resolve at Implementation Start)

1. **Does v0.1.7 support sum types with pattern matching?** Write a 5-line probe. Determines Frame representation approach.

2. **Does `gather` work on `tensor[n, bool]`?** If not, bool columns need a cast workaround for filtering.

3. **Can Chelis `Dict` hold values of different types?** `Dict[String, Column]` where Column is a sum type — does this unify? If not, the parallel-dicts approach is the only option.

4. **What does `where_indices(mask)` look like in Chelis?** The filter implementation needs "indices where mask is true." Is there a builtin, or do we build it from `cumsum(cast(mask, i64))` + `gather`?

5. **How does `Std.Io.Csv` represent parsed data?** Does it return `List[List[String]]` (rows of cells), or something typed? The CSV→Frame bridge layer depends on the answer.

6. **Can `Coral.Frame.describe` import from Nautilus optionally?** If Chelis doesn't support conditional/optional imports, `describe` either always depends on Nautilus (making it a required dependency) or lives in a separate `Coral.Stats` module that consumers import explicitly.

7. **Should the persistent column dict (HAMT) be pure Chelis or Rust runtime?** A pure-Chelis HAMT composes with `grad` automatically (it's tensor ops all the way down). A Rust-side HAMT is faster but opaque to AD — frame operations that use the Rust dict internals wouldn't be differentiable. The pure-Chelis path is strongly preferred for AD compatibility. Verify that the performance is acceptable for frames with 50-100 columns (the HAMT only stores column references, not column data, so the tree is small).

---

## 14. Deferred Items (Recorded, Not Gating)

- **Parquet I/O** — Phase B, requires upstream `parquet2` integration
- **Query optimization** — no predicate pushdown, no projection pushdown. If Coral gets large enough to need this, it's a compiler-level optimization on the tensor DAG, not a Coral-level feature.
- **Multi-column GroupBy** — group by a tuple of columns. Deferred to avoid combinatorial complexity in the first implementation.
- **Outer join** — inner and left join ship first. Outer join adds complexity around NaN fill for non-matching rows.
- **Streaming / out-of-core** — Coral is in-memory only. No chunked reading, no spilling to disk.
- **String column methods** — str.contains, str.split, regex. Host-side string operations are not the priority.
- **Excel I/O** — via a Rust crate (calamine) in the runtime. Low priority.
- **Time series index** — datetime-indexed DataFrames with resample/shift/lag. Would depend on Std.Time. Useful for finance but not in the first implementation.
- **GPU benchmarks** — benchmark Coral on HIP to validate the "GPU-accelerated columns" claim. Requires a real workload + GPU hardware.

---

## 15. Execution Summary

```
Phase A (pure Chelis, no upstream blockers):
├── A1-A6: Core Frame (construction → concat)
├── A7: GroupBy
├── A8: Window
├── A9: IO (CSV/JSON)
├── A10: Join
├── A11: Reshape
└── A12: Describe (optional Nautilus dep)

Phase B (upstream runtime addition):
└── B1: Parquet IO

Parallel with all phases:
├── SKILL.md (written at dev start, updated as modules ship)
├── mdBook (structure at dev start, chapters as modules ship)
└── CI (validation for SKILL.md + mdBook + golden assertions)
```

Target: ~140 pandas-parity assertions. Ship Coral v0.1.0 when Phase A is green.
