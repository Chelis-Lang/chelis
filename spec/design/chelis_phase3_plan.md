# Phase 3: Language Completeness - Expanded Implementation Plan

## Context

Phase 2 shipped a feature-rich tensor computation language: effects, linearity, macros,
`vmap`, tuples, Tide APIs, LSP, and Cove. Phase `3a` shipped the package system.
Phases `3b` and `3b-ii` shipped Python interop, safetensors, and direct execution.
Phase `3e` shipped the pipe-first public Surf idiom.

But Chelis still depends on Python for every non-tensor part of a real AI program. It
can define a transformer forward pass and compute gradients, but it still cannot:

- represent first-class scalar integers/floats/bools outside tensors
- process strings as ordinary language values
- store variable-length sequences and maps
- iterate over non-tensor data
- read text/config/data files directly
- tokenize text and batch it into model inputs

That means Chelis is still a sophisticated tensor-compute DSL rather than a
self-sufficient AI programming language.

**Phase 3 deliverable:** a researcher can write, in pure Chelis, a program that reads
text, tokenizes it, batches and pads it, runs a model, computes loss and gradients, and
prints results, without dropping to Python for preprocessing or orchestration.

**Phase 3 does NOT deliver:** sparse tensors, complex numbers, research type
extensions, or Lean formalization. Those move to Phase `5e`, `5f`, `5g`, and `5h`.
Phase 3 is the pragmatic language-completeness phase.

---

## Dependency Graph

```text
Shipped foundations

3e: Pipe-First Style Pass
3a: Package System
3b: Python FFI
3b-ii: Direct Python Execution + NumPy
3c: Scalar & String Foundation
3d: Collections & Iteration

Remaining work

3h: Core Numeric Primitives -> 3g: Data Loading & Tokenization -> 3i: Std.Time & Std.Decimal -> 3f: SKILL.md v2 Redo
```

**Recommended execution order:**

1. `3h`: Core numeric primitives
2. `3g`: Data loading and tokenization
3. `3i`: `Std.Time` and `Std.Decimal`
4. `3f`: SKILL.md v2 redo

Shipped Phase 3 foundations stay in place and continue to constrain the remaining work:

- `3e` defines the public Surf idiom new examples must follow
- `3a` defines the package/distribution story new libraries should use
- `3b` / `3b-ii` define the Python interop boundary the fuller language must still fit

`3h` follows the shipped `3c`/`3d` foundations because real AI model code still needs
practical tensor-language primitives such as `einsum`, `concat`, `gather`, and
`clamp`. `3g` depends on that fuller host-and-tensor surface because tokenization,
batching, and model ingress should not force awkward library workarounds. `3i` comes
after `3g` because time/exact-decimal support rounds out the standard library rather
than blocking the pure-AI workflow milestone. `3f` goes last because the teaching
surface should describe the real full Phase 3 language, not a partially complete
midpoint.

---

## Shipped Foundations

### 3e: Pipe-First Style Pass

**Status:** shipped.

This remains the public style foundation for all remaining Phase 3 work:

- pipe-first decompiler output
- short-form block bindings
- width-aware multiline pipe layout
- examples and docs that read like human-written Surf rather than typed Deep debug text

All new examples introduced in `3h`, `3g`, `3i`, and `3f` should continue to follow
this style.

### 3a: Package System (Shells + Reef)

**Status:** shipped.

The package system remains in scope as infrastructure, not as remaining work. New
Phase 3 library surfaces such as tokenizer helpers or text/data modules should be
documented as package-friendly APIs that fit the existing Reef / `.chb` model.

### 3b: Python FFI

**Status:** shipped.

Python interop remains a supporting boundary, not the solution to Chelis's remaining
language gaps. Phase 3 is successful only when preprocessing and tokenization no
longer require Python for ordinary use.

### 3b-ii: Direct Python Execution + NumPy Guarantee

**Status:** shipped.

Direct execution remains the bridge for embedding Chelis in Python workflows, but the
remaining Phase 3 work is about making that embedding optional for end-to-end AI
program authoring.

---

## 3c: Scalar and String Foundation

**Status:** shipped.

**Goal:** make Chelis a real programming language for AI workflows by adding first-class
scalar values and strings outside the tensor-only world.

### Why This Matters

Without first-class scalars and strings, Chelis cannot naturally express:

- loop counters and vocabulary indices
- loss-threshold checks and training decisions
- file paths and config keys
- labels, tokens, and log messages
- tokenizer state and text-derived metadata

### Required Surface

**Scalar types as ordinary values:**

- unrestricted `Int`, `Float`, and `Bool`
- arithmetic, comparison, `%` / `mod`, and basic integer bitwise helpers such as
  `bitand`, `bitor`, `bitxor`, `shl`, and `shr`
- scalar values are distinct from rank-0 tensors
- explicit scalar/tensor conversions where needed

**String values as ordinary values:**

- immutable UTF-8 `String`
- length, concat, slice, contains, starts/ends-with, trim
- string length is defined in characters, not bytes
- parse/format helpers such as `to_int`, `to_float`, and `to_string`
- string-producing/logging use cases must work without Python

**Control and observability:**

- scalar `if cond then a else b`
- `print(x)` and `debug(x)` as minimal IO-backed debugging tools with explicit `IO`
  effect tracking
- tensor shape queries such as `shape`, `rank`, and `numel`
- `Option[T]` promoted as the practical failure-returning surface for parse/lookups

### Implementation Shape

- extend Surf/Deep/type docs and implementation for scalar/string literals and scalar
  conditionals
- extend type inference/checking for non-tensor scalar and string operations
- extend the evaluator for scalar/string execution
- lower scalar/string/`Option`/print logic through a compiled host-value lane so
  `chelis build` does not fall back to evaluator-only behavior for mixed programs
- keep that host-value lane on CPU for both C and HIP targets; tensor kernels still use
  the existing tensor DAG/device paths
- use tagged execution values on the Tide/Python wire surface
- keep tensor computation semantics unchanged: no implicit scalar/tensor blending

### Acceptance Oracle

A pure Chelis training-step-style program can:

- compute scalar stopping criteria
- build or format a checkpoint/log path as a string
- print progress without Python
- compile with `chelis build --target c`, run as a native binary, and match `chelis eval`

---

## 3d: Collections and Iteration

**Status:** shipped.

**Goal:** add the variable-length data structures and functional iteration primitives
required for preprocessing and dataset plumbing.

### Why This Matters

Real AI programs need to represent:

- lists of token ids
- lists of sentences with different lengths
- dictionaries for vocabularies and configs
- dataset rows and intermediate preprocessing results

Tensors alone cannot express that variable-length host-side structure.

### Required Surface

**Collections:**

- immutable `List[T]`
- immutable `Dict[K, V]` with practical key types such as `String` and `Int`

**Current executable slice:**

- shipped compiled collection slice: list literals, `List[T]`, `len`, `index`,
  `append`, `concat`, `take`, `drop`, `chunk`, `flatten`, `range`, `zip`,
  `enumerate`, numeric `to_tensor`, practical rank-1 `to_list`, `pad_sequences`, and
  the first immutable `Dict[K, V]` builtins: `dict_of`, `dict_get`,
  `dict_contains`, `dict_remove`, `dict_insert`, `dict_merge`, `dict_keys`,
  `dict_values`, `dict_entries`, plus compiled higher-order iteration (`map`,
  `filter`, `fold`, `scan`, `partition`, `flat_map`)
- this now covers the practical compiled collection surface needed to hand off cleanly
  to `3g` tokenization/data-loading work rather than leaving obvious batching or
  nested-list gaps behind

**Iteration primitives:**

- `map`, `filter`, `fold`, `scan`, `partition`, `flat_map`, `zip`, `enumerate`,
  `range`
- collection length (lists and dicts), indexing, append/concat, sequence truncation
  (`take`, `drop`), batching (`chunk`), nested-list flattening, key lookup, immutable
  dict remove/update/overlay, key/value/entry enumeration, and dataset-friendly
  helpers such as cumulative scans, stable boolean partitioning, and callback-driven
  list expansion
- effect propagation through iteration
  this is now covered by checker tests for callback-driven `IO` and `Random`

**Collection/tensor bridge:**

- list-to-tensor conversion for numeric lists
- practical rank-1 tensor-to-list conversion
- stacking and padding helpers
- `pad_sequences` as the critical bridge from variable-length token lists to batched
  tensor inputs

### Implementation Shape

- add collection types and type inference rules
- add collection literals and built-in iteration APIs
- add evaluator/runtime support for immutable collections
- document linearly typed tensor elements inside collections without making collections
  themselves linear by default
- compile collections through the same host-value lane used by `3c` in both C and HIP
  builds; collections stay host-side as CPU data structures, but they are not
  evaluator-only
- keep tensors as the fixed-shape compute substrate and make the list/tensor bridge
  explicit

### Acceptance Oracle

A pure Chelis preprocessing program can:

- build a vocabulary/config map
- transform a list of examples with functional iteration
- truncate a variable-length sequence before tensorization
- immutably extend or overlay a config/vocabulary dictionary
- pad variable-length integer sequences into a batched tensor input

Current shipped oracles for the executable collection slices:

- `examples/list_foundation.ch` survives `chelis fmt`, `chelis check`, `chelis eval`,
  and `chelis build --target c`, and the compiled binary matches eval output
- `examples/dict_foundation.ch` survives `chelis fmt`, `chelis check`, `chelis eval`,
  and `chelis build --target c`, and the compiled binary matches eval output
- `examples/iter_foundation.ch` survives `chelis fmt`, `chelis check`, `chelis eval`,
  and `chelis build --target c`, and the compiled binary matches eval output

---

## 3h: Core Numeric Primitives

**Goal:** close the most important tensor-language expressiveness gaps between the
initial RISC + derived surface and the operations real models and numerical libraries
expect to use directly.

### Why This Matters

The shipped scalar/string/collection/data foundations make Chelis a host language, but
serious AI/model code still needs a richer tensor core than the original minimal set.
Without this cut, users still hit avoidable walls around:

- Einstein-style contractions and batched linear algebra
- skip connections and multi-head packing/unpacking flows
- embedding lookup and index-driven tensor access
- masked/conditional tensor updates
- cumulative and order-sensitive tensor statistics
- diagonal/trace style linear-algebra utilities
- basic value clipping and training-control helpers

### Required Surface

**Core primitives added in this phase:**

- `einsum`
- `concat` / `split`
- `gather` / `scatter`
- `where`
- `cumsum`
- `sort`
- `diagonal` / `trace`
- `clamp`

**Priority notes:**

- `einsum` is the single highest-impact addition because it covers matmul, batched
  matmul, transpose, trace, outer products, and common contraction patterns in one
  primitive
- `concat` / `split` are the minimal structural operations needed for modern
  transformer-style model blocks
- `diagonal` / `trace` and `clamp` are small but broadly useful numerical surfaces

**Standard-library addition unlocked here:**

- `Std.Nn.Embedding` as the explicit named surface over `gather`

### Implementation Shape

- extend the type/checking/lowering/backend docs and implementation for the new tensor
  primitives
- keep `Std.Nn.Embedding` in the standard library rather than the compiler core, but
  make the underlying `gather` primitive part of this phase
- preserve the Phase 0/1 invariants: explicit shapes, explicit broadcasting, and
  backend agreement with the reference C path
- reject deterministic literal-driven `3h` value errors during `chelis check` when the
  offending extents/indices are statically concrete; runtime-only bad values still fail
  during execution, but compiled C exits non-zero rather than aborting

### Acceptance Oracle

A pure Chelis model program can:

- express common contraction-heavy blocks using `einsum`
- concatenate and split tensor features without falling back to Python
- perform embedding lookup through `Std.Nn.Embedding`
- clamp or trace intermediate tensors in compiled programs on the C path

Current shipped oracle for the executable `3h` slice:

- `cargo test -p chelis-cli phase3h_numeric_acceptance_oracle -- --nocapture`
  which proves both the compiled tensor-structural example path
  (`examples/tensor_structural_ops.ch`, including `einsum`) and the Reef package path
  (`Std.Nn.Embedding` imported from `chelis-std`) build to valid C artifacts

---

## 3g: Data Loading and Tokenization

**Goal:** make Chelis self-sufficient for the preprocessing path every serious AI
program needs.

### Why This Matters

Without file I/O and tokenization, Chelis can only consume already-prepared tensors.
That leaves the most basic LLM/data workflow steps in Python:

- read text
- parse configs/data
- tokenize text
- batch and pad tokens

Phase 3 is not complete until those steps are expressible in pure Chelis.

### Required Surface

**Data loading:**

- text file I/O: read/write text files
- CSV loading for small/medium structured datasets
- JSON loading for configs, vocabularies, and metadata

**Tokenization:**

- a BPE tokenizer as the minimum practical tokenizer surface
- ability to load an external vocabulary/merge artifact format used in real workflows
- encode/decode text to and from integer token sequences

**Batching bridge:**

- batch encode helpers
- padding to fixed sequence length
- clean handoff from `List[List[Int]]` to model-ready tensors

### Implementation Shape

- model file I/O as explicit IO-effect operations
- document tokenizer/state formats and library placement without inventing unnecessary
  registry or remote-service machinery
- keep the first tokenizer target pragmatic: enough to ingest ordinary HuggingFace-style
  BPE assets rather than inventing a novel Chelis-native tokenizer format

### Acceptance Oracle

A pure Chelis program can:

- read a text file
- tokenize its contents into integer sequences
- batch and pad those sequences into tensors
- feed them into a model without Python preprocessing

---

## 3i: `Std.Time` and `Std.Decimal`

**Goal:** round out the standard-library host-language surface with time and
exact-arithmetic utilities that real workflows need but the compiler core should not own.

### Why This Matters

Pure Chelis workflows still need ordinary application scaffolding around the model:

- dates and durations in schedules, checkpoints, and reporting
- exact decimal arithmetic for money/config/reporting cases where binary floats are the
  wrong surface

These do not justify new compiler intrinsics, but they do belong in the Phase 3
language-completeness story rather than an indefinite backlog.

### Required Surface

- `Std.Time` for dates, timestamps, durations, and basic time arithmetic
- `Std.Decimal` for exact decimal values and arithmetic

### Implementation Shape

- ship both as standard-library modules, not new core-language primitives
- keep the APIs package-friendly under the existing Reef / `chelis-std` model
- sequence this after `3g` so data/tokenization remains the critical practical
  milestone and before `3f` so the teaching surface can cover the complete host-side
  standard stack

### Acceptance Oracle

A pure Chelis workflow can:

- represent and format a timestamp or duration without Python
- represent exact decimal configuration/reporting values without binary-float drift
- use those values in package-friendly standard-library code

---

## 3j: School — Numerical Methods, Statistics, and Optimization

**Goal:** A reef package providing the numerical methods that sit between raw tensor
primitives and domain applications. The equivalent of scipy.stats + scipy.optimize +
scipy.integrate for Chelis.

**Prerequisite:** 3h (core numeric primitives), 3i (Std.Time for time-series stats).

### Modules

| Module | Contents | Key Dependencies |
|---|---|---|
| `School.Stats` | Descriptive statistics (variance, skew, kurtosis, median), correlation, covariance, shrinkage estimators | 3h: sort, quantile, einsum |
| `School.Distributions` | Normal, LogNormal, Uniform, Student-t, Chi-squared — PDF, CDF, inverse CDF, sampling | `Random` effect, scalar math |
| `School.Optim` | Convex optimization solvers (QP, SOCP, LP). Differentiable optimization via implicit differentiation through KKT conditions. NOT neural network optimizers (those are `Std.Optim`). | einsum, linear algebra |
| `School.Interpolation` | Linear, cubic, spline interpolation | sort, gather |
| `School.LinAlg` | SVD, PCA, eigendecomposition, Cholesky — higher-level wrappers over tensor primitives + einsum | einsum, existing BLAS |
| `School.Testing` | Hypothesis testing, confidence intervals, p-values | `School.Distributions`, `School.Stats` |
| `School.ODE` | ODE solvers (Euler, RK4, adaptive step). Composes with `grad` for neural ODE support. | cumsum, host control flow |
| `School.SDE` | SDE solvers (Euler-Maruyama, Milstein). Uses `Random` effect. | `School.ODE`, `Random`, cumsum |
| `School.Integrate` | Numerical integration (trapezoidal, Simpson's, Gaussian quadrature) | fold, scalar math |
| `School.Roots` | Root finding / nonlinear equations (Newton-Raphson, bisection, Brent) | scalar math, host control flow |
| `School.Signal` | Signal processing (FFT, STFT, filtering). **Blocked by complex numbers (Phase 5f) — stub in 3j.** | Phase 5f complex tensors |

### Implementation Strategy

All modules are pure Chelis programs built from tensor primitives, scalar math, and
collections. No C FFI, no compiler special-casing. The QP solver is the most complex
module (iterative algorithm with convergence checking); everything else is functional
composition over existing operations.

`School.Signal` ships as a typed API stub in 3j (like `Std.IO.Safetensors` was in 3a) —
correct signatures and documentation, but implementation blocked by complex number
support in Phase 5f.

`grad` through ODE solvers is the highest-value composition test:
`grad(solve_ode(f, x0, t), wrt=x0)` must work for neural ODE research.

### Test Plan

- Each module has at least 3 positive tests comparing against scipy/numpy reference
  values (within tolerance)
- Negative tests: wrong input shapes, unsupported types
- AD composition test: `grad` through `School.ODE.rk4`, `School.Optim.solve_qp`,
  `School.Interpolation.cubic`
- Effect propagation: `School.Distributions.sample` propagates `Random`,
  `School.Signal` stub propagates correct effect annotations
- Package gate: `chelis reef build` produces a valid `.chb`, consumer imports and
  type-checks

### Acceptance Oracle

`cargo test -p chelis-cli phase3j_school_oracle -- --exact` — builds school from source,
imports it in a consumer, runs a statistical analysis pipeline (generate data from a
distribution, fit ODE, compute confidence interval).

**Effort:** large. The QP solver and ODE integrator are the bulk; stats and
distributions are straightforward.

---

## 3k: Coral — Typed Dataframes

**Goal:** A reef package for structured tabular data where numeric columns are
GPU-accelerable tensors and string columns are host-side lists. The unique feature: AD
flows through dataframe operations, enabling sensitivity analysis no existing dataframe
library supports.

**Prerequisite:** 3h (gather, scatter, argsort for sort-by/group-by), 3d (collections
for string columns), 3g (Std.IO.Csv/Json for data loading).

### Core Design

A DataFrame is `Dict[String, Column]` where:

```chelis
type Column =
  | IntCol(tensor[n, int64])
  | FloatCol(tensor[n, f32])
  | StringCol(List[String])
  | BoolCol(tensor[n, bool])
```

Numeric columns are tensors on the lazy RISC DAG — they go through the tensor lane, get
GPU-accelerated, support AD, and benefit from the compiler's operation fusion. A
`filter → mutate → aggregate` pipeline on numeric columns may compile to a single fused
kernel. String columns are host-side lists — they go through the host lane and execute
eagerly. The type system tracks which columns are which. No query optimizer — numeric
optimization comes from the tensor compiler's existing fusion passes, not a
dataframe-specific planner.

### Modules

| Module | Contents | Key Primitives Used |
|---|---|---|
| `Coral.Frame` | DataFrame construction, column selection, row filtering (boolean mask → `gather`), sorting by column (`argsort` → `gather` all columns), mutation (add computed column), column type queries | gather, argsort, where |
| `Coral.GroupBy` | Group-by via `argsort` + run-length detection, aggregation (sum, mean, count, min, max per group) via segmented `scatter(..., "add")` | argsort, scatter, cumsum |
| `Coral.Join` | Sort-merge join on typed key columns, left/inner/outer join variants | argsort, gather, concat |
| `Coral.Reshape` | Pivot (long → wide), melt (wide → long), stack/unstack | Dict manipulation, tensor reshape |
| `Coral.IO` | DataFrame-aware CSV loading (wraps `Std.IO.Csv`, auto-detects column types, returns typed DataFrame). DataFrame-aware JSON loading. DataFrame → CSV export. | `Std.IO.Csv`, `Std.IO.Json`, string parsing |

### AD Through Dataframes

The key differentiator. Because filter is `gather` and aggregation is
`scatter(..., "add")` + `sum`/`mean`, the entire filter → aggregate pipeline is
differentiable:

```chelis
def portfolio_risk(prices: Coral.Frame, threshold: f32) -> f32 = {
  -- filter: gather (differentiable)
  high_vol = Coral.filter(prices, \row -> get_float(row, "volatility") > threshold)
  -- aggregate: mean over tensor column (differentiable)
  Coral.mean_col(high_vol, "return")
}

-- Sensitivity of risk measure to threshold
grad(portfolio_risk, wrt=threshold)  -- works because filter → gather → AD
```

No existing dataframe library supports this.

### Test Plan

- `Coral.Frame`: construct from columns, select, filter, sort_by, mutate — all produce
  correct results
- `Coral.GroupBy`: group_by + sum/mean/count matches pandas.groupby reference on test
  data
- `Coral.Join`: inner join matches pandas.merge on test data, key type enforcement works
- `Coral.IO`: CSV round-trip (load → export → reload) preserves data and column types
- AD: `grad` through filter + aggregate pipeline produces correct gradients
- GPU: numeric column operations compile to HIP and produce correct results (manual gate)
- Negative: wrong column name errors, type mismatch errors, join key type mismatch errors

### Acceptance Oracle

`cargo test -p chelis-cli phase3k_coral_oracle -- --exact` — loads a CSV into a
DataFrame, filters rows, groups by a column, aggregates, and verifies results match
expected values. Plus a separate AD test computing `grad` through a filter-aggregate
pipeline.

**Effort:** medium. The core Frame/GroupBy/IO modules are the priority; Join and Reshape
can ship with minimal implementations and grow.

---

## 3l: Treasure — Finance

**Goal:** A reef package for quantitative finance. Pricing models, risk measures, yield
curves, stochastic processes, order books. Built entirely on `chelis-std` + `school` +
`coral`. Contains only finance-specific logic.

**Prerequisite:** 3j (school — distributions, optimization, SDE solvers), 3k (coral —
for loading/manipulating financial data), 3i (Std.Time for dates, Std.Decimal for cash
amounts).

### Modules

| Module | Contents | Key Dependencies |
|---|---|---|
| `Treasure.Pricing` | Black-Scholes analytical, Heston semi-analytical, SABR calibration, Monte Carlo engines with variance reduction. Greeks via `grad` for free — write the pricing function, `grad(price, wrt=(spot, vol, rate))` gives delta/vega/rho automatically. | `School.Distributions`, `School.SDE`, `Random` effect, cumsum |
| `Treasure.Risk` | VaR (parametric, historical, Monte Carlo), CVaR/expected shortfall, stress testing, scenario generation | `School.Stats`, sort/quantile, `Random` effect |
| `Treasure.Curves` | Yield curve construction (bootstrap from market instruments), interpolation (linear, cubic, Nelson-Siegel), day count conventions (ACT/360, ACT/365, 30/360) | `School.Interpolation`, `School.Roots`, `Std.Time` |
| `Treasure.Stochastic` | SDE models: GBM, Heston, SABR, jump-diffusion. Path generation using cumsum + `School.SDE`. Variance reduction (antithetic, control variates). | `School.SDE`, `Random`, cumsum, einsum |
| `Treasure.Orderbook` | Limit order book representation (price-priority sorted collections), matching logic, bid/ask spread computation, VWAP | Host-side collections, sort, `Std.Decimal` |

### What Makes This Work in Chelis

- **Greeks for free:** `grad(black_scholes_price, wrt=(spot, vol, rate, T))` gives all
  four first-order Greeks in one backward pass. `vmap(grad(...))` gives per-instrument
  Greeks for a portfolio. No bump-and-reprice, no finite differences.
- **Reproducible Monte Carlo:** The `Random` effect with `withSeed` handlers means every
  simulation is exactly reproducible. Two runs with the same seed produce identical
  paths. This is a regulatory requirement.
- **Typed market data:** Named tensor dimensions like `tensor[instrument, scenario, f32]`
  prevent accidentally multiplying a `[portfolio, maturity]` matrix by a
  `[maturity, scenario]` matrix when the dimensions don't match.
- **Effect-tracked data provenance:** A function that reads from a market data feed has
  `IO` effect. A function using Monte Carlo has `Random` effect. The type system tracks
  what each computation depends on.

### Test Plan

- `Treasure.Pricing`: Black-Scholes price matches analytical formula (< 1e-10 error)
- `Treasure.Pricing`: Monte Carlo price converges to Black-Scholes analytical for
  vanilla European call (< 1% error with 100K paths)
- `Treasure.Pricing`: Greeks via `grad` match analytical Black-Scholes Greeks
  (< 1e-6 error)
- `Treasure.Risk`: Parametric VaR matches `School.Distributions.Normal.ppf` at standard
  confidence levels
- `Treasure.Curves`: Bootstrap reproduces known market instrument prices (< 1bp error)
- `Treasure.Stochastic`: GBM paths satisfy known statistical properties
  (mean = spot * exp(mu*T), variance matches theory)
- `Treasure.Orderbook`: matching logic satisfies price-time priority invariant
- Effect tracking: MC pricing propagates `Random`, curve construction propagates `IO`
  for market data
- Reproducibility: same seed produces identical prices across runs

### Acceptance Oracle

`cargo test -p chelis-cli phase3l_treasure_oracle -- --exact` — prices a European call
option via Black-Scholes and Monte Carlo, verifies convergence, computes Greeks via
`grad`, loads market data via `coral`, and produces a risk report.

**Effort:** medium. Black-Scholes + Monte Carlo + basic risk is the core; curves and
order book are smaller. The bulk of the work is composing existing primitives (`school`
solvers, `coral` dataframes, tensor ops), not implementing new infrastructure.

---

## 3f: SKILL.md v2

**Goal:** Update the teaching surface for the full Phase 2 + Phase 3 language, including
domain shells.

**Prerequisite:** All other Phase 3 sub-phases complete.

### Required Surface

The refreshed skill should teach:

- the shipped pipe-first Surf idiom from `3e`
- effects, linearity, macros, `vmap`, and tuples
- handler syntax (`withSeed`, `withDevice`)
- scalar types (`Int`, `Float`, `Bool`) and operations
- strings and string operations
- collections (`List`, `Dict`) and functional iteration (`map`, `filter`, `fold`)
- `Option` type and pattern matching
- core numeric primitives (`einsum`, `concat`, `gather`, `cumsum`, `sort`, etc.)
- list/tensor bridge (`pad_sequences`, `stack`, `to_tensor`)
- file I/O and `IO` effect
- CSV/JSON parsing
- tokenizer usage
- `Std.Time` and `Std.Decimal` host-program idioms
- package imports (`Std.*`, `School.*`, `Coral.*`, `Treasure.*`)
- dataframe operations (`coral`)
- numerical methods (`school`)
- financial models (`treasure` overview, not exhaustive)
- the boundary between host-side preprocessing and tensor compute inside Chelis itself

### Acceptance Oracle

All SKILL.md examples validated via `skill_suite.rs` against the current compiler. Eval
on target base models to measure improvement over SKILL.md v1.

---

## Shell Ecosystem (Phase 3 Sub-Phases)

The domain shells are now proper Phase 3 sub-phases (3j, 3k, 3l) with full
specifications above, not deferred post-phase work. The tier structure is:

- `chelis-std` (core) — standard library shipped as the first Reef package
- `school` (3j) — general numerical methods on top of `chelis-std`
- `coral` (3k) — typed dataframes on top of `chelis-std`
- `treasure` (3l) — finance on top of `chelis-std` + `school` + `coral`

---

## Phase 3 Red-Team Checkpoint

Before calling Phase 3 healthy enough to continue, red-team these concrete surfaces:

**Scalar/string foundation (`3c`):**

- scalar arithmetic and conditionals work without silently becoming tensor operations
- strings are usable for paths, labels, and logs
- `print` / `debug` are honest about their IO/effect behavior

**Collections/iteration (`3d`):**

- collection APIs handle variable-length data without hidden mutation
- effect propagation through `map` / `fold` is correct
- list/tensor bridging rejects malformed shape cases clearly

**Core numeric primitives (`3h`):**

- `einsum` agrees with the reference backend on the supported contraction corpus
- `concat` / `split`, `diagonal` / `trace`, and `clamp` behave consistently across
  evaluator and compiled paths
- `Std.Nn.Embedding` exercises the real `gather` path rather than a fake host-side stub

**Data loading/tokenization (`3g`):**

- text/CSV/JSON loading returns the documented structures
- tokenizer encode/decode is deterministic against the documented assets
- batching/padding produces the expected tensor shapes and values

**Standard library host types (`3i`):**

- `Std.Time` and `Std.Decimal` stay standard-library scoped rather than leaking
  compiler-intrinsic assumptions
- examples/docs do not overclaim backend or tensor-kernel relevance for these modules

**Numerical methods (`3j` School):**

- stats functions agree with scipy reference values within tolerance
- `grad` through ODE solvers produces correct gradients
- `School.Signal` stub has correct type signatures but clearly errors at runtime
- `School.Optim` solvers converge on well-conditioned problems and reject ill-conditioned
  inputs

**Dataframes (`3k` Coral):**

- filter/group-by/join produce correct results against pandas reference
- AD flows through filter → aggregate pipelines
- GPU compilation of numeric column operations works (manual gate)
- wrong column names and type mismatches produce clear errors

**Finance (`3l` Treasure):**

- Black-Scholes price matches analytical formula
- Greeks via `grad` match analytical Greeks
- Monte Carlo converges to analytical for vanilla options
- same seed produces identical prices across runs

**Teaching surface (`3f`):**

- `SKILL.md` teaches the real executable language, not a stale tensor-only subset
- examples align with the package/style/python foundations already shipped
- coverage includes domain shells (School, Coral, Treasure)

---

## Phase 3 Dependency and Size Summary

| Sub-phase | Size | Dependencies | Nature |
|---|---|---|---|
| `3e`: Pipe-first style pass | shipped | none | Engineering |
| `3a`: Package system | shipped | `3e` | Engineering |
| `3b`: Python FFI interop core | shipped | `3a` | Engineering |
| `3b-ii`: Direct execution + NumPy | shipped | `3b` | Engineering |
| `3c`: Scalar and string foundation | shipped | `3e` | Engineering |
| `3d`: Collections and iteration | shipped | `3c` | Engineering |
| `3h`: Core numeric primitives | medium | `3d` | Engineering (RISC ops + AD + backends) |
| `3g`: Data loading and tokenization | medium | `3h` | Engineering (I/O + pure Chelis libraries) |
| `3i`: `Std.Time` and `Std.Decimal` | small | `3c`, `3a` | Engineering (pure Chelis std modules) |
| `3j`: School | large | `3h`, `3i` | Pure Chelis library (stats + optim + ODE/SDE) |
| `3k`: Coral | medium | `3h`, `3d`, `3g` | Pure Chelis library (dataframes) |
| `3l`: Treasure | medium | `3j`, `3k`, `3i` | Pure Chelis library (finance) |
| `3f`: SKILL.md v2 | small | all above | Documentation |

This phase is intentionally sequential and pragmatic. The remaining work is about making
Chelis usable, not publishable. `3j` and `3k` can overlap (no mutual dependency). `3l`
depends on both. `3f` goes truly last because it must cover the complete ecosystem
including the domain shells.

---

## Phase 3 Completion Check

Before calling Phase 3 complete:

- `3a`, `3b`, `3b-ii`, and `3e` remain honest shipped foundations
- `3c` provides practical scalar/string programming without Python fallback
- `3d` provides collections and iteration for variable-length host-side data
- `3h` provides the expanded tensor-language surface needed for real model code
- `3g` provides text/config/data loading plus tokenizer and batching support
- `3i` provides `Std.Time` and `Std.Decimal` as practical standard-library host types
- `3j` provides numerical methods (stats, optimization, ODE/SDE) as a Reef package
- `3k` provides typed dataframes with AD through tabular operations as a Reef package
- `3l` provides finance-specific pricing, risk, and stochastic process tools as a Reef
  package
- `3f` reflects the full post-shell language in `SKILL.md` and examples, including
  `School.*`, `Coral.*`, and `Treasure.*` package imports
- a pure Chelis program can read text, tokenize it, batch/pad it, run a model, compute
  loss and gradients, and print results without Python
- domain shells compose correctly: `treasure` depends on `school` + `coral`, all build
  and import through the Reef pipeline
