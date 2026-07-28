# Chelis Capability Surface (canonical)

The authoritative inventory of what the Chelis language, compiler, and bundled
`chelis-std` actually provide. This file is the **upstream source of truth**; the
per-shell `docs/CHELIS_SURFACE.md` required by
[`spec/design/shell_repo_contract.md`](../spec/design/shell_repo_contract.md) §3
is a domain-scoped, `@pin`/`@upstream`-annotated *view* derived from this one and
should cite it rather than re-deriving the surface.

> **Tracks:** `main` · **Current release:** chelis 0.9.0 · **Last refreshed:** 2026-06-23

**Read this before designing around a suspected language gap.** Most historical
downstream narrowness (recursive list-walks in a tensor-first language, per-rank
verb copies, parameters frozen out of training) traces to contributors not knowing
this surface.

**Authoritative sources** this digest is built from — if this file and one of these
disagree, the source wins and this file is the bug:

| Surface | Source of truth |
|---|---|
| Operation taxonomy + AD adjoints | [`spec/05-risc-primitives.md`](../spec/05-risc-primitives.md) |
| Tier-1 op set (the RISC DAG) | `RiscOp` enum, `crates/chelis-ir/src/dag.rs` |
| Closed builtin vocabulary | `BUILTIN_NAMES`, `crates/chelis-types/src/builtins.rs` |
| Type system, precisions, accumulators | [`spec/04-type-system.md`](../spec/04-type-system.md) |
| Transformations (`grad`/`vmap`/`jit`/`realize`) | [`spec/06-transformations.md`](../spec/06-transformations.md) |
| Backends + reject lists | [`spec/08-backends.md`](../spec/08-backends.md), `crates/chelis-backend-{c,hip,metal}` |
| Scope taxonomy (core vs std vs shell) | [`spec/design/chelis_canonical_reference.md`](../spec/design/chelis_canonical_reference.md) §8.5 |

---

## 0. Two lowering lanes (read this first)

Not everything lowers to the RISC DAG. A checked program is lowered into **two
lanes**, and which lane an operation takes determines its backend reach and whether
it differentiates:

- **DAG lane** (`crates/chelis-ir/src/dag.rs`, `lower.rs`, `tier2.rs`) — the
  tensor-compute lane. Tier-1 primitives have their own `RiscOp` node; Tier-2
  derived ops decompose into Tier-1 during IR construction. The DAG is what the
  C/HIP/Metal backends codegen, what the reference evaluator runs, what `grad`
  differentiates, and what the prover can reason about. **Reaches all backends**
  (subject to per-backend coverage gaps in §8).
- **Host lane** (`crates/chelis-ir/src/host.rs`, `chelis-compiler-api/src/runtime/host_ops.rs`,
  `chelis-backend-c/src/host_emit.rs`) — operations that don't fit the closed RISC
  vocabulary: higher-order combinators (`fold`/`scan`/`map`), data-dependent control
  flow (`sort`), sequential prefix (`cumsum`), meta-ops (`einsum`), and all
  collection / string / I/O builtins. These are emitted as **direct `chelis_*` C
  runtime calls** or run by the host interpreter. **C backend only** — the GPU
  backends reject any program containing a host-lane op (§8).

The host lane is the *outer* program; tensor math is carved out of it into DAG
regions. Type/effect/linearity checking runs **before** the split, so both lanes are
equally checked; AD, GPU codegen, and SMT proof only reach the DAG lane.

Generic ADT layout and polymorphism classification on the host lane consume the
checker's alias-resolved ADT registry and authored-signature metadata. They are
not reconstructed from source declarations. Nested dimension parameters remain
representation-erased (chelis#940), while stored tensor dtype parameters stay
concrete through ADT nesting and beta reduction (chelis#948).

**Lane legend used in the tables below:**

| Mark | Meaning |
|---|---|
| `DAG` | Lowers to the RiscOp DAG; reaches C/HIP/Metal (per §8). |
| `Host` | Direct `chelis_*` C call / host interpreter; **C backend only**. |
| `DAG+Host` | Has both lowerings; the DAG form is used in tensor context, the host form in host context. |

---

## 1. Tier-1 RISC primitives (the DAG)

The irreducible set. Each has a `RiscOp` variant and a defined reverse-mode adjoint
(`spec/05` §2). Signatures use `D` for a dimension list and `p` for precision; tensor
inputs are borrow-typed (`&tensor`, auto-borrowed at call sites).

### 1.1 Elementwise binary — `spec/05` §2.1

| Name | Signature | AD adjoint (given upstream `g`) |
|---|---|---|
| `add` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | `(g, g)` |
| `mul` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | `(g*y, g*x)` |
| `div` | `(&tensor[D,p_float], &tensor[D,p_float]) -> tensor[D,p_float]` | `(g/b, -g*y/b)`; IEEE-754, **float operands only** (chelis#178) |
| `floor_div` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | **non-differentiable** — `grad` rejects; round quotient toward −∞ (Python `//`); ints and floats |
| `trunc_div` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | **non-differentiable** — `grad` rejects; round toward zero (C `/`); **integer operands only** |
| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | `(g*(x>=y), g*(x<y))` |
| `cmplt` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,bool]` | zero gradient (by design) |

`div`/`recip` are native Tier-1 (IEEE-754, correct on the full real line) — **not** an
`exp(neg(log))` decomposition. **chelis#178:** `div` is now float-only; integer `div`
(and the `/` operator on ints) is a type error citing `spec/05` §2.1 and pointing at
`floor_div` (round toward −∞, matching torch/JAX/numpy `floor_divide` / Python `//`)
or `trunc_div` (round toward zero, the C `/` quotient). Neither matches the torch/JAX
float upcast (their default `divide`); use a `cast` first for that.

### 1.2 Elementwise unary — `spec/05` §2.2 (float types only)

| Name | AD adjoint |
|---|---|
| `neg` | `-g` |
| `recip` | `-g*y*y` (= `-g/x²`) |
| `exp` | `g*exp(x)` |
| `log` | `g/x` |
| `sin` | `g*cos(x)` |
| `cos` | `-g*sin(x)` |
| `tan` | `g/cos²(x)` |
| `atan` | `g/(1+x²)` |
| `sqrt` | `g/(2*sqrt(x))` |
| `abs` | `g*sign(x)` (0 at x=0) |
| `floor` | **non-differentiable** — `grad` rejects (`PiecewiseConstant`) |
| `ceil` | **non-differentiable** — `grad` rejects (`PiecewiseConstant`) |
| `round` | round-to-nearest-ties-to-even (banker's rounding); **non-differentiable** — `grad` rejects (`PiecewiseConstant`) |

### 1.3 Reduction — `spec/05` §2.3

| Name | Signature | AD adjoint |
|---|---|---|
| `sum` | `(&tensor[..,p], axis: int, accumulator: prec = default(p)) -> tensor[..,acc]` | `expand(g, axis)` |
| `max_reduce` | `(&tensor[..,p], axis: int) -> tensor[..,p]` | `g * one_hot(argmax)` |
| `min_reduce` | `(&tensor[..,p], axis: int) -> tensor[..,p]` | `g * one_hot(argmin)` |
| `prod_reduce` | `(&tensor[..,p], axis: int) -> tensor[..,p]` | per-slice product/quotient |
| `argmax_reduce` | `(&tensor[..,p], axis: int) -> tensor[..,int64]` | **non-differentiable** (index output) |
| `argmin_reduce` | `(&tensor[..,p], axis: int) -> tensor[..,int64]` | **non-differentiable** (index output) |

- **Axis must be a compile-time constant** (literal, or `cast(N,int32)` of a literal).
  A runtime-axis reduction is a check-time error (chelis#259). Negative axes index
  from the end (`-1` = last).
- **`accumulator` (sum only)** controls running-sum precision and result dtype.
  Defaults (no implicit promotion): bf16/f16→f32, f32→f32, f64→f64, int8/int16→int32,
  int32→int32, int64→int64. Full table: `spec/04` §5.7.1.
- `sum` uses a stride-4 ILP cascade — bit-exact with torch CPU `row_sum` for n≤16,
  **not** bit-exact with `numpy.sum`. GPU reduction kernels may differ at ~1 ULP.

### 1.4 Windowed reduction — `spec/05` §2.3.1 (Valid padding only)

| Name | Signature |
|---|---|
| `reduce_window_max` / `_min` / `_sum` / `_mean` | `(&tensor[..,p], window_shape: List[int32], strides: List[int32]) -> tensor[..,p]` |

Tier-1 in its own right (single `RiscOp::ReduceWindow{reducer,window_shape,strides}`;
adjoint via `RiscOp::ReduceWindowGrad`). Output extent per windowed axis is
`floor((d - window)/stride) + 1`. **Same padding is not implemented** — `pad`
explicitly first. Windowed extents must be statically known on the build path.
**C backend only; HIP/Metal codegen deferred and rejected** at build (§8).

### 1.5 Movement — `spec/05` §2.4

| Name | Signature | AD adjoint |
|---|---|---|
| `reshape` | `(&tensor[D_old,p], shape) -> tensor[D_new,p]` | `reshape(g, old_shape)` |
| `permute` | `(&tensor[..,p], axes) -> tensor[..,p]` | `permute(g, inverse_axes)` |
| `expand` | `(&tensor[D_small,p], axis, size) -> tensor[D_large,p]` | `sum(g, expanded_axes)` |
| `pad` | `(&tensor[D,p], padding, fill) -> tensor[D',p]` | `shrink(g, inverse_padding)` |
| `shrink` | `(&tensor[D,p], bounds) -> tensor[D',p]` | `pad(g, inverse_bounds)` |
| `stride` | `(&tensor[D,p], strides) -> tensor[D',p]` | expand/scatter |

`expand` does not copy data (stride-0 on the expanded axis) and addresses its insert
point by name; its insert axis must be a compile-time constant.

### 1.6 Memory & effectful — `spec/05` §2.5–2.6

| Name | Signature | AD / effect |
|---|---|---|
| `const` | `(value, shape...) -> tensor[shape,p]` | zero gradient |
| `load` | `(source, shape...) -> tensor[shape,p]` | zero gradient |
| `dropout` | `(&tensor[D,f32], rate: f32) -> tensor[D,f32]` | differentiable (mask fixed wrt seed); introduces `Random`. **Eval-only — not codegen'd by C/HIP/Metal build yet.** |
| `uniform_like` | `(&tensor[D,f32], lo: f32, hi: f32) -> tensor[D,f32]` | zero gradient; introduces `Random`; seeded via `with seed(Ni64) { }` |

Internal-only `RiscOp`s not directly callable from Surf: `Store`, `Copy`, `Drop`,
`Realize`, `Cast`, `FusedElem`, `OneHot`, `BlasMatmul` (the `matmul` specialization
target), `Gather`/`ScatterAdd`/`Scatter` (the sparse nodes below).

### 1.7 Sparse tensor-lane nodes — `spec/05` §3.5

First-class `RiscOp`s with evaluator/verifier/AD/C+HIP support:

| Surf builtin | Lowers to | Signature | AD adjoint |
|---|---|---|---|
| `gather` | `RiscOp::Gather{axis}` | `(&values, &indices, axis) -> tensor` | `ScatterAdd` |
| `scatter_replace` | `RiscOp::Scatter{axis}` | `(&base, &indices, &updates, axis) -> tensor` | **no_grad** (`NonDeterministicAtDuplicateIndices`) |
| `scatter_elements` | `RiscOp::ScatterElements{axis}` | `(&data, &indices, &updates, axis) -> tensor` | **no_grad** (`NonDeterministicAtDuplicateIndices`) |
| _(scatter-add internal)_ | `RiscOp::ScatterAdd{axis}` | — | `Gather` |

`scatter_replace` is last-write-wins (hyperplane shape) with a deterministic row-major
order rule; `scatter_elements` is the ONNX element-wise variant (`indices.shape ==
updates.shape`, `output.shape == data.shape`, `spec/05` §3.5.1) with the same
determinism and AD policy. Every backend (eval/C/HIP) observes it. Distinct from the
host-lane `scatter` (§3).

---

## 2. Tier-2 derived built-ins (DAG)

Convenience functions the desugarer emits and the IR pass decomposes into Tier-1
during construction (`crates/chelis-ir/src/tier2.rs`). They are **not** separate DAG
nodes — they exist in Deep AST only and end up as Tier-1 compositions, so they reach
all backends and differentiate via their decomposition. `spec/05` §3–4.

| Name | Lowering | AD |
|---|---|---|
| `sub` | `add(a, neg(b))` | differentiable |
| `eq`,`neq`,`gt`,`gte`,`lte`,`lt` | `cmplt` compositions (`spec/05` §3.2) | zero-grad (bool out) |
| `and`,`or`,`not` | `mul` / `max_elem` / `neg` on bools | zero-grad (bool) |
| `relu` | `max_elem(x, 0)` | differentiable (subgradient) |
| `sigmoid` | `recip(add(1, exp(neg(x))))` | differentiable |
| `tanh`,`silu`,`gelu` | `tier2.rs` decompositions | differentiable |
| `matmul` | `expand`+`mul`+`sum`, pattern-matched to BLAS (`spec/05` §4.1); optional `accumulator` | differentiable |
| `mean` | `div(sum(x,axis), count)` | differentiable |
| `softmax` | max-shift + `exp` + `sum` + `div` (`spec/05` §4.2) | differentiable |
| `min_elem` | `neg(max_elem(neg(a), neg(b)))` | differentiable |
| `layer_norm` | mean/var normalize + affine (`spec/05` §4.4) | differentiable |
| `conv2d` | `im2col` → `matmul` → `reshape` (`spec/05` §4.5) | differentiable |

The derived activations and arithmetic also have a host-lane C-emit path
(`host_emit.rs`) used when they appear inside the host lane — i.e. they are
effectively `DAG+Host`, but their canonical lowering is the DAG.

**Documented composite lowerings** that are spec-level recipes, not builtins in the
closed vocabulary (provided via desugaring or `chelis-std`/shell libraries):
`linear`, `cross_entropy`, `embedding`, `multi_head_attention`, `im2col`, `argmax`
(`spec/05` §3.5, §4.3–4.7).

> **`normalize`** is accepted by the type checker but has **no specified lowering**
> (`spec/05` §3.4 note). Do **not** treat it as a stable builtin until the spec and IR
> lowering are aligned.

---

## 3. Host-lane operations (C backend only)

These do **not** enter the RISC DAG. They are emitted as direct `chelis_*` C runtime
calls (`host_emit.rs`) or run by the host interpreter (`host_ops.rs`). **No AD
adjoint** (not differentiable through), and **GPU backends reject any program
containing them** (§8). This is the part most often missed when scanning for a
capability.

### 3.1 Data-dependent / meta tensor ops (Phase 3h, `spec/05` §1)

| Name | Signature | Notes |
|---|---|---|
| `cumsum` | `(&tensor, axis) -> tensor` | cumulative sum along axis; precision-preserving per dtype |
| `sort` | `(&tensor, axis) -> (values, indices)` | returns sorted values + int32 index tensor |
| `einsum` | `(equation: string, &lhs, &rhs) -> tensor` | 2-operand only today; no ellipsis; static-extent errors rejected at check |
| `diagonal` | `(&tensor, _, _) -> tensor` | diagonal extraction |
| `trace` | `(&tensor, _, _) -> tensor` | matrix trace |
| `where` | `(&cond, &a, &b) -> tensor` | `DAG+Host`: also has the `add(mul(cond,a),mul(neg(cond),b))` DAG form (`spec/05` §3.5) |
| `clamp` | `(&tensor, lo, hi) -> tensor` | elementwise clip |
| `concat` | `(lists/tensors, axis) -> tensor` | join along axis |
| `split` | `(&tensor, sizes, axis) -> list` | partition along axis |
| `scatter` | `(base, indices, updates, axis, mode: string) -> tensor` | host pentaop form (string mode); distinct from tensor-lane `scatter_replace` (§1.7) |
| `pad_sequences`, `pad_sequences_to` | variable-length padding | runtime-derived extents |

### 3.2 Host-runtime tensor builder — `spec/05` §3.6

| Name | Signature | Notes |
|---|---|---|
| `tensor_scan` | `(initial: T, fn: (T,int64)->T, n: int64) -> tensor[n, T]` | **host-only**, no AD; **`chelis build` (C/HIP) rejects it whole-program**; `grad`/`vmap` over a body reaching it are rejected (reachability-scoped). The sanctioned init-time per-index tensor constructor (`T` scalar; accumulator is f64-backed, exact to 2^53). |

**`tensor_scan` (tensor lane, init-only) ≠ `scan` (list combinator, §3.3).** Don't
put `tensor_scan` in a loss, forward, or anything `grad`/`build` must reach.

### 3.3 Higher-order list / sequence combinators

`map`, `filter`, `fold`, `scan`, `partition`, `flat_map`, `flatten`, `zip`,
`enumerate`, `chunk`, `take`, `drop`, `range`, `append`, `index`, `len`, `concat`.
Higher-order ones (`map`/`filter`/`fold`/`scan`/`partition`/`flat_map`) take callback
functions — the DAG has no function-pointer node, which is why they are host-lane.
The host lane is eager (no lazy list fusion).

### 3.4 Collections, strings, conversions

- **Dict:** `dict_of`, `dict_get`, `dict_contains`, `dict_remove`, `dict_insert`,
  `dict_merge`, `dict_keys`, `dict_values`, `dict_entries`.
- **String:** `string_len`, `string_concat`, `string_slice`, `string_contains`,
  `string_starts_with`, `string_ends_with`, `string_trim`, `to_string`.
- **Scalar coercion:** `to_int`, `to_float`.
- **Tensor↔host bridges & queries:** `rank`, `shape`, `numel`, `tensor_to_scalar`,
  `scalar_to_tensor`, `to_tensor`, `to_list`. `shape` returns a runtime list;
  reductions/expands still need compile-time-constant axes regardless.

### 3.5 I/O and process — introduces `Io`

`read_file`, `write_file`, `read_lines`, `read_bytes`, `file_exists`, `list_dir`,
`mmap_file`, `mmap_read`, `mmap_len`, `process_run`.

| Name | Signature | Notes |
|---|---|---|
| `process_run` | `(cmd: string, args: List[string]) -> (int64, string, string)` | argv, no shell. **Eval/test-only** — C/HIP/Metal build reject it (chelis#267). |

### 3.6 Diagnostics & test — `Test` effect on asserts

`print`, `fail`, `debug`, and the `test_assert*` family: `test_assert`,
`test_assert_eq_f32`, `test_assert_eq_int`, `test_assert_eq_bool`,
`test_assert_eq_string`, `test_assert_close_tensor`, `test_assert_eq_tensor_int64`.

### 3.7 Integer / bitwise elementwise

`mod`, `bitand`, `bitor`, `bitxor`, `shl`, `shr` — integer-only, host-lane,
no AD. Shifts use declared-width two's-complement semantics; counts at or
above the width fully shift out the value, while negative counts trap
([04-NUM-13]).

### 3.8 JSON I/O + decimal rounding — **eval-only** (chelis#890)

Native JSON over the prelude `Json` ADT
(`Json = JNull | JBool bool | JNum f64 | JStr string | JList List[Json] |
JDict Dict[string, Json]`). **Eval/test-only**: `chelis build` rejects every
name below whole-program (`EVAL_ONLY_HOST_BUILTINS`), like `process_run`.
All failures (malformed JSON, missing path, type mismatch, non-finite
number) are loud eval errors — no silent defaults.

| Name | Signature | Notes |
|---|---|---|
| `parse_json` | `(s: string) -> Json` | strict RFC 8259; a leading UTF-8 BOM is ignored (§8.1, matching `parse_csv`); **int-vs-float is decided at parse time** — a token with `.`/`e`/`E` becomes `JNum f64`, anything else becomes `JInt int64` (the `Std.Io.Json` and Python `json` rule), so integers stay exact; an integer literal too wide for int64 falls back to `JNum` and is the one remaining lossy case; duplicate keys: first position, last value; depth cap 512 |
| `to_json` | `(v: Json) -> string` | compact, **insertion-order keys**, `JInt` emitted exactly (no decimal point, no f64 round-trip), **shortest-round-trip f64** for `JNum` (Rust `{:?}` formatter — deliberately NOT the print channel, chelis#748/#723/#734); NaN/inf fail; non-ASCII emitted as raw UTF-8; byte-stable |
| `json_f64` | `(j: Json, path: string) -> f64` | dot-path: segment = dict key, or strictly all-digits list index (no sign, no leading zeros); fails name the missing key and list available keys; **widens `JInt` transparently** (lossy above 2^53 — use `json_int` for exactness) |
| `json_int` | `(j: Json, path: string) -> int64` | exact integer read. Refuses a `JNum` rather than truncating it, naming `json_f64` as the remedy (§C1.1: no silent narrowing) |
| `json_str` | `(j: Json, path: string) -> string` | |
| `json_list` | `(j: Json, path: string) -> List[Json]` | elements re-enter the accessors (element-relative paths) |
| `json_f64s` | `(j: Json, path: string) -> List[f64]` | list of numbers at path; `JInt` elements widen as in `json_f64`; any non-number element fails with its index |
| `jnum` | `(x: f64) -> Json` | **exactly f64** (bare literals are f32 per §5.3 and are rejected loudly — suffix them `0.1f64` or use `cast(n, f64)`; an f32 would quantize through the byte-exact serializer) |
| `jint` | `(n: int64) -> Json` | **exactly int64** (narrower integers rejected loudly — suffix `1i64` or `cast(n, int64)`), mirroring `jnum`'s width guard |
| `jstr` | `(s: string) -> Json` | |
| `jlist` | `(items: List[Json]) -> Json` | |
| `jdict` | `(entries: List[(string, Json)]) -> Json` | insertion order; duplicate keys upsert |
| `json_set` | `(j: Json, path: string, v: Json) -> Json` | returns updated value; missing intermediate dict keys auto-create nested dicts (output assembly); list segments replace existing elements only; path segments and the built result respect the 512 depth cap |
| `round_to` | `(x: f64, places: int) -> f64` | decimal rounding, **ties-to-even** on the exact binary value (= Python `round`): `round_to(2.5, 0) = 2.0`, `round_to(2.675, 2) = 2.67`; `places` in 0..=100; non-finite passes through. **f64-only** until its per-dtype semantics are authored in spec/05 (chelis#891 review: the widen/round/re-narrow path at f32 computes at other than the declared width, non-conforming under [04-NUM-8]) |

Composes with §3.5: `read_file |> parse_json`, accessors + tensor builtins
for compute, `jdict`/`json_set`/`to_json` + `write_file` for nested output.
Matching on the `Json` constructors is available for power users; the
accessors are the primary agent surface.

### 3.9 CSV I/O — **eval-only** (chelis#903)

Native CSV, first row = header, riding the `Json` ADT: a **Csv document**
is the fixed-shape Json value
`JDict {"columns": JList[JStr], "rows": JList[JDict]}` (cells from
`parse_csv` are `JStr` — no silent numeric coercion at parse time), so
every §3.8 accessor works on it (`json_list(c, "rows")`, `json_str(c,
"rows.0.px")`, `to_json(c)` for debugging) and there is deliberately no
`Csv` prelude type. **Eval/test-only** like §3.8 (`EVAL_ONLY_HOST_BUILTINS`).
All failures are loud eval errors — no silent NaN/defaults.

| Name | Signature | Notes |
|---|---|---|
| `parse_csv` | `(s: string) -> Json` | RFC-4180-ish: quoted fields, doubled embedded quotes, commas/newlines literal inside quotes; LF or CRLF (mixed ok); leading UTF-8 BOM stripped; blank rows only at EOF. Errors name the 1-based row, plus the 1-based column where one applies (quote/separator errors): unclosed quote, content after closing quote, bare `"` in an unquoted field, bare CR, ragged row (row-level), interior blank row (row-level), **duplicate header names** |
| `to_csv` | `(c: Json) -> string` | serializes the exact document shape `parse_csv` returns; the round-trip is **values-as-text**: every cell re-reads as the `JStr` of its serialized field text (all-`JStr` documents round-trip identically; `JNum` re-reads bit-exactly via `csv_f64`; cell *types* other than `JStr` do not survive — CSV is untyped). Cells may be `JStr`/`JNum` (**shortest-round-trip f64**, the same `format_f64_json` as `to_json`)/`JBool`/`JNull` (empty cell); minimal quoting (incl. a BOM-leading first header field), LF rows, trailing newline, byte-stable; loud on non-finite numbers, container cells, unexpected top-level keys, a row missing a declared column or carrying an undeclared/duplicate key |
| `csv_f64s` | `(c: Json, col: string) -> List[f64]` | whole column as numbers; strict JSON number grammar per cell — literally `parse_json`'s scanner (surrounding spaces/tabs tolerated, matching Python `float()`); empty or non-numeric cells fail naming the column, 0-based data row, and offending text; finite `JNum` cells (assembled docs) read directly, non-finite fail; `JBool`/`JNull` cells fail per-type |
| `csv_strs` | `(c: Json, col: string) -> List[string]` | whole column verbatim (`JStr` cells only; other cell types fail per-type — no cross-type coercion) |
| `csv_nrows` | `(c: Json) -> int64` | data rows (header excluded) |
| `csv_cols` | `(c: Json) -> List[string]` | header names in file order (preserved even for zero-row files) |
| `csv_f64` | `(c: Json, row: int, col: string) -> f64` | one cell as a number; `row` is a 0-based data-row index, any integer precision (bare literals work) |
| `csv_str` | `(c: Json, row: int, col: string) -> string` | one cell verbatim |

Missing columns fail naming the column **and listing the available
columns**. Document-shape validation is eager and uniform: every accessor
(including `csv_nrows`/`csv_cols`) checks that `columns` is a list of
unique strings and every row a JDict; cell types are checked at read.

**Integer precision (chelis#904 review):** `csv_f64`/`csv_f64s` parse
through the same number grammar into f64, so an integer above 2^53
**silently loses precision** — `9007199254740993` reads back as
`9007199254740992.0`, with no diagnostic. CSV columns of int64 IDs are
the *common* case, more so than in JSON. The exact escape hatch is
**`csv_str` + `to_int`** (or `csv_strs` + a `map` of `to_int`): cells
come out of `parse_csv` as `JStr`, so the text is preserved verbatim
until you choose the numeric read. Same limit and the same reason as
§3.8's `json_f64`; the exact-int64 story upstream is chelis#729.
Composes end-to-end with §3.5/§3.8:
`read_file |> parse_csv` → `csv_f64s` + tensor builtins → `round_to` →
`jdict`/`json_set`/`to_json` (or `to_csv`) + `write_file`.

---

## 4. Complete closed vocabulary (completeness check)

Every name in `BUILTIN_NAMES` (`crates/chelis-types/src/builtins.rs`), verbatim. The
block below mirrors the array exactly and is locked to it by the
`doc_surface_section_4_mirrors_builtin_names` test (`crates/chelis-types/src/builtins.rs`),
so it cannot silently drift. This is the audit surface — when bumping, diff
`BUILTIN_NAMES` against this block.

Three capabilities live *outside* the array and are intentionally absent below: `const`
and `load` are `RiscOp` memory nodes produced during lowering (not name-callable
builtins), and `dropout` is registered straight into the builtin type env (`builtin_env`)
rather than the array. All three are documented in §1.6. With those exceptions: if a name
is not in the block, it is not a builtin (it's `chelis-std`, a shell library, or
undefined).

```
Tier-1 DAG:   add mul div floor_div trunc_div max_elem cmplt neg recip exp log sin cos tan atan sqrt
              abs floor ceil round sum max_reduce min_reduce prod_reduce argmax_reduce
              argmin_reduce reduce_window_max reduce_window_min reduce_window_sum
              reduce_window_mean reshape permute expand pad shrink stride
              uniform_like gather scatter_replace scatter_elements
Tier-2 DAG:   sub eq neq lt gt lte gte and or not relu sigmoid tanh silu gelu
              softmax normalize mean matmul min_elem layer_norm conv2d
Host lane:    cumsum sort einsum diagonal trace where clamp concat split scatter
              pad_sequences pad_sequences_to tensor_scan
              map filter fold scan partition flat_map flatten zip enumerate chunk
              take drop range append index len
              dict_of dict_get dict_contains dict_remove dict_insert dict_merge
              dict_keys dict_values dict_entries
              string_len string_concat string_slice string_contains
              string_starts_with string_ends_with string_trim to_string to_int
              to_float rank shape numel tensor_to_scalar scalar_to_tensor to_tensor
              to_list
              read_file write_file read_lines read_bytes file_exists list_dir
              mmap_file mmap_read mmap_len process_run
              parse_json to_json json_f64 json_int json_str json_list json_f64s
              jnum jint jstr jlist jdict json_set round_to
              parse_csv to_csv csv_f64s csv_strs csv_nrows csv_cols
              csv_f64 csv_str
              print fail debug test_assert test_assert_eq_f32 test_assert_eq_int
              test_assert_eq_bool test_assert_eq_string test_assert_close_tensor
              test_assert_eq_tensor_int64
              mod bitand bitor bitxor shl shr
```

`normalize` is listed because it is in `BUILTIN_NAMES`, but it has **no specified
lowering** (`spec/05` §3.4) — treat it as unstable, not a stable builtin (see §2).

Prelude ADTs/constructors (also in scope): `Option`/`Some`/`None`,
`List`/`Cons`/`Nil`, `MappedFile`, and
`Json`/`JNull`/`JBool`/`JNum`/`JStr`/`JList`/`JDict` (§3.8).

---

## 5. Autodiff status summary — `spec/05` §5

`grad` is reverse-mode AD over the DAG; it differentiates any composition of Tier-1
primitives (Tier-2 inherit via decomposition).

- **Differentiable:** all Tier-1 except those below; all Tier-2 except comparisons/logicals.
- **Zero-gradient by design:** `cmplt` + comparisons (`eq`/`neq`/`lt`/`gt`/`lte`/`gte`),
  `and`/`or`/`not`, `const`, `load`, `uniform_like`.
- **Non-differentiable — `grad` rejects with a structured `AdError`:** `floor`, `ceil`,
  `round` (`PiecewiseConstant`); `argmax_reduce`, `argmin_reduce` (index output);
  `scatter_replace`, `scatter_elements` (`NonDeterministicAtDuplicateIndices`).
- **No AD (host lane):** every op in §3 — `cumsum`, `sort`, `einsum`, `fold`, `scan`,
  `tensor_scan`, etc. A differentiable path must stay in the DAG lane.
- `if/then/else` differentiates (chosen branch); loops/recursion do not differentiate
  *through* (grad-per-step inside a driver is fine). ADT/record/`match`/list bodies are
  not yet differentiable args — see the D1–D5 roadmap in
  [`spec/design/differentiable_language.md`](../spec/design/differentiable_language.md).

---

## 6. Backends — `spec/08-backends.md`

Three real codegen backends (`crates/chelis-backend-{c,hip,metal}`). **None invoke the
native compiler** — `chelis build` emits source + flags; the user runs `gcc`/`hipcc`/`clang++`.

| Target | Emits | Status |
|---|---|---|
| `c` (default) | `func.c` + `func.h` + `chelis_runtime.h`/`chelis_blas.h` + `libchelis_runtime.a` + flags; OpenMP loops, BLAS (`cblas_sgemm`/`dgemm`) calls | full reference; the numerical ground truth |
| `hip` | `func_hip.cpp` with embedded HIP kernel strings (JIT via `hiprtc`) + host runtime; rocBLAS for matmul | real; DAG-lane subset (rejects below) |
| `metal` | `func_metal.mm` with embedded MSL kernel strings (JIT via `MTLDevice`) + host runtime | real; DAG-lane subset (rejects below) |

### 6.1 The host-lane rule

> **Any program containing a host-lane op (§3) can only build to `c`.** The GPU
> backends accept a pure RiscOp DAG; if the program needs the host backend
> (`host_program_requires_host_backend`), `--target hip`/`metal` fall back to C
> codegen for the whole program.

### 6.2 DAG-lane GPU coverage gaps

| Op / feature | C | HIP | Metal |
|---|---|---|---|
| Elementwise, reductions, movement, memory, `FusedElem`, `Cast` | ✓ | ✓ | ✓ (no f64) |
| `BlasMatmul` | ✓ | ✓ (rocBLAS; bf16/f16 via GemmEx) | ✓ (tiled MSL) |
| `ReduceWindow` / `ReduceWindowGrad` | ✓ | ✗ rejected | ✗ rejected |
| `Pad`, `Shrink` | ✓ | ✗ rejected | ✗ rejected |
| `dropout` | ✗ (eval-only) | ✗ | ✗ |
| `Gather`/`ScatterAdd`/`Scatter` | ✓ (all precisions) | ✓ **f32 payloads only**, int32/int64 indices, indices must come from `load` (not computed) | ✗ deferred |
| `f64` | ✓ | ✓ | ✗ **hard-rejected** (Apple Silicon lacks FP64 ALUs) |

Rejections are clean `unsupported_feature` diagnostics at compile time, not silent
fallbacks (`reject_unsupported_hip_ops` / `reject_unsupported_metal_ops`). Reduce-window
build also rejects runtime-symbolic windowed axes and bf16/f16 (cast to f32 first).

---

## 7. Type system — `spec/04-type-system.md`

- **Precisions:** floats `f32`, `f64`, `bf16`, `f16`; ints `int8`, `int16`, `int32`,
  `int64`; `bool`; `string`. (`f8e4m3` reserved, not admitted. No unsigned ints.)
- **Literal defaults:** integer literals → `int32`, float literals → `f32`.
- **No implicit precision promotion** — widening requires an explicit `cast`.
- **Named dimensions match by name**; symbolic dims (`batch`, `seq`) for runtime-varying
  axes, concrete dims for fixed architecture. Wildcard `*` for length-poly elements.
- **No broadcasting, ever** — operands' dims must match; use `expand`/`reshape`
  explicitly. (Masks fatal dimension errors in generated code — a design rule, not a gap.)
- **Linearity (use-once)** with borrow types: read-only primitive params are `&tensor`,
  auto-borrowed at call sites; `grad` consumes its targets (defensive `copy()` at grad
  sites). `realize` and explicit `drop` consume owned params.
- **Rank polymorphism** (`..r`): Tier-2 shape-identity bodies and Tier-3 named-axis
  reductions (`sum`/`mean`) — see [`spec/design/rank_polymorphism.md`](../spec/design/rank_polymorphism.md); owning issue chelis#258. Body discipline limits which builtins are admissible (`shape_class` in `builtins.rs`).

---

## 8. Effects — `crates/chelis-effects`

| Effect | Introduced by | Handled by |
|---|---|---|
| `Random` | `dropout`, `uniform_like` | `with seed(Ni64) { ... }` |
| `Io` | file ops, `mmap_*`, `process_run`, `print` | root / runtime |
| `Test` | `test_assert*` | pinned at root, no handler |
| `Accum` | accumulation contexts | — |
| `Resource(String)` | device/resource pinning | `with device("gpu:0"|"cpu") { ... }` |

Effects are inferred and checked after types, before lowering. The style gate and
`chelis check` report effect rows per function.

---

## 9. Transformations — `spec/06-transformations.md`

IR-level transforms a user applies (AD is an IR transform, not a library):

| Transform | Semantics |
|---|---|
| `grad` | Reverse-mode AD; returns gradients (per-arg tuple `.0`,`.1`,…); `wrt=(...)` restricts. Scalar-returning, pure-tensor-op target. |
| `vmap` | Vectorize over a named batch dimension (adds a batch axis through the DAG). |
| `jit` | JIT compile + cache. |
| `realize` | Force materialization of a (lazy) tensor; a consuming operation. |

`grad`/`vmap` over a body reaching a host-only builder (`tensor_scan`) are rejected at
the transform boundary (reachability-scoped).

---

## 10. `chelis` CLI surface — `crates/chelis-cli`

| Command | Purpose | Style gate? |
|---|---|---|
| `build` | Compile to C/HIP/Metal (`--target {c\|hip\|metal}`, default `c`) | yes |
| `check` | Type/effect/linearity front-end (`--show-inferred`) | yes |
| `validate` | Syntax validation (`--surf`/`--deep`/`--desugar`) | file subject |
| `eval` | Evaluate expr or `--file` (`--json`) | yes (file) |
| `fmt` | Canonical formatter (`--check`, `--inplace`) | gate subject |
| `lint` | Naming/style rules (`--check` for CI) — `spec/01-nomenclature.md` | gate subject |
| `cost` | Report lowered-IR copy cost (`--json`) | no |
| `deep` | Desugar Surf → Deep s-expr (`--annotate`) | no |
| `surf` | Decompile Deep → Surf (best-effort) | no |
| `prove` | `@property` checker (`--tier {auto\|fuzz-only\|smt-only\|type-only}`, `--samples`, `--seed`, `--smt-timeout`) | no |
| `test` | Run `tests/` Chelis-native tests (`--filter`, `--json`, `--jobs`) | no |
| `tide` | REPL / HTTP API / MCP / LSP server (`serve`, `mcp`, `lsp`) | no |
| `cove` | Terminal UI (`--file`) | no |
| `reef` | Package manager (`init`, `build`, `publish`, `install`) | no |

The style gate (`fmt --check` + blocking `lint`) runs inside `build`, `check`,
`validate`, and `eval --file`. Bypass with `--allow-style-violations` (never in CI) or
`CHELIS_STYLE_GATE_DISABLE=1` (test corpus only).

---

## 11. `chelis-std` module surface — `packages/chelis-std`

Bundled with the compiler (cannot be bumped independently of the chelis pin). The ML
modules (`Nn`/`Loss`/`Optim`/`Schedule`) were **cut from std to the shell layer** in
chelis-std 0.4.0 — there is no upstream NN fallback. Use these; do not reimplement:

| Module | Key exports |
|---|---|
| `Std.Tensor.Construct` | `linspace`, `arange`, `stack`, `squeeze`, `unsqueeze` |
| `Std.Tensor.Mask` | `where_indices` |
| `Std.Init.{Random,Xavier,Kaiming,XavierExt}` | `normal_like`, `kaiming_*`, `xavier_*`, `trunc_normal` (seeded, Box-Muller) |
| `Std.Sort` | `sort_1d`, `sort_2d` |
| `Std.Scan` | `scan_list` (list lane; tensor lane is the `tensor_scan` builtin) |
| `Std.Index` | `list_index`, `take_list`, `drop_list` |
| `Std.Io.{Csv,Json,Parquet,Safetensors}` | `read_csv`, `save_tensors`/`load_tensors`, … |
| `Std.Test` | `assert_*`, `assert_close*`, `assert_shape`, `fail` |
| `Std.Time`, `Std.Decimal`, `Std.Tokenizer`, `Std.Process`, `Std.Contracts` | dates, fixed-point, tokenization, `run`/`run_chelis`, contract predicates |

`Std.Tensor.Reduce` was removed in chelis#333: its `min`/`prod`/`argmax`/`argmin`
were bodyless sigs taking a *runtime* `int32` axis, but the `*_reduce` builtins they
would forward to require a *compile-time-constant* axis, so they were unimplementable
as declared and never had a runtime function. Call the builtins directly with a const
axis instead: `min_reduce(x, cast(1, int32))`, `prod_reduce`, `argmax_reduce`,
`argmin_reduce`.

---

## 12. `@property` / `prove` — `spec/design/chelis_property_spec.md`

`@property` declarations desugar to a `bool`-returning `def` tagged
`chelis_role: "property"`, with typed binders and optional `where` preconditions and
`with tolerance|seed|samples|contract` metadata. `chelis prove` discharges them in
tiers:

- **Tier A (fuzz):** deterministic sampling from binder types; precondition filtering;
  reports the first counterexample.
- **Tier B (SMT):** cvc5; lowers arithmetic invariants/contracts to SMT-LIB (the
  `CVC5_LOWERABLE` subset). Properties over host-lane ops are **not** SMT-provable and
  fall back to Tier A.
- **Tier C:** contract-driven SMT assumptions + obligation discharge.

`--tier auto` tries A→B→C. The 0.9.0 release added the prove honesty/discharge layer
and verdict taxonomy (`DisprovedModuloRealArithmetic` etc.) — the proofs are explicit
about their own strength. SMT/GPU codegen reach only the DAG lane; the host C runtime
is a tested-not-proven trusted base.

---

## 13. Where to read more

| Doc | Why |
|---|---|
| [`spec/05-risc-primitives.md`](../spec/05-risc-primitives.md) | the §1–2 tables digest this; AD adjoints |
| [`spec/04-type-system.md`](../spec/04-type-system.md) | named dims, precision, accumulators, linearity |
| [`spec/02-surf-syntax.md`](../spec/02-surf-syntax.md) | the language you write |
| [`spec/06-transformations.md`](../spec/06-transformations.md) | `grad`/`vmap`/`jit`/`realize` |
| [`spec/08-backends.md`](../spec/08-backends.md) | per-target coverage and reject lists |
| [`spec/design/differentiable_language.md`](../spec/design/differentiable_language.md) | D1–D5 AD roadmap — what NOT to design around |
| [`spec/design/rank_polymorphism.md`](../spec/design/rank_polymorphism.md) | `..r` polymorphism |
| [`spec/design/implicit_linearity.md`](../spec/design/implicit_linearity.md) | why `copy()`/`drop()` exist |
| [`spec/design/chelis_canonical_reference.md`](../spec/design/chelis_canonical_reference.md) | core vs std vs shell scope taxonomy |
| [`spec/design/shell_repo_contract.md`](../spec/design/shell_repo_contract.md) | what a downstream shell's `CHELIS_SURFACE.md` view must carry |
