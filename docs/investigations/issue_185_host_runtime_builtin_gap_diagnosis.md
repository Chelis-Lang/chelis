# Issue #185: BUILTIN_NAMES-vs-`eval_builtin` host-runtime gap

## Symptom

Several names appear in `BUILTIN_NAMES`
(`crates/chelis-types/src/builtins.rs:14-148`) so `chelis check` accepts
them, but the host-runtime evaluator
(`crates/chelis-compiler-api/src/runtime.rs::eval_builtin`, the match
arms inside the function spanning roughly lines 1488-2497) has no
dispatch arm. Calling such a builtin through `chelis test` / `chelis
eval` falls into the function's fallthrough arm and surfaces

```
"unsupported builtin `<name>` in host runtime"
```

For the bool logical ops (`and` / `or` / `not`) the arms exist but only
accept scalar bool inputs; tensor-bool inputs fall into the scalar-only
helpers' error path

```
"bool op expects bool arg, got Tensor(..)"
```

## Missing set (pre-flight at base `df9a894`)

Computed by diffing `BUILTIN_NAMES` against arm names extracted from
`eval_builtin`'s match. 14 names total:

| Group | Builtins | Spec |
|---|---|---|
| A — Unary RISC | `abs`, `cos`, `tan`, `floor`, `ceil`, `atan` | §2.2 |
| B — Reductions | `max_reduce` | §2.2 |
| C — Binary Tier 2 | `max_elem`, `min_elem` | §3.4 |
| E — Composed Tier 2 | `mean`, `layer_norm`, `conv2d` | §3.4, §4.4, §4.5 |
| F — Tensor-bool | `and`, `or`, `not` (tensor arms only) | §3.2 |
| Allowlist | `normalize` (pending-spec-stability), `scatter_replace` (lowered-before-eval) | — |

The original issue lists 12 missing; sibling sweep at pre-flight HEAD
adds `atan`, `min_elem`, `mean`, `conv2d` (which had been in the
planning enumeration but were worth re-verifying at branch base). `pad`
was on the original list but landed as part of PR #214 (issue #187) and
is now correctly dispatched.

## Delegation strategy

The IR evaluator at `crates/chelis-ir/src/eval.rs` is the canonical
oracle for numerical correctness (per
`feedback_evaluator_byte_identical_gate`: backend dtype-add acceptance
oracles must include byte-identical / in-tolerance agreement vs
evaluator; self-consistency is insufficient). The host-runtime arms
therefore delegate to the evaluator's existing math rather than
reimplementing per-builtin semantics:

- **Group A**: use the existing `float_unop_with_tensor(args, f64_op,
  f32_op)` helper that handles scalar + tensor inputs and routes
  tensors through `tensor_float_unop_f32` (the same f32-rounding
  profile the C backend emits via `cosf`/`tanf`/etc.). `abs`,
  `floor`, `ceil`, `atan` follow the same pattern with their libm
  counterparts.

- **Group B**: extend the existing `ReduceOp` enum with a `Max`
  variant, threaded through `tensor_reduce_host` symmetrically with
  the existing `Min` arm.

- **Group C**: element-wise binary ops via `numeric_binop` with
  `f64::max` / `f64::min` closures, matching the IR evaluator's
  `binary_map(.., f64::max)` for `RiscOp::MaxElem` and its
  derived-Tier-2 cousin `MinElem`.

- **Group E**: build a small ad-hoc DAG (`Load` for each tensor input
  + the appropriate `tier2::lower_mean` / `lower_layer_norm` /
  `lower_conv2d` decomposition), then forward-eval it through
  `eval_tensor_with`. This delegates to the canonical IR-lowering
  decomposition rather than re-deriving the math in two places.

- **Group F**: separate tensor arms (`bool_tensor_binop` /
  `bool_tensor_unop`) that operate on bool-precision tensors directly
  and produce bool-precision tensor results. Per the brief's pinned
  decision: do NOT broaden the scalar arms with tensor branches —
  the tensor-bool case needs explicit shape-checking that scalar
  broadcasting hides.

## Allowlist policy

The invariant test `builtin_dispatch_invariant.rs` enforces that every
`BUILTIN_NAMES` entry has either (i) an arm in `eval_builtin` or
(ii) an entry in the documented allowlist. Allowlist reasons are drawn
from the closed vocabulary:

- `type-only` — exists only for typing / signature lookups
- `target=<backend>-only` — only emitted on a specific backend target
- `lowered-before-eval` — IR lowering converts the call to a RISC op
  before host eval runs
- `pending-spec-stability` — name reserved but semantics not yet
  pinned

Pre-existing entries:

- `scatter_replace` — `lowered-before-eval`: the sparse-IR Surf-facing
  name always lowers to `RiscOp::Scatter` via
  `chelis-ir/src/lower.rs:lower_scatter_replace_uses_sparse_ir_node`.
  The host runtime never sees the symbolic builtin call; the DAG
  evaluator handles it directly.
- `normalize` — `pending-spec-stability`: marked unstable in
  `crates/chelis-types/src/builtins.rs:55-56`. Follow-up issue tracks
  semantics. Per `feedback_no_partial_impl_during_escalation`: don't
  ship a placeholder; track separately and exclude.

## Commit plan

This investigation pinned the gap as a 3-commit PR per the brief:

1. `test(compiler-api): pin failing fixtures for chelis#185
   host-runtime gap` — failing positive + negative tests for every
   group, plus the standing invariant.
2. `docs(compiler-api): diagnosis paragraph for chelis#185
   host-runtime gap` — this document.
3. `fix(compiler-api): implement host-runtime arms for chelis#185
   groups A-F` — the actual delegation arms.
