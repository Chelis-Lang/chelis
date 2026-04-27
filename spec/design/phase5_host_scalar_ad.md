# Phase 5 — Host-lane scalar AD (deferred)

**Status:** Deferred. No current downstream user needs this; the canonical
work-around (locally-bound fn over tensor input, returning a scalar) covers
every active use case. Revisit when a real customer or paper requires scalar
AD; do **not** start implementation without a focused design session.

## Background

`chelis build --target c` has a two-lane backend split:

- **Tensor lane** (`crates/chelis-ir/src/lower.rs`) — pure tensor expressions
  lower to a RISC DAG; the gradient transform in `chelis-deep` /
  `chelis-ir::grad` produces a backward DAG. AD is reverse-mode and operates
  over an immutable DAG of pure ops.
- **Host lane** (`crates/chelis-ir/src/host.rs`) — scalars, control flow,
  list/dict/tuple ops, ADTs, list combinators (`map`, `fold`, `filter`,
  `scan`, `partition`). Lowers to a `HostProgram` IR. **No AD transform
  exists for this lane.**

Top-level functions whose return type is a scalar `f32` (e.g.
`def square(x: f32) -> f32 = mul(x, x)`) land in the host lane, even though
their bodies use only arithmetic. `grad(square)` therefore has no lowering
target, and `cmd_build` rejects with a diagnostic that points users at the
supported tensor-lane pattern (the `host_program_unresolved_call_sites`
guard in `crates/chelis-cli/src/main.rs:903`).

## What works today

The supported `grad` shape is encoded by the regression test
`build_c_tensor_grad_local_wrapper_over_function_param_builds` in
`crates/chelis-cli/tests/cli.rs:1374`:

```chelis
def jac_row[n](
  model: tensor[n, f32] -> f32 -> f32 -> f32,
  theta: tensor[n, f32], x: f32, y: f32
) -> tensor[n, f32] = {
  target = fn (theta_local: tensor[n, f32]) -> model(theta_local, x, y)
  grad(target, wrt=(theta_local))(theta)
}
```

Constraints enforced by the lowering:

- Differentiated function is **locally bound** (a `fn` literal or function
  parameter), not a top-level def.
- Input type is **a tensor**, return type is a scalar — keeps the gradient
  computation entirely on the tensor lane.
- Body uses only pure tensor ops (`add`, `mul`, `einsum`, `sum`, etc.) — no
  `if`/`fold`/`map` over host-lane values.

This shape covers every active use case downstream:

- **Nautilus** — `lm_scalar_nparam`'s Jacobian assembly differentiates a
  tensor-parameterized model.
- **Shoals** (planned) — Greeks computed from pricing functions that are
  vectorized over strikes/paths and return tensors.
- **Coral** — frame-aggregation runtime; `grad` appears in docs as a
  documented future capability, not a runtime path.

## What does NOT work

- `grad(top_level_scalar_fn)(scalar_arg)` — the canonical
  `dsquare(x: f32) -> f32 = grad(square)(x)` shape.
- `grad(local_fn_with_mixed_params)` where `wrt` is a scalar parameter and
  the body uses host-lane operations.
- `grad` through host-lane combinators (`fold`, `map`, `filter`) on
  scalar/list/ADT values — the ad transform doesn't have rules for these.

## Design decision (locked when implementation starts)

**Recommendation: forward-mode dual numbers in the host runtime.**

The decision matters because the implementation cost differs by ~5×.

### Forward-mode

Each `f32` becomes a `(value: f32, derivative: f32)` struct at the runtime
level. Each scalar op gets a `_dual` variant in the host runtime
(`add_dual`, `mul_dual`, `exp_dual`, `sin_dual`, ...). The compiler
transforms `grad(f)` into a call to `f`'s dual version, with the seed
derivative set to 1.0 for the active `wrt` parameter and 0.0 for the rest.

**Cost estimate**: ~50 dual variants in the host runtime + a compiler
transform + tests. Likely 2–3 days for someone familiar with the AD codebase.

**Trade-offs**:
- One directional derivative per pass. Vector-input scalar-output (e.g.
  `f(x: tensor[100, f32]) -> f32`) needs 100 forward passes. Not the right
  fit for ML-style optimization, but **fine for the canonical scalar AD
  use case** (single-variable functions like `dsquare(x) = grad(square)(x)`).
- Composes cleanly with the existing tensor-lane reverse-mode AD because the
  two lanes don't interact at the AD level.
- No runtime tape; no replay; minimal new infrastructure.

### Reverse-mode (alternative)

A runtime tape data structure records every scalar op during forward
execution. After the forward pass, the tape is replayed in reverse with
adjoint propagation, producing one full gradient (one value per `wrt`
input) per call.

**Cost estimate**: tape data structure + tape-recording variants of every
scalar op + replay engine + memory management. Likely 5–7 days for someone
familiar with the AD codebase, plus design review for the tape ABI.

**Trade-offs**:
- Single pass produces all gradients (better for vector-input scalar-output).
- Invasive — every scalar op must push to a tape during forward execution.
  Even non-`grad` callers pay the bookkeeping cost unless the compiler
  emits two variants of every function (clean and tape-recording).
- Matches the tensor lane's design philosophy, but the tensor lane reverses
  over a static DAG, not a per-call tape; the tape pattern is a real
  departure.

### Hybrid (do not pick first)

Forward mode for ≤ N `wrt` parameters, reverse mode for more. Strictly more
implementation work than picking one. Don't pursue without a use case that
clearly needs both.

## Recommendation, when this comes up

Forward-mode. Single-variable scalar AD covers the canonical use cases
documented in chelis-std and the illustrative examples; multi-variable
gradients are best handled by lifting to the tensor lane (which is already
supported).

Implementation outline (not a commitment):

1. Define `chelis_dual_f32 { f32 value; f32 derivative; }` in the host
   runtime header.
2. Generate dual variants in the host runtime for every scalar op the
   compiler currently emits. Naming convention: `add_dual`, `mul_dual`,
   `exp_dual`, `sin_dual`, `cast_int32_to_f32_dual`, etc.
3. Add a compiler transform that walks the host program: when a function
   `f: f32 -> f32` is the operand of `grad`, emit a parallel `f_dual:
   chelis_dual_f32 -> chelis_dual_f32` derived from `f`'s body via
   per-op rewrite.
4. For multi-parameter scalar functions with `wrt=(p1, p2, ...)`, emit one
   dual call per `wrt` parameter; combine results into the gradient tuple.
5. Reject (with a clear diagnostic) calls where `wrt` includes a host-lane
   list/dict/ADT type — those need first-class container AD which is
   genuinely structural.
6. Test corpus: `dsquare(x) = grad(square)(x)`, mixed scalar + tensor
   parameters, nested local-fn `grad`, pure host-lane chains.

## What to NOT change in this push

- **Don't** weaken the `cmd_build` guard. The current rejection is correct;
  silently emitting broken C is worse than an explicit error pointing at
  the workaround.
- **Don't** add forward-mode plumbing speculatively. The decision is locked
  in this doc; the work starts when a real driver shows up.
- **Don't** promote scalar-AD examples in user-facing docs that suggest the
  pattern works. The transformations spec's existing examples
  (`f(x) = sum(x * x)` over `tensor[3, f32]`) are correct — they show the
  supported tensor-lane pattern. Keep new examples in that shape.

## References

- Working pattern test: `crates/chelis-cli/tests/cli.rs:1374`
  (`build_c_tensor_grad_local_wrapper_over_function_param_builds`).
- Rejection guard: `crates/chelis-cli/src/main.rs:903`.
- Unresolved-call detector: `crates/chelis-ir/src/host.rs:538`
  (`host_program_unresolved_call_sites`).
- Tensor-lane AD definition: `spec/06-transformations.md` §2.
- Coral's tracking entry: `/home/jeff/Documents/scratch/coral/docs/UPSTREAM_BUGS.md`
  ("Upstream blocker (v0.1.21, still present v0.2.0, still present v0.2.1):
  `grad` type-checks but fails to build").
