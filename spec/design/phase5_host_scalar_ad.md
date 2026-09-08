# Phase 5 — Host-lane scalar AD

**Status:** Implemented (chelis#405). Forward-mode dual numbers, per the
locked design below. `grad(f, wrt=...)(args)` over a scalar `f32 -> f32`
(or multi-scalar-param) top-level def now lowers on the host lane and builds
to C. The previous unresolved-transform-marker rejection (chelis#841: now the unspellable `HOST_UNRESOLVED_TRANSFORM_MARKER`) is retained for the cases
the dual transform does not cover (container `wrt`, unsupported scalar ops);
the tensor-lane reverse-mode AD path is untouched.

The dual transform now follows the full shape of the canonical driver:

- single-expression scalar bodies (the original push);
- `let`-block bodies (`{ d1 = ...; mul(s, d1) }`) — bindings are
  dual-evaluated in sequence and threaded through the environment;
- calls to user-defined scalar defs (`d1(...)`, `normal_cdf(...)`) — the
  callee's body is inlined into the dual tree, with the chain rule carried by
  the argument derivatives;
- the host scalar builtins listed in `dual_eval_app`
  (`add`/`sub`/`mul`/`div`/`neg`/`exp`/`log`/`sin`/`cos`/`tanh`/`sqrt`,
  constant-exponent `pow`, scalar `cast`).

This is enough to differentiate the Black-Scholes scalar Greeks named in the
issue (`delta`, `vega` over a `call_price` built from `d1`/`d2`/`normal_cdf`,
`let`-blocks, and `log`/`sqrt`/`exp`). A `(mutually) recursive scalar callee
fails closed at `MAX_DUAL_INLINE_DEPTH` — the transform returns `None` and the
build falls through to the same unresolved-transform-marker rejection rather than
looping.

Implementation: `try_lower_scalar_grad_app` / `dual_eval` / `dual_eval_let` /
`dual_eval_user_call` in `crates/chelis-ir/src/host.rs` (forward-mode dual
transform at host-IR lowering time, emitting parallel value/derivative trees
over the existing host scalar builtins — no new runtime struct or C builtin).
Oracle: `build_c_scalar_grad_builds_and_is_numerically_correct` in
`crates/chelis-cli/tests/cli.rs`, with multi-param, container-rejection,
block-body, user-call, recursion-fail-closed, and Black-Scholes-Greeks parity
tests beside it.

The original deferral rationale and design analysis are preserved below for
context.

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
  grad(target, wrt=theta_local)(theta)
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

## What does NOT work (current boundary)

The two sections above describe the original deferral. With the implementation
shipped, the canonical `grad(top_level_scalar_fn)(scalar_arg)` shape — and the
`let`-block / user-defined-call composition the Black-Scholes Greeks need —
now lower. The remaining gaps the dual transform does not cover (it returns
`None`, falling through to the unresolved-transform-marker rejection):

- `grad` where `wrt` is a host container (`List`/`dict`/ADT/tuple) parameter —
  the genuine container-AD escalation in step 5 below. Rejected, with a test.
- `grad` through host-lane combinators (`fold`, `map`, `filter`, `scan`) or
  control flow (`if`/`match`) in the differentiated body — the dual transform
  has no rules for these constructs.
- `grad` through a scalar op not in the `dual_eval_app` table, or a
  non-constant `pow` exponent (rejected to stay correct, see the `pow` arm).
- a (mutually) recursive scalar callee — bounded by `MAX_DUAL_INLINE_DEPTH`
  and rejected rather than inlined.

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
