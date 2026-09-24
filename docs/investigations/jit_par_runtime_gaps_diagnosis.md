# `jit` and `par` runtime-evaluator and C-backend gaps diagnosis

Diagnosis pass for the bundled fix on branch `fix/jit-par-runtime-arms`.
Findings 1+2 of the 0.7.6 toolchain hygiene red-team (PR #51).

Current status: chelis#2388 found that the host `par` arm still erases
effectful non-final children. The checker now fences `par` under chelis#2503;
the implementation record below is historical and is not an acceptance claim.

Cross-references:

- IR-side fix that PR #40 landed:
  `docs/investigations/jit_par_spec_impl_mismatch_diagnosis.md`.
- IR-side pinning tests already on `main`:
  `crates/chelis-types/tests/jit_par_passthrough.rs` (`#[test]
  jit_wrapping_a_value_type_checks_and_carries_inner_type`,
  `par_is_rejected_with_the_typed_issue_fence`).
- Failing-test pins for this branch:
  `crates/chelis-cli/tests/jit_par_runtime_gap.rs` (commit 1 of this
  branch).

No code changes in this commit.

## Spec sections (quoted, do not edit)

`spec/03-deep-syntax.md` §2.3 (Expressions table):

```
| `par` | `(par {} expr₁ expr₂ ...)` | Parallel evaluation (v1: sequential) |
```

`spec/03-deep-syntax.md` §2.7 (Transforms table):

```
| `jit` | `(jit {} expr)` | Compilation trigger |
```

`jit` is semantically a no-op at evaluation (the JIT effect, if any,
lives in metadata, not in the computed value). `par` v1 promises
sequential composition: evaluate each child in order, return the value
of the last child.

## What PR #40 did

PR #40 ("fix: jit pass-through and par sequential semantics per spec
(Agent 1, D+E)", merged commit `9e9e36e`) extended the IR layer:

1. `crates/chelis-types/src/infer.rs`: removed the `jit` / `par`
   rejection from `validate_ir_expr`; added a `jit` arm in `infer_expr`
   that returns the inner type, mirroring the existing `par` arm.
2. `crates/chelis-ir/src/lower.rs`:
   - Dropped `par` and `jit` from `assert_ir_lowerable`'s reject set.
   - Added `lower_jit` (pass-through on the inner expression).
   - Replaced `lower_par`'s body to lower each child in order and
     return the `LoweredValue` of the last child.
3. Added `regression_tests::jit_is_passthrough_at_lowering` and
   `par_is_sequential_at_lowering` in `lower.rs`.

This is enough to make a top-level scalar `result: f32 = jit(1.5)` and
`result: f32 = par {1.0; 2.0; 3.0}` work via `chelis eval --file`: the
def classifies as "lowered" (DAG-evaluable) by `top_level_lowering_map`
and the DAG eval path handles the lowered nodes.

## What PR #40 did NOT do

PR #40 did **not** extend two adjacent layers that also need to know
about `jit` and `par`:

### Gap A — Runtime evaluator (`eval_list` dispatch)

File: `crates/chelis-compiler-api/src/runtime.rs`.

The host evaluator's `EvalContext::eval_list` is the dispatch point
that turns a Deep `(<tag> {} ...)` form into a `RuntimeValue`. It has
explicit arms for every supported tag and a catch-all that errors with
`host runtime does not support \`<tag>\``:

```rust
// crates/chelis-compiler-api/src/runtime.rs (around line 564)
other => Err(format!(
    "host runtime does not support `{}`",
    other.unwrap_or("?")
)),
```

The arms in scope today (as of `main` at `c392811`) include `lit`,
`var`, `app`, `if`, `let`, `tuple`, `copy`, `borrow`, `record`,
`access`, `tuple-get`, `match`, `fn`, `pipe`, `cast`, `realize`,
`grad`, `vmap`, and `handle-effect`. There is no `jit` arm and no
`par` arm.

Whether `eval_list` gets called for a given def depends on
`top_level_lowering_map` (`crates/chelis-ir/src/lower.rs`), which
classifies each top-level def as either "lowered" (DAG path, no host
eval) or "host" (host runtime). The classification is driven by
`expr_requires_host_runtime`, which marks a body as host if it
contains `to_tensor`, `print`, `to_list`, certain match/record/access
shapes, or other host-only primitives.

So:

- Scalar body, no host-forcing primitives: DAG path. `lower_jit` /
  `lower_par` from PR #40 handle it. Works.
- Tensor body via `to_tensor`: host path. `eval_list` falls through to
  the catch-all. Errors `host runtime does not support 'jit'/'par'`.

Empirical reproduction on `main`:

```
$ cat tensor_jit.ch
result = jit(to_tensor([1.5, 2.5, 3.5]))
$ chelis eval --file tensor_jit.ch
error: host runtime does not support `jit`

$ cat tensor_par.ch
result = par {
  to_tensor([1.0, 2.0]);
  to_tensor([3.0, 4.0, 5.0])
}
$ chelis eval --file tensor_par.ch
error: host runtime does not support `par`
```

### Gap B — C backend host-lane lowerer

File: `crates/chelis-ir/src/host.rs`.

The host-lane lowerer `lower_host_expr_kind` translates Deep expressions
into the `HostExpr` IR used by `chelis-backend-c::host_emit`. Its
dispatch is a giant match over `tag(list)`. It covers `tuple`, `record`,
`lit`, `var`, `if`, `match`, `let`, `tuple-get`, `access`, `cast`,
`app`, `pipe`, `handle-effect`, `copy`, and `realize`. There is no
`jit` arm and no `par` arm. The catch-all is silent:

```rust
// crates/chelis-ir/src/host.rs (around line 3326)
_ => HostExpr::new(HostExprKind::Unit),
```

So a host-lane `jit(<inner>)` or `par {<children>}` lowers to
`HostExpr::Unit`, and the emitted C assigns the binding to `0` and
prints `result = ()`. The red-team brief reported this as a segfault;
empirically it is silent data loss (the binary exits 0 with `result =
()`).

Empirical reproduction:

```
$ chelis build --target c tensor_jit.ch
$ gcc -O2 -fopenmp tensor_jit.c -L. -lchelis_runtime -lm -lpthread -ldl -o tensor_jit
$ ./tensor_jit
result = ()
```

The relevant C snippet:

```c
int main(void) {
    int __binding_0_value;
    __binding_0_value = 0;
    int result = __binding_0_value;
    printf("%s = ", "result");
    printf("()");
    printf("\n");
    return 0;
}
```

## "par-in-fn returns 0.0" — separate pre-existing bug

The red-team brief mentions a sub-bug:

> `par`-in-fn returns `0.0` instead of the last value.

Empirical reproduction on `main`:

```
$ cat par_in_fn.ch
def go -> f32 = par {
  1.0;
  2.0;
  3.0
}
result: f32 = go()
$ chelis eval --file par_in_fn.ch
tensor(shape=[], data=[0.0])
```

But the same shape with a non-par body shows the same:

```
$ cat baseline.ch
def go -> f32 = 7.5
result = go()
$ chelis eval --file baseline.ch
tensor(shape=[], data=[0.0])
```

So this is a pre-existing scalar-fn-eval bug for `def go -> T = <lit>;
result = go()` — `result` evaluates to `0.0` independent of `go`'s
body. It is **not** a `par`-specific bug; the red-team finding
exposed it through a `par` body but the root cause is upstream of
`par`/`jit` and out of scope for this branch's bundle.

For the tensor-returning fn-scoped case, the eval simply errors out
(same Gap A as the top-level case):

```
$ cat par_in_fn_tensor.ch
def go() -> tensor[three, f32] = par {
  to_tensor([1.0, 2.0, 3.0]);
  to_tensor([4.0, 5.0, 6.0])
}
result = go()
$ chelis eval --file par_in_fn_tensor.ch
error: host runtime does not support `par`
```

So fixing Gap A (and Gap B) is the right call for the par-in-fn
tensor case; fixing the scalar `0.0` regression belongs in a separate
investigation. Tracked as a separate finding — not closing it on this
branch.

## Fix shape (for commit 3)

1. **Gap A** (host runtime evaluator,
   `crates/chelis-compiler-api/src/runtime.rs::eval_list`): add a
   `Some("jit")` arm that evaluates the body expression and returns
   its `RuntimeValue`, and a `Some("par")` arm that evaluates each
   child in order and returns the last child's `RuntimeValue`.
   This mirrors `lower_jit` and `lower_par` in IR.

2. **Gap B** (C backend host lane,
   `crates/chelis-ir/src/host.rs::lower_host_expr_kind`): add a `jit`
   arm that lowers its inner expression and returns the result, and a
   `par` arm that lowers each child in order and returns the last
   child's `HostExpr`. The intermediate children's `HostExpr` values
   are not threaded into a sequence node — `HostExpr` doesn't have one
   today and the spec-blessed v1 par semantics don't require
   side-effect preservation across non-IO children (every IO primitive
   has a Bucket-1 host arm). If a v1 par body contains a `realize` or
   `print`, those are first-class host-lane forms that get their own
   `HostExpr` nodes and get emitted on their own.

   Risk: dropping non-last children from the C emit could miss
   side-effecting primitives that aren't explicit IO. The conservative
   shape is to lower each child to a `HostExpr`, push them all into a
   block-like sequence, and bind the last as the value. If that
   requires a new `HostExpr` kind, that's a wider change and should be
   escalated to the orchestrator before this commit lands.

   For this branch, follow the same shape as `lower_realize`:
   pass-through to the inner expression for `jit`, and last-child
   for `par`. This is consistent with how the IR DAG lowering already
   behaves (`lower_par` adds every intermediate node to the DAG and
   returns the last) and is the simplest fix that closes Findings
   1+2 without expanding scope. Document the side-effects-across-par
   caveat in the runtime-arm doc comment.

3. **Flip the four ignored fixtures** in
   `crates/chelis-cli/tests/jit_par_runtime_gap.rs` to running.

4. **No spec change.** `spec/03-deep-syntax.md` §§2.3 and 2.7 are
   already correct; the work is making the runtime evaluator and C
   backend match what the spec says (and what IR DAG lowering already
   does).
