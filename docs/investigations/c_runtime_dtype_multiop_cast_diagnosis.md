# C runtime multi-op cast composition: PR-3-surfaced bug no-repro on current main

## Context

`CRuntime-F32Coupling` workstream W2 PR 4 (0.7.8 compiler cleanup). PR 3 (W2
Agent B, merged as #87) reported a DAG-side f64 corruption surfaced while
migrating host_emit code generation:

> Separately, I verified that an unrelated DAG-side f64 corruption exists in
> `chelis build --target c` output for `add(cast(t, f64), cast(t, f64))`
> programs (result tensor renders as garbage like `134217762.5625`). This bug
> is pre-existing on `origin/main` (reproduced before applying my changes via
> `git stash`) and is the same `CRuntime-F32Coupling` class manifesting in a
> different code path; it is not introduced by this PR and is out of scope.

PR 4's first task per the dispatch brief: reproduce on current `main`
(post-PR #87 merge), characterize, and either fix in-scope or recommend a §5
follow-on with an `#[ignore]`d fixture.

## Reproduction attempt on current main (commit `b539429`)

Tested four variants of the bug shape against `chelis eval` and
`chelis build --target c` on `main` at `b539429` (PR #87 merged):

1. `def composed(x: tensor[3, f32]) -> tensor[3, f64] = { a = cast(x, f64); b = cast(x, f64); add(a, b) }` with input `[1.5, 2.5, 3.5]`.
   * `chelis eval`: `result = tensor(shape=[3], data=[3.0, 5.0, 7.0])`
   * gcc-compiled C: `3 5 7`
   * Agreement: byte-exact.

2. `mul(cast(x, f64), cast(x, f64))` with input `[1.5, 2.5, 3.5]`.
   * Eval: `[2.25, 6.25, 12.25]`
   * C-backend: `2.25 6.25 12.25`
   * Agreement: byte-exact.

3. Top-level binding form: `t = to_tensor([1.0, 2.0, 3.0]); a = cast(t, f64); b = cast(t, f64); result = add(a, b)`.
   * Eval: `result = tensor(shape=[3], data=[2.0, 4.0, 6.0])`
   * C-backend: same.

4. Larger inputs (`100.5, 200.5, 300.5`) and mixed-precision compositions:
   all byte-exact agreement.

No reproduction.

## Root cause analysis: the bug is closed

Inspecting the generated C for the test programs shows the cast-then-add
chain is emitted through DAG-specialized kernels, not the host_emit
elementwise helpers. The emitted code already uses typed pointer accesses:

```c
((double*)t3->data)[i] = ((double*)t1->data)[idx_a] + ((double*)t2->data)[idx_b];
```

Re-running the same probes against commit `be04ddf` (the parent of PR #87,
pre-host_emit migration) shows the same typed-pointer emission. The
DAG-specialized binary elementwise path was already typed-pointer-aware
before PR #87 landed.

The host_emit `assign_tensor_binary_elementwise` is the fallback for
non-DAG-specialized programs. PR #87's host_emit dispatch tests at
`crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs` cover that
fallback via direct `HostProgram` construction. The 60-fixture matrix at
`crates/chelis-e2e/tests/dtype_op_matrix.rs` covers the runtime accessor
sites.

PR 3's agent reported the bug after running `git stash` on their own staged
changes. Their PR was the host_emit migration (PR #87) and the stash put
host_emit back to the pre-migration state. The bug they observed and the bug
they fixed were therefore the same one. Once their changes landed as #87,
the bug class was closed.

## Disposition

**No-repro on current main. PR #87 closed the bug class.** No fix needed
in W2 PR 4.

PR 4 adds 17 multi-op composition fixtures to
`crates/chelis-e2e/tests/dtype_op_matrix.rs` and 5 cast+arithmetic
composition fixtures at
`crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs`. The
matrix fixtures exercise runtime accessor chains (concat then gather,
where then einsum, cumsum then clamp, etc.); the cbackend fixtures
exercise the full `chelis build --target c` plus gcc compile plus run
chain for cast-mixed-with-arithmetic programs. Both lock the byte-exact
f64 mantissa invariant across the compositions reported by PR 3's agent.

No §5 follow-on is needed for this disposition; the bug class is already
captured by the existing `CRuntime-F32Coupling` entry which is closed by
PRs #84, #86, and #87.

## Cross-references

* PR #87 commit message and diff: `crates/chelis-backend-c/src/host_emit.rs`.
* W2 PR 3 dispatch fixtures: `crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs`.
* Cross-validation matrix: `crates/chelis-e2e/tests/dtype_op_matrix.rs`.
* Multi-op composition fixtures added by PR 4 (this PR): same matrix file
  plus `crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs`.
* Surface-bug regression locks: `crates/chelis-cli/tests/cbackend_cast_memcpy.rs`,
  `cbackend_reshape_memcpy.rs`, `cbackend_print_tensor_f64.rs`.
