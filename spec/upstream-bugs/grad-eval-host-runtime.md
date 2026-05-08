# grad-eval-host-runtime: `chelis test` host runtime does not support `grad`

**Status:** **CLOSED 2026-05-07** (Bucket 1 of compiler-vs-interpreter
closure campaign).
**Filed:** 2026-04-29
**Closed:** 2026-05-07 (commit `412fa61`)
**Owning phase:** chelis-core (AD), chelis-cli (test runner)
**Discovered by:** Phase 3l Shoals fix-up #2

## Closure summary (2026-05-07)

The host runtime now supports `grad`, `vmap`, and `realize` via
delegation to the C backend's lowering machinery. `chelis test` and
`chelis eval` route those forms through `lower_subexpr_program` +
`eval_tensor_roots_with_strict` (the same machinery `chelis build
--target c` uses), so all three execution lanes now agree on programs
that pass `chelis check`. All three syntactic forms are accepted:
`grad(f)(x)`, `let g = grad(f); g(x)`, and the wrapper-fn-param form.

Implementation lives in
`crates/chelis-compiler-api/src/runtime.rs::apply_transform`. The
Phase-5 forward-mode dual-numbers design at
`spec/design/phase5_host_scalar_ad.md` remains relevant only as a
future performance optimisation; it is no longer load-bearing for
correctness.

Regression coverage in `crates/chelis-cli/tests/cli.rs`:
`eval_realize_is_identity_in_host_runtime`,
`eval_grad_inline_form_in_host_runtime`,
`eval_grad_locally_bound_form_in_host_runtime`,
`eval_grad_wrapper_fn_param_form_in_host_runtime`,
`eval_vmap_in_host_runtime`, plus per-form host-vs-C parity probes.

## Original deferral rationale (historical, now resolved)

This bug surfaced the host-lane scalar AD gap. The host evaluator did
not have `grad` support because host-lane scalar AD was not yet
implemented. The architectural decision was locked at
`spec/design/phase5_host_scalar_ad.md` (recommendation: forward-mode
dual numbers; deferred until a real driver appears). The 2026-05-07
closure resolved this by reusing the C backend's autodiff lowering
inside the IR evaluator, sidestepping the need for a separate
host-lane AD implementation. The rest of this file documents the
original repro and impact for historical reference.

## Summary

Functions defined in terms of `grad(...)` type-check successfully under
`chelis check` but fail at runtime under `chelis test` with the error:

```
host runtime does not support `grad`
```

This blocks the Phase 3l acceptance criterion "Greeks via `grad` match
analytical Black-Scholes Greeks (< 1e-6 error)" from being executed by
the in-language test runner.

## Minimal repro

`reef.toml`:

```toml
[package]
name = "repro"
version = "0.0.1"
compiler = "=0.4.0"
module_prefix = "Repro"

[dependencies]
chelis-std = { version = "0.1.0" }
```

`src/lib.ch`:

```chelis
module Repro.Lib
export (call_total, deltas)
def call_total[n](spots: tensor[n, f32]) -> f32 = {
  squared = to_tensor(map(fn (s: f32) -> mul(s, s), to_list(copy(spots))))
  tensor_to_scalar(sum(squared, 0))
}
def deltas[n](spots: tensor[n, f32]) -> tensor[n, f32] = {
  obj = fn (sx: tensor[n, f32]) -> call_total(sx)
  grad(obj, wrt=sx)(spots)
}
```

`tests/grad.ch`:

```chelis
module Repro.Tests.Grad
import Std.Test (assert_close)
import Repro.Lib (deltas)
def test_grad_evaluates() -> unit ! { Test } = {
  spots = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
  ds = deltas(spots)
  d0 = index(to_list(ds), cast(0, int64))
  assert_close(d0, cast(2.0, f32), cast(0.001, f32), "d/dx0 of sum(x^2) at x0=1 is 2")
}
```

## Observed behavior

`chelis check src/lib.ch` → score 1.0, all four components green:

```
{
  "score": 1,
  "components": { "parse": 1, "structure": 1, "names": 1, "types": 1 },
  "typed_nodes": 42082,
  "untyped_nodes": 0,
  "total_nodes": 42082,
  "unresolved_names": [],
  "errors": []
}
```

`chelis test tests/` → exit code 1 with the canonical failure line:

```
tests/grad.ch
  test_grad_evaluates ........... FAIL (host runtime does not support `grad`)

0 passed, 1 failed
```

The error string `host runtime does not support \`grad\`` is emitted by
the `chelis test` runner when its host evaluator hits a `grad` IR node.

## Expected behavior

`chelis test` should evaluate `grad(f, wrt=x)(input)` and return a
tensor of partial derivatives matching what the C/HIP backend would
emit, so that AD-derived numerical assertions can run on the inner
test loop without dropping out to a manual gate.

For the repro above, the expected output of `deltas([1.0, 2.0, 3.0])`
is `[2.0, 4.0, 6.0]` (the gradient of `sum(x^2)` is `2*x`).

## Speculated cause categories

1. **AD compile-time transform produces an IR op that the host
   evaluator does not implement.** The transform may emit a
   `RiscOp::Grad` or similar that has a backend lowering but no
   evaluator branch. This would explain the disagreement between
   `chelis check` (which is an HM type-check that does not look
   inside `grad`) and `chelis test` (which actually invokes the
   evaluator).
2. **AD transform happens lazily and never fires under the host
   evaluator code path.** The grad node is left in the IR as a
   placeholder and the evaluator legitimately rejects it as
   unsupported. This matches the current error string most directly.
3. **`chelis test`-vs-`chelis eval` evaluation path mismatch.** It
   is possible that `chelis eval --file` would succeed where
   `chelis test` fails (or vice versa), if the two CLI surfaces
   route through different evaluator implementations. The repro
   above only exercises `chelis test`; the eval-side surface is not
   checked.

## Downstream impact

Shoals v0.1.0-alpha defines four `grad`-derived Greek functions in
`src/pricing.ch` (`deltas_call`, `deltas_put`, `vegas_call`, plus a
helper `call_total_per_sigma`). All of them compile cleanly under
`chelis check` (score 1.0) but cannot be exercised under
`chelis test`. Shoals's runtime Greeks tests use central
finite-difference approximations as a stand-in until this bug is
resolved.

When the upstream eval path lands, the property
`properties/pricing.ch::matches_textbook_reference` will swap from
finite-difference comparison to grad-vs-textbook comparison
automatically (one-line change to the property body), and the
chelis-cli phase oracle's `#[ignore = "blocked on grad eval"]` test
guard becomes a one-line removal.
