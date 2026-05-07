# Linearity-checker divergence: root-cause analysis

Branch: `worktree-agent-a9b871b6475810eee` (worktree off `main`).
Investigation HEAD parent: `bd27b2f` (Phase G' fix host-runtime + skip no-match
filter; defer worker wire-up + linearity).

## Reproduction

Investigation test: `crates/chelis-types/tests/rt_lin_div_diagnosis.rs`.

Run: `cargo test -p chelis-types --test rt_lin_div_diagnosis -- --nocapture`.

The test mirrors the chelis-std `test_linspace_endpoints` shape with a small
synthetic library (linspace returns `tensor[4, f32]`, `assert_shape` and
`assert_close_tensor` each take a tensor by value) and runs the same
library + new-code pair through six different chelis-types entry points.

Verdict matrix (single test, six probes):

| # | Probe | check_phase0e API used | Type env source | Linearity verdict |
|---|---|---|---|---|
| 1 | Path A (cache) | `_with_context` (new code only) | non-empty (`build_type_env_from_library`) | **REJECT** (use-after-consume on `actual`) |
| 2 | Path B (format-reparse) | monolithic on `format_program(library + new)` | n/a | **ACCEPT** |
| 3 | Combined no-format | monolithic over library + new deep | n/a | **ACCEPT** |
| 4 | Combined via empty ctx | `_with_context` on combined | `TypeEnv::empty()` | **ACCEPT** (falls back to `annotate_phase0e_program`) |
| 5 | Combined via non-empty ctx | `_with_context` on combined | `build_type_env_from_library` | **REJECT** |
| 6 | New only via non-empty ctx + monolithic linearity | `_with_context` for type, then `check_linearity` | non-empty | **REJECT** |

Confirms divergence reproduces minimally, and isolates the deciding
factor to **the `check_phase0e` path's prelude-ADT registration**, NOT
to format-reparse, NOT to the `check_linearity_with_context` walker
strategy.

### Pretty-printed annotated_exprs comparison

The same `test_linspace_endpoints` body (last `def` in each program)
prints differently between paths. Excerpt:

Path A new_checked.annotated_exprs[1] (rejects):
```
(let {}
  (bind {} actual
    (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}
      (var {} linspace) ...))
  (let {}
    (bind {} __chelis_tmp0
      (app {type: (t-prim {} int32)}
        (var {} assert_shape) (var {} actual) ...))
    (app {type: (t-prim {} int32)}
      (var {} assert_close_tensor) (var {} actual) (var {} expected) ...)))
```

Path B / Combined annotated_exprs[7] (accepts):
```
(let {}
  (bind {} actual
    (app {} (var {} linspace) ...))         ; <-- empty type metadata
  (let {}
    (bind {} __chelis_tmp0
      (app {} (var {} assert_shape) (var {} actual) ...))   ; <-- empty
    (app {} (var {} assert_close_tensor) (var {} actual) (var {} expected) ...)))
```

In Path A the `(app linspace ...)` carries
`{type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}` — so `actual` is
declared with a tensor type and the linearity checker correctly detects
linear consumption at the first call site.

In Path B / Combined the `(app linspace ...)` carries empty metadata;
`actual` is declared with no type; `expr_is_linear` returns false; no
consumption fires; the reuse passes silently.

## Diagnosis: Case 2 (with one structural-asymmetry kicker)

Per the decision tree, this is **Case 2 — format-reparse / monolithic
flow accidentally normalizes annotation in a way that hides a real
linearity violation.** The `_with_context` path is the more correct
answer.

### Mechanism

`crates/chelis-types/src/infer.rs:1960-2006` defines
`annotate_phase0e_program`, which is called from `check_phase0e_program`
(the monolithic flow Path B and combined_no_format use). It builds
state from `builtins::builtin_env() + AdtRegistry::new()` — i.e., a
**fresh, empty ADT registry**. The note at lines 1961-1969 explicitly
acknowledges the asymmetry with `infer_phase0e_program_with_env`:

> "Preserve historical behavior: build a fresh annotation state from
> `builtin_env()` + an EMPTY ADT registry (no prelude registration).
> This is asymmetric with `infer_phase0e_program_with_env` (which DOES
> register prelude ADTs)..."

`crates/chelis-types/src/context.rs:99-115` `TypeEnv::empty()`, in
contrast, calls `register_prelude_adts` on its inner ADT registry
before sealing the type-env. `build_type_env_from_library` (in
`infer.rs:136`) starts from `TypeEnv::empty()` and so the type-env it
returns has prelude ADTs registered. When that type-env is fed to
`check_phase0e_with_context` AND `library_def_count() > 0`, the
non-fallback branch
(`annotate_phase0e_program_with_context`, `infer.rs:2017-2065`) clones
the prelude-ADT-registered state and the resulting annotation
correctly resolves `Cons`/`Nil` in `to_tensor([...])`, propagating a
concrete `tensor[n, f32]` return type from `linspace`'s body all the
way back to the call site's `app` node.

The monolithic `check_phase0e_program` (and Path B's
`compile_source` → `check_phase0e_program`) does NOT register prelude
ADTs. The `linspace` body (`to_tensor([start, stop, ...])`) fails to
unify because `Cons` is unresolved. linspace's def emerges with return
type `(t-var {} _)`. The downstream `(app linspace ...)` call-site
inference falls through with no concrete type. Because the linearity
checker's `expr_is_linear` is type-directed
(`linearity.rs:578-581`), `actual` is treated as non-linear and
neither call consumes it.

This means the legacy Path B has been silently masking a real linearity
violation in the chelis-std test corpus for as long as `linspace` (and
similarly `to_tensor`-returning helpers whose bodies use prelude-ADT
list literals) have existed. The Path A cache path is the first
pipeline in this repo that actually exercises proper type inference on
new code referencing those helpers.

### Why Case 2 (not Case 1)

Case 1 (span loss) was the original Phase G' hypothesis. The error
message offset reads `offset 0` for both consume and reuse sites,
which superficially looks like span loss. But the offsets being
zero is just because `Span` is `Default::default` on
the macro-expanded sub-nodes — the linearity checker is correctly
identifying the SAME variable `actual` (by name) across both call
sites; it's not using span equality to determine binding identity. The
verdict difference does not come from spans. Confirming this:

- Both paths surface the same error message kind (`UseAfterConsume`)
  on Path A, with the same offsets.
- The annotated_expr structural diff above has nothing to do with
  spans — it's about the `{type: ...}` metadata entry on the `app`
  node.

So span loss is not driving the verdict. (There may be a separate
cosmetic span-loss issue worth addressing, but it is not the cause of
the linearity divergence.)

### Why Case 3 (let / underscore desugaring) is also ruled out

Both paths desugar `_ = expr` to a `let` binding to a fresh
`__chelis_tmp0`. The diff shows this is identical:

```
(let {} (bind {} __chelis_tmp0 (app ...)) <continuation>)
```

shape on both sides. Different desugaring is not the cause.

## Verification probe — `chelis check`

Per the decision-tree probe instructions for Case 2:

```
cd packages/chelis-std
chelis check tests/tensor/construct.ch
```

`chelis check` is single-file and does not resolve reef imports, so it
emits `UnboundVariable` errors on `linspace` / `assert_shape` / etc.,
without ever reaching linearity. So `chelis check` cannot produce a
direct yes/no on this question.

The closest equivalent that DOES exercise both paths is the harness
test in this investigation. Combined with the source reading of
`assert_shape`'s declaration —

```
def assert_shape[n](t: tensor[n, f32], expected_n: int64, label: string)
  -> unit ! { Test } = { ... }
```

— `t` is taken by value, NOT by borrow. The chelis linearity rule
unambiguously says calling `assert_shape(actual, ...)` consumes
`actual`. A subsequent `assert_close_tensor(actual, ...)` is then a
real use-after-consume. **Path A's REJECT is the linearity-correct
verdict; Path B's ACCEPT is masking a real bug.**

## Tests in chelis-std needing rewrite

Each call to `assert_shape(actual, ...)` followed by another
tensor-consuming call on `actual` is a real use-after-consume. Either
add `&` at the first call site (the `assert_shape` body uses `t` only
through the observational `shape(t, 0)` builtin, so `&t` is sound) or
wrap the first use with `copy(actual)`.

The pattern audit (grep results in this worktree) finds:

- `packages/chelis-std/tests/tensor/construct.ch:23` —
  `_ = assert_shape(actual, cast(1, int64), "linspace count=1 length")`
  followed by `assert_close_tensor(actual, expected, ...)` on line 24.
- `packages/chelis-std/tests/tensor/construct.ch:29` —
  `_ = assert_shape(actual, cast(3, int64), "linspace count=3 length")`
  followed by `assert_close_tensor(actual, expected, ...)` on line 30.
- `packages/chelis-std/tests/init/kaiming.ch:50` —
  `assert_shape(out, cast(8, int64), "...")` is a tail-position
  consuming use; if `out` is not used afterwards in that test, it is
  fine. Audit each call site for a follow-on tensor use.
- `packages/chelis-std/tests/init/kaiming.ch:54` — same pattern.
- `packages/chelis-std/tests/init/xavierext.ch:59,63,67` — same; verify.
- `packages/chelis-std/tests/nn/rmsnorm.ch:34` — same; verify.

For each `_ = assert_shape(t, ...); next_call(t, ...)` pair, the
recommended source fix is one of:

1. **Borrow the assert** (preferred): change to `_ = assert_shape(&t, ...)`.
   The body's only use of `t` is `shape(t, 0)` which is observational,
   so this is sound.
2. **Copy on the consuming use**: change to
   `_ = assert_shape(t, ...); next_call(copy(t), ...)`.
3. **Inline the assert away** if it isn't load-bearing.

Option 1 is the cleanest and preserves test intent without runtime
cost.

## Architectural note (informational, not the recommended fix)

There is a separate, real bug in
`crates/chelis-types/src/infer.rs:1960-2006`: `annotate_phase0e_program`
should register prelude ADTs. Fixing that would tighten the monolithic
flow to match `_with_context` on this and probably other corner cases.
Doing so is a cross-cutting change (the comment at 1961-1969 explicitly
warns that downstream tooling depends on the asymmetric annotation
shape — the fix would need to verify that `expr_type` readers,
`extend_root_names_from_value`, etc., still behave correctly). Even
with that fix, the chelis-std tests still need to be rewritten, because
the underlying linearity rule is correct and the tests really are
use-after-consume — fixing the type-inference bug just makes the
monolithic path stop hiding the violation.

The recommended sequence is therefore:
1. Rewrite the chelis-std tests (Option 3 of the original plan) —
   these are real linearity violations and would be caught by any
   linearity checker that has access to a correctly-typed AST.
2. THEN, separately, consider fixing `annotate_phase0e_program` to
   register prelude ADTs (and adjust whatever downstream tooling needs
   the asymmetric shape). This is correctness hygiene, not blocking.
3. Then wire `eval_in_context` into the `cmd_test` worker (Phase G'
   Scope 3). With chelis-std rewritten, the in-context path will
   accept, the cache will be live for `chelis test`, and the legacy
   `compile_with_reef_graph + format_program + prepare_eval` path can
   be removed or kept as a fallback.

## Branch + commit SHA

Branch: `worktree-agent-a9b871b6475810eee`.
Investigation commit: see commit titled `lin-rca: linearity divergence
root-cause analysis` on this branch. The commit body includes the
case classification.
