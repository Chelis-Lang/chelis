# 0.7.6 Toolchain Hygiene Red-Team Pass v2

**Status:** Adversarial validation report.
**Tested HEAD:** `21c6386b068565e4ef7d686ab6317f74650fd5ce` (`origin/main`,
after PRs #52-#56).
**Methodology:** ran `chelis fmt`, `chelis eval --file`, `chelis check`,
`chelis build --target c`, and `chelis lint` directly against the release
binary. Compiled C-backend output with `gcc -O2 -march=native -fopenmp`
and ran the produced binary to confirm runtime values, not just compile
success.

## Executive Summary

Two HIGH-severity gaps remain on main after the v1 fix sweep, plus one
MEDIUM autofix regression and one MEDIUM evaluator-vs-backend
divergence. None of the four originally-filed v1 issues regressed; jit /
par pass-through, fmt idempotency, `x |> copy` typing, and prefer-pipe
autofix output all hold up under broader probing. The remaining issues
are pre-existing surfaces that the hygiene sweep did not touch but are
on the same control plane.

| ID | Severity | Surface | One-line |
| --- | --- | --- | --- |
| V2-F1 | HIGH | runtime evaluator | zero-arity scalar `def go() -> f32 = 7.5; y = go()` silently returns `0.0` |
| V2-F2 | HIGH | runtime evaluator vs C-backend | `cast(tensor[D, p], q)` perfect-score in `check`, runs in C-backend, hard error in `eval` |
| V2-F3 | MEDIUM | prefer-pipe-operator autofix | warning fires on `add(mul(x,x), x)` shape but autofix never rewrites; infinite-warning loop |
| V2-F4 | MEDIUM | implicit linearity | `y = x; a = f(y); b = f(x)` rejects with `UseAfterConsume` instead of inserting Copy at consuming fan-out |

## Findings

### V2-F1 — Zero-arity scalar def returning a literal evaluates to 0.0 (HIGH)

**Surface:** `crates/chelis-compiler-api/src/runtime.rs` — runtime
evaluator's app/call handling for zero-arity definitions.

**Reproducer:**

```chelis
def go() -> f32 = 7.5
y = go()
```

```
$ chelis fmt --inplace probe.ch    # accepted
$ chelis check probe.ch            # score 1.0, no errors
$ chelis eval --file probe.ch
tensor(shape=[], data=[0.0])
```

The bug is observed for every primitive return type tried:

| body | `y = go()` result |
| --- | --- |
| `def go() -> f32 = 7.5` | `0.0` |
| `def go() -> int32 = 42` | `0.0` |
| `def go() -> bool = true` | `0.0` (numeric) |
| `def go() -> f32 = 0.0` | `0.0` (accidentally correct) |
| `def go() -> f32 = -0.3` | `-0.3` (unary expr, not a literal) |
| `def go() -> f32 = add(1.0, 2.7)` | `3.7` (correct) |

Control: when the call appears as an argument the value is correct.

```chelis
def go() -> f32 = 7.5
def use_it(x: f32) -> f32 = mul(x, 2.0)
y = use_it(go())   ;; evaluates to 15  (so go() did return 7.5 here)
```

And the C backend embeds `chelis_fill_f32(t0, 7.50000000f)` correctly;
the bug is `eval`-only, but the program is silently miscompiled there.
This is the same severity class as the original jit/par finding: the
evaluator returns a clean tensor and exits 0 with a wrong number.

**Hypothesis:** the let-binding `y = go()` lowers the RHS through a path
that allocates a default-zero output for the zero-arity callee and
forgets to bind the body's literal expression. The `use_it(go())` path
goes through a different lowering branch that does correctly evaluate
the inner literal.

**Fix sketch:** the runtime evaluator's app-of-zero-arity-def case
should evaluate the def body in an empty scope and return its value,
not just allocate the result slot. The simplest test fixture is a
single-line file `def go() -> f32 = 7.5\ny = go()\n` asserting `y ==
7.5`.

### V2-F2 — `cast(tensor[D, p], q)` is silently miscompiled (HIGH)

**Surface:** `crates/chelis-compiler-api/src/runtime.rs:1107-1128`
(`eval_cast`).

**Reproducer:**

```chelis
x = to_tensor([1.5, 2.7, -0.3])
y = cast(x, f32)
```

```
$ chelis check probe.ch
{ "score": 1, "errors": [] }
$ chelis build probe.ch --target c --output out
... (succeeds, runs, prints x and y correctly)
$ chelis eval --file probe.ch
error: unsupported cast from Tensor(...)
```

`spec/04-type-system.md §4.3` and §5.2 list `cast(x: tensor[D, p],
new_p) : tensor[D, new_p]` as a first-class supported form. `eval_cast`
hard-coded match only handles `RuntimeValue::Int`, `Float`, `Bool`, and
`String`; the `Tensor` and `Tuple` arms fall into the catch-all `Err`.

This is an evaluator-vs-backend agreement failure. `chelis check`
reports score `1.0` for a program that the evaluator immediately
rejects. Same precision casts (`f32 -> f32`) and widening casts (`f32
-> f64`) and narrowing-to-int (`f32 -> int32`) all hit the same arm and
all fail.

This isn't a regression from the hygiene workstream but it's the same
class of bug (silent perfect-score from `check`, runtime divergence
between `eval` and `build`). The first red-team didn't catch it.

**Fix sketch:** add `RuntimeValue::Tensor(_)` arms that produce the
correct precision-tagged tensor, preserving shape. Lock with a check /
eval / build cross-test similar to the jit/par fixture in
`crates/chelis-cli/tests/jit_par_runtime_gap.rs`.

### V2-F3 — `prefer-pipe-operator` flags shapes it cannot autofix (MEDIUM)

**Surface:** `crates/chelis-lint/src/rules/prefer_pipe_operator*` (the
rule and its autofix).

**Reproducer (minimal):**

```chelis
x = to_tensor([1.5, 2.7, -0.3])
y = add(mul(x, x), x)
```

```
$ chelis lint --rule prefer-pipe-operator --fix probe.ch
warning: probe.ch:2:5: prefer-pipe-operator (§3.6): nested first-argument call chain can be written with `|>`
$ chelis lint --rule prefer-pipe-operator --fix probe.ch
warning: probe.ch:2:5: prefer-pipe-operator (§3.6): nested first-argument call chain can be written with `|>`
$ # file unchanged across N passes
```

The lint fires on every pass; the autofix performs zero rewrites; CI
hooks that fail on advisory warnings (or developers that re-run `--fix`
expecting convergence) see a permanently-noisy file. Also occurs on
`sum(mul(x, x), 0)` (this exact shape is in
`examples/illustrative/linear_regression.ch`-style code and in the
v2-probe corpus's `def sq(x) = sum(mul(x, x), 0)`).

The autofix DOES correctly handle other shapes:

- `z = relu(add(x, a))` (single chain, no fan-out): fixed on pass 1
- `w = mul(a, relu(add(x, a)))` (chain in second-arg position): fixed
  on pass 1
- `w = mul(relu(add(x, a)), a)` (chain in first-arg position, outer
  call has additional args): **flagged, never fixed** (same class as
  the minimal repro)

**Hypothesis:** the rule treats "the LHS of an outer call is a nested
chain" as an autofix candidate, but the autofix has a guard that
suppresses rewriting when the outer call has additional siblings that
would have to move (because rewriting `add(mul(x,x), x)` to `mul(x,x)
|> add(x)` requires the outer `add` to be applied in
first-arg-insertion form, which works in spec but the autofix isn't
emitting it). The rule fires regardless of whether the fix is
emittable.

PR #55 fixed the output-correctness of the rewrites that DO happen.
The remaining gap is that the rule's trigger predicate is wider than
the autofix's emit predicate, so the warning sticks even after `--fix`.

**Fix sketch:** narrow `prefer-pipe-operator`'s detection to the
shapes the autofix actually rewrites, OR emit the missing rewrite for
the "outer call has additional args" case. Either keeps `--fix`
convergent.

### V2-F4 — Cross-statement var-RHS fan-out rejected instead of auto-copied (MEDIUM)

**Surface:** implicit-linearity inserter for var-RHS let bindings.

**Reproducer:**

```chelis
x = to_tensor([1.5, 2.7, -0.3])
y = x
a = mul(y, to_tensor([2.0, 2.0, 2.0]))
b = mul(x, to_tensor([3.0, 3.0, 3.0]))
```

```
$ chelis eval --file probe.ch
error: variable `x` was already consumed by use at offset 0; later use at offset 0 is invalid
$ chelis check probe.ch
{ "score": 0.8, "errors": [{"kind":"UseAfterConsume",...}] }
```

Per `spec/design/implicit_linearity.md`:

> The compiler inserts `Copy` for source-level consuming fan-out: a
> value used in more than one non-borrow consuming position. Copies are
> inserted for all consume sites before the final consume site; the
> final consume site takes the original.

`x` has two consuming uses (transitively through `y`, and directly in
`b`). The spec says compiler **should** insert a Copy on the first use.
Workaround `y = copy(x)` makes the program compile and eval correctly.

Direct-fan-out in the same call site (`mul(x, x)`) is correctly
auto-copied. The gap is cross-statement, var-RHS-aliased fan-out.

The brief lists this surface explicitly (Surface 1 — "implicit-linearity
copy insertion for var-RHS let-bindings + cross-statement consume
fan-out"). I count it as a real gap, not "out of scope": rejected
without an actionable user message for spec-conformant source.

The closest pre-existing tracker is `Linearity-F1`/`Linearity-F2` per
PR #30; the cross-statement var-RHS specific case may already be in
scope there.

**Fix sketch:** when checking var-RHS let bindings, walk forward and
collect all transitive consume sites of the bound name and the alias;
insert `Copy` at the first consume of either alias when the count is
>1. Tests: the minimal repro above; also a 3-alias chain (`y = x; z =
y; a = mul(z, ...); b = mul(x, ...)`) which should also auto-copy.

## Negative-Result Probes (clean surfaces)

The following surfaces were probed with non-trivial inputs and held
green; no findings filed.

- **jit / par runtime evaluator (PR #56).** All four shapes from
  `jit_par_runtime_gap.rs` plus jit-wrapping-par, par-of-jits, jit
  inside fn body, par inside fn body. Eval and C-backend both return
  the inner-or-last value with non-trivial tensor data
  `[1.5, 2.7, -0.3]`.
- **fmt idempotency on pipes (PR #53).** 3-, 5-stage pipe chains,
  pipes nested inside block bodies, pipes mixed inside non-pipe calls
  (`mul(x |> relu |> add(a), a)`). Full `examples/` corpus formatted
  twice in two parallel trees and `diff -r` was empty.
- **`x |> copy` typing (PR #54).** Tensor with statically-known dim
  passes; auto-copy followed by a consuming `mul` evaluates correctly.
- **prefer-pipe-operator autofix correctness (PR #55).** Every output
  that DID get rewritten (in V2-F3 above and elsewhere) passes `fmt
  --check` and the resulting program evaluates correctly.
- **CHANGELOG (PR #52).** Present and dated; not validated for prose
  completeness.
- **CLI silent-no-output (PR #50).** `def`-only programs emit the
  expected stderr warning and exit 0.
- **Pipe stages.** `x |> relu`, `x |> add(a)`, `x |> grad(f)`,
  `xs |> vmap(grad(f))`, `def f(g, x) = x |> g`, `x |> realize`,
  `x |> copy`, `x |> tensor_to_scalar` all evaluate and C-build
  correctly with `[1.5, 2.7, -0.3]`-style input.
- **Lints:** `deep-user-symbol-charset` accepts canonical tags
  (`pat-var`, `t-fn`, `t-prim`) and rejects non-vocab names like
  `pat-name`. `module-pascal-components` accepts `nautilus`, `coral`,
  `shoals`, `octant`, `capstone`. `no-em-dash-in-public-strings` fires
  on em-dashes inside string literals and not on em-dashes in
  comments.
- **Inlining recursion guard (PR #39).** `outer(doubler, x) =
  doubler(doubler(x))` evaluates to 6.0 for `doubler(x) = add(x, x)`
  and `x = 1.5`. True self-recursion (`def loop_rec(x) = loop_rec(x)`)
  errors loudly with `CycleDetected` and a usable suggestion.
- **Sibling rejections.** First-class `grad` outside an `app`
  position emits a clean stderr message ("`grad` is not supported by
  IR evaluation yet; use `chelis build --target c` instead") and exits
  1.

## Recommended Follow-ons

1. **File V2-F1 as a blocking bug.** Zero-arity scalar def silent
   miscompile is the same severity class as the original jit/par C
   miscompile. Add an `eval` /  `build c` cross-test for every
   primitive return type (f32, f64, int32, int64, bool).
2. **File V2-F2 alongside.** Eval lacks a `Tensor` arm in `eval_cast`.
   Lock with `eval` / `build c` agreement test for f32->f32,
   f32->f64, f32->int32 on a `tensor[3, f32]`.
3. **Tighten `prefer-pipe-operator` trigger or extend its
   autofix.** Either fix is acceptable. Lock with a "lint --fix
   converges in one pass" invariant on the example corpus.
4. **V2-F4: surface var-RHS implicit-copy gap.** If this is already
   tracked under `Linearity-F1`/`F2`, add the cross-statement
   var-RHS-aliased reproducer to its fixture; if not, file as new.

## Methodology Note

The brief required testing against `origin/main` HEAD. Verified
`HEAD == origin/main == 21c6386b068565e4ef7d686ab6317f74650fd5ce` before
any probe ran. All `eval`, `build --target c`, `fmt`, `check`, and
`lint` invocations went through `target/release/chelis` built from
that HEAD. Where possible, every claim is backed by both the
runtime-evaluator output AND the compile-and-run output of the
generated C program (so the silent-zeros vs working-binary class is
not assumed away).
