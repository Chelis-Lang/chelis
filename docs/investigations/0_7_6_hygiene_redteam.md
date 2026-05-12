# 0.7.6 Toolchain Hygiene Red Team Report

**Date:** 2026-05-12
**Branch under review:** `fan-out-fix` (22 commits ahead of `main`)
**Reviewer role:** adversarial; instructed to find problems, not validate.
**Binary tested:** `target/debug/chelis` (reports version `0.7.6`).

## Executive Summary

Five surfaces of the 0.7.6 hygiene workstream exhibit spec/implementation
gaps that ship as silently incorrect or loudly inconsistent behavior in the
`chelis eval` and `chelis build --target c` paths:

1. **`jit` pass-through (#40) only fixed lowering, not the host evaluator
   or the C-backend runtime.** `chelis eval --file` with `jit(...)` either
   rejects with `host runtime does not support jit` or, when the body is
   tensor-typed and routes through IR, the emitted C binary segfaults at
   run time.
2. **`par` sequential semantics (#40) is regressed for tensor children.**
   `par { f32-lit; f32-lit; f32-lit }` at top level evaluates to the last
   value, but `par { to_tensor(...); to_tensor(...); to_tensor(...) }`
   rejects with `host runtime does not support par`, and the C-backend
   build for the analogous fn-wrapped form segfaults at run time.
3. **`prefer-pipe-operator` autofix produces non-canonical output.** For
   any input nested-call chain of three or more stages, `chelis lint --fix`
   emits `x |> f |> g |> h` on one line; the formatter rejects it on the
   very next `chelis fmt --check`, so the autofix breaks the style gate it
   was meant to satisfy. The underlying root cause is that
   **`chelis fmt` itself is non-idempotent** for >=3-stage pipes: pass 1
   produces a multi-line braced form, pass 2 collapses it to a different
   one-line shape.
4. **Bare-keyword pipe stage `x |> copy` parses but fails type-check** with
   `copy requires tensor input, got ?<freshvar>`, even when the pipe input
   is statically tensor-typed. The brief lists `x |> copy` as one of the
   supported new shapes; the parser fix (PR #35) did not extend the type
   checker.
5. **`chelis eval --file` silent-no-output bug (G7) is unfixed on this
   branch.** A `def`-only Surf input exits 0 with no stdout and no stderr.
   The fix commit (`2c2bba6`) exists on a separate branch
   (`fix/cli-eval-warn-on-empty-roots`) and was never merged into
   `fan-out-fix`.

CHANGELOG honesty has degraded: the 0.7.6 entry still says
`chelis lint --fix` "no longer rewrites `redundant-linearity-call` or
`prefer-pipe-operator`," but both autofixes were re-enabled by PRs #34 and
#42 on this branch. PR #40 added no CHANGELOG entry for jit/par.

A name-collision class bug in `chelis build --target c` was found that is
out of strict 0.7.6 scope but worth recording: any source defining
`def main(...)` produces uncompilable C because of clash with the runtime
`int main(void)` entry. The `examples/hello_tensor.ch` `def main(...)`
shape is shielded only by parity tests labelling it "library-only" and
never linking the binary.

## Methodology

- Spec read: `spec/01-nomenclature.md` §1.4-1.5, §6.1-6.3, §8.6, §11.1;
  `spec/03-deep-syntax.md` §2, §5, §6 (canonical form); `spec/02-surf-syntax.md`
  pipe + type syntax; `spec/04-type-system.md` t-fn shape;
  `spec/06-transformations.md` §4 (jit); `spec/07-concurrency.md` (par);
  `spec/design/implicit_linearity.md`. Implementer-perspective files under
  `docs/investigations/` were intentionally not consulted.
- Code paths inspected:
  - `crates/chelis-deep/src/validate.rs` (`VALID_TAGS`, 61-entry vocabulary)
  - `crates/chelis-lint/src/rules/deep_user_symbol_charset.rs` (`CLOSED_TAGS`)
  - `crates/chelis-lint/src/rules/redundant_linearity_call.rs`
  - `crates/chelis-lint/src/rules/prefer_pipe_operator.rs`
  - `crates/chelis-ir/src/lower.rs` (`lower_par`, `lower_jit`, `top_level_lowering_map`)
  - `crates/chelis-compiler-api/src/runtime.rs` (`eval_list` tag dispatch)
  - `crates/chelis-types/src/infer.rs` (copy/realize type-check arms)
  - `crates/chelis-cli/src/main.rs` (`cmd_eval`, `run_eval_emit`)
  - `crates/chelis-surf/src/parser.rs` (`parse_par`, pipe-stage routing)
- Binary built once via `cargo build -p chelis-cli --bin chelis`. All
  command-line probes use absolute paths to that binary. Fresh test
  fixtures live under `/tmp/redteam/` (throwaway; some referenced below).
- Test workflow: write `.ch` source; run `chelis fmt --inplace`; run
  `chelis eval --file` (and where relevant `chelis build --target c` +
  `gcc` + run the binary); compare actual output to spec-required output.

## Findings

Ordered by severity. Each section gives the test source, command, expected
result, actual result, and severity.

### Finding 1 (HIGH): `jit(...)` rejected by host evaluator and segfaults under C-build

**Test source** (`/tmp/redteam/jit_only.ch`):

```chelis
def compute(x: tensor[3, f32]) -> tensor[3, f32] = jit(mul(x, to_tensor([2.0, 2.0, 2.0])))

a = to_tensor([1.5, 2.7, -0.3])
jit_result = compute(a)
```

**Command and actual output:**

```
$ chelis eval --file /tmp/redteam/jit_only.ch
error: host runtime does not support `jit`
EXIT=1

$ chelis check /tmp/redteam/jit_only.ch
{ "score": 1, ..., "errors": [] }
EXIT=0

$ chelis build --target c /tmp/redteam/build_jit.ch
Wrote ./build_jit.c and ./build_jit.h ...
$ gcc -O2 ... ./build_jit.c ... -o ./build_jit
GCC-EXIT=0
$ ./build_jit
Segmentation fault (core dumped)
RUN-EXIT=139
```

**Expected:** `spec/06-transformations.md` §4 says `jit(f)(x) = f(x)` for
all `x`; `spec/03-deep-syntax.md` §2.7 marks jit as a compilation trigger
that is semantically a no-op. The host evaluator should accept and pass
through the body; the C backend should emit code that runs without
crashing.

**Actual:** PR #40 added `lower_jit` and the type-inference arm, but
neither `crates/chelis-compiler-api/src/runtime.rs::eval_list` nor the
C-backend tensor-realization path was updated. `eval_list` falls through
to the catch-all `host runtime does not support` error. Tensor-typed
jit bodies route through IR lowering (because the top-level def's body
type makes it lowerable) and produce a C binary that runs but segfaults.

**Severity:** HIGH. The fix is partial. `chelis check` cleanly accepts
`jit(...)` so a user has every reason to believe the program is valid;
both downstream paths (`eval`, `build --target c`) then fail in distinct
ways. The C-backend runtime crash is the most dangerous because it
appears to compile cleanly.

---

### Finding 2 (HIGH): `par` sequential semantics rejects tensor children and segfaults under C-build

**Test source A** — top-level f32 par works:

```chelis
result = par { 1.0; 2.0; 3.0 }
```

```
$ chelis eval --file ...
tensor(shape=[], data=[3.0])
EXIT=0
```

**Test source B** — top-level tensor par rejects:

```chelis
result = par { to_tensor([1.0, 1.0, 1.0]); to_tensor([2.0, 2.0, 2.0]); to_tensor([3.0, 3.0, 3.0]) }
```

```
$ chelis eval --file ...
error: host runtime does not support `par`
EXIT=1
```

**Test source C** — par-in-function returns wrong value:

```chelis
def compute() -> f32 = par { 1.0; 2.0; 3.0 }

result = compute()
```

```
$ chelis eval --file ...
tensor(shape=[], data=[0.0])
EXIT=0
```

(Spec says par returns the last expression's value, which is `3.0`.)

**Test source D** — tensor par-in-fn segfaults under C build:

```chelis
def compute() -> tensor[3, f32] = par { to_tensor([1.0, 1.0, 1.0]); to_tensor([2.0, 2.0, 2.0]); to_tensor([3.0, 3.0, 3.0]) }

result = compute()
```

```
$ chelis build --target c ... && gcc ... && ./build_par_t
Segmentation fault (core dumped)
RUN-EXIT=139
```

**Expected:** `spec/03-deep-syntax.md` §2.3 / §5: `par` is sequential in
v1; the value of the par is the value of its last child. Empty par or
last-child-eval failure is the only path that should diverge from a tuple
of children.

**Actual:** Same root cause as Finding 1 — PR #40 added `lower_par` but
not the host evaluator arm. `par` falls through to the `host runtime does
not support` error for any code path that doesn't route through IR
lowering, and the IR-lowered path itself produces the `0.0` fallback when
the par expression is reached inside a fn body (the `lower_par` skip-2
iteration sees no children in that lowering context). The C build
segfaults at runtime, identically to jit.

**Severity:** HIGH. Same shape as Finding 1.

---

### Finding 3 (HIGH): `chelis fmt` is non-idempotent for >=3-stage pipes, breaking the `prefer-pipe-operator` autofix

**Test source** (`/tmp/redteam/autofix_3pipe.ch`):

```chelis
def f(x: tensor[3, f32]) -> tensor[3, f32] = neg(x)
def composed(x: tensor[3, f32]) -> tensor[3, f32] = f(f(f(x)))

a = to_tensor([1.5, 2.7, -0.3])
result = composed(a)
```

**Command sequence:**

```
$ chelis lint --fix .../autofix_3pipe.ch
fixed 1 replacement(s) under .../autofix_3pipe.ch

$ chelis fmt --check .../autofix_3pipe.ch
error: ... is not canonically formatted
EXIT=1

$ chelis fmt .../autofix_3pipe.ch          # pass 1
def composed(...) -> ... = {
  x
  |> f
  |> f
  |> f
}

$ chelis fmt <pass1-output>                # pass 2
def composed(...) -> ... = { x
|> f
|> f
|> f }
```

(pass 3 is stable to pass 2.)

**Expected:**
- `spec/03-deep-syntax.md` §6: "Deep has exactly one textual representation
  per program." Surf's canonical form, while not literally Deep, is meant
  to be the unique formatter output for a given program.
- Autofix invariant from the brief: "when `chelis lint --fix` rewrites
  `f(g(x))` to `g(x) |> f`, the result must still parse, type-check, and
  evaluate identically." Implicit: the result must also pass the style
  gate; otherwise the user's clean source is replaced with a file that
  fails `chelis check`/`chelis build`.

**Actual:** Both the formatter and the autofix are buggy:
- The autofix emits a single-line pipe `x |> f |> f |> f`, which the
  formatter does not consider canonical.
- The formatter itself oscillates: input → multi-line braced → one-line
  braced-with-newlines → stable. `fmt --check` rejects the formatter's
  own output.

The same class of non-idempotency was reproduced with `match` inside
braces in `/tmp/redteam/match_eval.ch`.

**Severity:** HIGH. The autofix path leaves the user in a state where
`chelis lint --fix` was supposed to be a green operation but instead
breaks the style gate, requiring a follow-up `chelis fmt --inplace` to
recover. The formatter idempotency violation is a contract-invariant
break (§6 canonical-form spec).

---

### Finding 4 (MEDIUM): bare-keyword pipe stage `x |> copy` fails type check even with typed tensor input

**Test source** (`/tmp/redteam/pipe_copy_simple.ch`):

```chelis
def main(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy

a = to_tensor([1.5, 2.7, -0.3])
result = main(a)
```

**Command and actual output:**

```
$ chelis eval --file ...
error: Type errors: [CheckError { kind: TypeMismatch,
  message: "copy requires tensor input, got ?283",
  suggestions: ["Wrap only tensor values in copy"], ... }]
EXIT=1
```

In contrast, `def main(x: tensor[3, f32]) -> tensor[3, f32] = copy(x)`
evaluates cleanly. And `def main(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize`
evaluates cleanly. The parser fix in PR #35 ("parser accepts bare keyword
tokens as pipe stages") routes `x |> copy` through a synthesized lambda
`fn (__chelis_pipe) -> copy(__chelis_pipe)`. The type inferrer's `copy`
arm in `crates/chelis-types/src/infer.rs:4523-4554` calls `subst.apply`
on the lambda parameter and gets an unresolved type variable
(`?283`), then matches the `_ =>` arm and emits the "copy requires
tensor input" error.

**Expected:** The brief lists `x |> copy` as a supported new shape on
par with `x |> realize`.

**Actual:** Only `x |> realize` survives the type checker because
`realize`'s inference arm is permissive (returns the inner type
unconditionally); `copy`'s arm rejects unless `subst.apply` yields a
concrete tensor type, which it does not when the operand is a fresh
lambda parameter.

**Severity:** MEDIUM. The user sees a loud rejection (so this is not
silent miscompile), but the rejection message references an internal
type variable name and the brief documents this shape as fixed.

---

### Finding 5 (MEDIUM): `chelis eval --file` decls-only files exit 0 silently; G7 fix never merged

**Test source A** (`/tmp/redteam/decls_only.ch`):

```chelis
def helper(x: f32) -> f32 = mul(x, 2.0)
def other(x: f32) -> f32 = add(x, 1.0)
```

**Test source B** — same shape inside a reef package
(`/tmp/redteam/reef_test/src/lib.ch`):

```chelis
module Rt.Lib
def helper(x: f32) -> f32 = mul(x, 2.0)
```

```
$ chelis eval --file .../decls_only.ch  1>out.log  2>err.log
EXIT=0
$ cat out.log   # empty
$ cat err.log   # empty
```

**Expected** (from brief): "Should emit a stderr warning, exit 0. Verify
both behaviors. Verify also in reef-package context."

**Actual:** Both paths exit 0 with no stdout and no stderr. The pinned
regression test in `crates/chelis-cli/tests/cli.rs` (added by
`5128031 test: pin chelis eval --file silently succeeds on def-only
programs`) is still gated `#[ignore]`. The fix commit `2c2bba6 fix: warn
on stderr when chelis eval --file finds no evaluable roots` exists on
branch `fix/cli-eval-warn-on-empty-roots` but was not merged into the
0.7.6 hygiene branch.

**Severity:** MEDIUM. Spec-vs-impl gap with a surfaced symptom (silence),
not a wrong-answer bug. The brief assumed the fix landed; it did not.

---

### Finding 6 (MEDIUM): `def main(...)` collides with C-backend runtime entry

**Test source** (`/tmp/redteam/use_main.ch`):

```chelis
def main(x: tensor[3, f32]) -> tensor[3, f32] = x |> add(to_tensor([1.0, 1.0, 1.0]))

a = to_tensor([1.5, 2.7, -0.3])
result = main(a)
```

**Command and actual output:**

```
$ chelis build --target c .../use_main.ch
Wrote ./use_main.c and ./use_main.h ...
$ gcc -O2 ... ./use_main.c ... -o ./use_main
./use_main.c: In function 'main':
./use_main.c:238:25: error: too many arguments to function 'main';
   expected 0, have 1
   __binding_1_value = main(__call_arg0_6);
./use_main.c:209:5: note: declared here
   int main(void) { ...
```

**Expected:** Either the C emitter renames user-defined `main` to avoid
collision with the runtime entry, or `def main` is documented as
reserved.

**Actual:** No rename and no diagnostic. The C output emits both a user
`main` (called from the runtime entry) and the runtime `int main(void)`;
gcc rejects with the call-arity mismatch. This is technically out of
strict 0.7.6 scope (not introduced by this workstream), but it is
reachable from the `eval`-vs-`build` parity work the brief requested,
and `examples/hello_tensor.ch` actively uses `def main(...)`; the
existing parity test (`parity_hello_tensor_library_only`) shields the
example by labelling it library-only and only running the build to an
object file, never linking.

**Severity:** MEDIUM. Affects any user who follows the example-corpus
naming convention and tries to actually run the C output.

---

### Finding 7 (LOW): CHANGELOG entry for 0.7.6 contradicts current branch state

**File:** `CHANGELOG.md` lines 9-17.

The 0.7.6 entry says:

> `chelis lint --fix` no longer rewrites `redundant-linearity-call` or
> `prefer-pipe-operator` warnings. Those rules remain visible as
> warnings, but their source rewrites are disabled until the fixer can
> prove that removing explicit linearity calls or converting nested
> calls to pipes preserves type, ownership, and call-argument behavior.

But on the current branch (`fan-out-fix`):
- PR #34 (`4982ce3`) re-enabled `redundant-linearity-call` autofix.
- PR #42 (`29347c8`) re-enabled `prefer-pipe-operator` autofix.

Neither PR updated the CHANGELOG entry. PR #40 (jit/par) also did not
add a CHANGELOG entry.

**Severity:** LOW. Documentation drift; visible to anyone consulting
the changelog to understand what 0.7.6 actually ships.

---

## Negative-Result Probes

These surfaces were probed and behaved correctly; the tests below pass.
They are listed so a follow-on reviewer knows what was covered.

### Pipe-stage lowering (positive cases)

| Shape | File | Result |
|---|---|---|
| `x \|> relu`, `x \|> sigmoid` | `/tmp/redteam/pipe_bare4.ch` | matches `relu(x)` / `sigmoid(x)` element-wise on `[-0.3, 1.5, 2.7]` |
| `x \|> add(to_tensor([1.0,1.0,1.0]))` | `/tmp/redteam/pipe_first_arg.ch` | `[2.5, 3.7, 0.7]` |
| `x \|> grad(loss)` | `/tmp/redteam/pipe_grad.ch` | `[3.0, 5.4, -0.6]` (=2x for `sum(x*x)`) |
| `xs \|> vmap(grad(loss))` | `/tmp/redteam/pipe_vmap_grad.ch` | per-row 2x: `[[3.0,5.4,-0.6],[4.0,-2.0,1.0]]` |
| `x \|> f`, `x \|> f \|> f` with fn-typed param | `/tmp/redteam/pipe_fn_param.ch` | matches `f(x)` / `f(f(x))` |
| `outer(doubler, x)` where `outer(f,x)=f(f(x))` | `/tmp/redteam/inlining_nested.ch` | `[6.0, 10.8, -1.2]` (=`doubler(doubler(x))`, NOT one-level inlining) |
| `x \|> realize` | `/tmp/redteam/pipe_realize.ch` | identity on tensor |

### Lint rules

- `deep-user-symbol-charset` accepts canonical compound tags (`t-ref`,
  `t-tensor`, `t-fn`, `t-adt`, `t-prim`, `t-var`, `d-lit`, `d-name`,
  `pat-ctor`, `arm`, `deftype`, `variant`, `kv`, `bind`, `params`,
  etc.) and rejects user-defined hyphenated names (`my-bad-name`)
  emitted into Deep. Files tested: `/tmp/redteam/borrow_types.dp`
  (clean), `/tmp/redteam/adts.dp` (clean), `/tmp/redteam/dims_test.dp`
  (clean), `/tmp/redteam/bad_user_symbol.dp` (rejected as expected).
- `module-pascal-components` accepts `Chelis`, `Nautilus`, `Coral`,
  `Shoals`, `Octant`, `Capstone`, `CEarchin`; rejects `aBC`,
  `lowercase`, `SCREAMING`.
- `no-em-dash-in-public-strings` fires on `"hello—world"`, passes on a
  `#` comment line containing the same em-dash.

### Implicit linearity (var-RHS let-binding fan-out)

- `alias = x; mul(x, alias)` evaluates to `x*x` (`[2.25, 7.29, 0.09]`
  for input `[1.5, 2.7, -0.3]`). File:
  `/tmp/redteam/lin_var_rhs.ch`.
- AD parity: `grad(loss)` with implicit-alias body produces the same
  `[3.0, 5.4, -0.6]` as the explicit-`copy()` form. File:
  `/tmp/redteam/lin_var_rhs_grad.ch`.
- 3-level alias chain still produces correct grad of `x^4` (i.e. `4x^3
  = [13.5, 78.732, -0.108]`). File: `/tmp/redteam/lin_alias_chain.ch`.

### Loud rejections preserved (no-regression checks)

- `let g = grad(loss); g(a)` rejects with "grad is not supported by IR
  evaluation yet". File: `/tmp/redteam/grad_apply.ch`.
- `let vg = vmap(grad(loss)); vg(a)` rejects with "vmap is not
  supported by IR evaluation yet". File: `/tmp/redteam/vmap_grad_fc.ch`.
- True self-recursion `def loop(x) = loop(x)` is loudly rejected at
  type-check with `CycleDetected: def loop is trivially
  non-terminating`. File: `/tmp/redteam/true_recurse.ch`.
- Existing examples (`examples/*.ch`, `examples/illustrative/*.ch`) are
  fmt-idempotent. The non-idempotency in Finding 3 was reproduced only
  with synthesized adversarial fixtures.

### `redundant-linearity-call` autofix semantic parity

For the corpus in `/tmp/redteam/lint_corpus/c01..c06_*.ch`, the
`copy()` stripping preserves eval output. Each program was eval'd
before `lint --fix`, again after, and the outputs match byte-for-byte
for c01-c03 (which contained `copy()` to strip). c04-c06 trigger only
`prefer-pipe-operator` autofix which has Finding 3.

## Recommended Follow-Ons

Issues that should be tracked as their own work items, not folded into
this fix wave:

1. **Wire jit/par into the host runtime** (`runtime.rs::eval_list`) and
   into whatever C-backend runtime currently dereferences the resulting
   node. Likely scope: add `Some("jit") => self.eval_expr(child)` and
   `Some("par") => last child eval`. Confirm the segfaulting C path
   actually emits valid C for an identity transform; spec/06 §4.3 says
   jit inserts a "compilation boundary" — the segfault suggests the
   C-backend tensor-allocation path is reading from an uninitialized or
   freed pointer after the transform-skip.
2. **Fix `chelis fmt` idempotency for 3+ stage pipes and for `match`
   inside braces.** The canonical-form invariant is load-bearing for
   `lint --fix` to remain a useful operation (and is named as such in
   §6 of the spec).
3. **Either fix `x |> copy` type inference or remove `copy` from the
   bare-keyword pipe-stage allowlist.** The brief explicitly listed it
   as supported; the parser accepts it; only the type-check arm
   rejects, producing a confusing message that references an internal
   type variable.
4. **Land the G7 silent-no-output fix on `main`** (commit `2c2bba6`,
   currently sitting on `fix/cli-eval-warn-on-empty-roots`) or update
   the pinned `#[ignore]` test to match the actual ship state.
5. **Reserve or alias `def main`** in the C backend, and add an example-
   corpus test that actually links `examples/hello_tensor.ch` so this
   surface is exercised end-to-end.
