# Parser pipe bare-keyword pipe stage — diagnosis

Diagnoses Item 2b (G11) of the 0.7.6 toolchain hygiene workstream
(`/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`,
`docs/investigations/item2_sibling_sweep_findings.md` G11).

## Bug shape (confirmed empirically)

`crates/chelis-surf/src/parser.rs:1000-1003` drives the pipe loop:

```rust
while *self.peek() == TokenKind::Pipe {
    self.advance();
    let stage = self.parse_prefix()?;
    stages.push(stage);
}
```

`parse_prefix` (`parser.rs:1080-1190`) dispatches every reserved-keyword
token to a dedicated parser that **unconditionally** expects the
keyword's normal-form continuation:

| Token         | Parser fn                | Required next token | parser.rs line |
|---------------|--------------------------|---------------------|----------------|
| `Realize`     | `parse_realize`          | `LParen`            | 1510-1516      |
| `Copy`        | `parse_copy`             | `LParen`            | 1518-1524      |
| `Grad`        | `parse_grad`             | `LParen`            | 1415-1438      |
| `Vmap`        | `parse_vmap`             | `LParen`            | 1440-1500      |
| `Jit`         | `parse_jit`              | `LParen`            | 1502-1508      |
| `Cast`        | `parse_cast`             | `LParen`            | 1405-1413      |
| `Par`         | `parse_par`              | `LBrace`            | 1545-1583      |
| `If`          | `parse_if`               | full conditional    | varies         |
| `Match`       | `parse_match`            | full match form     | varies         |
| `Fn`          | `parse_lambda`           | full lambda form    | varies         |
| `With`        | `parse_with_handler`     | handler name        | 1526-1543      |

Empirical reproduction (verified, all `expected LParen` or similar
shape, byte 45 for the canonical fixture
`def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> <keyword>`):

```
realize-bare: ERR - expected LParen, found Eof
copy-bare:    ERR - expected LParen, found Eof
grad-bare:    ERR - expected LParen, found Eof
vmap-bare:    ERR - expected LParen, found Eof
cast-bare:    ERR - expected LParen, found Eof
jit-bare:     ERR - expected LParen, found Eof
fn-bare:      ERR - expected LParen, found Eof
par-bare:     ERR - expected LBrace, found Eof
with-bare:    ERR - expected identifier, found Eof
if-bare:      ERR - unexpected end of input
match-bare:   ERR - unexpected end of input
```

Control: `x |> relu` parses cleanly because `relu` is an `Ident` token,
not a reserved keyword.

## Spec semantics per keyword

Per `spec/01-nomenclature.md` §3.6, a bare pipe stage `x |> f` passes
the piped value as the only argument to `f`. The bare form is therefore
**semantically meaningful only when the keyword is a unary callable** —
i.e., when `f(x)` is a valid expression for the single piped value.

| Keyword     | Arity                                    | Bare `x \|> kw` meaningful? | Spec interpretation                              |
|-------------|------------------------------------------|------------------------------|---------------------------------------------------|
| `realize`   | unary builtin                            | **Yes**                      | `realize(x)`                                      |
| `copy`      | unary builtin                            | **Yes**                      | `copy(x)`                                         |
| `grad`      | takes a function, returns a function     | No                           | use arg form: `x \|> grad(f)` ≡ `grad(f)(x)`      |
| `vmap`      | takes a function (+ axis)                | No                           | use arg form: `x \|> vmap(f)` ≡ `vmap(f)(x)`      |
| `jit`       | takes a function                         | No                           | use arg form: `x \|> jit(f)` ≡ `jit(f)(x)`        |
| `cast`      | binary (value, precision)                | No                           | use arg form: `x \|> cast(f32)` ≡ `cast(x, f32)`  |
| `with`      | handler + body                           | No                           | no callable-form interpretation                   |
| `par`       | block of expressions                     | No                           | structural form, not a callable                   |
| `if`        | conditional                              | No                           | structural form                                   |
| `match`     | pattern match                            | No                           | structural form                                   |
| `fn`        | lambda literal                           | No                           | bare `fn` is the lambda intro keyword             |

The structural fix's surface is therefore the **two unary builtins**
(`realize`, `copy`). The remaining keywords' bare-form rejections are
preserved as the correct (if currently confusing) diagnostic — they have
no valid bare-pipe-stage interpretation per spec.

For completeness: the keywords whose **arg form** is meaningful (`grad`,
`vmap`, `jit`, `cast`) already work as pipe stages today via existing
lowering (Item 2 closed `grad`/`vmap` pipe stages; `jit`/`cast` follow the
same `Expr::Apply`-or-similar shape). Item 2b is specifically the bare
form.

## Canonical lowering for the fix

`x |> realize` lowers per spec §3.6 / spec `03-deep-syntax.md` §5 to:

```
(pipe {} <seed> (fn {} (params {} __chelis_pipe) (realize {} (var {} __chelis_pipe))))
```

This mirrors the desugar pattern at
`crates/chelis-surf/src/desugar.rs:511-531` for `Expr::Apply(func, args)`
pipe stages, which already synthesizes a fresh `__chelis_pipe` lambda
parameter for the piped value (`fresh_pipe_param_name`,
`desugar.rs:341-355`).

`x |> copy` lowers analogously with `(copy {} (var {} __chelis_pipe))`
as the lambda body.

## Fix architecture

Two architectural options were considered:

### Option A — parser-side synthesis (chosen)

Factor the pipe-loop body at `parser.rs:1000-1003` into a new
`parse_pipe_stage()` helper. The helper dispatches on the leading token:

- If the next token is `Realize` or `Copy` and the token after it is
  **not** `LParen` (i.e., the bare form), advance past the keyword and
  synthesize an explicit lambda:
  ```rust
  Expr::Lambda(
      vec![Param { name: "__chelis_pipe", ty: None, span }],
      Box::new(Expr::Realize(
          Box::new(Expr::Var("__chelis_pipe".into(), span)),
          span,
      )),
      span,
  )
  ```
- Else, fall through to `parse_prefix()` (unchanged).

The desugarer needs no changes: `Expr::Lambda` already desugars to the
canonical `(fn {} (params {} __chelis_pipe) (realize {} (var {} __chelis_pipe)))`
shape, which lowers correctly via the existing pipe-stage
`CallableExpr::Plain` arm.

### Option B — desugar-side synthesis (rejected)

Have the parser produce `Expr::Var("realize", span)` for the bare form
and special-case the unary-builtin names in `desugar_pipe_stage`. This
was rejected because it overloads `Expr::Var` to mean both "user-defined
name reference" and "builtin keyword reference", which would conflict
with the linearity checker and lowering's resolver (which look up `Var`
names in `program_defs` / `local_callables`). Option A keeps the Surf
AST shape unambiguous.

## Canary status

Today the parser-level pin
`bare_realize_pipe_stage_rejection_is_pinned`
(`crates/chelis-surf/tests/integration.rs`) asserts the **current**
`expected LParen` rejection. When the fix lands, this pin must be
inverted (assert successful parse) and the three `#[ignore]` fixtures
(`bare_realize_as_pipe_stage_parses`, `bare_copy_as_pipe_stage_parses`,
`chained_bare_keyword_with_named_pipe_stages_parses`) must flip to
running.

## Sibling sweep

Other places the parser might reject valid-per-spec forms:

| Site                         | Symptom                                                                     | Status                                                       |
|------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------|
| `let f = realize`            | Top-level bare keyword reference. Currently fails for the same reason.       | **Out of scope.** Item 2b is pipe-stage only.                |
| `let f = copy`               | Same shape.                                                                  | **Out of scope.**                                            |
| `x |> grad(f)`               | Arg form. Works post-Item-2.                                                 | Already fixed.                                               |
| `x |> vmap(f)`               | Arg form. Works post-Item-2.                                                 | Already fixed.                                               |
| `x |> cast(f32)`             | Arg form. Today fails with `expected Comma, found RParen` because `parse_cast` is two-arg (`cast(expr, type)`). First-arg insertion would require teaching `cast(f32)` as a one-arg form when in pipe context. | **Out of scope.** Separate gap; recommend §5 entry (see below). |

### Recommended future workstream (not filed here)

A unified "Surf parser bare-keyword in callable position" pass would
factor in:

1. Bare keyword as top-level reference (`let f = realize`).
2. Bare keyword as juxtaposition argument (already partially handled via
   the LParen postfix path in `parse_primary_atom`).
3. `cast(type)` as a one-arg pipe-stage form (per spec §3.6).

The diagnosis note **recommends** a §5 entry to `docs/gap_synthesis.md`
covering this broader keyword-callable disambiguation. Filing is an
orchestrator decision per the dispatch boilerplate.

## Files touched (fix commit)

- `crates/chelis-surf/src/parser.rs` — factor the pipe-loop body into
  `parse_pipe_stage()`, add the bare-keyword recognition.
- `crates/chelis-surf/tests/integration.rs` — flip the three `#[ignore]`
  fixtures to running and invert the regression pin (or replace it with
  a pin on a still-unsupported shape, e.g., bare `if` as a pipe stage).
