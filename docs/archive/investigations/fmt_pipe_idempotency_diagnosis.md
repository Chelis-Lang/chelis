# `chelis fmt` non-idempotency on multi-stage pipe chains — diagnosis

Diagnoses Finding 3a from the 0.7.6 red-team pass (PR #51).
Tests pinning the bug: see commit `ae9dbb6` (three `#[ignore]` cases in
`crates/chelis-surf/tests/integration.rs`).

## Bug shape (confirmed empirically)

`chelis fmt` is not a fixed point for any function body that is a pipe
chain with three or more `|>` stages. Pass 1 and pass 2 produce
different outputs; `chelis fmt --check` rejects the formatter's own
pass-1 output.

Reproducer:

```text
$ cat foo.ch
def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> neg |> abs |> sigmoid

$ chelis fmt --inplace foo.ch    # pass 1
$ cat foo.ch
def f(x: tensor[3, f32]) -> tensor[3, f32] = {
  x
  |> neg
  |> abs
  |> sigmoid
}

$ chelis fmt --inplace foo.ch    # pass 2
$ cat foo.ch
def f(x: tensor[3, f32]) -> tensor[3, f32] = { x
|> neg
|> abs
|> sigmoid }

$ chelis fmt --check foo.ch.pass1    # rejects formatter's own output
error: foo.ch.pass1 is not canonically formatted
```

Same shape for 4-stage chains and for 3-stage chains alongside
neighboring decls. Two-stage chains (`x |> f |> g`, total chain length
3 incl. seed) are fine because the flat-form fast path applies.

## Trace through the formatter

All line numbers refer to `crates/chelis-surf/src/format.rs` at
`ae9dbb6`.

### Pass 1: `Expr::Pipe` body

Pass-0 input parses as `FunDef { body: Expr::Pipe(seed, stages) }`.
`format_decl` for `FunDef` calls `format_function_body(body)` (line 87,
defined at 261):

```rust
fn format_function_body(expr: &Expr) -> String {
    if matches!(expr, Expr::Block(_, _, _)) {
        format_expr(expr)
    } else {
        let rendered = format_expr(expr);
        if rendered.contains('\n') {
            format!("{{\n{}\n}}", indent_lines(&rendered, 2))
        } else {
            rendered
        }
    }
}
```

`format_expr(Pipe)` (line 302) calls `format_pipe_expr` →
`format_pipe_layout(binding_head=None, seed, stages)` (line 538):

```rust
let total_stages = 1 + stages.len();
// ... build flat_chain ...
if total_stages <= 3 && flat.chars().count() <= WIDTH {
    return flat;
}
// else: multi-line, no binding head
let mut lines = vec![seed];
lines.extend(stages.iter().map(|stage| format!("|> {stage}")));
lines.join("\n")
```

For 3 stages, `total_stages = 4`, so the flat-form short-circuit at
line 556 does NOT fire. The function returns the multi-line form:

```text
x
|> neg
|> abs
|> sigmoid
```

Back in `format_function_body`, `rendered.contains('\n')` is true, so
it wraps with `{\n  ...\n}` and 2-space indentation. Final pass-1
output:

```text
def f(...) -> ... = {
  x
  |> neg
  |> abs
  |> sigmoid
}
```

This is reasonable canonical output. The bug is not in pass 1 alone;
pass 1 just sets up the AST that pass 2 mis-formats.

### Pass 2: `Expr::Block` body with empty bindings

The parser (`crates/chelis-surf/src/parser.rs:1763-1782`,
`parse_block`) sees `{ <expr> }` and produces
`Expr::Block(bindings=[], body=<expr>)`. So pass-1 output re-parses
as:

```text
FunDef { body: Block(bindings=[], body=Pipe(x, [neg, abs, sigmoid])) }
```

`format_function_body` matches `Expr::Block(_, _, _)` and calls
`format_expr(block)`, which dispatches to `format_block` (line 472):

```rust
fn format_block(bindings: &[LetBinding], body: &Expr) -> String {
    if bindings.is_empty() {
        return format!("{{ {} }}", format_expr(body));
    }
    // ... non-empty bindings path indents per-line ...
}
```

For empty bindings, the single-line `{ body }` template is used
unconditionally. `format_expr(body)` is the same multi-line pipe
string `"x\n|> neg\n|> abs\n|> sigmoid"`. Substituting:

```text
{ x
|> neg
|> abs
|> sigmoid }
```

which becomes the broken pass-2 line. The opening `{` glues to the
seed `x` on line 1; subsequent stage lines have no indentation; the
closing `}` glues to the last stage on the final line. None of that
re-parses to the same AST shape either (it does happen to re-parse as
the same Block, because Surf is brace-and-newline-tolerant), but
`fmt --check` compares byte text, and the bytes differ from pass 1.

## Why two-stage chains are fine

For a 2-stage chain `x |> f |> g`, `stages.len() == 2` so
`total_stages == 3`, which satisfies `total_stages <= 3` at line 556.
Provided the flat form fits in 80 columns it returns the single-line
flat chain, which never triggers the brace-wrap in
`format_function_body`. No `Expr::Block` ever materializes, so the
pass-2 mis-format path is never entered.

The threshold is a hard `<=` on stage count, not a width-only test.
Three stages tip into the multi-line path even when the flat form
trivially fits.

## Why other multi-line bodies do not regress

`format_block`'s `{ body }` single-line template predates the
multi-stage pipe case. Today it is only safe for `format_expr(body)`
results that do not contain newlines. Most expressions that survive
`format_expr` are single-line: literals, var refs, applications,
binops, short conditionals, etc. The only common single-expression
body shape that goes multi-line through `format_expr` is the
multi-stage `Expr::Pipe`. That is the surface this fix addresses; the
formatter contract is that the single-line `{ ... }` form is only
emitted when `format_expr(body)` itself is single-line.

## Fix sketch

Make `format_block`'s empty-bindings path mirror the non-empty path:
when `format_expr(body)` returns multi-line text, emit
`{\n  <indented body>\n}` instead of `{ <body> }`. The non-empty path
already does this (line 482-485). After the fix:

- Pass 1 (Pipe body) → wraps multi-line: `{\n  x\n  |> neg\n  ...\n}`.
- Pass 2 (Block body, empty bindings, Pipe inside) → also emits
  `{\n  x\n  |> neg\n  ...\n}`.

Both passes produce identical bytes, so `fmt(fmt(src)) == fmt(src)`
and `fmt --check` accepts the pass-1 output.

Pass N for N ≥ 2 stays idempotent because the AST shape stabilizes at
`Block(empty, Pipe)` and `format_block` always emits the same bytes
for that shape.

## Scope and corpus check

After the fix, sweep `examples/`, `packages/chelis-std/`, and the
in-tree fixture corpora to confirm no other `.ch` file relied on the
old `{ <multi-line> }` shape. The fix only changes the empty-bindings
branch in `format_block`, so the non-empty Block layout is unchanged.

## Why not change `format_pipe_layout`?

An alternative fix is to widen the flat-form fast path so 3- and
4-stage chains stay single-line when they fit within 80 columns.
That treats one symptom (the over-eager multi-line wrap for short
chains) but leaves the underlying bug: `format_block`'s single-line
empty-bindings template still mis-emits any other future
single-expression body that happens to be multi-line. The chosen fix
removes the structural bug at its source. Pipe-chain layout is
orthogonal and stays under the existing rule.
