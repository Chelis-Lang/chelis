# Pipe autofix re-enable + bare-keyword sibling-sweep extras — diagnosis

Diagnoses Agent 2's bundled fix (F + H) on the 0.7.6 toolchain hygiene
workstream (`/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`):

- **F**: re-enable the `prefer-pipe-operator` autofix that 477bd0d
  disabled, reusing the Path 1B mechanism from Item 5 (PR #34).
- **H**: three sibling-sweep findings from Item 2b's diagnosis note
  (`docs/archive/investigations/parser_pipe_bare_keyword_diagnosis.md`,
  "Sibling sweep" table) — bare unary-builtin reference at top level
  (H1), bare unary-builtin keyword as juxtaposition argument (H2), and
  one-argument `cast(type)` pipe-stage form (H3).

## F — `prefer-pipe-operator` autofix re-enable

### Disable context

Commit 477bd0d (PR #22) stripped the `fix()` body and supporting
`Replacement` machinery from
`crates/chelis-lint/src/rules/prefer_pipe_operator.rs`. The reason
recorded in `spec/01-nomenclature.md` §12 (lines 1007-1014):

> `redundant-linearity-call` and `prefer-pipe-operator` do not expose
> auto-fixes until the fixer can prove the rewrite preserves semantics.
> ... For pipe rewrites, that proof requires knowing that the
> expression is a true first-argument dataflow chain.

The source-only walker in `find_pipe_candidates` / `pipe_candidate_at`
can detect a nested-first-argument call chain shape, but it cannot
prove the rewrite preserves dataflow semantics (the inner call could
have effects, the outer call could have side-tracked argument ordering
that only the type pipeline knows about, etc.).

### Mechanism: reuse Path 1B from Item 5

Item 5 (PR #34) added a non-breaking opt-in trait method
`Rule::fix_requires_typed_pipeline_check() -> bool` (default `false`)
and a corresponding gate in the CLI fix driver
(`crates/chelis-cli/src/main.rs::apply_lint_fixes`, lines 5145-5250).
The driver computes the candidate post-fix source in memory and runs
the same typed pipeline `chelis check` uses
(`typed_pipeline_accepts_surf` at L5015-5033): parse → desugar →
macro-expand → IR fitness → type-check → effect-check → linearity-check.
Replacements from rules whose `fix_requires_typed_pipeline_check()`
returns `true` only land on disk when the candidate passes every stage.

The architectural rationale lives in
`docs/archive/investigations/redundant_linearity_autofix_architecture.md`. The
trait surface is unchanged: `prefer-pipe-operator` simply overrides
`fix_requires_typed_pipeline_check` to `true`, restores its original
`fix()` body from commit `85e4248`, and the CLI gate handles the proof.

### Restored fix shape

The original autofix body (from `85e4248`) rewrites a nested
first-argument call chain to a pipe chain in canonical first-argument-
insertion form:

| Source                          | Rewrite                  |
|---------------------------------|--------------------------|
| `relu(neg(x))`                  | `x \|> neg \|> relu`     |
| `add(neg(x), bias)`             | `x \|> neg \|> add(bias)`|
| `outer(inner(x), scale)`        | `x \|> inner \|> outer(scale)` |

The rule's source-text walker already detects each candidate's start
and end byte ranges and tracks the call's name and trailing argument
list. The original `fix()` body builds the pipe form by:

1. Bottom-most stage: the innermost argument (e.g., `x`).
2. Each enclosing call contributes `|> name(rest...)`, where `rest`
   omits the carried first argument.
3. The final replacement spans `call.start..call.end` and substitutes
   the assembled pipe expression.

Path 1B's typed-pipeline gate catches the cases the source walker
cannot prove safe. For example, if the original source references
undefined `outer`, `inner`, `scale`, the typed pipeline rejects both
the original and the rewrite, the gate silently drops the replacement,
and the source is preserved.

### Why this re-enable is safe now

The 477bd0d disable predates Item 5's Path 1B infrastructure. With the
driver-level gate already shipped for `redundant-linearity-call`, the
re-enable for `prefer-pipe-operator` is a per-rule opt-in
(`fix_requires_typed_pipeline_check(&self) -> true` plus the original
`fix()` body). No new infrastructure is required.

### Anti-scope

The fix deliberately does **not**:

- modify the source-text walker's flagging criterion. The `check()`
  method's behavior is unchanged; only `fix()` is restored.
- extract the typed-pipeline gate into a separate crate. The pipeline
  lives in `chelis-cli` and stays there.
- touch `redundant-linearity-call`. That rule already shipped Path 1B
  in PR #34.
- change Item 2b's parser surface. The H findings are addressed
  separately in this same PR but they don't interact with F.

## H — Three sibling-sweep findings

Item 2b's diagnosis note enumerated the three sibling-sweep candidates
(`parser_pipe_bare_keyword_diagnosis.md`, "Sibling sweep" table). All
three reproduce empirically against the current parser (verified with
ad-hoc probe before drafting fixtures):

| Finding | Source                                          | Failure today                                              |
|---------|-------------------------------------------------|------------------------------------------------------------|
| H1      | `f = realize` (top-level let-binding RHS)       | `expected LParen, found Eof at byte 4`                     |
| H2      | `apply_fn(realize)` (juxtaposition argument)    | `expected LParen, found RParen at byte 35`                 |
| H3      | `x \|> cast(f32)` (one-arg pipe-stage cast)     | `expected Comma, found RParen at byte 58`                  |

All three share the same root cause as Item 2b: `parse_prefix`
dispatches every reserved-keyword token to a dedicated parser that
unconditionally expects the keyword's normal-form continuation
(`LParen` for `Realize`/`Copy`/`Cast`, etc.). Item 2b factored
`parse_pipe_stage` out of the pipe-loop and added bare-keyword
recognition there; the analogous fixes here apply to the non-pipe
prefix surfaces (H1/H2) and to the pipe-stage cast surface (H3).

### H1 — top-level bare unary-builtin reference

**Source**: `parser.rs::parse_prefix` (lines 1268-1269) routes
`TokenKind::Realize` / `TokenKind::Copy` to `parse_realize` /
`parse_copy`, which each call `expect(&TokenKind::LParen)?`
unconditionally. The same dispatch fires inside
`parse_let_def::parse_expr_until_decl_separator`, so a top-level
`f = realize` rejects with `expected LParen, found Eof`.

**Spec semantics** (per `01-nomenclature.md` §3.6): a bare unary builtin
in expression position is the function itself. The canonical
η-expansion is `fn (v) -> realize(v)`. That's exactly the same lambda
Item 2b's `parse_pipe_stage` synthesizes today for bare pipe stages.

**Fix surface**: factor a shared helper that recognizes bare unary
builtin keyword tokens (`Realize`, `Copy`) and synthesizes an explicit
lambda. The most surgical placement is to special-case these tokens
inside `parse_prefix` BEFORE dispatching to `parse_realize` /
`parse_copy`. If the next token is not `LParen`, take the bare-keyword
path; otherwise fall through to the existing call-form parser.

### H2 — bare unary-builtin keyword as juxtaposition argument

**Source**: `parser.rs::parse_primary_atom` (lines 1332-1410) covers
`Ident`, `TypeIdent`, `LParen`, `LBracket`, `If`, `Amp` — but NOT the
reserved keyword tokens (`Realize`, `Copy`, `Grad`, etc.). When the
juxtaposition-arg parser hits `apply_fn(realize)`, control flows from
`parse_juxtaposition_args` → `parse_expr_list` → `parse_expr(0)` →
`parse_prefix` → `parse_realize`, which expects `LParen` (the next
token is `RParen`, hence the error).

**Spec semantics**: identical to H1. A bare unary builtin passed as a
function argument is the η-expanded form.

**Fix surface**: the H1 fix in `parse_prefix` automatically resolves
H2 because `parse_expr_list` calls `parse_expr` which goes through
`parse_prefix`. If `parse_prefix` accepts a bare unary builtin and
returns the synthesized lambda, the juxtaposition-argument path
inherits the fix. No separate code change is needed for H2 once H1's
fix is in place — H1 + H2 share the same `parse_prefix` surface.

### H3 — one-arg `cast(type)` pipe-stage form (parser scope only)

**Spec language** (`01-nomenclature.md` §3.6, lines 311-313):

> A bare stage (`x \|> f`) passes the piped value as the only argument
> to `f`. A call stage (`x \|> f(y, z)`) inserts the piped value before
> the written arguments.

So `x |> cast(f32)` ≡ `cast(x, f32)`. The piped value fills the first
slot, the type argument fills the second. This is **exactly**
first-argument insertion, so per spec the form should parse.

**Source**: `parser.rs::parse_cast` (lines 1498-1506) hard-codes the
two-argument shape: `LParen`, expr, `Comma`, ident, `RParen`. When
called via `parse_pipe_stage → parse_prefix → parse_cast` for
`x |> cast(f32)`, it parses `f32` as the first expression (it's an
`Ident`, so it becomes `Var("f32")`), then expects `Comma` but gets
`RParen`.

**Fix surface**: Item 2b's `parse_pipe_stage` already has a
bare-keyword recognizer (for `Realize`/`Copy`). Extend it to also
recognize `Cast` followed by `LParen` followed by a single-arg
form, then synthesize an explicit lambda over a fresh `__chelis_pipe`
parameter:

```rust
// In parse_pipe_stage, after the bare-keyword check:
// Look-ahead for `cast (ident)` (one-arg form).
if matches!(self.peek(), TokenKind::Cast)
    && matches!(self.peek_after_current(), TokenKind::LParen)
{
    let cast_tok = self.advance();
    let cast_span = cast_tok.span;
    self.expect(&TokenKind::LParen)?;
    // If this is the one-arg form (ident, RParen), build the lambda;
    // otherwise rewind and let parse_prefix → parse_cast handle the
    // two-arg form.
    // ... synthesize Expr::Lambda([__chelis_pipe], cast(var __chelis_pipe, f32))
}
```

An alternative is to teach `parse_cast` itself a one-arg shape — but
that overloads `parse_cast`'s behavior (it would mean `cast(f32)` is a
valid expression in non-pipe context too, which has no spec interpretation
because the value-to-cast is missing). Keep the special form local to
`parse_pipe_stage` so the non-pipe `cast` parser stays unambiguous.

**Downstream type-inference limitation** (escalated to orchestrator):
With the parse-level fix in place, `x |> cast(f32)` parses and desugars
correctly to the canonical lambda over `__chelis_pipe`. However, the
type-checker's `infer_cast` (`crates/chelis-types/src/infer.rs:9019`)
requires the inner expression to resolve to `Type::Tensor(_,_)` or
`Type::Prim(_)` at inference time. The synthesized lambda's parameter
is initially a fresh type variable; `infer_cast` evaluates the body
before the pipe-stage unification (`infer_pipe` at L8880-8932) binds
the parameter. The result is a `CastNonTensor` error against the
unresolved type variable.

`realize` and `copy` do not have this issue because their inference
arms (L4489-4505 and L4506+) pass the inner type through unchanged,
allowing the pipe-stage unification to bind the parameter. `cast` is
structurally different: its target type is part of the node, and its
inference branches on the inner-expression type.

The parse-level fix in this PR is correct per spec §3.6. The
type-inference behavior is a separate structural concern (bidirectional
or two-pass inference for `cast` inside synthesized lambdas, or
delayed `infer_cast` resolution). Recommend filing as its own
follow-on workstream rather than expanding this PR's scope — the
H3 parser surface is closed by this fix, but end-to-end usage of
`x |> cast(f32)` will continue to surface a type-checker error until
that follow-on lands.

Item 2b's diagnosis note already noted H3 as a separate gap; this PR
fixes the parser half and documents the type-checker half as the
remaining work.

### Lambda synthesis approach (H1/H2/H3)

All three findings synthesize an `Expr::Lambda` that the existing
desugarer + lowerer already handle. The lambda over `__chelis_pipe`
mirrors the pattern in `parse_pipe_stage` from Item 2b. For H1/H2,
the lambda is used in non-pipe contexts (let RHS, juxtaposition arg);
this is fine because `Expr::Lambda` is a first-class value expression.

### Why not extend `parse_realize` / `parse_copy` to handle bare form

That would change the meaning of `realize` / `copy` everywhere they
appear — but the bare form is only spec-meaningful in expression
position, not in statement-prefix position (e.g., not inside a
type-ascription or as the lhs of `=` in a function definition). The
narrower fix at `parse_prefix` keeps the existing rejection sites
intact for those contexts.

## Sibling sweep — other bare-keyword-in-callable-position surfaces

After applying the H1/H2/H3 fixes, audit the remaining reserved-keyword
parsers in `parser.rs` for the same shape (callable-form spec but
rejected as bare):

| Keyword            | Spec arity                                  | Bare form spec-meaningful?           |
|--------------------|----------------------------------------------|---------------------------------------|
| `realize`, `copy`  | unary builtin                                | **Yes** (H1/H2 — fixed in this PR)    |
| `grad`             | takes a function, returns a function         | No (per Item 2b diagnosis)            |
| `vmap`             | takes a function (+ axis)                    | No                                    |
| `jit`              | takes a function                             | No                                    |
| `cast`             | binary (value, precision)                    | Pipe-stage one-arg form (H3 in this PR) |
| `with`             | handler + body                               | No                                    |
| `par`              | block of expressions                         | No                                    |
| `if`/`match`/`fn`  | structural                                   | No                                    |

The structural keywords (`if`, `match`, `fn`, `par`, `with`) have no
callable-form interpretation. Their bare-form rejection is the correct
behavior. The transform keywords (`grad`, `vmap`, `jit`) are not
unary builtins — their bare form has no spec-meaningful η-expansion
because they take a function-typed argument. Their arg form works
today via the existing pipe lowering (Item 2, PR #26).

Net: after H1/H2/H3 land, only `cast` retains a "valid-per-spec but
parser-rejected" arg-form for its pipe-stage one-arg shape (H3). Once
that lands, the bare-keyword sibling sweep on `parser.rs` is closed
for the unary-builtin and pipe-cast families.

## Why these are bugs (vs. expected diagnostics)

Per spec §3.6, all three forms have a defined meaning:

- H1: `f = realize` ≡ `f = fn (v) -> realize(v)` (η-expansion).
- H2: `apply_fn(realize)` passes the η-expanded form as an argument.
- H3: `x |> cast(f32)` ≡ `cast(x, f32)` (first-argument insertion).

The parser-level rejections today emit confusing "expected LParen"
errors that don't surface the spec-level meaning. Each fix replaces a
confusing rejection with a spec-canonical parse, matching the same
pattern Item 2b applied for bare-keyword pipe stages.

## Tests added in this PR

- F: `crates/chelis-cli/tests/cli.rs`
  - `lint_fix_prefer_pipe_operator_keeps_when_typed_pipeline_rejects`
    (replaces `lint_fix_preserves_unproven_prefer_pipe_operator_rewrites`):
    same Surf shape, same final assertion — the rewrite is silently
    dropped because the typed pipeline rejects the program. Validates
    Path 1B as the safety gate, not the disabled-fix state.
  - `lint_fix_prefer_pipe_operator_rewrites_when_typed_pipeline_accepts`
    (positive, `#[ignore]` until fix): `relu(neg(x))` → `x |> neg |> relu`.
  - `lint_fix_prefer_pipe_operator_rewrites_multi_arg_outer_stage`
    (positive, `#[ignore]` until fix): `add(neg(x), bias)` →
    `x |> neg |> add(bias)`.

- H: `crates/chelis-surf/tests/integration.rs`
  - `top_level_bare_unary_builtin_reference_parses` (`#[ignore]`).
  - `bare_unary_builtin_as_juxtaposition_argument_parses` (`#[ignore]`).
  - `one_arg_cast_pipe_stage_parses` (`#[ignore]`).

All three H fixtures fail today with the exact error shapes documented
in the table above, confirming the bugs are real.

## Files touched (fix commit)

- `crates/chelis-lint/src/rules/prefer_pipe_operator.rs` — restore
  `fix()` body, restore `name` field on `Call`, restore `replacement`
  field on `Candidate`, restore `render_stage` helper, override
  `fix_requires_typed_pipeline_check` to `true`.
- `crates/chelis-surf/src/parser.rs` — extend `parse_prefix` to
  accept bare `Realize`/`Copy` (H1/H2); extend `parse_pipe_stage` to
  recognize the one-arg `cast(type)` form (H3).
- `crates/chelis-cli/tests/cli.rs` — flip the F `#[ignore]` fixtures
  to running.
- `crates/chelis-surf/tests/integration.rs` — flip the H `#[ignore]`
  fixtures to running.

## References

- Plan: `.claude/plans/build-up-a-plan-mossy-meteor.md` Agent 2 (F + H).
- Item 5 architecture: `docs/archive/investigations/redundant_linearity_autofix_architecture.md`.
- Item 2b sibling sweep: `docs/archive/investigations/parser_pipe_bare_keyword_diagnosis.md`
  (the three findings here appear in its "Sibling sweep" table).
- Spec safety bar: `spec/01-nomenclature.md` lines 1007-1014.
- Spec first-argument insertion: `spec/01-nomenclature.md` §3.6.
- 477bd0d disable commit (PR #22).
- Original autofix: commit `85e4248` (PR #19).
- Item 2b fix: commit `6364546` (PR #35).
- Item 5 fix: commit `4982ce3` (PR #34).
