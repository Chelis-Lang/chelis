# `jit` and `par` spec-impl mismatch diagnosis

Diagnosis pass for the bundled fix dispatched by
`/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md` (Agent 1 —
D+E). Cross-references:

- Sweep findings: `docs/investigations/item2_sibling_sweep_findings.md`
  sections **G6** (`par`) and **G8** (`jit`).
- Pinning tests: `crates/chelis-types/tests/jit_par_passthrough.rs`
  (commit `3137c18`).

No code changes in this commit.

## Spec sections (quoted, do not edit)

`spec/03-deep-syntax.md` §2.7 (Transforms table):

```
| `jit` | `(jit {} expr)` | Compilation trigger |
```

`spec/03-deep-syntax.md` §2.3 (Expressions table):

```
| `par` | `(par {} expr₁ expr₂ ...)` | Parallel evaluation (v1: sequential) |
```

Both forms are first-class in the 61-tag closed Deep vocabulary. `jit`
is a compilation-trigger transform that is a semantic no-op at
evaluation (the JIT effect, if any, lives in metadata, not in the
computed value). `par` v1 explicitly promises sequential semantics.

`spec/01-nomenclature.md` §1.3 also lists both `jit` and `par` as
reserved keywords; Surf parses them; the surf decompiler emits them.

## Empirical reproduction (current `main` + this branch's commit 1)

### `jit`

Surf snippet (`jit(x)` — bare, no further application):

```
def f(x: tensor[3, f32]) -> tensor[3, f32] = jit(x)
```

`chelis check`:

```
"errors": [{"kind":"Other","message":"`jit` is not supported by IR lowering","severity":0.5}]
```

Surf snippet (`jit(relu)(x)`):

```
def f(x: tensor[3, f32]) -> tensor[3, f32] = jit(relu)(x)
def main(x: tensor[3, f32]) -> tensor[3, f32] = f(x)
```

`chelis check`: passes (score 1.0). `chelis eval`:

```
error: `jit` is not supported by IR evaluation yet; use `chelis build --target c` instead
```

The asymmetry is because the checker validator (`validate_ir_expr`,
`crates/chelis-types/src/infer.rs:2344`) only inspects the `app` arg
list, not the head of an `app`. When `jit` is wrapped by `app`, the
checker walks past it and the rejection happens later at lowering.

### `par`

Surf snippet:

```
def f(x: tensor[3, f32], y: tensor[3, f32]) -> tensor[3, f32] = par {
  x;
  y
}
```

`chelis check`:

```
"errors": [{"kind":"Other","message":"`par` is not supported by IR lowering","severity":0.5}]
```

`par` is rejected in the checker because the validator's generic
fallthrough loop (line 2465 of `infer.rs`) walks all elements
including the inner par body, and the dedicated `par` typing arm in
`infer_expr` at `infer.rs:4471` is reached before validation runs, so
the rejection at L2406 fires.

## Located rejection / handling sites

| Site | File:line | Role | Today | Spec-aligned target |
|---|---|---|---|---|
| Checker validator rejection | `crates/chelis-types/src/infer.rs:2405-2412` | `validate_ir_expr` raises an `Other` error for `tag = par \| jit` | rejects | drop the rejection arm |
| Checker `par` typing arm | `crates/chelis-types/src/infer.rs:4471-4488` | `infer_expr` already types `par` as the type of its last child | already correct | keep as-is |
| Checker `jit` typing arm | `crates/chelis-types/src/infer.rs` | no dedicated arm; `jit` falls through `infer_expr` generic recursion | unknown but currently unreachable past the validator | add a pass-through arm that returns the inner child's type |
| IR pre-lowering walk | `crates/chelis-ir/src/lower.rs:1665-1680` | `assert_ir_lowerable` rejects `match \| par \| jit` before lowering | rejects | drop `par` and `jit` from the matches!, keep `match` |
| `par` lowering arm | `crates/chelis-ir/src/lower.rs:2513` → `lower_par` (L4997-L5000) | `lower_par` routes to `lower_unrepresentable("par", ...)` | rejects | lower each child and return the value of the last (sequential composition) |
| `jit` lowering arm | `crates/chelis-ir/src/lower.rs:2521` (`"vmap" \| "jit" => lower_unsupported`) | routes to `lower_unsupported("jit", ...)` | rejects | lower the inner expression `elems[2]` (pass-through) |
| Existing IR canary | `crates/chelis-ir/src/lower.rs:6500-6508` (`fix4_jit_is_rejected_before_lowering`) | pins the rejection | currently green | replace with a pass-through-lowering test |

The sweep findings (G8) noted the plan's line numbers (`lower.rs:2447`)
were stale; the live dispatch is at `lower.rs:2521`. The
`assert_ir_lowerable` site at L1670 was not enumerated in the plan but
also rejects `par`/`jit` before lowering runs. It is in scope.

## Sibling sweep — places that touch `par`/`jit`

Searched `crates/` for `"par"` and `"jit"` symbol references. Outside
the rejection chain above, the keywords are correctly handled:

| Site | Status |
|---|---|
| `crates/chelis-surf/src/lexer.rs:387,393` | tokens; no change |
| `crates/chelis-surf/src/desugar.rs:242,262,1056,1088` | Surf → Deep desugar; emits `(jit {} ...)` and `(par {} ...)`; no change |
| `crates/chelis-surf/src/decompile.rs:733,750,1620,1653` | Deep → Surf decompile; round-trips correctly; no change |
| `crates/chelis-deep/src/validate.rs:32,60` | closed-tag vocabulary; no change |
| `crates/chelis-validate/src/lib.rs:48,71` | grammar validator; no change |
| `crates/chelis-surf/tests/integration.rs:418` | exercises Surf/Deep round-trip including `par`; no change |
| `crates/chelis-cli/tests/cli.rs:4747` | tree-sitter validate accepts `par { a; b }`; no change |

The desugar and decompile sides already treat both as first-class
Phase-0 constructs; the gap is entirely between the validator
rejection and the lowering rejection.

C backend (`chelis-backend-c`) and host-emit have no `par`/`jit`
literal references; once lowering passes them through to the DAG (jit
as identity, par as sequence-yielding-last) the backend sees ordinary
nodes and emits them without further work.

## Chosen fix shape

Three coordinated changes in commit 3:

1. **`crates/chelis-types/src/infer.rs:2405-2412`** — remove the
   `matches!(tag, "par" | "jit")` rejection arm in `validate_ir_expr`.
   Both forms are spec-blessed.
2. **`crates/chelis-types/src/infer.rs` `infer_expr`** — add a `jit`
   typing arm next to the existing `par` arm (L4471). Spec says `jit`
   is a compilation trigger over an expression; the inferred type of
   `(jit {} inner)` is the type of `inner`.
3. **`crates/chelis-ir/src/lower.rs:1665-1680`** — drop `"par"` and
   `"jit"` from the `assert_ir_lowerable` reject list; keep `"match"`
   (still genuinely unrepresentable).
4. **`crates/chelis-ir/src/lower.rs:2513-2521`** — split the `jit`
   arm out of `"vmap" | "jit"` so `vmap` keeps its current
   `lower_unsupported` routing while `jit` gets its own pass-through
   helper. Replace `lower_par`'s body to lower each child in order and
   return the last `LoweredValue`.
5. **`crates/chelis-ir/src/lower.rs:6500-6508`** — flip
   `fix4_jit_is_rejected_before_lowering` from a rejection canary into
   a pass-through-lowering test, and add a sibling `lower_par_is_sequential`
   test. (The `match` rejection canary at
   `fix4_grad_is_rejected_before_lowering`'s neighborhood stays.)
6. **`crates/chelis-types/tests/jit_par_passthrough.rs`** — drop the
   `#[ignore]` attribute from the two positive fixtures. Drop the two
   negative-rejection fixtures, since the rejection is the bug.

The choice is forced by the spec: `par` v1 is sequential, `jit` is a
compilation trigger that is a no-op at eval. Both reduce to operations
the DAG already represents (sequenced node construction in `par`'s
case; identity on the inner value in `jit`'s case). No new IR op,
backend op, or public API is required. No structural workaround.

## Out-of-scope items

- **Metadata persistence of the `jit` effect**: per
  `spec/03-deep-syntax.md` §2.7 `jit` is a compilation trigger; its
  effect could later be threaded into emitted code metadata so a
  downstream toolchain knows to JIT-compile that region. This is
  Phase-1+ work and not promised at Phase-0 eval. Pass-through is the
  spec-compliant v1 semantic.
- **Real parallel scheduling for `par`**: the spec explicitly bounds
  v1 to sequential semantics. Sub-Phase 1+ DAG fork/join is
  out-of-scope.
- **`vmap` (also routed to `lower_unsupported` today)**: the sweep
  treats `vmap` standalone (G2) as a separate ticket because its fix
  requires resolving the inner callable; it is not a one-line
  pass-through.
- **`match` rejection**: still genuinely unrepresentable at Phase 0;
  the `assert_ir_lowerable` reject for `match` stays.
