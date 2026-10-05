# Item 2 — sibling-sweep findings (G1–G12)

Pure discovery pass for the orchestrator. Each gap below was reproduced
empirically (Surf source or in-tree IR test) against the merged Item 2
fix at `origin/main` (`6487d35`). No code changes. No fixes.

## One-paragraph summary

Of the twelve gaps walked, **five are real blockers for routine 0.7.6
user code** and warrant active dispatches (G9/G10 — same gap, two
labels; G11, G12). **Four are spec-blessed IR-lowering gaps** that hold
through to today only because the host lane catches them speculatively
(`if` on non-float — G5; `match` — G7), or because the type-checker
intercepts them before lowering ever fires (`par` — G6; `jit` — G8): the
`lower_unrepresentable` sites still exist, but no Surf path reaches
them with the type-checker enabled. **Three are internal corner paths**
that are unreachable from user-level Surf today because the
type-checker rejects them before lowering (`grad`/`vmap`/`vmap(grad)` of
a function-valued parameter at the IR level — G1/G2/G4; only G1/G2 fire
through `chelis build` because the C-host lane currently rejects them
with a clean diagnostic — see G12 for the related host-lane rejection).
G3 is reachable only through hand-crafted Deep that bypasses the
checker, same shape as G2.

The recommended dispatch matrix at the end groups the live blockers
into one PR (G9+G10), one parser PR (G11), one C-host-lane PR (G12),
and treats G1/G2/G3/G4/G5/G6/G7/G8 as reserved-for-later (spec or
unreachable-by-construction).

---

### G1: `grad` outside function application (unresolvable callable)

- **Location**: `crates/chelis-ir/src/lower.rs:2818` —
  `lower_grad_callable_with_nodes` early-returns when
  `extract_fn_parts(fn_expr)` returns `None`.
- **Trigger shape (concrete Surf)**:
  ```
  def h(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] = grad(model)(x)
  ```
  `chelis build` and `chelis eval` both emit
  `` `grad` is not supported by IR evaluation yet; use `chelis build --target c` instead``.
- **Trigger shape (concrete Deep)**:
  `(app {} (grad {} (var {} model)) (var {} x))`, where `model` is a
  function-valued parameter (so neither `local_callables` nor
  `program_defs` resolves it).
- **Sub-bugs**: one. The bug is "grad's inner callable cannot be
  resolved to a concrete `(fn (params...) body)` shape." Symmetric to
  G2 (vmap) and G4 (vmap-grad), differing only in which lowering
  helper does the rejection.
- **Fix surface estimate**: substantial. To support `grad` of a
  function-valued parameter, lowering would need to either (a)
  monomorphize on each callsite (specialize `h` for every caller's
  `model` argument), or (b) preserve grad as an IR construct and lower
  per-callsite at host-lane lowering — the same architectural fork
  G12 will face. Touches `lower_grad_callable_with_nodes`,
  `extract_fn_parts`, and the `CallableExpr` resolution path.
- **Spec status**: spec-blessed (`grad` is in `spec/03-deep-syntax.md`
  §2 as a first-class IR transform). However, the IR-eval rejection is
  documented as "use `chelis build --target c` instead", which is the
  same diagnostic family the merged Item 2 PR partially closed for
  pipe stages. Lowering owes the user support eventually; the
  C-target-via-host-lane path (G12) needs to handle this case too
  before this gap can close.

### G2: `vmap` standalone (unresolvable callable)

- **Location**: `crates/chelis-ir/src/lower.rs:3006` —
  `lower_vmap_callable_with_nodes` early-returns on `extract_fn_parts`
  failure.
- **Trigger shape (concrete Surf)**:
  ```
  def h(model: tensor[3, f32] -> tensor[3, f32], xs: tensor[2, 3, f32]) -> tensor[2, 3, f32] =
    vmap(model)(xs)
  ```
  `chelis build` and `chelis eval` both emit
  `` `vmap` is not supported by IR evaluation yet; use `chelis build --target c` instead``.
- **Trigger shape (concrete Deep)**:
  `(app {} (vmap {} (var {} model) (lit {} 0)) (var {} xs))`.
- **Sub-bugs**: one. Mirror of G1.
- **Fix surface estimate**: same as G1 — architectural fork between
  monomorphization and IR-level transform preservation. Helpers
  needed: `lower_vmap_callable_with_nodes`, `extract_fn_parts`. Pipe
  variant `xs |> vmap(model)` would hit G9 first since `model` is a
  parameter.
- **Spec status**: spec-blessed (Deep tag `vmap`). Lowering owes the
  user support; same gating concern as G1.

### G3: `vmap` with no tensor arguments

- **Location**: `crates/chelis-ir/src/lower.rs:3051-3055` —
  `lower_vmap_callable_with_nodes` early-returns when no argument's
  rank exceeds `axis`.
- **Trigger shape (concrete Surf)**: **not directly reachable**. The
  type-checker rejects `vmap(f)` over scalar arguments with
  "def 'h' body doesn't match declared signature" before lowering
  fires. The reduced empirical repro requires either hand-written Deep
  or a checker bypass.
- **Trigger shape (concrete Deep)**:
  `(app {} (vmap {} (fn {} (params {} (x {type: (t-prim {} f32)})) (mul {} (var {} x) (var {} x))) (lit {} 0)) (lit {} 1.0))`.
  Parsed via direct `chelis_deep::parser::parse_str` and run through
  `parse_and_lower_unchecked` it would fire L3051.
- **Sub-bugs**: one. The "no tensor arguments" branch covers axis
  out-of-rank for every argument — semantically equivalent for
  scalar-only argument lists and for tensor argument lists where
  every tensor has rank ≤ axis.
- **Fix surface estimate**: defensive only; the type-checker already
  forbids the shape. The code path exists for safety against
  unchecked Deep input. No fix needed unless we want either a clearer
  diagnostic or a checker-aligned earlier rejection.
- **Spec status**: rejection is correct per spec — vmap requires at
  least one tensor argument carrying the batch dim. No support owed.

### G4: `vmap(grad)` with no tensor arguments

- **Location**: `crates/chelis-ir/src/lower.rs:3215-3218` —
  `lower_vmap_grad_callable_with_nodes` early-returns on the same
  axis-vs-rank check as G3.
- **Trigger shape (concrete Surf)**: **not directly reachable**.
  Type-checker rejects `vmap(grad(f))` over scalar arguments
  similarly to G3.
- **Trigger shape (concrete Deep)**: analogous to G3, with `vmap`
  wrapping `grad`.
- **Sub-bugs**: one. Same shape as G3.
- **Fix surface estimate**: defensive only; type-checker handles the
  user-level case. No fix owed.
- **Spec status**: rejection correct. No support owed.

### G5: `if` expressions (non-float branches)

- **Location**: `crates/chelis-ir/src/lower.rs:4717` — `lower_if`
  rejects when output precision is not `is_float()`.
- **Trigger shape (concrete Surf)**: **the `if`-on-tensors-of-int
  shape reproduces the IR rejection but is intercepted by the C
  backend's host-lane fallback**, so `chelis build` succeeds. Direct
  trigger requires hand-written Deep or a test that bypasses the host
  lane. The IR test
  `non_float_if_is_rejected_before_lowering`
  (`lower.rs:6401`) pins the rejection.
- **Trigger shape (concrete Deep)**:
  `(if {type: (t-prim {} bool)} (lit {type: (t-prim {} bool)} true) (lit {type: (t-prim {} bool)} true) (lit {type: (t-prim {} bool)} false))`.
- **Sub-bugs**: one. The `if` lowering uses arithmetic-on-floats
  (`Mul`/`Add`/`Neg`) to implement masked select; non-float output
  precision has no representable implementation.
- **Fix surface estimate**: extend `lower_if` with an integer-mask
  variant (multiply-by-mask in the target precision domain, similar
  to the float path but using integer arithmetic), or add a separate
  branching IR node `Select` that lowers to a C-level ternary.
- **Spec status**: spec-blessed (`if` is in the Deep vocabulary,
  `spec/03-deep-syntax.md:265`). The IR rejection is a Phase 0
  implementation limit; lowering owes support for non-float types but
  the C-target host-lane path already covers the user-visible case.
  Reserved-for-later until a Surf shape arises that needs IR-level
  non-float `if`.

### G6: `par` (parallelism)

- **Location**: `crates/chelis-ir/src/lower.rs:4772` — `lower_par`
  unconditionally rejects.
- **Trigger shape (concrete Surf)**:
  ```
  def f(x: tensor[3, f32], y: tensor[3, f32]) -> tensor[3, f32] = par { x; y }
  ```
  But the rejection comes from the **type-checker**, not lowering:
  `crates/chelis-types/src/infer.rs:2406-2412` emits
  `` `par` is not supported by IR lowering`` and rejects the program
  before lowering fires. The `lower_par` site is defensive only.
- **Trigger shape (concrete Deep)**: `(par {} (var {} x) (var {} y))`
  through `parse_and_lower_unchecked`.
- **Sub-bugs**: one. Two layers of defense (checker + IR) but a single
  semantic gap.
- **Fix surface estimate**: substantial — `par` is a parallelism
  primitive with no DAG representation today. Closing requires either
  (a) sequential-semantics lowering (par→sequence; documented in
  spec/03 as the v1 semantics) wired in, or (b) a DAG-level
  fork/join construct. The checker rejection at infer.rs:2406 is the
  primary user-visible barrier; lifting that and routing to a
  sequential-semantics lowering at `lower_par` would close the
  Phase-0 promise in the spec.
- **Spec status**: spec-blessed but explicitly Phase-deferred —
  `spec/03-deep-syntax.md:275` reads "Parallel evaluation (v1:
  sequential)". The spec promises sequential semantics in v1;
  current code rejects rather than running sequentially.
  **Spec-implementation mismatch** worth surfacing to the
  orchestrator, separate from the routine-lowering work.

### G7: `match` (pattern matching)

- **Location**: `crates/chelis-ir/src/lower.rs:4898` — `lower_match`
  unconditionally rejects.
- **Trigger shape (concrete Surf)**:
  ```
  type R = | A | B
  def f(x: tensor[3, f32], y: tensor[3, f32], v: R) -> tensor[3, f32] =
    relu(match v with { | A => x | B => y })
  ```
  `chelis build` succeeds via host-lane fallback; `chelis eval`
  silently returns no value (the IR-eval path swallows the
  rejection). IR test `unsupported_match_is_rejected_before_lowering`
  (`lower.rs:6415`) pins the direct rejection.
- **Trigger shape (concrete Deep)**:
  `(match {} (var {} x) (arm {} (pat-var {} y) () (var {} y)))`.
- **Sub-bugs**: at least two:
  1. The direct IR-lowering rejection (no DAG representation for
     pattern matching).
  2. The host-lane silent success on builds vs IR-eval's silent
     no-output. The eval-side behavior is itself a separate bug — the
     diagnostic should surface rather than producing no output.
- **Fix surface estimate**: very large. `match` requires either tag-
  comparison-and-branch (similar to G5's `if` extension) or a richer
  IR node family. Bigger than Item 2's pipe-stage scope. The
  silent-no-output eval bug is much smaller and is reproducible as a
  separate dispatch.
- **Spec status**: spec-blessed (`match` is in the Deep vocabulary,
  `spec/03-deep-syntax.md:263`). Lowering owes support; reserved for
  a dedicated phase.

### G8: `jit` operator

- **Location**: `crates/chelis-ir/src/lower.rs:2426` —
  `"vmap" | "jit" => self.lower_unsupported(tag, elems)`. Note the
  plan listed line 2447 but actual is L2426. (L2447 is the unrelated
  start of `lower_def`.) Routes to `lower_unsupported` at L4921, not
  `lower_unrepresentable`.
- **Trigger shape (concrete Surf)**:
  ```
  def f(x: tensor[3, f32]) -> tensor[3, f32] = jit(relu)(x)
  ```
  `chelis build` and `chelis eval` both emit
  `` `jit` is not supported by IR evaluation yet; use `chelis build --target c` instead``.
  But again the type-checker also rejects at `infer.rs:2406`. IR test
  `fix4_jit_is_rejected_before_lowering` (`lower.rs:6275`) pins the
  direct rejection.
- **Trigger shape (concrete Deep)**: `(jit {} (var {} f))`.
- **Sub-bugs**: one. Routing-only — `jit` is a compilation hint
  that should be transparent (semantics identical to its argument,
  modulo compiler scheduling decisions), so lowering it as a passthrough
  is straightforward; the current code chooses to fail loudly instead.
- **Fix surface estimate**: small — `jit` semantics in the v1 spec are
  pass-through (the JIT effect is recorded in metadata, not lowering).
  Replacing the rejection with `self.lower_expr(&elems[2])` would
  unblock it, modulo deciding what metadata persists. Touches the
  one dispatch line at `lower.rs:2426` and the checker rejection at
  `infer.rs:2406`.
- **Spec status**: spec-blessed (`jit` is in the Deep vocabulary,
  `spec/03-deep-syntax.md:317`). Spec-implementation mismatch — spec
  promises a compilation trigger that is semantically a no-op at
  evaluation time; implementation rejects unconditionally.

### G9: `lower_pipe()` `None`-resolution fallthrough

- **Location**: `crates/chelis-ir/src/lower.rs:4631` (post-Item-2;
  diagnosis brief estimated `4587`). The arm fires when
  `resolve_callable_expr(func_expr)` returns `None` — i.e., the pipe
  stage is a callable that cannot be resolved through
  `local_callables`, `program_defs`, or the `var`/`vmap`/`grad`
  recursive resolution path in `resolve_callable_expr_inner`
  (`lower.rs:2694-2758`).
- **Trigger shape (concrete Surf)**: see G10 below — the canonical
  user-facing shape is a function-valued parameter used as a pipe
  stage.
- **Trigger shape (concrete Deep)**: the canary test at
  `lower.rs:6428` uses
  `(pipe {} (lit {type: (t-prim {} f32)} 1.0) (grad {} (var {} f)))`
  with no `program_defs` entry for `f` (the diagnosis note at
  `docs/archive/investigations/pipe_grad_stage_diagnosis.md` explains this
  hits the `None`-fallthrough, not the `CallableExpr::Grad` arm).
- **Sub-bugs**: at least three failure subclasses share this single
  rejection site:
  1. Function-valued parameter as pipe stage (G10 — `def h(f, x) = x |> f`).
  2. Inline `fn`-literal not threaded through `resolve_callable_expr`
     (uncommon in user code; would need a literal lambda as a pipe
     stage that the resolver doesn't recognize).
  3. Pipe stage with an unresolved (typo'd or out-of-scope) callable
     name — the canary's shape.
- **Fix surface estimate**: medium. Subclasses 1+2 share a common fix:
  introduce a notion of a parameter-callable carrying a typed function
  signature, and synthesize a host-lane call from the pipe stage when
  the resolver returns "parameter-callable known by name". Subclass 3
  is a real diagnostic that must remain.
- **Spec status**: spec-blessed for subclasses 1+2 (Surf supports
  function-valued parameters; pipe is meant to compose freely with
  them per `spec/01-nomenclature.md` §3.6). Subclass 3 is correctly
  rejected.

### G10 (Item 2-extended): pipe stage = function-valued parameter

- **Location**: same as G9 (`crates/chelis-ir/src/lower.rs:4631`).
- **Trigger shape (concrete Surf)** (empirically verified):
  ```
  def double_apply(f: &tensor[3, f32] -> tensor[3, f32], x: &tensor[3, f32]) -> tensor[3, f32] =
    x |> f |> f
  def relu1(x: &tensor[3, f32]) -> tensor[3, f32] = relu(x)
  def test_double_apply(seed: &tensor[3, f32]) -> tensor[3, f32] =
    double_apply(relu1, seed)
  ```
  `chelis check` passes (types OK).
  `chelis eval --file ...` emits
  `pipe stage is not supported by IR evaluation yet; use `chelis build --target c` instead at source span `surf:100..101``.
  `chelis build` emits the same diagnostic (the host lane reaches the
  same path).
- **Trigger shape (concrete Deep)**:
  `(pipe {span: ...} (var {} x) (var {} f) (var {} f))` where `f` is
  a parameter of the enclosing `def`. `resolve_callable_expr_inner`
  recurses into `(var {} f)`, looks up `f` in `local_callables` (not
  there) and `program_defs` (not there — `f` is a parameter), returns
  `None`, and falls into the L4631 arm.
- **Sub-bugs**: this **is** G9 subclass 1. They are not two distinct
  bugs; G10 is the user-facing label and G9 is the code-level label.
  Confirmed by reading `resolve_callable_expr_inner` and tracing the
  `Some("var")` arm at `lower.rs:2704-2717`: function-valued
  parameters don't live in either of the two lookup tables.
- **Fix surface estimate**: medium. Extend lowering with a new
  `CallableExpr::Parameter { name, signature }` variant; populate it
  when `resolve_callable_expr_inner` finds a parameter shadowed by a
  function-typed argument; in `lower_pipe`, route that variant to a
  host-lane call (similar to how `chelis-ir::host` already handles
  parameter-callables via the `call` builtin). The host lane already
  has the machinery — see
  `host_program_unresolved_call_sites`
  (`crates/chelis-ir/src/host.rs:1314`) — so the fix is plumbing,
  not new infrastructure.
- **Spec status**: spec-blessed. Pipe and first-class functions are
  both spec features; their composition should work. This is a
  routine-lowering blocker for hello-chelis-class corpora.

### G11 (Item 2b): parser rejects bare keyword as pipe stage

- **Location**: `crates/chelis-surf/src/parser.rs:1175` — `parse_prefix`
  dispatches `TokenKind::Realize` to `parse_realize`
  (`parser.rs:1510-1516`), which **unconditionally** expects an
  `LParen` after the keyword. Same shape applies to `Copy`, `Grad`,
  `Vmap`, `Jit`, `Par`, `Cast`, `If`, `Match`, `Fn`, `With` (parser
  dispatch table at `parser.rs:1168-1179`).
- **Trigger shape (concrete Surf)** (empirically verified):
  ```
  def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize
  ```
  `chelis check`, `chelis eval`, `chelis deep` all emit
  `expected LParen, found Eof at byte 45`. Same for `copy`, `grad`,
  `vmap`, `jit`; `par` emits `expected LBrace`.
- **Trigger shape (concrete Deep)**: not applicable — the gap is
  before Deep is produced. (Note that the Surf desugarer at
  `crates/chelis-surf/src/desugar.rs:511` already handles a bare
  `Var` in a pipe stage by wrapping in a synthesized lambda;
  same wrapping would suffice for these keyword forms if the parser
  produced an `Expr::Var`/`Expr::Realize`-applied-to-nothing.)
- **Sub-bugs**: one parser-level gap with multiple keyword
  manifestations. The gap is **structural**: the pipe-stage parser
  (`parser.rs:1000-1003`) calls `parse_prefix()`, which interprets
  every reserved keyword as "the keyword's full statement form
  must follow." It does not allow bare keyword reference. Affected
  keywords: `realize`, `copy`, `grad`, `vmap`, `jit`, `par`, `cast`,
  `if`, `match`, `fn`, `with`.
- **Fix surface estimate**: medium. Two architectural options:
  1. **Parser-side**: detect when a keyword is in a pipe-stage
     context (or any context that doesn't need its full form) and
     produce an `Expr::Var`-like reference. The pipe loop already
     has the context; the parser would need to peek "is this stage
     just a bare keyword token followed by another `|>` or end of
     expression?" and produce a synthesized identifier in that case.
  2. **Desugar-side**: lift the special handling to `desugar.rs`,
     have `parse_prefix` produce a generic `Expr::Apply(keyword, [])`
     when LParen is absent in a pipe-stage context.
- **Spec status**: spec-blessed. `|> realize` is canonical Surf style
  per `spec/01-nomenclature.md` §3.6 (first-argument insertion for
  unary builtins). The parser is mis-rejecting valid Surf.

### G12 (Item 2c): C backend rejects `grad` over piped body

- **Location**: `crates/chelis-cli/src/main.rs:1571-1588`. Triggered
  when `chelis_ir::host::host_program_unresolved_call_sites`
  (`crates/chelis-ir/src/host.rs:1314`) returns a non-empty list,
  meaning the host body contains an unresolved generic `call(...)`
  builtin or `__unresolved_*` builtin — the host lane couldn't
  resolve the callee at lowering time.
- **Trigger shape (concrete Surf)** (empirically verified):
  ```
  def sumsq(theta: tensor[3, f32]) -> f32 =
    mul(theta, theta) |> sum(0) |> tensor_to_scalar
  def main(theta: tensor[3, f32]) -> tensor[3, f32] = grad(sumsq)(theta)
  ```
  `chelis check` passes.
  `chelis build --target c` rejects with the long-form
  ``can't lower these defs. Their body applies/binds `grad` (or `vmap`)
  in a position the host lane can't resolve...`` diagnostic.
- **Trigger shape (concrete Deep)**: the `main` def lowers to a host
  body that includes `(app {} (grad {} (var {} sumsq)) (var {} theta))`.
  Because `sumsq`'s body contains pipe stages, the host-lane lowering
  emits an unresolved generic `call("sumsq", ...)` rather than an
  inlinable tensor body.
- **Sub-bugs**: at least two:
  1. **Host-lane summary doesn't handle piped function bodies**.
     Non-pipe equivalent (e.g., `def sumsq(theta) = tensor_to_scalar(sum(mul(theta, theta), 0))`
     followed by `grad(sumsq)(theta)`) **does** compile cleanly
     (verified). The pipe rewrite breaks the summary path.
  2. **Diagnostic asymmetry**: the same shape with `grad` replaced
     by a plain call (`def main(theta) = sumsq(theta)`) compiles
     fine. The blocker is specifically the combination of
     pipe-in-body + `grad`-over-the-binding.
- **Fix surface estimate**: medium-to-large. The
  `derive_host_function_specialization` /
  `try_summarize_sparse_helper` chain
  (`crates/chelis-ir/src/host.rs:1329-1505` and surrounding) walks
  the host body of `sumsq` looking for a single-`TensorCall` shape;
  a pipe-built body lowers to a `Let`-chain that the summarizer
  doesn't recognize, so it falls through to "host call with
  unresolved callee." Closing this requires teaching the summarizer
  about pipe-lowered Let chains, or rewriting them to a flat
  composition before summarization.
- **Spec status**: spec-blessed. Pipe is a canonical Surf style;
  `grad` is a canonical IR transform; their composition should
  compile. This is a routine-lowering blocker for hello-chelis-class
  corpora.

---

## Recommended dispatch matrix

The orchestrator decides; this is a recommendation based on shared
fix surface.

| Bundle | Gaps | Rationale | Live blocker? |
|--------|------|-----------|----------------|
| **A: Pipe with parameter callables** | G9, G10 | Same code site (`lower.rs:4631`); G10 is the user-facing label and G9 is the internal label. Single fix in `resolve_callable_expr_inner` + `lower_pipe`. | Yes (G10 blocks hello-chelis). |
| **B: Parser bare-keyword pipe stages** | G11 | Single parser site (`parse_prefix` keyword dispatch); affects 11 keywords with identical shape. | Yes (G11 blocks hello-chelis). |
| **C: C-host-lane summary over piped bodies** | G12 | Distinct subsystem (`chelis-ir::host` summarizer); independent of A/B. | Yes (G12 blocks hello-chelis). |
| **D: `jit` is semantically a no-op** | G8 | Spec-implementation mismatch; small fix (pass-through lowering + lifted checker rejection). Likely worth its own small dispatch. | No (spec deviation, not user-facing blocker). |
| **E: `par` v1-sequential semantics** | G6 | Spec-implementation mismatch; spec promises sequential semantics, code rejects. Independent of D. | No (spec deviation). |
| **F: IR `if` non-float lowering** | G5 | Reserved for later — host lane covers the user case; no user-visible blocker today. | No. |
| **G: IR `match` lowering** | G7 | Reserved for a dedicated phase. Sub-bug 2 (eval-side silent no-output) is a smaller, separable diagnostic fix. | No. |
| **H: IR-level transforms over parameter callables** | G1, G2 | Same architectural fork as G10/G12 — once C-host-lane can resolve, IR-eval can follow. Reserved until A+C are in. | Borderline (only via `chelis eval`; `chelis build` for G1/G2 already rejects with a clear diagnostic — see G12). |
| **I: Defensive scalar-vmap paths** | G3, G4 | Type-checker already rejects user-level cases. No fix owed. | No. |

The three recommended next dispatches are **A**, **B**, **C** (the
real hello-chelis blockers). D/E (jit/par spec compliance) are
small-and-independent — orchestrator may bundle or split.
F/G/H/I are reserved-for-later with the rationale documented above.

---

*Discovery only. No code changes outside this file.*
