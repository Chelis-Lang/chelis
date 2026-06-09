# Issue #338 implementation handoff (session stopped mid-gate)

**Date:** 2026-06-09. **Branch:** `fix/338-eval-named-axis` (off `main` @ `9e498f1`,
the 0.7.24 release commit). **ALL CHANGES ARE UNCOMMITTED** in the working tree.
The plan of record is `~/.claude/plans/can-you-please-review-polished-salamander.md`.

## What is done (implementation complete, tests green locally)

`chelis eval` now evaluates named-axis reductions with backend-identical
numerics by routing through `lower_subexpr_program` + the forward DAG
evaluator (the `grad`/`vmap` lane). `cargo nextest run -p chelis-cli --test
rank_poly_tier3` = **22/22 pass**, including the restored eval-vs-backend
agreement oracle at ranks 2/3/4 (non-square), the exact issue repro
(`[3.0, 7.0]`), and a parity-corners suite.

### Files changed (uncommitted)

- `crates/chelis-compiler-api/src/runtime.rs` — the whole fix:
  - **Site A** (`eval_named_axis_reduction_app`): a reduction app whose axis
    arg is a bare `(var name)` is intercepted in `eval_app` BEFORE generic arg
    evaluation; the operand is interpreter-evaluated, staged as a placeholder
    typed from its *static type* (checker `{type:}` annotation on the node →
    frame `binding_types` → top-level `type_env`), and the app is lowered +
    DAG-evaluated. Checker guarantees bare-var axis ⇒ named axis
    (`check_reduction_signature`, infer.rs ~13548-13651).
  - **Site B** (`try_named_axis_def_call`): calls to defs that *require*
    routing (`def_requires_named_axis_routing`: direct bare-var-axis reduction
    in body, or reference to a rank-poly-sig def that needs routing) are routed
    at the def-call boundary with placeholders typed from the callee's declared
    formal param types (mirrors build's host-lane call of a sig-typed compiled
    fn). Per-arg fallback to the arg expr's static type (covers rank-poly
    formals / the closure-alias corner). Deterministic ladder: not-lowerable →
    `Ok(None)` → ordinary interpretation → Site A inside; `NamedAxisRouteError`
    splits `NotLowerable` (recoverable) from `Fatal` (terminal). No silent
    fallback: terminal failures emit a targeted chelis#338 diagnostic.
  - Shared core `route_named_axis_expr` (host-only-builtin guard, lower,
    strict DAG eval with load closure over staged → tensor_bindings → frame
    bindings, pack with precision from root `output_type`,
    `pre_resolve_top_level_value_refs` for lazy top-level refs).
  - `unwrap_declared_scalar_return`: routed `-> f32` defs come back rank-0;
    converted to a host scalar via `ScalarBits::from_f64_as`.
  - **`binding_types` plumbing**: `EvalContext.binding_types:
    HashMap<String, Option<Expr>>` mirrors `bindings` lifecycle exactly
    (closure apply / let / match save-swap-restore). `RuntimeValue::Closure`
    gained `param_types: Vec<Option<Expr>>` (from annotated fn params).
    Explicit `None` markers mask same-named top-level types (shadow safety).
    `eval_let` records the bound expr's static type.
  - **Pipe threading**: `eval_pipe` threads the head's static type through
    stages via `pipe_stage_output_type` (Identity-class builtins pass the type
    through — uses `chelis_types::shape_class` — def stages yield declared
    return types; bare-var stages handled). `apply_resolved_callable` →
    `apply_resolved_callable_with_arg_types` (old name delegates).
  - New helpers: `REDUCTION_BUILTIN_NAMES`, `app_reduces_named_axis`,
    `scan_expr_for_named_axis_reduction`, `collect_var_names`,
    `strip_type_wrappers`, `declared_tensor_type_for_value` (zips declared
    dims with runtime shape; rejects `d-rank`), `param_decl_type_expr`,
    `param_type_expr_at` (refactored out of `param_precision_at`).
- `crates/chelis-types/src/lib.rs` — one line: re-export `ShapeClass`,
  `shape_class` (needed for pipe Identity threading).
- `crates/chelis-cli/tests/rank_poly_tier3.rs` — eval helpers
  (`eval_stdout`, `eval_stderr_expecting_failure`,
  `assert_eval_agrees_with_backend`); agreement folded into
  `named_reduce_builds_and_runs_nonsquare_at_ranks_2_3_4`;
  `eval_resolves_named_axis_issue_repro` (pins exact single-line output —
  note: single-root eval prints WITHOUT the `name = ` prefix);
  `concrete_rank_named_reduce_eval_matches_backend` (sum/max/min/prod;
  argmax/argmin EXCLUDED, see backend bug below);
  `named_axis_eval_parity_corners` (one program: List-param def, top-level
  direct reduce, let-block, closure alias, in-def pipe, root pipe with
  Identity stage, scalar return, grad-over-named-reduce);
  `pipe_rewriting_stage_then_named_reduce_is_a_pinned_gap` (see below);
  stale L304-313 comment updated.
- `spec/design/rank_polymorphism.md` — "Eval support (chelis#338, FIXED)" note.
- `spec/design/rank_polymorphism_tier3_followups.md` — **untracked, needs
  `git add`** — #338 marked DONE; spike verdict recorded (#340 lanes diverge,
  stays separate).

### Probe matrix (all verified by hand against build+run binaries)

p1 List-param def ✓, p2 top-level `sum(y, seq)` ✓, p3 let-block ✓, p4 in-def
pipe ✓, p5 root pipe ✓, p6 `g = use2; g(x)` alias ✓, p7 `-> f32` scalar ✓,
p9 `grad(total)` ✓ (`[1,1,1]` both lanes), p10 `y |> relu |> sum(seq)` ✓.
Known pinned gap: `y |> permute(1,0) |> sum(seq)` — build green, eval declines
with the targeted chelis#338 diagnostic (non-Identity stages drop the threaded
type). `grad` over a tensor-output reduce def is check-rejected (scalar rule).

## What REMAINS (in order)

1. **`cargo fmt` was NOT run** on the new code (the command was interrupted).
   Run `cargo fmt --all` first; expect drift in runtime.rs/rank_poly_tier3.rs.
2. **Full gate**: `python3 scripts/gate.py` (workspace build, clippy
   `--workspace --all-targets -D warnings` — per memory, per-crate clippy
   misses lints — fmt check, `chelis lint --check .` (§8.6 no-em-dash applies
   to Rust string literals; new literals were written without em-dashes but
   verify), `cargo nextest run --workspace --profile ci`). Reap stale
   cargo/rustc processes first. One `LEAK` was observed on
   `concrete_rank_named_reduce_clean` in a local nextest run (status still
   PASS; likely transient child-process reaping — re-check in the gate).
3. **Fresh-context red team** (`redteam-exec` skill; CLAUDE.md protocol:
   clean stale subagents, spawn a fresh local subagent that RUNS code).
   Suggested targets: shadowing corners of `binding_types` (local let
   shadowing a top-level of the same name with/without annotations), match-arm
   pattern bindings feeding reductions, library (chelis-std) defs in the
   routed universe, `chelis test` path (`Std.Test` asserts over routed
   results), precision-poly sigs through `declared_tensor_type_for_value`,
   double-effect risk on the rung-1→descend ladder (args must evaluate exactly
   once — they do: values are reused), the pinned permute-pipe gap, and
   eval-vs-backend on a larger composed program.
4. **Fresh-context code review**, then commit + PR "Fixes #338".
   NO AI-authorship trailers (commit hook rejects). Don't commit
   `packages/chelis-std/reef.lock`. Rebase on `main` if anything landed.
   macOS Smoke is the authoritative CI oracle; Linux jobs flake on disk-full.
5. **File a NEW issue: argmax/argmin C-backend output bug** (found while
   probing, orthogonal to #338 — reproduces with a plain int axis):
   `def am(x: &tensor[batch, seq, f32]) -> tensor[batch, int64] = argmax_reduce(x, 1)`
   → backend prints `data=[1065353216.0, 0.0]` (f32 bit-pattern of 1.0
   reinterpreted); eval correctly prints `[1.0, 0.0]`. This is why
   argmax/argmin are excluded from the tier-3 agreement test (comment in the
   test points here).
6. Optional cleanup: the #345 bisect agent left worktree
   `.claude/worktrees/agent-a5859ff13513aa89f` (contains an untracked `.venv`
   symlink) — `git worktree remove --force` it. Probe scratch lives in
   `/tmp/probe338/` and `/tmp/ice345/`.

## Issue #345 assessment (separate question, COMPLETE — not yet posted)

Verdict: **NOT rank polymorphism.** Bisect (sig-form repro, build per commit):
v0.7.23 PASS → ... → `207a9a2` (#327) PASS → **`1fa71e7` (PR #326, the #319
precision-poly attention grad fix) FIRST BAD** → #286/#337 inherit the
failure. Mechanism: #326's `annotate_fn_children` seeds separate-`sig` def
bodies with fresh checker dim vars (`dN`) in node type metadata; during grad
inlining `type_from_meta` applies precision/rank but NOT dim substitutions, so
relu's backward `Const 0.0` mask keeps `Named("dN", None)`; nothing rebinds
it (call-site `tensor_dim_substitutions` is keyed on the formal's user-facing
dim names, not the checker-renamed `dN`) and the old guard (`dag.rs:1198`,
from PR #9) panics. 0.7.23 passed by ACCIDENT (un-annotated sig bodies fell
back to the operand node's concrete type); the inline-annotated form ICEs
even on 0.7.23 (latent). Also found: `Expand { size: Sym(_) }` escapes the
guard's scan entirely. Suggested fix: the dim-var analogue of #326's own
precision-var fix (recover renamed dim vars at the call site; make
`type_from_meta` apply dim substitutions), or have the checker preserve
user-facing sig dim names. Full detail in the session transcript; the verdict
text is ready to be posted as an issue comment on #345 if wanted.
