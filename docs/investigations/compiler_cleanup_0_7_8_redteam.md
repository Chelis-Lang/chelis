# 0.7.8 Compiler Cleanup terminal red-team report

Wave 3 terminal red team for the 0.7.8 compiler-cleanup workstream
(plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`).

Run on branch `redteam/0_7_8_terminal` off `main` at commit `9996be5`
(post-PR-#95 redundant-linearity-call autofix re-coverage).

Per CLAUDE.md "Red Team Protocol", every finding here was validated by
running tests and commands. Adversarial fixtures were added under:

- `crates/chelis-types/tests/linearity_alias_destructure_adversarial.rs` (9 linearity fixtures)
- `crates/chelis-cli/tests/red_team_0_7_8.rs` (7 CLI fixtures)
- `crates/chelis-ir/tests/implicit_copy_fanout_shape_a_adversarial.rs` (6 implicit-copy v3 fixtures)
- `crates/chelis-e2e/tests/runtime_dtype_coupling_adversarial.rs` (9 CRuntime fixtures)

Total: 31 adversarial fixtures. All pass on the starting commit.

## Pre-flight + full-gate

- Pre-flight on `9996be5`: `git status` clean; `cargo build --workspace
  --all-targets` green; `cargo test --workspace` 2690 passed, 0 failed,
  176 ignored; `chelis lint --check .` exit 0.
- Full gate before push: same four commands re-run after fixture additions.
  Workspace test count rises to 2721 passed (+31 fixtures from this red
  team); zero failures; ignored count unchanged.

## §5 audit summary

Each Closed-by-0.7.8 §5 entry was checked for: (a) cited code anchor
exists at the claimed location; (b) cited regression tests exist and
pass on the starting commit. All passed.

| §5 entry | Anchor verified | Regression tests pass | Notes |
|---|---|---|---|
| `HostEval-ScalarFn-F1` | `lower.rs:2712` (lower_app), `2722` (`elems.len() < 3` guard) | 5 fixtures in `host_eval_scalar_fn_call.rs` | All pass |
| `Linearity-F1` | `linearity.rs:43` (`enum ConsumeKind`), `:59` (`struct ConsumeSite`) | 4 fixtures in `linearity_typed_consumekind.rs` | All pass |
| `Linearity-F2` | `linearity.rs::tuple_get_element_type`, `:224` (`destructure_scope_depth`) | Same file plus `linearity_aliased_consume.rs` | All pass |
| `Linearity-AliasedConsume-F1` | `linearity.rs:79` (`aliases: HashMap<...>`), `:185` (`resolve_alias_chain`) | 2 fixtures in `linearity_aliased_consume.rs` | All pass |
| `CRuntime-F32Coupling` | `runtime/src/lib.rs:50` (`unsafe trait TensorElement`), `:103-115` (5 impls), `:680` (`chelis_tensor_to_f64` dispatch) | 77 fixtures `dtype_op_matrix.rs` + 11 `host_emit_dtype_dispatch.rs` + 5 `cbackend_cast_arithmetic_composition.rs` | All pass |
| Item 1 v3 | `infer.rs` relaxed-retry path, `linearity.rs:1170` (`callee_is_observational_higher_order`) | 6 fixtures in `implicit_copy_fanout_v3.rs` | All pass |
| Items 2/3/4/5 | `prefix_namespace.rs`, `module_pascal_components.rs`, `doc_filename_convention.rs`, `no_em_dash_in_public_strings.rs` plus spec §7.1.1 / §8.3 / §8.5 / §8.6 | No regression fixtures dedicated; corpus sweep | Spec edits are canonical, not narrative |
| Item 7 | `crates/chelis-cli/src/main.rs` lint path resolution + canonicalization | 2 fixtures in `lint_path_walk_consistency.rs` | All pass |
| Item 6 (PR #95) | rule logic unchanged; coverage unlock via PR #91 upstream | 9 fixtures in `redundant_linearity_call_autofix.rs` (5 new in W5) | All pass |

The 11 intentional `*mut f32` references in `crates/chelis-runtime/`
enumerated in PR #88's audit were independently grepped and match
exactly. Zero in `chelis-backend-c/`, `chelis-backend-hip/`,
`chelis-backend-metal/`, `chelis-ir/`. Audit is honest.

## Adversarial coverage delta

### §3.1 HostEval-ScalarFn-F1

PR #80's fixtures cover bare zero-arg `result = go()` for each scalar
return type, plus a one-arg negative control. They do not cover:

- Nested zero-arg (`def outer -> i32 = inner()`).
- Three-level chain (`def deepest() = ...; def middle() = deepest(); def outer() = middle()`).
- Zero-arg call as a subexpression (`result = add(go(), 1)`).
- Zero-arg call inside another zero-arg fn's body.
- Large i64 above f32 representable range.

All six new fixtures pass. No new bugs surfaced.

### §3.2 + §3.3 Linearity

PR #83 plus PR #90's fixtures cover single-level alias, single-level
tuple-destructure, and the two-level alias bypass. They do not cover:

- 3-level and 4-level alias chains. Both pass with correct
  `UseAfterConsume` on the original source.
- Pipe-stage on alias (`y |> realize` after `y = x`). Passes with
  correct error.
- Nested destructure: `(inner, c) = pair; (a, b) = inner; realize(a);
  realize(a)`. Passes with correct error on the inner-component `a`.
- Triple-nested destructure (3 levels of tuple). Passes correctly.
- Nested destructure followed by alias then consume. Passes correctly.
- 4-level chain consume at chain tail only (positive control). Passes
  with no errors.

All nine new fixtures pass. The recursive-tuple type lookup via
`tuple_get_element_type` is durable beyond single-level destructure.

### §3.4 CRuntime-F32Coupling

PR #84's negative-coverage fixtures cover `<T>::data_ptr` mismatch for
i64-vs-f64 and a couple of other pairings. They do not cover the
complete cross-pairing matrix for `f32, f64, i64`.

Added 6 dtype-mismatch rejection fixtures:
- `f32::data_ptr` against F64 and I64 tensors.
- `f64::data_ptr` against F32 tensor.
- `i64::data_ptr` against F64, I32, BOOL tensors.

Added 3 round-trip preservation fixtures:
- i64 round-trip with value within f64 exact-int range.
- i64 above 2^53 with documented f64 precision loss (locks current
  read-path behavior so the read doesn't truncate to f32).
- f64 round-trip with a value beyond f32 representable precision
  (1.234567890123456_7 — 16 significant decimal digits).

All nine pass. No silent precision loss surfaced.

### §3.5 Implicit-copy v3 Shapes A and B

PR #91 explicitly defers two Shape A variants per the diagnosis:
- `def f(x: &T) -> T = let y = x in y` (let tail-return).
- `def f(x: &T) -> T = match ...` (match tail-return).
- `def f(x: &T) -> T = if ... then x else x` (if tail-return).

Added two adversarial fixtures pinning these as `expect_err`. If the
broader Shape A fix ships later, these assertions flip — the test
catches the change in either direction.

Added four Shape B adversarial fixtures:
- 3-grad fan-out with trailing borrow-read (the v3 PR tested 2-grad
  fan-out only).
- Mixed grad + vmap fan-out.
- `jit(f)(args)` with trailing borrow — diagnosis claims jit is NOT
  treated as observational; current behavior is recorded.
- 3-grad fan-out without trailing borrow (positive control).

All six pass. The observational-higher-order classifier covers grad
and vmap symmetrically; jit and par stay unchanged as documented.

### §3.6 Lint CLI consistency (Item 7) + §5 sibling-sweep

Confirmed PR #93 closure: `chelis lint --check .` and explicit subtree
walk produce identical `doc-filename-convention` error counts.

Confirmed the parallel `Lint-ExceptionPathRoot-F1` §5 entry still
reproduces exactly as filed:

```
chelis lint --check crates docs examples packages
```

produces 6 false-positive `surf-def-arrow-form` errors on
`crates/chelis-surf/tests/fixtures/*.ch` files (block_binding_expr.ch,
lambda.ch, match_expr.ch, operators.ch, pipe.ch, simple_def.ch) that
`chelis lint --check .` excepts. Subtree exits 1; `.` exits 0. The
§5 entry is correctly filed and not yet resolved.

### §3.7 Lint precision Items 2/3/4/5

- Item 4 (§8.5 mdBook escape): verified by creating a temp directory
  with `book.toml` at the root and a kebab-case `.md` inside `src/`.
  Lint passes inside the mdBook tree; outside it (same filename in a
  regular `docs/` directory without `book.toml`), it fails with
  `doc-filename-convention`. Escape works correctly.

- Item 5 (§8.6 docstring exclusion): verified by creating a Python
  file with em-dashes in (a) module docstring, (b) function docstring,
  (c) `print(...)` call, (d) `raise ValueError(...)` call. Only (c)
  and (d) flag. Docstrings exempt correctly.

- Items 2 and 3: corpus-level — `chelis lint --check .` on this repo
  emits 243 advisory warnings and 0 errors, matching PR #92's
  measurement. No newly-flagged correct code surfaced.

### §3.8 redundant-linearity-call autofix

PR #95's 5 new fixtures cover Shape A + Shape B sources. Re-execution
confirms 9/9 autofix fixtures pass (4 pre-existing + 5 new).

Added adversarial probe: `realize(copy(x))` autofixes cleanly to
`realize(x)`; the stripped source type-checks cleanly. Bare-pipe
trailing `... |> copy` does not match the rule's `copy(...)` call form
and is correctly left alone.

## CHANGELOG verification

The `[Unreleased]` section of `CHANGELOG.md` mentions **only** PR #95's
`redundant-linearity-call` autofix coverage. It does NOT mention:

- `HostEval-ScalarFn-F1` (PR #80) closure.
- `Linearity-F1` typed `ConsumeKind` refactor (PR #83) closure.
- `Linearity-F2` tuple-destructure detection + cascade (PRs #83 + #90)
  closure.
- `Linearity-AliasedConsume-F1` alias chain forwarding (PR #83) closure.
- `CRuntime-F32Coupling` four-PR migration (#84/#86/#87/#88) closure.
- Item 1 v3 implicit-copy Shape A + Shape B (PR #91).
- Lint precision bundle Items 2/3/4/5 (PR #92).
- Lint CLI path-walk consistency Item 7 (PR #93).

This is a documentation/audit gap. The customer-visible surface for
each of these landed in 0.7.8, but `CHANGELOG.md`'s `[Unreleased]`
section will not communicate the change set when 0.7.8 cuts. See
Finding F1 below.

## Findings

### F1 — CHANGELOG `[Unreleased]` missing 8 of 9 0.7.8 §5 closures

- **Severity:** MEDIUM.
- **Shape:** Documentation invariant violation. The 0.7.8 release will
  ship with a `CHANGELOG.md` that documents only PR #95; the other
  eight closed §5 entries are user-facing changes (silent miscompile
  fixes, linearity spec correctness, C runtime dtype safety, lint
  precision improvements) and the release notes will not convey them.
- **Verification:** `grep -E "0\.7\.8|HostEval|F32Coupling|TensorElement|ConsumeKind|alias.*consume|tuple.*destructure|lint precision|path-walk|dtype.*coupling" CHANGELOG.md` returns zero matches in the `[Unreleased]` section. The release entry for 0.7.7 references unrelated content.
- **Recommended disposition:** **follow-on PR** before the 0.7.8 tag.
  The release-cut PR (whichever wraps `[Unreleased]` -> `[0.7.8] -
  YYYY-MM-DD`) must add CHANGELOG entries for the 7 silent gaps
  (HostEval, Linearity-F1, Linearity-F2, Linearity-AliasedConsume-F1,
  CRuntime-F32Coupling, Item 1 v3, Items 2/3/4/5, Item 7). Cite the
  closing PR for each.
- **Does not block the merge of this red-team branch**; it blocks the
  0.7.8 release-cut commit.

### F2 — Broader Shape A scope not filed as a §5 follow-on

- **Severity:** LOW.
- **Shape:** PR #91's diagnosis explicitly notes that `let/if/match`
  tail-return Shape A is OUT of v3 scope and a §5-candidate follow-on,
  but I find no §5 entry for it in `docs/gap_synthesis.md`. The
  diagnosis says "Out of v3 scope; document as a candidate for a
  follow-on entry" but no entry was opened.
- **Verification:** `grep -i "shape.a.broader\|Shape-A-Broader\|Let-Tail-Return\|broader.shape.a" docs/gap_synthesis.md` returns zero hits. The implicit-copy v3 diagnosis at
  `docs/investigations/implicit_copy_fanout_v3_diagnosis.md` lines
  117-122 names three deferred body shapes (`bare var`, `let y = x in
  y`, `if/match tail`) and calls them §5 candidates.
- **Recommended disposition:** **new §5 entry** — `ImplicitCopy-ShapeA-Broader-F1` or similar.  This is an orchestrator filing decision, but the unfiled candidate is a real gap. Adversarial fixtures
  `shape_a_let_tail_return_currently_rejects` and
  `shape_a_if_tail_return_currently_rejects` pin the current reject
  behavior so a future fix lands with regression coverage.

### F3 — `host_eval_zero_arg_in_subexpression` and `host_eval_zero_arg_inside_zero_arg_body` not in PR #80's fixture set

- **Severity:** LOW.
- **Shape:** Coverage gap. The W3 fix is exactly correct (one-line
  arity guard), but the regression fixtures only exercise the
  top-level `result = go()` shape. The same guard fires for nested
  forms (`add(go(), 1)`) and zero-arg-inside-zero-arg
  (`def go() = add(helper(), 3)`). A future regression that re-tightens
  the guard would still pass the existing fixtures.
- **Verification:** Inspected `host_eval_scalar_fn_call.rs` — all five
  fixtures use the top-level shape.
- **Recommended disposition:** **already addressed in this PR** via
  the two new fixtures in `crates/chelis-cli/tests/red_team_0_7_8.rs`.
  No further action needed; these are now pinned regression coverage.

### F4 — Lint-ExceptionPathRoot-F1 reproducibility confirmed

- **Severity:** LOW-MEDIUM (matches the §5 entry's filing).
- **Shape:** The §5 entry is correctly filed; the bug reproduces
  exactly as described. `chelis lint --check crates docs examples
  packages` produces 6 false-positive `surf-def-arrow-form` errors on
  `crates/chelis-surf/tests/fixtures/*.ch` that `chelis lint --check .`
  excepts. Subtree invocation exits 1; `.` invocation exits 0.
- **Verification:** Direct `diff` of two lint outputs (see §3.6 above).
- **Recommended disposition:** **no new action**; the §5 entry already
  captures it. Pinned regression fixture is the
  `lint_subtree_invocation_matches_dot_for_doc_filename_convention`
  test in `crates/chelis-cli/tests/red_team_0_7_8.rs` (note: it tests
  the doc-filename-convention closure, not the exception-path-root
  bug; the exception-path-root bug is documented here as a separate
  finding that the §5 entry already covers).

### F5 — Adversarial corpus surveys confirm zero new destructure errors

- **Severity:** none (positive confirmation).
- **Shape:** PR #90's cascade survey of `examples/` and `packages/`
  reported zero in-tree tuple-destructure violations to clean up; the
  warnings channel was removed and surfaced violations route through
  `errors` directly. Re-survey on the current commit: `chelis check`
  against `examples/` and `packages/chelis-std/src/` produces pre-existing
  unrelated errors (e.g., `unbound variable: dot`, `DimensionMismatch`
  in `examples/`) but **no** `UseAfterConsume on destructured` errors.
  The destructure-cascade is honestly closed.

## Pre-flight + full-gate result

- Pre-flight on `9996be5`: PASS (clean status, build green, 2690 tests
  pass, lint exit 0).
- Full gate after adding 31 adversarial fixtures: PASS
  (`cargo build --workspace --all-targets` green, `cargo test
  --workspace` 2721 passed 0 failed 176 ignored, `cargo clippy
  --workspace --all-targets -- -D warnings` clean, `cargo fmt --all --
  --check` clean, `chelis lint --check .` exit 0).

## Total findings

- 5 total findings.
- 1 material (MEDIUM): F1 CHANGELOG gap (recommend follow-on PR before
  0.7.8 tag).
- 1 LOW-MEDIUM: F4 Lint-ExceptionPathRoot-F1 — already correctly filed
  as §5; no new action.
- 1 LOW: F2 Broader Shape A scope not filed as §5 — recommend new §5
  entry.
- 1 LOW: F3 HostEval coverage thin — addressed in this PR via fixtures.
- 1 positive confirmation (F5).

## Disposition summary

| Finding | Severity | Disposition |
|---|---|---|
| F1 CHANGELOG `[Unreleased]` missing 8 closures | MEDIUM | **Follow-on PR before 0.7.8 tag** |
| F2 Broader Shape A scope unfiled | LOW | **Recommend new §5 entry** (`ImplicitCopy-ShapeABroader-F1` or similar) |
| F3 HostEval fixture coverage thin | LOW | **Addressed in this red-team PR** via new fixtures |
| F4 Lint-ExceptionPathRoot-F1 reproduces | LOW-MEDIUM | **§5 already filed**; no action |
| F5 destructure cascade clean | none | positive confirmation |

The orchestrator decides whether to file F2 as a §5 entry (per the
`§5 filing is orchestrator decision` standing rule). This report
recommends it; the diagnosis at
`docs/investigations/implicit_copy_fanout_v3_diagnosis.md` already
documents the deferred candidate.
