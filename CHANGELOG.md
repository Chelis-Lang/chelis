# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed - backend integration test suite parsimony pass

Trimmed redundant gcc/codegen invocations across the backend-crate
integration test cluster without losing coverage.

- `crates/chelis-backend-c/tests/simd_reductions.rs`: collapsed the
  30-test 5-op-by-6-size matrix into 5 tests (one per reduction op).
  Each test now emits one C program that sweeps every size internally,
  cutting gcc compile-and-run invocations from 30 to 5 while still
  checking scalar-vs-SIMD agreement at every AVX2 lane-boundary size.
- `crates/chelis-backend-hip/tests/perf_f1_strided_batched_default.rs`:
  removed; its two Perf-F1 structural acceptance cases were moved into
  `red_team_w5_strided_batched.rs` (the keep-by-default strided-batched
  lock file), since the W5 adversarial cases did not subsume the rank-3
  literal-stride dispatch and broadcasted-lhs helper-loop fallback.
- `crates/chelis-backend-metal/tests/codegen_structure.rs`: removed the
  three `wsm1_{f32,f16,bf16}_matmul_routes_to_*` tests, which duplicated
  the per-dtype routing assertions in
  `dtype_matrix.rs::matmul_{f32,f16,bf16}_*` (the canonical per-dtype
  home). The non-routing `wsm1` tests (subgraph folding, M/N/K uniform
  packing) stay.
- `crates/chelis-backend-hip/tests/gpu_correctness.rs`: removed
  `ws_a3_hip_admits_bf16_matmul_at_codegen`, a "does not panic" subset
  of `ws_a3_bf16_f16_matmul.rs::bf16_matmul_default_accumulator_emits_bf16_gemm_wrapper`,
  which additionally asserts the emitted bf16 GemmEx wrapper and link
  flag.

### Changed - phase3 integration test parsimony pass

Trimmed redundant tests from the `chelis-cli` `phase3*` integration
test cluster. Deleted
`phase3j_pre_oracle_build_path_repros_uniform_like_seed_succeeds` from
`crates/chelis-cli/tests/phase3j_pre_std.rs`: it asserted the seed=7
`uniform_like` vector, a strict subset of
`phase3j_pre_oracle_build_path_repros_uniform_like_seed_distinct_seeds_differ`,
which builds seed 7 and seed 42 in one program and asserts the
identical seed=7 vector. Deleted
`reef_std_parquet_module_resolves_and_type_checks` and
`reef_std_parquet_write_resolves_and_type_checks` from
`crates/chelis-cli/tests/phase3g_io.rs`: both were strict subsets of
`reef_std_parquet_module_builds_cleanly`, which imports both
`read_parquet` and `write_parquet`, type-checks a `def` for each, and
also runs `chelis build`. No invariant coverage was lost.

### Changed - e2e parsimony pass (ir/types/api/e2e/effects test cluster)

Trimmed redundant integration-test boilerplate in the
chelis-ir / chelis-types / chelis-compiler-api / chelis-e2e cluster
without dropping any pinned invariant.

- Deleted `crates/chelis-types/tests/rt_lin_div_diagnosis.rs`, a
  `println!`-only Phase G' investigation scaffold with zero
  assertions. The Phase G' linearity behavior it probed stays pinned
  by `phase_e_linearity_with_context.rs`.
- Deleted two subset tests in `crates/chelis-e2e/tests/bench_phase1e.rs`
  (`bench_phase1e_missing_pytorch_interpreter_emits_structured_skip_report`,
  `bench_phase1e_hidden_hip_device_emits_structured_skip_report`); both
  skip-reason assertions are already covered by
  `bench_phase1e_linreg_smoke_emits_structured_json`.
- Folded `parity_pair_1/2/3` in `phase_f_with_context.rs` and the three
  `eval_in_context_matches_prepare_eval_*` tests in
  `phase_g_compiled_context.rs` into table-driven tests with identical
  coverage.
- Deleted two looser duplicates in `rt1_adversarial.rs`
  (`cast_scalar_to_u8_rejected_at_check_time`,
  `out_of_i32_range_literal_default_behavior`); the exact-diagnostic
  versions in `ws_a0_rt1_unsigned_rejection.rs` and
  `ws_a0_rt1_int_overflow.rs` assert a strict superset.
  `d1_diagnostic_mentions_i64_suffix_and_cast` is kept because it
  exercises an int64-return-position input no `ws_a0_*` test covers.
- `#[ignore]`-gated the wall-clock perf-ratio test
  `rt_g_compose.rs::g5_cold_path_overhead_at_most_2x_monolithic`,
  separating it from the correctness pass per the CLAUDE.md
  perf-vs-correctness rule, and documented it in `docs/manual_gates.md`.
### Changed - e2e parsimony pass over the wsa/wsc/rt3a integration test cluster

Trimmed redundant coverage in the `chelis-cli` `wsa*` / `wsc*` /
`rt3a*` integration test cluster without losing any pinned invariant.
Three cross-cluster triplicate tests (the WS-C-blocker reproducer and
two integer-matmul-rejection copies) were removed because the
invariant is already pinned by keep-by-default regression locks in
`rt3a_adversarial.rs`, `rt4_adversarial.rs`, and
`wsa5_precision_polymorphism.rs`. The WS-A6 and WS-A7 dtype-matrix
re-tests were consolidated into their owning files
(`wsa6_def_annotation_desugar.rs`, `wsa7_bareref_return_inference.rs`),
which now carry the full arithmetic-dtype matrix. Near-identical tests
in `wsc_v3_stdlib_finish.rs` and `wsc_stdlib_generalization.rs` were
collapsed into table-driven tests with identical coverage, and two
literal-duplicate fixtures in `rt3a_adversarial.rs` were merged into
one test that keeps the union of both assertions.
### Added - test toolchain footgun guards

Three standing guards that turn recurring CI footguns into
enforcement. See `docs/investigations/test_toolchain_guards_design.md`.

- Test-timing budget: a `ci` nextest profile (`.config/nextest.toml`)
  writes per-test JUnit timing XML; `scripts/test_timing_check.py`
  flags tests that regressed past their committed budget or that are
  new and over the absolute ceiling. Thresholds are config
  (`scripts/test_timing_config.json`,
  `scripts/test_timing_baseline.json`); the baseline is hand-curated
  and regenerated explicitly via
  `python3 scripts/test_timing_check.py --update-baseline`. CI runs it
  as an informational, non-failing step.
- Em-dash visibility: `chelis lint --check` now buckets output by
  severity so blocking errors print last under a delimited header plus
  a summary line instead of being buried in advisory-warning noise.
  Investigation confirmed the `no-em-dash-in-public-strings` (§8.6)
  rule already catches em dashes in raw strings, `format!` arguments,
  and multi-line strings; the recurring failure was visibility, not a
  parser gap.
- CI-gate parity: `scripts/gate.py` is the single source of truth for
  the per-PR developer gate; `.github/workflows/ci.yml` calls it per
  stage and `scripts/test_gate.py` asserts the workflow inlines no
  gate command the script does not produce. `AGENTS.md` now points at
  `python3 scripts/gate.py` and is corrected to run `cargo nextest`
  and include `chelis lint --check .`.

## [0.7.9] — 2026-05-14

### Fixed - chelis check advisory warnings now match chelis lint --check

`chelis check` (the workflow `chelis reef build` invokes on `.ch`
files) flooded with `redundant-linearity-call` and
`prefer-pipe-operator` advisory warnings that `chelis lint --check`
already suppressed. The two emit paths had diverged: `cmd_lint`
applied `should_suppress_unfixable_violation` before printing kept
violations, but `emit_advisory_lint_warnings_for_file` (the path
`chelis check` invokes) applied only the path-glob exception filter.
For rules opted into `check_mirrors_fix=true`, a non-actionable
warning whose autofix the typed-pipeline gate rejects therefore still
fired through `chelis check`. Fixed by threading
`should_suppress_unfixable_violation` into the advisory-emit path so
both code paths apply the same gate. A genuinely-redundant `copy()`
on an owned tensor still warns (the autofix is safe). Closes
`Lint-CheckMirrorsFixAdvisoryEmitLeak-F1`. Regression tests at
`crates/chelis-cli/tests/red_team_0_7_9.rs` (`lp_leak_a`, `lp_leak_b`,
`lp_leak_fix_*` positive control).

### Fixed - lint workspace-root detection uses a real Cargo workspace probe

`detect_lint_workspace_root` was implemented as
`canonicalize(current_working_directory)` with no workspace probe, so
the workspace-rooted exception matching shipped in the previous fix
held only when `chelis lint` was invoked from the workspace root.
Running `chelis lint --check .` from a subdirectory, or `chelis lint
--check /abs/workspace` from an unrelated directory, re-surfaced the
false-positive `surf-def-arrow-form` errors. Fixed by detecting the
workspace root with `cargo locate-project --workspace`, Cargo's own
canonical workspace-locating probe, run with its working directory
set to the lint target's directory rather than the process CWD. When
the targets are not inside any Cargo workspace, no workspace-rooted
exception glob can apply, so violations pass through unfiltered
(linting a loose file outside a workspace remains supported). The
same detection is now shared by `cmd_lint`, the advisory-emit path,
and the build-time style gate. Closes
`Lint-WorkspaceRootCwdAssumption-F1`. Regression tests at
`crates/chelis-cli/tests/red_team_0_7_9.rs` (`le_leak_a`,
`le_leak_fix_sibling_path_invocation_matches_workspace_root_invocation`).

### Fixed - type checker rejects divergent declared dimension parameters

`chelis check` previously accepted a function whose declared return type
and body type differed in dimension *identity* but matched in dimension
*rank*. A definition like
`def f[n, m](x: &tensor[n, f32], y: &tensor[m, f32]) -> tensor[n, f32] = y`
type-checked with `score=1` and then evaluated with a runtime shape
mismatch (HIGH severity silent miscompilation, red-team finding
SR-LEAK-A). Two independent leak paths are closed:

- `types_structurally_equal` (the Shape A relaxed-retry guard) compared
  tensor dimensions by rank only. It now compares dimension *identity*:
  two dims match iff same concrete name, same literal, or the same dim
  variable. `Dim::Wildcard` still matches anything. Distinct dim
  variables (`n` vs `m`) no longer satisfy the structural guard.
- A plain owned-tensor body bypassed the relaxed-retry entirely: the
  post-body signature unification collapsed two distinct declared dim
  parameters via free dim-variable unification. A new post-body rigidity
  check, `check_declared_dvars_rigid`, flags this `Var->Var` collapse
  (and the previously-handled `Var->Lit` pin) as a `DimensionMismatch`.
  Declared dimension parameters are rigid within the def body.

Closes `TypeCheck-FreeDimVarUnification-F1` (red-team finding
SR-LEAK-A, `docs/investigations/terminal_redteam_0_7_9.md`). Diagnosis:
`docs/investigations/typecheck_dim_identity_diagnosis.md`. Regression
tests at `crates/chelis-cli/tests/red_team_0_7_9.rs` (the SR-LEAK-A
fixtures, divergent-dim negatives plus same-dim positive controls for
both paths). Spec completion: `spec/04-type-system.md` §4.4 now states
the dim-parameter rigidity rule the fix enforces.

### Fixed - redundant-linearity-call false-positive on copy(borrow)

The `redundant-linearity-call` advisory warning previously fired on
`copy()` calls where stripping the call would leave the program with
a borrow-to-owned type mismatch the implicit-copy inserter cannot
bridge. The CLI driver's typed-pipeline gate correctly suppressed the
`[fix]` marker, but the warning itself still fired, leaving 137+
false positives in Nautilus `src/linalg.ch` against 0.7.8. Fixed by
opting the rule into `check_mirrors_fix=true` so the CLI driver's
`should_suppress_unfixable_violation` helper suppresses the warning
when the typed-pipeline gate rejects the proposed strip. Closes
`Lint-RedundantLinearityCopyOnBorrowWarn-F1`. Regression tests at
`crates/chelis-lint/tests/redundant_linearity_call_autofix.rs` (F10,
F11 negative control).

### Fixed - redundant-linearity-call false-positive on 2-arg list primitives in pipe form

The pipe form `xs |> drop(n)` puts a literal single-arg `drop(n)` in
the source text, but semantically it is the 2-arg list-drop with the
first argument piped in. The rule's syntactic
`has_single_top_level_argument()` accepted the pipe-form shape and
fired a redundant-linearity-call warning, which the user could not
satisfy without breaking the program (Coral reported the pattern
against 0.7.7 in `src/internal/hamt.ch` and `src/internal/window.ch`
and worked around it by writing the 2-arg form directly). Closed
transparently by the same `check_mirrors_fix` opt-in above: the
typed-pipeline gate rejects the strip `drop(n) -> n` and the warning
is suppressed. Closes `Lint-PreferPipeRedundantLinearityPair-F1`.
Regression tests at
`crates/chelis-lint/tests/redundant_linearity_call_autofix.rs` (F12,
F13 negative control).

### Fixed - lint exception path matching anchored at workspace root

`chelis lint --check crates docs examples packages` produced 6
false-positive `surf-def-arrow-form` errors against
`crates/chelis-surf/tests/fixtures/*.ch` that `chelis lint --check .`
correctly excepted. CI invokes the `.` form so the gate was not
broken, but developers linting sub-trees saw spurious errors. Root
cause: `apply_exceptions` at
`crates/chelis-lint/src/exceptions.rs::is_excepted`
strip-prefixed the violation path against the walk-target root rather
than a shared workspace root, so workspace-rooted exception patterns
silently failed to match under sub-directory walks.

Fixed by adding a `workspace_root` parameter to `apply_exceptions`
and `is_excepted`, distinct from the walk root, and anchoring
prefix-stripping against it. The CLI detects the workspace root by
canonicalizing the current working directory at the boundary,
reusing the canonicalize-at-CLI-boundary pattern PR #93 established
for walk targets. No walk-up filesystem search is introduced (per
`feedback_no_walkup_filesystem_detection.md`).

Updates four call sites:
`cmd_lint`, `apply_lint_fixes`,
`emit_advisory_lint_warnings_for_file`, and
`style_gate::run_lint_for_single_file`. Closes
`Lint-ExceptionPathRoot-F1`. Regression tests at
`crates/chelis-cli/tests/lint_path_walk_consistency.rs`
(`lint_cli_exception_pattern_matches_under_subtree_and_cwd_walks`
and `lint_cli_exception_pattern_matches_under_multi_subtree_walk`)
plus a unit test in
`crates/chelis-lint/src/exceptions.rs::tests::apply_exceptions_anchors_on_workspace_root_not_walk_target`.

### Fixed - implicit-copy Shape A covers let/if/match tail-position returns

PR #91's W4-A relaxed-retry in `shape_a_relaxed_return`
(`crates/chelis-types/src/infer.rs`) only accepted a bare
`(fn (params...) (var x))` body, so

```surf
def f[n](x: &tensor[n, f32]) -> tensor[n, f32] = { y = x; y }
def g[n](c: bool, x: &tensor[n, f32]) -> tensor[n, f32] = if c then x else x
def h[n](c: Choice, x: &tensor[n, f32]) -> tensor[n, f32] =
  match c with { | Left => x | Right => x }
```

all surfaced `def 'f' body doesn't match declared signature` even
though the tail expression of every desugared body is the same
bare-var ref the v3 fix already accepts. Added
`descend_to_tail_var` next to `shape_a_relaxed_return`; the walker
descends through `(let bind body)`, `(if cond then_e else_e)`, and
`(match scrutinee arm ...)` and returns `Some(name)` only when every
sibling branch resolves to the same bare-var name. The structural
type-equality check and relaxed-type construction are unchanged, so
the broader gate stays as conservative as PR #91's. Closes
`Linearity-ShapeABroadReturn-F1`. Regression fixtures at
`crates/chelis-ir/tests/implicit_copy_shape_a_broader_return.rs`.

## [0.7.8] — 2026-05-13

### Fixed - host-eval scalar zero-arg fn-call silent miscompilation (#80)

`def go -> f32 = 7.5; result = go()` previously evaluated `result`
as `0.0` via `chelis eval --file` instead of `7.5`. Root cause was
an off-by-one arity guard in IR lowering: `LowerCtx::lower_app` at
`crates/chelis-ir/src/lower.rs:2712` rejected zero-arg
`(app {meta} (var fn-name))` forms (3 elements: tag + meta + fn)
via `if elems.len() < 4`, emitting `RiscOp::Const { value: 0.0 }`
before any callable resolution. Fixed by relaxing the guard to
`if elems.len() < 3`; the downstream arms already handle empty
arg slices correctly. Closes `HostEval-ScalarFn-F1`. Regression
tests at `crates/chelis-cli/tests/host_eval_scalar_fn_call.rs`
cover f32/f64/i64/bool zero-arg return types plus a one-arg
negative control.

### Changed - linearity checker uses typed `ConsumeKind` discrimination (#83)

`ConsumeSite` now carries an `enum ConsumeKind { Aliasing,
Structural }` field. The previous string-prefix check
(`descriptor.starts_with("binding ")`) at
`crates/chelis-types/src/linearity.rs::read_or_error` is replaced
by `matches!(site.kind, ConsumeKind::Aliasing)`. All eight
consume-site producers set `kind` explicitly. Closes
`Linearity-F1`.

### Fixed - alias-consume linearity bypass (#83)

`let y = x; let z = realize(y); add(x, z)` previously passed
`chelis check` silently because `y`'s linearity state was tracked
per-bound-name, not per-underlying-value. Fixed by adding a
`LinearScope.aliases: HashMap<String, Vec<Option<String>>>`
parallel to `bindings` and `types`. `Structural` consumes
forward through `resolve_alias_chain` to the underlying source
name's scope entry; multi-level chains (`let z = y; let y = x;
consume(z)`) walk to `x`. Closes `Linearity-AliasedConsume-F1`.
Regression tests at
`crates/chelis-types/tests/linearity_aliased_consume.rs`.

### Fixed - tuple-destructure linearity false-negative now errors (#83 + #90)

`let (a, b) = pair; realize(a); realize(a)` previously passed
silently because `chelis-surf` desugar synthesized `__chelis_tmp_N`
intermediaries without type metadata, making
`expr_is_owned_linear` return false and skipping the entire
destructure chain. Fixed in PR #83 via a local
`tuple_get_element_type` helper that indexes into the underlying
tuple var's `t-tuple` scope-type at linearity-check time. PR #90
swept the production corpus (zero warnings surfaced), then
removed the temporary `LinearityInfo::warnings` cascade channel
and routed destructured-component use-after-consume directly
through `errors`. Closes `Linearity-F2`. Regression tests at
`crates/chelis-types/tests/linearity_typed_consumekind.rs` and
`linearity_aliased_consume.rs`.

### Changed - C runtime tensor data is dtype-aware (#84 + #86 + #87 + #88)

`chelis_tensor.data` is now `*mut u8`, accessed through a new
`TensorElement` trait with checked `data_ptr`, unchecked
`data_ptr_unchecked` (debug_assert), and a default `fill` lifting
the cast-then-loop pattern from `chelis_fill_i64`. Trait impls
for `f32, f64, i32, i64` (`bool` and `i32` route through `f32`
internally because the runtime stores them f32-encoded today —
tracked separately as `CRuntime-BoolStorage-F1` and
`CRuntime-I32Storage-F1`). `CHELIS_*` constants promoted to
`pub const`.

Migration shipped in four PRs: PR #84 introduced the trait +
struct change + first anchor migrations; PR #86 migrated 31
runtime call sites across 14 ops; PR #87 migrated 6 host_emit
code-generation sites; PR #88 added 22 multi-op composition
fixtures and the workstream-wide sibling sweep audit.
**110 dtype-coupling fixtures lock the migration**:
`crates/chelis-e2e/tests/dtype_op_matrix.rs` (77),
`crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs` (11),
`crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs`
(5), plus the four 0.7.6 surface-fix regression locks.
Sibling-sweep audit: 11 intentional `*mut f32` references remain
in `crates/chelis-runtime/` (each enumerated with justification);
0 in `crates/chelis-backend-c/`, HIP, Metal, IR. Closes
`CRuntime-F32Coupling`. Audit note:
`docs/investigations/c_runtime_dtype_coupling_workstream_audit.md`.

### Fixed - implicit-copy inserter handles return-position borrow-to-owned and grad/vmap fan-out (#91)

Two fan-out shapes that previously failed:

- **Shape A** (return-position borrow-to-owned): `def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = x`
  previously failed with a type mismatch. Fixed at
  `crates/chelis-types/src/infer.rs::check_top_level` via a
  relaxed-retry path in the def-body unify that recognizes
  body-as-bare-var with `Ref(T)` inferred and `T` declared.
  Limited to the bare `(var x)` body shape; broader return-position
  coercions (`let`/`if`/`match` tails) are tracked as a follow-on.
- **Shape B** (grad/vmap-callee fan-out): `dw = grad(f, wrt=w)(args); db = grad(f, wrt=b)(args)`
  previously failed with `UseAfterConsume` on the fanned-out args.
  Fixed at
  `crates/chelis-types/src/linearity.rs::arg_is_borrowed` via a
  new `callee_is_observational_higher_order` helper that treats
  every arg of a `(grad ...)` or `(vmap ...)` callee as borrowed
  rather than consumed. Covers piped grad/vmap stages too
  (pipe-stage path routes through the same `arg_is_borrowed`).

Closes Item 1 v3 of the implicit-copy inserter rollout (after
PR #29 v1 and the v2 fix-up). Six fixtures at
`crates/chelis-ir/tests/implicit_copy_fanout_v3.rs`.

### Changed - lint precision: math/ML prefixes, ecosystem allowlist, kebab-case docs, docstring em-dash (#92)

Four lint-rule precision fixes:

- **§7.1.1** `prefix-namespace` accepts a curated math/ML
  well-known prefix list (`exp_`, `log_`, `sin_`, `cos_`, `tan_`,
  `sqrt_`, `lin_`, `std_`, `var_`, `min_`, `max_`, `sum_`,
  `mean_`, `relu_`, `gelu_`, `silu_`, `tanh_`, `sigmoid_`) in
  addition to module-domain-derived prefixes. Module-level
  allowlist annotations are also supported.
- **§6.3** `module-pascal-components` `KNOWN_SINGLE_WORDS`
  extended by 18 entries (4 from the hello-chelis report —
  `Linearity`, `Hypothesis`, `Integration`, `Optimize` — plus a
  math/ML sweep adding 14 more). `Vocabulary-F2` structural
  closure (auto-generation from spec §2.6) remains pending.
- **§8.3** `doc-filename-convention` accepts kebab-case in
  mdBook-rooted narrative-doc trees (detected by walking
  ancestors for `book.toml`) and accepts hyphens when the
  filename stem matches a workspace Cargo `[package].name`
  (e.g., `c-earchin.md` matches package `c-earchin`).
- **§8.6** `no-em-dash-in-public-strings` excludes Python
  docstrings (first non-whitespace triple-quoted string on its
  line). Mid-line triple-quoted strings (`print("""…""")`)
  remain in scope.

Spec text updated at `spec/01-nomenclature.md` §7.1.1, §8.3, §8.5,
§8.6.

### Fixed - lint CLI produces identical output for CWD walk and explicit-path walk (#93)

`chelis lint --check src/ tests/ verify/ scripts/ docs/`
previously returned a different error count than
`chelis lint --check .` because
`doc-filename-convention::classify_doc` did substring matching on
the walker's path output. `walkdir::WalkDir::new(root)` yields
paths prefixed with the root passed in, so `.` produced
`./docs/file.md` (substring `/docs/` matched) while `docs/`
produced `docs/file.md` (no leading slash, no match). Fixed by
canonicalizing user-supplied target paths to absolute paths at
the CLI boundary in `cmd_lint`. Companion fix to
`crates/chelis-lint/src/walker.rs::is_skip_dir` ensures the
worktree-scoped skip filter never matches the walk-root entry
itself.

Sibling-sweep finding: `chelis-lint/src/exceptions.rs::is_excepted`
has parallel path-spelling sensitivity in the opposite direction
(`crates/chelis-surf/tests/fixtures/*.ch` patterns fail when the
CLI walks a sub-directory). Filed as `Lint-ExceptionPathRoot-F1`.

### Added - `redundant-linearity-call` autofix covers implicit-copy v3 shapes (#95)

The implicit-copy fan-out v3 fix (Shape A borrow-to-owned at return
position and Shape B grad/vmap fan-out across observational
higher-order calls) closes a coverage gap for the
`redundant-linearity-call` autofix. The Path 1B safety gate
(typed-pipeline-accepts) now accepts strip candidates for these
shapes, so `chelis lint --check` emits the `[fix]` marker and
`chelis lint --fix` rewrites them. No rule logic changed; the
unlock comes entirely from the upstream typecheck and linearity
fixes. See
`docs/investigations/redundant_linearity_autofix_recoverage_diagnosis.md`
for the diagnosis.

### Fixed - `numel(to_tensor([]))` returns 0 for empty input (#100)

Three clamp sites inflated a zero-element shape product back to one:
`eval_builtin "numel"` did `shape.iter().product::<usize>().max(1)`,
and both `chelis_alloc_tensor` and `chelis_alloc_view` had
`if size == 0 { size = 1 }`. Shape inference was correct
(`to_tensor([])` produces rank-1 shape `[0]`); the bug was in
`numel` reporting only. Dropping the three clamps lets the
empty-product identity (`[].iter().product() == 1`) handle the
scalar case naturally while rank-1 zero-element tensors correctly
report `numel = 0`. Allocator safety preserved by the existing
`bytes.max(1)` on `posix_memalign`. Closes
`Runtime-EmptyTensorNumel-F1`, the upstream root cause of the
Coral-reported `filter`/`head`/`tail`/`slice` crashes on empty
results. Regression tests at
`crates/chelis-cli/tests/eval_empty_tensor_numel.rs` and
`crates/chelis-cli/tests/cbackend_empty_tensor_numel.rs` (eval-vs-C
agreement).

## [0.7.7] — 2026-05-12

### Added - pipe-stage callable surface

`chelis check`, `chelis eval`, and `chelis build` now accept callable
shapes as pipe stages that previously required an explicit lambda or
rewrote-to-application form:

- `x |> grad(f)` and `xs |> vmap(grad(f))` lower through the existing
  application path. Equivalent to `grad(f)(x)` and `vmap(grad(f))(xs)`.
- `x |> f` where `f` is a function-valued parameter of a callable.
- `x |> tensor_to_scalar` and other non-elementwise unary primitives
  (the in-IR lowering now mirrors the host lane's beta-reduction path).
- `x |> realize` as a bare-keyword pipe stage (parser synthesizes a
  lambda; the existing realize inference accepts the resulting form).

### Added - implicit-copy fan-out covers var-RHS let-bindings

The linearity checker now permits implicit copy insertion for
cross-statement var-RHS let-aliasing such as `let alias = x; mul(x, alias)`.
The DAG-level Copy node is inserted automatically when fan-out across
non-borrow consume sites is detected; explicit `copy(x)` is no longer
required in this shape.

### Added - lint allowlist now matches canonical sources

- `deep-user-symbol-charset` accepts `t-ref` (the canonical compound
  tag for read-only borrow types in §1.4 / §2.5). `chelis deep` output
  no longer self-conflicts with `chelis lint --check`.
- `module-pascal-components` accepts `Capstone` as a top-level
  ecosystem module prefix; `spec/01-nomenclature.md` §2.6 documents it
  alongside the other ecosystem packages.

### Added - `chelis eval --file` warns when there is nothing to evaluate

Programs whose `--file` contains only `def` declarations no longer
silently return success with no output. A stderr warning is emitted
(`warning: input contains only def declarations; nothing to evaluate`)
and the process exits 0. Scripted consumers are unaffected.

### Fixed - lint auto-fix re-enabled with typed-pipeline proof

`chelis lint --fix` now applies `redundant-linearity-call` and
`prefer-pipe-operator` rewrites again. The fixer drives proposed
rewrites through the typed pipeline and only writes the result when
parsing, type-checking, and linearity all accept it. The 0.7.6
limitation that disabled these auto-fixes is closed.

### Internal

- `inlining_names` recursion guard narrowed: legitimate nested
  fn-typed parameter applications (`f(f(x))`) no longer trip the
  inlining guard's false-positive rejection. True self-recursion
  still terminates with the documented fallback.
- Dead `t-dims` parsing in `chelis-ir::lower` removed (vestige of the
  pre-flattened `t-tensor` form).

## [0.7.6] - 2026-05-10

### Fixed - conservative lint auto-fix rollout

`chelis lint --fix` no longer rewrites `redundant-linearity-call` or
`prefer-pipe-operator` warnings. Those rules remain visible as
warnings, but their source rewrites are disabled until the fixer can
prove that removing explicit linearity calls or converting nested calls
to pipes preserves type, ownership, and call-argument behavior.

## [0.7.5] — 2026-05-10

### Fixed - context lowering for shell test runs

Context-based evaluation now uses the same library-aware lowering map
when lowering new code that the compiler API uses when counting new-code
roots. This fixes `chelis test` failures in downstream shells where a
test wrapper called a host-only library helper returning a scalar value.

## [0.7.4] — 2026-05-10

### Added - lint auto-fix and cleanup rules

`chelis lint` now supports `--fix`, `--rules`, and `--list`. The lint
engine records rule severity, applies non-overlapping source fixes
in-place, and distinguishes `chelis-lint: allow` from
`chelis-lint: keep`: allow suppresses diagnostics and fixes, while
keep preserves the source but can still warn.

The first fixable cleanup rules are `redundant-linearity-call` and
`prefer-pipe-operator`. Public string literals are now checked by the
blocking `no-em-dash-in-public-strings` rule; the initial rollout
cleaned existing source strings while leaving Markdown-prose
enforcement queued for a later doc-corpus pass.

## [0.7.3] — 2026-05-10

### Fixed — lowering diagnostics and release test coverage

Chelis lowering now exposes structured diagnostics for valid language forms that
are not supported by IR evaluation instead of letting internal lowering panics
escape through user-facing commands. Diagnostics include source offsets or span
IDs when available and point users at `chelis build --target c` when the C host
backend is the supported path.

`chelis test` now has regression coverage for pipe stages that must fall back to
the host runtime, and the release workflow runs the same pipe-stage fixture
against the built release binary before publishing assets.

## [0.7.2] — 2026-05-10

### Fixed — bridge provenance diagnostics

`chelis prove` now resolves c-earchin `.spans.json` manifests for Deep bridge
properties. Human failure/error diagnostics show the originating EARS file,
line, column, requirement ID, and requirement text; JSON output includes the
same data under `source.requirement`.

## [0.7.1] — 2026-05-10

### Added — first-class Surf properties and `chelis prove`

Chelis now supports Level 2 executable properties with canonical Surf syntax:
`@property NAME forall(params...) where ...: expr`. Properties desugar to
ordinary Deep `defsig` plus `def` forms and carry canonical property metadata:
`chelis_role`, `property_source_kind`, `property_quantifiers`, and
`property_preconditions`.

The new `chelis prove` subcommand discovers Surf properties and Deep bridge
witnesses, runs deterministic type-directed sampling, filters false
preconditions without calling the predicate, and reports stable human or NDJSON
output. V1 supports scalar binders and fixed-shape numeric tensors; unsupported
selected properties exit `2`, failures exit `1`, and setup/input errors exit
`3`.

Deep bridge compatibility accepts legacy `c_earchin_role:
"property_witness"` metadata while c-earchin moves to the canonical Chelis
property metadata contract.

### Added — style gate on `chelis build` / `chelis check` / `chelis validate` / `chelis eval --file`

The four CLI ingestion paths now enforce `chelis fmt --check` and the
full `chelis lint` rule set on the user-supplied source file before
the front-end pipeline runs. Non-canonical formatting and any lint
violation fail the command with a one-line-per-issue diagnostic and
an actionable hint (`run \`chelis fmt --inplace <file>\` to fix`).

The new `--allow-style-violations` flag bypasses the gate with a
stderr warning. CI must not pass it; it exists for emergency builds
and one-off migrations. The opaque environment variable
`CHELIS_STYLE_GATE_DISABLE=1` does the same and is reserved for
integration-test harnesses that synthesize ad-hoc Surf to exercise
type/effect/linearity behavior independently of style.

`chelis eval EXPR` (the inline-expression form) is unchanged — the
expression has no on-disk source to check.

### Added — four new lint rules to reflect §3 and §10.1 of the style guide

- `surf-type-pascal-case` (§3.1): `type Name` declarations must be PascalCase.
- `surf-value-snake-case` (§3.2): `def name` declarations must be snake_case (no leading underscores, no uppercase letters).
- `surf-test-name-prefix` (§10.1): functions carrying the `Test` effect must be named `test_*` or `example_*`.
- `surf-def-arrow-form` (§3.5, new spec section): `def name(params) -> T = expr` is the canonical signature form; the `def name(params) : T = expr` colon variant is flagged.

The total `chelis lint` rule set is now 13. `spec/01-nomenclature.md`
gains a new §3.5 documenting the def-arrow-form preference.

## [0.6.1] — 2026-05-06

Bootstrap-list patch. Updates DEFAULT_BOOTSTRAP_LIST in
crates/chelis-reef/src/lib.rs to point at the post-rename shell
tags (nautilus v0.6.0, coral v0.6.0, shoals v0.3.0, octant v0.4.1).
Required reef bundle rebuild against compiler =0.6.1 (from =0.6.0)
and propagated test-fixture pin updates.

No API or runtime change beyond the bootstrap-list and pin bumps.

## [0.6.0] — 2026-05-06

Ecosystem-wide naming-convention sweep. Codifies the cross-shell
identifier conventions in `spec/01-nomenclature.md`, ships a new
`chelis lint` subcommand backed by the `chelis-lint` crate that
enforces them, and bumps `chelis-std` from 0.1.0 to 0.2.0 alongside
breaking module renames.

### Added — `chelis lint` subcommand

New `chelis lint [paths...]` and `chelis lint --check` (gating mode,
exits nonzero on any violation). Backed by the new `chelis-lint`
workspace crate, which implements 9 rules covering `.ch`, `.dp`,
`.rs`, `.py`, `.sh`, `.md`, `.snap`, `Cargo.toml`/`reef.toml`, and
`.github/workflows/*.yml` surfaces. Each rule cites a section of
`spec/01-nomenclature.md`. Exception entries carry a mandatory
`cross_ref: SectionRef` field per §12 — the schema rejects free-form
prose so post-hoc justifications can't accrete in the lint config.

CI gate (`.github/workflows/ci.yml`) runs `chelis lint --check`. The
post-sweep state of chelis main reports zero violations.

### Changed (breaking) — `chelis-std` runtime renamed `Std.IO` → `Std.Io`

The `Std.IO`, `Std.IO.Csv`, `Std.IO.Json`, `Std.IO.Parquet`, and
`Std.IO.Safetensors` modules are renamed to `Std.Io.*` per §6.2 of
the recorded style guide (Title-case compound, never ALL-CAPS
abbreviation). Downstream `import Std.IO ...` statements must update
to `import Std.Io ...`. The `chelis-std` reef package version bumps
from 0.1.0 to 0.2.0 to reflect this. The bundled artifacts in
`crates/chelis-std-bundle/dist/` are regenerated.

### Changed — `Hellotensor` → `HelloTensor` in `examples/`

The `module Hellotensor` declaration in the canonical hello-world
example becomes `module HelloTensor` per §6.3 (PascalCase per
component, including each word inside a compound). The on-disk
filename `examples/hello_tensor.ch` is unchanged.

### Changed — `phaseA_*.rs` → `phase_a_*.rs` integration tests

Five `crates/chelis-cli/tests/phaseA_*.rs` files renamed to
`phase_a_*` per §9.1 (lowercase phase-letter form). Function names
inside the tests retain their `phaseA_*` style — only the filenames
moved.

### Changed — `.github/scripts/smoke_macos_accelerate.sh` ported to `.py`

Per §2.9 (no shell scripts; Python only). The macOS smoke job in
CI invokes `python3 .github/scripts/smoke_macos_accelerate.py`.

### Spec — `spec/01-nomenclature.md` recorded as canonical source

The empty 8-section glossary becomes a 13-section style guide
covering hard language constraints (§1), filesystem and manifest
naming (§2), Surf/Rust/Python identifier conventions (§§3-5),
module conventions (§6), function naming patterns (§7) including
§7.1.1 model/algorithm sub-namespaces (`bs_/mc_/gbm_/fd_/lm_/cg_/
airy_/beta_/chi_/det_/eig_/inv_/erf_`) and §7.2 type-suffix policy
with the parser/converter idiom carve-out, documentation
conventions (§8) including §8.5 mdBook source-tree exception with
`SUMMARY.md`/`README.md` tool-required exemptions, project-cutting
conventions (§9), test naming (§10), resolved escalations from the
May 2026 cleanup (§11), and the lint enforcement surface (§12).

The §11.1 Surf-vs-Deep hyphen asymmetry is recorded as intentional
structure (Deep's compound tag vocabulary uses hyphens like `t-fn`,
`pat-ctor`, `d-name`; user-defined Deep symbols inherit Surf's
no-hyphen rule via desugaring). The lint enforces a narrow guard
that any non-tag Deep symbol must satisfy the Surf identifier
charset.

The §11.2 Rust kebab-package / snake-lib hyphen→underscore
shoreline is documented in §2.2 as a Rust-language-norm crossing.

### Fixed — `scripts/regenerate_chelis_std_bundle.py`

The script invoked `cargo build -p chelis`, but the workspace
package is `chelis-cli` (the binary is named `chelis`). Fixed.

### Migration

Downstream shell-package authors:

- Update `import Std.IO ...` to `import Std.Io ...` (and any
  `Std.IO.<sub>` imports likewise).
- Bump the `chelis-std` pin in your `reef.toml` from `0.1.0` to
  `0.2.0` and the `compiler` pin from `=0.5.0` to `=0.6.0`.
- The shell repos `nautilus`, `coral`, `octant`, `shoals` are
  republishing in lockstep with this release; update their pins to
  the new tags.
