# Negative-axis and rank-0 standalone-parameter IR-lowering diagnosis

Diagnosis for the `softmax axis requires a statically known axis in IR
lowering` panic that blocked `chelis eval` / `chelis test` on every
package linking bundled chelis-std. A downstream agent escalated it;
this note records the root-cause analysis behind the fix.

Cross-references:

- Fix branch: `fix/negative-axis-normalization`.
- Pinning tests: `crates/chelis-ir/tests/negative_axis_normalization.rs`
  and `crates/chelis-cli/tests/downstream_chelis_std_axis_oracle.rs`.
- Spec clarified: `spec/05-risc-primitives.md` (the **Axis** paragraph).

## Bug surface

```
$ chelis eval --file packages/chelis-std/src/nn/attention.ch
error: softmax axis requires a statically known axis in IR lowering
```

Reproduced on the v0.7.9 release tag, so both bugs below are
pre-existing and the fix is forward-only. The named error comes from
`require_dim` in `crates/chelis-ir/src/tier2.rs`: it `panic!`s when an
axis index is absent from the operand's dimension list. The dispatch's
stated root cause was Bug B alone; instrumenting the lowering path
showed the *reproduced* panic is actually Bug A, and the two are
distinct.

## Bug B - negative axes mapped to `usize::MAX`

`extract_axis` in `crates/chelis-ir/src/lower.rs` did `*n as usize` on
the axis literal. A negative literal like `-1` became `usize::MAX`,
which `tier2::lower_softmax` / `lower_*reduce` then looked up against
the operand's dims (`dims.get(usize::MAX)` -> `None`) and `require_dim`
panicked. Negative axes were never normalized in lowering.

The type checker was internally inconsistent about negative axes:

- `gather` / `scatter` normalized them via `normalize_static_axis`
  (`rank + axis`), but only on the *static-evaluator* path; the regular
  `infer_gather_result_type` call site dropped the sign via
  `extract_axis_literal` and silently fell back to axis 0.
- the reductions (`sum`, `mean`, `max_reduce`, `min_reduce`,
  `prod_reduce`, `argmax_reduce`, `argmin_reduce`) rejected negative
  axes outright in `check_reduction_signature` with
  `"<op> requires non-negative axis"`.
- `softmax` never validated its axis at all, so `softmax(_, -1)` passed
  the checker and only blew up in lowering.

`spec/05-risc-primitives.md` line 85 said "Zero-indexed integer. Must
be a valid axis for the input rank" while the canonical formula
examples in the same file (lines 365 / 383 / 438) and canonical
stdlib (`attention.ch`) already use `axis=-1`. The orchestrator
resolved the contradiction: negative axes index from the end,
uniformly, for every axis-taking op.

### Bug B fix

- `extract_axis` -> `extract_axis_raw` (raw `i64`). New rank-aware
  `normalize_axis` maps `axis < 0` to `rank + axis` and turns a
  still-out-of-range axis into a `raise_lowering_error` diagnostic, so
  `require_dim` is unreachable from user input. Wired into all 7
  `extract_axis` call sites in `lower.rs`.
- The type checker normalizes the same way so the IR only ever sees
  non-negative axes: `check_reduction_signature` normalizes instead of
  rejecting; `softmax` gained axis validation+normalization; shared
  `resolve_builtin_axis` / `resolve_axis_pair_member` helpers carry the
  normalization for gather, scatter, scatter_replace, cumsum, sort,
  split, trace, and diagonal.
- `spec/05-risc-primitives.md`: the **Axis** paragraph now states the
  convention explicitly.

## Bug A - rank-0 standalone parameters

A top-level def whose parameter types come from a *separate* `sig`
declaration (`sig loss: &tensor[a, b, f32] -> ...` then
`def loss(logits, labels) = ...`) desugars to a `(fn (params logits
labels) ...)` node with **bare, untyped** params: `desugar_fun_def`
only attaches type metadata to params with an *inline* annotation, and
the standalone `sig` becomes its own `(defsig ...)` node. The type
checker uses the sig to check the body but never wrote the parameter
types back onto the `(params ...)` node.

When IR lowering lowers such a def standalone (the library-compile
path does this for every chelis-std def), `lower_fn`'s
`param_name_and_type_expr` found no type and bound the param to a
rank-0 `default_type()` Load. Any shape-sensitive op on that param -
`softmax(logits, 1)` in `Std.Loss.CrossEntropy.loss` - then did
`dims.get(1)` on a rank-0 type, got `None`, and `require_dim` panicked.
This is axis-sign-independent: the positive-axis `softmax(logits, 1)`
panicked for exactly this reason, and it is what poisoned the linked
chelis-std context for every downstream `chelis test`.

### Bug A fix

The type checker already resolves these parameter types (it must, to
check the body). It just did not write them where lowering reads them.
`annotate_params_node` (`crates/chelis-types/src/infer.rs`) now stamps
each def's declared `sig` parameter type *expressions* onto its
`(params ...)` node.

Two subtleties drove the final shape:

- The declared types are copied verbatim as Deep type *expressions*,
  not round-tripped through the inferred `Type`. The inferred function
  type of a bare `fn` literal loses `&` borrow wrappers; stamping a
  bare-tensor type onto a borrowed param made the linearity checker
  treat it as owned and reject a second read as use-after-consume
  (surfaced first by `Std.Nn.RmsNorm.rms_scale`). Copying the `sig`'s
  `(t-ref ...)` expression verbatim preserves the borrow.
- Only params from a real `defsig` are stamped. The synthesized sig
  that `desugar_fun_def` emits for a partially-annotated def uses
  `(t-var _)` placeholders for its untyped params; stamping those, or
  stamping inferred types onto a sig-less def, pre-empts the
  read-only/borrow inference that `infer_signature_metadata` performs
  on bare params (surfaced by
  `check_show_inferred_prints_signature_inference_metadata`). The
  declared `sig` parameter type expressions are gathered into a
  thread-local map (`DECLARED_SIG_PARAM_TYPES`) installed for the
  duration of each annotation pass - `annotate_ir_program`,
  `annotate_ir_program_with_context`, `build_type_env_from_library`,
  `build_compiled_library_context`, and
  `build_compiled_library_context_with_base` - so every annotation
  entry point behaves consistently and the library-compile path (where
  chelis-std's separate-`sig` defs are annotated) is covered.

## Acceptance oracle

`chelis eval --file` on both `packages/chelis-std/src/loss/
crossentropy.ch` (Bug A) and `packages/chelis-std/src/nn/attention.ch`
(Bug B) lowers without panicking, and a minimal downstream package
that imports chelis-std runs `chelis test` clean
(`downstream_chelis_std_axis_oracle.rs`, on the `ci` profile).

## Incidental: `redteam_typecheck_cache.rs` repair

`crates/chelis-compiler-api/tests/redteam_typecheck_cache.rs` did not
compile on `main` against the post-#130 `cache_file_name` signature
(which gained a `&CacheIdentity` argument), which blocked
`cargo build --workspace --all-targets` and therefore the whole gate.
The file's `rt1_*` tests were red-team coverage of the RT-1
cache-collision *defect*; #130 ("package-identity + compiler-version
in compiled-context cache keys") is the *fix* for that defect but did
not update this file. The tests were repaired to compile and re-pinned
as negative parity for the #130 fix (distinct package roots no longer
collide; `load_if_fresh` rejects a `CacheIdentity` mismatch). This is
incidental to the axis fix - it is #128/#130 cleanup the gate forced -
and is called out for the #130 owners' review.
