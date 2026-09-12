# chelis#1835 design report: make the kernel-decision memo structural

Reviewed against `origin/main` at `0e858f184`, read-only. Line numbers are `crates/chelis-ir/src/host.rs`.

## 1. Question 1: must reading a callee's summary lower it?

**Yes, once; the "once" is the whole repair.** `top_level_fn_helper_summary_rejects` (11133) is *defined* by the sparse/BLAS summarizer's verdict on the callee's lowered tensor-helper DAGs: `HostTensorHelper.summary_rejection` is set at helper construction (4959), and the classes (688-798: `MultipleRoots`, `NonLoadOperand`, `NonContiguousLayout`) are DAG-shape facts after callee inlining. No syntactic surrogate exists, and helpers come from both kernel and host-body paths, so the probe cannot lower less than the callee. A cheaper second predicate is a second answer to one question, the drift class `runtime_extents.md:674` records for this decision.

Rejected: a pre-pass over Deep syntax (second definition, drifts); a summary on `CheckedProgram` (chelis-types would own a chelis-ir fact; the program is serialized); a memo keyed on something stable (§2).

Two orderings. The first is no longer a hypothesis: the #1829 implementer measured it on current `main`, where `def_body_decision_impl` runs the callee probe at 3999 one line *before* the declared-tensor-result check at 4000 that discards the result, so a non-tensor def (all of `Std.Io.Json`) pays a probe it can never use; and `lower_app_host_expr` re-runs `staged_def_kernel` per call site (9873), the probe's per-site shape.

## 2. Question 2: structural, not conventional

The flag is a lifetime bound in disguise: every memo is keyed on `program as *const CheckedProgram as usize` (11152, 12617), and the guard's clear-on-begin/drop keeps a reused address from serving a stale entry. #935 introduced it for two caches, #1332 grew it to seven, #1531 bypassed it when `host_def_kernel` became a second entry.

Proposal: `pub struct HostLoweringSession<'a> { program: &'a CheckedProgram, facts: RefCell<DefLaneFacts> }` with `Deref<Target = CheckedProgram>`. The seven memos become fields (per-def: lane decision, summary-rejects, type-polymorphic, staging applicability; per-program: defs, call graph, effect rows, subexpr context, dynamic-`to_tensor` summaries). Every internal `program: &CheckedProgram` parameter becomes `&HostLoweringSession<'_>`; bodies are unchanged through `Deref`. `HOST_LOWERING_CACHE_ACTIVE`, `HostLoweringCacheGuard`, the pointer keys, and #1829's refcount go. `try_lower_compiled_program` builds its own session; `host_def_kernel`/`host_def_evaluation_plan` take `&HostLoweringSession`, the bare `&CheckedProgram` forms are removed, and `EvalContext` owns one session for its `program`'s lifetime (context-bound and Random kernels re-lower per application, 448-457, so a per-call session repays the probe per application).

What makes it structural: no key, so nothing is stale; the borrow checker binds the session to the program, so nothing outlives it; nested entries are two sessions sharing nothing, so `diag-1829-report.md` §8's nesting hazard has no mechanism; and an entry point that forgets the session does not compile. #1693's one squash carried two independent mechanisms, each days to attribute; arming a flag at one more entry is correct only until the next entry, which is how the interpreter reached this. The push/pop stacks (`INLINING_STACK`, `MONO_SPECIALIZATIONS`, `ACTIVE_TYPE_SUBST`, `TENSOR_HELPER_PREFLIGHT_STACK`) stay thread-local; D1 probe isolation is untouched.

## 3. Question 3: the counted receipt

`issue_1205_host_lowering_work_is_linear` (16716) is the shape. Its sibling `issue_1835_kernel_decision_work_is_linear` lives in the same `#[cfg(test)]` module because `HostWorkProfile` is crate-private, which is also why it runs: `gate.py ci-fast` runs `cargo nextest run --workspace --lib --bins` in the per-PR `Fast Tests (Linux)` job (`ci.yml:556-598`, `ci_test_targets.py:69`). An integration file needs a `ci-test-targets.toml` row or runs only nightly in `full-workspace`, cancelled at 60 minutes (#1819); a feature-gated accessor like `lowering-trace` is nightly-only (`gate.py:529`).

Fixture: `f_i(x: tensor[4, f32]) -> tensor[4, f32] = add(f_{i+1}(x), f_{i+1}(x))`, depth 12, leaf `mul(x, x)`, built as `issue_1205_host_profile` does. Drive `host_def_kernel` per def, root first, then a second full pass. Assert counts: `helper_summary_builds <= 12` after pass one, unchanged after pass two; `program_def_collections == 1`; each def's decision equals the C lane's from `try_lower_compiled_program` (the B2h agreement). A depth-6 row shows +6 builds, against the re-expanding tree's own depth-6 count. **Correction, 2026-09-12: the figure below was attached to the wrong fixture.** This paragraph originally read "Today it reads **398,574**", a number measured by the chelis#1829 implementer on a *scalar* chain, which I folded in here as a correction to my own about-2^12 estimate without checking that the two fixtures were the same shape. They are not. On the base, chelis#1835's implementer measures the **tensor** fixture above at **78** and the **scalar** chain at **398,574**. Both shapes exhibit the defect; they differ by orders of magnitude because the scalar chain's re-expansion is not bounded by tensor-helper structure. The receipt in PR #1910 carries both, each figure attributed to its own chain, which is the right shape for it. The mechanism the original sentence described is unchanged and correct: each probe's own lowering re-probes its callees, so the expansion is not one tree of depth 12 but a tree whose every node re-expands. Only the number's subject was wrong. A receipt should name which chain it measures. It stays red after #1829, which arms the guard in `runtime/mod.rs`, not `host_def_kernel`; the receipt separates the issues. Record the red count in the PR before the fix.

## 4. Question 4: numbered spec

No amendment is owed. The contract exists, stated in observables rather than mechanisms: `spec/04` guards ("Every execution mode places guards by this rule", "observes the same values and traps"), `spec/08` §2 (every lane owes its results to `spec/04` §9), `[05-OBS-9..11]` (target-aware root lane assignment). Which defs are kernels is how the lanes meet those sentences; when both conform it is unobservable, and asymptotics are pure implementation. The decision leaks into behavior only through chelis#1515 (C falls through a failed kernel lowering to host code, eval errors, `eval.rs:400-409`); if #1515 resolves toward an error on C, that is `[05-UNS-1]`'s question for its owner. The B2h row in `runtime_extents.md` is its home.

## 5. Sequencing

Land #1829 first; it closes the user-visible regression, and its `runtime/mod.rs` edit is what this replaces. Then one #1835 PR, commit boundaries: red receipt; session type and memo move, flag and guard deleted; `EvalContext` owns the session; receipt green; predicate reordering with its own count. Rebase conflicts against #1829 are deletions. If #1829 stalls, #1835 subsumes it; the orchestrator's call.

## 6. Undecided

- Whether `try_lower_tensor_helper_call` duplicates or shares an inlined callee's sub-DAG; if it duplicates, a fan-out-2 kernel is 2^depth nodes even with a linear probe. Decide with `tensor_helper_input_nodes` across depths 6/12/18; #1205's family, not absorbed.
- Whether the per-site `staged_def_kernel` re-ask at 9873 is measurable on tensor-returning callees; decide with `tensor_helper_attempts` on the same fixture. It is a caller re-asking a per-def fact; the staged lowering's own defects (#1775, #1779, #1693's other mechanism) are outside this design, which does not touch `host/staged.rs`.
- Whether MONO state shared by nested sessions on one thread needs isolation. Decide with an eval-inside-lowering fixture under D1's mutation control.

## 7. A guard this design must satisfy, discovered on #1843 after this report was written

The chelis#893 runtime-representation **Phase 0 inventory** flags new arithmetic in
`crates/chelis-ir/src/host.rs` under the kind `normalized-key-arithmetic`. On #1843's
head it flagged two rows: the **refcount decrement in `HostLoweringCacheGuard::drop`**
and a **counter increment in `top_level_fn_helper_summary_rejects`**. *(That sentence is
refuted; neither owner has an inventory row. The correction is at the end of this
section.)* An unclassified row
is `UNCLASSIFIED_FAILURE` from `scripts/runtime_representation_oracle.py`, not a warning,
so each needs a sanctioned classification in the baseline or a restructuring that removes
it. The oracle runs in `heavy-e2e.yml`, never per pull request, so this surfaces after
merge unless it is dispatched deliberately.

**Correction, 2026-09-12.** This section originally asserted, of the flagged rows above:

> the first flagged row **stops existing** rather than needing a classification

and the paragraph making that argument is deleted. The claim it rested on is the one
still standing earlier in this section, now marked there: that
`HostLoweringCacheGuard::drop`'s refcount decrement and
`top_level_fn_helper_summary_rejects`'s counter were flagged Phase 0 rows, so deleting the
guard would remove one. Measured against
`spec/design/runtime_representation_phase0_inventory.json` on `main`: **neither owner
appears anywhere in the inventory** (`HostLoweringCacheGuard` 0 occurrences,
`top_level_fn_helper_summary_rejects` 0). The refcount was hand-spelled in a way the
method-name rule does not match, so no row ever existed for it.

`crates/chelis-ir/src/host.rs` **does** carry three `normalized-key-arithmetic` owners -
`infer_app_expr_host_type`, `mono_specialization_symbol`, and
`try_lower_general_list_grad_app`, each present in both `foundation_rows` and
`active_debt` - so "host.rs has no rows" would be equally wrong. The three that exist are
simply not the two I named.

So the honest claim for an implementation is **"adds no flagged row"**, not "removes one",
and an implementer adding arithmetic here should check whether it lands under one of those
three existing owners or creates a fourth. The rest of §2's argument for the structural
repair stands on its own terms and never depended on this.

**But §3's counted receipt meets the same guard.** `HostWorkProfile.helper_summary_builds`
is a counter increment in this exact file, and `issue_1835_kernel_decision_work_is_linear`
requires it to be incremented on a path the oracle scans. Whoever implements this owes
one of:

- a sanctioned classification for the counter, decided before writing it rather than
  after the nightly goes red;
- or a counter that lives outside the scanned owner, if the profile can be moved without
  weakening the receipt.

Do not discover this from a red nightly. Run
`.venv/bin/python scripts/runtime_representation_oracle.py --phase 0 --skip-mutations`
on the candidate before pushing, and dispatch `heavy-e2e.yml` on it when claiming
completion. Establishing the classification question *before* the counter is written is
cheap; rewriting the receipt afterwards to dodge a guard is how a receipt gets weakened
to fit a tool, which is the failure this whole issue exists to correct.
