# `redundant-linearity-call` autofix re-enablement: architectural path

## Context

Commit 477bd0d (PR #22) disabled the autofix for the
`redundant-linearity-call` and `prefer-pipe-operator` lint rules. The
reason given in `spec/01-nomenclature.md` §12 (lines 1007-1014):

> `redundant-linearity-call` and `prefer-pipe-operator` do not expose
> auto-fixes until the fixer can prove the rewrite preserves semantics.
> For `copy()` / `drop()`, that proof requires the type and linearity
> pipeline, not source-text matching.

Item 1 of the 0.7.6 toolchain hygiene workstream (PR #29) extended
implicit linearity to allow var-RHS let-bindings to fan out without an
explicit `copy()`. That makes the broad class of programs the autofix
would target safe to rewrite — but the proof still has to be expressed
in code, not assumed.

Item 5's plan (`build-up-a-plan-mossy-meteor.md` §5.2) names two paths:

1. **Path 1**: autofix proposes the rewrite, then runs the typed
   pipeline on the result. Accept iff the pipeline accepts.
2. **Path 2**: autofix consults linearity metadata from the already-
   checked program to decide whether each `copy(x)` is safe to strip.

This note picks the path, names the implementation surface, and notes
the sibling re-enablement (`prefer-pipe-operator`) that should follow.

## Current architecture

Three constraints frame the decision:

1. **`chelis-lint` is dep-pure today.** Its `Cargo.toml` lists only
   `regex` and `walkdir` (plus `tempfile` for dev). It does not depend
   on `chelis-surf`, `chelis-deep`, `chelis-types`, `chelis-effects`,
   `chelis-macros`, or `chelis-ir`. The crate is a source-text walker
   designed to be embeddable in tools that don't have the full compiler.
2. **`Rule::fix()` returns `Option<Replacement>` from
   `Context { root, path, source, surface }`.** No typed metadata is
   plumbed into the lint context today. Replacements are byte-range
   substitutions; the driver applies them and writes the file.
3. **The fix driver lives in `crates/chelis-cli/src/main.rs`**, in
   `apply_lint_fixes()` (around L5112). The CLI already depends on
   `chelis-surf` + `chelis-types` + `chelis-effects` + `chelis-ir` +
   `chelis-macros` + `chelis-deep`, so the typed pipeline is naturally
   available there. The pipeline used by `chelis check` flows through
   `chelis_surf::desugar::desugar_program` →
   `chelis_macros::expand_program` →
   `chelis_types::check_ir_program` → `chelis_effects::check_program` →
   `chelis_types::check_linearity` (see `cmd_check_one` in `main.rs`).

The implication: Path 2 (typed metadata in `Context`) would require
threading `LinearityInfo` and probably `CheckedProgram` from the
already-checked source into the lint `Context`, which in turn means
adding `chelis-types` (and transitively `chelis-deep`) as deps on
`chelis-lint`. That expands the dep graph of a library that is
deliberately minimal.

Path 1 has two sub-paths:

- **Path 1A**: the rule's `fix()` itself parses, desugars, type-checks,
  effect-checks, and linearity-checks the candidate source. Requires
  the same dep-graph expansion as Path 2.
- **Path 1B**: the rule's `fix()` returns the proposed replacement
  unchanged. The CLI driver (`apply_lint_fixes`) verifies the post-fix
  source still passes the typed pipeline. Rules opt in via a new trait
  method, default `false`, so existing rules don't run the gate.

## Chosen path: Path 1B

Path 1B is selected because:

1. **No `chelis-lint` dep-graph expansion.** chelis-lint stays a pure
   source/text library. Editor integrations, language-server consumers,
   and the lint registry tooling that don't carry the compiler stay
   unchanged.
2. **Single verification site.** The CLI's `apply_lint_fixes` is the
   one entry point that actually writes to disk. Putting the gate
   there means every rewrite a user receives has been verified.
3. **Opt-in per rule.** The `Rule` trait gains one method:
   `fn fix_requires_typed_pipeline_check(&self) -> bool { false }`.
   `RedundantLinearityCall` overrides to `true`. Other rules — the
   naming-convention rules, charset rules, allowlist rules — keep the
   `false` default and skip the (otherwise wasted) re-check.
4. **Mirrors the rollback contract.** The driver applies the
   replacement in-memory, runs the typed pipeline against the candidate
   text, and only writes to disk if the pipeline accepts. If any
   flagged occurrence fails the post-check, that single replacement is
   dropped while the others (if any) proceed.
5. **Matches the spec text.** Spec line 1008-1010 says "the proof
   requires the type and linearity pipeline" — Path 1B literally runs
   that pipeline on the candidate, satisfying the safety bar as
   written.

### Where the proof runs

`apply_lint_fixes` already loops once per pass and once per file with
the source in memory. The driver's flow becomes:

1. Run `chelis_lint::lint` to collect violations.
2. For each replacement: if the rule's
   `fix_requires_typed_pipeline_check()` is `true`, apply the
   replacement to a candidate `String`, then run the same pipeline
   `cmd_check_one` uses. Keep the replacement iff the pipeline accepts;
   drop it iff it doesn't.
3. Sort, dedupe, and write the surviving replacements as before.

The typed pipeline body already lives in `cmd_check_one` and
`checked_program_with_effects`. The fix driver can extract a small
helper that takes a `String` source and returns `Result<(), _>` —
parse, desugar, expand macros, type-check, effect-check, linearity-
check, and lower. The candidate is rejected if any step fails.

## Why not Path 2

Path 2 (LinearityInfo in lint Context) is rejected because:

1. It requires `chelis-lint` to depend on `chelis-types`, which
   transitively pulls `chelis-deep`. That triples the lint crate's dep
   graph for one rule.
2. The metadata that proves "stripping `copy(x)` here is safe" is not
   a single bit on the source-level `copy()` site — it is "the program
   still type/effect/linearity-checks after the rewrite." Reconstructing
   that judgment from `LinearityInfo` alone would require carrying
   counterfactual information ("what would this program look like
   without this `copy()`?") through the linearity checker, which is
   architecturally heavier than just re-running the pipeline.
3. Path 2 would still need a fallback for the case where the typed
   pipeline's verdict on the post-strip program differs from the
   pre-strip metadata's prediction. That fallback is exactly Path 1.

## Why not Path 1A

Path 1A (rule's `fix()` does the proof) has the same dep-graph
expansion as Path 2, with no upside over Path 1B. The CLI driver
already has the deps; pushing the work into the rule duplicates the
work for every rule that needs it.

## Implementation skeleton (for the Fix commit)

```rust
// crates/chelis-lint/src/lib.rs
pub trait Rule: Send + Sync {
    // ...existing methods...

    /// Whether the CLI fix driver must verify the post-fix source
    /// against the typed/linearity pipeline before writing the
    /// replacement to disk. Default `false` for rules whose fix is
    /// purely structural (case rename, allowlist update, etc.).
    fn fix_requires_typed_pipeline_check(&self) -> bool {
        false
    }
}

// crates/chelis-lint/src/rules/redundant_linearity_call.rs
impl Rule for RedundantLinearityCall {
    // ...existing methods...

    fn fix_requires_typed_pipeline_check(&self) -> bool {
        true
    }

    fn fix(&self, ctx: &Context<'_>, violation: &Violation) -> Option<Replacement> {
        // Source-only fix: strip `copy(x)` -> `x`. Original logic from
        // commit 85e4248; the typed-pipeline gate is enforced by the
        // CLI driver in `apply_lint_fixes`.
    }
}

// crates/chelis-cli/src/main.rs (apply_lint_fixes)
fn verify_fix_with_typed_pipeline(candidate: &str) -> Result<(), String> {
    // Same flow as cmd_check_one: parse Surf -> desugar -> expand
    // macros -> check_ir_program -> check_program (effects) ->
    // check_linearity. Return Ok(()) iff all stages succeed.
}
```

The CLI driver, when it has a replacement from a rule whose
`fix_requires_typed_pipeline_check()` is `true`:

1. Compute the candidate `String` by applying the replacement.
2. Call `verify_fix_with_typed_pipeline(&candidate)`. If `Ok`, keep the
   replacement. If `Err`, drop it.

The cost: each `redundant-linearity-call` replacement triggers one
parse-and-check. For the existing corpus, that's a handful per file.
The pipeline is fast (sub-millisecond per file in practice) and only
runs when the rule opts in.

## Architectural invariant (fixture 4)

The plan's fixture 4 ("if the lint flags any `copy()`, the autofix
must produce output that re-passes the typed pipeline") is satisfied
by Path 1B by construction: the driver gates the write on the typed
pipeline. Any flagged occurrence whose strip would break the typed
pipeline is silently kept. The invariant therefore tests **both** the
lint's flagging criterion and the driver's gate. If a flagged `copy()`
turns out to break the pipeline, the gate keeps it (so the program
stays valid) and the corresponding fixture asserts that the resulting
file still parses, type-checks, and evaluates identically — even if
not every `copy(` was stripped.

The F4 corpus probe in the test re-runs `chelis lint --check` after the
fix; if any `copy()` survives, fixture 4 still passes provided the
resulting source is valid. The assertion `!after.contains("copy(")` in
the F1/F2/F3 fixtures is stricter — those simple shapes should always
strip cleanly. F4's broader invariant is "after fix, file still passes
check + eval identically."

The F1/F2/F3 trio confirms the typical re-enablement scope; F4 covers
the residual safety bar.

## Sibling sweep: `prefer-pipe-operator`

Commit 477bd0d disabled the `prefer-pipe-operator` autofix at
`crates/chelis-lint/src/rules/prefer_pipe_operator.rs:50-52` for a
parallel reason: the source-only lint walker can't prove the rewrite
preserves dataflow semantics. The same Path 1B mechanism re-enables it
once Item 2 of the workstream lands (PR #26 — pipe lowering for
grad/vmap-grad stages).

**Recommendation to the orchestrator**: file a §5 entry in
`docs/archive/reports/gap_synthesis.md` tracking `prefer-pipe-operator` autofix
re-enablement as a follow-on workstream. The trait surface added in
Item 5 will be reusable; the per-rule opt-in is just
`fix_requires_typed_pipeline_check(&self) -> true` on the pipe rule.
The sibling sweep itself is out of scope for this PR.

## Anti-scope

This investigation deliberately does **not**:

- propose a separate `chelis-check-api` crate to host the gate. The
  pipeline lives in `chelis-cli` today; extracting it would be a
  larger refactor and isn't required for re-enablement. If a future
  workstream (e.g., for IDE/LSP integration) needs the same gate from
  outside the CLI, that's the time to extract.
- touch `prefer-pipe-operator`. That's the sibling sweep follow-on.
- modify `spec/01-nomenclature.md`. The lines 1007-1014 about the
  autofix safety bar are canonical — Path 1B satisfies them by running
  exactly the type-and-linearity pipeline the spec names.

## References

- Plan: `.claude/plans/build-up-a-plan-mossy-meteor.md` §"Item 5 —
  `redundant-linearity-call` autofix re-enable"
- Item 1 diagnosis: `docs/investigations/var_rhs_let_fanout_diagnosis.md`
- Disabled stub: `crates/chelis-lint/src/rules/redundant_linearity_call.rs:80-82`
- Original autofix: commit `85e4248` (PR #19)
- Disable commit: `477bd0d` (PR #22)
- Spec safety bar: `spec/01-nomenclature.md` lines 1007-1014
- CLI fix driver: `crates/chelis-cli/src/main.rs:5112` (`apply_lint_fixes`)
- CLI check pipeline: `crates/chelis-cli/src/main.rs:1177`
  (`cmd_check_one`); `crates/chelis-cli/src/main.rs:4543`
  (`checked_program_with_effects`)
