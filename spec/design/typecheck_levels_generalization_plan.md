# chelis#1207 — Level-based generalization implementation plan

**Issue:** [chelis#1207](https://github.com/Chelis-Lang/chelis/issues/1207)

**Evidence:**
[`docs/investigations/typecheck_generalize_superlinear_diagnosis.md`](../../docs/investigations/typecheck_generalize_superlinear_diagnosis.md)

## Scope and controlling contract

This plan replaces the binding-count cost of `Env::generalize` with eager,
solver-local levels. It does not change Chelis generalization semantics.
`spec/04-type-system.md` remains the controlling language contract, including
[04-INF-1..3], ordinary-lambda generalization, deferred-shape restrictions, and
uniform recursive instantiation.

The implementation must satisfy the binding review in the complete issue thread:

- do not add `ena`; existing substitution path compression is sufficient for this
  change;
- cover `TypeVar`, `DimVar`, and `RankVar`, including precision-slot type variables
  and dimensions reachable through rank bindings;
- lower younger variables whenever unification makes them reachable from an older
  variable;
- make every generalization boundary explicit;
- keep level state inside transactional solver state so a rejected speculative
  candidate cannot leak lowering;
- preserve deferred-shape monomorphism and recursive-instantiation pins;
- invalidate every persisted payload that contains `TypeEnv`; and
- retain the sweep implementation as a test-only exact reference until the
  authoritative oracle passes.

Chelis#1205's deeply nested lowering regime and the separate SCC/reachability cost
are outside this change.

## 1. Current implementation facts

### 1.1 Generalization and boundaries

`Env::generalize` in `crates/chelis-types/src/env.rs` applies the substitution to
every environment binding three times through `free_tvars`, `free_dvars`, and
`free_rvars`. Its policy filters are also part of current behavior: a type variable
with a deferred shape constraint or an active recursive-instantiation pin is not
quantified.

The seven call sites represent five boundary classes:

1. sequential block-binding RHS inference in `infer/expr_function.rs`;
2. local transformed `def` inference in `infer/expr_transform.rs`;
3. ordinary top-level inference in `infer/common.rs`;
4. implicit-generic signature and compiler-metadata resolution in
   `infer/common.rs` and `infer/program.rs`; and
5. recursive-SCC completion in two inference drivers in `infer/program.rs`.

The two recursive drivers call `prebind_recursive_function_schemes` before body
inference (`program.rs:105` and `program.rs:1081` on the reviewed source). Those
provisional variables must therefore be minted inside the SCC level, not before the
boundary begins.

### 1.2 Unification choke points

Production variable binding funnels through `bind_tvar`, `bind_dvar`, and
`bind_rvar` in `unify.rs`, plus the direct rank-alias insertion in
`unify_row_against_row`. `bind_tvar` also covers precision-slot variables reached
through `unify_tensor_prec`.

`unify` applies the current substitution before those binds. The lowering traversal
therefore walks the post-substitution composite value, including dimensions from a
previously expanded ground rank binding. `bind_rvar` itself rejects rank-containing
runs; rank-to-rank aliases are handled by `unify_row_against_row` and need their own
test path.

### 1.3 Transactional and persisted state

`Subst` is serialized, deserialized, and manually cloned. Deferred-shape resolution
tries candidates on a cloned `Subst` and commits only a successful clone. Level
changes must participate in exactly that transaction.

`TypeEnv` persists `Env`, `VarGen`, and `Subst`. Current non-empty stacked inference
clones it at both of these entry points in
`crates/chelis-types/src/infer/program.rs`:

- `build_compiled_library_context_with_base_in_session` (`base.inner().clone()`);
- `check_ir_with_signature_context_in_session` (`context.inner().clone()`).

The compiler API's `compile_new_source_in_context` is a caller of the latter path,
not a third `TypeEnv` clone point.

Three persisted payload owners contain `TypeEnv`:

- compiled context in `crates/chelis-compiler-api/src/context.rs`;
- stdlib context in `crates/chelis-compiler-api/src/stdlib_cache.rs`; and
- dependency-library context in
  `crates/chelis-compiler-api/src/library_cache.rs`.

The shared `cache_envelope` wraps payload bytes but does not describe their bincode
shape.

## 2. Solver-level design

### 2.1 Level state belongs to `Subst`

Add serialized fields to `Subst`, include them in its manual `Clone`, and exercise
them through the same trial/commit flow as all other substitution state:

- `current_level: u32`, with zero as the outermost level;
- an ordered level-transition log recording every enter, parent-restoring leave, and
  resume transition together with the next type, dimension, and rank IDs from
  `VarGen`;
- sparse lowered-level maps for type, dimension, and rank variables; and
- a resume floor for each ID class, below which variables imported from a persisted
  context are treated as level zero.

`VarGen` IDs are monotonic and not recycled. A variable's mint level can therefore
be recovered from the last transition whose corresponding next-ID watermark is at
or below its ID. When several transitions share a watermark, the last transition
wins. This makes an enter followed by a leave with no intervening allocation assign
the next variable to the restored parent, rather than to the departed child. Sparse
overrides record only lowering below that mint level. This avoids editing every
`fresh_*` call while keeping all semantic state in `Subst`.

`enter_level` returns a LIFO token and appends the new child's transition at the
current watermarks. `leave_level` consumes that token, restores the recorded parent,
and appends a parent transition at the then-current watermarks. No exit path may
restore `current_level` without recording that leave transition.

Required internal operations:

- enter a level using the current `VarGen` watermarks and return a LIFO token;
- leave a level with that token and the current watermarks, guarded so every early
  return restores the parent level and records the restoration;
- query a type, dimension, or rank variable's effective level;
- lower every free variable reachable through a type, dimension, or ground rank
  run to a target level; and
- resume a persisted solver for a new check before any new variable is minted. At
  resume, the current next-ID watermarks become the old-ID floors; all IDs below
  their class floor normalize to level zero; obsolete transitions and lowered-map
  entries below those floors are discarded; and exactly one level-zero transition
  is seeded at the new watermarks. Substitution bindings remain intact. This bounds
  serialized level metadata by transitions and lowering performed since the most
  recent resume, rather than by the lifetime of the context cache.

### 2.2 Lowering rules

After the occurs check and before storing a binding:

- `bind_tvar(v, ty)` lowers every free variable in `ty` to
  `level_of_tvar(v)`;
- `bind_dvar(v, dim)` lowers every free variable in `dim` to
  `level_of_dvar(v)`;
- `bind_rvar(r, dims)` lowers every free dimension variable in the ground run to
  `level_of_rvar(r)`; and
- the rank-alias arm in `unify_row_against_row` applies the equivalent rule when
  two rank variables are related.

The structural type walk includes `TensorPrec::Var`. The dimension/rank walk must
also see dimensions that become inline only after applying a previously stored
ground rank binding.

An environment binding at level L may ordinarily contain only unquantified free
variables at level L or older. Audit every `Scheme::mono` insertion in inference.
The PP1 bind-on-first-use path needs explicit lowering to the enclosing binding
level before insertion.

Recursive provisional bindings are the deliberate temporary exception: they are
created and bound at the SCC's inner level, remain there while every member is
checked, and are removed before leaving that level. **Do not lower provisional SCC
variables to the outer level**; doing so would prevent the completed members from
generalizing after the group closes.

### 2.3 Generalization boundaries

For an ordinary boundary:

```text
enter child level
infer or resolve the boundary body
leave child level
generalize variables whose effective level is above the parent level
```

Apply that pattern to sequential bindings, local transformed definitions, ordinary
top-level definitions, and implicit-generic signature/metadata resolution. Any path
that defers top-level binding to an SCC must not independently close or generalize
the group's variables.

For each recursive group, in both driver copies, the order is stricter:

```text
enter one SCC level
snapshot every group member's prior environment binding
prebind_recursive_function_schemes  # provisional vars are minted here
begin_group
infer every member body
finish_group                        # validate uniformity and clear pins
remove every deferred group-member binding
leave the SCC level
generalize each completed member at the outer level
bind the completed schemes
```

"Every deferred group-member binding" includes inferred members and members whose
authored signature or metadata caused prebinding to skip them. It is deliberately
broader than "bindings created by prebinding": neither driver may leave an authored
or metadata-backed group member installed while generalizing its peers.

Add `recursion::abort_group()` beside `finish_group`. It takes and discards the
thread-local group context and its pins without validating an incomplete group. A
structured SCC scope owns the level token, the complete member-name set, and the
prior-binding snapshot. Every cancellation or error exit must, before returning or
breaking the schedule:

```text
abort_group
remove temporary bindings and restore every prior group-member binding
leave the SCC level, recording the restored-parent transition
return without generalizing any member
```

The normal path consumes the same scope only after `finish_group`, complete
group-member removal, and level exit. `finish_group` remains before generalization
because it owns recursion-pin validation on a completed group; `abort_group` is the
non-validating cleanup for an incomplete one.

### 2.4 Level-based `Env::generalize`

Keep the public signature stable. The production body:

1. applies the substitution to the candidate type;
2. collects its free type, dimension, and rank variables in deterministic ID order;
3. keeps only variables above `subst.current_level`; and
4. preserves the existing deferred-shape and recursion-pin exclusions.

It must not enumerate environment bindings. Move the three sweep collectors behind
the internal `generalize-sweep-oracle` feature after the production path flips.
Quantifier ordering and the resulting `Scheme` body must remain byte-for-byte
comparable with the reference result.

## 3. Persisted-context resumption and cache compatibility

Add one typed operation:

```rust
pub(crate) fn TypeEnv::resume_for_new_check(&self) -> TypeEnvInner
```

It clones the inner state, resets the new check to level zero, and records the
persisted `VarGen` watermarks as resume floors before any new variable can be minted.
The resume operation also performs the level-metadata compaction specified in
section 2.1: old IDs normalize to level zero, obsolete marks and lowering overrides
are discarded, and one level-zero transition is seeded at the current watermarks.
Replace both non-empty stacked clone sites with this operation:

- `build_compiled_library_context_with_base_in_session`;
- `check_ir_with_signature_context_in_session`.

Tests must exercise both entry paths. Empty-context construction may continue using
an ordinary clone because it has no persisted free monomorphic solver variables from
a preceding check. Builtin and prelude construction can consume `VarGen` IDs; the
exception is justified by the absence of cross-check monomorphic state, not by an
absence of allocated inference-variable IDs.

Adding serialized level fields changes the three payload formats, not the shared
envelope format. In the implementation change:

- compiled context: `CACHE_MAGIC` `CHELIS_CTX_V9` → `CHELIS_CTX_V10` and
  `CACHE_FORMAT_VERSION` 9 → 10;
- stdlib context: `STDLIB_CACHE_FORMAT_VERSION` 5 → 6;
- dependency-library context: `LIBRARY_CACHE_FORMAT_VERSION` 2 → 3; and
- shared `cache_envelope::ENVELOPE_FORMAT_VERSION` remains 1.

Update the exact version-lock test owned by each payload and add stale-payload
rejection coverage. The expected user-visible effect is one cold rebuild for each
cache family after upgrade, never an attempted decode of the old bincode shape.

## 4. Spec-first tests

Write these tests before changing production generalization.

### 4.1 Type-variable scope and ordinary lambdas

- Reuse the existing escape negative in
  `crates/chelis-types/src/infer/tests/more.rs`: an outer lambda parameter is bound
  as `let y = x` and then used at incompatible types. It must reject for the same
  reason before and after the change.
- Add the ignored-argument positive:
  `fn x -> let y = fn z -> x in (y(1), y(true))`. It must accept because `z` is
  generalized while the outer `x` remains one monomorphic result type shared by
  both calls.
- Cover a direct younger-type-variable escape and assert its level is lowered to
  the older variable's level before inner generalization.
- For each of type, dimension, and rank IDs: enter a child, mint a child variable,
  leave, mint a parent variable, cross another boundary, and assert the child and
  restored-parent mint levels are exact. This locks the leave-transition mark and
  catches an implementation that records only enter/resume watermarks.

### 4.2 Dimension, rank, and precision paths

- `bind_dvar`: bind an older dimension variable directly to `Dim::Var(younger)`;
  assert lowering and non-generalization.
- `bind_rvar`: bind an older rank variable to a **ground** `Vec<Dim>` containing a
  younger `Dim::Var`; assert lowering and non-generalization. Do not put
  `Dim::Rank` in this test because `bind_rvar` rejects rank-containing runs.
- `unify_row_against_row`: relate an older rank variable to a younger rank alias and
  assert the alias is lowered.
- Expanded-inline path: first bind a rank variable to a ground run, then apply that
  binding before a later row unification; assert the younger dimensions exposed by
  expansion are lowered.
- `TensorPrec::Var`: cover variable-to-variable and variable-to-concrete unification,
  plus correct quantification at an ordinary boundary.

### 4.3 Boundary and policy preservation

- sequential block binding, local transformed `def`, ordinary top-level definition,
  and both implicit-generic resolver sites;
- PP1 monomorphic bind followed by a sibling binding, proving the sibling cannot
  quantify the PP1 variable;
- both recursive drivers, proving provisional variables are minted at the SCC level,
  stay monomorphic during the group, and generalize only after `finish_group`,
  complete group-member removal, and level exit; cover both inferred members and
  authored-signature/metadata members;
- cancel or error midway through a recursive SCC in each driver and, before any
  next-check reset, assert that the parent level is restored, the recursion group
  context and pins are empty, every temporary member binding is gone, and every
  shadowed prior binding is restored; then run a second check successfully;
- deferred expand/reshape results stay monomorphic;
- recursive-instantiation pins stay monomorphic until group validation clears them;
  and
- a rejected speculative shape candidate leaves every level field and map in the
  committed `Subst` identical to its pre-trial value.

### 4.4 Serialization and cache tests

- serialize and deserialize a `TypeEnv` containing nontrivial level state;
- resume it through `build_compiled_library_context_with_base_in_session` and prove
  old variables are older than variables minted by the added library layer;
- resume it through `check_ir_with_signature_context_in_session` and prove the same
  property for new source, including the compiler API caller;
- repeat serialize/resume cycles and assert transition-log and lowered-map sizes stay
  bounded by work since the latest resume while generalization results remain
  identical;
- assert exact compiled-context V10, stdlib V6, and dependency-library V3 versions;
- reject fixtures with each preceding payload version; and
- retain the shared envelope V1 exact-version test.

### 4.5 Structural elimination test

Add a test-only environment-visit counter at the sweep enumeration point. A generated
independent-binding checker test must:

1. disable the feature-gated reference sweep for its thread;
2. reset the visit counter;
3. invoke the real production `Env::generalize` path across the fixture; and
4. assert exactly zero environment-binding visits.

Disabling the reference hook is essential: reference work used by parity validation
must not contaminate the structural production assertion.

## 5. Authoritative completion oracle

The implementation change adds the internal Cargo feature
`chelis-types/generalize-sweep-oracle`. Under that feature, every production
generalization computes both the level result and the retained sweep result and
asserts exact equality of ordered `tvars`, `dvars`, `rvars`, and the scheme body.

The phase's one authoritative oracle is the named **Typecheck Level
Generalization Oracle** suite. Its one-shot local runner is:

```sh
cargo nextest run --workspace \
  --ignore-default-filter \
  --features chelis-types/generalize-sweep-oracle \
  --no-fail-fast
```

The corpus boundary is every nextest-discovered, non-ignored workspace test, with
nextest's repository default filter explicitly bypassed. Every acceptance test in
section 4 must be a non-ignored nextest test. An ignored test or a doctest is outside
this oracle unless a future reviewed change adds a named leg to the authoritative
runner; it cannot be cited as completion evidence merely because it exists. The
expected result is exit code zero with:

- exact sweep-versus-level equality at every observed generalization;
- all scope, rank, transactional, resumption, and cache tests passing; and
- the independent-binding structural test observing zero production environment
  visits.

CI executes the same selection as four deterministic, disjoint nextest hash
partitions:

```sh
cargo nextest run --workspace \
  --ignore-default-filter \
  --features chelis-types/generalize-sweep-oracle \
  --no-fail-fast \
  --partition hash:1/4

cargo nextest run --workspace \
  --ignore-default-filter \
  --features chelis-types/generalize-sweep-oracle \
  --no-fail-fast \
  --partition hash:2/4

cargo nextest run --workspace \
  --ignore-default-filter \
  --features chelis-types/generalize-sweep-oracle \
  --no-fail-fast \
  --partition hash:3/4

cargo nextest run --workspace \
  --ignore-default-filter \
  --features chelis-types/generalize-sweep-oracle \
  --no-fail-fast \
  --partition hash:4/4
```

The CI matrix must contain all four partition indices exactly once and must disable
matrix fail-fast so one failure cannot hide the other partition's result. A
fail-closed aggregate job keeps the stable **Typecheck Level Generalization
Oracle** status and succeeds only when all four partitions succeed. The union of
the four partitions is the oracle; no partition is independent completion
evidence. This is an execution split of one corpus, not four acceptance oracles.

The one-shot runner remains manually reproducible even though it intentionally
re-runs the old sweeps and may exceed the local inner-loop budget. Document the
runner and its exact-head result in the implementation PR and current-state
evidence. `python3 scripts/gate.py --local` remains the pre-push gate, but it does
not replace this oracle.

Wall-clock runs of the repository fixture generator at N=200/400/800 and the Shoals
module are supporting performance evidence only. Expected support: the binding-count
term disappears and `lets`/`letsind` approach `flat` scaling. Re-profile the
`check`-versus-`build` gap afterward instead of attributing that separate gap to this
change without evidence.

## 6. Implementation sequence

1. Add the tests in section 4 and lock the current sweep behavior.
2. Add transactional level state, watermark accessors, lowering operations,
   resumption, serialization, and the three payload-version bumps while production
   still uses sweeps.
3. Add lowering at every bind choke point, PP1 handling, all ordinary boundaries,
   and the corrected SCC order in both drivers.
4. Add the oracle feature and its dedicated fail-closed CI suite.
5. Flip `Env::generalize` to levels; retain sweeps only behind the oracle feature.
6. Run the local gate, then the authoritative oracle.
7. Record supporting timings and file separate follow-ups for SCC/reachability and
   any surviving checker-lane gap.

Do not add a runtime switch between algorithms. The old algorithm is an assertion
oracle, not a second supported production mode.

## 7. Divergence protocol

If exact parity reveals user-visible behavior that cannot be explained by an
implementation bug, stop before the production flip and reduce it to positive and
negative tests. The controlling numbered specification decides the result. If the
numbered specification is incomplete, amend its normative rule first in the same
reviewed change; neither this design document nor reviewer preference may choose
Chelis language semantics.

## 8. Risk register

| risk | required control |
|---|---|
| Outer variable captures a younger composite variable | Lowering at all four bind/alias choke points plus direct escape tests |
| Rank escape hides behind alias or prior expansion | Separate ground-run, alias, and expanded-inline tests |
| Provisional SCC variables become permanently monomorphic | Enter before prebind; never lower provisional variables to outer level; remove before leave |
| A leave without allocations assigns later variables to the departed child | Record enter and leave transitions; exact three-ID-class exit regression |
| Cancellation leaks a level, recursion pins, or group bindings | Explicit `abort_group`; structured cleanup of level, group context, pins, and all inferred/authored member bindings before control exits |
| Rejected shape trial leaks lowering | Level state in cloned `Subst` plus byte-identical rollback test |
| Persisted variable is re-generalized by new code | Typed resumption at both non-empty entry points plus stacked tests |
| Repeated persisted resumption grows obsolete level history | Resume-floor compaction plus repeated-cycle size and semantic-equivalence tests |
| An old bincode payload is decoded as new state | Exact V10/V6/V3 payload bumps and stale-version tests; envelope stays V1 |
| Parity misses heavy checker tests behind nextest's default filter | `--ignore-default-filter`, non-ignored acceptance tests, and a dedicated four-partition CI suite with a fail-closed aggregate |
| Parity passes while production still sweeps | Independent reference-disabled zero-visit assertion in the authoritative oracle |
| Quantifier reordering creates false parity failures | Deterministic ID ordering and exact ordered-scheme equality |
| Performance claim expands into chelis#1205 | Generated binding corpus and explicit nested-lowering exclusion |
