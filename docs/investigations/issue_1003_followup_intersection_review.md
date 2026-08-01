# Post-#1003 intersection review: #908, #912, and the remediation roadmap

**Review date:** 2026-08-01

**Original reviewed main:** `88fe034e1c7321b51a22dcf7d3cc1923de3d2db7`

**Reconciliation base:** `da486cadfc1813c77f33d23f548ead8a8dd1f80c`

**Tracking work item:** [#1023](https://github.com/Chelis-Lang/chelis/issues/1023)

**Scope:** Jeff's merged PRs #1014-#1022 after #1011, plus the post-review
#1026 update, compared with the requested changes on #1003 and the active
remediation roadmap.

This is an exact-head source, issue-thread, and CI review. It is not a
fresh-context red-team certification.

**Ownership note:** #1026's Binder/Syntax/Selector behavior, `spec/03` sync,
`unrepresentable_ast_domain.md` status, and implementation tests remain with
Jeff for a follow-up. This change records the interaction but edits only the
cross-roadmap contracts owned by [#1023].

## Executive conclusion

Jeff's post-#1003 work is mostly legitimate sibling work for #908 and #912,
not an ownership collision with the five remediation classes. Several choices
respond directly and constructively to the #1003 review:

- #1017 migrated consumers before switching the producer;
- no second public `List.tag` authority was added;
- `Atom::Tag` was retained while the legacy bridge still needs it;
- #1019 routes desugar construction through validated `Node::new`;
- #1016 strengthened the #912 data model with one `HostReason` and
  compile-fail coverage;
- #1018 fixed the standalone arrow-form root instance; and
- #1022 began moving public `.dp` ingestion onto the stamp pass; #1026 then
  corrected the Binder/Syntax/Selector role model and moved the principal CLI
  `.dp` surfaces onto that pass.

Those changes are progress, but they do not deliver the structural end states
requested on #1003. The #908 path still converts validated Nodes back into the
legacy public List representation, skips the new variants in permanent
validators/oracles, and leaves compilation ingress on the old parser. The #912
path still has a raw-string partial tag authority, does not make the manifest
the required observation boundary, and leaves much of its original acceptance
surface ignored or unfinished.

Current main remains red after #1026: #1019's recursive Node/List normalization
is on the failure path for two macOS stack-safety tests. The exact-head CI run
at `da486cad` stopped with 642 tests unexecuted behind nextest fail-fast.

The resulting status is therefore:

1. **#908 is an in-progress representation migration, not yet class
   elimination.**
2. **#912 delivered a useful narrowed checklist, not its complete original
   tracker/design acceptance scope.**
3. **The five remediation classes retain their own work.** The new #908/#912
   structures must become stronger successors or derived inputs to those plans,
   not parallel permanent authorities.

## What the #1003 review requested

The [#1003 review](https://github.com/Chelis-Lang/chelis/pull/1003#issuecomment-5149649165)
requested four changes before the #908 producer cut:

1. Preserve one decoded-tag authority. Do not add a second public tag field to
   legacy `List`; retain `Atom::Tag` until the List path can be deleted
   atomically, or move directly to private validated `Node`.
2. Migrate success-path consumers before switching a producer. Require real
   exhaustive tag semantics, recursive validator/oracle coverage, explicit
   `BareList`/`UnknownForm` dispositions, and stamped public ingress.
3. Rebase onto #994 and replace its raw-string partial `KNOWN_TAGS` dispatcher
   with typed exhaustive classification, including positive and negative
   controls for Node, `BareList`, and `UnknownForm`.
4. Resolve the diagnostic, oracle, design-document, and exact-head CI
   interlocks rather than weakening or bypassing them.

It also recorded two roadmap handoffs outside #1003 itself:

- #912's realizability/capability declarations must become derived or
  transitional views of #729 Phase 4 and #730 Phase 3 rather than competing
  permanent capability authorities.
- `[05-OBS-6]` requires coordinated amendments to #732's observation plan,
  dtype §C4, and the release roadmap because it introduces another eval-output
  migration after the roadmap's per-lane freeze.

## PR-by-PR intersection

| PR | Primary area | What it resolves from #1003 | What remains or collides |
|---|---|---|---|
| [#1014](https://github.com/Chelis-Lang/chelis/pull/1014) | Prove induction | No direct overlap | Independent product work |
| [#1015](https://github.com/Chelis-Lang/chelis/pull/1015) | 0.18.1 preparation | No direct overlap | Its stated rationale does not identify the urgent downstream defect that the roadmap names as the out-of-band patch-release exception |
| [#1016](https://github.com/Chelis-Lang/chelis/pull/1016) | #912 enforcement | Unifies `HostReason`; adds `BuiltinDecl` compile-fail coverage and a tag-table test | Removes the zero-disagreement comparator before the #908 migration is complete; the later #908 interaction immediately invalidated the proven agreement |
| [#1017](https://github.com/Chelis-Lang/chelis/pull/1017) | #908 consumers | Correct consumer-before-producer sequencing; real Node tag dispatch through the bridge; retains `Atom::Tag` | The semantics still depend on rebuilding a public List; validators and permanent oracles still skip Node-domain data |
| [#1018](https://github.com/Chelis-Lang/chelis/pull/1018) | #912/#947 | Fixes the standalone arrow-form root instance | Does not establish manifest completeness or dotted-root behavior |
| [#1020](https://github.com/Chelis-Lang/chelis/pull/1020) | #912/#908 interaction | Repairs false-Host results exposed by #908 and adds a realizability-based lowering-map entry point | Expands a manual raw-string table rather than replacing it with a typed total `DeepTag` classification; the old classifier remains active |
| [#1021](https://github.com/Chelis-Lang/chelis/pull/1021) | #912 wiring | Exposes manifest data through targeted eval JSON; adds a `requires_main` parameter to the build-result seam | Every production caller passes `None`, so the build decision still searches generated C text; manifest ownership is not enforced |
| [#1019](https://github.com/Chelis-Lang/chelis/pull/1019) | #908 producer | Runs desugar-created vocabulary nodes through validated `Node::new`; deletes `Atom::Keyword` after its parser rejection | Immediately normalizes Nodes back to Lists; introduces an unguarded recursive normalization pass and the current macOS failure path |
| [#1022](https://github.com/Chelis-Lang/chelis/pull/1022) | #908 ingress | Moves `.dp` formatting onto `parse_and_stamp_file` | Check, build, eval, cost, and compiler-api ingress remain on `parse_str_strict`; the PR documents an unresolved Binder/Syntax stamping problem |
| [#1026](https://github.com/Chelis-Lang/chelis/pull/1026) | #908 roles and ingress | Corrects vocabulary-headed Binder/Syntax/Selector stamping; moves surf, fmt, check, build, cost, and prove CLI paths onto `parse_and_stamp_file` | Eval validates a stamped tree but then hands source to an engine that reparses it; compiler-api still uses `parse_str_strict`; validators/oracles and the legacy carrier cut remain; the controlling spec/design reconciliation is deferred to Jeff's follow-up |

## #908: what is live and what is still transitional

### Delivered mechanism

`Node` is a meaningful new boundary. Its fields are private, both constructors
validate, custom deserialization routes through validation, and construction
rejects a `Name` in a declared `RuntimeExpr` child slot. #1017 also avoided the
worst sequencing failure from the abandoned #1003 head: it moved semantic
consumers before #1019 made Node construction reachable.

That supports a narrow claim:

> Desugar-created vocabulary nodes pass through the new role/arity constructor
> gate before they are returned from the internal construction step.

### Not yet delivered

It does not yet support the broader claim that the invalid Deep AST domain is
unrepresentable throughout the public/compiler pipeline:

- [`Expr`](../../crates/chelis-deep/src/ast.rs) still publicly contains
  `List`, `Node`, `BareList`, and `UnknownForm`.
- `Expr::node` still produces `List + Atom::Tag`.
- [`Node::to_list`](../../crates/chelis-deep/src/node.rs) reconstructs that
  legacy representation with cloned public elements.
- [`desugar_program`](../../crates/chelis-surf/src/desugar.rs) constructs Nodes
  internally and then normalizes the result back to Lists.
- [`chelis-types` program entry](../../crates/chelis-types/src/infer/program.rs)
  recursively normalizes Node, List, BareList, Map, MetaExpr, and UnknownForm
  into another legacy tree before the main checking passes.
- [`chelis_deep::validate`](../../crates/chelis-deep/src/validate.rs) treats
  Node, BareList, and UnknownForm as requiring no recursive validation.
- The permanent raw-vocabulary walker also returns immediately for those three
  variants rather than traversing Node metadata and role-relevant children.
- #1026 fixes the observed Binder/Syntax/Selector false-errors and routes the
  principal CLI entry points through the stamped pass. The eval arm still
  discards that stamped result and sends source to a non-strict engine parser,
  and the compiler-api entry points still call `parse_str_strict` directly.
- #1026 changed the role/arity model without updating the controlling Deep
  syntax chapter or #908 design/evidence. That follow-up is explicitly left
  with Jeff rather than folded into this cross-roadmap reconciliation.

The remaining work is therefore not merely deletion. The stamp-role model,
public ingestion, validators, permanent oracles, recursive stack behavior, and
downstream representation ownership must all be corrected before the legacy
path can be removed honestly.

### Current CI regression

#1019 added `normalize_nodes_to_lists` and `normalize_node_to_list` to the
checker entry path. The function recursively rebuilds existing Lists as well as
Nodes and does not use the checker's per-site stack guard. The macOS acceptance
tests deliberately run a 4,000-deep List tree on a reduced grown segment to
prove that exhaustion becomes a typed diagnostic. The new pre-check recursion
can exhaust that segment before a guarded checker walker runs.

At the reconciliation base, the
[CI run](https://github.com/Chelis-Lang/chelis/actions/runs/30709700379)
aborts these tests with signal 10:

- `deep_app_chain_is_rejected_never_silently_passes`
- `deep_app_chain_yields_stack_budget_diagnostic_not_sigsegv`

Nextest then reports 642 tests not run because of fail-fast. Main is not green,
and the status of those hidden tests is unknown.

## #912: interpreting “all scope delivered”

The issue state and the completion statement should be separated.

1. In the [post-#1020 update](https://github.com/Chelis-Lang/chelis/issues/912#issuecomment-5151757598),
   Jeff reduced the remaining checklist to eval JSON manifest exposure and the
   build Compile-vs-object decision.
2. #1021's PR body said `Closes #912 scope`; its commit said `Closes the
   remaining #912 wiring scope`. GitHub recognized the keyword/reference and
   automatically closed #912 one second after the merge. This was not evidence
   of a separate manual close decision.
3. Eleven seconds later, Jeff posted
   [“All scope delivered”](https://github.com/Chelis-Lang/chelis/issues/912#issuecomment-5151911128)
   and enumerated the three child fixes, `[05-OBS-6]`, realizability
   infrastructure, the expanded tag table, JSON manifest output, and the build
   API accepting `requires_main`.
4. Later the same day, Jeff posted a
   [“Final remaining item” update](https://github.com/Chelis-Lang/chelis/issues/912#issuecomment-5152354928)
   that identifies target-aware f64 build routing across the layered, pruned,
   and preserve-host build entry points as an unfinished focused refactor.

The most charitable contextual reading is:

> All items in Jeff's narrowed post-#1020 delivery checklist have a landed
> implementation seam.

It should not be read as proof that every item in the original #912 tracker and
design plan is complete. Even within the narrowed checklist, “build pipeline
accepts manifest requires_main” means the function signature accepts
`Option<bool>`; #1021's own commit records that callers pass `None` and the
manifest will be threaded later.

### Delivered #912 value

- `[05-OBS-6]` now authors the always-labelled root contract in the numbered
  spec.
- #820, #947, and the cited #848 instance have landed repairs.
- `Target`, `Lane`, per-builtin declarations, `HostReason`, realizability
  inference, and root-manifest types provide useful common vocabulary.
- A missing `BuiltinDecl` classification fails to compile.
- Explicit `eval --target c --json` output includes manifest information.

### Original acceptance scope still open in the implementation

- [`KNOWN_TAGS`](../../crates/chelis-types/src/known_tags.rs) is a manual table
  keyed by `&'static str`; its test manually lists an expression subset rather
  than making a new `DeepTag` variant require a compiler-level disposition.
- [`expr_needs_host`](../../crates/chelis-effects/src/realizability.rs) returns
  `false` for `BareList` and `UnknownForm` without a `HostReason`.
- `compute_root_manifest` says it expands tuple/ADT roots, but its implementation
  still contains a TODO and emits one entry per def.
- [`ManifestedProgram`](../../crates/chelis-types/src/manifest.rs) exists but is
  not used by production observation paths, so it does not prevent observation
  before manifest computation.
- All production `cmd_build_c_result` calls pass `None`; generated-C string
  search remains the actual `main` decision.
- Target-aware f64 build routing is not threaded through every build entry;
  Jeff's latest #912 update records this as remaining work.
- Six of seven tests in
  [`issue_912_root_boundary.rs`](../../crates/chelis-cli/tests/issue_912_root_boundary.rs)
  remain ignored, including tuple/dotted-root, precision, cohabitation, and
  cross-lane completeness cases. The eval and C completeness bodies are still
  `todo!`.
- `[05-OBS-6]` requires an owed-but-unavailable root to emit `[05-UNS-1]` with
  the root, lane, and reason; the manifest acceptance surface does not yet prove
  that requirement.

## Interaction with the remediation roadmap

### Ownership is compatible

The [roadmap](../../spec/design/remediation_roadmap.md) explicitly places #908
and #912 beside the five remediation plans because those plans cannot deliver
their domains:

- #908 owns AST-domain states that #731's checker exhaustiveness cannot make
  unrepresentable.
- #912 owns root existence, naming, order, and artifact decisions; the root
  envelope and unavailable-root behavior are now authored by `[05-OBS-6]`.

There is therefore no reason to reject the work merely because it is not a
phase of #729-#733.

### Structural-successor interlock with #731

#731's permanent contract currently names decode-once `Atom::Tag`, typed
`DeepTag` dispositions, and a standing raw-tag oracle. #908 may replace that
representation, but the successor must be at least as strong. A private Node
constructor followed by conversion into a public List, plus validators that
skip the new domain, is not yet that successor. The controlling documents also
remain stale: [`checker_totality.md`](../../spec/design/checker_totality.md)
still describes the old permanent representation, while
[`unrepresentable_ast_domain.md`](../../spec/design/unrepresentable_ast_domain.md)
still says implementation is pending and requires deletion of `Atom::Tag` and
`Expr::List`.

### Capability-authority interlock with #729 and #730

`BuiltinDecl.realizability` and target capability sets are useful inputs for
#912 routing. They overlap the future exact builtin x surface x dtype x backend
authority owned by #729 Phase 4 and the gate deduplication owned by #730 Phase
3. This reconciliation names them as pre-table routing projections and
requires them to derive from Tables A/B at Phase 4. Root unavailability consumes
#730's typed failure channel rather than creating a parallel diagnostic
vocabulary.

`KNOWN_TAGS` is a different handoff: it classifies Deep syntax, not numeric
capability. It belongs to #908/#731's exhaustive typed `DeepTag` dispositions
and must not be generated from the numeric tables.

### Observation-freeze interlock with #732 and dtype §C4

The numbered specification now controls: `[05-OBS-6]` requires every root to
render as `name = value`. This reconciliation follows it through
[`faithful_observation.md`](../../spec/design/faithful_observation.md),
[`dtype_semantics.md`](../../spec/design/dtype_semantics.md) §C4, and the
release roadmap. It distinguishes the root envelope from #732's frozen numeric
payload formatter and schedules the remaining manifested boundary as a v0.19
contract migration tracked by #1023.

## Recommended boundary from here

### Immediate stabilization

1. Fix or remove #1019's unguarded recursive normalization before making any
   further completion claim.
2. Run the exact required main matrix and a full non-fail-fast workspace rerun
   so the 642 hidden tests receive a result.

### Complete #908 as a structural cut

1. Reconcile #1026's Binder/Syntax/Selector role and arity rules with the
   controlling Deep syntax chapter, #908 design status, and both polarities of
   the executable matrix (Jeff follow-up).
2. Make every CLI and compiler-api `.dp` ingress consume the stamped value it
   validated; a validate-then-reparse path is not decode-once.
3. Make validators and permanent oracles traverse Node metadata and all
   enforcement-relevant children.
4. Give `BareList` and `UnknownForm` explicit, tested dispositions at each
   semantic boundary.
5. Delete `Expr::List`, `List`, `Atom::Tag`, and `Node::to_list` atomically only
   after no public or compiler path depends on them.
6. Update the #731 successor contract, #908 design status/evidence, continuous
   oracle, and retrospective in the same change set.

### Restore the remaining #912 contract

Reopening #912 is the cleanest tracker disposition because the automatic close
was caused by a keyword and the original tracker is the parent for the class.
Whichever issue state is chosen, the remaining acceptance scope should stay
explicit:

1. replace the raw-string partial classification with a typed total
   disposition;
2. make `BareList`/`UnknownForm` fail closed or carry an explicit reason;
3. finish target-aware f64 routing through every layered/pruned/preserve-host
   build path;
4. implement dotted tuple/ADT manifest expansion;
5. thread `requires_main` from the computed manifest to every build caller;
6. make observation consume a manifested program rather than merely computing
   and discarding a manifest;
7. implement `[05-UNS-1]` for owed unavailable roots; and
8. replace the ignored/TODO acceptance surface with green positive and negative
   controls.

### Reconcile the roadmap (this change)

1. #732, dtype §C4, and the release roadmap now acknowledge `[05-OBS-6]`, keep
   payload formatting frozen, and name the separate root-envelope migration.
2. The current realizability and target-capability structures are recorded as
   pre-table projections that derive from #729 Tables A/B; #730 remains the
   typed failure authority.
3. #731 now permits #908's Node carrier only as an at-least-as-strong successor
   with stamped ingress consumption, recursive validators/oracles, exhaustive
   dispositions, and an atomic legacy deletion cut.

## Bottom line

Jeff's work should be retained as useful progress, not reverted wholesale. The
consumer-first sequencing, validated Node constructor, realizability model,
root instances, and JSON observability are productive foundations. The needed
correction is to stop treating the transitional seams as proof that the defect
classes are already eliminated. This change supplies the cross-roadmap contract
coordination; Jeff's follow-up retains the #1026 behavior/spec sync and the
remaining #908/#912 implementation cuts. Main must still be made green before
either structural class is declared complete.

[#1023]: https://github.com/Chelis-Lang/chelis/issues/1023
