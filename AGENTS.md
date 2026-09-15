# Chelis Agent Contract

Canonical agent instructions for this repository.
`CLAUDE.md` should resolve to this file so Claude-style and Codex-style entry points do
not drift.

## Quality Standards

### Spec-First Development

- Before writing implementation, write test stubs derived from the owning spec.
- Every spec requirement should have a corresponding test before the code exists.
- If the spec says "X is a type error," write the failing test before implementing the checker.
- A phase is not done until every active spec requirement has both positive and negative
  coverage.

### Negative Test Parity

- For every test that checks something works, add the corresponding failure test.
- If you cannot name the failure case, the spec understanding is still weak.

### Do Not Trust Green

- Passing tests prove alignment with the tests, not necessarily with the spec.
- After green CI, check what active requirements still lack tests.
- Audit silent fallbacks, default values, empty error vectors, and `unwrap_or` paths.

### Guard Inventories

A guard list carries a reviewed disposition per row or is regenerated from the
source tree. Do not hand-maintain a second copy of tree membership. Regeneration
may discover rows; it must never assign reviewed semantic authority or erase a
required coverage floor.

### Phase Completion Criteria

- Do not claim a phase is done based on crate-local green or narrative progress.
- A phase is done when the executable acceptance surface is green, docs are honest, and
  the red team can only find minor residual issues.
- Budget for at least one adversarial validation pass that runs code, not just source review.

### One Acceptance Oracle Per Phase

The runtime representation host/C vocabulary and checked-metadata gate is
`.venv/bin/python scripts/runtime_representation_oracle.py --phase 1`. It includes
Phase 0 inventory/mutations and requires fresh frozen-selection/framework receipts.
The hosted `runtime-representation` stage in `heavy-e2e.yml` owns daily and
manually dispatched enforcement; ordinary PR success does not certify it, so
dispatch that workflow on the candidate when claiming Phase 1 completion.
`--fast` remains the pre-push gate and `--local` remains optional. Phase 1 does
not validate the later generated ABI, native Python/DLPack or device execution
obligations.


- Every phase must name one authoritative completion oracle.
- That oracle may be a single command, a named suite, or a documented manual runner, but
  it must be explicit.
- Supporting evidence may exist, but it does not replace the oracle.
- If the oracle is manual or long-running, document it in the owning phase plan and the
  current-state docs.

## Red Team Protocol

### Baseline

- Red team against the spec, the code, the tests, the examples, and the CLI behavior.
- Execute tests and commands; do not treat source inspection as sufficient proof.
- For this repository, a red-team round is a fresh local subagent reviewing from an
  inline brief. Verifying a fix is not a round: the reviewer that reported the finding
  checks the repair. A main-thread validation pass is neither, except the rerun of a
  departed reviewer's exact reproduction, and a phase or pull request is red-teamed only
  when the subagent actually ran the validation work.

### Required Red-Team Behaviors

- Write and run adversarial tests when coverage is missing.
- Verify that inputs which should fail do fail, and with the right reason.
- Verify that inputs which should pass do pass, with exact outputs where applicable.
- Check docs and phase claims against the shipped behavior, not just intent.
- Stay inside the brief: its head, files, in-scope claims, and any explicit limits bound
  the round.
  Record the exact reviewed commit and the worktree's baseline status, and use the
  worktree and warm target the brief hands you rather than rebuilding cold.
- Report a P3 in one line, without reproduction. A construct no maintainer would write
  is not a finding.

### Fresh-Context Enforcement

- A pull request gets at most three fresh rounds by default; a fourth needs the user's
  explicit approval. A prose-only pull request, design documents included, gets one,
  and a second needs the same approval. Rounds run from any platform count, and the
  pull request's round record is the counter. Verification does not count against the
  cap; the end-of-pull-request round does.
- A confirmed in-scope P0 or P1 does not by itself earn a fresh round. The reviewer that
  reported it stays alive and verifies the fix: it holds the context, and verification
  is a few turns. A fresh round is owed only when the fix introduces a new mechanism or
  touches files the standing reviewer did not read, or once at the end of a long pull
  request before ready-for-review. A one-word or one-line repair inside the files the
  reviewer read never earns one.
- Every round runs from an inline brief in the shape the `redteam-exec` skill carries:
  exact head, changed files, the pull request's in-scope claims, the worktree and target
  it may use with pasted busy-signal output for that target, the report-length budget,
  and the delivery channel. Context budgets and time limits are optional, with no
  default. When set, include them in the brief. The brief does not open with "read
  `AGENTS.md`".
- Freshness applies to the subagent's review context, not to the filesystem. Hand the
  reviewer an existing worktree and its warm target cache when the worktree is at the
  exact review head, has a known clean baseline, and has no concurrent writer or build
  owner. The brief does not assert that last condition; it pastes evidence bearing on
  it. Run
  `.venv/bin/python scripts/worktree_status.py [--path PATH] [--json] [--quiet]`
  and paste its output, with the time you took it, into the brief. That pasted output,
  not the sentence around it, is the heavyweight-command handshake of
  [Build Concurrency And Process Hygiene](#build-concurrency-and-process-hygiene) for
  that target. Create a new worktree or target only when those reuse conditions do not
  hold.
- Read what that signal actually says. It answers free, busy, unknown, or not clean,
  and where those conflict the more cautious answer wins. Missing evidence withholds
  free by design rather than standing in for an empty machine: a probe that could not
  read the process table answers unknown, so unknown is a reason to wait rather than a
  reason to proceed. A `--local` or full gate run is lease-proven and kernel-backed. A
  run that takes no lease, `--fast` among them, is caught only by a scan of the
  processes it spawned, so it is the weakest of the signals. A finished gate report is
  printed under a history label and is never current state. Free is the probe's best
  answer rather than a proof: it documents the residual case its own fail-safe does not
  reach.
- A reviewer whose probes mutate tracked source does not share a worktree with anything
  that compiles, whatever the signal reports. This is an exception to warm reuse rather
  than a caveat on it: sequencing narrows the window in which one agent's inserted
  variant reaches another agent's build, and a separate worktree closes it.
  [`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §7
  has the collisions both rules come from.
- Before spawning a fresh round, inventory subagent handles created in your own current
  session. Retire only stale or failed handles that will not be used again; a standing
  reviewer awaiting a fix is neither. Do not disturb another developer's handles. If the
  spawn routes to remote infrastructure, errors, or comes back broken, retire that
  handle and retry until you have a working fresh local subagent or can state that
  red-team validation is blocked because fresh local subagent execution is unavailable.
- A subagent that reuses a worktree must restore its temporary probes or mutations and
  report the final worktree status, unless the task explicitly asks to retain them.

### Pull Request Review Gate

- Before starting work on a pull request, and again before merging it, read
  [the PR-author guide for tests and protected contracts](docs/guard_changes_for_pr_authors.md).
  Follow its change-specific instructions and use the checker's required
  acknowledgement lines in the PR description.
- Every pull request, documentation-only work included, gets at least one compliant
  red-team round before merge. Push first, after `python3 scripts/gate.py --fast`: the
  round reviews the pushed head while CI runs on it. Applicable CI checks must pass on
  the head that goes ready-for-review, including any repairs made during review.
  `--local` is optional for troubleshooting or additional local validation; it is not
  a pull-request readiness requirement. Record the reviewed head and CI evidence in
  the pull request. Any supporting local evidence must also name the head it covered;
  evidence from before a repair does not validate the repaired candidate.
  [`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §8
  has the runs behind this rule.
- `PR Contract Acknowledgements` is the required owner for pull-request-description
  acknowledgements. A title or description edit reruns that check without cancelling
  or replacing compiler validation for the same commit. Generic edits do not enter
  compiler or Hull workflows. Retargeting the pull request's base is an implementation
  change: `PR Base Retarget Validation` holds the head pending while trusted
  coordination dispatches fresh compiler and Hull workflows against the exact new
  synthetic merge. Wait for that receipt and the refreshed acknowledgement/changelog
  checks before proceeding.
- After reviews and repairs are complete and no further content change is planned,
  dispatch `PR Package Expansion` with the pull request number and exact head SHA.
  Start it alongside the final required implementation checks rather than waiting for
  them; merge only after both the required checks and the expansion report have been
  inspected. Record the reviewed SHA and run link in the pull request. A content change,
  hand-resolved conflict, base-changing rebase, or base-branch retarget creates a new
  synthetic candidate and requires a fresh dispatch. Do not create that invalidation
  merely to refresh a branch after `origin/main` advances: when GitHub can merge the
  exact reviewed head safely, preserve that head and its evidence as
  [Worktree And Branch Discipline](#worktree-and-branch-discipline) requires. Resolve
  failures introduced by the candidate; identify inherited failures and any missing,
  timed-out or otherwise incomplete coverage explicitly.
- Classify every finding against the pull request's stated scope. A finding is in scope
  only when the pull request introduces it, worsens it, or claims to correct it. Mere
  discovery during review, including a pre-existing spec/implementation mismatch in an
  unrelated surface, does not bring a finding into scope.
- State a pull request's or phase's claim at the granularity its oracle proves. An
  unbounded universal claim invites sampling in every round and can never be closed.
- For documentation and design reviews, assign severity by the contract impact rather
  than by the mere presence of an inaccurate sentence. A wrong normative rule, or a
  plan that cannot close a named in-scope instance or acceptance requirement, is P1. A
  design document that misdescribes current `main`, or exposes a sequencing seam between
  delivery slices while leaving the normative contract and named deliverable achievable,
  is P2 and must be recorded as residual work rather than promoted to a merge blocker.
  Staleness against a sibling pull request's moving head, wording, line-level accuracy
  of the pull request body, and anything whose fix would add text without a necessity
  sentence are out of scope.
- A confirmed in-scope P0 (critical) or P1 (high/major) finding blocks merge until the
  reviewer that reported it verifies the fix. Repeat fix and verification until that
  reviewer reports no in-scope P0 or P1;
  [Fresh-Context Enforcement](#fresh-context-enforcement) names the cases that owe a
  fresh round instead. When that reviewer is no longer available, the orchestrator reruns
  its exact reproduction against the fixed head and records the result as "reviewer
  unavailable"; that record closes the finding, and the end-of-pull-request round, when
  one is owed, re-checks it.
- Absent an in-scope P0 or P1 finding, scale rounds to the change. Minor updates, bug
  fixes, and textual changes do not inherently merit another round. A rebase whose
  overlap with your work is significant, either in changed lines or in semantics, may
  merit a fresh round on the intersection. A rebase that only picks up an atom clearly
  consistent with, or irrelevant to, the files you are working on does not, and neither
  does one whose only hand-resolved conflicts are generated or digest lines that the
  owning script resolves (see
  [Worktree And Branch Discipline](#worktree-and-branch-discipline)). Use your best
  judgement.
- A non-major finding does not block merge, but if you are already rebasing or fixing
  something else, fold in the other relevant issues reviewers raised.
- Repairs under this gate may correct, remove, or narrow the pull request's existing
  content. They must not add new design scope, implementation responsibilities,
  inventories, mechanisms, or promises merely to absorb a finding. When a correction
  would require that expansion, reduce the claim and track the additional work outside
  the pull request.
- Separability sizes a pull request. Slices that must ship together are commits
  inside one pull request; slices that can ship apart are separate pull requests. A
  line count does not decide it. About 1,000 hand-written changed lines, excluding
  regenerated artifacts such as embedded skill copies and generated registries, is the
  point at which you owe a sentence justifying that the work is still one shippable
  slice. It is not a threshold the next line breaches, and arguing it as one argues
  about the wrong quantity: a pull request already past the figure invites the
  sunk-cost reading that finishing is cheaper than splitting, which is not a judgement.
- The sharper signal is the ratio of cases a claim covers to cases its tests prove.
  A pull request whose oracle proves a small fraction of what its claim asserts is too
  big for that oracle at any line count, and the repair is to narrow the claim or
  extend the oracle. This is the measurable form of the claim-granularity rule above.
  Put size inside the reviewing round's scope and invite the reviewer to disagree with
  it; a reviewer told that size is settled cannot raise the finding that matters here.
  [`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §6 has the
  branches that outgrew the figure and the ratio that caught it.
- A finding class is the underlying defect category or unmet obligation, not its file,
  line, or wording instance. Every round record names the class of each finding; when a
  report leaves one unlabeled, the orchestrator assigns it while recording the round.
  When two consecutive rounds report the same class, or replace a repaired finding with
  a different class, stop patching witnesses: change the representation, the oracle,
  the claim, or the brief before running another round. Do not keep expanding the pull
  request to satisfy a moving brief.
- Route confirmed in-scope findings back to the original implementation agent when it
  is still available so the fix retains its build context. Do not fix out-of-scope
  findings in the pull request; link an existing issue or file one if the defect is not
  already tracked. Record every round's exact head, verdict, commands, finding classes
  and scope classifications, accepted-no-action observations, linked issues, residual
  scope, and any explicit deadline in the pull request.

## Documentation And Spec Sync

### Documentation Authority

Authority is subject-specific during the migration from numbered chapters to OpenSpec capabilities.

1. `spec/design/chelis_canonical_reference.md` controls cross-subject architecture and project boundaries.
2. For a transferred chapter, its named `openspec/specs/<capability>/spec.md` controls the subject.
3. An untransferred `spec/00-12*.md` chapter controls its subject.
4. `spec/design/chelis_project_plan.md` controls project sequence that a higher authority does not define.
5. `spec/design/archive/` is historical reference only.

A chapter transfers only through a reviewed change that records the transfer. The chapter must mark itself superseded and link the controlling capability.

If a chapter has no complete transfer record, the numbered chapter remains controlling. The complete transfer contract is proposed, not in force: it lives in the active change at `openspec/changes/migrate-spec-authority/specs/spec-authority-migration/spec.md`.

If active documents disagree, correct the document that controls the subject. Do not add a third explanation.

That ordering is for **project-level** questions: what Chelis is, what it is for, what
the roadmap says. It is not the ordering for language semantics, and the canonical
reference says so itself: "Language semantics still belong in the numbered spec
documents."

### Normative Specs Are Timeless Contracts

Normative specifications describe the decided Chelis architecture and language
semantics, irrespective of how completely any compiler version implements them. This
rule applies to an untransferred numbered chapter, a controlling
`openspec/specs/<capability>/spec.md`, and every normative `spec.md` delta under an
active OpenSpec change.

- Do not put project or implementation status in normative specs. Prohibited material
  includes status/version-draft banners, phase or milestone labels, completion claims,
  delivery histories, release or PR inventories, acceptance-oracle results, temporary
  workarounds, and descriptions of what the implementation currently happens to do.
- State the fully decided rule without weakening it to match a bug, an incomplete
  backend, or a temporary restriction. In particular, never narrow an operation to one
  dtype merely because that is the only dtype a lane implements today. The
  implementation must move toward the spec; the spec must not move toward bad behavior.
- When an implementation gap would materially mislead a reader, one short
  non-normative parenthetical may say that the requirement is not fully implemented and
  link its owning issue. It must not describe the current workaround, phase, release,
  partial inventory, or completion percentage, and it must not qualify the normative
  requirement.
- Put sequencing and status in `spec/design/`, `docs/`, GitHub trackers, or OpenSpec
  proposals/designs/tasks. A chapter-to-capability transfer records implementation
  divergence in the owning OpenSpec change, never by copying that status into the
  receiving capability spec.

### Numbered Specs Decide; Design Docs Implement

- **`spec/00-12*.md` is the authority on WHAT the language does and HOW it must
  behave.** Any decision about semantics, types, dtypes, syntax, effects, op behavior,
  diagnostics, or another user-visible contract belongs here. This is the only tier
  that outlives the work that produced it. (For a chapter with a recorded transfer,
  the Documentation Authority rules above hand that chapter's subject to its
  controlling `openspec/` capability spec - the tier boundary is unchanged: the
  normative tier decides, design docs implement.)
- **`spec/design/*.md` is the authority on how we IMPLEMENT and SEQUENCE those
  decisions**: phase plans, oracles, module layout, privacy contracts, migration
  order, consumer maps, evidence. A design doc may elaborate a numbered-spec rule and
  should record the reasoning behind it, but it does not get to decide one.
- **Where the two disagree, the numbered spec wins and the design doc has a bug.** Say
  so in the doc when you find it rather than reconciling silently in code.

Two rules follow, and both are cheap:

1. **When a design doc states a rule that is really a language decision, lift it into
   the numbered spec and leave a pointer behind.**
2. **Watch for permission-to-mandate escalation.** "X is a conforming implementation"
   in a spec does not license "therefore we do X" in a design doc, and neither
   licenses "therefore we do X everywhere" in code. If your implementation needs a
   stronger rule than the spec states, amend the spec first and say so in the PR.

A design doc is a working artifact: read constantly while its phases are in flight,
unread once they ship, so a decision parked in one does not survive the work that made
it. The 2026-07 three-level f64 drift that produced both rules, and the worked examples
of each, are in [`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §1.

### Challenge Written Designs Before Implementation

- A design doc is not correct merely because it was written down. Before implementing
  it, test its claims against the controlling normative spec, hardware and ecosystem
  reality, and Chelis's stated principles.
- If a design is wrong, over-broad, or drifted, stop and amend the controlling document
  and tracker before writing code. Do not faithfully compound a bad decision or
  reconcile a design/spec conflict silently in implementation.
- Treat permission-to-mandate escalation, compatibility assumptions without a current
  requirement, and deferred structural fixes as reasons to challenge the design.

### Explicit And Structural Design Bias

- When the controlling normative spec leaves a real design choice, prefer explicit
  spelling over contextual inference. Chelis is machine-generation-first, so human
  typing convenience does not outweigh unambiguous types, effects, dtypes, versions,
  or adaptation rules.
- Chelis is pre-compatibility unless a controlling contract says otherwise. Prefer the
  clean comprehensive design that makes a defect class structurally impossible over a
  smaller-blast-radius patch, a legacy default, a versionless compatibility fallback,
  or unnecessary phase deferral.
- If a design doc permits both a structural solution and a permissive stopgap, amend it
  to select the structural contract before implementation. This bias never overrides a
  normative semantic rule; amend that rule first when the language decision must
  change.

**Normative inventory registries.** A file under `spec/registry/` is
numbered-spec-tier content, not a design doc: each is incorporated by
reference into its owning `[05-OP-N]` atom, carries identity-keyed rows with
no semantic ordinals, and is amended only as a numbered-spec change under the
same review discipline and guard oracles as the chapter that owns it. The
atom keeps the semantic rules; the registry holds the enumerable identities,
and neither may duplicate the other's content.

### Numeric Surface Discipline

The numeric remediation's covered-family surface ratchet and typed entry
edges (`spec/design/dtype_semantics.md` §C6) bind every change that touches
numeric data, whether or not you have read that document:

- **No numeric channel outside the tagged carrier.** A public ADT variant, wire
  field, exported C signature, exported C data declaration, or binding parameter
  or result that carries numeric values as
  bare `f64`/`double`, or that takes a raw integer dtype id, is a review-blocking
  finding. Zero-exception classification is the landing rule: every discovered row
  must end in exactly one final authority class - structurally nonnumeric, a
  structurally recognized exact tagged carrier/transport, or an exact numeric
  operation registration. Historical foundation snapshots record the old 39
  grandfather rows, three successor overrides, and 155 permanent plain rows; the
  executable primary census and guard retain none of those disposition lists or
  admission paths. The active primary baseline has completed
  that migration: all 313 rows have final authority as 74 exact structurally
  nonnumeric rows, 16 structurally recognized tagged carriers/transports, and
  223 registered numeric operations (the [05-OP-35] stdlib registrations among
  them). It retains no grandfather, successor-override, permanent-disposition,
  integer-plumbing, or other transition rows. The wire baseline has 96 final
  numeric leaves: 79 verified transports and 17 exact numeric operations.
  WireDag v9 includes the `WireDagNode.shape_deps: u64` transport and the fixed
  `NonnegativeExtent` carrier's literal-witness requirement role. Its
  private verifier requires current graph, codec/admission, cache, publication
  and mutation execution; neither a static descriptor nor a baseline grants
  wire authority. Nine source/name/path/vocabulary and opaque-handle bindings have final
  nonnumeric authority. Seven bindings have final tagged-transport authority: four
  compiler-JSON functions, the compiled-model tensor call, and the two DLPack methods.
  `NativeTensor.shape` has exact numeric-operation authority under [05-OP-45].
  The binding census retains no legacy cohort; every discovered row requires current
  graph and execution authority. The backend-header baseline has ten final rows:
  the exact generated `chelis_gpu_tensor` tagged transport and nine [05-OP-33]
  device-owner operations, discovered through the complete HIP support root
  under the committed Phase-0 SDK stubs, a fixed target and freestanding
  standard-library-free include search. Canonical include attribution fails if
  it escapes that declared universe. The recursively discovered HIP header set,
  not a basename allowlist, selects attributed files before row extraction, so
  every declaration in a reached nested support header reaches authority
  comparison while SDK fixtures remain inputs only. The complete recursively
  discovered published Metal header set exports no ABI row today; an executable
  enrollment gate scans every `.h` with the shared C-family lexer and fails when
  any first does, so braces in comments or literal payloads cannot hide a later
  declaration and no vacuous Metal lane or workstation SDK can stand in for
  coverage. A bare
  numeric carrier has no citation or maintainer-override path:
  redesign it onto the tagged carrier or remove it. Opening a fresh issue does not
  authorize capacity debt. No grandfather, permanent-disposition,
  successor-override, or integer-plumbing path is part of the final contract. A language-level stdlib ADT
  field such as `t-prim f64` remains type-tagged by its declared Chelis type; the ADT
  constructor is therefore a `numeric-op` requiring exact semantic registration,
  not a C-style untagged carrier seam. Integer fields follow the same rule. Every
  such constructor, irrespective of age, owes the semantic registration below.
  Source-faithful ingestion still preserves distinct variants such as
  `JsonInt(int64)` and `JsonFloat(f64)`; one float funnel is not an equivalent
  tagged source model. The
  complete inventory is exact and bijective with its authority map; regeneration
  cannot bless an unclassified row. A rename, signature change, or reclassification
  removes the old identity and adds a successor that independently satisfies the
  final rule. The pre-Phase-1 baseline is deliberately partial: wire-schema and PyO3
  coverage become mandatory only through their named executable entry gates in §C6,
  and Phase 1 may not start before both are green. Coverage state comes from the
  test's typed `coverage_manifest()` (artifact, enumerator, command, expected
  success, and mutations), never an editable field in the baseline JSON.
- **Every new or changed numeric op requires an exact semantic registration in the
  same change set.** The owning family
  registry binds the callable's exact canonical identity to
  one verbatim, existing `[05-OP-N]` authority in
  `spec/05-risc-primitives.md`. A callable without an existing governing atom authors
  that atom and its mapping together. A chapter substring, `[05-OBS-1]`, an absent
  atom, or a Rust doc comment
  is not authority; existence means a normative definition line beginning
  `> **[05-OP-N]**`, not a cross-reference elsewhere. The atom states the signature,
  per-dtype semantics at [04-NUM-8]'s declared widths, adjoint or
  non-differentiability rule, and accumulator rule where applicable; it may
  hold its enumerable identity table in the normative `spec/registry/` file
  it incorporates by reference, and rows there are part of the atom's
  normative content. Tooling validates
  the atom group and existence; reviewers validate that the selected atom's normative
  text actually governs the callable. Review does not confer semantic authority: if
  no existing atom governs it, amend the numbered spec first and register that new
  atom. Before allocating its number, re-check the highest existing `[05-OP-N]` on
  current `main`. The chelis#1288/#1293 prerequisites apply the same requirement to
  every surviving legacy row; age and an old census disposition are not authority.
  Authoring the atom also requires running
  `.venv/bin/python scripts/generate_rejection_registries.py --write` and committing
  the resulting `crates/chelis-types/src/rejection_registry_generated.rs`; that
  generated membership artifact is required in addition to, and is not a substitute
  for, the exact `SemanticRegistration`.
- **Never silently narrow at ingress.** Choosing a lossy dtype for ingested data
  (int64 IDs into an f32 tensor) is a decision: use the named lossy form (the
  chelis#759 pattern) or the exact dtype, never a quiet convenience cast. Better
  still, type the boundary so the checker can defend it: ingestion APIs preserve
  the source format's numeric distinctions as ADT variants (`io/json`'s
  `JsonInt(int64)` beside `JsonFloat(f64)` is the precedent), never one float
  funnel.
- **A new surface KIND** (a new serialization format, IPC channel, or export
  mechanism that can carry numbers) extends the §C6 enumerators in the same change
  set, or does not land.
- **Published C ABI is configuration-invariant, and every declaration in it is
  attributable.** Public declarations may not vary
  by preprocessor feature/context. The §C6 header leg enforces that prohibition and
  compares toolchain-stable canonical declaration identities, never a
  preprocessor's whitespace or pretty-print spelling. Published headers may contain
  no `#line` directive or hand-written linemarker - those rewrite the file
  attribution the census reads back from `cc -E`, which can delete a real callable
  export from the inventory - and every `.h` in the published include directory must
  be reachable from a declared root.
- **C numeric-callable classification is conservative.** A non-boolean,
  non-character built-in arithmetic value type - including bare `int`, `short`,
  `long`, signed/unsigned forms, pointer-sized integers, and the exact-width integer
  types - makes a callable `numeric-op`. Names and parameter-name heuristics never
  turn a callable into plumbing. The executable primary census has no
  integer-plumbing exceptions; every arithmetic declaration follows the conservative
  classification rule. The final rule
  registers extents, allocation sizes, indices, and dtype selectors as numeric
  operations; raw dtype selectors are forbidden.
  Conditional macro definitions
  likewise taint their whole connected local-include component, by either include
  spelling: a public declaration consuming a tainted token is rejected even when the
  definition lives in another header. These conservative classification and
  context rules are locked by PR #956 commit
  `6ddf1a72d6dea6770a330d5c2ef3b8fa7d023c43`; weakening either rule requires
  changing this contract and the negative controls together.
- **An arithmetic spelling the census does not recognize is a BUILD FAILURE, not an
  unflagged row.** The closed list is the non-numeric one
  (`NON_NUMERIC_C_TYPE_WORDS`): qualifier and aggregate keywords plus the two
  non-arithmetic value spellings. An allowlist of arithmetic spellings can never be
  complete - `_Float16`, `__fp16`, `__bf16`, `_Decimal64`, and `__int128` were all
  classifying as dtype-free - so the rule is inverted. If you add a type word to a
  published header and the census rejects it, decide which list it belongs in; do
  not route around it. The same lock exists on the language side: adding a variant
  to `chelis_types::Prim` stops the tripwire compiling until the new dtype is
  classified. Typedef aliases are resolved before classifying, including array
  aliases (`typedef double chelis_vec4[4];`), and a typedef shape the resolver
  cannot read is rejected rather than skipped.
- **An exported stdlib `def` declares its signature.** The census reads a public
  def's numeric capacity off its `defsig`, so an exported def that declares none is
  numeric surface nobody can see. Declare it, or stop exporting the binding.
- The census, tripwire, and oracle files are guard artifacts: editing one to make
  your change pass is never the fix. The failure message names the sanctioned
  actions; take one of those.

### Public-Surface Change Rule

When behavior changes, update the owning code, tests, docs, and examples in the same
change set:

- parser / type system / IR / backend tests
- CLI integration tests
- executable examples in `examples/`
- active specs and current-state docs

## OpenSpec (captured capabilities and advisory validation)

The `openspec/` tree is the canonical OpenSpec planning and capability root.
OpenSpec planning is optional. OpenSpec validation does not gate merges for
human-reviewed changes. `spec/design/spec_provenance.md` describes the future
governance regime. This regime is not active.

### Submitting an OpenSpec document change

**Push the branch. That is the whole workflow.**

```sh
git switch -c openspec/add-thing
# edit openspec/** only
git commit -am "docs(openspec): add the thing"
git push -u origin HEAD
```

A push to any branch except `main` starts `openspec-autoland`: a
permissionless signal workflow fires a `workflow_run`, the trusted
controller re-derives the branch, commit, and repository from the API,
classifies that exact commit with default-branch code, and opens or reuses
one internal pull request. The merge worker then waits for the required
checks on that commit and merges it. No local command is needed and no
human approval is involved.

`openspec-submit` (inside Devenv) is an optional local helper that
validates before you push and reports the outcome in your terminal. It is
not required, and it is not how a change lands.

Autoland opens the pull request with a short-lived installation token,
minted per run from the `chelis-openspec` GitHub App, which exists for this
mechanism alone (`vars.OPENSPEC_APP_ID`,
`secrets.OPENSPEC_APP_PRIVATE_KEY` -- not the shared `CI_APP_*`
credentials, whose installation grants no pull-request write), scoped to
this repository and to `Pull requests: write` with `Contents: read`, and
revoked when the job ends. The
built-in `GITHUB_TOKEN` cannot be used, because a pull request opened with
it starts none of the required checks. If the App is not configured, or its
installation lacks that permission here, the controller reports the push
blocked and writes nothing -- it never opens a pull request that could not
merge. `README.md` has the details.

A push that touches anything outside the document set is classified
`review`, nothing is written, and the change follows the ordinary path. Do
not mix a document change with code in one branch if you want it to land
automatically; split them.

The document set is `openspec/project.md`, Markdown under
`openspec/specs/`, and Markdown or `.openspec.yaml` under
`openspec/changes/`. Normative capability specifications are included, per
the maintainer authorization in `spec/design/spec_provenance.md`
§ Automated acceptance of OpenSpec documents.

Everything else keeps the ordinary review and test path, including any
change to `openspec/config.yaml`, `scripts/openspec_*.py`,
`scripts/check_openspec.py`, or any `.github/workflows/openspec-autoland*`
file.

`scripts/openspec_acceptance.py` is the single decision point, and it fails
closed: an empty change set, an unreadable diff record, a symlink, an
executable bit, a submodule pointer, a copy or type-change record, a
deleted capability specification, or a rename crossing the boundary all
route to human review. Do not add a path to its allowlist to make your
change land; that file is itself outside the automatic path.

Your branch must also carry byte-identical `.github`, `scripts`,
`openspec/config.yaml`, Devenv, Nix, and toolchain content, compared by Git
object id. A branch that predates a change to one of those is refused until
you rebase -- deliberately, because a diff cannot tell "did not touch the
workflow" apart from "carries an older workflow".

**Automatic acceptance is not correctness evidence.** It proves the path
boundary and schema validity. Where an accepted OpenSpec artifact contradicts
`spec/**` or an executable oracle, the owning authority controls and the
artifact is corrected.

- The `openspec-validate` workflow uses a pinned `Chelis-Lang/ci` action.
  The action supplies Node 24.18.0 and the locked OpenSpec 1.6.0 package.
  `scripts/check_openspec.py` runs structural validation only.
  Schema findings produce warnings because the action uses advisory mode.
  Operational failures stay nonzero. The workflow uses only `contents: read`.
- `openspec validate` enumerates active changes and specifications only.
  It does not enumerate artifacts under `openspec/changes/archive/`.
- Local validation requires OpenSpec 1.6.0:

```sh
openspec validate --all --strict --no-interactive
```

- The captured capabilities cite their source chapters. Capture alone does not
  transfer authority.
- No numbered chapter has been transferred. The numbered chapters therefore remain
  controlling, and their captured capabilities are reference material.
- Use the documentation authority rules above for all chapter and capability
  disagreements.
- `spec/design/spec_provenance.md` § OpenSpec boundary blocks the first chapter
  transfer until a separate change amends that boundary.
- Do not run `openspec init`'s tool generation. `.claude/skills` and
  `.codex/skills` are symlinks to `agent-skills/`, so `--tools claude,codex`
  writes generated skill trees into the shared skill library through both
  paths. OpenSpec is not yet wired into the agent workflow — drive the CLI
  directly.

## Issue Tracking Conventions

### One Tracking Issue Per Class

A recurring defect class gets **one tracking issue**, which is also the GitHub
**sub-issue parent** for every instance, and no second issue beside it: do not file a
separate META issue to hold the class statement or the instance list. Its body carries the plan (phases,
oracles, freeze points, the class statement); its evidence lives in the owning
design doc under `spec/design/` or in `docs/investigations/`.

Rules:

- When filing an issue, always check to see if it should be grouped under a relevant
  tracking issue.
- An issue has **one** parent. When a defect splits across classes (the #689
  shape: a silent half and a support half), parent it to whichever class's
  **oracle turns green when it is fixed**, and add an explicit `Also part of #N`
  line or comment for the other. Do not leave the second half implicit; that is
  how a half gets dropped when the first parent closes.
- A tracking issue carries the `tracking` label so it can be excluded from the
  work queue. **`-label:tracking` is the work queue.**
- A class without a design doc is legitimate. Say so in the tracker ("no design
  doc is planned; this is a parent, not a plan") rather than leaving a reader to
  wonder which doc they failed to find.

### Labels

`tracking` and the `area:*` family carry navigation; `bug` and `soundness`
carry severity. Prefer the specific one - an issue labelled only `soundness`
is not findable by anyone who does not already know it exists.

| label | means |
|---|---|
| `tracking` | a hub: a class or a plan, not a work item |
| `spec-gap` | normative text was never authored. Distinct from `design-discussion`, which means the decision exists and is contested |
| `soundness` | semantics divergence, type-safety, or a wrong answer. **Not** CI tooling |
| `area:eval` | the `chelis eval` interpreter lane |
| `area:runtime` | `chelis-runtime` and the C ABI surface |
| `area:backend` | backend codegen: C, HIP, Metal, and IR lowering |
| `area:prove` | `chelis-prove`, SMT, contract discharge |
| `area:bindings` | Python bindings and the compiler-api embedding surface |
| `area:ecosystem` | shell repos, `reef conform`, ecosystem drift |
| `area:perf` | performance and benchmarking |
| `area:ci` | CI workflows, coverage, mutation testing, dev infrastructure |

`type-system` and `cli` predate this table and keep their existing meanings.

## Contract Invariants

Machine-facing contracts should be expressed as invariants and locked with tests.

Examples:

- if a command reports perfect success, its error list must be empty
- formatter output must remain parseable on the supported corpus
- decompiler output must round-trip through the supported parser path
- executable examples must remain executable after canonical formatting
- status docs must not claim a stronger guarantee than the repo actually proves

When a public surface has an implicit invariant, make it explicit and test it.

## Example Corpus Policy

- `examples/` means executable Phase 0 examples that should survive `chelis fmt` and
  `chelis check` cleanly.
- `examples/illustrative/` is for syntax or design examples that are useful but not on
  the executable Phase 0 path.
- Do not mix those meanings in tests or docs.
- Decide the executable-vs-illustrative split early in a phase, not after examples have
  already been used as proof artifacts.

## Manual Gates

- Every manual acceptance gate must have a documented command, expected success condition,
  and owning phase.
- If default CI does not run the gate, the docs must say so directly.
- Ignored tests are allowed only when they clearly mirror a documented manual gate or an
  environment-dependent prerequisite.
- Phase summaries must not imply that a manual gate is part of the default workspace pass
  when it is not.

## CLI Surface Discipline

- CLI commands are part of the product surface, not wrappers around library tests.
- Formatter, decompiler, evaluator, checker, and build-command behavior should be tested
  against a corpus, not only single happy-path examples.
- For machine-facing CLI output, test both shape and semantic invariants.

## Scripting Language Policy

- **Python** for all scripts, utilities, report generators, and automation helpers.
  Write tests for them.
- **Rust** where the task naturally fits a compiled workspace member.
- **Never shell.** Do not write `.sh` scripts. If a CI step needs a one-liner, invoke
  Python instead. Shell is fragile and untestable.
- **The exception list has one entry, and `spec/01-nomenclature.md` §2.9 controls it:**
  `crates/chelisup/bootstrap/chelisup.sh`, the `curl ... | sh` one-liner that runs on a
  bare machine before Chelis, Cargo, or Python exists. It is the only committed `.sh`
  file in the repository and the only entry the `no-shell-scripts` lint exempts
  (`style_gate.rs::exceptions()`). Adding a second entry is a numbered-spec change, not
  a judgment call.
- **Three other sanctioned shell artifacts exist and never reach that list**, because
  the lint cannot see any of them: the Nix `chelisup` launcher is generated at build
  time rather than committed, and `.githooks/commit-msg` and
  `.cargo-husky/hooks/commit-msg` have no file extension, so `chelis_lint`'s
  `Surface::classify` - which keys on the extension - never classifies them. All three
  must still be minimal POSIX `sh` and `shellcheck`-clean, and you have to verify that
  yourself: CI checks only the two hooks, with `sh -n`. Nothing checks the bootstrap,
  and the launcher's `chelisupLauncherLint` uses `bash -n` rather than `sh -n` and has
  not run on a pull request since chelis#1450. All other scripts remain Python.
- Existing `scripts/` directory uses Python; follow that convention.
- **Use a uv-managed Python**, not the system Python, for every script and every
  ad-hoc invocation. [Build Toolchain](#build-toolchain) owns provisioning, the
  resolution order, and the direct-invocation forms.
- **Two diagnostics are deliberately bootstrap-free.**
  `scripts/reap_orphans.py` and `scripts/preflight_exec_probe.py` are invoked
  as bare `python3` on purpose: they run *before* and independently of a
  working project environment, which is exactly when a uv re-exec would be
  the thing that is broken. They therefore MUST stay standard-library only
  and MUST keep parsing on the oldest system Python a supported workstation
  ships (macOS still ships 3.9), which is what the
  `from __future__ import annotations` header in each buys. Neither
  exemption extends to any other script: everything else, and all ad-hoc
  scripting, uses a uv-managed interpreter.
  `scripts/test_bootstrapless_scripts.py` locks both properties.

## Worktree And Branch Discipline

- Treat the primary checkout for the current clone (the main worktree listed by
  `git worktree list --porcelain`) as live developer state. Its path is specific to
  the developer and machine and must not be hard-coded in repository policy. Agents
  may run read-only queries there, but must not switch branches, edit files, build, or
  create or remove scratch artifacts in it.
- Create a dedicated worktree before the first write for every task, including small
  documentation edits and throwaway probes. Keep its branch, target, and scratch state
  task-owned, and give it its own environment per
  [Build Toolchain](#build-toolchain); never copy or symlink the primary `.venv`.
- A worktree isolates the working tree, the index, and its own HEAD reflog. It does not
  isolate the stash stack, `.git/info/exclude`, the hooks directory, or branch reflogs:
  those are per repository, so every worktree on a clone shares one copy of each, and a
  write to any of them is a write to every sibling's state. Having worked out that a
  worktree's `.git` is a file pointing at the shared directory is not the same as
  acting on it.
- Do not run `git stash` in a shared clone. The stack is one stack for the whole
  repository: `push` with a pathspec that matches nothing is a silent no-op, the
  paired `pop` then takes whatever a peer session left on top, and `pop` says nothing
  about whose work it just applied to your tree. To discard your own changes use
  `git checkout -- <paths>`. To park them, copy the files to task-owned scratch space.
  [`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §5 records the
  incident.
- Do not repurpose an unrelated worktree because it appears idle. Reuse is allowed only
  for the same PR or immediate follow-up work after checking ownership, exact head,
  status, and active processes.
- When a PR is otherwise ready to merge and `origin/main` has advanced, do not rebase
  merely to refresh its base. Fetch current refs, check GitHub's current mergeability,
  and inspect the prospective merge result with `git merge-tree` or an equivalent
  temporary integration. If GitHub's merge produces the intended semantic and
  structural result without a dangerous conflict, preserve the reviewed head and its CI
  evidence. Rebase only when that result differs, is unsafe or unclear, or another
  identified semantic or structural issue requires a changed head.
- A non-trivial rebase or hand-resolved conflict requires review of the resolution
  before any history rewrite is published. Run `python3 scripts/gate.py --fast` on the
  result for a non-documentation change or the focused documentation checks for a
  docs-only change. A content-preserving rebase may retain supporting local evidence;
  record that its covered head differs, and require CI on the new head. Never
  force-push a red gate. Obtain approval, then use an exact-head
  `--force-with-lease`. A clean mechanical rebase needs no resolution review, and
  neither does one whose only hand-resolved conflicts are generated registry lines:
  regenerate `rejection_registry_generated.rs` with
  `scripts/generate_rejection_registries.py --write` and treat that result as
  mechanical. Atom/region text hashes have been retired; contract changes still
  owe their required clauses, semantic checks and exact PR acknowledgements.

## Build Toolchain

A uv-managed Python is a hard prerequisite on every platform: `chelis-python` links
against `libpython`, and the gate scripts and several tests need an interpreter. Install
uv from <https://docs.astral.sh/uv/getting-started/installation/>, verify it with
`uv --version`, and provision the version with `uv python install 3.11`.
`py/pyproject.toml` pins `requires-python = ">=3.11"`. See [`README.md`](README.md) for
the full setup.

**Primary path (uv).** Create a checkout's environment once with `uv venv --python 3.11`,
at the root of the primary checkout and at the root of every dedicated worktree. Never
copy or symlink another checkout's `.venv`. Three invocation forms follow, and this
repository uses them consistently: every script is `.venv/bin/python scripts/<name>.py`;
the gate is always `python3 scripts/gate.py --fast`, `--local`, or `--list`, never
`.venv/bin/python scripts/gate.py`; and the
`uv run --managed-python --python 3.11 --no-project python ...` form appears only where a
document explains what the gate re-executes to, and as the fallback when no `.venv`
exists, never as a routine invocation.

**Resolution.** Rust test and gate code resolves an interpreter through
`tests/support/managed_python.rs`: an explicit `PYO3_PYTHON` wins outright, and the
checkout's `.venv/bin/python` is the fallback. Outside Devenv, `.cargo/config.toml`
points `PYO3_PYTHON` at `.venv/bin/python`; Devenv overrides it with
`.devenv/state/venv/bin/python`. For direct Cargo commands in a dedicated worktree,
export `PYO3_PYTHON="$(uv python find 3.11)"`. An explicit `PYO3_PYTHON` is
authoritative: an invalid path must fail loudly rather than fall back.

**The gate.** `scripts/gate.py` is stdlib-only and is the one `python3` entry point that
self-heals: when its runtime is not already uv- or Devenv-managed it re-executes itself
through the uv command above before running gate logic, so `python3 scripts/gate.py` is
correct in every environment. It exports its selected interpreter as `PYO3_PYTHON` to
every child command, never requires a checkout-local `.venv`, and its `--fast` and
`--local` preflight warns when the worktree has no `.venv/bin/python` (create it with
`uv venv --python 3.11`); only direct cargo and nextest runs outside the gate need it,
because they fall back to it when `PYO3_PYTHON` is unset.

**Alternative path (Devenv).** Inside an activated Devenv shell the environment at
`.devenv/state/venv` serves the same purpose as `.venv` and needs no separate step: invoke
scripts as `python scripts/<name>.py`, and `chelis-gate` forwards every argument to
`python3 scripts/gate.py`. Devenv is optional and, because each command pays shell
evaluation and activation unless it runs inside one persistent shell, slower per command
than the primary path; agent fleets use the primary path. `README.md` holds the Devenv
detail.

**macOS:** Apple's bundled Python reports a stale `sysconfig.LIBDIR` path. Do not route
PyO3 to it.

**C front end.** The chelis#893 Phase 0 oracle and the `chelis-repr-inventory` tests read
the registered C and Objective-C headers through a `clang` binary on PATH (any clang that
prints `-ast-dump=json`), in addition to the `cc` the capacity census already requires;
a missing `clang` fails the scan loudly rather than skipping it.

## Local Git Hook

`.githooks/commit-msg` is the tracked commit-msg hook. It runs
`scripts/check_commit_message.py` through Devenv, `.venv`, or
`uv run --managed-python --python 3.11 --no-project`, in that precedence order.

Devenv **copies** it into the shared hooks directory on shell entry, resolved
with `git rev-parse --git-common-dir` so a linked worktree installs to the same
place. The installed copy names no worktree and resolves its repository at run
time, so one copy is correct from every worktree and on every branch, including
branches predating it.

Do not reach the hook through `core.hooksPath` instead. That config is
repository-scoped while a tracked file is branch-scoped, so a worktree on a
branch without `.githooks` would run no hook at all and accept the commit
silently, with the `.git/hooks` fallback disabled by the same config. Do not
reinstate an installer that writes an absolute path either: that names one
worktree for every worktree, and all of them lose the ability to commit once it
is deleted (chelis#1409).
`scripts/test_commit_hook.py` locks the tracked path, the absence of any
absolute path, and the accept/reject behavior.

Cargo-husky remains the fallback for the manual setup: `cargo test` installs
`.cargo-husky/hooks/commit-msg`, a POSIX wrapper that runs the same checker through
Devenv, `.venv`, or `uv run --managed-python --python 3.11 --no-project`. It is
worktree-agnostic for the same reason the tracked hook is, and the two are compatible
because both resolve the repository at run time rather than naming one worktree.

All formatting and lint hooks remain disabled. CI remains the remote
enforcement boundary.

## Commit And Pull Request Hygiene

### Changelog Fragments

- A behavior-changing PR must add a fragment in `changelog.d/`, using
  `<pr-or-slug>.<added|changed|fixed>[.breaking].md`. Write the entry without its
  outer bullet; mark breaking changes through the filename suffix.
- Correct an existing pending fragment when follow-up work changes its claim.
- Internal work with no release-note value may use the `no-changelog` PR label.
  It suppresses only the missing-fragment requirement. Malformed fragments and
  direct changelog edits still fail the required `Changelog` CI check.
- Reserve `CHANGELOG.md` edits for release assembly. The release author runs
  `.venv/bin/python scripts/changelog.py build --version VERSION --date YYYY-MM-DD`,
  reviews the preview, then repeats with `--write`. Commit the assembled notes,
  fragment deletions, and version bump together. Never recreate `[Unreleased]`.
- [The fragment contract](changelog.d/README.md) owns the format, examples,
  publishing behavior, and focused acceptance command.

### Commit Messages

- Use plain conventional commit messages with the configured human author. Do not add
  `Claude-Session`, Codex/Claude attribution, AI co-authorship markers, or AI-session
  links to commit messages or PR bodies.

### Issues Are Closed Manually

Automatic issue closure is disabled for this repository. `Closes #N`, `Fixes #N`, and
`Resolves #N` have no effect from a pull request body, from a branch commit message, or
from the squashed default-branch commit. **Merging a pull request never closes an
issue.**

Close an issue as a separate, deliberate step, once the work is done and verified:
`gh issue close N` with a comment naming the merged pull request, or the same action in
the web UI. A pull request that finishes an issue still says so in its body; that
sentence is a claim addressed to a reviewer, not an instruction to GitHub.

Prefer `Part of #N` or `Addresses #N` when a pull request advances an issue without
finishing it. That is the accurate phrasing either way, and it keeps intent legible if
the repository setting is ever restored.

## Build And Gate Commands

Agent gates for non-documentation changes:

```sh
python3 scripts/gate.py --fast    # before every push: fixes in place, then checks
python3 scripts/gate.py --local   # optional troubleshooting and local validation
python3 scripts/gate.py --detach --local   # optional run, detached
python3 scripts/gate.py --status [HANDLE]  # the detached run's real verdict
```

`scripts/gate.py` is the single source of truth for the per-PR gate. `--fast` is the
pre-push gate: fix-in-place, run before every push. CI on the pushed candidate owns
routine PR validation and must pass before ready-for-review. `--local` is an optional
way to reproduce checks on the developer's machine; no per-PR run is required.
The full workspace and broad phase oracles run in `heavy-e2e.yml` daily at
03:17 UTC and on manual dispatch. Passing PR checks does not certify those
phase acceptance oracles; dispatch them on the candidate when claiming completion.
The PR test worker calls `python3 scripts/gate.py ci-fast`: every default-feature
lib/bin unit target plus the reviewed integrations in `.config/ci-test-targets.toml`.
The legacy/full/manual gate selections remain available. See
[`docs/ci_validation.md`](docs/ci_validation.md) for cadence and artifacts.
`scripts/test_gate.py` pins the complete ordered set of
single-line `run:` commands permitted in those gate-owned jobs, so shell syntax
cannot hide an unreviewed command. To see the canonical full list and the
fast/local/CI ownership annotations:

Before `--fast`, `--local`, or another long local validation, fetch `origin/main` so the
changed-crate selection and inherited-failure comparison use current evidence. If the
branch is materially behind, reconcile it deliberately before spending hours on a
stale tree; do not rewrite shared history without the rebase review gate above. When an
unrelated failure appears, reproduce or compare it on current `origin/main` before
diagnosing it as branch-owned.

```sh
python3 scripts/gate.py --list
```

Each printed command carries one of four annotations: `fast + local + ci` (the lint
row, which `--fast` and `--local` share), `local + ci`, `ci-owned`, and `full gate; CI
coverage split` (the workspace nextest row). Two trailing `#` notes describe the
dynamic stages: what `--fast` runs, and the per-crate nextest `--local` appends. The
gate's own output is the only authoritative list; this file does not transcribe it.

`python3` is only the gate bootstrap. An unmanaged invocation re-executes via
`uv run --managed-python --python 3.11 --no-project`; an active Devenv or
uv-created environment is preserved. If uv is missing, the gate exits with
installation and Python-provisioning commands.

That re-exec drops `UV_PYTHON_PREFERENCE` from the child environment. uv reads
it as `--python-preference` and rejects it beside the `--managed-python` the
gate passes deliberately, so an ambient setting (Devenv exports `only-system`)
otherwise turns the self-healing bootstrap into a hard `error: the argument
--managed-python cannot be used with --python-preference`. Nothing else in the
environment is altered: `UV_PYTHON_DOWNLOADS=never` still fails loudly rather
than letting the gate fetch an interpreter behind that choice.

Inside Devenv, `chelis-gate` runs the gate under the activated
`.devenv/state/venv` interpreter, the same one `PYO3_PYTHON` names and the same
one a developer gets by typing `python`. A Devenv script can only name a Nix
package, so the launcher starts under the bare store CPython and hands the
script the activated interpreter; without that handoff the gate reads its own
runtime as unmanaged and re-executes when it should not.

The gate runs `cargo nextest run --no-fail-fast` (CI's actual runner), not
`cargo test --workspace`, and includes `chelis lint --check .` (the §8.6 /
§12 naming gate). The sanitizer, macOS-smoke, LOC-report, no-AI-authorship, docs, and
smt-build CI jobs are out of scope for this script by design.

The three explicit rustdoc stages exist because `cargo nextest` does not
execute doctests, and the gate deliberately does not use `--workspace --doc`.
`scripts/gate.py`'s module docstring records why, and what each stage covers.

**Doctests only run where something invokes them.** The canonical gate
invokes doctests for `chelis-types`, `chelis-compiler-api`, and
`chelis-pipeline-core`. The `backend-sanitizers` job also runs
`cargo test -p chelis-backend-c --lib` and an explicit `--doc` invocation.
Full backend sanitizer integrations run nightly in `heavy-e2e.yml`. A `compile_fail` oracle in
another crate runs nowhere until that crate gains an equivalent invocation in
the same change set.

`scripts/unrepresentable_domain_oracle.py` is chelis#908's authoritative
completion oracle, and the tracker requires every fix in that class to run
it continuously. The full command runs in `heavy-e2e.yml` daily at 03:17 UTC
or on manual dispatch, not in the required PR integration job. Acceptance is
exit 0 with a final `ORACLE: PASS` line.

`--local` (chelis#360) is an optional local validation command. It runs
two of the three workspace clippy configurations (`-D warnings`, compile-only): the
default row and the solver-free-features row. The `--no-default-features` row is
CI-owned through `gate.py lint-and-unit`, because `check_configuration_closure.py`
reconciles every repository `.rs` file against the dep-info in the worktree's target,
`crates/chelis-prove/src/clarabel_sos.rs` is compiled per pull request only by the
solver-free row, and the no-default row compiles a strict subset of the default row.
It then runs `cargo fmt --check`, `chelis lint --check .`, the deterministic
std-bundle regeneration check, the explicit rustdoc commands, the checkpoint and
hash-order compile-fail fixtures, the configuration-closure check, both pipeline-core
guards, the chelis#908 unrepresentable-domain oracle, the runtime-representation
Phase 0 oracle, and `cargo nextest run -p <crate> --no-fail-fast` for each crate
changed vs `origin/main` (committed diff plus uncommitted work; owning packages are
resolved from each member's `Cargo.toml`, not the directory name). The derived crate
list is always printed; "no crate changes detected" means the per-crate stage was
skipped, not silently empty. The workspace nextest stage is CI-owned: run `--fast`
before every push, push before the review round so the reviewer and CI see the same
head, and require CI on the candidate before ready-for-review. Mac workspace
validation runs daily at 04:17 UTC in `macos-nightly.yml` and on manual dispatch.
It is not a required PR check; dispatch it on the branch for Mac-specific changes. See [`docs/local_macos_environment.md`](docs/local_macos_environment.md)
for why the workspace suite does not belong in the local loop on macOS.

If a cold `--local` run is useful, launch it with
`python3 scripts/gate.py --detach --local` and collect the
result with `python3 scripts/gate.py --status [HANDLE]`. The launcher's exit code is a
launch verdict and nothing more: the run's own exit code, exit 4 for a lease timeout
included, arrives through `--status`. Record the `--status` verdict and the head it
covered, never the launch.
[`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §8
has the runs behind the evidence and invocation rules.

`--fast` is the inner-loop pass. It fixes in place and prints what it changed:
`scripts/regen_all.py --tier 0` (the rejection registry, the embedded conformance
skill assets, and the opaque-invariants corpus) and `cargo fmt --all`, then
`chelis lint --check .`, `cargo clippy -p <crate> --tests -- -D warnings` for each
changed crate, one `cargo nextest run` over the drift tripwires (atom partition,
generated dtype header, compiler pins, opaque corpus, loud-unsupported, payload census,
bundled std loader, conformance manifest, asset drift, skill-set uniformity, phase-3
gate inventory, stack-guard coverage, runtime-extent target manifest), and, when
a `packages/chelis-std/` or `crates/chelis-std-bundle/` path changed,
`regen_all.py --tier 1` right after tier 0 (so every check sees the regenerated
bundle) and `cargo nextest run -p chelis-std-bundle --lib` after the tripwires.
Every writer runs before every check. It
exits non-zero for any failing stage (fmt, regeneration, lint, per-crate clippy, the
tripwire run, or the std-bundle self-test) and never for a file it fixed; a regenerated
`dist/` or `reef.lock` is reported as a changed file to commit, never as a failure.
Changed files are reported from content
hashes of the porcelain set before and after the run, so a file that was already dirty
and that fmt changed further is still listed. It never runs a workspace clippy row,
the chelis#908 oracle, or the runtime-representation oracle, and it never takes the
lease.

`scripts/regen_all.py` is the regeneration entry point on its own as well. Default
tiers 0 and 1 write (tier 1 is the std bundle and needs cargo); `--check` reports every
stale artifact; `--full` adds tier 2, the capacity census and the
runtime-representation inventory. The Python-binding baseline stores only stable
reviewed rows, flags, authorities, and contracts; current graph identities are
execution evidence from the dedicated binding acceptance test and are never persisted
or regenerated. Full regeneration exits 2 naming the manual action when a census row
lands with citation `TODO` or the regenerated inventory's digest differs from the
reviewed `FREEZE_SHA256`. It never writes a frozen digest, the loud-unsupported
`BASELINE`, the binding or wire census JSON, the dtype C header, or the tree-sitter
parsers.

`--fast`, `--local`, and the bare full gate run a preflight before the first command:
the environment checks (`PYO3_PYTHON`, `CARGO_TARGET_DIR` containment, an explicit
oracle handoff; exit 2 on failure), the git facts for the run summary (never fatal), a
warning when the worktree has no `.venv/bin/python`, and, on macOS only,
`scripts/preflight_exec_probe.py` as a subprocess: probe exit 0 proceeds, exit 1
(wedged first exec) stops the gate with exit 3 and the termination class
`preflight-stop`, exit 3 (slow) and exit 2 (could not run) warn and proceed. On Linux
the probe is skipped and the summary records why; the environment checks, the `.venv`
warning, the lease, and the summary behave the same on both platforms. CI stage runs
skip the preflight.

`--local` and the bare full gate hold a workstation-wide `flock` on `gate.lock`
under `$CHELIS_GATE_LEASE_DIR`, else `$XDG_CACHE_HOME/chelis`, else `~/.cache/chelis`.
Updated runners acquire in ticket-registration order. Cancellation, timeout, and
process death release queue places; kernel locks establish liveness, including after
SIGKILL. Holder sidecars are descriptive only.

Waits are indefinite by default, polling every 10 seconds; 60-second heartbeats show
holder details and queue position. `--no-wait` exits 4 on lease/queue contention;
`--lease-timeout SECONDS` caps the wait (exit 4); `--no-lease` bypasses; queue errors
exit 2. `--fast` only reports a holder. Update active worktrees to honor the queue.
Queue mechanics live in `scripts/gate.py`; acceptance:
`.venv/bin/python -m unittest scripts.test_gate_queue`.

Every command's combined stdout and stderr streams live. On failure the gate
retains the complete transcript under `target/gate-failures/`, replays the
final 200 lines, and prints the stage/index, duration, exit code or signal,
relevant environment, exact rerun command, and transcript path. Successful
command transcripts are removed. Every run other than `--list`, pass or fail, also
writes `target/gate-reports/<utc-timestamp>-<pid>-<mode>.json` (or under
`$CHELIS_GATE_REPORT_DIR`): the mode, git facts, preflight and lease records,
per-stage seconds, the first failing stage, the termination class (`pass`,
`stage-failure`, `signal`, `environment`, `preflight-stop`, `lease-timeout`,
`user-cancel`, `internal-error`), and the files a `--fast` run changed; it then prints
one summary line with the stage count, seconds, verdict, and report path. Record the
seconds from that file when citing an optional `--local` run in the pull request.

The gate normalizes `CARGO_TARGET_DIR` to an absolute path inside the current
worktree and rejects paths outside it. It also sets
`CARGO_HUSKY_DONT_INSTALL_HOOKS=1` for child builds, preventing cargo-husky
from mutating the clone's shared `.git/hooks` while sibling worktrees run.
These controls isolate writable state; concurrent agents may still contend
for CPU and make each other slower. The advisory lease serializes `--local` and full
runs across worktrees on one workstation; it never kills another process.

Documentation-only changes require applicable CI on the candidate head. Hosted
docs-only classification is routing evidence, not proof that every changed Markdown
control artifact has an owning validator in that workflow.

Markdown parsed, embedded, mirrored, or used as agent instructions is a control artifact,
not inert prose. Its focused validators must run even when CI reports
`docs_only=true`. The always-run Docs job owns shared-agent-skill validation:

- `scripts/check_agent_skills.py` validates metadata, the registered shared set,
  source/embedded byte agreement, and Claude/Codex red-team wrapper agreement;
- the validator and CI-routing tests exercise failure cases;
- `chelis-conformance`'s `asset_drift_tripwire` and `skill_set_uniformity` tests
  exercise compiled assets and downstream distribution.

Local reruns of these checks are optional. Docs also builds mdBook and validates the
package skill/examples. Phase-specific acceptance oracles and manual gates remain
required; making `--local` optional does not replace those named obligations.

Default-gate discipline:

- Run `cargo check -p <crate> --tests` before the first nextest of a change, then use
  focused `cargo nextest run -p <crate> --test <file>` commands for the inner
  development loop: each compiles only what it names. Do not substitute a
  workspace-wide `cargo test` run for the canonical gate.
- tests that are slow or require heavyweight local prerequisites should be
  `#[ignore]` by default and invoked through a documented manual gate
- every ignored test must have a concrete manual command and expected success condition in
  the owning phase docs

Phase-specific manual gates must be called out explicitly when they are not part of the
default workspace run.

## Build Concurrency And Process Hygiene

- Concurrent agents or subagents that build or test MUST use an isolated target
  directory: set `CARGO_TARGET_DIR=target/agents/<name>`, or use a separate git
  worktree with its own `target/`. A relative `CARGO_TARGET_DIR` resolves
  against the invocation cwd, not the workspace root, so run cargo from the
  repo root or use an absolute path. Never share the primary `target/` with a
  session that may be building concurrently; cargo's target-dir lock serializes
  the builds and feature/profile differences invalidate each other's caches.
- Before building, list orphaned cargo/rustc/cargo-nextest/chelis processes with
  `python3 scripts/reap_orphans.py` and reap them with
  `python3 scripts/reap_orphans.py --kill`. Inside Devenv, use
  `chelis-reap-orphans` and `chelis-reap-orphans --kill`. Review the dry-run
  listing first:
  ppid==1 cannot distinguish an abandoned build from a deliberately detached
  one (`nohup cargo build` you are still tailing). Run it from the checkout
  whose `target/` you are about to use; scoping is per-checkout. Orphaned runs
  keep burning CPU and hold the cargo lock across sessions.
- Before a subagent starts a heavyweight Cargo command, it must report the exact command
  and expected weight to the orchestrator. The orchestrator checks active processes and
  load, then runs, staggers, or declines it; isolated targets prevent state corruption
  but do not eliminate CPU starvation.
- Contention diagnostic: several unrelated tests FAILing at near-identical
  wall-clock times (for example all ~217s, nextest's slow-kill) means CPU
  starvation, not code breakage. Re-run on a quiet machine before treating those
  as real failures; [`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §2 has the
  measured factor.
- At session end, verify that task-owned background cargo, rustc, and nextest processes
  are gone. A stopped wrapper is not proof that its reparented children stopped; use
  the scoped `reap_orphans.py` dry run and kill only confirmed task-owned stragglers.
- macOS workstation only: first-exec assessment can degrade under mass
  fresh-binary bursts and stall multi-binary test runs at ~0 CPU (chelis#356).
  Probe with `python3 scripts/preflight_exec_probe.py` (exit 1 wedged, exit 3
  slow). Inside Devenv, use `chelis-exec-preflight`. Run the probe before the
  local workspace nextest stage. If the probe reports degradation, use the
  manually dispatched `macos-nightly.yml` workflow per
  [`docs/local_macos_environment.md`](docs/local_macos_environment.md).

### Post-Merge Cleanup

- After a PR merges, remove its associated git worktrees and task-owned target builds
  unless they are expected to support immediate follow-up work. Retained artifacts are
  temporary: remove them as soon as that follow-up finishes.
- Before removal, verify the PR is merged, the worktree has no uncommitted work worth
  preserving, and no active process owns its target. Remove only the exact PR-owned
  worktree and regenerable target paths; preserve the primary checkout and unrelated
  worktrees, targets, and user changes.
- Audit cleanup targets with full branch names and live PR state. Because this repository
  squash-merges, "commits ahead of `origin/main`" and missing commit subjects in main do
  not prove that work is unmerged. For an ambiguous old worktree, compare stable patch
  IDs against the PR commits and confirm the added symbols on current main.
- Use `git worktree remove <path>` followed by `git worktree prune`. For a separate
  task-owned target, prefer `cargo clean --target-dir <exact-path>` after inspecting it.
- Remove worktrees with individual explicit commands, never a blanket loop. Worktree
  removal keeps the branch ref; branch deletion is a separate later decision. If macOS
  leaves a partially removed target or `.DS_Store`, re-inspect the exact path before an
  equally narrow cleanup command.
- Close the issues the PR actually resolved, as a separate step; merging does not do it
  (see [Issues Are Closed Manually](#issues-are-closed-manually)). Use
  `gh issue close <N> --comment "resolved by #<PR>"`.
- A merged PR is not proof its issues are resolved: confirm the behavior on current
  `main` first, and when the PR only advanced an issue, leave it open with a comment on
  what remains. Close a tracking hub only when every sub-issue is closed and the
  condition the hub itself names is met.

## Local HIP Environment

HIP manual gates are runnable on this workstation. See
[`docs/local_hip_environment.md`](docs/local_hip_environment.md) for the authoritative
runbook and the environment detail.

**Run every HIP manual gate through** `scripts/hip_test.py` — for example
`scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1`
— or `chelis-hip-test` inside Devenv. Plain `cargo test --ignored` inherits an
incomplete environment and segfaults hipBLAS-linked binaries at process exit with empty
output. That failure looks like a code regression and is not one.

## Subagent Coordination And Delivery

- Every subagent prompt must name the delivery mechanism and the complete expected
  report. A locally written or plain-text report that is not sent through the platform's
  parent-message/final-report channel has not been delivered.
- A subagent must not end its turn merely to wait for a background build, monitor, or
  notification that cannot wake it. Keep ownership of a long command through the
  platform's synchronous wait/poll mechanism, or return the honest partial result and
  unfinished work. A reviewer that has delivered its round report is not waiting: it
  stays available, and the orchestrator resumes it with the fix to verify.
- CI is watched by at most one background waiter whose exit wakes the session, or by
  nobody, in which case the turn ends and the result arrives later. Never watch CI from
  a foreground sleep or poll loop.
- If an agent returns "waiting" or goes idle without the deliverable, the orchestrator
  resumes it immediately with the exact missing items. Prefer a clearly labelled partial
  report over silence or an overstated completion claim, and deduplicate repeated reports
  that race with a resume nudge.

### Fan-Out Budget

A fleet's running cost is the sum of its live contexts plus a shared usage window, and
nothing reports either total. Meter both at the spawn, which is where the decision is
actually made. [`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §3
has the fleet run these numbers come from.

- Spawning more than five subagents live at once under one orchestrator needs the
  user's explicit approval and a stated reason. Five is not a certified safe width. It
  is the widest fan-out measured working here, against seven as the width that broke,
  and the threshold exists to force the decision into the open rather than to bless
  anything under it. This is a separate budget from the CPU one in
  [Build Concurrency And Process Hygiene](#build-concurrency-and-process-hygiene): a
  fan-out the workstation can schedule comfortably can still exhaust a usage window in
  minutes.
- Every spawn names the model tier it runs on and says in one clause why that tier
  fits the work. Reserve the expensive tier for judgement whose errors are costly to
  detect, and give mechanical work the cheap tier: waiting on CI, polling, mirroring
  bytes between files, transcribing a result. A standing instruction to economize is
  not a substitute. The tier is named per spawn, or it was never chosen.
- Every brief states a numeric report-length budget. A numeric context budget is also
  required except for red-team rounds, where it is optional. An agent that will exceed
  an explicitly set context budget says so and returns what it has rather than
  continuing silently, and a report over its length budget is a defect in the report
  rather than evidence that the budget was too small.
- An orchestrator states its own context size in the message that announces a spawn,
  so the fleet's live total is visible to the user without anyone having to ask for
  it.

### Briefs

- A brief that carries facts the orchestrator has already established says so: that
  they are verified, what head or artifact they were verified against, and that the
  agent must not re-derive them. It also names what the agent still has to establish
  for itself, so "trust the brief" does not read as "trust everything". Without those
  two lines an agent re-reads the sources behind the brief, and the orchestrator never
  sees that it happened. Re-spawning is where this bites hardest: a brief reissued
  byte-identically to a replacement agent has usually lost the exploration the
  original spawn was built on, and the replacement pays for it again.
- A brief longer than a few paragraphs is written to a file and passed as a pointer.
  The prompt itself stays short enough to read: the pointer, the task, and the
  delivery channel. Three properties follow, and the third is not obvious. A brief
  that outlives the turn can be re-sent verbatim when an agent is killed or cut off,
  so the work resumes without being re-authored from memory. A shared common-rules
  file lets one set of verified facts serve every brief in a fleet without being
  retyped into each. And inter-agent messages truncate at roughly four kilobytes,
  silently, so a brief or a report pasted inline costs one round trip to discover the
  truncation and another to resend. Where this contract calls a brief "inline" it means
  self-contained, the opposite of "read `AGENTS.md`", and not that the text must sit in
  the spawn message; a pointer to a self-contained file satisfies it.
- The brief file outlives both the agent that reads it and the session that wrote it,
  or the resume property above is imaginary. Put it where both parties can still read
  it after either one restarts, and pass an absolute path. A brief parked in a
  per-session scratchpad leaves its reader holding a pointer to nothing the moment the
  author is killed, which is the failure the file was supposed to survive.
- Reports come back the same way, and "the same way" is specific: the agent writes the
  report to a file and replies with the absolute path and a one-line summary, and that
  reply is the delivery. Writing the file and saying nothing is not delivery, as the
  first rule in this section already says.

[`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md) §4 has the measured cost of
re-derivation, and the kills and cut-offs that file-based briefs were resumed from.

## Style Gate

`chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` invoke `chelis fmt --check` and the full
blocking `chelis lint` rule set on the input file before the
front-end pipeline runs. Style failures block the build by default and
emit a one-issue-per-line diagnostic to stderr. Advisory lint warnings
report valid-but-non-preferred source and do not fail `lint --check` or
the built-in gate.

- Authoritative source of truth: `spec/01-nomenclature.md` (the rule
  spec) and `crates/chelis-lint/src/rules/` (the executable
  enforcement). The canonical formatter is `chelis_surf::format` for
  `.ch` and `chelis_deep::printer` for `.dp`.
- Override flag: `--allow-style-violations` bypasses the gate with a
  stderr warning. Use only for emergency local builds. CI must not pass
  it, and it is not a way around a formatter or lint failure during a
  migration: migrate the source instead. The flag bypasses only the
  style gate, not parse, type, effect, validation, evaluation, or
  backend errors.
- Test override env var: `CHELIS_STYLE_GATE_DISABLE=1` disables the
  gate process-wide. Reserved for the integration-test corpus that
  synthesizes ad-hoc Surf to exercise type/effect/linearity behavior;
  do not set it in production CI or in user-facing scripts.

When writing new code or fixtures, run `chelis fmt --inplace <file>`
and `chelis lint --check` before pushing. The gate replaces the older
manual checklist of "remember to run fmt". Run `python3 scripts/gate.py --fast`
before pushing a non-documentation change, use focused nextest commands during
development, and leave routine workspace execution to hosted CI.

## Surf Style Guide

The authority is `spec/02-surf-syntax.md` §0.1
(canonical forms and the bidirectional contract); §P10-P12 define the
wider set of input spellings the parser still *accepts* but the
formatter rewrites.

When writing or rewriting Surf in this repository:

- prefer `def ... -> T = ...` over `def ... : T = ...` (enforced by the
  `surf-def-arrow-form` lint rule, §3.5)
- put types on function parameters, not on load-style top-level bindings
- use symbolic dimensions such as `batch` and `seq` for runtime-varying axes
- keep fixed architecture dimensions concrete
- do not annotate intermediate expressions when inference already determines the type
- keep meaningful intermediates like `h1`, `logits`, `probs`, and `loss`
- combine short tensor operations when the composed expression is clearer than over-decomposed single-op bindings
- pipe stages use first-argument insertion: `x |> f(y)` means
  `f(x, y)`. Use `x |> fn (v) -> f(y, v)` when the piped value belongs
  in a later argument position.
- treat decompiler-generated verbose load chains and checker-inserted ascriptions as debug output, not example style
- user code typically writes neither explicit `&` nor explicit `copy()` for fan-out
  into read-only primitives; auto-borrow handles it. Write `&x` when an exported
  API or dense signature benefits from clarity. Write `copy(x)` only when forking
  ownership for downstream consumption.
- existing fixtures and migration baselines may keep explicit `copy()`
  or `drop()` calls when they prove compatibility or preserve baseline
  evidence. `redundant-linearity-call` is advisory and should not be
  papered over with blocking-rule exceptions.
- lowered IR now carries compiler-inserted `Copy` and `Drop` nodes for implicit
  linearity. If auto-copy/auto-drop produces unexpected IR, treat it as a
  structural blocker and escalate against `spec/design/implicit_linearity.md`
  rather than papering over it as a routine fixture bug.
- type identifiers are PascalCase (`surf-type-pascal-case`, §3.1)
- function/value identifiers are snake_case (`surf-value-snake-case`, §3.2)
- functions carrying the `Test` effect are named `test_*` or `example_*`
  (`surf-test-name-prefix`, §10.1)
- **A nullary definition needs `()`: `def name() -> T`, not `def name ->
  T`.** This is the highest-frequency breakage — it turned every fixture
  in chelis#1176 into a hard parse error (`expected function parameter
  list `()`, found Arrow`) on rebase. Expect it on any branch predating
  #1031.
- Applying a returned value needs explicit grouping: `(f(x))(y)`.
  Ungrouped `f(x)(y)` is rejected — Chelis has flat multi-argument
  application and no implicit currying. Juxtaposition stays rejected.
- Zero-arity decoration is dropped in *types*, kept in *expressions*:
  `Ctor()` is a Deep `app`, bare `Ctor` is a Deep `var`, and a zero-field
  record keeps `{}` to distinguish `record`/`pat-record`. `! {}` is a
  declared-pure upper bound, semantically distinct from an omitted
  effect clause.
- The unit value is `()`, the unit type is `unit`, and a singleton tuple
  is `(x,)` — that comma is semantic, not cosmetic.
- Effects carry exact casing: `Diff`, `Random`, `Accum`, `IO`, `Test`,
  `Resource(...)`.
- Non-primary transform arguments are named: `grad(f, wrt=x)`,
  `vmap(f, axis=n)`; axis zero is bare `vmap(f)`.
- Canonical output omits trailing separators and prints the canonical
  literal spelling (shortest round-trippable float, no digit separators
  or redundant zeroes).

## Chelis-Specific Rules

### Deep AST

- Every Deep node is a 3-tuple: `(tag {} children...)`
- Metadata map is always present at element 1
- 62-tag closed vocabulary; see `spec/03-deep-syntax.md`
- Function application is `app`, names are `var`, literals are `lit`
- RISC primitives are built-in functions, not tags
- **Decompilation routes through a typed Deep-to-Surf resugaring
  boundary** (chelis#1031) with a total disposition for every public
  Deep tag, printed by the shared Surf printer. It fails closed on
  malformed Deep, invalid surface identifiers, unknown `surf_*`
  metadata, incompatible literal metadata, and non-finite constructed
  values — a resugaring failure is a real defect, not output to work
  around. Canonical Deep and canonical Surf are two representations of
  one public language, bound by the three executable laws in
  `spec/02-surf-syntax.md` §0.1 (`desugar(resugar(·))`, formatter
  idempotence, and the semantic retraction). If you change either
  printer or the desugarer, those laws are the oracle.

### Type System

- No implicit precision promotion
- Named tensor dimensions match by name
- No implicit broadcasting; explicit `expand` only
- Integer literals default to `int32`, float literals to `f32`

### Build Reality

- `chelis build` emits C, header, and runtime artifacts plus compile flags (default target)
- `chelis build --target hip` emits C/HIP host code with embedded GPU kernel strings
- Neither target invokes the native compiler — the user runs `gcc`/`hipcc` manually
- On this workstation specifically, the local HIP toolchain was
  reconciled 2026-05-01 and HIP manual gates are runnable. See
  [`docs/local_hip_environment.md`](docs/local_hip_environment.md) for the authoritative
  runbook.

## Toolchain And Packaging Orchestration

Chelis has a rustup-style install/version layer and a one-command project
orchestrator. The authoritative design is
[`spec/design/chelis_packaging_and_install.md`](spec/design/chelis_packaging_and_install.md);
the user-facing guide is [`docs/book/src/install.md`](docs/book/src/install.md)
and [`docs/book/src/reef.md`](docs/book/src/reef.md).

- **`chelisup`** (`crates/chelisup`) is the toolchain installer and the
  pin-resolving `chelis` **shim**. It owns the toolchain lifecycle: bootstrap
  (`crates/chelisup/bootstrap/chelisup.sh`, the one permitted shell script),
  `install` / `default` / `list-installed` / `which` / `show` / `uninstall` /
  `self uninstall`. The store is `$CHELIS_HOME` (default `~/.chelis`):
  `toolchains/<ver>`, `bin/{chelis,chelisup}`, `reef/`, `src/`.
- **Nix `chelisup` closure:** the Nix package uses `bin/chelisup` as a wrapper
  around `libexec/chelisup`. Before `install`, the wrapper creates the staging
  root `$CHELIS_HOME/nix-gcroots/chelisup.next`. After success, it promotes
  `$CHELIS_HOME/nix-gcroots/chelisup`. A failed install preserves the prior root.
  If the install copied a new binary, the wrapper promotes
  `$CHELIS_HOME/nix-gcroots/chelisup.partial`. A new attempt recovers a stale
  staging root before it changes that root. After success, the Nix wrapper
  restores itself at `$CHELIS_HOME/bin/chelisup`. The generic Rust installer
  contains no Nix root path or cleanup logic. The installed Nix wrapper removes
  all three roots after the real `self uninstall` command succeeds. The
  `chelisupLauncherLint` flake check runs `bash -n` plus `shellcheck` over the
  generated launcher, but since chelis#1450 it fires only on `workflow_dispatch` or a
  published release, so it gates nothing on a pull request or a push to `main`.
- **Shim resolution order** (first match wins): `+<ver>` arg → `CHELIS_TOOLCHAIN`
  → nearest `chelis-toolchain` file → nearest `reef.toml` `compiler =` pin →
  recorded default. A resolved-but-not-installed version is a loud error naming
  `chelisup install <ver>`; the shim never auto-installs on `cd` and never
  silently falls back. `+latest` is not a supported reference (concrete pins
  only).
- **`chelis reef setup [--path]`** (`cmd_reef_setup` in `crates/chelis-cli`) is
  the orchestrator: ensure the pinned toolchain (auto-install via the chelisup
  binary), `reef install --from-lockfile`, `reef src sync` when `[chelis-src]`
  is present, then a `reef doctor` summary. `reef doctor` is its read-only
  counterpart across all classes (toolchain, source crates, binary artifacts).
- **Shim-corruption trap (do not regress):** `reef setup`'s toolchain step MUST
  subprocess the real `chelisup` binary. Never call `chelisup::install::install`
  in-process from `chelis-cli`: that helper copies `current_exe()` into
  `<home>/bin/{chelis,chelisup}`, which from the `chelis` binary overwrites the
  shim with the compiler. **This is enforced at compile time:** chelisup's
  `install` and `ensure_shim_installed` are `pub(crate)`, so a call from
  `chelis-cli` is an `E0603` build error caught by the normal
  clippy/build/test stages. Keep that visibility (and the call-site comment);
  do not widen it to `pub`.
- **§5.4 invariant:** `setup`'s auto-install is *explicit* provisioning and is
  therefore exempt from the "no auto-install" rule, which governs only the
  *implicit* shim. The unknown-subcommand hint augments only clap's
  `InvalidSubcommand`.

Use the `packaging-install` skill when changing or validating any of this.

## Shared Local Skills

Project-local skills live in `agent-skills/`.
`.claude/skills` and `.codex/skills` should resolve to that same directory so both tool
surfaces load the same skill library.
Command wrappers should stay mirrored too: `.claude/commands/` and `.codex/commands/`
should stay byte-identical so slash-command access does not drift between tool
surfaces. Keep a `red-team` alias wired to `redteam-exec` with two entry modes. A fresh
round retires your own stale or failed current-session handles, spawns a fresh local
subagent, and hands it the inline brief; `verify` sends the fix to the standing reviewer.
Fresh context is an agent property: the wrapper reuses a clean, exact-head, idle worktree
and its warm target artifacts instead of forcing a cold checkout and rebuild.

Current shared skill set:

- `redteam-exec`
- `spec-sync`
- `phase-gate`
- `backend-numerics`
- `example-corpus`
- `cli-surface`
- `packaging-install`
- `issue-resolution`

## Downstream Shell Contract

Downstream shell repos (every repo in the canonical-reference §Shell
Ecosystem table) inherit this `AGENTS.md` through a **stamped pointer
managed block** (not a verbatim copy; machine-local environment sections
excepted — see the contract §1) AND must satisfy
[`spec/design/shell_repo_contract.md`](spec/design/shell_repo_contract.md):
pin hygiene with a mechanical multi-location consistency guard, a
per-shell `docs/CHELIS_SURFACE.md` capability inventory, the
narrowing-citation rule (every workaround cites `chelis#NNN` at the site —
file upstream, never silently work around), expected-to-fail blocker
probes under `tests_blocked/` re-run at every pin bump, negative-test
sidecars, the shared skill set materialized as an upstream pointer, and
uv-managed Python.

The contract ships **in the toolchain** as `chelis reef conform`
(audit / init / sync / bump / bump-check), backed by the
`chelis-conformance` crate — the machine-readable form of the contract's
§11 table (`MANIFEST`) and the §Shell-Ecosystem registry (`REGISTRY`),
tripwire-locked to this doc and `.github/workflows/ecosystem-drift.yml`.
Shells wire `conform audit` + `conform bump-check` into CI; pin bumps land
as `conform bump` PRs (never direct to `main`). `Chelis-Lang/school` is the
reference implementation. Changes to the contract land here first (edit the
doc **and** `MANIFEST`/`REGISTRY` in lockstep, or the tripwire fails) and
propagate to every shell via `conform sync` per the scaffolding drift rule.

### Conform Bump Wave Checklist

`chelis +<new-version> reef conform bump <new-version>` is a mechanical starter,
not a green-PR oracle. Run it from a fresh shell worktree with the new toolchain
selected explicitly, then audit every item below:

- Inspect both its exit status and `git status`. Pre-conformance repos can fail or exit
  zero after a partial edit and materialize orphaned `.claude/`, `.codex/`, or
  `agent-skills/` content; remove that half-retrofit and defer full adoption to a
  separate `conform init` change.
- Cascade every dependency surface manually: sibling-shell versions in `reef.toml`,
  sibling tag/version variables in workflows, `[chelis-src]`'s exact
  `CHELIS_PIN_COMMIT`, and every non-frozen nested project `reef.toml`. The standard pin
  guard does not prove tag-to-SHA agreement or cover sibling and nested pins.
- Audit the shell's own package version, CHANGELOG convention, and hard-coded version
  strings in both CI and release workflows before tagging. A workflow at the tagged
  commit cannot be repaired by merely rerunning the failed release.
- Treat `reef.lock` by entry authority. Regenerate `bundled` toolchain entries. Never
  commit `local_registry` entries without `remote_origin` or hashes produced by a
  private local registry. Keep published dependency entries at the last published
  release during a cascade block, and ensure `reef build` precedes any `chelis test` or
  `chelis prove` step that reads the committed lock.
- Re-run `conform audit` and repair every live `docs/UPSTREAM_BUGS.md` entry to carry its
  own `chelis#NNN` or `docs/issue_drafts/<file>` citation. Nested detail bullets must not
  accidentally parse as uncited independent entries.
- Verify a claimed acceptance command by opening the workflow and locating the exact
  step. If the authoritative campaign is expensive, compare a bounded pilot at the old
  and new pins on identical sources; only the delta supports a "no new regressions"
  claim.
- When a skipped version range crosses canonical Surf v0.19, run
  `chelis migrate surf --from 0.18 --inplace` over maintained `.ch` sources, then handle
  semantic migrations the tool cannot choose: explicit literal suffixes, int64 extents,
  and checked `cast` versus truncating `cast_trunc`.
- `chelis test` deliberately does not run the style gate. A shell that generates Chelis
  source must run `chelis check` or `chelis fmt --check` over emitted files; do not hide
  formatter drift behind a measured threshold in the generator.
- Finish with `reef build`, the shell's real CI-equivalent gates, and exact review of the
  generated diff. A successful `conform bump` alone is never completion evidence.
