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

### Phase Completion Criteria

- Do not claim a phase is done based on crate-local green or narrative progress.
- A phase is done when the executable acceptance surface is green, docs are honest, and
  the red team can only find minor residual issues.
- Budget for at least one adversarial validation pass that runs code, not just source review.

### One Acceptance Oracle Per Phase

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
- For this repository, a requested "red team agent" means a fresh local subagent in a
  new context. A main-thread validation pass does **not** satisfy that request.

### Required Red-Team Behaviors

- Write and run adversarial tests when coverage is missing.
- Verify that inputs which should fail do fail, and with the right reason.
- Verify that inputs which should pass do pass, with exact outputs where applicable.
- Check docs and phase claims against the shipped behavior, not just intent.
- Record the exact reviewed commit and the worktree's baseline status. A fresh reviewer
  may reuse an existing exact-head worktree and its target build artifacts so the pass
  does not require a cold rebuild; fresh context does not require a fresh checkout.

### Fresh-Context Enforcement

- When asked to run a red team or "spawn a red team agent", first inventory subagent
  handles created in your own current session. Stop or interrupt and retire only stale
  or failed handles that will not be used again; do not disturb another developer's
  handles or a handle reserved for follow-up work. Then spawn a new local subagent with
  fresh context. A platform need not support deleting a retired handle from its listing.
- Freshness applies to the subagent's review context, not to the filesystem. Hand the
  fresh subagent an existing worktree and its warm target cache when the worktree is at
  the exact review head, has a known clean baseline, and has no concurrent writer or
  build owner. Create a new worktree or target only when those reuse conditions do not
  hold.
- If the first spawn attempt routes to remote infrastructure, errors, or comes back in a
  broken state, stop or interrupt and retire that handle, then retry until you have
  either:
  1. a working fresh local subagent, or
  2. an explicit statement that red-team validation is blocked because fresh local
     subagent execution is unavailable.
- Do not substitute main-thread validation and call it a red team.
- Do not mark a phase as red-teamed unless the fresh-context subagent actually ran the
  validation work.
- A subagent that reuses a worktree must restore its temporary probes or mutations and
  report the final worktree status, unless the task explicitly asks to retain them.

### Pull Request Review Gate

- Every pull request creation workflow, including documentation-only work, must include
  at least one compliant red-team review of the exact PR head before merge.
- Classify every finding against the pull request's stated scope. A finding is in scope
  only when the pull request introduces it, worsens it, or claims to correct it. Mere
  discovery during review, including a pre-existing spec/implementation mismatch in an
  unrelated surface, does not bring a finding into scope.
- For documentation and design reviews, assign severity by the contract impact rather
  than by the mere presence of an inaccurate sentence. A wrong normative rule, or a
  plan that cannot close a named in-scope instance or acceptance requirement, is P1. A
  design document that misdescribes current `main`, or exposes a sequencing seam between
  delivery slices while leaving the normative contract and named deliverable achievable,
  is P2 and must be recorded as residual work rather than promoted to a merge blocker.
- Only a confirmed in-scope P0 (critical) or P1 (high/major) finding blocks merge. Fix
  that finding in the pull request, push the updated head, and run another fresh-context
  red-team review against that exact head.
- Repeat the fix-and-review cycle until the most recent exact-head red-team review
  reports no in-scope P0 or P1 findings. A review of an earlier head does not satisfy
  this gate. An out-of-scope P0 or P1 finding does not require a change to the pull
  request or another red-team review.
- Repairs under this gate may correct, remove, or narrow the pull request's existing
  content. They must not add new design scope, implementation responsibilities,
  inventories, mechanisms, or promises merely to absorb a finding. When a correction
  would require that expansion, reduce the claim and track the additional work outside
  the pull request.
- A finding class is the underlying defect category or unmet obligation, not its file,
  line, or wording instance. If consecutive review rounds replace a repaired finding
  with a different class of finding, stop the repair loop and correct or narrow the
  review brief before running another round. Do not keep expanding the pull request to
  satisfy a moving brief.
- Route confirmed in-scope findings back to the original implementation agent when it
  is still available so the fix retains its build context. Do not fix out-of-scope
  findings in the pull request; link an existing issue or file one if the defect is not
  already tracked. Record every round's exact head, verdict, commands, finding-scope
  classifications, accepted-no-action observations, linked issues, and residual scope
  in the pull request.

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

**Why this matters, with a measured instance.** A design doc is a working artifact: it
is read constantly while its phases are in flight and stops being read the moment they
ship. A decision parked in one does not survive the work that made it. In 2026-07 that
produced a three-level drift. `spec/04-type-system.md` [04-NUM-2] PERMITTED one narrow
thing ("computing a single op in f64 and rounding once is a conforming
implementation"); `spec/design/dtype_semantics.md` cited that permission to MANDATE
f64 computation for every float op; the evaluator then extended the mandate to
multi-step reductions and to comparison operands. Each step was a reasonable reading
of the one above it, nobody re-checked against the numbered spec, and the result was a
language that computed f32 programs in f64.

Two rules follow, and both are cheap:

1. **When a design doc states a rule that is really a language decision, lift it into
   the numbered spec and leave a pointer behind.** `spec/05-risc-primitives.md` §8's
   [05-OBS-1..5] is the worked example: the observation contract moved out of
   `faithful_observation.md` and now survives independently of it. `spec/04` §9's
   [04-NUM-9..11] and §9.1 followed, for the trap contract, the exactness guarantee,
   and the per-dtype value table.
2. **Watch for permission-to-mandate escalation.** "X is a conforming implementation"
   in a spec does not license "therefore we do X" in a design doc, and neither
   licenses "therefore we do X everywhere" in code. If your implementation needs a
   stronger rule than the spec states, amend the spec first and say so in the PR;
   `spec/design/dtype_semantics.md` §B1 calls that "the protocol, not a failure."

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
  field, exported C signature, exported C data declaration, or binding parameter that
  carries numeric values as
  bare `f64`/`double`, or that takes a raw integer dtype id, is a review-blocking
  finding. Zero-exception classification is the landing rule: every discovered row
  must end in exactly one final authority class - structurally nonnumeric, a
  structurally recognized exact tagged carrier/transport, or an exact numeric
  operation registration. The immutable foundation-era universe retains the old 39
  grandfather rows, three successor overrides, and 155 permanent plain rows as
  deletion debt owned by chelis#1288. The active primary baseline has already
  moved 18 exact structurally nonnumeric rows, the eight tagged-carrier
  declarations, and 123 registered numeric operations (the [05-OP-35]
  stdlib surface registrations among them) to final authority; it
  therefore retains no pre-ratchet grandfather rows, no successors, and 63
  permanent plain rows as active debt, and the obsolete prelude `Json` row
  is deleted. The
  84 typed-wire and 17 registered-PyO3 baseline rows remain sealed legacy cohorts,
  while Count's wire field is final-registered. Those lists confer no authorization
  for a new, renamed, reclassified, or otherwise changed row, and a change touching one
  must move it to a final authority class rather than copy its disposition. A bare
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
  turn a callable into plumbing. The executable census still contains three exact
  integer-plumbing exceptions as chelis#1288 deletion debt; they cannot be copied,
  widened, renamed, or used to authorize any changed declaration. The final rule
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
OpenSpec planning is optional. OpenSpec validation does not gate merges.
`spec/design/spec_provenance.md` describes the future governance regime.
This regime is not active.

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
**sub-issue parent** for every instance. Its body carries the plan (phases,
oracles, freeze points, the class statement); its evidence lives in the owning
design doc under `spec/design/` or in `docs/investigations/`.

**Do not create a separate META issue alongside it.** The existing META/tracker
pairs ([#727]/[#729], [#703]/[#730], and siblings) are historical, not a pattern
to copy: the METAs were filed during the 2026-07 numeric audit as evidence
records, and the trackers were filed later, when the design docs were written,
as delivery contracts. Three reasons the split has stopped paying for itself:

1. **It has already broken down.** Three of the five "METAs" are closed. #709 and
   #710 closed while their class continues under an open #731, and neither was
   written as a META - both are instance reports the class map promoted after the
   fact. #728 closed alongside its own tracker #732, so the pair carried no
   information the tracker did not.
2. **Sub-issues do the job the pairing was improvising.** When the tracker is
   the parent, "what belongs to this class" is a structural fact. A second issue
   whose content is a list of instances duplicates the child list and drifts
   from it.
3. **Two bodies means two things to keep honest**, and the evidence half has a
   better home: `docs/investigations/` already holds the probe corpus and the
   audit record.

Rules:

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

## Scripting Language Policy

- **Python** for all scripts, utilities, report generators, and automation helpers.
  Write tests for them.
- **Rust** where the task naturally fits a compiled workspace member.
- **Never shell.** Do not write `.sh` scripts. If a CI step needs a one-liner, invoke
  Python instead. Shell is fragile and untestable.
- **Shell exceptions.** Shell is permitted only for these artifacts:
  - The published `chelisup.sh` bootstrap runs before Chelis, Cargo, or Python exists.
  - The generated Nix `chelisup` launcher runs with only its package closure.
  - The cargo-husky `commit-msg` hook locates a repository-managed Python interpreter.
  Each artifact MUST use minimal POSIX `sh` and MUST pass `shellcheck` when
  available. The bootstrap and the hook are checked with `sh -n`; the generated Nix
  launcher is checked with `bash -n` by the `chelisupLauncherLint` flake check, so a
  bash-ism there passes CI while still violating the POSIX-`sh` requirement above.
  All other scripts remain Python.
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
  before any history rewrite is published. Run `python3 scripts/gate.py --local` for a
  non-documentation change or the focused documentation checks for a docs-only change.
  Never force-push a red gate. Obtain approval, then use an exact-head
  `--force-with-lease`; clean mechanical rebases do not need the additional resolution
  review.

## Build Toolchain

A uv-managed Python is a hard prerequisite on every platform: `chelis-python` links
against `libpython`, and the gate scripts and several tests need an interpreter. Install
uv from <https://docs.astral.sh/uv/getting-started/installation/>, verify it with
`uv --version`, and provision the version with `uv python install 3.11`.
`py/pyproject.toml` pins `requires-python = ">=3.11"`. See [`README.md`](README.md) for
the full setup.

**Provisioning.** Create a checkout's environment once with `uv venv --python 3.11`, at
the root of the primary checkout and at the root of every dedicated worktree. Never copy
or symlink another checkout's `.venv`. Inside Devenv, the activated environment at
`.devenv/state/venv` serves the same purpose and needs no separate step.

**Resolution.** Rust test and gate code resolves an interpreter through
`tests/support/managed_python.rs`: an explicit `PYO3_PYTHON` wins outright, and the
checkout's `.venv/bin/python` is the fallback. Outside Devenv, `.cargo/config.toml`
points `PYO3_PYTHON` at `.venv/bin/python`; Devenv overrides it with
`.devenv/state/venv/bin/python`. For direct Cargo commands in a dedicated worktree,
export `PYO3_PYTHON="$(uv python find 3.11)"`. An explicit `PYO3_PYTHON` is
authoritative: an invalid path must fail loudly rather than fall back.

**Invoking Python directly.** Outside Devenv use `.venv/bin/python` or
`uv run --managed-python --python 3.11 --no-project python`; inside Devenv use the
activated environment. `scripts/gate.py` is the one `python3` entry point that
self-heals, re-executing through that uv command before running gate logic, so a bare
`python3 scripts/gate.py` is always safe; it also sets `PYO3_PYTHON` to its uv-selected
interpreter for every child command.

**macOS:** Apple's bundled Python reports a stale `sysconfig.LIBDIR` path. Do not route
PyO3 to it.

## Local Git Hook

Devenv installs the `no-ai-authorship` hook at the `commit-msg` stage. The hook
runs `scripts/check_commit_message.py`.

Cargo-husky remains the fallback for the manual setup. `cargo test` installs its
POSIX wrapper, which runs the same Python checker through Devenv, `.venv`, or
`uv run --managed-python --python 3.11 --no-project` in that precedence order.
All formatting and lint hooks remain disabled. CI remains the remote
enforcement boundary.

## Commit And Pull Request Hygiene

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

Agent pre-push gate for non-documentation changes:

```sh
python3 scripts/gate.py --local
```

`scripts/gate.py` is the single source of truth for the per-PR
gate. Agents run only the `--local` subset before pushing; the bare full gate is
CI-owned for routine PR validation. CI calls `python3 scripts/gate.py <stage>`
for each split job. `scripts/test_gate.py` pins the complete ordered set of
single-line `run:` commands permitted in those gate-owned jobs, so shell syntax
cannot hide an unreviewed command. To see the canonical full list and the
local/CI ownership annotations:

Before `--local` or another long local validation, fetch `origin/main` so the
changed-crate selection and inherited-failure comparison use current evidence. If the
branch is materially behind, reconcile it deliberately before spending hours on a
stale tree; do not rewrite shared history without the rebase review gate above. When an
unrelated failure appears, reproduce or compare it on current `origin/main` before
diagnosing it as branch-owned.

```sh
python3 scripts/gate.py --list
# cargo clippy --workspace --all-targets -- -D warnings  # local + ci
# cargo fmt --all -- --check  # local + ci
# cargo run -p chelis-cli --bin chelis --quiet -- lint --check .  # local + ci
# <managed-python> scripts/regenerate_chelis_std_bundle.py --debug --check  # local + ci
# cargo test -p chelis-types --doc  # local + ci
# cargo test -p chelis-compiler-api --doc  # local + ci
# cargo test -p chelis-pipeline-core --doc  # local + ci
# <managed-python> scripts/check_checkpoint_compile_fail.py  # local + ci
# <managed-python> scripts/pipeline_core_dependency_guard.py  # local + ci
# <managed-python> scripts/pipeline_core_documentation_guard.py  # local + ci
# <managed-python> scripts/check_pipeline_core_compile_fail.py  # ci-owned
# cargo nextest run --workspace --no-fail-fast  # full gate; CI coverage split
# <managed-python> scripts/compiler_front_end_performance.py  # ci-owned
# <managed-python> scripts/unrepresentable_domain_oracle.py  # local + ci
# # --local also runs: cargo nextest run -p <crate> --no-fail-fast for each crate changed vs origin/main
```

`python3` is only the gate bootstrap. An unmanaged invocation re-executes via
`uv run --managed-python --python 3.11 --no-project`; an active Devenv or
uv-created environment is preserved. If uv is missing, the gate exits with
installation and Python-provisioning commands.

The gate runs `cargo nextest run --no-fail-fast` (CI's actual runner), not
`cargo test --workspace`, and includes `chelis lint --check .` (the §8.6 /
§12 naming gate). All nextest profiles also set `fail-fast = false`, so direct
and non-gate CI nextest runs expose every failure instead of cancelling the
remainder. The sanitizer, macOS-smoke, LOC-report, no-AI-authorship, docs, and
smt-build CI jobs are out of scope for this script by design.

The three explicit rustdoc stages exist because `cargo nextest` does not
execute doctests, and the gate deliberately does not use `--workspace --doc`.
`scripts/gate.py`'s module docstring records why, and what each stage covers.

**Doctests only run where something invokes them.** The canonical gate
invokes doctests for `chelis-types`, `chelis-compiler-api`, and
`chelis-pipeline-core`. The `backend-sanitizers` job also runs
`cargo test -p chelis-backend-c` without a filter. A `compile_fail` oracle in another crate runs nowhere until that
crate gains an equivalent invocation in the same change set.

`scripts/unrepresentable_domain_oracle.py` is chelis#908's authoritative
completion oracle, and the tracker requires every fix in that class to run
it in a continuous job. Acceptance is exit 0 with a final `ORACLE: PASS`
line. Every obligation drives compiled artifacts: the built `chelis`
binary over `.dp` fixtures, and compiled test binaries through `cargo
nextest`. Its Python unit tests patch the command runners, so they are
evidence about the script's decision logic and never a substitute for
running it. It runs in the `integration` stage, which hosted CI executes
exactly once on the second `workspace-tests-shard` worker before the
fail-closed `Workspace Tests (Linux)` aggregate, on every non-docs-only pull
request, and in the `--local` pre-push subset. `scripts/gate.py`'s docstring
records why that stage rather than `lint-and-unit`, and how the
`CHELIS_ORACLE_BINARY` build handoff avoids a redundant rebuild.

`--local` (chelis#360) runs the developer pre-push subset: workspace clippy
(`-D warnings`, compile-only), `cargo fmt --check`, `chelis lint
--check .`, the deterministic std-bundle regeneration check, all three
explicit rustdoc commands, the checkpoint fixture, both pipeline-core
guards, the chelis#908 unrepresentable-domain oracle, and
`cargo nextest run -p <crate> --no-fail-fast` for each
crate changed vs `origin/main` (committed diff plus uncommitted work;
owning packages are resolved from each member's `Cargo.toml`, not the
directory name). The derived crate list is always printed; "no crate
changes detected" means the per-crate stage was skipped, not silently
empty. The workspace nextest stage is CI-owned: run `--local` before
pushing, open a draft PR early, and let CI (macOS Smoke is the
authoritative workspace oracle) run the full suite. See
[`docs/local_macos_environment.md`](docs/local_macos_environment.md)
for why the workspace suite does not belong in the local loop on
macOS.

Every command's combined stdout and stderr streams live. On failure the gate
retains the complete transcript under `target/gate-failures/`, replays the
final 200 lines, and prints the stage/index, duration, exit code or signal,
relevant environment, exact rerun command, and transcript path. Successful
command transcripts are removed.

The gate normalizes `CARGO_TARGET_DIR` to an absolute path inside the current
worktree and rejects paths outside it. It also sets
`CARGO_HUSKY_DONT_INSTALL_HOOKS=1` for child builds, preventing cargo-husky
from mutating the clone's shared `.git/hooks` while sibling worktrees run.
These controls isolate writable state; concurrent agents may still contend
for CPU and make each other slower.

Prose-only changes with no code, fixture, example, or structurally consumed Markdown
are exempt from `--local`: run the focused documentation checks, push, and require green
CI. Hosted docs-only classification is routing evidence, not proof that every changed
Markdown control artifact has an owning validator in that workflow.

Markdown parsed, embedded, mirrored, or used as agent instructions is a control artifact,
not inert prose. Run its focused validators even when CI reports `docs_only=true`. For a
shared `agent-skills/*/SKILL.md` or red-team command-wrapper change, at minimum:

- run the platform's skill-schema validator against every changed `SKILL.md` (in Codex,
  use the `skill-creator` `quick_validate.py` helper),
- run `scripts/regenerate_conformance_assets.py --check` through the uv-managed Python,
- compare the live and embedded skill bytes and the Claude/Codex wrapper bytes, and
- run the `chelis-conformance` `asset_drift_tripwire` and `skill_set_uniformity` tests
  with the worktree's managed Python environment.

The always-run Docs job builds mdBook and validates the package skill/examples; it does
not replace these shared-agent-skill checks.

Default-gate discipline:

- Use focused `cargo nextest run -p <crate> --test <file>` commands for the inner
  development loop. Do not substitute a workspace-wide `cargo test` run for the
  canonical gate.
- For non-documentation changes, `python3 scripts/gate.py --local` is the pre-push
  checkpoint; routine workspace execution is hosted-CI-owned.
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
  starvation, not code breakage. Measured 2026-06-10: the 25-test
  `rank_poly_tier3` suite took 2,434s under contention vs 24s on a quiet
  machine. Re-run on a quiet machine before treating those as real failures.
- At session end, verify that task-owned background cargo, rustc, and nextest processes
  are gone. A stopped wrapper is not proof that its reparented children stopped; use
  the scoped `reap_orphans.py` dry run and kill only confirmed task-owned stragglers.
- macOS workstation only: first-exec assessment can degrade under mass
  fresh-binary bursts and stall multi-binary test runs at ~0 CPU (chelis#356).
  Probe with `python3 scripts/preflight_exec_probe.py` (exit 1 wedged, exit 3
  slow). Inside Devenv, use `chelis-exec-preflight`. Run the probe before the
  local workspace nextest stage. If the probe reports degradation, use the
  macOS Smoke CI stage per
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

## Subagent Coordination And Delivery

- Every subagent prompt must name the delivery mechanism and the complete expected
  report. A locally written or plain-text report that is not sent through the platform's
  parent-message/final-report channel has not been delivered.
- A subagent must not end its turn merely to wait for a background build, monitor, CI,
  or notification that cannot wake it. Keep ownership of a long command through the
  platform's synchronous wait/poll mechanism, or return the honest partial result and
  unfinished work.
- If an agent returns "waiting" or goes idle without the deliverable, the orchestrator
  resumes it immediately with the exact missing items. Prefer a clearly labelled partial
  report over silence or an overstated completion claim, and deduplicate repeated reports
  that race with a resume nudge.

## Local HIP Environment

This workstation has a reconciled AMD/ROCm HIP setup, so HIP manual gates are locally
runnable. [`docs/local_hip_environment.md`](docs/local_hip_environment.md) is the
authoritative runbook and holds the environment detail: the wheel ROCm stack, the
`environment.d/hip.conf` overrides, and how to probe the local GPU.

**Run every HIP manual gate through** `scripts/hip_test.py` — for example
`scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1`
— or `chelis-hip-test` inside Devenv. Plain `cargo test --ignored` inherits an
incomplete environment and segfaults hipBLAS-linked binaries at process exit with empty
output. That failure looks like a code regression and is not one.

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
  stderr warning. Use only for emergency local builds and one-off
  migrations. CI must not pass it. The flag bypasses only the style
  gate, not parse, type, effect, validation, evaluation, or backend
  errors.
- Test override env var: `CHELIS_STYLE_GATE_DISABLE=1` disables the
  gate process-wide. Reserved for the integration-test corpus that
  synthesizes ad-hoc Surf to exercise type/effect/linearity behavior;
  do not set it in production CI or in user-facing scripts.

When writing new code or fixtures, run `chelis fmt --inplace <file>`
and `chelis lint --check` before pushing. The gate replaces the older
manual checklist of "remember to run fmt". Run `python3 scripts/gate.py --local`
before pushing a non-documentation change, use focused nextest commands during
development, and leave routine workspace execution to hosted CI.

## Surf Style Guide

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

### Canonical Surf v0.19 (chelis#1031, shipped 0.18.4)

**The grammar changed, and the style gate enforces it.** `chelis fmt
--check` runs ahead of `build`, `check`, `validate`, and `eval --file`,
so Surf that was canonical under 0.18.3 can now fail before the
front-end pipeline runs. The authority is `spec/02-surf-syntax.md` §0.1
(canonical forms and the bidirectional contract); §P10-P12 define the
wider set of input spellings the parser still *accepts* but the
formatter rewrites.

Read that section before authoring Surf or debugging a parse error that
"should" work. The forms that most often bite:

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

**Migrating an existing tree:** `chelis migrate surf --from 0.18 --check
<paths>...` reports, `--inplace` rewrites as a preflighted batch
transaction (whole batch validated before any write; atomic replacement
with rollback; symlinks and multiply-hard-linked files rejected).
`--from 0.18` is the only accepted value. Identifiers that became
reserved words are **not** guessed — the migrator stops and names the
byte offset, and the rename is yours to author. Semantic boundaries are
not migrated for you: suffix adoption, overflow, non-finite values,
literal patterns, raw controls, invalid escapes, and structurally
ambiguous legacy forms still reject.

Do not reach for `--allow-style-violations` or
`CHELIS_STYLE_GATE_DISABLE=1` to get past a v0.19 failure. Those exist
for emergency local builds and the ad-hoc integration corpus
respectively; migrate the source instead.

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
  `chelisupLauncherLint` flake check gates the generated launcher with `bash -n`
  plus `shellcheck`.
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
should stay behaviorally aligned so slash-command access does not drift between tool
surfaces. Keep a `red-team` alias wired to `redteam-exec`, and make that wrapper enforce
retirement of your own current-session stale or failed handles that will not be reused,
plus a fresh local subagent before any validation is counted as a red team. Fresh context
is an agent property: the wrapper should reuse a clean, exact-head, idle worktree and its
warm target artifacts when available instead of forcing a cold checkout and rebuild.

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
