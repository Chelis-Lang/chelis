# Chelis Agent Contract

Keep this file concise and relevant to every agent working in this repository.
Each added token is read tens of thousands of times. State a rule once, link the
document that owns the detail, and put the explanation in that document, not here.

`CLAUDE.md` is a symlink to this file so Claude-style and Codex-style entry points do
not drift.

## What Chelis Is

Chelis is a numerical computing language for code that agents write and people
supervise. Tensors carry named dimensions and precision in their type; the compiler
checks shapes, precision, effects, and ownership before anything runs, and `chelis
prove` checks the properties an author states, naming the method behind each result.
The bet is that numerical code an agent can reason about, and a person can review
through its types and properties, beats code whose mistakes first surface at run time.
Chelis is general purpose within numerical computing; the worked examples come from
quantitative finance. Differentiation and machine-learning programs are research
directions, not the definition of the language. It is not a systems language, a web
framework, a deep-learning framework, or a general scripting replacement for Python.
`spec/00-context.md` and `spec/design/chelis_canonical_reference.md` own the full
statement; their specifics may lag, their intent does not. When a tradeoff
appears, apply these in order:

1. **Unambiguity over ergonomics.** The author is an agent. The friction a human feels
   spelling out every type, effect, dtype, and dimension is not worth a reading the
   compiler has to guess at.
2. **Composition over special cases.** A new capability composes existing primitives
   before it earns a new one.
3. **Inference over annotation.** Where the checker determines something uniquely, the
   author does not repeat it; intermediates carry no ascription.
4. **Machine generation first.** A convenience that exists only for a human typist is
   not a reason to add syntax, a default, or a fallback.
5. **Additive sugar only.** Every surface form desugars to the core; nothing in the
   surface has semantics the core lacks.
6. **Explicit over implicit.** No implicit broadcasting (`expand` only), no implicit
   precision promotion, no implicit currying or partial application, no silent
   narrowing at ingress, no hidden effects. Where intent cannot be determined uniquely,
   the compiler rejects.
7. **Small language, big library.** The compiler knows only the closed RISC primitive
   set and its derived built-ins; everything else is a library. The canonical reference
   §8.5 has the core/standard-library/external-library taxonomy.
8. **Future-proof without over-building.** Decide the rule fully now, implement what
   the phase needs, and never narrow a rule to what a lane implements today.

Two corollaries govern how the compiler itself is changed. Chelis is pre-compatibility
unless a controlling contract says otherwise, so prefer the structural design that
makes a defect class impossible over a smaller-blast-radius patch, a legacy default, a
versionless compatibility fallback, or phase deferral; close the class, not the
instance. And determinism is part of the contract: for fixed program text, compiler
build, target, and declared inputs, every check, evaluation, and build result is a
function of those inputs, and feedback that varies between identical runs is a defect.

## Quality Standards

### Spec-First Development

- Before writing implementation, write test stubs derived from the owning spec.
- Every spec requirement should have a corresponding test before the code exists.
- If the spec says "X is a type error," write the failing test before implementing
  the checker.

### Negative Test Parity

- For every test that checks something works, add the corresponding failure test.
- If you cannot name the failure case, the spec understanding is still weak.

### Do Not Trust Green

- Passing tests prove alignment with the tests, not necessarily with the spec.
- After green CI, check what active requirements still lack tests.
- Audit silent fallbacks, default values, empty error vectors, and `unwrap_or` paths.

## Review And Merge

### Red Team Rounds

Run every round through the [`redteam-exec` skill](agent-skills/redteam-exec/SKILL.md).
It carries the brief shape, the worktree-reuse rules, and the verify mode.

- Red team against the spec, the code, the tests, the examples, and the CLI behavior.
  Execute tests and commands; source inspection is not proof.
- Every pull request, documentation-only work included, gets at least one round before
  merge. A round is the whole live back-and-forth between one reviewer and the author,
  not a single review pass:
  1. A fresh local subagent reviews the exact head from an inline brief and reports
     its findings.
  2. The reviewer stays alive. The author repairs the findings in the worktree.
  3. The author hands the repair back, and the same reviewer verifies it and looks for
     similar issues the repair may have missed or introduced.
  4. Any further finding goes back to the author, and steps 2 and 3 repeat.
  5. The round ends only when that reviewer states it is satisfied.
  A reviewer that has reported is not finished; it is waiting for the fix. Ending the
  loop after the first report, or verifying a repair with a different reviewer, is not
  a round.
- A pull request gets at most three fresh rounds; a fourth needs the user's explicit
  approval. A prose-only pull request gets one, and a second needs the same approval.
  The pull request's round record is the counter. Verification by the standing reviewer
  does not count; the end-of-pull-request round does.
- A finding is in scope only when the pull request introduces it, worsens it, or claims
  to correct it. Discovery during review does not bring a pre-existing defect into scope.
- A confirmed in-scope P0 or P1 merits a fresh round after the current round finishes,
  within the cap. Unmigrated assertions, old comments, and minor documentation drift do
  not. A rebase does not by itself merit a round: send a hand-resolved intersection that
  stays within files the standing reviewer already read to that reviewer for focused
  verification, and use a fresh round only when the rebase introduces a new mechanism or
  touches files that reviewer did not read. A targeted review of substantial rebase
  overlap needs no permission and never counts toward the cap.
- For documentation and design reviews, severity follows contract impact. A wrong
  normative rule, or a plan that cannot close a named in-scope deliverable, is P0 or P1.
  A design document that misdescribes current `main`, or exposes a sequencing seam while
  the contract stays achievable, is P2 or P3 and is recorded as residual work, never
  promoted to a merge blocker. Wording, line-level accuracy of the pull request body,
  staleness against a sibling pull request's moving head, and anything whose fix would
  add text without a necessity sentence are out of scope.
- State a pull request's claim at the granularity its oracle proves. An unbounded
  universal claim invites sampling in every round and can never be closed.
- A finding class is the defect category, not its file, line, or wording instance.
  Every round record names the class of each finding. When two consecutive rounds
  report the same class, or replace a repaired finding with a different class, stop
  patching witnesses: change the representation, the oracle, the claim, or the brief
  before running another round.
- Repairs may correct, remove, or narrow the pull request's content. They must not add
  design scope, mechanisms, inventories, or promises merely to absorb a finding. When a
  correction would need that, reduce the claim and track the rest outside the pull
  request.
- Freshness is a property of the reviewer's context, not the filesystem. Hand a
  reviewer an existing worktree and its warm target only when it is at the exact review
  head, has a known clean baseline, and has no concurrent writer, and paste the output of
  `.venv/bin/python scripts/worktree_status.py` into the brief as the evidence. Unknown
  or not-clean means wait, and free is the probe's best answer rather than a proof: a
  run that takes no lease, `--fast` among them, is caught only by a scan of the
  processes it spawned. A reviewer whose probes mutate tracked source gets its own
  worktree, and you never edit a worktree a reviewer is reading.
- Before spawning a fresh round, retire only your own stale or failed subagent handles;
  a standing reviewer awaiting a fix is neither. If a spawn routes to remote
  infrastructure, errors, or comes back broken, retire it and retry until you have a
  working fresh local subagent, or state that red-team validation is blocked.

### Pull Request Lifecycle

The mechanics are expensive to get wrong, and
[the PR-author guide](docs/guard_changes_for_pr_authors.md) owns them. Read it before
starting and again before merging, and use the acknowledgement lines it requires.

1. Fetch `origin/main` and base the branch on it. Run `python3 scripts/gate.py --fast`,
   push, and open the pull request before the first red-team round so CI runs in
   parallel. Add a `changelog.d/` fragment. Record the reviewed head and CI evidence in
   the pull request.
2. During a round, consolidate repairs into a local commit and hand that unpushed head to
   the standing reviewer. Push only when the round is closed. If a push or CI rerun is
   already needed, fix every known P2-or-lower finding before it.
3. Do not rebase or merge the base merely because `main` advanced. Fetch, check GitHub's
   mergeability, and inspect the prospective merge with `git merge-tree`. Rebase only
   when that result differs, is unsafe or unclear, or an identified semantic or
   structural issue requires a changed head. If a rebase is already planned, do it before
   any other pushed change. When a base update is necessary, rebase onto the target
   rather than merging the target in, and prefer that rebase to closing and reopening the
   pull request or rerunning old CI: a rerun reuses the old synthetic merge, and a reopen
   reruns CI anyway without moving the head.
4. Every base merge, base-changing rebase, or other force-pushed rewrite is declared in
   the PR body with exactly one head-bound line before it is pushed:
   `Candidate-base-update: <new-head-sha> <specific conflict or semantic reason>` for a
   base update, or `Candidate-history-rewrite: <new-head-sha> <specific reason>`
   for any other rewrite. The preflight rejects a missing, duplicate, stale-head, or
   empty line. If it rejects an already-pushed head, repair the body and rerun that same
   workflow; do not manufacture another change to satisfy the guard.
5. Never force-push a red gate. Force-push only with an exact-head `--force-with-lease`.
6. Documentation-only changes still require applicable CI on the candidate head.
7. When reviews and repairs are complete and no further content change is planned,
   dispatch `PR Package Expansion` with the pull request number and exact head SHA,
   alongside the final required checks. Merge only after both the required checks and
   the expansion report have been inspected; read its introduced, inherited, and unrun
   counts as the guide describes. An ordinary content change, a base retarget, or a
   rebase the trusted verifier does not accept requires a fresh dispatch. Record the
   reviewed SHA and run link in the pull request.
8. Retargeting the base is an implementation change: `PR Base Retarget Validation`
   holds the head pending fresh compiler and Hull runs. A title or body edit reruns only
   `PR Contract Acknowledgements`.
9. Merge with a squash. Merging closes no issue; close the issues the pull request
   actually resolved as a separate step (see [Issue Tracking](#issue-tracking)), then
   remove the pull request's worktree and task-owned target once no follow-up needs them.

## Spec Authority And Design Discipline

### Documentation Authority

For project-level questions, what Chelis is, what it is for, and what the roadmap says:

1. `spec/design/chelis_canonical_reference.md` controls cross-subject architecture and
   project boundaries.
2. For a transferred chapter, its named capability in the pinned `chelis-plans`
   store controls the subject.
3. An untransferred `spec/00-12*.md` chapter controls its subject.
4. `spec/design/chelis_project_plan.md` controls project sequence that a higher
   authority does not define.
5. `spec/design/archive/` is historical reference only.

Language semantics belong in the numbered spec documents. A chapter transfers only
through a reviewed change that records the transfer; no chapter has transferred, so the
numbered chapters control and the pinned store's captured `chelis-*` capabilities
are reference. If active documents disagree, correct the document that controls the subject. Do not add a
third explanation.

### Normative Specs Are Timeless Contracts

A numbered chapter, a controlling capability spec, and every normative `spec.md` delta
under an active OpenSpec change describe the decided architecture and semantics,
irrespective of how completely any compiler version implements them.

- No project or implementation status in normative specs: no status banners, phase or
  milestone labels, completion claims, delivery histories, PR inventories, oracle
  results, temporary workarounds, or descriptions of what the implementation happens
  to do today.
- State the fully decided rule without weakening it to match a bug, an incomplete
  backend, or a temporary restriction. Never narrow an operation to one dtype because
  that is the only dtype a lane implements today. The implementation moves toward the
  spec; the spec never moves toward bad behavior.
- When an implementation gap would materially mislead a reader, one short
  non-normative parenthetical may say the requirement is not fully implemented and
  link its owning issue. It must not describe the workaround or qualify the rule.
- Sequencing and status live in `spec/design/`, `docs/`, GitHub trackers, or OpenSpec
  proposals, designs, and tasks.

### Numbered Specs Decide; Design Docs Implement

- `spec/00-12*.md` is the authority on WHAT the language does and HOW it must behave:
  semantics, types, dtypes, syntax, effects, op behavior, diagnostics, every
  user-visible contract. `spec/registry/` files are numbered-spec-tier content
  incorporated by reference into their owning `[05-OP-N]` atom, amended under the same
  review discipline.
- `spec/design/*.md` is the authority on how we IMPLEMENT and SEQUENCE those decisions.
  A design doc may elaborate a rule and record its reasoning, but it does not decide one.
- Where the two disagree, the numbered spec wins and the design doc has a bug. Say so in
  the doc rather than reconciling silently in code. When a design doc states a rule that
  is really a language decision, lift it into the numbered spec and leave a pointer.
- Watch for permission-to-mandate escalation. "X is a conforming implementation" in a
  spec does not license "therefore we do X" in a design doc, nor "we do X everywhere" in
  code. If your implementation needs a stronger rule than the spec states, amend the spec
  first and say so in the PR.
- A design doc is not correct merely because it was written down. Before implementing
  it, test its claims against the controlling spec, hardware and ecosystem reality, and
  Chelis's stated principles. If it is wrong, over-broad, or drifted, amend the
  controlling document and tracker before writing code.
- Where the spec leaves a real choice, the tenets above decide it: explicit spelling
  over contextual inference, and the structural design over the patch. If a design doc
  permits both, amend it to select the structural contract before implementing. This
  bias never overrides a normative semantic rule; amend that rule first when the
  language decision must change.

### Numeric Surface Discipline

`spec/design/dtype_semantics.md` §C6 binds every change that touches numeric data,
whether or not you have read it. The census failure messages name the sanctioned
actions; the census, tripwire, and oracle files are guard artifacts, and editing one to
make your change pass is never the fix.

- No numeric channel outside the tagged carrier. A public ADT variant, wire field,
  exported C signature or data declaration, or binding parameter or result that carries
  numbers as bare `f64`/`double`, or takes a raw integer dtype id, is a review-blocking
  finding with no citation or override path: redesign it onto the tagged carrier or
  remove it. Opening a fresh issue does not authorize capacity debt.
  No grandfather, permanent-disposition, successor-override, or integer-plumbing path
  is part of the final contract. Every discovered row ends in exactly one class:
  structurally nonnumeric, a recognized exact tagged carrier, or an exact numeric
  operation registration.
- Every new or changed numeric op, every stdlib ADT constructor with a numeric field
  included, requires an exact semantic registration in the same change set, binding its
  canonical identity to one verbatim existing `[05-OP-N]` atom
  (a definition line beginning `> **[05-OP-N]**`) in `spec/05-risc-primitives.md`. No
  governing atom means you author the atom first; re-check the highest existing number
  on current `main` before allocating. Then run
  `.venv/bin/python scripts/generate_rejection_registries.py --write` and commit the
  generated registry; that artifact is required in addition to the registration.
- Never silently narrow at ingress. A lossy dtype for ingested data is a decision: use
  the named lossy form (the chelis#759 pattern) or the exact dtype, and preserve source
  numeric distinctions as ADT variants (`JsonInt(i64)` beside `JsonFloat(f64)`), never
  one float funnel.
- A new surface kind that can carry numbers (serialization format, IPC channel, export
  mechanism) extends the §C6 enumerators in the same change set, or does not land.
- Published C ABI is configuration-invariant and every declaration is attributable: no
  preprocessor-varying public declarations, no `#line` directives, every published
  header reachable from a declared root. A built-in arithmetic type, bare `int`
  included, makes a callable `numeric-op`; names and parameter-name heuristics never
  turn a callable into plumbing; extents, allocation sizes, indices, and dtype
  selectors are numeric operations, and raw dtype selectors are forbidden. A
  conditional macro definition taints its whole connected local-include component.
  An arithmetic spelling the census does not recognize is a build failure, and so is
  an unclassified new `chelis_types::Prim` variant. PR #956's negative controls lock
  these rules; weakening one changes this contract and those controls together.
- An exported stdlib `def` declares its signature via `defsig`, or stops being exported.

### Public-Surface Change Rule

When behavior changes, update the owning code, tests, docs, and examples in the same
change set: parser, type system, IR, and backend tests; CLI integration tests; the
executable examples in `examples/`; and the active specs and current-state docs. The
[`spec-sync` skill](agent-skills/spec-sync/SKILL.md) walks the surfaces.

### OpenSpec

Chelis plans live in [Chelis-Lang/openspec](https://github.com/Chelis-Lang/openspec),
under `chelis-*` change and capability IDs. This checkout's `openspec/config.yaml`
is only a `chelis-plans` store pointer; `openspec/store.lock.yaml` pins the revision
validated by CI. Register the store before using `openspec`; a missing store is an
error, not permission to recreate a local planning tree. Author and submit planning
changes in a dedicated store worktree; implementation PRs cite the change ID and
accepted store commit. Planning remains optional and Phase 0 remains inactive.
`README.md` owns setup, review order, and the document-only acceptance boundary.
Do not run `openspec init`'s tool generation: `.claude/skills` and `.codex/skills`
are symlinks to `agent-skills/`.

## Change Hygiene

- **Changelog fragments.** A behavior-changing PR adds a fragment in `changelog.d/`
  named `<pr-or-slug>.<added|changed|fixed>[.breaking].md`; correct a pending fragment
  when follow-up work changes its claim. Internal work may use the `no-changelog` label.
  Reserve `CHANGELOG.md` edits for release assembly: the release author runs
  `.venv/bin/python scripts/changelog.py build --version VERSION --date YYYY-MM-DD`,
  reviews the preview, repeats with `--write`, and commits the notes, fragment
  deletions, and version bump together, never recreating `[Unreleased]`.
  [The fragment contract](changelog.d/README.md) owns the format.
- **Commit messages.** Plain conventional commits with the configured human author. No
  `Claude-Session` trailers, Codex or Claude attribution, AI co-authorship markers, or
  AI-session links in commit messages or PR bodies. The tracked commit-msg hook rejects
  them; `README.md` explains how it is installed.
- **Documents describe the current state.** A new or substantially revised document reads
  as a description of how things are, for a reader who never saw the history: no dated
  status ("as of", "currently", "no longer"), delivery narrative, or decision attribution;
  history lives in the changelog and its fragments, pull requests, and issues. A design
  doc may state the problem it addresses and the alternatives it rejected; investigation
  records and the archive keep their dates. No absolute machine paths or personal email addresses in
  tracked files.
- **Issues are closed manually.** Automatic closure is disabled: `Closes #N` in a PR body
  or commit has no effect, and merging never closes an issue. Close one deliberately
  with `gh issue close N --comment "resolved by #<PR>"` once the behavior is confirmed
  on current `main`. Write `Part of #N` or `Addresses #N` when a PR advances an issue
  without finishing it. Close a tracking hub only when every sub-issue is closed and the
  condition the hub names is met.
- **Contract invariants.** Express machine-facing contracts as invariants and lock them
  with tests: perfect success means an empty error list, formatter output stays
  parseable, decompiler output round-trips, executable examples stay executable after
  canonical formatting, status docs claim no more than the repo proves.
- **Manual gates.** Every manual acceptance gate has a documented command, expected
  success condition, and owning phase in [`docs/manual_gates.md`](docs/manual_gates.md);
  if default CI does not run it, the docs say so. Ignored tests are allowed only when
  they clearly mirror a documented manual gate or an environment-dependent prerequisite.
- **CLI surface.** Commands are product surface, not wrappers around library tests. Test
  formatter, decompiler, evaluator, checker, and build behavior against a corpus, and
  test machine-facing output for both shape and semantic invariants. The
  [`cli-surface` skill](agent-skills/cli-surface/SKILL.md) has the corpus rules.

## Issue Tracking

Agents file many issues; use the [`issue-resolution` skill](agent-skills/issue-resolution/SKILL.md)
when picking one up.

- A recurring defect class gets **one tracking issue**, which is also the GitHub
  sub-issue parent for every instance. Its body carries the plan; its evidence lives in
  the owning design doc or `docs/investigations/`. Set the parent link explicitly;
  `Part of #N` in a body creates none.
- Before filing, check whether the issue belongs under an existing tracking issue. Not
  every issue needs a parent.
- An issue has one parent. When a defect splits across classes, parent it to the class
  whose oracle turns green when it is fixed, and add an explicit `Also part of #N` line
  for the other.
- A class without a design doc is legitimate; say so in the tracker.
- `-label:tracking` is the work queue. Prefer the specific label: an issue labelled only
  `soundness` is not findable by anyone who does not already know it exists.

| label | means |
|---|---|
| `tracking` | a hub: a class or a plan, not a work item |
| `bug` | something is not working |
| `soundness` | semantics divergence, type-safety, or a wrong answer; not CI tooling |
| `spec-gap` | normative text was never authored |
| `design-discussion` | the decision exists and is contested, or a design wart |
| `enhancement` | new feature or request (`feature request` is a legacy duplicate) |
| `usability` | developer or user experience |
| `documentation` | docs additions or corrections |
| `failing-on-main` | red on main outside the per-PR checks, nightly- or dispatch-owned; open means still red |
| `nightly-failure` | a nightly workflow failure |
| `no-changelog` | internal change; suppresses only the missing-fragment check |
| `area:eval` | the `chelis eval` interpreter lane |
| `area:runtime` | `chelis-runtime` and the C ABI surface |
| `area:backend` | backend codegen: C, HIP, Metal, and IR lowering |
| `area:prove` | `chelis-prove`, SMT, contract discharge |
| `area:bindings` | Python bindings and the compiler-api embedding surface |
| `area:ecosystem` | shell repos, `reef conform`, ecosystem drift |
| `area:perf` | performance and benchmarking |
| `area:ci` | CI workflows, coverage, mutation testing, dev infrastructure |
| `type-system`, `cli`, `lint` | the type system, the CLI surface, and `chelis-lint` rules |
| `launch:p1` / `launch:p2` / `launch:p3` | launch triage: core promise silently false; loud core failure that ships as a known issue; roughness that does not gate launch |
| `launch:required` | a non-defect launch deliverable or authored decision |
| `demo-path` | breaks a release corpus case; gates launch regardless of priority |
| `fence-ok` | a typed documented rejection is an acceptable P1 resolution |
| `freeze` | a public contract decision that must be authored before launch |
| `scope:core` / `scope:experimental` / `scope:unreleased` | covered by the 0.19 core promise; ships as experimental; outside the public 0.19 release |

## Environment And Tooling

### Worktree And Branch Discipline

- The primary checkout (the main worktree in `git worktree list --porcelain`) is live
  developer state. Read-only queries are fine there; never switch branches, edit, build,
  or create scratch artifacts in it.
- Create a dedicated worktree before the first write of every task, including small
  documentation edits and throwaway probes, and give it its own `.venv` with
  `uv venv --python 3.11`; never copy or symlink another checkout's `.venv`.
- A worktree isolates the working tree, the index, and its HEAD reflog. The stash
  stack, `.git/info/exclude`, the hooks directory, and branch reflogs are shared by
  every worktree on the clone. Do not run `git stash` in a shared clone: to discard your
  own changes use `git checkout -- <paths>`; to park them, copy the files to task-owned
  scratch space or commit them on your branch.
- Do not repurpose an unrelated worktree because it appears idle. Reuse only for the
  same PR or immediate follow-up after checking ownership, exact head, status, and active
  processes. Never share a worktree with a reviewer while either of you writes to it.
- After a PR merges, remove its worktree and task-owned target with individual
  `git worktree remove <path>` and `cargo clean --target-dir <path>` commands, never a
  blanket loop, after confirming the PR is merged, nothing uncommitted is worth keeping,
  and no process owns the target. Squash merges mean "commits ahead of `origin/main`"
  proves nothing; compare patch ids when in doubt. Branch deletion is a separate decision.

### Python And Scripts

- A uv-managed Python 3.11 is a hard prerequisite on every platform. Create each
  checkout's environment once with `uv venv --python 3.11`; `README.md` has the setup.
  Every script is `.venv/bin/python scripts/<name>.py` (`python scripts/<name>.py`
  inside Devenv), and every ad-hoc invocation uses that interpreter too, never the
  system Python. `python3` appears only as the gate bootstrap and for the two
  bootstrap-free diagnostics `scripts/reap_orphans.py` and
  `scripts/preflight_exec_probe.py`, which stay standard-library only.
- Scripts, utilities, report generators, and automation helpers are Python, with tests.
  Rust where the task fits a compiled workspace member. Never shell: the only permitted
  `.sh` is `crates/chelisup/bootstrap/chelisup.sh`. A new `scripts/*.py` needs a
  `[[path_rule]]` or the CI planner fails closed on it.

### Build And Gate Commands

[`docs/local_gate.md`](docs/local_gate.md) records everything `scripts/gate.py` does.
The rules:

```sh
python3 scripts/gate.py --fast         # before every push: fixes in place, then checks
python3 scripts/gate.py --validation   # optional troubleshooting and extra validation
python3 scripts/gate.py --detach --validation   # optional run, detached
python3 scripts/gate.py --status [HANDLE]  # the detached run's real verdict
python3 scripts/gate.py --list         # the canonical command list with ownership annotations
```

- Fetch `origin/main` before any long local validation. Run `--fast` before every push
  of a non-documentation change; trivial changes may warrant focused tests or none.
  `--fast` passing means "no unrouted path and no failing tripwire in my tree", not
  "CI will accept this". CI on the pushed candidate owns routine validation, and the
  workspace suite runs only there and nightly; passing PR checks never certifies a
  phase acceptance oracle, so dispatch `heavy-e2e.yml` on the candidate when claiming
  completion.
- Use focused `cargo check -p <crate> --tests` and `cargo nextest run -p <crate>
  --test <file>` for the inner loop; never a workspace-wide `cargo test`. The same
  applies to a check you want early evidence for: run its owning test locally, for
  example `cargo nextest run -p chelis-cli --test capacity_census_tripwire`, rather
  than dispatching a large workflow such as `heavy-e2e.yml` ad hoc because it happens
  to contain that test. `.config/ci-test-targets.toml` names each test's owner.
- On failure the gate keeps the transcript under `target/gate-failures/` and prints the
  exact rerun command. A detached run's verdict comes from `--status`, never from the
  launch exit code.
- Concurrent builds use an isolated target: `CARGO_TARGET_DIR=target/agents/<name>` as
  an absolute path, or a separate worktree. Before building, run
  `python3 scripts/reap_orphans.py`, review the listing, and reap with `--kill`; at
  session end confirm your cargo, rustc, and nextest processes are gone. Several
  unrelated tests failing at near-identical wall-clock times is CPU starvation, not
  code breakage; rerun on a quiet machine.
- HIP manual gates run only through `scripts/hip_test.py` (or `chelis-hip-test` in
  Devenv); plain `cargo test --ignored` segfaults at exit and looks like a regression.
  [`docs/local_hip_environment.md`](docs/local_hip_environment.md) is the runbook,
  and [`docs/local_macos_environment.md`](docs/local_macos_environment.md) covers the
  macOS first-exec stall (chelis#356).

## Subagents

[`docs/investigations/agent_contract_rationale.md`](docs/investigations/agent_contract_rationale.md)
holds the measurements behind these rules.

- Every subagent prompt names the delivery mechanism and the complete expected report.
  A report that is not sent through the platform's final-report channel has not been
  delivered. A subagent never ends its turn merely to wait for a background build or
  notification that cannot wake it: keep ownership through a synchronous wait, or return
  an honest partial result. A reviewer that has delivered its round report is not
  waiting; it stays available for the orchestrator to resume with the fix.
- CI is watched by at most one background waiter whose exit wakes the session, or by
  nobody. Never watch CI from a foreground sleep or poll loop.
- If an agent returns "waiting" or goes idle without the deliverable, resume it
  immediately with the exact missing items. Prefer a labelled partial report over
  silence or an overstated completion claim, and deduplicate repeated reports that
  race with a resume nudge.
- Every follow-up message to a running subagent, and every message to a peer session,
  ends by asking the recipient to acknowledge it and confirm what it will do. No
  acknowledgement by the recipient's next reply means the message was not received:
  resend it, consolidated.
- More than five subagents live at once under one orchestrator needs the user's
  explicit approval and a stated reason. Five is the widest fan-out measured working
  here, not a certified safe width, and it is a separate budget from the CPU one above.
- Every spawn names its model tier and says in one clause why that tier fits: the
  expensive tier for judgement whose errors are costly to detect, the cheap tier for
  mechanical work such as waiting on CI, polling, or transcribing a result. The
  orchestrator states its own context size in the message that announces a spawn.
- Every brief states a numeric report-length budget, and a numeric context budget except
  for red-team rounds. An agent that will exceed its context budget says so and returns
  what it has.
- A brief says which facts the orchestrator has already verified, against what head, and
  that the agent must not re-derive them, and it names what the agent still has to
  establish itself.
- A brief longer than a few paragraphs is a file passed by absolute path, stored where
  it outlives both the agent and the session, never in a per-session scratchpad.
  Inter-agent messages truncate silently near four kilobytes. "Inline" means
  self-contained, the opposite of "read `AGENTS.md`", not pasted into the spawn message.
- Reports come back the same way: the agent writes the report to a file and replies with
  the absolute path and a one-line summary. That reply is the delivery.
- A subagent that reuses a worktree restores its temporary probes and reports the final
  worktree status unless asked to retain them. Before a heavyweight cargo command it
  reports the exact command and expected weight to the orchestrator.

## Writing Chelis Source

Load the [`example-corpus` skill](agent-skills/example-corpus/SKILL.md) before writing
any `.ch`; it carries the Surf style rules, the parse-breaking spellings, and the Deep
AST contract. `spec/02-surf-syntax.md` §0.1 is the authority.

- `chelis build`, `check`, `validate`, and `eval --file` run `chelis fmt --check` and the
  blocking `chelis lint` rules before the front end; style failures block the build.
  `--allow-style-violations` is for emergency local builds only, never CI, and
  `CHELIS_STYLE_GATE_DISABLE=1` is reserved for the integration-test corpus. Run
  `chelis fmt --inplace <file>` and `chelis lint --check` before pushing.
- Type system: no implicit precision promotion, named tensor dimensions match by name,
  no implicit broadcasting (explicit `expand` only), integer literals default to `i32`
  and float literals to `f32`.
- `chelis build` invokes the native compiler for C, HIP, or Metal and produces an
  executable or static library, retaining sources and runtime artifacts. `--emit-c`
  stops after source emission. CPU is the acceptance priority; GPU targets remain
  prerelease. In the chelis repository, `docs/book/src/backends.md` covers C
  builds and `docs/gpu_backends.md` covers HIP and Metal.

## Book

The user-facing book for this shell is the mdBook whose `book.toml` is under
`docs/`. chelis.ch mirrors it page for page at
`https://chelis.ch/docs/<package name>/` (chelis: https://chelis.ch/docs/chelis/),
and the chelis.ch text is canonical: book pages are rendered from the site by
the website's `scripts/sync_books.py`, so edit prose on the site and re-render,
or make the same edit in both places in the same change.

The reader is an engineer, or an AI coding agent, writing Chelis code against
this shell. They know the domain but not this repo's internals or history, and
they want to call the API correctly the first time. Every page teaches: what the
API does, a runnable example with its real output, the contract (inputs, domain,
shapes, precision, errors) and the pitfalls.

Never in the book: issue or PR numbers, repo-internal paths (`spec/`, `src/`
internals, `scripts/`, `tests/`, maintainer docs), maintainer or CI commands,
contributor history, process talk (gates, red teams, agent instructions),
status words (planned, not yet, stub, phase, milestone), "see the source" in
place of documentation, em-dashes, and the word "load-bearing".
`scripts/check_book.py` enforces the mechanical part in CI.

A change that alters user-visible behavior says so in its changelog entry.
The book documents the latest release: the chelis.ch page and this book take
the change when that release is documented.

## The Chelis-Lang Repositories

The public repositories, one line each. The conformance `REGISTRY` in
`crates/chelis-conformance` is the authority on which shells the conformance tooling
binds. Every shell consumes the compiler-bundled `chelis-std` runtime and is bound by
the shell contract; the registry records whether it does so through reef, a Cargo
workspace, or Docker.

| Repository | Contains |
|---|---|
| `chelis` | This repository: the compiler, runtime, CLI, `chelis-std`, `reef`, `chelisup`, the numbered spec, and the conformance tooling. |
| `nautilus` | Shell: numerical methods, statistics, linear algebra, optimization, ODE/SDE solvers, special functions. The scipy analogue. |
| `coral` | Shell: typed dataframes whose numeric columns are tensors. The pandas analogue. |
| `shoals` | Shell: quantitative finance on nautilus and coral: pricing, risk, curves, stochastic processes. |
| `hello-chelis` | Example programs; the smallest conforming shell. |
| `economoist` | Verified economic and dynamic-programming models in Chelis. |
| `arb-sys` | Rust bindings to the Arb arbitrary-precision library. |
| `shoreleave` | Bed package: bank, public and market holiday calendars from dated official sources, as `Std.Datetime.Business` calendars. |
| `meridian` | Bed package: the IANA time-zone database as `Std.Datetime.Zone` time zones. |

## Pointers

- **Shared skills** live in `agent-skills/`; `.claude/skills` and `.codex/skills` are
  symlinks to that one authored tree, `.claude/commands/` and `.codex/commands/` stay byte-identical, and the
  `red-team` alias is wired to `redteam-exec` with its fresh-round and verify modes. The
  set: `redteam-exec`, `spec-sync`, `phase-gate`, `backend-numerics`, `example-corpus`,
  `cli-surface`, `packaging-install`, `issue-resolution`. Shells also receive
  `chelis-std`, the downstream-authoring skill authored at
  `packages/chelis-std/SKILL.md`.
- **Toolchain and packaging.** `chelisup` is the installer and pin-resolving `chelis`
  shim; `chelis reef setup` is the orchestrator. Use the
  [`packaging-install` skill](agent-skills/packaging-install/SKILL.md) for any change
  there. One trap it enforces at compile time: `reef setup` subprocesses the real
  `chelisup` binary, never `chelisup::install::install` in-process, because that helper
  copies `current_exe()` over the shim. Design:
  [`spec/design/chelis_packaging_and_install.md`](spec/design/chelis_packaging_and_install.md).
- **Downstream shells** inherit this complete contract through a stamped managed block
  and must satisfy [`spec/design/shell_repo_contract.md`](spec/design/shell_repo_contract.md),
  shipped in the toolchain as `chelis reef conform`. Full inheritance is the default,
  but each shell decides which portions apply. Shell-owned additions stay outside the
  block and should remain when they are relevant and current. To omit an inherited
  section, put its exact ATX heading in a shell-owned span such as
  `<!-- shell-local:exclude:begin -->`,
  `<!-- ### Numeric Surface Discipline -->`, `<!-- shell-local:exclude:end -->`; sync
  removes that heading and its section, while deleting the selector restores it. The
  root `# Chelis Agent Contract` selector omits the entire inherited body.
  Contract changes land here first, editing the doc and the conformance
  `MANIFEST`/`REGISTRY` in lockstep. §7.1 of that doc is the audit a `conform bump` wave
  still owes after the mechanical starter runs.
