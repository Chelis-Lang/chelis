# Agent Quality Architecture

**Status:** Adopted guidance + implementation backlog (tracking issue
chelis#740). Written 2026-07-16 out of the numeric audit
(chelis#680-#734), whose failures were not discipline failures but
predictable *agent* failure modes: local context windows, task-local
optimization, no knowledge of what already exists. The five design plans
(#729-#733) fix the discovered classes; this document is the standing
answer to "how do we prevent the NEXT scattered implementation, parallel
duplicate build, or spec-silent feature" - for new work, on any branch,
written by agents.

## The enforcement ladder (the one organizing rule)

A rule's reliability is set by the channel that delivers it:

> **compile error > gate/lint failure > tripwire test > generated
> checklist > PR template > CLAUDE.md prose > tribal knowledge**

Everything below "gate failure" is advisory to an agent under token
pressure - the audit measured this: the repo had unusually strong prose
process and still produced four sightings of "the correct helper exists
next door, uncalled", drifted duplicate gates, and one storage decision
declared at four layers. **Standing rule: push every rule as far up the
ladder as it can go, and treat "it's documented" as the floor, not the
fix.** A compile error is the only context-delivery mechanism that
reaches an agent 100% of the time, in-context, at the moment it matters.

Corollary, from the audit's evidence about agents specifically: they
trust comments (comments hid the only unpredicted bug), they stop at the
reported symptom, they invent plausible fallbacks when a function cannot
fail, and they re-implement what they cannot see. Each mechanism below
targets one of those.

## The mechanisms

### 1. Tables and chokepoints (compile-error tier)

Wherever completeness matters - lanes, dtypes, tags, effect kinds,
gates - there must be ONE table or enum whose consumers are
exhaustive-matched or generated, so partial work does not build. New
features extend tables; they do not add parallel dispatch. Owned by the
plan set: the capability table (`spec/design/capability_table.md`),
`DeepTag`/`EffectKind` (#731/#730), the dtype finalizer (#729), the
generated formatter (#732).

### 2. Repo-specific lint rules (gate tier)

`chelis-lint` lints Rust source and runs in the gate (the §8.6 rule
proves the pattern). **Every incident closes with a lint rule when one
is expressible.** In flight: `rust-no-wildcard-dispatch` (#730 Phase 2),
the `spec-provenance` family (#733 Phase 1). The closing-move checklist
for any future incident: fix, test, THEN ask "what lint rule makes this
unwritable?"

### 3. Tripwires and the duplicate registry (test tier)

The conform MANIFEST and `scripts/test_gate.py` are the precedent:
anything that must exist in two places gets a test that diffs them. The
drifted gate pair (#697/#698) existed because nothing knew they were
copies. Backlog: a `duplicates registry` test enumerating every
intentional copy (spec table <-> code, CLI gate <-> compiler-api gate)
and failing on divergence; the #730 token tripwire generalizes the
pattern to known-bad code shapes.

### 4. Context injection where agents actually look (checklist tier)

- **Nested `CLAUDE.md` files per crate.** Claude Code loads a
  directory's CLAUDE.md when working there - guidance delivered at edit
  time, which root-level prose cannot do. Backlog: guardrail files for
  `crates/chelis-backend-c/` (emitted helpers are generated, never
  hand-write a dtype switch; the failure channel is Result-typed),
  `crates/chelis-ir/` (lowering must raise, never placeholder;
  fold rules), `crates/chelis-types/` (Type::Error discipline),
  `crates/chelis-lint/` (rule-registration protocol).
- **The mechanism index** (below): the canonical-helpers list an agent
  must consult before writing numeric/dispatch/gate code. The audit's
  best search heuristic, inverted into prevention.
- **Change-shape skills** in `agent-skills/`: `add-a-builtin`,
  `add-a-dtype`, `add-a-deep-tag`, `touch-a-gate` - each a checklist of
  every surface the Public-Surface Change Rule implies, generated from
  the tables of mechanism 1 where possible.

### 5. Review tuned for agent failure modes (process tier)

Measured, not stylistic (see the audit's scoring table in
`docs/investigations/numeric_audit_next_sweeps.md`): comments hide bugs;
code-reading invents bugs; controls catch overclaims; probe what was
cleared. Additions to the existing red-team protocol:

- **The sibling check**: any PR adding a helper, gate, or dispatch arm
  must show the search for an existing equivalent and either use it or
  justify divergence in the PR body. The direct countermeasure to
  scattered implementation; checkable by a reviewer agent.
- **Contract-first for multi-session features**: any feature spanning
  sessions or branches lands its spec/design contract (phase-handoff
  style: inherit / deliver / frozen-at-exit / not-yours / oracle) as a
  docs-only PR FIRST. Parallel agents then implement against frozen
  interfaces instead of colliding; every implementation PR names the
  contract it implements (enforced by #733 Phase 0's PR gate).
- Claim-before-work stays as the `issue-resolution` skill states
  (assignee set before branching).

### 5b. Archive evidence once; never keep a second copy of a test

Two rules that met in practice the day the probe corpus landed:

1. **No duplicate without a tripwire** (mechanism 3): once every probe
   row gained a committed test twin, the ~110 archived `.ch` fixtures
   were an intentional duplicate of the tests' embedded programs with
   nothing diffing them - a stale-copy hazard of exactly the #694 kind.
   Resolution: the fixtures were DELETED (tests are the single source;
   git history keeps the point-in-time set; the drivers and battery
   definitions stay as reusable sweep tooling). Rule: negative results
   are archived as TESTS, not as parallel fixture copies.
2. **An artifact that is never edited again should cost the gate
   nothing**: the fixtures were also the lint stage's long pole
   (`chelis lint --check .` re-walks the tree per checked file - the
   known quadratic behavior). The dedup resolved this instance; the
   general lint-side ignore mechanism for genuinely un-editable archive
   directories stays on chelis#740's backlog for the next case.

### 6. Standing detection (because prevention leaks)

- **Ratchet metrics in CI**: counts that may only decrease - `_ =>`
  arms over the named enums, `unwrap_or(` in lowering paths,
  `#[ignore]`s without issue numbers. Same delivery shape as the
  existing LOC-report job.
- **Scheduled audit agents**: the sweep methodology is written down and
  reusable (`docs/investigations/numeric_audit_next_sweeps.md` "The one
  rule" + the probe corpus). A periodic fresh-context agent runs the
  conformance matrices, the domain-validity invariant, and the debt
  reports against the week's landings. Execute everything; promote
  nothing from reading. Must include **ignored-suite drift
  monitoring**: run the `#[ignore]`d suites and diff observed failure
  values against each ignore note's documented value - CI never runs
  the red set, so a known-broken cell whose wrongness CHANGES is
  otherwise invisible, and pinning wrong values as assertions is
  forbidden (never document a bug as intended). Scheduled observation
  is the only honest monitor for known-broken cells.
- **Coverage honesty**: the committed matrices are POINT coverage
  (specific ops x dtypes x values, mostly rank-1 contiguous), not
  property coverage - clearances generalize by induction. The
  property-level guards are the plan set's Phase 0 invariants
  (domain-validity, round-trip, totality) and the endpoint is the
  generated conformance matrix (#729 Phase 4); until those land,
  treat "the matrix is green" as "the probed cells are green".
- **Downstream**: shells currently have no compiled-lane numerical
  validation at all - tracked as chelis#738 (conform amendment
  candidate).

## The mechanism index

Consult BEFORE writing numeric, dispatch, gate, formatting, or
error-path code. If what you need is here, use it; if you believe you
need a variant, say why in the PR (the sibling check). Additions to this
table are part of landing any new canonical mechanism.

| mechanism | where | use it for | do NOT |
|---|---|---|---|
| `checked_int_binop` | `chelis-compiler-api/src/runtime/host_ops.rs` | exact integer scalar arithmetic with traps | route integers through `Fn(f64,f64)` kernels (#680; superseded by #729's kernel split when it lands) |
| `raise_lowering_error` | `chelis-ir/src/lower.rs` | ANY unsupported case met during lowering | emit `RiscOp::Const {0.0}` placeholders (#699) or `unwrap_or_default()` extractions (#725) |
| `fold_static_size`'s `checked_i64` walk | `chelis-ir/src/lower.rs` | compile-time integer folding (decline on overflow) | f64 condition folding (#711/#720) |
| `chelis_int_div_guard` | `chelis-runtime/include/chelis_runtime.h` | the branded-trap message pattern | inventing new trap message shapes (see #730 §C2 / spec/05 §7 atoms) |
| `tensor_float_unop_f32` | `host_ops.rs` | (historical) f32 lane agreement | applying to non-f32 dtypes (#717); superseded by #729's finalizer |
| `lower_unsupported` | `chelis-ir/src/lower.rs` | the correct unsupported-tag response | writing new catch-alls that return values (#703) |
| HIP narrow-float rejection / Metal f64 rejection | `chelis-backend-hip` gate / `chelis-backend-metal` dtype | the calibration examples for unsupported diagnostics | silent `ElemKind` fallbacks (#689) |
| eval/c lane drivers (verbatim strings) | `crates/chelis-cli/tests/precision_matrix.rs` + sweep files, `docs/investigations/probes/` | any cross-lane numeric assertion | comparing through f64/tolerance (#687) or trusting printed tensors for int64/f16 (#723/#716) until #732 lands |
| the conform MANIFEST tripwire | `chelis-conformance` | any doc<->machine-form lockstep | hand-mirroring a doc into code with no diff test |
| spec atoms + status banners | spec/04 §9-§10, spec/05 §7-§8 | citing decided semantics; adding new normative text | writing MUST/SHALL prose outside atoms in atomized files (#733) |

## Relationship to the plan set

Mechanisms 1-2 are delivered BY the five plans; this document owns 3-6
and the index, tracked as an implementation backlog on the tracking
issue. The sequencing hook: mechanism 4's skills and nested guardrails
should land alongside Wave 1 of `spec/design/remediation_roadmap.md`
(they reference the failure channel and atom sections that Wave 0/1
create), and mechanism 6's ratchets are cheap enough to land with
Wave 0.
