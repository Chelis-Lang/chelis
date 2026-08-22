# Numeric Remediation Roadmap: sequencing, ownership, and the ledgers

**Status:** Living coordination document for the plan set that came out of
the 2026-07 numeric audit. This doc owns four things nothing else owns:
the **global sequencing** across the five plans, the **release boundaries**
(where the version cuts fall, sliced to keep each downstream-shell bump
digestible and never temporary), the **unclaimed-issue ledger** (every filed
issue that no plan's kill table claims, with its assigned home), and the
**deferred-evidence ledger** (claims still resting on inspection, each with
its verification task). It contains NO contracts
of its own - contracts live in the five plans; when this doc and a plan
disagree, the plan wins and this doc has a bug.

The plan set: [`dtype_semantics.md`](dtype_semantics.md) ([#729]), [`loud_unsupported.md`](loud_unsupported.md) ([#730]),
[`checker_totality.md`](checker_totality.md) ([#731]), [`faithful_observation.md`](faithful_observation.md) ([#732]),
[`spec_provenance.md`](spec_provenance.md) ([#733]), plus the [`capability_table.md`](capability_table.md) schema (rides
[#729] Phase 4) and the audit record under `docs/investigations/`.

## The class map

| meta (the class) | method (design spec) | tracker |
|---|---|---|
| [#727] no dtype's semantics enforced at any single point ([#695] = its integer instance) | [`dtype_semantics.md`](dtype_semantics.md) - per-dtype finalizer behind private constructors, int/float kernel split, one storage decision, generated backend dispatch | [#729] |
| [#703] unsupported cases substitute values instead of failing | [`loud_unsupported.md`](loud_unsupported.md) - Result-typed failure channels, the census sweep, dependency-bottom closed identities, fail-closed HostType/ABI states, structured emission, gates demoted to UX, and (2026-07-30) §C7 ratchet totality: derived-universe guards including Python/C consumers, typed/live exclusion authority with review-owned relevance, the typed kind/authority channel, and a non-shippable mutation-based panic-surfacing oracle | [#730] |
| [#709] unrecognized constructs silently exempt from checking (+[#710]'s silent half) | [`checker_totality.md`](checker_totality.md) - loud wildcard + handle-effect case, ErrorWitness token (silent Type::Error unconstructible), totality invariant, DeepTag exhaustiveness | [#731] |
| [#728] the observation channel is not dtype-faithful (**CLOSED 2026-08-21**) | [`faithful_observation.md`](faithful_observation.md) - one Rust formatter, generated C print helper, round-trip invariant, tolerance table; landable before [#729]; unblocks [#687] | [#732] (**CLOSED 2026-08-21**, on its own stated condition; [#728] closed with it) |
| spec silence + stale claims ([#694]; the unauthored cells) | [`spec_provenance.md`](spec_provenance.md) - OpenSpec plans changes after Phase 0 activation, while a pinned Buoy shell and one-way Chelis adapter provide repository-independent authority, freshness, coverage, and impact enforcement; design/fixtures may proceed now, advisory execution waits for Buoy's final oracle, and blocking waits for the adapter and Chelis configuration oracles | [#733] |

This map is scoped to the 2026-07 numeric audit's plan set and stays that way.
The `meta (the class)` column names a META issue per row; that pairing is
historical and is NOT the pattern for a new class - `AGENTS.md` section Issue
Tracking Conventions owns the one-tracker-per-class rule now.

**Sibling classes, tracked on their own issues, not here.** Listed only so a
reader of the five plans above does not go looking inside them for work they
structurally cannot deliver. Each row is a limit of one of the five and the
issue that took it:

| what one of the five cannot deliver | handled at |
|---|---|
| [#731] makes a silent `Type::Error` unconstructible by gating its constructor. The same move is unavailable for `Type::Unit`, which is an ordinary type with no constructor to gate (its own section C3 says so), and `DeepTag` exhaustiveness forces *a* disposition, not a correct one. Neither reaches the `Atom` / list-head domain | [#908] ([#885], [#887] Tier 2). [#887] Tier 1 stays with [#731], which does deliver *diagnosed*: the sub-issue link puts it under [#874], inside the [#731] subtree |
| [#730] makes callable rejections loud and explicitly non-goals making them WORK ("that is [#729]'s or an op-owner's work"), so a C-host function-value ABI has no owner anywhere in the five | [#909] ([#866], [#867], [#879]). [#868] sits in [#883]'s subtree with the rest of the span work; section C2 remains [#730]'s contract, and the parent link records where the fix lands |
| [#729] seals numeric construction behind private Rust constructors. It has no reach into the C runtime, where `chelis_tensor.data` is a `pub` untyped `*mut u8`; Phases 0-4 never touch it | [#893] ([#899], [#889]). [#892]'s bool storage still rides [#729]'s v0.19 cut |
| [#730] section C2 declares the diagnostic span normative and [#731] owns checker diagnostics, but neither has a phase that threads one: `Unsupported::with_span` has zero call sites and `CheckError::with_span_id` has none outside its own builder test | [#883] ([#868], [#886], [#916], [#1172]). [#730] section C2 keeps the span *contract*; the sub-issue links say where the *fix* lands |
| [05-OBS-1..5] were each conditioned on a stored numeric value reaching an exit, so root existence, naming, order, and the `build` artifact obligation were outside [#732]. [05-OBS-6] now authors the always-labelled manifest-order envelope and the unavailable-root [05-UNS-1] requirement; complete manifested observation/build consumption is still not delivered by the formatter plan | [#912] ([#820], [#862]), with the post-[#1003] integration/acceptance residue tracked at [#1023]. [#775]'s shape half was authored as [05-OBS-4] under [#732] |
| no numeric plan touches `chelis reef conform`'s audit surface, which the four bump waves below keep regenerating gaps in | [#788] ([#814], [#825], [#845]) |

None carries a wave assignment; they are not sequenced against the waves.

Supporting: [`capability_table.md`](capability_table.md) (schema; rides [#729] Phase 4),
[`docs/agent_quality_architecture.md`](../../docs/agent_quality_architecture.md) ([#740]), the seeded atoms
(spec/04 §9-§10, spec/05 §7-§8), and [PR #696](https://github.com/Chelis-Lang/chelis/pull/696) (the acceptance surface).

## Global sequencing

The plans are deliberately independently landable - every pairwise
interlock is pinned in both landing orders (each doc's §I1). The
*recommended* order optimizes for detectors-before-fixes and for
small-wins-early:

**Wave 0 - all five Phase 0s, in any order, immediately.** STATUS
2026-07-20: four of five LANDED 2026-07-17 - the domain-validity
invariant + [#687] oracle lanes ([#729] P0, PR #758; named receipt
`.venv/bin/python scripts/dtype_phase0_oracle.py`, continuously inherited by
the Phase 1/2 chain), the substitution
census verification + token tripwire ([#730] P0, PR #746), the
`Type::Error` census + red totality invariant ([#731] P0, PR #757 -
`issue_731_totality_invariant.rs`), the round-trip harness + exit
census ([#732] P0, PR #752). The one outstanding Wave 0 item is OpenSpec
activation + PR review routing + `spec/**` signoff ([#733] P0). The design PR
does not activate that workflow; Phase 0's configuration, instructions, and
oracle do. Phase 0 makes no Buoy assurance claim and keeps OpenSpec and
provider metadata outside canonical product authority. The two ordering
handshakes
(the `%.16g`/`%.1f` grep pattern landing in [#730] P0's tripwire; [#729]
P0 and [#732] P0 sharing lane drivers) were honored and are now
historical.

**Wave 1 - the small loud fixes.** STATUS 2026-07-20: COMPLETE. [#731]
Phase 1 (the checker holes: loud wildcard, the handle-effect case, the
seed-literal diagnostic, the MalformedForm sweep) merged as PR #793;
[#730] Phase 1 (the Unsupported Result channel with the speculative-
sub-lowering laundering path closed structurally, every live census row
converted) merged as PR #791. [#732] Phase 1 (the formatter + eval
adoption + the eval-side migration, a Wave 2 item entered early per the
standing exception) merged as PR #792 the same day. All three were
fresh-context red-teamed (QUALIFIED PASS x3) with every confirmed
finding folded in before merge; the red teams' discoveries are filed as
[#794]/[#795]/[#796].

**Wave 2 - the independently-landable value work.** [#732] Phase 1
landed with Wave 1 (above); [#732] Phase 2 also LANDED (PR #863) and
delivered the generated C side, fixing [#716]/[#723] outright and giving the
refactor its byte-exact instrument; its one authoritative completion
oracle is
`.venv/bin/python scripts/faithful_observation_phase2_oracle.py`,
accepted at exit 0 with final line `PHASE 2 ORACLE: PASS` - it carries
the known-red ledger that keeps every annexed [#729]-family cell
re-executed rather than silently skipped, and fails when one goes
green, so the upstream repair's landing forces the un-ignore in the
same change set. That protocol has now retired the [#864] cell: current
execution showed its root metadata was already F64, while the static
`to_tensor` lowering shortcut discarded the checked dtype of each literal
leaf before widening it. The construction fix carries NO typed leaf: static
extraction keeps [#856]'s exact `RawScalar` and finalizes only FLOAT leaves,
at the width read from each leaf's own type metadata. Routing every leaf
through an f64 value field - the shape an earlier draft proposed - would
have reintroduced the above-2^53 integer loss that exact lane prevents.
The original assertion covers enclosing-tensor and scalar-widening shapes,
and the repaired row is locked into the oracle's unconditional must-run
inventory after leaving the known-red ledger. The same protocol retired the
[#684] row ([#1078]) and annexed [#1110], whose compiled-lane half now leaves
the ledger through [#729] Phase 3: host lowering finalizes a suffixed literal
at its checker-stamped width before an enclosing widening cast. This is a
narrow [#717]/[#729]-family value repair and does not claim the atomic
per-dtype-storage Phase 1 migration. It ran in parallel with [#731] Phases 2-3 (the
witness token + DeepTag). [#730] Phase 2 is COMPLETE AND
ACCEPTED (PR [#799], merged 2026-07-24): closed vocabularies, staged
HostType/ABI separation, and structured emission, with the authoritative
oracle green (`PHASE 2 ORACLE: PASS`) and the plan-set's fresh-context
adversarial review run and dispositioned
(`docs/investigations/pr799_returned_function_values_redteam.md`). Its
initial source-lint approach was explicitly re-planned after execution
showed incomplete and false-positive behavior; the lint was extracted to
PR [#815] (since closed unmerged) and is not a Wave 2 dependency. [#733] Phase 1's Buoy shell-side design
and fixture preparation may ride alongside: OpenSpec still plans new
normative text and atom IDs remain stable. The executable advisory pilot
waits for the selected Buoy revision's standalone `devenv test` final
oracle and the shell-adapter prerequisites; once admitted, it reports
malformed authorities and stale registrations without blocking existing
Chelis commands or coupling `buoy-core` back to Chelis.
[#732] Phase 2 additionally gates the ECOSYSTEM's compiled-lane
validation: no shell runs a compiled binary today, and [#754]'s
cross-lane agreement gate (the mechanism [#738]'s conform row points
at) is hard-gated on byte-identical rendering.

**Wave 3 - the semantics refactor.** [#729] Phases 1-3 in order (the
module + storage decision; the kernel split + prove; backend adoption),
validated by everything Waves 0-2 built. Phases 1 and 2 landed through PRs
#1049 and #1054; PR #1065 is the bounded Phase 3 integer-`abs` seed, and this
revision delivers the remaining Phase 3 C value work. The required Linux
Integration job runs the nested
`.venv/bin/python scripts/dtype_phase3_oracle.py`, making the Phase 0-3
acceptance chain continuous. Entry gate: the [#729] §C6
covered-family capacity census/tripwire (PR #956) lands BEFORE Phase 1
entry. This change completes the two typed hard edges: the wire-schema and
PyO3 binding commands in [#729] §C6 are implemented with reviewed baselines
and red mutations (`capacity_census_wire` and `capacity_census_bindings`,
respectively). Phase 1 re-derives its §C3 layer set from that completed
census. These remain explicit thick-red DAG edges below, not editable
coverage metadata (PR #950 red teams P2-4 and re-P1). The exact PR #956 follow-up
`6ddf1a72d6dea6770a330d5c2ef3b8fa7d023c43` is part of that entry gate:
conditional macro definitions taint their connected local-include
component, bare and pointer-sized integer C callables classify
conservatively as `numeric-op`, and only the three byte-frozen
pre-ratchet plumbing declarations named in [#729] §C6 are exempt. The
2026-07-31 round-4 fold-in completes that entry gate: an arithmetic
spelling the census does not recognize is a build failure rather than an
unflagged row, and adding a `chelis_types::Prim` variant stops the
tripwire compiling until the new dtype is classified.

**Wave 4 - the permanent guards.** [#729] Phase 4 delivers the capability
table per [`capability_table.md`](capability_table.md); [#733]
Phase 3 (the first blocking provenance ratchet) ships with it only after
the advisory Buoy pilot and change-impact phases are green. **[#732] is
CLOSED as of 2026-08-21, and it is the first plan in this set to close.**
Phase 3 (the tolerance table + the [#687] handshake) shipped in v0.18.3
through PRs #1099/#1115/#1118 and is not v0.20 payload; [#754] remains its
external shell consumer. [#997]'s direct diagnostic-rendering contract debt,
the tracker's own stated closing condition, was retired on 2026-08-21 by the
`FO-DIAG` migration (PR #1250), and the tracker closed the same evening with
zero open children, an evidence record on the thread, and both owning
documents audited by full read. Its META [#728] closed with it. [#1059] was
re-homed from [#732] to [#1170] on 2026-08-21 before the close: its
acceptance is a compiled-lane capability event, which is [#1170]'s subject,
and an `Also part of #732` comment records the provenance. It stays open and
CI-pinned; the class closure loosened nothing. Wave 4 therefore has three
live slots, not four.
[#730] Phase 3 (gates become UX; amended 2026-07-30 to also deliver the
typed diagnostic-kind and rejection-authority work) keeps Wave 4 as its
recommended slot but is NOT gated on the capability table or the Buoy
pilot - its own deliverable stands on "let the emitter's channel speak"
whether or not the table has landed. This doc previously read as if it
were gated; per this doc's own rule the plan won and the sentence was
corrected 2026-07-30. As of 2026-08-05 the focused typed-authority and sealed
diagnostic-kind slices are implemented, followed by the first gate-contract
slice: one typed compiler-api policy now serves both public build paths, the
stale C precision preflights are gone, shared gates accept only the closed
`BuildTarget` vocabulary, and the exact syntactic `reject_*` manifest walks
free functions and `impl` methods across both crate source trees. The HIP
contract keeps
direct-load f16/bf16 `BlasMatmul` admitted but rejects real narrow-float
operand compute before emission. The source manifest is intentionally scoped
to syntactic `reject_*` definitions; it does not claim to detect a semantic
reimplementation hidden under an unrelated name. The final gate-contract
slice moves Metal/effect policy into those same typed compiler-api
definitions, including one shared traversal for host tensor-helper DAGs, and
scopes pure-DAG effect rejection to the entry actually emitted. It closes the
former host-helper `dropout` emitter panic, cites the compiled-kernel owner
[#1192], removes the no-op CLI reject hook, and corrects stale closed-#616
Metal/HIP diagnostic prose to [05-MOV-1]. The single Phase 3 runner is
`scripts/loud_unsupported_phase3_oracle.py`; it remains red only at the
independently owned [#912] root-realizability leg, so Phase 3 is not complete.
[#730] Phase 4 (ratchet totality, added
2026-07-30: product-source-manifest ratchets with non-Cargo language
adapters, typed/live exclusion references, the non-product
mutation-based panic contract, and the change-gated + nightly
structural-authority jobs) is
guard work that may land any time after [#730] Phase 2 and carries no
wave assignment. Its user-visible halves are decided by [05-UNS-5..6]
(spec/05 §7, authored 2026-07-30); the plan implements them. Citation
presence may land before blocking freshness and coverage, but every
selected capability row ultimately binds one current controlling atom
revision.

Standing exception: any Wave may be entered early for a cell that becomes
urgent - the interlock sections make that safe; this ordering is advice,
not law.

## The dependency DAG

The sequencing above as a graph. NOTE: GitHub renders this as one tall
cascade - a degraded but still correct view. For the intended
side-by-side column layout, open it in VS Code's markdown preview.
Solid arrows are the within-plan phase chains - the only hard sequencing.
Dashed arrows are soft interlocks with a recommended direction; the
alternative order is pinned in the owning docs' §I1 sections. Dotted
arrowless links are shared-component coordination with no inherent
order: [#730] Phase 2 delivers `EffectKind` and [#731] Phase 1 consumes
it (string-match + loud else if it lands first). `DeepTag` exhaustiveness is
owned entirely by [#731] Phase 3; it has no dependency on the extracted
[#730] source lint. The thick red edges are the hard dependencies: inside the
plan set, [#719]'s fix precedes [#732] Phase 3's `sqrt = 0` tolerance
row (SATISFIED 2026-07-17: PR #760 merged, [#719] closed - the row may
be authored when Phase 3 arrives); downstream of the set, [#754]'s
ecosystem gate is hard-gated on [#732] Phase 2's byte-identical
rendering. [#729] Phase 1 is separately hard-gated on PR #956's
covered-family tripwire and both typed-leg oracles. Not drawn
(for legibility): OpenSpec remains [#733]'s planning workflow while a
pinned Buoy shell and one-way Chelis adapter provide enforcement.
The graph is acyclic. Node colors
are the waves above: grey = Wave 0, green = Wave 1, blue = Wave 2, orange =
Wave 3, purple = Wave 4 (so [#733]'s advisory Buoy pilot, blue, may ride Wave
2); white boxes with dashed borders are standalone fixes outside the wave
structure. LANDED marks the four Phase 0s merged 2026-07-17, the three
Phase 1s merged 2026-07-20 (PRs #793/#792/#791), and [#730] Phase 2
merged and accepted 2026-07-24 (PR [#799]). [#730] Phase 4 (ratchet
totality, added 2026-07-30) hangs off its Phase 2 and interleaves
freely with its Phase 3.

```mermaid
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 45, "rankSpacing": 42}}}%%
flowchart TB
  classDef w0 fill:#ececec,stroke:#808080,color:#1a1a1a
  classDef w1 fill:#d9ead3,stroke:#5a8a4a,color:#1a1a1a
  classDef w2 fill:#d0e0f2,stroke:#4a78a8,color:#1a1a1a
  classDef w3 fill:#fce5cd,stroke:#c07f2f,color:#1a1a1a
  classDef w4 fill:#e2d5ea,stroke:#8e5ea8,color:#1a1a1a
  classDef ext fill:#ffffff,stroke:#999999,stroke-dasharray:4 3,color:#1a1a1a

  subgraph S733["#733 spec provenance"]
    direction TB
    n733p0["P0 · OpenSpec adoption<br/>+ PR review routing"]:::w0
    n733p1["P1 · advisory Buoy shell<br/>+ Chelis adapter"]:::w2
    n733p2["P2 · neutral change impact"]:::w2
    n733p3["P3 · blocking coverage<br/>+ surface ratchet"]:::w4
    n733p0 --> n733p1 --> n733p2 --> n733p3
  end

  subgraph S730["#730 loud unsupported"]
    direction TB
    n730p0["P0 · census re-verify + token tripwire<br/>+ rejected-cells corpus stub (LANDED)"]:::w0
    n730p1["P1 · Result channel + live-site sweep (LANDED)"]:::w1
    n730p2["P2 · typed vocabularies + Host ABI<br/>+ structured emission (LANDED)"]:::w2
    n730p3["P3 · gates become UX<br/>+ typed kind/authority"]:::w4
    n730p4["P4 · ratchet totality"]:::w4
    n730p0 --> n730p1 --> n730p2 --> n730p3
  end

  subgraph S731["#731 checker totality"]
    direction TB
    n731p0["P0 · Type::Error census<br/>+ red totality invariant (LANDED)"]:::w0
    n731p1["P1 · loud wildcard + handle-effect case<br/>+ #710 guard sweep (LANDED)"]:::w1
    n731p2["P2 · ErrorWitness token,<br/>invariant always-on"]:::w2
    n731p3["P3 · DeepTag exhaustive dispatch"]:::w2
    n731p0 --> n731p1 --> n731p2 --> n731p3
  end

  subgraph S732["#732 faithful observation (CLOSED 2026-08-21)"]
    direction TB
    n732p0["P0 · round-trip harness + exit census (LANDED)"]:::w0
    n732p1["P1 · format_element + eval adoption<br/>+ eval-side migration (LANDED)"]:::w2
    n732p2["P2 · generated C helper + to_list arms<br/>+ C-side migration (LANDED)"]:::w2
    n732p3["P3 · tolerance table + #687 handshake<br/>(LANDED, v0.18.3)"]:::w4
    n732p0 --> n732p1 --> n732p2 --> n732p3
  end

  subgraph S729["#729 dtype semantics"]
    direction TB
    n729p0["P0 · domain checker + #687 oracle lanes (LANDED)"]:::w0
    n729c6["C6 · covered-family capacity tripwire<br/>(PR #956)"]:::w3
    n729c6wire["C6 · wire-schema enumerator<br/>+ mutation oracle (THIS CHANGE)"]:::w3
    n729c6binding["C6 · PyO3 enumerator<br/>+ mutation oracle (THIS CHANGE)"]:::w3
    n729p1["P1 · semantics module + storage decision<br/>+ eval adoption"]:::w3
    n729p2["P2 · kernel split + traps + prove"]:::w3
    n729p3["P3 · C backend adoption<br/>+ generated observation"]:::w3
    n729p4["P4 · capability table"]:::w4
    n729p0 --> n729c6
    n729c6 ==>|"HARD: covered-family oracle"| n729c6wire
    n729c6 ==>|"HARD: covered-family oracle"| n729c6binding
    n729c6wire ==>|"HARD: wire leg green"| n729p1
    n729c6binding ==>|"HARD: binding leg green"| n729p1
    n729p1 --> n729p2 --> n729p3 --> n729p4
  end

  n719["#719 sqrt fix (FIXED: PR #760)"]:::ext
  n683["#683 i64::MIN literal (standalone)"]:::ext
  n713["#713 pad_sequences dtype (standalone)"]:::ext
  n754["#754 cross-lane gate (ecosystem)"]:::ext

  n730p0 -.->|"tripwire hosts the<br/>%.16g / %.1f pattern"| n732p0
  n730p2 -.-|"EffectKind (either first)"| n731p1
  n731p3 -.-|"DeepTag (either order)"| n730p2
  n730p1 -.->|"census rows 6-7 raise first<br/>(else 729.P3 absorbs them)"| n729p3
  n732p1 -.->|"format_element Rust side<br/>(I1: either order pinned)"| n729p1
  n732p2 -.->|"generated print helper<br/>(I1: either order pinned)"| n729p3
  n729p3 -.->|"value divergences shrink<br/>before the oracle turns on"| n732p3
  n729p4 -.->|"Rejected(reason) cells<br/>feed gate derivation"| n730p3
  n719 ==>|"HARD: precedes the<br/>sqrt = 0 tolerance row"| n732p3
  n683 -.->|"natural moment"| n729p2
  n713 -.->|"natural moment"| n729p3
  n732p2 ==>|"HARD: byte-identical rendering<br/>is the gate's prerequisite"| n754
  n730p2 --> n730p4

  style S733 fill:#FFFFFF,stroke:#C3CCD3
  style S730 fill:#FFFFFF,stroke:#C3CCD3
  style S731 fill:#FFFFFF,stroke:#C3CCD3
  style S732 fill:#FFFFFF,stroke:#C3CCD3
  style S729 fill:#FFFFFF,stroke:#C3CCD3
  %% linkStyle indices are 0-based over EVERY edge in declaration order, and a
  %% chain (a --> b --> c) contributes one index per arrow. There are 33 edges
  %% here (0-32; index 32 is the appended #730 P2->P4 phase edge, default
  %% styling). 21,22 are the two `-.-` either-order links; 13,14,15,16,28,31
  %% are exactly the six `==>` HARD edges. Adding or removing any edge above
  %% renumbers everything after it - recount before editing these two lines.
  linkStyle default stroke-width:2.5px
  linkStyle 21,22 stroke:#9AA7B0,color:#7A8894,stroke-width:2.5px
  linkStyle 13,14,15,16,28,31 stroke:#B3362B,stroke-width:5px,color:#B3362B
```

## Release slicing: where the version cuts fall

Sequencing above is about *phase dependencies*; this section is about *release
boundaries* - a different axis. The plans are independently landable, so a
version cut can fall wherever a coherent set of phases is green; the thing that
decides *where* is **downstream-shell migration cost**, not the internal wave
numbers.

The governing rule: **every release moves a shell-facing surface to its FINAL
state, or does not touch it at all.** The failure mode to avoid is a surface
that lands in an intermediate state a shell must adapt to and then re-adapt to
- silent to loud to supported in three separate bumps, `seed(42)` to `42i64`
to some third form, render-shape A to B to C. A one-time break shells absorb
once is cheap; a break they make and later unmake is the expensive kind. Like
the sequencing above, this ordering is advice, not law; a cut may move if a
cell becomes urgent.

Current baseline (2026-08-05): **v0.18.4 is shipped**. Three further patch cuts
landed on the v0.18 line after v0.18.1, and the last two of them are real
migration cuts - do not read the patch-level version as "mechanical". v0.18.4
in particular is breaking on four boundaries at once, two of which land on the
anti-churn invariants below in ways those invariants do not settle.

**v0.18.2** (2026-08-03) was additive: the `chelis-std` CSV/JSON serializers
(chelis#928), the eval-lane JSON/CSV builtin families (chelis#890, chelis#903),
`chelis eval --timeout` (chelis#914, chelis#930), Nix packages and a Devenv
shell (chelis#907), [#729] Phase 2's typed integer semantics, and exact
trapping integer `abs` in C (chelis#1065). Its one source-visible delta went
the wrong way and shipped as a **recorded divergence**: eval's `shape` returned
int32 against [05-DIM-2]'s int64 SHALL, released with an explicit
do-not-migrate-onto note rather than a migration instruction (chelis#1120).

**v0.18.3** (2026-08-04) is both a **source migration** and an **exact-output
migration**, per its own release record:

- [05-DIM-1]/[05-DIM-2] extent dtypes shipped (chelis#1130, for chelis#1112):
  `shape()` now returns int64 and the movement bounds (`shrink`/`pad`/`stride`/
  `expand` extents) take int64. Bare int32 extent call sites become check
  errors - extents get `i64` suffixes, while axis parameters stay int32. This
  also retires 0.18.2's chelis#1120 divergence, so a shell that skipped 0.18.2
  migrates once rather than twice.
- Compiled-lane `round` now half-ties to even (chelis#1142): an exact-output
  change on half-tie values that brings C into conformance with
  roundTiesToEven. Eval was already conforming.
- `cast_trunc` exists as [05-OP-6] and is a new reserved word (chelis#1144):
  identifiers named `cast_trunc` no longer parse, and shells blocked by
  [04-NUM-14]'s fractional-cast trap migrate each site onto it.
- [#732] Phase 3 shipped its tolerance table, shared comparator, [#687]
  handshake, and continuously nested oracle (PRs #1099/#1115/#1118). This was
  behavior-preserving guard work and adds no shell migration.

**v0.18.4** (2026-08-05) is a **source migration**, an **ABI break**, and a
**wire break** in one patch cut, per its own release record. Four boundaries
move:

- **The published C ABI carries extents as `int64_t`** (chelis#1149, part of
  [#1112]): `chelis_tensor`'s `shape`/`strides`/`size`, `chelis_alloc`, and
  `chelis_alloc_view` widen, while `chelis_tensor_shape`'s `axis` parameter
  NARROWS to `int32_t` as [05-DIM-1]'s axis-domain half. `ndim` and `dtype`
  stay `int`. Anything linking against `chelis_runtime.h`, or compiling or
  consuming emitted C, rebuilds. `ChelisGpuTensor` deliberately does not
  move, which is why [#1112]'s GPU half stays open.
- **The Surf grammar moves to canonical v0.19** (PR #1031, the [#1024] track):
  one canonical repository and producer spelling, a strictly wider accepted
  input set, and decompilation routed through a typed Deep-to-Surf resugaring
  boundary. Because the style gate runs `chelis fmt --check` ahead of `build`,
  `check`, `validate`, and `eval --file`, source that was canonical under
  0.18.3 can now fail before the front end runs. Migration is
  `chelis migrate surf --from 0.18 --inplace`; identifiers that became
  reserved words are named, not guessed, and the rename is authored.
- **`defsig` is a same-unit annotation and deferred inference must resolve at
  its declaration** ([04-INF-1], PR #1178): an orphan `defsig` no longer
  checks, and an unknown-constructor obligation still unresolved at its
  declaration boundary rejects. Programs leaning on an under-constrained
  signature hole or a cross-unit `defsig` need authored annotations.
- **WireDag payload schema 4 -> 5** (PR #1181): `Pad` fill values are sealed
  in `ScalarValue` end to end. Schema-4 payloads migrate on read; 0.18.4
  artifacts are not readable by older compilers.

Also user-visible without a migration: integer overflow in `scatter`,
`cumsum`, `trace`, and `einsum` now traps in release builds as well as debug
(PR #1181); checked `cast` is exhaustive over all 81 active dtype pairs and
elementwise traps select the lowest row-major flat index (PR #1189
implementing [04-NUM-15], closing [#1150] and [#1152]); a warm `chelis build`
against a library stops re-typechecking it (PR #1176), which invalidates
0.18.3 on-disk caches by design and reorders emitted C for multi-package reef
builds.

**Two of those four boundaries land on the anti-churn invariants below, in two
different ways, and neither is settled by the invariant text as written.** The
ABI widening is the exported-signature change invariant 7 makes v0.19 payload
*by default*; 0.18.4 was not a cut promised "mechanical", so the absolute
clause held and only the default was overridden - by a release decision, not by
an amendment here. The schema 4 -> 5 step is a scope question rather than an
override: invariant 1 governs [#729] §C3's storage decision, and schema 5
sealed the `Pad` carrier, a different surface that §C3's storage has still not
followed. Either way a binding consumer adapts twice. Record both as open
maintainer calls rather than reading either invariant as satisfied or as
violated.

The v0.17 and v0.18 rows
below are therefore historical records, while v0.19 and v0.20 remain planned
cuts. v0.18.1 already shipped [05-OBS-6]'s `name = ` prefix in both lanes. It
did not deliver the manifested root set/order, unavailable-root behavior, or
artifact boundary; that remaining root-topology migration rides the next
source-migration cut (v0.19). A cut's migration note is the **breaking-change
summary**, not the full release contents: most of what a cut carries is ordinary
work, and only the deltas called out below force a downstream change.

### What actually forces a downstream change

| class | shell-visible change | cost |
|---|---|---|
| A - syntax migration | `with seed(42)` -> `42i64` ([#731] P1) | one-time, final |
| B - loud rejection of silently-wrong code | [#730]/[#731]/[#729] loud paths, [#730] P2 host-type | shells fix a real bug; permanent |
| C - wire / binding break | [#729] §C3 per-dtype storage (schema + Python payload) | one-time; **must be atomic** (§C3 forbids partial adoption) |
| D1 - payload rendering change | [#732] canonical element/container grammar, then [#729] own-width eval completion | exactly two authorized steps: the shipped formatter grammar and one final eval value-width correction in v0.19; frozen afterward |
| D2a - root label prefix | [05-OBS-6] `name = value` | SHIPPED once in v0.18.1; frozen; does not reopen D1 |
| D2b - manifested root topology | [#912]/[#1023] complete root set/order, dotted expansion, unavailable-root diagnostics, and artifact routing | one coordinated post-v0.18.1 migration in both lanes; may not alter D1 or D2a |
| E - reject-now-support-later | [#730] loud reject -> [#729] kernel lands | the add-then-remove-workaround trap |

The class-E anti-churn tool is the capability *decision* itself: whether a cell
is supported or a stable, cited `Unimplemented { issue: #N }` rejection is a
behavior fact, and **every such decision ships in v0.19 with the dtype break** -
never deferred to the v0.20 table. A shell then writes the `chelis#N` narrowing
citation once, against 0.19, and removes it only when the kernel actually lands
- a real event, not a slicing artifact. The v0.20 table only *mechanizes*
decisions 0.19 already made, so it is behavior-preserving by construction.

### The cuts

| cut | carries | shell impact | notes |
|---|---|---|---|
| **v0.17.0 - loud checking + canonical eval rendering** (SHIPPED) | [#730] P1 + [#731] P1 + [#732] P1, and everything else merged since 0.16.1 | **source migration** (wave 1) | breaking deltas only: loud rejections (incl. the new loud compiled-lane `test_*` assert, [#796] - the old inert-`0` stub is gone, so anything leaning on it fails heavily in E2E), `with seed(n)` -> `42i64` ([#731] P1 - the Shoals / Whale / hello-chelis HEAD canaries fail on unsuffixed seeds), and dtype-faithful eval rendering. Seed form frozen ([#735] changes only meaning); eval payload render frozen (C matches it at 0.18) |
| **v0.18.0 - checker totality, DeepTag, host-type/ABI boundary, compiled rendering** (SHIPPED) | [#731] P2 (PR #800) + [#731] P3 (DeepTag) + [#730] P2 (PR #799 vocab + host-type/ABI state) + [#732] P2 (compiled render) | **mechanical** for shells | no wire break ([#730] P2 preserves the `CHELIS_*` ids); the added loudness lands on already-broken code, so no *expected* source migration. Completes byte-identical payload rendering. The release-hygiene requirement is that the tarball ships `chelis_runtime_dtype.h`, which public `chelis_runtime.h` includes |
| **v0.18.1 - always-labelled root prefix** (SHIPPED) | [05-OBS-6]'s `name = value` prefix from #994, with the stale repo expectations synchronized in #1011 | **exact-output migration** | shipped the prefix once without changing payload digits or value shape; it did not prove manifest completeness, dotted expansion/order, unavailable-root diagnostics, or artifact routing |
| **v0.18.3 - extent/cast migration + faithful-observation guards** (SHIPPED) | [05-DIM-1/2] extent migration + `round` parity + `cast_trunc` + [#732] P3 (PRs #1099/#1115/#1118) | **source + exact-output migration** for the extent/round/cast changes; [#732] P3 itself is behavior-preserving | Phase 3's tolerance table and shared comparator shipped here and must not be scheduled again in v0.20 |
| **v0.18.4 - canonical Surf + int64 C ABI + declaration contracts + WireDag 5** (SHIPPED) | PR #1031's canonical Surf v0.19 grammar and total Deep resugaring ([#1024]) + chelis#1149's int64 extent ABI (part of [#1112]) + PR #1178's [04-INF-1] declaration contracts + PR #1181's WireDag schema 5, with PR #1189's exhaustive checked cast ([#1150], [#1152]) and PR #1176's dependency typecheck cache riding along | **source migration + ABI break + wire break** | four boundaries in one patch cut. Migration: `chelis migrate surf --from 0.18 --inplace`, rebuild against the new `chelis_runtime.h`, author annotations for orphan `defsig` and unresolved deferred inference; WireDag payloads migrate on read one way only. The ABI widening overrode invariant 7's default and the schema step raises a scope question invariant 1 does not answer, so the "bindings adapt once" promise needs re-adjudicating before v0.19 |
| **v0.19.0 - grounded dtype storage break + every behavior-changing capability decision + manifested root completion** | [#729] P1-P3 landed atomic per §C3, **plus every capability decision that changes behavior** - integer-overflow traps and each supported-vs-`Unimplemented` disposition (int `mean` [#724], bool arithmetic [#726], the [#715] rows, the HIP/Metal/C reject cells, and, of the cells seeded 2026-08-04, only [#722]'s integer-unary B-cells still ride this cut - the other three are delivered: [#704]'s scalar activation c-host B-cell and the scalar `tan`/`atan`/`recip` row landed on `main` after the v0.18.4 tag in PR #1188, which closed [#704], [#712], and [#715], and are carried by the 0.18.5 cut; [#937]'s per-dtype `uniform_like` emission shipped in v0.18.4 with PR #1181) - plus the prelude JSON/CSV integer-capacity decision (a `JInt`-shaped variant and integer accessors) for any prelude numeric channel this cut admits - invariant 7 HOLDS such channels out of every earlier release unless they land integer-capable from the start ([#729] §C3's amended census) - any published-ABI signature change deferred here by anti-churn invariant 7, and [#912]/[#1023]'s complete manifested root boundary | **source migration** (wave 2) | the storage break and the root-topology expectation migration are coordinated here; class E resolves here, not at the 0.20 table. Canonical Surf and the int64 ABI already shipped at 0.18.4, so this cut no longer carries them. Bindings adapt to the per-dtype payload once *from here*; capability behavior and root topology are final; the v0.18.1 prefix does not move again |
| **v0.20.0 - behavior-preserving permanent guards** | [#729] P4 (the capability *table*, mechanizing 0.19's decisions) + [#730] P3 (gates -> UX) + [#733] P3 (first blocking provenance ratchet) | **mechanical** for shells | guaranteed behavior-preserving: no decision, rejection, or rendered byte changes here - 0.19 shipped them all. [#732] P3 already shipped in v0.18.3. [#733] P0 lands independently before this cut, while its P1-P2 advisory integration is a prerequisite rather than v0.20 release payload. `tests_blocked/` probes are re-adjudicated against the now-standing table |

Net downstream shape: the original four-cut `conform` model assumed one bump,
probe re-run, and inventory refresh per minor cut. Actual history inserted
three additional shipped contract patches: **v0.18.1** changed exact
root-output expectations by adding the [05-OBS-6] prefix, **v0.18.3** changed
both source (int64 extents) and exact output (compiled `round` half-ties), and
**v0.18.4** changed source (canonical Surf, `defsig`/deferred-inference
rejection), the published C ABI, and the WireDag schema at once. The
source-visible waves are now shipped **v0.17** (seed suffix + fixing
loud-rejected code), shipped **v0.18.1** (root label prefix), shipped
**v0.18.3** (int64 extents, `round` half-ties, the `cast_trunc` reserved word),
shipped **v0.18.4** (canonical Surf v0.19, the int64 C ABI, the declaration
contracts, WireDag 5), and planned **v0.19** (the remaining
capability behavior + [#729] §C3 storage + manifested root topology). v0.20
remains mechanical. No later wave may undo or restyle an earlier one - and a
patch-level version number is not by itself evidence that a cut is mechanical,
which four consecutive patch cuts have now demonstrated rather than predicted.

Cadence for non-contract work: an **internal-only** change (a refactor, a
checker-internal fix, doc-only work) normally **rides the next planned cut**
rather than getting its own release; only an **urgent downstream bug fix** - a
shell blocked on a real defect - justifies an out-of-band patch release. PR
#819's `compile_and_load` metadata fix did not ship in 0.18; it now rides the
next compatible planned cut unless a current shell blocker justifies a separate
patch.

### The anti-churn invariants

1. **Atomic wire break.** [#729] §C3 is all-layers-or-nothing; never split the
   storage decision across releases or binding consumers adapt N times. The
   invariant is intact on its own terms - §C3's per-dtype storage has not
   shipped - but v0.18.4 moved a DIFFERENT wire surface first (WireDag payload
   schema 4 -> 5, sealing the `Pad` carrier), so a consumer now adapts at
   0.18.4 and again at the storage break. Decide whether this invariant governs
   the WireDag schema at all, or only §C3's storage layout, before the next cut
   has to rely on the answer.
2. **Payload grammar once; own-width completion once; root-prefix once;
   root-topology once.** [#732] froze the numeric grammar and
   container/scalar shape for eval at P1 and C at P2. [#729] Phase 1 is the
   one permitted follow-up to eval payload digits: per-dtype storage completes
   shortest-round-trip rendering at the declared width (for example f32
   `1.2247449159622192` becomes `1.2247449`). No later [#729] phase may change
   those digits or shapes. [05-OBS-6] later authored a separate envelope around those frozen
   payloads; its `name = ` prefix shipped once in both lanes at v0.18.1. The
   remaining manifested root set/order, unavailable-root, and artifact cut
   rides v0.19 once. Neither later step may reopen [#732]'s formatter decisions
   or restyle the shipped prefix.
3. **Frozen seed form.** [#731] P1's `i64` suffix is the final syntax; [#735]
   authors only meaning. Safe to ship to shells at 0.17.
4. **Decisions before tables.** Every behavior-changing capability decision
   (supported vs a stable, cited `Unimplemented { issue }`) ships in v0.19; the
   v0.20 table only mechanizes them. Never leave a bare loud error in one cut
   that a later cut reclassifies - a shell cites once, against 0.19.
5. **v0.20 is mechanically behavior-preserving.** If any change in the guards
   cut would alter a result, a rejection, or a rendered byte, it belongs in 0.19
   instead. That guarantee is what lets shells treat the 0.20 bump as pin-only.
6. **Ship what the public headers include.** A release whose public runtime
   header gains an `#include` (0.18's `chelis_runtime_dtype.h`) must add that
   file to every tarball; a published-artifact smoke that compiles a trivial C
   unit against the shipped `chelis_runtime.h` catches the omission before it
   reaches a downstream build.
7. **Published-ABI signatures freeze between cuts.** From now until the 0.19
   storage break, a change to an exported signature in the published runtime
   headers is 0.19 payload by default - it rides the one budgeted ABI break,
   never a cut promised "mechanical". A mechanical cut must be able to assert
   the header-signature inventory unchanged; [#729]'s §C6 capacity tripwire is
   the mechanism once it lands. Motivating instance: PR #891's
   `chelis_pad_sequences` gaining an `int pad_dtype` parameter (flagged in the
   2026-07-30 sweep, unmerged) inside the window the 0.18 note promises "no
   wire break". The same freeze governs prelude numeric channels: a PR adding
   one (the #891 `Json` shape) ships in NO release before 0.19 unless it lands
   with its final integer-capable form from the start - "if it lands before
   the break" is a fact pattern, not a hold; THIS sentence is the hold (PR
   #950 red team P2-5). The in-tree precedent is already final-from-the-start:
   `packages/chelis-std/src/io/json.ch` carries `JsonInt(int64)` beside
   `JsonFloat(f64)`.

   **v0.18.4 shipped an exported-signature change ahead of the 0.19 storage
   break** (chelis#1149: `chelis_tensor`'s extent fields, `chelis_alloc`,
   `chelis_alloc_view`, and `chelis_tensor_shape`'s `axis`). The cut was not
   promised mechanical and named the rebuild in its release note, so no shell
   was misled - but the invariant's stated default ("0.19 payload by default")
   did not hold, and it was overridden by a release decision rather than by an
   amendment here. Either amend the invariant to say what actually governs an
   ABI change in a breaking patch cut, or record why 0.18.4 was the exception;
   leaving it as written invites the next reader to treat the freeze as
   binding when it is not.

### Per-cut conform checklist

Every minor cut gets a mechanical `conform bump` PR wave across the shells;
the 0.17 and 0.19 source-migration cuts additionally carry real source edits.
The shipped v0.18.1 patch was an extra exact-output expectation migration, the
shipped v0.18.3 patch carried both a source migration (int64 extents) and an
exact-output one (compiled `round` half-ties), and the shipped v0.18.4 patch
carried a source migration (canonical Surf, the declaration contracts), an ABI
break, and a wire break together. Each owed the same probe and inventory
refresh, and 0.18.3 and 0.18.4 owed real source edits as well - 0.18.4's are
tool-assisted (`chelis migrate surf --from 0.18 --inplace`) except for the
reserved-word renames it deliberately refuses to guess. Each wave carries:

- a migration note as the **breaking-change summary** (the delta below), the
  `chelis#NNN` refs it closes, and the exact surface that changed - not the full
  changelog;
- a re-run of every shell's `tests_blocked/` probes (a probe flipping green =
  remove the narrowing citation for that ref);
- the multi-location pin-consistency guard and the `docs/CHELIS_SURFACE.md`
  capability-inventory refresh;
- a published-artifact smoke that compiles a trivial C unit against the shipped
  `chelis_runtime.h`, so a header the release forgot to package (0.18's
  `chelis_runtime_dtype.h`) fails the release, not the downstream build;
- at a wire-schema bump: acknowledgement at each Python consumer. This was
  written as "0.19 only"; 0.18.4's WireDag 4 -> 5 already owed it.

Migration-note stubs (the breaking delta per cut):

- **0.17** (source migration) - "`check`/`build` now fail loudly where they
  silently substituted (incl. compiled-lane `test_*` asserts, which no longer
  stub to an inert `0`); `with seed(n)` requires an `i64`-suffixed literal
  (`seed(42i64)`); eval output is dtype-faithful (integers print as integers).
  Known HEAD-canary casualties: Shoals, Whale, hello-chelis (unsuffixed seeds),
  plus any E2E leaning on the old assert stub."
- **0.18.0** (mechanical) - "pin bump only: more previously-silent errors are
  caught (bogus casts, non-record field access, malformed host types) but on
  already-broken code; compiled and eval output now render byte-identically; the
  runtime tarball gains `chelis_runtime_dtype.h`. The public runtime header
  also drops the unused `chelis_print_f32` export ([#732] Phase 2 - a tensor
  print with no emitter in any backend, so no compiled program could reach
  it); a shell that declared it directly loses a symbol it could never have
  usefully called."
  NOTE, against the natural instinct to bundle: [#894]'s
  `chelis_fill_bool_bits(t, uint32_t)` -> `chelis_fill_bool(t, uint8_t)` rename
  does NOT ride this cut. It is a bool STORAGE change (4-byte f32 encoding to
  one native byte), so anti-churn invariant 1 puts it in **0.19** with the rest
  of [#729]'s storage break - the int32 decode fixes go the other way, because
  completing [#730] §C6.2 is 0.18 payload. Two C-ABI deltas in one header do
  not justify merging two cuts; the storage decision is all-layers-or-nothing
  and the header is only one of its layers.
- **0.18.1** (exact-output migration, shipped) - "every emitted root now uses
  the `name = value` prefix in both lanes. Update exact stdout expectations;
  payload digits and value shape are unchanged. This does not yet promise the
  complete manifested root set/order or unavailable-root diagnostics."
- **0.18.3** (source + exact-output migration, shipped) - "`shape()` returns
  int64 and the movement bounds (`shrink`/`pad`/`stride`/`expand` extents) take
  int64 per [05-DIM-1]/[05-DIM-2]; bare int32 extent call sites are now check
  errors, so suffix extents `i64`. Axis parameters are unchanged at int32.
  Compiled `round` half-ties to even, an exact-output change on half-tie values
  only. `cast_trunc` is a new reserved word: rename any identifier using it, and
  migrate sites blocked by [04-NUM-14]'s fractional-cast trap onto it. This cut
  also retires 0.18.2's recorded int32-`shape` divergence, so do not migrate
  onto 0.18.2's shape return."
- **0.18.4** (source migration + ABI break + wire break, shipped) - "four
  boundaries move at once. Run `chelis migrate surf --from 0.18 --inplace`
  across the tree: Surf's canonical form changed and `chelis fmt --check` runs
  ahead of `build`/`check`/`validate`/`eval --file`, so previously canonical
  source now fails the gate. Identifiers that became reserved words are named
  by the migrator, not rewritten - author those renames. Rebuild anything that
  links `chelis_runtime.h` or consumes emitted C: `chelis_tensor`'s extents,
  `chelis_alloc`, and `chelis_alloc_view` are `int64_t`, while
  `chelis_tensor_shape`'s `axis` narrows to `int32_t`. An orphan `defsig` no
  longer checks and an unresolved deferred-inference obligation rejects at its
  declaration boundary; add authored annotations. WireDag payloads move to
  schema 5 and migrate on read - 0.18.4 artifacts are not readable by older
  compilers. Integer overflow in `scatter`/`cumsum`/`trace`/`einsum` now traps
  in release as well as debug, and on-disk typecheck caches from 0.18.3 are
  invalidated by design (the first build after upgrading is cold)."
- **0.19** (source migration) - "dtype semantics are grounded: integer overflow
  traps instead of wrapping, per-dtype tensor storage (wire-format v2, Python
  payload shape changed), narrow dtypes preserved end-to-end. Eval float tensor
  elements now render shortest-round-trip at their own width, so an f32 tensor
  prints `1.2247449` where it printed `1.2247449159622192`; every op x dtype
  capability decision is now fixed (supported, or a cited stable rejection).
  Checked casts now trap instead of choosing an implicit conversion for
  fractional float-to-integer values (`cast(3.5, int32)`) and non-member bool
  values (`cast(2, bool)`); apply `floor` or `round` before the integer cast,
  and produce exactly 0 or 1 before a bool cast. Python `np.uint64` ingress now
  raises `ChelisError` instead of silently producing an f64 payload; choose an
  explicit int64 or f64 conversion.
  The v0.18.1 root prefix is unchanged; the complete manifested root set now
  appears in manifest order in both lanes. Update expectations for added or
  reordered dotted roots, and treat an unavailable owed root as a named
  diagnostic rather than a missing line or missing `main`. Surf's canonical
  form and the int64 C ABI are NOT part of this cut - both shipped at 0.18.4
  and must not be migrated twice."
- **0.20** (mechanical) - "pin bump only: the capability table, gates-as-UX,
  and [#733] Phase 3 blocking provenance ratchet land; [#732]'s cross-lane
  oracle already shipped in v0.18.3. Phase 0 landed independently and Phases
  1-2 were advisory prerequisites. All encoding decisions already shipped in
  0.19 - no behavior change."

### Historical tag gate and the 0.17 sequence

A source-migration cut separates two kinds of breakage, handled differently:

- **Repo-owned** failures gate the tag. The heavy E2E failures from the new
  loud compiled-lane `test_*` rejection ([#796]) are the repo's own tests and
  **must be green before tagging** - a cut is never tagged over a red repo E2E.
- **Shell-owned** failures do not gate the tag; they are the *expected*
  migration signal (the Shoals / Whale / hello-chelis HEAD canaries failing on
  unsuffixed seeds). Their `conform` bump fixes are **prepared before the tag**
  so shells migrate promptly once it lands.

The planned 0.17 sequence was:

1. Merge the corrected release-slicing change (this PR).
2. Once the repo-owned `test_*` E2E is green, make the short-lived 0.17 release
   commit and tag it.
3. Verify the published assets (the header smoke: every `#include`d runtime
   header is in the tarball).
4. Start the shell `conform` bump migrations (the seed-suffix fixes, prepared
   in advance).
5. Merge PR #800 (the 0.18 witness work) only afterward - it does not ride 0.17.

## The unclaimed-issue ledger

Filed issues no plan's kill table claimed, each now with an owner. Rule:
an entry leaves this ledger only by appearing in a plan's issue map, in
the capability table's seed decisions, or by being closed.

**Scope note.** This ledger covers issues in the numeric audit's families.
Issues outside them - the Python bindings cluster, the 0.17.1 parser
regression, the prove surface, the host-lane I/O proposals, the stdlib assert
gap - are carried by the `area:*` labels and their own trackers, not here.
The rows the 2026-08-04 validation pass added
(`docs/investigations/remediation_status_2026_08_04.md`) apply that scope by
PARENT rather than by subsystem: [#870] and [#872] surface on the prove
lane but are parented to [#730], so their disposition is owned here while
the rest of that surface stays with its own tracker.

**Assignment now lives in the GitHub sub-issue graph**: every issue below is
parented to the class that owns it, so "which plan claims this" is a
structural fact rather than a table entry. What this ledger still owns is
the **disposition prose** - the two-part splits, the natural moments, the
raise-or-prove caveats - which a parent link cannot carry. A row whose
disposition is fully captured by its parent may be deleted; a row carrying
a judgement may not.

| issue | what | assigned home |
|---|---|---|
| [#681] | Std.Decimal negative `result_scale` leaks an unbranded error | standalone small fix; diagnostic text should conform to [#730] §C2 when touched. No plan dependency. |
| [#683] | `i64::MIN` not writable as a literal | standalone front-end fix; natural moment is [#729] Phase 2 (the exact int lane makes the round-trip testable), but nothing blocks doing it sooner |
| [#689] | HIP int64 ops emit F32 kernels | two-part: the SILENT half dies at [#730] Phase 1 (`elem_kind` raises); the SUPPORT half is owned by capability-table B-cells `Unimplemented { issue: #689 }` until int64 kernel templates land (see [`capability_table.md`](capability_table.md) seed decisions) |
| [#690] | HIP has no integer div-by-zero guard | rides the same HIP B-cell work as [#689]; the guard is part of `Implemented` for HIP int division cells, not a separate errand, and the kernel-template guard is the whole fix. Deliberately NOT labelled a missed migration: what already holds this is a RECORDED CELL ([`capability_table.md`](capability_table.md) names the div-guard gap on the [#689] rows), not an executable mechanism, and Table-B mechanization is Phase 4 work - so until then the ownership is documentary and the row is a STRUCTURAL GAP whose mechanism is scheduled rather than standing. Future detector, once the GPU lanes join [#754]'s cross-lane gate (both it and [#737] still open; [#736] is closed): eval aborts branded on a zero divisor where HIP returns exit-0 garbage, so lane agreement re-catches a regression here without a hand-written HIP-only assertion |
| [#691] | C DAG lane emits fmaxf/fabsf for int64 | repaired in [#729] Phase 3: integer min/max/abs dispatch through exact checked integer paths; the original rows remain ordinary regressions. The [#730] rejection-authority liveness pin held the ISSUE open while emitter sites cited it as live authority; PR #1164 rehomed those citations to historical prose (`chelis-backend-c/src/emit.rs:821` and `:3271` today describe the repair rather than a live rejection), `spec/design/loud_unsupported_issue_manifest.json` no longer lists it, and [#691] CLOSED 2026-08-04 with the `rejection-authority-liveness` job green. The pin itself is unchanged and still binds every OTHER cited authority: `scripts/validate_rejection_issue_manifest.py` fails closed on a closed authority, which the issue-closing reference in PR #1151's own body demonstrated when that PR merged |
| [#693] | Metal int64 `abs` zero emission | root cause is [#699] (confirmed by emission); Metal B-cell `Unimplemented { issue: #693 }` until the MSL integer path is wired post-[#699]-fix |
| [#704] | nine builtins compiled to the literal `0` on a SCALAR operand while the tensor forms were correct: `relu`/`sigmoid`/`silu`/`gelu` errored in eval (hard error became a wrong number), and `tan`/`atan`/`floor`/`ceil`/`round` were CORRECT in eval (the lanes disagreed silently, which was worse) | Repaired by the 2026-08-05 pre-table scalar numeric slice under [#729]: the checker applies explicit numeric-domain policy, eval admits the decided scalar activation family, and C host emission dispatches every named operation without a silent-zero fallback at every admitted width. The same slice completes [#712]/[#715]'s scalar integer identity and min/max rows, authors the timeless scalar rules in spec/05, and makes the hand-curated cross-lane matrix unconditional with negative parity. This was a MISSED MIGRATION onto already-decided semantics, not a new local rule. [#729] Phase 4 still owns the permanent generated A/B projections and conformance product; it cannot omit the scalar cells the interim matrix missed here and at [#937] |
| [#713] | `pad_sequences` allocates int32 output for int64 input | repaired in [#729] Phase 3 at the typed runtime allocation/copy boundary; the int64 row is in the authoritative Phase 3 oracle |
| [#734] | `to_string` on tensors/lists compiles to the literal `<value>` | [#730] census row 3; the substitution died at [#730] Phase 1 and is unwritable after Phase 2. CLOSED 2026-08-04: the compiled catch-all is now a §C2 `Unsupported`, so the placeholder is gone; the remaining support half - actually rendering tensors and lists via [#732]'s formatter in the C host lane - is tracked at the open [#1059], re-homed from [#732] to [#1170] on 2026-08-21 when that tracker closed |
| [#751] | generated C emits uncompilable / sign-losing float constants (f64::MAX as integer literal; -0.0 as `-0`) | repaired in [#729] Phase 3 by exact-bit C literal emission; all four former `C_LANE_EXCLUDED` rows returned to the always-run corpus and the exclusion ledger is empty |
| [#754] | shell-invokable cross-lane agreement gate (owner: brittonr) | downstream consumer, not plan-set work: hard-gated on [#732] Phase 2; consumes Phase 3's tolerance artifact and [#729] Phase 4's capability table (cell skipping); GPU lanes join after [#736]/[#737]; [#738] is its consumer; the one-comparator rule is pinned in [#732]'s §C4.3. Scope boundary (2026-07): a verdict proves lane agreement for its RECORDED (target triple, C toolchain + flags incl. -ffp-contract, libm identity) only - never cross-platform determinism by itself; the platform axis compares verdicts across CI matrix entries under the same tolerance table. Platform priority (Jeff, 2026-07-20): server-side Linux x86-64 is the primary verdict platform before any wider matrix. The -ffp-contract entry in the provenance flag set now has a measured in-house exemplar: the pre-[#770]-fix `uniform_like` affine was contraction-dependent (PR #779 removed the sensitivity at the source) |
| [#763] | `chelis lane-check` - the exact-only, Nix-hermetic first slice of [#754] (owner: brittonr) | child of the [#754] row: same one-comparator rule and proof-scope boundary; for its exact-safe corpus, byte-identity holds TODAY. [#732] Phase 2 unlocked corpus expansion, and [#729] Phase 3 returned the former [#751]/[#761] curated gaps as ordinary regression rows |
| [#761] | C lane flushes f32 subnormal literals to zero at ingress (the to_tensor route; found by [#719]'s fix session) | repaired in [#729] Phase 3 by exact f32 bit emission at the C ingress; the direct and cast-mediated rows are unconditional oracle fixtures |
| [#775] | scalar top-level roots render as a rank-0 tensor in eval but a bare scalar in compiled C | CLOSED 2026-08-04 after re-verification. Phase 1 decided the bare scalar/rank-0 form on 2026-07-20 and ratified it as [05-OBS-4]; Phase 2/3 regressions now prove both eval roots and compiled C exits use that form. The divergence as filed and its former C-lane mirror no longer reproduce. The separate complete-root-set/order/artifact contract remains [#912], not residue of [#775] |
| [#780] | matmul shape checking lost through an unannotated lambda parameter - a Surf-reachable false green (found in [#773]'s red team; pre-existing on both sides of the [#773] fix) | ADDRESSED by [#731] PP1 and `spec/04-type-system.md` [04-INF-1]. A checker-owned obligation ledger records every shape-computed operation reached from a lambda-owned unknown constructor, including synthesized/authored bare type holes and projection-derived variables; prevents let-generalization for that lambda; replays the ordinary rule when first use within the declaration binds it; and rejects any residual obligation at that declaration's own boundary. A top-level shape helper therefore declares its parameter's outer constructor instead of borrowing a later declaration's call site; a result annotation alone is not a remedy. The exact false-green spellings and projection cousin are named §C4.4 regressions, while consistent and ordinary-polymorphic controls stay green. This is the class mechanism, not a `matmul` retry |
| [#783] | annotation writeback degrades an unresolved Var to a rank-0 default and clobbers a concrete annotation (a silent [#703]-class substitution; the enabler of the transient [#773]-fix conv2d ICE, hotfixed same day) | ADDRESSED inside [#731] PP1. One writeback information-ordering gate now prevents every Var-, Error-, or partial-dimension-derived candidate from replacing a more informative existing type expression. Symbolic metadata is still legal when no richer annotation exists, preserving generic checked programs. The byte-preservation regression and the pre-existing eleven shape-override guards cover both channel layers; this is no longer an inventory of point checks |
| [#794] | `.dp`-reachable lowering-side value substitutions the PR #793 red team confirmed: `extract_f64_value`'s catch-all folds a `(par ...)` bound's FIRST child (spec/03 says last), and `extract_usize_value` silently maps a negative `.dp` int64 seed to 0 | [#730] census extension rows. The checker side is already closed (PR #793 narrowed its accept-set to `lit` and rejects negative literal seeds), so both are checker-unreachable today - the lowering fix is defense-in-depth per §C1.4 |
| [#795] | conv2d's present-but-non-literal stride/padding fall to `unwrap_or(1)`/`unwrap_or(0)` in lower.rs (the [#776] value-default shape; census row 23, discovered in [#730] Phase 1's sweep) | TRUE MISSED MIGRATION under [#730], not a standalone rule. `loud_unsupported.md` census row 23 requires one optional-static-argument resolver that distinguishes absent, resolved, and present-but-unresolvable; defaults apply only to absence, and unresolved present arguments raise through the frozen typed channel. The same change audits every optional compile-time argument in `lower.rs`, lands absent/present negative parity, and shrinks the numeric-unwrap tripwire. A conv2d-only branch leaves the class open |
| [#796] | compiled-lane `test_*` assertion builtins: pre-[#730]-P1 binaries compiled assertions to inert `0` stubs (could never fail); now loudly rejected | RESOLVED BY DECISION 2026-08-04: the row's own second option ("an authored eval-only contract") is already authored, in `spec/05-risc-primitives.md` §3.6.1 - the `test_*` family is host-only, has no compiled-lane emission arm, and its rejection is deliberately liveness-scoped through the C host emitter's catch-all rather than the whole-program host-only gate. So the DEFECT this issue filed is dead: PR #791 removed the inert stub, and a compiled program that calls an assertion on a reachable path is rejected loudly. What survives is a FEATURE request - a compiled binary that can fail its own assertions - which §3.6.1's closing sentence already specifies (real C assertion helpers). CLOSED 2026-08-04 citing §3.6.1; leaving it open under a class tracker would have made the class read incomplete when its actual defect is fixed, which is the inverse of the honesty this plan set exists for. The compiled-assertion FEATURE and the broader finding it surfaced - that nothing tells a user what the C lane cannot do, because the answer lives in four hand-maintained lists plus an emitter catch-all that have already drifted - are carried by [#1170] |
| [#840] | `chelis build` reports success but emits non-compiling C for defs named after C keywords (`double`, `long`, ...) - the identifier cousin of [#751] | standalone `chelis-backend-c` emission fix at the `CIdentifier`/`EmittedExpr` chokepoint (mangle or reject loudly, never uncompilable C from exit 0); joins [#751]/[#761] in [#763]'s corpus curation; found by the PR [#799] red team (F2), likely pre-existing |
| [#847] | `grad` over a function-valued model parameter unifies independently declared rigid dim params `n` and `m` at check time (the generic Jacobian-row wrapper is rejected; the concrete-dim variant checks) | VERIFIED STALE/MISSED MIGRATION at the PP1 baseline: the exact `jacobian_row` reproducer already checked clean on current `origin/main`, before PP1 changed code. No `grad` exception or rigid-unification patch was added. [04-INF-1] records the channel rule, the exact program is now a permanent positive regression, and a body that genuinely requires `n = m` remains the negative control. Close the instance as non-reproducing without pretending it supplied a second mechanism |
| [#850] | a call to a `sig`-declared export with no `def` body passes `chelis check` at score 1.0 with an EMPTY error list, and `chelis build` lowers it to an undeclared C function that fails only at the native toolchain (surfaced via Std.Io.Parquet) | CHECK HALF ADDRESSED under [#731]: `spec/03` §2.2 now makes `defsig` a same-unit annotation for a same-name `def`, and declaration collection rejects the general orphan class independently of persistent context. The §C4.4 corpus has the orphan negative and paired positive. A full stdlib sweep found and backed all three signature-only regions (Parquet, SafeTensors, Xavier) with explicit fail-loud Chelis bodies. The BUILD half remains [#730]/[#763] work; this change does not claim that half or close [#850] as a whole |
| [#851] | match-arm pattern binders leak into top-level cycle detection: a valid program whose top-level binding name matches a pattern binder anywhere in its call graph is rejected with a false `binding cycle` (0.17.1) | ADDRESSED as the predicted MISSED MIGRATION. `collect_eager_refs` now treats each arm as a lexical scope and removes binders from guard and body references. The exhaustive binder walk moved to one shared `chelis-deep` helper consumed by authoring, macros, and the checker, so no fourth permissive copy exists. Reef's Surf `Pattern` helper is deliberately separate because it consumes a different typed AST, with that boundary documented. Exact, nested-pattern, and genuine-cycle controls are pinned |
| [#888] | `DimExpr::normalized_key` folds concrete dimension factors with `saturating_mul`, so two different tensor sizes get the same key and C/HIP memory planning reuses a slot at the wrong capacity (VERIFIED by execution, Surf-reachable) | standalone. Kept here because it is the compiler-internal face of this plan set's own trap contract: [04-NUM-3] makes USER integer overflow trap in every lane while the compiler's size arithmetic saturates here and wraps in [#889]. That is an accident, not an exemption, and should not be read as one. [#889] rides [#893]'s runtime hardening |
| [#870] | `chelis-prove` aborts (SIGABRT, via the wait-timeout SIGCHLD self-pipe) when the environment denies `sendto`, instead of degrading | [#730] LU1, one supervised external-process degradation path. Both current Beacon subprocess launch/wait paths route through a single closed `Completed | TimedOut | Degraded(Unsupported)` supervisor; direct `wait-timeout` use elsewhere is forbidden. The oracle denies each supervisor mechanism in turn and requires a branded nonzero result with no abort, plus normal-completion and real-timeout controls. Catching `sendto` at the reported site is not the fix |
| [#872] | `chelis-prove`'s `type_from_deep_depth` depth-32 fuse projects to `Type::Unit`, so an opaque-type producer nested deeper than 32 drops out of the obligation set | [#730] LU2, total proof-type traversal. The same region also has `type_contains_depth`'s depth-16 `false` fuse, so a one-site raise would leave the same defect class alive. Both fuses are replaced by one iterative, cycle-aware worklist: cycles converge by visited identity, resource exhaustion is a typed error, and neither condition fabricates a type or boolean. The oracle crosses both former limits, exercises a cycle, and plants a low budget |
| [#878] | `RiscOp::Pad { fill: f64 }` carries the fill through f64, so an int64 fill above 2^53 collapses | repaired on the [#729] sealed-carrier path: IR and Wire Pad fills are `ScalarValue`, lowering finalizes at the padded tensor dtype, WireDag v5 rejects a raw-number spelling, and the legacy migration obtains dtype only from the owning node's output type. Exact >2^53 tests cover lowering, eval, wire round-trip, and C emission; wrong-dtype IR is rejected. Both capacity censuses SHRINK rather than cite an exemption. Context/stdlib caches move to V7/4 with the serialized shape. Beacon remains an explicit downstream consumer mismatch under [#708]: its advertised 1-3 set does not accept v5, so negotiation must reject before dispatch |
| [#901] | `TensorValue`'s mixed integer/float equality arm computes `i.abs()` ad hoc for its exact-representability test and panics on `i64::MIN` (`chelis-ir/src/eval.rs:164`) | repaired as the intended MISSED MIGRATION: `integer_is_exactly_representable` lives in the sealed semantics module, is total over `i64::MIN`, and derives significand width from the actual float dtype (bf16/f16/f32/f64). `TensorValue::PartialEq` consumes that predicate in both operand orders. The boundary locks deliberately distinguish exact powers of two (`i64::MIN` at f64) from inexact neighbours (`i64::MAX`, `2^24+1` at f32), so a lossy fixture cannot hide an exact stored value |
| [#906] | eval aborts with a stack overflow on flat list literals of roughly 2-4k elements, killing the literal-baking data path | [#730] LU3, canonical iterative compiler list spines. Literal baking and every equivalent compiler-authored `Cons`/`Nil` traversal share one iterative spine iterator/folder with a typed improper-tail result. The oracle crosses the reported 2k-5k range and structurally forbids a second recursive canonical-spine walker. This does not absorb user-authored recursion [#257] or general lowering/cost-model recursion [#409], whose oracles are different. Raising the stack only moves the cliff |
| [#937] | `emit_uniform_like` writes f32 samples through `float *data` with no dtype dispatch, so f64 `uniform_like` returns near-zero garbage in compiled C while eval is correct | repaired under [05-OP-8]'s all-active-float decision. The shared sampler computes f64 output with one f64 FMA from exactly widened f32 bounds, f32 with one f32 FMA, and f16/bf16 with that f32 result rounded once. Eval consumes it directly; C emits all four active float widths, and HIP emits matching typed f32/f64 samplers while its reduced-float cells remain explicit unsupported cells under [#174]. IR verification rejects a non-float template. Structural tests forbid f32 stores into f64 output, the Linux C gate compares raw f64 bits, and the HIP manual gate does the same on-device. The permanent recurrence guard remains Phase 4's generated cell product; this repair supplies the concrete cell it will consume |
| [#942] | `cast` rejects any tensor whose element type came from inference (`expand`, `uniform_like` results) - a `_ =>` wildcard turning an unresolved-but-legal type-state into a false rejection | repaired structurally: the cast result-state match is exhaustive, and positional `expand` records a monomorphic two-shape obligation on its unresolved result. Declared results and later tensor consumers may select same-rank replacement or rank-plus-one insertion; unification validates the selected shape, while a shape-neutral `cast` materializes the spec/04 default (same-rank on an existing axis, insertion when `axis == rank`). Before a final checked program is annotated, every still-unselected result materializes that same default and the checker refreshes earlier owner stamps. A reusable library context instead serializes the obligation so downstream code retains the first shape-bearing choice after a cache round trip. Locks cover both shapes, later-consumer insertion, scalar/trailing insertion, otherwise-unconsumed results, serialized-context selection, cast after `expand` and `uniform_like`, rejection of an unrelated result rank, rejection of an axis beyond the trailing position, and rejection when two uses try to choose different shapes for one binding; genuinely unconstrained and function-valued cast operands remain negative controls. Hull's `RExpand` rows remain the acceptance oracle. A future type-state addition is now a compile error at the cast match rather than another false rejection hidden by `_ =>` |
| [#955] | backend-c host lane cannot lower nested `Option`/`List` composites ([05-UNS-1]); separately, nothing in CI builds a std-importing program, so eval-only std code ships green | Two structural [#730] items, not two patches. LU4 derives nested host ABI support recursively from the closed constructor table in [`capability_table.md`](capability_table.md); a represented `List`, `Option`, and inner type imply both nesting orders, while an unimplemented constructor returns its exact authority. LU5 derives a build/link/run-or-reject cell for every exported stdlib module from the export manifest. The issue closes only when both oracles are green; a special nested-shape arm or one hand-picked std importing test satisfies neither class |
| [#980] | integer overflow semantics differ by op, build profile, and lane | VERIFIED and repaired against [04-NUM-3]. The reproducer confirmed scatter add wrapped explicitly while cumsum/trace/einsum depended on debug-overflow settings. Their shared runtime arithmetic trait now uses checked add/multiply for int32/int64 and native IEEE arithmetic for floats, emitting `numeric trap: overflow in <op> at <prim>` at the FFI boundary. Subprocess controls exercise all four ops at both widths; the recurrence commands are `cargo nextest run -p chelis-runtime --test issue_980_integer_overflow --no-fail-fast` and the same command with `--release`, so profile-dependent behavior cannot return |
| [#1009] | `pad_sequences` / `pad_sequences_to` have no normative spec text (surfaced by PR #891) | repaired as the ratchet's backfill case: spec/05 owns [05-OP-9] and [05-OP-10], including signatures, extents, per-dtype movement semantics, truncation-at-width for `_to`, non-differentiability, and the absence of an accumulator. The exact public callable identities are registered to those atoms in the capacity census, and the generated rejection registry is refreshed. Phase 4 consumes these authored cells; there is no grandfathered exemption |
| [#1113] | axis-argument dtype enforcement is inconsistent across ops: `sum` rejects int64 axes while cumsum/concat accept both | repaired at both authority and mechanism. [05-DIM-3] records the already-decided split: every axis-domain argument is int32, while `reduce_window_*` window shapes and strides are `List[int64]` extent-domain values. Every `BuiltinDecl` must now classify its axis layout (including an explicit `NoAxes`), so adding a builtin without the classification is a compile error; a structural test pins the complete non-empty identity set. Generic inference screens every unambiguous registered axis, concat consumes the shared guard, and scatter_elements receives the same positive/negative enforcement. This metadata is the bridge to Phase 4's Table-A-derived acceptance, where the hand-routed checks are deleted |
| [#1131] | the Deep validator accepts a `lit` whose atom kind contradicts its declared prim family (an Int atom under `t-prim f32`); `check` scores 1.0 | ADDRESSED under [#731] by the general rule, not the reported pair. `spec/03` §6.4 and [04-LIT-1] define the full atom/primitive matrix and its sole explicit exception: an exact Int payload marked `literal_source: integer` when an integer spelling binds directly at a float dtype. The checker rejects every unmarked cross-family pair and every malformed marker after alias resolution. Deep/Surf producers emit the marked form, and both tensor/DAG and scalar host-eval constructors preserve [04-NUM-14]'s direct target-width rounding instead of converting through f64. Positive, malformed-marker, full cross-family, producer, exact-rounding, and §C4.4 score controls are pinned |
| [#1147] | `scatter_elements` has no inference arm, so string axes, f32 indices, and string operands all check clean | ADDRESSED by [#731] PP2. `scatter_elements` has a complete shape-computed rule for data/indices/updates/axis, including rank, update equality, precision, bounds, and non-axis extents. More importantly, every `BuiltinDecl` now requires an inference disposition beside registration: an exact checked route or a reasoned generic acceptance. Bidirectional manifest checks prevent an unowned declaration/route pair, and runtime execution witnesses prevent a checked application from silently falling through after its real semantic arm is removed. The four bad programs plus a positive control live in §C4.4, and the gather/scatter diagnostic helper names its real caller |
| [#1150] | C host lane: a checked cast on a host-built tensor emits NO conversion - the f32 buffer is reinterpreted at the target dtype (`host_emit.rs:2572`) | repaired structurally across [#730] LU6 and [#729]: C host emission maps its closed ABI vocabulary onto `CheckedCastPlan`, emits a typed tensor conversion for every active source/target pair, and permits identity only on the exact same-`Prim` diagonal. The `_ => input` fallback is deleted. The generated 9 x 9 x scalar/tensor-DAG/tensor-host matrix executes in eval and compiled C with exact observations, while the host-built mixed-offender reproducer now raises the selected branded trap instead of reinterpreting storage |
| [#1152] | checked cast: which trap KIND a multi-offender tensor reports diverges eval-vs-C, race-dependent under OpenMP on Linux while eval is domain-biased by its whole-buffer pre-pass | repaired through [#729], `Also part of #730`: tensor eval reduces explicit `IndexedTrapCandidate { flat_index, trap }` values instead of performing a domain-first pre-pass. Parallel C workers only reduce the minimum offending flat index, valid conversions remain parallel, and the selected element is reclassified after the region before its exact trap is rendered. Both Domain/Overflow orderings run through eval, C DAG, and C host at thread counts 1, 2, 4, and 8; the complete active-pair product and plan-level negative matrix remain in the Phase 3 oracle |

Also tracked to closure but already claimed (listed for completeness):
[#680]/[#684]/[#685]/[#686]/[#688]/[#704] -> [#729]; [#682]/[#692]/[#697]/[#698]/[#699]/[#705]/[#725] ->
[#730] (both halves: the builtin-coverage half at Phase 1's typed emitter,
the gate half at Phase 3's dedupe); [#709]/[#710]/[#833] -> [#731]
([#833] is the declaration-only `total_nodes == 0` score hole; §C4.4's
fitness-honesty corpus row at Phase 1); [#716]/[#723] -> [#732]; [#694] -> [#733]; [#711]/[#720] ->
[#729] Phase 2 / [#720]'s own note; [#712]/[#715]/[#724]/[#726] -> capability table
seed decisions; [#722] -> [#730] Phase 1 (loud) then [#729]/table (computed).
Wave 0's own discoveries, claimed at filing: [#744]/[#745] -> [#730]
(census rows 13/19, Phase 1 raises); [#748]/[#749] -> [#732] Phase 2
(the generated formatter kills both); [#753] -> [#729] (spec/04
[04-NUM-7] + the capability seed row); [#755]/[#756] -> [#731] (census
extensions; die at the Phase 1-2 sweep and witness migration).

Closed by the 2026-07-18/19 fix batch and removed from the ledger
above: [#747] (PR #767), [#721] (PR #768), [#750] (PR #774). Filed and
fixed inside the same batch, never ledgered: the [#735] sweep's
discoveries [#770] (PR #779), [#771] (PR #777, lane-agreement half;
the reject-diagnostic half stays [#731] Phase 1), and [#776] (PR #782),
plus the standalone [#706] (PR #769) and [#707] (PR #772).

## The deferred-evidence ledger

Claims resting on inspection or partial execution, per the audit's own
standard ("execute everything"). Each has a tracked task.

| item | current evidence | task |
|---|---|---|
| HIP runtime behavior ([#689]/[#690] symptoms at runtime) | **executed on gfx1151** (HIP 7.13, chelis 0.16.1, via `scripts/hip_test.py`): [#689] confirmed - `neg`/`sum` run an `_f32` kernel over the `CHELIS_I64` buffer and return garbage (e.g. `sum([10¹²,2·10¹²,3·10¹²,4·10¹²])` = 3567587328 vs 10¹³; `neg` also drops the upper lanes), while `add`/`mul` (correct `_i64` kernels) and f32 `neg` are exact. [#690] confirmed - HIP `trunc_div` by zero returns silently (exit 0, garbage) where the evaluator aborts branded. Repro archived at `docs/investigations/probes/hip_runtime/`; red-team re-ran the probes (PASS) and the runtime results are posted on [#689](https://github.com/Chelis-Lang/chelis/issues/689#issuecomment-5011825763), [#690](https://github.com/Chelis-Lang/chelis/issues/690#issuecomment-5011825818), and [#736](https://github.com/Chelis-Lang/chelis/issues/736#issuecomment-5011825877) | [#736]: probes executed + posted; both issues off the emission-only caveat |
| Metal runtime execution (typed kernels actually computing) | emission-locked (`metal_dtype_emission_and_bool_add.rs`); never executed | [#737]: a small driver harness (main.mm + chelis runtime link) on an arm64 Mac; promote the emission locks to run locks |
| [#688]'s opaque produced-value chokepoint (`flatten_field_value`) | the CLASS is executed (spurious fuzz-tier counterexample); the cited site is not | tracked on [#688] itself: needs `--features smt` + an `@opaque` int64-field type; exact repro sketch is in the issue comments |
| `with seed` / `with device` semantics (cross-lane seeding reproducibility) | the `with seed` half is SWEPT (2026-07-18 sweep + 2026-07-19 re-sweep on [#735]: eval vs compiled-C host lane, all probes f32-bit-identical post the [#770]/[#771]/[#776] fixes, independently re-run 253/253); [05-RNG-1] shipped (PR #781) | [#735]: remaining authoring - `with device` selection, vmap x seed, grad-through-seeded-draws numerics; GPU/kernel-lane RNG sweep is the [#736] follow-on; the unconditional cross-lane atom's blocker cleared 2026-07-20 ([#731] Phase 1 merged, PR #793) - re-sweep + atom (b) shipment unblocked |
| shell repos' compiled-lane numerics (school validates eval-only) | audit note, unexecuted downstream | [#738]: conform-contract amendment proposal - shells gain a compiled-lane numerical row |
| census rows 13-16 of [#730] (dead-by-probe placeholder sites) | probed dead or dead-by-inspection | re-verified mechanically at [#730] Phase 0; §C1.4 raise-or-prove applies regardless |

(Filed: [#735] effects, [#736] HIP runtime, [#737] Metal harness, [#738] shell
lanes.)

## Effects: split into decided-needs-authoring and genuinely open

Called out beyond the ledger because it is a spec-silence case (the
exact [#733] shape) and not merely missing evidence. Re-scoped per
Jeff's 2026-07 pass on [#735], which grounded it against the LaCaDiLE
formal development (the canonical calculus; our spec text lags it):

- **The effect DISCIPLINE is decided** - handler scoping is syntactic
  (the effect discharges for the body's scope only), nesting shadows
  innermost-first, `with seed`/`with device` compose via distinct
  labels, and grad rejects an unhandled `Random` effect - all backed by
  LaCaDiLE Theorem 3 (mechanized handler calculus). That half is
  spec-atom TRANSCRIPTION behind [#733], not design. Coordination note
  for [#733]: these atoms cite theorems in ANOTHER repo's formal
  development. Buoy's atom revision covers the normative spec text, not the
  external theorem, so the Chelis/Buoy shell integration must define a separate
  cross-repo theorem binding before these atoms land.
- **The RNG-reproducibility call is MADE and partly shipped** (Jeff,
  2026-07-18, the [decision memo on
  #735](https://github.com/Chelis-Lang/chelis/issues/735#issuecomment-5013424269)):
  the contract is same-seed same-stream, byte-identical, across all
  supported lanes, for any range - the measured two-tier behavior
  (exact for [0,1), 1-ULP elsewhere) was rejected as an ambiguous
  contract and fixed as an implementation defect instead. Status: the
  formerly never-swept cross-lane gap has now been swept twice by
  execution (2026-07-18 sweep, 2026-07-19 re-sweep; all probes
  f32-bit-identical after the [#770]/[#771]/[#776] fixes, independently
  re-run). The per-lane determinism-at-rest atom **[05-RNG-1]** shipped
  (PR #781, spec/05 §2.6) - grounded in the sweep evidence, deliberately
  NOT in LaCaDiLE, whose `Random` effect is abstract and says nothing
  about stream values, so the cross-repo citation question does not
  arise for it. The unconditional CROSS-LANE atom's sole blocker CLEARED
  2026-07-20: [#731] Phase 1 merged (PR #793) with the
  out-of-range/unsuffixed seed-literal diagnostic plus the
  negative-int64-literal rejection its red team forced - the
  confirmation re-sweep and the atom (b) shipment are now fully
  unblocked (noted on [#735]; the `.dp`-only lowering-side
  negative-seed fallback is checker-unreachable and tracked at
  [#794]).
- **Genuinely open, still owed**: what `with device` selects
  operationally, vmap x seed (not in LaCaDiLE at all; its own sweep),
  the numerics of grad THROUGH seeded draws (LaCaDiLE yields only
  seed-before-grad), and the GPU/kernel RNG lanes (the [#736]
  follow-on). All stay [#735] authoring.

[#731] makes the BODIES type-checked (with the T-Handle shape as the
formal target, per its §C1.5) and [#730] makes unknown KINDS loud;
[#735]'s remaining output is atoms plus the parked sweeps, not code.

[#680]: https://github.com/Chelis-Lang/chelis/issues/680
[#681]: https://github.com/Chelis-Lang/chelis/issues/681
[#682]: https://github.com/Chelis-Lang/chelis/issues/682
[#683]: https://github.com/Chelis-Lang/chelis/issues/683
[#684]: https://github.com/Chelis-Lang/chelis/issues/684
[#685]: https://github.com/Chelis-Lang/chelis/issues/685
[#686]: https://github.com/Chelis-Lang/chelis/issues/686
[#257]: https://github.com/Chelis-Lang/chelis/issues/257
[#409]: https://github.com/Chelis-Lang/chelis/issues/409
[#687]: https://github.com/Chelis-Lang/chelis/issues/687
[#688]: https://github.com/Chelis-Lang/chelis/issues/688
[#689]: https://github.com/Chelis-Lang/chelis/issues/689
[#690]: https://github.com/Chelis-Lang/chelis/issues/690
[#691]: https://github.com/Chelis-Lang/chelis/issues/691
[#692]: https://github.com/Chelis-Lang/chelis/issues/692
[#693]: https://github.com/Chelis-Lang/chelis/issues/693
[#694]: https://github.com/Chelis-Lang/chelis/issues/694
[#695]: https://github.com/Chelis-Lang/chelis/issues/695
[#696]: https://github.com/Chelis-Lang/chelis/pull/696
[#697]: https://github.com/Chelis-Lang/chelis/issues/697
[#698]: https://github.com/Chelis-Lang/chelis/issues/698
[#699]: https://github.com/Chelis-Lang/chelis/issues/699
[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#704]: https://github.com/Chelis-Lang/chelis/issues/704
[#705]: https://github.com/Chelis-Lang/chelis/issues/705
[#706]: https://github.com/Chelis-Lang/chelis/issues/706
[#707]: https://github.com/Chelis-Lang/chelis/issues/707
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#710]: https://github.com/Chelis-Lang/chelis/issues/710
[#711]: https://github.com/Chelis-Lang/chelis/issues/711
[#712]: https://github.com/Chelis-Lang/chelis/issues/712
[#713]: https://github.com/Chelis-Lang/chelis/issues/713
[#715]: https://github.com/Chelis-Lang/chelis/issues/715
[#716]: https://github.com/Chelis-Lang/chelis/issues/716
[#719]: https://github.com/Chelis-Lang/chelis/issues/719
[#720]: https://github.com/Chelis-Lang/chelis/issues/720
[#721]: https://github.com/Chelis-Lang/chelis/issues/721
[#722]: https://github.com/Chelis-Lang/chelis/issues/722
[#723]: https://github.com/Chelis-Lang/chelis/issues/723
[#724]: https://github.com/Chelis-Lang/chelis/issues/724
[#725]: https://github.com/Chelis-Lang/chelis/issues/725
[#726]: https://github.com/Chelis-Lang/chelis/issues/726
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#708]: https://github.com/Chelis-Lang/chelis/issues/708
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
[#733]: https://github.com/Chelis-Lang/chelis/issues/733
[#734]: https://github.com/Chelis-Lang/chelis/issues/734
[#735]: https://github.com/Chelis-Lang/chelis/issues/735
[#736]: https://github.com/Chelis-Lang/chelis/issues/736
[#737]: https://github.com/Chelis-Lang/chelis/issues/737
[#738]: https://github.com/Chelis-Lang/chelis/issues/738
[#740]: https://github.com/Chelis-Lang/chelis/issues/740
[#744]: https://github.com/Chelis-Lang/chelis/issues/744
[#745]: https://github.com/Chelis-Lang/chelis/issues/745
[#747]: https://github.com/Chelis-Lang/chelis/issues/747
[#748]: https://github.com/Chelis-Lang/chelis/issues/748
[#749]: https://github.com/Chelis-Lang/chelis/issues/749
[#750]: https://github.com/Chelis-Lang/chelis/issues/750
[#751]: https://github.com/Chelis-Lang/chelis/issues/751
[#753]: https://github.com/Chelis-Lang/chelis/issues/753
[#754]: https://github.com/Chelis-Lang/chelis/issues/754
[#755]: https://github.com/Chelis-Lang/chelis/issues/755
[#756]: https://github.com/Chelis-Lang/chelis/issues/756
[#761]: https://github.com/Chelis-Lang/chelis/issues/761
[#763]: https://github.com/Chelis-Lang/chelis/issues/763
[#770]: https://github.com/Chelis-Lang/chelis/issues/770
[#771]: https://github.com/Chelis-Lang/chelis/issues/771
[#773]: https://github.com/Chelis-Lang/chelis/issues/773
[#775]: https://github.com/Chelis-Lang/chelis/issues/775
[#776]: https://github.com/Chelis-Lang/chelis/issues/776
[#780]: https://github.com/Chelis-Lang/chelis/issues/780
[#783]: https://github.com/Chelis-Lang/chelis/issues/783
[#794]: https://github.com/Chelis-Lang/chelis/issues/794
[#795]: https://github.com/Chelis-Lang/chelis/issues/795
[#796]: https://github.com/Chelis-Lang/chelis/issues/796
[#799]: https://github.com/Chelis-Lang/chelis/pull/799
[#815]: https://github.com/Chelis-Lang/chelis/pull/815
[#833]: https://github.com/Chelis-Lang/chelis/issues/833
[#840]: https://github.com/Chelis-Lang/chelis/issues/840
[#847]: https://github.com/Chelis-Lang/chelis/issues/847
[#850]: https://github.com/Chelis-Lang/chelis/issues/850
[#851]: https://github.com/Chelis-Lang/chelis/issues/851
[#888]: https://github.com/Chelis-Lang/chelis/issues/888
[#889]: https://github.com/Chelis-Lang/chelis/issues/889
[#893]: https://github.com/Chelis-Lang/chelis/issues/893
[#1003]: https://github.com/Chelis-Lang/chelis/pull/1003
[#912]: https://github.com/Chelis-Lang/chelis/issues/912
[#990]: https://github.com/Chelis-Lang/chelis/issues/990
[#997]: https://github.com/Chelis-Lang/chelis/issues/997
[#1023]: https://github.com/Chelis-Lang/chelis/issues/1023
[#1059]: https://github.com/Chelis-Lang/chelis/issues/1059
[#788]: https://github.com/Chelis-Lang/chelis/issues/788
[#814]: https://github.com/Chelis-Lang/chelis/issues/814
[#820]: https://github.com/Chelis-Lang/chelis/issues/820
[#825]: https://github.com/Chelis-Lang/chelis/issues/825
[#845]: https://github.com/Chelis-Lang/chelis/issues/845
[#862]: https://github.com/Chelis-Lang/chelis/issues/862
[#866]: https://github.com/Chelis-Lang/chelis/issues/866
[#867]: https://github.com/Chelis-Lang/chelis/issues/867
[#868]: https://github.com/Chelis-Lang/chelis/issues/868
[#879]: https://github.com/Chelis-Lang/chelis/issues/879
[#883]: https://github.com/Chelis-Lang/chelis/issues/883
[#885]: https://github.com/Chelis-Lang/chelis/issues/885
[#886]: https://github.com/Chelis-Lang/chelis/issues/886
[#887]: https://github.com/Chelis-Lang/chelis/issues/887
[#892]: https://github.com/Chelis-Lang/chelis/issues/892
[#899]: https://github.com/Chelis-Lang/chelis/issues/899
[#908]: https://github.com/Chelis-Lang/chelis/issues/908
[#909]: https://github.com/Chelis-Lang/chelis/issues/909
[#894]: https://github.com/Chelis-Lang/chelis/pull/894
[#174]: https://github.com/Chelis-Lang/chelis/issues/174
[#717]: https://github.com/Chelis-Lang/chelis/issues/717
[#856]: https://github.com/Chelis-Lang/chelis/issues/856
[#864]: https://github.com/Chelis-Lang/chelis/issues/864
[#1078]: https://github.com/Chelis-Lang/chelis/issues/1078
[#1110]: https://github.com/Chelis-Lang/chelis/issues/1110
[#870]: https://github.com/Chelis-Lang/chelis/issues/870
[#872]: https://github.com/Chelis-Lang/chelis/issues/872
[#878]: https://github.com/Chelis-Lang/chelis/issues/878
[#901]: https://github.com/Chelis-Lang/chelis/issues/901
[#906]: https://github.com/Chelis-Lang/chelis/issues/906
[#937]: https://github.com/Chelis-Lang/chelis/issues/937
[#942]: https://github.com/Chelis-Lang/chelis/issues/942
[#955]: https://github.com/Chelis-Lang/chelis/issues/955
[#980]: https://github.com/Chelis-Lang/chelis/issues/980
[#1009]: https://github.com/Chelis-Lang/chelis/issues/1009
[#1113]: https://github.com/Chelis-Lang/chelis/issues/1113
[#1131]: https://github.com/Chelis-Lang/chelis/issues/1131
[#1132]: https://github.com/Chelis-Lang/chelis/issues/1132
[#1136]: https://github.com/Chelis-Lang/chelis/pull/1136
[#1147]: https://github.com/Chelis-Lang/chelis/issues/1147
[#1150]: https://github.com/Chelis-Lang/chelis/issues/1150
[#1152]: https://github.com/Chelis-Lang/chelis/issues/1152
[#1170]: https://github.com/Chelis-Lang/chelis/issues/1170
[#1192]: https://github.com/Chelis-Lang/chelis/issues/1192
[#874]: https://github.com/Chelis-Lang/chelis/issues/874
[#916]: https://github.com/Chelis-Lang/chelis/issues/916
[#1024]: https://github.com/Chelis-Lang/chelis/issues/1024
[#1112]: https://github.com/Chelis-Lang/chelis/issues/1112
[#1172]: https://github.com/Chelis-Lang/chelis/issues/1172
