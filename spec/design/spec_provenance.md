# Spec Provenance: Buoy-backed authority, evidence, and change impact

**Status:** Phase 0 OpenSpec adoption contract + future Buoy-backed tooling
design. This PR defines but does not activate Phase 0. Once Phase 0's
configuration, instructions, review routing, validation suite, and PR gate
land, OpenSpec becomes the required planning and agent-communication workflow.
The standalone repository-independent [Buoy] workspace currently reports
version 0.2.0 and its initial acceptance oracle has been exercised, but no
Chelis integration-tool or Buoy revision is pinned yet. Buoy production
hardening and the future integration tool remain active work; neither is a
blocking Chelis dependency today.

Tracking issue: [#733].

**Owning specs:** every file under `spec/`; the capability-table,
checker-totality, unsupported-behavior, observation, and dtype-semantics
designs; the repository's Spec-First Development and Negative Test Parity
contracts.

**Class fixed:** specification silence, stale claims, incomplete decision
surfaces, and assurance laundering. The system must expose both “no decision
was authored for this cell” and “this evidence still points at an obsolete
revision” without treating an annotation, waiver, review, or discovered test
as proof that behavior is correct.

## Decision

Chelis will not implement a second provenance graph, revision algorithm,
coverage engine, change-impact checker, or direct Buoy adapter inside
`chelis-lint` or another Chelis crate.

A future OpenSpec-type provenance tool, distinct from the current OpenSpec
planning workflow, will integrate Buoy's repository-independent Rust APIs.
Chelis will pin and invoke that tool rather than depend on Buoy directly:

- Buoy owns repository-independent graph values, deterministic identity,
  lifecycle, freshness, coverage-policy evaluation, impact closure, and
  conservative assurance aggregation.
- The future tool owns repository snapshotting, syntax-aware adapters,
  configuration loading, oracle orchestration, and stable report surfaces over
  Buoy.
- Chelis owns normative text, stable ID namespaces, complete capability
  domains, host payload schemas, evidence scopes, approval authorities,
  policies, debt, waivers, and the checked-in configuration passed to the tool.
- `buoy-core` never depends on Chelis, and the future tool must express
  Chelis-specific semantics through versioned configuration and payloads rather
  than hard-coded repository knowledge.
- Chelis lint, CI, and any future editor surface consume the same pinned tool
  reports; they do not independently reimplement provenance.

This supersedes the 2026-07-20 plan to leave atom/hash enforcement as a future
`chelis-lint` growth tier. The concrete standalone Buoy implementation now
exists, so the remaining problem is a stable OpenSpec-type integration tool,
readiness, and Chelis-owned configuration—not invention of another provenance
engine inside Chelis.

## Disposition of the previous #733 contracts

This table makes the supersession explicit. A retained or replaced contract is
not implemented merely because it appears here; the named phase owns its exact
fixtures, format, and oracle. No previous contract survives by implication.

| Previous #733 contract | Disposition in this design | Owning phase or section |
|---|---|---|
| PR trailers, docs-only detection, `spec-exempt`, `CODEOWNERS`, and the PR template | **Retained as provider-level review routing.** Exact parsing, authorization, reporting, and planted cases remain Phase 0 deliverables. Provider state never becomes canonical authority. | Phase 0; §C7 |
| Stable Chelis atom IDs | **Retained.** Existing IDs remain allocated; Phase 1 freezes the registry, namespaces, and migration mapping. | Phase 1; §C1 |
| Markdown blockquote atoms | **Retained only as migration input.** Existing blocks remain normative Chelis text until selected; final Buoy authority uses a parser-backed, fixture-proven form. | Phase 1; §C1 |
| EARS-shaped normative statements | **Retained as authoring guidance, not parser authority.** Normative meaning lives in the atom statement and its approved scope/kind. | Phase 1; §C1 |
| Eight-hex `xxh3-64` revisions | **Superseded.** Identity uses the complete canonical XXH3-128 value under the active versioned Buoy envelope. | §C2 |
| `@spec ID rev HASH` comment annotations and a `chelis-lint` provenance rule family | **Superseded.** The future OpenSpec-type tool uses syntax-aware adapters backed by Buoy to attach typed metadata to eligible items; Chelis does not build a second provenance engine. | Phase 1; §C3 |
| One in-repo extractor plus `--spec-report` | **Replaced by the pinned future tool over shared Buoy semantics.** Phase 1 freezes the external command/configuration/report boundary without duplicating authority or freshness logic in Chelis. | Phase 1; §C6 |
| Spec-edit freshness by mechanically comparing carrier revs | **Strengthened.** Current revisions are necessary, and Phase 2 additionally requires complete provider-neutral transitive impact dispositions. | Phase 2; §C2 and §C6 |
| Green test or issue-linked ignored test as the two coverage states | **Superseded.** Registration, freshness, selection, five-state execution verdict, debt, waiver, and assurance class remain independent facts. | §C3 and §C4 |
| Blocking coverage manifest and advisory debt reports | **Replaced by versioned coverage policies, adoption ratchets, repository-owned debt, and repository-owned waivers.** Phase 3 freezes initial policy IDs, selectors, required roles, reports, and negative controls. | Phase 3; §C4 |
| Capability-row, Deep-tag, tolerance-row, and diagnostic citations | **Retained and generalized as governed structural surfaces.** Selected members bind one current controlling atom revision; intentional duplicates also require derivations. | Phase 3; §C5 |
| OpenSpec proving inadequate as the trigger for provenance work | **Superseded.** OpenSpec becomes required for planning and agent communication when Phase 0's executable adoption surface lands; a nonblocking pilot through the future Buoy-backed tool may begin once Phase 1 fixtures and pins exist. Blocking still waits for the readiness rule. | Phase 0; Phases 1 and 3 |
| Executable atoms through `chelis prove` | **Retained as optional stronger evidence, not a replacement for ordinary carriers.** Exact properties, models, proofs, assumptions, and exclusions are bound only when justified. | Phase 4 |

## Phase 0 OpenSpec activation contract

Once Phase 0's executable adoption surface is green, OpenSpec becomes the
required planning and agent-communication workflow for Chelis changes. This
requirement continues after Buoy-backed blocking adoption: provenance
enforcement does not replace change planning. Every agent-authored feature or
behavior change SHALL:

1. create or update an OpenSpec change before implementation;
2. identify the affected specification requirements;
3. record positive and negative scenarios before production code;
4. keep proposal, design, specification deltas, and tasks synchronized; and
5. link the active OpenSpec change from its implementation pull request.

Implementation may begin only after the OpenSpec proposal and requirement
deltas have entered the human review queue. Human acceptance of an OpenSpec
plan is review-only evidence: it authorizes the planned change but does not
prove implementation correctness.

An implementation agent SHALL name the active OpenSpec change and the specific
requirement or design section it implements. “Follow the spec” without an
addressable requirement is not a valid implementation claim.

## OpenSpec boundary

OpenSpec remains the proposal and review workflow for changes. It is not:

- the runtime authority format;
- the atom lifecycle engine;
- the freshness or coverage oracle;
- an input to canonical graph identity; or
- a substitute for repository-owned approval and change records.

The staged discipline is therefore **OpenSpec for planning, then a future
OpenSpec-type tool backed by Buoy for enforcement**. Current OpenSpec
validation proves only that planning artifacts are internally valid. The
future tool may consume explicit repository-owned links to planning artifacts,
but current OpenSpec state, lifecycle, and provider metadata never become
canonical Buoy inputs.

## Architecture

```text
OpenSpec proposal
       |
       | planning and review only
       v
Chelis repository-owned authorities and records
       |
       | saved repository + pinned configuration
       v
Future OpenSpec-type provenance tool
       |
       | repository-independent Rust APIs
       v
Buoy static graph and change-impact engine
       |
       +--> advisory/blocking policy report
       +--> deterministic change report
       +--> effectful oracle execution report
       |
       v
Chelis gate and reviewer decision
```

Provider state sits outside this graph. GitHub may help materialize candidate
records, but labels, users, timestamps, issue state, and live review APIs do
not determine canonical authority or graph identity.

---

# Part I — normative integration contracts

## C1. Atom authority and lifecycle

A live atom has:

- one stable Chelis-owned ID;
- exactly one explicit authority;
- one lifecycle state;
- one kind and nonempty scope;
- one self-contained normative statement; and
- repository-owned approval when ratified.

The existing IDs such as `04-NUM-1`, `04-TOT-1`, `05-UNS-1`, `05-OBS-1`,
and `05-RNG-1` remain allocated and are never silently renamed.

Current blockquote atoms remain normative under the existing Chelis spec
contract until migrated, but they are not silently treated as Buoy
authorities. A selected section enters Buoy policy only after its authorities
have been converted or handled by a fixture-proven adapter in the pinned future
tool.

The preferred authority form is Buoy's parser-backed fenced object:

````markdown
```spec-provenance-atom-v1
id = "04-NUM-3"
state = "ratified"
kind = "behavioral"
scope = ["chelis.numeric.integer-overflow"]
statement = """
Integer op results that are not exactly representable in their declared
width SHALL trap; ordinary arithmetic SHALL NOT wrap, saturate, or silently
widen.
"""

[approval]
authority = "chelis-spec-maintainers"
record = "spec/reviews/04-NUM-3.md"
digest = "xxh3-128:<full-review-record-digest>"
```
````

`spec-provenance-atom-v1` identifies the Markdown syntax contract. Active
semantic identity uses the separately versioned `spec-provenance/v2`
profile.

Allowed lifecycle transitions are closed. Withdrawn and superseded IDs remain
permanently reserved by tombstones; deletion does not make an ID reusable.

## C2. Semantic revisions

A stable atom ID denotes identity. A semantic revision is the full canonical
XXH3-128 digest of the atom's normative kind, scope, and statement under the
active versioned envelope.

The revision excludes source location, formatting trivia, lifecycle state,
approval, and the stable ID itself. Moving unchanged authority preserves the
atom revision while changing repository source-map and complete-graph
identity.

The former eight-hex `xxh3-64` proposal is superseded. Truncated display
hashes never establish identity.

XXH3 is non-cryptographic. It answers “did these canonical semantic bytes
change?” It does not establish authenticity, authorization, natural-language
truth, collision impossibility, or supply-chain integrity.

## C3. Evidence and navigation

Evidence metadata must be attached to an eligible executable item by a
syntax-aware adapter owned and versioned by the future integration tool.
Comments, strings, prose mentions, file-level annotations, and nearby atom IDs
do not create carriers.

Every carrier declares an explicit role:

- positive;
- negative;
- property;
- exhaustive-model;
- formal-property; or
- structural.

The role is never inferred from a test name. Coverage policy states which
roles are required, preserving the repository's positive/negative parity
contract mechanically.

A carrier binds its exact atom revision, parsed item, oracle, configuration,
corpus, claim scope, assumptions, and exclusions.

Registration means only `linked`. Execution separately reports `not-run`,
`pass`, `fail`, `error`, or `timeout`. A discovered or fresh test is not
`tested` until every applicable required carrier is selected and passing
within its declared scope.

Implementation links remain navigational. They support ownership and impact
analysis but never satisfy behavioral coverage.

## C4. Coverage policy, debt, and waivers

Coverage is controlled by versioned Chelis-owned policies. A policy declares:

- a finite atom and decision-surface scope;
- required evidence roles;
- advisory or blocking enforcement;
- an adoption-ratchet identity; and
- any authority or derivation requirements.

Adoption starts advisory. Promotion to blocking retains the same ratchet and
occurs only after every selected obligation is visible and current.

Debt and waivers are repository-owned, digest-bound objects. An issue link may
be informational, but issue state is not canonical. A waiver may permit a
process gate to continue; it does not satisfy, freshen, execute, test, prove,
or structurally enforce the waived obligation.

Known failures retain their actual `fail` or `not-run` state. They are not
converted into passing coverage because an ignored test or issue exists.

## C5. Governed structural surfaces

1. **Capability rows:** every member of the complete operation × surface ×
   dtype × lane domain remains present, including prohibited, deferred, and
   not-applicable members. No adapter predicate may filter difficult cells
   before completeness checking. Every disposition cites one controlling
   current atom revision.
2. **Deep tags:** every member of the closed Deep vocabulary receives exactly
   one checker disposition and controlling atom revision.
3. **Tolerance rows:** every governed cross-lane tolerance entry is explicit,
   authority-bound, and independently covered. Missing rows are not inferred
   from current implementation behavior.
4. **Generated or mirrored artifacts:** intentional copies—including spec
   tables, generated capability data, duplicated gates, and checked-in
   reports—carry a derivation binding source, target, tool, configuration,
   relation, consistency oracle, and exclusions.

A disposition proves that a decision exists. It does not prove that the
implementation honors the decision.

## C6. Static, change, and execution checks

The future tool's Buoy-backed integration preserves three distinct surfaces:

- **static:** build and validate the current saved-repository graph;
- **change:** compare explicit base and head graphs and require complete impact
  dispositions; and
- **execute:** run selected effectful oracles and normalize their results.

Static graph identity never changes because an oracle runs.

A changed atom or other bound object produces a transitive impact set over
carriers, implementation links, surface dispositions, policies, derivations,
assumptions, properties, models, and proofs.

Every required impact receives exactly one disposition:

- `reaffirmed`;
- `revised`;
- `retired`; or
- `waived`.

Reaffirmation binds exact base and head values and records a deliberate
reread. It is review-only evidence. It cannot make stale evidence fresh, and
wildcard or bulk “reaffirm everything” records are invalid.

## C7. Pull-request and provider integration

`CODEOWNERS` on `spec/**`, PR-template citations, and GitHub labels remain
useful review-routing controls, but they are provider-level process evidence.

`Spec-Atoms:` and `Spec-Design:` trailers may seed a candidate neutral change
manifest. They do not replace that committed manifest once change checking is
blocking.

Likewise, a `spec-exempt` label may support the pre-integration process gate,
but blocking provenance adoption requires a repository-owned debt or waiver
record. Provider state cannot be the only durable exemption ledger.

---

# Part II — staged adoption

## Phase 0 — activate OpenSpec planning and review routing

Deliver the repository's OpenSpec configuration, change templates and agent
instructions, strict validation, human review queue, PR template, `spec/**`
review routing, citation checks, and temporary-exemption path.

This PR defines the phase; the OpenSpec requirement becomes active only when
the phase oracle is green. The phase makes no Buoy assurance claim and does
not make OpenSpec or provider metadata canonical product authority.

**Oracle:** the named `openspec_adoption` suite plus the PR-gate suite prove
that valid changes pass strict OpenSpec validation; missing requirement
deltas, positive or negative scenarios, and implementation-PR change links
fail; and cited, docs-only, and explicitly exempted PR cases follow their
defined paths.

## Phase 1 — advisory Buoy-backed tool pilot

Select and pin one exact revision of the future OpenSpec-type provenance tool;
that tool revision must pin one exact Buoy revision and compatibility profile.
Before production tool configuration or adapter code, freeze positive and
negative fixtures for:

- existing atom IDs and authority migration;
- malformed and duplicate authorities;
- stale carrier revisions;
- unattached carrier metadata;
- provider/root/order independence; and
- an expected-authority selector that produces no authorities.

Adopt the pinned tool in advisory mode with Chelis-owned configuration and a
small initial scope. Do not add a direct Buoy dependency, provenance parser, or
parallel graph implementation to the Chelis workspace. No existing Chelis
command becomes blocking.

**Oracle:** the named `chelis_provenance_advisory` integration suite invokes
the pinned tool, proves byte-identical reports under separate roots, rejects
planted malformed/stale inputs, verifies the tool and Buoy pins, and leaves the
existing Chelis gate unchanged.

## Phase 2 — provider-neutral change impact

Through the pinned tool, materialize repository-owned approvals, review
records, debt, waivers, reaffirmations, and neutral base-to-head manifests.
Enable advisory change reports.

**Oracle:** the named `chelis_provenance_change_impact` suite proves that a
planted atom edit enumerates every transitive dependent claim; it rejects a
missing, duplicate, stale, or inapplicable disposition and accepts one exact
complete manifest.

## Phase 3 — first blocking ratchet

Select the first finite surfaces—initially the atomized spec/04 and spec/05
sections plus their corresponding capability/tag/tolerance domains. Require
positive and negative roles where applicable.

Integrate the ordinary required oracles through the pinned tool in this phase:
selection, effectful execution, five-state result normalization, and exact
binding back to carriers. This is the minimum execution surface needed for a
blocking coverage claim; stronger model/property/proof evidence remains Phase
4.

Promotion requires:

- the pinned Buoy acceptance oracle green at the exact revision selected by the
  pinned future tool;
- all future-tool adapter and Chelis-configuration positive and negative
  fixtures green;
- no missing or filtered selected surface member;
- every selected carrier current;
- every required oracle selected and passing; and
- policy weakening and waiver negative controls green.

**Oracle:** the named `chelis_provenance_blocking_policy` suite proves that a
planted uncovered atom, missing surface member, stale carrier, and silent
policy weakening each fail while the complete selected scope passes.

## Phase 4 — stronger evidence

Add bounded models, sampled properties, formal properties, and proof records
only where their exact scope justifies the maintenance cost. Reuse Phase 3's
execution and result-normalization path rather than creating another identity
or report surface. Preserve the boundary that source-level proof does not prove
rustc, LLVM, emitted C/HIP/Metal, or machine code.

**Oracle:** the named `chelis_provenance_stronger_evidence` suite proves exact
model/property/proof scopes, assumptions, exclusions, negative controls, and
current bindings while leaving static identity independent of execution.

## Readiness rule

An advisory pilot may begin before Buoy's production-hardening program is
complete because it cannot block Chelis or replace existing acceptance
evidence.

Blocking adoption may not begin merely because the initial Buoy repository
gate is green. It requires a reviewed, pinned future-tool revision, that tool's
exact pinned Buoy revision with relevant hardening obligations complete, plus
the Chelis-specific Phase 3 oracle.

This design records the target; it does not claim that condition is satisfied
today.

## Non-goals

- Reimplementing Buoy semantics or a direct Buoy adapter in any Chelis crate.
- Making OpenSpec or GitHub state canonical.
- Treating annotations, implementation links, waivers, review records, test
  discovery, or passing siblings as proof.
- Atomizing all existing prose in one pass.
- Inferring a decision surface from whatever atoms or tests happen to exist.
- Claiming that provenance proves natural-language truth or emitted backend
  correctness.

## Issue map

| phase | what becomes impossible |
|---|---|
| 0 | a PR with no visible spec story; unsignposted spec edits |
| 1 | silently malformed authorities or stale registrations inside the advisory scope |
| 2 | an atom change whose transitive dependents receive no explicit disposition |
| 3 | missing selected surface members, uncovered required roles, and silent policy weakening |
| 4 | model, property, or proof evidence being reported outside its exact bindings and scope |

## Open decisions

The Phase 1 adoption proposal must freeze:

1. the exact future-tool revision, its exact Buoy revision, and compatibility policy;
2. the external command, configuration, and report-schema boundary Chelis consumes;
3. the first advisory authority selectors and finite decision surfaces;
4. the existing blockquote-atom migration strategy; and
5. the initial derivation inventory for generated and mirrored artifacts.

[Buoy]: https://github.com/Chelis-Lang/buoy
[#694]: https://github.com/Chelis-Lang/chelis/issues/694
[#733]: https://github.com/Chelis-Lang/chelis/issues/733
