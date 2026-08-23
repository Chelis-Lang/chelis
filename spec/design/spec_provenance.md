# Spec Provenance: Buoy-backed authority, evidence, and change impact

**Status:** Phase 0 OpenSpec adoption contract + direct Buoy shell-side
integration design. This PR defines but does not activate Phase 0. Once Phase
0's configuration, instructions, review routing, validation suite, and PR gate
land, OpenSpec becomes the required planning and agent-communication workflow.
At the reviewed [Buoy snapshot `61b2c25a`][Buoy-61b2c25a] (workspace version
0.2.0), Buoy itself exposes the `static` / `change` / `execute` CLI, the
versioned `buoy.adapter-sdk/v1` shell contract, deterministic reports, and the
repository-independent `buoy-core`. That snapshot is design evidence, not the
Chelis adoption pin: no Buoy revision, Chelis adapter, configuration, or command
integration is pinned here. Buoy production hardening and the parsed-item
protocol needed for Chelis host syntax remain incomplete; Buoy is not a
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

Chelis will consume Buoy directly through a one-way shell-side integration;
it will not implement a second provenance graph, revision algorithm, coverage
engine, change-impact checker, or report protocol inside `chelis-lint` or a
compiler-semantic crate:

- `buoy-core` owns repository-independent graph values, deterministic identity,
  lifecycle, freshness, coverage-policy evaluation, impact closure, and
  conservative assurance aggregation. It remains `no_std` and never depends on
  Chelis, parsers, filesystems, processes, providers, or clocks.
- The root `buoy` crate owns the generic imperative shell: saved-repository
  snapshotting, configuration, the built-in Markdown/Rust adapters, the public
  `buoy.adapter-sdk/v1` registry, oracle orchestration, CLI exit classes, and
  stable reports.
- A Chelis adapter and optional Chelis command facade live in Chelis or another
  shell-side integration crate and depend inward on those published Buoy
  interfaces. Buoy does not import Chelis semantics.
- Chelis owns normative text, stable ID namespaces, complete capability
  domains, host payload schemas, evidence scopes, approval authorities,
  policies, debt, waivers, and checked-in configuration.
- Chelis lint, CI, and any future editor surface consume the same saved-snapshot
  semantic library and reports; they do not independently reimplement
  provenance, and unsaved/editor/provider state cannot satisfy CI authority.

The built-in `markdown/v1` and `rust/v1` adapters may cover selected existing
Markdown and Rust sources. A future Chelis-language adapter must use the public
shell SDK and cannot label `.ch`/`.dp` syntax as the frozen `MarkdownFence` or
`RustItem` kinds or hide it in an opaque payload. Its implementation is blocked
until a separately approved canonical schema version defines the host-item
identity, historical reader, migration, fixtures, and compatibility mapping.

This supersedes both the 2026-07-20 `chelis-lint` growth-tier plan and this PR's
earlier extra-tool layer. Buoy is the tool and shell contract; the remaining
work is to finish the relevant Buoy hardening, resolve the host-item protocol,
and implement Chelis-owned configuration and the one-way shell adapter without
inventing another provenance engine.

## Disposition of the previous #733 contracts

This table makes the supersession explicit. A retained or replaced contract is
not implemented merely because it appears here; the named phase owns its exact
fixtures, format, and oracle. No previous contract survives by implication.

| Previous #733 contract | Disposition in this design | Owning phase or section |
|---|---|---|
| PR trailers, nonnormative docs-only detection, `spec-exempt`, `CODEOWNERS`, and the PR template | **Retained as provider-level review routing, with the docs-only bypass narrowed.** A PR touching `spec/**` always follows the cited OpenSpec path even when every changed file is Markdown; only changes outside `spec/**` may qualify as nonnormative docs-only. Exact parsing, authorization, reporting, and planted cases remain Phase 0 deliverables. Provider state never becomes canonical authority. | Phase 0; §C7 |
| Stable Chelis atom IDs | **Retained.** Existing IDs remain allocated; Phase 1 freezes the registry, namespaces, and migration mapping. | Phase 1; §C1 |
| Markdown blockquote atoms | **Retained only as migration input.** Existing blocks remain normative Chelis text until selected; final Buoy authority uses a parser-backed, fixture-proven form. | Phase 1; §C1 |
| EARS-shaped normative statements | **Retained as authoring guidance, not parser authority.** Normative meaning lives in the atom statement and its approved scope/kind. | Phase 1; §C1 |
| Eight-hex `xxh3-64` revisions | **Superseded.** Identity uses the complete canonical XXH3-128 value under the active versioned Buoy envelope. | §C2 |
| `@spec ID rev HASH` comment annotations and a `chelis-lint` provenance rule family | **Superseded.** The pinned Buoy shell and its syntax-aware built-in or Chelis adapters attach typed metadata to eligible items; Chelis does not build a second provenance engine. | Phase 1; §C3 |
| One in-repo extractor plus `--spec-report` | **Replaced by the pinned Buoy CLI and shell-side integration.** Phase 1 freezes Buoy's command/configuration/report and adapter-SDK boundaries without duplicating authority or freshness logic in Chelis. | Phase 1; §C6 |
| Spec-edit freshness by mechanically comparing carrier revs | **Strengthened.** Current revisions are necessary, and Phase 2 additionally requires complete provider-neutral transitive impact dispositions. | Phase 2; §C2 and §C6 |
| Green test or issue-linked ignored test as the two coverage states | **Superseded.** Registration, freshness, selection, five-state execution verdict, debt, waiver, and assurance class remain independent facts. | §C3 and §C4 |
| Blocking coverage manifest and advisory debt reports | **Replaced by versioned coverage policies, adoption ratchets, repository-owned debt, and repository-owned waivers.** Phase 3 freezes initial policy IDs, selectors, required roles, reports, and negative controls. | Phase 3; §C4 |
| Capability-row, Deep-tag, tolerance-row, and diagnostic citations | **Retained and generalized as governed structural surfaces.** Numeric Table A binds each complete semantic cell to its controlling atom. Table B preserves that binding while adding a typed kernel, implementation issue, or rejected-by-design authority per backend. Host constructors, sibling callable registries, external target dispositions, and derived exported-stdlib dependencies remain separate governed domains. Intentional duplicates also require derivations. | Phase 3; §C5 |
| OpenSpec proving inadequate as the trigger for provenance work | **Superseded.** OpenSpec becomes required for planning and agent communication when Phase 0's executable adoption surface lands; a nonblocking pilot through pinned Buoy shell interfaces may begin once Phase 1 fixtures, boundaries, and pins exist. Blocking still waits for the readiness rule. | Phase 0; Phases 1 and 3 |
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

The staged discipline is therefore **OpenSpec for planning, then pinned Buoy
shell interfaces plus a Chelis-owned adapter for enforcement**. Current
OpenSpec validation proves only that planning artifacts are internally valid.
The Chelis integration may consume explicit repository-owned links to planning
artifacts, but current OpenSpec state, lifecycle, and provider metadata never
become canonical Buoy inputs.

## Architecture

```text
OpenSpec proposal
       |
       | planning and review only
       v
Chelis repository-owned authorities and records
       |
       | saved bytes + explicit configuration
       v
Chelis shell adapter / command facade
       |
       | buoy.adapter-sdk/v1 + complete immutable values
       v
Buoy std shell -----------------> static / change / execute reports
       |
       | repository-independent values only
       v
buoy-core (`no_std`) ------------> identity / policy / impact closure
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
have been converted or handled by a fixture-proven built-in or Chelis adapter
under the pinned Buoy shell contract.

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
syntax-aware adapter registered through the pinned Buoy shell SDK. Built-in
Markdown/Rust adapters and any Chelis adapter retain distinct immutable
parser/schema identities. Comments, strings, prose mentions, file-level
annotations, and nearby atom IDs do not create carriers.

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

1. **Capability rows:** numeric Table A covers the complete finite
   `BuiltinId × SurfaceClass × operand Prim × SemanticParams` domain, where
   `SurfaceClass = Scalar | Tensor`.
   Every A cell remains present, including rejections, and binds one current
   controlling atom through its typed `SpecAtomRef`; no adapter predicate may
   filter difficult cells before completeness checking. Table B then covers
   every A-`Supported` cell times the exact backend set
   `eval | c-host | c-dag | hip | metal`. Its typed disposition is an
   implemented `KernelId`, an unimplemented `IssueRef` plus `DiagnosticKind`,
   or a rejected-by-design `SpecAtomRef` plus `DiagnosticKind`. Compact
   authoring macros expand before completeness validation; missing and
   duplicate rows fail rather than being filtered. Recursive host
   constructors form a separate complete constructor × position × backend
   domain. Container and boundary builtins use the sibling
   `BuiltinId × SiblingDomain × SiblingCaseId × SemanticParams` semantic
   registry and its supported-row × backend product, selected by their
   declaration. Runtime, stdlib, and binding callables instead retain §C6's
   exact semantic registry keyed by
   `ExternalCallableFamily × CanonicalCallableId`. Every runtime C export and
   PyO3 binding additionally receives one total target disposition keyed by
   `ExternalCallableFamily × CanonicalCallableId × ExternalTargetContext`;
   its typed cell is an implementation identity, an open issue plus
   diagnostic, or a rejected-by-design atom plus diagnostic. Exported stdlib
   definitions have no authored target cell: their per-backend result is a
   generated transitive dependency closure over the checked body, and an
   unresolved dependency or empty-by-default result fails construction. None
   is forced into numeric `SurfaceClass` rows.
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

The pinned Buoy CLI or a Chelis command facade over the same shell library
preserves three distinct surfaces:

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

Mechanically docs-only detection applies only when no changed path is under
`spec/**`. A `spec/**` edit is normative even when every changed file is
Markdown and must follow the cited OpenSpec path. Phase 0 includes a planted
negative case proving that such an edit cannot take the docs-only bypass.

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
deltas, positive or negative scenarios, implementation-PR change links, or
specific requirement/design citations fail; planted proposal/design/spec/task
drift fails; and an implementation PR whose linked change has not entered the
human review queue—or whose production commit predates that entry—fails.
Cited PRs, nonnormative docs-only PRs that do not touch `spec/**`, and
explicitly exempted PRs follow their defined paths, while a docs-only `spec/**`
edit is rejected from the docs-only path.

## Phase 1 — advisory Buoy shell-side pilot

Select and pin one exact Buoy revision whose standalone `devenv test` final
oracle is green, plus its compatibility profile, `buoy.adapter-sdk/v1`
contract, parser/schema identities, and ownership for the Chelis shell adapter
and command namespace. Resolve the parsed-item protocol through a separately
approved canonical schema before emitting any Chelis host item. Design and
fixture preparation may precede those prerequisites; production configuration,
advisory execution, and adapter code may not. Freeze positive and negative
fixtures for:

- existing atom IDs and authority migration;
- malformed and duplicate authorities;
- stale carrier revisions and unattached metadata;
- provider/root/registration-order independence;
- an expected-authority selector that produces no authorities; and
- rejection of Chelis syntax mislabeled as Rust, Markdown, or opaque data.

Adopt the pinned Buoy shell interfaces in advisory mode with Chelis-owned
configuration and a small initial scope. A shell-side integration crate may
depend inward on `buoy` / `buoy-core`; compiler-semantic crates may not, and
`buoy-core` must remain Chelis-free. Do not add a second provenance parser,
graph, or report implementation. No existing Chelis command becomes blocking.

**Oracle:** the named `chelis_provenance_advisory` integration suite invokes
the pinned Buoy static surface or Chelis facade, proves byte-identical reports
under separate roots and adapter registration orders, rejects planted
malformed/stale/mislabeled inputs, verifies the Buoy and SDK pins, and leaves
the existing Chelis gate unchanged.

## Phase 2 — provider-neutral change impact

Through the pinned Chelis/Buoy shell integration, materialize repository-owned
approvals, review records, debt, waivers, reaffirmations, and neutral
base-to-head manifests. Enable advisory change reports.

**Oracle:** the named `chelis_provenance_change_impact` suite proves that a
planted atom edit enumerates every transitive dependent claim; it rejects a
missing, duplicate, stale, or inapplicable disposition and accepts one exact
complete manifest.

## Phase 3 — first blocking ratchet

Select the first finite surfaces—initially the atomized spec/04 and spec/05
sections plus their corresponding capability/tag/tolerance domains. Require
positive and negative roles where applicable.

Integrate the ordinary required oracles through the pinned Buoy shell in this
phase: selection, effectful execution, five-state result normalization, and
exact binding back to carriers. This is the minimum execution surface needed
for a blocking coverage claim; stronger model/property/proof evidence remains
Phase 4.

For the [#729] capability domains, this phase consumes the owning dtype
oracles and their exact receipts. It does not redefine, weaken, or replace
`dtype_phase4c_oracle.py`, `dtype_phase4d_oracle.py`, or the authoritative
`dtype_phase4_oracle.py`; provenance proves their selection, execution state,
and binding to governed rows, while [#729] continues to own what they execute.

Promotion requires:

- the standalone Buoy `devenv test` oracle green at the exact selected
  revision;
- the published adapter SDK, approved host-item schema/migration mapping, and
  all Chelis adapter/configuration positive and negative fixtures green;
- no missing or filtered selected surface member;
- every selected carrier current;
- every required oracle selected and passing; and
- policy weakening and waiver negative controls green.

**Oracle:** the named `chelis_provenance_blocking_policy` suite proves that a
planted uncovered atom, missing surface member, stale carrier, and silent
policy weakening each fail. It also proves that a required oracle which is
unselected or reports `not-run`, `fail`, `error`, or `timeout` cannot satisfy
the obligation; an active waiver over any nonpassing verdict remains visible
but cannot synthesize satisfaction; and only the complete selected scope with
every required oracle selected, exactly bound, and passing succeeds.

## Phase 4 — stronger evidence

Add bounded models, sampled properties, formal properties, and proof records
only where their exact scope justifies the maintenance cost. Reuse Phase 3's
execution and result-normalization path rather than creating another identity
or report surface. Preserve the boundary that source-level proof does not prove
rustc, LLVM, emitted C/HIP/Metal, or machine code.

**Oracle:** the named `chelis_provenance_stronger_evidence` suite proves exact
model/property/proof scopes, assumptions, exclusions, negative controls, and
current bindings while leaving static identity independent of execution.

## Final integration oracle

The authoritative completion oracle for the full Chelis integration is
`devenv test` executed from the pinned Chelis integration worktree. The focused
phase suites above must be registered under that command; they do not replace
it. Success requires standalone Buoy compatibility, Chelis fixture polarity,
complete surfaces, adapter totality, cross-root/offline equality, advisory and
blocking ratchets, neutral impact closure, derivations, oracle normalization,
mutation controls, and every selected formal binding to be current and green
without network or provider authority.

This future integration oracle is not part of Chelis's current developer gate;
Phase 1 must materialize and document the pinned integration environment before
the command can count as evidence.

## Readiness rule

Phase 0 and Phase 1 design/fixture preparation may proceed while Buoy hardening
is incomplete, but no advisory pilot counts as Chelis integration evidence
until the exact selected Buoy revision's standalone `devenv test` final oracle
is green. A Chelis-language adapter may not emit host parsed-item identity until
the separately approved schema/migration blocker is resolved.

Blocking adoption additionally requires the published adapter-SDK and
host-item compatibility contracts, all Phase 1-2 advisory obligations visible
and current, and the Chelis-specific Phase 3 oracle. Neither a partial Buoy gate
nor a green built-in-adapter slice satisfies this rule.

This design records the target; it does not claim that condition is satisfied
today.

## Non-goals

- Reimplementing Buoy semantics, coupling `buoy-core` back to Chelis, or
  placing provenance dependencies in compiler-semantic crates.
- Encoding Chelis host syntax as `RustItem`, `MarkdownFence`, or opaque data.
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

1. the exact Buoy revision, compatibility policy, adapter-SDK contract, and
   parser/schema identities;
2. the Chelis shell-side crate and command namespace plus the configuration and
   report-schema boundary;
3. the approved host parsed-item schema, historical reader, migration, fixture,
   and compatibility mapping;
4. the first advisory selectors, finite decision surfaces, and blockquote-atom
   migration strategy; and
5. the initial derivation inventory for generated and mirrored artifacts.

[Buoy]: https://github.com/Chelis-Lang/buoy
[Buoy-61b2c25a]: https://github.com/Chelis-Lang/buoy/commit/61b2c25afd55f8295715868cb854d2b4777de9a2
[#694]: https://github.com/Chelis-Lang/chelis/issues/694
[#733]: https://github.com/Chelis-Lang/chelis/issues/733
