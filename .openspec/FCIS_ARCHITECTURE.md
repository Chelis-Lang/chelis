# Chelis FCIS Architecture Contract

This document defines the cross-change laws used by every Functional Core / Imperative
Shell migration under `.openspec/changes/`. Domain protocols remain separate: this is a
shared vocabulary and invariant set, not a generic effect framework.

This is a migration design shared by the active OpenSpec changes, not an additional tier in
the repository's normative documentation hierarchy. A completed migration must promote the
accepted, minimal FCIS laws into `spec/design/chelis_canonical_reference.md` or the owning
active `spec/00-12*.md` document. Until then, this file must not override those sources.

## Boundary rule

Every change names a designated core crate or exact module set and its adapter set before
implementation. The core's dependency graph must make forbidden host capabilities
unavailable where practical. Source architecture checks supplement crate boundaries; they
do not replace behavioral determinism, denial, parity, and replay tests.

A designated core may mutate invocation-owned local state. It may not obtain filesystem,
environment, process, network, wall-clock, terminal, entropy, thread-scheduling,
global-panic-hook, mutable-static, thread-local semantic state, unsafe FFI, or other
platform-dependent semantic capabilities directly or through aliases, re-exports, traits,
callbacks, function pointers, macros, or adapter imports. A domain may retain a narrowly
named implementation-safety mechanism, such as fallible allocation or stack growth, only
when it cannot alter semantic acceptance and reports failure through the host-failure
channel.

Architecture enforcement prefers a dependency-minimal crate boundary whenever the domain
can support one without a compatibility break. Where a temporary module boundary is
necessary, an exact production-module manifest supplements dependency allowlists,
resolved-API checks where available, and negative fixtures for specifically claimed bypass
forms. Lexical source scans alone are not a transitive purity proof, and fixture coverage
must not be described as complete detection of arbitrary macro expansion, dynamic dispatch,
or future Rust syntax. Public core interfaces must not accept unsealed callbacks or traits
capable of smuggling forbidden host capabilities. Every architecture gate names its actual
enforcement mechanism, threat model, proven fixture classes, and known blind spots before
implementation begins.

## Decision and workflow algebra

Semantic rejection, protocol corruption, and host execution failure are different states.
Public core results use domain-owned sum types that make contradictory states
unrepresentable. The following are law-shaped examples, not shared production generics:

```text
Decision<T, D> = Accepted(T) | Rejected(D)
Execution<T, D, H> = Decision<T, D> | HostFailure(H)
Step<T, D, R, S, P, H> =
    Completed(Decision<T, D>)
  | Suspended { request: R, state: S }
  | ProtocolFailure(P)
  | HostFailure(H)
```

A domain may add an internal non-returned `Continue` state or an effect-execution report
algebra for partial/indeterminate external state, but its public result must still preserve
the distinctions above. Each owning change gives an explicit table mapping expected host
observations, policy denial, semantic rejection, protocol corruption, executor failure, and
partial or indeterminate effects into its domain result variants.

Expected domain-relevant host outcomes such as not-found, permission denial, timeout,
unavailable capability, or a child-process exit are typed observations when the core must
decide their meaning. A malformed handoff, correlation failure, impossible response kind,
or transcript-integrity failure is a protocol failure. An adapter or implementation failure
that cannot be interpreted as a domain observation is a host failure.

A successful value cannot contain error diagnostics. A rejected value cannot claim a
complete successful artifact set. Protocol and host failures are not cached or rendered as
semantic source rejection. Unrecoverable process aborts, such as an allocator abort that
Rust cannot report, are outside the returned-result guarantee and must not be relabeled by
panic catch.

## Request and observation laws

A workflow whose next semantic decision depends on a host result uses a domain-specific,
closed request/observation protocol.

- Request identities are deterministic invocation-local sequence numbers starting at zero.
  A random UUID, process-global counter, clock, address, or thread identity must not affect
  semantic output or replay identity.
- A suspension is consumed by value, is not cloneable, and has exactly one outstanding
  request. Response validation occurs before continuation state changes.
- Response kind, request identity, domain identity, exact versioned payload/path/argument/
  entry/transcript bounds, and integrity metadata are validated. A request-workflow change
  pins its v1 defaults and total-memory/collection ceilings before implementation rather than
  leaving `bounded` as an executor-selected promise. Duplicate, stale, mismatched, malformed,
  and oversized observations fail closed.
- Replaying a recorded transcript means starting a new machine from equal initial inputs and
  supplying the normalized observations in order. It never means reusing a consumed live
  suspension.
- Requests and observations contain structured data. A live observation may transiently
  carry sensitive domain payload needed by the core, such as bytes read from an authorized
  file. Secret or non-persistable payload is excluded from normalized persistent
  transcripts, diagnostics, notices, plans, hashes, logs, and semantic identities; its
  presence disables persistence rather than being hidden by hashing it.
- Adapters execute requests and report observations. They do not choose semantic follow-up
  requests, target policy, dependency policy, proof trust, or verdicts.
- The v1 protocols in these changes intentionally allow one outstanding request. A domain
  that needs concurrency may later yield one canonically ordered request batch under one
  continuation; adapters still may not invent or reorder semantic work.

## Identity taxonomy

Identity and persistence are conditional capabilities, not prerequisites for an FCIS
boundary. A change that reads or writes a semantic cache, persistent replay record, stable
idempotency key, or externally compared digest distinguishes three identities:

1. **Query key**: authoritative initial semantic inputs available before execution. This is
   the only identity suitable for locating cache candidates.
2. **Decision or evidence digest**: the query key plus the normalized observation transcript
   or other evidence used to produce a returned decision.
3. **Report identity**: optional measurement and rendering metadata such as elapsed duration,
   cache location, terminal mode, or adapter logs.

A query-key hit does not by itself authorize reuse of an observation-dependent result. The
owning domain must verify the stored evidence digest, replay or validate the normalized
observations under the current policy, enforce freshness/revocation rules where relevant,
and revalidate mutable prerequisites before effects. Domains that cannot do so disable final
outcome caching.

Report metadata never changes a query key or semantic decision. Observation transcripts may
change a decision/evidence digest but cannot be required to discover a cache entry before
those observations exist. When a change actually ships an identity, its encoding is
versioned, domain-separated, length-aware, canonical before hashing, and covered by golden
vectors. An identity is computed by the owning core from the value it identifies or is
validated against that value before use; a caller-asserted digest, descriptor fingerprint,
prepared-program identity, or blob identity is not authoritative by construction. A change
that does not ship persistence or identity omits these types and tests rather than inventing
a speculative hash contract.

## Capability and trust rule

Host capabilities and trust authorization are explicit but are not self-asserted by an
adapter. An adapter-provided descriptor may identify its implementation, configuration, and
transport; a separately trusted policy determines what that descriptor is authorized to do
or claim. Unknown capabilities fail closed. Unknown proof engines receive no green-producing
trust authorization.

Secrets are passed only to the executor authorized to use them. They are never embedded in
plans, requests intended for persistence, normalized persistent observations, notices,
diagnostics, cache keys, logs, or replay records. A transient live observation that must
carry sensitive domain payload follows the non-persistence rule above.

## Determinism and ordering

Every returned collection with user-visible or identity-bearing meaning has a specified
total canonical ordering, including mixed representable/unrepresentable path locators,
tombstones, failures, ties, and duplicate candidates. Equal semantic inputs and equal
normalized observation sequences produce equal decisions. Host enumeration order, randomized
hash-map state, scheduling, entropy, thread stack size, elapsed time, locale, and process
environment are not semantic tie-breakers. Every identity-bearing logical path specifies
separator, Unicode/non-Unicode, case, invalid-component, and cross-platform ordering behavior
rather than inheriting host path accidents.

## Compatibility projections

Parity is evaluated only after declared preparation and preflight differences. Each public
surface matrix states supported operations, targets or capabilities, style/preflight policy,
and unsupported behavior. Compatibility projections name byte-stable fields and normalized
fields; they do not erase real semantic differences or count unsupported pairs as evidence.
Every behavior-bearing capability matrix has one typed owner; Markdown, frontend adapters,
and acceptance corpora are generated projections or tripwire-checked consumers rather than
independent authorities.

## Mechanical evidence contract

The `establish-fcis-contract-mechanics` change owns evidence infrastructure shared by these
migrations. It does not define production effect or workflow types. Before a domain
implementation begins, its active requirements and planned evidence are registered in a
versioned machine manifest:

- stable change, capability, requirement, scenario, fixture, slice, and oracle identities;
- explicit positive/negative scenario polarity and complete
  scenario-to-fixture-to-slice-to-final-oracle reachability;
- an acyclic prerequisite graph whose child gates run current-revision prerequisite
  regressions rather than trusting stale completion records;
- exact core/adapter boundaries, forbidden capability classes, architecture enforcement
  layers, proven fixture forms, threat model, and known blind spots;
- one typed owner for every builtin/rule/engine/action registry, surface matrix, lock rank,
  report state, identity domain, or other acceptance-bearing closed vocabulary; and
- exact v1 resource and protocol bounds plus explicit persistent-identity scope.

The canonical runner is `.venv/bin/python scripts/fcis_gate.py <oracle> [--slice <slice>]`.
Its validated command plans contain argv arrays, workspace-relative directories, and declared
environment policy, never shell strings. Registered but missing evidence is blocked or failed,
not skipped. Focused slices may support incremental review but cannot produce a final
completion claim. A successful final report has an empty error list by construction; failed
or blocked reports have nonempty structured errors. Semantic report order excludes absolute
checkout roots, elapsed duration, terminal mode, process identity, and localized OS text.

Architecture evidence is layered: resolved dependency allowlists and crate boundaries first,
exact mixed-module manifests where temporarily necessary, resolved forbidden-API/macro checks
where available, compile-fail fixtures for claimed bypass forms, public-interface capability
checks, and behavioral determinism/denial/replay/parity suites. The shared checker rejects a
lexical or bounded-fixture mechanism described as a complete transitive purity proof.

## Delivery and evidence

The migration proceeds in risk-reducing order:

1. establish the FCIS manifest/checker, fail-closed oracle registry, and failing evidence
   registration needed by each domain before its implementation begins;
2. land proof forced-result removal, Tide deny-external hardening, and removal of Reef's
   implicit `gh auth token` fallback as independently completable security changes;
3. use lint snapshots as the one-shot FCIS and architecture-gate proving ground;
4. make compiler stage state explicit before beginning the canonical facade and
   target/build-planning slice;
5. migrate proof and evaluator request/observation protocols through bounded
   builtin/engine-family slices; and
6. migrate Reef read-only analysis, local transactions, conformance/source repair, and
   remote publication through separately gated slices in that order.

Security corrections and unrelated product-surface changes land as independent OpenSpec
changes. A tightly coupled migration may retain one umbrella acceptance oracle only when its
design states the cross-slice invariant that requires end-to-end completion; each slice must
still be independently reviewable, have a focused gate, preserve a rollback boundary, and
avoid claiming umbrella completion. Focused commands never satisfy the final completion claim
unless the focused command is itself the named oracle of a separately scoped change. A shared
test harness may own manifest parsing, resolved forbidden-API checks, canonical identity
helpers for domains that ship identities, and negative-fixture execution, but production
domains do not share a generic effect or workflow framework without a separate concrete
proposal.
