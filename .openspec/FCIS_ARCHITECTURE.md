# Chelis FCIS Architecture Contract

This document defines the cross-change laws used by every Functional Core / Imperative
Shell migration under `.openspec/changes/`. Domain protocols remain separate: this is a
shared vocabulary and invariant set, not a generic effect framework.

## Boundary rule

Every change names a designated core crate or exact module set and its adapter set before
implementation. The core's dependency graph must make forbidden host capabilities
unavailable where practical. Source architecture checks supplement crate boundaries; they
do not replace behavioral determinism, denial, parity, and replay tests.

A designated core may mutate invocation-owned local state. It may not obtain filesystem,
environment, process, network, wall-clock, terminal, global-panic-hook, mutable-static, or
thread-local semantic capabilities directly or through aliases, re-exports, traits,
callbacks, function pointers, or adapter imports.

## Decision algebra

Semantic rejection and host execution failure are different states. Public core results use
sum types that make contradictory states unrepresentable:

```text
Decision<T, D> = Accepted(T) | Rejected(D)
Execution<T, D, H> = Decision<T, D> | HostFailure(H)
```

A successful value cannot contain error diagnostics. A rejected value cannot claim a
complete successful artifact set. A host failure is not cached or rendered as a semantic
source rejection. Unrecoverable process aborts, such as an allocator abort that Rust cannot
report, are outside the returned-result guarantee and must not be relabeled by panic catch.

## Request and observation laws

A workflow whose next semantic decision depends on a host result uses a domain-specific,
closed request/observation protocol.

- Request identities are deterministic invocation-local sequence numbers starting at zero.
  A random UUID, process-global counter, clock, address, or thread identity must not affect
  semantic output or replay identity.
- A suspension is consumed by value, is not cloneable, and has exactly one outstanding
  request. Response validation occurs before continuation state changes.
- Response kind, request identity, domain identity, payload bounds, and integrity metadata
  are validated. Duplicate, stale, mismatched, malformed, and oversized observations fail
  closed.
- Replaying a recorded transcript means starting a new machine from equal initial inputs and
  supplying the normalized observations in order. It never means reusing a consumed live
  suspension.
- Requests and observations contain structured data. Secrets, terminal strings, elapsed
  durations, ephemeral absolute paths, and process-local handles are excluded from
  persistent transcripts and semantic identities.
- Adapters execute requests and report observations. They do not choose semantic follow-up
  requests, target policy, dependency policy, proof trust, or verdicts.

## Identity taxonomy

Every change distinguishes three identities:

1. **Query key**: authoritative initial semantic inputs available before execution. This is
   the only identity suitable for cache lookup.
2. **Decision or evidence digest**: the query key plus the normalized observation transcript
   or other evidence used to produce a returned decision.
3. **Report identity**: optional measurement and rendering metadata such as elapsed duration,
   cache location, terminal mode, or adapter logs.

Report metadata never changes a query key or semantic decision. Observation transcripts may
change a decision/evidence digest but cannot be required to discover a cache entry before
those observations exist.

## Capability and trust rule

Host capabilities and trust authorization are explicit but are not self-asserted by an
adapter. An adapter-provided descriptor may identify its implementation, configuration, and
transport; a separately trusted policy determines what that descriptor is authorized to do
or claim. Unknown capabilities fail closed. Unknown proof engines receive no green-producing
trust authorization.

Secrets are passed only to the executor authorized to use them. They are never embedded in
plans, requests intended for persistence, observations, notices, diagnostics, cache keys, or
replay records.

## Determinism and ordering

Every returned collection with user-visible or identity-bearing meaning has a specified
canonical ordering. Equal semantic inputs and equal normalized observation sequences produce
equal decisions. Host enumeration order, hash-map order, scheduling, thread stack size,
elapsed time, and process environment are not semantic tie-breakers.

## Compatibility projections

Parity is evaluated only after declared preparation and preflight differences. Each public
surface matrix states supported operations, targets or capabilities, style/preflight policy,
and unsupported behavior. Compatibility projections name byte-stable fields and normalized
fields; they do not erase real semantic differences or count unsupported pairs as evidence.
