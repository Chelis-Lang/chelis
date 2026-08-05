# Compiler Pipeline Architecture Delta: canonicalize-library-proof-identity

## ADDED Requirements

### Requirement: Canonical library proof identity
The library proof identity SHALL be derived from the final semantically accepted program's source, after the effect and linearity passes, so the identity a checked library carries corresponds to the exact source that library carries.

The identity SHALL be path-independent: a monolithic build and a layered build of the same whole library source SHALL produce the same library proof identity. Library composition SHALL recompute the composed identity canonically from the whole composed source with a top-level context, and SHALL rebind the composed type environment to that identity so the type environment and program continue to agree.

Composition SHALL continue to reject an extension that does not retain the exact library identity it was checked against, and contextual lowering SHALL continue to reject a lowered library with another proof identity.

#### Scenario: Identity corresponds to the final accepted source
- **WHEN** a checked library is built and its identity is read
- **THEN** the identity equals the identity derived from the final accepted program's own source and top-level context

#### Scenario: Monolithic and layered identities agree
- **WHEN** the same whole library source is built through the monolithic path and the layered path
- **THEN** both checked libraries carry the same library proof identity

#### Scenario: Composition still rejects a foreign extension
- **WHEN** an extension that recorded a different library identity is composed
- **THEN** composition returns no checked program, unchanged from the sealed behavior

#### Scenario: Chained composition matches the canonical identity
- **WHEN** an extension is checked against a composed library and then composed onto it
- **THEN** the extension's recorded context identity equals the composed library's canonical identity and composition succeeds

### Requirement: Cache decode recomputes the library proof identity
Cache decoding SHALL recompute the library proof identity from the decoded program's own source and top-level context, and SHALL reject a stored identity that does not equal the recomputed value, before it constructs a `CheckedLibrary`.

This recomputation SHALL be in addition to the existing declared-type-map match and the rerun of effect and linearity checks. It SHALL NOT repeat type inference. Both the compiler-API library context and the chelis-std sub-context decode paths SHALL perform it. The identity constructor SHALL remain private, and only the cache decode path SHALL reach the recompute entry.

#### Scenario: A stored identity is forged but self-consistent
- **WHEN** cache decoding receives a library whose stored identity is carried consistently on both the type environment and the checked program but is not equal to the identity recomputed from the decoded source
- **THEN** the parser rejects the pair before it constructs the checked library

#### Scenario: A legitimately produced cache still decodes
- **WHEN** cache decoding receives a library whose stored identity equals the identity recomputed from its own decoded source
- **THEN** the parser accepts it and constructs the checked library

#### Scenario: Recomputation does not repeat type inference
- **WHEN** cache decoding recomputes the library proof identity
- **THEN** it derives the identity from the decoded source without running a type-inference session, and it still reruns effect and linearity checks
