# Type System Delta: add-bounded-monomorphization

## ADDED Requirements

### Requirement: Uniform recursive generic instantiation

A top-level generic function — a `def` whose checker-recorded signature carries a type
variable — MAY recurse, directly or through a mutually recursive group. Every recursive
call inside the recursive binding group SHALL be typed at the caller's own instantiation
of the group's type parameters: a type argument satisfies the requirement when it
resolves to the caller's own type parameter or remains unconstrained (and is thereby
chosen as it), and — for signature type variables introduced by inference rather than
authored binders — when it is fully concrete, since a variable-free argument cannot grow
the instantiation set. A recursive call whose type application is not so admitted
(polymorphic recursion) SHALL be a type error, reported by the
checker as the earliest competent stage per `spec/05-risc-primitives.md` [05-UNS-2],
naming the function and the two disagreeing instantiations, and citing its deciding
`spec/04-type-system.md` §3.1 atom per [05-UNS-5]. The rule is lane-uniform: a program
rejected under this requirement SHALL be rejected identically by eval, C, HIP, and Metal
paths, because the rejection happens at check time before any lane-specific stage runs.
This requirement makes the instantiation set of every accepted program finite, which is
the boundedness precondition the `generic-monomorphization` capability relies on.

#### Scenario: Direct uniform recursion is accepted

- **WHEN** a generic function `loop` over `Box[a]` calls `loop` on a value of the same
  `Box[a]` instantiation and the program applies `loop` at `Box[int32]`
- **THEN** the checker accepts the program with `loop`'s recursive call typed at
  `Box[int32]`

#### Scenario: Mutual uniform recursion is accepted

- **WHEN** generic functions `even_len` and `odd_len` over `List[a]` call each other
  with the caller's own element type
- **THEN** the checker accepts the program and both members of the group carry the same
  instantiation at every call site reached from one application

#### Scenario: Polymorphic recursion is a check-time type error

- **WHEN** a generic function `f` over `a` recursively calls `f` at `Box[a]`
- **THEN** the checker rejects the program with a diagnostic naming `f`, the caller
  instantiation, the differing recursive instantiation, and the deciding §3.1 atom, and
  no lowering or evaluation stage runs

#### Scenario: Rejection is lane-uniform

- **WHEN** the same polymorphic-recursive program is submitted to `chelis eval --file`
  and to `chelis build`
- **THEN** both invocations report the identical check-time type error and neither
  reaches a lane-specific stage
