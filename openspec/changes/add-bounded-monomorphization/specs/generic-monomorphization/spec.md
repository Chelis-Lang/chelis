# Generic Monomorphization: add-bounded-monomorphization

## ADDED Requirements

### Requirement: Recursive generic host calls compile via bounded memoized specialization

When a native build target lowers a checked call to a top-level generic host function
whose recursive binding group survives to lowering, the compiler SHALL emit one
specialized definition per distinct checked type application of that function, keyed by
the checker-recorded instantiation. The compiler SHALL memoize specializations so that
an already-emitted or in-progress `(function, instantiation)` pair is reused, never
re-expanded, and SHALL lower every recursive edge inside the group as an ordinary call
to the owning specialized symbol. Specialization SHALL be driven by checked call-site
type applications only; the compiler SHALL NOT perform eager unbounded AST expansion.
Boundedness is guaranteed by the type system's uniform recursive instantiation
requirement (`type-system` capability; `spec/04-type-system.md` §3.1), which the
lowering stage MAY assume and SHALL NOT re-litigate.

#### Scenario: Direct recursion at one instantiation

- **WHEN** `chelis build` compiles a program where a recursive generic `loop` over
  `Box[a]` is applied at `Box[int32]`
- **THEN** the emitted C contains exactly one specialized `loop` definition for
  `Box[int32]`, its recursive call targets that same symbol, and the compiled binary
  runs with the checked program's exact output

#### Scenario: Distinct instantiations get distinct specializations

- **WHEN** the same recursive generic function is applied at two different
  instantiations in one program
- **THEN** the emitted C contains exactly two specialized definitions, one per
  instantiation, and each recursive edge targets its own group's symbol

#### Scenario: Mutual recursion specializes as a group

- **WHEN** mutually recursive generic functions are applied at one instantiation
- **THEN** every member of the group receives a specialized definition at that
  instantiation, cross-calls target the group's specialized symbols, and the binary
  links and runs

#### Scenario: Memoization terminates compilation

- **WHEN** lowering encounters the recursive edge of a specialization it is currently
  emitting
- **THEN** it reuses the in-progress symbol instead of recursing into another
  expansion, and compilation terminates

### Requirement: Emitted native code references no omitted generic definition

Generated native artifacts SHALL NOT contain a reference to a generic definition the
emitter omitted. Every call emitted for a build target SHALL resolve within the emitted
artifact set or the declared runtime ABI, so that the documented manual compile step
(`gcc`/`hipcc`) links without undefined symbols. If lowering cannot produce a
specialized definition for a surviving generic call, it SHALL reject through the
unsupported-case response contract (`spec/05-risc-primitives.md` §7, [05-UNS-1..5]) —
naming the call and carrying its authority — rather than emit a dangling reference.
While any such rejection remains not-yet-implemented rather than deliberate, its
diagnostic SHALL cite an open tracking issue per [05-UNS-5].

#### Scenario: Build artifacts link cleanly

- **WHEN** `chelis build` succeeds on a program containing recursive generic host calls
  and the user compiles the emitted C with the reported flags
- **THEN** the native compile step completes with no undefined-symbol error and the
  binary's observed output matches the eval lane on the same program

#### Scenario: A surviving unsupported call fails closed

- **WHEN** lowering reaches a generic call for which no specialized definition can be
  produced
- **THEN** the build fails with a branded `unsupported:` diagnostic carrying its
  authority, and no C referencing the omitted definition is written

#### Scenario: No shipped diagnostic cites a closed tracking issue

- **WHEN** any remaining not-yet-implemented rejection on this surface is emitted
- **THEN** the issue it cites is open (chelis#1158 until this capability lands), never
  the closed chelis#941
