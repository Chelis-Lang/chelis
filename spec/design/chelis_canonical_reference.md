# Chelis: Canonical Project Reference

This reference describes Chelis's purpose, architecture, and package boundaries.
The [language specifications](../00-context.md) define syntax and semantics;
the [Chelis Guide](../../docs/book/src/README.md) explains how to use the toolchain.

## 1. What Chelis Is

Chelis is a numerical computing language for code that agents write and people
supervise. Tensor types carry named dimensions and precision. The compiler checks
shapes, precision, effects, and ownership before execution. `chelis prove` checks
stated properties and names the method behind each result.

Chelis covers numerical methods, simulation, statistics, dataframes, and finance.
Differentiation and machine-learning programs are applications within that scope.
Chelis is not a systems language, web framework, deep-learning framework, or general
scripting replacement for Python.

The [examples](../../examples/) contain complete programs. The [first-program
guide](../../docs/book/src/first-program.md) shows how to check and run one.

## 2. Core Bet

Numerical code is easier to review when types carry the facts numerical errors
depend on, and when a tool can check the properties its author states. A person can
supervise an agent-written program through its types, diagnostics, and properties
without re-deriving every operation.

`chelis check` returns structured diagnostics with source locations. For fixed
source, compiler build, target, and declared inputs, checking and execution have
deterministic results. The exact contract is in [the context
specification](../00-context.md#5-design-principles).

## 3. Design Principles

Chelis applies these principles in order when they conflict:

1. Unambiguity over ergonomics.
2. Composition over special cases.
3. Inference over annotation when the answer is unique.
4. Machine generation first.
5. Surface syntax adds no semantics absent from the core.
6. Explicit broadcasting, precision conversion, effects, and ownership.
7. A small compiler-known core and a broad library ecosystem.
8. Fully decided rules without implementing speculative machinery.

In particular, tensor dimensions do not broadcast implicitly; `expand` spells out
expansion. Dtypes do not promote implicitly, and calls do not curry or partially
apply implicitly. The [type](../04-type-system.md) and
[operation](../05-risc-primitives.md) specifications own the precise rules.

## 4. Dual Syntax Architecture

**Surf** (`.ch`) is the readable source form for writing and reviewing programs.
**Deep** (`.dp`) is the canonical structural form used by the compiler and editing
tools. Surf desugars to Deep; the supported Deep surface resugars to
formatter-canonical Surf. Invalid or unrepresentable Deep fails validation instead
of producing plausible source. The [Surf](../02-surf-syntax.md) and
[Deep](../03-deep-syntax.md) specifications define the round-trip contract.

Every public Deep node has a tag, a metadata map, and its children:

```lisp
(tag {} children...)
```

Tags come from a closed vocabulary. Function calls, names, and literals have
explicit structural forms. The metadata slot can retain source provenance.
`chelis deep`, `chelis surf`, and `chelis validate` expose the representations;
the [CLI guide](../../docs/book/src/cli.md) gives runnable commands.

Chelis uses `.ch` for Surf, `.dp` for Deep, and `.chb` for package metadata.
The package artifact contract lives in [Reef distribution](reef_distribution.md).

## 5. Ecosystem Names

Surf and Deep name the two source forms. A **shell** is a Chelis package; **Reef**
manages packages through `reef.toml`. **Tide** is the interactive interface, and
**Cove** is the terminal coding environment. Chelis is pronounced **CHEL-is**.

### 5.1 Shell Ecosystem

The [shell repository contract](shell_repo_contract.md) defines how downstream
packages use the toolchain. The
[`chelis-conformance` registry](../../crates/chelis-conformance/src/registry.rs)
records shell repositories and their dependencies; its active entries are
checked by the ecosystem canary. The registry owns package membership and
status. This document describes the main domain roles:

| Package | Role |
|---|---|
| Nautilus | Numerical methods |
| Coral | Dataframes |
| Shoals | Quantitative finance |
| Octant | Mathematical notation and Deep |
| School | Machine learning |
| Hull | An executable language specification |

Other shells can provide tools or domain libraries without becoming compiler
primitives. A shell may depend on another shell through Reef. The
[Reef guide](../../docs/book/src/reef.md) covers package use.

### 5.4 Bundled Standard Library

`chelis-std` is bundled with the compiler. It is the `Std.*` library used by
Chelis programs, not an independently substituted shell. Its package pin follows
the project's compiler pin. [Reef distribution](reef_distribution.md) defines
the bundle, lockfile entry, and loading behavior. The [standard-library
guide](../../docs/book/src/stdlib.md) describes the available modules and
their limits.

## 6. CLI Surface

The `chelis` CLI formats, validates, checks, evaluates, proves, and builds
programs. It also exposes Surf/Deep conversion, native tests, Reef package
commands, Tide, and Cove. The [CLI guide](../../docs/book/src/cli.md) contains
executable command examples and describes the output of each command.

`chelis build` targets C by default; HIP and Metal are available as experimental
targets. `chelis eval` uses the evaluator rather than executing a generated
native binary. Checking a program does not certify that every backend can build
it. See the [backend guide](../../docs/book/src/backends.md).

`chelis build`, `chelis check`, `chelis validate`, and `chelis eval --file`
apply the [Surf style](../01-nomenclature.md) rules before compilation.
`chelis fmt` and `chelis lint --check` let an author check the same surface
directly. The CLI guide documents the supported override.

## 7. Type System Scope

The [type system](../04-type-system.md) includes algebraic data types, pattern
matching, inference, named tensor dimensions, explicit numeric precision, effects,
and ownership. Scalars, strings, lists, and dictionaries let numerical programs
prepare data and express host-side logic. Borrowing and explicit ownership
operations govern resource use. Typed diagnostics explain rejected programs.

Random operations take explicit keys. Randomness is not an effect or an ambient
seeded region. The [effects guide](../../docs/book/src/effects.md) gives working
examples; the numbered specs define their meaning.

## 8. Computational Model

The compiler lowers typed numeric operations to a graph of primitive operations.
Derived built-ins compose those primitives while preserving their named identity
through the semantic transformations that require it. The
[operation specification](../05-risc-primitives.md) defines their domains,
results, failures, and differentiation rules.

`sub` and `min_elem` are direct RISC primitives governed by [05-OP-41] and
[05-OP-40], respectively. They retain their own identities through lowering;
neither is rewritten as a composition of other arithmetic operations.

`grad` differentiates eligible functions and `vmap` maps functions over a named
axis. Both require compiler cooperation. The
[transform specification](../06-transformations.md) owns their exact behavior;
the [transforms guide](../../docs/book/src/transforms.md) shows executable uses.

### 8.5. Scope Boundaries

The compiler core contains operations and transforms whose semantics require
compiler knowledge: primitive operations, derived built-ins, and transformations
such as `grad` and `vmap`. An ordinary library function is built from these
operations without introducing a new compiler-recognized identity or adjoint.

The bundled **standard library** supplies reusable `Std.*` modules, including
I/O, tensor helpers, dates and times, and exact decimal arithmetic. A standard
library module is packaged with the toolchain even when its behavior is written
in Chelis. Module availability and known limits are documented in the
[standard-library guide](../../docs/book/src/stdlib.md).

**External shells** supply domain knowledge and tools. Nautilus owns numerical
methods; Coral owns tabular data; Shoals owns finance; School owns machine
learning; Octant bridges mathematical notation and Deep. A new algorithm or
model belongs in a library when it can compose the existing operations. A new
primitive requires a language contract in the numbered specifications.

This boundary concerns compiler knowledge, not performance by itself. The
compiler may recognize and optimize a library function without making its
public API part of the language. The [operation specification](../05-risc-primitives.md)
owns compiler-recognized numeric identities.

## 9. Backend Strategy

The default backend emits portable C and links the bundled runtime. HIP and
Metal generate host code and device kernels for supported programs. Backend
capabilities differ, so the [backend specification](../08-backends.md) and
[guide](../../docs/book/src/backends.md) define target selection and limits.
Python bindings provide another way to use the compiler; they do not change
the language's core semantics.

## 10. Interactive Execution

`chelis eval` and Tide use the evaluator for interactive results. `chelis build`
produces native artifacts through the selected backend. The [Tide
specification](../09-tide.md) and [CLI guide](../../docs/book/src/cli.md)
describe the interfaces and outputs.

## 11. Compiler Implementation

Chelis is a Rust workspace. Its parser, type checker, intermediate representation,
backends, runtime, CLI, and tooling are organized as workspace crates. The
[architecture guide](../../ARCHITECTURE.md) maps these components to source
directories. Implementation choices follow the language contracts rather than
defining them.

## 12. Agent Coding Assistance

An agent can write Surf or Deep, use structured compiler feedback to repair a
program, and call check, evaluation, proving, and editing tools through Tide.
Deep gives editing tools explicit nodes and a stable metadata location; the
[editing-surface design](chelis_agent_editing_surface.md) defines its validation
and failure behavior. The [Tide specification](../09-tide.md) describes the
interactive tools.

The tools report their own checks. A successful type check does not imply a
successful proof or a successful build on every target. The [proving
guide](../../docs/book/src/proving.md) explains methods and qualifiers.

## 13. Documentation Hierarchy

Use this reference for project purpose and the core, standard-library, and shell
boundary. The [numbered specifications](../00-context.md) control language and
CLI semantics. The [Chelis Guide](../../docs/book/src/README.md) explains
usage; focused documents in this directory describe implementation choices; the
[project plan](chelis_project_plan.md) records sequencing and acceptance work.
Documents under [`spec/design/archive/`](archive/) preserve historical
decisions and do not define the current contract.
