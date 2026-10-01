# Chelis: Ecosystem and Communications Context

**Status:** Active context memo.
This file is for positioning, devrel, and ecosystem-facing summaries.
It is not the language spec. `spec/design/chelis_canonical_reference.md` controls
what Chelis is; this memo explains how to talk about it.

---

## What Chelis Is

Chelis is a numerical computing language for code that agents write and people
supervise. Tensors carry named dimensions and precision in their type, and a proof
stack checks the properties an author states about the code.

The points to lead with:

- named tensor dimensions and explicit precision, checked before anything runs, with
  no implicit broadcasting and no implicit precision promotion
- effects and ownership in the type, so randomness, I/O, and aliasing are visible in a
  signature
- structured, deterministic diagnostics an agent can act on, with suggested repairs
- Surf for readable supervision, Deep for canonical machine-facing structure
- executable properties as spec: `@property` functions checked by `chelis prove`,
  with each result naming the method behind it (type checking, SMT, or seeded
  sampling)
- Hull, a second checker that cross-checks the compiler, and a Lean 4 mechanization of
  a core calculus of Chelis

Chelis is general purpose within numerical computing. The worked examples come from
quantitative finance because that is where a silent wrong number is most expensive,
not because the language is limited to it. Chelis is not a systems language, a web
framework, a deep-learning framework, or a general scripting replacement for Python.

Differentiation and machine-learning programs are research directions. Describe them
in general terms, never as the purpose of the language.

---

## Executable Properties as Spec

Properties are Chelis functions annotated with `@property`. They state what correct
means ("put-call parity holds for all valid inputs"), and `chelis prove` checks the
generated implementation against them. A supervisor reviews properties, which are
short domain facts, instead of re-reading long optimized code.

Domain shells ship reference implementations alongside properties. Shoals carries
standard pricing models, so a user writes references only for proprietary models, and
`chelis prove` checks that an optimized implementation agrees with its reference.

Full design: `chelis_trust_stack.md`, `chelis_reference_implementations_spec.md`,
`chelis_property_spec.md`.

---

## Why It Matters

1. Coding agents are becoming the primary authors of numerical code.
2. Numerical bugs (a broadcast that stretches the wrong axis, a silent precision
   change, an index that wraps) produce plausible wrong numbers instead of crashes.
3. A person cannot re-derive every generated line, but can read types and properties.

Chelis answers all three together: the compiler rejects the bug classes, the prover
checks stated properties, and the diagnostics close the loop for the agent.

---

## How to Talk About Agent Coding

Agents write Chelis through a first-party `SKILL.md` and the Tide MCP server
(`chelis tide mcp`), which exposes check, eval, prove, and structural Deep edits as
tools. The compiler is the agent's feedback loop: every rejection names what went
wrong and where.

A small local coding model trained on Deep is a research direction, not a product
commitment. Do not describe the coding strategy as "fine-tune a model."

---

## Positioning Notes

- Lead with the premise: agents write, the compiler and prover check, people supervise.
- Say "numerical computing language"; "tensor language" is acceptable shorthand.
- Use quantitative finance for examples, and keep the general-purpose scope explicit.
- Name C as the build target.
- Do not lead with autodiff, neural networks, or model training.
