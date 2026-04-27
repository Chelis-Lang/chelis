# Chelis: Ecosystem and Communications Context

**Status:** Active context memo.
This file is for positioning, devrel, and ecosystem-facing summaries.
It is not the language spec and it is not a historical design note.

---

## What Chelis Is

Chelis is a functional programming language for AI research.
It is designed around a workflow where coding agents write most of the code and humans
supervise, review, and steer.

Its core technical differentiators are:

- Surf for readable supervision, Deep for canonical machine-facing structure
- named tensor dimensions and explicit precision tracking
- compiler feedback as graded training signal
- a compact RISC DAG that keeps transforms and backend work tractable
- executable properties as spec — customer-authored `@property` functions verified
  against AI-generated implementations on random inputs (`chelis fuzz`)

Chelis is not trying to be a general-purpose language or a Python replacement.
It targets the model-definition and compiler layer for AI workloads.

---

## Executable Properties as Spec

No other platform lets you write "for all valid inputs, put-call parity holds" as
executable code and have the toolchain verify it on 100,000 random inputs against the
AI-generated implementation. Properties are Chelis functions annotated with
`@property`. They define correctness. The toolchain enforces it. The customer reviews
properties (simple, one-line domain facts), not generated code (complex, optimized,
opaque). This directly addresses the consequential computing concern: "I can't define
a spec and ensure it's in the generated code." The answer: "You define the spec as
properties. We verify the code satisfies them."

Domain shells ship reference implementations alongside properties. The customer
doesn't write a reference for standard models — Shoals provides Black-Scholes,
Heston, Vasicek; future vertical shells provide their domain's standard models. The
customer writes references only for proprietary models. The toolchain proves
optimized implementations agree with references on random inputs. The customer
reviews properties and references (short, declarative, human-readable), not generated
code (long, optimized, opaque).

Where this fits relative to the rest of the verification stack — and how it compares
to the prevailing alternative:

| Platform | Spec verification |
|---|---|
| Other AI codegen platforms | None (manual review, PDF specs that rot) |
| Chelis | Executable properties + reference implementations: shell-provided references for standard models, customer-written properties for invariants and reference-correspondence, all verified by `chelis fuzz` on random inputs |

Full design: `chelis_trust_stack.md`, `chelis_reference_implementations_spec.md`,
`chelis_fuzz_spec.md`.

---

## Why It Matters

Chelis is built around three converging realities:

1. coding agents are becoming the primary authors of code
2. AI research still suffers from weak shape, precision, and correctness guarantees
3. programs-as-data matters for architecture search, program synthesis, and learned
   systems

The dual-syntax architecture and compiler feedback loop are the answer to those three
pressures taken together.

---

## How to Talk About the Coding Capability

Do **not** describe the Chelis coding strategy as "just fine-tune a model."
The current strategy has two explicit tracks:

### Track 1

Use a first-party `SKILL.md` plus the Tide MCP server with frontier models.
This is the Phase 2 story: compiler-in-the-loop, in-context learning, validated against
the current compiler surface.
Then late Phase 3 rewrites that skill file for the fuller language-complete surface:
pipe-first Surf, scalar/string code, collections, iteration, and tokenization/data-
loading workflows.

### Track 2

Ship a local model with the toolchain.
This is the Phase 4 story: SSD for distributional shaping, complexity-aware trajectory
collection with compiler feedback, empirically chosen fine-tuning, then quantize and
ship as GGUF.
LoRA is the default starting point, SDFT is the anti-forgetting fallback if forgetting
is measured, and RLVR is optional final polish if needed.

The important change is that the local model is a required deliverable, not optional
hedging.

---

## Dependencies and Prerequisites

The critical dependency for Track 1 is the Tide MCP server.
That is the interface where the compiler-as-teacher loop runs.

A second non-negotiable prerequisite is a seed corpus of 50-100 hand-written or
supervised-interaction programs covering core Chelis patterns.
Those programs serve as:

- few-shot examples
- trajectory seeds
- evaluation anchors

The corpus should be deliberately stratified by problem complexity (~20 single-op, ~40
single-layer, ~30 multi-layer, ~10 full models) so that all bands are represented as
controls when measuring the ICL effect across complexity levels.

Before local-model training starts, measure the ICL effect by running the SKILL
evaluation with and without the spec in context.
That tells us whether distillation-style methods are even worth attempting.

Anti-forgetting is the hard constraint for Track 2:

- Chelis training must not erase PyTorch/JAX semantic knowledge

---

## What Success Looks Like

### Phase 1-2 Milestone

Frontier models can write useful Chelis through `SKILL.md` + MCP, and that workflow is
documented, tested, and repeatable.
The skill path has already validated at 9/10 tasks on a local Qwen 35B MoE setup.

### Longer-Term

Chelis ships a first-party local coding model as part of the toolchain.
The point is not only to prove frontier-model usability.
The point is to make AI generation of correct Chelis code a standard product capability,
including offline/local use.

---

## Positioning Notes

- "First-party coding capability" is the right umbrella phrase.
- "Compiler-as-teacher" is more accurate than "compiler-as-evaluator only."
- "SKILL.md + MCP" is the Phase 2 story.
- "Package system and Python interop are shipped foundations; remaining Phase 3 work is
  language completeness" is the key sequencing story.
- "Pipe-first examples first, then scalar/string/collection/tokenization support, then
  `SKILL.md` v2 redo" is the Phase 3 public-surface story.
- "Local model ships with toolchain" is the Phase 4 story.
- "A language for AIs that doesn't include an AI is an incomplete product" is now part
  of the core framing.
