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

Chelis is not trying to be a general-purpose language or a Python replacement.
It targets the model-definition and compiler layer for AI workloads.

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

### Track 2

Ship a local model with the toolchain.
This is the Phase 3 story: SSD for distributional shaping, trajectory collection with
compiler feedback, LoRA fine-tune, then quantize and ship as GGUF.

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
- "Local model ships with toolchain" is the Phase 3 story.
- "A language for AIs that doesn't include an AI is an incomplete product" is now part
  of the core framing.
