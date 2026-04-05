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

Do **not** describe the Chelis coding strategy as "train a fine-tuned model" by default.
The current strategy is staged:

### Stage A

Use a first-party `SKILL.md` plus the Tide MCP server with frontier models.
This is compiler-in-the-loop, in-context learning first.

### Stage B

Collect trajectories from models attempting Chelis tasks with compiler feedback.
Failures are useful training data, not waste.

### Stage C

Fine-tune only if the earlier stages do not meet the quality bar.

This keeps the story aligned with both the project plan and the evidence that
frontier-model in-context learning may already be sufficient for a narrow, highly
structured language like Chelis.

---

## Dependencies and Prerequisites

The critical dependency for first-party AI generation is the Tide MCP server.
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

### Longer-Term

Chelis has a first-party coding capability, delivered through the cheapest mechanism
that works:

- skill file
- trajectory collection
- or a fine-tuned model if needed

The point is not to force a specific ML strategy.
The point is to make AI generation of correct Chelis code reliable enough to be part of
the language's core value proposition.

---

## Positioning Notes

- "First-party coding capability" is the right umbrella phrase.
- "Compiler-as-teacher" is more accurate than "compiler-as-evaluator only."
- "Fine-tuned Chelis model" is a possible later outcome, not the default present-tense
  description.
