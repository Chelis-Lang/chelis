# Chelis: Project Context for Ecosystem & Communications

---

## What Chelis Is

Chelis is a new programming language for AI research. It's a functional language with a type system designed around AI primitives — tensor shapes, numeric precision, differentiability, and neural architecture composition. The compiler catches an entire class of bugs (shape mismatches, precision errors, non-differentiable operations inside gradient computation) at compile time that current tools only surface as runtime crashes after minutes or hours of GPU time.

The defining feature: Chelis has two syntax layers. **Surf** is a clean, modern functional syntax that humans read and review. **Deep** is an s-expression representation that AI coding agents generate and manipulate. Programs round-trip between the two. This dual-layer design is built on the thesis that the next generation of AI programs will be written primarily by AI agents, with humans in a supervisory role — reading, reviewing, and steering, not typing.

The tagline: **Written by AIs, for AIs, to write AI programs. Turtles all the way down.**

The name comes from *chelys* (Greek: turtle). Domain: **chelis.ch**.

---

## Why It Exists

Three converging trends:

**1. AI agents are becoming the primary authors of code.** Claude Code, Cursor, Windsurf, Devin — the trajectory is clear. Within 2-3 years, most new code will be agent-generated with human review. But no programming language has been designed for this. Every existing language optimizes for human authoring ergonomics. Chelis is the first language designed from the ground up where the primary author is a machine.

**2. AI research is bottlenecked by Python's type system (or lack thereof).** A PhD student implementing a novel transformer variant in PyTorch loses days to: shape errors discovered after 20 minutes of training, silent numerical degradation from wrong precision, custom backward passes that are subtly wrong, and GPU memory leaks from Python reference cycles. These are all type system problems. Chelis catches them at compile time.

**3. The next frontier in AI is programs that write programs.** Neural architecture search, evolutionary program synthesis, neurosymbolic AI, learned function compilation — all require treating programs as data that can be inspected, mutated, type-checked, and evolved. Chelis's homoiconic representation (programs are typed data structures) makes this natural. The compiler's graded fitness feedback (not just pass/fail, but a 0-1 score with structured repair suggestions) turns the type checker into a reward function for AI agents doing RL or evolution over program space.

---

## Who It's For

**Primary:** ML researchers and AI systems engineers who build novel architectures, training methods, and AI infrastructure. The NeurIPS/ICML/ICLR crowd. People who currently use PyTorch, JAX, or tinygrad and fight their tooling daily.

**Secondary:** AI agent developers building coding agents, architecture search systems, or neurosymbolic pipelines. These users care about the programs-as-data property and the compiler-as-reward-signal.

**Tertiary:** Research labs and AI companies evaluating next-generation toolchains for AI development at scale. The "what comes after PyTorch" conversation.

**Explicitly not for (at launch):** Web developers, systems programmers, general-purpose scripting, DevOps. Chelis is a domain language for AI research and deployment, not a general-purpose language.

---

## Key Technical Differentiators

For communications purposes, these are the talking points. You don't need to understand the implementation — just the value proposition.

**1. Compile-time shape and precision checking.** Tensor dimensions are named and tracked through the type system. `tensor[batch, seq, hidden, bf16]` is a type. Passing a `tensor[batch, hidden, seq, bf16]` (transposed) is a compile error, not a runtime crash 20 minutes into training. Mixed precision (f32 + bf16) is caught at compile time — no silent numerical degradation.

**2. The compiler is a training signal.** Every compilation attempt produces a fitness score (0-1) with structured error details and repair suggestions. An AI agent evolving neural architectures can use the compiler as a reward function: programs that almost type-check score higher than programs that are completely broken. This is unique — no other language compiler produces graded feedback.

**3. Dual syntax: Surf for humans, Deep for machines.** Surf is clean, readable, modern functional syntax. Deep is a regular s-expression format optimized for AI generation and structural manipulation. Programs round-trip between the two. This means an AI agent generates in Deep (low ambiguity, high regularity), a human reviews in Surf (readable, familiar), and the agent continues working in Deep.

**4. Programs as typed data.** Because Deep is homoiconic (programs are data structures), you can write programs that inspect, transform, and generate other programs — with full type checking. This enables evolutionary architecture search, neural architecture search, and neurosymbolic AI at the language level, not as a hack on top of Python strings.

**5. RISC computational model.** All tensor computation decomposes into ~12 primitive operations (inspired by tinygrad). This means the language is simple to learn (for both humans and AI), simple to optimize (the compiler sees the full computation graph), and simple to target new hardware (implementing 12 ops on a new backend is a tractable project).

**6. Differentiable by design.** Automatic differentiation (`grad`) is a first-class language feature, not a library bolted on. The type system tracks which operations are differentiable and catches errors like "you can't differentiate through argmax" at compile time, not at runtime.

---

## The Competitive Landscape

**PyTorch / JAX:** The incumbents. Python-based, dynamically typed, massive ecosystems. Chelis doesn't compete on ecosystem size — it competes on correctness, AI-native design, and the programs-as-data property. The pitch is "use Chelis for the model definition and training, Python for everything else" (there's a Python FFI). Long-term, if AI agents write most code, the ecosystem advantage of Python diminishes because agents can generate libraries.

**Mojo:** Chris Lattner's language. Positions as "Python but fast" via MLIR. Mojo is a systems language (manual memory management, low-level control) targeting performance parity with C/CUDA. Chelis is a research language targeting correctness and AI-native semantics. Different audiences, different bets. Mojo says "make Python faster"; Chelis says "make AI programs correct."

**tinygrad:** George Hotz's minimal framework. Chelis borrows tinygrad's RISC primitive philosophy (~12 ops) but adds a type system, named dimensions, AD as a language feature, and the dual-syntax architecture. tinygrad is a framework; Chelis is a language.

**Dex / Futhark:** Research languages from Google (Dex) and University of Copenhagen (Futhark). Dex pioneered algebraic effects for automatic differentiation. Futhark pioneered purity-driven GPU compilation from functional code. Chelis synthesizes ideas from both but targets AI research specifically, not general array programming. Neither Dex nor Futhark has the dual-syntax / AI-generation-first design.

**Triton / CUDA:** Kernel languages. Chelis is higher-level — it compiles down to GPU kernels (via a Futhark-style architecture) rather than requiring users to write them. The Chelis compiler owns fusion, parallelization, and memory planning.

---

## Brand and Nomenclature

### Name: Chelis
From *chelys* (Greek: turtle). The turtle metaphor runs deep:

- The language is "turtles all the way down" — AIs writing AI programs that write AI programs.
- Turtles carry their homes on their backs (self-contained, portable programs).
- Turtles are ancient and persistent (the language aims for long-term relevance, not hype cycles).

### Domain
**chelis.ch** — Swiss .ch TLD. Clean, memorable, appropriate for a language named Chelis.

### Ecosystem Nomenclature

All ecosystem naming follows a turtle/ocean theme:

| Concept | Name | Rationale |
|---|---|---|
| Surface syntax | **Surf** | The surface of the ocean — what you see |
| S-expression syntax | **Deep** | The ocean depths — what's underneath |
| Packages | **Shells** | Turtles have shells; self-contained units |
| Package registry | **Reef** (reef.chelis.ch) | Where shells live; a living ecosystem |
| Interactive mode / REPL | **Tide** | Comes and goes; interactive, rhythmic |
| Project manifest | `reef.toml` | A project's place in the reef |
| `.ch` files | Surf source | Human-readable |
| `.dp` files | Deep source | Machine-readable |
| `.chb` files | Binary AST | Compiled, distributable |

### Pronunciation
CHEL-is (like "jealous" but with a CH). Two syllables. Not "chee-lis."

---

## Project Status and Timeline

### Current Phase: 0 (Foundation)
Building the compiler from scratch in Rust. Six crates: Deep parser, Surf parser, type checker, intermediate representation, C code generation backend, CLI.

### Phase 0: Foundation
- Deep parser (s-expression parsing, round-tripping) — done or in progress
- Surf parser + desugaring to Deep
- Type checker (the core intellectual contribution)
- RISC DAG construction (the ~12 primitive operations)
- C backend (compile Chelis → C → executable)
- Automatic differentiation (grad as a compiler transform)
- **MNIST end-to-end on CPU** — the proof-of-concept: a Chelis program that trains a neural network

### Phase 1: GPU Backend
Futhark-style source-to-source compilation. The Chelis compiler outputs C host code + embedded CUDA/OpenCL kernel strings. Vendor JIT compilers handle the last mile. Same Chelis program runs on NVIDIA, AMD, Intel, Apple silicon.

### Phase 2: Language Maturity
Algebraic effects (tracking differentiability and randomness in the type system), linear types for tensors (preventing GPU memory leaks at compile time), macro system, the Tide Agent API (HTTP/JSON + MCP server for AI agent integration), `vmap` transform.

### Phase 3: Ecosystem
Package system (Shells + Reef), StableHLO backend (Google TPU access), PyTorch FX integration, Python FFI, IDE support (LSP), research-grade type system extensions (distribution types, equivariance constraints).

---

## Academic Positioning

Chelis has multiple publishable contributions:

**1. The type system.** Named tensor dimensions with compile-time checking, numeric precision types with no implicit promotion, and (Phase 2) algebraic effects for differentiability tracking. Each of these is a paper-worthy contribution. The combination is novel.

**2. The dual-syntax architecture.** No existing language has been designed with separate human-readable and machine-writable syntax layers that round-trip. This is a PL design contribution worth a workshop paper at minimum.

**3. The compiler-as-reward-function.** Using graded type checking feedback as a training signal for AI agents doing program synthesis. This bridges PL and ML research.

**4. The programs-as-typed-data property.** Homoiconic typed AST enabling evolutionary methods with type-system-guided fitness. This connects to neurosymbolic AI, NAS, and program synthesis research.

**Target venues:** NeurIPS (systems track), ICML, POPL/PLDI (PL track), ICFP, ML systems workshops (MLSys). The language touches both the PL and ML communities — positioning matters.

---

## Team

**Jeff Smith** — Founder/designer. Background: early PyTorch team at Meta/FAIR, led the CICERO diplomacy AI project and ESMFold protein structure prediction, co-authored a Nature paper on neuromotor interfaces shipping in Meta's Orion AR glasses. Two Manning books. Previously co-founded 2nd Set AI (venture-funded). Contributor to the Elixir ML stack (Nx, Axon, Livebook). Deep expertise in Elixir/BEAM and Rust.

The team is small and technical. The language is being built by coding agents under human supervision — dogfooding the thesis from day one.

---

## Key Relationships and Context

- Jeff has C-level relationships at several AI labs and research companies through prior work.
- The project intersects with but is distinct from Jeff's other work (not detailed here — Chelis stands alone).
- The language compiler is open-source (likely Apache 2.0 or MIT). Commercial strategy TBD but likely involves enterprise tooling and managed services around the language, not the language itself.

---

## What Success Looks Like

**Phase 0 milestone:** A working compiler that can train MNIST on CPU, with a GPU backend in progress. A published preprint or workshop paper on the type system or dual-syntax design. A handful of early-adopter researchers using Chelis for real experiments.

**Phase 1-2 milestone:** GPU backend working. A NeurIPS or ICML paper. A small but engaged community (dozens of active users, not thousands). Several published shells on the Reef. AI coding agents (Claude Code, etc.) can target Chelis via the Tide MCP server.

**Longer-term:** Chelis is a credible alternative to PyTorch/JAX for specific workloads (architecture search, neurosymbolic AI, precision-sensitive training). A sustainable open-source community. Commercial partnerships with AI labs or cloud providers.

The language does NOT need mass adoption to succeed. It needs to be the best tool for a specific, high-value niche (AI researchers building novel architectures) and to demonstrate that AI-native language design is a viable paradigm.
