# Chelis: Complete Brand Mapping

## The Name

**Chelis** (CHEL-is). Respelling of chelys, Greek for turtle/tortoise. Six characters. The turtle that holds up the world — turtles all the way down. AI writing AI writing AI. No bottom. No human in the foundational loop.

Secondary meaning: Chelys was also the first lyre in Greek mythology — Hermes made it from a turtle shell. A language is an instrument. This language was made from the turtle.

Tertiary collision: Chelis is a genus of tiger moths (family Erebidae). Not ideal but not harmful. The moth connection is obscure enough that it won't dominate search results once the PL exists. If it comes up, lean in: "Our turtle eats your moth."

---

## File Extension

**`.ch`** — the natural choice. Two characters, matches the name's opening. The only conflict is the Ch interpreter (an embeddable C/C++ scripting engine from SoftIntegration, extremely obscure — last significant activity years ago). Go uses `.go`, Rust uses `.rs`, Zig uses `.zig`. The `.ch` extension is effectively unclaimed in the PL space.

Fallback: `.che` if `.ch` proves contested.

---

## CLI and Tooling

```
chelis build          # compile a project
chelis run            # compile and execute
chelis test           # run test suite
chelis tide           # interactive mode (Tide)
chelis deep           # emit Deep s-expression AST
chelis fmt            # format source code
chelis check          # type-check without compiling
chelis bench          # benchmark suite
chelis add <pkg>      # add dependency
chelis publish        # publish to registry
```

The binary is `chelis`. Typed once, tab-completed forever. No ambiguity with any existing CLI tool.

Build file: `reef.toml` (the project's place in the Reef).

---

## Package Ecosystem

**Registry:** `reef.chelis.ch` — where turtles gather. The Reef.

**Packages** are called **shells** (turtle shells, also: self-contained units). `chelis add shell-name`. A shell contains modules. This gives the ecosystem a natural vocabulary:

- "Have you tried the `linalg` shell?"
- "Published a new shell for probabilistic programming"
- "Shell dependencies resolved"

If "shells" feels too cute, just call them packages. The vocabulary can evolve.

---

## Standard Library Naming

```
chelis.prelude        # fundamental types, traits, pattern matching
chelis.tensor         # tensor types, named dimensions, precision
chelis.ad             # automatic differentiation (Diff effect)
chelis.random         # stochastic computations (Random effect)
chelis.device         # GPU resource management (Resource effect)
chelis.io             # file I/O, data loading
chelis.nn             # neural network primitives
chelis.optim          # optimizers
chelis.par            # concurrency primitives (par only in v1)
chelis.ffi            # Python interop (DLPack, PyO3)
```

---

## Branding Narrative

### One-liner
"The programming language designed as a substrate for machine intelligence."

### Elevator pitch
"Chelis is a functional programming language where AI writes the programs, the programs are themselves AI, and those AIs write more programs. Named for the Greek word for turtle — because it's turtles all the way down. Humans see Scala-like syntax; machines see a homoiconic s-expression core they can read, mutate, type-check, and evolve. The type system tracks tensor shapes, numeric precision, differentiability, and memory ownership at compile time. Phase 0 targets C + BLAS on CPU; Phase 1 is Futhark-style GPU compilation (own the compilation); StableHLO/FX are additive later."

### Visual identity direction
Turtle iconography. Clean, geometric, not cartoonish. A turtle shell viewed from above is a hexagonal tessellation — good geometry for a logo. Colors: deep ocean tones (the chelys is a sea creature). The icon should work at 16x16 favicon size.

### Tagline options
- "Turtles all the way down."
- "The substrate for machine intelligence."
- "Programs that write programs that write programs."

---

## Why You'd Be Happy in 5 Years

**The name doesn't constrain the language's evolution.** Unlike "Turten" (turtle + tensor), Chelis doesn't lock you into tensors as the core identity. If the language evolves into symbolic AI, program synthesis, neurosymbolic reasoning, or domains nobody has imagined yet — the name still works. "Chelis" is a vessel, not a descriptor.

**Academic credibility.** Greek-derived names signal seriousness without pretension. The NeurIPS/ICFP audience will parse "chelys = turtle" and appreciate the reference. Compare: Dex (index), Futhark (runes), Idris (Egyptian pharaoh). Chelis fits this tier.

**SEO is a blank canvas.** Right now, "chelis" returns a Nigerian education company, a Texas restaurant, and a moth genus. Within weeks of a launch, "chelis programming language" owns the top slot. Within months, just "chelis" does. This is the ideal starting position — there's nothing to displace, no incumbent to fight.

**The metaphor scales.** "Shell" for the REPL/packages, "carapace" for the type system's protective layer, "chelonian" as a demonym for community members, "plastron" for the runtime. These are available if you want them, ignorable if you don't.

**Pronunciation is settled.** CHEL-is. No ambiguity. Works in English, Mandarin, Hindi, Spanish, German, Japanese. Two syllables, hard-C start, clean ending.

**No one can take it from you.** The Nigerian education company uses "The Chelis Group" (different trademark class). The moth genus is taxonomic nomenclature, not a trademark. The namespace is genuinely clear.

---

## Why You Might Regret It

**Requires explanation.** For the first 1-2 years, every mention needs the gloss: "Chelis — Greek for turtle. Turtles all the way down." This is a front-loaded cost that decreases as the language gains recognition. Python needed the same gloss for Monty Python, Rust for oxidation, Julia for the name Julia. The cost is real but temporary.

**Doesn't signal "AI" or "ML."** The name is domain-neutral. Someone hearing "Chelis" for the first time gets no hint that it's an AI language. Counter-argument: this is a feature. "Python" doesn't signal "web scraping" or "data science" either. Names that signal a specific domain age badly as the domain evolves.

**The moth thing.** Someone will find the Chelis moth genus and make a meme. You'll see a tweet: "TIL the Chelis programming language shares its name with a genus of tiger moths." This is mildly annoying, not harmful. It gives you an excuse for good-natured community jokes.

**"Chela" proximity in Spanish.** Mexican developers might make "let's grab a chela and write some Chelis" jokes. This is... actually fine? Maybe even good? A language that inspires beer-related wordplay has worse problems.

**Six characters.** Longer than Go/Nim/Zig, same as Python/Kotlin/Elixir. Fine. Not a real concern.

---

## The 10-Year Question

In 10 years, Chelis either succeeded or it didn't. If it succeeded, the name is the name — nobody remembers or cares about the etymology of Python, Rust, Java, or Scala. The name IS the language. The turtle metaphor becomes invisible background, surfacing only in conference talks and documentation headers.

If it failed, the name is irrelevant. No language ever failed because of its name.

The question is whether the name creates unnecessary friction during the critical adoption window (years 1-3). The answer: less friction than any other surviving candidate. Kurnel has permanent homophone friction. Turt has crowded namespace friction. Chelis has one-time explanation friction that decreases with every new user who learns the word.

---

## Compared to the Alternatives

| Factor | Chelis | Kurnel | Turt |
|---|---|---|---|
| Pronunciation clarity | Unambiguous | = "kernel" (permanent) | Unambiguous |
| SEO starting position | Blank (ideal) | Polluted by "kernel" | Crowded (turt.io, Turtl) |
| Explanation cost | One-time ("Greek for turtle") | One-time + permanent spelling | Low ("short for turtle") |
| Domain evolution | Unconstrained | Unconstrained | Unconstrained |
| Academic tone | Strong | Strong | Informal |
| Autocomplete risk | → "chellis" (moth, harmless) | → "kernel" (fatal) | → "turtle" (noisy) |
| File extension | .ch (clean) | .kn (clean) | .trt (clean) |
| 10-year durability | High | Medium (homophone ages badly) | Medium (informality ages badly) |
