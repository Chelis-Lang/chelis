---
name: example-corpus
description: Use when adding, moving, or validating Chelis example programs, and before writing any Surf (`.ch`) source, including fixtures and probes. Carries the Surf style rules, the parse-breaking spellings, and the Deep AST contract; preserves the executable-vs-illustrative split and keeps example-related tests honest.
---

# Example Corpus

Use this skill whenever touching `examples/` or tests/docs that reference examples.

## Policy

- `examples/` is the executable corpus.
- `examples/illustrative/` is for syntax/design examples that are not on the executable
  phase path.
- Make this split explicit early in the phase instead of retrofitting it after examples
  have already been used as proof.

## Rules

- If an example is used as phase proof, it belongs in `examples/`.
- If it intentionally uses unsupported or future-phase constructs, move it to
  `examples/illustrative/`.
- Update tests when moving examples so executable-corpus tests do not silently become parse-only.

## Verification

For executable examples:

- parse succeeds
- `chelis fmt --inplace` preserves validity
- `chelis check` returns score `1.0` with no errors

For illustrative examples:

- parsing/round-trip expectations may still apply
- docs and tests must not imply they are executable phase proof

## Writing Surf

These rules apply to every `.ch` you write in this repository: examples,
fixtures, and probes alike. The authority is `spec/02-surf-syntax.md` §0.1
(canonical forms and the bidirectional contract); §P10-P12 define the wider
set of input spellings the parser still accepts but the formatter rewrites.
`spec/01-nomenclature.md` is the rule spec behind `chelis lint`, and
`crates/chelis-lint/src/rules/` is its executable enforcement; the canonical
formatter is `chelis_surf::format` for `.ch` and `chelis_deep::printer` for `.dp`.
Run `chelis fmt --inplace <file>` and `chelis lint --check` before pushing.

Spellings that are hard errors, not style:

- A nullary definition needs `()`: `def name() -> T`, not `def name -> T`.
  This is the highest-frequency breakage; expect it on any branch predating
  chelis#1031 (`expected function parameter list `()`, found Arrow`).
- Applying a returned value needs explicit grouping: `(f(x))(y)`. Ungrouped
  `f(x)(y)` is rejected; Chelis has flat multi-argument application and no
  implicit currying, and juxtaposition stays rejected.
- The unit value is `()`, the unit type is `unit`, and a singleton tuple is
  `(x,)`; that comma is semantic.
- Effects carry exact casing: `Diff`, `Accum`, `IO`, `Test`,
  `Resource(...)`.
- Randomness has no effect and no handler: a draw takes an explicit key,
  `dropout(key_from_seed(42i64), x, 0.5f32)`. `with seed(...)` and
  `! { Random }` are the typed `RetiredRandomness` parse error.
- Non-primary transform arguments are named: `grad(f, wrt=x)`,
  `vmap(f, axis=n)`; axis zero is bare `vmap(f)`.
- `sum`, `cumsum`, `trace`, and `einsum` over `i8` or `i16` return `i32`
  (`spec/04` §5.7.1); declare the result as `i32`, or narrow it with an
  explicit `cast`.
- Pipe stages use first-argument insertion: `x |> f(y)` means `f(x, y)`. Use
  `x |> fn (v) -> f(y, v)` when the piped value belongs in a later position.

Style the lint rules enforce:

- prefer `def ... -> T = ...` over `def ... : T = ...` (`surf-def-arrow-form`,
  §3.5)
- type identifiers are PascalCase (`surf-type-pascal-case`, §3.1);
  function/value identifiers are snake_case (`surf-value-snake-case`, §3.2)
- functions carrying the `Test` effect are named `test_*` or `example_*`
  (`surf-test-name-prefix`, §10.1)
- zero-arity decoration is dropped in types and kept in expressions: `Ctor()`
  is a Deep `app`, bare `Ctor` is a Deep `var`, and a zero-field record keeps
  `{}`. `! {}` is a declared-pure upper bound, distinct from an omitted effect
  clause.
- canonical output omits trailing separators and prints the canonical literal
  spelling (shortest round-trippable float, no digit separators or redundant
  zeroes)

Style the formatter does not enforce:

- put types on function parameters, not on load-style top-level bindings
- use symbolic dimensions such as `batch` and `seq` for runtime-varying axes;
  keep fixed architecture dimensions concrete
- do not annotate intermediate expressions when inference already determines
  the type; keep meaningful intermediates like `h1`, `logits`, `probs`, and
  `loss`
- combine short tensor operations when the composed expression is clearer
  than over-decomposed single-op bindings
- treat decompiler-generated verbose load chains and checker-inserted
  ascriptions as debug output, not example style
- user code typically writes neither explicit `&` nor explicit `copy()` for
  fan-out into read-only primitives; auto-borrow handles it. Write `&x` when
  an exported API or dense signature benefits from clarity, and `copy(x)`
  only when forking ownership for downstream consumption. Existing fixtures
  and migration baselines may keep explicit `copy()` or `drop()` calls when
  they prove compatibility or preserve baseline evidence.
- lowered IR carries compiler-inserted `Copy` and `Drop` nodes for implicit
  linearity. If auto-copy/auto-drop produces unexpected IR, treat it as a
  structural blocker and escalate against `spec/design/implicit_linearity.md`
  rather than papering over it as a routine fixture bug.

## Deep AST

- Every Deep node is a 3-tuple: `(tag {} children...)`, with the metadata map
  always present at element 1. The vocabulary is a closed set of 62 tags; see
  `spec/03-deep-syntax.md`. Function application is `app`, names are `var`,
  literals are `lit`, and RISC primitives are built-in functions, not tags.
- Decompilation routes through a typed Deep-to-Surf resugaring boundary
  (chelis#1031) with a total disposition for every public Deep tag, printed by
  the shared Surf printer. It fails closed on malformed Deep, invalid surface
  identifiers, unknown `surf_*` metadata, incompatible literal metadata, and
  non-finite constructed values; a resugaring failure is a real defect, not
  output to work around. Canonical Deep and canonical Surf are two
  representations of one public language, bound by the three executable laws
  in `spec/02-surf-syntax.md` §0.1 (`desugar(resugar(·))`, formatter
  idempotence, and the semantic retraction). If you change either printer or
  the desugarer, those laws are the oracle.
