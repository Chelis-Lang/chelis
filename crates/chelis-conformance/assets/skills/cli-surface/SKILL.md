---
name: cli-surface
description: Use when changing or validating Chelis CLI behavior. Focuses on corpus-based testing for chelis fmt, surf, deep, check, eval, build, and tide, plus machine-facing output invariants.
---

# CLI Surface

Use this skill for any change or validation pass involving `chelis` commands.

## Default Principle

Do not trust a CLI surface that only passed one happy-path test.
Exercise it across a corpus and check both parseability and semantic invariants.

## Required Surfaces

- `chelis fmt`
- `chelis lint`
- `chelis surf`
- `chelis deep`
- `chelis check`
- `chelis validate`
- `chelis eval`
- `chelis build`
- `chelis tide` when interactive behavior is relevant
- The **style gate** (the implicit `fmt --check` + `lint --check` step
  that runs inside `build`/`check`/`validate`/`eval --file`)

## Validation Rules

- `fmt` output must remain parseable on the supported corpus
- `deep` then `surf` output must re-enter the compiler path cleanly where that path is promised
- machine-facing output should obey contract invariants
  - perfect success must not coexist with errors
  - emitted JSON should remain stable enough for downstream tools
- executable examples should keep working after formatting
- non-executable examples must not be mistaken for CLI proof artifacts
- the style gate must fail by default on non-canonical or
  lint-violating input, must succeed with `--allow-style-violations`
  (with a stderr warning), and must be silently disabled by
  `CHELIS_STYLE_GATE_DISABLE=1` for the test corpus only — verify
  each branch when changing the gate

## Chelis-Specific Checks

- run CLI integration tests, including
  `crates/chelis-cli/tests/style_gate.rs` for the gate itself
- probe both executable and illustrative examples when the behavior touches formatting or decompilation
- compare CLI claims against actual backend/evaluator behavior when the command is a thin wrapper
- when adding a new lint rule, register it in
  `crates/chelis-lint/src/registry.rs` and confirm the gate picks it
  up by running `chelis check` on a deliberately violating fixture

## Authoritative References

- User-facing CLI doc: `docs/book/src/cli.md`
- Style guide spec: `spec/01-nomenclature.md`
- Gate implementation: `crates/chelis-cli/src/style_gate.rs`
- Gate behavior tests: `crates/chelis-cli/tests/style_gate.rs`
