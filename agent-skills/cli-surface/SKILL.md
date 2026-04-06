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
- `chelis surf`
- `chelis deep`
- `chelis check`
- `chelis eval`
- `chelis build`
- `chelis tide` when interactive behavior is relevant

## Validation Rules

- `fmt` output must remain parseable on the supported corpus
- `deep` then `surf` output must re-enter the compiler path cleanly where that path is promised
- machine-facing output should obey contract invariants
  - perfect success must not coexist with errors
  - emitted JSON should remain stable enough for downstream tools
- executable examples should keep working after formatting
- non-executable examples must not be mistaken for CLI proof artifacts

## Chelis-Specific Checks

- run CLI integration tests
- probe both executable and illustrative examples when the behavior touches formatting or decompilation
- compare CLI claims against actual backend/evaluator behavior when the command is a thin wrapper
