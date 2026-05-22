# module-pascal-components-flags-single-words: §6.3 flags real single English words as suspected compounds

**Status:** open; immediate fix applied (rule demoted to advisory)
**Filed:** 2026-05-22
**Owning phase:** chelis-lint (`crates/chelis-lint/src/rules/module_pascal_components.rs`)
**Discovered by:** Calcify shell `chelis lint --check .` reporting 30 blocking
`module-pascal-components` errors

## Summary

The `module-pascal-components` rule (§6.3) is meant to catch unCapitalized
compounds: `Examplerootfind` should be `ExampleRootFind`, `Hellotensor` should
be `HelloTensor`. Its detection heuristic is "a component with a long run of
lowercase letters (>= 7) and no internal capital is a suspected multi-word
compound". That heuristic cannot distinguish a flattened compound from an
ordinary long English word, so it false-positives on real single words such
as `Optimization`, `Comprehension`, `Conditional`, `Exception`, `Translation`,
and `Translated`. Every one of those is a single PascalCase English word and
fully conformant with §6.3, yet the rule fires on each.

## The rule's intent

§6.3 requires every component of a module ladder to be PascalCase per word,
including each word inside a compound. The genuine defect the rule targets is
the sibling-sweep mutation where a compound's internal capital is flattened:
`ExampleRootFind` becomes `Examplerootfind`. Catching that is correct and
worth keeping.

## The failure mode

The heuristic at `component_violation` ("max lowercase run >= 7 with no
internal capital") flags *any* sufficiently long single word. It has no way
to tell `Examplerootfind` (a real flattened compound) from `Optimization`
(a real single word) because both are "long, all-lowercase after the leading
cap". The rule's accuracy depends entirely on the `KNOWN_SINGLE_WORDS`
allowlist absorbing every long English word that ever appears as a module
component.

## Why the allowlist is not a solution

`KNOWN_SINGLE_WORDS` requires enumerating every long English word that any
shell in the ecosystem might ever use as a module component. The maintainers
have already been doing exactly this, ad-hoc, commit by commit: the list has
accreted `Nautilus`, `Distributions`, `Activation`, `Embedding`, `Generate`,
`Schedule`, `Capstone`, `Hydronnx`, `Linearity`, `Hypothesis`, `Integration`,
`Optimize`, `Classification`, `Regression`, and roughly forty more, each added
because some real word tripped the heuristic. This change set adds five more
(`Calcify`, `Translation`, `Comprehension`, `Conditional`, `Exception`) for
the Calcify shell. There is no end to this list: it is the English language.

## The structural problem

A lint rule whose correctness depends on a hand-maintained English dictionary
is going to keep failing. Every new shell, every new module name, every new
long word is a latent false positive that surfaces as a blocking CI error in
a downstream repo that did nothing wrong. The allowlist is not converging on
a fixed set; it grows monotonically with the corpus.

## Suggested resolutions, ordered by ambition

1. **Demote to advisory.** The rule still runs and still reports, but it does
   not fail `chelis lint --check` or the style gate. A false positive then
   costs a reviewer one glance instead of a blocked build. *Applied in this
   change set* as the immediate fix: `module_pascal_components` moved from
   `registry::all_rules` to `registry::non_blocking_rules`, with
   `severity()` returning `Severity::Advisory`.
2. **Loosen the heuristic to require a real compound-boundary signal.** Only
   flag a component when there is positive evidence of a flattened compound,
   for example when the component matches the lowercase-after-first form of a
   name in `KNOWN_PASCAL_COMPOUNDS` (this check already exists and is precise),
   or when the component is a recognized compound stem. Drop the bare
   long-lowercase-run heuristic, which has no positive signal at all.
3. **Consult a word list.** Ship or reference an English word list and only
   flag a long lowercase component when it is *not* a known word. This trades
   the hand-maintained allowlist for a maintained dictionary, which at least
   does converge, but couples the linter to a dictionary artifact.
4. **Require explicit compound delimiters.** Make §6.3 require that compounds
   be written with internal capitals and treat the absence of internal
   capitals on a long component as conformant-by-construction (a single word).
   This removes the heuristic entirely; the rule would then only catch the
   `KNOWN_PASCAL_COMPOUNDS` flattening case, which is detectable precisely.

Resolution (1) is applied now. (2) is the recommended structural follow-up:
it keeps the rule useful (it still catches the genuine sibling-sweep defect)
without needing an English dictionary.

## Code references

- `crates/chelis-lint/src/rules/module_pascal_components.rs`
  - `component_violation` — the long-lowercase-run heuristic that
    false-positives.
  - `KNOWN_SINGLE_WORDS` — the ad-hoc allowlist that absorbs the false
    positives.
  - `KNOWN_PASCAL_COMPOUNDS` / `known_compound_lowercase_form` — the precise
    compound-detection path that resolution (2) would build on.
- `crates/chelis-lint/src/registry.rs` — `module_pascal_components` now
  registered in `non_blocking_rules`.
