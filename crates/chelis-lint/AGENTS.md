# chelis-lint crate guardrails

These instructions refine the repository-root `AGENTS.md` for changes under
`crates/chelis-lint/`.

## Rule registration protocol

1. Write positive and negative tests from `spec/01-nomenclature.md` before the
   implementation. Include the corresponding failure case for every accepted
   case.
2. Add the rule module under `src/rules/` and export it from `src/rules/mod.rs`.
3. Register blocking rules in `registry::all_rules`; register warning or
   advisory rules in `registry::non_blocking_rules`. A rule is not shipped
   merely because its module compiles.
4. Lock CLI behavior when the rule or policy changes the public `chelis lint`
   surface. Test output shape and exit status, not only direct `Rule::check`.
5. Update the owning spec, docs, examples or fixtures, and changelog in the
   same change when behavior changes.

## Traversal and corpus state

- `walker::walk` is the only recursive filesystem discovery mechanism. A rule
  must not create its own tree walker or consult `.gitignore`, machine-local
  ignores, or hard-coded skip paths.
- Use `Rule::prepare_run` for corpus-wide state. Build it once from the
  canonical entries supplied by the lint driver, keep it invocation-local,
  and consume it through `check_prepared`.
- Whole-tree exclusions belong in the shipped or repository
  `chelis-lint.toml` policy with a valid class and spec cross-reference.
  Rule-specific exceptions and inline `allow`/`keep` remain post-traversal.
- Policy discovery and referenced specs fail closed. Do not add silent
  fallbacks for malformed, unreadable, escaping, or machine-local inputs.
- Ancillary files consulted by a rule must come from the canonical entry set
  or pass a parent-aware `TraversalPolicy` admission check. An excluded path
  must not influence an admitted entry indirectly.

## Minimum focused gate

Run the `chelis-lint` crate tests, the affected standalone CLI integration
test, `cargo fmt --all -- --check`, and clippy with `-D warnings` in an isolated
`CARGO_TARGET_DIR`.
