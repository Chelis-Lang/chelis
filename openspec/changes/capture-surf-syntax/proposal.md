## Why

`spec/02-surf-syntax.md` is the authoritative record of the Surf language surface: keywords,
operator precedence, module/import/export, dimensions, type signatures, blocks, records,
pattern matching, tuples, transforms, literals, and opaque types. No OpenSpec capability
currently records these as testable requirements with negative parity.

## What Changes

- Introduce a `surf-syntax` capability recording the Surf grammar, desugaring shapes, and
  the normative parse/type rules attached to each surface form.
- Capture the value/type case-split overrides, explicit declaration binders, rank
  variables, literal defaults and suffixes, contextual tensor-literal inference, and the
  opaque-type / invariant surface.
- Capture the parse-error and type-error cases (non-associative chaining, integer `/`,
  bare-transform, non-exhaustive match, bare non-tail statements, etc.).

## Capabilities

### New Capabilities
- `surf-syntax`: Surf keywords, operator precedence, module system, imports/exports,
  dimensions and rank variables, type signatures and effect annotations, blocks, effect
  handlers, macros, records, pattern matching, tuples, transforms, numeric/string literals,
  type aliases, and opaque types with declared invariants.

### Modified Capabilities

## Impact

- Source: `spec/02-surf-syntax.md` (read-only) is the authority for this capability.
- Surface: `crates/chelis-surf` (lexer/parser/desugarer/formatter) and the Surf→Deep
  desugaring contract.
- No code changes; this change records current behavior as an OpenSpec capability spec.
