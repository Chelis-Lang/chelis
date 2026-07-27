## Why

`spec/03-deep-syntax.md` is the authoritative record of Deep, the primary machine interface:
the universal 3-tuple node structure, the closed 62-tag vocabulary, metadata contract,
canonical form, and validation rules. No OpenSpec capability records these as testable
requirements with negative parity.

## What Changes

- Introduce a `deep-syntax` capability recording the node structure, metadata keys, span
  trust-boundary rules, closed tag vocabulary, module identity, type/dimension resolution,
  built-in scope, application/pipe semantics, canonical form, and validation rules.
- Capture the parse-error and type-error cases (unknown tags, arity violations, duplicate
  modules, reserved linker names, malformed type expressions, forbidden span characters).

## Capabilities

### New Capabilities
- `deep-syntax`: Deep AST node structure and metadata, span/producer-string trust boundary,
  the closed 62-tag vocabulary, module identity, type-expression and dimension resolution,
  built-in scope, function application and pipe semantics, canonical form and literal
  normalization, and structural/arity/vocabulary validation.

### Modified Capabilities

## Impact

- Source: `spec/03-deep-syntax.md` (read-only) is the authority for this capability.
- Surface: `crates/chelis-deep` (lexer/parser/validate/printer), `chelis-ir` newtypes,
  `chelis validate --deep`.
- No code changes; this change records current behavior as an OpenSpec capability spec.
