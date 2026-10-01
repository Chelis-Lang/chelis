# Integer Dtype Spelling Migration

**Status:** Delivery plan for [#1592].
**Owning specs:** `spec/02-surf-syntax.md`, `spec/03-deep-syntax.md`,
`spec/04-type-system.md`, and `spec/10-serialization.md`.

## Scope

This change delivers one coordinated spelling migration:

- canonical Surf, canonical Deep, diagnostics, examples, the standard library,
  and checked-in language fixtures use `i8`, `i16`, `i32`, and `i64`;
- normal Surf and Deep ingress reject the retired v0.18 `int*` spellings with
  migration guidance;
- `chelis migrate surf --from 0.18` and
  `chelis migrate deep --from 0.18` perform value-preserving, syntax-aware
  rewrites; and
- existing JSON, WireDag, cache, C ABI, and external dtype vocabulary keep
  their versioned `int*` identities.

The language and interchange vocabularies are separate production components.
`Prim::name` and `Prim::parse_name` own language text.
`Prim::interchange_name` and `Prim::parse_interchange_name` own the existing
machine-facing spelling. Wire encoders and decoders must use the latter pair.

## Migration boundaries

Surf migration rewrites authored dtype positions, literal suffixes, and cast
targets without rewriting arbitrary identifiers or string contents. Deep
migration rewrites only the primitive child of `t-prim`. Normal parsing never
performs either rewrite implicitly.

This is a breaking language-text change but not a wire-schema or ABI version
change. Historical cache fixtures remain byte-identical; current compilation
tests migrate their historical source before invoking the current parser.

## Acceptance oracle

The authoritative completion oracle is:

```sh
.venv/bin/python scripts/integer_dtype_spelling_oracle.py
```

It requires canonical and rejection coverage at both ingresses, both explicit
migrations, a retired-spelling-free tracked `.ch`/`.dp` corpus, and unchanged
execution-wire, WireDag-vocabulary, and historical-cache contracts. Acceptance
is exit zero with final line:

```text
INTEGER DTYPE SPELLING ORACLE: PASS
```

The repository fast gate and hosted pull-request checks remain required
supporting evidence; they do not replace this oracle.
