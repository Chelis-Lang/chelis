**checker/CLI: nominal type applications now enforce parameter
kinds and concrete dimensions (chelis#1247, chelis#1258).** Declaration
headers acquire checker-owned `Type`/`Dimension` kinds by a deterministic
fixed point, so integer arguments reject at ordinary type parameters and
constrain dimension parameters instead of becoming fresh wildcards.
`Column[3]` now rejects a two-element constructor with
`DimensionMismatch`; `Option[3]` rejects with `TypeMismatch`. Surf emits a
structural `d-lit` for the integer argument, restoring `deep`/`surf` and
`migrate surf` round trips, while aliases, records, matches, eval, generated
C, `chelis check`, and `chelis test` preserve the same checked extent.
Structured inferred JSON wraps nominal dimensions as
`{kind:"dimension", dim:{...}}` without changing existing type-argument
objects. Serialized compiler state changes accordingly: compiled-context
cache v12 → v13, stdlib cache v8 → v9, library cache v5 → v6,
and Reef prepared-graph cache v2 → v3; stale entries rebuild
automatically.
