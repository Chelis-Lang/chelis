# Black-Scholes Octant Fixture

Span-attributed Deep + spans manifest produced by Octant from
`references/black_scholes_call.tex` (the canonical hero-example LaTeX in the
Octant repo). Used by the chelis span-survival pipeline tests as the
load-bearing audit canary: every span ID in `call_price.dp` must round-trip
through the chelis compile pipeline and surface in the generated C output.

## Files

- `call_price.dp` — span-attributed Deep AST (20 nodes, IDs `n_001`..`n_020`)
- `call_price.spans.json` — sidecar mapping each `deep_node_id` to its LaTeX byte range
- `README.md` — this file

## Regeneration

These artifacts are produced by Octant's `translate` subcommand. They are
committed as static fixtures so chelis tests do not depend on Octant being
installed. To regenerate:

```
octant translate <octant>/references/black_scholes_call.tex \
  --output call_price.dp \
  --spans call_price.spans.json
```

Then sanitize `source` in `call_price.spans.json` from an absolute path to
`references/black_scholes_call.tex` (relative to the Octant repo). The
`source_hash` field stays as Octant emitted it.

## Spec

See `spec/design/chelis_span_survival.md` for the contract this fixture
exercises and `spec/03-deep-syntax.md` §1.1 for the documented `span` metadata
key.
