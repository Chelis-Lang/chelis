The ownership verifier's `inconsistent live owners` error now carries the Surf
source span of each owner it names, completing chelis#2122's ask that the
message locate the disagreement in source rather than by owner id alone:

```text
block b1 in `roots` is reached with inconsistent live owners: live only on this path: %2[carry]@surf:120..135; live only on the path already verified: none (2 live here, 1 earlier)
```

Ownership lowering records the span of the expression an owner is minted for,
taken from `HostExpr::span_id` per `spec/design/chelis_span_survival.md`. A
node without its own span keeps the nearest enclosing one, since that region
still locates the owner. An owner minted outside any expression, such as a unit
parameter, prints as before with no span and no placeholder.

The same label is used by the ownership dump, so spans appear there too.
