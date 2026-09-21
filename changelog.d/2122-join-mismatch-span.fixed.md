The ownership verifier's `inconsistent live owners` error now carries the Surf
source span of each owner it names, completing chelis#2122's ask that the
message locate the disagreement in source rather than by owner id alone:

```text
block b1 in `roots` is reached with inconsistent live owners: live only on this path: %2[carry]@surf:120..135; live only on the path already verified: none (2 live here, 1 earlier)
```

Ownership lowering records the span of the expression an owner is minted for,
taken from `HostExpr::span_id` per `spec/design/chelis_span_survival.md`, on
every path that lowers an expression: ordinary expressions, and the arguments
of a direct call to a user-defined `def`, including an argument admitted as a
raw literal. An owner minted for a function reference takes the enclosing
region instead, because the value outlives the expression that produced it
(chelis#2319).

A node without its own span keeps the nearest enclosing one, since that region
still locates the owner. An owner minted outside any expression, such as a unit
parameter, prints as before with no span and no placeholder. A span id is an
opaque producer-issued string, so one that is empty, or that carries whitespace
or the `;` and `,` separators this single-line message is built from, renders as
no span rather than as corrupted output.

The ownership dump shares the same label, so owner-definition lines show spans
there too. Lines for applications, copies and block parameters print the bare
owner id as before.
