**Deep nominal arguments now use the recursive type grammar at ordinary
`.dp` ingress (chelis#1125, part of chelis#731).** `chelis surf` and
`chelis validate --deep` reject a rank spread such as `Rows[..r]` instead of
accepting a `d-rank` below `t-adt`. Concrete dimension arguments remain
valid, and rank spreads remain valid in tensor-axis slots.
