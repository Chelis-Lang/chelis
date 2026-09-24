Selecting evaluation roots now scopes which nodes the evaluator may seed as
live. Lowering makes **every** top-level `def` a DAG root, and a def's
parameters become `Load` nodes built by the same code that builds a genuine
entry input, so an uncalled `def g(z)` leaves `z` in the DAG
indistinguishable from the `main(x)` being run. Value reachability ignores
it; a seed does not, because marking nodes the roots cannot reach is what a
seed is for. Any seed therefore pulled another declaration's subgraph into
the run, and the strict-load check then demanded that declaration's
parameters — the `missing required input` shape of
[#991](https://github.com/Chelis-Lang/chelis/issues/991), re-armed. A node
owned only by a root the caller did not select is now excluded from seeding;
one the selection also reaches is never dropped, and selecting every root or
none is a no-op. This scopes the `[05-OP-68]` abort seed only: a scoped draw
is already selected by the entered-region set its caller derives from those
same roots, and a draw of a region a selected root enters must still execute
for its handler's ordinal. The scoping covers the root-reachable half: a
node an unselected declaration *discards* is an ancestor of no root, so it
is owned by no unselected root either and still seeds — separating that from
a discarded node in the selected root would need per-declaration attribution
the DAG does not carry. Part of
[#2476](https://github.com/Chelis-Lang/chelis/issues/2476).
