`chelis check` and the C lane of `chelis build` no longer do work at each
`let` binding for every binding around it. Several lexical walks kept one table
of bound names per enclosing binding and now keep a single table: the walks
that find a body's free variables (in host lowering and the linearity check),
realizability's top-level dependency walk, and the checker's declaration-graph
and parameter-use walks. When host lowering renames a body's top-level reads,
it no longer copies the rest of the body at each binding. Host type refinement
now shares its type scope with each nested binding instead of copying it.
Results, diagnostics and emitted C are unchanged. For a definition whose body
chains 4,000 bindings, host lowering drops from about 12.6 G to 2.4 G
instructions, and from about 433 G to 2.7 G when the bindings read a top-level
value. `chelis check` drops from about 19.8 G to 13.9 G, and
`chelis build --emit-c` from about 68.5 G to 47.7 G. See
[#3797](https://github.com/Chelis-Lang/chelis/issues/3797).
