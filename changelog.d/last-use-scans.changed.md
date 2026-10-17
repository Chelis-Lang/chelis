`chelis build` spends less time placing and checking scope-exit drops. The
last-use scheduler and its canonical re-check now answer each owner's
questions (where it is defined, which operations, terminators and edges use
it, which edges enter a block, which operation a drop follows) from tables
built once per function body, answer post-dominance from a post-dominator tree
instead of a search per owner, and re-anchor scheduled drops in the host-site
map from one index instead of a scan of the map per drop. A body with 2,000
bindings read by one final list retires about 30% fewer instructions, and the
ownership scheduler drops from about a quarter of that build's samples to
under 1%. Programs, diagnostics and emitted C are unchanged. See
[#3755](https://github.com/Chelis-Lang/chelis/issues/3755).
