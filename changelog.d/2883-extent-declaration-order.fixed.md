`chelis build` compiles a program that projects the record half of a masked
`(Col[n], tensor[n, bool])` result, such as `col_xs(masked(t).0)` where `masked`
builds the record with `where` and helper results. Each helper's dimension
binders now take the caller's `n`, which its parameter declares before any guard
reads it. Previously a `--target c` build succeeded and wrote C that failed to
compile with an undeclared identifier. The C emitter, for `--target c` and for
the host-side tensor helpers of a `--target hip` build, still refuses any
compiled tensor function that would read a run-time extent before declaring it,
with a typed "rendered before it is declared" rejection under
[#1277](https://github.com/Chelis-Lang/chelis/issues/1277), rather than writing C
that does not compile. See
[#2883](https://github.com/Chelis-Lang/chelis/issues/2883).
