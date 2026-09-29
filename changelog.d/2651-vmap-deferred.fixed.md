The checker now determines `vmap` batching after an enclosing recursive group
has solved the mapped function's types. Calls to inline and locally bound
untyped row lambdas infer their parameter type from a slice of the mapped
actual. A mapped result claim cannot supply an unknown row result, and
incorrect scalar claims reject during checking. See
[#2651](https://github.com/Chelis-Lang/chelis/issues/2651) and
[#1887](https://github.com/Chelis-Lang/chelis/issues/1887).
