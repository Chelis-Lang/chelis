CLI manifest, lowering-root, and symbolic-dimension readers and the compiler
host-runtime evaluator now consume the total Deep expression carrier directly.
Successor nodes and transitional lists preserve the same behavior, while
structural, undecodable, metadata, and malformed carriers are handled
explicitly instead of disappearing through shallow list bridges. See
[#1125](https://github.com/Chelis-Lang/chelis/issues/1125).
