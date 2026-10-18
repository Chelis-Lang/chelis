`chelis fmt` on Deep (`.dp`) files, and `chelis deep` output, cost time in
proportion to the printed text on deep nesting. The printer copied every
nested line once per enclosing level and rendered each node's whole subtree
to measure its width, which was cubic in the depth: `fmt --check` on a tuple
nested 2,000 deep took 22 G instructions and 643 MB, and the style gate that
`check` and `build` run paid the same. It now takes 0.24 G and 33 MB, and a
16,000-deep tuple formats in 5.9 G. The printed text is unchanged except
where a string holds a raw control character
([#3852](https://github.com/Chelis-Lang/chelis/issues/3852)).
