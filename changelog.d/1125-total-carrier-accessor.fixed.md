Deep AST readers can use the public carrier-total `Expr::carrier` and
`ExprCarrier` interface, which distinguishes decoded nodes, structural lists,
undecodable heads, leaves, metadata carriers, and malformed legacy lists. The
tensor-precision checker now reads successor `Node` trees directly instead of
silently skipping nested precision nodes through a shallow legacy-list bridge,
preserves structural-list role across legacy normalization, retains span IDs on
undecodable legacy heads, and uses order-independent declaration ownership for
one unsupported-precision diagnostic per affected declaration occurrence.
Deep `prove` also reports these checker defects when strict Deep validation
rejects the same input. See
[#1125](https://github.com/Chelis-Lang/chelis/issues/1125).
