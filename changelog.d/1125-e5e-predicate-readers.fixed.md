Invariant-predicate checking now consumes the total Deep expression carrier
directly, preserving successor and legacy behavior without rebuilding nodes
through `Node::to_list`.
See [#1125](https://github.com/Chelis-Lang/chelis/issues/1125).
