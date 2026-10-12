C builds preserve the runtime length and order of gradients for numeric List
fields nested inside fixed model ADTs, including tensor-valued block fields
with checked trailing extents. Unused discrete columns need no tensor shape;
live columns without an extent witness receive a typed build refusal.
Eval and C check List index bounds only on the executed branch of a gradient.
See [#3480](https://github.com/Chelis-Lang/chelis/issues/3480).
