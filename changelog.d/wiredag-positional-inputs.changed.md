`WireDag` admission (`WireDag::from_validated_json`, and the validation that
`Serialize` and the compiler's own WireDag production run) resolves the inputs of
Cumsum, FloatClassify, Mod, Bitwise, ReLU, ReLU-adjoint and Count nodes, and every
runtime-dimension input slot, by position instead of searching the earlier nodes.
Admitting a long chain of such nodes now costs time linear in its length: a
40,000-node ReLU chain took 22.0 G instructions and now takes 2.8 G, the same as a
chain of Exp nodes. Admitted graphs and rejection messages are unchanged. See
[#3747](https://github.com/Chelis-Lang/chelis/issues/3747).
