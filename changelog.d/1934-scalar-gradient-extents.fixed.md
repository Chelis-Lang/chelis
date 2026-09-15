Compiled C gradients accept primitive scalar targets alongside tensor and
aggregate targets, retain forward extent failures even for zero cotangents,
and reconstruct complete results in written target order, including repeated
selectors. Primitive scalars and rank-zero tensor results keep their distinct
public types. Fixes [#1934](https://github.com/Chelis-Lang/chelis/issues/1934).
