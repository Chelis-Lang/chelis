A Deep program now has one in-memory representation whichever door it enters
through. Surf desugaring, the checker's entries, the prover and the CLI no
longer rewrite trees into a second, transitional list spelling, so a check or
transformation that previously read only one spelling now runs whether the
program came from a `.ch` file, from the `.dp` text it prints to, through
`chelis check`, or through `chelis prove`. Cached compiled library and stdlib
contexts from earlier builds are rebuilt, because their format versions
advance. See [#1125](https://github.com/Chelis-Lang/chelis/issues/1125) and
[#1029](https://github.com/Chelis-Lang/chelis/issues/1029).
