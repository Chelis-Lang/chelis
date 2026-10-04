In compiled C programs, `to_float` and `to_int` now follow [05-OP-59] and agree
with `chelis eval`. A finite `to_float` spelling that overflows f64, such as
`"1e400"`, yields a signed infinity instead of `None`. `inf`, `infinity`, and
`nan` are accepted in any letter case with an optional sign. Both parsers trim
every surrounding Unicode whitespace character, including newlines and
non-breaking spaces, rather than only spaces and tabs. Previously a compiled
program that substituted a default for `None` silently computed with that
default on these inputs. The runtime gains the C entries `chelis_to_int` and
`chelis_to_float`, which share one implementation with the evaluator;
`chelis_parse_scalar` keeps its [05-OP-31] contract. See
[#2870](https://github.com/Chelis-Lang/chelis/issues/2870).
