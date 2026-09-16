A float literal may be written with any finite decimal body that decodes to its
value, so a constant transcribed from a reference at published precision no
longer has to be shortened by hand. `0.319381530f64`, `1.10f64` and
`687.0600000000f32` now parse, and `chelis fmt` prints the canonical shortest
spelling for them. Previously the parser rejected each one and named the
spelling to use, but because `chelis fmt` parses through the same validator it
failed with that same error rather than applying the fix, one literal at a
time. See [#2119](https://github.com/Chelis-Lang/chelis/issues/2119).

A float suffix now also accepts an integer body: `42f32` and `8000000f64` bind
the exact integer at the suffix width, and the formatter keeps the body you
wrote. This is a different literal from the decimal-bodied `42.0f32`, which
decodes a decimal instead; the compiler already implemented the distinction, but
Surf had no way to spell it. An integer body whose value rounds to infinity at
its width, such as `65520f16`, is rejected as non-finite.
