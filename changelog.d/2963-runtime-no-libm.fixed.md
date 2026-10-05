Built executables no longer import host `exp`, `log`, `log2` or `cbrt` through
the runtime archive. The exact decimal-to-binary parse behind `to_float` and
scalar-carrier parsing, and the exact decimal rounding of `round_to`, now use
the runtime's own integer arithmetic instead of num-bigint, whose buffer-size
and root estimates called those functions, and a
test fails if the carried runtime archive imports any host math-library
transcendental. Parsed and rounded results are unchanged
([#2963](https://github.com/Chelis-Lang/chelis/issues/2963)).
