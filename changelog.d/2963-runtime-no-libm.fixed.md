Built executables no longer import host `exp`, `log`, `log2` or `cbrt` through
the runtime archive. The exact decimal-to-binary parse behind `to_float` and
scalar-carrier parsing now uses the runtime's own integer arithmetic instead of
num-bigint, whose buffer-size and root estimates called those functions, and a
test fails if the carried runtime archive imports any host math-library
transcendental. Parsed results are unchanged
([#2963](https://github.com/Chelis-Lang/chelis/issues/2963)).
