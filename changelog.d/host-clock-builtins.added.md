Two host clock builtins, `clock_wall_read()` and `clock_monotonic_read()`,
return `(seconds, nanoseconds)` as `(i64, i64)` from a single host reading,
with nanoseconds in `[0, 10^9)`. The wall clock is on the POSIX timescale;
the monotonic clock never runs backwards and has an unspecified origin. Both
carry the `IO` effect, so a function declared without `IO` cannot call them.
A host clock error, or a reading whose seconds lie outside
`-377705030401..253402214400`, fails with `<builtin>: io: <detail>`.
`chelis eval` and `chelis test` run them through the evaluator's
policy-checked system port. Compiled builds reject them with the same
diagnostic as `process_run` until compiled host execution lands
([#1297](https://github.com/Chelis-Lang/chelis/issues/1297)). [05-OP-75]
in `spec/05-risc-primitives.md` defines them.
