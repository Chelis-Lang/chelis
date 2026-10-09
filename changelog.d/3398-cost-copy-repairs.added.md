`chelis cost --json` reports `copy_repairs`: every copy the compiler inserts to
repair consuming fan-out in the entry program, with the span of the consume that
receives the copy and the kind and span of each later use that forced it. The text
output prints one `copy_repair` line per copy. The report is deterministic for a
given program. See [#3398](https://github.com/Chelis-Lang/chelis/issues/3398).
