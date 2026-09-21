`chelis build --target c` no longer emits a public entry for a definition that
performs a `Random` effect no handler inside it discharges. The public entry ABI
carries no RNG frame, so the generated wrapper could only hand the body an
inactive RNG state: the tensor-DAG lane then returned values drawn against it,
unreproducible from any `with seed(...)` the author wrote, and the C host lane
aborted at the first call.

Such a definition is now emitted for the program's own seeded callers but is not
published, and the withheld wrapper records why in the generated C. When the
program opens no `with seed(...)` region anywhere, nothing in it can call the
definition under a handler, so the build is rejected with a diagnostic naming
the cause and both remedies rather than withholding it silently. A
definition carrying an undischarged `Random` remains an ordinary function whose
caller supplies the handler, exactly as the standard library's random helpers
do, and `chelis check` continues to accept it.

This adds `spec/04-type-system.md` [04-EFF-3], which states the rule for every
emission path. See [#2318](https://github.com/Chelis-Lang/chelis/issues/2318).
