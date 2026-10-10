`grad` now differentiates a `match` arm whose pattern nests constructor,
record, tuple, `@`, or string-literal sub-patterns, such as
`Outer { inner: Inner { w }, s }`, in the evaluator and in C builds. The
constructors and strings of a compile-time-known value decide the arm at every
pattern depth, so a nested constructor that differs from the value's skips the
arm.
Before, `grad` refused every nested sub-pattern other than a variable or `_`,
and the same pattern had to be written as one `match` per level.

A `match` whose arm selection needs a run-time test is still refused: a
runtime scrutinee, a guarded arm, or a nested literal or constructor pattern
tested against a value known only at run time (a numeric or `bool` field, for
example). These refusals are now `unsupported:` diagnostics that cite the
open [#618](https://github.com/Chelis-Lang/chelis/issues/618), where they
cited the closed #520. See
[#3463](https://github.com/Chelis-Lang/chelis/issues/3463).
