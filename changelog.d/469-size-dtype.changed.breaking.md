An `expand` or `insert` size that names a value binding keeps that binding's own
type, so an `i32` binding is rejected with "expects an i64 size". The checker
used to retype any name in the size slot as an `i64` extent, which admitted an
`i32` binding the provenance rule happened to accept. Write the binding as
`i64` or `cast(name, i64)`. A size naming something that is both a value and
an in-scope dimension, such as a local or parameter spelled like a binder of
the enclosing definition, is a type error naming both meanings, including
programs that previously ran: `eval` and C read such a name differently and
printed different extents. Rename one of them. See
[#469](https://github.com/Chelis-Lang/chelis/issues/469).
