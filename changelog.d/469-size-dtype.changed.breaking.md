An `expand` or `insert` size that names a value binding keeps that binding's own
type, so an `i32` binding is rejected with "expects an i64 size". The checker
used to retype any name in the size slot as an `i64` extent, which admitted an
`i32` binding the provenance rule happened to accept. Write the binding as
`i64` or `cast(name, i64)`. See
[#469](https://github.com/Chelis-Lang/chelis/issues/469).
