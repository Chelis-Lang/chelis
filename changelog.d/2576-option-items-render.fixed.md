A compiled program now prints an `Option` held inside a list or a data-type
value, directly or through a tuple nested in one, such as the result of
`map(fn (x: string) -> Some(x), xs)`, as `chelis eval` prints it. Previously the
program printed the root's label and aborted in `chelis_print_list` with
`validate_value rejects unknown tags`. An `Option` that is a root's own value or
one of a tuple root's components is not yet printable; see
[#2597](https://github.com/Chelis-Lang/chelis/issues/2597). See
[#2576](https://github.com/Chelis-Lang/chelis/issues/2576).
