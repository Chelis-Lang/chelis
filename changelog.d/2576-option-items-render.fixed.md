A compiled program now prints an `Option` held inside a list, tuple or data-type
value, such as the result of `map(fn (x: string) -> Some(x), xs)`, as `chelis
eval` prints it. Previously the program printed the root's label and aborted in
`chelis_print_list` with `validate_value rejects unknown tags`. See
[#2576](https://github.com/Chelis-Lang/chelis/issues/2576).
