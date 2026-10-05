`char_code`, `char_from_code` and `clamp` now report a domain failure as a
numeric trap in [04-NUM-9]'s form, with identical text in `chelis eval` and
compiled programs. A context line names the offending value, for example
`char_code operand has 2 Unicode scalar values, expected exactly one`. The
trap line follows it, for example `numeric trap: domain in char_code at i64`
or `numeric trap: domain in clamp at f32`. The failure used to be a
`Domain: ...` line with a spec citation. A `clamp` bound whose runtime shape
is neither a scalar nor the operand's shape now names both shapes.
