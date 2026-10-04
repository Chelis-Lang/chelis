`chelis eval` no longer prints a `hint:` line after a `cast` domain trap. A
failing cast now ends on the trap line, as it does in compiled programs, which
[04-NUM-9] requires. The hint described only fractional sources, so it was
wrong for `bool` and non-finite sources. The book's type reference now
introduces the named conversions `cast_trunc`, `cast_saturate` and
`cast_wrap` as the way to narrow on purpose.
