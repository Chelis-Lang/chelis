Compiler wire dimensions and literal runtime extents use a validated int64
carrier. Decoding rejects negative, oversized, and non-integer extents even
outside a complete DAG. Window sizes and vocabulary extents retain their
operation-specific positivity checks, and valid JSON values keep their existing
representation. Part of [#1288](https://github.com/Chelis-Lang/chelis/issues/1288).
