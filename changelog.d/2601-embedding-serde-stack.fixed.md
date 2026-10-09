Ordinary Serde serialization and deserialization of deeply nested embedding
`ExecutionValue` objects grow the native stack as needed, so deep JSON input
and output do not abort on a small worker stack. The execution-value JSON shape
stays the same. See
[#2601](https://github.com/Chelis-Lang/chelis/issues/2601).
