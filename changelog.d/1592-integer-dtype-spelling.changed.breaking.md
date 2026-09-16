Chelis source and canonical Deep now spell signed integer dtypes `i8`, `i16`,
`i32`, and `i64`. The retired `int8`, `int16`, `int32`, and `int64` language
spellings are rejected with migration guidance. Run
`chelis migrate surf --from 0.18 --inplace` for `.ch` source and
`chelis migrate deep --from 0.18 --inplace` for `.dp` source. Existing
compiler-API JSON, WireDag, cache, and C ABI dtype names remain unchanged.
See [#1592](https://github.com/Chelis-Lang/chelis/issues/1592).
