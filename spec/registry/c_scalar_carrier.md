# Registry: public C scalar and dtype carriers ([05-OP-31])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-31] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| callable | exact C signature |
|---|---|
| dtype storage size | `int64_t chelis_dtype_size(chelis_dtype dtype)` |
| scalar validation/construction | `chelis_scalar chelis_scalar_from_bits(chelis_dtype dtype, uint64_t bits)` |
| value boxing | `chelis_value chelis_value_from_scalar(chelis_scalar value)` |
| value extraction | `chelis_scalar chelis_value_as_scalar(chelis_value value)` |
| rank-zero tensor construction | `chelis_tensor *chelis_scalar_tensor(chelis_scalar value)` |
| rank-zero tensor extraction | `chelis_scalar chelis_tensor_to_scalar(const chelis_tensor *tensor)` |
| tensor fill | `void chelis_fill_scalar(chelis_tensor *tensor, chelis_scalar value)` |
| scalar rendering | `chelis_string chelis_string_from_scalar(chelis_scalar value)` |
| scalar parsing | `chelis_option_scalar chelis_parse_scalar(chelis_string text, chelis_dtype dtype)` |
| exact dictionary scalar lookup | `chelis_option_scalar chelis_dict_get_scalar(const chelis_dict *dict, chelis_value key, chelis_dtype dtype)` |
