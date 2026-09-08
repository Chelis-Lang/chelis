# Registry: public C heap-handle and lifetime callables ([05-OP-44])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-44] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| callable | exact C signature |
|---|---|
| string retain | `void chelis_string_retain(chelis_string value)` |
| string release | `void chelis_string_release(chelis_string value)` |
| tensor retain | `void chelis_tensor_retain(const chelis_tensor *tensor)` |
| tensor release | `void chelis_tensor_release(const chelis_tensor *tensor)` |
| list retain | `void chelis_list_retain(const chelis_list *list)` |
| list release | `void chelis_list_release(const chelis_list *list)` |
| tuple retain | `void chelis_tuple_retain(const chelis_tuple *tuple)` |
| tuple release | `void chelis_tuple_release(const chelis_tuple *tuple)` |
| dictionary retain | `void chelis_dict_retain(const chelis_dict *dict)` |
| dictionary release | `void chelis_dict_release(const chelis_dict *dict)` |
| ADT retain | `void chelis_adt_retain(const chelis_adt *adt)` |
| ADT release | `void chelis_adt_release(const chelis_adt *adt)` |
| option retain | `void chelis_option_retain(const chelis_option *option)` |
| option release | `void chelis_option_release(const chelis_option *option)` |
| mapped-file retain | `void chelis_mapped_file_retain(const chelis_mapped_file *mapped)` |
| mapped-file release | `void chelis_mapped_file_release(const chelis_mapped_file *mapped)` |
| value clone | `chelis_value chelis_value_clone(chelis_value value)` |
| value release | `void chelis_value_release(chelis_value value)` |
| string take into value | `chelis_value chelis_value_take_string(chelis_string value)` |
| tensor take into value | `chelis_value chelis_value_take_tensor(chelis_tensor *tensor)` |
| list take into value | `chelis_value chelis_value_take_list(chelis_list *list)` |
| tuple take into value | `chelis_value chelis_value_take_tuple(chelis_tuple *tuple)` |
| dictionary take into value | `chelis_value chelis_value_take_dict(chelis_dict *dict)` |
| ADT take into value | `chelis_value chelis_value_take_adt(chelis_adt *adt)` |
| option take into value | `chelis_value chelis_value_take_option(chelis_option *option)` |
| mapped-file take into value | `chelis_value chelis_value_take_mapped_file(chelis_mapped_file *mapped)` |
| string take out of value | `chelis_string chelis_string_take_value(chelis_value value)` |
| string borrow from value | `chelis_string chelis_string_borrow_value(chelis_value value)` |
| tensor take out of value | `chelis_tensor *chelis_tensor_take_value(chelis_value value)` |
| tensor borrow from value | `const chelis_tensor *chelis_tensor_borrow_value(chelis_value value)` |
| list take out of value | `chelis_list *chelis_list_take_value(chelis_value value)` |
| list borrow from value | `const chelis_list *chelis_list_borrow_value(chelis_value value)` |
| tuple take out of value | `chelis_tuple *chelis_tuple_take_value(chelis_value value)` |
| tuple borrow from value | `const chelis_tuple *chelis_tuple_borrow_value(chelis_value value)` |
| dictionary take out of value | `chelis_dict *chelis_dict_take_value(chelis_value value)` |
| dictionary borrow from value | `const chelis_dict *chelis_dict_borrow_value(chelis_value value)` |
| ADT take out of value | `chelis_adt *chelis_adt_take_value(chelis_value value)` |
| ADT borrow from value | `const chelis_adt *chelis_adt_borrow_value(chelis_value value)` |
| option take out of value | `chelis_option *chelis_option_take_value(chelis_value value)` |
| option borrow from value | `const chelis_option *chelis_option_borrow_value(chelis_value value)` |
| mapped-file take out of value | `chelis_mapped_file *chelis_mapped_file_take_value(chelis_value value)` |
| mapped-file borrow from value | `const chelis_mapped_file *chelis_mapped_file_borrow_value(chelis_value value)` |
| option none | `chelis_option *chelis_option_none(void)` |
| option some | `chelis_option *chelis_option_some(chelis_value value)` |
| option discriminant | `bool chelis_option_is_some(const chelis_option *option)` |
| option unwrap | `chelis_value chelis_option_unwrap(const chelis_option *option)` |
| tensor entry borrow | `chelis_tensor *chelis_tensor_entry_borrow(int32_t rank, const int64_t *shape, chelis_dtype dtype, const void *data, int64_t byte_capacity)` |
| tensor read view | `chelis_read_view chelis_tensor_read_view(const chelis_tensor *tensor)` |
| tensor begin write | `chelis_tensor_write *chelis_tensor_begin_write(chelis_tensor *tensor)` |
| tensor write view | `chelis_write_view chelis_tensor_write_view(const chelis_tensor_write *guard)` |
| tensor end write | `void chelis_tensor_end_write(chelis_tensor_write *guard)` |
| tensor repurpose | `void chelis_tensor_repurpose(chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape)` |
