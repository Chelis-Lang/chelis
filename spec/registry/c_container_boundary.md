# Registry: public C container and boundary callables ([05-OP-32])

This file is normative-tier content under the AGENTS.md Documentation
Authority rules: it is incorporated by reference into
`spec/05-risc-primitives.md` [05-OP-32] and is amended only as a
numbered-spec change under the same review discipline. Rows are keyed by
identity; row order is not semantic and no ordinal is part of any identity.

| callable | exact C signature |
|---|---|
| length-aware string construction | `chelis_string chelis_string_from_utf8(const uint8_t *value, int64_t len)` |
| character code | `int64_t chelis_char_code(chelis_string value)` |
| character from code | `chelis_string chelis_char_from_code(int64_t value)` |
| string print | `void chelis_print_string(chelis_string value)` |
| string length | `int64_t chelis_string_len(chelis_string value)` |
| string slice | `chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len)` |
| list length | `int64_t chelis_list_len(const chelis_list *list)` |
| list construction | `chelis_list *chelis_list_from_values(const chelis_value *items, int64_t len)` |
| list index | `chelis_value chelis_list_index(const chelis_list *list, int64_t index)` |
| list take | `chelis_list *chelis_list_take(const chelis_list *list, int64_t count)` |
| list drop | `chelis_list *chelis_list_drop(const chelis_list *list, int64_t count)` |
| list chunk | `chelis_list *chelis_list_chunk(const chelis_list *list, int64_t size)` |
| integer range | `chelis_list *chelis_range_i64(int64_t start, int64_t end)` |
| list enumerate | `chelis_list *chelis_list_enumerate(const chelis_list *list)` |
| tuple length | `int64_t chelis_tuple_len(const chelis_tuple *tuple)` |
| tuple construction | `chelis_tuple *chelis_tuple_from_values(const chelis_value *items, int64_t len)` |
| tuple index | `chelis_value chelis_tuple_get(const chelis_tuple *tuple, int64_t index)` |
| ADT construction | `chelis_adt *chelis_adt_construct(chelis_string ctor, const chelis_value *fields, int64_t len)` |
| ADT field count | `int64_t chelis_adt_field_count(const chelis_adt *adt)` |
| ADT field index | `chelis_value chelis_adt_get_field(const chelis_adt *adt, int64_t index)` |
| dictionary length | `int64_t chelis_dict_len(const chelis_dict *dict)` |
| dictionary construction | `chelis_dict *chelis_dict_from_pairs(const chelis_list *pairs)` |
| dictionary membership | `bool chelis_dict_contains(const chelis_dict *dict, chelis_value key)` |
| dictionary lookup | `chelis_option *chelis_dict_get(const chelis_dict *dict, chelis_value key)` |
| dictionary removal | `chelis_dict *chelis_dict_remove(const chelis_dict *dict, chelis_value key)` |
| dictionary insertion | `chelis_dict *chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value)` |
| dictionary merge | `chelis_dict *chelis_dict_merge(const chelis_dict *left, const chelis_dict *right)` |
| byte-file read | `chelis_list *chelis_read_bytes(chelis_string path)` |
| mapped byte read | `chelis_list *chelis_mmap_read(const chelis_mapped_file *mapped, int64_t offset, int64_t len)` |
| mapped byte length | `int64_t chelis_mmap_len(const chelis_mapped_file *mapped)` |
| list print | `void chelis_print_list(const chelis_list *list)` |
| tuple print | `void chelis_print_tuple(const chelis_tuple *tuple)` |
| dictionary print | `void chelis_print_dict(const chelis_dict *dict)` |
| ADT print | `void chelis_print_adt(const chelis_adt *adt)` |
