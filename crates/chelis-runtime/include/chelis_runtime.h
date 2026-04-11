#ifndef CHELIS_RUNTIME_H
#define CHELIS_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHELIS_F32 0
#define CHELIS_F64 1
#define CHELIS_I32 2
#define CHELIS_BOOL 3
#define CHELIS_MAX_DIM 8

typedef struct {
    float *data;
    int shape[CHELIS_MAX_DIM];
    int strides[CHELIS_MAX_DIM];
    int ndim;
    int dtype;
    int size;
    int owns_data;
} chelis_tensor;

typedef struct {
    void *handle;
} chelis_string;

typedef struct chelis_list chelis_list;
typedef struct chelis_tuple chelis_tuple;
typedef struct chelis_dict chelis_dict;
typedef struct chelis_adt chelis_adt;
typedef struct chelis_mapped_file chelis_mapped_file;

typedef enum {
    CHELIS_VALUE_INT64,
    CHELIS_VALUE_FLOAT64,
    CHELIS_VALUE_BOOL,
    CHELIS_VALUE_STRING,
    CHELIS_VALUE_TENSOR,
    CHELIS_VALUE_LIST,
    CHELIS_VALUE_TUPLE,
    CHELIS_VALUE_DICT,
    CHELIS_VALUE_ADT
} chelis_value_tag;

typedef struct {
    chelis_value_tag tag;
    union {
        int64_t i64;
        double f64;
        bool boolean;
        chelis_string string;
        chelis_tensor *tensor;
        chelis_list *list;
        chelis_tuple *tuple;
        chelis_dict *dict;
        chelis_adt *adt;
    } as;
} chelis_value;

typedef struct {
    chelis_value key;
    chelis_value value;
} chelis_dict_entry;

typedef struct {
    bool is_some;
    int64_t value;
} chelis_option_i64;

typedef struct {
    bool is_some;
    double value;
} chelis_option_f64;

typedef struct {
    bool is_some;
    chelis_value value;
} chelis_option_value;

#ifdef __cplusplus
extern "C" {
#endif

chelis_tensor *chelis_alloc(int ndim, const int *shape, int dtype);
chelis_tensor *chelis_alloc_view(int ndim, const int *shape, int dtype, float *data);
void chelis_free(chelis_tensor *t);
void chelis_fill_f32(chelis_tensor *t, float val);
chelis_tensor *chelis_scalar_tensor_from_i64(int64_t value);
chelis_tensor *chelis_scalar_tensor_from_f64(double value);
double chelis_tensor_to_f64(const chelis_tensor *t);
int64_t chelis_tensor_rank(const chelis_tensor *t);
int64_t chelis_tensor_shape(const chelis_tensor *t, int64_t axis);
int64_t chelis_tensor_numel(const chelis_tensor *t);

chelis_string chelis_string_from_cstr(const char *value);
const char *chelis_string_data(chelis_string value);
void chelis_string_retain(chelis_string value);
void chelis_string_release(chelis_string value);
chelis_string chelis_string_concat(chelis_string lhs, chelis_string rhs);
chelis_string chelis_string_trim(chelis_string value);
chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len);
bool chelis_string_eq(chelis_string lhs, chelis_string rhs);
bool chelis_string_contains(chelis_string haystack, chelis_string needle);
bool chelis_string_starts_with(chelis_string value, chelis_string prefix);
bool chelis_string_ends_with(chelis_string value, chelis_string suffix);
int64_t chelis_string_len(chelis_string value);
chelis_string chelis_string_from_int64(int64_t value);
chelis_string chelis_string_from_f64(double value);
chelis_string chelis_string_from_bool(bool value);
chelis_option_i64 chelis_parse_int64(chelis_string value);
chelis_option_f64 chelis_parse_f64(chelis_string value);

void chelis_list_retain(const chelis_list *list);
void chelis_list_release(const chelis_list *list);
int64_t chelis_list_len(const chelis_list *list);
void chelis_tuple_retain(const chelis_tuple *tuple);
void chelis_tuple_release(const chelis_tuple *tuple);
int64_t chelis_tuple_len(const chelis_tuple *tuple);
void chelis_dict_retain(const chelis_dict *dict);
void chelis_dict_release(const chelis_dict *dict);
int64_t chelis_dict_len(const chelis_dict *dict);
void chelis_adt_retain(const chelis_adt *adt);
void chelis_adt_release(const chelis_adt *adt);
chelis_adt *chelis_adt_construct(chelis_string ctor, const chelis_value *fields, int64_t len);
chelis_string chelis_adt_get_tag(const chelis_adt *adt);
bool chelis_adt_tag_equals(const chelis_adt *adt, chelis_string ctor);
int64_t chelis_adt_field_count(const chelis_adt *adt);
chelis_value chelis_adt_get_field(const chelis_adt *adt, int64_t index);

chelis_value chelis_value_from_int64(int64_t value);
chelis_value chelis_value_from_f64(double value);
chelis_value chelis_value_from_bool(bool value);
chelis_value chelis_value_from_string(chelis_string value);
chelis_value chelis_value_from_tensor(chelis_tensor *value);
chelis_value chelis_value_from_list(chelis_list *value);
chelis_value chelis_value_from_tuple(chelis_tuple *value);
chelis_value chelis_value_from_dict(chelis_dict *value);
chelis_value chelis_value_from_adt(chelis_adt *value);
void chelis_value_retain(chelis_value value);
void chelis_value_release(chelis_value value);
int64_t chelis_value_as_int64(chelis_value value);
double chelis_value_as_f64(chelis_value value);
bool chelis_value_as_bool(chelis_value value);
chelis_string chelis_value_as_string(chelis_value value);
chelis_tensor *chelis_value_as_tensor(chelis_value value);
chelis_list *chelis_value_as_list(chelis_value value);
chelis_tuple *chelis_value_as_tuple(chelis_value value);
chelis_dict *chelis_value_as_dict(chelis_value value);
chelis_adt *chelis_value_as_adt(chelis_value value);

chelis_list *chelis_list_empty(void);
chelis_list *chelis_list_from_values(const chelis_value *items, int64_t len);
chelis_value chelis_list_index(const chelis_list *list, int64_t index);
chelis_list *chelis_list_append(const chelis_list *list, chelis_value value);
chelis_list *chelis_list_concat(const chelis_list *lhs, const chelis_list *rhs);
chelis_list *chelis_list_take(const chelis_list *list, int64_t count);
chelis_list *chelis_list_drop(const chelis_list *list, int64_t count);
chelis_list *chelis_list_chunk(const chelis_list *list, int64_t size);
chelis_list *chelis_list_flatten(const chelis_list *list);
chelis_list *chelis_range_i64(int64_t start, int64_t end);
chelis_list *chelis_list_zip(const chelis_list *lhs, const chelis_list *rhs);
chelis_list *chelis_list_enumerate(const chelis_list *list);
chelis_tuple *chelis_tuple_from_values(const chelis_value *items, int64_t len);
chelis_value chelis_tuple_get(const chelis_tuple *tuple, int64_t index);
chelis_dict *chelis_dict_from_pairs(const chelis_list *pairs);
bool chelis_dict_contains(const chelis_dict *dict, chelis_value key);
chelis_option_value chelis_dict_get(const chelis_dict *dict, chelis_value key);
chelis_option_i64 chelis_dict_get_i64(const chelis_dict *dict, chelis_value key);
chelis_option_f64 chelis_dict_get_f64(const chelis_dict *dict, chelis_value key);
chelis_dict *chelis_dict_remove(const chelis_dict *dict, chelis_value key);
chelis_dict *chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value);
chelis_dict *chelis_dict_merge(const chelis_dict *lhs, const chelis_dict *rhs);
chelis_list *chelis_dict_keys(const chelis_dict *dict);
chelis_list *chelis_dict_values(const chelis_dict *dict);
chelis_list *chelis_dict_entries(const chelis_dict *dict);
chelis_tensor *chelis_tensor_from_value_list(const chelis_list *list);
chelis_list *chelis_list_from_tensor(const chelis_tensor *tensor);
chelis_tensor *chelis_pad_sequences(const chelis_list *sequences, chelis_value pad_value);
chelis_tensor *chelis_pad_sequences_to(const chelis_list *sequences, int64_t width, chelis_value pad_value);
chelis_tensor *chelis_tensor_concat(const chelis_list *parts, int64_t axis);
chelis_list *chelis_tensor_split(const chelis_tensor *tensor, int64_t axis, const chelis_list *sizes);
chelis_tensor *chelis_tensor_gather(const chelis_tensor *tensor, const chelis_tensor *indices, int64_t axis);
chelis_tensor *chelis_tensor_cmplt(const chelis_tensor *lhs, const chelis_tensor *rhs);
chelis_tensor *chelis_tensor_scatter(
    const chelis_tensor *base,
    const chelis_tensor *indices,
    const chelis_tensor *updates,
    int64_t axis,
    chelis_string mode
);
chelis_tensor *chelis_tensor_where(
    const chelis_tensor *cond,
    const chelis_tensor *then_tensor,
    const chelis_tensor *else_tensor
);
chelis_tensor *chelis_tensor_cumsum(const chelis_tensor *tensor, int64_t axis);
chelis_tuple *chelis_tensor_sort(const chelis_tensor *tensor, int64_t axis);
chelis_tensor *chelis_tensor_diagonal(const chelis_tensor *tensor, int64_t axis1, int64_t axis2);
chelis_tensor *chelis_tensor_trace(const chelis_tensor *tensor, int64_t axis1, int64_t axis2);
chelis_tensor *chelis_tensor_clamp(
    const chelis_tensor *tensor,
    const chelis_tensor *lo,
    const chelis_tensor *hi
);
chelis_tensor *chelis_tensor_einsum(
    chelis_string equation,
    const chelis_tensor *lhs,
    const chelis_tensor *rhs
);
void chelis_print_list(const chelis_list *list);
void chelis_print_tuple(const chelis_tuple *tuple);
void chelis_print_dict(const chelis_dict *dict);
void chelis_print_adt(const chelis_adt *adt);
chelis_string chelis_read_file(chelis_string path);
void chelis_write_file(chelis_string path, chelis_string contents);
chelis_list *chelis_read_lines(chelis_string path);
chelis_list *chelis_read_bytes(chelis_string path);
bool chelis_file_exists(chelis_string path);
chelis_list *chelis_list_dir(chelis_string path);
chelis_mapped_file *chelis_mmap_file(chelis_string path);
chelis_list *chelis_mmap_read(const chelis_mapped_file *mapped, int64_t offset, int64_t len);
int64_t chelis_mmap_len(const chelis_mapped_file *mapped);

static inline void chelis_flat_to_indices(int flat, const int *shape, int ndim, int *out) {
    for (int d = ndim - 1; d >= 0; d--) {
        out[d] = flat % shape[d];
        flat /= shape[d];
    }
}

static inline int chelis_indices_to_flat(const int *indices, const int *strides, int ndim) {
    int flat = 0;
    for (int d = 0; d < ndim; d++) {
        flat += indices[d] * strides[d];
    }
    return flat;
}

static inline int chelis_is_contiguous(const chelis_tensor *t) {
    int expected = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        if (t->strides[d] != expected) return 0;
        expected *= t->shape[d];
    }
    return 1;
}

chelis_tensor *chelis_contiguous(const chelis_tensor *t);
void chelis_print_f32(const chelis_tensor *t);

#ifdef __cplusplus
}
#endif

#endif
