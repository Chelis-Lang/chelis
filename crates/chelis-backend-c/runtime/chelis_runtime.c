#include "chelis_runtime.h"
#include <ctype.h>

#define CHELIS_RUNTIME_FAIL(...) \
    do { \
        fprintf(stderr, __VA_ARGS__); \
        exit(1); \
    } while (0)

static void chelis_init_tensor(chelis_tensor *t, int ndim, const int *shape, int dtype) {
    t->ndim = ndim;
    t->dtype = dtype;
    t->size = 1;
    t->owns_data = 1;
    for (int d = 0; d < ndim; d++) {
        t->shape[d] = shape[d];
        t->size *= shape[d];
    }
    /* Row-major strides */
    for (int d = ndim - 1; d >= 0; d--) {
        t->strides[d] = (d == ndim - 1) ? 1 : t->strides[d + 1] * t->shape[d + 1];
    }
    if (t->size == 0) t->size = 1; /* scalar */
}

chelis_tensor* chelis_alloc(int ndim, const int *shape, int dtype) {
    chelis_tensor *t = (chelis_tensor*)calloc(1, sizeof(chelis_tensor));
    chelis_init_tensor(t, ndim, shape, dtype);
    t->data = (float*)calloc(t->size, sizeof(float));
    return t;
}

chelis_tensor* chelis_alloc_view(int ndim, const int *shape, int dtype, float *data) {
    chelis_tensor *t = (chelis_tensor*)calloc(1, sizeof(chelis_tensor));
    chelis_init_tensor(t, ndim, shape, dtype);
    t->data = data;
    t->owns_data = 0;
    return t;
}

void chelis_free(chelis_tensor *t) {
    if (t) {
        if (t->owns_data) free(t->data);
        free(t);
    }
}

void chelis_fill_f32(chelis_tensor *t, float val) {
    for (int i = 0; i < t->size; i++) t->data[i] = val;
}

chelis_tensor* chelis_scalar_tensor_from_i64(int64_t value) {
    chelis_tensor *tensor = chelis_alloc(0, NULL, CHELIS_I32);
    tensor->data[0] = (float)value;
    return tensor;
}

chelis_tensor* chelis_scalar_tensor_from_f64(double value) {
    chelis_tensor *tensor = chelis_alloc(0, NULL, CHELIS_F64);
    tensor->data[0] = (float)value;
    return tensor;
}

double chelis_tensor_to_f64(const chelis_tensor *t) {
    if (!t || t->ndim != 0) {
        fprintf(stderr, "chelis_tensor_to_f64 expects a rank-0 tensor\n");
        abort();
    }
    return (double)t->data[0];
}

int64_t chelis_tensor_rank(const chelis_tensor *t) {
    return t ? t->ndim : 0;
}

int64_t chelis_tensor_shape(const chelis_tensor *t, int64_t axis) {
    if (!t || axis < 0 || axis >= t->ndim) {
        fprintf(stderr, "chelis_tensor_shape axis out of bounds\n");
        abort();
    }
    return t->shape[axis];
}

int64_t chelis_tensor_numel(const chelis_tensor *t) {
    return t ? t->size : 0;
}

static chelis_string chelis_string_from_buffer(const char *value, size_t len) {
    chelis_string out;
    out.data = (char*)calloc(len + 1, sizeof(char));
    memcpy(out.data, value, len);
    out.data[len] = '\0';
    return out;
}

static size_t chelis_utf8_char_width(unsigned char byte) {
    if ((byte & 0x80) == 0x00) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 1;
}

static size_t chelis_utf8_offset_for_char(const char *value, size_t char_index) {
    size_t byte_index = 0;
    size_t current = 0;
    while (value[byte_index] != '\0' && current < char_index) {
        byte_index += chelis_utf8_char_width((unsigned char)value[byte_index]);
        current++;
    }
    return byte_index;
}

chelis_string chelis_string_from_cstr(const char *value) {
    return chelis_string_from_buffer(value, strlen(value));
}

chelis_string chelis_string_concat(chelis_string lhs, chelis_string rhs) {
    size_t lhs_len = strlen(lhs.data);
    size_t rhs_len = strlen(rhs.data);
    chelis_string out;
    out.data = (char*)calloc(lhs_len + rhs_len + 1, sizeof(char));
    memcpy(out.data, lhs.data, lhs_len);
    memcpy(out.data + lhs_len, rhs.data, rhs_len);
    out.data[lhs_len + rhs_len] = '\0';
    return out;
}

chelis_string chelis_string_trim(chelis_string value) {
    size_t len = strlen(value.data);
    size_t start = 0;
    while (start < len && isspace((unsigned char)value.data[start])) start++;
    size_t end = len;
    while (end > start && isspace((unsigned char)value.data[end - 1])) end--;
    return chelis_string_from_buffer(value.data + start, end - start);
}

chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len) {
    if (start < 0 || len < 0) {
        return chelis_string_from_cstr("");
    }
    size_t char_len = chelis_string_len(value);
    size_t s = (size_t)start;
    size_t n = (size_t)len;
    if (s >= char_len) {
        return chelis_string_from_cstr("");
    }
    size_t end_char = s + n;
    if (end_char > char_len) end_char = char_len;
    size_t start_byte = chelis_utf8_offset_for_char(value.data, s);
    size_t end_byte = chelis_utf8_offset_for_char(value.data, end_char);
    return chelis_string_from_buffer(value.data + start_byte, end_byte - start_byte);
}

bool chelis_string_eq(chelis_string lhs, chelis_string rhs) {
    return strcmp(lhs.data, rhs.data) == 0;
}

bool chelis_string_contains(chelis_string haystack, chelis_string needle) {
    return strstr(haystack.data, needle.data) != NULL;
}

bool chelis_string_starts_with(chelis_string value, chelis_string prefix) {
    size_t prefix_len = strlen(prefix.data);
    return strncmp(value.data, prefix.data, prefix_len) == 0;
}

bool chelis_string_ends_with(chelis_string value, chelis_string suffix) {
    size_t value_len = strlen(value.data);
    size_t suffix_len = strlen(suffix.data);
    if (suffix_len > value_len) return false;
    return strcmp(value.data + value_len - suffix_len, suffix.data) == 0;
}

int64_t chelis_string_len(chelis_string value) {
    int64_t count = 0;
    for (size_t i = 0; value.data[i] != '\0'; i++) {
        if (((unsigned char)value.data[i] & 0xC0) != 0x80) {
            count++;
        }
    }
    return count;
}

chelis_string chelis_string_from_int64(int64_t value) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%lld", (long long)value);
    return chelis_string_from_cstr(buffer);
}

chelis_string chelis_string_from_f64(double value) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%g", value);
    return chelis_string_from_cstr(buffer);
}

chelis_string chelis_string_from_bool(bool value) {
    return chelis_string_from_cstr(value ? "true" : "false");
}

chelis_option_i64 chelis_parse_int64(chelis_string value) {
    chelis_option_i64 out;
    char *end = NULL;
    long long parsed = strtoll(value.data, &end, 10);
    while (end && *end != '\0' && isspace((unsigned char)*end)) end++;
    out.is_some = end && *end == '\0';
    out.value = (int64_t)parsed;
    return out;
}

chelis_option_f64 chelis_parse_f64(chelis_string value) {
    chelis_option_f64 out;
    char *end = NULL;
    double parsed = strtod(value.data, &end);
    while (end && *end != '\0' && isspace((unsigned char)*end)) end++;
    out.is_some = end && *end == '\0';
    out.value = parsed;
    return out;
}

chelis_value chelis_value_from_int64(int64_t value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_INT64;
    out.as.i64 = value;
    return out;
}

chelis_value chelis_value_from_f64(double value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_FLOAT64;
    out.as.f64 = value;
    return out;
}

chelis_value chelis_value_from_bool(bool value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_BOOL;
    out.as.boolean = value;
    return out;
}

chelis_value chelis_value_from_string(chelis_string value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_STRING;
    out.as.string = value;
    return out;
}

chelis_value chelis_value_from_tensor(chelis_tensor *value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_TENSOR;
    out.as.tensor = value;
    return out;
}

chelis_value chelis_value_from_list(chelis_list *value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_LIST;
    out.as.list = value;
    return out;
}

chelis_value chelis_value_from_tuple(chelis_tuple *value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_TUPLE;
    out.as.tuple = value;
    return out;
}

chelis_value chelis_value_from_dict(chelis_dict *value) {
    chelis_value out;
    out.tag = CHELIS_VALUE_DICT;
    out.as.dict = value;
    return out;
}

int64_t chelis_value_as_int64(chelis_value value) {
    if (value.tag != CHELIS_VALUE_INT64) {
        fprintf(stderr, "expected int64 value\n");
        abort();
    }
    return value.as.i64;
}

double chelis_value_as_f64(chelis_value value) {
    if (value.tag == CHELIS_VALUE_INT64) return (double)value.as.i64;
    if (value.tag != CHELIS_VALUE_FLOAT64) {
        fprintf(stderr, "expected float64 value\n");
        abort();
    }
    return value.as.f64;
}

bool chelis_value_as_bool(chelis_value value) {
    if (value.tag != CHELIS_VALUE_BOOL) {
        fprintf(stderr, "expected bool value\n");
        abort();
    }
    return value.as.boolean;
}

chelis_string chelis_value_as_string(chelis_value value) {
    if (value.tag != CHELIS_VALUE_STRING) {
        fprintf(stderr, "expected string value\n");
        abort();
    }
    return value.as.string;
}

chelis_tensor* chelis_value_as_tensor(chelis_value value) {
    if (value.tag != CHELIS_VALUE_TENSOR) {
        fprintf(stderr, "expected tensor value\n");
        abort();
    }
    return value.as.tensor;
}

chelis_list* chelis_value_as_list(chelis_value value) {
    if (value.tag != CHELIS_VALUE_LIST) {
        fprintf(stderr, "expected list value\n");
        abort();
    }
    return value.as.list;
}

chelis_tuple* chelis_value_as_tuple(chelis_value value) {
    if (value.tag != CHELIS_VALUE_TUPLE) {
        fprintf(stderr, "expected tuple value\n");
        abort();
    }
    return value.as.tuple;
}

chelis_dict* chelis_value_as_dict(chelis_value value) {
    if (value.tag != CHELIS_VALUE_DICT) {
        fprintf(stderr, "expected dict value\n");
        abort();
    }
    return value.as.dict;
}

chelis_list* chelis_list_empty(void) {
    chelis_list *list = (chelis_list*)calloc(1, sizeof(chelis_list));
    list->items = NULL;
    list->len = 0;
    return list;
}

chelis_list* chelis_list_from_values(const chelis_value *items, int64_t len) {
    chelis_list *list = (chelis_list*)calloc(1, sizeof(chelis_list));
    list->len = len;
    list->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        list->items[i] = items[i];
    }
    return list;
}

int64_t chelis_list_len(const chelis_list *list) {
    return list ? list->len : 0;
}

chelis_value chelis_list_index(const chelis_list *list, int64_t index) {
    if (!list || index < 0 || index >= list->len) {
        fprintf(stderr, "list index out of bounds\n");
        abort();
    }
    return list->items[index];
}

chelis_list* chelis_list_append(const chelis_list *list, chelis_value value) {
    int64_t len = list ? list->len : 0;
    chelis_list *out = chelis_list_from_values(list ? list->items : NULL, len);
    out->items = (chelis_value*)realloc(out->items, (size_t)(len + 1) * sizeof(chelis_value));
    out->items[len] = value;
    out->len = len + 1;
    return out;
}

chelis_list* chelis_list_concat(const chelis_list *lhs, const chelis_list *rhs) {
    int64_t lhs_len = lhs ? lhs->len : 0;
    int64_t rhs_len = rhs ? rhs->len : 0;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = lhs_len + rhs_len;
    out->items = out->len > 0 ? (chelis_value*)calloc((size_t)out->len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < lhs_len; i++) out->items[i] = lhs->items[i];
    for (int64_t i = 0; i < rhs_len; i++) out->items[lhs_len + i] = rhs->items[i];
    return out;
}

chelis_list* chelis_list_take(const chelis_list *list, int64_t count) {
    if (count < 0) {
        fprintf(stderr, "take requires non-negative count\n");
        abort();
    }
    int64_t len = list ? list->len : 0;
    if (count < len) {
        len = count;
    }
    return chelis_list_from_values(list ? list->items : NULL, len);
}

chelis_list* chelis_list_drop(const chelis_list *list, int64_t count) {
    if (count < 0) {
        fprintf(stderr, "drop requires non-negative count\n");
        abort();
    }
    int64_t len = list ? list->len : 0;
    if (count >= len) {
        return chelis_list_empty();
    }
    return chelis_list_from_values(list->items + count, len - count);
}

chelis_list* chelis_list_chunk(const chelis_list *list, int64_t size) {
    if (size <= 0) {
        fprintf(stderr, "chunk requires positive size\n");
        abort();
    }
    int64_t len = list ? list->len : 0;
    int64_t outer_len = len == 0 ? 0 : (len + size - 1) / size;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = outer_len;
    out->items = outer_len > 0 ? (chelis_value*)calloc((size_t)outer_len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < outer_len; i++) {
        int64_t start = i * size;
        int64_t remaining = len - start;
        int64_t chunk_len = remaining < size ? remaining : size;
        out->items[i] = chelis_value_from_list(
            chelis_list_from_values(list->items + start, chunk_len)
        );
    }
    return out;
}

chelis_list* chelis_list_flatten(const chelis_list *list) {
    int64_t outer_len = list ? list->len : 0;
    int64_t total_len = 0;
    for (int64_t i = 0; i < outer_len; i++) {
        if (list->items[i].tag != CHELIS_VALUE_LIST) {
            fprintf(stderr, "flatten expects nested list input\n");
            abort();
        }
        chelis_list *inner = list->items[i].as.list;
        total_len += inner ? inner->len : 0;
    }
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = total_len;
    out->items = total_len > 0 ? (chelis_value*)calloc((size_t)total_len, sizeof(chelis_value)) : NULL;
    int64_t offset = 0;
    for (int64_t i = 0; i < outer_len; i++) {
        chelis_list *inner = list->items[i].as.list;
        int64_t inner_len = inner ? inner->len : 0;
        for (int64_t j = 0; j < inner_len; j++) {
            out->items[offset++] = inner->items[j];
        }
    }
    return out;
}

chelis_list* chelis_range_i64(int64_t start, int64_t end) {
    int64_t len = end > start ? end - start : 0;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = len;
    out->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        out->items[i] = chelis_value_from_int64(start + i);
    }
    return out;
}

chelis_tuple* chelis_tuple_from_values(const chelis_value *items, int64_t len) {
    chelis_tuple *tuple = (chelis_tuple*)calloc(1, sizeof(chelis_tuple));
    tuple->len = len;
    tuple->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        tuple->items[i] = items[i];
    }
    return tuple;
}

chelis_value chelis_tuple_get(const chelis_tuple *tuple, int64_t index) {
    if (!tuple || index < 0 || index >= tuple->len) {
        fprintf(stderr, "tuple index out of bounds\n");
        abort();
    }
    return tuple->items[index];
}

chelis_list* chelis_list_zip(const chelis_list *lhs, const chelis_list *rhs) {
    int64_t lhs_len = lhs ? lhs->len : 0;
    int64_t rhs_len = rhs ? rhs->len : 0;
    int64_t len = lhs_len < rhs_len ? lhs_len : rhs_len;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = len;
    out->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        chelis_value items[2] = { lhs->items[i], rhs->items[i] };
        out->items[i] = chelis_value_from_tuple(chelis_tuple_from_values(items, 2));
    }
    return out;
}

chelis_list* chelis_list_enumerate(const chelis_list *list) {
    int64_t len = list ? list->len : 0;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = len;
    out->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        chelis_value items[2] = {
            chelis_value_from_int64(i),
            list->items[i],
        };
        out->items[i] = chelis_value_from_tuple(chelis_tuple_from_values(items, 2));
    }
    return out;
}

static bool chelis_value_key_eq(chelis_value lhs, chelis_value rhs) {
    if (lhs.tag != rhs.tag) return false;
    switch (lhs.tag) {
        case CHELIS_VALUE_INT64:
            return lhs.as.i64 == rhs.as.i64;
        case CHELIS_VALUE_STRING:
            return chelis_string_eq(lhs.as.string, rhs.as.string);
        default:
            return false;
    }
}

chelis_dict* chelis_dict_from_pairs(const chelis_list *pairs) {
    chelis_dict *dict = (chelis_dict*)calloc(1, sizeof(chelis_dict));
    int64_t len = pairs ? pairs->len : 0;
    dict->entries = len > 0 ? (chelis_dict_entry*)calloc((size_t)len, sizeof(chelis_dict_entry)) : NULL;
    dict->len = 0;
    for (int64_t i = 0; i < len; i++) {
        if (pairs->items[i].tag != CHELIS_VALUE_TUPLE) {
            fprintf(stderr, "dict_of expects list entries to be tuples\n");
            abort();
        }
        chelis_tuple *entry = pairs->items[i].as.tuple;
        if (!entry || entry->len != 2) {
            fprintf(stderr, "dict_of expects 2-tuples\n");
            abort();
        }
        chelis_value key = entry->items[0];
        if (!(key.tag == CHELIS_VALUE_INT64 || key.tag == CHELIS_VALUE_STRING)) {
            fprintf(stderr, "dict_of keys must be int64 or string\n");
            abort();
        }
        bool replaced = false;
        for (int64_t j = 0; j < dict->len; j++) {
            if (chelis_value_key_eq(dict->entries[j].key, key)) {
                dict->entries[j].value = entry->items[1];
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            dict->entries[dict->len].key = key;
            dict->entries[dict->len].value = entry->items[1];
            dict->len += 1;
        }
    }
    return dict;
}

bool chelis_dict_contains(const chelis_dict *dict, chelis_value key) {
    if (!(key.tag == CHELIS_VALUE_INT64 || key.tag == CHELIS_VALUE_STRING)) {
        fprintf(stderr, "dict key must be int64 or string\n");
        abort();
    }
    for (int64_t i = 0; dict && i < dict->len; i++) {
        if (chelis_value_key_eq(dict->entries[i].key, key)) {
            return true;
        }
    }
    return false;
}

chelis_option_value chelis_dict_get(const chelis_dict *dict, chelis_value key) {
    chelis_option_value out;
    out.is_some = false;
    out.value = chelis_value_from_int64(0);
    if (!(key.tag == CHELIS_VALUE_INT64 || key.tag == CHELIS_VALUE_STRING)) {
        fprintf(stderr, "dict key must be int64 or string\n");
        abort();
    }
    for (int64_t i = 0; dict && i < dict->len; i++) {
        if (chelis_value_key_eq(dict->entries[i].key, key)) {
            out.is_some = true;
            out.value = dict->entries[i].value;
            return out;
        }
    }
    return out;
}

chelis_option_i64 chelis_dict_get_i64(const chelis_dict *dict, chelis_value key) {
    chelis_option_value value = chelis_dict_get(dict, key);
    chelis_option_i64 out;
    out.is_some = value.is_some;
    out.value = 0;
    if (!value.is_some) {
        return out;
    }
    if (value.value.tag != CHELIS_VALUE_INT64) {
        fprintf(stderr, "dict value is not int64\n");
        abort();
    }
    out.value = value.value.as.i64;
    return out;
}

chelis_option_f64 chelis_dict_get_f64(const chelis_dict *dict, chelis_value key) {
    chelis_option_value value = chelis_dict_get(dict, key);
    chelis_option_f64 out;
    out.is_some = value.is_some;
    out.value = 0.0;
    if (!value.is_some) {
        return out;
    }
    if (value.value.tag == CHELIS_VALUE_INT64) {
        out.value = (double)value.value.as.i64;
        return out;
    }
    if (value.value.tag != CHELIS_VALUE_FLOAT64) {
        fprintf(stderr, "dict value is not float64\n");
        abort();
    }
    out.value = value.value.as.f64;
    return out;
}

chelis_dict* chelis_dict_remove(const chelis_dict *dict, chelis_value key) {
    if (!(key.tag == CHELIS_VALUE_INT64 || key.tag == CHELIS_VALUE_STRING)) {
        fprintf(stderr, "dict key must be int64 or string\n");
        abort();
    }
    int64_t len = dict ? dict->len : 0;
    int64_t kept = 0;
    for (int64_t i = 0; i < len; i++) {
        if (!chelis_value_key_eq(dict->entries[i].key, key)) {
            kept += 1;
        }
    }
    chelis_dict *out = (chelis_dict*)calloc(1, sizeof(chelis_dict));
    out->entries = kept > 0 ? (chelis_dict_entry*)calloc((size_t)kept, sizeof(chelis_dict_entry)) : NULL;
    out->len = 0;
    for (int64_t i = 0; i < len; i++) {
        if (!chelis_value_key_eq(dict->entries[i].key, key)) {
            out->entries[out->len++] = dict->entries[i];
        }
    }
    return out;
}

chelis_dict* chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value) {
    if (!(key.tag == CHELIS_VALUE_INT64 || key.tag == CHELIS_VALUE_STRING)) {
        fprintf(stderr, "dict key must be int64 or string\n");
        abort();
    }
    int64_t len = dict ? dict->len : 0;
    chelis_dict *out = (chelis_dict*)calloc(1, sizeof(chelis_dict));
    out->entries = len > 0 ? (chelis_dict_entry*)calloc((size_t)(len + 1), sizeof(chelis_dict_entry)) : (chelis_dict_entry*)calloc(1, sizeof(chelis_dict_entry));
    out->len = len;
    for (int64_t i = 0; i < len; i++) {
        out->entries[i] = dict->entries[i];
    }
    for (int64_t i = 0; i < out->len; i++) {
        if (chelis_value_key_eq(out->entries[i].key, key)) {
            out->entries[i].value = value;
            return out;
        }
    }
    out->entries[out->len].key = key;
    out->entries[out->len].value = value;
    out->len += 1;
    return out;
}

chelis_dict* chelis_dict_merge(const chelis_dict *lhs, const chelis_dict *rhs) {
    chelis_dict *out = (chelis_dict*)calloc(1, sizeof(chelis_dict));
    int64_t lhs_len = lhs ? lhs->len : 0;
    out->entries = lhs_len > 0 ? (chelis_dict_entry*)calloc((size_t)lhs_len, sizeof(chelis_dict_entry)) : NULL;
    out->len = lhs_len;
    for (int64_t i = 0; i < lhs_len; i++) {
        out->entries[i] = lhs->entries[i];
    }
    for (int64_t i = 0; rhs && i < rhs->len; i++) {
        chelis_dict *next = chelis_dict_insert(out, rhs->entries[i].key, rhs->entries[i].value);
        out = next;
    }
    return out;
}

chelis_list* chelis_dict_keys(const chelis_dict *dict) {
    int64_t len = dict ? dict->len : 0;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = len;
    out->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        out->items[i] = dict->entries[i].key;
    }
    return out;
}

chelis_list* chelis_dict_values(const chelis_dict *dict) {
    int64_t len = dict ? dict->len : 0;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = len;
    out->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        out->items[i] = dict->entries[i].value;
    }
    return out;
}

chelis_list* chelis_dict_entries(const chelis_dict *dict) {
    int64_t len = dict ? dict->len : 0;
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = len;
    out->items = len > 0 ? (chelis_value*)calloc((size_t)len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < len; i++) {
        chelis_value pair_items[2] = {
            dict->entries[i].key,
            dict->entries[i].value,
        };
        out->items[i] = chelis_value_from_tuple(chelis_tuple_from_values(pair_items, 2));
    }
    return out;
}

chelis_tensor* chelis_tensor_from_value_list(const chelis_list *list) {
    int shape[1] = { (int)(list ? list->len : 0) };
    int dtype = CHELIS_F64;
    if (list && list->len > 0 && list->items[0].tag == CHELIS_VALUE_INT64) {
        dtype = CHELIS_I32;
    }
    chelis_tensor *out = chelis_alloc(1, shape, dtype);
    for (int64_t i = 0; list && i < list->len; i++) {
        switch (list->items[i].tag) {
            case CHELIS_VALUE_INT64:
                out->data[i] = (float)list->items[i].as.i64;
                break;
            case CHELIS_VALUE_FLOAT64:
                out->data[i] = (float)list->items[i].as.f64;
                break;
            default:
                fprintf(stderr, "to_tensor expects numeric list elements\n");
                abort();
        }
    }
    return out;
}

chelis_list* chelis_list_from_tensor(const chelis_tensor *tensor) {
    if (!tensor) {
        return chelis_list_empty();
    }
    if (tensor->ndim != 1) {
        fprintf(stderr, "to_list expects a rank-1 tensor\n");
        abort();
    }
    chelis_list *out = (chelis_list*)calloc(1, sizeof(chelis_list));
    out->len = tensor->shape[0];
    out->items = out->len > 0 ? (chelis_value*)calloc((size_t)out->len, sizeof(chelis_value)) : NULL;
    for (int64_t i = 0; i < out->len; i++) {
        float raw = tensor->data[i * tensor->strides[0]];
        switch (tensor->dtype) {
            case CHELIS_BOOL:
                out->items[i] = chelis_value_from_bool(raw != 0.0f);
                break;
            case CHELIS_I32:
                out->items[i] = chelis_value_from_int64((int64_t)raw);
                break;
            case CHELIS_F32:
            case CHELIS_F64:
                out->items[i] = chelis_value_from_f64((double)raw);
                break;
            default:
                fprintf(stderr, "to_list expects numeric or bool tensor input\n");
                abort();
        }
    }
    return out;
}

chelis_tensor* chelis_pad_sequences(const chelis_list *sequences, chelis_value pad_value) {
    int64_t batch = sequences ? sequences->len : 0;
    int64_t width = 0;
    for (int64_t i = 0; sequences && i < sequences->len; i++) {
        if (sequences->items[i].tag != CHELIS_VALUE_LIST) {
            fprintf(stderr, "pad_sequences expects nested lists\n");
            abort();
        }
        if (sequences->items[i].as.list->len > width) {
            width = sequences->items[i].as.list->len;
        }
    }
    int shape[2] = { (int)batch, (int)width };
    int dtype = pad_value.tag == CHELIS_VALUE_INT64 ? CHELIS_I32 : CHELIS_F64;
    chelis_tensor *out = chelis_alloc(2, shape, dtype);
    double pad = pad_value.tag == CHELIS_VALUE_INT64 ? (double)pad_value.as.i64 : pad_value.as.f64;
    for (int64_t row = 0; row < batch; row++) {
        chelis_list *seq = sequences->items[row].as.list;
        for (int64_t col = 0; col < width; col++) {
            int flat = (int)(row * width + col);
            if (col < seq->len) {
                switch (seq->items[col].tag) {
                    case CHELIS_VALUE_INT64:
                        out->data[flat] = (float)seq->items[col].as.i64;
                        break;
                    case CHELIS_VALUE_FLOAT64:
                        out->data[flat] = (float)seq->items[col].as.f64;
                        break;
                    default:
                        fprintf(stderr, "pad_sequences expects numeric nested lists\n");
                        abort();
                }
            } else {
                out->data[flat] = (float)pad;
            }
        }
    }
    return out;
}

static int chelis_tensor_normalize_axis(const chelis_tensor *tensor, int64_t axis, const char *op) {
    if (axis < 0 || axis >= tensor->ndim) {
        CHELIS_RUNTIME_FAIL(
            "%s axis %lld out of bounds for rank %d\n",
            op,
            (long long)axis,
            tensor->ndim
        );
    }
    return (int)axis;
}

static chelis_tensor* chelis_tensor_clone(const chelis_tensor *tensor) {
    chelis_tensor *out = chelis_alloc(tensor->ndim, tensor->shape, tensor->dtype);
    memcpy(out->data, tensor->data, (size_t)tensor->size * sizeof(float));
    return out;
}

static void chelis_require_same_tensor_shape(
    const chelis_tensor *lhs,
    const chelis_tensor *rhs,
    const char *op
) {
    if (lhs->ndim != rhs->ndim) {
        CHELIS_RUNTIME_FAIL("%s expects matching tensor rank\n", op);
    }
    for (int axis = 0; axis < lhs->ndim; axis++) {
        if (lhs->shape[axis] != rhs->shape[axis]) {
            CHELIS_RUNTIME_FAIL("%s expects matching tensor shape\n", op);
        }
    }
}

static int chelis_tensor_scalar_or_same_shape(
    const chelis_tensor *bound,
    const chelis_tensor *tensor
) {
    if (bound->ndim == 0) return 1;
    if (bound->ndim != tensor->ndim) return 0;
    for (int axis = 0; axis < tensor->ndim; axis++) {
        if (bound->shape[axis] != tensor->shape[axis]) return 0;
    }
    return 1;
}

static int64_t chelis_int_list_value(const chelis_list *list, int64_t index, const char *op) {
    if (!list || index < 0 || index >= list->len || list->items[index].tag != CHELIS_VALUE_INT64) {
        fprintf(stderr, "%s expects a list of int64 values\n", op);
        abort();
    }
    return list->items[index].as.i64;
}

chelis_tensor* chelis_tensor_concat(const chelis_list *parts, int64_t axis) {
    if (!parts || parts->len == 0) {
        fprintf(stderr, "concat expects at least one tensor part\n");
        abort();
    }
    chelis_tensor *first = chelis_value_as_tensor(parts->items[0]);
    int axis_i = chelis_tensor_normalize_axis(first, axis, "concat");
    int out_shape[CHELIS_MAX_DIM];
    memcpy(out_shape, first->shape, sizeof(int) * CHELIS_MAX_DIM);
    out_shape[axis_i] = 0;
    for (int64_t part_idx = 0; part_idx < parts->len; part_idx++) {
        chelis_tensor *tensor = chelis_value_as_tensor(parts->items[part_idx]);
        if (tensor->ndim != first->ndim || tensor->dtype != first->dtype) {
            fprintf(stderr, "concat expects matching tensor rank and dtype\n");
            abort();
        }
        for (int axis2 = 0; axis2 < tensor->ndim; axis2++) {
            if (axis2 != axis_i && tensor->shape[axis2] != first->shape[axis2]) {
                fprintf(stderr, "concat expects matching non-concatenated axes\n");
                abort();
            }
        }
        out_shape[axis_i] += tensor->shape[axis_i];
    }
    chelis_tensor *out = chelis_alloc(first->ndim, out_shape, first->dtype);
    int axis_offset = 0;
    int indices[CHELIS_MAX_DIM];
    for (int64_t part_idx = 0; part_idx < parts->len; part_idx++) {
        chelis_tensor *tensor = chelis_value_as_tensor(parts->items[part_idx]);
        for (int linear = 0; linear < tensor->size; linear++) {
            chelis_flat_to_indices(linear, tensor->shape, tensor->ndim, indices);
            indices[axis_i] += axis_offset;
            int out_linear = chelis_indices_to_flat(indices, out->strides, out->ndim);
            out->data[out_linear] = tensor->data[linear];
            indices[axis_i] -= axis_offset;
        }
        axis_offset += tensor->shape[axis_i];
    }
    return out;
}

chelis_list* chelis_tensor_split(const chelis_tensor *tensor, int64_t axis, const chelis_list *sizes) {
    int axis_i = chelis_tensor_normalize_axis(tensor, axis, "split");
    int64_t total = 0;
    for (int64_t i = 0; i < sizes->len; i++) {
        total += chelis_int_list_value(sizes, i, "split");
    }
    if (total != tensor->shape[axis_i]) {
        fprintf(stderr, "split sizes must sum to the selected axis extent\n");
        abort();
    }
    chelis_value *items = sizes->len > 0 ? (chelis_value*)calloc((size_t)sizes->len, sizeof(chelis_value)) : NULL;
    int axis_offset = 0;
    int indices[CHELIS_MAX_DIM];
    for (int64_t part_idx = 0; part_idx < sizes->len; part_idx++) {
        int part_size = (int)chelis_int_list_value(sizes, part_idx, "split");
        int shape[CHELIS_MAX_DIM];
        memcpy(shape, tensor->shape, sizeof(int) * CHELIS_MAX_DIM);
        shape[axis_i] = part_size;
        chelis_tensor *part = chelis_alloc(tensor->ndim, shape, tensor->dtype);
        for (int linear = 0; linear < part->size; linear++) {
            chelis_flat_to_indices(linear, part->shape, part->ndim, indices);
            indices[axis_i] += axis_offset;
            int src = chelis_indices_to_flat(indices, tensor->strides, tensor->ndim);
            part->data[linear] = tensor->data[src];
            indices[axis_i] -= axis_offset;
        }
        axis_offset += part_size;
        items[part_idx] = chelis_value_from_tensor(part);
    }
    return chelis_list_from_values(items, sizes->len);
}

chelis_tensor* chelis_tensor_gather(const chelis_tensor *tensor, const chelis_tensor *indices, int64_t axis) {
    int axis_i = chelis_tensor_normalize_axis(tensor, axis, "gather");
    int out_ndim = tensor->ndim - 1 + indices->ndim;
    int out_shape[CHELIS_MAX_DIM];
    int pos = 0;
    for (int i = 0; i < axis_i; i++) out_shape[pos++] = tensor->shape[i];
    for (int i = 0; i < indices->ndim; i++) out_shape[pos++] = indices->shape[i];
    for (int i = axis_i + 1; i < tensor->ndim; i++) out_shape[pos++] = tensor->shape[i];
    chelis_tensor *out = chelis_alloc(out_ndim, out_shape, tensor->dtype);
    int out_index[CHELIS_MAX_DIM];
    int src_index[CHELIS_MAX_DIM];
    int gather_index[CHELIS_MAX_DIM];
    for (int linear = 0; linear < out->size; linear++) {
        chelis_flat_to_indices(linear, out->shape, out->ndim, out_index);
        int src_pos = 0;
        for (int i = 0; i < axis_i; i++) src_index[src_pos++] = out_index[i];
        for (int i = 0; i < indices->ndim; i++) gather_index[i] = out_index[axis_i + i];
        int index_linear = chelis_indices_to_flat(gather_index, indices->strides, indices->ndim);
        int64_t gathered = (int64_t)indices->data[index_linear];
        if (gathered < 0 || gathered >= tensor->shape[axis_i]) {
            CHELIS_RUNTIME_FAIL("gather index %lld out of bounds\n", (long long)gathered);
        }
        src_index[src_pos++] = (int)gathered;
        for (int i = axis_i + 1; i < tensor->ndim; i++) {
            src_index[src_pos++] = out_index[axis_i + indices->ndim + (i - axis_i - 1)];
        }
        int src_linear = chelis_indices_to_flat(src_index, tensor->strides, tensor->ndim);
        out->data[linear] = tensor->data[src_linear];
    }
    return out;
}

chelis_tensor* chelis_tensor_cmplt(const chelis_tensor *lhs, const chelis_tensor *rhs) {
    chelis_require_same_tensor_shape(lhs, rhs, "cmplt");
    chelis_tensor *out = chelis_alloc(lhs->ndim, lhs->shape, CHELIS_BOOL);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = lhs->data[i] < rhs->data[i] ? 1.0f : 0.0f;
    }
    return out;
}

chelis_tensor* chelis_tensor_scatter(
    const chelis_tensor *base,
    const chelis_tensor *indices,
    const chelis_tensor *updates,
    int64_t axis,
    chelis_string mode
) {
    int axis_i = chelis_tensor_normalize_axis(base, axis, "scatter");
    chelis_tensor *expected = chelis_tensor_gather(base, indices, axis);
    chelis_require_same_tensor_shape(expected, updates, "scatter");
    chelis_free(expected);
    chelis_tensor *out = chelis_tensor_clone(base);
    bool replace_mode = strcmp(mode.data, "replace") == 0;
    bool add_mode = strcmp(mode.data, "add") == 0;
    if (!replace_mode && !add_mode) {
        CHELIS_RUNTIME_FAIL("scatter mode must be replace or add\n");
    }
    bool *seen = replace_mode ? (bool*)calloc((size_t)out->size, sizeof(bool)) : NULL;
    int update_index[CHELIS_MAX_DIM];
    int out_index[CHELIS_MAX_DIM];
    int gather_index[CHELIS_MAX_DIM];
    for (int linear = 0; linear < updates->size; linear++) {
        chelis_flat_to_indices(linear, updates->shape, updates->ndim, update_index);
        int out_pos = 0;
        for (int i = 0; i < axis_i; i++) out_index[out_pos++] = update_index[i];
        for (int i = 0; i < indices->ndim; i++) gather_index[i] = update_index[axis_i + i];
        int index_linear = chelis_indices_to_flat(gather_index, indices->strides, indices->ndim);
        int64_t gathered = (int64_t)indices->data[index_linear];
        if (gathered < 0 || gathered >= base->shape[axis_i]) {
            CHELIS_RUNTIME_FAIL("scatter index %lld out of bounds\n", (long long)gathered);
        }
        out_index[out_pos++] = (int)gathered;
        for (int i = axis_i + 1; i < base->ndim; i++) {
            out_index[out_pos++] = update_index[axis_i + indices->ndim + (i - axis_i - 1)];
        }
        int out_linear = chelis_indices_to_flat(out_index, out->strides, out->ndim);
        if (replace_mode) {
            if (seen[out_linear]) {
                CHELIS_RUNTIME_FAIL(
                    "scatter replace mode rejects duplicate target index %d\n",
                    out_linear
                );
            }
            seen[out_linear] = true;
            out->data[out_linear] = updates->data[linear];
        } else {
            out->data[out_linear] += updates->data[linear];
        }
    }
    free(seen);
    return out;
}

chelis_tensor* chelis_tensor_where(
    const chelis_tensor *cond,
    const chelis_tensor *then_tensor,
    const chelis_tensor *else_tensor
) {
    chelis_require_same_tensor_shape(cond, then_tensor, "where");
    chelis_require_same_tensor_shape(then_tensor, else_tensor, "where");
    chelis_tensor *out = chelis_alloc(then_tensor->ndim, then_tensor->shape, then_tensor->dtype);
    for (int i = 0; i < out->size; i++) {
        out->data[i] = cond->data[i] != 0.0f ? then_tensor->data[i] : else_tensor->data[i];
    }
    return out;
}

chelis_tensor* chelis_tensor_cumsum(const chelis_tensor *tensor, int64_t axis) {
    int axis_i = chelis_tensor_normalize_axis(tensor, axis, "cumsum");
    chelis_tensor *out = chelis_tensor_clone(tensor);
    int axis_size = tensor->shape[axis_i];
    int inner = 1;
    int outer = 1;
    for (int i = axis_i + 1; i < tensor->ndim; i++) inner *= tensor->shape[i];
    for (int i = 0; i < axis_i; i++) outer *= tensor->shape[i];
    for (int outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int inner_idx = 0; inner_idx < inner; inner_idx++) {
            float running = 0.0f;
            for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
                int linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                running += out->data[linear];
                out->data[linear] = running;
            }
        }
    }
    return out;
}

chelis_tuple* chelis_tensor_sort(const chelis_tensor *tensor, int64_t axis) {
    int axis_i = chelis_tensor_normalize_axis(tensor, axis, "sort");
    chelis_tensor *values = chelis_tensor_clone(tensor);
    chelis_tensor *indices = chelis_alloc(tensor->ndim, tensor->shape, CHELIS_I32);
    int axis_size = tensor->shape[axis_i];
    int inner = 1;
    int outer = 1;
    for (int i = axis_i + 1; i < tensor->ndim; i++) inner *= tensor->shape[i];
    for (int i = 0; i < axis_i; i++) outer *= tensor->shape[i];
    for (int outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int inner_idx = 0; inner_idx < inner; inner_idx++) {
            for (int i = 0; i < axis_size; i++) {
                int linear = (outer_idx * axis_size + i) * inner + inner_idx;
                indices->data[linear] = (float)i;
            }
            for (int i = 1; i < axis_size; i++) {
                int j = i;
                while (j > 0) {
                    int left = (outer_idx * axis_size + (j - 1)) * inner + inner_idx;
                    int right = (outer_idx * axis_size + j) * inner + inner_idx;
                    if (values->data[left] <= values->data[right]) break;
                    float tmpv = values->data[left];
                    values->data[left] = values->data[right];
                    values->data[right] = tmpv;
                    float tmpi = indices->data[left];
                    indices->data[left] = indices->data[right];
                    indices->data[right] = tmpi;
                    j--;
                }
            }
        }
    }
    chelis_value items[2];
    items[0] = chelis_value_from_tensor(values);
    items[1] = chelis_value_from_tensor(indices);
    return chelis_tuple_from_values(items, 2);
}

chelis_tensor* chelis_tensor_diagonal(const chelis_tensor *tensor, int64_t axis1, int64_t axis2) {
    int axis1_i = chelis_tensor_normalize_axis(tensor, axis1, "diagonal");
    int axis2_i = chelis_tensor_normalize_axis(tensor, axis2, "diagonal");
    if (axis1_i == axis2_i) {
        fprintf(stderr, "diagonal expects distinct axes\n");
        abort();
    }
    int diag = tensor->shape[axis1_i] < tensor->shape[axis2_i] ? tensor->shape[axis1_i] : tensor->shape[axis2_i];
    int out_shape[CHELIS_MAX_DIM];
    int pos = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        if (i == axis1_i) out_shape[pos++] = diag;
        else if (i != axis2_i) out_shape[pos++] = tensor->shape[i];
    }
    chelis_tensor *out = chelis_alloc(tensor->ndim - 1, out_shape, tensor->dtype);
    int out_index[CHELIS_MAX_DIM];
    int src_index[CHELIS_MAX_DIM];
    for (int linear = 0; linear < out->size; linear++) {
        chelis_flat_to_indices(linear, out->shape, out->ndim, out_index);
        int diag_idx = out_index[axis1_i];
        int out_pos = 0;
        for (int i = 0; i < tensor->ndim; i++) {
            if (i == axis1_i || i == axis2_i) src_index[i] = diag_idx;
            else src_index[i] = out_index[out_pos++];
        }
        int src = chelis_indices_to_flat(src_index, tensor->strides, tensor->ndim);
        out->data[linear] = tensor->data[src];
    }
    return out;
}

chelis_tensor* chelis_tensor_trace(const chelis_tensor *tensor, int64_t axis1, int64_t axis2) {
    chelis_tensor *diag = chelis_tensor_diagonal(tensor, axis1, axis2);
    int reduce_axis = (axis1 < axis2 ? axis1 : axis2);
    int axis_size = diag->shape[reduce_axis];
    int out_shape[CHELIS_MAX_DIM];
    int pos = 0;
    for (int i = 0; i < diag->ndim; i++) {
        if (i != reduce_axis) out_shape[pos++] = diag->shape[i];
    }
    chelis_tensor *out = chelis_alloc(diag->ndim - 1, out_shape, diag->dtype);
    int inner = 1;
    int outer = 1;
    for (int i = reduce_axis + 1; i < diag->ndim; i++) inner *= diag->shape[i];
    for (int i = 0; i < reduce_axis; i++) outer *= diag->shape[i];
    for (int outer_idx = 0; outer_idx < outer; outer_idx++) {
        for (int inner_idx = 0; inner_idx < inner; inner_idx++) {
            float sum = 0.0f;
            for (int axis_idx = 0; axis_idx < axis_size; axis_idx++) {
                int linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                sum += diag->data[linear];
            }
            out->data[outer_idx * inner + inner_idx] = sum;
        }
    }
    chelis_free(diag);
    return out;
}

chelis_tensor* chelis_tensor_clamp(
    const chelis_tensor *tensor,
    const chelis_tensor *lo,
    const chelis_tensor *hi
) {
    if (!chelis_tensor_scalar_or_same_shape(lo, tensor) || !chelis_tensor_scalar_or_same_shape(hi, tensor)) {
        fprintf(stderr, "clamp expects scalar bounds or matching-shape tensor bounds\n");
        abort();
    }
    chelis_tensor *out = chelis_alloc(tensor->ndim, tensor->shape, tensor->dtype);
    for (int i = 0; i < out->size; i++) {
        float low = lo->ndim == 0 ? lo->data[0] : lo->data[i];
        float high = hi->ndim == 0 ? hi->data[0] : hi->data[i];
        float value = tensor->data[i];
        if (value < low) value = low;
        if (value > high) value = high;
        out->data[i] = value;
    }
    return out;
}

chelis_tensor* chelis_tensor_einsum(
    chelis_string equation,
    const chelis_tensor *lhs,
    const chelis_tensor *rhs
) {
    if (strstr(equation.data, "...") != NULL) {
        CHELIS_RUNTIME_FAIL("einsum ellipsis support is deferred in 3h\n");
    }
    if (lhs->dtype != rhs->dtype) {
        CHELIS_RUNTIME_FAIL("einsum expects matching tensor dtype\n");
    }

    const char *arrow = strstr(equation.data, "->");
    if (arrow == NULL) {
        CHELIS_RUNTIME_FAIL("einsum equation must contain explicit output\n");
    }
    const char *comma = strchr(equation.data, ',');
    if (comma == NULL || comma > arrow) {
        CHELIS_RUNTIME_FAIL("einsum 3h currently supports exactly two operands\n");
    }
    if (strchr(comma + 1, ',') != NULL && strchr(comma + 1, ',') < arrow) {
        CHELIS_RUNTIME_FAIL("einsum 3h currently supports exactly two operands\n");
    }

    int lhs_rank = (int)(comma - equation.data);
    int rhs_rank = (int)(arrow - comma - 1);
    int out_rank = (int)strlen(arrow + 2);
    if (lhs_rank != lhs->ndim || rhs_rank != rhs->ndim) {
        CHELIS_RUNTIME_FAIL("einsum label count must match operand rank\n");
    }
    if (lhs_rank > CHELIS_MAX_DIM || rhs_rank > CHELIS_MAX_DIM || out_rank > CHELIS_MAX_DIM) {
        CHELIS_RUNTIME_FAIL("einsum rank exceeds CHELIS_MAX_DIM\n");
    }

    char lhs_labels[CHELIS_MAX_DIM];
    char rhs_labels[CHELIS_MAX_DIM];
    char out_labels[CHELIS_MAX_DIM];
    memcpy(lhs_labels, equation.data, lhs_rank);
    memcpy(rhs_labels, comma + 1, rhs_rank);
    memcpy(out_labels, arrow + 2, out_rank);

    int label_dims[256];
    int label_values[256];
    bool out_contains[256];
    bool reduction_seen[256];
    memset(label_dims, -1, sizeof(label_dims));
    memset(label_values, 0, sizeof(label_values));
    memset(out_contains, 0, sizeof(out_contains));
    memset(reduction_seen, 0, sizeof(reduction_seen));

    for (int i = 0; i < out_rank; i++) {
        unsigned char label = (unsigned char)out_labels[i];
        out_contains[label] = true;
    }

    for (int i = 0; i < lhs_rank; i++) {
        unsigned char label = (unsigned char)lhs_labels[i];
        if (label_dims[label] >= 0 && label_dims[label] != lhs->shape[i]) {
            CHELIS_RUNTIME_FAIL(
                "einsum label `%c` has inconsistent extents\n",
                lhs_labels[i]
            );
        }
        label_dims[label] = lhs->shape[i];
    }
    for (int i = 0; i < rhs_rank; i++) {
        unsigned char label = (unsigned char)rhs_labels[i];
        if (label_dims[label] >= 0 && label_dims[label] != rhs->shape[i]) {
            CHELIS_RUNTIME_FAIL(
                "einsum label `%c` has inconsistent extents\n",
                rhs_labels[i]
            );
        }
        label_dims[label] = rhs->shape[i];
    }

    int out_shape[CHELIS_MAX_DIM];
    for (int i = 0; i < out_rank; i++) {
        unsigned char label = (unsigned char)out_labels[i];
        if (label_dims[label] < 0) {
            CHELIS_RUNTIME_FAIL(
                "einsum output label `%c` missing from inputs\n",
                out_labels[i]
            );
        }
        out_shape[i] = label_dims[label];
    }

    char reduction_labels[CHELIS_MAX_DIM];
    int reduction_shape[CHELIS_MAX_DIM];
    int reduction_rank = 0;
    for (int i = 0; i < lhs_rank; i++) {
        unsigned char label = (unsigned char)lhs_labels[i];
        if (!out_contains[label] && !reduction_seen[label]) {
            reduction_seen[label] = true;
            reduction_labels[reduction_rank] = lhs_labels[i];
            reduction_shape[reduction_rank] = label_dims[label];
            reduction_rank++;
        }
    }
    for (int i = 0; i < rhs_rank; i++) {
        unsigned char label = (unsigned char)rhs_labels[i];
        if (!out_contains[label] && !reduction_seen[label]) {
            reduction_seen[label] = true;
            reduction_labels[reduction_rank] = rhs_labels[i];
            reduction_shape[reduction_rank] = label_dims[label];
            reduction_rank++;
        }
    }

    chelis_tensor *out = chelis_alloc(out_rank, out_shape, lhs->dtype);
    int out_index[CHELIS_MAX_DIM];
    int reduction_index[CHELIS_MAX_DIM];
    int lhs_index[CHELIS_MAX_DIM];
    int rhs_index[CHELIS_MAX_DIM];
    int reduction_total = 1;
    for (int i = 0; i < reduction_rank; i++) {
        reduction_total *= reduction_shape[i];
    }

    for (int out_linear = 0; out_linear < out->size; out_linear++) {
        if (out_rank > 0) {
            chelis_flat_to_indices(out_linear, out_shape, out_rank, out_index);
        }
        for (int i = 0; i < out_rank; i++) {
            unsigned char label = (unsigned char)out_labels[i];
            label_values[label] = out_index[i];
        }

        float acc = 0.0f;
        for (int reduction_linear = 0; reduction_linear < reduction_total; reduction_linear++) {
            if (reduction_rank > 0) {
                chelis_flat_to_indices(
                    reduction_linear,
                    reduction_shape,
                    reduction_rank,
                    reduction_index
                );
            }
            for (int i = 0; i < reduction_rank; i++) {
                unsigned char label = (unsigned char)reduction_labels[i];
                label_values[label] = reduction_index[i];
            }
            for (int i = 0; i < lhs_rank; i++) {
                lhs_index[i] = label_values[(unsigned char)lhs_labels[i]];
            }
            for (int i = 0; i < rhs_rank; i++) {
                rhs_index[i] = label_values[(unsigned char)rhs_labels[i]];
            }
            acc += lhs->data[chelis_indices_to_flat(lhs_index, lhs->strides, lhs_rank)]
                * rhs->data[chelis_indices_to_flat(rhs_index, rhs->strides, rhs_rank)];
        }
        out->data[out_linear] = acc;
    }
    return out;
}

static void chelis_print_value_inline(chelis_value value) {
    switch (value.tag) {
        case CHELIS_VALUE_INT64:
            printf("%lld", (long long)value.as.i64);
            break;
        case CHELIS_VALUE_FLOAT64:
            printf("%g", value.as.f64);
            break;
        case CHELIS_VALUE_BOOL:
            printf("%s", value.as.boolean ? "true" : "false");
            break;
        case CHELIS_VALUE_STRING:
            printf("%s", value.as.string.data);
            break;
        case CHELIS_VALUE_TENSOR:
            chelis_print_f32(value.as.tensor);
            break;
        case CHELIS_VALUE_LIST:
            chelis_print_list(value.as.list);
            break;
        case CHELIS_VALUE_TUPLE:
            chelis_print_tuple(value.as.tuple);
            break;
        case CHELIS_VALUE_DICT:
            chelis_print_dict(value.as.dict);
            break;
    }
}

void chelis_print_list(const chelis_list *list) {
    printf("[");
    for (int64_t i = 0; list && i < list->len; i++) {
        if (i) printf(", ");
        chelis_print_value_inline(list->items[i]);
    }
    printf("]");
}

void chelis_print_tuple(const chelis_tuple *tuple) {
    printf("(");
    for (int64_t i = 0; tuple && i < tuple->len; i++) {
        if (i) printf(", ");
        chelis_print_value_inline(tuple->items[i]);
    }
    printf(")");
}

void chelis_print_dict(const chelis_dict *dict) {
    printf("dict(");
    for (int64_t i = 0; dict && i < dict->len; i++) {
        if (i) printf(", ");
        chelis_print_value_inline(dict->entries[i].key);
        printf(": ");
        chelis_print_value_inline(dict->entries[i].value);
    }
    printf(")");
}

chelis_tensor* chelis_contiguous(const chelis_tensor *t) {
    if (chelis_is_contiguous(t)) {
        /* Already contiguous, just copy */
        chelis_tensor *out = chelis_alloc(t->ndim, t->shape, t->dtype);
        memcpy(out->data, t->data, t->size * sizeof(float));
        return out;
    }
    chelis_tensor *out = chelis_alloc(t->ndim, t->shape, t->dtype);
    int indices[CHELIS_MAX_DIM];
    for (int i = 0; i < out->size; i++) {
        chelis_flat_to_indices(i, out->shape, out->ndim, indices);
        int src = chelis_indices_to_flat(indices, t->strides, t->ndim);
        out->data[i] = t->data[src];
    }
    return out;
}

void chelis_print_f32(const chelis_tensor *t) {
    printf("tensor(shape=[");
    for (int d = 0; d < t->ndim; d++) { if (d) printf(", "); printf("%d", t->shape[d]); }
    printf("], data=[");
    int n = t->size < 10 ? t->size : 10;
    for (int i = 0; i < n; i++) {
        double value = t->data[i];
        if (i) printf(", ");
        if (fabs(value - round(value)) < 1e-9) {
            printf("%.1f", value);
        } else {
            printf("%g", value);
        }
    }
    if (t->size > 10) printf(", ...");
    printf("])");
}
