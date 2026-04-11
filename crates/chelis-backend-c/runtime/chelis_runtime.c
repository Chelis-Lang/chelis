#include "chelis_runtime.h"
#include <ctype.h>

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
    printf("])\n");
}
