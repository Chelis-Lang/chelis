#ifndef CHELIS_RUNTIME_H
#define CHELIS_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_simd.h"
#include "chelis_runtime_dtype.h"
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
/* Element size in bytes for the given CHELIS_* dtype tag. Mirrors the
 * per-dtype dispatch inside `chelis_alloc` and the GPU-side
 * `chelis_gpu_dtype_size`. Generated C code calls this when sizing
 * memcpys / per-element strides so the byte stride matches the storage
 * layout. RT-4 F2/F3 fix: replaces hardcoded `sizeof(float)` in the C
 * backend's reshape and cast emitters. */
int chelis_dtype_size(int dtype);
void chelis_free(chelis_tensor *t);
void chelis_fill_f32(chelis_tensor *t, float val);
void chelis_fill_i64(chelis_tensor *t, int64_t val);
void chelis_fill_f64(chelis_tensor *t, double val);
/* Issue #189: bit-pattern fill helpers for f32 / f64 Const emission.
 * Codegen computes the IEEE 754 bit pattern of the source value at
 * compile time (`f32::to_bits()` / `f64::to_bits()`) and emits
 * `chelis_fill_f32_bits(t, 0xXXXXXXXXu)` / `chelis_fill_f64_bits(t,
 * 0xXXXXXXXXXXXXXXXXuLL)`. The runtime bit-casts the integer pattern
 * back to the IEEE 754 value before filling, so the constant is
 * bit-identical to the source value -- avoiding the lossy
 * decimal-format-string round-trip that the pre-fix emitter used. The
 * shape mirrors `chelis_fill_bf16` / `chelis_fill_f16`. */
void chelis_fill_f32_bits(chelis_tensor *t, uint32_t bits);
void chelis_fill_f64_bits(chelis_tensor *t, uint64_t bits);
/* Issue #365: bit-pattern fill helper for Bool tensors. A Bool tensor uses
 * the same 4-byte f32-encoded storage (0.0 / 1.0) the comparison ops write,
 * but its dtype tag is CHELIS_BOOL. Filling it through
 * chelis_fill_f32_bits trips the debug-build dtype assertion; this helper
 * asserts CHELIS_BOOL and fills the f32-encoded storage so a debug-runtime
 * reduce/softmax/cross-entropy backward mask fill is dtype-correct. */
/* CHELIS_BOOL is one native byte; pass 0 or 1. Any non-zero byte is true.
   This was chelis_fill_bool_bits(t, uint32_t) taking an IEEE binary32 pattern,
   back when bool storage was an f32 payload (CRuntime-BoolStorage-F1). */
void chelis_fill_bool(chelis_tensor *t, uint8_t value);
/* Issue #248: scalar bit-pattern reconstruction helpers. The C backend
 * emits `chelis_uniform_sample_f32(..., chelis_f32_from_bits(0xXXXXXXXXu),
 * chelis_f32_from_bits(0xYYYYYYYYu))` so the runtime sees the byte-identical
 * f32 narrowing of the source `low` / `high` instead of a `%.8` decimal
 * round-trip. Symmetric with `chelis_fill_f32_bits` / `chelis_fill_f64_bits`
 * but for per-call scalar args rather than buffer fills, so a `static inline`
 * bit-cast suffices; no Rust-side `extern "C"` symbol is needed. */
static inline float chelis_f32_from_bits(uint32_t bits) {
    float v;
    memcpy(&v, &bits, sizeof(float));
    return v;
}
static inline double chelis_f64_from_bits(uint64_t bits) {
    double v;
    memcpy(&v, &bits, sizeof(double));
    return v;
}
/* Issue #387: portable integer division/remainder by-zero trap. The
 * evaluator halts with a clean diagnostic; the C backend must do the same
 * on every platform. Relying on the hardware fault is NOT portable: x86
 * raises SIGFPE on integer #DE, but ARM64 (e.g. macOS arm64) defines
 * integer division by zero to return a value and does NOT fault, so the
 * binary would silently compute a wrong answer -- the exact eval-vs-backend
 * divergence #387 exists to kill. Codegen calls this guard before every
 * INTEGER `div` / `mod`; it returns the divisor so the call composes inline
 * (`a / chelis_int_div_guard(b)`). Float division is IEEE-754 (`1.0/0.0 ==
 * inf`) and is never guarded. The message matches the evaluator's
 * `integer division or remainder by zero` exactly. */
static inline int64_t chelis_int_div_guard(int64_t divisor) {
    if (divisor == 0) {
        fprintf(stderr, "integer division or remainder by zero\n");
        abort();
    }
    return divisor;
}
/* WS-1 (dtype + Metal cleanup cycle): two-byte fill helpers for bf16
 * and f16 tensors. Codegen computes the exact 16-bit pattern from the
 * IR literal at compile time (the `half` crate's `to_bits()`) and
 * passes it as `bits`; the runtime writes that pattern into every
 * element so the storage round-trips exactly. */
void chelis_fill_bf16(chelis_tensor *t, uint16_t bits);
void chelis_fill_f16(chelis_tensor *t, uint16_t bits);
/* WS-1 buffer-conversion helpers used by the C backend's bf16/f16
 * matmul wrapper (convert-then-`cblas_sgemm`). The host-side
 * arithmetic story for bf16/f16 is "always go through f32"; matmul
 * batches the conversion to amortize the per-element cost across the
 * GEMM call. */
void chelis_bf16_buffer_to_f32(const uint16_t *src, float *dst, int64_t n);
void chelis_f32_buffer_to_bf16(const float *src, uint16_t *dst, int64_t n);
void chelis_f16_buffer_to_f32(const uint16_t *src, float *dst, int64_t n);
void chelis_f32_buffer_to_f16(const float *src, uint16_t *dst, int64_t n);
chelis_tensor *chelis_scalar_tensor_from_i64(int64_t value);
chelis_tensor *chelis_scalar_tensor_from_f64(double value);
chelis_tensor *chelis_scalar_tensor_from_f32(float value);
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
/* RT-4 F1: dtype-aware variant. Honors the declared destination dtype
 * for both allocation and per-element writes. The C backend calls this
 * when the surface-level annotation disambiguates storage width
 * (e.g. `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]`). */
chelis_tensor *chelis_tensor_from_value_list_typed(const chelis_list *list, int dst_dtype);
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
_Noreturn void chelis_fail(chelis_string message);
chelis_string chelis_read_file(chelis_string path);
void chelis_write_file(chelis_string path, chelis_string contents);
chelis_list *chelis_read_lines(chelis_string path);
chelis_list *chelis_read_bytes(chelis_string path);
bool chelis_file_exists(chelis_string path);
chelis_list *chelis_list_dir(chelis_string path);
chelis_mapped_file *chelis_mmap_file(chelis_string path);
chelis_list *chelis_mmap_read(const chelis_mapped_file *mapped, int64_t offset, int64_t len);
int64_t chelis_mmap_len(const chelis_mapped_file *mapped);

/*
 * Stable tuple ABI for generated C drivers:
 *
 * - Construct tuples with `chelis_tuple_from_values(...)` after boxing each item with
 *   the matching `chelis_value_from_*` helper.
 * - Extract typed items from tuple-returning Chelis functions with the helpers below.
 */
static inline int64_t chelis_tuple_get_int64(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_int64(chelis_tuple_get(tuple, index));
}

static inline double chelis_tuple_get_f64(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_f64(chelis_tuple_get(tuple, index));
}

static inline bool chelis_tuple_get_bool(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_bool(chelis_tuple_get(tuple, index));
}

static inline chelis_string chelis_tuple_get_string(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_string(chelis_tuple_get(tuple, index));
}

static inline chelis_tensor *chelis_tuple_get_tensor(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_tensor(chelis_tuple_get(tuple, index));
}

static inline chelis_list *chelis_tuple_get_list(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_list(chelis_tuple_get(tuple, index));
}

static inline chelis_tuple *chelis_tuple_get_tuple(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_tuple(chelis_tuple_get(tuple, index));
}

static inline chelis_dict *chelis_tuple_get_dict(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_dict(chelis_tuple_get(tuple, index));
}

static inline chelis_adt *chelis_tuple_get_adt(const chelis_tuple *tuple, int64_t index) {
    return chelis_value_as_adt(chelis_tuple_get(tuple, index));
}

/*
 * WS-1: bf16 / f16 per-element conversion to and from f32.
 *
 * Storage layout for both formats is a 16-bit unsigned integer. The
 * generated C kernels read a 16-bit element, convert to f32, perform
 * arithmetic in f32, convert back, and store. Per spec/04-type-system.md
 * §5.7.1 the matmul accumulator is f32 and the reduction accumulator
 * for bf16/f16 reduce_sum is also f32; both fall out naturally from
 * routing through these helpers.
 *
 * bf16 encoding: sign(1) exp(8) mantissa(7). The bit pattern matches
 * the upper half of an IEEE 754 binary32. Conversion to f32 is a
 * left-shift-by-16 into the high half of the bit pattern; conversion
 * from f32 rounds to nearest even.
 *
 * f16 encoding: sign(1) exp(5) mantissa(10) per IEEE 754 binary16.
 * Conversion handles subnormals, infinity, NaN, and the smaller
 * exponent range; conversion from f32 rounds to nearest even.
 *
 * NaN propagation: every conversion preserves NaN-ness (the result is
 * also a NaN), though the exact NaN payload is not preserved across
 * f32 -> reduced -> f32 round trips. Inf is preserved exactly.
 *
 * All helpers are `static inline` so the C compiler can fold the
 * conversion into the surrounding kernel loop and emit SIMD-friendly
 * code without a function call per element.
 */
static inline float chelis_bf16_to_f32(uint16_t bits) {
    uint32_t expanded = ((uint32_t)bits) << 16;
    float out;
    memcpy(&out, &expanded, sizeof(out));
    return out;
}

static inline uint16_t chelis_f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    /* NaN: preserve the most-significant mantissa bit so the result is
     * still a NaN (not silently coerced to inf). */
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) {
        return (uint16_t)((bits >> 16) | 0x0040u);
    }
    /* Round-to-nearest-even on the discarded 16 mantissa bits. */
    uint32_t lsb = (bits >> 16) & 1u;
    uint32_t rounded = bits + 0x7FFFu + lsb;
    return (uint16_t)(rounded >> 16);
}

static inline float chelis_f16_to_f32(uint16_t bits) {
    uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)bits & 0x7C00u) >> 10;
    uint32_t mant = (uint32_t)bits & 0x03FFu;
    uint32_t out_bits;
    if (exp == 0u) {
        if (mant == 0u) {
            out_bits = sign;
        } else {
            /* Subnormal: normalize. */
            int shift = 0;
            while ((mant & 0x0400u) == 0u) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x03FFu;
            uint32_t exp32 = (uint32_t)(127 - 15 - shift + 1);
            out_bits = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        /* Inf or NaN. */
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t exp32 = exp + (127u - 15u);
        out_bits = sign | (exp32 << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &out_bits, sizeof(out));
    return out;
}

static inline uint16_t chelis_f32_to_f16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {
        /* Inf / NaN: preserve. */
        if (mant != 0u) {
            return (uint16_t)(sign | 0x7E00u);
        }
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp >= 0x1F) {
        /* Overflow to inf. */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {
        /* Subnormal or zero in f16. */
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x00800000u) >> (1 - exp);
        uint32_t rounded = mant + 0x00001000u;
        return (uint16_t)(sign | (rounded >> 13));
    }
    /* Normal: round-to-nearest-even on the discarded 13 mantissa bits. */
    uint32_t lsb = (mant >> 13) & 1u;
    uint32_t rounded = mant + 0x00000FFFu + lsb;
    if ((rounded & 0x00800000u) != 0u) {
        rounded = 0u;
        exp += 1;
        if (exp >= 0x1F) {
            return (uint16_t)(sign | 0x7C00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (rounded >> 13));
}

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
