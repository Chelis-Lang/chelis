#ifndef CHELIS_RUNTIME_H
#define CHELIS_RUNTIME_H

#include <stdbool.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_simd.h"
#include "chelis_runtime_dtype.h"

typedef struct {
    void *handle;
} chelis_string;

typedef struct chelis_list chelis_list;
typedef struct chelis_tensor chelis_tensor;
typedef struct chelis_tensor_write chelis_tensor_write;
typedef struct chelis_tuple chelis_tuple;
typedef struct chelis_dict chelis_dict;
typedef struct chelis_adt chelis_adt;
typedef struct chelis_option chelis_option;
typedef struct chelis_mapped_file chelis_mapped_file;

typedef struct { chelis_dtype dtype; uint8_t reserved[7]; uint64_t bits; } chelis_scalar;
typedef struct { uint64_t bits; } chelis_key;
typedef uint8_t chelis_value_tag;
enum { CHELIS_VALUE_UNIT = 0, CHELIS_VALUE_SCALAR = 1, CHELIS_VALUE_STRING = 2, CHELIS_VALUE_TENSOR = 3, CHELIS_VALUE_LIST = 4, CHELIS_VALUE_TUPLE = 5, CHELIS_VALUE_DICT = 6, CHELIS_VALUE_ADT = 7, CHELIS_VALUE_OPTION = 8, CHELIS_VALUE_MAPPED_FILE = 9 };
typedef union { chelis_scalar scalar; void *handle; } chelis_value_payload;
typedef struct { chelis_value_tag tag; uint8_t reserved[7]; chelis_value_payload payload; } chelis_value;
#include "chelis_runtime_views.h"
typedef struct { chelis_value key; chelis_value value; } chelis_dict_entry;

#ifdef __cplusplus
extern "C" {
#endif

chelis_tensor *chelis_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype);
chelis_tensor *chelis_tensor_alloc_like(const chelis_tensor *input, chelis_scalar exemplar);
chelis_tensor *chelis_tensor_entry_borrow(int32_t rank, const int64_t *shape, chelis_dtype dtype, const void *data, int64_t byte_capacity);
void chelis_tensor_retain(const chelis_tensor *tensor);
void chelis_tensor_release(const chelis_tensor *tensor);
/* A successful chelis_tensor_begin_write invalidates every prior read view; dereferencing a stale view violates the caller precondition. */
chelis_read_view chelis_tensor_read_view(const chelis_tensor *tensor);
void chelis_tensor_repurpose(chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape);
chelis_tensor_write *chelis_tensor_begin_write(chelis_tensor *tensor);
chelis_write_view chelis_tensor_write_view(const chelis_tensor_write *guard);
void chelis_tensor_end_write(chelis_tensor_write *guard);
/* Element size in bytes for the given CHELIS_* dtype tag. Mirrors the
 * per-dtype dispatch inside `chelis_alloc` and the GPU-side
 * `chelis_gpu_dtype_size`. Generated C code calls this when sizing
 * memcpys / per-element strides so the byte stride matches the storage
 * layout. RT-4 F2/F3 fix: replaces hardcoded `sizeof(float)` in the C
 * backend's reshape and cast emitters. */
int64_t chelis_dtype_size(chelis_dtype dtype);
/* Issue #248: scalar bit-pattern reconstruction helpers. The C backend
 * emits `chelis_uniform_sample_f32(..., chelis_f32_from_bits(0xXXXXXXXXu),
 * chelis_f32_from_bits(0xYYYYYYYYu))` so the runtime sees the byte-identical
 * f32 narrowing of the source `low` / `high` instead of a `%.8` decimal
 * round-trip. A `static inline` bit-cast suffices; no Rust-side symbol is
 * needed. */
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
/* spec/04 section 4.7: output from preceding effects survives a later trap.
 * abort() need not flush C streams (notably on glibc). Preserve the original
 * failure even if a stream itself cannot be flushed. */
static inline void chelis_flush_and_abort(void) {
    (void)fflush(stdout);
    (void)fflush(stderr);
    abort();
}

static inline int64_t chelis_int_div_guard(int64_t divisor) {
    if (divisor == 0) {
        fprintf(stderr, "integer division or remainder by zero\n");
        chelis_flush_and_abort();
    }
    return divisor;
}
/* chelis#729 Phase 3: exact signed-integer absolute value. The generated
 * caller supplies the declared width and the frozen C2 diagnostic produced
 * from chelis_types::dtype_semantics::NumericTrap. Checking the minimum
 * before negation avoids C signed-overflow undefined behavior. */
static inline int64_t chelis_int_abs_guard(int64_t value, int bits,
                                          const char *trap_message) {
    int64_t minimum;
    switch (bits) {
        case 8: minimum = INT8_MIN; break;
        case 16: minimum = INT16_MIN; break;
        case 32: minimum = INT32_MIN; break;
        case 64: minimum = INT64_MIN; break;
        default:
            fprintf(stderr, "chelis internal error: invalid integer abs width %d\n", bits);
            chelis_flush_and_abort();
            /* Keep the published header warning-clean even when a C/C++
             * compiler does not infer abort's non-returning contract through
             * this inline wrapper. The return is unreachable. */
            return value;
    }
    if (value == minimum) {
        fprintf(stderr, "%s\n", trap_message);
        chelis_flush_and_abort();
    }
    return value < 0 ? -value : value;
}

/* chelis#729 Phase 3: checked signed arithmetic at the declared width.
 * Every check happens before the C operation, so signed-overflow undefined
 * behavior is unreachable. These are header-local implementation helpers,
 * not exported runtime callables. */
static inline void chelis_int_limits(int bits, int64_t *minimum, int64_t *maximum) {
    switch (bits) {
        case 8: *minimum = INT8_MIN; *maximum = INT8_MAX; break;
        case 16: *minimum = INT16_MIN; *maximum = INT16_MAX; break;
        case 32: *minimum = INT32_MIN; *maximum = INT32_MAX; break;
        case 64: *minimum = INT64_MIN; *maximum = INT64_MAX; break;
        default:
            fprintf(stderr, "chelis internal error: invalid integer width %d\n", bits);
            chelis_flush_and_abort();
    }
}

static inline void chelis_numeric_trap(const char *message) {
    fprintf(stderr, "%s\n", message);
    chelis_flush_and_abort();
}

static inline int64_t chelis_int_checked_add(int64_t lhs, int64_t rhs, int bits,
                                             const char *trap_message) {
    int64_t minimum, maximum;
    chelis_int_limits(bits, &minimum, &maximum);
    if ((rhs > 0 && lhs > maximum - rhs) || (rhs < 0 && lhs < minimum - rhs)) {
        chelis_numeric_trap(trap_message);
    }
    return lhs + rhs;
}

static inline int64_t chelis_int_checked_sub(int64_t lhs, int64_t rhs, int bits,
                                             const char *trap_message) {
    int64_t minimum, maximum;
    chelis_int_limits(bits, &minimum, &maximum);
    if ((rhs < 0 && lhs > maximum + rhs) || (rhs > 0 && lhs < minimum + rhs)) {
        chelis_numeric_trap(trap_message);
    }
    return lhs - rhs;
}

static inline int64_t chelis_int_checked_mul(int64_t lhs, int64_t rhs, int bits,
                                             const char *trap_message) {
    int64_t minimum, maximum;
    chelis_int_limits(bits, &minimum, &maximum);
    if (lhs != 0 && rhs != 0) {
        if ((lhs == -1 && rhs == minimum) || (rhs == -1 && lhs == minimum)) {
            chelis_numeric_trap(trap_message);
        }
        if ((lhs > 0 && rhs > 0 && lhs > maximum / rhs) ||
            (lhs > 0 && rhs < 0 && rhs < minimum / lhs) ||
            (lhs < 0 && rhs > 0 && lhs < minimum / rhs) ||
            (lhs < 0 && rhs < 0 && lhs < maximum / rhs)) {
            chelis_numeric_trap(trap_message);
        }
    }
    return lhs * rhs;
}

static inline int64_t chelis_int_checked_neg(int64_t value, int bits,
                                             const char *trap_message) {
    int64_t minimum, maximum;
    chelis_int_limits(bits, &minimum, &maximum);
    (void)maximum;
    if (value == minimum) chelis_numeric_trap(trap_message);
    return -value;
}

static inline int64_t chelis_int_checked_divisor(int64_t dividend, int64_t divisor,
                                                 int bits, const char *zero_message,
                                                 const char *overflow_message) {
    int64_t minimum, maximum;
    chelis_int_limits(bits, &minimum, &maximum);
    (void)maximum;
    if (divisor == 0) chelis_numeric_trap(zero_message);
    if (dividend == minimum && divisor == -1) chelis_numeric_trap(overflow_message);
    return divisor;
}

static inline int64_t chelis_checked_int_cast(int64_t value, int bits,
                                              const char *overflow_message) {
    int64_t minimum, maximum;
    chelis_int_limits(bits, &minimum, &maximum);
    if (value < minimum || value > maximum) chelis_numeric_trap(overflow_message);
    return value;
}

static inline int64_t chelis_checked_float_to_int(double value, int bits,
                                                  const char *domain_message,
                                                  const char *overflow_message) {
    if (!isfinite(value) || trunc(value) != value) chelis_numeric_trap(domain_message);
    if (bits == 64) {
        if (value < -9223372036854775808.0 || value >= 9223372036854775808.0) {
            chelis_numeric_trap(overflow_message);
        }
    } else {
        int64_t minimum, maximum;
        chelis_int_limits(bits, &minimum, &maximum);
        if (value < (double)minimum || value > (double)maximum) {
            chelis_numeric_trap(overflow_message);
        }
    }
    return (int64_t)value;
}

/* [05-OP-6] `cast_trunc`: truncate toward zero, then range-check at the
   target width. Unlike chelis_checked_float_to_int, a fractional value is
   the WHOLE point and is not a domain error; only a non-finite source is.
   Truncating BEFORE the range check is what makes the two lanes agree:
   the evaluator range-checks its already-truncated value too, so e.g.
   2147483647.9 -> int32 succeeds in both rather than overflowing here and
   succeeding there. */
static inline int64_t chelis_trunc_float_to_int(double value, int bits,
                                                const char *domain_message,
                                                const char *overflow_message) {
    double truncated;
    if (!isfinite(value)) chelis_numeric_trap(domain_message);
    truncated = trunc(value);
    if (bits == 64) {
        if (truncated < -9223372036854775808.0 || truncated >= 9223372036854775808.0) {
            chelis_numeric_trap(overflow_message);
        }
    } else {
        int64_t minimum, maximum;
        chelis_int_limits(bits, &minimum, &maximum);
        if (truncated < (double)minimum || truncated > (double)maximum) {
            chelis_numeric_trap(overflow_message);
        }
    }
    return (int64_t)truncated;
}

static inline bool chelis_checked_bool_from_int(int64_t value,
                                                const char *domain_message) {
    if (value != 0 && value != 1) chelis_numeric_trap(domain_message);
    return value == 1;
}

static inline bool chelis_checked_bool_from_float(double value,
                                                  const char *domain_message) {
    if (!isfinite(value) || (value != 0.0 && value != 1.0)) {
        chelis_numeric_trap(domain_message);
    }
    return value == 1.0;
}
/* [04-NUM-13] / chelis#682: width-bounded two's-complement shifts.
 * Generated code must not use C's signed shift operators directly:
 * left-shifting a negative value or into the sign bit is undefined, a
 * negative count is undefined, and a count at least the promoted width is
 * undefined. These helpers perform every bit movement on uint64_t, then
 * decode the declared-width two's-complement result without an
 * out-of-range unsigned-to-signed cast. */
static inline uint64_t chelis_int_width_mask(int bits) {
    if (bits == 64) return UINT64_MAX;
    return (UINT64_C(1) << bits) - UINT64_C(1);
}

static inline int64_t chelis_int_from_twos(uint64_t value, int bits) {
    uint64_t mask = chelis_int_width_mask(bits);
    uint64_t sign = UINT64_C(1) << (bits - 1);
    value &= mask;
    if ((value & sign) == 0) return (int64_t)value;
    uint64_t magnitude = ((~value) & mask) + UINT64_C(1);
    if (bits == 64 && magnitude == (UINT64_C(1) << 63)) return INT64_MIN;
    return -(int64_t)magnitude;
}

static inline void chelis_int_shift_validate(int64_t amount, int bits) {
    if (bits != 8 && bits != 16 && bits != 32 && bits != 64) {
        fprintf(stderr, "invalid integer shift width: %d\n", bits);
        chelis_flush_and_abort();
    }
    if (amount < 0) {
        fprintf(stderr, "shift amount must be non-negative, got %lld\n",
                (long long)amount);
        chelis_flush_and_abort();
    }
}

static inline int64_t chelis_int_shl(int64_t value, int64_t amount, int bits) {
    chelis_int_shift_validate(amount, bits);
    if (amount >= bits) return 0;
    uint64_t mask = chelis_int_width_mask(bits);
    uint64_t shifted = (((uint64_t)value & mask) << (uint32_t)amount) & mask;
    return chelis_int_from_twos(shifted, bits);
}

static inline int64_t chelis_int_shr(int64_t value, int64_t amount, int bits) {
    chelis_int_shift_validate(amount, bits);
    uint64_t mask = chelis_int_width_mask(bits);
    uint64_t raw = (uint64_t)value & mask;
    bool negative = (raw & (UINT64_C(1) << (bits - 1))) != 0;
    if (amount >= bits) return negative ? -1 : 0;
    uint32_t count = (uint32_t)amount;
    uint64_t shifted = raw >> count;
    if (negative && count != 0) shifted |= mask ^ (mask >> count);
    return chelis_int_from_twos(shifted, bits);
}
chelis_scalar chelis_scalar_from_bits(chelis_dtype dtype, uint64_t bits);
/* [05-OP-69]: the key of a seed, whose bits are the seed's two's-complement
 * bits with no mixing. A key is never a bare integer at the boundary: a public
 * entry takes and returns it as this carrier (spec/08 section 2). */
chelis_key chelis_key_from_seed(int64_t seed);
chelis_value chelis_value_box_scalar(chelis_scalar value);
chelis_scalar chelis_value_unbox_scalar(chelis_value value);
chelis_tensor *chelis_scalar_tensor(chelis_scalar value);
chelis_scalar chelis_tensor_to_scalar(const chelis_tensor *tensor);
void chelis_fill_scalar(chelis_tensor_write *guard, chelis_scalar value);
chelis_string chelis_string_from_scalar(chelis_scalar value);
chelis_option *chelis_parse_scalar(chelis_string text, chelis_dtype dtype);
chelis_option *chelis_dict_get_scalar(const chelis_dict *dict, chelis_value key, chelis_dtype dtype);
int32_t chelis_tensor_rank(const chelis_tensor *tensor);
int64_t chelis_tensor_shape(const chelis_tensor *tensor, int32_t axis);
int64_t chelis_tensor_numel(const chelis_tensor *tensor);
int64_t chelis_tensor_stride(const chelis_tensor *tensor, int32_t axis);
int64_t chelis_tensor_byte_count(const chelis_tensor *tensor);
int64_t chelis_tensor_elementwise_index_step(const chelis_tensor *input, const chelis_tensor *domain);
int64_t chelis_tensor_elementwise_index_step_for_shape(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *shape);
void chelis_tensor_unravel_index(const chelis_tensor *tensor, chelis_scalar index, chelis_scalar *coordinates);
int64_t chelis_tensor_flat_index(const chelis_tensor *tensor, const chelis_scalar *coordinates);
void chelis_tensor_check_permute(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape, const chelis_scalar *axes);
void chelis_tensor_check_expand(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape, int32_t axis);
void chelis_tensor_pad_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *before, const chelis_scalar *after, chelis_scalar *shape);
void chelis_tensor_shrink_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *start, const chelis_scalar *end, chelis_scalar *shape);
void chelis_tensor_stride_shape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *steps, chelis_scalar *shape);
int64_t chelis_tensor_affine_index(const chelis_tensor *tensor, const chelis_scalar *coordinates, const chelis_scalar *offsets, const chelis_scalar *steps);

/* OP33 checked metadata only: no tensor payload allocation or ownership. */
typedef struct chelis_metadata_plan chelis_metadata_plan;
chelis_metadata_plan *chelis_metadata_plan_new(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar exemplar);
chelis_metadata_plan *chelis_metadata_plan_view(chelis_scalar rank, const chelis_scalar *shape, const chelis_scalar *strides, chelis_scalar exemplar, chelis_scalar byte_capacity);
int32_t chelis_metadata_plan_rank(const chelis_metadata_plan *plan);
const int64_t *chelis_metadata_plan_shape(const chelis_metadata_plan *plan);
const int64_t *chelis_metadata_plan_strides(const chelis_metadata_plan *plan);
int64_t chelis_metadata_plan_count(const chelis_metadata_plan *plan);
int64_t chelis_metadata_plan_byte_count(const chelis_metadata_plan *plan);
int64_t chelis_metadata_plan_byte_offset(const chelis_metadata_plan *plan, chelis_scalar linear_index);
chelis_dtype chelis_metadata_plan_dtype(const chelis_metadata_plan *plan);
void chelis_metadata_plan_check_capacity(const chelis_metadata_plan *plan, chelis_scalar byte_capacity);
void chelis_metadata_plan_release(chelis_metadata_plan *plan);

typedef enum {
    CHELIS_REDUCE_SUM = 0, CHELIS_REDUCE_COUNT = 1, CHELIS_REDUCE_MAX = 2,
    CHELIS_REDUCE_MIN = 3, CHELIS_REDUCE_PROD = 4, CHELIS_REDUCE_ARGMAX = 5,
    CHELIS_REDUCE_ARGMIN = 6
} chelis_reduction_op;
typedef struct chelis_reduction_plan chelis_reduction_plan;
typedef enum {
    CHELIS_SPARSE_GATHER = 0, CHELIS_SPARSE_ADD = 1,
    CHELIS_SPARSE_REPLACE = 2, CHELIS_SPARSE_ELEMENTS = 3
} chelis_sparse_op;
void chelis_tensor_check_literal(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar exemplar, chelis_scalar count);
void chelis_tensor_write_literal(chelis_tensor_write *guard, chelis_scalar count, const chelis_scalar *values);

// Checked valid-padding geometry; plans retain no tensor storage.
typedef enum { CHELIS_WINDOW_SUM = 0, CHELIS_WINDOW_MEAN = 1, CHELIS_WINDOW_MAX = 2, CHELIS_WINDOW_MIN = 3, CHELIS_WINDOW_GRAD = 4 } chelis_window_op;
typedef enum { CHELIS_WINDOW_SOURCE = 0, CHELIS_WINDOW_RESULT = 1 } chelis_window_side;
typedef struct chelis_movement_plan chelis_movement_plan;
typedef enum { CHELIS_MOVEMENT_EXPAND = 0, CHELIS_MOVEMENT_INSERT = 1, CHELIS_MOVEMENT_PAD = 2, CHELIS_MOVEMENT_SHRINK = 3, CHELIS_MOVEMENT_STRIDE = 4 } chelis_movement_op;
typedef enum { CHELIS_MOVEMENT_SOURCE = 0, CHELIS_MOVEMENT_RESULT = 1 } chelis_movement_side;
chelis_movement_plan *chelis_tensor_permute_plan(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *axes);
chelis_movement_plan *chelis_tensor_expand_plan(const chelis_tensor *input, chelis_scalar axis, chelis_scalar size, chelis_movement_op operation);
chelis_movement_plan *chelis_tensor_affine_plan(const chelis_tensor *input, chelis_scalar rank, const chelis_scalar *first, const chelis_scalar *second, chelis_movement_op operation);
int64_t chelis_movement_extent(const chelis_movement_plan *plan, chelis_movement_side side, chelis_scalar axis);
int64_t chelis_movement_count(const chelis_movement_plan *plan);
int64_t chelis_movement_index(const chelis_movement_plan *plan, chelis_scalar linear);
void chelis_movement_check_target(const chelis_movement_plan *plan, chelis_scalar rank, const chelis_scalar *shape);
void chelis_movement_plan_release(chelis_movement_plan *plan);
typedef struct chelis_window_plan chelis_window_plan;
chelis_window_plan *chelis_tensor_window_plan(const chelis_tensor *input, chelis_scalar count, const chelis_scalar *window, const chelis_scalar *steps, chelis_window_op operation);
int64_t chelis_window_extent(const chelis_window_plan *plan, chelis_window_side side, chelis_scalar axis);
int64_t chelis_window_count(const chelis_window_plan *plan);
int64_t chelis_window_index(const chelis_window_plan *plan, chelis_scalar group, chelis_scalar leaf);
void chelis_window_check_tensor(const chelis_window_plan *plan, const chelis_tensor *tensor, chelis_window_side side);
void chelis_window_check_target(const chelis_window_plan *plan, chelis_window_side side, chelis_scalar rank, const chelis_scalar *shape);
void chelis_window_plan_release(chelis_window_plan *plan);

typedef enum { CHELIS_MATMUL_LEFT = 0, CHELIS_MATMUL_RIGHT = 1, CHELIS_MATMUL_RESULT = 2 } chelis_matmul_part;
typedef enum { CHELIS_MATMUL_ROWS = 0, CHELIS_MATMUL_COLUMNS = 1, CHELIS_MATMUL_REDUCTION = 2 } chelis_matmul_dimension_kind;
typedef struct chelis_matmul_plan chelis_matmul_plan;
chelis_matmul_plan *chelis_tensor_matmul_plan(const chelis_tensor *left, const chelis_tensor *right, chelis_scalar exemplar);
int64_t chelis_matmul_extent(const chelis_matmul_plan *plan, chelis_scalar axis);
int64_t chelis_matmul_dimension(const chelis_matmul_plan *plan, chelis_matmul_dimension_kind dimension);
int64_t chelis_matmul_batch_count(const chelis_matmul_plan *plan);
int64_t chelis_matmul_matrix_count(const chelis_matmul_plan *plan, chelis_matmul_part part);
int64_t chelis_matmul_index(const chelis_matmul_plan *plan, chelis_matmul_part part, chelis_scalar batch, chelis_scalar element);
void chelis_matmul_check_target(const chelis_matmul_plan *plan, chelis_scalar rank, const chelis_scalar *shape);
void chelis_matmul_check_scratch(const chelis_matmul_plan *plan, chelis_matmul_part part, chelis_scalar exemplar);
void chelis_matmul_check_vendor(const chelis_matmul_plan *plan, chelis_scalar maximum);
void chelis_matmul_plan_release(chelis_matmul_plan *plan);
typedef struct chelis_sparse_plan chelis_sparse_plan;
chelis_sparse_plan *chelis_tensor_sparse_plan(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, chelis_scalar axis, chelis_sparse_op operation);
int64_t chelis_sparse_extent(const chelis_sparse_plan *plan, chelis_scalar axis);
int64_t chelis_sparse_count(const chelis_sparse_plan *plan);
int64_t chelis_sparse_index_slot(const chelis_sparse_plan *plan, chelis_scalar linear);
int64_t chelis_sparse_data_index(const chelis_sparse_plan *plan, chelis_scalar linear, chelis_scalar selected);
void chelis_sparse_check_target(const chelis_sparse_plan *plan, chelis_scalar rank, const chelis_scalar *shape);
void chelis_sparse_plan_release(chelis_sparse_plan *plan);
chelis_reduction_plan *chelis_tensor_reduction_plan(const chelis_tensor *tensor, chelis_scalar axis_count, const chelis_scalar *axes, chelis_scalar exemplar, chelis_reduction_op operation);
chelis_reduction_plan *chelis_shape_reduction_plan(chelis_scalar rank, const chelis_scalar *shape, chelis_scalar axis_count, const chelis_scalar *axes, chelis_scalar exemplar, chelis_reduction_op operation);
int64_t chelis_reduction_count(const chelis_reduction_plan *plan);
int64_t chelis_reduction_extent(const chelis_reduction_plan *plan, chelis_scalar axis);
int64_t chelis_reduction_index(const chelis_reduction_plan *plan, chelis_scalar outer, chelis_scalar leaf);
void chelis_reduction_check_target(const chelis_reduction_plan *plan, chelis_scalar rank, const chelis_scalar *shape);
void chelis_reduction_check_scratch(const chelis_reduction_plan *plan, chelis_scalar exemplar);
void chelis_reduction_plan_release(chelis_reduction_plan *plan);
void chelis_tensor_check_reshape(const chelis_tensor *tensor, chelis_scalar rank, const chelis_scalar *shape);
chelis_tensor *chelis_tensor_reshape(const chelis_tensor *tensor, const chelis_list *shape);

chelis_string chelis_string_from_cstr(const char *value);
chelis_string chelis_string_from_utf8(const uint8_t *value, int64_t len);
const char *chelis_string_data(chelis_string value);
int64_t chelis_char_code(chelis_string value);
chelis_string chelis_char_from_code(int64_t value);
void chelis_print_string(chelis_string value);
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
void chelis_option_retain(const chelis_option *option);
void chelis_option_release(const chelis_option *option);
void chelis_mapped_file_retain(const chelis_mapped_file *mapped);
void chelis_mapped_file_release(const chelis_mapped_file *mapped);
chelis_adt *chelis_adt_construct(chelis_string ctor, const chelis_value *fields, int64_t len);
chelis_string chelis_adt_get_tag(const chelis_adt *adt);
bool chelis_adt_tag_equals(const chelis_adt *adt, chelis_string ctor);
int64_t chelis_adt_field_count(const chelis_adt *adt);
chelis_value chelis_adt_get_field(const chelis_adt *adt, int64_t index);

chelis_value chelis_value_take_string(chelis_string value);
chelis_value chelis_value_take_tensor(chelis_tensor *tensor);
chelis_value chelis_value_take_list(chelis_list *list);
chelis_value chelis_value_take_tuple(chelis_tuple *tuple);
chelis_value chelis_value_take_dict(chelis_dict *dict);
chelis_value chelis_value_take_adt(chelis_adt *adt);
chelis_value chelis_value_take_option(chelis_option *option);
chelis_value chelis_value_take_mapped_file(chelis_mapped_file *mapped);
chelis_value chelis_value_clone(chelis_value value);
void chelis_value_release(chelis_value value);

chelis_string chelis_string_take_value(chelis_value value);
chelis_string chelis_string_borrow_value(chelis_value value);
chelis_tensor *chelis_tensor_take_value(chelis_value value);
const chelis_tensor *chelis_tensor_borrow_value(chelis_value value);
chelis_list *chelis_list_take_value(chelis_value value);
const chelis_list *chelis_list_borrow_value(chelis_value value);
chelis_tuple *chelis_tuple_take_value(chelis_value value);
const chelis_tuple *chelis_tuple_borrow_value(chelis_value value);
chelis_dict *chelis_dict_take_value(chelis_value value);
const chelis_dict *chelis_dict_borrow_value(chelis_value value);
chelis_adt *chelis_adt_take_value(chelis_value value);
const chelis_adt *chelis_adt_borrow_value(chelis_value value);
chelis_option *chelis_option_take_value(chelis_value value);
const chelis_option *chelis_option_borrow_value(chelis_value value);
chelis_mapped_file *chelis_mapped_file_take_value(chelis_value value);
const chelis_mapped_file *chelis_mapped_file_borrow_value(chelis_value value);

chelis_option *chelis_option_none(void);
chelis_option *chelis_option_some(chelis_value value);
bool chelis_option_is_some(const chelis_option *option);
chelis_value chelis_option_unwrap(const chelis_option *option);

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
chelis_option *chelis_dict_get(const chelis_dict *dict, chelis_value key);
chelis_dict *chelis_dict_remove(const chelis_dict *dict, chelis_value key);
chelis_dict *chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value);
chelis_dict *chelis_dict_merge(const chelis_dict *left, const chelis_dict *right);
chelis_list *chelis_dict_keys(const chelis_dict *dict);
chelis_list *chelis_dict_values(const chelis_dict *dict);
chelis_list *chelis_dict_entries(const chelis_dict *dict);
chelis_tensor *chelis_tensor_from_values(const chelis_list *list, chelis_dtype dtype);
chelis_list *chelis_tensor_elements(const chelis_tensor *tensor);
chelis_tensor *chelis_pad_sequences(const chelis_list *sequences, chelis_scalar pad_value);
chelis_tensor *chelis_pad_sequences_to(const chelis_list *sequences, int64_t width, chelis_scalar pad_value);
chelis_tensor *chelis_tensor_concat(const chelis_list *parts, int32_t axis);
chelis_list *chelis_tensor_split(const chelis_tensor *tensor, int32_t axis, const chelis_list *sizes);
chelis_tensor *chelis_tensor_gather(const chelis_tensor *tensor, const chelis_tensor *indices, int32_t axis);
chelis_tensor *chelis_tensor_cmplt(const chelis_tensor *left, const chelis_tensor *right);
chelis_tensor *chelis_tensor_scatter_replace(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis);
chelis_tensor *chelis_tensor_scatter_add(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis);
chelis_tensor *chelis_tensor_where(
    const chelis_tensor *condition,
    const chelis_tensor *then_tensor,
    const chelis_tensor *else_tensor
);
chelis_tensor *chelis_tensor_cumsum(const chelis_tensor *tensor, int32_t axis);
chelis_tuple *chelis_tensor_sort(const chelis_tensor *tensor, int32_t axis);
chelis_tensor *chelis_tensor_diagonal(const chelis_tensor *tensor, int32_t axis1, int32_t axis2);
chelis_tensor *chelis_tensor_trace(const chelis_tensor *tensor, int32_t axis1, int32_t axis2);
chelis_tensor *chelis_tensor_clamp(
    const chelis_tensor *tensor,
    const chelis_tensor *lower,
    const chelis_tensor *upper
);
chelis_tensor *chelis_tensor_einsum(
    chelis_string equation,
    const chelis_tensor *left,
    const chelis_tensor *right,
    chelis_dtype accumulator
);
void chelis_print_list(const chelis_list *list);
void chelis_print_tuple(const chelis_tuple *tuple);
void chelis_print_dict(const chelis_dict *dict);
void chelis_print_adt(const chelis_adt *adt);
void chelis_fail(chelis_string message);
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
        uint32_t lsb = (mant >> 13) & 1u;
        uint32_t rounded = mant + 0x00000FFFu + lsb;
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

/* chelis#1112: a flat offset and a per-axis index are both bounded by
 * extents, so they live in the extent domain and carry int64_t. `ndim`
 * remains the axis-domain loop bound. */
static inline void chelis_flat_to_indices(int64_t flat, const int64_t *shape, int ndim,
                                          int64_t *out) {
    for (int d = ndim - 1; d >= 0; d--) {
        out[d] = flat % shape[d];
        flat /= shape[d];
    }
}

static inline int64_t chelis_indices_to_flat(const int64_t *indices, const int64_t *strides,
                                             int ndim) {
    int64_t flat = 0;
    for (int d = 0; d < ndim; d++) {
        flat += indices[d] * strides[d];
    }
    return flat;
}

chelis_tensor *chelis_contiguous(const chelis_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
