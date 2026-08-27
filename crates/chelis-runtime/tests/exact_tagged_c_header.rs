//! C compile/run and layout oracle for [05-OP-31]'s exact declarations.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("chelis-exact-tagged-abi-{nonce}"));
        fs::create_dir_all(&path).expect("create probe directory");
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn include_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("include")
}

#[test]
fn header_has_only_the_exact_tagged_dynamic_rank_abi() {
    let mut header = fs::read_to_string(include_dir().join("chelis_runtime.h")).expect("header");
    header.push_str(
        &fs::read_to_string(include_dir().join("chelis_runtime_dtype.h"))
            .expect("generated dtype header"),
    );
    for required in [
        "typedef uint8_t chelis_dtype;",
        "typedef struct { chelis_dtype dtype; uint8_t reserved[7]; uint64_t bits; } chelis_scalar;",
        "typedef struct { uint8_t is_some; uint8_t reserved[7]; chelis_scalar value; } chelis_option_scalar;",
        "typedef union { chelis_scalar scalar; void *handle; } chelis_value_payload;",
        "typedef struct { chelis_value_tag tag; uint8_t reserved[7]; chelis_value_payload payload; } chelis_value;",
        "typedef struct { uint8_t is_some; uint8_t reserved[7]; chelis_value value; } chelis_option_value;",
        "typedef struct { void *data; const int64_t *shape; const int64_t *strides; int64_t size; int64_t byte_capacity; int32_t rank; chelis_dtype dtype; uint8_t owns_data; uint8_t reserved[2]; } chelis_tensor;",
        "chelis_tensor *chelis_alloc_view(int32_t rank, const int64_t *shape, chelis_dtype dtype, void *data, int64_t byte_capacity);",
    ] {
        assert!(header.contains(required), "missing exact declaration: {required}");
    }

    for forbidden in [
        "CHELIS_MAX_DIM",
        "CHELIS_F32",
        "CHELIS_VALUE_INT64",
        "chelis_option_i64",
        "chelis_option_f64",
        "chelis_fill_f32",
        "chelis_fill_f64",
        "chelis_fill_bool_bits",
        "chelis_value_from_int64",
        "chelis_value_from_f64",
        "chelis_value_from_bool",
        "chelis_value_as_int64",
        "chelis_value_as_f64",
        "chelis_value_as_bool",
        "chelis_scalar_tensor_from_",
        "chelis_tensor_to_f64",
        "chelis_format_shortest",
        "chelis_bf16_buffer_to_f32",
        "chelis_f16_buffer_to_f32",
    ] {
        assert!(
            !header.contains(forbidden),
            "legacy public ABI spelling remains: {forbidden}"
        );
    }
}

#[test]
fn c_layout_matches_the_normative_fixed_width_field_order() {
    let temp = TempDir::new();
    let source = temp.0.join("probe.c");
    let binary = temp.0.join("probe");
    fs::write(
        &source,
        r#"
#include <stddef.h>
#include <stdint.h>
#include "chelis_runtime.h"

_Static_assert(sizeof(chelis_dtype) == 1, "dtype width");
_Static_assert(sizeof(chelis_scalar) == 16, "scalar size");
_Static_assert(offsetof(chelis_scalar, dtype) == 0, "scalar dtype offset");
_Static_assert(offsetof(chelis_scalar, bits) == 8, "scalar bits offset");
_Static_assert(sizeof(chelis_option_scalar) == 24, "option scalar size");
_Static_assert(sizeof(chelis_value) == 24, "value size");
_Static_assert(offsetof(chelis_value, payload) == 8, "value payload offset");
_Static_assert(sizeof(chelis_option_value) == 32, "option value size");
_Static_assert(offsetof(chelis_tensor, data) == 0, "tensor data offset");
_Static_assert(offsetof(chelis_tensor, shape) == sizeof(void *), "tensor shape offset");
_Static_assert(offsetof(chelis_tensor, strides) == 2 * sizeof(void *), "tensor strides offset");
_Static_assert(offsetof(chelis_tensor, size) == 3 * sizeof(void *), "tensor size offset");
_Static_assert(offsetof(chelis_tensor, byte_capacity) == 3 * sizeof(void *) + 8, "tensor capacity offset");

int main(void) {
    return CHELIS_DTYPE_BOOL == 3 && CHELIS_VALUE_SCALAR == 1 ? 0 : 1;
}
"#,
    )
    .expect("write C layout probe");
    let compile = Command::new("cc")
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Werror")
        .arg("-I")
        .arg(include_dir())
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .expect("compile C layout probe");
    assert!(
        compile.status.success(),
        "layout probe failed:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    assert!(Command::new(binary)
        .status()
        .expect("run layout probe")
        .success());
}
