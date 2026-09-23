//! C compile/run and layout oracle for [05-OP-31]'s exact declarations.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_DIR_NONCE: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        loop {
            // Concurrent tests share a process, so PID plus wall-clock time is
            // not a unique identity. The monotonic nonce separates those
            // threads; create_dir also rejects residue from a reused PID.
            let nonce = TEMP_DIR_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "chelis-exact-tagged-abi-{}-{nonce}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create probe directory {}: {error}", path.display()),
            }
        }
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

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn published_header_sources() -> String {
    [
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_runtime_views.h",
    ]
    .into_iter()
    .map(|name| {
        fs::read_to_string(include_dir().join(name))
            .unwrap_or_else(|error| panic!("read published header {name}: {error}"))
    })
    .collect::<Vec<_>>()
    .join("\n")
}

#[test]
fn concurrent_probe_directories_have_distinct_live_paths() {
    let handles = (0..4)
        .map(|_| std::thread::spawn(TempDir::new))
        .collect::<Vec<_>>();
    let directories = handles
        .into_iter()
        .map(|handle| handle.join().expect("create probe directory thread"))
        .collect::<Vec<_>>();
    let mut paths = directories
        .iter()
        .map(|directory| directory.0.clone())
        .collect::<Vec<_>>();
    paths.sort();
    paths.dedup();
    assert_eq!(paths.len(), directories.len(), "probe paths collided");
    assert!(
        directories.iter().all(|directory| directory.0.is_dir()),
        "a live probe directory was removed by another TempDir"
    );
}

#[test]
fn header_has_only_the_exact_tagged_dynamic_rank_abi() {
    let header = published_header_sources()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    for required in [
        "typedef uint8_t chelis_dtype;",
        "typedef struct { chelis_dtype dtype; uint8_t reserved[7]; uint64_t bits; } chelis_scalar;",
        "typedef struct chelis_tensor chelis_tensor;",
        "typedef struct chelis_tensor_write chelis_tensor_write;",
        "typedef struct chelis_option chelis_option;",
        "typedef union { chelis_scalar scalar; void *handle; } chelis_value_payload;",
        "typedef struct { chelis_value_tag tag; uint8_t reserved[7]; chelis_value_payload payload; } chelis_value;",
        "typedef struct { const void *data; int64_t count; chelis_dtype dtype; uint8_t reserved[7]; } chelis_read_view;",
        "typedef struct { void *data; int64_t count; chelis_dtype dtype; uint8_t reserved[7]; } chelis_write_view;",
        "chelis_tensor *chelis_tensor_entry_borrow(int32_t rank, const int64_t *shape, chelis_dtype dtype, const void *data, int64_t byte_capacity);",
        "A successful chelis_tensor_begin_write invalidates every prior read view; dereferencing a stale view violates the caller precondition.",
        "chelis_read_view chelis_tensor_read_view(const chelis_tensor *tensor);",
        "chelis_tensor_write *chelis_tensor_begin_write(chelis_tensor *tensor);",
        "chelis_write_view chelis_tensor_write_view(const chelis_tensor_write *guard);",
        "void chelis_tensor_end_write(chelis_tensor_write *guard);",
        "void chelis_fill_scalar(chelis_tensor_write *guard, chelis_scalar value);",
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
        "chelis_option_scalar",
        "chelis_option_value",
        "chelis_alloc_view",
        "chelis_free",
        "chelis_value_retain",
        "owns_data",
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
_Static_assert(sizeof(chelis_value) == 24, "value size");
_Static_assert(offsetof(chelis_value, payload) == 8, "value payload offset");
_Static_assert(sizeof(chelis_read_view) == 24, "read view size");
_Static_assert(offsetof(chelis_read_view, data) == 0, "read view data offset");
_Static_assert(offsetof(chelis_read_view, count) == 8, "read view count offset");
_Static_assert(offsetof(chelis_read_view, dtype) == 16, "read view dtype offset");
_Static_assert(sizeof(chelis_write_view) == 24, "write view size");
_Static_assert(offsetof(chelis_write_view, data) == 0, "write view data offset");
_Static_assert(offsetof(chelis_write_view, count) == 8, "write view count offset");
_Static_assert(offsetof(chelis_write_view, dtype) == 16, "write view dtype offset");

int main(void) {
    chelis_tensor *tensor = NULL;
    return CHELIS_DTYPE_BOOL == 3 && CHELIS_VALUE_OPTION == 8 &&
                   CHELIS_VALUE_MAPPED_FILE == 9 && tensor == NULL
               ? 0
               : 1;
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

#[test]
fn repository_runtime_consumer_compiles_against_the_published_header() {
    let source = repository_root().join("nix/tests/runtime-consumer.c");
    let compile = Command::new("cc")
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-I")
        .arg(include_dir())
        .arg("-fsyntax-only")
        .arg(&source)
        .output()
        .expect("compile repository runtime consumer");
    assert!(
        compile.status.success(),
        "{} did not compile against the published runtime headers:\n{}",
        source.display(),
        String::from_utf8_lossy(&compile.stderr)
    );
}

#[test]
fn published_runtime_header_compiles_as_c11_and_cxx17() {
    let temp = TempDir::new();
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    for (extension, compiler, standard) in
        [("c", "cc", "-std=c11"), ("cpp", cxx.as_str(), "-std=c++17")]
    {
        let source = temp.0.join(format!("runtime-header.{extension}"));
        fs::write(
            &source,
            "#include \"chelis_runtime.h\"\nvoid probe(void) {}\n",
        )
        .expect("write dual-language runtime-header probe");
        let compile = Command::new(compiler)
            .arg(standard)
            .arg("-Wall")
            .arg("-Wextra")
            .arg("-Werror")
            .arg("-pedantic-errors")
            // The Nix compiler wrapper injects linker flags that
            // `-fsyntax-only` never consumes; clang reports them as unused.
            .arg("-Wno-unused-command-line-argument")
            .arg("-I")
            .arg(include_dir())
            .arg("-fsyntax-only")
            .arg(&source)
            .output()
            .unwrap_or_else(|error| panic!("start {compiler} for {standard}: {error}"));
        assert!(
            compile.status.success(),
            "published runtime header failed under {standard}:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
    }
}

#[test]
fn cxx_header_probe_rejects_the_c_only_noreturn_spelling() {
    let temp = TempDir::new();
    for name in [
        "chelis_runtime.h",
        "chelis_runtime_dtype.h",
        "chelis_runtime_views.h",
        "chelis_simd.h",
    ] {
        let source = include_dir().join(name);
        let contents = fs::read_to_string(&source)
            .unwrap_or_else(|error| panic!("read published header {}: {error}", source.display()));
        let mutated = if name == "chelis_runtime.h" {
            let inline = contents.replace(
                "static inline void chelis_flush_and_abort(void)",
                "static inline _Noreturn void chelis_flush_and_abort(void)",
            );
            let mutated = inline.replace(
                "void chelis_fail(chelis_string message);",
                "_Noreturn void chelis_fail(chelis_string message);",
            );
            assert_ne!(mutated, contents, "runtime header mutation did not fire");
            mutated
        } else {
            contents
        };
        fs::write(temp.0.join(name), mutated)
            .unwrap_or_else(|error| panic!("write mutated published header {name}: {error}"));
    }
    let source = temp.0.join("c-only-noreturn.cpp");
    fs::write(&source, "#include \"chelis_runtime.h\"\n")
        .expect("write C++ negative-control probe");
    let compiler = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let compile = Command::new(&compiler)
        .args(["-std=c++17", "-pedantic-errors", "-fsyntax-only", "-I"])
        .arg(&temp.0)
        .arg(&source)
        .output()
        .unwrap_or_else(|error| panic!("start {compiler} for negative control: {error}"));
    assert!(
        !compile.status.success() && String::from_utf8_lossy(&compile.stderr).contains("_Noreturn"),
        "the C-only spelling must fail for its own reason under C++17:\n{}",
        String::from_utf8_lossy(&compile.stderr)
    );
}

#[test]
fn repository_runtime_consumer_cannot_regress_to_the_removed_free_api() {
    let temp = TempDir::new();
    let source = repository_root().join("nix/tests/runtime-consumer.c");
    let current = fs::read_to_string(&source).expect("repository runtime consumer");
    let obsolete = current.replace("chelis_tensor_release(tensor);", "chelis_free(tensor);");
    assert_ne!(
        obsolete, current,
        "runtime consumer does not exercise tensor release"
    );
    let probe = temp.0.join("obsolete-runtime-consumer.c");
    fs::write(&probe, obsolete).expect("write obsolete runtime consumer probe");
    let compile = Command::new("cc")
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-I")
        .arg(include_dir())
        .arg("-fsyntax-only")
        .arg(&probe)
        .output()
        .expect("compile obsolete runtime consumer probe");
    assert!(
        !compile.status.success(),
        "repository runtime consumer compiled after restoring removed chelis_free"
    );
}

#[test]
fn c_tensor_descriptor_rejects_public_field_access() {
    let temp = TempDir::new();
    let source = temp.0.join("opaque.c");
    fs::write(
        &source,
        r#"#include "chelis_runtime.h"
int main(void) {
    chelis_tensor *tensor = 0;
    return (int)tensor->rank;
}
"#,
    )
    .expect("write opaque-tensor probe");
    let compile = Command::new("cc")
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Werror")
        .arg("-I")
        .arg(include_dir())
        .arg(&source)
        .arg("-c")
        .arg("-o")
        .arg(temp.0.join("opaque.o"))
        .output()
        .expect("compile opaque-tensor probe");
    assert!(
        !compile.status.success(),
        "public tensor field access compiled despite the incomplete type"
    );
}
