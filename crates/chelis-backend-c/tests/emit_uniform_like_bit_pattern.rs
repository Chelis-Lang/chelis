//! Issue #248 (#189 follow-up): `emit_uniform_like` was the third
//! lossy `%.8`-format-string site in the C backend. The F32/F64 Const
//! and Pad sites landed in PR #243 (#189); this sibling on the
//! `chelis_uniform_sample_f32` call's `low` / `high` arguments stayed
//! unfixed and is closed here.
//!
//! Pre-fix the emitter wrote:
//!
//! ```text
//! t1->data[i] = chelis_uniform_sample_f32(t1_seed, (uint64_t)i,
//!                                         0.00000000f, 0.50000000f);
//! ```
//!
//! ...for `low = 1e-40`, narrowing the sub-normal-range value to f32
//! and then losing it entirely to the `%.8` format string (gcc parses
//! "0.00000000f" as 0.0f).
//!
//! Post-fix the emitter reconstructs each f32 argument from its exact
//! bit pattern via the `chelis_f32_from_bits` static-inline helper
//! declared in `chelis_runtime.h`, symmetric with PR #243's
//! `chelis_fill_f32_bits` mechanism.

use chelis_backend_c::emit::CEmitter;
use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;

fn f32_tensor(size: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(size)],
        precision: Prim::F32,
    }
}

fn build_uniform_like_dag(low: f64, high: f64, seed: u64) -> Dag {
    let mut dag = Dag::new();
    let template = dag.add_node(RiscOp::Const { value: 0.0 }, vec![], f32_tensor(4), None);
    dag.add_node(
        RiscOp::UniformLike { low, high, seed },
        vec![template],
        f32_tensor(4),
        None,
    );
    dag
}

/// Bit-pattern reproducer: the sub-normal-range `1e-40` reproducer
/// from the issue. The closest f32 to `1.0e-40_f64` is a denormal
/// (`f32::from_bits(0x000116c2)`), and the emitted C must reconstruct
/// that bit pattern verbatim via `chelis_f32_from_bits`.
#[test]
fn issue_248_uniform_like_low_arg_emits_exact_bit_pattern() {
    let low: f64 = 1.0e-40;
    let high: f64 = 0.5;
    let dag = build_uniform_like_dag(low, high, 42);
    let src = CEmitter::emit_dag(&dag, "test_fn").unwrap();

    let low_bits = (low as f32).to_bits();
    let high_bits = (high as f32).to_bits();
    let low_needle = format!("chelis_f32_from_bits(0x{low_bits:08x}u)");
    let high_needle = format!("chelis_f32_from_bits(0x{high_bits:08x}u)");

    assert!(
        src.contains(&low_needle),
        "uniform_like low arg must emit `{low_needle}`; emitted source:\n{src}"
    );
    assert!(
        src.contains(&high_needle),
        "uniform_like high arg must emit `{high_needle}`; emitted source:\n{src}"
    );
}

/// Negative-parity guard: the pre-fix `%.8f`-style decimal literal
/// must never re-emerge for ordinary or subnormal-range values. Stamps
/// the issue's reproducer plus a one-ULP-off ordinary value (`0.1`)
/// where `%.8` round-trips one ULP short of the source-narrowing.
#[test]
fn issue_248_uniform_like_does_not_use_lossy_format() {
    let cases: &[(f64, f64)] = &[
        (1.0e-40, 0.5),
        (0.1, 0.9),
        (f32::MIN_POSITIVE as f64, 1.0),
        // smallest positive f32 denormal
        (f32::from_bits(0x0000_0001) as f64, 1.0),
    ];
    for &(low, high) in cases {
        let dag = build_uniform_like_dag(low, high, 7);
        let src = CEmitter::emit_dag(&dag, "test_fn").unwrap();
        let low_bits = (low as f32).to_bits();
        let high_bits = (high as f32).to_bits();
        assert!(
            src.contains(&format!("0x{low_bits:08x}")),
            "uniform_like for low={low}: expected bit pattern `0x{low_bits:08x}`; emitted:\n{src}"
        );
        assert!(
            src.contains(&format!("0x{high_bits:08x}")),
            "uniform_like for high={high}: expected bit pattern `0x{high_bits:08x}`; emitted:\n{src}"
        );
        // Belt-and-braces: assert every call site of
        // `chelis_uniform_sample_f32` routes both scalar args through
        // `chelis_f32_from_bits`. Filter out the static-inline
        // definition line (the only non-call site that mentions the
        // symbol) by requiring the line to also reference a tensor
        // slot the call writes into. A future refactor that
        // hand-edits the `{:.8}f` literal back into the emit would
        // trip this on top of the bit-pattern assertions.
        for line in src.lines() {
            if line.contains("chelis_uniform_sample_f32(") && line.contains("->data[i]") {
                assert!(
                    line.contains("chelis_f32_from_bits"),
                    "uniform_like call line must route through `chelis_f32_from_bits`; got:\n  {line}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------
// End-to-end gcc-compile-and-run byte-identity test. Mirrors the
// issue_189_*_byte_identical_to_eval_under_gcc oracles: emit C,
// gcc-compile, run, and verify the sampler saw the byte-identical
// f32 narrowing of the source `low` / `high`. Linux-only because the
// emitted uniform_like kernel uses `-fopenmp`; macOS clang
// (masquerading as gcc) rejects that flag.
// ---------------------------------------------------------------

#[cfg(target_os = "linux")]
use chelis_backend_c::codegen;
#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::process::Command;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

#[cfg(target_os = "linux")]
fn runtime_include_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../chelis-runtime/include")
}

#[cfg(target_os = "linux")]
fn target_debug_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe failed");
    exe.parent()
        .and_then(Path::parent)
        .map(PathBuf::from)
        .expect("could not resolve target/debug dir from current_exe")
}

#[cfg(target_os = "linux")]
fn ensure_runtime_static_lib(canonical: &Path) -> std::io::Result<()> {
    if canonical.exists() {
        return Ok(());
    }
    let deps_dir = canonical
        .parent()
        .expect("canonical lib path has no parent")
        .join("deps");
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    if let Ok(entries) = fs::read_dir(&deps_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy().to_string();
            if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
                let meta = entry.metadata()?;
                let mtime = meta.modified()?;
                if newest.as_ref().is_none_or(|(cur, _)| mtime > *cur) {
                    newest = Some((mtime, entry.path()));
                }
            }
        }
    }
    let hashed = match newest {
        Some((_, p)) => p,
        None => {
            Command::new(env!("CARGO"))
                .args(["build", "-p", "chelis-runtime", "--lib"])
                .status()
                .map_err(|e| std::io::Error::other(format!("cargo build chelis-runtime: {e}")))?;
            let entries = fs::read_dir(&deps_dir)?;
            let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy().to_string();
                if name.starts_with("libchelis_runtime-") && name.ends_with(".a") {
                    let meta = entry.metadata()?;
                    let mtime = meta.modified()?;
                    if newest.as_ref().is_none_or(|(cur, _)| mtime > *cur) {
                        newest = Some((mtime, entry.path()));
                    }
                }
            }
            newest
                .map(|(_, p)| p)
                .ok_or_else(|| std::io::Error::other("no libchelis_runtime-*.a after rebuild"))?
        }
    };
    let tmp = canonical.with_extension(format!("a.tmp.{}", std::process::id()));
    fs::copy(&hashed, &tmp)?;
    match fs::rename(&tmp, canonical) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && canonical.exists() => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

#[cfg(target_os = "linux")]
fn runtime_lib_path() -> PathBuf {
    static PATH: OnceLock<PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let canonical = target_debug_dir().join("libchelis_runtime.a");
        ensure_runtime_static_lib(&canonical).unwrap_or_else(|e| {
            panic!(
                "failed to materialize libchelis_runtime.a at {}: {e}",
                canonical.display()
            )
        });
        canonical
    })
    .clone()
}

#[cfg(target_os = "linux")]
fn compile_and_run(test_name: &str, c_source: &str, harness: &str) -> Option<String> {
    let dir = std::env::temp_dir().join(format!("chelis_issue248_{test_name}"));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("kernel.c"), c_source).unwrap();
    fs::write(dir.join("main.c"), harness).unwrap();
    let include_dir = runtime_include_dir();
    for hdr in &[
        "chelis_runtime.h",
        "chelis_blas.h",
        "chelis_simd.h",
        "chelis_math.h",
    ] {
        let src = fs::read_to_string(include_dir.join(hdr)).unwrap();
        fs::write(dir.join(hdr), src).unwrap();
    }
    let bin = dir.join("test_bin");
    let runtime_lib = runtime_lib_path();
    let compile = Command::new("gcc")
        .args([
            "-O2",
            "-std=c11",
            "-fopenmp",
            "-I",
            dir.to_str().unwrap(),
            dir.join("kernel.c").to_str().unwrap(),
            dir.join("main.c").to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime_lib.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("failed to invoke gcc");
    if !compile.status.success() {
        eprintln!(
            "COMPILE FAILED [{test_name}]:\n{}",
            String::from_utf8_lossy(&compile.stderr)
        );
        eprintln!("Kernel C:\n{c_source}");
        return None;
    }
    let run = Command::new(&bin).output().expect("failed to run binary");
    if !run.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&run.stdout).into_owned())
}

/// Byte-identity acceptance oracle. Sets `low = high = 1e-40` so the
/// sampler returns exactly `low` for every element (with `high - low ==
/// 0` the lerp degenerates). Pre-fix that would print
/// `0x00000000` (the `%.8` literal collapsed to zero). Post-fix it
/// must print the f32 bit pattern of `(1.0e-40_f64 as f32)`
/// (`0x000116c2`, a denormal).
#[cfg(target_os = "linux")]
#[test]
fn issue_248_uniform_like_byte_identical_low_under_gcc() {
    let low: f64 = 1.0e-40;
    let high: f64 = 1.0e-40;
    let dag = build_uniform_like_dag(low, high, 0);
    let result = codegen(&dag, "test_uniform_like").unwrap();
    let src = &result.c_source;
    let want_bits = (low as f32).to_bits();
    let harness = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "chelis_runtime.h"

extern void test_uniform_like(chelis_tensor** inputs, int n_in, chelis_tensor** outputs, int n_out);

int main(void) {
    chelis_tensor* outputs[1] = { NULL };
    test_uniform_like(NULL, 0, outputs, 1);
    float v;
    memcpy(&v, outputs[0]->data, sizeof(float));
    uint32_t bits;
    memcpy(&bits, &v, sizeof(uint32_t));
    printf("0x%08x\n", bits);
    return 0;
}
"#;
    let Some(output) = compile_and_run("uniform_like_low_byte_id", src, harness) else {
        panic!("emitted C did not compile/run");
    };
    let expected = format!("0x{want_bits:08x}");
    assert!(
        output.trim() == expected,
        "uniform_like low byte-identical mismatch: expected `{expected}`, got `{}`. Source:\n{src}",
        output.trim()
    );
}
