//! Compiles the CORE-MATH amalgamation (spec/design/correctly_rounded_math.md section 4.1).
//!
//! The kernels are correctly rounded only under IEEE semantics, so the build owns its
//! compiler flags: it removes every ambient C-flag variable the `cc` crate would append
//! after its own flags (and clang's `CCC_OVERRIDE_OPTIONS`) before compiling, then
//! passes the strict profile explicitly. The amalgamation's `#error` guards fail the
//! build if fast math or excess precision still reaches the compiler.

const AMBIENT_FLAG_VARIABLES: &[&str] = &["CFLAGS", "TARGET_CFLAGS", "HOST_CFLAGS", "CCC_OVERRIDE_OPTIONS"];

fn is_ambient_flag_variable(key: &str) -> bool {
    AMBIENT_FLAG_VARIABLES.contains(&key) || key.starts_with("CFLAGS_")
}

fn main() {
    println!("cargo:rerun-if-changed=csrc/crmath_amalgamation.c");
    println!("cargo:rerun-if-changed=csrc/crmath_ffi.c");
    for key in AMBIENT_FLAG_VARIABLES {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let ambient: Vec<String> = std::env::vars_os()
        .filter_map(|(key, _)| key.into_string().ok())
        .filter(|key| is_ambient_flag_variable(key))
        .collect();
    for key in ambient {
        println!("cargo:rerun-if-env-changed={key}");
        // SAFETY: the build script is single-threaded at this point; no other thread
        // reads the environment while it is modified.
        unsafe { std::env::remove_var(&key) };
    }

    cc::Build::new()
        .file("csrc/crmath_ffi.c")
        .include("csrc")
        .flag("-std=c11")
        .flag("-O2")
        .flag("-ffp-contract=off")
        .flag("-fno-fast-math")
        .warnings(true)
        .extra_warnings(true)
        .compile("chelis_crmath");
}
