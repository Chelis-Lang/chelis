//! Link the `chelis` executable with a GNU build id on Linux.
//!
//! `chelis-image-id` identifies the running compiler by the build id of its
//! image. Without a build id, each process computes SHA-256 over the whole
//! executable (chelis#3659). Many Linux toolchains add a build id by default,
//! but the Nix toolchain does not. So this script asks the linker for one.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--build-id");
    }
}
