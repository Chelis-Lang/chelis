fn main() {
    // Detect Sleef via pkg-config
    if pkg_config::probe_library("sleef").is_ok() {
        println!("cargo:rustc-cfg=feature=\"sleef\"");
        println!("cargo:rustc-link-lib=sleef");
    }
}
