fn main() {
    println!("cargo::rustc-check-cfg=cfg(verus_only)");
    println!("cargo::rustc-check-cfg=cfg(verus_keep_ghost)");
}
