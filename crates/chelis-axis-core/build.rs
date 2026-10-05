fn main() {
    println!("cargo::rustc-check-cfg=cfg(kani)");
    println!("cargo::rustc-check-cfg=cfg(axis_false_postcondition)");
    println!("cargo::rustc-check-cfg=cfg(verus_only)");
    println!("cargo::rustc-check-cfg=cfg(verus_keep_ghost)");
}
