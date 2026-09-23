fn main() {
    println!("cargo:rerun-if-changed=inputs/value.txt");
    println!("cargo:rerun-if-env-changed=OPT_LEVEL");
    let value: u32 = std::fs::read_to_string("inputs/value.txt").expect("read required inputs/value.txt").trim().parse().unwrap();
    std::fs::write(std::path::Path::new(&std::env::var_os("OUT_DIR").unwrap()).join("value.rs"),
        format!("pub const GENERATED: u32 = {value};\n")).unwrap();
}
