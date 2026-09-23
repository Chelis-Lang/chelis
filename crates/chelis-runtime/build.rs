fn main() {
    println!("cargo:rerun-if-changed=include/chelis_runtime.h");
    chelis_runtime_identity_build::declare_producer(chelis_runtime_identity::RecordKind::Runtime)
        .unwrap_or_else(|error| panic!("runtime identity production failed: {error}"));
}
