fn main() {
    chelis_runtime_identity_build::declare_producer(chelis_runtime_identity::RecordKind::Python)
        .unwrap_or_else(|error| panic!("runtime identity production failed: {error}"));
}
