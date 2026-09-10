// Serialization alone cannot authorize a new durable numeric payload root.
fn exercise(path: &std::path::Path) {
    cache_envelope::save(path, [0; 32], &1.5_f64).unwrap();
}
