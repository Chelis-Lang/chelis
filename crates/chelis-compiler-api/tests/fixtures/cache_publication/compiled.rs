// CompiledContext must use its separate versioned context envelope.
fn exercise(path: &std::path::Path, value: &chelis_compiler_api::CompiledContext) {
    cache_envelope::save(path, [0; 32], value).unwrap();
}
