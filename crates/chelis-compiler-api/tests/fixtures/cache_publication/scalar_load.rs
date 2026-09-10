fn exercise(path: &std::path::Path) {
    let _: Option<f64> = cache_envelope::load(path, [0; 32]).unwrap();
}
