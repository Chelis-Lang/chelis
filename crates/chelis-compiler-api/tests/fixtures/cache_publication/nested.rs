fn exercise(path: &std::path::Path) {
    cache_envelope::save(path, [0; 32], &vec![9007199254740993_i64]).unwrap();
}
