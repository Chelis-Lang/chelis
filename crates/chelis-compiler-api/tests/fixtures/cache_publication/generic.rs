fn exercise<T: serde::Serialize + serde::de::DeserializeOwned>(path: &std::path::Path, value: &T) {
    cache_envelope::save(path, [0; 32], value).unwrap();
    let _: Option<T> = cache_envelope::load(path, [0; 32]).unwrap();
}
