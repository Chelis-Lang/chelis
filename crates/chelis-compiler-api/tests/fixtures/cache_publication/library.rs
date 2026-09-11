// spec/10 §3: the library cache retains its actual admitted payload owner.
fn exercise(path: &std::path::Path, value: &LibraryContext) {
    cache_envelope::save(path, [0; 32], value).unwrap();
    let _: Option<LibraryContext> = cache_envelope::load(path, [0; 32]).unwrap();
}

const _: () = assert!(<LibraryContext as cache_envelope::CachePayload>::FORMAT_VERSION == 14);
const _: () = assert!(same(
    <LibraryContext as cache_envelope::CachePayload>::KEY_DOMAIN,
    b"chelis_library_typecheck_v"
));
