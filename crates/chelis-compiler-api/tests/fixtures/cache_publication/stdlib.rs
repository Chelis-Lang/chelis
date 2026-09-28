// spec/10 §3: the stdlib cache retains its distinct admitted payload owner.
fn exercise(path: &std::path::Path, value: &StdLibContext) {
    cache_envelope::save(path, [0; 32], value).unwrap();
    let _: Option<StdLibContext> = cache_envelope::load(path, [0; 32]).unwrap();
}

const _: () = assert!(<StdLibContext as cache_envelope::CachePayload>::FORMAT_VERSION == 40);
const _: () = assert!(same(
    <StdLibContext as cache_envelope::CachePayload>::KEY_DOMAIN,
    b"chelis_std_typecheck_v"
));
