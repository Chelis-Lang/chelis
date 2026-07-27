use std::fs;
use std::path::PathBuf;

use chelis_runtime::dtype_header::render_runtime_dtype_c_header;

#[test]
fn checked_in_c_dtype_header_is_generated_from_the_rust_vocabulary() {
    let header = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("include")
        .join("chelis_runtime_dtype.h");
    let checked_in = fs::read_to_string(&header)
        .unwrap_or_else(|err| panic!("read generated dtype header {}: {err}", header.display()));
    assert_eq!(checked_in, render_runtime_dtype_c_header());
}

#[test]
fn generated_c_dtype_decoder_has_a_loud_default() {
    let generated = render_runtime_dtype_c_header();
    assert!(generated.contains("default:"));
    assert!(generated.contains("invalid Chelis runtime dtype id"));
    assert!(!generated.contains("default: return CHELIS_F32"));
    assert!(!generated.contains("default: return sizeof(float)"));
}

#[test]
fn public_runtime_header_includes_the_generated_dtype_contract() {
    let runtime_header = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("include")
        .join("chelis_runtime.h");
    let checked_in = fs::read_to_string(&runtime_header)
        .unwrap_or_else(|err| panic!("read runtime header {}: {err}", runtime_header.display()));
    assert!(checked_in.contains("#include \"chelis_runtime_dtype.h\""));
}
