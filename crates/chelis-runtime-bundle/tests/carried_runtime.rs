//! A linked test build carries the runtime static archive of its own build and
//! stages exactly those bytes (`spec/08-backends.md` §2.1).

use sha2::{Digest, Sha256};
use std::fs;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn staging_publishes_exactly_the_carried_archive() {
    let dir = tempfile::tempdir().expect("tempdir");
    let staged = chelis_runtime_bundle::stage(dir.path())
        .expect("a linked build carries a runtime and stages it when CHELIS_RUNTIME_DIR is unset");
    let carried = chelis_runtime_bundle::carried_sha256().expect("carried digest");

    let bytes = fs::read(&staged.archive).expect("staged archive");
    assert!(
        bytes.starts_with(b"!<arch>\n"),
        "the carried runtime is a static archive"
    );
    assert_eq!(hex(&Sha256::digest(&bytes)), carried);
    assert_eq!(staged.archive_sha256, carried);

    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(&staged.receipt).expect("receipt")).expect("JSON");
    assert_eq!(receipt["archive_sha256"], carried.as_str());
    assert_eq!(receipt["mode"], chelis_runtime_bundle::MODE);
}
