//! Bind every provider consumer to the package this compiler carries.

use std::fs;

fn main() {
    const MANIFEST: &str = "../../packages/chelis-clarabel/reef.toml";
    println!("cargo:rerun-if-changed={MANIFEST}");
    let manifest: toml::Value = fs::read_to_string(MANIFEST)
        .expect("read carried Clarabel package manifest")
        .parse()
        .expect("parse carried Clarabel package manifest");
    let name = manifest["package"]["name"]
        .as_str()
        .expect("Clarabel package name is a string");
    let version = manifest["package"]["version"]
        .as_str()
        .expect("Clarabel package version is a string");
    // The declaration's module and linked symbols belong to this package.
    assert_eq!(name, "chelis-clarabel");
    // Keep the Cargo directive one line; Reef validates the semantic version
    // when it admits the package into the dependency graph.
    assert!(!version.is_empty());
    assert!(
        version
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'+'))
    );
    println!("cargo:rustc-env=CHELIS_CLARABEL_PACKAGE_NAME={name}");
    println!("cargo:rustc-env=CHELIS_CLARABEL_PACKAGE_VERSION={version}");
}
