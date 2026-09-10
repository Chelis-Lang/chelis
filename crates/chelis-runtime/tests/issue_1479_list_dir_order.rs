//! chelis#1479 row 1: `chelis_list_dir` returns entries in `[05-HOST-4]` order.
//!
//! `spec/05-risc-primitives.md` [05-HOST-4] orders the result by the byte
//! sequence of the entry name the host reports, fixed before any conversion to
//! `string`. This is the compiled lane's half of that contract; the evaluator's
//! half is `crates/chelis-compiler-api/tests/issue_1479_list_dir_order.rs`, and
//! [05-HOST-1] requires the two to agree.
//!
//! Every fixture is created in an order that is not the required one, so a
//! build that forwards the host's enumeration order cannot pass by coincidence.
//! Proven to fire: deleting the `names.sort_by(...)` line from `chelis_list_dir`
//! fails the multi-entry cases here while leaving the empty-directory case
//! green.

use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use chelis_runtime::{
    chelis_list_dir, chelis_list_index, chelis_list_len, chelis_list_release, chelis_string,
    chelis_string_borrow_value, chelis_string_data, chelis_string_from_cstr, chelis_string_release,
};

/// A self-cleaning probe directory. `chelis-runtime` deliberately carries
/// almost no dev-dependencies (its one edge is documented in `Cargo.toml` as
/// test-only), so this uses the local idiom rather than adding `tempfile`
/// here.
///
/// The path needs entropy from all three of the process, the clock, and a
/// counter. `cargo nextest` runs every test in its OWN process, so
/// `PROBE_NONCE` reinitialises to `0` in each one and disambiguates nothing
/// across tests; the clock alone is not enough either, because two processes
/// spawned back to back can read the same coarse `as_nanos` value on a
/// virtualised macOS runner. Both tests then build the same directory and
/// each sees the other's fixture entries. The process id is what separates
/// them, which is why `runtime_dtype_c_probe.rs`, `ownership_ledger.rs`, and
/// `dim_carrier_int64.rs` all include it.
struct TempDir(PathBuf);

static PROBE_NONCE: AtomicU64 = AtomicU64::new(0);

impl TempDir {
    fn new() -> Self {
        let clock = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let seq = PROBE_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "chelis-1479-list-dir-{}-{clock}-{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create probe directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Build a runtime string at the real FFI boundary.
fn make(text: &str) -> chelis_string {
    let owned = CString::new(text).expect("test paths contain no interior NUL");
    unsafe { chelis_string_from_cstr(owned.as_ptr() as *const c_char) }
}

/// Call `chelis_list_dir` over FFI and read every element back as Rust text.
fn list_dir_names(dir: &Path) -> Vec<String> {
    let path = make(dir.to_str().expect("UTF-8 fixture path"));
    unsafe {
        let list = chelis_list_dir(path);
        assert!(!list.is_null(), "chelis_list_dir returned null");
        let len = chelis_list_len(list);
        let mut names = Vec::with_capacity(len as usize);
        for index in 0..len {
            let element = chelis_list_index(list, index);
            let text = chelis_string_borrow_value(element);
            names.push(
                CStr::from_ptr(chelis_string_data(text))
                    .to_str()
                    .expect("entry name round-trips as UTF-8")
                    .to_string(),
            );
        }
        chelis_list_release(list);
        chelis_string_release(path);
        names
    }
}

#[test]
fn chelis_list_dir_orders_entries_by_name_not_creation_order() {
    let dir = TempDir::new();
    // Written in reverse of the required order.
    for name in ["zulu.txt", "mike.txt", "delta.txt", "alpha.txt"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture entry");
    }

    assert_eq!(
        list_dir_names(dir.path()),
        vec!["alpha.txt", "delta.txt", "mike.txt", "zulu.txt"],
    );
}

#[test]
fn chelis_list_dir_orders_by_byte_sequence_not_case_insensitively() {
    let dir = TempDir::new();
    // 'M' is 0x4D and 'Z' is 0x5A, both below 'a' at 0x61. A case-folding
    // comparison would give apple, Mango, Zebra instead.
    //
    // These are deliberately not case variants of one another: the default
    // macOS filesystem is case-insensitive, so "Banana" and "banana" would be
    // one file rather than two.
    for name in ["apple", "Mango", "Zebra"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture entry");
    }

    assert_eq!(list_dir_names(dir.path()), vec!["Mango", "Zebra", "apple"]);
}

#[test]
fn chelis_list_dir_orders_dotfiles_and_digits_by_byte_sequence() {
    let dir = TempDir::new();
    // '.' is 0x2E, digits 0x30..0x39, uppercase 0x41.., lowercase 0x61...
    for name in ["b", "A", "10", "2", ".hidden"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture entry");
    }

    // "10" precedes "2" because this is byte order, not numeric order.
    assert_eq!(
        list_dir_names(dir.path()),
        vec![".hidden", "10", "2", "A", "b"],
    );
}

/// Actual invalid-name filesystem coverage runs on Linux; the pure conversion
/// controls also exercise these cases on macOS without creating host files.
#[cfg(target_os = "linux")]
#[test]
fn chelis_list_dir_rejects_non_utf8_names_in_host_order() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::process::Command;

    const CHILD_PATH: &str = "CHELIS_1479_INVALID_DIRECTORY";
    if let Some(path) = std::env::var_os(CHILD_PATH) {
        let _ = list_dir_names(Path::new(&path));
        panic!("invalid names returned a successful list");
    }
    let dir = TempDir::new();
    for raw in [
        b"a\xff".as_slice(),
        b"a\xfe".as_slice(),
        b"0-valid".as_slice(),
    ] {
        fs::write(dir.path().join(OsStr::from_bytes(raw)), b"x").expect("write fixture entry");
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "chelis_list_dir_rejects_non_utf8_names_in_host_order",
            "--nocapture",
        ])
        .env(CHILD_PATH, dir.path())
        .output()
        .expect("run FFI failure child");
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        format!(
            "IO trap in list_dir: directory b\"{}\", entry b\"a\\xfe\": name is not valid UTF-8\n",
            dir.path().as_os_str().as_encoded_bytes().escape_ascii()
        )
    );
}

#[test]
fn chelis_list_dir_on_an_empty_directory_returns_the_empty_list() {
    let dir = TempDir::new();
    let empty = dir.path().join("empty");
    fs::create_dir(&empty).expect("create empty fixture directory");

    assert!(list_dir_names(&empty).is_empty());
}

#[test]
fn chelis_list_dir_excludes_the_current_and_parent_directory_links() {
    let dir = TempDir::new();
    fs::create_dir(dir.path().join("child")).expect("create child directory");
    fs::write(dir.path().join("file"), b"x").expect("write fixture entry");

    let names = list_dir_names(dir.path());
    assert_eq!(names, vec!["child", "file"]);
    assert!(!names.iter().any(|name| name == "." || name == ".."));
}

/// [05-HOST-1] requires the same value result in every language execution mode.
/// This pins the compiled lane against the exact list the evaluator test asserts
/// for the identical fixture, so the two cannot drift apart silently without
/// one of the two suites turning red.
#[test]
fn chelis_list_dir_agrees_with_the_evaluator_lanes_expected_order() {
    let dir = TempDir::new();
    for name in ["zulu.txt", "mike.txt", "delta.txt", "alpha.txt"] {
        fs::write(dir.path().join(name), b"x").expect("write fixture entry");
    }

    // Identical fixture and identical expectation as
    // `list_dir_orders_entries_by_name_not_creation_order` in
    // crates/chelis-compiler-api/tests/issue_1479_list_dir_order.rs.
    assert_eq!(
        list_dir_names(dir.path()),
        vec!["alpha.txt", "delta.txt", "mike.txt", "zulu.txt"],
    );
}
