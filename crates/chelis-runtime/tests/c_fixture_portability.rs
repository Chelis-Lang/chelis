//! chelis#2496: test-authored C fixtures meet GCC only on the Linux lanes.
//!
//! Apple clang accepts both defects that chelis#1864 merged: a format macro
//! whose header arrives only through a macOS framework, and an unbraced `if`
//! followed by a second statement on its line. A Mac run of a fixture is
//! therefore no portability evidence. These controls run on every pull request
//! and prove that the lane compiles with GCC and that the strict fixture flags
//! still reject both shapes, so that evidence cannot quietly become clang's.
#![cfg(target_os = "linux")]

use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
};

/// The flags a fixture opts into when a warning must fail its build.
const STRICT_FIXTURE_FLAGS: [&str; 4] = ["-std=c11", "-Wall", "-Wextra", "-Werror"];

struct Scratch(PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "chelis-c-fixture-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn compile(&self, source: &str) -> Output {
        let file = self.0.join("fixture.c");
        fs::write(&file, source).unwrap();
        Command::new("cc")
            .args(STRICT_FIXTURE_FLAGS)
            .arg("-I")
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include"))
            .arg("-c")
            .arg(&file)
            .arg("-o")
            .arg(self.0.join("fixture.o"))
            .output()
            .expect("the fixture C compiler `cc` must run")
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn assert_rejected_for(output: &Output, reason: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && stderr.contains(reason),
        "GCC must reject this fixture for `{reason}`; status {:?}, stderr:\n{stderr}",
        output.status
    );
}

fn assert_compiles(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The native Random observer prelude's include set. `PRIu64` comes only from
/// `<inttypes.h>`: on macOS `chelis_math.h` reaches it through Accelerate, and
/// on Linux nothing here does.
const FORMAT_MACRO_HEADERS: &str = "#include \"chelis_runtime.h\"
#include <assert.h>
#include <math.h>
#include \"chelis_math.h\"
#include <stdio.h>
#include <string.h>
";
const FORMAT_MACRO_USE: &str = "
int print_count(uint64_t count) { return printf(\"%\" PRIu64 \"\\n\", count); }
";

/// GCC's `-Wmisleading-indentation`, inside `-Wall`, warns when a guard that
/// begins its line is followed by a second statement on the same line.
const SAME_LINE_GUARD: &str = "#include <stdlib.h>
static int calls;
int record(int input_count) {
    if (input_count < 1) abort(); ++calls;
    return calls;
}
";
const OWN_LINE_GUARD: &str = "#include <stdlib.h>
static int calls;
int record(int input_count) {
    if (input_count < 1) abort();
    ++calls;
    return calls;
}
";

#[test]
fn linux_fixture_compiler_is_gcc() {
    let output = Command::new("cc")
        .args(["-dM", "-E", "-x", "c", "/dev/null"])
        .output()
        .expect("the fixture C compiler `cc` must run");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let macros = String::from_utf8_lossy(&output.stdout);
    let defines = |name: &str| {
        macros
            .lines()
            .any(|line| line.split_whitespace().nth(1) == Some(name))
    };
    assert!(
        defines("__GNUC__") && !defines("__clang__"),
        "the Linux lanes are the only GCC evidence for C fixtures (chelis#2496), \
         but this lane's `cc` is not GCC"
    );
}

#[test]
fn format_macro_without_its_direct_include_is_rejected() {
    let output = Scratch::new("format-missing")
        .compile(&format!("{FORMAT_MACRO_HEADERS}{FORMAT_MACRO_USE}"));
    assert_rejected_for(&output, "PRIu64");
}

#[test]
fn format_macro_with_its_direct_include_compiles() {
    let output = Scratch::new("format-direct").compile(&format!(
        "{FORMAT_MACRO_HEADERS}#include <inttypes.h>\n{FORMAT_MACRO_USE}"
    ));
    assert_compiles(&output);
}

#[test]
fn unbraced_guard_with_a_same_line_statement_is_rejected() {
    let output = Scratch::new("guard-same-line").compile(SAME_LINE_GUARD);
    assert_rejected_for(&output, "misleading-indentation");
}

#[test]
fn unbraced_guard_with_the_next_statement_on_its_own_line_compiles() {
    let output = Scratch::new("guard-own-line").compile(OWN_LINE_GUARD);
    assert_compiles(&output);
}
