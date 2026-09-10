//! #1591 / spec/04 §4.7: earlier output survives a later trap on buffered streams.
use std::{
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

struct Probe(PathBuf);
impl Probe {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("chelis-abort-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn compile(&self, nonflushing_abort: bool) -> PathBuf {
        let source = self.0.join("probe.c");
        let binary = self.0.join(if nonflushing_abort {
            "nonflushing"
        } else {
            "native"
        });
        fs::write(&source, SOURCE).unwrap();
        let mut command = Command::new("cc");
        command
            .args(["-std=c11", "-Wall", "-Werror", "-I"])
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("include"));
        if nonflushing_abort {
            command.arg("-DNONFLUSHING_ABORT");
        }
        let result = command
            .arg(&source)
            .arg("-o")
            .arg(&binary)
            .output()
            .expect("C compiler required");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        binary
    }
}
impl Drop for Probe {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const SOURCE: &str = r#"
#include <stdlib.h>
#ifdef NONFLUSHING_ABORT
/* Darwin's abort can flush itself. This twin isolates the runtime's own
 * obligation; the native twin below exercises the real libc, including Linux. */
static _Noreturn void nonflushing_abort(void) { _Exit(99); }
#define abort nonflushing_abort
#endif
#include "chelis_runtime.h"
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    static char output_buffer[4096], error_buffer[4096];
    if (setvbuf(stdout, output_buffer, _IOFBF, sizeof output_buffer) != 0) return 3;
    if (setvbuf(stderr, error_buffer, _IOFBF, sizeof error_buffer) != 0) return 4;
    if (strcmp(argv[2], "before") == 0) fputs("effect\n", stdout);
    if (strcmp(argv[1], "numeric") == 0) chelis_numeric_trap("numeric trap");
    else if (strcmp(argv[1], "division") == 0) chelis_int_div_guard(0);
    else if (strcmp(argv[1], "abs") == 0) chelis_int_abs_guard(INT8_MIN, 8, "abs trap");
    else if (strcmp(argv[1], "abs_width") == 0) chelis_int_abs_guard(1, 3, "abs trap");
    else if (strcmp(argv[1], "limits") == 0) {
        int64_t lo, hi; chelis_int_limits(3, &lo, &hi);
    }
    else if (strcmp(argv[1], "shift_width") == 0) chelis_int_shift_validate(1, 3);
    else if (strcmp(argv[1], "shift_amount") == 0) chelis_int_shift_validate(-1, 8);
    else if (strcmp(argv[1], "success") != 0) return 5;
    if (strcmp(argv[2], "after") == 0) fputs("effect\n", stdout);
    return 0;
}
"#;

fn check_buffered_output(nonflushing_abort: bool) {
    let probe = Probe::new();
    let binary = probe.compile(nonflushing_abort);
    for (case, diagnostic) in [
        ("numeric", "numeric trap"),
        ("division", "integer division or remainder by zero"),
        ("abs", "abs trap"),
        ("abs_width", "invalid integer abs width"),
        ("limits", "invalid integer width"),
        ("shift_width", "invalid integer shift width"),
        ("shift_amount", "shift amount must be non-negative"),
        ("success", ""),
    ] {
        for position in ["before", "after"] {
            for file_sink in [false, true] {
                let output_path = probe.0.join("stdout");
                let mut command = Command::new(&binary);
                command.args([case, position]);
                if file_sink {
                    command.stdout(Stdio::from(fs::File::create(&output_path).unwrap()));
                }
                let result = command.output().expect("execute abort probe");
                let stdout = if file_sink {
                    fs::read(&output_path).unwrap()
                } else {
                    result.stdout
                };
                let expected = if case == "success" || position == "before" {
                    b"effect\n".as_slice()
                } else {
                    b"".as_slice()
                };
                assert_eq!(
                    stdout, expected,
                    "{case}/{position}/file={file_sink}/nonflushing={nonflushing_abort}"
                );
                assert_eq!(result.status.success(), case == "success", "{case}");
                if case == "success" {
                    assert!(result.stderr.is_empty());
                } else {
                    assert!(
                        String::from_utf8_lossy(&result.stderr).contains(diagnostic),
                        "{case}: {:?}",
                        result.stderr
                    );
                }
            }
        }
    }
}

#[test]
fn runtime_flushes_before_a_nonflushing_abort() {
    check_buffered_output(true);
}

#[test]
fn buffered_output_survives_native_libc_abort() {
    check_buffered_output(false);
}
