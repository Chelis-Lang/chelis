use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use chelis_runtime::dtype_header::render_runtime_dtype_c_header;
use chelis_runtime::RuntimeDType;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "chelis-runtime-{label}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&path)
            .unwrap_or_else(|error| panic!("create probe directory {}: {error}", path.display()));
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct CProbe {
    _directory: TempDir,
    executable: PathBuf,
}

impl CProbe {
    fn compile(header: &str) -> Self {
        let directory = TempDir::new("dtype-probe");
        let header_path = directory.0.join("chelis_runtime_dtype.h");
        let source_path = directory.0.join("probe.c");
        let executable = directory.0.join("probe");

        fs::write(&header_path, header).unwrap_or_else(|error| {
            panic!("write probe header {}: {error}", header_path.display())
        });
        fs::write(
            &source_path,
            r#"#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "chelis_runtime_dtype.h"

int main(int argc, char **argv) {
    char *end = NULL;
    long parsed;

    if (argc != 2) {
        return 64;
    }
    parsed = strtol(argv[1], &end, 10);
    if (*argv[1] == '\0' || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX) {
        return 65;
    }
    printf("%zu\n", chelis_runtime_dtype_size_checked((int)parsed));
    return 0;
}
"#,
        )
        .unwrap_or_else(|error| panic!("write probe source {}: {error}", source_path.display()));

        let compile = Command::new("gcc")
            .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
            .arg(&source_path)
            .arg("-o")
            .arg(&executable)
            .current_dir(&directory.0)
            .output()
            .expect("invoke the repository gcc command");
        assert!(
            compile.status.success(),
            "compile dtype probe with gcc\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&compile.stdout),
            String::from_utf8_lossy(&compile.stderr)
        );

        Self {
            _directory: directory,
            executable,
        }
    }

    fn run(&self, id: i32) -> Output {
        Command::new(&self.executable)
            .arg(id.to_string())
            .current_dir(self.executable.parent().expect("probe directory"))
            .output()
            .unwrap_or_else(|error| panic!("execute C dtype probe for {id}: {error}"))
    }
}

fn compare_valid_widths(header: &str) -> Result<(), String> {
    let probe = CProbe::compile(header);
    for dtype in RuntimeDType::ALL {
        let output = probe.run(dtype.id());
        if !output.status.success() {
            return Err(format!(
                "C probe rejected valid dtype {dtype:?} ({}) with {}: {}",
                dtype.id(),
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let stdout = String::from_utf8(output.stdout)
            .map_err(|error| format!("C probe emitted invalid UTF-8 for {dtype:?}: {error}"))?;
        let c_width = stdout.trim().parse::<usize>().map_err(|error| {
            format!("C probe emitted an invalid width for {dtype:?}: {stdout:?}: {error}")
        })?;
        if c_width != dtype.byte_width() {
            return Err(format!(
                "C probe width for {dtype:?} was {c_width}; Rust width was {}",
                dtype.byte_width()
            ));
        }
    }
    Ok(())
}

fn replace_case_width(header: &str, dtype: RuntimeDType, wrong_width: usize) -> String {
    let expected = format!("case {}: return {};", dtype.c_macro(), dtype.byte_width());
    let replacement = format!("case {}: return {wrong_width};", dtype.c_macro());
    assert_eq!(
        header.matches(&expected).count(),
        1,
        "the generated header must contain one case for {dtype:?}"
    );
    header.replacen(&expected, &replacement, 1)
}

#[test]
fn every_valid_c_dtype_tag_returns_the_rust_width() {
    compare_valid_widths(&render_runtime_dtype_c_header())
        .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn invalid_c_dtype_tag_exits_unsuccessfully_without_a_fallback_width() {
    let probe = CProbe::compile(&render_runtime_dtype_c_header());
    let output = probe.run(-1);

    assert!(!output.status.success(), "the invalid tag must fail");
    assert!(
        output.stdout.is_empty(),
        "the invalid tag must not print a fallback width: {:?}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("invalid Chelis runtime dtype id: -1"),
        "the invalid tag must report the rejected value: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn c_probe_detects_a_deliberately_incorrect_generated_case() {
    let generated = render_runtime_dtype_c_header();
    let mutated = replace_case_width(&generated, RuntimeDType::F32, 5);
    let error = compare_valid_widths(&mutated).expect_err("the wrong C width must be rejected");

    assert!(
        error.contains("F32"),
        "unexpected comparison error: {error}"
    );
    assert!(
        error.contains("was 5; Rust width was 4"),
        "unexpected comparison error: {error}"
    );
}
