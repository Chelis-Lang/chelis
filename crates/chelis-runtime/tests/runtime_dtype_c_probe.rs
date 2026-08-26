use std::fs;
use std::path::PathBuf;
use std::process::Command;
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

fn compile_and_run(header: &str) -> String {
    let directory = TempDir::new("dtype-probe");
    let header_path = directory.0.join("chelis_runtime_dtype.h");
    let source_path = directory.0.join("probe.c");
    let executable = directory.0.join("probe");

    fs::write(&header_path, header)
        .unwrap_or_else(|error| panic!("write probe header {}: {error}", header_path.display()));
    let mut source = String::from(
        "#include <stdio.h>\n#include \"chelis_runtime_dtype.h\"\n\nint main(void) {\n",
    );
    source.push_str("    printf(\"%zu\", sizeof(chelis_dtype));\n");
    for dtype in RuntimeDType::ALL {
        source.push_str(&format!(
            "    printf(\" %u\", (unsigned){});\n",
            dtype.c_macro()
        ));
    }
    source.push_str("    printf(\"\\n\");\n    return 0;\n}\n");
    fs::write(&source_path, source)
        .unwrap_or_else(|error| panic!("write probe source {}: {error}", source_path.display()));

    let compile = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg(&source_path)
        .arg("-o")
        .arg(&executable)
        .current_dir(&directory.0)
        .output()
        .expect("invoke C compiler");
    assert!(
        compile.status.success(),
        "compile dtype probe\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );
    let output = Command::new(&executable)
        .current_dir(&directory.0)
        .output()
        .expect("execute C dtype probe");
    assert!(output.status.success(), "dtype probe exited nonzero");
    String::from_utf8(output.stdout).expect("dtype probe emits UTF-8")
}

fn compare_exact_tags(header: &str) -> Result<(), String> {
    let output = compile_and_run(header);
    let observed = output
        .split_whitespace()
        .map(|field| field.parse::<usize>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if observed.first() != Some(&1) {
        return Err(format!(
            "C chelis_dtype width was {:?}; exact carrier width is 1",
            observed.first()
        ));
    }
    for (index, dtype) in RuntimeDType::ALL.iter().enumerate() {
        let actual = observed.get(index + 1).copied();
        if actual != Some(dtype.id() as usize) {
            return Err(format!(
                "C tag for {dtype:?} was {actual:?}; Rust tag was {}",
                dtype.id()
            ));
        }
    }
    Ok(())
}

fn replace_tag(header: &str, dtype: RuntimeDType, wrong_tag: u8) -> String {
    let expected = format!("{} = {}", dtype.c_macro(), dtype.id());
    let replacement = format!("{} = {wrong_tag}", dtype.c_macro());
    assert_eq!(
        header.matches(&expected).count(),
        1,
        "the generated header must contain one tag for {dtype:?}"
    );
    header.replacen(&expected, &replacement, 1)
}

#[test]
fn every_c_dtype_tag_and_the_carrier_width_match_rust() {
    compare_exact_tags(&render_runtime_dtype_c_header()).unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn c_probe_detects_a_deliberately_incorrect_generated_tag() {
    let generated = render_runtime_dtype_c_header();
    let mutated = replace_tag(&generated, RuntimeDType::F32, 9);
    let error = compare_exact_tags(&mutated).expect_err("the wrong C tag must be rejected");
    assert!(
        error.contains("F32"),
        "unexpected comparison error: {error}"
    );
    assert!(
        error.contains("was Some(9); Rust tag was 0"),
        "unexpected comparison error: {error}"
    );
}
