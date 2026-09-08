//! Bounded C2.2 emitter authority control. Compile the production snapshot
//! method in a small recorder, then repeat with the retired arithmetic restored.
//! The companion generated-DAG and host tests execute optimized C with UBSan.
use std::{fs, process::Command};

fn snapshot_method() -> &'static str {
    let source = include_str!("../src/emit.rs");
    let start = source
        .find("    fn emit_tensor_snapshot(")
        .expect("snapshot method");
    let end = source[start..]
        .find("    fn emit_owned_tensor(")
        .expect("next method");
    &source[start..start + end]
}

fn run_snapshot(method: &str) -> std::process::Output {
    let temp = tempfile::tempdir().unwrap();
    let source = format!(
        r#"
use std::collections::BTreeSet;
#[derive(Default)]
struct Emitter {{ lines: Vec<String>, write_nodes: BTreeSet<usize> }}
impl Emitter {{
    fn line(&mut self, line: &str) {{ self.lines.push(line.to_string()); }}
    {method}
}}
fn main() {{
    for writable in [false, true] {{
        let mut emitter = Emitter::default();
        emitter.emit_tensor_snapshot(7, writable);
        let strides: Vec<_> = emitter.lines.iter().filter(|line| line.contains("t7_strides[__axis] =")).collect();
        assert_eq!(strides.len(), 1, "exactly one stride projection");
        assert!(strides[0].ends_with("t7_strides[__axis] = chelis_tensor_stride(t7, __axis);"),
                "stride authority must be the runtime owner");
        let bytes: Vec<_> = emitter.lines.iter().filter(|line| line.starts_with("int64_t t7_byte_capacity =")).collect();
        assert_eq!(bytes, ["int64_t t7_byte_capacity = chelis_tensor_byte_count(t7);"],
                   "byte authority must be the runtime owner");
        assert_eq!(emitter.write_nodes.contains(&7), writable);
    }}
}}
"#
    );
    let path = temp.path().join("snapshot.rs");
    fs::write(&path, source).unwrap();
    let binary = temp.path().join("snapshot");
    let compiled = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "-O"])
        .arg(&path)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "snapshot probe did not compile: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    Command::new(binary).output().unwrap()
}

#[test]
fn snapshot_delegation_rejects_raw_stride_and_byte_reconstruction() {
    let method = snapshot_method();
    let baseline = run_snapshot(method);
    assert!(
        baseline.status.success(),
        "{}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    for (from, to, message) in [
        (
            "int64_t t{id}_byte_capacity = chelis_tensor_byte_count(t{id});",
            "int64_t t{id}_byte_capacity = t{id}_size * chelis_dtype_size(t{id}_dtype);",
            "byte authority must be the runtime owner",
        ),
        (
            "for (int32_t __axis = 0; __axis < t{id}_rank; ++__axis) t{id}_strides[__axis] = chelis_tensor_stride(t{id}, __axis);",
            "int64_t stride = 1; for (int32_t __axis = t{id}_rank; __axis-- > 0;) {{ t{id}_strides[__axis] = stride; stride *= t{id}_shape[__axis]; }}",
            "stride authority must be the runtime owner",
        ),
    ] {
        assert_eq!(
            method.matches(from).count(),
            1,
            "unique production mutation anchor"
        );
        let run = run_snapshot(&method.replace(from, to));
        assert!(
            !run.status.success(),
            "restored raw arithmetic escaped the control"
        );
        assert!(String::from_utf8_lossy(&run.stderr).contains(message));
    }
}
