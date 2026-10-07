use std::io::Write;
use std::process::{Command, Output};

fn nested_application(depth: usize) -> String {
    let body = format!(
        "{}(lit {{type: (t-prim {{}} i32)}} 1){}",
        "(app {} (var {} id) ".repeat(depth),
        ")".repeat(depth)
    );
    format!(
        "(defsig {{}} id (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} i32)))\n\
         (def {{}} id (fn {{}} (params {{}} x) (var {{}} x)))\n\
         (defsig {{}} main (t-fn {{}} (t-prim {{}} i32)))\n\
         (def {{}} main (fn {{}} (params {{}}) {body}))\n"
    )
}

fn validate(depth: usize, canonical: bool) -> Output {
    let mut file = tempfile::Builder::new()
        .suffix(".dp")
        .tempfile()
        .expect("create Deep file");
    file.write_all(nested_application(depth).as_bytes())
        .expect("write Deep file");
    file.flush().expect("flush Deep file");
    if canonical {
        let output = Command::new(env!("CARGO_BIN_EXE_chelis"))
            .args(["fmt", "--inplace"])
            .arg(file.path())
            .output()
            .expect("format Deep file");
        assert!(
            output.status.success(),
            "formatter must accept Deep input: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let mut command = Command::new(env!("CARGO_BIN_EXE_chelis"));
    command.args(["validate", "--deep"]);
    if !canonical {
        command.arg("--allow-style-violations");
    }
    command
        .arg(file.path())
        .output()
        .expect("run Deep validator")
}

#[test]
fn deep_validator_accepts_nested_applications_without_abort() {
    let output = validate(1_500, true);
    assert!(
        output.status.success(),
        "validator must accept parser-supported nesting; status={:?}, stdout={}, stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn deep_validator_reports_parser_nesting_limit_without_abort() {
    let output = validate(40_000, false);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(1),
        "status={:?}; {stderr}",
        output.status
    );
    assert!(
        stderr.contains("Deep input nests deeper than the parser supports at byte "),
        "{stderr}"
    );
}
