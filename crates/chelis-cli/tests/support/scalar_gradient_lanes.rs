// chelis#2993 and chelis#3017: shared by the standing canary
// `issue_1741_scalar_gradient_cli` and its nightly sweep
// `issue_3017_scalar_gradient_sweep`. `chelis build` computes a top-level
// scalar `grad` at the operand dtype ([04-NUM-8]) through the same
// reverse-mode DAG `chelis eval` evaluates (spec/06 section 2.3), so the
// executable prints what eval prints.

fn width_program(dtype: &str, body: &str, root: &str) -> String {
    let body = body.replace("{t}", dtype);
    let root = root.replace("{t}", dtype);
    format!("module Probe.Case\ndef f(x: {dtype}) -> {dtype} = {body}\nout = {root}\n")
}

fn chelis(args: &[&str]) -> std::process::Output {
    Command::new(assert_cmd::cargo_bin!("chelis"))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args)
        .output()
        .unwrap()
}

fn eval_stdout(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("p.ch");
    common::write_file(&path, source);
    let output = chelis(&["eval", "--file", path.to_str().unwrap()]);
    assert!(output.status.success(), "{source}\n{output:?}");
    String::from_utf8(output.stdout).unwrap()
}
