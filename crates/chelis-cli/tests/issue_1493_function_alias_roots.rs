//! [05-OBS-7,11]: checked aliases do not become internal lowering errors.
mod common;
use std::{fs, process::Command};
use tempfile::TempDir;

fn fixture(dir: &TempDir, source: &str) -> std::path::PathBuf {
    let file = dir.path().join("alias.ch");
    fs::write(&file, source).unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_chelis"))
            .args(["fmt", "--inplace"])
            .arg(&file)
            .status()
            .unwrap()
            .success()
    );
    file
}

#[test]
fn function_alias_program_checks_evaluates_and_builds_its_concrete_result() {
    for annotated in [false, true] {
        let dir = TempDir::new().unwrap();
        let source = format!(
            "module Aliases\ndef anchor(x: i32) -> i32 = x\n{} = anchor\ndef user() -> i32 = alias(1)\n",
            if annotated {
                "alias: (i32) -> i32"
            } else {
                "alias"
            }
        );
        let file = fixture(&dir, &source);
        let checked = Command::new(env!("CARGO_BIN_EXE_chelis"))
            .arg("check")
            .arg(&file)
            .output()
            .unwrap();
        assert!(checked.status.success(), "{:?}", checked);
        let evaluated = Command::new(env!("CARGO_BIN_EXE_chelis"))
            .args(["eval", "--file"])
            .arg(&file)
            .output()
            .unwrap();
        assert!(evaluated.status.success(), "{evaluated:?}");
        assert_eq!(String::from_utf8(evaluated.stdout).unwrap(), "user = 1\n");
        let built = Command::new(env!("CARGO_BIN_EXE_chelis"))
            .arg("build")
            .arg("--emit-c")
            .arg(&file)
            .args(["--target", "c", "-o"])
            .arg(dir.path())
            .output()
            .unwrap();
        assert!(built.status.success(), "{built:?}");
        assert_eq!(run_generated(&dir), "user = 1\n");
    }
}

#[test]
fn ordinary_aliases_have_distinct_roots_on_eval_and_c() {
    for (value, expected) in [
        ("1", "value = 1\nother = 1\nuser = 1\n"),
        (
            "to_tensor([1, 2], i32)",
            "value = tensor(shape=[2], data=[1, 2])\nother = tensor(shape=[2], data=[1, 2])\nuser = tensor(shape=[2], data=[1, 2])\n",
        ),
    ] {
        let dir = TempDir::new().unwrap();
        let file = fixture(
            &dir,
            &format!("module Aliases\nvalue = {value}\nother = value\ndef user() = other\n"),
        );
        let result = Command::new(env!("CARGO_BIN_EXE_chelis"))
            .args(["eval", "--file"])
            .arg(&file)
            .output()
            .unwrap();
        assert!(result.status.success(), "{result:?}");
        assert_eq!(String::from_utf8(result.stdout).unwrap(), expected);
        let built = Command::new(env!("CARGO_BIN_EXE_chelis"))
            .arg("build")
            .arg("--emit-c")
            .arg(&file)
            .args(["--target", "c", "-o"])
            .arg(dir.path())
            .output()
            .unwrap();
        assert!(built.status.success(), "{built:?}");
        assert_eq!(run_generated(&dir), expected);
    }
}

fn run_generated(dir: &TempDir) -> String {
    let c_file = fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|file| file.extension().is_some_and(|ext| ext == "c"))
        .unwrap();
    assert!(
        common::link_generated(
            dir.path(),
            c_file.file_name().unwrap().to_str().unwrap(),
            "program"
        )
        .success()
    );
    let run = Command::new(dir.path().join("program")).output().unwrap();
    assert!(run.status.success(), "{run:?}");
    String::from_utf8(run.stdout).unwrap()
}
