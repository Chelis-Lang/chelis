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
fn check_accepts_function_alias_but_observation_names_unavailable_root() {
    for annotated in [false, true] {
        let dir = TempDir::new().unwrap();
        let source = format!(
            "module Aliases\ndef anchor(x: int32) -> int32 = x\n{} = anchor\ndef user() -> int32 = alias(1)\n",
            if annotated {
                "alias: (int32) -> int32"
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
        for command in ["eval", "build"] {
            let mut process = Command::new(env!("CARGO_BIN_EXE_chelis"));
            process.arg(command);
            if command == "eval" {
                process.arg("--file");
            }
            process.arg(&file);
            let destination = dir.path().join("output");
            if command == "build" {
                process.args(["--target", "c", "-o"]).arg(&destination);
            }
            let result = process.output().unwrap();
            assert!(!result.status.success(), "{result:?}");
            let message = String::from_utf8_lossy(&result.stderr);
            for required in ["[05-UNS-1]", "alias", "Host", "function"] {
                assert!(message.contains(required), "{message}");
            }
            assert!(!destination.exists(), "failed build left an artifact");
            assert!(result.stdout.is_empty(), "partial output: {result:?}");
        }
    }
}

#[test]
fn ordinary_aliases_have_distinct_roots_on_eval_and_c() {
    for (value, expected) in [
        ("1", "value = 1\nother = 1\nuser = 1\n"),
        (
            "to_tensor([1, 2])",
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
            .arg(&file)
            .args(["--target", "c", "-o"])
            .arg(dir.path())
            .output()
            .unwrap();
        assert!(built.status.success(), "{built:?}");
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
        assert_eq!(String::from_utf8(run.stdout).unwrap(), expected);
    }
}
