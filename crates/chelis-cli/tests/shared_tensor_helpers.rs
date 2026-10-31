//! spec/08 §2: extraction-site growth does not duplicate identical kernels.
mod common;
use std::{fs, path::Path, process::Command};

fn source(count: usize, chain: bool) -> String {
    let mut source = "def forward(x: tensor[4, f32]) -> tensor[4, f32] ! { IO } = {\n".to_string();
    for site in 0..count {
        let input = if chain && site > 0 {
            format!("value_{}", site - 1)
        } else {
            "x".into()
        };
        source.push_str(&format!(
            "  value_{site} = -{input}\n  _ = print(\"site\")\n"
        ));
    }
    source.push_str(&format!("  value_{}\n}}\n", count - 1));
    source
}

fn chelis(dir: &Path, args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_chelis"))
        .current_dir(dir)
        .env_remove("CHELIS_STYLE_GATE_DISABLE")
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "chelis {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn repeated_sites_share_but_distinct_input_labels_do_not() {
    for count in [1, 8, 32] {
        for chain in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            fs::write(dir.path().join("sites.ch"), source(count, chain)).unwrap();
            chelis(dir.path(), &["fmt", "--inplace", "sites.ch"]);
            chelis(dir.path(), &["build", "--emit-c", "sites.ch", "-o", "out"]);
            let c = fs::read_to_string(dir.path().join("out/sites.c")).unwrap();
            let implementations = c
                .lines()
                .filter(|line| {
                    line.starts_with("static __attribute__((noinline)) void __chelis_tensor_impl_")
                })
                .count();
            assert_eq!(
                implementations,
                if chain { count } else { 1 },
                "sites={count}, chain={chain}"
            );
            for site in 0..count {
                assert!(c.contains(&format!(
                    "static void chelis_fn_666f7277617264__tensor_{site}__private("
                )));
            }
        }
    }
}

#[test]
fn built_shared_pipeline_matches_evaluation_and_effect_order() {
    let dir = tempfile::tempdir().unwrap();
    let mut text = source(8, false);
    text.push_str("result = forward(to_tensor([1.0f32, -2.0f32, 3.0f32, -4.0f32]))\n");
    fs::write(dir.path().join("sites.ch"), text).unwrap();
    chelis(dir.path(), &["fmt", "--inplace", "sites.ch"]);
    let evaluated = chelis(dir.path(), &["eval", "--file", "sites.ch"]);
    chelis(dir.path(), &["build", "sites.ch", "-o", "out"]);
    let output = Command::new(dir.path().join("out/sites")).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let native = String::from_utf8(output.stdout).unwrap();
    assert_eq!(native, evaluated);
    assert_eq!(native.lines().filter(|line| *line == "site").count(), 8);
}

#[test]
fn kernel_locator_follows_the_site_and_excludes_decoys() {
    let source = concat!(
        "// static __attribute__((noinline)) void __chelis_tensor_impl_7() { decoy }\n",
        "const char *text = \"static void __chelis_tensor_impl_7() { decoy }\";\n",
        "static void __chelis_tensor_impl_7();\n",
        "static void decoy__chelis_tensor_impl_7() {\nwrong();\n}\n",
        "static __attribute__((noinline)) void __chelis_tensor_impl_7() {\n    real_f32();\n    if (n_in != 1) {\n        abort();\n    }\n    if (chelis_tensor_read_view(inputs[0]).dtype != CHELIS_DTYPE_KEY) {\n        abort();\n    }\n}\n",
        "static __attribute__((noinline)) void __chelis_tensor_impl_8() {\n    double forbidden;\n    if (n_in != 2) {\n        abort();\n    }\n    if (chelis_tensor_read_view(inputs[0]).dtype != CHELIS_DTYPE_F32) {\n        abort();\n    }\n}\n",
        "static void site() {\n    __chelis_tensor_impl_7();\n}\n",
        "static void direct() {\nidentity();\n}\n",
    );
    let selected = common::tensor_kernel_definition(source, "site");
    assert!(
        selected.contains("real_f32()")
            && !selected.contains("decoy")
            && !selected.contains("double")
    );
    assert!(selected.contains("if (n_in != 1) {"));
    assert!(selected.contains(").dtype != CHELIS_DTYPE_KEY) {"));
    assert!(common::tensor_kernel_definition(source, "direct").contains("identity()"));
    let changed_call = source.replace(
        "    __chelis_tensor_impl_7();",
        "    __chelis_tensor_impl_8();",
    );
    let changed_kernel = common::tensor_kernel_definition(&changed_call, "site");
    assert!(
        changed_kernel.contains("double forbidden"),
        "the dtype oracle must inspect the actual callee"
    );
    assert!(!changed_kernel.contains("if (n_in != 1) {"));
    assert!(!changed_kernel.contains(").dtype != CHELIS_DTYPE_KEY) {"));
}

#[test]
fn kernel_locator_rejects_missing_or_ambiguous_callees() {
    for source in [
        "static void site() {\n    __chelis_tensor_impl_7();\n}\n",
        "static void __chelis_tensor_impl_7();\nstatic void site() {\n    __chelis_tensor_impl_7();\n}\n",
        "static void site() {\n    __chelis_tensor_impl_bad();\n}\n",
        "static void site() {\n    __chelis_tensor_impl_7();\n    __chelis_tensor_impl_8();\n}\n",
    ] {
        assert!(
            std::panic::catch_unwind(|| common::tensor_kernel_definition(source, "site")).is_err(),
            "{source}"
        );
    }
}
