//! #1741: checked scalar gradients must also execute in eval and property fuzzing.
use std::process::Command;

const PROPERTY: &str = "module M\n@property exp_grad_positive forall(x: f32)\nwhere x > 0.5, x < 9.5:\n  (grad(fn (xx: f32) -> exp(xx), wrt=xx)(x) > 0.0)\n";

#[test]
fn original_scalar_gradient_property_passes_auto_and_fuzz() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("property.ch");
    std::fs::write(&path, PROPERTY).unwrap();
    for tier in ["auto", "fuzz-only"] {
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["prove", path.to_str().unwrap(), "--tier", tier])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{tier}: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout)
                .contains("1 passed, 0 failed, 0 unsupported, 0 errors"),
            "{output:?}"
        );
    }
}

#[test]
fn check_and_eval_agree_for_scalar_gradients_and_reject_real_mixed_surfaces() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gradient.ch");
    for dtype in ["f32", "f64"] {
        std::fs::write(&path, format!("def g(x: {dtype}) -> bool = gt(grad(fn (xx: {dtype}) -> exp(xx), wrt=xx)(x), 0.0{dtype})\nout = g(1.0{dtype})\n")).unwrap();
        for args in [
            vec!["check", path.to_str().unwrap()],
            vec!["eval", "--file", path.to_str().unwrap()],
        ] {
            let output = Command::new(assert_cmd::cargo_bin!("chelis"))
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(&args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{args:?}: {} {}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            if args[0] == "eval" {
                assert_eq!(
                    String::from_utf8(output.stdout).unwrap().trim(),
                    "out = true"
                );
            }
        }
    }
    for tensor in ["scalar_to_tensor(1.0f32)", "to_tensor([1.0f32])"] {
        std::fs::write(&path, format!("out = gt({tensor}, 0.0f32)\n")).unwrap();
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{output:?}");
    }
}
