//! C4.4: a backward shape observation does not redeclare a forward extent.
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use tempfile::tempdir;

const SOURCE: &str = r#"module Repro.Restored_Extent_Grad
type Model[c] =
  | Model { stem: tensor[c, 1, 2, 2, f32] }
def reshape_nchw[a, c, s](x: tensor[a, c, s, s, f32]) -> tensor[a, c, s, s, f32] = {
  n_a = shape(x, 0i32)
  n_c = shape(x, 1i32)
  n_s = shape(x, 2i32)
  n_rows = mul(n_a, mul(n_s, n_s))
  y = neg(reshape(permute(x, 0i32, 2i32, 3i32, 1i32), [n_rows, n_c]))
  permute(reshape(y, [n_a, n_s, n_s, n_c]), 0i32, 3i32, 1i32, 2i32)
}
def loss[a, c, s](p: Model[c], x: tensor[a, 1, s, s, f32]) -> f32 = {
  y = reshape_nchw(conv(x, p.stem, [2i64, 2i64], [(0i64, 0i64), (0i64, 0i64)]))
  tensor_to_scalar(sum(sum(sum(sum(y, 3i32), 2i32), 1i32), 0i32))
}
def step[a, c, s](p: Model[c], x: tensor[a, 1, s, s, f32]) -> Model[c] = grad(loss, wrt=p)(p, x)
def main() -> Model[2] = {
  p = Model { stem: reshape(insert(scalar_to_tensor(0.25f32), 0i32, 8i64), [2i64, 1i64, 2i64, 2i64]) }
  x = reshape(insert(scalar_to_tensor(1.0f32), 0i32, 16i64), [1i64, 1i64, 4i64, 4i64])
  step(p, x)
}
"#;

#[test]
fn inverse_reshape_after_conv_preserves_forward_extent_and_exact_gradient() {
    let dir = tempdir().unwrap();
    for channels in [2, 3] {
        let source = SOURCE
            .replace("Model[2]", &format!("Model[{channels}]"))
            .replace(
                "8i64), [2i64, 1i64, 2i64, 2i64]",
                &format!("{}i64), [{channels}i64, 1i64, 2i64, 2i64]", channels * 4),
            );
        let path = dir.path().join("restored_extent.ch");
        write_file(&path, &source);
        let eval = Command::cargo_bin("chelis")
            .unwrap()
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            eval.status.success(),
            "{}",
            String::from_utf8_lossy(&eval.stderr)
        );
        let data = vec!["-4.0"; channels * 4].join(", ");
        let expected =
            format!("main = Model(tensor(shape=[{channels}, 1, 2, 2], data=[{data}]))\n");
        assert_eq!(String::from_utf8(eval.stdout).unwrap(), expected);
        if gcc_available() {
            assert_eq!(build_and_run(&source, "restored_extent"), expected);
        }
    }
}
