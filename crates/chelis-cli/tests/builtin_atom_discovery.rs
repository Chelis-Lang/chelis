//! Executable boundary witnesses for newly numbered builtin contracts.
use assert_cmd::Command;

fn evaluate(source: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("semantic_boundary.ch");
    std::fs::write(&file, source).unwrap();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(file)
        .output()
        .unwrap()
}

#[test]
fn shifts_at_and_above_each_signed_width_follow_num_13() {
    for (dtype, width) in [("i8", 8), ("i16", 16), ("i32", 32), ("i64", 64)] {
        for count in [width, width + 1] {
            let source = format!(
                "left = shl(cast(1, {dtype}), cast({count}, {dtype}))\npositive = shr(cast(1, {dtype}), cast({count}, {dtype}))\nnegative = shr(cast(-1, {dtype}), cast({count}, {dtype}))\n"
            );
            let output = evaluate(&source);
            assert!(
                output.status.success(),
                "{source}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(
                String::from_utf8(output.stdout).unwrap().trim(),
                "left = 0\npositive = 0\nnegative = -1",
                "{source}"
            );
        }
    }
}

#[test]
fn negative_shift_counts_fail_with_the_num_13_diagnostic() {
    for operation in ["shl", "shr"] {
        let output = evaluate(&format!("result = {operation}(1, -1)\n"));
        assert!(!output.status.success());
        let diagnostic = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            diagnostic.contains("shift amount must be non-negative, got -1"),
            "{diagnostic}"
        );
    }
}

// Build rectangular values by the same recursive shape relation that the
// controlling type-system chapter admits; no rank-two special case.
fn rectangular_list(shape: &[usize], dtype: &str, next: &mut usize) -> String {
    if shape.is_empty() {
        *next += 1;
        return if dtype == "bool" {
            (*next % 2 == 1).to_string()
        } else {
            format!("cast({}, {dtype})", *next)
        };
    }
    format!(
        "[{}]",
        (0..shape[0])
            .map(|_| rectangular_list(&shape[1..], dtype, next))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[test]
fn rectangular_list_ingress_preserves_recursive_shape_and_every_element_dtype() {
    for dtype in [
        "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64", "bool",
    ] {
        for shape in [&[2][..], &[2, 2], &[2, 1, 2], &[1, 2, 1, 2]] {
            let mut count = 0;
            let value = rectangular_list(shape, dtype, &mut count);
            let output = evaluate(&format!("result = to_tensor({value})\n"));
            assert!(
                output.status.success(),
                "{dtype} {shape:?}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let data = (1..=count)
                .map(|n| match dtype {
                    "bool" => (n % 2 == 1).to_string(),
                    "f16" | "bf16" | "f32" | "f64" => format!("{n}.0"),
                    _ => n.to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            assert_eq!(
                String::from_utf8(output.stdout).unwrap().trim(),
                format!("result = tensor(shape={shape:?}, data=[{data}])"),
                "{dtype} {shape:?}"
            );
        }
    }
}

#[test]
fn recursive_list_ingress_rejects_ragged_shapes_and_nonnumeric_leaves() {
    for depth in 0..4 {
        // Computed rows keep this a runtime ingress witness; known literal
        // raggedness is rejected statically by the checker.
        let mut ragged = "[row([1.0f32]), row([2.0f32, 3.0f32])]".to_string();
        for _ in 0..depth {
            ragged = format!("[{ragged}]");
        }
        let output = evaluate(&format!(
            "def row(xs: List[f32]) -> List[f32] = xs\nresult = to_tensor({ragged})\n"
        ));
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("numeric trap: domain in to_tensor at i64"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut invalid = "[\"text\"]".to_string();
        for _ in 0..depth {
            invalid = format!("[{invalid}]");
        }
        let output = evaluate(&format!("result = to_tensor({invalid})\n"));
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    }
}
