//! WS-1 (dtype + Metal cleanup cycle): the C backend now admits bf16
//! and f16 at every active op surface; the CLI gate must no longer
//! reject programs that mention these dtypes when targeting `c`.
//! f8e4m3 stays rejected at type-check per spec/04-type-system.md
//! §1.1.1.
//!
//! These tests do NOT invoke gcc; they just confirm `chelis build
//! --target c` completes successfully (or fails for the right reason
//! in the negative case). The byte-identical / numerical-correctness
//! oracle is in `crates/chelis-backend-c/tests/dtype_matrix_bf16_f16.rs`.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn write_and_build_c_target(source: &str, name: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let kernel_c = dir.path().join(format!("{name}.c"));
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            kernel_c.to_str().unwrap(),
        ])
        .output()
        .expect("invoke chelis build")
}

#[test]
fn cli_admits_bf16_program_targeting_c() {
    // A trivial bf16 program: load + binary add + store. The CLI's
    // C-backend precision gate (`c_backend_supports_precision` +
    // `reject_unsupported_c_precisions`) previously rejected this
    // pre-WS-1 with a "bf16/f16 are admitted only on --target hip"
    // diagnostic; post-WS-1 the build must succeed and emit a
    // kernel that contains the convert-on-load helper.
    let source =
        "def add_bf16(a: tensor[4, bf16], b: tensor[4, bf16]) -> tensor[4, bf16] = a + b\n";
    let out = write_and_build_c_target(source, "add_bf16");
    assert!(
        out.status.success(),
        "chelis build --target c on bf16 program failed: stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cli_admits_f16_program_targeting_c() {
    // f16 sibling of the above. Same gate, same admit-list.
    let source = "def add_f16(a: tensor[4, f16], b: tensor[4, f16]) -> tensor[4, f16] = a + b\n";
    let out = write_and_build_c_target(source, "add_f16");
    assert!(
        out.status.success(),
        "chelis build --target c on f16 program failed: stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn cli_still_rejects_f8e4m3_on_target_c() {
    // f8e4m3 is deferred at the language level per spec §1.1.1; the
    // type checker rejects the syntax before the C backend even
    // sees it. The exact diagnostic source moves over time (lexer,
    // type checker, etc.), but the build MUST fail.
    let source = "def to_f8(x: tensor[4, f32]) -> tensor[4, f8e4m3] = cast(x, f8e4m3)\n";
    let out = write_and_build_c_target(source, "to_f8");
    assert!(
        !out.status.success(),
        "chelis build --target c should reject f8e4m3 (deferred per spec/04-type-system.md \
         §1.1.1) but the build succeeded: stdout={}, stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined.contains("f8e4m3"),
        "rejection diagnostic must name f8e4m3 so users land on the right spec section; \
         got: {combined}"
    );
}
