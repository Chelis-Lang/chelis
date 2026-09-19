//! Phase 3 gate-contract regressions for chelis#730.
//!
//! Gates may make an emitter rejection earlier or more specific, but they
//! must not fork target policy by public entry path. These probes lock the
//! two drifted policies reported by chelis#697 and chelis#698.

use std::fs;

use assert_cmd::Command;
use chelis_compiler_api::{
    compiler::{BuildTarget, compile, compile_for_execution},
    schema::{CompileRequest, CompileTarget, SourceKind},
};
use tempfile::tempdir;

fn build(source: &str, stem: &str, target: &str) -> assert_cmd::assert::Assert {
    let dir = tempdir().expect("tempdir");
    let source_path = dir.path().join(format!("{stem}.ch"));
    let output_path = dir.path().join("out");
    fs::write(&source_path, source).expect("write source");

    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1").args([
        "build",
        source_path.to_str().expect("utf-8 source path"),
        "--target",
        target,
        "--output",
        output_path.to_str().expect("utf-8 output path"),
    ]);
    command.assert()
}

fn build_deep(source: &str, stem: &str, target: &str) -> assert_cmd::assert::Assert {
    let dir = tempdir().expect("tempdir");
    let source_path = dir.path().join(format!("{stem}.dp"));
    let output_path = dir.path().join("out");
    fs::write(&source_path, source).expect("write Deep source");

    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1").args([
        "build",
        source_path.to_str().expect("utf-8 source path"),
        "--deep",
        "--target",
        target,
        "--output",
        output_path.to_str().expect("utf-8 output path"),
    ]);
    command.assert()
}

/// Shared gate APIs accept only the closed build-target vocabulary. A caller
/// cannot smuggle an unknown spelling through a stringly fallback that turns
/// the gate into a silent no-op.
#[test]
fn shared_gate_target_parser_rejects_unknown_spellings() {
    assert_eq!(BuildTarget::try_from("c"), Ok(BuildTarget::C));
    assert_eq!(BuildTarget::try_from("hip"), Ok(BuildTarget::Hip));
    assert_eq!(BuildTarget::try_from("metal"), Ok(BuildTarget::Metal));
    assert_eq!(
        BuildTarget::try_from("vulkan").expect_err("unknown target must not be admitted"),
        "unknown target 'vulkan': expected 'c', 'hip', or 'metal'"
    );
}

/// chelis#697: C dtype admission is a property of the lowered operation and
/// backend, never of whether an unrelated declaration happens to need the
/// host lane. Both forms use an active, exactly-supported i64 scalar.
#[test]
fn c_dtype_admission_does_not_depend_on_an_unrelated_host_function() {
    let tensor_only = "def value() -> i64 = cast(1, i64)\n";
    let with_unrelated_host =
        "def label() -> string = \"unrelated\"\ndef value() -> i64 = cast(1, i64)\n";

    build(tensor_only, "int64_tensor_only", "c").success();
    build(with_unrelated_host, "int64_with_host", "c").success();
}

/// Negative parity for the C-gate deletion: removing a stale early gate must
/// not admit a dtype the language still defers. The checker remains the
/// competent rejection boundary for f8e4m3.
#[test]
fn c_build_still_rejects_the_deferred_f8e4m3_dtype() {
    build(
        "def value(x: tensor[1, f8e4m3]) -> tensor[1, f8e4m3] = x\n",
        "deferred_f8e4m3",
        "c",
    )
    .failure()
    .stderr(predicates::str::contains("f8e4m3"));
}

/// chelis#698: the compiler API's stale copy rejected f64 before the HIP
/// emitter, while the CLI copy admitted the backend's typed f64 kernels. One
/// shared policy must admit the same supported program through both entries.
#[test]
fn hip_f64_admission_agrees_across_public_build_paths() {
    let source = "def add64(a: tensor[2, f64], b: tensor[2, f64]) -> tensor[2, f64] = \
                  add(a, b)\n";

    compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::Hip,
        entry_name: Some("add64".to_string()),
    })
    .expect("compiler API must admit the HIP backend's typed f64 add kernel");
    build(source, "hip_f64_add", "hip").success();
}

/// Negative parity for the shared HIP policy: a capability cell that remains
/// absent is rejected by the same typed early gate in the shipped CLI path.
#[test]
fn hip_scatter_elements_rejects_an_unimplemented_f64_payload_cell() {
    build(
        "def apply_scatter(data: tensor[2, 2, f64], indices: tensor[2, 2, i32], \
         updates: tensor[2, 2, f64]) -> tensor[2, 2, f64] = \
         scatter_elements(data, indices, updates, 0)\n",
        "hip_scatter_elements_f64",
        "hip",
    )
    .failure()
    .stderr(predicates::str::contains("scatter_elements"))
    .stderr(predicates::str::contains("early capability gate"));
}

/// A BLAS matmul only subsumes its own operation. Narrow-float arithmetic in
/// either operand is real upstream compute and must be rejected by the shared
/// gate before the HIP emitter reaches its missing elementwise suffix.
#[test]
fn hip_narrow_matmul_operand_compute_rejects_without_panicking_across_build_paths() {
    for precision in ["f16", "bf16"] {
        let source = format!(
            "def mm(a: tensor[2, 2, {precision}], b: tensor[2, 2, {precision}], \
             c: tensor[2, 2, {precision}]) -> tensor[2, 2, {precision}] = \
             matmul(add(a, b), c)\n"
        );

        let result = std::panic::catch_unwind(|| {
            compile(CompileRequest {
                source_kind: SourceKind::Surf,
                source: source.clone(),
                target: CompileTarget::Hip,
                entry_name: Some(format!("hip_{precision}_matmul_operand_compute")),
            })
        });
        let error = result
            .unwrap_or_else(|_| panic!("compiler API panicked for {precision} operand compute"))
            .expect_err("narrow-float operand compute must be rejected");
        let message = &error.errors[0].message;
        assert!(message.contains("early capability gate"), "{message}");
        assert!(message.contains(precision), "{message}");
        assert!(message.contains("Add"), "{message}");

        build(
            &source,
            &format!("hip_{precision}_matmul_operand_compute"),
            "hip",
        )
        .failure()
        .stderr(predicates::str::contains("early capability gate"))
        .stderr(predicates::str::contains("Add"));
    }
}

/// Positive parity: the supported narrow-float cell is the `BlasMatmul`
/// itself over loaded operands. Removing the erroneous input-cone walk must
/// not restore the compiler API's former over-strict rejection.
#[test]
fn hip_narrow_blas_matmul_stays_admitted_across_build_paths() {
    for precision in ["f16", "bf16"] {
        let source = format!(
            "def mm(a: tensor[2, 2, {precision}], b: tensor[2, 2, {precision}]) \
             -> tensor[2, 2, {precision}] = matmul(a, b)\n"
        );

        compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            target: CompileTarget::Hip,
            entry_name: Some(format!("hip_{precision}_blas_matmul")),
        })
        .unwrap_or_else(|error| panic!("compiler API rejected {precision} BLAS matmul: {error:?}"));
        build(&source, &format!("hip_{precision}_blas_matmul"), "hip").success();
    }
}

/// Sealed fixed-control C support and the unsupported device lanes are each
/// independent of the public entry path (#1872, [05-OP-37]).
#[test]
fn compiled_dropout_rejection_agrees_across_public_build_paths() {
    let source = "def noisy(x: tensor[4, f32]) -> tensor[4, f32] = \
                  with seed(42i64) { dropout(x, 0.5) }\n";

    compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some("noisy".to_string()),
    })
    .expect("fixed-control C source entry must retain its sealed plan");
    build(source, "dropout_c", "c").success();

    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::Hip,
        entry_name: Some("noisy".to_string()),
    })
    .expect_err("compiled dropout must be rejected by the shared effect gate");
    let diagnostic = &error.errors[0];
    assert_eq!(diagnostic.kind().as_str(), "unsupported_feature");
    assert!(diagnostic.message.contains("early capability gate"));
    assert!(diagnostic.message.contains("unimplemented chelis#1192"));
    assert!(
        diagnostic
            .message
            .contains("run this program with `chelis eval`")
    );
    assert!(!diagnostic.message.contains("with seed(...)` instead"));

    for target in ["hip", "metal"] {
        build(source, &format!("dropout_{target}"), target)
            .failure()
            .stderr(predicates::str::contains("unsupported:"))
            .stderr(predicates::str::contains("early capability gate"))
            .stderr(predicates::str::contains("unimplemented chelis#1192"))
            .stderr(predicates::str::contains(
                "run this program with `chelis eval`",
            ));
    }
}

/// #1872: static rate is not a closed entry; the public ABI supplies no RNG.
#[test]
fn bare_inherited_random_c_surf_entry_rejects_across_public_paths() {
    assert_bare_inherited_random_rejects(SourceKind::Surf);
}

#[test]
fn bare_inherited_random_c_deep_entry_rejects_across_public_paths() {
    assert_bare_inherited_random_rejects(SourceKind::Deep);
}

fn assert_bare_inherited_random_rejects(kind: SourceKind) {
    let source = "def sample(x: tensor[4, f32]) -> tensor[4, f32] = dropout(x, 0.5f32)\n";
    let decls = chelis_surf::parser::parse_str(source).unwrap();
    let deep = chelis_deep::printer::print_canonical(
        &chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar"),
    );
    let text = match kind {
        SourceKind::Surf => source,
        SourceKind::Deep => deep.as_str(),
    };
    for execution in [false, true] {
        let request = CompileRequest {
            source_kind: kind,
            source: text.to_string(),
            target: CompileTarget::C,
            entry_name: Some("sample".into()),
        };
        let errors = if execution {
            compile_for_execution(request)
                .expect_err("public entry has no ambient RNG")
                .errors
        } else {
            compile(request)
                .expect_err("public entry has no ambient RNG")
                .errors
        };
        assert_eq!(errors[0].kind().as_str(), "unsupported_feature");
        assert!(errors[0].message.contains("inherited Random"), "{errors:?}");
    }
    let result = match kind {
        SourceKind::Surf => build(source, "inherited_surf", "c"),
        SourceKind::Deep => build_deep(&deep, "inherited_deep", "c"),
    };
    result
        .failure()
        .stderr(predicates::str::contains("unsupported:"));
}

/// Entry-scoped compilation admits both the pure and source-fixed C entry;
/// selecting a runtime-rate entry still rejects it.
#[test]
fn compiled_dropout_gate_follows_the_emitted_entry_scope() {
    let source = "def clean(x: tensor[4, f32]) -> tensor[4, f32] = add(x, x)\n\
                  def noisy(x: tensor[4, f32]) -> tensor[4, f32] = \
                  with seed(42i64) { dropout(x, 0.5) }\n";

    compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some("clean".to_string()),
    })
    .expect("an un-emitted dropout sibling must not block the selected clean entry");

    compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some("noisy".to_string()),
    })
    .expect("selecting the source-fixed C dropout entry must retain its sealed plan");

    let runtime = source.replace("dropout(x, 0.5)", "dropout(x, tensor_to_scalar(sum(x, 0)))");
    let error = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: runtime,
        target: CompileTarget::C,
        entry_name: Some("noisy".to_string()),
    })
    .expect_err("selected runtime-rate dropout must remain unsupported");
    assert!(
        error.errors[0].message.contains("RuntimeRate")
            || error.errors[0]
                .message
                .contains("statically-resolvable rate"),
        "{error:?}"
    );
}

/// A scalar activation uses the host-expression lane, but callable
/// compilation is still entry-scoped. An unsupported tensor helper owned by
/// an unselected sibling must not enter either the typed gates or emission.
#[test]
fn scalar_activation_entry_ignores_an_unemitted_dropout_sibling() {
    let source = "def activate(x: f64) -> f64 = gelu(x)\n\
                  def clean(x: f64) -> f64 = activate(x)\n\
                  def noisy(x: tensor[4, f32]) -> tensor[4, f32] = \
                  with seed(42i64) { dropout(x, 0.5) }\n";

    let artifact = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some("clean".to_string()),
    })
    .expect("an un-emitted dropout sibling must not block the selected scalar entry");

    let emitted = artifact
        .compile_result
        .files
        .iter()
        .map(|file| file.contents.as_str())
        .collect::<String>();
    assert!(
        emitted.contains("chelis_host_gelu_f64"),
        "the selected scalar activation must be emitted:\n{emitted}"
    );
    assert!(
        !emitted.contains("dropout") && !emitted.contains("noisy"),
        "the unselected sibling must be absent from the artifact:\n{emitted}"
    );
}

/// A parameter is a lexical binding, even when its spelling collides with an
/// unselected top-level function. Entry projection must not retain that
/// sibling merely because the selected body reads the parameter.
#[test]
fn scalar_activation_entry_respects_parameter_shadowing() {
    let source = "def noisy(x: tensor[4, f32]) -> tensor[4, f32] = \
                  with seed(42i64) { dropout(x, 0.5) }\n\
                  def clean(noisy: f64) -> f64 = gelu(noisy)\n";

    let artifact = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some("clean".to_string()),
    })
    .expect("a shadowed un-emitted sibling must not block the selected scalar entry");

    let emitted = artifact
        .compile_result
        .files
        .iter()
        .map(|file| file.contents.as_str())
        .collect::<String>();
    assert!(
        emitted.contains("chelis_host_gelu_f64"),
        "the selected scalar activation must be emitted:\n{emitted}"
    );
    assert!(
        !emitted.contains("dropout"),
        "the shadowed unselected sibling must be absent from the artifact:\n{emitted}"
    );
}

/// The Deep ingestion branch used to carry its own call sites to the
/// CLI-local policy. It must now reach the same typed compiler-api gate as
/// Surf rather than preserving a second stringly rejection path.
#[test]
fn deep_dropout_uses_the_shared_typed_effect_gate() {
    let source = include_str!("fixtures/phase3_seeded_dropout.dp");
    build_deep(source, "deep_dropout_c", "c").success();
    for target in ["hip", "metal"] {
        build_deep(source, &format!("deep_dropout_{target}"), target)
            .failure()
            .stderr(predicates::str::contains("unsupported:"))
            .stderr(predicates::str::contains("early capability gate"))
            .stderr(predicates::str::contains("unimplemented chelis#1192"));
    }
}

/// A host-lane cohabitant changes where the tensor def is stored, not the
/// target policy. The shared effect gate must inspect host tensor-helper DAGs
/// so this shape cannot fall through to an emitter panic.
#[test]
fn host_tensor_helper_dropout_uses_the_shared_typed_effect_gate() {
    let source = "def label() -> string = \"host\"\n\
                  def noisy(x: tensor[4, f32]) -> tensor[4, f32] = \
                  with seed(42i64) { dropout(x, 0.5) }\n";

    build(source, "host_helper_dropout", "c").success();
    for target in ["hip", "metal"] {
        build(source, &format!("host_helper_dropout_{target}"), target)
            .failure()
            .stderr(predicates::str::contains("unsupported:"))
            .stderr(predicates::str::contains("early capability gate"))
            .stderr(predicates::str::contains("unimplemented chelis#1192"));
    }
}
