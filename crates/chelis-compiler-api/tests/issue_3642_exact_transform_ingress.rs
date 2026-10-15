//! spec/06 §2.7: discrete primals retain exact values despite unit cotangents.
use chelis_compiler_api::{
    compiler::eval,
    schema::{EvalRequest, ExecutionValue, SourceKind},
};

fn assert_true(source: String) {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.clone(),
        bindings: Default::default(),
    })
    .unwrap_or_else(|error| panic!("{source}: {error:?}"));
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("out"))
        .unwrap();
    assert!(
        matches!(root.value, ExecutionValue::Bool { value: true }),
        "{source}: {result:?}"
    );
}

#[test]
fn discrete_adt_primals_select_exact_branches() {
    for target in [9007199254740993i64, i64::MAX, i64::MIN] {
        for nested in [false, true] {
            let (ty, field, argument) = if nested {
                (
                    "List[Model]",
                    "index(p, 0i64)",
                    format!("[Model {{ scale: 3.0f32, target: {target}i64 }}]"),
                )
            } else {
                (
                    "Model",
                    "p",
                    format!("Model {{ scale: 3.0f32, target: {target}i64 }}"),
                )
            };
            // The adjacent value is deliberately indistinguishable after f64 rounding.
            let adjacent = if target == i64::MIN {
                target + 1
            } else {
                target - 1
            };
            for (comparison, expected) in [(target, "6.0f32"), (adjacent, "1.0f32")] {
                let result = if nested {
                    "index(g, 0i64).scale"
                } else {
                    "g.scale"
                };
                assert_true(format!(
                    "type Model = | Model {{ scale: f32, target: i64 }}\ndef loss(p: {ty}) -> f32 = {{ m = {field}\n if eq(m.target, {comparison}i64) then mul(m.scale, m.scale) else m.scale }}\ng = grad(loss)({argument})\nout = eq({result}, {expected})\n"
                ));
            }
        }
    }
}

#[test]
fn vmap_discrete_scalar_arguments_remain_exact() {
    for target in [9007199254740993i64, i64::MAX, i64::MIN] {
        let adjacent = if target == i64::MIN {
            target + 1
        } else {
            target - 1
        };
        for (comparison, expected) in [(target, "9.0f32"), (adjacent, "3.0f32")] {
            assert_true(format!(
                "def choose(x: tensor[f32], target: i64) -> tensor[f32] = if eq(target, {comparison}i64) then mul(x, x) else x\ny = vmap(choose)(to_tensor([3.0f32]), {target}i64)\nout = eq(tensor_to_scalar(sum(y, 0i32)), {expected})\n"
            ));
        }
    }
}

#[test]
fn grad_nontarget_integer_arguments_remain_exact() {
    for target in [9007199254740993i64, i64::MAX, i64::MIN] {
        let adjacent = if target == i64::MIN {
            target + 1
        } else {
            target - 1
        };
        for (comparison, expected) in [(target, "6.0f32"), (adjacent, "1.0f32")] {
            assert_true(format!(
                "def loss(x: f32, target: i64) -> f32 = if eq(target, {comparison}i64) then mul(x, x) else x\ng = grad(loss, wrt=x)(3.0f32, {target}i64)\nout = eq(g, {expected})\n"
            ));
        }
    }
}

#[test]
fn i64_max_adt_gradient_matches_executed_c() {
    use chelis_compiler_api::{
        compiler::compile,
        schema::{CompileRequest, CompileTarget},
    };
    use std::{fs, process::Command};
    let source = "type Model = | Model { scale: f32, target: i64 }\ndef loss(p: Model) -> f32 = if gt(p.target, 0i64) then mul(p.scale, p.scale) else p.scale\ndef step(p: Model) -> Model = grad(loss)(p)\nout = step(Model { scale: 3.0f32, target: 9223372036854775807i64 })\n";
    let evaluated = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    })
    .unwrap();
    let expected = evaluated
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("out"))
        .unwrap()
        .display
        .as_ref()
        .unwrap();
    assert_eq!(expected, "Model(6.0, ())");
    let compiled = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("ingress".into()),
    })
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    for file in &compiled.files {
        fs::write(dir.path().join(&file.path), &file.contents).unwrap();
    }
    let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
    let binary = dir.path().join("ingress");
    let mut cc = Command::new("cc");
    cc.arg(dir.path().join("ingress.c"))
        .arg("-I")
        .arg(dir.path())
        .arg(staged.archive)
        .args(["-lm", "-lpthread"]);
    if cfg!(target_os = "macos") {
        cc.args([
            "-framework",
            "Accelerate",
            "-framework",
            "Security",
            "-framework",
            "CoreFoundation",
        ]);
    } else {
        cc.arg("-ldl");
    }
    let output = cc.arg("-o").arg(&binary).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(binary).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!("out = {expected}\n")
    );
}
