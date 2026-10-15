//! OP33 public transform path: runtime-sized scans, their adjoints and batching.
mod ownership_support;
use chelis_compiler_api::compiler::{compile, eval};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;

// [04-ADT-4], spec/04 §4.7: a transform result still owes the outer
// declaration's claim, at its primitive producer rather than after return.
fn claimed_scan_gradient(size: usize, claim: usize) -> String {
    format!(
        "type Box[n] = | Box {{ v: tensor[n, f32] }}\n\
         def loss(x: tensor[*, f32]) -> tensor[f32] = sum(cumsum(x, 0i32), 0i32)\n\
         def produce(size: i64) -> tensor[*, f32] = grad(loss)(insert(scalar_to_tensor(0.0f32), 0i32, size))\n\
         def mk(size: i64) -> Box[{claim}] = Box {{ v: produce(size) }}\n\
         out = mk(string_len(\"{}\"))\n",
        "x".repeat(size)
    )
}

#[test]
fn transform_result_claim_accepts_matching_and_empty_axes() {
    for size in [0, 1, 3] {
        let source = claimed_scan_gradient(size, size);
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        })
        .unwrap();
        let displayed = result.roots[0].display.as_ref().unwrap();
        assert!(
            displayed.contains(&format!("shape=[{size}]")),
            "{displayed}"
        );
        let generated = ownership_support::emit(&source, "matching_transform_claim");
        let (ledger, compiled) = ownership_support::run_program(&generated);
        ownership_support::balanced(&ledger);
        assert!(compiled.contains(displayed), "{compiled}");
    }
}

#[test]
fn transform_result_claim_rejects_mismatching_and_empty_axes() {
    for size in [0, 5] {
        let source = claimed_scan_gradient(size, 3);
        let error = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        })
        .expect_err("outer literal result claim must survive grad");
        let error = format!("{error:?}");
        assert!(error.contains("numeric trap: domain in"), "{error}");
        assert!(error.contains("claimed = 3"), "{error}");
        let generated = ownership_support::emit(&source, "rejecting_transform_claim");
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        assert!(stderr.contains("numeric trap: domain in"), "{stderr}");
        assert!(stderr.contains("claimed = 3"), "{stderr}");
    }
}

#[test]
fn transform_claims_follow_direct_alias_callback_and_nested_results() {
    let preamble = "type Box[n] = | Box { v: tensor[n, f32] }\n\
        type Wrap[n] = | Wrap { box: Box[n] }\n\
        def loss(x: tensor[*, f32]) -> tensor[f32] = sum(cumsum(x, 0i32), 0i32)\n\
        def produce(size: i64) -> tensor[*, f32] = grad(loss)(insert(scalar_to_tensor(0.0f32), 0i32, size))\n";
    for definition in [
        "def mk(size: i64) -> tensor[3, f32] = produce(size)",
        "def mk(size: i64) -> Box[3] = {
 g = grad(loss)
 Box { v: g(insert(scalar_to_tensor(0.0f32), 0i32, size)) }
}",
        "def mk(size: i64) -> Wrap[3] = Wrap { box: Box { v: produce(size) } }",
        "def mk(size: i64) -> List[Box[3]] = map(fn (n: i64) -> Box { v: produce(n) }, [size])",
    ] {
        for (text, agrees) in [("xxx", true), ("xxxxx", false)] {
            let source = format!("{preamble}{definition}\nout = mk(string_len(\"{text}\"))\n");
            let result = eval(EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.clone(),
                bindings: BTreeMap::new(),
            });
            assert_eq!(result.is_ok(), agrees, "{source}\n{result:?}");
            let generated = ownership_support::emit(&source, "projected_transform_claim");
            if agrees {
                let (ledger, _) = ownership_support::run_program(&generated);
                ownership_support::balanced(&ledger);
            } else {
                let error = format!("{:?}", result.unwrap_err());
                assert!(error.contains("claimed = 3"), "{error}");
                let stderr = ownership_support::run_failure_stderr(&generated, "");
                assert!(stderr.contains("claimed = 3"), "{stderr}");
            }
        }
    }
}

#[test]
fn transform_claims_project_multi_target_gradient_roots() {
    for text in ["xxx", "xxxxx"] {
        let source = format!(
            "def loss(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[f32] = add(sum(cumsum(x, 0i32), 0i32), sum(cumsum(y, 0i32), 0i32))\n\
            def mk(size: i64) -> (tensor[3, f32], tensor[3, f32]) = grad(loss)(insert(scalar_to_tensor(0.0f32), 0i32, 3i64), insert(scalar_to_tensor(0.0f32), 0i32, size))\n\
            out = mk(string_len(\"{text}\"))\n"
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        });
        assert_eq!(result.is_ok(), text.len() == 3, "{source}\n{result:?}");
        let generated = ownership_support::emit(&source, "multi_root_transform_claim");
        if text.len() == 3 {
            let (ledger, _) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
        } else {
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(stderr.contains("claimed = 3"), "{stderr}");
        }
    }
}

#[test]
fn unclaimed_multi_target_gradient_keeps_its_native_helper_contract() {
    let source = "def loss(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[f32] = add(sum(cumsum(x, 0i32), 0i32), sum(cumsum(y, 0i32), 0i32))\n\
        out = grad(loss)(to_tensor([1.0f32]), to_tensor([1.0f32, 2.0f32]))\n";
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .unwrap();
    let generated = ownership_support::emit(source, "unclaimed_multi_gradient");
    let (ledger, output) = ownership_support::run_program(&generated);
    ownership_support::balanced(&ledger);
    assert_eq!(result.roots.len(), 2);
    assert!(output.contains("data=[2.0, 1.0]"), "{output}");
}

#[test]
fn transform_result_guard_runs_before_later_host_effects() {
    for size in [3, 5] {
        let source = claimed_scan_gradient(size, 3).replace(
            "def mk(size: i64) -> Box[3] = Box { v: produce(size) }",
            "def mk(size: i64) -> Box[3] ! { IO } = {\n _ = print(\"before\")\n value = Box { v: produce(size) }\n alias = value\n _ = print(\"after\")\n alias\n}",
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        });
        let generated = ownership_support::emit(&source, "transform_claim_effect_order");
        if size == 3 {
            assert_eq!(result.unwrap().transcript, ["before", "after"]);
            let (ledger, output) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
            assert!(output.starts_with("before\nafter\n"), "{output}");
        } else {
            let error = result.unwrap_err();
            assert_eq!(error.transcript, ["before"], "{error:?}");
            let (output, stderr) = ownership_support::run_failure_output(&generated, "");
            assert_eq!(output, "before\n");
            assert!(stderr.contains("claimed = 3"), "{stderr}");
        }
    }
}

#[test]
fn independent_earlier_trap_precedes_transform_result_guard() {
    for (prefix, expected) in [
        ("0", "numeric trap: domain in reshape at i64"),
        (
            "-9223372036854775808",
            "numeric trap: overflow in neg at i64",
        ),
    ] {
        let source = claimed_scan_gradient(5, 3).replace(
            "def mk(size: i64) -> Box[3] = Box { v: produce(size) }",
            &format!("def mk(size: i64) -> Box[3] ! {{ IO }} = {{\n _ = print(\"before\")\n _ = neg(scalar_to_tensor({prefix}i64))\n value = Box {{ v: produce(size) }}\n _ = print(\"after\")\n value\n}}"),
        );
        let error = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        })
        .unwrap_err();
        assert_eq!(error.transcript, ["before"], "{error:?}");
        assert!(format!("{error:?}").contains(expected), "{error:?}");
        let generated = ownership_support::emit(&source, "earlier_trap_transform_claim");
        let (output, stderr) = ownership_support::run_failure_output(&generated, "");
        assert_eq!(output, "before\n");
        assert!(stderr.contains(expected), "{stderr}");
    }
}

#[test]
fn differentiated_body_trap_precedes_outer_cotangent_claim() {
    for (index, expected) in [
        (0, "numeric trap: domain in reshape at i64"),
        (99, "numeric trap: domain in gather at i64"),
    ] {
        let source = claimed_scan_gradient(5, 3).replace(
            "= sum(cumsum(x, 0i32), 0i32)",
            &format!("= {{\n _ = gather(x, to_tensor([{index}i64]), 0i32)\n sum(cumsum(x, 0i32), 0i32)\n}}"),
        );
        let error = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        })
        .unwrap_err();
        assert!(format!("{error:?}").contains(expected), "{error:?}");
        let generated = ownership_support::emit(&source, "forward_trap_transform_claim");
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        assert!(stderr.contains(expected), "{stderr}");
    }
}

#[test]
fn named_transform_result_claims_remain_caller_local() {
    for size in [3, 5] {
        let source = format!(
            "def loss(x: tensor[*, f32]) -> tensor[f32] = sum(cumsum(x, 0i32), 0i32)\n\
            def mk[n](reference: tensor[n, f32], size: i64) -> tensor[n, f32] = grad(loss)(insert(scalar_to_tensor(0.0f32), 0i32, size))\n\
            first = mk(to_tensor([1.0f32, 2.0f32, 3.0f32]), string_len(\"xxx\"))\n\
            second = mk(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32]), string_len(\"{}\"))\n",
            "x".repeat(size)
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        });
        assert_eq!(result.is_ok(), size == 5, "{result:?}");
        let generated = ownership_support::emit(&source, "caller_local_transform_claim");
        if size == 5 {
            let (ledger, _) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
        } else {
            let error = format!("{:?}", result.unwrap_err());
            assert!(error.contains("reference axis 0 = 5"), "{error}");
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(stderr.contains("reference axis 0 = 5"), "{stderr}");
        }
    }
}

#[test]
fn transform_result_claim_projects_nominal_gradient_layout() {
    for size in [3, 5] {
        let source = format!(
            "type Box[n] = | Box {{ v: tensor[n, f32] }}\n\
            def loss[n](x: Box[n]) -> tensor[f32] = sum(cumsum(x.v, 0i32), 0i32)\n\
            def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\n\
            def mk(size: i64) -> Box[3] = grad(loss)(Box {{ v: produce(size) }})\n\
            out = mk(string_len(\"{}\"))\n",
            "x".repeat(size)
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        });
        assert_eq!(result.is_ok(), size == 3, "{source}\n{result:?}");
        let generated = ownership_support::emit(&source, "nominal_gradient_result_claim");
        if size == 3 {
            let (ledger, _) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
        } else {
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(stderr.contains("claimed = 3"), "{stderr}");
        }
    }
}

#[test]
fn transform_result_claim_projects_list_gradient_layout() {
    for size in [3, 5] {
        let source = format!(
            "def list_index[item](values: List[item], idx: i64) -> item = index(values, idx)\n\
            type Box[n] = | Box {{ v: tensor[n, f32] }}\n\
            type Model[n] = | Model {{ blocks: List[Box[n]] }}\n\
            def loss[n](x: Model[n]) -> tensor[f32] = sum(cumsum(list_index(x.blocks, 0i64).v, 0i32), 0i32)\n\
            def produce(size: i64) -> tensor[*, f32] = insert(scalar_to_tensor(0.0f32), 0i32, size)\n\
            def mk(size: i64) -> Model[3] = grad(loss)(Model {{ blocks: [Box {{ v: produce(size) }}] }})\n\
            out = mk(string_len(\"{}\"))\n",
            "x".repeat(size)
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        });
        assert_eq!(result.is_ok(), size == 3, "{source}\n{result:?}");
        let generated = ownership_support::emit(&source, "list_gradient_result_claim");
        if size == 3 {
            let (ledger, _) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
        } else {
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(stderr.contains("claimed = 3"), "{stderr}");
        }
    }
}

#[test]
fn original_file_sized_scan_gradient_preserves_outer_claim() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("size.txt");
    for size in [3, 5] {
        std::fs::write(&input, "x".repeat(size)).unwrap();
        let source = claimed_scan_gradient(size, 3).replace(
            &format!("string_len(\"{}\")", "x".repeat(size)),
            &format!("string_len(read_file({:?}))", input.to_str().unwrap()),
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        });
        assert_eq!(result.is_ok(), size == 3, "{result:?}");
        let generated = ownership_support::emit(&source, "original_file_scan_claim");
        if size == 3 {
            let (ledger, output) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
            assert!(output.contains("data=[3.0, 2.0, 1.0]"), "{output}");
        } else {
            let error = format!("{:?}", result.unwrap_err());
            assert!(
                error.contains("numeric trap: domain in reshape at i64"),
                "{error}"
            );
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(
                stderr.contains("numeric trap: domain in reshape at i64"),
                "{stderr}"
            );
        }
    }
}

#[test]
fn vmap_result_claim_uses_the_same_producer_schedule() {
    for size in [3, 5] {
        let source = format!(
            "def square(x: tensor[f32]) -> tensor[f32] = mul(x, x)\n\
            def mk(size: i64) -> tensor[3, f32] = vmap(square)(insert(scalar_to_tensor(2.0f32), 0i32, size))\n\
            out = mk(string_len(\"{}\"))\n",
            "x".repeat(size)
        );
        let result = eval(EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
            bindings: BTreeMap::new(),
        });
        assert_eq!(result.is_ok(), size == 3, "{source}\n{result:?}");
        let generated = ownership_support::emit(&source, "vmap_result_claim");
        if size == 3 {
            let (ledger, _) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
        } else {
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(stderr.contains("claimed = 3"), "{stderr}");
        }
    }
}

#[test]
fn runtime_scan_grad_and_vmap_execute_in_eval_and_c() {
    let source = include_str!("../../../examples/grad_cumsum.ch");
    let actual = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .unwrap();
    let displayed = actual
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().unwrap(),
                root.display.as_deref().unwrap()
            )
        })
        .collect::<String>();
    assert!(displayed.contains("data=[3.0, 2.0, 1.0]"), "{displayed}");
    assert!(displayed.contains("6.0, 4.0, 9.0, 15.0]"), "{displayed}");
    assert!(
        displayed.contains("data=[127, 128, 127, 2, 5, 9]"),
        "{displayed}"
    );
    let generated = ownership_support::emit(source, "scan_lanes");
    let (ledger, compiled) = ownership_support::run_program(&generated);
    ownership_support::balanced(&ledger);
    assert_eq!(compiled, displayed);
}

#[test]
fn scan_bool_and_integer_gradient_are_rejected_before_execution() {
    for source in [
        "module Demo.Main\nout = cumsum(to_tensor([true, false]), 0i32)\n",
        "module Demo.Main\ndef loss(x: tensor[3, i32]) -> tensor[i32] = sum(cumsum(x, 0i32), 0i32)\nout = grad(loss)(to_tensor([1i32, 2i32, 3i32]))\n",
    ] {
        assert!(
            eval(EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
                bindings: BTreeMap::new()
            })
            .is_err()
        );
        assert!(
            compile(CompileRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
                target: CompileTarget::C,
                entry_name: Some("scan_negative".into())
            })
            .is_err()
        );
    }
}
