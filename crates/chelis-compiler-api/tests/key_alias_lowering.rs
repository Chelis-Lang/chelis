//! [04-INF-9]: checked key-builtin values retain their identity when applied.
mod key_reference;
mod ownership_support;

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::pipeline::{
    LoweringMode, PipelineGoal, PipelineOutcome, PipelineRequest, run_source,
};
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use chelis_ir::dag::RiscOp;
use std::collections::BTreeMap;

fn eval_main(source: &str) -> Result<Vec<String>, String> {
    eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .map(|result| {
        result
            .roots
            .iter()
            .map(|root| root.display.clone().expect("display"))
            .collect()
    })
    .map_err(|error| format!("{error:?}"))
}

#[test]
fn typed_scalar_key_builtin_aliases_execute_in_eval_and_c() {
    let source = "def main() = {\n  seed: i64 -> key = key_from_seed\n  fork: key -> (key, key) = split_key\n  children: key -> i64 -> tensor[2, key] = split_keys\n  mix: key -> i64 -> key = fold_in\n  (left, right) = fork(seed(7i64))\n  (left, right, mix(seed(7i64), -1i64), children(seed(7i64), 2i64))\n}\n";
    let (left, right) = key_reference::split(7);
    let expected = [
        format!("key({left:016x})"),
        format!("key({right:016x})"),
        format!("key({:016x})", key_reference::fold_in(7, -1)),
        format!(
            "tensor(shape=[2], data=[key({:016x}), key({:016x})])",
            key_reference::fold_in(7, 0),
            key_reference::fold_in(7, 1)
        ),
    ];
    assert_eq!(eval_main(source).unwrap(), expected, "Eval");
    let generated = ownership_support::emit(source, "typed scalar key aliases");
    let (ledger, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&ledger);
    for (index, display) in expected.iter().enumerate() {
        assert!(
            stdout.contains(&format!("main.{index} = {display}")),
            "{stdout}"
        );
    }
}

#[test]
fn typed_key_alias_rejects_wrong_input_before_lowering() {
    let source = "def main() = {\n  fork: key -> (key, key) = split_key\n  fork(7i64)\n}\n";
    let error = eval_main(source).expect_err("an i64 is not a key");
    assert!(
        error.contains("PrecisionMismatch") || error.contains("precision mismatch"),
        "{error}"
    );
}

#[test]
fn typed_tensor_key_builtin_aliases_execute_in_eval_and_c() {
    let source = "def main() = {\n  seed: tensor[2, i64] -> tensor[2, key] = key_from_seed\n  fork: tensor[2, key] -> (tensor[2, key], tensor[2, key]) = split_key\n  children: tensor[2, key] -> i64 -> tensor[2, 3, key] = split_keys\n  mix: tensor[2, key] -> tensor[2, i64] -> tensor[2, key] = fold_in\n  seeds = to_tensor([1i64, -1i64])\n  (left, right) = fork(seed(seeds))\n  (left, right, mix(seed(seeds), seeds), children(seed(seeds), 3i64))\n}\n";
    let seeds = [1u64, u64::MAX];
    let left = seeds
        .iter()
        .map(|key| key_reference::split(*key).0)
        .collect::<Vec<_>>();
    let right = seeds
        .iter()
        .map(|key| key_reference::split(*key).1)
        .collect::<Vec<_>>();
    let folded = [
        key_reference::fold_in(1, 1),
        key_reference::fold_in(u64::MAX, -1),
    ];
    let children = seeds
        .iter()
        .flat_map(|key| (0..3).map(move |index| key_reference::fold_in(*key, index)))
        .collect::<Vec<_>>();
    let tensor = |shape: &str, keys: &[u64]| {
        format!(
            "tensor(shape={shape}, data=[{}])",
            keys.iter()
                .map(|key| format!("key({key:016x})"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let expected = [
        tensor("[2]", &left),
        tensor("[2]", &right),
        tensor("[2]", &folded),
        tensor("[2, 3]", &children),
    ];
    assert_eq!(eval_main(source).unwrap(), expected, "Eval");
    let generated = ownership_support::emit(source, "typed tensor key aliases");
    let (ledger, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&ledger);
    for (index, display) in expected.iter().enumerate() {
        assert!(
            stdout.contains(&format!("main.{index} = {display}")),
            "{stdout}"
        );
    }
}

#[test]
fn typed_key_builtin_alias_survives_higher_order_passage() {
    let source = "def apply(f: i64 -> key, seed: i64) -> key = f(seed)\ndef main() = {\n  make: i64 -> key = key_from_seed\n  apply(make, -1i64)\n}\n";
    let expected = format!("key({:016x})", u64::MAX);
    assert_eq!(eval_main(source).unwrap(), [expected.clone()], "Eval");
    let generated = ownership_support::emit(source, "key alias callback");
    let (ledger, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&ledger);
    assert!(stdout.contains(&format!("main = {expected}")), "{stdout}");
}

#[test]
fn typed_key_builtin_alias_chain_retains_original_identity() {
    let source = "def main() = {\n  first: i64 -> key = key_from_seed\n  second: i64 -> key = first\n  second(-1i64)\n}\n";
    let expected = format!("key({:016x})", u64::MAX);
    assert_eq!(eval_main(source).unwrap(), [expected.clone()], "Eval");
    let generated = ownership_support::emit(source, "key alias chain");
    let (ledger, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&ledger);
    assert!(stdout.contains(&format!("main = {expected}")), "{stdout}");
}

#[test]
fn unannotated_alias_selects_scalar_and_tensor_independently_in_eval() {
    let source = "def main() = {\n  seed = key_from_seed\n  (seed(-1i64), seed(to_tensor([1i64, 2i64])))\n}\n";
    assert_eq!(
        eval_main(source).unwrap(),
        [
            format!("key({:016x})", u64::MAX),
            "tensor(shape=[2], data=[key(0000000000000001), key(0000000000000002)])".to_string(),
        ]
    );
}

#[test]
fn typed_tensor_fold_alias_rejects_runtime_shape_mismatch() {
    let source = "def folded(k: tensor[*, key], n: tensor[*, i64]) -> (tensor[*, key], unit) = {\n  combine: tensor[*, key] -> tensor[*, i64] -> tensor[*, key] = fold_in\n  (combine(k, n), ())\n}\ndef main() = folded(key_from_seed(to_tensor([1i64, 2i64])), to_tensor([3i64, 4i64, 5i64]))\n";
    let error = eval_main(source).expect_err("unequal runtime shapes");
    assert!(
        error.contains("numeric trap: domain in fold_in at i64"),
        "{error}"
    );
    let generated = ownership_support::emit(source, "fold alias shape mismatch");
    let stderr = ownership_support::run_failure_stderr(&generated, "");
    assert!(
        stderr.contains("numeric trap: domain in fold_in at i64"),
        "{stderr}"
    );
}

#[test]
fn typed_key_aliases_lower_to_their_dag_primitives() {
    let source = "def main() = {\n  seed: tensor[2, i64] -> tensor[2, key] = key_from_seed\n  fork: tensor[2, key] -> (tensor[2, key], tensor[2, key]) = split_key\n  children: tensor[2, key] -> i64 -> tensor[2, 3, key] = split_keys\n  mix: tensor[2, key] -> tensor[2, i64] -> tensor[2, key] = fold_in\n  xs = to_tensor([1i64, 2i64])\n  (left, right) = fork(seed(xs))\n  (left, right, children(seed(xs), 3i64), mix(seed(xs), xs))\n}\n";
    let outcome = run_source(PipelineRequest {
        source_kind: SourceKind::Surf,
        source,
        entry: None,
        goal: PipelineGoal::Lower(LoweringMode::Strict),
    })
    .expect("checked alias program lowers");
    let PipelineOutcome::Lowered(lowered) = outcome else {
        panic!("lower goal returns DAG");
    };
    let ops = lowered
        .dag()
        .nodes()
        .iter()
        .map(|node| &node.op)
        .collect::<Vec<_>>();
    assert!(
        ops.iter().any(|op| matches!(op, RiscOp::KeyFromSeed)),
        "{ops:?}"
    );
    assert_eq!(
        ops.iter()
            .filter(|op| matches!(op, RiscOp::Split { .. }))
            .count(),
        2,
        "{ops:?}"
    );
    assert!(
        ops.iter().any(|op| matches!(op, RiscOp::SplitN { .. })),
        "{ops:?}"
    );
    assert!(ops.iter().any(|op| matches!(op, RiscOp::FoldIn)), "{ops:?}");
}
