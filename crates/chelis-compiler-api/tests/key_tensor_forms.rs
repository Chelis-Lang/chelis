//! [05-OP-69]..[05-OP-72]: tensor operations equal per-element scalar derivations.
mod key_reference;
mod ownership_support;

use chelis_compiler_api::compiler::{compile, eval_selected};
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, EvalRequest, SourceKind};
use std::collections::BTreeMap;

fn tensor(shape: &[usize], keys: &[u64]) -> String {
    if shape.is_empty() {
        return format!("key({:016x})", keys[0]);
    }
    format!(
        "tensor(shape={shape:?}, data=[{}])",
        keys.iter()
            .map(|k| format!("key({k:016x})"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[test]
fn tensor_key_forms_match_the_independent_scalar_reference_in_eval_and_c() {
    for (seeds, shape, values) in [
        ("-1i64", vec![], vec![-1i64]),
        ("scalar_to_tensor(-1i64)", vec![], vec![-1i64]),
        ("to_tensor([1i64, -1i64])", vec![2], vec![1, -1]),
        (
            "to_tensor([[9007199254740993i64, -1i64], [9223372036854775807i64, -9223372036854775808i64]])",
            vec![2, 2],
            vec![9007199254740993, -1, i64::MAX, i64::MIN],
        ),
        ("to_tensor(range(0i64, 0i64))", vec![0], vec![]),
        (
            "reshape(to_tensor(range(0i64, 0i64)), [2i64, 0i64])",
            vec![2, 0],
            vec![],
        ),
    ] {
        let source = format!(
            "def main() = {{\n seeds = {seeds}\n (left, right) = split_key(key_from_seed(seeds))\n (key_from_seed(seeds), left, right, fold_in(key_from_seed(seeds), seeds), split_keys(key_from_seed(seeds), 3i64))\n}}\n"
        );
        let mut sources = vec![source];
        if seeds != "-1i64" {
            sources.push(format!(
                "def derived[r](seeds: tensor[..r, i64]) -> (tensor[..r, key], tensor[..r, key], tensor[..r, key], tensor[..r, key], tensor[..r, 3, key]) = {{\n (left, right) = split_key(key_from_seed(seeds))\n (key_from_seed(seeds), left, right, fold_in(key_from_seed(seeds), seeds), split_keys(key_from_seed(seeds), 3i64))\n}}\ndef main() = derived({seeds})\n"
            ));
        }
        let keys = values.iter().map(|v| *v as u64).collect::<Vec<_>>();
        let left = keys
            .iter()
            .map(|k| key_reference::split(*k).0)
            .collect::<Vec<_>>();
        let right = keys
            .iter()
            .map(|k| key_reference::split(*k).1)
            .collect::<Vec<_>>();
        let folded = keys
            .iter()
            .zip(&values)
            .map(|(k, n)| key_reference::fold_in(*k, *n))
            .collect::<Vec<_>>();
        let children = keys
            .iter()
            .flat_map(|k| (0..3).map(move |n| key_reference::fold_in(*k, n)))
            .collect::<Vec<_>>();
        let mut child_shape = shape.clone();
        child_shape.push(3);
        let expected = [
            tensor(&shape, &keys),
            tensor(&shape, &left),
            tensor(&shape, &right),
            tensor(&shape, &folded),
            tensor(&child_shape, &children),
        ]
        .iter()
        .enumerate()
        .map(|(i, s)| format!("main.{i} = {s}"))
        .collect::<Vec<_>>();
        for source in sources {
            let result = eval_selected(
                EvalRequest {
                    source_kind: SourceKind::Surf,
                    source: source.clone(),
                    bindings: BTreeMap::new(),
                },
                &["main".into()],
            )
            .unwrap_or_else(|e| panic!("{source}\n{e:?}"));
            let actual = result
                .roots
                .iter()
                .map(|r| {
                    format!(
                        "{} = {}",
                        r.name.as_deref().unwrap(),
                        r.display.as_deref().unwrap()
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "eval {source}");
            let generated =
                ownership_support::emit(&source, &format!("key_tensor_forms: {source}"));
            let (ledger, stdout) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
            assert_eq!(stdout.lines().collect::<Vec<_>>(), expected, "C {source}");
        }
    }
}

#[test]
fn tensor_key_split_checks_runtime_count_before_allocating() {
    for count in [-1, 0, i64::MAX] {
        let source = format!(
            "def children(k: tensor[2, key], n: i64) -> (tensor[2, *, key], unit) = (split_keys(k, n), ())\ndef main() = children(key_from_seed(to_tensor([1i64, 2i64])), {count}i64)\n"
        );
        let result = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.clone(),
                bindings: BTreeMap::new(),
            },
            &["main".into()],
        );
        let generated = ownership_support::emit(&source, "key_tensor_count");
        if count < 0 {
            let error = result.expect_err("negative runtime count must trap");
            assert!(format!("{error:?}").contains("numeric trap: domain in split_keys at i64"));
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(
                stderr.contains("numeric trap: domain in split_keys at i64"),
                "{stderr}"
            );
        } else if count > 0 {
            let error = result.expect_err("overflowing result must reject before allocation");
            assert!(
                format!("{error:?}").contains("overflow in split_keys at i64"),
                "{error:?}"
            );
            let stderr = ownership_support::run_failure_stderr(&generated, "");
            assert!(
                stderr.contains("Overflow") || stderr.contains("overflow in split_keys at i64"),
                "{stderr}"
            );
        } else {
            let result = result.expect("zero count is empty");
            assert!(
                result
                    .roots
                    .iter()
                    .any(|r| r.display.as_deref() == Some("tensor(shape=[2, 0], data=[])")),
                "{result:?}"
            );
            let (ledger, stdout) = ownership_support::run_program(&generated);
            ownership_support::balanced(&ledger);
            assert!(stdout.contains("tensor(shape=[2, 0], data=[])"), "{stdout}");
        }
    }
}

#[test]
fn tensor_fold_rejects_runtime_shape_mismatch_without_broadcasting() {
    for source in [
        "def folded(k: tensor[*, key], n: tensor[*, i64]) -> (tensor[*, key], unit) = (fold_in(k, n), ())\ndef main() = folded(key_from_seed(to_tensor([1i64, 2i64])), to_tensor([3i64, 4i64, 5i64]))\n",
        "def folded(k: tensor[*, *, key], n: tensor[*, *, i64]) -> (tensor[*, *, key], unit) = (fold_in(k, n), ())\ndef main() = folded(key_from_seed(to_tensor([[1i64, 2i64]])), to_tensor([[3i64], [4i64]]))\n",
    ] {
        let error = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
                bindings: BTreeMap::new(),
            },
            &["main".into()],
        )
        .expect_err("different runtime extents must reject");
        assert!(
            format!("{error:?}").contains("numeric trap: domain in fold_in at i64"),
            "{error:?}"
        );
        let generated = ownership_support::emit(source, "key_tensor_fold_shape");
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        assert!(
            stderr.contains("numeric trap: domain in fold_in at i64"),
            "{stderr}"
        );
    }
}

/// #2709: expectations come from [05-RNG-2], independently of the alias
/// lowering and the implementations that both executable lanes call.
fn alias_expected(shape: &[usize], seeds: &[i64]) -> Vec<String> {
    let keys = seeds
        .iter()
        .map(|seed| key_reference::key_from_seed(*seed))
        .collect::<Vec<_>>();
    let left = keys
        .iter()
        .map(|key| key_reference::split(*key).0)
        .collect::<Vec<_>>();
    let right = keys
        .iter()
        .map(|key| key_reference::split(*key).1)
        .collect::<Vec<_>>();
    let folded = keys
        .iter()
        .zip(seeds)
        .map(|(key, seed)| key_reference::fold_in(*key, *seed))
        .collect::<Vec<_>>();
    let children = keys
        .iter()
        .flat_map(|key| (0..2).map(move |index| key_reference::fold_in(*key, index)))
        .collect::<Vec<_>>();
    let mut child_shape = shape.to_vec();
    child_shape.push(2);
    [
        tensor(shape, &keys),
        tensor(shape, &left),
        tensor(shape, &right),
        tensor(shape, &folded),
        tensor(&child_shape, &children),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, value)| format!("main.{index} = {value}"))
    .collect()
}

fn assert_alias_eval_c(source: &str, expected: &[String]) {
    let result = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .unwrap_or_else(|error| panic!("alias Eval failed: {source}\n{error:?}"));
    let actual = result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}",
                root.name.as_deref().unwrap(),
                root.display.as_deref().unwrap()
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "alias Eval: {source}");
    // Exercise ordinary entry selection as well as the authored-host route.
    // A host-only fix must not conceal the DAG builtin-as-value rejection.
    for generated in [
        ownership_support::emit_selected(source, "key-alias-selected"),
        ownership_support::emit(source, "key-alias-host"),
    ] {
        let (ledger, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&ledger);
        assert_eq!(
            stdout.lines().collect::<Vec<_>>(),
            expected.iter().map(String::as_str).collect::<Vec<_>>(),
            "alias C: {source}"
        );
    }
}

fn alias_program(seeds: &str, declarations: &str) -> String {
    format!(
        "def main() = {{\n{declarations}\n  seeds = {seeds}\n  (left, right) = halves(seed(seeds))\n  (seed(seeds), left, right, folded(seed(seeds), seeds), children(seed(seeds), 2i64))\n}}\n"
    )
}

/// [04-INF-9], [05-OP-69]..[05-OP-72]: usable aliases need executable
/// parity as well as checker acceptance.
#[test]
fn unannotated_key_builtin_aliases_execute_in_eval_and_c() {
    let declarations =
        "  seed = key_from_seed\n  halves = split_key\n  folded = fold_in\n  children = split_keys";
    for (seeds, shape, values) in [
        ("-1i64", vec![], vec![-1i64]),
        ("scalar_to_tensor(-1i64)", vec![], vec![-1i64]),
        ("to_tensor([1i64, -1i64])", vec![2], vec![1, -1]),
        (
            "to_tensor([[9007199254740993i64, -1i64], [9223372036854775807i64, -9223372036854775808i64]])",
            vec![2, 2],
            vec![9007199254740993, -1, i64::MAX, i64::MIN],
        ),
        ("to_tensor(range(0i64, 0i64))", vec![0], vec![]),
    ] {
        assert_alias_eval_c(
            &alias_program(seeds, declarations),
            &alias_expected(&shape, &values),
        );
    }
}

/// One closed alias chooses each checked call independently, including the
/// tensor rank-zero surface, which is distinct from a scalar key.
#[test]
fn one_unannotated_key_alias_selects_multiple_shapes_in_eval_and_c() {
    let source = "def main() = {\n  seed = key_from_seed\n  (seed(7i64), seed(scalar_to_tensor(8i64)), seed(to_tensor([1i64, 2i64])))\n}\n";
    assert_alias_eval_c(
        source,
        &[
            "main.0 = key(0000000000000007)".into(),
            "main.1 = key(0000000000000008)".into(),
            "main.2 = tensor(shape=[2], data=[key(0000000000000001), key(0000000000000002)])"
                .into(),
        ],
    );
}

/// Typed controls retain the concrete callback path beside independent
/// unannotated instantiation.
#[test]
fn concretely_typed_key_builtin_aliases_execute_in_eval_and_c() {
    for (seeds, shape, values, seed_ty, key_ty, children_ty) in [
        ("-1i64", vec![], vec![-1i64], "i64", "key", "tensor[2, key]"),
        (
            "to_tensor([1i64, -1i64])",
            vec![2],
            vec![1, -1],
            "tensor[2, i64]",
            "tensor[2, key]",
            "tensor[2, 2, key]",
        ),
    ] {
        let declarations = format!(
            "  seed: {seed_ty} -> {key_ty} = key_from_seed\n  halves: {key_ty} -> ({key_ty}, {key_ty}) = split_key\n  folded: {key_ty} -> {seed_ty} -> {key_ty} = fold_in\n  children: {key_ty} -> i64 -> {children_ty} = split_keys"
        );
        assert_alias_eval_c(
            &alias_program(seeds, &declarations),
            &alias_expected(&shape, &values),
        );
    }
}

/// [04-INF-9]: preserve the resolved builtin through a chain, aggregate,
/// and concrete higher-order parameter in executable lanes.
#[test]
fn transported_key_builtin_aliases_execute_in_eval_and_c() {
    let (left, right) = key_reference::split(key_reference::key_from_seed(-1));
    let expected = [
        format!("main.0 = key({left:016x})"),
        format!("main.1 = key({right:016x})"),
    ];
    for source in [
        "def main() = {\n  first = split_key\n  derive = first\n  derive(key_from_seed(-1i64))\n}\n",
        "def main() = {\n  ops = (split_key, fold_in)\n  derive = ops.0\n  derive(key_from_seed(-1i64))\n}\n",
        "def main() = {\n  derive = split_key\n  invoke = fn (f: key -> (key, key), k: key) -> f(k)\n  invoke(derive, key_from_seed(-1i64))\n}\n",
    ] {
        assert_alias_eval_c(source, &expected);
    }
}

/// A public aggregate cannot export unspecialized builtin identity through
/// a zero-payload callable witness. Local aggregates remain executable above.
#[test]
fn exported_aggregate_of_key_callables_rejects_before_public_c_abi() {
    let source = "def exported() = (split_key, fold_in)\ndef main() = {\n  ops = exported()\n  derive = ops.0\n  mix = ops.1\n  (derive(key_from_seed(7i64)), mix(key_from_seed(8i64), 1i64))\n}\n";
    let eval_error = eval_selected(
        EvalRequest {
            source_kind: SourceKind::Surf,
            source: source.to_string(),
            bindings: BTreeMap::new(),
        },
        &["main".into()],
    )
    .err()
    .expect("Eval must reject an exported unspecialized aggregate");
    assert!(
        format!("{eval_error:?}")
            .contains("builtin `split_key` is not supported by IR evaluation as a value"),
        "{eval_error:?}"
    );
    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some("key-aggregate-export".into()),
    })
    .err()
    .expect("an exported aggregate must not erase callable identity at the C ABI");
    assert!(
        format!("{error:?}")
            .contains("host type did not resolve before the code-generation boundary"),
        "{error:?}"
    );
}

/// A lexical function named like the builtin keeps the lexical meaning.
/// Choosing an operation from alias text would produce split(7) instead.
#[test]
fn shadowed_key_builtin_name_does_not_select_builtin_carrier() {
    let source = "def main() = {\n  split_key = fn (k: key) -> {\n    _ = drop(k)\n    (key_from_seed(1i64), key_from_seed(2i64))\n  }\n  split_key(key_from_seed(7i64))\n}\n";
    assert_alias_eval_c(
        source,
        &[
            "main.0 = key(0000000000000001)".into(),
            "main.1 = key(0000000000000002)".into(),
        ],
    );
}

/// Negative runtime twins: alias transport must not erase dynamic domain
/// checks once the checker admits the operand types.
#[test]
fn key_builtin_aliases_preserve_runtime_domain_rejections() {
    for (source, trap) in [
        (
            "def run(k: tensor[2, key], n: i64) -> (tensor[2, *, key], unit) = {\n  derive = split_keys\n  (derive(k, n), ())\n}\ndef main() = run(key_from_seed(to_tensor([1i64, 2i64])), -1i64)\n",
            "numeric trap: domain in split_keys at i64",
        ),
        (
            "def run(k: tensor[*, key], n: tensor[*, i64]) -> (tensor[*, key], unit) = {\n  derive = fold_in\n  (derive(k, n), ())\n}\ndef main() = run(key_from_seed(to_tensor([1i64, 2i64])), to_tensor([3i64]))\n",
            "numeric trap: domain in fold_in at i64",
        ),
    ] {
        let error = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: source.into(),
                bindings: BTreeMap::new(),
            },
            &["main".into()],
        )
        .expect_err("the aliased operation must trap at runtime");
        assert!(format!("{error:?}").contains(trap), "{source}\n{error:?}");
        let generated = ownership_support::emit(source, "key-alias-domain");
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        assert!(stderr.contains(trap), "{source}\n{stderr}");
    }
}
