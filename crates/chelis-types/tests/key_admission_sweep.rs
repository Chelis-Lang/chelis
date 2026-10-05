//! chelis#2413: key admission is a closed allow-list ([04-LIN-9], spec/04
//! section 1.1; `chelis_types::key_admission`). A key-carrying value reaches
//! an operation only when the list admits it: a key primitive's key operand,
//! `drop`, a branch's join, construction and destructuring, a move, a
//! builtin case that routes each value to one consumer, or a parameter whose
//! declared type carries a key. Everything else refuses it.
//!
//! The sweeps are generated from the vocabulary, not hand-listed: every Deep
//! tag ([`DeepTag::ALL`], witnesses from a match with no wildcard arm) and
//! every builtin ([`BUILTINS`], arity from its scheme). The IR half of the
//! same list is swept over every graph operation in `chelis-ir`'s verifier
//! tests (`every_risc_op_admits_a_key_exactly_where_the_allow_list_does`).
//!
//! Evidentiary status: REGRESSION TESTS. At `f4eeca363` the tag sweep failed
//! on `realize` (accepted), and the builtin sweep's `realize`-free rows
//! passed; see the per-test notes.

use chelis_deep::DeepTag;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::key_admission::{
    KeyAdmission, KeyPrimitive, TagKeys, builtin_key_operand, tag_keys,
};
use chelis_types::types::Type;
use chelis_types::{
    BUILTINS, BuiltinSiblingCaseId, builtin_env, check_linearity, check_typed_program,
};

fn verdict(source: &str) -> Result<(), Vec<CheckError>> {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    let deep = desugar_program(&decls).expect("fixture desugars");
    let checked = check_typed_program(&deep).map_err(|result| result.errors)?;
    check_linearity(&checked).map(|_| ())
}

/// A diagnostic that refuses a key: the allow-list's refusal, a key reuse or
/// read ([04-LIN-9]), or the type checker's key rules, never an unrelated
/// mismatch in the witness.
fn refuses_a_key(error: &CheckError) -> bool {
    error.message.to_lowercase().contains("key")
        && !error.message.contains("declared signature")
        && matches!(
            error.kind,
            CheckErrorKind::KeyReuse
                | CheckErrorKind::PrecisionMismatch
                | CheckErrorKind::TypeMismatch
                | CheckErrorKind::UnsupportedTensorPrecision
        )
}

const HOLDER: &str = "type Holder =\n  | Holder { k: key, n: i64 }\n";

/// What one tag witness shows.
enum Shows {
    /// The key reaches the tag's operand through a program that checks.
    Key(String),
    /// The checker refuses the tag's form itself before any key rule runs;
    /// the string is a fragment of that diagnostic.
    FormRefused(String, &'static str),
}

/// One witness per way a key-carrying value can reach the tag's runtime
/// operands. Exhaustive, with no wildcard arm: a new tag is a compile error
/// here until it has witnesses. An empty list means no key-carrying value
/// can be the tag's operand, and the reason is stated.
fn tag_witnesses(tag: DeepTag) -> Vec<Shows> {
    let key = |source: &str| Shows::Key(source.to_string());
    let holder = |source: &str| Shows::Key(format!("{HOLDER}{source}"));
    match tag {
        // A user function's key parameter, and a constructor's key field.
        DeepTag::App => vec![
            key(
                "def g(k: key) -> tensor[2, key] = split_keys(k, 2i64)\n\ndef f(k: key) -> tensor[2, key] = g(k)\n",
            ),
            key("def f(k: key) -> Option[key] = Some(k)\n"),
        ],
        DeepTag::Pipe => vec![key(
            "def f(k: key) -> tensor[2, key] = k |> split_keys(2i64)\n",
        )],
        DeepTag::Let => vec![key("def f(k: key) -> key = {\n  j = k\n  j\n}\n")],
        DeepTag::Block => vec![key("def f(k: key) -> key = do { k }\n")],
        DeepTag::Fn => vec![key("def f(k: key) -> key = (fn (j: key) -> j)(k)\n")],
        DeepTag::If => vec![key(
            "def f(c: bool, k: key, j: key) -> key = if c then k else j\n",
        )],
        DeepTag::Match => vec![key(
            "def f(o: Option[key]) -> key = match o with {\n  | Some(k) => k\n  | None => key_from_seed(0i64)\n}\n",
        )],
        DeepTag::Record => vec![holder("def f(k: key) -> Holder = Holder { k, n: 1i64 }\n")],
        DeepTag::Access => vec![holder("def f(h: Holder) -> key = h.k\n")],
        DeepTag::Tuple => vec![key("def f(k: key) -> (key, i64) = (k, 1i64)\n")],
        DeepTag::TupleGet => vec![key(
            "def f(p: (key, i64)) -> key = {\n  (k, n) = p\n  k\n}\n",
        )],
        DeepTag::RecordUpdate => vec![holder(
            "def f(h: Holder, k: key) -> Holder = h with { k }\n",
        )],
        DeepTag::Borrow => vec![key("def f(ks: tensor[2, key]) -> i64 = numel(&ks)\n")],
        DeepTag::Copy => vec![key(
            "def f(ks: tensor[2, key]) -> tensor[2, key] = copy(ks)\n",
        )],
        DeepTag::Realize => vec![
            key("def f(k: key) -> key = realize(k)\n"),
            key(
                "def main() = (realize(key_from_seed(3i64)), realize(split_keys(key_from_seed(4i64), 2i64)))\n",
            ),
            key("def f(ks: List[key]) -> List[key] = map(fn (k: key) -> realize(k), ks)\n"),
        ],
        DeepTag::Cast => vec![key("def f(k: key) -> i64 = cast(k, i64)\n")],
        DeepTag::Par => vec![Shows::FormRefused(
            "def f(k: key, j: key) -> (key, key) = par { k; j }\n".to_string(),
            "`par` expression",
        )],
        DeepTag::Quote => vec![Shows::FormRefused(
            "def f(k: key) = quote(k)\n".to_string(),
            "Deep tag `quote`",
        )],
        DeepTag::Unquote => vec![Shows::FormRefused(
            "def f(k: key) = quote(unquote(k))\n".to_string(),
            "Deep tag `quote`",
        )],
        DeepTag::Splice => vec![Shows::FormRefused(
            "def f(ks: List[key]) = quote(splice(ks))\n".to_string(),
            "Deep tag `quote`",
        )],
        // A handler region's result moves out as a block's does.
        DeepTag::HandleEffect => vec![
            key("def f(k: key) -> key = with device(\"cpu\") { fold_in(k, 1i64) }\n"),
            key("def f(k: key) -> (key, key) = with device(\"cpu\") { split_key(k) }\n"),
        ],
        // A transform's own operand is a function, which carries no key
        // (spec/04 section 8.4.1). A key reaches the transform's application,
        // an `App` whose callee is the transform, through the target's key
        // parameters.
        DeepTag::Grad | DeepTag::Vmap | DeepTag::Jit => vec![],
        DeepTag::Var
        | DeepTag::Lit
        | DeepTag::Module
        | DeepTag::Import
        | DeepTag::ImportAll
        | DeepTag::Export
        | DeepTag::Def
        | DeepTag::Defsig
        | DeepTag::Deftype
        | DeepTag::Typealias
        | DeepTag::Variant
        | DeepTag::Field
        | DeepTag::Defdim
        | DeepTag::Arm
        | DeepTag::PatVar
        | DeepTag::PatLit
        | DeepTag::PatCtor
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatAs
        | DeepTag::TPrim
        | DeepTag::TFn
        | DeepTag::TTensor
        | DeepTag::TRef
        | DeepTag::TAdt
        | DeepTag::TVar
        | DeepTag::TUnit
        | DeepTag::TTuple
        | DeepTag::DName
        | DeepTag::DVar
        | DeepTag::DLit
        | DeepTag::DRank
        | DeepTag::Params
        | DeepTag::Bind
        | DeepTag::Kv
        | DeepTag::Effects
        | DeepTag::Resource => vec![],
    }
}

/// Every Deep tag answers a key-carrying operand as the allow-list says: an
/// admitting tag checks, and a refusing tag is rejected with a diagnostic
/// about the key that names the tag (or, where the checker refuses the form
/// itself first, with that refusal).
///
/// Evidentiary status: REGRESSION TEST. At `f4eeca363` the three `realize`
/// rows checked (the first two then failed in the evaluator's key rules and
/// in `chelis build`'s ownership lowering).
#[test]
fn every_deep_tag_answers_a_key_carrying_operand_as_the_allow_list_says() {
    let mut failures = Vec::new();
    let mut swept = 0;
    for tag in DeepTag::ALL {
        let witnesses = tag_witnesses(tag);
        let admits = matches!(tag_keys(tag), TagKeys::Admits(_) | TagKeys::ByCallee);
        if tag_keys(tag) == TagKeys::NoOperand && !witnesses.is_empty() {
            failures.push(format!(
                "{tag:?}: witnesses for a tag with no runtime operand"
            ));
        }
        for witness in witnesses {
            swept += 1;
            match witness {
                Shows::Key(source) => match (verdict(&source), admits) {
                    (Ok(()), true) => {}
                    (Ok(()), false) => failures.push(format!("{tag:?}: accepted\n{source}")),
                    (Err(errors), true) => {
                        failures.push(format!("{tag:?}: rejected {errors:?}\n{source}"))
                    }
                    (Err(errors), false) => {
                        let named = format!("`{}`", tag.as_str());
                        let about = errors.iter().any(|error| {
                            refuses_a_key(error)
                                && (error.message.contains(&named)
                                    || error.message.contains("cannot be borrowed")
                                    || error.message.contains("cannot be copied")
                                    || error.message.contains("`key` source"))
                        });
                        if !about {
                            failures.push(format!(
                                "{tag:?}: rejected for another reason {errors:?}\n{source}"
                            ));
                        }
                    }
                },
                Shows::FormRefused(source, fragment) => {
                    if admits {
                        failures.push(format!("{tag:?}: an admitting tag has a refused form"));
                    }
                    match verdict(&source) {
                        Err(errors) if errors.iter().any(|e| e.message.contains(fragment)) => {}
                        other => failures.push(format!(
                            "{tag:?}: expected the form's own refusal `{fragment}`, got {other:?}"
                        )),
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(swept >= 20, "the tag sweep ran only {swept} witnesses");
}

/// The allow-list's tags, pinned: the brief's list is construction and
/// destructuring, joins, moves (a handler region's result among them), and
/// applications (whose callee decides).
/// Every transform tag (`grad`, `vmap`, `jit`, `realize`, `cast`, `copy`,
/// `borrow`) refuses a key operand of its own.
#[test]
fn the_admitting_tags_are_exactly_moves_joins_aggregates_and_applications() {
    let admitting: Vec<&str> = DeepTag::ALL
        .into_iter()
        .filter(|tag| matches!(tag_keys(*tag), TagKeys::Admits(_) | TagKeys::ByCallee))
        .map(DeepTag::as_str)
        .collect();
    assert_eq!(
        admitting,
        [
            "fn",
            "app",
            "let",
            "match",
            "if",
            "record",
            "access",
            "pipe",
            "block",
            "tuple",
            "tuple-get",
            "record-update",
            "handle-effect"
        ]
    );
    for tag in [
        DeepTag::Grad,
        DeepTag::Vmap,
        DeepTag::Jit,
        DeepTag::Realize,
        DeepTag::Cast,
        DeepTag::Copy,
        DeepTag::Borrow,
    ] {
        assert!(
            matches!(tag_keys(tag), TagKeys::Refuses(_)),
            "{tag:?} must refuse a key operand of its own"
        );
    }
}

/// A key reaches a transform's application through its target's key
/// parameters: `grad(f)(k, ..)`, `vmap(f)(ks)` and `jit(f)(k)` check.
///
/// Evidentiary status: disposition lock (each checked at `f4eeca363`).
#[test]
fn a_transform_application_passes_keys_to_its_targets_key_parameters() {
    for source in [
        "def g(k: key, x: f32) -> f32 = {\n  _ = drop(k)\n  mul(x, x)\n}\n\ndef f(k: key) -> f32 = grad(g, wrt=x)(k, 1.0)\n",
        "def g(k: key) -> tensor[2, f32] = uniform_like(k, to_tensor([0.0, 0.0]), 0.0, 1.0)\n\ndef f(k: key) -> tensor[2, 2, f32] = vmap(g)(split_keys(k, 2i64))\n",
        "def g(k: key) -> tensor[2, key] = split_keys(k, 2i64)\n\ndef f(k: key) -> tensor[2, key] = jit(g)(k)\n",
    ] {
        verdict(source).unwrap_or_else(|errors| panic!("{errors:?}\n{source}"));
    }
}

/// The key-carrying argument shapes the builtin sweep feeds.
const SHAPES: [&str; 3] = ["key", "tensor[2, key]", "List[key]"];

/// No builtin checks with a key-carrying value at an operand the allow-list
/// does not admit. For every builtin, at its scheme's arity (or every arity
/// up to four when its scheme is not a function type), and for each shape,
/// the sweep calls it with every operand key-carrying; a program that checks
/// must have some sibling case (or its key primitive) that admits every
/// operand.
///
/// Evidentiary status: lock. At `f4eeca363` no row broke this; the tag
/// sweep and the `realize` rows carry the regression.
#[test]
fn no_builtin_accepts_a_key_at_an_operand_the_allow_list_does_not_admit() {
    let (env, _) = builtin_env();
    let mut failures = Vec::new();
    let mut swept = 0;
    for decl in BUILTINS {
        let arities: Vec<usize> = match env.lookup(decl.name).map(|scheme| &scheme.body) {
            Some(Type::Fn(params, _)) => vec![params.len()],
            _ => (1..=4).collect(),
        };
        let cases: Vec<BuiltinSiblingCaseId> = decl
            .capability
            .sibling_cases
            .iter()
            .map(|case| case.case)
            .collect();
        let admitted_everywhere = |arity: usize| {
            let all = |cases: &[BuiltinSiblingCaseId]| {
                (0..arity).all(|index| builtin_key_operand(decl.name, cases, index).is_ok())
            };
            all(&cases) || cases.iter().any(|case| all(std::slice::from_ref(case)))
        };
        for arity in arities {
            for shape in SHAPES {
                let params: Vec<String> = (0..arity).map(|i| format!("k{i}: {shape}")).collect();
                let args: Vec<String> = (0..arity).map(|i| format!("k{i}")).collect();
                let source = format!(
                    "def w({}) = {}({})\n",
                    params.join(", "),
                    decl.name,
                    args.join(", ")
                );
                swept += 1;
                if verdict(&source).is_ok() && !admitted_everywhere(arity) {
                    failures.push(format!("{}: accepted\n{source}", decl.name));
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(
        swept >= 3 * BUILTINS.len(),
        "the builtin sweep ran only {swept} programs"
    );
}

/// Every key primitive admits a key at its key operand and nowhere else, and
/// `drop` admits any key-carrying value; the routing cases' positives are
/// `key_builtin_cases`'s sweep.
///
/// Evidentiary status: the key-primitive positives lock existing acceptance;
/// the non-key-operand negatives lock existing refusals.
#[test]
fn every_key_primitive_admits_exactly_its_key_operand() {
    for primitive in KeyPrimitive::ALL {
        let (good, bad) = match primitive {
            KeyPrimitive::SplitKey => ("def w(k: key) -> (key, key) = split_key(k)\n", None),
            KeyPrimitive::SplitKeys => (
                "def w(k: key) -> tensor[2, key] = split_keys(k, 2i64)\n",
                Some("def w(k: key, j: key) -> tensor[2, key] = split_keys(k, j)\n"),
            ),
            KeyPrimitive::FoldIn => (
                "def w(k: key) -> key = fold_in(k, 1i64)\n",
                Some("def w(k: key, j: key) -> key = fold_in(k, j)\n"),
            ),
            KeyPrimitive::Dropout => (
                "def w(k: key, x: tensor[2, f32]) -> tensor[2, f32] = dropout(k, x, 0.5)\n",
                Some("def w(k: key, x: tensor[2, key]) -> tensor[2, key] = dropout(k, x, 0.5)\n"),
            ),
            KeyPrimitive::UniformLike => (
                "def w(k: key, x: tensor[2, f32]) -> tensor[2, f32] = uniform_like(k, x, 0.0, 1.0)\n",
                Some(
                    "def w(k: key, x: tensor[2, key]) -> tensor[2, f32] = uniform_like(k, x, 0.0, 1.0)\n",
                ),
            ),
        };
        assert_eq!(
            builtin_key_operand(primitive.builtin(), &[], KeyPrimitive::KEY_OPERAND),
            Ok(KeyAdmission::Primitive(primitive))
        );
        verdict(good).unwrap_or_else(|errors| panic!("{primitive:?}: {errors:?}\n{good}"));
        if let Some(bad) = bad {
            let errors = verdict(bad).expect_err(&format!("{primitive:?}: expected a rejection"));
            assert!(
                errors.iter().any(refuses_a_key),
                "{primitive:?}: rejected for another reason {errors:?}"
            );
        }
    }
    verdict("def w(k: key) = drop(k)\n").expect("drop consumes a key");
    verdict("def w(p: (key, tensor[2, key])) = drop(p)\n").expect("drop consumes a key holder");
}

/// A builtin named as a value is judged at the parameters of the function
/// type it is instantiated at, as a call is: `split_key` as a callback
/// checks, and a builtin value whose parameter admits no key is refused
/// there, with a diagnostic that names it. Only a builtin whose whole rule
/// travels with the value is a value at all ([04-INF-9], chelis#3149):
/// `drop` and `to_int` are applicable only by name, so naming either as a
/// value is refused before any key judgement.
///
/// Evidentiary status: the `split_key` positive locks existing acceptance;
/// the `test_assert_eq` negative is a REGRESSION TEST for the key judgement,
/// and the by-name refusals lock chelis#3149.
#[test]
fn a_builtin_named_as_a_value_admits_a_key_only_where_a_call_would() {
    verdict("def w(ks: List[key]) -> List[(key, key)] = map(split_key, ks)\n")
        .unwrap_or_else(|errors| panic!("{errors:?}"));
    let errors = verdict(
        "def w(c: bool, a: key, b: key) -> unit = \
         (if c then test_assert_eq else test_assert_eq)(a, b, \"same\")\n",
    )
    .expect_err("a key reaching `test_assert_eq` as a value is refused");
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::KeyReuse)
                && error.message.contains("`test_assert_eq`")
                && error.message.contains("key-carrying")
        }),
        "expected a key refusal naming `test_assert_eq`, got {errors:?}"
    );
    for (source, builtin) in [
        (
            "def w(ks: List[key]) -> List[unit] = map(drop, ks)\n",
            "drop",
        ),
        (
            "def w(ks: List[key]) -> List[i64] = map(to_int, ks)\n",
            "to_int",
        ),
        (
            "def w(c: bool, k: key) -> i64 = (if c then to_int else to_int)(k)\n",
            "to_int",
        ),
    ] {
        let errors = verdict(source).expect_err("a builtin applicable only by name is refused");
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::TypeMismatch)
                    && error
                        .message
                        .contains(&format!("builtin `{builtin}` is applicable only by name"))
            }),
            "expected a by-name refusal naming `{builtin}`, got {errors:?}\n{source}"
        );
    }
}

/// A key tensor's extent is not key material ([04-LIN-9]): `shape` and
/// `numel` admit a key-carrying operand at operand 0 as an extent read,
/// which is not a use, so the key is still consumed exactly once afterwards.
/// Their other operands admit none.
///
/// Evidentiary status: REGRESSION TEST. At `096daea8c` every positive was
/// refused with "`shape` does not admit a key-carrying operand at argument
/// 0" (or `numel`).
#[test]
fn an_extent_read_admits_a_key_tensor_and_leaves_the_key_live() {
    for builtin in ["shape", "numel"] {
        assert_eq!(
            builtin_key_operand(builtin, &[], 0),
            Ok(KeyAdmission::ExtentObservation)
        );
        assert!(builtin_key_operand(builtin, &[], 1).is_err(), "{builtin}");
    }
    for source in [
        "def w(ks: tensor[2, key]) -> (i64, tensor[2, key]) = {\n  n = shape(ks, 0i32)\n  (n, ks)\n}\n",
        "def w(ks: tensor[2, key]) -> (i64, tensor[2, key]) = {\n  n = numel(ks)\n  (n, ks)\n}\n",
        "def w(ks: tensor[2, key], xs: tensor[2, f32]) -> (i64, tensor[2, f32]) = {\n  n = shape(ks, 0i32)\n  (n, vmap(fn (j: key, v: tensor[f32]) -> uniform_like(j, v, 0.0, 1.0))(ks, xs))\n}\n",
    ] {
        verdict(source).unwrap_or_else(|errors| panic!("{errors:?}\n{source}"));
    }
}

/// Negative parity: an extent read is still a read, so it must precede the
/// key's one use, and it does not license a second use.
#[test]
fn an_extent_read_neither_follows_nor_repeats_the_keys_one_use() {
    let draw = "vmap(fn (j: key, v: tensor[f32]) -> uniform_like(j, v, 0.0, 1.0))";
    for (source, fragment) in [
        (
            format!(
                "def w(ks: tensor[2, key], xs: tensor[2, f32]) -> (tensor[2, f32], i64) = {{\n  ys = {draw}(ks, xs)\n  (ys, shape(ks, 0i32))\n}}\n"
            ),
            "the extent of key-carrying variable `ks` is read",
        ),
        (
            format!(
                "def w(ks: tensor[2, key], xs: tensor[2, f32]) = {{\n  n = numel(ks)\n  a = {draw}(ks, copy(xs))\n  (n, a, {draw}(ks, xs))\n}}\n"
            ),
            "key-carrying variable `ks` was already consumed",
        ),
    ] {
        let errors = verdict(&source).expect_err(&source);
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::KeyReuse) && error.message.contains(fragment)
            }),
            "expected a KeyReuse containing {fragment:?}, got {errors:?}\n{source}"
        );
    }
    let source = "def w(ks: tensor[2, key], k: key) -> i64 = {\n  _ = drop(ks)\n  shape(to_tensor([1.0]), k)\n}\n";
    let errors = verdict(source).expect_err(source);
    assert!(errors.iter().any(refuses_a_key), "{errors:?}\n{source}");
}
