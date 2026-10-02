//! chelis#2413 round 1: every Container and Boundary builtin case's key rule
//! ([`case_keys`]) is the checker's verdict on a key-carrying instantiation.
//!
//! The sweep is generated from the declaration table, not hand-listed: it
//! walks [`BUILTINS`], and for every sibling case it takes one witness program
//! per declared type parameter or refused operand from [`witnesses`], whose
//! match has no wildcard arm. A parameter routed to one consumer must check;
//! a borrowed one, one routed to the callback and the result, and a refused
//! operand must be rejected.
//!
//! [04-LIN-9], [04-LIN-10] and spec/04 section 1.1.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{
    BUILTINS, BuiltinSiblingCaseId, CaseKeys, KeyRouting, case_keys, check_linearity,
    check_typed_program,
};

fn verdict(source: &str) -> Result<(), Vec<CheckError>> {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    let deep = desugar_program(&decls).expect("fixture desugars");
    let checked = check_typed_program(&deep).map_err(|result| result.errors)?;
    check_linearity(&checked).map(|_| ())
}

const HOLDER: &str = "type Holder =\n  | Holder { k: key }\n";

/// One key-carrying witness per declared type parameter (for a
/// [`CaseKeys::Values`] case) or refused operand (for a
/// [`CaseKeys::Refused`] case), in declaration order. `None` marks a slot no
/// key-carrying type can reach. A [`CaseKeys::NoKeyOperand`] case has none.
fn witnesses(case: BuiltinSiblingCaseId) -> Vec<Option<String>> {
    use BuiltinSiblingCaseId as Case;
    let some = |source: &str| Some(source.to_string());
    match case {
        Case::FailString
        | Case::TestAssertBool
        | Case::ReadFile
        | Case::WriteFile
        | Case::ReadLines
        | Case::ReadBytes
        | Case::FileExists
        | Case::ListDir
        | Case::MmapFile
        | Case::MmapRead
        | Case::MmapLen
        | Case::ProcessRun
        | Case::ClockWallRead
        | Case::ClockMonotonicRead
        | Case::ParseCsv
        | Case::ToCsv
        | Case::CsvF64s
        | Case::CsvInts
        | Case::CsvStrs
        | Case::CsvNrows
        | Case::CsvCols
        | Case::CsvF64
        | Case::CsvInt
        | Case::CsvStr
        | Case::CharCode
        | Case::CharFromCode
        | Case::StringLen
        | Case::StringConcat
        | Case::StringSlice
        | Case::StringContains
        | Case::StringStartsWith
        | Case::StringEndsWith
        | Case::StringTrim
        | Case::ToInt
        | Case::ToFloat
        | Case::RangeList => vec![],
        Case::PrintRecursive => vec![some("def f(k: key) = print(k)\n")],
        Case::DebugRecursive => vec![some("def f(k: key) -> key = debug(k)\n")],
        Case::TestAssertEq => {
            let source = "def f(a: key, b: key) = test_assert_eq(a, b, \"same\")\n";
            vec![some(source), some(source)]
        }
        Case::TestAssertEqTensor => {
            let source = "def f(a: tensor[2, key], b: tensor[2, key]) = test_assert_eq_tensor(&a, \
                          &b, \"same\")\n";
            vec![some(source), some(source)]
        }
        Case::TestAssertCloseTensor => {
            let source = "def f(a: tensor[2, key], b: tensor[2, key]) = \
                          test_assert_close_tensor(&a, &b, 0.5f32, \"close\")\n";
            vec![some(source), some(source)]
        }
        Case::EqRecursive | Case::NeqRecursive => {
            let op = if case == Case::EqRecursive {
                "eq"
            } else {
                "neq"
            };
            let source = format!("def f(a: List[key], b: List[key]) -> bool = {op}(a, b)\n");
            vec![some(&source), some(&source)]
        }
        Case::ToStringUnit | Case::ToStringFunction => vec![None],
        Case::ToStringScalar => vec![some("def f(k: key) -> string = to_string(k)\n")],
        Case::ToStringTensor => vec![some(
            "def f(ks: tensor[2, key]) -> string = to_string(ks)\n",
        )],
        Case::ToStringList => vec![some("def f(ks: List[key]) -> string = to_string(ks)\n")],
        Case::ToStringTuple => vec![some("def f(p: (key, i64)) -> string = to_string(p)\n")],
        Case::ToStringDict => {
            vec![some(
                "def f(d: Dict[string, key]) -> string = to_string(d)\n",
            )]
        }
        Case::ToStringOption => vec![some("def f(o: Option[key]) -> string = to_string(o)\n")],
        Case::ToStringAdt => vec![Some(format!(
            "{HOLDER}def f(h: Holder) -> string = to_string(h)\n"
        ))],
        Case::LenList => vec![some("def f(ks: List[key]) -> i64 = len(ks)\n")],
        Case::LenDict => vec![some("def f(d: Dict[string, key]) -> i64 = len(d)\n")],
        Case::IndexList => vec![some("def f(ks: List[key]) -> key = index(ks, 0i64)\n")],
        Case::AppendList => {
            let source = "def f(ks: List[key], k: key) -> List[key] = append(ks, k)\n";
            vec![some(source), some(source)]
        }
        Case::ConcatList => {
            let source = "def f(a: List[key], b: List[key]) -> List[key] = concat(a, b)\n";
            vec![some(source), some(source)]
        }
        Case::TakeList => vec![some("def f(ks: List[key]) -> List[key] = take(ks, 1i64)\n")],
        Case::SkipList => vec![some("def f(ks: List[key]) -> List[key] = skip(ks, 1i64)\n")],
        Case::ChunkList => vec![some(
            "def f(ks: List[key]) -> List[List[key]] = chunk(ks, 1i64)\n",
        )],
        Case::FlattenList => vec![some(
            "def f(kss: List[List[key]]) -> List[key] = flatten(kss)\n",
        )],
        Case::EnumerateList => vec![some(
            "def f(ks: List[key]) -> List[(i64, key)] = enumerate(ks)\n",
        )],
        Case::ConcatTensors => vec![some(
            "def f(a: tensor[2, key], b: tensor[2, key]) -> tensor[4, key] = concat([a, b], \
             0i32)\n",
        )],
        Case::SplitTensor => vec![some(
            "def f(k: key) -> List[tensor[2, key]] = split(split_keys(k, 4i64), 0i32, [2i64, \
             2i64])\n",
        )],
        Case::ToTensorList => vec![some(
            "def f(a: key, b: key) -> tensor[2, key] = to_tensor([a, b])\n",
        )],
        Case::ToListTensor => vec![some(
            "def f(ks: tensor[2, key]) -> List[key] = to_list(ks)\n",
        )],
        Case::PadSequences => {
            let source = "def f(a: key, b: key) = pad_sequences([[a]], b)\n";
            vec![some(source), some(source)]
        }
        Case::PadSequencesTo => {
            let source = "def f(a: key, b: key) = pad_sequences_to([[a]], 2i64, b)\n";
            vec![some(source), some(source)]
        }
        Case::DropValue => vec![some("def f(k: key) = drop(k)\n")],
        Case::MapList => vec![
            some("def f(ks: List[key]) -> List[i64] = map(fn (j: key) -> 0i64, ks)\n"),
            some("def f(xs: List[i64]) -> List[key] = map(fn (n: i64) -> key_from_seed(n), xs)\n"),
        ],
        Case::FlatMapList => vec![
            some("def f(ks: List[key]) -> List[i64] = flat_map(fn (j: key) -> [0i64], ks)\n"),
            some(
                "def f(xs: List[i64]) -> List[key] = flat_map(fn (n: i64) -> [key_from_seed(n)], \
                 xs)\n",
            ),
        ],
        Case::FilterList => vec![some(
            "def f(ks: List[key]) -> List[key] = filter(fn (j: key) -> true, ks)\n",
        )],
        Case::PartitionList => vec![some(
            "def f(ks: List[key]) -> (List[key], List[key]) = partition(fn (j: key) -> true, \
             ks)\n",
        )],
        Case::FoldList => vec![
            some(
                "def f(k: key, xs: List[i64]) -> key = fold(fn (acc: key, n: i64) -> fold_in(acc, \
                 n), k, xs)\n",
            ),
            some("def f(ks: List[key]) -> i64 = fold(fn (acc: i64, j: key) -> acc, 0i64, ks)\n"),
        ],
        Case::ScanList => vec![
            some(
                "def f(k: key, xs: List[i64]) -> List[key] = scan(fn (acc: key, n: i64) -> \
                 fold_in(acc, n), k, xs)\n",
            ),
            some(
                "def f(ks: List[key]) -> List[i64] = scan(fn (acc: i64, j: key) -> acc, 0i64, \
                 ks)\n",
            ),
        ],
        Case::TensorScan => vec![some(
            "def main() = tensor_scan(key_from_seed(1i64), fn (acc: key, n: i64) -> fold_in(acc, \
             n), 3i64)\n",
        )],
        Case::ZipList => vec![
            some("def f(ks: List[key], xs: List[i64]) -> List[(key, i64)] = zip(ks, xs)\n"),
            some("def f(ks: List[key], xs: List[i64]) -> List[(i64, key)] = zip(xs, ks)\n"),
        ],
        Case::DictOf => vec![some(
            "def f(k: key) -> Dict[string, key] = dict_of([(\"x\", k)])\n",
        )],
        Case::DictGet => vec![some(
            "def f(d: Dict[string, key]) -> Option[key] = dict_get(d, \"x\")\n",
        )],
        Case::DictContains => vec![some(
            "def f(d: Dict[string, key]) -> bool = dict_contains(d, \"x\")\n",
        )],
        Case::DictRemove => vec![some(
            "def f(d: Dict[string, key]) -> Dict[string, key] = dict_remove(d, \"x\")\n",
        )],
        Case::DictKeys => vec![some(
            "def f(d: Dict[string, key]) -> List[string] = dict_keys(d)\n",
        )],
        Case::DictValues => vec![some(
            "def f(d: Dict[string, key]) -> List[key] = dict_values(d)\n",
        )],
        Case::DictEntries => vec![some(
            "def f(d: Dict[string, key]) -> List[(string, key)] = dict_entries(d)\n",
        )],
        Case::DictInsert => {
            let source = "def f(d: Dict[string, key], k: key) -> Dict[string, key] = \
                          dict_insert(d, \"y\", k)\n";
            vec![some(source), some(source)]
        }
        Case::DictMerge => {
            let source = "def f(a: Dict[string, key], b: Dict[string, key]) -> Dict[string, key] \
                          = dict_merge(a, b)\n";
            vec![some(source), some(source)]
        }
    }
}

/// What the checker must answer for one declared slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expected {
    Accepted,
    Rejected,
}

fn expected_slots(rule: CaseKeys) -> Vec<(String, Expected)> {
    match rule {
        CaseKeys::NoKeyOperand => vec![],
        CaseKeys::Refused(positions) => positions
            .iter()
            .map(|position| (format!("operand {position}"), Expected::Rejected))
            .collect(),
        CaseKeys::Values(parameters) => parameters
            .iter()
            .map(|parameter| {
                let expected = match parameter.routing {
                    KeyRouting::OneConsumer => Expected::Accepted,
                    KeyRouting::Borrowed | KeyRouting::CallbackAndResult => Expected::Rejected,
                };
                (
                    format!("parameter {} ({:?})", parameter.name, parameter.routing),
                    expected,
                )
            })
            .collect(),
    }
}

/// A diagnostic that refuses a key: a key reuse or borrow ([04-LIN-9]), or an
/// operation that does not admit one (spec/04 section 1.1), never an
/// unrelated mismatch in the witness.
fn refuses_a_key(error: &CheckError) -> bool {
    let about_key = error.message.to_lowercase().contains("key");
    let unrelated = error.message.contains("declared signature");
    about_key
        && !unrelated
        && matches!(
            error.kind,
            CheckErrorKind::KeyReuse
                | CheckErrorKind::PrecisionMismatch
                | CheckErrorKind::TypeMismatch
                | CheckErrorKind::UnsupportedTensorPrecision
        )
}

/// Evidentiary status: the filter, partition, scan and tensor_scan rows are
/// REGRESSION TESTS (each checked at `b47fdd7d3`), and so are the concat and
/// split tensor rows and the test_assert_eq rows; every other row locks the
/// verdict `b47fdd7d3` already gave.
#[test]
fn every_builtin_case_answers_a_key_carrying_instantiation_as_declared() {
    let mut failures = Vec::new();
    let mut swept = 0;
    for builtin in BUILTINS {
        for sibling in builtin.capability.sibling_cases {
            let rule = case_keys(sibling.case);
            let slots = expected_slots(rule);
            let programs = witnesses(sibling.case);
            if programs.len() != slots.len() {
                failures.push(format!(
                    "{}:{:?}: {} witnesses for {} declared slots",
                    builtin.name,
                    sibling.case,
                    programs.len(),
                    slots.len()
                ));
                continue;
            }
            for ((slot, expected), program) in slots.into_iter().zip(programs) {
                let Some(program) = program else {
                    continue;
                };
                swept += 1;
                let row = format!("{}:{:?} {slot}", builtin.name, sibling.case);
                match (verdict(&program), expected) {
                    (Ok(()), Expected::Accepted) => {}
                    // The rejection must be about the key, never an unrelated
                    // error in the witness.
                    (Err(errors), Expected::Rejected) => {
                        if !errors.iter().any(refuses_a_key) {
                            failures.push(format!(
                                "{row}: rejected for another reason {:?}\n{program}",
                                errors.iter().map(|e| &e.message).collect::<Vec<_>>()
                            ));
                        }
                    }
                    (Ok(()), Expected::Rejected) => {
                        failures.push(format!("{row}: accepted\n{program}"));
                    }
                    (Err(errors), Expected::Accepted) => failures.push(format!(
                        "{row}: rejected {:?}\n{program}",
                        errors.iter().map(|e| &e.message).collect::<Vec<_>>()
                    )),
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(swept >= 60, "the sweep ran only {swept} witnesses");
}

/// A builtin whose cases declare different key rules needs a selector in the
/// linearity checker; `concat` (list and tensor) is the only one. Any other
/// is checked against the union of its cases' refusals, so this test names a
/// new one before it is merely over-refused.
#[test]
fn every_builtin_with_differing_case_key_rules_has_a_selector() {
    let differing: Vec<&str> = BUILTINS
        .iter()
        .filter(|builtin| {
            let rules: Vec<CaseKeys> = builtin
                .capability
                .sibling_cases
                .iter()
                .map(|case| case_keys(case.case))
                .collect();
            rules.windows(2).any(|pair| pair[0] != pair[1])
        })
        .map(|builtin| builtin.name)
        .collect();
    assert_eq!(differing, ["concat"]);
}

/// P1-2: the higher-order builtins whose callback reads a value the result
/// also keeps refuse a key-carrying instantiation, with a diagnostic that
/// names the builtin and the parameter and suggests `split_keys`, never
/// `copy`. Named callbacks and lambdas alike.
///
/// Evidentiary status: REGRESSION TEST. At `b47fdd7d3` every negative checked.
#[test]
fn a_callback_builtin_never_hands_one_key_to_its_callback_and_its_result() {
    let keep = "def keep(j: key) -> bool = true\ndef step(acc: key, n: i64) -> key = fold_in(acc, \
                n)\n";
    for (builtin, parameter, source) in [
        (
            "filter",
            "T",
            "def bad(ks: List[key]) -> List[key] = filter(keep, ks)\n",
        ),
        (
            "filter",
            "T",
            "def bad(ks: List[key]) -> List[key] = filter(fn (j: key) -> keep(j), ks)\n",
        ),
        (
            "partition",
            "T",
            "def bad(ks: List[key]) -> (List[key], List[key]) = partition(keep, ks)\n",
        ),
        (
            "scan",
            "A",
            "def bad(k: key, xs: List[i64]) -> List[key] = scan(step, k, xs)\n",
        ),
        (
            "tensor_scan",
            "T",
            "def main() = tensor_scan(key_from_seed(1i64), step, 3i64)\n",
        ),
    ] {
        let source = format!("{keep}{source}");
        let errors = verdict(&source).expect_err(&format!("{builtin}: expected a rejection"));
        let error = errors
            .iter()
            .find(|error| matches!(error.kind, CheckErrorKind::KeyReuse))
            .unwrap_or_else(|| panic!("{builtin}: expected KeyReuse, got {errors:?}"));
        assert!(
            error.message.contains(&format!("`{builtin}`"))
                && error
                    .message
                    .contains(&format!("type parameter `{parameter}`"))
                && error.message.contains("[04-LIN-9]"),
            "{builtin}: the diagnostic must name the builtin and `{parameter}`: {error:?}"
        );
        let suggestions = error.suggestions.join(" ");
        assert!(
            suggestions.contains("split_keys") && !suggestions.contains("copy("),
            "{builtin}: the repair derives keys, never copies: {error:?}"
        );
    }
}

/// P2-4: key operand admission reads every builtin's registration, not only
/// the Numeric domain's: tensor `concat` ([05-OP-62]) and `split`
/// ([05-OP-53]) admit the nine data element dtypes, never `key`, while list
/// `concat` ([05-OP-54]) moves keys as values.
///
/// Evidentiary status: REGRESSION TEST for the two tensor negatives, which
/// checked at `b47fdd7d3`; the list twin locks existing acceptance.
#[test]
fn tensor_concat_and_split_refuse_key_tensors_and_list_concat_moves_keys() {
    for (builtin, source) in [
        (
            "concat",
            "def bad(k: key) -> tensor[4, key] = {\n  (a, b) = split_key(k)\n  \
             concat([split_keys(a, 2i64), split_keys(b, 2i64)], 0i32)\n}\n",
        ),
        (
            "split",
            "def bad(k: key) -> List[tensor[2, key]] = split(split_keys(k, 4i64), 0i32, [2i64, \
             2i64])\n",
        ),
    ] {
        let errors = verdict(source).expect_err(&format!("{builtin}: expected a rejection"));
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                    && error.message.contains(&format!(
                        "`{builtin}` does not admit a key-carrying operand at argument 0"
                    ))
                    && error.message.contains("section 1.1")
            }),
            "{builtin}: expected the key admission diagnostic, got {errors:?}"
        );
    }
    verdict("def good(k: key) -> List[key] = {\n  (a, b) = split_key(k)\n  concat([a], [b])\n}\n")
        .unwrap_or_else(|errors| panic!("list concat of keys: {errors:?}"));
    verdict(
        "def good(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[4, f32] = concat([x, y], 0i32)\n",
    )
    .unwrap_or_else(|errors| panic!("tensor concat of floats: {errors:?}"));
}
