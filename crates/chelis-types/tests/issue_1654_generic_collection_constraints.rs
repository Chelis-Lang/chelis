//! chelis#1654: collection admission is explicit at authored boundaries and
//! durable on already-checked function values.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::ast::Decl;
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{
    TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_typed_program,
};

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut messages = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>();
    messages.sort();
    messages
}

fn whole_program(source: &str) -> [(&'static str, Result<(), Vec<CheckError>>); 2] {
    let parsed = parse_str(source).expect("valid Surf");
    let desugared = desugar_program(&parsed).expect("Surf fixture must desugar");
    let expanded: Vec<Expr> = expand_program(&desugared, &ExpansionOptions::default())
        .expect("macro expansion")
        .into_exprs();
    [
        (
            "typed",
            check_typed_program(&desugared)
                .map(|_| ())
                .map_err(|report| report.errors),
        ),
        (
            "IR",
            check_ir_program(&expanded)
                .map(|_| ())
                .map_err(|report| report.errors),
        ),
    ]
}

fn serialized_context(source: &str) -> Result<(), Vec<CheckError>> {
    let declarations = parse_str(source).expect("valid Surf");
    let boundary = declarations
        .iter()
        .take_while(|decl| matches!(decl, Decl::FunDef { .. } | Decl::Sig { .. }))
        .count();
    let (library, caller) = declarations.split_at(boundary);
    let context =
        build_type_env_from_library(&desugar_program(library).expect("Surf fixture must desugar"))
            .map_err(|report| report.errors)?;
    let encoded = bincode::serialize(&context).expect("checked context serializes");
    let restored: TypeEnv = bincode::deserialize(&encoded).expect("checked context restores");
    check_ir_with_context(
        &restored,
        &desugar_program(caller).expect("Surf fixture must desugar"),
    )
    .map(|_| ())
    .map_err(|report| report.errors)
}

fn accepts(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        if let Err(errors) = result {
            panic!(
                "{label} [{entry}] over-rejected:\n{}\nsource:\n{source}",
                rendered(&errors).join("\n")
            );
        }
    }
}

fn rejects(label: &str, source: &str, operation: &str) {
    for (entry, result) in whole_program(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{entry}] unexpectedly accepted:\n{source}"),
        };
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
            "{label} [{entry}] expected TypeMismatch:\n{}",
            rendered(&errors).join("\n")
        );
        assert!(
            errors.iter().any(|error| error.message.contains(operation)),
            "{label} [{entry}] did not name `{operation}`:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

fn rejects_type_mismatch(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
            "{label} [{entry}] expected TypeMismatch:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

fn rejects_type_or_dimension(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{entry}] unexpectedly accepted"),
        };
        assert!(
            errors.iter().any(|error| {
                matches!(
                    error.kind,
                    CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
                )
            }),
            "{label} [{entry}] expected a type/dimension rejection:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

fn rejects_once(label: &str, source: &str, operation: &str) {
    for (entry, result) in whole_program(source) {
        let errors = result.unwrap_err();
        assert_eq!(
            errors.len(),
            1,
            "{label} [{entry}] expected one isolated rejection:\n{}",
            rendered(&errors).join("\n")
        );
        assert!(
            matches!(
                errors[0].kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
            ) && errors[0].message.contains(operation),
            "{label} [{entry}] did not retain the `{operation}` rejection:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

#[test]
fn authored_generic_wrappers_must_declare_collection_constructors() {
    for (operation, source) in [
        ("len", "def size(x) = len(x)\n"),
        ("len", "def size(x) = len(x)\nout = size([1i32])\n"),
        ("len", "def size[a](x: a) -> i64 = len(x)\n"),
        ("len", "measure = fn (x) -> len(x)\n"),
        ("len", "measure = len\ndef size(x) = measure(x)\n"),
        (
            "len",
            "def invoke(f, x) = f(x)\ndef size(x) = invoke(len, x)\n",
        ),
        ("index", "def at(xs, i) = index(xs, i)\n"),
        ("append", "def push(xs, x) = append(xs, x)\n"),
        ("concat", "def join(lhs, rhs) = concat(lhs, rhs)\n"),
    ] {
        rejects("insufficient collection contract", source, operation);
    }
    for (entry, result) in whole_program("def size(x) = len(x)\n") {
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message.contains("never determined within `size`")),
            "direct declaration diagnostic [{entry}] did not identify `size`:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

#[test]
fn explicit_list_and_dict_contracts_are_sufficient() {
    accepts(
        "explicit List length",
        "def size[a](xs: List[a]) -> i64 = len(xs)\nout = size([1i32])\n",
    );
    accepts(
        "explicit Dict length",
        "def size[k, v](xs: Dict[k, v]) -> i64 = len(xs)\n\
         keys: List[string] = [\"a\"]\n\
         values: List[i64] = [1i64]\n\
         out = size(dict_of(zip(keys, values)))\n",
    );
    accepts(
        "explicit collection relations",
        "def at[a](xs: List[a], i: i64) -> a = index(xs, i)\n\
         def push[a](xs: List[a], x: a) -> List[a] = append(xs, x)\n\
         def join[a](lhs: List[a], rhs: List[a]) -> List[a] = concat(lhs, rhs)\n\
         first = at([1i64], 0i64)\n\
         more = push([1i64], 2i64)\n\
         all = join([1i64], [2i64])\n",
    );
    rejects_type_mismatch(
        "checked user function type survives aliasing",
        "def size[a](xs: List[a]) -> i64 = len(xs)\n\
         measure = size\n\
         out = measure(1i64)\n",
    );
}

#[test]
fn local_first_application_remains_monomorphic() {
    accepts(
        "local list binding",
        "def use(xs: List[i64]) -> i64 = {\n\
         measure = fn (x) -> len(x)\n\
         measure(xs)\n\
         }\n",
    );
    rejects(
        "local scalar binding",
        "def use(x: i64) -> i64 = {\n\
         measure = fn (value) -> len(value)\n\
         measure(x)\n\
         }\n",
        "len",
    );
}

#[test]
fn builtin_contracts_survive_alias_return_aggregate_and_higher_order_passage() {
    for source in [
        "measure = len\nout = measure(1i64)\n",
        "def return_len() = len\nmeasure = return_len()\nout = measure(1i64)\n",
        "def pair() = (len, 1i32)\nmeasure = pair().0\nout = measure(1i64)\n",
        "def identity(f) = f\nmeasure = identity(len)\nout = measure(1i64)\n",
        "def invoke(f, x) = f(x)\nout = invoke(len, 1i64)\n",
        "def choose(flag: bool) = if flag then len else len\nmeasure = choose(true)\nout = measure(1i64)\n",
    ] {
        rejects("transported len contract", source, "len");
    }
    rejects_type_mismatch(
        "transported len result",
        "measure = len\nout: string = measure([1i64])\n",
    );
    for source in [
        "measure = len\nout = measure([1i64])\n",
        "measure = len\nout: i64 = measure([1i64])\n",
        "def identity(f) = f\nmeasure = identity(len)\nout = measure([1i64])\n",
        "def invoke(f, x) = f(x)\nout = invoke(len, [1i64])\n",
        "def choose(flag: bool) = if flag then len else len\nmeasure = choose(true)\nout = measure([1i64])\n",
        "def pair() = (len, 1i32)\nout = pair().1\n",
        "def ignore(f) = 1i32\nout = ignore(len)\n",
        "def ignore(f, x) = x\nout = ignore(len, 1i64)\n",
    ] {
        accepts("valid transported len contract", source);
    }
}

#[test]
fn each_collection_relation_survives_indirect_calls() {
    for (operation, source) in [
        ("index", "op = index\nout = op(1i64, 0i64)\n"),
        ("index", "op = index\nout = op([1i64], 0i32)\n"),
        ("index", "op = index\nout = op([1i64], 0i16)\n"),
        ("append", "op = append\nout = op([1i64], \"bad\")\n"),
        ("concat", "op = concat\nout = op([1i64], [1.0f32])\n"),
        ("concat", "op = concat\nout = op([1i64], 0i64)\n"),
    ] {
        rejects("transported relation", source, operation);
    }
    for source in [
        "op = index\nout: i64 = op([1i64], 0i64)\n",
        "op = append\nout: List[i64] = op([1i64], 2i64)\n",
        "op = concat\nout: List[i64] = op([1i64], [2i64])\n",
        "op = concat\nout = op([to_tensor([1.0f32])], 0i32)\n",
    ] {
        accepts("valid transported relation", source);
    }
    for source in [
        "op = index\nout: string = op([1i64], 0i64)\n",
        "op = append\nout: List[string] = op([1i64], 2i64)\n",
        "op = concat\nout: List[string] = op([1i64], [2i64])\n",
    ] {
        rejects_type_mismatch("transported result relation", source);
    }
}

#[test]
fn tensor_concat_transport_preserves_axis_and_exact_result_equations() {
    rejects_type_or_dimension(
        "direct tensor concat result",
        "def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = \
         concat([a, b], 1i32)\n",
    );
    for (route, source) in [
        (
            "alias",
            "def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = {\n\
             op = concat\n\
             op([a, b], 1i32)\n\
             }\n"
            .to_string(),
        ),
        (
            "higher-order return",
            "def return_concat() = concat\n\
             def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = {\n\
             op = return_concat()\n\
             op([a, b], 1i32)\n\
             }\n"
            .to_string(),
        ),
        (
            "higher-order passage",
            "def identity(f) = f\n\
             def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = {\n\
             op = identity(concat)\n\
             op([a, b], 1i32)\n\
             }\n"
            .to_string(),
        ),
        (
            "aggregate",
            "def pair() = (concat, 1i32)\n\
             def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = {\n\
             op = pair().0\n\
             op([a, b], 1i32)\n\
             }\n"
            .to_string(),
        ),
        (
            "recursive return",
            "def choose_concat(n: i32) \
             -> List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = \
             if n == 0i32 then concat else choose_concat(n - 1i32)\n\
             def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = {\n\
             op = choose_concat(1i32)\n\
             op([a, b], 1i32)\n\
             }\n"
            .to_string(),
        ),
        (
            "monomorphic annotated alias",
            "def bad(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 99, f32] = {\n\
             op: List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = concat\n\
             op([a, b], 1i32)\n\
             }\n"
            .to_string(),
        ),
        (
            "nested monomorphic function value",
            "def bad(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 99, f32] = {\n\
             op: List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = concat\n\
             nested = (op, 1i32).0\n\
             nested([a, b], 1i32)\n\
             }\n"
            .to_string(),
        ),
    ] {
        rejects_type_or_dimension(&format!("{route} tensor concat result"), &source);
    }

    for (route, source) in [
        (
            "direct",
            "def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = \
             concat([a, b], 9i32)\n"
                .to_string(),
        ),
        (
            "alias",
            "def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = {\n\
             op = concat\n\
             op([a, b], 9i32)\n\
             }\n"
            .to_string(),
        ),
        (
            "monomorphic annotated alias",
            "def bad(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 6, f32] = {\n\
             op: List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = concat\n\
             op([a, b], 9i32)\n\
             }\n"
            .to_string(),
        ),
    ] {
        rejects(&format!("{route} tensor concat axis"), &source, "concat");
    }

    for source in [
        "def good(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = {\n\
         op = concat\n\
         op([a, b], 1i32)\n\
         }\n",
        "def good(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = {\n\
         op = concat\n\
         op([a, b], -1i32)\n\
         }\n",
        "def good(a: tensor[2, 3, f32], b: tensor[4, 3, f32]) -> tensor[6, 3, f32] = {\n\
         op = concat\n\
         op([a, b], 0i32)\n\
         }\n",
        "def good(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 6, f32] = {\n\
         op: List[tensor[2, 3, f32]] -> i32 -> tensor[*, *, f32] = concat\n\
         op([a, b], 1i32)\n\
         }\n",
        "def dynamic(a: tensor[2, 3, f32], b: tensor[2, 3, f32], axis: i32) \
         -> tensor[*, *, f32] = {\n\
         op: List[tensor[2, 3, f32]] -> i32 -> tensor[*, *, f32] = concat\n\
         op([a, b], axis)\n\
         }\n",
        "def direct_dynamic(a: tensor[2, 3, f32], b: tensor[2, 3, f32], axis: i32) \
         -> tensor[*, *, f32] = concat([a, b], axis)\n",
        "def choose_concat(n: i32) \
         -> List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = \
         if n == 0i32 then concat else choose_concat(n - 1i32)\n\
         def good(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = {\n\
         op = choose_concat(1i32)\n\
         op([a, b], 1i32)\n\
         }\n",
    ] {
        accepts("valid transported tensor concat", source);
    }
}

#[test]
fn tensor_concat_call_evidence_is_owned_and_cleaned_up_per_application() {
    accepts(
        "one alias supports independent axis equations",
        "def both(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 6, f32] = {\n\
         op: List[tensor[2, 3, f32]] -> i32 -> tensor[*, *, f32] = concat\n\
         rows: tensor[4, 3, f32] = op([a, b], 0i32)\n\
         cols: tensor[2, 6, f32] = op([a, b], 1i32)\n\
         cols\n\
         }\n",
    );
    accepts(
        "independently specialized aliases do not share evidence",
        "def both(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 6, f32] = {\n\
         rows_op: List[tensor[2, 3, f32]] -> i32 -> tensor[*, 3, f32] = concat\n\
         cols_op: List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = concat\n\
         rows: tensor[4, 3, f32] = rows_op([a, b], 0i32)\n\
         cols: tensor[2, 6, f32] = cols_op([a, b], 1i32)\n\
         cols\n\
         }\n",
    );
    rejects_once(
        "a failed call does not poison the following valid call",
        "def mixed(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 6, f32] = {\n\
         op: List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = concat\n\
         bad = op([a, b], 9i32)\n\
         good: tensor[2, 6, f32] = op([a, b], 1i32)\n\
         good\n\
         }\n",
        "concat",
    );
}

#[test]
fn checked_contracts_survive_serialized_library_contexts() {
    for source in [
        "def exported() = len\nmeasure = exported()\nout = measure(1i64)\n",
        "def exported() = concat\njoin = exported()\nout = join([1i64], [1.0f32])\n",
    ] {
        let errors = serialized_context(source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
            "{}",
            rendered(&errors).join("\n")
        );
    }
    for source in [
        "def exported() = len\nmeasure = exported()\nout = measure([1i64])\n",
        "def exported() = concat\njoin = exported()\nout = join([1i64], [2i64])\n",
    ] {
        serialized_context(source)
            .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));
    }

    let invalid_tensor_concat = "def exported() = concat\n\
         join = exported()\n\
         def bad(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 99, f32] = \
         join([a, b], 1i32)\n";
    let errors = serialized_context(invalid_tensor_concat).unwrap_err();
    assert!(
        errors.iter().any(|error| {
            matches!(
                error.kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
            )
        }),
        "{}",
        rendered(&errors).join("\n")
    );
    serialized_context(
        "def exported() = concat\n\
         join = exported()\n\
         def good(a: tensor[2, 3, f32], b: tensor[2, 4, f32]) -> tensor[2, 7, f32] = \
         join([a, b], 1i32)\n",
    )
    .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));

    let invalid_monomorphic_tensor_concat = "def exported() = {\n\
         op: List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = concat\n\
         op\n\
         }\n\
         join = exported()\n\
         def bad(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 99, f32] = \
         join([a, b], 1i32)\n";
    let errors = serialized_context(invalid_monomorphic_tensor_concat).unwrap_err();
    assert!(
        errors.iter().any(|error| {
            matches!(
                error.kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::DimensionMismatch
            )
        }),
        "{}",
        rendered(&errors).join("\n")
    );
    serialized_context(
        "def exported() = {\n\
         op: List[tensor[2, 3, f32]] -> i32 -> tensor[2, *, f32] = concat\n\
         op\n\
         }\n\
         join = exported()\n\
         def good(a: tensor[2, 3, f32], b: tensor[2, 3, f32]) -> tensor[2, 6, f32] = \
         join([a, b], 1i32)\n",
    )
    .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));
}

#[test]
fn recursive_indirect_calls_keep_the_checked_contract() {
    accepts(
        "recursive indirect list call",
        "def invoke(n: i32, f: List[i64] -> i64, xs: List[i64]) -> i64 =\n\
           if n == 0i32 then f(xs) else invoke(n - 1i32, f, xs)\n\
         out = invoke(1i32, len, [1i64])\n",
    );
    rejects(
        "recursive indirect scalar call",
        "def invoke(n: i32, f: i64 -> i64, x: i64) -> i64 =\n\
           if n == 0i32 then f(x) else invoke(n - 1i32, f, x)\n\
         out = invoke(1i32, len, 1i64)\n",
        "len",
    );
}

#[test]
fn same_scc_returns_propagate_checked_collection_contracts() {
    let mutual_scalar = "def left(n) = if n == 0i32 then len else right(n - 1i32)\n\
         def right(n) = if n == 0i32 then left(0i32) else left(n - 1i32)\n\
         measure = right(1i32)\n\
         out = measure(1i64)\n";
    let mutual_list = "def left(n) = if n == 0i32 then len else right(n - 1i32)\n\
         def right(n) = if n == 0i32 then left(0i32) else left(n - 1i32)\n\
         measure = right(1i32)\n\
         out = measure([1i64])\n";
    rejects(
        "mutual recursion retains returned len contract",
        mutual_scalar,
        "len",
    );
    accepts("mutual recursion accepts matching List call", mutual_list);
    let errors = serialized_context(mutual_scalar).unwrap_err();
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::TypeMismatch) && error.message.contains("len")
        }),
        "{}",
        rendered(&errors).join("\n")
    );
    serialized_context(mutual_list)
        .unwrap_or_else(|errors| panic!("{}", rendered(&errors).join("\n")));

    rejects(
        "four-member recursion propagates returned len contract",
        "def first(n) = if n == 0i32 then len else fourth(n - 1i32)\n\
         def second(n) = if n == 0i32 then first(0i32) else first(n - 1i32)\n\
         def third(n) = if n == 0i32 then second(0i32) else second(n - 1i32)\n\
         def fourth(n) = if n == 0i32 then third(0i32) else third(n - 1i32)\n\
         measure = fourth(1i32)\n\
         out = measure(1i64)\n",
        "len",
    );
    accepts(
        "four-member recursion accepts matching List call",
        "def first(n) = if n == 0i32 then len else fourth(n - 1i32)\n\
         def second(n) = if n == 0i32 then first(0i32) else first(n - 1i32)\n\
         def third(n) = if n == 0i32 then second(0i32) else second(n - 1i32)\n\
         def fourth(n) = if n == 0i32 then third(0i32) else third(n - 1i32)\n\
         measure = fourth(1i32)\n\
         out = measure([1i64])\n",
    );
}
