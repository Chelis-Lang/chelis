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
    let desugared = desugar_program(&parsed);
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
        build_type_env_from_library(&desugar_program(library)).map_err(|report| report.errors)?;
    let encoded = bincode::serialize(&context).expect("checked context serializes");
    let restored: TypeEnv = bincode::deserialize(&encoded).expect("checked context restores");
    check_ir_with_context(&restored, &desugar_program(caller))
        .map(|_| ())
        .map_err(|report| report.errors)
}

fn accepts(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        if let Err(errors) = result {
            panic!(
                "{label} [{entry}] over-rejected:\n{}",
                rendered(&errors).join("\n")
            );
        }
    }
}

fn rejects(label: &str, source: &str, operation: &str) {
    for (entry, result) in whole_program(source) {
        let errors = result.unwrap_err();
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

#[test]
fn authored_generic_wrappers_must_declare_collection_constructors() {
    for (operation, source) in [
        ("len", "def size(x) = len(x)\n"),
        ("len", "def size(x) = len(x)\nout = size([1i32])\n"),
        ("len", "def size[a](x: a) -> int64 = len(x)\n"),
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
                .any(|error| error.message.contains("in `size` at declaration boundary")),
            "direct declaration diagnostic [{entry}] did not identify `size`:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

#[test]
fn explicit_list_and_dict_contracts_are_sufficient() {
    accepts(
        "explicit List length",
        "def size[a](xs: List[a]) -> int64 = len(xs)\nout = size([1i32])\n",
    );
    accepts(
        "explicit Dict length",
        "def size[k, v](xs: Dict[k, v]) -> int64 = len(xs)\n\
         keys: List[string] = [\"a\"]\n\
         values: List[int64] = [1i64]\n\
         out = size(dict_of(zip(keys, values)))\n",
    );
    accepts(
        "explicit collection relations",
        "def at[a](xs: List[a], i: int32) -> a = index(xs, i)\n\
         def push[a](xs: List[a], x: a) -> List[a] = append(xs, x)\n\
         def join[a](lhs: List[a], rhs: List[a]) -> List[a] = concat(lhs, rhs)\n\
         first = at([1i64], 0i32)\n\
         more = push([1i64], 2i64)\n\
         all = join([1i64], [2i64])\n",
    );
    rejects_type_mismatch(
        "checked user function type survives aliasing",
        "def size[a](xs: List[a]) -> int64 = len(xs)\n\
         measure = size\n\
         out = measure(1i64)\n",
    );
}

#[test]
fn local_first_application_remains_monomorphic() {
    accepts(
        "local list binding",
        "def use(xs: List[int64]) -> int64 = {\n\
         measure = fn (x) -> len(x)\n\
         measure(xs)\n\
         }\n",
    );
    rejects(
        "local scalar binding",
        "def use(x: int64) -> int64 = {\n\
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
        "measure = len\nout: int64 = measure([1i64])\n",
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
        ("index", "op = index\nout = op(1i64, 0i32)\n"),
        ("append", "op = append\nout = op([1i64], \"bad\")\n"),
        ("concat", "op = concat\nout = op([1i64], [1.0f32])\n"),
        ("concat", "op = concat\nout = op([1i64], 0i64)\n"),
    ] {
        rejects("transported relation", source, operation);
    }
    for source in [
        "op = index\nout: int64 = op([1i64], 0i32)\n",
        "op = append\nout: List[int64] = op([1i64], 2i64)\n",
        "op = concat\nout: List[int64] = op([1i64], [2i64])\n",
        "op = concat\nout = op([to_tensor([1.0f32])], 0i32)\n",
    ] {
        accepts("valid transported relation", source);
    }
    for source in [
        "op = index\nout: string = op([1i64], 0i32)\n",
        "op = append\nout: List[string] = op([1i64], 2i64)\n",
        "op = concat\nout: List[string] = op([1i64], [2i64])\n",
    ] {
        rejects_type_mismatch("transported result relation", source);
    }
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
}

#[test]
fn recursive_indirect_calls_keep_the_checked_contract() {
    accepts(
        "recursive indirect list call",
        "def invoke(n: int32, f: List[int64] -> int64, xs: List[int64]) -> int64 =\n\
           if n == 0i32 then f(xs) else invoke(n - 1i32, f, xs)\n\
         out = invoke(1i32, len, [1i64])\n",
    );
    rejects(
        "recursive indirect scalar call",
        "def invoke(n: int32, f: int64 -> int64, x: int64) -> int64 =\n\
           if n == 0i32 then f(x) else invoke(n - 1i32, f, x)\n\
         out = invoke(1i32, len, 1i64)\n",
        "len",
    );
}
