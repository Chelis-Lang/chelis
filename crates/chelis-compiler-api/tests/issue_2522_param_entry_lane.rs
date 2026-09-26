//! chelis#2522 item 3: a definition whose parameter the DAG entry ABI cannot
//! carry keeps its authored C function.
//!
//! The whole-program DAG entry takes each parameter as an input tensor, a
//! numeric or bool scalar as a rank-0 one. A constant body made a definition
//! DAG-lowerable whatever its parameters were, so a program whose only
//! definition was `def f(x: Named) -> i64 = 1i64` emitted only the zero-input
//! direct entry, and `f` with its parameter vanished. A tensor or scalar
//! parameter is still a DAG input: an unused one is dropped from the direct
//! entry exactly as an unused tensor is, and a used bool scalar keeps the
//! exact bool storage chelis#1294 locks.

use chelis_backend_c::GeneratedHeader;
use chelis_compiler_api::compiler::compile;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

const NAMED: &str = "type Named =\n  | Named(string, i64)\n";

/// Each parameter the DAG entry cannot carry, with its expected C spelling.
const PARAMS: &[(&str, &str)] = &[
    ("string", "chelis_string x"),
    ("Named", "chelis_adt* x"),
    ("List[i64]", "chelis_list* x"),
    ("tensor[3, i64]", "chelis_tensor* x"),
];

fn header(source: &str) -> GeneratedHeader {
    let artifact = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some("fixture".into()),
    })
    .unwrap_or_else(|error| panic!("{source}: {error:?}"));
    let header = artifact
        .files
        .iter()
        .find(|file| file.path == "fixture.h")
        .expect("generated C header");
    GeneratedHeader::parse(&header.contents).expect("sealed generated header")
}

fn authored_declaration(source: &str) -> Result<String, String> {
    header(source)
        .declaration("f")
        .map(|declaration| declaration.declaration().to_string())
        .ok_or_else(|| format!("no authored declaration for `f` in:\n{source}"))
}

/// REGRESSION TEST for the only-definition cases with a string, ADT or list
/// parameter; without the rule each emitted only the direct entry. The
/// non-f32 tensor parameter (an existing rule) and every beside-another-
/// definition case already had a host wrapper and are locks.
#[test]
fn a_constant_definition_keeps_every_parameter_the_dag_entry_cannot_carry() {
    let mut failures = Vec::new();
    for (ty, expected) in PARAMS {
        let body = format!("{NAMED}def f(x: {ty}) -> i64 = 1i64\n");
        for (shape, source) in [
            ("only definition", body.clone()),
            (
                "beside another definition",
                format!("{body}def g(y: i64) -> i64 = y\n"),
            ),
        ] {
            match authored_declaration(&source) {
                Ok(declaration) if declaration.contains(expected) => {}
                Ok(declaration) => failures.push(format!(
                    "{ty}, {shape}: declaration `{declaration}` lacks `{expected}`"
                )),
                Err(error) => failures.push(format!("{ty}, {shape}: {error}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The siblings that must not move: a parameter the DAG entry carries as an
/// input keeps the definition the four-argument direct entry. At `a884b5ddf`
/// the rule moved scalar parameters too, and the `chelis#1294` bool-scalar
/// kernel program lost its `main`.
#[test]
fn a_definition_whose_parameters_are_dag_inputs_stays_a_dag_entry() {
    let mut failures = Vec::new();
    for source in [
        "def f(x: tensor[3, f32]) -> tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])\n",
        "def f(x: i64) -> i64 = 1i64\n",
        "def f(s: bool) -> tensor[1, bool] = insert(scalar_to_tensor(s), 0, 1i64)\n",
        "def f(x: tensor[3, f32], k: f32) -> tensor[3, f32] = x\n",
    ] {
        let header = header(source);
        if header.declaration("f").is_some() {
            failures.push(format!("a host wrapper for a DAG entry:\n{source}"));
            continue;
        }
        match header.declaration("fixture") {
            Some(direct)
                if direct
                    .declaration()
                    .contains("chelis_tensor **inputs, int n_in") => {}
            other => failures.push(format!(
                "no four-argument direct entry ({:?}):\n{source}",
                other.map(|declaration| declaration.declaration().to_string())
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The rule leaves a definition that inherits Random on its lane: as the only
/// definition it is still a public tensor entry, which refuses the inherited
/// draw rather than drawing from seed zero.
#[test]
fn an_inherited_random_definition_is_still_refused_at_its_public_entry() {
    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: "def keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\n"
            .to_string(),
        target: CompileTarget::C,
        entry_name: Some("fixture".into()),
    })
    .expect_err("an inherited draw has no handler at a public entry");
    assert!(
        format!("{error:?}").contains("public tensor entry cannot receive inherited Random"),
        "{error:?}"
    );
}
