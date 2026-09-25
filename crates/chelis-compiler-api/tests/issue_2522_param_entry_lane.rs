//! chelis#2522 item 3: a definition whose parameter the DAG entry ABI cannot
//! carry keeps its authored C function.
//!
//! The whole-program DAG entry takes only f32 and bool tensors. A constant
//! body made a definition DAG-lowerable whatever its parameters were, so a
//! program whose only definition was `def f(x: i64) -> i64 = 1i64` emitted
//! only the zero-input direct entry, and `f` with its parameter vanished. An
//! f32-tensor definition with a constant body is still a DAG entry.

use chelis_backend_c::GeneratedHeader;
use chelis_compiler_api::compiler::compile;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

const NAMED: &str = "type Named =\n  | Named(string, i64)\n";

/// Each parameter the DAG entry cannot carry, with its expected C spelling.
const PARAMS: &[(&str, &str)] = &[
    ("i64", "int64_t x"),
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

/// REGRESSION TEST for the only-definition cases with an i64, string, ADT or
/// list parameter; without the rule each emitted only the direct entry. The
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

/// The sibling that must not move: an f32-tensor parameter is carried by the
/// DAG entry, so the constant definition stays the four-argument direct entry.
#[test]
fn a_constant_f32_tensor_definition_stays_a_dag_entry() {
    let header =
        header("def f(x: tensor[3, f32]) -> tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])\n");
    assert!(
        header.declaration("f").is_none(),
        "no host wrapper for a DAG entry"
    );
    let direct = header
        .declaration("fixture")
        .expect("the direct DAG entry")
        .declaration()
        .to_string();
    assert!(
        direct.contains("chelis_tensor **inputs, int n_in"),
        "{direct}"
    );
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
