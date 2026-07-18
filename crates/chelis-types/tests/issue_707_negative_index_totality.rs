//! Adversarial keeper (fresh-context red team, chelis#707 / PR #772).
//!
//! The tuple-projection index fix (`tuple_get_index` + `None => return
//! Type::Error`) introduces a SILENT `Type::Error` for a negative
//! bare-atom index. Pre-fix, `deep::Atom::Int(-1)` matched the bare-atom
//! arm and evaluated `-1i64 as usize` (a huge index), producing a LOUD
//! `TupleIndexOutOfBounds`. Post-fix, `tuple_get_index` returns `None`
//! (`usize::try_from(-1)` fails) and the sole caller does a bare
//! `return Type::Error` with NO diagnostic pushed, so `check_ir_program`
//! reports success with an EMPTY error vector.
//!
//! This is precisely the [04-TOT-2] pattern (an undiagnosed `Type::Error`
//! under an empty error vector) the PR claims to close. Reachability is
//! hand-authored Deep only — the Surf `.N` desugar never emits a negative,
//! float, or symbol projection index — so no Surf program hits it, but the
//! checker's totality invariant (chelis#731) is over all Deep.
//!
//! These tests assert a diagnostic IS produced. They FAIL against PR #772
//! as shipped (documenting the hole) and PASS once the `None` arm pushes a
//! real diagnostic (e.g. `TupleIndexOutOfBounds` / an "invalid tuple
//! index" `TypeMismatch`) instead of a bare `return Type::Error`.

use chelis_deep::parser::parse_str;
use chelis_types::check_ir_program;

fn errors_for(deep_src: &str) -> Vec<String> {
    let exprs = parse_str(deep_src).expect("deep parse");
    match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(rep) => rep.errors.iter().map(|e| e.message.clone()).collect(),
    }
}

const T2: &str =
    "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))";

#[test]
fn negative_bare_atom_index_is_diagnosed_not_silent() {
    let src = format!("{T2}\n(def {{}} x (tuple-get {{}} (var {{}} t) -1))");
    let errs = errors_for(&src);
    assert!(
        !errs.is_empty(),
        "negative bare-atom tuple index must produce a diagnostic, not a \
         silent Type::Error (regressed from a loud TupleIndexOutOfBounds \
         pre-fix); got an empty error vector"
    );
}

#[test]
fn negative_bare_atom_index_on_nontuple_is_diagnosed() {
    let src = "(def {} x (tuple-get {} (lit {type: (t-prim {} int32)} 5) -1))";
    let errs = errors_for(src);
    assert!(
        !errs.is_empty(),
        "negative bare-atom tuple index on a non-tuple must produce a \
         diagnostic (pre-fix: 'expected tuple type'); got an empty error \
         vector"
    );
}

#[test]
fn float_lit_node_index_is_diagnosed_not_silent() {
    let src = format!(
        "{T2}\n(def {{}} x (tuple-get {{}} (var {{}} t) (lit {{type: (t-prim {{}} f32)}} 1.5)))"
    );
    let errs = errors_for(&src);
    assert!(
        !errs.is_empty(),
        "float tuple index must produce a diagnostic, not a silent \
         Type::Error; got an empty error vector"
    );
}
