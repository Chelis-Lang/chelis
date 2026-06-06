//! Issue Chelis-Lang/chelis#319 (checker half) — a `def` declared with a
//! SEPARATE `sig` (no inline parameter annotations) must have its BODY
//! annotated with the resolved tensor types implied by that sig, exactly
//! as an inline-annotated `def` already is.
//!
//! Before the fix `annotate_fn_children` inferred the bare `fn` literal's
//! body against fresh unconstrained parameter tvars (the params carry no
//! inline annotation — their types live only in the separate `sig`), so a
//! shape-sensitive body op such as `permute`/`matmul` was stamped with a
//! bare `(t-var …)` type rather than a `(t-tensor …)`. IR lowering then
//! read a rank-0 `default_type()` off that annotation and
//! `tier2::lower_matmul` panicked with `expects rank >= 2`. The fix seeds
//! the body-inference scope with the declared sig parameter types.
//!
//! Spec authority: spec/04-type-system.md §5.7 / §5.8.

use chelis_deep::printer::print_canonical;
use chelis_deep::{Atom, Expr};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::check_ir_program;

/// True when `expr` is the checked `(def {…} <def_name> (fn …))` node.
/// The def's name is its first non-meta child symbol.
fn is_named_def(expr: &Expr, def_name: &str) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first() else {
        return false;
    };
    if tag != "def" {
        return false;
    }
    // elements[1] = meta map, elements[2] = name symbol.
    matches!(list.elements.get(2), Some(Expr::Atom(Atom::Symbol(name), _)) if name == def_name)
}

fn checked_def_text(src: &str, def_name: &str) -> String {
    let decls = parse_str(src).expect("surf parse");
    let deep = desugar_program(&decls);
    let checked = check_ir_program(&deep).expect("check");
    for e in checked.exprs() {
        if is_named_def(e, def_name) {
            return print_canonical(std::slice::from_ref(e));
        }
    }
    panic!("def `{def_name}` not found in checked program");
}

const SEPARATE_SIG_SDPA: &str = "\
sig sdpa: tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, d, p] -> tensor[s, s, p] -> tensor[s, d, p]
def sdpa(q, k, v, scale) = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  weights = softmax(mul(scores, scale), -1)
  matmul(weights, v)
}
";

// The inline reimplementation the issue points at: identical body, but
// concrete `f32` parameter annotations written inline on the `def` (no
// separate `sig`). The ordinary annotation path already resolves its body
// to tensor types.
const INLINE_TYPED_SDPA: &str = "\
def sdpa(q: tensor[s, d, f32], k: tensor[s, d, f32], v: tensor[s, d, f32], scale: tensor[s, s, f32]) -> tensor[s, d, f32] = {
  kt = permute(k, 1, 0)
  scores = matmul(q, kt)
  weights = softmax(mul(scores, scale), -1)
  matmul(weights, v)
}
";

#[test]
fn issue_319_separate_sig_body_permute_is_a_tensor_not_bare_tvar() {
    let text = checked_def_text(SEPARATE_SIG_SDPA, "sdpa");
    // The `kt = permute(...)` binding's annotated type must be a rank-2
    // tensor, not a bare `(t-var …)`. We locate the permute app and
    // assert its enclosing `type:` is a `t-tensor`.
    let permute_pos = text.find("permute").expect("permute app present");
    let before = &text[..permute_pos];
    // The nearest preceding `type:` belongs to the permute `app` node.
    let type_pos = before
        .rfind("type:")
        .expect("permute app carries a type annotation");
    let type_slice = &text[type_pos..permute_pos];
    assert!(
        type_slice.contains("t-tensor"),
        "issue #319: separate-sig body `permute` must annotate as a tensor, \
         not a bare type variable; got `{type_slice}`",
    );
}

#[test]
fn issue_319_separate_sig_body_carries_no_bare_tvar_app_type() {
    // Negative parity: no body `app` node may be left with a bare
    // `(t-var …)` *as its whole annotated type* — every shape-bearing op
    // in the body resolves to a concrete tensor type when the sig is
    // present. (`(t-prim {} int32)` axis literals are fine; the guard is
    // specifically about an app result type collapsing to a lone tvar.)
    let text = checked_def_text(SEPARATE_SIG_SDPA, "sdpa");
    assert!(
        !text.contains("type: (t-var"),
        "issue #319: a separate-sig verb body must not leave an app result \
         type as a bare type variable; checked def:\n{text}",
    );
}

#[test]
fn issue_319_inline_typed_control_also_has_tensor_body_types() {
    // Control: the inline-typed form already resolves its body to tensor
    // types. If this regresses, the fix broke the ordinary annotation
    // path rather than the separate-sig one.
    let text = checked_def_text(INLINE_TYPED_SDPA, "sdpa");
    assert!(
        text.contains("permute") && !text.contains("type: (t-var"),
        "issue #319 control: inline-typed verb body must resolve to tensor \
         types; checked def:\n{text}",
    );
}
