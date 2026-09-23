//! chelis#1125 / chelis#1029: the Surf and Deep producers build the same tree.
//!
//! With one in-memory spelling per construct, what stays checkable is that
//! the two producers agree (`spec/design/checker_totality.md`, PP7,
//! "Completion: one spelling per construct"). For every executable example,
//! the Deep that `chelis check` hands the checker from a `.ch` file must equal,
//! spans aside, the tree the `.dp` front end stamps from that Deep's canonical
//! print.
//!
//! The comparison walks both trees carrier by carrier. Comparing printed text
//! would compare the printer with itself: a structural list and an unknown
//! form with the same head print identically, which
//! `comparator_reports_a_carrier_swap_the_printer_cannot_see` demonstrates.

// The parity harness's discovery is the corpus definition; its membership
// checks are unused here.
#[allow(dead_code)]
#[path = "../../chelis-cli/tests/common/parity_corpus.rs"]
mod parity_corpus;

use chelis_compiler_api::pipeline::prepare_surf_decls;
use chelis_deep::node::Node;
use chelis_deep::printer::{print_canonical, print_expr_flat};
use chelis_deep::{Atom, DeepTag, Expr, Metadata, UnknownFormData};
use serde_json::Value;
use std::path::PathBuf;

fn examples_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .expect("examples directory should exist")
}

/// The executable examples, discovered exactly as `parity.rs` discovers them.
fn executable_examples() -> Vec<String> {
    let names = parity_corpus::discover(&examples_root())
        .unwrap_or_else(|error| panic!("example discovery failed: {error}"));
    assert!(!names.is_empty(), "the executable example corpus is empty");
    names.into_iter().collect()
}

/// The tree `chelis check` hands the checker for a single-file `.ch`: the
/// Surf parse, then desugaring and macro expansion in `prepare_surf_decls`.
fn surf_front_end(name: &str) -> Vec<Expr> {
    let path = examples_root().join(name);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    let decls = chelis_surf::parser::parse_str(&source)
        .unwrap_or_else(|error| panic!("{name} does not parse: {error}"));
    prepare_surf_decls(&decls, None)
        .unwrap_or_else(|error| panic!("{name} does not prepare: {error}"))
        .into_expanded_deep()
}

/// The `.dp` front end `chelis check` runs, applied to the canonical print.
fn deep_front_end(exprs: &[Expr]) -> Result<Vec<Expr>, String> {
    let printed = print_canonical(exprs);
    chelis_deep::parse_and_stamp_file(&printed)
        .map_err(|error| format!("the printed Deep does not stamp: {error}"))
}

struct Difference {
    path: String,
    left: String,
    right: String,
}

fn describe(expr: &Expr) -> String {
    let carrier = match expr {
        Expr::Atom(..) => "Atom",
        Expr::Map(..) => "Map",
        Expr::MetaExpr(..) => "MetaExpr",
        Expr::Node(..) => "Node",
        Expr::BareList(..) => "BareList",
        Expr::UnknownForm(..) => "UnknownForm",
    };
    let printed = print_expr_flat(expr);
    let mut shown: String = printed.chars().take(400).collect();
    if shown.len() < printed.len() {
        shown.push_str(" ...");
    }
    format!("{carrier} {shown}")
}

/// `chelis_deep::Span` serializes as exactly `{offset, len}`, and no other
/// type in the Deep wire model has that field set.
fn is_span(value: &Value) -> bool {
    let Value::Object(fields) = value else {
        return false;
    };
    fields.len() == 2
        && fields.get("offset").is_some_and(Value::is_u64)
        && fields.get("len").is_some_and(Value::is_u64)
}

fn erase_spans(value: &mut Value) {
    if is_span(value) {
        *value = Value::Null;
        return;
    }
    match value {
        Value::Object(fields) => fields.values_mut().for_each(erase_spans),
        Value::Array(items) => items.iter_mut().for_each(erase_spans),
        _ => {}
    }
}

/// Metadata payloads are typed and carry spans throughout, so they are
/// compared through their serde form with every span erased: a normalizing
/// copy of the data, which keeps each nested carrier's variant name.
fn spanless_metadata(metadata: &Metadata) -> Value {
    let mut value = serde_json::to_value(metadata).expect("metadata serializes");
    erase_spans(&mut value);
    value
}

fn metadata_difference(path: &str, left: &Metadata, right: &Metadata) -> Option<Difference> {
    let (left, right) = (spanless_metadata(left), spanless_metadata(right));
    (left != right).then(|| Difference {
        path: format!("{path}.meta"),
        left: left.to_string(),
        right: right.to_string(),
    })
}

fn sequence_difference(path: &str, left: &[Expr], right: &[Expr]) -> Option<Difference> {
    for (index, (left, right)) in left.iter().zip(right).enumerate() {
        if let Some(difference) = expr_difference(&format!("{path}[{index}]"), left, right) {
            return Some(difference);
        }
    }
    (left.len() != right.len()).then(|| Difference {
        path: format!("{path}.len"),
        left: left.len().to_string(),
        right: right.len().to_string(),
    })
}

fn expr_difference(path: &str, left: &Expr, right: &Expr) -> Option<Difference> {
    let differ = |what: &str| {
        Some(Difference {
            path: format!("{path}{what}"),
            left: describe(left),
            right: describe(right),
        })
    };
    match (left, right) {
        (Expr::Atom(l, _), Expr::Atom(r, _)) => {
            if l == r {
                None
            } else {
                differ(".atom")
            }
        }
        (Expr::Map(l, _), Expr::Map(r, _)) => metadata_difference(path, l, r),
        (Expr::MetaExpr(l, _), Expr::MetaExpr(r, _)) => {
            metadata_difference(path, &l.metadata, &r.metadata)
                .or_else(|| expr_difference(&format!("{path}.expr"), &l.expr, &r.expr))
        }
        (Expr::Node(l, _), Expr::Node(r, _)) => {
            if l.tag() != r.tag() {
                return differ(".tag");
            }
            let here = format!("{path}({})", l.tag().as_str());
            metadata_difference(&here, l.meta(), r.meta())
                .or_else(|| sequence_difference(&here, l.children_slice(), r.children_slice()))
        }
        (Expr::BareList(l, _), Expr::BareList(r, _)) => {
            sequence_difference(&format!("{path}(bare-list)"), l, r)
        }
        (Expr::UnknownForm(l), Expr::UnknownForm(r)) => {
            if l.head != r.head {
                return differ(".head");
            }
            let here = format!("{path}(unknown {})", l.head);
            metadata_difference(&here, &l.meta, &r.meta)
                .or_else(|| sequence_difference(&here, &l.children, &r.children))
        }
        _ => differ(".carrier"),
    }
}

/// The first place, spans aside, where two programs differ.
fn first_difference(left: &[Expr], right: &[Expr]) -> Option<Difference> {
    sequence_difference("program", left, right)
}

#[test]
fn surf_and_deep_producers_build_the_same_tree_for_every_executable_example() {
    let examples = executable_examples();
    let mut failures = Vec::new();
    let mut compared = 0;
    for name in &examples {
        let from_surf = surf_front_end(name);
        assert!(!from_surf.is_empty(), "{name} produced no Deep");
        match deep_front_end(&from_surf) {
            Ok(from_deep) => {
                if let Some(difference) = first_difference(&from_surf, &from_deep) {
                    failures.push(format!(
                        "{name}: first difference at {}\n  from .ch: {}\n  from .dp: {}",
                        difference.path, difference.left, difference.right
                    ));
                }
            }
            Err(error) => failures.push(format!("{name}: {error}")),
        }
        compared += 1;
    }
    println!("compared {compared} executable examples");
    assert_eq!(compared, examples.len());
    assert!(
        failures.is_empty(),
        "{} of {compared} examples disagree across producers:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Rebuild `expr` with `rewrite` applied at the first preorder site whose
/// enclosing node accepts the result.
fn rewrite_first(expr: &Expr, rewrite: &dyn Fn(&Expr) -> Option<Expr>) -> Option<Expr> {
    if let Some(replacement) = rewrite(expr) {
        return Some(replacement);
    }
    match expr {
        Expr::Node(node, span) => {
            node.children_slice()
                .iter()
                .enumerate()
                .find_map(|(index, child)| {
                    let replacement = rewrite_first(child, rewrite)?;
                    let mut rebuilt = (**node).clone();
                    rebuilt.try_replace_child(index, replacement).ok()?;
                    Some(Expr::Node(Box::new(rebuilt), *span))
                })
        }
        Expr::BareList(items, span) => items.iter().enumerate().find_map(|(index, item)| {
            let replacement = rewrite_first(item, rewrite)?;
            let mut items = items.clone();
            items[index] = replacement;
            Some(Expr::BareList(items, *span))
        }),
        _ => None,
    }
}

/// The first example whose `.ch` tree accepts `rewrite`, with the original
/// and the rewritten program.
fn rewritten_example(rewrite: &dyn Fn(&Expr) -> Option<Expr>) -> (String, Vec<Expr>, Vec<Expr>) {
    for name in executable_examples() {
        let original = surf_front_end(&name);
        for (index, expr) in original.iter().enumerate() {
            if let Some(replacement) = rewrite_first(expr, rewrite) {
                let mut rewritten = original.clone();
                rewritten[index] = replacement;
                return (name, original, rewritten);
            }
        }
    }
    panic!("no executable example has a site for this control; it did not run");
}

fn assert_reported(control: &str, name: &str, original: &[Expr], rewritten: &[Expr]) {
    let difference = first_difference(original, rewritten)
        .unwrap_or_else(|| panic!("{control} on {name} was not reported"));
    println!(
        "{control} on {name}: reported at {}\n  original: {}\n  rewritten: {}",
        difference.path, difference.left, difference.right
    );
}

#[test]
fn comparator_reports_a_wrapped_structural_list_child() {
    let (name, original, rewritten) = rewritten_example(&|expr| {
        let Expr::BareList(items, span) = expr else {
            return None;
        };
        let index = items
            .iter()
            .position(|item| matches!(item, Expr::Atom(Atom::Name(_), _)))?;
        let wrapped = Node::try_new(
            DeepTag::Var,
            Metadata::default(),
            vec![items[index].clone()],
        )
        .ok()?;
        let mut items = items.clone();
        items[index] = Expr::Node(Box::new(wrapped), *span);
        Some(Expr::BareList(items, *span))
    });
    assert_reported(
        "wrapping a structural-list child",
        &name,
        &original,
        &rewritten,
    );
}

#[test]
fn comparator_reports_a_dropped_metadata_entry() {
    let (name, original, rewritten) = rewritten_example(&|expr| {
        let Expr::Node(node, span) = expr else {
            return None;
        };
        let key = node.meta().values().next()?.key();
        let mut meta = node.meta().clone();
        meta.remove(key);
        let dropped = Node::try_new(node.tag(), meta, node.children_slice().to_vec()).ok()?;
        Some(Expr::Node(Box::new(dropped), *span))
    });
    assert_reported("dropping one metadata entry", &name, &original, &rewritten);
}

/// The first subtree, in corpus and preorder order, that `select` accepts.
fn first_subtree(select: &dyn Fn(&Expr) -> bool) -> (String, Expr) {
    fn find(expr: &Expr, select: &dyn Fn(&Expr) -> bool) -> Option<Expr> {
        if select(expr) {
            return Some(expr.clone());
        }
        match expr {
            Expr::Node(node, _) => node.children_slice().iter().find_map(|c| find(c, select)),
            Expr::BareList(items, _) => items.iter().find_map(|item| find(item, select)),
            Expr::UnknownForm(data) => data.children.iter().find_map(|c| find(c, select)),
            Expr::MetaExpr(meta, _) => find(&meta.expr, select),
            Expr::Atom(..) | Expr::Map(..) => None,
        }
    }
    for name in executable_examples() {
        if let Some(found) = surf_front_end(&name).iter().find_map(|e| find(e, select)) {
            return (name, found);
        }
    }
    panic!("no executable example has a site for this control; it did not run");
}

#[test]
fn comparator_reports_a_carrier_swap_the_printer_cannot_see() {
    // A real structural list `(name {..} ..)` and the unknown form with the
    // same head, metadata and children, each compared as a one-form program
    // so that no enclosing node has to admit the swapped carrier.
    let (name, structural) = first_subtree(&|expr| {
        matches!(expr, Expr::BareList(items, _)
            if matches!(items.as_slice(), [Expr::Atom(Atom::Name(_), _), Expr::Map(..), ..]))
    });
    let Expr::BareList(items, span) = &structural else {
        unreachable!("the selector admits only structural lists");
    };
    let [
        Expr::Atom(Atom::Name(head), _),
        Expr::Map(meta, _),
        children @ ..,
    ] = items.as_slice()
    else {
        unreachable!("the selector admits only a name and a map first");
    };
    let unknown = Expr::UnknownForm(Box::new(UnknownFormData {
        head: head.clone(),
        meta: meta.clone(),
        children: children.to_vec(),
        span: *span,
    }));
    let (structural, unknown) = ([structural], [unknown]);
    assert_eq!(
        print_canonical(&structural),
        print_canonical(&unknown),
        "the two carriers must print identically for this control to mean anything"
    );
    let difference = first_difference(&structural, &unknown)
        .unwrap_or_else(|| panic!("the carrier swap on {name} was not reported"));
    assert_eq!(difference.path, "program[0].carrier");
    println!(
        "swapping a structural list for an unknown form on {name}: reported at {}\n  original: {}\n  rewritten: {}",
        difference.path, difference.left, difference.right
    );
}
