//! Canonical printer for the Chelis Deep syntax.
//!
//! Converts `&[Expr]` into the canonical string representation used for `.dp`
//! files and `chelis deep`.

use crate::annotations_codec::{
    WireExpr, WireMetaExpr as MetaExpr, WireMetadata as MetaMap, WireNode, WireUnknown,
};
use crate::ast::{Atom, Expr};
use crate::tag::DeepTag;

const MAX_LINE: usize = 80;
const INDENT_STEP: usize = 2;

/// Print a slice of top-level expressions in canonical pretty form.
///
/// Expressions are separated by blank lines and the output ends with a
/// single newline.
pub fn print_canonical(exprs: &[Expr]) -> String {
    let printer = Printer::new();
    print_program(exprs, |wire, out| out.push_str(&printer.fmt_expr(wire, 0)))
}

/// Print a slice of top-level expressions in flat canonical form.
pub fn print_canonical_flat(exprs: &[Expr]) -> String {
    print_program(exprs, |wire, out| write_flat(Flat::Expr(wire), out))
}

/// Print a single expression in canonical pretty form (no trailing newline).
pub fn print_expr(expr: &Expr) -> String {
    with_wire(expr, |wire| Printer::new().fmt_expr(wire, 0))
}

/// Print a single expression in canonical flat form (no trailing newline).
pub fn print_expr_flat(expr: &Expr) -> String {
    with_wire(expr, |wire| render_flat(Flat::Expr(wire)))
}

pub fn print_macro_source(source: &crate::annotations::MacroSource) -> String {
    render_flat(Flat::Expr(&WireExpr::from_value(
        &crate::annotations::MetadataValue::Source(source.clone()),
    )))
}

fn print_program(exprs: &[Expr], mut write: impl FnMut(&WireExpr, &mut String)) -> String {
    if exprs.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (index, expr) in exprs.iter().enumerate() {
        if index > 0 {
            out.push_str("\n\n");
        }
        with_wire(expr, |wire| write(wire, &mut out));
    }
    out.push('\n');
    out
}

/// Run `f` on the wire shadow of `expr`, then release the shadow without the
/// per-level recursion of its derived drop glue (chelis#2424).
fn with_wire<R>(expr: &Expr, f: impl FnOnce(&WireExpr) -> R) -> R {
    let wire = WireExpr::from_ast(expr);
    let result = f(&wire);
    wire.dispose();
    result
}

/// One pending piece of flat canonical output.
enum Flat<'a> {
    Text(&'a str),
    Expr(&'a WireExpr),
    Node(&'a WireNode),
    Map(&'a MetaMap),
    MetaExpr(&'a MetaExpr),
    MetaEntries(&'a [(String, WireExpr)]),
}

/// The flat canonical form of `item`.
fn render_flat(item: Flat<'_>) -> String {
    let mut out = String::new();
    write_flat(item, &mut out);
    out
}

/// Append the flat canonical form of `item` to `out`.
///
/// Flat output is one token stream, so a heap stack of pending pieces
/// replaces the per-level recursion: the depth of the tree no longer bounds
/// the native stack, and each byte is written once instead of being copied
/// into every enclosing level's string (chelis#2424).
fn write_flat(item: Flat<'_>, out: &mut String) {
    let mut stack = vec![item];
    while let Some(item) = stack.pop() {
        match item {
            Flat::Text(text) => out.push_str(text),
            Flat::Expr(expr) => match expr {
                WireExpr::ExtensionData(data) => out.push_str(data.syntax()),
                WireExpr::Atom(atom, _) => write_atom(out, atom),
                WireExpr::Map(map, _) => stack.push(Flat::Map(map)),
                WireExpr::MetaExpr(meta, _) => stack.push(Flat::MetaExpr(meta)),
                WireExpr::Node(node, _) => stack.push(Flat::Node(node)),
                WireExpr::BareList(items, _) => {
                    out.push('(');
                    stack.push(Flat::Text(")"));
                    for (index, item) in items.iter().enumerate().rev() {
                        stack.push(Flat::Expr(item));
                        if index > 0 {
                            stack.push(Flat::Text(" "));
                        }
                    }
                }
                WireExpr::UnknownForm(form) => {
                    open_form(&form.head, &form.meta, &form.children, out, &mut stack);
                }
            },
            Flat::Node(node) => {
                open_form(
                    node.tag.as_str(),
                    &node.meta,
                    &node.children,
                    out,
                    &mut stack,
                );
            }
            Flat::Map(map) => {
                out.push('{');
                stack.push(Flat::Text("}"));
                let entries = sorted_entries(&map.entries);
                for (index, (key, value)) in entries.into_iter().enumerate().rev() {
                    stack.push(Flat::Expr(value));
                    stack.push(Flat::Text(": "));
                    stack.push(Flat::Text(key));
                    if index > 0 {
                        stack.push(Flat::Text(", "));
                    }
                }
            }
            Flat::MetaExpr(meta) => {
                stack.push(Flat::Expr(&meta.expr));
                stack.push(Flat::Text(" "));
                stack.push(Flat::MetaEntries(&meta.entries));
            }
            Flat::MetaEntries(entries) => {
                out.push_str("^{");
                stack.push(Flat::Text("}"));
                for (index, (key, value)) in sorted_entries(entries).into_iter().enumerate().rev() {
                    stack.push(Flat::Expr(value));
                    stack.push(Flat::Text(" "));
                    stack.push(Flat::Text(key));
                    stack.push(Flat::Text(":"));
                    if index > 0 {
                        stack.push(Flat::Text(" "));
                    }
                }
            }
        }
    }
}

/// Write a form's `(head ` and queue the `{meta} child...)` that follows it.
fn open_form<'a>(
    head: &str,
    meta: &'a MetaMap,
    children: &'a [WireExpr],
    out: &mut String,
    stack: &mut Vec<Flat<'a>>,
) {
    out.push('(');
    out.push_str(head);
    out.push(' ');
    stack.push(Flat::Text(")"));
    for child in children.iter().rev() {
        stack.push(Flat::Expr(child));
        stack.push(Flat::Text(" "));
    }
    stack.push(Flat::Map(meta));
}

/// Entries in canonical key order. The sort is stable, so entries with equal
/// keys keep their order.
fn sorted_entries(entries: &[(String, WireExpr)]) -> Vec<(&str, &WireExpr)> {
    let mut sorted: Vec<_> = entries
        .iter()
        .map(|(key, value)| (key.as_str(), value))
        .collect();
    sorted.sort_by_key(|(key, _)| *key);
    sorted
}

fn write_atom(out: &mut String, atom: &Atom) {
    use std::fmt::Write as _;
    match atom {
        Atom::Name(s) => out.push_str(s),
        Atom::Int(n) => write!(out, "{n}").expect("writing to a String cannot fail"),
        Atom::Float(f) => {
            let start = out.len();
            write!(out, "{f}").expect("writing to a String cannot fail");
            if !out[start..].contains('.') {
                out.push_str(".0");
            }
        }
        Atom::Str(s) => {
            out.reserve(s.len() + 2);
            out.push('"');
            for ch in s.chars() {
                match ch {
                    '\\' => out.push_str("\\\\"),
                    '"' => out.push_str("\\\""),
                    '\n' => out.push_str("\\n"),
                    '\t' => out.push_str("\\t"),
                    '\r' => out.push_str("\\r"),
                    '\0' => out.push_str("\\0"),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Atom::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
    }
}

/// The canonical text of one atom.
fn atom_text(atom: &Atom) -> String {
    let mut out = String::new();
    write_atom(&mut out, atom);
    out
}

/// The pretty layout. Every width it measures is a flat rendering from
/// [`write_flat`]; only breaking a line recurses.
struct Printer {
    max_width: usize,
    indent_step: usize,
}

impl Printer {
    fn new() -> Self {
        Self {
            max_width: MAX_LINE,
            indent_step: INDENT_STEP,
        }
    }

    fn fmt_expr(&self, expr: &WireExpr, indent: usize) -> String {
        match expr {
            WireExpr::ExtensionData(data) => data.syntax().to_string(),
            WireExpr::Atom(atom, _) => atom_text(atom),
            WireExpr::Map(map, _) => self.fmt_map(map, indent),
            WireExpr::MetaExpr(meta, _) => self.fmt_meta_expr(meta, indent),
            WireExpr::Node(node, _) => self.fmt_node(node, indent),
            WireExpr::BareList(items, _) => self.fmt_list(expr, items, indent),
            WireExpr::UnknownForm(form) => self.fmt_unknown_form(expr, form, indent),
        }
    }

    /// A stamped node prints as its canonical `(tag {meta} children...)`
    /// form, with the tag regenerated from the decoded [`DeepTag`].
    fn fmt_node(&self, node: &WireNode, indent: usize) -> String {
        let flat = render_flat(Flat::Node(node));
        if force_flat_tag(node.tag) {
            return flat;
        }
        if !force_break_tag(node.tag) && indent + flat.len() <= self.max_width {
            return flat;
        }
        self.fmt_canonical_node(node, indent)
    }

    fn fmt_canonical_node(&self, node: &WireNode, indent: usize) -> String {
        let tag = node.tag.as_str();
        let children = &node.children;
        let meta_indent = indent + 1 + tag.len() + 1;
        let meta_text = self.fmt_map(&node.meta, meta_indent);
        let header = format!("({tag} {meta_text}");

        if children.is_empty() {
            return format!("{header})");
        }

        let child_indent = indent + self.indent_step;
        let prefix = " ".repeat(child_indent);
        let mut lines: Vec<String> = header.lines().map(ToString::to_string).collect();
        for child in children {
            push_child_lines(&mut lines, &prefix, &self.fmt_expr(child, child_indent));
        }
        lines
            .last_mut()
            .expect("canonical node has at least one child line")
            .push(')');
        lines.join("\n")
    }

    /// A structural list prints flat when it fits, else broken after its
    /// first element.
    fn fmt_list(&self, list: &WireExpr, items: &[WireExpr], indent: usize) -> String {
        let flat = render_flat(Flat::Expr(list));
        let Some((first, rest)) = items.split_first() else {
            return flat;
        };
        if indent + flat.len() <= self.max_width {
            return flat;
        }
        let first = self.fmt_expr(first, indent + self.indent_step);
        self.fmt_broken_list(&first, None, rest, indent)
    }

    /// An unknown form prints as the list of its head symbol, its metadata
    /// map and its children, laid out from the borrowed form.
    fn fmt_unknown_form(&self, form_expr: &WireExpr, form: &WireUnknown, indent: usize) -> String {
        let flat = render_flat(Flat::Expr(form_expr));
        if indent + flat.len() <= self.max_width {
            return flat;
        }
        self.fmt_broken_list(&form.head, Some(&form.meta), &form.children, indent)
    }

    /// A list too wide for one line: `first` follows the opening parenthesis,
    /// and the metadata map, when there is one, and every item start their
    /// own lines.
    fn fmt_broken_list(
        &self,
        first: &str,
        meta: Option<&MetaMap>,
        items: &[WireExpr],
        indent: usize,
    ) -> String {
        let child_indent = indent + self.indent_step;
        let prefix = " ".repeat(child_indent);
        let mut lines = vec![format!("({first}")];
        if let Some(meta) = meta {
            push_child_lines(&mut lines, &prefix, &self.fmt_map(meta, child_indent));
        }
        for item in items {
            push_child_lines(&mut lines, &prefix, &self.fmt_expr(item, child_indent));
        }
        lines
            .last_mut()
            .expect("generic list always has a header line")
            .push(')');
        lines.join("\n")
    }

    fn fmt_map(&self, map: &MetaMap, indent: usize) -> String {
        let flat = render_flat(Flat::Map(map));
        if indent + flat.len() <= self.max_width {
            return flat;
        }

        let sorted = sorted_entries(&map.entries);
        let mut lines = Vec::new();
        for (index, (key, value)) in sorted.iter().enumerate() {
            let entry_indent = indent + self.indent_step;
            let value_indent = entry_indent + key.len() + 2;
            let rendered = self.fmt_expr(value, value_indent);
            let mut value_lines = rendered.lines();
            let first = value_lines.next().unwrap_or("");
            let line = if index == 0 {
                format!("{{{}: {}", key, first)
            } else {
                format!("{}{}: {}", " ".repeat(entry_indent), key, first)
            };
            lines.push(line);
            for continuation in value_lines {
                lines.push(continuation.to_string());
            }
            if index + 1 != sorted.len()
                && let Some(last) = lines.last_mut()
            {
                last.push(',');
            }
        }
        lines.push(format!("{}}}", " ".repeat(indent)));
        lines.join("\n")
    }

    fn fmt_meta_expr(&self, meta: &MetaExpr, indent: usize) -> String {
        let flat = render_flat(Flat::MetaExpr(meta));
        if indent + flat.len() <= self.max_width {
            return flat;
        }

        let meta_part = render_flat(Flat::MetaEntries(&meta.entries));
        let expr_indent = indent + meta_part.len() + 1;
        let expr_text = self.fmt_expr(&meta.expr, expr_indent);
        let mut lines = expr_text.lines();
        let first = lines.next().unwrap_or("");
        let mut output = format!("{meta_part} {first}");
        for line in lines {
            output.push('\n');
            output.push_str(line);
        }
        output
    }
}

/// Append a rendered child: its first line indented by `prefix`, and its
/// continuation lines as rendered.
fn push_child_lines(lines: &mut Vec<String>, prefix: &str, rendered: &str) {
    let mut child_lines = rendered.lines();
    if let Some(first) = child_lines.next() {
        lines.push(format!("{prefix}{first}"));
    }
    lines.extend(child_lines.map(ToString::to_string));
}

/// Leaf tags that always print on one line.
fn force_flat_tag(tag: DeepTag) -> bool {
    matches!(
        tag,
        DeepTag::Var
            | DeepTag::Lit
            | DeepTag::DName
            | DeepTag::DVar
            | DeepTag::DLit
            | DeepTag::DRank
            | DeepTag::TPrim
    )
}

/// Binding forms that always break their children onto separate lines.
fn force_break_tag(tag: DeepTag) -> bool {
    matches!(tag, DeepTag::Fn | DeepTag::Let | DeepTag::Bind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Atom, Expr, MetaExpr, Metadata};
    use crate::span::Span;

    fn sp() -> Span {
        Span { offset: 0, len: 0 }
    }

    fn atom_expr(atom: Atom) -> Expr {
        Expr::Atom(atom, sp())
    }

    fn sym(name: &str) -> Expr {
        atom_expr(Atom::Name(name.to_string()))
    }

    fn metadata(values: Vec<(&str, Expr)>) -> Metadata {
        let mut metadata = Metadata::default();
        for (key, value) in values {
            if key == "type" {
                metadata
                    .insert(crate::annotations::MetadataValue::Type(
                        crate::annotations::TypeSyntax::try_new(value).unwrap(),
                    ))
                    .unwrap();
            } else {
                metadata
                    .extensions_mut()
                    .insert(
                        key.into(),
                        crate::ExtensionData::parse(&print_expr_flat(&value)).unwrap(),
                    )
                    .unwrap();
            }
        }
        metadata
    }
    fn node(tag: &str, meta: Vec<(&str, Expr)>, children: Vec<Expr>) -> Expr {
        let tag = DeepTag::parse(tag).expect("fixture tags are in the closed vocabulary");
        Expr::node(tag, metadata(meta), children, sp())
    }

    fn generic_list(elements: Vec<Expr>) -> Expr {
        Expr::BareList(elements, sp())
    }

    fn meta_expr(entries: Vec<(&str, Expr)>, expr: Expr) -> Expr {
        Expr::MetaExpr(
            MetaExpr {
                metadata: metadata(entries),
                expr: Box::new(expr),
            },
            sp(),
        )
    }

    #[test]
    fn test_atom_string_escapes() {
        assert_eq!(
            print_expr(&atom_expr(Atom::Str("a \"quote\"\n".into()))),
            r#""a \"quote\"\n""#
        );
    }

    #[test]
    fn leaf_nodes_stay_single_line() {
        let expr = node(
            "lit",
            vec![("type", node("t-prim", vec![], vec![sym("f32")]))],
            vec![atom_expr(Atom::Float(3.125))],
        );
        assert_eq!(print_expr(&expr), "(lit {type: (t-prim {} f32)} 3.125)");
    }

    #[test]
    fn short_expr_stays_single_line() {
        let expr = node(
            "app",
            vec![],
            vec![
                node("var", vec![], vec![sym("add")]),
                node("var", vec![], vec![sym("x")]),
                node("var", vec![], vec![sym("y")]),
            ],
        );
        assert_eq!(
            print_expr(&expr),
            "(app {} (var {} add) (var {} x) (var {} y))"
        );
    }

    #[test]
    fn long_expr_breaks_children_under_tag_and_meta() {
        let expr = node(
            "app",
            vec![],
            vec![
                node("var", vec![], vec![sym("mean")]),
                node(
                    "app",
                    vec![],
                    vec![
                        node("var", vec![], vec![sym("neg")]),
                        node(
                            "app",
                            vec![],
                            vec![
                                node("var", vec![], vec![sym("sum")]),
                                node("var", vec![], vec![sym("very_long_intermediate_name")]),
                                node(
                                    "lit",
                                    vec![("type", node("t-prim", vec![], vec![sym("i32")]))],
                                    vec![atom_expr(Atom::Int(0))],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        );
        assert_eq!(
            print_expr(&expr),
            "(app {}\n  (var {} mean)\n  (app {}\n    (var {} neg)\n    (app {}\n      (var {} sum)\n      (var {} very_long_intermediate_name)\n      (lit {type: (t-prim {} i32)} 0))))"
        );
    }

    #[test]
    fn fn_layout_puts_params_and_body_on_separate_lines() {
        let expr = node(
            "fn",
            vec![],
            vec![
                node("params", vec![], vec![sym("x"), sym("y")]),
                node(
                    "app",
                    vec![],
                    vec![
                        node("var", vec![], vec![sym("add")]),
                        node("var", vec![], vec![sym("x")]),
                        node("var", vec![], vec![sym("y")]),
                    ],
                ),
            ],
        );
        assert_eq!(
            print_expr(&expr),
            "(fn {}\n  (params {} x y)\n  (app {} (var {} add) (var {} x) (var {} y)))"
        );
    }

    #[test]
    fn let_layout_puts_bind_and_body_on_separate_lines() {
        let expr = node(
            "let",
            vec![],
            vec![
                node(
                    "bind",
                    vec![],
                    vec![
                        sym("hidden"),
                        node(
                            "app",
                            vec![],
                            vec![
                                node("var", vec![], vec![sym("relu")]),
                                node("var", vec![], vec![sym("very_long_intermediate_name")]),
                            ],
                        ),
                    ],
                ),
                node("var", vec![], vec![sym("hidden")]),
            ],
        );
        assert_eq!(
            print_expr(&expr),
            "(let {}\n  (bind {}\n    hidden\n    (app {} (var {} relu) (var {} very_long_intermediate_name)))\n  (var {} hidden))"
        );
    }

    #[test]
    fn long_metadata_breaks_inside_map() {
        let expr = node(
            "fn",
            vec![(
                "type",
                node(
                    "t-fn",
                    vec![],
                    vec![
                        node(
                            "t-tensor",
                            vec![],
                            vec![
                                node("d-name", vec![], vec![sym("batch")]),
                                node("d-name", vec![], vec![sym("hidden")]),
                                node("t-prim", vec![], vec![sym("f32")]),
                            ],
                        ),
                        node(
                            "t-tensor",
                            vec![],
                            vec![
                                node("d-name", vec![], vec![sym("batch")]),
                                node("t-prim", vec![], vec![sym("f32")]),
                            ],
                        ),
                    ],
                ),
            )],
            vec![
                node("params", vec![], vec![sym("x")]),
                node("var", vec![], vec![sym("x")]),
            ],
        );
        let output = print_expr(&expr);
        assert!(output.contains("{type: (t-fn {}"));
        assert!(output.contains("\n              (t-tensor {}"));
        assert!(output.contains("\n  (params {} x)"));
    }

    #[test]
    fn meta_expr_sorts_keys() {
        let expr = meta_expr(
            vec![
                ("z_key", atom_expr(Atom::Int(1))),
                ("a_key", atom_expr(Atom::Int(2))),
            ],
            atom_expr(Atom::Name("body".into())),
        );
        assert_eq!(print_expr(&expr), "^{:a_key 2 :z_key 1} body");
    }

    #[test]
    fn flat_mode_preserves_single_line_node() {
        let expr = node(
            "app",
            vec![],
            vec![
                node("var", vec![], vec![sym("mean")]),
                node(
                    "app",
                    vec![],
                    vec![
                        node("var", vec![], vec![sym("neg")]),
                        node("var", vec![], vec![sym("very_long_intermediate_name")]),
                    ],
                ),
            ],
        );
        assert!(!print_expr_flat(&expr).contains('\n'));
    }

    #[test]
    fn generic_lists_still_round_trip() {
        let expr = generic_list(vec![sym(":"), atom_expr(Atom::Int(42)), sym("i32")]);
        assert_eq!(print_expr(&expr), "(: 42 i32)");
    }

    #[test]
    fn multiple_top_level_exprs_have_blank_line_between() {
        let exprs = vec![sym("one"), sym("two")];
        assert_eq!(print_canonical(&exprs), "one\n\ntwo\n");
    }

    /// chelis#2424: flat printing, with the Deep-to-wire conversion and the
    /// release of the converted tree, does not recurse once per nesting
    /// level. A 100,000-deep chain cycling through every carrier that nests
    /// (`Node`, `BareList`, `MetaExpr`, `UnknownForm`) prints byte for byte on
    /// a 256 KiB thread. Building, comparing and dropping the input recurse,
    /// so they run on a large-stack thread; only the printer runs on the
    /// small one.
    #[test]
    fn a_deep_mixed_chain_prints_flat_on_a_small_stack() {
        std::thread::Builder::new()
            .stack_size(512 * 1024 * 1024)
            .spawn(print_a_deep_mixed_chain_on_a_small_stack)
            .expect("spawn the large-stack harness")
            .join()
            .expect("the large-stack harness must not abort");
    }

    fn print_a_deep_mixed_chain_on_a_small_stack() {
        const DEPTH: usize = 100_000;
        let var = |name: &str| node("var", vec![], vec![sym(name)]);
        let mut input = var("x");
        let mut prefixes = Vec::with_capacity(DEPTH);
        let mut suffixes = String::with_capacity(DEPTH);
        for level in 1..=DEPTH {
            let (wrapped, prefix, suffix) = match level % 4 {
                0 => (
                    node("app", vec![], vec![var("g"), input]),
                    "(app {} (var {} g) ",
                    ")",
                ),
                1 => (generic_list(vec![sym("g"), input]), "(g ", ")"),
                2 => (meta_expr(vec![], input), "^{} ", ""),
                _ => (
                    Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
                        head: "u".into(),
                        meta: Metadata::default(),
                        children: vec![input],
                        span: sp(),
                    })),
                    "(u {} ",
                    ")",
                ),
            };
            input = wrapped;
            prefixes.push(prefix);
            suffixes.push_str(suffix);
        }
        let expected: String =
            prefixes.iter().rev().copied().collect::<String>() + "(var {} x)" + &suffixes;

        let (program, expr) = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(256 * 1024)
                .spawn_scoped(scope, || {
                    (
                        print_canonical_flat(std::slice::from_ref(&input)),
                        print_expr_flat(&input),
                    )
                })
                .expect("spawn the small-stack printer")
                .join()
                .expect("flat printing must not exhaust a small native stack")
        });
        assert!(program == format!("{expected}\n"), "program output differs");
        assert!(expr == expected, "expression output differs");
    }
}
