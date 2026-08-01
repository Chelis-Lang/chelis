//! Canonical printer for the Chelis Deep syntax.
//!
//! Converts `&[Expr]` into the canonical string representation used for `.dp`
//! files and `chelis deep`.

use crate::ast::{Atom, Expr, List, MetaExpr, MetaMap};
use crate::span::Span;
use crate::tag::DeepTag;

const MAX_LINE: usize = 80;
const INDENT_STEP: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrintMode {
    Pretty,
    Flat,
}

/// Print a slice of top-level expressions in canonical pretty form.
///
/// Expressions are separated by blank lines and the output ends with a
/// single newline.
pub fn print_canonical(exprs: &[Expr]) -> String {
    print_program(exprs, PrintMode::Pretty)
}

/// Print a slice of top-level expressions in flat canonical form.
pub fn print_canonical_flat(exprs: &[Expr]) -> String {
    print_program(exprs, PrintMode::Flat)
}

/// Print a single expression in canonical pretty form (no trailing newline).
pub fn print_expr(expr: &Expr) -> String {
    Printer::pretty().fmt_expr(expr, 0)
}

/// Print a single expression in canonical flat form (no trailing newline).
pub fn print_expr_flat(expr: &Expr) -> String {
    Printer::flat().fmt_expr(expr, 0)
}

fn print_program(exprs: &[Expr], mode: PrintMode) -> String {
    if exprs.is_empty() {
        return String::new();
    }
    let printer = Printer::new(mode);
    let parts: Vec<String> = exprs.iter().map(|expr| printer.fmt_expr(expr, 0)).collect();
    let mut out = parts.join("\n\n");
    out.push('\n');
    out
}

struct Printer {
    mode: PrintMode,
    max_width: usize,
    indent_step: usize,
}

impl Printer {
    fn new(mode: PrintMode) -> Self {
        Self {
            mode,
            max_width: MAX_LINE,
            indent_step: INDENT_STEP,
        }
    }

    fn pretty() -> Self {
        Self::new(PrintMode::Pretty)
    }

    fn flat() -> Self {
        Self::new(PrintMode::Flat)
    }

    fn fmt_expr(&self, expr: &Expr, indent: usize) -> String {
        match expr {
            Expr::Atom(atom, _) => Self::fmt_atom(atom),
            Expr::Map(map, _) => self.fmt_map(map, indent),
            Expr::MetaExpr(meta, _) => self.fmt_meta_expr(meta, indent),
            Expr::List(list, _) => self.fmt_list(list, indent),
            Expr::Node(node, _) => {
                // Render a stamped Node back as its canonical list form.
                let list = node_to_list(node.as_ref());
                self.fmt_list(&list, indent)
            }
            Expr::BareList(elems, _) => {
                let list = List {
                    elements: elems.clone(),
                };
                self.fmt_list(&list, indent)
            }
            Expr::UnknownForm(data) => {
                let list = unknown_form_to_list(&data.head, &data.meta, &data.children);
                self.fmt_list(&list, indent)
            }
        }
    }

    fn fmt_expr_flat(&self, expr: &Expr) -> String {
        match expr {
            Expr::Atom(atom, _) => Self::fmt_atom(atom),
            Expr::Map(map, _) => Self::fmt_map_flat(map),
            Expr::MetaExpr(meta, _) => self.fmt_meta_expr_flat(meta),
            Expr::List(list, _) => self.fmt_list_flat(list),
            Expr::Node(node, _) => {
                let list = node_to_list(node.as_ref());
                self.fmt_list_flat(&list)
            }
            Expr::BareList(elems, _) => {
                let list = List {
                    elements: elems.clone(),
                };
                self.fmt_list_flat(&list)
            }
            Expr::UnknownForm(data) => {
                let list = unknown_form_to_list(&data.head, &data.meta, &data.children);
                self.fmt_list_flat(&list)
            }
        }
    }

    fn fmt_atom(atom: &Atom) -> String {
        match atom {
            Atom::Name(s) => s.clone(),
            Atom::Int(n) => n.to_string(),
            Atom::Float(f) => {
                let s = f.to_string();
                if s.contains('.') { s } else { format!("{s}.0") }
            }
            Atom::Str(s) => {
                let mut out = String::with_capacity(s.len() + 2);
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
                out
            }
            Atom::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        }
    }

    fn fmt_list(&self, list: &List, indent: usize) -> String {
        let flat = self.fmt_list_flat(list);
        if self.mode == PrintMode::Flat || self.force_flat_list(list) {
            return flat;
        }
        if !self.force_break_list(list) && indent + flat.len() <= self.max_width {
            return flat;
        }

        if let Some(output) = self.fmt_canonical_node(list, indent) {
            return output;
        }

        self.fmt_generic_list_broken(list, indent)
    }

    fn fmt_list_flat(&self, list: &List) -> String {
        let parts: Vec<String> = list
            .elements
            .iter()
            .map(|expr| self.fmt_expr_flat(expr))
            .collect();
        format!("({})", parts.join(" "))
    }

    fn fmt_canonical_node(&self, list: &List, indent: usize) -> Option<String> {
        let (tag, meta, children) = canonical_node_parts(list)?;
        let tag = tag.as_str();
        let meta_indent = indent + 1 + tag.len() + 1;
        let meta_text = self.fmt_map(meta, meta_indent);
        let header = format!("({tag} {meta_text}");

        if children.is_empty() {
            return Some(format!("{header})"));
        }

        let child_indent = indent + self.indent_step;
        let prefix = " ".repeat(child_indent);
        let mut lines: Vec<String> = header.lines().map(ToString::to_string).collect();
        for child in children {
            let rendered = self.fmt_expr(child, child_indent);
            let mut child_lines = rendered.lines();
            if let Some(first) = child_lines.next() {
                lines.push(format!("{prefix}{first}"));
            }
            for line in child_lines {
                lines.push(line.to_string());
            }
        }
        lines
            .last_mut()
            .expect("canonical node has at least one child line")
            .push(')');
        Some(lines.join("\n"))
    }

    fn fmt_generic_list_broken(&self, list: &List, indent: usize) -> String {
        if list.elements.is_empty() {
            return "()".to_string();
        }

        let child_indent = indent + self.indent_step;
        let prefix = " ".repeat(child_indent);
        let mut lines = vec![format!(
            "({}",
            self.fmt_expr(&list.elements[0], child_indent)
        )];
        for child in &list.elements[1..] {
            let rendered = self.fmt_expr(child, child_indent);
            let mut child_lines = rendered.lines();
            if let Some(first) = child_lines.next() {
                lines.push(format!("{prefix}{first}"));
            }
            for line in child_lines {
                lines.push(line.to_string());
            }
        }
        lines
            .last_mut()
            .expect("generic list always has a header line")
            .push(')');
        lines.join("\n")
    }

    fn fmt_map(&self, map: &MetaMap, indent: usize) -> String {
        let flat = Self::fmt_map_flat_with(self, map);
        if self.mode == PrintMode::Flat || indent + flat.len() <= self.max_width {
            return flat;
        }

        let mut sorted: Vec<_> = map.entries.iter().collect();
        sorted.sort_by_key(|(key, _)| key.as_str());

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

    fn fmt_map_flat(map: &MetaMap) -> String {
        Self::fmt_map_flat_with(&Self::flat(), map)
    }

    fn fmt_map_flat_with(&self, map: &MetaMap) -> String {
        if map.entries.is_empty() {
            return "{}".to_string();
        }
        let mut sorted: Vec<_> = map.entries.iter().collect();
        sorted.sort_by_key(|(key, _)| key.as_str());
        let parts: Vec<String> = sorted
            .iter()
            .map(|(key, value)| format!("{key}: {}", self.fmt_expr_flat(value)))
            .collect();
        format!("{{{}}}", parts.join(", "))
    }

    fn fmt_meta_expr(&self, meta: &MetaExpr, indent: usize) -> String {
        let flat = self.fmt_meta_expr_flat(meta);
        if self.mode == PrintMode::Flat || indent + flat.len() <= self.max_width {
            return flat;
        }

        let meta_part = self.fmt_meta_entries(&meta.entries);
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

    fn fmt_meta_expr_flat(&self, meta: &MetaExpr) -> String {
        let meta_part = self.fmt_meta_entries(&meta.entries);
        format!("{meta_part} {}", self.fmt_expr_flat(&meta.expr))
    }

    fn fmt_meta_entries(&self, entries: &[(String, Expr)]) -> String {
        let mut sorted: Vec<_> = entries.iter().collect();
        sorted.sort_by_key(|(key, _)| key.as_str());
        let parts: Vec<String> = sorted
            .iter()
            .map(|(key, value)| format!(":{key} {}", self.fmt_expr_flat(value)))
            .collect();
        format!("^{{{}}}", parts.join(" "))
    }

    fn force_flat_list(&self, list: &List) -> bool {
        let Some((tag, _, _)) = canonical_node_parts(list) else {
            return false;
        };
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

    fn force_break_list(&self, list: &List) -> bool {
        let Some((tag, _, _)) = canonical_node_parts(list) else {
            return false;
        };
        matches!(tag, DeepTag::Fn | DeepTag::Let | DeepTag::Bind)
    }
}

fn canonical_node_parts(list: &List) -> Option<(DeepTag, &MetaMap, &[Expr])> {
    match list.elements.as_slice() {
        [
            Expr::Atom(Atom::Name(tag_str), _),
            Expr::Map(meta, _),
            children @ ..,
        ] => {
            let tag = DeepTag::parse(tag_str)?;
            Some((tag, meta, children))
        }
        _ => None,
    }
}

/// Reconstruct a canonical `List` from a stamped `Node` for printing.
fn node_to_list(node: &crate::node::Node) -> List {
    use crate::node::ChildRef;
    let span = Span::new(0, 0);
    let mut elements = Vec::with_capacity(node.child_count() + 2);
    elements.push(Expr::Atom(Atom::Name(node.tag().as_str().to_string()), span));
    elements.push(Expr::Map(node.meta().clone(), span));
    for child_ref in node.children_iter() {
        match child_ref {
            ChildRef::Expr(e)
            | ChildRef::Syntax(e)
            | ChildRef::Type(e)
            | ChildRef::EffectHandler(e)
            | ChildRef::Bypass(e) => elements.push(e.clone()),
            ChildRef::Binder(s) => {
                elements.push(Expr::Atom(Atom::Name(s.to_string()), span));
            }
            ChildRef::Selector(s) => {
                elements.push(Expr::Atom(Atom::Name(s.to_string()), span));
            }
        }
    }
    List { elements }
}

/// Reconstruct a `List` from an `UnknownForm` for printing.
fn unknown_form_to_list(head: &str, meta: &MetaMap, children: &[Expr]) -> List {
    let span = Span::new(0, 0);
    let mut elements = Vec::with_capacity(children.len() + 2);
    elements.push(Expr::Atom(Atom::Name(head.to_string()), span));
    elements.push(Expr::Map(meta.clone(), span));
    elements.extend(children.iter().cloned());
    List { elements }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Atom, Expr, MetaExpr, MetaMap};
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

    fn node(tag: &str, meta: Vec<(&str, Expr)>, children: Vec<Expr>) -> Expr {
        use crate::tag::DeepTag;
        let meta_map = MetaMap {
            entries: meta
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        };
        let deep_tag = DeepTag::parse(tag).expect("test uses valid tag");
        Expr::node(deep_tag, meta_map, children, sp())
    }

    fn generic_list(elements: Vec<Expr>) -> Expr {
        Expr::BareList(elements, sp())
    }

    fn meta_expr(entries: Vec<(&str, Expr)>, expr: Expr) -> Expr {
        Expr::MetaExpr(
            MetaExpr {
                entries: entries
                    .into_iter()
                    .map(|(key, value)| (key.to_string(), value))
                    .collect(),
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
                ("z-key", atom_expr(Atom::Int(1))),
                ("a-key", atom_expr(Atom::Int(2))),
            ],
            atom_expr(Atom::Name("body".into())),
        );
        assert_eq!(print_expr(&expr), "^{:a-key 2 :z-key 1} body");
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
}
