//! Canonical printer for the Chelis Deep syntax.
//!
//! Converts `&[Expr]` into a canonical string representation as defined in
//! the Chelis spec §5.

use crate::ast::{Atom, Expr, List, MetaExpr, MetaMap};

const MAX_LINE: usize = 80;

/// Print a slice of top-level expressions in canonical form.
///
/// Expressions are separated by blank lines and the output ends with a
/// single newline.
pub fn print_canonical(exprs: &[Expr]) -> String {
    if exprs.is_empty() {
        return String::new();
    }
    let mut printer = Printer { indent: 0 };
    let parts: Vec<String> = exprs.iter().map(|e| printer.fmt_expr(e)).collect();
    let mut out = parts.join("\n\n");
    out.push('\n');
    out
}

/// Print a single expression in canonical form (no trailing newline).
pub fn print_expr(expr: &Expr) -> String {
    let mut printer = Printer { indent: 0 };
    printer.fmt_expr(expr)
}

struct Printer {
    indent: usize,
}

impl Printer {
    fn fmt_expr(&mut self, expr: &Expr) -> String {
        match expr {
            Expr::Atom(atom, _) => Self::fmt_atom(atom),
            Expr::List(list, _) => self.fmt_list(list),
            Expr::Map(map, _) => Self::fmt_map(map),
            Expr::MetaExpr(meta, _) => self.fmt_meta_expr(meta),
        }
    }

    fn fmt_map(map: &MetaMap) -> String {
        if map.entries.is_empty() {
            return "{}".to_string();
        }
        let mut sorted: Vec<_> = map.entries.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());
        let mut printer = Printer { indent: 0 };
        let parts: Vec<String> = sorted
            .iter()
            .map(|(k, v)| format!("{}: {}", k, printer.fmt_expr(v)))
            .collect();
        format!("{{{}}}", parts.join(", "))
    }

    fn fmt_atom(atom: &Atom) -> String {
        match atom {
            Atom::Symbol(s) => s.clone(),
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
            Atom::Keyword(k) => format!(":{k}"),
            Atom::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        }
    }

    fn fmt_list(&mut self, list: &List) -> String {
        // First try single-line.
        let single = self.fmt_list_single_line(list);
        if self.indent + single.len() <= MAX_LINE {
            return single;
        }
        // Multi-line: tag + first child on opening line, rest indented.
        self.fmt_list_multi_line(list)
    }

    fn fmt_list_single_line(&mut self, list: &List) -> String {
        let parts: Vec<String> = list.elements.iter().map(|e| self.fmt_expr(e)).collect();
        format!("({})", parts.join(" "))
    }

    fn fmt_list_multi_line(&mut self, list: &List) -> String {
        // elements[0] is the "tag", rest are children
        if list.elements.len() <= 1 {
            // Just a tag with no children -- shouldn't reach multi_line, but handle it
            return self.fmt_list_single_line(list);
        }

        let child_indent = self.indent + 2;
        let prefix = " ".repeat(child_indent);
        let old_indent = self.indent;
        self.indent = child_indent;

        let tag_str = self.fmt_expr(&list.elements[0]);

        if list.elements.len() == 2 {
            // (tag single_child)
            let child_str = self.fmt_expr(&list.elements[1]);
            self.indent = old_indent;
            return format!("({} {})", tag_str, child_str);
        }

        // Multiple children: tag + first child on opening line, rest indented
        let mut lines = Vec::new();
        let first_child = self.fmt_expr(&list.elements[1]);
        lines.push(format!("({} {}", tag_str, first_child));

        for elem in &list.elements[2..] {
            lines.push(format!("{}{}", prefix, self.fmt_expr(elem)));
        }

        self.indent = old_indent;
        let last = lines.len() - 1;
        lines[last].push(')');
        lines.join("\n")
    }

    fn fmt_meta_expr(&mut self, meta: &MetaExpr) -> String {
        // Sort entries alphabetically by key.
        let mut sorted: Vec<(&String, &Expr)> = meta.entries.iter().map(|(k, v)| (k, v)).collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        let mut meta_parts = Vec::new();
        for (key, val) in &sorted {
            meta_parts.push(format!(":{}", key));
            meta_parts.push(self.fmt_expr(val));
        }
        let meta_str = format!("^{{{}}}", meta_parts.join(" "));
        let expr_str = self.fmt_expr(&meta.expr);

        let combined = format!("{} {}", meta_str, expr_str);
        if self.indent + combined.len() <= MAX_LINE {
            combined
        } else {
            format!("{}\n{}{}", meta_str, " ".repeat(self.indent), expr_str)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Atom, Expr, List, MetaExpr};
    use crate::span::Span;

    fn sp() -> Span {
        Span { offset: 0, len: 0 }
    }

    fn atom_expr(a: Atom) -> Expr {
        Expr::Atom(a, sp())
    }

    fn list_expr(tag: &str, children: Vec<Expr>) -> Expr {
        let mut elements = vec![atom_expr(Atom::Symbol(tag.to_string()))];
        elements.extend(children);
        Expr::List(List { elements }, sp())
    }

    fn meta_expr(entries: Vec<(&str, Expr)>, expr: Expr) -> Expr {
        Expr::MetaExpr(
            MetaExpr {
                entries: entries
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
                expr: Box::new(expr),
            },
            sp(),
        )
    }

    // --- Atom tests ---

    #[test]
    fn test_atom_symbol() {
        assert_eq!(print_expr(&atom_expr(Atom::Symbol("foo".into()))), "foo");
    }

    #[test]
    fn test_atom_int() {
        assert_eq!(print_expr(&atom_expr(Atom::Int(42))), "42");
        assert_eq!(print_expr(&atom_expr(Atom::Int(0))), "0");
        assert_eq!(print_expr(&atom_expr(Atom::Int(-7))), "-7");
    }

    #[test]
    fn test_atom_float() {
        assert_eq!(print_expr(&atom_expr(Atom::Float(1.0))), "1.0");
        assert_eq!(print_expr(&atom_expr(Atom::Float(3.125))), "3.125");
        assert_eq!(print_expr(&atom_expr(Atom::Float(-0.5))), "-0.5");
    }

    #[test]
    fn test_atom_string() {
        assert_eq!(
            print_expr(&atom_expr(Atom::Str("hello".into()))),
            r#""hello""#
        );
        assert_eq!(
            print_expr(&atom_expr(Atom::Str("line\nnewline".into()))),
            r#""line\nnewline""#
        );
        assert_eq!(
            print_expr(&atom_expr(Atom::Str(r#"a "quote""#.into()))),
            r#""a \"quote\"""#
        );
        assert_eq!(
            print_expr(&atom_expr(Atom::Str("tab\there".into()))),
            r#""tab\there""#
        );
    }

    #[test]
    fn test_atom_keyword() {
        assert_eq!(
            print_expr(&atom_expr(Atom::Keyword("axis".into()))),
            ":axis"
        );
    }

    #[test]
    fn test_atom_bool() {
        assert_eq!(print_expr(&atom_expr(Atom::Bool(true))), "true");
        assert_eq!(print_expr(&atom_expr(Atom::Bool(false))), "false");
    }

    // --- List tests ---

    #[test]
    fn test_simple_list_single_line() {
        let expr = list_expr(
            "add",
            vec![atom_expr(Atom::Int(1)), atom_expr(Atom::Int(2))],
        );
        assert_eq!(print_expr(&expr), "(add 1 2)");
    }

    #[test]
    fn test_empty_list() {
        let expr = list_expr("nop", vec![]);
        assert_eq!(print_expr(&expr), "(nop)");
    }

    #[test]
    fn test_long_list_wraps() {
        // Build a list that exceeds 80 chars on one line.
        let children: Vec<Expr> = (0..10)
            .map(|i| atom_expr(Atom::Symbol(format!("very-long-name-{}", i))))
            .collect();
        let expr = list_expr("define", children);
        let output = print_expr(&expr);
        // Should be multi-line.
        assert!(output.contains('\n'), "expected multi-line output");
        // First line starts with `(define `.
        assert!(output.starts_with("(define "));
        // Subsequent lines indented by 2 spaces.
        let lines: Vec<&str> = output.lines().collect();
        for line in &lines[1..] {
            assert!(
                line.starts_with("  "),
                "expected 2-space indent, got: {:?}",
                line
            );
        }
        // Closing paren at end of last line.
        assert!(output.ends_with(')'));
    }

    // --- Metadata tests ---

    #[test]
    fn test_meta_sorted_keys() {
        let expr = meta_expr(
            vec![
                ("z-key", atom_expr(Atom::Int(1))),
                ("a-key", atom_expr(Atom::Int(2))),
            ],
            atom_expr(Atom::Symbol("body".into())),
        );
        let output = print_expr(&expr);
        assert_eq!(output, "^{:a-key 2 :z-key 1} body");
    }

    // --- Nested structures ---

    #[test]
    fn test_nested_lists() {
        let inner = list_expr(
            "mul",
            vec![atom_expr(Atom::Int(3)), atom_expr(Atom::Int(4))],
        );
        let outer = list_expr("add", vec![atom_expr(Atom::Int(1)), inner]);
        assert_eq!(print_expr(&outer), "(add 1 (mul 3 4))");
    }

    // --- Multiple top-level expressions ---

    #[test]
    fn test_multiple_top_level() {
        let exprs = vec![
            atom_expr(Atom::Int(1)),
            atom_expr(Atom::Int(2)),
            atom_expr(Atom::Int(3)),
        ];
        let output = print_canonical(&exprs);
        assert_eq!(output, "1\n\n2\n\n3\n");
    }

    #[test]
    fn test_empty_input() {
        let output = print_canonical(&[]);
        assert_eq!(output, "");
    }

    #[test]
    fn test_single_top_level() {
        let exprs = vec![atom_expr(Atom::Symbol("hello".into()))];
        let output = print_canonical(&exprs);
        assert_eq!(output, "hello\n");
    }
}
