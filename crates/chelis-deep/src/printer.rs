//! Canonical printer for the Chelis Deep syntax.
//!
//! Converts `&[Expr]` into the canonical string representation used for `.dp`
//! files and `chelis deep`.

#[cfg(test)]
use alloc::{boxed::Box, string::ToString};
use alloc::{string::String, vec::Vec};

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
    print_program(exprs, |wire, out| printer.write_pretty(wire, out))
}

/// Print a slice of top-level expressions in flat canonical form.
pub fn print_canonical_flat(exprs: &[Expr]) -> String {
    print_program(exprs, |wire, out| write_flat(Flat::Expr(wire), out))
}

/// Print a single expression in canonical pretty form (no trailing newline).
pub fn print_expr(expr: &Expr) -> String {
    with_wire(expr, |wire| {
        let mut out = String::new();
        Printer::new().write_pretty(wire, &mut out);
        out
    })
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
        write_flat_step(item, out, &mut stack);
    }
}

/// Write one pending piece of flat output, queueing the pieces it contains.
fn write_flat_step<'a>(item: Flat<'a>, out: &mut String, stack: &mut Vec<Flat<'a>>) {
    match item {
        Flat::Text(text) => out.push_str(text),
        Flat::Expr(expr) => match expr {
            WireExpr::ExtensionData(data) => write_extension(out, data.syntax()),
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
                open_form(&form.head, &form.meta, &form.children, out, stack);
            }
        },
        Flat::Node(node) => {
            open_form(node.tag.as_str(), &node.meta, &node.children, out, stack);
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

/// Append the flat canonical form of `item` to `out` if it is at most
/// `budget` bytes wide, and report whether it was. A form that does not fit
/// is abandoned as soon as it passes the budget and `out` is left as it was,
/// so measuring a node's width costs at most the budget rather than the
/// node's whole subtree (chelis#3852).
fn write_flat_within(item: Flat<'_>, out: &mut String, budget: Option<usize>) -> bool {
    let Some(budget) = budget else {
        return false;
    };
    let start = out.len();
    let mut stack = vec![item];
    while let Some(item) = stack.pop() {
        write_flat_step(item, out, &mut stack);
        if out.len() - start > budget {
            out.truncate(start);
            return false;
        }
    }
    true
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
    use core::fmt::Write as _;
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
                    control if control.is_control() => write_control_escape(out, control),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Atom::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
    }
}

/// The canonical text of one atom.
#[cfg(test)]
fn atom_text(atom: &Atom) -> String {
    let mut out = String::new();
    write_atom(&mut out, atom);
    out
}

/// The pretty layout, written once into one buffer.
///
/// An earlier layout rendered each child to its own string and copied it,
/// split into lines, into every enclosing level, so a byte nested `d` deep
/// was copied `d` times, and it rendered each node's whole subtree flat to
/// compare it with the line width. On deep nesting that was cubic in the
/// depth (chelis#3852). Here each byte is written once, and a width test
/// stops at the line width.
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
}

impl Printer {
    fn write_pretty(&self, expr: &WireExpr, out: &mut String) {
        self.write_expr(expr, 0, out);
    }

    /// The width left on a line at `indent`, or `None` when nothing fits.
    fn budget(&self, indent: usize) -> Option<usize> {
        self.max_width.checked_sub(indent)
    }

    fn write_expr(&self, expr: &WireExpr, indent: usize, out: &mut String) {
        match expr {
            WireExpr::ExtensionData(data) => write_extension(out, data.syntax()),
            WireExpr::Atom(atom, _) => write_atom(out, atom),
            WireExpr::Map(map, _) => self.write_map(map, indent, out),
            WireExpr::MetaExpr(meta, _) => self.write_meta_expr(meta, indent, out),
            WireExpr::Node(node, _) => self.write_node(node, indent, out),
            WireExpr::BareList(items, _) => self.write_list(expr, items, indent, out),
            WireExpr::UnknownForm(form) => self.write_unknown_form(expr, form, indent, out),
        }
    }

    fn write_node(&self, node: &WireNode, indent: usize, out: &mut String) {
        if force_flat_tag(node.tag) {
            write_flat(Flat::Node(node), out);
            return;
        }
        if !force_break_tag(node.tag)
            && write_flat_within(Flat::Node(node), out, self.budget(indent))
        {
            return;
        }
        let tag = node.tag.as_str();
        out.push('(');
        out.push_str(tag);
        out.push(' ');
        self.write_map(&node.meta, indent + 1 + tag.len() + 1, out);
        let child_indent = indent + self.indent_step;
        for child in &node.children {
            self.write_child_line(child_indent, out, |out| {
                self.write_expr(child, child_indent, out);
            });
        }
        out.push(')');
    }

    fn write_list(&self, list: &WireExpr, items: &[WireExpr], indent: usize, out: &mut String) {
        let Some((first, rest)) = items.split_first() else {
            write_flat(Flat::Expr(list), out);
            return;
        };
        if write_flat_within(Flat::Expr(list), out, self.budget(indent)) {
            return;
        }
        out.push('(');
        self.write_expr(first, indent + self.indent_step, out);
        self.write_broken_items(None, rest, indent, out);
    }

    fn write_unknown_form(
        &self,
        form_expr: &WireExpr,
        form: &WireUnknown,
        indent: usize,
        out: &mut String,
    ) {
        if write_flat_within(Flat::Expr(form_expr), out, self.budget(indent)) {
            return;
        }
        out.push('(');
        out.push_str(&form.head);
        self.write_broken_items(Some(&form.meta), &form.children, indent, out);
    }

    /// The lines of a broken list after its first element.
    fn write_broken_items(
        &self,
        meta: Option<&MetaMap>,
        items: &[WireExpr],
        indent: usize,
        out: &mut String,
    ) {
        let child_indent = indent + self.indent_step;
        if let Some(meta) = meta {
            self.write_child_line(child_indent, out, |out| {
                self.write_map(meta, child_indent, out);
            });
        }
        for item in items {
            self.write_child_line(child_indent, out, |out| {
                self.write_expr(item, child_indent, out);
            });
        }
        out.push(')');
    }

    fn write_map(&self, map: &MetaMap, indent: usize, out: &mut String) {
        if map.entries.is_empty() {
            write_flat(Flat::Map(map), out);
            return;
        }
        if write_flat_within(Flat::Map(map), out, self.budget(indent)) {
            return;
        }
        let sorted = sorted_entries(&map.entries);
        let entry_indent = indent + self.indent_step;
        for (index, (key, value)) in sorted.iter().enumerate() {
            if index == 0 {
                out.push('{');
            } else {
                push_line_start(out, entry_indent);
            }
            out.push_str(key);
            out.push_str(": ");
            self.write_expr(value, entry_indent + key.len() + 2, out);
            if index + 1 != sorted.len() {
                out.push(',');
            }
        }
        push_line_start(out, indent);
        out.push('}');
    }

    fn write_meta_expr(&self, meta: &MetaExpr, indent: usize, out: &mut String) {
        if write_flat_within(Flat::MetaExpr(meta), out, self.budget(indent)) {
            return;
        }
        let start = out.len();
        write_flat(Flat::MetaEntries(&meta.entries), out);
        let expr_indent = indent + (out.len() - start) + 1;
        out.push(' ');
        self.write_expr(&meta.expr, expr_indent, out);
    }

    /// Put a child on a line of its own, indented to `indent`.
    fn write_child_line(&self, indent: usize, out: &mut String, write: impl FnOnce(&mut String)) {
        push_line_start(out, indent);
        write(out);
    }
}

/// Start a new line indented to `indent`.
fn push_line_start(out: &mut String, indent: usize) {
    out.push('\n');
    out.extend(core::iter::repeat_n(' ', indent));
}

/// Write extension data's spelling, with every raw control character inside
/// a string token written as its canonical String escape (spec/03 §1.1). The
/// tokens and the separators between them are already canonical, and an
/// escape the author wrote is kept as written, so the decoded data is the
/// same; canonical Deep therefore holds no raw line break or carriage
/// return inside extension data, which a line-based reader would otherwise
/// split or drop (chelis#3863).
fn write_extension(out: &mut String, syntax: &str) {
    let mut in_string = false;
    let mut escaped = false;
    for ch in syntax.chars() {
        if !in_string {
            in_string = ch == '"';
            out.push(ch);
            continue;
        }
        if escaped {
            escaped = false;
            out.push(ch);
            continue;
        }
        match ch {
            '\\' => {
                escaped = true;
                out.push(ch);
            }
            '"' => {
                in_string = false;
                out.push(ch);
            }
            control if control.is_control() => write_control_escape(out, control),
            other => out.push(other),
        }
    }
}

/// The canonical String escape of a control character (spec/03 §6.4): its
/// named escape where one exists, otherwise minimal lowercase `\u{h}`.
fn write_control_escape(out: &mut String, control: char) {
    use core::fmt::Write as _;
    match control {
        '\n' => out.push_str("\\n"),
        '\t' => out.push_str("\\t"),
        '\r' => out.push_str("\\r"),
        '\0' => out.push_str("\\0"),
        other => {
            write!(out, "\\u{{{:x}}}", u32::from(other)).expect("writing to a String cannot fail")
        }
    }
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

    /// The layout before chelis#3852, which returned each child's text and
    /// split it into lines at every enclosing level. Kept only as the byte
    /// oracle the streaming layout must reproduce.
    struct Returning {
        max_width: usize,
        indent_step: usize,
    }

    impl Returning {
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
        fn fmt_unknown_form(
            &self,
            form_expr: &WireExpr,
            form: &WireUnknown,
            indent: usize,
        ) -> String {
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
            // An empty map has no entry to break before, so `{}` stays flat at any
            // indent; breaking it would print only its closing brace.
            if map.entries.is_empty() || indent + flat.len() <= self.max_width {
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

        fn fmt_meta_expr(&self, meta: &super::MetaExpr, indent: usize) -> String {
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

    fn returning_layout(exprs: &[Expr]) -> String {
        let printer = Returning {
            max_width: MAX_LINE,
            indent_step: INDENT_STEP,
        };
        print_program(exprs, |wire, out| out.push_str(&printer.fmt_expr(wire, 0)))
    }

    fn streaming_layout(exprs: &[Expr]) -> String {
        print_canonical(exprs)
    }

    const I32: &str = "(lit {type: (t-prim {} i32)} 1)";

    fn tuple_chain(depth: usize) -> String {
        let body = format!("(tuple {{}} {I32} ").repeat(depth) + I32 + &")".repeat(depth);
        format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
    }

    fn let_chain(depth: usize) -> String {
        let mut body = String::new();
        for index in 0..depth {
            body.push_str(&format!("(let {{}} (bind {{}} x{index} {I32}) "));
        }
        body.push_str("(var {} x0)");
        body.push_str(&")".repeat(depth));
        format!("(def {{}} main (fn {{}} (params {{}}) {body}))\n")
    }

    fn deep_type(depth: usize) -> String {
        "(t-tuple {} (t-prim {} f32) ".repeat(depth) + "(t-prim {} i64)" + &")".repeat(depth)
    }

    /// Typed parameters and an annotated body: maps that must break.
    fn typed_program(depth: usize) -> String {
        let ty = deep_type(depth);
        format!(
            "(def {{type: (t-fn {{}} {ty} {ty})}} f (fn {{}} (params {{}} (x {{type: {ty}}})) \
             (var {{type: {ty}}} x)))\n"
        )
    }

    /// Structural lists, unknown forms, legacy prefix metadata, extension
    /// data and long names, as list Deep.
    fn list_program(width: usize, seed: u64) -> String {
        let mut state = seed;
        let mut next = move |bound: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % bound
        };
        let mut out = String::new();
        for item in 0..width {
            let name = "n".repeat(1 + next(90) as usize);
            match next(6) {
                0 => out.push_str(&format!(
                    "({name} {item} \"s{item}\" {}) ",
                    "(a b) ".repeat(next(12) as usize)
                )),
                1 => out.push_str(&format!(
                    "(frob {{custom: ({}), k{item}: {name}}} {name}) ",
                    "x ".repeat(next(30) as usize)
                )),
                2 => out.push_str(&format!(
                    "^{{:custom ({}) :doc \"d\"}} {name} ",
                    "y ".repeat(next(30) as usize)
                )),
                3 => out.push_str(&format!(
                    "{{b: {item}, a: ({}), c: {name}}} ",
                    "z ".repeat(next(40) as usize)
                )),
                4 => out.push_str(&format!(
                    "({}{}) ",
                    "(".repeat(next(8) as usize),
                    ")".repeat(0)
                )),
                _ => out.push_str(&format!("{name} ")),
            }
        }
        let balanced: String = {
            // Close any parenthesis case 4 opened.
            let open = out.matches('(').count();
            let close = out.matches(')').count();
            out.clone() + &")".repeat(open.saturating_sub(close))
        };
        format!("(top {balanced})\n")
    }

    /// A seeded generator of arbitrary trees for the byte oracle: every node
    /// shape the layout breaks, zero-child nodes included, metadata maps
    /// and prefix metadata whose extension spellings hold carriage returns,
    /// line feeds and multibyte text, and names and strings that cross the
    /// line width.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) % bound.max(1)
        }
        fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
            &xs[self.next(xs.len() as u64) as usize]
        }
    }
    const PIECES: &[&str] = &[
        "a",
        "b",
        "z9",
        "\u{e9}",
        "\u{4e2d}",
        "\u{2460}",
        "\u{1f600}",
        " ",
        "\t",
        "-",
        "_",
    ];
    /// Control characters without a named escape: C0, DEL and C1.
    const CONTROLS: &[&str] = &[
        "\u{1}", "\u{8}", "\u{1b}", "\u{1f}", "\u{7f}", "\u{85}", "\u{9f}",
    ];
    std::thread_local! { /// Generated extension lists whose separators held a control character.
    static CONTROL_SEPARATORS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
    const RAW_PIECES: &[&str] = &[
        "\r", "\n", "\r\n", "\r\r\n", "\t", "\u{e9}", "\u{4e2d}", "\u{2460}", "q", "w", " ",
        "\n\n", "\u{1}", "\u{1b}", "\\n", "\\r",
    ];
    fn rname(r: &mut Rng) -> String {
        let n = 1 + {
            let c = r.next(4) == 0;
            r.next(if c { 70 } else { 12 })
        } as usize;
        let mut s = String::from("n");
        for _ in 1..n {
            s.push_str(match r.next(10) {
                0 => "\u{e9}",
                1 => "_",
                2 => "9",
                _ => "x",
            });
        }
        s
    }
    fn rstr(r: &mut Rng) -> String {
        let n = {
            let c = r.next(4) == 0;
            r.next(if c { 60 } else { 10 })
        } as usize;
        (0..n).map(|_| r.pick(PIECES).to_string()).collect()
    }
    fn rext(r: &mut Rng, depth: u32) -> String {
        match r.next(if depth > 2 { 3 } else { 7 }) {
            0 => {
                let n = {
                    let c = r.next(3) == 0;
                    r.next(if c { 70 } else { 8 })
                } as usize;
                let body: String = (0..n).map(|_| r.pick(RAW_PIECES).to_string()).collect();
                format!("\"{body}\"")
            }
            1 => rname(r),
            2 => format!("{}", r.next(100000)),
            3 => {
                let n = r.next(5);
                let items: Vec<String> = (0..n).map(|_| rext(r, depth + 1)).collect();
                let sep = *r.pick(&[" ", "\n ", "\r\n  ", " ", "\t", "\r\n\t"]);
                if n > 1 && sep.chars().any(char::is_control) {
                    CONTROL_SEPARATORS.with(|count| count.set(count.get() + 1));
                }
                format!("({})", items.join(sep))
            }
            4 => {
                let n = r.next(4);
                let items: Vec<String> = (0..n)
                    .map(|i| format!("k{i}: {}", rext(r, depth + 1)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
            5 => format!("^{{:k {}}} {}", rext(r, depth + 1), rext(r, depth + 1)),
            _ => format!("\"{}\"", rstr(r).replace('\t', "\\t")),
        }
    }
    fn rmeta(r: &mut Rng, depth: u32) -> Metadata {
        let mut m = Metadata::default();
        let n = match r.next(4) {
            0 | 1 => 0,
            2 => 1,
            _ => 1 + r.next(4),
        };
        for i in 0..n {
            let key = format!(
                "{}{}",
                r.pick(&["custom", "k", "zz", "a_very_long_metadata_key_name"]),
                i
            );
            let data = if r.next(8) == 0 {
                // Spellings parsing cannot produce: empty, ending in a line
                // feed, ending in a carriage return.
                Some(crate::ExtensionData::with_spelling_for_tests(
                    r.pick(&["", "tail\n", "tail\r", "a\r\n", "\n"]),
                ))
            } else {
                crate::ExtensionData::parse(&rext(r, depth)).ok()
            };
            if let Some(data) = data {
                let _ = m.extensions_mut().insert(key, data);
            }
        }
        if r.next(5) == 0
            && let Ok(ty) = crate::annotations::TypeSyntax::try_new(rtype(r, 0))
        {
            let _ = m.insert(crate::annotations::MetadataValue::Type(ty));
        }
        m
    }
    /// A type built from `t-prim` and `t-tuple`, the structured type maps
    /// real programs break.
    fn rtype(r: &mut Rng, depth: u32) -> Expr {
        if depth > 2 || r.next(3) == 0 {
            return Expr::node(
                DeepTag::TPrim,
                Metadata::default(),
                vec![sym(r.pick(&["f32", "i32", "i64", "bool"]))],
                sp(),
            );
        }
        let width = 1 + r.next(3) as usize;
        Expr::node(
            DeepTag::TTuple,
            Metadata::default(),
            (0..width).map(|_| rtype(r, depth + 1)).collect(),
            sp(),
        )
    }
    fn ratom(r: &mut Rng) -> Expr {
        match r.next(6) {
            0 => sym(&rname(r)),
            // Names parsing cannot produce: empty, ending in a line feed,
            // ending in a carriage return that meets the next line break.
            5 => sym(r.pick(&[
                "",
                "tail\n",
                "tail\r",
                "mid\r\nname",
                "esc\u{1b}",
                "del\u{7f}",
            ])),
            1 => atom_expr(Atom::Str(
                rstr(r)
                    + if r.next(3) == 0 { "\r\n\0\"\\" } else { "" }
                    + if r.next(3) == 0 { r.pick(CONTROLS) } else { "" },
            )),
            2 => atom_expr(Atom::Int(r.next(1_000_000_000) as i64 - 500)),
            3 => atom_expr(Atom::Float(r.next(10000) as f64 / 7.0)),
            _ => atom_expr(Atom::Bool(r.next(2) == 0)),
        }
    }
    fn rexpr(r: &mut Rng, depth: u32) -> Expr {
        if depth > 7 || r.next(10) < 3 {
            return ratom(r);
        }
        let nkids = {
            let c = r.next(3) == 0;
            r.next(if c { 6 } else { 3 })
        } as usize;
        match r.next(8) {
            0..=2 => {
                let tag = *r.pick(&[
                    DeepTag::Module,
                    DeepTag::Tuple,
                    DeepTag::TTuple,
                    DeepTag::Params,
                    DeepTag::Bind,
                    DeepTag::Effects,
                    DeepTag::PatTuple,
                    DeepTag::App,
                    DeepTag::Var,
                    DeepTag::TPrim,
                    DeepTag::Fn,
                    DeepTag::Let,
                    DeepTag::TUnit,
                    DeepTag::PatWild,
                    DeepTag::TFn,
                ]);
                let n = match tag {
                    DeepTag::Var | DeepTag::TPrim => 1,
                    DeepTag::Fn | DeepTag::Let => 2,
                    DeepTag::TUnit | DeepTag::PatWild => 0,
                    _ => nkids,
                };
                let kids: Vec<Expr> = (0..n)
                    .map(|_| {
                        if matches!(tag, DeepTag::Var | DeepTag::TPrim) {
                            sym(&rname(r))
                        } else {
                            rexpr(r, depth + 1)
                        }
                    })
                    .collect();
                let meta = rmeta(r, depth);
                match crate::node::Node::try_new(tag, meta, kids) {
                    Ok(n) => Expr::Node(Box::new(n), sp()),
                    Err(_) => ratom(r),
                }
            }
            3 => Expr::BareList((0..nkids).map(|_| rexpr(r, depth + 1)).collect(), sp()),
            4 => Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
                head: rname(r),
                meta: rmeta(r, depth),
                children: (0..nkids).map(|_| rexpr(r, depth + 1)).collect(),
                span: sp(),
            })),
            5 => Expr::Map(rmeta(r, depth), sp()),
            _ => Expr::MetaExpr(
                MetaExpr {
                    metadata: rmeta(r, depth),
                    expr: Box::new(rexpr(r, depth + 1)),
                },
                sp(),
            ),
        }
    }

    /// Where an extension value sits: in a node's metadata, a metadata map,
    /// legacy prefix metadata, or an unknown form's metadata.
    const OWNERS: [&str; 4] = ["node", "map", "prefix", "unknown form"];

    /// Every extension value in `exprs`, in tree order, as its lexed tokens,
    /// so two spellings of the same data compare equal; and, for each value
    /// holding a raw carriage return, its owner and nesting depth.
    fn extension_data(exprs: &[Expr]) -> (Vec<Vec<crate::lexer::TokenKind>>, Vec<(usize, usize)>) {
        let mut data = Vec::new();
        let mut carriage_returns = Vec::new();
        let mut pending: Vec<(&Expr, usize)> = exprs.iter().rev().map(|expr| (expr, 0)).collect();
        let mut note = |meta: &Metadata, owner: usize, depth: usize| {
            for (_, value) in meta.extensions().iter() {
                if value.syntax().contains('\r') {
                    carriage_returns.push((owner, depth));
                }
                data.push(
                    crate::lexer::lex(value.syntax())
                        .map(|tokens| tokens.into_iter().map(|token| token.kind).collect())
                        .unwrap_or_default(),
                );
            }
        };
        while let Some((expr, depth)) = pending.pop() {
            let children: Vec<&Expr> = match expr {
                Expr::Atom(..) => Vec::new(),
                Expr::Node(node, _) => {
                    note(node.meta(), 0, depth);
                    node.children_slice().iter().collect()
                }
                Expr::Map(meta, _) => {
                    note(meta, 1, depth);
                    Vec::new()
                }
                Expr::MetaExpr(meta, _) => {
                    note(&meta.metadata, 2, depth);
                    vec![&*meta.expr]
                }
                Expr::BareList(items, _) => items.iter().collect(),
                Expr::UnknownForm(form) => {
                    note(&form.meta, 3, depth);
                    form.children.iter().collect()
                }
            };
            pending.extend(children.into_iter().rev().map(|child| (child, depth + 1)));
        }
        (data, carriage_returns)
    }

    /// Whether `exprs` holds a leaf that parsing cannot produce: a name that
    /// is empty or holds a control character, or an extension spelling the
    /// extension parser would not give back.
    fn has_api_only_leaf(exprs: &[Expr]) -> bool {
        let mut found = false;
        let mut pending: Vec<&Expr> = exprs.iter().collect();
        let check_meta = |meta: &Metadata| {
            for (_, value) in meta.extensions().iter() {
                if crate::ExtensionData::parse(value.syntax())
                    .map_or(true, |parsed| parsed.syntax() != value.syntax())
                {
                    return true;
                }
            }
            false
        };
        // A name parses back only if it lexes as itself.
        let unlexable = |name: &str| {
            !matches!(
                crate::lexer::lex(name).as_deref(),
                Ok([crate::lexer::Token { kind: crate::lexer::TokenKind::Symbol(lexed), .. }])
                    if lexed == name
            )
        };
        while let Some(expr) = pending.pop() {
            match expr {
                Expr::Atom(Atom::Name(name), _) => found |= unlexable(name),
                Expr::Atom(..) => {}
                Expr::Node(node, _) => {
                    found |= check_meta(node.meta());
                    pending.extend(node.children_slice());
                }
                Expr::Map(meta, _) => found |= check_meta(meta),
                Expr::MetaExpr(meta, _) => {
                    found |= check_meta(&meta.metadata);
                    pending.push(&meta.expr);
                }
                Expr::BareList(items, _) => pending.extend(items),
                Expr::UnknownForm(form) => {
                    found |= check_meta(&form.meta) || unlexable(&form.head);
                    pending.extend(&form.children);
                }
            }
        }
        found
    }

    /// Whether some extension value spells a raw control character.
    fn has_raw_control_in_extension(exprs: &[Expr]) -> bool {
        exprs.iter().any(|expr| {
            with_wire(expr, |wire| {
                let mut found = false;
                fn visit(expr: &WireExpr, found: &mut bool) {
                    let meta = match expr {
                        WireExpr::ExtensionData(data) => {
                            *found |= data.syntax().chars().any(char::is_control);
                            return;
                        }
                        WireExpr::Map(map, _) => Some(map),
                        WireExpr::Node(node, _) => Some(&node.meta),
                        WireExpr::UnknownForm(form) => Some(&form.meta),
                        _ => None,
                    };
                    if let Some(meta) = meta {
                        for (_, value) in &meta.entries {
                            visit(value, found);
                        }
                    }
                    match expr {
                        WireExpr::Node(node, _) => {
                            node.children.iter().for_each(|c| visit(c, found))
                        }
                        WireExpr::UnknownForm(form) => {
                            form.children.iter().for_each(|c| visit(c, found))
                        }
                        WireExpr::BareList(items, _) => items.iter().for_each(|c| visit(c, found)),
                        WireExpr::MetaExpr(meta, _) => {
                            for (_, value) in &meta.entries {
                                visit(value, found);
                            }
                            visit(&meta.expr, found);
                        }
                        _ => {}
                    }
                }
                visit(wire, &mut found);
                found
            })
        })
    }

    /// The properties canonical printing must have on `exprs` (chelis#3852,
    /// chelis#3863), or the first one that fails. Reports whether the
    /// round-trip properties applied: they need a tree the parser can
    /// produce.
    fn printing_properties(exprs: &[Expr]) -> Result<bool, String> {
        let printed = print_canonical(exprs);
        for expr in exprs {
            if print_expr(expr) + "\n" != print_canonical(core::slice::from_ref(expr)) {
                return Err("print_expr disagrees with print_canonical".into());
            }
        }
        if has_api_only_leaf(exprs) {
            // Only the Rust API builds these, and they are printed as they
            // are; the properties below are about parsed data.
            return Ok(false);
        }
        // (c) The style gate reads canonical Deep as LF-only text.
        if printed.contains('\r') {
            return Err(format!("raw carriage return in {printed:?}"));
        }
        // (d) With no raw control character to escape, the bytes are the
        // returning layout's.
        if !has_raw_control_in_extension(exprs) && printed != returning_layout(exprs) {
            return Err(format!("layout changed: {printed:?}"));
        }
        // A tree the parser can produce: its flat printing parses back to a
        // tree that prints the same. Other shapes only the Rust API builds.
        let flat = print_canonical_flat(exprs);
        if crate::parser::parse_str(&flat, &chelis_syntax_host::NativeStack)
            .map_or(true, |parsed| print_canonical_flat(&parsed) != flat)
        {
            return Ok(false);
        }
        // (a) The printed text parses back to the same data.
        let reparsed = crate::parser::parse_str(&printed, &chelis_syntax_host::NativeStack)
            .map_err(|error| format!("{error:?} reparsing {printed:?}"))?;
        if extension_data(&reparsed).0 != extension_data(exprs).0 {
            return Err(format!("extension data changed through {printed:?}"));
        }
        if print_canonical_flat(&reparsed) != flat {
            return Err(format!("the tree changed through {printed:?}"));
        }
        // (b) Printing is idempotent.
        if print_canonical(&reparsed) != printed {
            return Err(format!("not idempotent on {printed:?}"));
        }
        Ok(true)
    }

    #[test]
    fn issue_3852_generated_trees_have_every_printing_property() {
        let mut failures = Vec::new();
        let mut round_trips = 0;
        let mut reach = [[0usize; 8]; OWNERS.len()];
        CONTROL_SEPARATORS.with(|count| count.set(0));
        for seed in 0..10_000u64 {
            let mut r = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xdead);
            let count = 1 + r.next(3) as usize;
            let exprs: Vec<Expr> = (0..count).map(|_| rexpr(&mut r, 0)).collect();
            if !has_api_only_leaf(&exprs) {
                for (owner, depth) in extension_data(&exprs).1 {
                    reach[owner][depth.min(7)] += 1;
                }
            }
            match printing_properties(&exprs) {
                Ok(true) => round_trips += 1,
                Ok(false) => {}
                Err(failure) => {
                    if failures.len() < 3 {
                        failures.push(format!("seed {seed}: {failure}"));
                    }
                }
            }
        }
        std::eprintln!("carriage-return extension values by owner {OWNERS:?} and depth: {reach:?}");
        std::eprintln!("trees that round-trip through the parser: {round_trips}");
        let separators = CONTROL_SEPARATORS.with(std::cell::Cell::get);
        std::eprintln!("extension lists separated by control characters: {separators}");
        assert!(
            separators > 500,
            "only {separators} control-character separators"
        );
        assert!(
            round_trips > 2000,
            "only {round_trips} trees reached the round trip"
        );
        // Carriage returns reach every owner, and every depth through 5.
        for (owner, depths) in OWNERS.iter().zip(reach) {
            assert!(
                depths.iter().any(|&count| count > 0),
                "no carriage return in a {owner}"
            );
        }
        for depth in 0..=5 {
            assert!(
                reach.iter().map(|depths| depths[depth]).sum::<usize>() > 0,
                "no carriage return at depth {depth}"
            );
        }
        assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    }

    #[test]
    fn issue_3852_widths_at_the_line_limit_have_every_printing_property() {
        // A node whose flat text is one byte under, at, and one byte over the
        // width left at its indent, nested at several indents, once with a
        // carriage return in its metadata and once without.
        let custom = |text: &str| {
            let mut metadata = Metadata::default();
            metadata
                .extensions_mut()
                .insert(
                    "custom".into(),
                    crate::ExtensionData::parse(&format!("\"{text}\"")).unwrap(),
                )
                .unwrap();
            metadata
        };
        let mut compared = 0;
        for text in ["a\r\nb", "a--b"] {
            for nesting in 0..5usize {
                let indent = nesting * INDENT_STEP;
                for over in [-1i64, 0, 1] {
                    let target = (MAX_LINE - indent) as i64 + over;
                    for children in [0usize, 1, 2] {
                        let build = |pad: usize| {
                            Expr::node(
                                DeepTag::Tuple,
                                custom(&format!("{text}{}", "x".repeat(pad))),
                                (0..children).map(|_| atom_expr(Atom::Int(7))).collect(),
                                sp(),
                            )
                        };
                        let base = print_expr_flat(&build(0)).len() as i64;
                        let Ok(pad) = usize::try_from(target - base) else {
                            continue;
                        };
                        let node = build(pad);
                        assert_eq!(print_expr_flat(&node).len() as i64, target);
                        let mut expr = node;
                        for _ in 0..nesting {
                            expr = generic_list(vec![sym(&"w".repeat(90)), expr]);
                        }
                        let exprs = vec![expr];
                        let round_trip = printing_properties(&exprs).unwrap_or_else(|failure| {
                            panic!(
                                "nesting {nesting}, width {target}, {children} children: {failure}"
                            )
                        });
                        assert!(round_trip, "the sweep's trees parse back");
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared >= 80, "compared {compared} widths");
        // The zero-child node whose 72-byte map takes it to 81.
        let module = Expr::node(
            DeepTag::Module,
            custom(&format!("a\r\nb{}", "x".repeat(56))),
            vec![],
            sp(),
        );
        assert!(printing_properties(&[module]).unwrap());
    }

    #[test]
    fn issue_3852_streaming_layout_reproduces_the_returning_layout() {
        let mut stamped = Vec::new();
        for depth in [0, 1, 2, 3, 5, 8, 13, 21, 40, 80] {
            stamped.push(tuple_chain(depth));
            stamped.push(let_chain(depth));
            stamped.push(typed_program(depth));
        }
        let mut compared = 0;
        let on_large_stack = std::thread::Builder::new().stack_size(256 * 1024 * 1024);
        let handle = on_large_stack
            .spawn(move || {
                for source in &stamped {
                    let exprs = crate::parser::parse_and_stamp_file(
                        source,
                        &chelis_syntax_host::NativeStack,
                    )
                    .expect("stamped probe parses");
                    assert_eq!(
                        streaming_layout(&exprs),
                        returning_layout(&exprs),
                        "{source}"
                    );
                    compared += 1;
                }
                for width in [1, 2, 4, 8, 16, 32] {
                    for seed in 0..40 {
                        let source = list_program(width, seed);
                        let exprs =
                            crate::parser::parse_str(&source, &chelis_syntax_host::NativeStack)
                                .expect("list probe parses");
                        assert_eq!(
                            streaming_layout(&exprs),
                            returning_layout(&exprs),
                            "{source}"
                        );
                        compared += 1;
                    }
                }
                compared
            })
            .expect("spawn the comparing thread");
        let compared = handle.join().expect("the comparison finishes");
        assert!(compared > 250, "compared {compared} programs");
    }

    #[test]
    fn issue_3863_extension_strings_print_their_control_characters_escaped() {
        // A raw carriage return, line feed or tab inside an extension string
        // prints as its named escape at every layout position, and an escape
        // the author wrote keeps its spelling.
        let long = "w ".repeat(40);
        for raw in [
            "\"a\r\nb\"",
            "\"a\nb\"",
            "\"a\r\r\nb\"",
            "\"t\tab\"",
            "\"kept\\r\\n\"",
        ] {
            for source in [
                format!("(def {{}} x (lit {{custom: {raw}}} 1))\n"),
                format!(
                    "(def {{}} x (tuple {{}} (lit {{custom: {raw}}} 1) (var {{}} {})))\n",
                    "y".repeat(60)
                ),
                format!("(top {{custom: {raw}}} ({long}) {raw})\n"),
                format!(
                    "(module {{custom: {raw}{}}})\n",
                    "x".repeat(56).replace('x', "")
                ),
                format!("(module {{custom: (a {raw} \"{}\")}})\n", "x".repeat(56)),
            ] {
                let exprs = crate::parser::parse_str(&source, &chelis_syntax_host::NativeStack)
                    .expect("the probe parses");
                printing_properties(&exprs)
                    .unwrap_or_else(|failure| panic!("{source:?}: {failure}"));
                let printed = print_canonical(&exprs);
                assert!(
                    !printed.contains('\t') || !raw.contains('\t'),
                    "{printed:?}"
                );
                if raw.contains("\\r") {
                    assert!(printed.contains("kept\\r\\n"), "{printed:?}");
                }
            }
        }
        // The round-1 witness: a broken zero-child module.
        let witness = format!("(module {{custom: \"a\r\nb{}\"}})\n", "x".repeat(56));
        let exprs = crate::parser::parse_str(&witness, &chelis_syntax_host::NativeStack)
            .expect("the witness parses");
        let printed = print_canonical(&exprs);
        assert!(printed.contains("a\\r\\nb"), "{printed:?}");
        assert!(printing_properties(&exprs).unwrap());
    }

    #[test]
    fn issue_3863_strings_print_the_canonical_escape_of_every_control_character() {
        // A string atom: named escapes where they exist, minimal lowercase
        // `\u{h}` for every other control character, and a `\u{h}` spelling
        // in the input, padded or uppercase, read as its value.
        let source =
            "(def {} s (lit {} \"\u{0041}\\u{000A}\\u{7F}\u{1b}\\u{85}\u{9f}\\u{1F600}\"))\n";
        let exprs = crate::parser::parse_str(source, &chelis_syntax_host::NativeStack)
            .expect("the string parses");
        let printed = print_canonical(&exprs);
        assert_eq!(
            printed,
            "(def {} s (lit {} \"A\\n\\u{7f}\\u{1b}\\u{85}\\u{9f}\u{1f600}\"))\n"
        );
        assert!(printing_properties(&exprs).unwrap());
        // In extension data the author's escape keeps its spelling, and a raw
        // control character takes its canonical escape.
        let source = "(def {} x (lit {custom: \"\\u{0041}\u{1b}\u{7f}\\u{A}\u{85}\"} 1))\n";
        let exprs = crate::parser::parse_str(source, &chelis_syntax_host::NativeStack)
            .expect("the extension parses");
        let printed = print_canonical(&exprs);
        assert!(
            printed.contains("\"\\u{0041}\\u{1b}\\u{7f}\\u{A}\\u{85}\""),
            "{printed:?}"
        );
        assert!(printing_properties(&exprs).unwrap());
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
    fn empty_metadata_past_the_line_width_prints_flat_and_reparses() {
        // Nest broken `app` nodes until a `params` node's empty metadata
        // starts beyond the line width, as deeply nested Surf produces.
        let mut expr = node(
            "fn",
            vec![],
            vec![
                node("params", vec![], vec![sym("__chelis_pipe")]),
                node("var", vec![], vec![sym("__chelis_pipe")]),
            ],
        );
        for _ in 0..45 {
            expr = node(
                "app",
                vec![],
                vec![
                    node("var", vec![], vec![sym("a_long_enough_function_name")]),
                    expr,
                ],
            );
        }
        let printed = print_expr(&expr);
        assert!(
            printed.contains("(params {}"),
            "empty metadata must keep both braces:\n{printed}"
        );
        let reparsed = crate::parser::parse_str(&printed, &chelis_syntax_host::NativeStack)
            .expect("printed Deep reparses");
        assert_eq!(reparsed.len(), 1);
        assert_eq!(print_expr(&reparsed[0]), printed);
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
                        print_canonical_flat(core::slice::from_ref(&input)),
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
