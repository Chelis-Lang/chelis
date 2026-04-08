use anyhow::{Context, Result};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use std::cmp::{max, min};
use std::fs;
use std::path::{Path, PathBuf};
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HighlightKind {
    Comment,
    Keyword,
    Constant,
    Namespace,
    Function,
    FunctionCall,
    Type,
    TypeKeyword,
    BuiltinType,
    Parameter,
    Variable,
    Property,
    Number,
    String,
    Operator,
    DeepTag,
}

#[derive(Debug, Clone)]
pub struct HighlightSpan {
    pub start: usize,
    pub end: usize,
    pub kind: HighlightKind,
}

pub struct Highlighter {
    parser: Parser,
    query: Query,
    capture_kinds: Vec<Option<HighlightKind>>,
}

pub enum LanguageMode {
    Surf,
    Deep,
}

impl Highlighter {
    pub fn new(mode: LanguageMode) -> Result<Self> {
        let (language, query_source) = match mode {
            LanguageMode::Surf => (
                tree_sitter_chelis::surf_language(),
                include_str!("../../../grammars/tree-sitter-chelis-surf/queries/highlights.scm"),
            ),
            LanguageMode::Deep => (
                tree_sitter_chelis::deep_language(),
                include_str!("../../../grammars/tree-sitter-chelis-deep/queries/highlights.scm"),
            ),
        };
        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .context("configure Chelis tree-sitter language")?;
        let query =
            Query::new(&language, query_source).context("compile Chelis highlight query")?;
        let capture_kinds = query
            .capture_names()
            .iter()
            .map(|name| map_capture(name))
            .collect();
        Ok(Self {
            parser,
            query,
            capture_kinds,
        })
    }

    pub fn highlight(&mut self, source: &str) -> Vec<HighlightSpan> {
        let Some(tree) = self.parser.parse(source, None) else {
            return Vec::new();
        };
        let mut cursor = QueryCursor::new();
        let mut spans = Vec::new();
        let mut matches = cursor.matches(&self.query, tree.root_node(), source.as_bytes());
        while let Some(capture) = matches.next() {
            for entry in capture.captures {
                if let Some(kind) = self
                    .capture_kinds
                    .get(entry.index as usize)
                    .and_then(|v| *v)
                {
                    spans.push(HighlightSpan {
                        start: entry.node.start_byte(),
                        end: entry.node.end_byte(),
                        kind,
                    });
                }
            }
        }
        spans
    }
}

fn map_capture(name: &str) -> Option<HighlightKind> {
    Some(match name {
        "comment" | "comment.line" | "comment.block" => HighlightKind::Comment,
        "keyword" => HighlightKind::Keyword,
        "constant.builtin" => HighlightKind::Constant,
        "namespace" => HighlightKind::Namespace,
        "function.special" => HighlightKind::DeepTag,
        "function" | "function.method" => HighlightKind::Function,
        "function.call" => HighlightKind::FunctionCall,
        "type" => HighlightKind::Type,
        "type.keyword" => HighlightKind::TypeKeyword,
        "type.builtin" => HighlightKind::BuiltinType,
        "parameter" => HighlightKind::Parameter,
        "variable" => HighlightKind::Variable,
        "identifier" => HighlightKind::Variable,
        "property" => HighlightKind::Property,
        "number" | "number.float" => HighlightKind::Number,
        "string" => HighlightKind::String,
        "boolean" => HighlightKind::Constant,
        "operator" => HighlightKind::Operator,
        "constructor" => HighlightKind::Type,
        _ => return None,
    })
}

#[derive(Debug, Clone)]
pub struct TextBuffer {
    pub path: Option<PathBuf>,
    text: String,
    cursor: usize,
    preferred_column: Option<usize>,
    pub scroll_x: usize,
    pub scroll_y: usize,
    pub dirty: bool,
}

impl TextBuffer {
    pub fn new(path: Option<PathBuf>, text: String) -> Self {
        Self {
            path,
            text,
            cursor: 0,
            preferred_column: None,
            scroll_x: 0,
            scroll_y: 0,
            dirty: false,
        }
    }

    pub fn open(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("read Cove file {}", path.display()))?;
        Ok(Self::new(Some(path.to_path_buf()), text))
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn insert_char(&mut self, ch: char) {
        self.text.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.preferred_column = None;
        self.dirty = true;
    }

    pub fn insert_str(&mut self, value: &str) {
        self.text.insert_str(self.cursor, value);
        self.cursor += value.len();
        self.preferred_column = None;
        self.dirty = true;
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = prev_boundary(&self.text, self.cursor);
        self.text.replace_range(prev..self.cursor, "");
        self.cursor = prev;
        self.preferred_column = None;
        self.dirty = true;
    }

    pub fn delete(&mut self) {
        if self.cursor >= self.text.len() {
            return;
        }
        let next = next_boundary(&self.text, self.cursor);
        self.text.replace_range(self.cursor..next, "");
        self.dirty = true;
    }

    pub fn move_left(&mut self) {
        self.cursor = prev_boundary(&self.text, self.cursor);
        self.preferred_column = None;
    }

    pub fn move_right(&mut self) {
        self.cursor = next_boundary(&self.text, self.cursor);
        self.preferred_column = None;
    }

    pub fn move_home(&mut self) {
        let (line, _) = self.cursor_line_col();
        self.cursor = self.line_start(line);
        self.preferred_column = None;
    }

    pub fn move_end(&mut self) {
        let (line, _) = self.cursor_line_col();
        self.cursor = self.line_end(line);
        self.preferred_column = None;
    }

    pub fn move_up(&mut self) {
        let (line, col) = self.cursor_line_col();
        if line == 0 {
            self.cursor = 0;
            return;
        }
        let target_col = self.preferred_column.unwrap_or(col);
        self.cursor = self.offset_for_line_col(line - 1, target_col);
        self.preferred_column = Some(target_col);
    }

    pub fn move_down(&mut self) {
        let (line, col) = self.cursor_line_col();
        if line + 1 >= self.line_count() {
            self.cursor = self.text.len();
            return;
        }
        let target_col = self.preferred_column.unwrap_or(col);
        self.cursor = self.offset_for_line_col(line + 1, target_col);
        self.preferred_column = Some(target_col);
    }

    pub fn save(&mut self) -> Result<()> {
        let path = self
            .path
            .as_ref()
            .context("cannot save an unsaved Cove buffer; launch with --file")?;
        fs::write(path, &self.text).with_context(|| format!("write {}", path.display()))?;
        self.dirty = false;
        Ok(())
    }

    pub fn cursor_line_col(&self) -> (usize, usize) {
        let line = self.text[..self.cursor]
            .bytes()
            .filter(|b| *b == b'\n')
            .count();
        let line_start = self.line_start(line);
        let column = self.text[line_start..self.cursor].chars().count();
        (line, column)
    }

    pub fn ensure_cursor_visible(&mut self, height: usize, width: usize) {
        let (line, col) = self.cursor_line_col();
        if line < self.scroll_y {
            self.scroll_y = line;
        } else if line >= self.scroll_y.saturating_add(height) {
            self.scroll_y = line.saturating_sub(height.saturating_sub(1));
        }
        if col < self.scroll_x {
            self.scroll_x = col;
        } else if col >= self.scroll_x.saturating_add(width) {
            self.scroll_x = col.saturating_sub(width.saturating_sub(1));
        }
    }

    pub fn line_count(&self) -> usize {
        max(1, self.text.lines().count())
    }

    fn line_start(&self, line: usize) -> usize {
        if line == 0 {
            return 0;
        }
        let mut current_line = 0;
        for (idx, ch) in self.text.char_indices() {
            if ch == '\n' {
                current_line += 1;
                if current_line == line {
                    return idx + 1;
                }
            }
        }
        self.text.len()
    }

    fn line_end(&self, line: usize) -> usize {
        let start = self.line_start(line);
        let rest = &self.text[start..];
        rest.find('\n')
            .map(|offset| start + offset)
            .unwrap_or(self.text.len())
    }

    fn offset_for_line_col(&self, line: usize, col: usize) -> usize {
        let start = self.line_start(line);
        let end = self.line_end(line);
        let mut offset = start;
        for (current, (idx, _)) in self.text[start..end].char_indices().enumerate() {
            if current == col {
                return start + idx;
            }
            offset = start + idx;
        }
        if col == 0 { start } else { end.max(offset) }
    }
}

fn prev_boundary(text: &str, cursor: usize) -> usize {
    text[..cursor]
        .char_indices()
        .last()
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn next_boundary(text: &str, cursor: usize) -> usize {
    if cursor >= text.len() {
        return text.len();
    }
    let mut iter = text[cursor..].char_indices();
    iter.next();
    iter.next()
        .map(|(idx, _)| cursor + idx)
        .unwrap_or(text.len())
}

pub fn render_highlighted_lines(
    source: &str,
    spans: &[HighlightSpan],
    scroll_y: usize,
    scroll_x: usize,
    height: usize,
    width: usize,
) -> Vec<Line<'static>> {
    let styles = build_style_table(source, spans);
    let mut lines = Vec::new();
    for (line_index, line) in source.lines().enumerate().skip(scroll_y).take(height) {
        let line_start = byte_offset_for_line(source, line_index);
        lines.push(render_line(line, line_start, scroll_x, width, &styles));
    }
    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

pub fn render_highlighted_lines_wrapped(
    source: &str,
    spans: &[HighlightSpan],
) -> Vec<Line<'static>> {
    let styles = build_style_table(source, spans);
    let mut lines = Vec::new();
    for (line_index, line) in source.lines().enumerate() {
        let line_start = byte_offset_for_line(source, line_index);
        lines.push(render_line_full(line, line_start, &styles));
    }
    if lines.is_empty() || source.ends_with('\n') {
        lines.push(Line::from(""));
    }
    lines
}

fn render_line(
    line: &str,
    line_start: usize,
    scroll_x: usize,
    width: usize,
    styles: &[Option<HighlightKind>],
) -> Line<'static> {
    let mut spans_out = Vec::new();
    let mut visible = String::new();
    let mut current_style = style_for(None);
    let mut current_col = 0usize;
    let mut emitted_cols = 0usize;
    for (idx, ch) in line.char_indices() {
        if current_col < scroll_x {
            current_col += 1;
            continue;
        }
        if emitted_cols >= width {
            break;
        }
        let kind = styles.get(line_start + idx).copied().flatten();
        let target_style = style_for(kind);
        if target_style != current_style && !visible.is_empty() {
            spans_out.push(Span::styled(std::mem::take(&mut visible), current_style));
        }
        current_style = target_style;
        visible.push(ch);
        current_col += 1;
        emitted_cols += 1;
    }
    if !visible.is_empty() {
        spans_out.push(Span::styled(visible, current_style));
    }
    if spans_out.is_empty() {
        spans_out.push(Span::raw(""));
    }
    Line::from(spans_out)
}

fn render_line_full(
    line: &str,
    line_start: usize,
    styles: &[Option<HighlightKind>],
) -> Line<'static> {
    let mut spans_out = Vec::new();
    let mut visible = String::new();
    let mut current_style = style_for(None);
    for (idx, ch) in line.char_indices() {
        let kind = styles.get(line_start + idx).copied().flatten();
        let target_style = style_for(kind);
        if target_style != current_style && !visible.is_empty() {
            spans_out.push(Span::styled(std::mem::take(&mut visible), current_style));
        }
        current_style = target_style;
        visible.push(ch);
    }
    if !visible.is_empty() {
        spans_out.push(Span::styled(visible, current_style));
    }
    if spans_out.is_empty() {
        spans_out.push(Span::raw(""));
    }
    Line::from(spans_out)
}

fn build_style_table(source: &str, spans: &[HighlightSpan]) -> Vec<Option<HighlightKind>> {
    let mut table = vec![None; source.len().max(1)];
    for span in spans {
        let span_priority = priority(span.kind);
        let end = min(span.end, table.len());
        for slot in &mut table[span.start..end] {
            if slot.as_ref().copied().map(priority).unwrap_or_default() <= span_priority {
                *slot = Some(span.kind);
            }
        }
    }
    table
}

fn priority(kind: HighlightKind) -> usize {
    match kind {
        HighlightKind::Comment => 1,
        HighlightKind::Operator => 2,
        HighlightKind::Variable => 3,
        HighlightKind::Parameter => 4,
        HighlightKind::Number | HighlightKind::String | HighlightKind::Constant => 5,
        HighlightKind::Property => 6,
        HighlightKind::Type | HighlightKind::TypeKeyword | HighlightKind::BuiltinType => 7,
        HighlightKind::FunctionCall | HighlightKind::Function | HighlightKind::DeepTag => 8,
        HighlightKind::Namespace => 9,
        HighlightKind::Keyword => 10,
    }
}

fn style_for(kind: Option<HighlightKind>) -> Style {
    match kind {
        Some(HighlightKind::Comment) => Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
        Some(HighlightKind::Keyword) => Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::Constant) => Style::default()
            .fg(Color::LightYellow)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::Namespace) => Style::default()
            .fg(Color::LightCyan)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::Function) => Style::default()
            .fg(Color::LightGreen)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::FunctionCall) => Style::default().fg(Color::Green),
        Some(HighlightKind::Type) => Style::default()
            .fg(Color::LightBlue)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::TypeKeyword) => Style::default()
            .fg(Color::Blue)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::BuiltinType) => Style::default()
            .fg(Color::LightYellow)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::Parameter) => Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::Variable) => Style::default().fg(Color::Gray),
        Some(HighlightKind::Property) => Style::default()
            .fg(Color::LightMagenta)
            .add_modifier(Modifier::BOLD),
        Some(HighlightKind::Number) => Style::default().fg(Color::LightRed),
        Some(HighlightKind::String) => Style::default().fg(Color::Green),
        Some(HighlightKind::Operator) => Style::default().fg(Color::Gray),
        Some(HighlightKind::DeepTag) => Style::default()
            .fg(Color::LightMagenta)
            .add_modifier(Modifier::BOLD),
        None => Style::default(),
    }
}

fn byte_offset_for_line(source: &str, line: usize) -> usize {
    if line == 0 {
        return 0;
    }
    let mut current_line = 0;
    for (idx, ch) in source.char_indices() {
        if ch == '\n' {
            current_line += 1;
            if current_line == line {
                return idx + 1;
            }
        }
    }
    source.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_sitter_highlights_mnist_without_error() {
        let mut highlighter = Highlighter::new(LanguageMode::Surf).expect("highlighter");
        let source = include_str!("../../../examples/mnist.ch");
        let spans = highlighter.highlight(source);
        assert!(!spans.is_empty());
        assert!(spans.iter().any(|span| span.kind == HighlightKind::Keyword));
        assert!(
            spans
                .iter()
                .any(|span| span.kind == HighlightKind::FunctionCall)
        );
        assert!(
            spans
                .iter()
                .any(|span| span.kind == HighlightKind::TypeKeyword)
        );
        assert!(
            spans
                .iter()
                .any(|span| span.kind == HighlightKind::BuiltinType)
        );
    }

    #[test]
    fn tree_sitter_highlights_deep_tags_without_error() {
        let mut highlighter = Highlighter::new(LanguageMode::Deep).expect("highlighter");
        let source = "(def {name loss} loss (app {} (var {} mean) (var {} xs)))";
        let spans = highlighter.highlight(source);
        assert!(!spans.is_empty());
        assert!(spans.iter().any(|span| span.kind == HighlightKind::DeepTag));
        assert!(
            spans
                .iter()
                .any(|span| span.kind == HighlightKind::Function)
        );
        assert!(
            spans
                .iter()
                .any(|span| span.kind == HighlightKind::Parameter)
        );
        assert!(
            spans
                .iter()
                .any(|span| span.kind == HighlightKind::Property)
        );
    }

    #[test]
    fn buffer_moves_between_lines() {
        let mut buffer = TextBuffer::new(None, "abc\ndef".to_string());
        buffer.move_end();
        buffer.move_down();
        let (line, col) = buffer.cursor_line_col();
        assert_eq!((line, col), (1, 3));
    }

    #[test]
    fn wrapped_renderer_keeps_long_line_content() {
        let spans = vec![HighlightSpan {
            start: 0,
            end: 12,
            kind: HighlightKind::DeepTag,
        }];
        let lines = render_highlighted_lines_wrapped("(def {} foo)", &spans);
        assert_eq!(lines.len(), 1);
        let rendered: String = lines[0]
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert_eq!(rendered, "(def {} foo)");
    }
}
