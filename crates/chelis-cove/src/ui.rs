use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Text;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::app::App;
use crate::editor::{render_highlighted_lines, render_highlighted_lines_wrapped};
use crate::live::diagnostics_text;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(frame.area());
    let body = root[0];
    let footer = root[1];

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(56), Constraint::Percentage(44)])
        .split(body);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(38),
            Constraint::Percentage(31),
            Constraint::Percentage(31),
        ])
        .split(columns[1]);

    draw_editor(frame, app, columns[0]);
    draw_deep(frame, app, right[0]);
    draw_diagnostics(frame, app, right[1]);
    draw_output(frame, app, right[2]);
    draw_footer(frame, app, footer);
}

fn draw_editor(frame: &mut Frame, app: &mut App, area: Rect) {
    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2) as usize;
    app.editor
        .ensure_cursor_visible(inner_height.max(1), inner_width.max(1));
    let lines = render_highlighted_lines(
        app.editor.text(),
        &app.editor_spans,
        app.editor.scroll_y,
        app.editor.scroll_x,
        inner_height.max(1),
        inner_width.max(1),
    );
    let title = app
        .editor
        .path
        .as_ref()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "[unsaved]".to_string());
    let paragraph = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);

    let (line, col) = app.editor.cursor_line_col();
    if line >= app.editor.scroll_y
        && col >= app.editor.scroll_x
        && inner_width > 0
        && inner_height > 0
    {
        let x = area.x + 1 + (col - app.editor.scroll_x) as u16;
        let y = area.y + 1 + (line - app.editor.scroll_y) as u16;
        if x < area.right() && y < area.bottom() {
            frame.set_cursor_position((x, y));
        }
    }
}

fn draw_deep(frame: &mut Frame, app: &mut App, area: Rect) {
    let lines = render_highlighted_lines_wrapped(&app.deep_source, &app.deep_spans);
    let paragraph = Paragraph::new(Text::from(lines))
        .block(Block::default().borders(Borders::ALL).title("Deep"))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_diagnostics(frame: &mut Frame, app: &App, area: Rect) {
    let text = diagnostics_text(&app.analysis);
    let paragraph = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL).title("Diagnostics"))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_output(frame: &mut Frame, app: &App, area: Rect) {
    let paragraph = Paragraph::new(app.output.clone())
        .block(Block::default().borders(Borders::ALL).title("Output"))
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let status = format!(
        "{}  {}  Ctrl-S save  Ctrl-R compile  Ctrl-E eval  F5/F6 too  Ctrl-Q quit",
        app.analysis.status_text(),
        cursor_status(app)
    );
    let paragraph = Paragraph::new(status).style(
        Style::default()
            .fg(Color::Black)
            .bg(Color::LightCyan)
            .add_modifier(Modifier::BOLD),
    );
    frame.render_widget(paragraph, area);
}

fn cursor_status(app: &App) -> String {
    let (line, col) = app.editor.cursor_line_col();
    format!("Ln {}, Col {}", line + 1, col + 1)
}
