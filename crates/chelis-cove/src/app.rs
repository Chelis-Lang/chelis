use std::io::{self, IsTerminal, Stdout};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::editor::{HighlightSpan, Highlighter, LanguageMode, TextBuffer};
use crate::live::{LiveAnalysis, analyze, compile_output, eval_output};
use crate::ui;

#[derive(Debug, Clone, Default)]
pub struct CoveOptions {
    pub file: Option<PathBuf>,
}

pub struct App {
    pub editor: TextBuffer,
    pub analysis: LiveAnalysis,
    pub output: String,
    pub deep_source: String,
    pub editor_spans: Vec<HighlightSpan>,
    pub deep_spans: Vec<HighlightSpan>,
    surf_highlighter: Highlighter,
    deep_highlighter: Highlighter,
}

impl App {
    pub fn new(options: CoveOptions) -> Result<Self> {
        let editor = match options.file {
            Some(path) => TextBuffer::open(&path)?,
            None => TextBuffer::new(None, String::new()),
        };
        let surf_highlighter = Highlighter::new(LanguageMode::Surf)?;
        let deep_highlighter = Highlighter::new(LanguageMode::Deep)?;
        let mut app = Self {
            editor,
            analysis: analyze(""),
            output: "Compile or evaluate from the current buffer.".to_string(),
            deep_source: String::new(),
            editor_spans: Vec::new(),
            deep_spans: Vec::new(),
            surf_highlighter,
            deep_highlighter,
        };
        app.refresh();
        Ok(app)
    }

    pub fn refresh(&mut self) {
        self.analysis = analyze(self.editor.text());
        self.deep_source = self
            .analysis
            .deep_text
            .clone()
            .unwrap_or_else(|| "Deep unavailable until Surf parses.".to_string());
        self.editor_spans = self.surf_highlighter.highlight(self.editor.text());
        self.deep_spans = self.deep_highlighter.highlight(&self.deep_source);
    }

    pub fn save(&mut self) {
        self.output = match self.editor.save() {
            Ok(()) => "saved".to_string(),
            Err(err) => format!("save failed\n{err:#}"),
        };
    }

    pub fn compile(&mut self) {
        self.output = compile_output(self.editor.text());
    }

    pub fn eval(&mut self) {
        self.output = eval_output(self.editor.text());
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('q') => return true,
                KeyCode::Char('s') => {
                    self.save();
                    return false;
                }
                KeyCode::Char('r') => {
                    self.compile();
                    return false;
                }
                KeyCode::Char('e') => {
                    self.eval();
                    return false;
                }
                _ => {}
            }
        }

        let mut changed = false;
        match key.code {
            KeyCode::F(5) => {
                self.compile();
                return false;
            }
            KeyCode::F(6) => {
                self.eval();
                return false;
            }
            KeyCode::Char(ch)
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT =>
            {
                self.editor.insert_char(ch);
                changed = true;
            }
            KeyCode::Enter => {
                self.editor.insert_char('\n');
                changed = true;
            }
            KeyCode::Tab => {
                self.editor.insert_str("    ");
                changed = true;
            }
            KeyCode::Backspace => {
                self.editor.backspace();
                changed = true;
            }
            KeyCode::Delete => {
                self.editor.delete();
                changed = true;
            }
            KeyCode::Left => self.editor.move_left(),
            KeyCode::Right => self.editor.move_right(),
            KeyCode::Up => self.editor.move_up(),
            KeyCode::Down => self.editor.move_down(),
            KeyCode::Home => self.editor.move_home(),
            KeyCode::End => self.editor.move_end(),
            _ => {}
        }
        if changed {
            self.refresh();
        }
        false
    }
}

pub fn run(options: CoveOptions) -> Result<()> {
    let mut app = App::new(options)?;
    if !io::stdout().is_terminal() {
        anyhow::bail!("chelis cove requires an interactive terminal");
    }
    let mut terminal = setup_terminal()?;
    let run_result = (|| -> Result<()> {
        loop {
            terminal.draw(|frame| ui::draw(frame, &mut app))?;
            if event::poll(Duration::from_millis(100)).context("poll Cove terminal events")? {
                match event::read().context("read Cove terminal event")? {
                    Event::Key(key) => {
                        if app.handle_key(key) {
                            break;
                        }
                    }
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    let restore_result = restore_terminal(&mut terminal);
    run_result.and(restore_result)
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).context("create Cove terminal")
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen).context("leave alternate screen")?;
    terminal.show_cursor().context("show cursor")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn app_loads_file_contents() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("sample.ch");
        std::fs::write(&path, "let x = 1\n").expect("write");
        let app = App::new(CoveOptions {
            file: Some(path.clone()),
        })
        .expect("app");
        assert!(app.editor.text().contains("let x"));
        assert_eq!(app.editor.path.as_deref(), Some(path.as_path()));
    }
}
