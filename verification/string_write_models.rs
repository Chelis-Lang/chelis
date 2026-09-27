//! `impl fmt::Write for String` (Aeneas.Std Core/FmtWithSlice.lean) as chelis-ir
//! `ownership::render` uses it: `writeln!`/`write!` into a `String` given back through its
//! mutable borrow, with `Display` and `Debug` placeholders and their options, a literal-only
//! template, and a `Display` impl that fails (the status is returned and the text written before
//! the failure stays); then `write_str` and `write_char` directly.
use std::fmt::{self, Write as _};

#[derive(Debug, Clone, Copy)]
pub enum Kind {
    Entry,
    Helper { arity: u32 },
}

pub struct Block {
    pub id: u32,
    pub ops: Vec<String>,
}

pub struct Unit {
    pub name: String,
    pub kind: Kind,
    pub blocks: Vec<Block>,
}

/// As `render::render`: one line per unit, block and op.
pub fn render(units: &Vec<Unit>) -> String {
    let mut out = String::new();
    for unit in units {
        let _ = writeln!(out, "unit {} {:?}", unit.name, unit.kind);
        for block in &unit.blocks {
            let _ = writeln!(out, "  b{} ({}):", block.id, block.ops.len());
            for op in &block.ops {
                let _ = writeln!(out, "    {op}");
            }
        }
    }
    out
}

/// Placeholder options, and a literal-only template, into a caller's string.
pub fn append(out: &mut String, n: u32, name: &str) -> bool {
    let first = write!(out, "[{n:>4}|{n:<3}|{n:03}|{name:^7}|{name:.2}]");
    let second = write!(out, "done");
    first.is_ok() && second.is_ok()
}

/// A value whose `Display` writes a prefix, then fails when `fail` is set.
pub struct Flaky {
    pub fail: bool,
}

impl fmt::Display for Flaky {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("pre")?;
        if self.fail { Err(fmt::Error) } else { f.write_str("-ok") }
    }
}

/// The status of `write!` and the string given back: on a failure, the later pieces are not
/// written.
pub fn write_flaky(out: &mut String, fail: bool) -> bool {
    write!(out, "<{}>{}", Flaky { fail }, 7).is_ok()
}

/// `write_str` and `write_char` directly.
pub fn pieces(out: &mut String) -> (bool, bool) {
    let a = out.write_str("ab").is_ok();
    let b = out.write_char('é').is_ok();
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders() {
        let units = vec![
            Unit {
                name: "main".to_string(),
                kind: Kind::Entry,
                blocks: vec![
                    Block { id: 0, ops: vec!["move %1".to_string(), "drop %2".to_string()] },
                    Block { id: 1, ops: vec![] },
                ],
            },
            Unit { name: "h".to_string(), kind: Kind::Helper { arity: 2 }, blocks: vec![] },
        ];
        assert_eq!(
            render(&units),
            "unit main Entry\n  b0 (2):\n    move %1\n    drop %2\n  b1 (0):\nunit h Helper { arity: 2 }\n"
        );
        assert_eq!(render(&Vec::new()), "");
    }

    #[test]
    fn appends() {
        let mut out = "x".to_string();
        assert!(append(&mut out, 7, "abc"));
        assert_eq!(out, "x[   7|7  |007|  abc  |ab]done");
    }

    #[test]
    fn flaky() {
        let mut out = "s".to_string();
        assert!(write_flaky(&mut out, false));
        assert_eq!(out, "s<pre-ok>7");
        assert!(!write_flaky(&mut out, true));
        assert_eq!(out, "s<pre-ok>7<pre");
    }

    #[test]
    fn direct() {
        let mut out = "z".to_string();
        assert_eq!(pieces(&mut out), (true, true));
        assert_eq!(out, "zabé");
    }
}
