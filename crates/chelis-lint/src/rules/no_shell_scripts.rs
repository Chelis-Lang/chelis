//! Rule `no-shell-scripts` — project policy is "never shell" (§2.9).
//! Any `.sh` file is a violation.

use crate::{Context, Rule, Surface, Violation};

pub struct NoShellScripts;

impl Rule for NoShellScripts {
    fn id(&self) -> &str {
        "no-shell-scripts"
    }

    fn spec_ref(&self) -> &str {
        "§2.9"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::ShellScript]
    }

    fn summary(&self) -> &str {
        "shell scripts are prohibited; port to Python (CLAUDE.md Scripting Language Policy)"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        vec![Violation {
            rule_id: self.id().to_string(),
            spec_ref: self.spec_ref().to_string(),
            path: ctx.path.to_path_buf(),
            line: None,
            col: None,
            message:
                "shell script disallowed by project policy; port to Python or remove (see §2.9)"
                    .to_string(),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn fire_on(path: &str, surface: Surface) -> Vec<Violation> {
        let p = Path::new(path);
        let ctx = Context {
            root: Path::new("/"),
            path: p,
            source: None,
            surface,
        };
        NoShellScripts.check(&ctx)
    }

    #[test]
    fn flags_shell_script() {
        let v = fire_on(
            "nautilus/scripts/install_chelis_std.sh",
            Surface::ShellScript,
        );
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].rule_id, "no-shell-scripts");
        assert_eq!(v[0].spec_ref, "§2.9");
    }

    #[test]
    fn driver_does_not_dispatch_to_python() {
        // Negative parity: a Python file is not Surface::ShellScript, so the
        // driver wouldn't dispatch this rule. Test the rule's `applies_to`
        // claim directly.
        assert_eq!(NoShellScripts.applies_to(), &[Surface::ShellScript]);
        assert!(!NoShellScripts.applies_to().contains(&Surface::PythonSource));
    }
}
