//! The bundled runtime does not expose shell-level initializers, a tokenizer, or
//! the `Std.Time` module that `Std.Datetime` replaced.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{make_app, write_file};

#[test]
fn removed_shell_modules_cannot_be_imported_from_chelis_std() {
    for (module, names) in [
        ("Std.Init.Random", "normal_like"),
        ("Std.Init.Kaiming", "kaiming_uniform"),
        ("Std.Init.XavierExt", "xavier_uniform"),
        ("Std.Tokenizer", "Tokenizer, encode"),
        ("Std.Time", "date"),
    ] {
        let (_dir, reef_home, app_pkg) = make_app("removed-stdlib-module");
        write_file(
            &app_pkg.join("src/main.ch"),
            &format!(
                "module Demo.Main\nimport {module} ({names})\nvalue = {first}\n",
                first = names.split(',').next().unwrap()
            ),
        );

        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .current_dir(&app_pkg)
            .args(["check", "src/main.ch"])
            .output()
            .expect("chelis check runs");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !output.status.success(),
            "removed module {module} unexpectedly type-checked:\n{text}"
        );
        assert!(
            text.contains(module),
            "diagnostic should name removed module {module}:\n{text}"
        );
    }
}
