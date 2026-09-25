//! Locate the static archive produced beside the linked runtime library.

use std::path::{Path, PathBuf};

/// What the running compilation can embed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Located {
    /// The compilation emits linkable output. This archive came from the
    /// rustc invocation that produced the linked `.rlib`.
    Archive(PathBuf),
    /// The compilation emits no linkable output: metadata only, rustdoc, or a
    /// process that is not rustc compiling a crate. It carries a placeholder,
    /// which staging refuses.
    NoLinkableOutput,
}

/// Classify the running compiler's arguments.
///
/// `args` are this process's arguments, `extern_name` is the dependency whose
/// archive to take, and `is_file` reports whether a path names a regular file.
pub(crate) fn locate(
    args: &[String],
    extern_name: &str,
    is_file: impl Fn(&Path) -> bool,
) -> Result<Located, String> {
    let Some(program) = args.first() else {
        return Ok(Located::NoLinkableOutput);
    };
    let program_name = Path::new(program)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if program_name.starts_with("rustdoc") || !has_flag(args, "--crate-name") || !emits_link(args) {
        return Ok(Located::NoLinkableOutput);
    }

    let mut libraries = extern_paths(args, extern_name);
    libraries.dedup();
    let rlib = match libraries.as_slice() {
        [] => {
            return Err(format!(
                "this compilation emits linkable output, but rustc received no \
                 `--extern {extern_name}=<path>`; the invoking crate must depend on \
                 `{extern_name}` directly"
            ));
        }
        [only] => only,
        many => {
            let listed = many
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "rustc received conflicting `--extern {extern_name}` libraries: {listed}"
            ));
        }
    };
    if rlib.extension().and_then(|ext| ext.to_str()) != Some("rlib") {
        return Err(format!(
            "`--extern {extern_name}` names {}, not an `.rlib`; the static archive \
             exists only once the library's compilation has finished",
            rlib.display()
        ));
    }
    let archive = rlib.with_extension("a");
    if !is_file(&archive) {
        return Err(format!(
            "no static archive {} was produced beside {}; `{extern_name}` must build \
             crate type `staticlib` in the same compilation as its `rlib`",
            archive.display(),
            rlib.display()
        ));
    }
    Ok(Located::Archive(archive))
}

/// This process's arguments as rustc reads them: an `@path` argument after the
/// program is replaced by the lines of `path`. Cargo passes rustc its arguments
/// through such a file when the command line is too long; classifying the
/// unexpanded `@path` would mistake a linkable compilation for one without
/// linkable output. `read` returns a file's contents.
pub(crate) fn expand_argfiles(
    args: Vec<String>,
    read: impl Fn(&Path) -> std::io::Result<String>,
) -> Result<Vec<String>, String> {
    let mut expanded = Vec::with_capacity(args.len());
    for (index, arg) in args.into_iter().enumerate() {
        match arg.strip_prefix('@').filter(|_| index > 0) {
            Some(file) if file.starts_with("shell:") => {
                return Err(format!(
                    "cannot locate the runtime archive: rustc read shell-quoted arguments from `{file}`"
                ));
            }
            Some(file) => {
                let text = read(Path::new(file)).map_err(|error| {
                    format!("cannot read the rustc argument file {file}: {error}")
                })?;
                expanded.extend(text.lines().map(str::to_owned));
            }
            None => expanded.push(arg),
        }
    }
    Ok(expanded)
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| {
        arg == flag
            || arg
                .strip_prefix(flag)
                .is_some_and(|rest| rest.starts_with('='))
    })
}

/// The values of every `--emit` flag, in either `--emit=…` or `--emit …` form.
fn flag_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    let mut values = Vec::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == flag {
            if let Some(value) = args.get(index + 1) {
                values.push(value.as_str());
            }
            index += 2;
            continue;
        }
        if let Some(value) = arg
            .strip_prefix(flag)
            .and_then(|rest| rest.strip_prefix('='))
        {
            values.push(value);
        }
        index += 1;
    }
    values
}

/// rustc emits linkable output by default; an explicit `--emit` list emits it
/// only when one of its entries is `link` (optionally `link=<path>`).
fn emits_link(args: &[String]) -> bool {
    let emits = flag_values(args, "--emit");
    emits.is_empty()
        || emits.iter().any(|list| {
            list.split(',')
                .any(|entry| entry.split('=').next() == Some("link"))
        })
}

/// Paths given for `extern_name` by `--extern [modifiers:]name=path`.
fn extern_paths(args: &[String], extern_name: &str) -> Vec<PathBuf> {
    flag_values(args, "--extern")
        .into_iter()
        .filter_map(|value| {
            let (spec, path) = value.split_once('=')?;
            let name = spec.rsplit(':').next().unwrap_or(spec);
            (name == extern_name).then(|| PathBuf::from(path))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RLIB: &str = "/t/deps/libchelis_runtime-0123abcd.rlib";
    const ARCHIVE: &str = "/t/deps/libchelis_runtime-0123abcd.a";

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| (*arg).to_owned()).collect()
    }

    fn archive_exists(path: &Path) -> bool {
        path == Path::new(ARCHIVE)
    }

    fn classify(list: &[String]) -> Result<Located, String> {
        locate(list, "chelis_runtime", archive_exists)
    }

    /// A rustc compilation of the bundle crate with `extra` arguments appended.
    fn link_compile(extra: &[&str]) -> Vec<String> {
        let mut list = args(&["rustc", "--crate-name", "chelis_runtime_bundle"]);
        list.extend(extra.iter().map(|arg| (*arg).to_owned()));
        list
    }

    #[test]
    fn linkable_compilation_takes_the_archive_beside_the_rlib() {
        let extern_arg = format!("chelis_runtime={RLIB}");
        for emit in [
            "--emit=dep-info,metadata,link",
            "--emit=dep-info,link=/out/x.rlib",
        ] {
            let list = link_compile(&[emit, "--extern", &extern_arg]);
            assert_eq!(
                classify(&list),
                Ok(Located::Archive(PathBuf::from(ARCHIVE))),
                "{emit}"
            );
        }
    }

    #[test]
    fn rustc_links_by_default_when_emit_is_absent() {
        // nixpkgs' buildRustCrate passes no --emit.
        let extern_arg = format!("--extern=chelis_runtime={RLIB}");
        let list = link_compile(&[&extern_arg]);
        assert_eq!(
            classify(&list),
            Ok(Located::Archive(PathBuf::from(ARCHIVE)))
        );
    }

    #[test]
    fn extern_modifiers_do_not_hide_the_library() {
        let extern_arg = format!("priv,noprelude:chelis_runtime={RLIB}");
        let list = link_compile(&["--emit", "link", "--extern", &extern_arg]);
        assert_eq!(
            classify(&list),
            Ok(Located::Archive(PathBuf::from(ARCHIVE)))
        );
    }

    #[test]
    fn compilations_without_linkable_output_carry_a_placeholder() {
        let rmeta = "chelis_runtime=/t/deps/libchelis_runtime-0123abcd.rmeta";
        let metadata_only = link_compile(&["--emit=dep-info,metadata", "--extern", rmeta]);
        assert_eq!(classify(&metadata_only), Ok(Located::NoLinkableOutput));

        let rustdoc = args(&["/toolchain/bin/rustdoc", "--crate-name", "x"]);
        assert_eq!(classify(&rustdoc), Ok(Located::NoLinkableOutput));

        // rust-analyzer's proc-macro server is not rustc compiling a crate.
        let server = args(&["/toolchain/libexec/rust-analyzer-proc-macro-srv"]);
        assert_eq!(classify(&server), Ok(Located::NoLinkableOutput));
    }

    #[test]
    fn linkable_compilation_without_the_library_fails() {
        let list = link_compile(&["--emit=link", "--extern", "other=/t/deps/libother.rlib"]);
        let error = classify(&list).unwrap_err();
        assert!(error.contains("no `--extern chelis_runtime"), "{error}");
    }

    #[test]
    fn linkable_compilation_given_only_metadata_fails() {
        let rmeta = "chelis_runtime=/t/deps/libchelis_runtime-0123abcd.rmeta";
        let list = link_compile(&["--emit=dep-info,metadata,link", "--extern", rmeta]);
        let error = classify(&list).unwrap_err();
        assert!(error.contains("not an `.rlib`"), "{error}");
    }

    #[test]
    fn missing_archive_beside_the_rlib_fails() {
        let extern_arg = "chelis_runtime=/elsewhere/libchelis_runtime-0123abcd.rlib";
        let list = link_compile(&["--emit=link", "--extern", extern_arg]);
        let error = classify(&list).unwrap_err();
        assert!(error.contains("no static archive"), "{error}");
    }

    #[test]
    fn conflicting_libraries_fail() {
        let first = format!("chelis_runtime={RLIB}");
        let second = "chelis_runtime=/other/libchelis_runtime-ffff.rlib";
        let list = link_compile(&["--emit=link", "--extern", &first, "--extern", second]);
        let error = classify(&list).unwrap_err();
        assert!(error.contains("conflicting"), "{error}");
    }

    #[test]
    fn argfile_arguments_are_classified_as_rustc_reads_them() {
        let contents = format!(
            "--crate-name\nchelis_runtime_bundle\n--emit=link\n--extern\nchelis_runtime={RLIB}\n"
        );
        let expanded = expand_argfiles(args(&["rustc", "@/t/args"]), |path| {
            assert_eq!(path, Path::new("/t/args"));
            Ok(contents.clone())
        })
        .expect("argument file");
        assert_eq!(
            classify(&expanded),
            Ok(Located::Archive(PathBuf::from(ARCHIVE)))
        );
    }

    #[test]
    fn unreadable_or_shell_quoted_argfiles_fail() {
        let missing = expand_argfiles(args(&["rustc", "@/t/missing"]), |_| {
            Err(std::io::Error::from(std::io::ErrorKind::NotFound))
        });
        assert!(missing.unwrap_err().contains("/t/missing"));
        let shell = expand_argfiles(args(&["rustc", "@shell:/t/args"]), |_| Ok(String::new()));
        assert!(shell.unwrap_err().contains("shell-quoted"));
    }
}
