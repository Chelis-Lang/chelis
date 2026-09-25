//! A macro that embeds the runtime static archive produced by the same rustc
//! invocation as the `chelis_runtime` library the invoking crate links
//! (`spec/08-backends.md` §2.1).
//!
//! Cargo and nixpkgs' `buildRustCrate` compile all crate types of a crate in one
//! rustc invocation under one `-C extra-filename`, so `lib<name>-<hash>.rlib` and
//! `lib<name>-<hash>.a` share a directory and a stem, and a dependent receives
//! the `.rlib` through `--extern`. Cargo starts a dependent of a crate with a
//! `staticlib` type only after that invocation has finished, so the archive
//! exists when the macro expands. The macro reads the arguments of the rustc
//! process expanding it; a compilation whose arguments do not locate the
//! archive fails instead of carrying another one.
//!
//! The expansion embeds bytes only, never a path. Compilation caches such as
//! kache key the invoking crate on the content of its `--extern` artifacts, not
//! their paths, and the `.rlib` changes with the archive built beside it, so a
//! cached expansion always carries the archive of the runtime it was keyed on.

mod locate;

use locate::Located;
use proc_macro::TokenStream;
use sha2::{Digest, Sha256};
use std::path::Path;

/// `runtime_archive!(chelis_runtime)` expands to
/// `Option<(&'static [u8], [u8; 32])>`: the static archive produced beside the
/// linked `chelis_runtime` library and its SHA-256, or `None` in a compilation
/// that emits no linkable output. The bytes are read with `include_bytes!`, so
/// the archive becomes a tracked input of the invoking crate.
#[proc_macro]
pub fn runtime_archive(input: TokenStream) -> TokenStream {
    expand(input, |archive| {
        let bytes = std::fs::read(archive)
            .map_err(|error| format!("cannot read {}: {error}", archive.display()))?;
        let digest = Sha256::digest(&bytes)
            .iter()
            .map(|byte| format!("{byte:#04x}"))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!(
            "::core::option::Option::Some((::core::include_bytes!({}) as &'static [u8], [{digest}]))",
            path_literal(archive)?
        ))
    })
}

fn expand(input: TokenStream, found: impl FnOnce(&Path) -> Result<String, String>) -> TokenStream {
    let extern_name = input.to_string();
    let extern_name = extern_name.trim();
    let expansion = if extern_name.is_empty()
        || !extern_name
            .chars()
            .all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        Err(format!(
            "expected the extern crate name of the runtime library, found `{extern_name}`"
        ))
    } else {
        let args = std::env::args_os()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        locate::locate(&args, extern_name, Path::is_file).and_then(|located| match located {
            Located::Archive(archive) => found(&archive),
            Located::NoLinkableOutput => Ok("::core::option::Option::None".to_owned()),
        })
    };
    let source = match expansion {
        Ok(source) => source,
        Err(message) => format!("::core::compile_error!({message:?})"),
    };
    source
        .parse()
        .expect("runtime archive macros emit well-formed Rust tokens")
}

fn path_literal(path: &Path) -> Result<String, String> {
    path.to_str()
        .map(|text| format!("{text:?}"))
        .ok_or_else(|| format!("the archive path {} is not UTF-8", path.display()))
}
