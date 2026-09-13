Derive the unimplemented-rejection issue manifest from exact production source
citations, rejecting missing, stale, dynamic, or malformed issue authority
before generated Rust membership is accepted. Production source discovery now
uses Cargo target roots plus the existing Rust parser's fail-closed module
graph, and every Rust-source edit retriggers live issue validation.
