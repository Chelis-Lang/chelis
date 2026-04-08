fn main() {
    println!("cargo:rerun-if-changed=../grammars/tree-sitter-chelis-surf/grammar.js");
    println!("cargo:rerun-if-changed=../grammars/tree-sitter-chelis-surf/src/parser.c");
    println!("cargo:rerun-if-changed=../grammars/tree-sitter-chelis-surf/src/tree_sitter/parser.h");
    println!("cargo:rerun-if-changed=../grammars/tree-sitter-chelis-deep/grammar.js");
    println!("cargo:rerun-if-changed=../grammars/tree-sitter-chelis-deep/src/parser.c");
    println!("cargo:rerun-if-changed=../grammars/tree-sitter-chelis-deep/src/tree_sitter/parser.h");

    cc::Build::new()
        .include("../grammars/tree-sitter-chelis-surf/src")
        .file("../grammars/tree-sitter-chelis-surf/src/parser.c")
        .warnings(false)
        .compile("tree-sitter-chelis-surf");

    cc::Build::new()
        .include("../grammars/tree-sitter-chelis-deep/src")
        .file("../grammars/tree-sitter-chelis-deep/src/parser.c")
        .warnings(false)
        .compile("tree-sitter-chelis-deep");
}
