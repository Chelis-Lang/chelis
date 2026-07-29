{
  lib,
  root,
  includeRoots ? null,
}:
let
  rootString = toString root;
  defaultRoots = [
    ".cargo"
    "Cargo.lock"
    "Cargo.toml"
    "LICENSE"
    "README.md"
    "chelis-lint.toml"
    "crates"
    "docs"
    "examples"
    "grammars"
    "packages"
    "rust-toolchain.toml"
    "spec"
    "tree-sitter-chelis"
  ];
  allowedRoots = if includeRoots == null then defaultRoots else includeRoots;
  rejectedNames = [
    ".devenv"
    ".git"
    ".venv"
    ".work"
    "__pycache__"
    "node_modules"
    "target"
  ];
  filter =
    path: type:
    let
      pathString = toString path;
      relative = lib.removePrefix "${rootString}/" pathString;
      components = lib.splitString "/" relative;
      top = if components == [ ] then "" else builtins.head components;
      name = baseNameOf pathString;
    in
    pathString == rootString
    || (lib.elem top allowedRoots && !(lib.elem name rejectedNames) && lib.cleanSourceFilter path type);
in
lib.cleanSourceWith {
  name = "chelis-source";
  src = root;
  inherit filter;
}
