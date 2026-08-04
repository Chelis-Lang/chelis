{
  pkgs,
  crateSource,
  toolchain,
  cvc5,
}:
cratePkgs:
let
  chelisOverrides = {
    "chelis-cli" = attrs: {
      src = crateSource;
      sourceRoot = "chelis-source/crates/chelis-cli";
    };
    "chelis-compiler-api" = attrs: {
      src = crateSource;
      sourceRoot = "chelis-source/crates/chelis-compiler-api";
    };
    "chelis-cove" = attrs: {
      src = crateSource;
      sourceRoot = "chelis-source/crates/chelis-cove";
    };
    "cvc5-sys" = attrs: {
      nativeBuildInputs = (attrs.nativeBuildInputs or [ ]) ++ [
        pkgs.llvmPackages.libclang
        pkgs.pkg-config
      ];
      CVC5_DIR = "${cvc5.dir}";
      LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
    };
    "tree-sitter-chelis" = attrs: {
      src = crateSource;
      sourceRoot = "chelis-source/tree-sitter-chelis";
    };
  };
  requiredOverrideNames = [
    "chelis-cli"
    "chelis-compiler-api"
    "chelis-cove"
    "cvc5-sys"
    "tree-sitter-chelis"
  ];
  missingOverrides = builtins.filter (
    name: !(builtins.hasAttr name chelisOverrides)
  ) requiredOverrideNames;
in
assert missingOverrides == [ ];
cratePkgs.buildRustCrate.override {
  cargo = toolchain;
  rustc = toolchain;
  defaultCrateOverrides = cratePkgs.defaultCrateOverrides // chelisOverrides;
}
