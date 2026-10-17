{ lib }:
let
  # This file is nix/source.nix, so its parent directory is the workspace
  # root. lib.fileset needs this path value, not the store path string of the
  # flake.
  root = ./..;
  rootString = toString root;
  # Build-facing subset only: prose edits must not invalidate packages.
  # This source feeds package metadata and package behavior checks.
  packageRoots = [
    "Cargo.lock"
    "Cargo.toml"
    "crates"
    "rust-toolchain.toml"
    "tree-sitter-chelis"
  ];
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
    || (lib.elem top packageRoots && !(lib.elem name rejectedNames) && lib.cleanSourceFilter path type);

  # Cargo compiles these directories only for test, bench, and example
  # targets. A crate derivation builds none of these targets.
  testOnlyDirectories = [
    "tests"
    "benches"
    "examples"
  ];

  # A view of the workspace that holds only `fileset`, with the relative
  # layout of the workspace.
  workspaceView =
    fileset:
    lib.cleanSourceWith {
      name = "chelis-source";
      src = lib.fileset.toSource {
        inherit root fileset;
      };
    };

  # The source of one workspace crate. `src` is the crate source from the
  # committed Cargo.nix. The result keeps that source without its test-only
  # directories. It adds `buildReads`, the workspace-relative paths that the
  # crate compile reads outside that remainder. A crate that reads outside its
  # directory gets a view of the workspace with the same relative layout. Its
  # `sourceRoot` names the crate directory in that view. `fileset` is the set
  # of files in the source.
  crate =
    {
      src,
      buildReads ? [ ],
    }:
    let
      directory = src.origSrc;
      relativeDirectory = lib.removePrefix "./" (lib.path.removePrefix root directory);
      excluded = map (name: directory + "/${name}") testOnlyDirectories;
      inExcluded = path: lib.any (excludedPath: lib.path.hasPrefix excludedPath path) excluded;
      reads = map (
        relative:
        let
          path = root + "/${relative}";
        in
        if lib.path.hasPrefix directory path && !inExcluded path then
          throw "nix/source.nix: the source of ${relativeDirectory} contains ${relative} already"
        else
          path
      ) buildReads;
      external = lib.filter (path: !lib.path.hasPrefix directory path) reads;
      own = lib.fileset.difference (lib.fileset.fromSource src) (
        lib.fileset.unions (map lib.fileset.maybeMissing excluded)
      );
      fileset = lib.fileset.unions ([ own ] ++ reads);
    in
    if external == [ ] then
      {
        inherit fileset;
        src = lib.fileset.toSource {
          root = directory;
          inherit fileset;
        };
      }
    else
      {
        inherit fileset;
        src = workspaceView fileset;
        sourceRoot = "chelis-source/${relativeDirectory}";
      };
in
{
  inherit crate root workspaceView;
  package = lib.cleanSourceWith {
    name = "chelis-source";
    src = root;
    inherit filter;
  };
}
