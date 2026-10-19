{
  pkgs,
  lib,
  version,
  chelis,
  buildChelisPackage,
}:
let
  librarySource = pkgs.runCommand "nix-builder-library-source" { } ''
    mkdir -p "$out/src"
    cat > "$out/reef.toml" <<'EOF'
    schema = "3"
    [package]
    name = "nix-builder-library"
    version = "0.1.0"
    compiler = "=${version}"
    module_prefix = "BuilderLibrary"
    resolver = "2"
    EOF
    cat > "$out/src/math.ch" <<'EOF'
    module BuilderLibrary.Math
    export (answer)

    def answer() -> i32 = 42
    EOF
  '';
  library = buildChelisPackage {
    inherit pkgs chelis;
    pname = "nix-builder-library";
    version = "0.1.0";
    src = librarySource;
    reefPackage = true;
  };
  programSource = pkgs.runCommand "nix-builder-program-source" { } ''
    mkdir -p "$out/src"
    cat > "$out/reef.toml" <<'EOF'
    schema = "3"
    [package]
    name = "nix-builder-program"
    version = "0.1.0"
    compiler = "=${version}"
    module_prefix = "BuilderApp"
    resolver = "2"

    [dependencies]
    nix-builder-library = { version = "=0.1.0" }
    EOF
    cat > "$out/src/main.ch" <<'EOF'
    module BuilderApp.Main
    import BuilderLibrary.Math (answer)
    main = answer()
    EOF
    archive_sha256="$(sha256sum ${library}/share/reef/nix-builder-library-0.1.0.tar.zst | cut -d ' ' -f 1)"
    shell_sha256="$(sha256sum ${library}/share/reef/nix-builder-library-0.1.0.chb | cut -d ' ' -f 1)"
    cat > "$out/reef.lock" <<EOF
    schema = "1"
    [package]
    name = "nix-builder-program"
    version = "0.1.0"

    [[dependencies]]
    name = "nix-builder-library"
    version = "0.1.0"
    compiler = "=${version}"
    archive_sha256 = "$archive_sha256"
    shell_sha256 = "$shell_sha256"

    [dependencies.source]
    kind = "local_registry"
    remote_origin = "github://test-org/nix-builder-library@v0.1.0"
    EOF
  '';
  program = buildChelisPackage {
    inherit pkgs chelis;
    pname = "nix-builder-program";
    version = "0.1.0";
    src = programSource;
    artifacts = "${library}/share/reef";
  };
  literalProgram = buildChelisPackage {
    inherit pkgs chelis;
    pname = "nix-builder-literal-entry";
    version = "0.1.0";
    src = pkgs.writeTextDir "src/main$tag.ch" "main = 43\n";
    entryPoint = "src/main$tag.ch";
  };
  deepProgram = buildChelisPackage {
    inherit pkgs chelis;
    pname = "nix-builder-deep-entry";
    version = "0.1.0";
    src = pkgs.writeTextDir "src/main.ch.dp" "(def {} main (lit {type: (t-prim {} i32)} 47))\n";
    entryPoint = "src/main.ch.dp";
  };
in
pkgs.runCommand "chelis-reef-builder" { passthru = { inherit library program; }; } ''
  actual="$(${lib.escapeShellArg program}/bin/nix-builder-program)"
  if [ "$actual" != 'main = 42' ]; then
    printf 'unexpected builder output: %s\n' "$actual" >&2
    exit 1
  fi

  test "$(${literalProgram}/bin/nix-builder-literal-entry)" = 'main = 43'
  test "$(${deepProgram}/bin/nix-builder-deep-entry)" = 'main = 47'

  ${pkgs.zstd}/bin/zstd --decompress --stdout \
    ${library}/share/reef/nix-builder-library-0.1.0.tar.zst > "$TMPDIR/library.tar"

  ${pkgs.python311}/bin/python3 - "$TMPDIR/library.tar" <<'PY'

  import sys
  import tarfile

  with tarfile.open(sys.argv[1]) as archive:
      for member in archive:
          assert member.mtime == 0, member.name
  PY

  printf '%s\n' "$actual" > "$out"
''
