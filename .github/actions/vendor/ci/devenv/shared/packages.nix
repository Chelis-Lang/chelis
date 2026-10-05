{
  ciPkgs,
  toolchain,
}:

let
  inherit (ciPkgs) lib;
  contracts = import ./contracts.nix;
  pins = import ./pins.nix;
  profileRecords = (contracts.validateProfiles toolchain.devenv-profiles).profiles;
  profileVersion = name: profileRecords.${name}.version;
  profileRuntimeVersion = name: runtime: profileRecords.${name}.runtime.${runtime};
  system = ciPkgs.stdenv.hostPlatform.system;

  exactPackage =
    version: package:
    assert package.version == version;
    package;

  selectSource =
    name: sources:
    sources.${system} or (throw "devenv-profile:package:${name}:unsupported-system:${system}");

  mkWheelApplication =
    {
      pname,
      pinnedVersion,
      version,
      sources,
    }:
    assert version == pinnedVersion;
    let
      source = selectSource pname sources;
    in
    ciPkgs.python311Packages.buildPythonApplication {
      inherit pname version;
      format = "wheel";
      src = ciPkgs.fetchurl {
        inherit (source) url hash;
      };
      dependencies = [ ];
      doCheck = false;
      dontUsePythonRuntimeDepsCheck = true;
      nativeBuildInputs = lib.optionals ciPkgs.stdenv.hostPlatform.isLinux [
        ciPkgs.patchelf
      ];
      postFixup = lib.optionalString ciPkgs.stdenv.hostPlatform.isLinux ''
        patchelf \
          --set-interpreter ${ciPkgs.stdenv.cc.bintools.dynamicLinker} \
          --set-rpath ${lib.makeLibraryPath [ ciPkgs.stdenv.cc.cc.lib ]} \
          "$out/bin/${pname}"
      '';
      passthru.provenance = source;
      meta = {
        mainProgram = pname;
        platforms = builtins.attrNames sources;
      };
    };

  mkJoinedPackage =
    {
      name,
      version,
      paths,
      platforms,
      provenance ? null,
    }:
    ciPkgs.symlinkJoin {
      name = "${name}-${version}";
      inherit paths;
      passthru = {
        inherit version provenance;
      };
      meta.platforms = platforms;
    };

  ruffPackage = mkWheelApplication {
    pname = "ruff";
    pinnedVersion = "0.12.5";
    version = profileVersion "ruff";
    sources = {
      aarch64-darwin = {
        url = "https://files.pythonhosted.org/packages/c5/de/c6bec1dce5ead9f9e6a946ea15e8d698c35f19edc508289d70a577921b30/ruff-0.12.5-py3-none-macosx_11_0_arm64.whl";
        hash = "sha256-lid17Vsnx6o/3A2PTUQz3q52We+Z6iD3g9Zm53M4uM8=";
      };
      x86_64-linux = {
        url = "https://files.pythonhosted.org/packages/24/ff/96058f6506aac0fbc0d0fc0d60b0d0bd746240a0594657a2d94ad28033ba/ruff-0.12.5-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
        hash = "sha256-LEfepq45QhhRaFFBupc0dn+WARPVHoP9e7mVjVvodjo=";
      };
    };
  };

  zizmorPackage = mkWheelApplication {
    pname = "zizmor";
    pinnedVersion = "1.28.0";
    version = profileVersion "zizmor";
    sources.x86_64-linux = {
      url = "https://files.pythonhosted.org/packages/5b/b4/f823bd2a1ba6dc432fdcbd249d4c442ce79bd137d34d986fce5c181f2800/zizmor-1.28.0-py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-riyrZ85xPnYODRthrXSdN0aT6isxAzeqsRzURnSCZ/M=";
    };
  };

  tyPackage = mkWheelApplication {
    pname = "ty";
    pinnedVersion = "0.0.63";
    version = profileVersion "check-python-types";
    sources.x86_64-linux = {
      url = "https://files.pythonhosted.org/packages/f5/be/e280ad095b050778f16f493d49a073fa5f6d8f301d3e2e59be6a672ba05c/ty-0.0.63-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      hash = "sha256-UExEV/OmKv6DbB8mosmhJUkpkJX5zEFGd4VY31GnUVw=";
    };
  };

  openspecSource = ciPkgs.runCommand "openspec-npm-source-1.6.0" { } ''
    mkdir "$out"
    cp ${./npm/openspec/package.json} "$out/package.json"
    cp ${./npm/openspec/package-lock.json} "$out/package-lock.json"
  '';
  openspecProvenance = {
    url = "https://registry.npmjs.org/@fission-ai/openspec/-/openspec-1.6.0.tgz";
    hash = pins.openspec.tarballHash;
    nodeVersion = profileRuntimeVersion "openspec-governance" "node";
  };
  openspecCli =
    assert profileVersion "openspec-governance" == "1.6.0";
    ciPkgs.buildNpmPackage {
      pname = "openspec";
      version = profileVersion "openspec-governance";
      src = openspecSource;
      nodejs = exactPackage (profileRuntimeVersion "openspec-governance" "node") ciPkgs.nodejs_24;
      npmDepsHash = pins.openspec.npmDepsHash;
      npmInstallFlags = [ "--ignore-scripts" ];
      dontNpmBuild = true;
      postInstall = ''
        mkdir -p "$out/bin"
        ln -s ../lib/node_modules/chelis-openspec-governance-action/node_modules/.bin/openspec "$out/bin/openspec"
      '';
      passthru.provenance = openspecProvenance;
      meta = {
        mainProgram = "openspec";
        platforms = [
          "aarch64-darwin"
          "x86_64-linux"
        ];
      };
    };

  # nixpkgs fetches crate2nix with the same content hash as pins.crate2nix.
  # The package can therefore use the binary from cache.nixos.org.
  crate2nixPackage =
    assert profileVersion "check-cargo-nix" == "0.15.0";
    assert ciPkgs.crate2nix.src.outputHash == pins.crate2nix.hash;
    exactPackage (profileVersion "check-cargo-nix") ciPkgs.crate2nix;

  allSystems = [
    "aarch64-darwin"
    "x86_64-linux"
  ];
  linuxSystems = [ "x86_64-linux" ];
in
{
  common = [
    ciPkgs.bash
    ciPkgs.coreutils
    ciPkgs.diffutils
    ciPkgs.findutils
    ciPkgs.gawk
    ciPkgs.git
    ciPkgs.gnugrep
    ciPkgs.gnused
    ciPkgs.python311
    ciPkgs.which
  ];

  "nixpkgs-actionlint" = exactPackage (profileVersion "actionlint") ciPkgs.actionlint;
  "nixpkgs-cargo-deny" = exactPackage (profileVersion "cargo-deny") ciPkgs.cargo-deny;
  "nixpkgs-cargo-llvm-cov" = exactPackage (profileVersion "cargo-llvm-cov") ciPkgs.cargo-llvm-cov;
  "nixpkgs-cargo-machete" = exactPackage (profileVersion "cargo-machete") ciPkgs.cargo-machete;
  "nixpkgs-cargo-mutants" = exactPackage (profileVersion "cargo-mutants") ciPkgs.cargo-mutants;
  "nixpkgs-cargo-criterion" =
    exactPackage (profileVersion "setup-cargo-criterion") ciPkgs.cargo-criterion;
  "nixpkgs-cargo-flamegraph" =
    exactPackage (profileVersion "setup-flamegraph") ciPkgs.cargo-flamegraph;
  "nixpkgs-hyperfine" = exactPackage (profileVersion "setup-hyperfine") ciPkgs.hyperfine;
  "crate2nix-source" = crate2nixPackage;
  "ty-wheel" = tyPackage;
  "container-tools" = mkJoinedPackage {
    name = "container-tools";
    version = profileVersion "container-image";
    paths = [
      (exactPackage (profileVersion "container-image") ciPkgs.nix)
      ciPkgs.podman
      ciPkgs.python311
    ];
    platforms = linuxSystems;
  };
  "openspec-npm" = mkJoinedPackage {
    name = "openspec";
    version = profileVersion "openspec-governance";
    paths = [
      (exactPackage (profileRuntimeVersion "openspec-governance" "node") ciPkgs.nodejs_24)
      openspecCli
    ];
    platforms = allSystems;
    provenance = openspecProvenance;
  };
  "ruff-wheel" = ruffPackage;
  "rust-quality-helpers" = mkJoinedPackage {
    name = "rust-quality-helpers";
    version = profileVersion "rust-lint";
    paths = [ (exactPackage (profileVersion "rust-lint") ciPkgs.python311) ];
    platforms = linuxSystems;
  };
  "rust-release-helpers" = mkJoinedPackage {
    name = "rust-release-helpers";
    version = profileVersion "rust-release-binary";
    paths = [ (exactPackage (profileVersion "rust-release-binary") ciPkgs.python311) ];
    platforms = linuxSystems;
  };
  "nixpkgs-typos" = exactPackage (profileVersion "typos") ciPkgs.typos;
  "zizmor-wheel" = zizmorPackage;
}
