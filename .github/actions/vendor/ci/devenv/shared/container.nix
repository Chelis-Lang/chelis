{
  nix2container,
  nixpkgs,
}:

let
  containerImageUid = 1000;
  containerImageGid = 1000;
  containerImageMaxLayers = 8;
  containerImageCreated = "0001-01-01T00:00:00Z";
in
rec {
  inherit
    containerImageUid
    containerImageGid
    containerImageMaxLayers
    containerImageCreated
    ;

  mkContainerCiImage =
    {
      pkgs,
      helperRevision,
      systemPackages,
      toolchainPackages,
      consumerRoot,
    }:
    assert builtins.match "[0-9a-f]{40}" helperRevision != null;
    assert builtins.isList systemPackages;
    assert builtins.isList toolchainPackages;
    assert builtins.all pkgs.lib.isDerivation systemPackages;
    assert builtins.all pkgs.lib.isDerivation toolchainPackages;
    assert pkgs.lib.isDerivation consumerRoot;
    let
      container = (import nix2container { inherit pkgs; }).nix2container;
      systemRoot = pkgs.buildEnv {
        name = "container-ci-system-root";
        paths = systemPackages;
        pathsToLink = [
          "/bin"
          "/etc"
          "/lib"
          "/share"
        ];
      };
      toolchainRoot = pkgs.buildEnv {
        name = "container-ci-toolchain-root";
        paths = toolchainPackages;
        pathsToLink = [
          "/bin"
          "/include"
          "/lib"
          "/share"
        ];
      };
      userRoot = pkgs.runCommand "container-ci-user-root" { } ''
        mkdir -p "$out/etc" "$out/home/ci"
        ln -s /share/chelis "$out/home/ci/.chelis"
        printf '%s\n' 'ci:x:${toString containerImageUid}:${toString containerImageGid}:CI user:/home/ci:/bin/sh' > "$out/etc/passwd"
        printf '%s\n' 'ci:x:${toString containerImageGid}:' > "$out/etc/group"
      '';
      systemLayer = container.buildLayer {
        deps = systemPackages;
        copyToRoot = systemRoot;
        maxLayers = 4;
        reproducible = true;
        metadata = {
          created_by = "mkContainerCiImage:system";
          author = "Chelis-Lang/ci";
          comment = "stable system closure";
        };
      };
      toolchainLayer = container.buildLayer {
        deps = toolchainPackages;
        copyToRoot = toolchainRoot;
        layers = [ systemLayer ];
        maxLayers = 2;
        reproducible = true;
        metadata = {
          created_by = "mkContainerCiImage:toolchain";
          author = "Chelis-Lang/ci";
          comment = "stable toolchain closure";
        };
      };
      consumerLayer = container.buildLayer {
        copyToRoot = [
          userRoot
          consumerRoot
        ];
        layers = [
          systemLayer
          toolchainLayer
        ];
        maxLayers = 1;
        reproducible = true;
        perms = [
          {
            path = userRoot;
            regex = "/home/ci";
            mode = "0755";
            uid = containerImageUid;
            gid = containerImageGid;
            uname = "ci";
            gname = "ci";
          }
          {
            path = consumerRoot;
            regex = "/workspace(/.*)?";
            mode = "0755";
            uid = containerImageUid;
            gid = containerImageGid;
            uname = "ci";
            gname = "ci";
          }
        ];
        metadata = {
          created_by = "mkContainerCiImage:consumer";
          author = "consumer";
          comment = "volatile consumer closure";
        };
      };
      image = container.buildImage {
        name = "chelis-container-ci";
        arch = "amd64";
        created = containerImageCreated;
        maxLayers = 1;
        layers = [
          systemLayer
          toolchainLayer
          consumerLayer
        ];
        config = {
          User = "${toString containerImageUid}:${toString containerImageGid}";
          WorkingDir = "/workspace";
          Env = [
            "HOME=/home/ci"
            "LIBRARY_PATH=/lib"
            "PATH=/bin"
          ];
        };
        meta = {
          contract = "chelis-container-nix-contract/v1";
          inherit helperRevision;
        };
      };
      contract = {
        schema = "chelis-container-nix-contract/v1";
        output = "packages.x86_64-linux.container-ci-image";
        inherit helperRevision;
        lockedInputs = [
          {
            name = "nix2container";
            revision = nix2container.rev;
            narHash = nix2container.narHash;
          }
          {
            name = "nixpkgs";
            revision = nixpkgs.rev;
            narHash = nixpkgs.narHash;
          }
        ];
        architecture = "amd64";
        created = containerImageCreated;
        defaultUser = "${toString containerImageUid}:${toString containerImageGid}";
        workingDirectory = "/workspace";
        maxLayers = containerImageMaxLayers;
        stableLayers = [
          {
            name = "system";
            derivation = systemLayer.drvPath;
          }
          {
            name = "toolchain";
            derivation = toolchainLayer.drvPath;
          }
        ];
        consumerLayer = {
          name = "consumer";
          derivation = consumerLayer.drvPath;
        };
        overrideInputs = [ ];
        daemonRequired = false;
        networkRequired = false;
      };
      contractFile = pkgs.writeText "container-ci-contract.json" (builtins.toJSON contract);
    in
    image.overrideAttrs (old: {
      passthru = (old.passthru or { }) // {
        containerCiContract = contract;
        containerCiContractFile = contractFile;
        containerCiStableLayers = {
          system = systemLayer;
          toolchain = toolchainLayer;
        };
        inherit consumerLayer;
      };
    });

  mkSyntheticContainerImage =
    pkgs: marker:
    mkContainerCiImage {
      inherit pkgs;
      helperRevision = "ffffffffffffffffffffffffffffffffffffffff";
      systemPackages = [
        pkgs.bashInteractive
        pkgs.cacert
        pkgs.coreutils
      ];
      toolchainPackages = [ pkgs.python311 ];
      consumerRoot = pkgs.writeTextDir "workspace/fixture.txt" marker;
    };
}
