{
  actionPackageAuthority,
  ciPkgs,
  inputs,
  lib,
  pkgs,
  toolchain,
  ...
}:

let
  inherit (lib) mkOption types;
  contracts = import ./contracts.nix;
  profileData = contracts.validateProfiles toolchain.devenv-profiles;
  system = pkgs.stdenv.hostPlatform.system;
  compatibleProfiles = lib.filterAttrs (
    _: profile: builtins.elem system profile.systems
  ) profileData.profiles;
  actionProfilePackages = builtins.mapAttrs (
    name: profile:
    let
      packageSystem =
        if builtins.elem system profile.systems then system else builtins.head profile.systems;
      packagePkgs =
        if packageSystem == system then ciPkgs else import inputs.ci-nixpkgs { system = packageSystem; };
      packageAuthority =
        if packageSystem == system then
          actionPackageAuthority
        else
          import ./packages.nix {
            ciPkgs = packagePkgs;
            inherit toolchain;
          };
    in
    packagePkgs.symlinkJoin {
      name = "chelis-action-profile-${name}-${profile.version}";
      paths = [ packageAuthority.${profile.constructor} ];
      passthru = {
        inherit profile;
        version = profile.version;
      };
      meta.platforms = profile.systems;
    }
  ) profileData.profiles;
  compatibleActionProfilePackages = lib.getAttrs (builtins.attrNames compatibleProfiles) actionProfilePackages;
  canonicalProfiles = builtins.mapAttrs (name: profile: {
    schema = profileData.schema;
    profile = name;
    inherit (profile)
      action
      systems
      constructor
      version
      digest
      runtime
      ;
  }) compatibleProfiles;
  # A profile whose record declares the nix C toolchain extends this profile;
  # every other shell keeps the base, which carries no C compiler.
  nixCToolchainProfile = "c-toolchain-nix";
  actionProfileDeclarations = builtins.mapAttrs (name: package: {
    extends = lib.optional (compatibleProfiles.${name}."c-toolchain" == "nix") nixCToolchainProfile;
    module = {
      packages = [ package ];
      files.".devenv/chelis-action-profile.json".json = canonicalProfiles.${name};
    };
  }) compatibleActionProfilePackages;
  # Devenv's own defaults, which the shared base replaces: the pinned stdenv
  # compiler wrapper and, on macOS, the Apple SDK.
  nixCToolchainDeclaration = {
    module = {
      stdenv = pkgs.stdenv;
      apple.sdk = if pkgs.stdenv.hostPlatform.isDarwin then pkgs.apple-sdk else null;
    };
  };
  versionType = types.strMatching "[0-9]+\\.[0-9]+\\.[0-9]+";
  profileType = types.submodule {
    options = {
      action = mkOption { type = types.strMatching "actions/[a-z0-9-]+"; };
      systems = mkOption {
        type = types.listOf (types.enum contracts.supportedSystems);
      };
      "c-toolchain" = mkOption { type = types.enum contracts.cToolchains; };
      constructor = mkOption {
        type = types.enum (builtins.attrValues contracts.profileConstructors);
      };
      version = mkOption { type = versionType; };
      digest = mkOption {
        type = types.nullOr (types.strMatching "sha256:[0-9a-f]{64}");
      };
      runtime = mkOption {
        type = types.attrsOf versionType;
        default = { };
      };
    };
  };
in
{
  options.chelis = {
    actionProfileSchema = mkOption {
      type = types.enum [ contracts.profileSchema ];
      description = "Schema for the closed action-profile inventory.";
    };
    actionProfiles = mkOption {
      type = types.attrsOf profileType;
      description = "Closed action profiles with typed package authority.";
    };
  };

  config = {
    _module.args.actionProfilePackages = actionProfilePackages;
    chelis.actionProfileSchema = profileData.schema;
    chelis.actionProfiles = profileData.profiles;
    profiles = actionProfileDeclarations // {
      ${nixCToolchainProfile} = nixCToolchainDeclaration;
    };
  };
}
