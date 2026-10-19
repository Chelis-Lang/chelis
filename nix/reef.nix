{ lib, chelisForSystem }:
let
  platforms = {
    x86_64-linux = "linux-x86_64";
    aarch64-linux = "linux-arm64";
    aarch64-darwin = "darwin-arm64";
  };
  fail = message: throw "Chelis Nix builder: ${message}";
  matches = pattern: value: builtins.isString value && builtins.match pattern value != null;
  hash = value: if matches "[0-9a-f]{64}" value then value else fail "invalid SHA-256 in reef.lock";
  assetName =
    value:
    if matches "[A-Za-z0-9._~-]+" value && value != "." && value != ".." then
      value
    else
      fail "unsafe release asset name in reef.lock";
  origin =
    value:
    let
      parts =
        if builtins.isString value then builtins.match "github://([^/]+)/([^@]+)@(.+)" value else null;
      org = builtins.elemAt parts 0;
      repo = builtins.elemAt parts 1;
      tag = builtins.elemAt parts 2;
    in
    if parts == null then
      fail "a locked download needs a github:// organization, repository, and tag"
    else if
      !(matches "[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?" org)
      || builtins.stringLength org > 39
      || !(matches "[a-z0-9]+(-[a-z0-9]+)*" repo)
      || !(matches "v?[0-9]+[.][0-9]+[.][0-9]+(-[A-Za-z0-9.-]+)?" tag)
    then
      fail "invalid GitHub release origin in reef.lock"
    else
      {
        inherit org repo tag;
        version = lib.removePrefix "v" tag;
      };
  download = source: file: sha256: {
    file = assetName file;
    sha256 = hash sha256;
    url = "https://github.com/${source.org}/${source.repo}/releases/download/${source.tag}/${assetName file}";
  };
  lockedDownloads =
    platform: dependency:
    let
      source = dependency.source;
      release = origin (source.remote_origin or null);
      sameVersion = release.version == dependency.version;
    in
    if source.kind == "path" || source.kind == "bundled" then
      [ ]
    else if source.kind == "local_registry" && dependency.name == "chelis-std" then
      [ ]
    else if source.kind == "local_registry" then
      if !sameVersion then
        fail "the locked package version differs from its release tag"
      else
        [
          (download release "${release.repo}-${release.version}.tar.zst" dependency.archive_sha256)
          (download release "${release.repo}-${release.version}.chb" dependency.shell_sha256)
        ]
    else if source.kind == "binary" then
      if source.platform != platform then
        [ ]
      else if !sameVersion then
        fail "the locked binary version differs from its release tag"
      else if source.sha256 != dependency.archive_sha256 || dependency.shell_sha256 != "" then
        fail "the binary hash fields in reef.lock disagree"
      else
        [ (download release source.asset source.sha256) ]
    else
      fail "unknown source kind in reef.lock";
  fetchReefLock =
    {
      pkgs,
      lockFile,
      system ? pkgs.stdenv.hostPlatform.system,
    }:
    let
      lock = builtins.fromTOML (builtins.readFile lockFile);
      platform = platforms.${system} or (fail "unsupported system ${system}");
      downloads = lib.concatMap (lockedDownloads platform) (lock.dependencies or [ ]);
      files = lib.foldl' (
        result: item:
        if result ? ${item.file} then
          fail "two locked downloads use the file name ${item.file}"
        else
          result // { ${item.file} = item; }
      ) { } downloads;
    in
    if
      !(lib.elem (lock.schema or "0") [
        "0"
        "1"
      ])
    then
      fail "unsupported reef.lock schema"
    else
      pkgs.linkFarm "reef-lock-artifacts" (
        lib.mapAttrsToList (name: item: {
          inherit name;
          path = pkgs.fetchurl {
            inherit (item) url sha256;
            name = item.file;
          };
        }) files
      );
  buildChelisPackage =
    {
      pkgs,
      pname,
      version,
      src,
      chelis ? chelisForSystem pkgs.stdenv.hostPlatform.system,
      entryPoint ? "src/main.ch",
      reefPackage ? false,
      lockFile ? null,
      artifacts ? if lockFile == null then null else fetchReefLock { inherit pkgs lockFile; },
      sourceDateEpoch ? 0,
      nativeBuildInputs ? [ ],
      ...
    }@args:
    let
      program = lib.removeSuffix ".ch" (baseNameOf entryPoint);
      remaining = builtins.removeAttrs args [
        "pkgs"
        "chelis"
        "entryPoint"
        "reefPackage"
        "lockFile"
        "artifacts"
        "sourceDateEpoch"
        "nativeBuildInputs"
      ];
    in
    pkgs.stdenvNoCC.mkDerivation (
      remaining
      // {
        inherit pname version src;
        nativeBuildInputs = [ chelis ] ++ nativeBuildInputs;
        strictDeps = true;
        dontConfigure = true;
        SOURCE_DATE_EPOCH = toString sourceDateEpoch;
        CHELIS_OFFLINE = "1";
        buildPhase = ''
          export HOME="$TMPDIR/home"
          export CHELIS_HOME="$TMPDIR/chelis-home"
          mkdir -p "$HOME" "$CHELIS_HOME"
          export SOURCE_DATE_EPOCH=${lib.escapeShellArg (toString sourceDateEpoch)}
          runHook preBuild
          ${lib.optionalString (lockFile != null) ''
            cp ${lib.escapeShellArg "${lockFile}"} reef.lock
            chmod u+w reef.lock
          ''}
          ${lib.optionalString (artifacts != null) ''
            chelis reef install --from-lockfile --artifact-dir ${lib.escapeShellArg "${artifacts}"}
          ''}
          ${
            if reefPackage then
              "chelis reef build --no-auto-fetch ."
            else
              "chelis build ${lib.escapeShellArg entryPoint} --output build"
          }
          runHook postBuild
        '';
        installPhase = ''
          runHook preInstall
          ${
            if reefPackage then
              ''
                mkdir -p "$out/share/reef"
                cp "dist/${pname}-${version}.tar.zst" "dist/${pname}-${version}.chb" "$out/share/reef/"
              ''
            else
              ''
                install -Dm755 ${lib.escapeShellArg "build/${program}"} "$out/bin/${pname}"
              ''
          }
          runHook postInstall
        '';
      }
    );
in
{
  inherit fetchReefLock buildChelisPackage;
}
