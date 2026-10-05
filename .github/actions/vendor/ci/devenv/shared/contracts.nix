let
  profileSchema = "chelis-devenv-profiles/v1";
  supportedSystems = [
    "aarch64-darwin"
    "x86_64-linux"
  ];
  profileActions = {
    actionlint = "actions/actionlint";
    "cargo-deny" = "actions/cargo-deny";
    "cargo-llvm-cov" = "actions/cargo-llvm-cov";
    "cargo-machete" = "actions/cargo-machete";
    "cargo-mutants" = "actions/cargo-mutants";
    "check-cargo-nix" = "actions/check-cargo-nix";
    "check-python-types" = "actions/check-python-types";
    "container-image" = "actions/container-image";
    "openspec-governance" = "actions/openspec-governance";
    ruff = "actions/ruff";
    "rust-lint" = "actions/rust-lint";
    "rust-release-binary" = "actions/rust-release-binary";
    "setup-cargo-criterion" = "actions/setup-cargo-criterion";
    "setup-flamegraph" = "actions/setup-flamegraph";
    "setup-hyperfine" = "actions/setup-hyperfine";
    typos = "actions/typos";
    zizmor = "actions/zizmor";
  };
  profileConstructors = {
    actionlint = "nixpkgs-actionlint";
    "cargo-deny" = "nixpkgs-cargo-deny";
    "cargo-llvm-cov" = "nixpkgs-cargo-llvm-cov";
    "cargo-machete" = "nixpkgs-cargo-machete";
    "cargo-mutants" = "nixpkgs-cargo-mutants";
    "check-cargo-nix" = "crate2nix-source";
    "check-python-types" = "ty-wheel";
    "container-image" = "container-tools";
    "openspec-governance" = "openspec-npm";
    ruff = "ruff-wheel";
    "rust-lint" = "rust-quality-helpers";
    "rust-release-binary" = "rust-release-helpers";
    "setup-cargo-criterion" = "nixpkgs-cargo-criterion";
    "setup-flamegraph" = "nixpkgs-cargo-flamegraph";
    "setup-hyperfine" = "nixpkgs-hyperfine";
    typos = "nixpkgs-typos";
    zizmor = "zizmor-wheel";
  };
  crossSystemProfiles = [
    "check-cargo-nix"
    "openspec-governance"
    "ruff"
  ];
  profileSystems = builtins.mapAttrs (
    name: _: if builtins.elem name crossSystemProfiles then supportedSystems else [ "x86_64-linux" ]
  ) profileActions;
  digestProfiles = [
    "actionlint"
    "cargo-deny"
    "cargo-llvm-cov"
    "cargo-machete"
    "cargo-mutants"
  ];
  profileRuntimeKeys = {
    "openspec-governance" = [ "node" ];
  };
  recordFields = [
    "action"
    "constructor"
    "digest"
    "runtime"
    "systems"
    "version"
  ];
  sharedModuleOwners = [
    {
      path = "devenv/shared/packages.nix";
      concerns = [ "common-packages" ];
    }
    {
      path = "devenv/shared/profiles.nix";
      concerns = [ "action-profiles" ];
    }
    {
      path = "devenv/shared/container.nix";
      concerns = [ "container-constructor" ];
    }
  ];
  outputConstructors = {
    "action-profile-packages" = "action-profiles";
    "check-actions" = "ci-tools";
    "ci-tools" = "ci-tools";
    "container-ci-image" = "container-constructor";
    "container-contract-fixtures" = "container-constructor";
  };
  outputFields = [
    "constructor"
    "credentialInputs"
    "mutable"
    "name"
  ];
  ciTaskDependencies = {
    "ci:tracked-before" = [ ];
    "ci:toolchain-versions" = [
      "ci:tracked-before"
      "devenv:python:uv"
    ];
    "ci:devenv-contract" = [
      "ci:tracked-before"
      "devenv:python:uv"
    ];
    "ci:apple-container-contract" = [
      "ci:tracked-before"
      "devenv:python:uv"
    ];
    "ci:toolchain" = [
      "ci:toolchain-versions"
      "ci:devenv-contract"
      "ci:apple-container-contract"
    ];
    "ci:architecture" = [
      "ci:tracked-before"
      "devenv:python:uv"
    ];
    "ci:nar-hash" = [ "ci:tracked-before" ];
    "ci:deterministic" = [
      "ci:toolchain"
      "ci:python-tests"
      "ci:nar-hash"
    ];
    "ci:container-image" = [ "ci:deterministic" ];
    "ci:governance-check" = [
      "ci:tracked-before"
      "devenv:python:uv"
    ];
    "ci:governance" = [
      "ci:deterministic"
      "ci:governance-check"
    ];
    "ci:type-check" = [ "ci:deterministic" ];
    "ci:lint" = [ "ci:deterministic" ];
    "ci:format-check" = [ "ci:deterministic" ];
    "ci:readme-check" = [ "ci:deterministic" ];
    "ci:warm-cache" = [ "ci:deterministic" ];
    "ci:warm-nix-cache" = [ "ci:deterministic" ];
    "ci:parity" = [
      "ci:toolchain-versions"
      "ci:devenv-contract"
      "ci:apple-container-contract"
      "ci:toolchain"
      "ci:architecture"
      "ci:deterministic"
      "ci:governance-check"
      "ci:governance"
      "ci:type-check"
      "ci:lint"
      "ci:format-check"
      "ci:readme-check"
      "ci:warm-cache"
      "ci:warm-nix-cache"
    ];
    "devenv:git-hooks:run" = [ "ci:parity" ];
    "ci:tracked-after" = [ "devenv:git-hooks:run" ];
  };
  taskReaches =
    tasks: target: current: visited:
    if current == target then
      true
    else if builtins.elem current visited then
      false
    else
      builtins.any (dependency: taskReaches tasks target dependency (visited ++ [ current ])) (
        tasks.${current} or [ ]
      );
  versionMatches =
    value: builtins.isString value && builtins.match "[0-9]+\\.[0-9]+\\.[0-9]+" value != null;
  digestMatches =
    value: builtins.isString value && builtins.match "sha256:[0-9a-f]{64}" value != null;
  unique =
    values:
    builtins.foldl' (
      result: value: if builtins.elem value result then result else result ++ [ value ]
    ) [ ] values;
  concatMap = function: values: builtins.concatLists (map function values);
  testModuleName = name: builtins.match "test_.*\\.py" name != null;
  matchesSuitePattern =
    pattern: name: if pattern == "test_*.py" then testModuleName name else name == pattern;
  safeSuiteDirectory =
    value:
    builtins.isString value
    && builtins.match "(actions|tools)(/[a-z0-9_-]+)*" value != null
    && builtins.match ".*(^|/)\\.\\.(/|$).*" value == null;
  readDirectory =
    path:
    let
      result = builtins.tryEval (builtins.readDir path);
    in
    if result.success then result.value else throw "chelis-contract:suite-boundary";
  discoverTestModules =
    root: relative:
    let
      entries = readDirectory (root + "/${relative}");
    in
    concatMap (
      name:
      let
        type = entries.${name};
        child = "${relative}/${name}";
      in
      if type == "directory" then
        discoverTestModules root child
      else if type == "regular" && testModuleName name then
        [ child ]
      else if type == "symlink" && testModuleName name then
        throw "chelis-contract:suite-boundary"
      else
        [ ]
    ) (builtins.attrNames entries);
  suiteModules =
    root: suite:
    let
      entries = readDirectory (root + "/${suite.directory}");
    in
    concatMap (
      name:
      if matchesSuitePattern suite.pattern name then
        if entries.${name} == "regular" then
          [ "${suite.directory}/${name}" ]
        else
          throw "chelis-contract:suite-boundary"
      else
        [ ]
    ) (builtins.attrNames entries);
  validRecord =
    name: record:
    let
      needsDigest = builtins.elem name digestProfiles;
      runtimeKeys = profileRuntimeKeys.${name} or [ ];
    in
    builtins.isAttrs record
    && builtins.attrNames record == recordFields
    && record.action == profileActions.${name}
    && record.constructor == profileConstructors.${name}
    && record.systems == profileSystems.${name}
    && versionMatches record.version
    && builtins.isAttrs record.runtime
    && builtins.attrNames record.runtime == runtimeKeys
    && builtins.all versionMatches (builtins.attrValues record.runtime)
    && (
      if needsDigest then
        digestMatches record.digest
      else
        builtins.isString record.digest && record.digest == ""
    );
  normalizeRecord =
    _: record: record // { digest = if record.digest == "" then null else record.digest; };
in
rec {
  inherit
    profileSchema
    supportedSystems
    profileActions
    profileConstructors
    profileSystems
    digestProfiles
    profileRuntimeKeys
    ciTaskDependencies
    ;

  validateProfiles =
    value:
    let
      validEnvelope =
        builtins.isAttrs value
        &&
          builtins.attrNames value == [
            "profiles"
            "schema"
          ]
        && value.schema == profileSchema
        && builtins.isAttrs value.profiles;
      names = if validEnvelope then builtins.attrNames value.profiles else [ ];
      expectedNames = builtins.attrNames profileActions;
      namesAreValid = names == expectedNames && builtins.length names <= 32;
      recordsAreValid =
        namesAreValid && builtins.all (name: validRecord name value.profiles.${name}) names;
      actions = if recordsAreValid then map (name: value.profiles.${name}.action) names else [ ];
      actionsAreUnique = builtins.length (unique actions) == builtins.length actions;
    in
    if !validEnvelope then
      throw "devenv-profile:inventory:envelope"
    else if !namesAreValid then
      throw "devenv-profile:inventory:names"
    else if !recordsAreValid then
      throw "devenv-profile:inventory:record"
    else if !actionsAreUnique then
      throw "devenv-profile:inventory:duplicate-action"
    else
      {
        schema = profileSchema;
        profiles = builtins.mapAttrs normalizeRecord value.profiles;
      };

  validateSuiteInventory =
    value:
    let
      validEnvelope =
        builtins.isAttrs value
        &&
          builtins.attrNames value == [
            "credentialVariables"
            "root"
            "suites"
          ]
        && builtins.isString value.root
        && builtins.isList value.suites
        && builtins.isList value.credentialVariables;
      validSuite =
        suite:
        builtins.isAttrs suite
        &&
          builtins.attrNames suite == [
            "directory"
            "name"
            "owner"
            "pattern"
          ]
        && builtins.isString suite.name
        && builtins.match "[a-z0-9][a-z0-9-]{0,63}" suite.name != null
        && safeSuiteDirectory suite.directory
        && builtins.isString suite.pattern
        && (suite.pattern == "test_*.py" || builtins.match "test_[a-z0-9_]+\\.py" suite.pattern != null)
        && builtins.elem suite.owner [
          "architecture"
          "driver"
          "governance"
          "toolchain"
        ];
      recordsAreValid =
        validEnvelope
        && builtins.length value.suites > 0
        && builtins.length value.suites <= 64
        && builtins.all validSuite value.suites;
      suiteNames = if recordsAreValid then map (suite: suite.name) value.suites else [ ];
      root = if recordsAreValid then builtins.toPath value.root else null;
      modulesBySuite = if recordsAreValid then map (suite: suiteModules root suite) value.suites else [ ];
      ownedModules = builtins.concatLists modulesBySuite;
      driverSuites =
        if recordsAreValid then builtins.filter (suite: suite.owner == "driver") value.suites else [ ];
      driverModules =
        if recordsAreValid then
          builtins.concatLists (map (suite: suiteModules root suite) driverSuites)
        else
          [ ];
      actualModules =
        if recordsAreValid then
          builtins.concatLists (
            map
              (
                relative:
                if builtins.pathExists (root + "/${relative}") then discoverTestModules root relative else [ ]
              )
              [
                "actions"
                "tools"
              ]
          )
        else
          [ ];
      namesAreUnique = builtins.length (unique suiteNames) == builtins.length suiteNames;
      modulesAreUnique = builtins.length (unique ownedModules) == builtins.length ownedModules;
      modulesAreComplete =
        builtins.sort builtins.lessThan ownedModules == builtins.sort builtins.lessThan actualModules;
      modulesAreBounded = builtins.length ownedModules > 0 && builtins.length ownedModules <= 256;
      governanceSuites =
        if recordsAreValid then
          builtins.filter (suite: suite.directory == "actions/openspec-governance") value.suites
        else
          [ ];
      governanceOwnerIsValid = builtins.all (
        suite:
        suite.name == "openspec-governance" && suite.pattern == "test_*.py" && suite.owner == "governance"
      ) governanceSuites;
      credentialsAreSafe =
        validEnvelope
        &&
          value.credentialVariables == [
            "GH_TOKEN"
            "GITHUB_TOKEN"
            "CHELIS_RELEASE_TOKEN"
          ];
    in
    if !validEnvelope || !recordsAreValid then
      throw "chelis-contract:suite-boundary"
    else if !credentialsAreSafe then
      throw "chelis-contract:suite-credentials"
    else if !governanceOwnerIsValid then
      throw "chelis-contract:suite-governance-owner"
    else if !namesAreUnique || !modulesAreUnique || !modulesAreComplete || !modulesAreBounded then
      throw "chelis-contract:suite-inventory"
    else
      {
        suites = value.suites;
        suiteCount = builtins.length value.suites;
        moduleCount = builtins.length ownedModules;
        driverSuiteCount = builtins.length driverSuites;
        driverModuleCount = builtins.length driverModules;
      };

  validateSuiteCounts =
    value:
    if
      builtins.isAttrs value
      &&
        builtins.attrNames value == [
          "driverModuleCount"
          "driverSuiteCount"
          "moduleCount"
          "suiteCount"
        ]
      &&
        value == {
          driverModuleCount = 57;
          driverSuiteCount = 36;
          moduleCount = 63;
          suiteCount = 41;
        }
    then
      value
    else
      throw "chelis-contract:suite-count";

  validateTaskGraph =
    value:
    let
      validEnvelope =
        builtins.isAttrs value
        && builtins.attrNames value == [ "tasks" ]
        && builtins.isAttrs value.tasks
        && builtins.length (builtins.attrNames value.tasks) <= 64;
      validDependencies =
        validEnvelope
        && builtins.all (
          name:
          builtins.isList value.tasks.${name}
          && builtins.length value.tasks.${name} <= 32
          && builtins.all builtins.isString value.tasks.${name}
          && builtins.length (unique value.tasks.${name}) == builtins.length value.tasks.${name}
        ) (builtins.attrNames value.tasks);
      imageIsInParity = validDependencies && taskReaches value.tasks "ci:container-image" "ci:parity" [ ];
      governanceLeafIsFocused =
        validDependencies
        &&
          (value.tasks."ci:governance-check" or [ ]) == [
            "ci:tracked-before"
            "devenv:python:uv"
          ];
      governanceAggregateIsComplete =
        validDependencies
        &&
          (value.tasks."ci:governance" or [ ]) == [
            "ci:deterministic"
            "ci:governance-check"
          ];
      governanceParityIsComplete =
        validDependencies
        && builtins.elem "ci:governance-check" (value.tasks."ci:parity" or [ ])
        && builtins.elem "ci:governance" (value.tasks."ci:parity" or [ ]);
    in
    if !validDependencies then
      throw "chelis-contract:task-graph"
    else if imageIsInParity then
      throw "chelis-contract:task-graph-image"
    else if !governanceLeafIsFocused then
      throw "chelis-contract:task-graph-governance-leaf"
    else if !governanceAggregateIsComplete then
      throw "chelis-contract:task-graph-governance-aggregate"
    else if !governanceParityIsComplete then
      throw "chelis-contract:task-graph-governance-parity"
    else if value.tasks != ciTaskDependencies then
      throw "chelis-contract:task-graph"
    else
      {
        tasks = value.tasks;
        requiredTaskCount = builtins.length value.tasks."ci:parity";
      };

  validateDeclarations =
    value:
    let
      validEnvelope =
        builtins.isAttrs value
        &&
          builtins.attrNames value == [
            "moduleOwners"
            "outputs"
            "portablePackageAuthority"
          ]
        && builtins.isList value.moduleOwners
        && builtins.isList value.outputs;
      ownersAreValid = validEnvelope && value.moduleOwners == sharedModuleOwners;
      outputNames =
        if validEnvelope then
          map (record: if builtins.isAttrs record then record.name or null else null) value.outputs
        else
          [ ];
      expectedOutputNames = builtins.attrNames outputConstructors;
      outputInventoryIsValid =
        builtins.length outputNames <= 16
        && outputNames == expectedOutputNames
        && builtins.length (unique outputNames) == builtins.length outputNames;
      validOutput =
        record:
        builtins.isAttrs record
        && builtins.attrNames record == outputFields
        && builtins.hasAttr record.name outputConstructors
        && record.constructor == outputConstructors.${record.name}
        && builtins.isList record.credentialInputs
        && record.credentialInputs == [ ]
        && builtins.isBool record.mutable
        && !record.mutable;
      outputsArePure = outputInventoryIsValid && builtins.all validOutput value.outputs;
    in
    if !validEnvelope then
      throw "chelis-contract:declaration-envelope"
    else if !ownersAreValid then
      throw "chelis-contract:shared-module-owner"
    else if value.portablePackageAuthority != "shared" then
      throw "chelis-contract:copied-logic"
    else if !outputInventoryIsValid then
      throw "chelis-contract:output-inventory"
    else if !outputsArePure then
      throw "chelis-contract:output-purity"
    else
      {
        moduleOwners = map (record: record.path) sharedModuleOwners;
        outputs = expectedOutputNames;
      };
}
