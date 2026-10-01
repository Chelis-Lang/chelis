# Canonical source revisions and content hashes for tools that ci pins by more
# than a plain version string. Both the action-profile authority
# (devenv/shared/packages.nix) and the consumer devenv module
# (devenv/consumer/devenv.nix) read these, so a pin cannot drift between the
# CI-action surface and the shared consumer surface.
{
  crate2nix = {
    version = "0.15.0";
    rev = "7c33e664668faecf7655fa53861d7a80c9e464a2";
    hash = "sha256-SUuruvw1/moNzCZosHaa60QMTL+L9huWdsCBN6XZIic=";
  };
  openspec = {
    version = "1.6.0";
    tarballHash = "sha256-TTy06AdEKgr6xfcvJidQQJw662VWjtzLEZew8yUfezY=";
    npmDepsHash = "sha256-vEEIvL2OC5UiapCrOofJxwIPN8sqJw/in36Kvl2jxDk=";
    nodeVersion = "24.18.0";
  };
}
