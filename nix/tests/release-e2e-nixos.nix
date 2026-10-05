# Boots two NixOS VMs and runs scripts/release_e2e.py against a release's
# Linux assets in each. `stock` is the default configuration; `nixld` also
# enables programs.nix-ld. Both machines carry Python, GCC, and binutils, which
# the harness and `chelis build` need. The VMs have no network, so the online
# steps skip. The test records each outcome without failing on it; the
# Release E2E workflow checks the reports.
#
#   nix build --file nix/tests/release-e2e-nixos.nix \
#     --arg assets /abs/assets --arg gates /abs/gates --arg scripts /abs/scripts \
#     --argstr tag vX.Y.Z --argstr archiveLabel vX.Y.Z|dev-<sha8> \
#     --argstr sourceSha <40-hex commit>
{
  assets,
  gates,
  # The repository's scripts/ directory: the harness and the canary modules it
  # imports.
  scripts,
  tag,
  archiveLabel ? tag,
  sourceSha,
  # nixos-26.05 on 2026-10-04.
  nixpkgs ? fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/825e2028c29b702a4a5f085f08095d12099784f2.tar.gz";
    sha256 = "12wb45knkzlkaipb5nyi8p7y5209zbn90wxmqqnrz2bc9npq6lgw";
  },
}:
let
  pkgs = import nixpkgs {
    system = "x86_64-linux";
    config = { };
    overlays = [ ];
  };
  inherit (pkgs) lib;
  # Only the Linux assets enter the store the VMs share.
  linuxAssets = builtins.path {
    name = "chelis-release-assets";
    path = assets;
    filter = path: _type: !(lib.hasInfix "darwin" (baseNameOf path));
  };
  gateTree = builtins.path {
    name = "chelis-release-gates";
    path = gates;
  };
  harness = builtins.path {
    name = "chelis-release-e2e";
    path = scripts;
    filter =
      path: _type:
      builtins.elem (baseNameOf path) [
        "release_e2e.py"
        "installed_artifact_canary.py"
        "verify_runtime_package.py"
      ];
  };
  host =
    { pkgs, ... }:
    {
      virtualisation = {
        memorySize = 4096;
        diskSize = 8192;
        cores = 2;
      };
      environment.systemPackages = [
        pkgs.python3
        pkgs.gcc
        pkgs.binutils
        pkgs.gnutar
        pkgs.gzip
      ];
    };
in
pkgs.testers.runNixOSTest {
  name = "chelis-release-e2e";
  nodes = {
    stock = host;
    nixld = {
      imports = [ host ];
      programs.nix-ld.enable = true;
    };
  };
  testScript = ''
    start_all()
    for machine in (stock, nixld):
        machine.wait_for_unit("multi-user.target")
        status, _ = machine.execute(
            f"python3 ${harness}/release_e2e.py run --label 'NixOS 26.05 ({machine.name})'"
            " --tag ${tag} --archive-label ${archiveLabel} --source-sha ${sourceSha}"
            " --assets ${linuxAssets} --gates ${gateTree}"
            " --evidence /tmp/evidence > /tmp/console.log 2>&1",
            timeout=1800,
        )
        print(machine.succeed("cat /tmp/console.log"), flush=True)
        # Only the reports and logs leave the VM. The installed toolchains
        # under /tmp/evidence/work run to hundreds of megabytes, and writing
        # them through the 9p shared directory has filled the host disk and
        # crashed the guest kernel (netfs).
        machine.succeed(
            f"mkdir -p /tmp/evidence && echo {status} > /tmp/evidence/exit-status"
            " && cp /tmp/console.log /tmp/evidence/console.log"
            " && tar -C /tmp --ignore-failed-read -czf /tmp/evidence.tar.gz"
            " evidence/report.json evidence/logs evidence/canary"
            " evidence/exit-status evidence/console.log"
        )
        machine.copy_from_machine("/tmp/evidence.tar.gz", machine.name)
  '';
}
