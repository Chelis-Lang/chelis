# Boots two NixOS VMs and runs scripts/release_e2e.py against the published
# Linux release assets in each. `stock` is the default configuration;
# `nixld` also enables programs.nix-ld. Both machines carry Python, GCC, and
# binutils, which the harness and `chelis build` need. The test records each
# outcome without failing on it; the workflow checks the reports.
#
#   nix build --file nix/tests/release-e2e-nixos.nix \
#     --arg assets /abs/assets --arg gates /abs/gates \
#     --arg harness /abs/scripts/release_e2e.py \
#     --argstr tag vX.Y.Z --argstr sourceSha <40-hex commit>
{
  assets,
  gates,
  harness,
  tag,
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
  harnessScript = builtins.path {
    name = "release_e2e.py";
    path = harness;
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
            f"python3 ${harnessScript} run --label 'NixOS 26.05 ({machine.name})'"
            " --tag ${tag} --source-sha ${sourceSha}"
            " --assets ${linuxAssets} --gates ${gateTree}"
            " --evidence /tmp/evidence > /tmp/console.log 2>&1",
            timeout=1800,
        )
        print(machine.succeed("cat /tmp/console.log"))
        machine.succeed(
            f"mkdir -p /tmp/evidence && echo {status} > /tmp/evidence/exit-status"
            " && cp /tmp/console.log /tmp/evidence/console.log"
            " && tar -C /tmp -czf /tmp/evidence.tar.gz evidence"
        )
        machine.copy_from_vm("/tmp/evidence.tar.gz", machine.name)
  '';
}
