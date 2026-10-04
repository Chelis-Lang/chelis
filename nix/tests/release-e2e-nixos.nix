# Boots two NixOS VMs and runs scripts/release_e2e.py against the published
# Linux release assets in each. `stock` is the default configuration;
# `nixld` also enables programs.nix-ld. Both machines carry Python, GCC, and
# binutils, which the harness and `chelis build` need. The test records each
# outcome without failing on it; the workflow checks the reports.
#
#   nix build --file nix/tests/release-e2e-nixos.nix \
#     --arg assets /abs/assets --arg gates /abs/gates \
#     --arg harness /abs/scripts/release_e2e.py \
#     --argstr tag vX.Y.Z --argstr archiveLabel vX.Y.Z|dev-<sha8> \
#     --argstr sourceSha <40-hex commit>
#
# `--arg online true` adds the bootstrap and the GitHub API install, through
# QEMU's user-mode network. The Nix sandbox has no network, so build the
# `driver` attribute instead and run it on the host with GITHUB_TOKEN set:
#
#   nix build --file nix/tests/release-e2e-nixos.nix driver --arg online true \
#     --argstr repo OWNER/NAME ...
#   GITHUB_TOKEN=... ./result/bin/nixos-test-driver -o "$PWD/nixos-result"
{
  assets,
  gates,
  harness,
  tag,
  archiveLabel ? tag,
  sourceSha,
  online ? false,
  repo ? "Chelis-Lang/chelis",
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
      ]
      ++ lib.optional online pkgs.gh;
    }
    // lib.optionalAttrs online {
      # The test framework leaves the user-mode NIC unconfigured; DHCP gives it
      # QEMU's address, gateway, and DNS forwarder.
      networking.interfaces.eth0.useDHCP = true;
    };
  onlineFlags = lib.optionalString online " --online --repo ${repo}";
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
    import os
    import tempfile

    token_path: str | None = None
    if ${if online then "True" else "False"}:
        with tempfile.NamedTemporaryFile("w", delete=False) as token_file:
            token_file.write(os.environ["GITHUB_TOKEN"])
        token_path = token_file.name

    start_all()
    for machine in (stock, nixld):
        machine.wait_for_unit("multi-user.target")
        prefix = ""
        if token_path is not None:
            machine.wait_until_succeeds("ip route show default | grep -q .", timeout=180)
            machine.copy_from_host(token_path, "/tmp/github-token")
            prefix = "GITHUB_TOKEN=$(cat /tmp/github-token) GH_TOKEN=$(cat /tmp/github-token) "
        status, _ = machine.execute(
            prefix
            + f"python3 ${harnessScript} run --label 'NixOS 26.05 ({machine.name})'"
            " --tag ${tag} --archive-label ${archiveLabel} --source-sha ${sourceSha}"
            " --assets ${linuxAssets} --gates ${gateTree}${onlineFlags}"
            " --evidence /tmp/evidence > /tmp/console.log 2>&1",
            timeout=1800,
        )
        machine.succeed("rm -f /tmp/github-token")
        print(machine.succeed("cat /tmp/console.log"))
        machine.succeed(
            f"mkdir -p /tmp/evidence && echo {status} > /tmp/evidence/exit-status"
            " && cp /tmp/console.log /tmp/evidence/console.log"
            " && tar -C /tmp -czf /tmp/evidence.tar.gz evidence"
        )
        machine.copy_from_vm("/tmp/evidence.tar.gz", machine.name)
  '';
}
