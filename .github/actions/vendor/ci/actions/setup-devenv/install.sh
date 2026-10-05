#!/usr/bin/env bash
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP is not set}"
: "${GITHUB_ACTION_PATH:?GITHUB_ACTION_PATH is not set}"
: "${GITHUB_PATH:?GITHUB_PATH is not set}"
: "${GITHUB_ENV:?GITHUB_ENV is not set}"
: "${CHELIS_SETUP_REPOSITORY:?CHELIS_SETUP_REPOSITORY is not set}"
: "${CHELIS_SETUP_REVISION:?CHELIS_SETUP_REVISION is not set}"

fail_output() {
  printf 'devenv-output:%s\n' "$1" >&2
  exit 64
}

setup_repository=$CHELIS_SETUP_REPOSITORY
setup_revision=$CHELIS_SETUP_REVISION

# Devenv v2.2.3
# Devenv main of 2026-09-04. The release line 2.2.1 embeds a Nix build with a
# use-after-free on the evaluator read-only flag (cachix/devenv#3064). This
# revision embeds the fixed build. Return to the first release with the fix.
devenv_revision="360b5eb1397291383d10845a63a0247981bd5598"
devenv_version="2.2.3"
# The reviewed output of packages.<system>.devenv at that revision. The
# manifest devenv-toolchain.toml holds the same paths, and the contract test
# requires equality. Setup substitutes the path and evaluates no flake.
devenv_output_x86_64_linux="/nix/store/xwhcf2bx93i76i59bgzvqr19kv6w6yab-devenv-wrapped-2.2.3"
devenv_output_aarch64_darwin="/nix/store/x06aza11qrhx7gvrqjhxprqbx7pi72kd-devenv-wrapped-2.2.3"
profile_root="$RUNNER_TEMP/chelis-ci-devenv-profile"
environment_root="$RUNNER_TEMP/chelis-ci-devenv-environment"
wrapper_root="$RUNNER_TEMP/chelis-ci-devenv-bin"
# The shared installer core owns profile installation and the bounded
# environment prewarm. The central cache seed commit realizes its planned
# roots through the same core.
installer_core="$GITHUB_ACTION_PATH/core.sh"

case "$(uname -s):$(uname -m)" in
  Linux:x86_64)
    system=x86_64-linux
    devenv_output=$devenv_output_x86_64_linux
    ;;
  Darwin:arm64 | Darwin:aarch64)
    system=aarch64-darwin
    devenv_output=$devenv_output_aarch64_darwin
    ;;
  *) fail_output "unsupported-system" ;;
esac

rm -rf "$profile_root" "$environment_root" "$wrapper_root"
mkdir -p "$environment_root/devenv" "$wrapper_root"
cp -R "$GITHUB_ACTION_PATH/environment/." "$environment_root/"
cp -R "$GITHUB_ACTION_PATH/../../devenv/shared" "$environment_root/devenv/shared"
cp "$GITHUB_ACTION_PATH/../../devenv-toolchain.toml" "$environment_root/devenv-toolchain.toml"
cp "$GITHUB_ACTION_PATH/devenv-ci" "$wrapper_root/devenv-ci"
cp "$GITHUB_ACTION_PATH/devenv-retry" "$wrapper_root/devenv-retry"
chmod 0755 "$wrapper_root/devenv-ci" "$wrapper_root/devenv-retry"

expected_version="devenv ${devenv_version}+${devenv_revision:0:7} (${system})"
bash "$installer_core" add-profile "$profile_root" "$devenv_output" "$expected_version"
bash "$installer_core" prewarm "$environment_root" "$profile_root/bin/devenv"

printf '%s\n' "$profile_root/bin" "$wrapper_root" >> "$GITHUB_PATH"
{
  printf 'CHELIS_DEVENV_BIN=%s\n' "$profile_root/bin/devenv"
  printf 'CHELIS_DEVENV_ROOT=%s\n' "$environment_root"
  printf 'CHELIS_SETUP_REPOSITORY=%s\n' "$setup_repository"
  printf 'CHELIS_SETUP_REVISION=%s\n' "$setup_revision"
} >> "$GITHUB_ENV"
