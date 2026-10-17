#!/usr/bin/env bash
# Shared installer core for portable setup.
#
# Usage:
#   core.sh add-profile <profile_root> <devenv_output> <expected_version>
#   core.sh prewarm <environment_root> <devenv_binary>
#
# `add-profile` substitutes the pinned Devenv output into an isolated Nix
# profile and requires the exact reported version. `prewarm` evaluates the
# portable environment through the daemon with a bounded retry and executes
# only `true`. Every other argument shape fails closed.
set -euo pipefail

fail_output() {
  printf 'devenv-output:%s\n' "$1" >&2
  exit 64
}

core_add_profile() {
  local profile_root=$1 devenv_output=$2 expected_version=$3 reported_version
  if ! nix profile add --profile "$profile_root" "$devenv_output"; then
    fail_output "unavailable"
  fi
  reported_version="$("$profile_root/bin/devenv" --version)"
  if [[ $reported_version != "$expected_version" ]]; then
    fail_output "version-mismatch"
  fi
}

core_prewarm() {
  local environment_root=$1 devenv_binary=$2 attempt
  local prewarm_attempts=3
  for ((attempt = 1; attempt <= prewarm_attempts; attempt++)); do
    if (
      cd "$environment_root"
      "$devenv_binary" --no-tui shell -- true
    ); then
      return 0
    fi
    if ((attempt == prewarm_attempts)); then
      printf 'Portable Devenv evaluation failed after %s attempts.\n' "$prewarm_attempts" >&2
      exit 1
    fi
    printf 'Portable Devenv evaluation attempt %s failed. Retry the evaluation.\n' "$attempt" >&2
  done
}

case ${1:-} in
  add-profile)
    [[ $# -eq 4 ]] || fail_output "core-usage"
    core_add_profile "$2" "$3" "$4"
    ;;
  prewarm)
    [[ $# -eq 3 ]] || fail_output "core-usage"
    core_prewarm "$2" "$3"
    ;;
  *)
    fail_output "core-usage"
    ;;
esac
