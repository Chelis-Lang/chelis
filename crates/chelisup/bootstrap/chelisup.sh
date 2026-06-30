#!/bin/sh
# chelisup bootstrap installer.
#
# The ONE shell script permitted in this repository (see the Scripting
# Language Policy carve-out in AGENTS.md). It runs on a bare machine
# BEFORE any chelis, cargo, or even Python exists, so it cannot be
# written in any of those. Keep it minimal POSIX sh and shellcheck-clean.
#
# It fetches the prebuilt `chelisup` binary for the host platform from
# the latest Chelis-Lang/chelis GitHub release and installs it at
# $CHELIS_HOME/bin/chelisup (default ~/.chelis/bin/chelisup).
#
# Release asset-name convention (published by the chelis release tooling,
# matching crates/chelisup slugs): a BARE executable named
#   chelisup-<slug>   with slug in { darwin-arm64, darwin-x86_64, linux-x86_64 }
# attached to each GitHub release (no version in the name, no tarball).
#
# Private-repo era (chelis#164): while Chelis-Lang/chelis is private the
# public release URL does not serve asset bytes, so an authenticated `gh`
# is REQUIRED. The curl path below works once releases are public.
set -eu

repo="Chelis-Lang/chelis"

# 1. Detect the host platform slug (matches crates/chelisup install.rs).
os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
  Darwin/arm64) slug="darwin-arm64" ;;
  Darwin/x86_64) slug="darwin-x86_64" ;;
  Linux/x86_64) slug="linux-x86_64" ;;
  *)
    printf 'chelisup bootstrap: unsupported host %s/%s; chelis ships darwin-arm64, darwin-x86_64, linux-x86_64\n' "$os" "$arch" >&2
    exit 1
    ;;
esac
asset="chelisup-$slug"

# 2. Resolve the install location ($CHELIS_HOME overrides ~/.chelis).
chelis_home="${CHELIS_HOME:-$HOME/.chelis}"
bin_dir="$chelis_home/bin"
target="$bin_dir/chelisup"
tmp="$target.download"

mkdir -p "$bin_dir"

# 3. Download the prebuilt for this host into a temp file.
if command -v gh >/dev/null 2>&1; then
  printf 'chelisup bootstrap: downloading %s via gh (latest release of %s)\n' "$asset" "$repo"
  gh release download --repo "$repo" --pattern "$asset" --output "$tmp" --clobber
else
  url="https://github.com/$repo/releases/latest/download/$asset"
  printf 'chelisup bootstrap: while %s is private this needs an authenticated gh (https://cli.github.com); trying the public URL anyway\n' "$repo" >&2
  printf 'chelisup bootstrap: downloading %s\n' "$url"
  curl -fsSL -o "$tmp" "$url"
fi

# 4. Install atomically and make it executable.
chmod +x "$tmp"
mv "$tmp" "$target"
printf 'chelisup bootstrap: installed %s\n' "$target"

# 5. PATH guidance (never auto-edit shell rc files).
case ":$PATH:" in
  *":$bin_dir:"*)
    printf 'chelisup bootstrap: done. Next: chelisup install <ver>\n'
    ;;
  *)
    printf 'chelisup bootstrap: add %s to your PATH, then run: chelisup install <ver>\n' "$bin_dir"
    # The $PATH below is literal text for the user to copy, not an expansion.
    # shellcheck disable=SC2016
    printf 'chelisup bootstrap:   export PATH="%s:$PATH"   (add to ~/.profile, ~/.bashrc, or ~/.zshrc)\n' "$bin_dir"
    ;;
esac
