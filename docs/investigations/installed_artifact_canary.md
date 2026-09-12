# Installed artifact callable gate

The release workflow's `installed-artifact-canary` matrix downloads the staged
Linux x86-64 and macOS arm64 artifacts from the same workflow run, before
publication. It uses the staged installer, not a Cargo binary or an existing
user installation. `publish-release` requires this matrix in addition to all
three existing build jobs; workflow dispatch still does not publish anything.

The script verifies archive and installer checksum sidecars (including their
filenames), rejects ambiguous/link/escaping archive members, inventories every
member payload, and requires the public runtime/header closure. It installs
through `chelisup` in a fresh isolated home, verifies installed bytes and shim
identity, and checks actual version and pin routing. No real home/default or
installer behavior changes. Checksums provide content identity, not authenticity
or a proof that source produced a binary; workflow/source provenance remains
trusted and is recorded separately.

Dispatch archives are named `chelis-dev-SHA8-*`, but the current installer only
accepts concrete versions and a `chelis-v*` archive root. For that lane only,
the canary records an explicit root-name conversion into a local archive and
checks every resulting payload against the original downloaded archive. Both
container hashes are retained. This is not a published release or a byte-identical
container. Published `vX.Y.Z` archives are copied unchanged. No alias is installed
outside this run's evidence directory.

The executed program and bitwise driver reuse the maintained manifested callable
smoke fixture. The installed CLI generates C; runtime/header outputs must match
the installed archive. C is copied to a header-free consumer directory, compiled
with the **shipped** headers, and linked with the **shipped** static runtime.
Three native invocations check every result bit, dtype, ordered shape and both
unchanged input buffers. An unavailable automatic root must reject before leaving
an artifact. Two separately compiled caller-side fault injections corrupt the
last result coordinate and an input buffer; both must fail the unchanged driver.
These injected faults are controls, not valid source-language execution.

Every process has retained argv/cwd, stdout/stderr, exit and timeout evidence.
Signals, timeouts, unexpected positive exits and missing required runs cannot
be counted as successful legs. `report.json` retains failures and unrun legs;
the evidence directory must not already exist. Inputs and installed payloads
are checked again after execution. The workflow uploads this directory even
on failure.

Local replay with **downloaded** artifacts and their checksum sidecars:

```
.venv/bin/python scripts/installed_artifact_canary.py --artifacts TOOLCHAIN_DIR --installer-assets INSTALLER_DIR --source-sha FULL_COMMIT --version X.Y.Z --build-label vX.Y.Z --evidence NEW_DIRECTORY
```

For a staged dispatch use its `dev-SHA8` build label instead. The workflow fills
in its exact source SHA, build label, version and run/attempt identity. A local
assembly can test this harness but must not be reported as a downloaded release.

PR Python ownership includes `scripts.test_installed_artifact_canary` via ordinary
script discovery (checksum/archive/mapping and real exit/signal/timeout controls).
The **installed native gate** runs on release/dispatch, not ordinary PR CI. Its
success is a bounded packaging and callable-ABI check: no reverse-mode AD, Hull
agreement, fixed-path Random, School adoption, glibc-2.31 runtime acceptance or
complete LaCaDiLE replacement is implied. Those acceptance legs remain separate.
