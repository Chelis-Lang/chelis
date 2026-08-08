## MODIFIED Requirements

### Requirement: The release workflow publishes the Devenv output
The release workflow MUST define one `build-chelisup-release` matrix job for `linux-x86_64` and `darwin-arm64`.

The job MUST invoke the pinned shared Devenv setup before it builds the output. It MUST authenticate Nix for the private ci input before the first project Devenv command.

The job MUST run `devenv build --no-tui outputs.release-chelisup`. It MUST upload only the exact executable and sidecar for its matrix platform.

No release workflow job MUST run `cargo build` for package `chelisup`. Toolchain publication for both platforms MUST follow the `portable-chelis-release` capability.

The publish job MUST depend on the release-output matrix. It MUST retain the current manual tag dispatch and publication conditions.

#### Scenario: A manual release builds portable installers
- **WHEN** the release workflow runs at a supported tag
- **THEN** the matrix builds both Devenv release outputs
- **AND** the publish job receives both executables and both sidecars

#### Scenario: A full toolchain job builds chelisup directly
- **WHEN** the release workflow contains `cargo build` for package `chelisup`
- **THEN** the release workflow contract test fails

#### Scenario: The matrix omits private input authentication
- **WHEN** the matrix reaches a project Devenv command without private ci authentication
- **THEN** the release workflow contract test fails

#### Scenario: Release publication policy changes
- **WHEN** the workflow publishes outside a manual dispatch at a `v*` tag
- **THEN** the current release workflow policy test fails
