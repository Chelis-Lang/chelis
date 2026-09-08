# Compiler Pipeline Architecture Delta: harden-pipeline-core-cache-and-lowering

## ADDED Requirements

### Requirement: Contextual host lowering matches isolated host lowering
Contextual lowering SHALL apply the same nonfatal-rejection host policy as isolated lowering. Under `AllowHostBackend`, a nonfatal lower rejection SHALL produce an empty host result on both the isolated and contextual paths.

The two paths SHALL agree for equivalent inputs: a nonfatal lower rejection under a selected host backend SHALL yield an empty `NamedRoots` product rather than a lowering error on either path. `AllowHostOnly` SHALL retain its stricter guard on both paths, accepting a nonfatal rejection only when no tensor root name is declared.

#### Scenario: Contextual host backend accepts a nonfatal rejection
- **WHEN** contextual lowering under `AllowHostBackend` receives a nonfatal lower rejection
- **THEN** it produces an empty `NamedRoots` host result instead of returning a lowering error

#### Scenario: Isolated and contextual host lowering agree
- **WHEN** the same nonfatal lower rejection under `AllowHostBackend` is lowered on the isolated path and the contextual path
- **THEN** both paths produce an empty host result with an empty `NamedRoots`

#### Scenario: Contextual host-only keeps its tensor-name guard
- **WHEN** contextual lowering under `AllowHostOnly` receives a nonfatal lower rejection while a tensor root name is declared
- **THEN** it returns the lowering error rather than an empty host result, matching the isolated host-only guard
