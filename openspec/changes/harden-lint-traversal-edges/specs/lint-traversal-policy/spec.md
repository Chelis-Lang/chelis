# Lint Traversal Policy Delta: harden-lint-traversal-edges

## MODIFIED Requirements

### Requirement: Explicit targets override only traversal exclusions at depth zero
A file or directory explicitly supplied as a lint root SHALL override a policy exclusion matching that root. The root MUST still be a regular file or directory, or a link resolving to the same kind inside the repository policy boundary. Policy exclusions SHALL continue to apply to matching descendants beneath an explicit directory root. An explicit root that exists but fails depth-zero admission — a socket, FIFO, device, or other non-regular entry, or a link resolving to a different entry kind or outside the repository policy boundary — MUST fail the lint invocation with the root path and rejection reason rather than producing a successful empty entry set. A nonexistent explicit root MUST also fail loudly; the two outcomes SHALL be consistent, never silent. Discovered (non-explicit) inadmissible entries remain silently omitted.

#### Scenario: Explicit excluded file is linted
- **WHEN** the user names an otherwise excluded regular source file directly
- **THEN** applicable rules check that file

#### Scenario: Explicit excluded directory is entered
- **WHEN** the user names an otherwise excluded directory directly
- **THEN** the directory is entered while separately excluded nested descendants remain pruned, including for resolved symlink targets

#### Scenario: Explicit special file fails loudly
- **WHEN** an explicitly named source-shaped path is a socket, FIFO, device, or another non-regular entry
- **THEN** the lint invocation fails with the root path and rejection reason instead of exiting successfully with no entries

#### Scenario: Explicit escaping symlink directory fails loudly
- **WHEN** an explicitly named directory link resolves outside the repository policy root
- **THEN** the walker does not enter or import the target tree and the lint invocation fails with the root path and rejection reason

#### Scenario: Explicit rejection is loud through the CLI
- **WHEN** `chelis lint --check` is invoked on an existing explicit root that fails depth-zero admission
- **THEN** the command exits nonzero and reports the root path and rejection reason, matching the nonexistent-root failure mode

#### Scenario: Discovered inadmissible entries stay silent
- **WHEN** a socket, FIFO, escaping symlink, or other inadmissible entry is discovered below an admitted lint root rather than named explicitly
- **THEN** the entry is omitted without failing the invocation

### Requirement: Traversal exclusions come from structured policy
`chelis-lint` SHALL obtain traversal exclusions from versioned TOML policy artifacts rather than path-name literals in `walker.rs`. A built-in baseline policy SHALL provide cross-repository infrastructure exclusions, and the nearest ancestor `chelis-lint.toml` SHALL add repository-specific exclusions. Nearest-ancestor discovery MUST be performed on the lint target resolved against the current working directory, so a relative target spelled from any cwd inside a policy root discovers the same repository policy as the equivalent absolute target. Discovery MUST NOT depend on how the target path was spelled.

#### Scenario: Repository policy extends built-in defaults
- **WHEN** a lint target has an ancestor `chelis-lint.toml`
- **THEN** the canonical walker applies both the built-in baseline exclusions and that nearest repository policy

#### Scenario: Loose target uses deterministic defaults
- **WHEN** a lint target has no `chelis-lint.toml` ancestor after cwd resolution
- **THEN** the canonical walker applies only the built-in baseline policy without consulting Git or machine-local files

#### Scenario: Relative target from a subdirectory cwd discovers repository policy
- **WHEN** lint is invoked with a relative target such as `.` from a working directory below a repository policy root
- **THEN** the repository `chelis-lint.toml` is discovered and its exclusions apply exactly as they would for the equivalent absolute target

#### Scenario: Relative and absolute spellings agree
- **WHEN** the same filesystem location is linted once via a relative path and once via an absolute path
- **THEN** both invocations resolve the same policy and admit the same entry set
