## ADDED Requirements

### Requirement: Traversal exclusions come from structured policy
`chelis-lint` SHALL obtain traversal exclusions from versioned TOML policy artifacts rather than path-name literals in `walker.rs`. A shipped baseline policy SHALL provide cross-repository infrastructure exclusions, and the nearest ancestor `chelis-lint.toml` SHALL add repository-specific exclusions.

#### Scenario: Repository policy extends shipped defaults
- **WHEN** a lint target has an ancestor `chelis-lint.toml`
- **THEN** the canonical walker applies both the shipped baseline exclusions and that nearest repository policy

#### Scenario: Loose target uses deterministic defaults
- **WHEN** a lint target has no `chelis-lint.toml` ancestor
- **THEN** the canonical walker applies only the shipped baseline policy without consulting Git or machine-local files

### Requirement: Policy schema is fail-closed and explainable
Each policy SHALL declare `version = 1`. Every exclusion MUST contain one gitignore-style `pattern`, one closed `class` value (`infrastructure`, `build`, `dependency`, `generated`, or `immutable`), and one `cross_ref`. Repository policy SHALL name the active spec document used to validate its cross-references. Unsupported versions, unknown fields or classes, invalid patterns, missing spec files, and unresolved cross-references MUST fail the lint invocation with the policy path and reason.

#### Scenario: Valid exclusion loads
- **WHEN** a policy entry has a valid pattern, known class, and cross-reference resolving in its declared spec
- **THEN** policy loading succeeds and matching can report the entry's pattern, class, and cross-reference

#### Scenario: Unknown field fails
- **WHEN** a policy contains a misspelled or unsupported field
- **THEN** lint fails instead of silently ignoring the field

#### Scenario: Unresolved cross-reference fails
- **WHEN** an exclusion's `cross_ref` does not resolve in the policy's declared spec document
- **THEN** lint fails before walking the target

### Requirement: Traversal is deterministic across machines
The `ignore` traversal engine MUST disable hidden-file filtering, `.gitignore`, `.ignore`, parent ignore files, global Git ignores, and `.git/info/exclude`. Only shipped and repository `chelis-lint` policies may exclude descendants.

#### Scenario: Global Git ignore cannot alter lint
- **WHEN** a user global ignore excludes a classifiable source path that no lint policy excludes
- **THEN** the canonical walker still admits that path

#### Scenario: Hidden source remains visible
- **WHEN** `.github/workflows/ci.yml` or another hidden classifiable path is not excluded by lint policy
- **THEN** the canonical walker admits it

#### Scenario: Repository gitignore is not lint policy
- **WHEN** `.gitignore` excludes a classifiable path that no lint policy excludes
- **THEN** the canonical walker still admits that path

### Requirement: Policy matching is workspace-anchored and prunes descendants
Repository patterns SHALL be interpreted relative to the directory containing `chelis-lint.toml`. A matching nested directory SHALL be omitted and not descended into. The same admitted entry set SHALL feed rule preparation and per-entry checks.

#### Scenario: Nested excluded directory is pruned
- **WHEN** a repository exclusion matches a nested generated directory
- **THEN** neither that directory's entries nor their source contents reach any rule or prepared catalog

#### Scenario: Subdirectory target uses workspace policy
- **WHEN** lint is invoked on a subdirectory beneath a policy root
- **THEN** patterns remain anchored to the policy root rather than the target subdirectory

#### Scenario: Sibling remains admitted
- **WHEN** one directory matches an exclusion and a sibling does not
- **THEN** the walker prunes only the matching directory

### Requirement: Explicit targets override only traversal exclusions at depth zero
A file or directory explicitly supplied as a lint root SHALL override a policy exclusion matching that root. The root MUST still be a regular file or directory, or a link resolving to the same kind inside the repository policy boundary. Policy exclusions SHALL continue to apply to matching descendants beneath an explicit directory root.

#### Scenario: Explicit excluded file is linted
- **WHEN** the user names an otherwise excluded regular source file directly
- **THEN** applicable rules check that file

#### Scenario: Explicit excluded directory is entered
- **WHEN** the user names an otherwise excluded directory directly
- **THEN** the directory is entered while separately excluded nested descendants remain pruned

#### Scenario: Explicit special file is rejected
- **WHEN** an explicitly named source-shaped path is a socket, FIFO, device, or another non-regular entry
- **THEN** the walker omits it before any rule can read it

#### Scenario: Explicit symlink directory cannot escape policy
- **WHEN** an explicitly named directory link resolves outside the repository policy root
- **THEN** the walker does not enter or import the target tree

### Requirement: Traversal exclusions remain distinct from diagnostic exceptions
A traversal exclusion SHALL prevent all rule dispatch for its matched descendants. A rule-specific `Exception`, inline `allow`, or inline `keep` SHALL NOT alter traversal and SHALL retain its existing diagnostic or autofix semantics. Ancillary files consulted by a rule MUST come from the canonical admitted entry set or pass parent-aware traversal-policy admission, so excluded content cannot change an admitted entry's verdict indirectly. When repository policy exists, paths above its root remain machine-local even when they are symlinks resolving inward.

#### Scenario: Rule exception does not prune corpus discovery
- **WHEN** a source path has a rule-specific exception but no traversal exclusion
- **THEN** the path remains in the canonical entry set and can contribute to other rules' prepared state

#### Scenario: Traversal exclusion suppresses every rule
- **WHEN** a path is excluded by traversal policy
- **THEN** no rule checks it and no rule-specific exception is required

#### Scenario: Excluded ancillary manifest cannot affect an admitted document
- **WHEN** an excluded Cargo manifest names a package whose kebab-case name matches an admitted documentation filename
- **THEN** `doc-filename-convention` ignores that manifest and reports the same verdict as if the excluded tree were absent

#### Scenario: External ancillary link path remains ungoverned
- **WHEN** a Cargo manifest link path is above the repository policy root but resolves to an admitted file inside that root
- **THEN** `doc-filename-convention` rejects the link as machine-local and does not grant the package-name exception

#### Scenario: Admitted ancillary manifest retains its documented effect
- **WHEN** an admitted Cargo manifest names a package whose kebab-case name matches an admitted documentation filename
- **THEN** `doc-filename-convention` retains the package-name exception from §8.3
