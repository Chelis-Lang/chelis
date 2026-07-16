## ADDED Requirements

### Requirement: Analysis consumes immutable workspace snapshots
Repository-wide built-in lint analysis SHALL accept an immutable `WorkspaceSnapshot` as the complete source-content and workspace-metadata input. A snapshot SHALL contain a normalized logical root, stable path-sorted entries with bytes, decoded source, surface inputs, and only metadata declared required by active pure rules, plus identity-bearing rejected-entry tombstones for collection failures. Pure rule contexts SHALL NOT expose a live root that can be reopened.

#### Scenario: Equal snapshots produce equal analysis
- **WHEN** an analysis runs twice with structurally equal snapshots, rules, and exceptions
- **THEN** it returns equal ordered diagnostics and violations

#### Scenario: Host changes after collection are invisible
- **WHEN** a host file changes after snapshot collection but before analysis
- **THEN** analysis observes the collected snapshot content rather than reopening the file

### Requirement: Collection policy is explicit
Snapshot builders SHALL receive explicit ignore, symlink, surface-classification, UTF-8 decoding, size, stable-open, and path-normalization policy. The FCIS contract manifest SHALL pin exact v1 per-entry, entry-count, locator, and total-snapshot bounds before collector implementation. V1 logical paths SHALL be relative UTF-8 with `/` separators, preserved case, no `.`/`..` components, and bytewise UTF-8 ordering. A non-UTF-8 host path SHALL fail without lossy conversion and SHALL use a diagnostic-only `HostEntryLocator` containing the normalized UTF-8 parent prefix, platform tag, and length-prefixed raw bytes of the first unrepresentable component. Canonical order SHALL place representable logical locators first by UTF-8 path bytes and entry-kind rank, then diagnostic-only locators by parent prefix, platform tag, raw-component length, and raw bytes, with canonical diagnostic fields breaking ties. Directory symlinks SHALL NOT be followed; selected file symlinks SHALL use containment-safe handle-relative access or fail closed on unsupported platforms. The resulting snapshot SHALL record tombstones and collection diagnostics for unreadable, invalid, oversized, non-UTF-8, unstable, escaped, unsupported-symlink, or otherwise rejected required entries. Canonical diagnostic equality SHALL exclude locale-dependent OS text and absolute host paths. These diagnostics SHALL block `lint --check`; ignored and unselected entries remain omitted without diagnostics.

#### Scenario: Unreadable file is reported
- **WHEN** a selected workspace file cannot be read
- **THEN** snapshot collection returns a structured diagnostic naming the normalized logical path when representable, otherwise its diagnostic-only host-entry locator, plus a normalized failure class

#### Scenario: Escaping symlink is rejected by policy
- **WHEN** a symlink resolves outside the configured workspace root and policy forbids escape
- **THEN** collection records a structured rejection and does not include external file bytes

#### Scenario: Ignored file is omitted deterministically
- **WHEN** an entry matches the explicit ignore policy
- **THEN** the entry is omitted consistently regardless of filesystem enumeration order

#### Scenario: Non-UTF-8 path is not normalized lossily
- **WHEN** a selected host entry has a path component that is not valid UTF-8
- **THEN** collection records a blocking `non_utf8_path` tombstone with a stable diagnostic-only raw-component locator and does not invent a logical path by lossy conversion

#### Scenario: Entry replacement during collection fails closed
- **WHEN** a selected entry or symlink target changes identity while the collector is opening or reading it
- **THEN** collection records `entry_changed_during_collection` and includes no bytes from the unstable read

### Requirement: Rule requirements and public migration are explicit
Each pure lint rule SHALL declare the file classes and metadata it requires so snapshot collection can remain complete without collecting unrelated host state. One typed rule registry SHALL own rule order, severity, blocking status, surfaces, data requirements, shared indexes, and legacy status. A rule access to absent or undeclared required data SHALL return a named `MissingDeclaredInput` contract failure rather than `None`, empty text, or empty index fallback. Chelis SHALL introduce a versioned `SnapshotRule` interface. The existing root-exposing `Context`/`Rule` API MAY remain for one documented minor-version window only behind a named legacy path adapter outside the designated pure core and MUST NOT be counted as pure-core evidence.

#### Scenario: Collector satisfies rule descriptor
- **WHEN** active rules declare their source and Cargo-manifest requirements
- **THEN** collection includes those inputs once before analysis begins

#### Scenario: Existing external rule receives a bounded migration path
- **WHEN** an external caller implements the prior public rule interface
- **THEN** the named legacy adapter preserves it for the documented window while clearly excluding it from the pure-core guarantee

#### Scenario: Missing descriptor input fails loudly
- **WHEN** a pure rule requests data absent from the supplied snapshot
- **THEN** analysis returns a named missing-input diagnostic rather than silently treating the workspace as empty

### Requirement: Lint rules perform no host I/O
Built-in `SnapshotRule` implementations and shared lint indexes SHALL live in the dependency-minimal `chelis-lint-core` crate and consume only the current snapshot entry, immutable shared indexes, rule configuration, and explicit exceptions. They MUST NOT traverse directories, read files, inspect environment, execute processes, access a network, print, obtain entropy, spawn semantic work, use unsafe FFI, mutate global semantic state, or accept callbacks or traits that expose those capabilities. Arbitrary legacy rules are outside this guarantee. The architecture gate SHALL state the exact direct, alias, re-export, callback, trait, macro, and `cfg(test)` fixture forms it proves and MUST NOT claim complete detection beyond its documented mechanism.

#### Scenario: Cross-file rule uses a supplied index
- **WHEN** an opaque-type or naming rule needs declarations from other files
- **THEN** it queries a shared index built from the snapshot instead of walking the workspace

#### Scenario: Architecture gate rejects rule I/O
- **WHEN** a negative fixture adds a direct, aliased, re-exported, qualified, callback-hidden, function-pointer, macro-hidden, or trait-hidden filesystem, directory-walking, environment, process, network, terminal, entropy, scheduling, unsafe-FFI, or mutable-global capability to a lint rule
- **THEN** the lint architecture gate fails and identifies the forbidden dependency

### Requirement: Shared indexes are snapshot-derived
Cross-file catalogs and indexes SHALL be computed from the supplied snapshot before rule evaluation and SHALL be reused across applicable entries. Their contents and ordering SHALL depend only on snapshot data and analysis configuration.

#### Scenario: Catalog is complete before rule execution
- **WHEN** a cross-file rule runs for the first path-sorted entry
- **THEN** it can query declarations from every included snapshot entry

#### Scenario: Duplicate declarations are stable
- **WHEN** multiple entries contribute equal or conflicting catalog keys
- **THEN** the index reports them in normalized path order with deterministic diagnostics

### Requirement: Path adapters delegate to snapshot analysis
Normal path-based lint entry points MAY remain as compatibility adapters, but SHALL build a snapshot once and delegate to snapshot-based built-in analysis. They MUST NOT run a second hidden traversal from individual built-in rules. Only the separately named legacy-rule adapter may expose the old root-based contract during its compatibility window.

#### Scenario: Path and snapshot lint agree
- **WHEN** a workspace remains unchanged between collection and analysis
- **THEN** the path adapter and direct snapshot API return equal ordered violations and diagnostics

#### Scenario: One command uses one content snapshot
- **WHEN** a path-based lint command analyzes multiple rules
- **THEN** source collection occurs once for that analysis invocation

### Requirement: Snapshot analysis preserves public behavior
For the established executable corpus, snapshot-based lint analysis SHALL preserve rule codes, severities, normalized paths, ordering, and exception behavior. The newly blocking collection diagnostics specified above are the only approved behavior correction in this change.

#### Scenario: Existing positive corpus remains clean
- **WHEN** the executable example corpus is analyzed through snapshots
- **THEN** it retains its established clean result

#### Scenario: Existing negative corpus retains diagnostics
- **WHEN** the negative lint corpus is analyzed through snapshots without a collection failure
- **THEN** it retains the established structured diagnostic set and order

### Requirement: Snapshot equality excludes host and rendering accidents
`WorkspaceSnapshot` structural equality SHALL cover successful entries, canonical collection diagnostics, and rejected-entry tombstones while excluding absolute host root spelling, locale-dependent OS error text, collection time, terminal mode, and renderer configuration. This change SHALL NOT introduce a persistent lint-result cache, `LintQueryKey`, `LintOutcomeDigest`, or stable hash encoding.

#### Scenario: Equivalent roots have equal analysis inputs
- **WHEN** equal logical snapshots are collected under different absolute host root paths and differ only in noncanonical OS report text
- **THEN** the snapshots compare equal and produce equal outcomes for equal rules and exceptions

#### Scenario: Renderer changes no lint decision
- **WHEN** only terminal or machine rendering configuration changes
- **THEN** snapshot equality, diagnostics, and violations remain unchanged

#### Scenario: Collection failures change structural equality
- **WHEN** two collections have equal successful entries but different rejected-entry tombstones
- **THEN** their snapshots compare unequal and analysis cannot silently treat the partial collection as the complete one

#### Scenario: Speculative cache identity is absent
- **WHEN** callers use the snapshot analysis API introduced by this change
- **THEN** no public stable hash or persistent result-reuse contract is implied; adding one requires a separate proposal
