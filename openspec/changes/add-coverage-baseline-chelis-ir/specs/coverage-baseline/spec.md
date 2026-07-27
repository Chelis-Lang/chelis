# Coverage Baseline: add-coverage-baseline-chelis-ir

## ADDED Requirements

### Requirement: Coverage measurement is scoped to one named package
The coverage wrapper SHALL measure exactly one workspace package named on its command line. It SHALL drive `cargo llvm-cov nextest` so the measured run matches the runner that `scripts/gate.py` uses for the `integration` stage. It SHALL report LLVM source-region coverage, not line coverage, because region counts distinguish individual `match` arms. After parsing the LCOV output it SHALL discard every source path outside that package's own `src/` directory, so coverage attributed to dependency crates never enters the summary. The wrapper SHALL NOT accept a request to measure the whole workspace.

#### Scenario: Named package is measured and summarized per file
- **WHEN** the wrapper is invoked for `chelis-ir`
- **THEN** it runs `cargo llvm-cov nextest -p chelis-ir` and emits a per-file region-coverage summary in which every reported path is under `crates/chelis-ir/src/`

#### Scenario: Dependency sources are excluded from the summary
- **WHEN** an instrumented run attributes covered regions to `chelis-deep`, `chelis-surf`, or any other dependency of the measured package
- **THEN** those paths are absent from the summary and do not contribute to the reported totals

#### Scenario: Workspace-wide measurement is refused
- **WHEN** the wrapper is invoked without a package name, or with a request to measure every package
- **THEN** it exits nonzero and reports that measurement is scoped to a single named package

### Requirement: An empty measurement is a loud failure, never zero percent
When the source-path filter matches no file in the parsed LCOV output, the wrapper SHALL exit nonzero and name both the filter it applied and the LCOV path it read. It SHALL NOT report 0.0% coverage in this case. A path-filter defect, a crate rename, or a moved source root is indistinguishable from a total loss of coverage once it is rendered as a percentage, and the two demand opposite responses from the reader.

#### Scenario: Filter matches nothing
- **WHEN** the parsed LCOV output contains no source path under the measured package's `src/` directory
- **THEN** the wrapper exits nonzero and reports the applied filter and the LCOV path, and prints no coverage percentage

#### Scenario: Genuine zero coverage of a file is reported normally
- **WHEN** the filter matches a source file and that file has zero covered regions
- **THEN** the wrapper reports that file at 0.0% within an otherwise normal summary and exits zero

### Requirement: Broken toolchain and malformed input fail loudly
The wrapper SHALL exit nonzero, and SHALL name the offending input, when `cargo-llvm-cov` is not installed, when the named package is not a member of the workspace, or when the baseline file is absent or malformed. It SHALL NOT substitute a default, skip the check, or degrade to a partial result in any of these cases.

#### Scenario: cargo-llvm-cov is not installed
- **WHEN** the wrapper runs on a machine without the `cargo-llvm-cov` binary
- **THEN** it exits nonzero and reports the missing tool together with its installation command

#### Scenario: Package is not a workspace member
- **WHEN** the wrapper is invoked with a package name absent from the workspace
- **THEN** it exits nonzero and names the unknown package

#### Scenario: Baseline file is malformed
- **WHEN** a comparison is requested against a baseline file that is absent, is not valid JSON, or lacks its provenance block
- **THEN** it exits nonzero and names the baseline path and the defect

#### Scenario: Valid inputs produce a summary
- **WHEN** the tool is installed, the package is a workspace member, and the baseline is well formed
- **THEN** the wrapper completes and exits zero

### Requirement: The committed baseline carries provenance and is advisory
The committed baseline SHALL record `rustc_version`, `host_triple`, `recorded_at`, and the test runner used, alongside per-file region counts. The baseline is authoritative only for the environment named in its own provenance block. When the current environment differs from the recorded one, the wrapper SHALL print a warning, SHALL label its comparison output as cross-environment, and SHALL still exit zero. The baseline SHALL be regenerated only by an explicit maintainer action, and SHALL NOT be rewritten automatically by any workflow or merge. No coverage percentage SHALL be treated as a pass or fail threshold.

#### Scenario: Comparison in the recorded environment
- **WHEN** the wrapper compares a fresh run against the baseline on the host and toolchain named in the provenance block
- **THEN** it reports the per-file deltas without a cross-environment warning and exits zero

#### Scenario: Comparison from a different host or toolchain
- **WHEN** the wrapper compares a fresh run against the baseline on a different host triple or rustc version
- **THEN** it prints a warning, labels the reported deltas as cross-environment, and exits zero

#### Scenario: A coverage decrease does not fail the run
- **WHEN** a fresh run reports lower region coverage than the baseline for one or more files
- **THEN** the wrapper reports the decrease and exits zero, because no percentage is a threshold

#### Scenario: No workflow rewrites the baseline
- **WHEN** the scheduled coverage workflow completes
- **THEN** the committed baseline file is unchanged and no commit is produced

### Requirement: Coverage runs outside the per-PR gate
The coverage workflow SHALL declare `workflow_dispatch` and a schedule trigger only, and SHALL NOT declare a `pull_request` trigger. It SHALL NOT be a required status check. It SHALL be listed in `NON_GATE_WORKFLOWS` in `scripts/test_gate.py`, so that the existing scope-classification test forces the exclusion to be deliberate and visible. Because instrumented builds increase object size, the workflow SHALL apply the same disk mitigations the existing Linux jobs use for chelis#392: the free-disk action and the dev/test debuginfo strip. The wrapper SHALL NOT be added to any `scripts/gate.py` stage.

#### Scenario: Workflow never runs on a pull request
- **WHEN** a pull request is opened or updated
- **THEN** the coverage workflow does not run and reports no status context

#### Scenario: Workflow is scope-classified
- **WHEN** `scripts/test_gate.py` enumerates the files under `.github/workflows/`
- **THEN** the coverage workflow appears in `NON_GATE_WORKFLOWS` and the scope-classification test passes

#### Scenario: Gate parity is preserved
- **WHEN** `scripts/test_gate.py` checks that gate jobs run only commands produced by `scripts/gate.py`
- **THEN** the check passes, because no coverage command was added to any gate stage

#### Scenario: Manual dispatch produces a summary
- **WHEN** a maintainer dispatches the workflow
- **THEN** it runs the wrapper for `chelis-ir` on the recorded host and publishes the per-file summary, without committing any file
