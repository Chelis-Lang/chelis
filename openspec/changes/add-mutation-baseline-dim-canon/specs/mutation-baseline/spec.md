# Mutation Baseline: add-mutation-baseline-dim-canon

## ADDED Requirements

### Requirement: Mutation scope is an explicit function list within one named file
The wrapper SHALL restrict mutation to one named source file and to an explicit, source-controlled list of function names within it. It SHALL compose the tool's name filter from that list rather than accepting a free-form regular expression, so the scope is reviewable as a list of identifiers. It SHALL apply the filter before the run, not to the results afterwards, because mutation cost is dominated by the per-mutant build and test rather than by the report. It SHALL refuse a request to mutate a whole crate or the whole workspace.

#### Scenario: Named functions in the named file are mutated
- **WHEN** the wrapper runs against `crates/chelis-ir/src/dag.rs` with its list of canonicalizer function names
- **THEN** every generated mutant lies in that file and names one of the listed functions

#### Scenario: Out-of-scope functions in the same file are not mutated
- **WHEN** the target file contains functions absent from the list
- **THEN** no mutant is generated for them and no build or test cost is paid for them

#### Scenario: Crate-wide mutation is refused
- **WHEN** the wrapper is invoked without a file and function list, or with a request to mutate a whole crate
- **THEN** it exits nonzero and reports that scope is limited to an explicit function list in one file

### Requirement: An empty or under-populated mutant population is a loud failure
The wrapper SHALL enumerate mutants before running them, and SHALL exit nonzero when the enumeration is empty, naming the file filter and the composed name filter. It SHALL additionally exit nonzero when any function named in its list contributes no mutant, naming every absent function. It SHALL NOT report a successful or perfect result in either case. The underlying tool exits zero when every tested mutant was caught, and a filter that matches nothing therefore satisfies that condition vacuously; the wrapper exists in part to make that reading impossible.

#### Scenario: Empty enumeration fails
- **WHEN** the composed filters match no mutant
- **THEN** the wrapper exits nonzero, names both filters, and reports no catch rate

#### Scenario: A renamed function drops out of scope and fails
- **WHEN** one function in the wrapper's list no longer exists in the target file, while the others still generate mutants
- **THEN** the wrapper exits nonzero and names the function that contributed no mutant

#### Scenario: A fully populated scope proceeds
- **WHEN** every function in the list contributes at least one mutant
- **THEN** the wrapper proceeds to run the mutants

#### Scenario: Enumeration needs no build
- **WHEN** the wrapper performs its scope checks
- **THEN** it uses the tool's mutant listing and does not build or test any mutant to do so

### Requirement: Tool exit codes are mapped explicitly and the baseline is never skipped
The wrapper SHALL distinguish the tool's exit codes rather than testing for zero. Missed mutants SHALL NOT be treated as a wrapper failure, because a surviving-mutant list is the artifact this capability produces. A failing unmutated baseline SHALL be a loud failure distinct from every other outcome, because it means the suite was already red and no mutant result carries information. A usage error and an internal error SHALL each be loud failures naming the invocation. The wrapper SHALL NOT pass an option that skips the unmutated baseline run.

#### Scenario: Missed mutants are data, not an error
- **WHEN** the run completes with surviving mutants and a green baseline
- **THEN** the wrapper reports the survivors and does not treat their existence as a tool failure

#### Scenario: A failing baseline is reported distinctly
- **WHEN** the unmutated baseline fails its own tests
- **THEN** the wrapper exits nonzero, states that the baseline was red, and does not present any mutant outcome as meaningful

#### Scenario: Usage and internal errors are loud
- **WHEN** the tool reports a usage error or an internal error
- **THEN** the wrapper exits nonzero and names the invocation it issued

#### Scenario: The baseline is always run
- **WHEN** the wrapper composes the tool invocation
- **THEN** the invocation contains no option that skips the unmutated baseline

### Requirement: Timeouts are a distinct outcome and are never counted as caught
The wrapper SHALL report timed-out mutants as their own class, separate from caught, missed, and unviable. It SHALL NOT count a timeout as caught. A mutation that makes the suite hang was not detected by any assertion; the run simply did not finish, and the two facts warrant different responses. The committed baseline SHALL record timeouts separately from both caught and missed counts.

#### Scenario: A non-terminating mutant is classified as a timeout
- **WHEN** a mutation of the Euclidean loop in `usize_gcd` produces a run that exceeds the per-mutant timeout
- **THEN** the wrapper records that mutant as a timeout and not as caught

#### Scenario: Timeout counts appear separately in the report
- **WHEN** the run produces caught, missed, and timed-out mutants
- **THEN** the summary reports three separate counts and the catch rate excludes timeouts from the caught total

#### Scenario: A genuinely caught mutant is not reclassified
- **WHEN** a mutation causes a test to fail within the timeout
- **THEN** the wrapper records it as caught

### Requirement: Survivors are classified against region coverage
The wrapper SHALL cross-reference each surviving mutant's source line against a region-coverage export for the same span, and SHALL report a survivor in an unreached region separately from a survivor in a reached region. Only a survivor in a reached region SHALL be presented as evidence about assertion strength; a survivor in an unreached region restates a coverage gap already measurable more cheaply. The wrapper SHALL consume a freshly produced coverage export and SHALL NOT read the committed coverage baseline file. When no coverage export is available the wrapper SHALL exit nonzero rather than emitting an unclassified survivor list.

#### Scenario: A survivor in reached code is a finding about assertions
- **WHEN** a surviving mutant lies on a line whose region the coverage export records as executed
- **THEN** the wrapper classifies it as missed and presents it as evidence of a weak assertion

#### Scenario: A survivor in unreached code is classified separately
- **WHEN** a surviving mutant lies on a line whose region the coverage export records as never executed
- **THEN** the wrapper classifies it as unreached and excludes it from the assertion-strength findings

#### Scenario: A missing coverage export fails the run
- **WHEN** no region-coverage export is available for the target span
- **THEN** the wrapper exits nonzero and does not emit a survivor list

#### Scenario: The committed coverage baseline is not consulted
- **WHEN** the wrapper classifies survivors
- **THEN** it reads only the freshly produced export and does not depend on the committed coverage baseline file

### Requirement: The committed baseline carries provenance and triage, and is advisory
The committed baseline SHALL record `rustc_version`, `host_triple`, `recorded_at`, the test runner used, and the crates included in the test scope, alongside the mutant population and per-outcome counts. Every surviving mutant SHALL carry a triage reason of equivalent, unreached, or untested. The baseline SHALL be regenerated only by explicit maintainer action and SHALL NOT be rewritten by any workflow or merge. No catch rate SHALL be treated as a pass or fail threshold, and a decrease SHALL NOT fail a run.

#### Scenario: Baseline records its environment and scope
- **WHEN** a maintainer regenerates the baseline
- **THEN** it contains the toolchain, host, timestamp, runner, and the list of crates whose tests were run

#### Scenario: Every survivor is triaged
- **WHEN** the baseline is committed
- **THEN** each surviving mutant carries one of the three triage reasons and none is left unclassified

#### Scenario: A lower catch rate does not fail the run
- **WHEN** a fresh run reports a lower catch rate than the committed baseline
- **THEN** the wrapper reports the difference and does not fail on that basis

#### Scenario: No workflow rewrites the baseline
- **WHEN** the scheduled mutation workflow completes
- **THEN** the committed baseline is unchanged and no commit is produced

### Requirement: Mutation runs outside the per-PR gate
The mutation workflow SHALL declare `workflow_dispatch` and a schedule trigger only, and SHALL NOT declare a `pull_request` trigger. It SHALL NOT be a required status check. It SHALL be listed in `NON_GATE_WORKFLOWS` in `scripts/test_gate.py`, so the existing scope-classification test forces the exclusion to be deliberate. No mutation command SHALL be added to any `scripts/gate.py` stage.

#### Scenario: Workflow never runs on a pull request
- **WHEN** a pull request is opened or updated
- **THEN** the mutation workflow does not run and reports no status context

#### Scenario: Workflow is scope-classified
- **WHEN** `scripts/test_gate.py` enumerates the files under `.github/workflows/`
- **THEN** the mutation workflow appears in `NON_GATE_WORKFLOWS` and the scope-classification test passes

#### Scenario: Gate parity is preserved
- **WHEN** `scripts/test_gate.py` checks that gate jobs run only commands produced by `scripts/gate.py`
- **THEN** the check passes, because no mutation command was added to any gate stage

#### Scenario: Manual dispatch produces a report
- **WHEN** a maintainer dispatches the workflow
- **THEN** it runs the scoped mutation and publishes the classified survivor summary, without committing any file
