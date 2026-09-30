# Core-Fragment Eval/C Parity Corpus And Release Receipt

Owning issue: [chelis#2102](https://github.com/Chelis-Lang/chelis/issues/2102)
(`scope:core`, `launch:required`). Parent: chelis#1362, the 0.19 core-fragment
launch ledger.

This is a design document. It decides how the parity corpus is selected, pinned,
executed, and reported. It decides nothing about language semantics.

Every **observation** rule it consumes is normative in
`spec/05-risc-primitives.md` §8 and is cited at each use. The **trap** criterion
is not: it is harness policy with no authority in §8, and §2.3 says so rather
than borrowing an atom that does not govern it. Where this document and a
numbered chapter disagree, the chapter wins and this document has a bug.

## 1. What this deliverable is, and what it is not

chelis#1362 guarantee 2 is *"`eval` and `build --target c` agree on observations
and traps."* Nothing in the repository currently executes that guarantee over a
corpus of real programs. #2102 is the evidence artifact that does, and it is also
the **only authority for `demo-path` tagging** on the launch ledger: an issue
earns `demo-path` because it breaks a case this manifest pins, and for no other
reason.

It is deliberately narrower than chelis#754 and chelis#763. Out of scope here, by
#2102's own text: a new public `lane-check` command, Nix hermeticity, GPU lanes,
general capability inference, and a broad mutation harness. Those remain #754's
and #763's.

Two consequences of that narrowing are worth stating, because they are the
difference between this receipt and a hermetic gate:

- A receipt verdict is evidence about the **recorded** platform, toolchain, and
  flag set, never about cross-platform determinism. #754's scope boundary
  already says this for the wider gate; it applies here unchanged.
- The receipt is **not** a substitute for #763's comparator surface. It reuses
  #763's comparison *predicate* (§3.3) rather than reimplementing a second one.

## 2. The observation channel

### 2.1 The decision: stdout against stdout

The receipt compares **the complete stdout of
`chelis eval --file P --target c` against the complete stdout of the compiled
binary built from `P`, byte for byte.**

`--target c` is load-bearing and not an implementation detail: it manifests the
eval lane under C backend constraints, which is what makes the two lanes'
root sets comparable at all (`[05-OBS-7]` scopes roots to a selected target and
`[05-OBS-10]`'s lane assignment is target-aware).

This is not a convenience choice. It is the channel the normative observation
rules already bind on both sides:

- `[05-OBS-6]`: every root renders with a `name = value` label at every exit in
  both lanes, in manifest entry order.
- `[05-OBS-1]`, `[05-OBS-2]` and §8.1: the number grammar, integer exactness,
  `bool` spelling, `key` spelling, and NaN class-level round-tripping are
  identical across lanes.
- `[05-OBS-4]`: a scalar-typed value renders bare at every exit in both lanes.

Byte equality of stdout is therefore **entailed by** those four atoms together
with §8.1. No single sentence in the chapter states it, and this document does
not claim one does. In particular the sentence *"Every lane produces
byte-identical text for the same admitted stored value"* is **`[05-OP-25]`**
(`to_string`, `spec/05-risc-primitives.md:1670`) and is scoped to that
callable's admitted stored values, not to the stdout observation channel; it is
consistent with the conclusion here but is not its authority.

Two qualifications belong with the claim rather than further down:

- `[05-OBS-3]` **permits** a cross-lane value difference of 1 ULP for `atan`,
  `cos`, `exp`, `log`, `sin` and `tan` — that is, permits different bytes. Byte
  equality is the operative contract only because that atom's arithmetic-width
  precondition is currently unmet; §3.2 states why. The corpus uses `exp`,
  `log` and `sqrt` throughout, so when chelis#897 lands this section needs
  revisiting in the same change set, and the receipt becomes over-strict for
  those six until it is.
- A receipt that compared any other channel would be measuring something the
  spec does not bind across these two lanes (§2.2).

### 2.2 Why not `eval --json` against scraped binary stdout

`chelis eval --json` emits the `EvalResult` wire form. The compiled binary has no
`--json`: it prints only the human rendering. Comparing the two therefore
requires converting one channel into the other, and that conversion is where a
harness accumulates its own defects rather than the compiler's.

This is measured, not predicted. The existing Voyage probe
(`Chelis-Lang/Voyage`, `probes/qcb-compiled/driver.py`, `_qx_to_eval_json`) takes
exactly that route, and its hand-written converter fails on lists of tuples and
on tensor printing for about eight programs — failures attributable to the
converter, not to either lane. Comparing stdout to stdout removes the converter
from the system entirely, and a re-measurement over 44 of those captures,
including all four the converter had failed on, found the two lanes' stdout
byte-identical in 44 of 44.

The two channels are also not equivalent in fidelity, which is the deeper reason:

- `[05-OBS-5]`: **every exit in both lanes** truncates tensor element rendering
  after 32 elements and marks the cut with `, ...`.
- The same atom: *"`to_list` and the wire schema never truncate: full-element
  fidelity is theirs."*

So the wire form carries all elements and the print form carries at most 32.
Comparing `eval --json` against compiled stdout compares a complete observation
against a truncated one. Comparing stdout against stdout compares two
observations the spec requires to be identical. §5.2 records what the 32-element
bound costs this receipt, and why that cost must be measured rather than assumed
away.

### 2.3 The trap channel

The one part of this receipt that is **harness policy rather than a normative
contract**, and it is labelled as such because the distinction was previously
blurred here.

The criterion the receipt applies: two lanes agree on a trap when their process
exit statuses are equal and their complete stderr is byte-identical.

**That criterion has no authority in `spec/05` §8.** `[05-OBS-6]`'s
*"A lane that cannot produce a root it owes SHALL emit [05-UNS-1] naming that
root, the lane, and the reason"* does not supply it: "lane" there is the
Tensor/Host lane (`[05-OBS-9]`: *"assign each entry its Tensor or Host lane"*),
not eval-versus-compiled. Read as this document previously read it, the atom's
requirement to name *the lane* would make byte-identical stderr impossible
rather than mandatory. Nothing in §8 requires the eval process's and the
compiled binary's diagnostic bytes to match.

The criterion is also already known to be too strong for the obvious candidate
cases. Measured at the pinned revisions, a program both lanes reject on the same
unbound variable produces:

```text
eval : error: Type errors:\n  UnboundVariable: ...
build: error: Check errors: Type errors:\n  UnboundVariable: ...
```

— the same rejection for the same reason, differing by one prefix. Under the
criterion above that is a `trap-reason-mismatch` on a spec-conforming pair.

Nothing is mis-verdicted today because manifest version 3 declares no trap case
(§10). What the receipt does claim, and what does rest on #1362 guarantee 2
rather than on a formatting choice, is the asymmetric case: **a case where one
lane traps and the other returns a value is a divergence**, and it is the most
severe verdict the receipt produces. That judgement needs no byte comparison.

Deciding the trap criterion properly — whether it should require comparable
pipeline positions, a normalised diagnostic identity, or a typed error code
rather than raw bytes — is owed before the first trap case is admitted, and §10
records it as residual rather than this section pretending it is settled.

Nothing in the repository compares trap reasons across lanes today; the Voyage
probe records exit status only. So this is new coverage, but coverage of a
criterion that is not yet the right one.

## 3. The comparison predicate

### 3.1 One predicate, already shipped

`chelis_types::agreement::compare_exact_observations` is the exact branch of the
one Phase 3 comparator (chelis#732). Its body is, verbatim:

```rust
if reference.as_bytes() == candidate.as_bytes() {
    Ok(AgreementOutcome::ByteExact)
} else {
    Err(AgreementError::ExactMismatch { .. })
}
```

Byte equality of the two captured streams is therefore the same predicate, not
an approximation of it. The receipt runner states that equivalence as a claim and
`scripts/test_core_fragment_parity_receipt.py` locks it: if that function ever
stops being byte-exact, the test fails and the runner's claim has to be re-earned
rather than silently rotting.

### 3.2 No tolerance, and why this receipt cannot grant one

`[05-OBS-3]` permits a cross-lane value difference only for the seven listed
transcendental operations, only within the listed bound, and only when **both
lanes compute at `[04-NUM-8]`'s declared arithmetic width**. chelis#897 records
that the evaluator currently computes float operations in f64 regardless of the
declared width, so that precondition is unmet for float arithmetic. The atom is
explicit that a lane pair failing the precondition *"is a failed comparison and
may not be laundered through the table."*

The receipt therefore runs with **no tolerance at all**, and this is the spec's
instruction rather than a strictness preference. A float mismatch is a named,
issue-linked failure.

### 3.3 The comparison is per case, over complete streams

The receipt compares whole streams, never per-root parsed values. A difference in
line count, root label, render order, shape text, element count, numeric
spelling, truncation marker, or any other byte is a divergence. There is no
per-root re-parse to go wrong, and no blanket tolerant fallback to reach for.

## 4. Pins

A verdict with an unpinned input is not evidence. Three classes of input are
pinned, in two different places, for a reason.

### 4.1 Corpus revisions — pinned *in* the manifest

Each corpus contributes at one exact revision, recorded in the manifest, because
the case set is a property of the manifest version:

| corpus | repo | kind |
|---|---|---|
| C Note | `Chelis-Lang/c-note` | committed `.ch` files |
| Sonar | `Chelis-Lang/sonar` | committed `.ch` files |
| Voyage | `Chelis-Lang/Voyage` | **derived** programs, generated by a pinned probe |

The Voyage row is the one that forces a schema decision. Its cases are not
committed files: they are captured at run time by `probes/qcb-compiled/driver.py`
during the `interp` phase, which intercepts each `chelis eval --file … --json`
call and writes `cap/<task>/<n>/k.ch` plus its `inputs/`. A path into the Voyage
tree cannot name such a case. The manifest therefore carries two source kinds
(§6.2): a committed case is pinned by `(repo, rev, path)`, and a derived case is
pinned by `(repo, rev, generator, task, index)`.

### 4.2 Chelis revision — recorded *by* the receipt, not fixed by the manifest

The manifest does **not** pin a Chelis revision, and this is deliberate. The
manifest declares which cases are required; the receipt records which compiler it
ran them against. A launch receipt is then "the receipt at the release-candidate
revision", and the same manifest version keeps producing comparable receipts
across the repair work #1362 sequences.

Baking a compiler pin into the manifest would force one of two bad outcomes: pin
an old release and the receipt stops describing the thing being shipped, or bump
the pin on every merge and the manifest version stops meaning anything.

Each receipt records, for the compiler under test:

- the git revision and `--version` string;
- the complete staging receipt, which carries the build mode, the runtime
  archive's sha256, and every published header's digest; and
- the resolution path by which the binary was found.

The build mode is read from the compiler's own declared channel, not inferred.
`chelis_runtime_bundle` writes `chelis_runtime.receipt.json` beside every staged
runtime — its own comment says "Staging receipts record the mode" — carrying
`"mode": "sealed"` or `"mode": "development"`. The receipt performs one throwaway
build during pin collection and reads that file. An unreadable or unrecognised
mode is `null`, never "sealed": a probe that cannot tell must not silently stop
the check firing.

The mode is not bookkeeping. An unsealed development build re-checks its source
checkout for runtime-bundle freshness on every `build` invocation, so a
concurrent writer to that checkout can change the binary's behaviour mid-run
while its hash stays constant. This is observed, not hypothetical: a pinned
before/after comparison over the Voyage corpus lost 68 programs mid-run to a
validation gate mutating the checkout, with both binaries' hashes unchanged
throughout. A hash alone does not pin an unsealed build, so the receipt fails on
a development-mode compiler rather than merely noting it.

The resolution path is recorded because the failure it guards against is silent:
the bare `chelis` shim resolves through the pin order in `AGENTS.md`
§"Shim resolution order", which outside a reef package on this workstation lands
on 0.13.0. A receipt that does not say which binary it ran is not a receipt.

### 4.3 The C toolchain and its flags — pinned per receipt, and part of the contract

The compiled lane is `build --target c` **plus** a compile and link step the user
performs. The flags of that step change the observations. The receipt therefore
records the compiler identity, version, target triple, and the exact compile and
link command, and treats the flag set as part of the verdict's scope.

This is measured. chelis#2782: `chelis build --target c` prints a compile line
carrying `-O2 -march=native` and no `-ffp-contract=off`
(`crates/chelis-backend-c/src/toolchain.rs:52`). GNU C defaults to
`-ffp-contract=fast`, so gcc fuses `a*b + c` into an FMA where the target has
one; the interpreter does not fuse. Over the 190 Voyage programs that build, **91
differ from `eval` in at least one value under the printed line, and 91 of 91
match bit-for-bit once `-ffp-contract=off` is added.**

Two things follow, and they belong to different owners:

- **This document's part:** the receipt records the flag set and reports under it.
  A receipt is scoped to the profile it ran.
- **chelis#2782's part:** whether the *shipped* printed line should disable
  contraction is a product decision about the emitted command, and #2782 owns it.
  This document does not decide it and must not be read as having decided it.

What #2102 does decide is the consequence for the ledger: because #2782 breaks
cases this manifest pins, it is a `demo-path` row (§7).

## 5. Non-vacuity

#2102 requires "non-vacuous case discovery". A receipt that discovers nothing,
compares nothing, or compares only cases that cannot distinguish the lanes is the
failure mode this section exists to make impossible.

### 5.1 Structural non-vacuity

The receipt exits nonzero, with a named reason, when any of these holds:

1. the manifest declares zero required cases;
2. any required case is missing from the discovered set;
3. any discovered case is absent from the manifest — an unclassified case is a
   build failure, never an implicit skip;
4. zero required cases produced a comparable observation; or
5. the per-corpus required-case counts do not match the manifest's declared
   expected counts.

Rule 3 is the one that matters most in practice and the reason the manifest is
per case rather than a directory glob. Both C Note and Sonar carry directories of
deliberately-failing probes (`c-note/fixtures/dischargeability/probes/`,
`sonar/recon/probes/`). A glob would promote those into the required set, and a
deliberately-failing probe that fails identically in both lanes would then be
counted as parity evidence — the exact vacuous pass this receipt exists to
prevent. Under §6.2 such a file is either a required trap case with a declared
expected trap, or an excluded row with a reason. It cannot be neither.

Rule 5 is narrower than it looks, and it is worth saying which check it is not.
It compares the manifest's declared `expected_case_count` against the manifest's
own case rows, so it catches a hand-edited or regenerated row drifting from the
declared count. It does **not** compare either number against a measured
outcome, so it is not the count-level golden assert that the existing manual
Voyage census — which reproduces a pinned revision's exact build/accept/reject
totals — would become if it were wired.

What pins expected outcomes here is stronger than a count and weaker in a
different way: `expected` and `known_divergence` are recorded **per case**
(§6.2), so a case that changes verdict is named rather than absorbed into a
total that still adds up. A count-level assert remains owed for the derived
Voyage third, where cases are generated rather than committed and a changed
generator can alter the population itself.

### 5.2 Observation non-vacuity, and the 32-element bound

Structural counts are not sufficient. Because `[05-OBS-5]` truncates tensor
rendering at 32 elements in both lanes, a case whose roots are all large tensors
has at most 32 elements per root actually compared, and the remainder is
**unobserved by this channel** — not verified, and not claimed to be.

The receipt therefore records, per case, whether any root's rendering carried the
`, ...` truncation marker, and reports two figures rather than one:

- the number of required cases that agreed; and
- the number of required cases that agreed **with no truncated root**.

A corpus whose agreement rests entirely on truncated renderings is reported as
such. This is a disclosure, not a gate: `[05-OBS-5]` is the specified observation
contract, so a truncated rendering is the correct observation, and full-element
fidelity is `to_list`'s and the wire's job by the same atom. A case that needs
full-element evidence states its roots through `to_list`; the manifest records
which cases do.

### 5.3 Exclusions are recorded per case, with a reason

Every Chelis source discovered in a pinned corpus that is not a required case
carries an exclusion reason from a closed set (§6.3). A count is not a reason.
Discovery walks `.ch` and `.dp`, the two forms `chelis eval --file` accepts; it
previously globbed `.ch` alone, which made this sentence false for the two Deep
programs Sonar carries at its pinned revision.

The corpus already demonstrates why. Of the Voyage captures, six programs are
rejected by `eval` itself, and **four of those six are rejected on the `sig`
reserved word** (captures 48/0, 63/1, 73/1, 90/0; the other two are one type
error and one unexpected end of input). A case that cannot parse and a case whose
lanes disagree are the same shape from outside — a nonzero exit and no
comparable observation — and only a recorded reason distinguishes them.

The same hazard is live in the committed corpora at their pinned revisions:
23 of Sonar's 45 `.ch` files reference `int64`, retired in favour of `i64`, and
`c-note/docs/qa_evidence/ground_truth/gordon.ch` is rejected at head by the
§12.5 redundant-grouping rule for property preconditions. Neither is a lane
disagreement. Both would read as one without §6.3.

## 6. The manifest

Location: `tests/corpus/core_fragment_parity/manifest.json`.

### 6.1 Versioning

The manifest carries `manifest_version`, an integer incremented by any change to
**what the receipt is asked to prove**: the required-case set, a pinned corpus
revision, an expected outcome, a case's declared roots, the **row set** of
the exclusion ledger, or an exclusion row's **reason** — a reason is an
assertion about the corpus, not evidence about a row. Editing an exclusion row's
`notes` does not bump it —
notes are recorded evidence about a row, not part of the obligation — and
neither does a `known_divergence` issue number changing, which tracks a defect
rather than the case.

A receipt records the manifest version it ran. `demo-path` assignments cite the
manifest version that justifies them (§7).

Version 2 added the two Sonar `.dp` exclusion rows that discovery began finding
when it stopped globbing `.ch` alone (§5.1 rule 3) — a row-set change, hence the
bump — and corrected four exclusion notes, which on its own would not have
warranted one. Version 3 moved twelve rows from `unmeasurable-by-probe` to their
operative reason (§9), a reason change with the row set unchanged. The
required-case set is unchanged from version 1 throughout.

### 6.2 Case rows

Every row is one case. Required fields:

| field | meaning |
|---|---|
| `case_id` | stable identity, unique across corpora; never reused after removal |
| `corpus` | `c-note`, `sonar`, or `voyage` |
| `source` | `{kind: "committed", repo, rev, path}` or `{kind: "derived", repo, rev, generator, task, index}`. A committed case keeps its suffix when materialized, so a `.dp` program is handed to the compiler as Deep rather than parsed as Surf |
| `expected` | `value` or `trap` — see below |
| `roots` | the root names the case owes, in manifest entry order. Required and non-empty when `expected` is `value`: §5.2's blackout guard reads this field, so an empty list disarms it, and a value case owing no observation is a `library-only` exclusion. `roots` declares the **base** root name — the identity `[05-OBS-7]` gives the root — and that is the canonical spelling. A declared root is satisfied by its own `name = ` label **or** by any `name.`-prefixed label, because `[05-OBS-8]` expands a tuple-valued root into dotted positional names and a fixed-product ADT root into its field names. The dotted spelling is *accepted* in `roots` but is not canonical and should not be used: the same atom says an ADT whose constructor is not statically fixed "remains one bare root", so a dotted declaration turns red the moment that fixedness changes, while a base declaration survives either rendering |
| `truncating_roots` | root names whose rendering is expected to carry `, ...` |
| `notes` | free text; may cite issues |

`expected` is required and has no default. A case whose expected outcome is
unknown is not a case yet: it is an exclusion with reason `outcome-undetermined`
until someone determines it.

**Two named limits of the root-presence check**, because the guarantee in §5 is
weaker than it first reads and should not have to be rediscovered:

- **A passing `trap` case carries no observation.** Trap admission is sometimes
  described as what stops a deliberately-failing probe counting as parity
  evidence; that is not what it does. A trap case renders nothing, declares no
  roots, and passes on agreeing diagnostics alone — which is §5's third failure
  mode, "compares only cases that cannot distinguish the lanes", by
  construction. §5's non-vacuity figures therefore describe **value cases
  only**, and a manifest consisting entirely of trap cases would pass while
  comparing zero observations. Nothing prevents that today beyond the fact that
  no trap case exists (§10).
- **A tuple or ADT root is checked at arity one.** Because a declared base root
  is satisfied by any one `name.`-prefixed label (above), a regression that
  silences `result.1` while `result.0` still renders is invisible to the
  presence check. Closing it would require declaring the full dotted expansion,
  which `[05-OBS-8]` makes unknowable in advance for a non-fixed constructor.
  This is the price of the widening, recorded rather than hidden.

A row may carry `known_divergence: {issue: N}`. Such a case is still run and
still compared, and **a tracked divergence still fails the receipt**. #1362's
ship rule is "no `demo-path` row remains open", and this manifest is
`demo-path`'s only authority, so a receipt that passed while a known divergence
was open would contradict the rule it exists to serve. `known_divergence` buys
diagnosis, not a pass: it separates a tracked defect from a regression, and the
exit status carries that distinction.

| exit | meaning |
|---:|---|
| 0 | `RECEIPT: PASS`. The only acceptance. |
| 1 | at least one untracked failure: a new divergence, a vacuous run, or a pin mismatch |
| 2 | the run could not start — an unusable invocation or a malformed manifest — so no verdict exists. `argparse` shares this code for a usage error, and the table is stated at that granularity rather than pretending to distinguish them |
| 3 | every failure is a tracked known divergence |

A row may not carry `known_divergence` without an issue number. A case that
agrees although its row records a divergence is reported as **unexpected-pass**,
which is the signal to re-check the issue and drop its `demo-path` tag — never
something to swallow.

### 6.3 Exclusion rows

Every non-case Chelis source in a pinned corpus — `.ch` or `.dp` — is an
exclusion row with `path` and a `reason` from this closed set:

| reason | meaning |
|---|---|
| `library-only` | no roots; nothing to observe |
| `prove-only` | exercises `prove`; not an eval/C case |
| `parse-rejected` | rejected at ingress by the compiler under test |
| `retired-syntax` | uses a spelling the compiler under test has retired |
| `nondeterministic` | clock, randomness, filesystem, network, or external input |
| `c-unsupported` | `build --target c` refuses it; the refusal's issue is cited |
| `unmeasurable-by-probe` | the harness cannot capture it; the limitation is named |
| `outcome-undetermined` | nobody has yet decided what it should do |

`parse-rejected` and `retired-syntax` are separate on purpose: the first is a
property of the program, the second of the pin, and conflating them hides
migration debt behind an apparent corpus defect.

**When a file satisfies more than one reason, record the reason that would still
exclude it if every other named limitation were removed.** Closing the
vocabulary does not close the choice within it, and the corpus contains such
files in numbers: a `@property`-only file behind an unresolvable `import` is both
`unmeasurable-by-probe` and `prove-only`, and only the second survives the
counterfactual. Recording the proximate symptom instead is the one classification
error a mechanical audit cannot find, because the symptom is something the
compiler really does — twelve rows were recorded that way and passed two
mechanical audits before a root-ownership predicate caught them (§9).

An unrecognised reason is a receipt failure, not a warning. The set is closed for
the same reason the C census's non-numeric type list is closed: an open list
silently absorbs the case nobody classified.

## 7. `demo-path` assignment

`demo-path` is #1362's corpus modifier: *"The issue breaks a pinned release
corpus case and gates launch regardless of priority."* This manifest is its only
authority. The procedure:

1. The receipt runs and reports a divergence or trap-parity failure on a required
   case. **Read the failing-stage distribution the receipt prints before filing
   anything:** a local C toolchain that cannot build anything makes every case
   lane-split at the `compile` stage, and this procedure followed literally
   would then file one issue per case. The receipt reports the counts and draws
   no conclusion from them — failures concentrated at `compile` suggest the
   toolchain, one `compile` failure beside agreeing cases suggests a genuine
   invalid-C defect, and that judgement is this step's, not the runner's.
2. The defect is filed, or an existing issue is identified. The receipt links it
   rather than re-deriving it.
3. That issue receives `demo-path`, with a comment naming the `case_id`, the
   manifest version, and the receipt's recorded pins.
4. The case's row gains `known_divergence: {issue: N}` so subsequent receipts
   distinguish it from a regression.
5. The tag is removed when a receipt at the same manifest version shows the case
   agreeing.

No `demo-path` issue exists in the tracker today, because #2102 has never run.
**chelis#2782 is the first**, and a reader should not assume a precedent exists
for how the tag is applied. It qualifies on its own measurement: it breaks 91 of
the 190 Voyage cases that build, which are pinned cases, which is the modifier's
stated condition.

## 8. The acceptance oracle

One command, per `AGENTS.md` §"One Acceptance Oracle Per Phase":

```sh
.venv/bin/python scripts/core_fragment_parity_receipt.py \
    --chelis <a sealed-runtime chelis binary> \
    --corpus c-note=<pinned c-note checkout> \
    --corpus sonar=<pinned sonar checkout> \
    --out target/parity-receipt
```

`--chelis` and at least one `--corpus` are required; the manifest defaults to
`tests/corpus/core_fragment_parity/manifest.json`. Build the compiler under test
with `cargo build -p chelis-cli --bin chelis --features sealed-runtime`, and to a
target directory a default-feature build will not overwrite — §4.2's pin check
fails a development-mode binary, and an ordinary `cargo run -p chelis-cli`
elsewhere in the repository silently replaces `target/debug/chelis` with an
unsealed one.

Acceptance is **exit 0 with a final `RECEIPT: PASS` line**. The receipt artifact
is written to `--out` and records the pins of §4, the per-case verdicts, the
non-vacuity figures of §5, the exclusion ledger of §6.3, and every command
executed, verbatim, in execution order.

This oracle is **not** part of the default workspace pass. It needs the three
corpora at their pinned revisions, a C toolchain, and a built compiler, so it is
a documented manual gate under `AGENTS.md` §"Manual Gates". It is long-running by
construction.

`scripts/test_core_fragment_parity_receipt.py` is the separate, fast, default-run
oracle for the runner's own decision logic, including the negative rows
§5.1 and §6.3 require. It is evidence about the script and never a substitute for
running it.

## 9. First receipt, and what it measured

Manifest version 3, run on macOS `arm64` with a sealed-runtime `chelis` built
from this branch, staging receipt `mode: sealed`, runtime archive `de658fb0…`,
over `Chelis-Lang/c-note` at `960a9beb` and `Chelis-Lang/sonar` at `9b26133f`:

| | |
|---|---:|
| Chelis sources discovered across both corpora | 132 |
| required cases | 13 |
| exclusions, each with a recorded reason | 119 |
| unclassified files | 0 |
| cases agreeing byte-exactly | 11 |
| cases agreeing with no truncated root | 11 |
| lane splits | 2 |

Exit 3: both failures are tracked known divergences, and #1362's ship rule still
blocks while they are open.

Cases: 11 from C Note, 2 from Sonar. Exclusions by corpus and reason:

| reason | c-note | sonar |
|---|---:|---:|
| `prove-only` | 49 | 15 |
| `unmeasurable-by-probe` | 12 | 0 |
| `retired-syntax` | 0 | 23 |
| `parse-rejected` | 8 | 6 |
| `library-only` | 5 | 1 |

Two rows in that table are the near-term levers on the case count, and they are
harness debt and migration debt respectively, not language defects:

- **`unmeasurable-by-probe` is 12 of 12 mine**, and only those 12 are a lever.
  Each is a package member carrying an `import`
  (`import CnoteEval.BlackScholes (bs_call, …)`); the receipt materializes one
  file, so the import cannot resolve, and the resulting `UnboundVariable` is a
  property of the harness rather than of the program. Package-aware
  materialization would admit these 12 as candidate cases.

  This figure was 24 and was wrong. Twelve of those rows own no root under
  `[05-OBS-7]` — ten are `@property` declarations beside an `import` and nothing
  else, two declare only parameterised `def`s — so resolving the import would
  not admit them as cases at all. They now carry their operative reason
  (`prove-only`, `library-only`). Their `UnboundVariable` was real and was the
  first thing to go wrong, which is exactly why it was recorded: §6.3's
  counterfactual test is what distinguishes the proximate symptom from the
  disqualifying property, and it was applied after the fact rather than at
  classification time.
- **`retired-syntax` is 23 of 23 Sonar's**, all on the pre-0.19 integer dtype
  spelling; the diagnostic names its own migration
  (`chelis migrate surf --from 0.18`). Sonar contributes 2 cases out of 47 files
  almost entirely for this reason, so migrating it is the single largest
  available increase in the corpus.

### 9.1 What the two lane splits are, and what they are not

`greeks.ch` and `black_scholes/call.ch` both take a scalar-`wrt` gradient of a
named top-level `def` — the form #1362's direct-transform table lists as
"Supported; exact on both lanes" — and `build --target c` refuses while `eval`
returns a value. All six c-note build refusals emit the same text, because
chelis#2755 records that the refusal message is generic; they are not one
mechanism.

What is established, on both 0.18.11 and current `main`:

- **A `cast` node in the differentiated body is sufficient to cause the
  refusal.** `add(s, 1.0)` builds; `add(s, cast(1.0, f32))` does not. So do
  `add(cast(s, f32), k)`, `add(s, cast(1, f32))`,
  `def f(s: f32, k: f32) -> f32 = cast(1.0, f32)` (the `wrt` parameter unused),
  and `div(mul(s, k), add(k, cast(1.0, f32)))`. `mul`, `add`, `div`, `sub`, `neg`,
  nesting depth, and `add(k, k)` are all innocent.
- **Two propositions chelis#2379's text asserts are false.** Its body names
  local binding structure as the discriminator and offers
  `div(mul(s, k), add(k, cast(1.0, f32)))` as a control that builds on 0.18.11.
  Measured here: `{ a = add(s, k)  mul(a, k) }` builds, and that control does
  **not** build on 0.18.11 or on `main`. The refutation is of the propositions;
  that #2379 asserts them is a reading of its issue text, recorded here as such
  so a later reader can re-check the issue rather than this document.

What is **not** established, and what this document previously claimed: that
`cast` is *the* discriminator. It is not. `add(abs(s), k)`, `add(relu(s), k)`
and `if (s >= 0.0) then add(s, k) else k` produce the same refusal with no
`cast` present, and `greeks.ch` with every `cast(X, f32)` textually removed
still evals and still refuses to build. Its `if` inside `normal_cdf` is the sole
remaining refusing element, measured rather than inferred: cast-stripped
`greeks.ch` with both `if` expressions also removed **builds**, and a nested
user-`def` call, that call wrapping `exp(neg(mul(x, x)))`, the same through a
local binding, unary minus, and infix division were each cleared individually.
Only `call.ch`'s refusal is cast-attributable: with casts removed it builds and
prints

```text
price = 19.98864
delta = 0.11209473
```

The refusal message itself points at the real shape — *"make sure that function
uses only pure tensor ops (sum, add, mul, einsum, etc.)"* — so this is an
op/form allowlist in the C lane's grad lowering. This document does **not** state
that allowlist: the probes above are a handful of data points, not a sweep, and
asserting a complete list from them would repeat the error it is correcting.

### 9.2 chelis#2782 does not earn `demo-path` from these corpora

Every one of the 13 cases produces identical verdicts under the `emitted` and
`no-fp-contract` profiles, so on clang/arm64 these cases are
contraction-insensitive. That does not contradict chelis#2782, whose
91-of-190 measurement was gcc on Linux aarch64 over recursive EMA/RSI/ATR/MACD
series; c-note's closed-form pricing has no such recursion. It does mean **#2782
earns `demo-path` from the Voyage third of the corpus, not from these two**, and
the Voyage third is not in manifest version 3 (§10).

## 10. Residual scope

Named, so a reader does not mistake this document for more than it is:

- **The Voyage third is not in manifest version 3.** The schema carries the
  `derived` source kind that pins `(repo, rev, generator, task, index)` for it,
  and the runner materializes such a case, but no Voyage rows are declared yet:
  its cases are captured at run time by `probes/qcb-compiled/driver.py`'s
  `interp` phase, which has to run first. This receipt consumes captures and
  does not generate them. Until those rows land, #2782's `demo-path`
  justification rests on measurement recorded in that issue rather than on a row
  in this manifest (§9).
- **Package-aware materialization is unbuilt**, which is what makes the 24
  `unmeasurable-by-probe` rows unmeasurable (§9). Admitting them needs the
  receipt to materialize a case together with the package it imports from.
- **No structured observation channel for the compiled lane.** §2.2's argument
  removes the need for one here, but a compiled binary still has no `--json`, so
  any future comparison needing full-element fidelity on the compiled side has
  nothing to read. Not filed as of this document.
- **Migration of the committed corpora to the 0.19 spellings is not done.** §5.3
  records the measured burden at the pinned revisions. Until it is done, those
  files are `retired-syntax` exclusions rather than cases, and the required-case
  counts are correspondingly smaller than the file counts.
- **The trap criterion itself is undecided, not merely untested.** §2.3 records
  that byte-identical stderr has no authority in `spec/05` §8 and is already
  known to be too strong for the obvious candidate cases: two lanes rejecting the
  same program for the same reason differ by an `error: Check errors: ` prefix.
  Choosing the real criterion — comparable pipeline positions, a normalised
  diagnostic identity, or a typed error code — is owed before the first trap case
  is admitted.
- **Manifest version 3 declares no `trap` case, so the trap channel of §2.3 is
  exercised only by unit tests.** Every one of the 13 required cases is
  `expected: "value"`. The trap comparison is implemented and its decision logic
  is covered positively and negatively in
  `scripts/test_core_fragment_parity_receipt.py`, but no real corpus case has
  driven it end to end, and this document does not claim otherwise. The
  deliberately-failing probes in `c-note/fixtures/dischargeability/probes/` and
  `sonar/recon/probes/` are the natural source, and they are currently
  `prove-only` exclusions because they own no root to observe. Admitting a trap
  case needs someone to decide the expected trap per probe, which is §6.2's
  `outcome-undetermined` boundary doing its job rather than a gap in it.
- **A default-feature `cargo` invocation anywhere in the repository overwrites
  `target/debug/chelis` with an unsealed binary**, which silently invalidates a
  recorded receipt's pins. §8 says to build the compiler under test to a target
  directory such a rebuild will not reach. Nothing enforces that; the pin check
  catches the consequence on the next run rather than preventing it.
- **A compiled-lane refusal at `build` and a trap at `run` are recorded as
  distinct stages but classified alike.** The verdict carries
  `compiled_stage` (`build`, `compile`, or `run`) so a reader can tell them
  apart. Whether a trap case should require the two lanes to fail at
  *comparable* pipeline positions, rather than merely with identical diagnostic
  bytes, is a question the first real trap case should settle. Deciding it now,
  with no case to test against, would be inventing a taxonomy.
- **The manifest declares root *identities*, not expected *observations*, and
  that choice is why §5's non-vacuity guarantee is a presence test rather than a
  comparison.** A golden-stdout-per-case representation would dissolve both
  named limits in §6.2 at once — a partial blackout inside a tuple root would
  fail on bytes, a trap case would carry a recorded observation, no spelling
  decision would arise because the golden records whatever `[05-OBS-8]` renders,
  and the check would reuse §3.1's byte comparator instead of a second
  label-parsing model of how rendering works. It would also stop the guarantee
  depending on a hand-maintained field. It is **not** proposed here: it changes
  what a manifest is, and adopting it belongs to a change that decides that
  deliberately rather than to a repair. It is recorded so the next reader
  evaluates the representation instead of hardening the presence test again.
- **Cross-platform verdict comparison is #754's**, not this receipt's.
