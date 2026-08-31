# Why three rules in the agent contract exist

`AGENTS.md` states rules. This file holds the incidents that produced three of
them, because the evidence is instructive, ages differently from the rule, and
would otherwise be paid for by every agent that loads the contract whole.

Each section is referenced from the rule it explains. Nothing here is normative:
if this file and `AGENTS.md` disagree, `AGENTS.md` is the contract.

## 1. The 2026-07 f64 drift, behind "Numbered Specs Decide; Design Docs Implement"

A design doc is a working artifact. It is read constantly while its phases are in
flight and stops being read the moment they ship, so a decision parked in one does
not survive the work that made it.

In 2026-07 that produced a three-level drift:

1. `spec/04-type-system.md` [04-NUM-2] PERMITTED one narrow thing: "computing a
   single op in f64 and rounding once is a conforming implementation".
2. `spec/design/dtype_semantics.md` cited that permission to MANDATE f64
   computation for every float op.
3. The evaluator then extended the mandate to multi-step reductions and to
   comparison operands.

Each step was a reasonable reading of the one above it. Nobody re-checked against
the numbered spec, and the result was a language that computed f32 programs in
f64.

Two rules follow, and both are cheap. They are stated in `AGENTS.md`; the worked
examples are here:

- **Lift a language decision out of a design doc and leave a pointer.**
  `spec/05-risc-primitives.md` §8's [05-OBS-1..5] is the worked example: the
  observation contract moved out of `faithful_observation.md` and now survives
  independently of it. `spec/04` §9's [04-NUM-9..11] and §9.1 followed, for the
  trap contract, the exactness guarantee, and the per-dtype value table.
- **Watch for permission-to-mandate escalation.** "X is a conforming
  implementation" in a spec does not license "therefore we do X" in a design doc,
  and neither licenses "therefore we do X everywhere" in code.
  `spec/design/dtype_semantics.md` §B1 calls amending the spec first "the
  protocol, not a failure."

## 2. Why there is no META issue beside a tracker, behind "One Tracking Issue Per Class"

The existing META/tracker pairs ([#727]/[#729], [#703]/[#730], and siblings) are
historical, not a pattern to copy. The METAs were filed during the 2026-07 numeric
audit as evidence records; the trackers were filed later, when the design docs
were written, as delivery contracts.

Three reasons the split stopped paying for itself:

1. **It broke down.** Three of the five "METAs" are closed. #709 and #710 closed
   while their class continues under an open #731, and neither was written as a
   META: both are instance reports the class map promoted after the fact. #728
   closed alongside its own tracker #732, so that pair carried no information the
   tracker did not. (#732's closure was itself an accidental keyword auto-close
   fired by narrative prose in PR #1250's body, which is a second reason not to
   read the pattern as deliberate.)
2. **Sub-issues do the job the pairing was improvising.** When the tracker is the
   parent, "what belongs to this class" is a structural fact. A second issue whose
   content is a list of instances duplicates the child list and drifts from it.
3. **Two bodies means two things to keep honest**, and the evidence half has a
   better home: `docs/investigations/` already holds the probe corpus and the audit
   record.

Reason 1 is itself evidence for keeping this out of the contract: its counts have
already moved twice since it was written.

## 3. The build-contention measurement, behind "Build Concurrency And Process Hygiene"

Several unrelated tests failing at near-identical wall-clock times, for example all
around 217s (nextest's slow-kill), means CPU starvation rather than code breakage.

Measured 2026-06-10: the 25-test `rank_poly_tier3` suite took 2,434s under
contention against 24s on a quiet machine, a factor of about 100. That is the
number to keep in mind before treating a wave of simultaneous timeouts as real
failures.

[#703]: https://github.com/Chelis-Lang/chelis/issues/703
[#709]: https://github.com/Chelis-Lang/chelis/issues/709
[#710]: https://github.com/Chelis-Lang/chelis/issues/710
[#727]: https://github.com/Chelis-Lang/chelis/issues/727
[#728]: https://github.com/Chelis-Lang/chelis/issues/728
[#729]: https://github.com/Chelis-Lang/chelis/issues/729
[#730]: https://github.com/Chelis-Lang/chelis/issues/730
[#731]: https://github.com/Chelis-Lang/chelis/issues/731
[#732]: https://github.com/Chelis-Lang/chelis/issues/732
