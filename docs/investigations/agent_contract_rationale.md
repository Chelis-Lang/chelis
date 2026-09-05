# Why some rules in the agent contract exist

`AGENTS.md` states rules. This file holds the incidents that produced some of
them, because the evidence is instructive, ages differently from the rule, and
would otherwise be paid for by every agent that loads the contract whole.

Each section is referenced from the rule it explains. Nothing here is normative:
if this file and `AGENTS.md` disagree, `AGENTS.md` is the contract.

Sections are numbered in the order they were added and keep their numbers, so a
citation stays valid. The numbering is not the contract's running order.

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

## 2. The build-contention measurement, behind "Build Concurrency And Process Hygiene"

Several unrelated tests failing at near-identical wall-clock times, for example all
around 217s (nextest's slow-kill), means CPU starvation rather than code breakage.

Measured 2026-06-10: the 25-test `rank_poly_tier3` suite took 2,434s under
contention against 24s on a quiet machine, a factor of about 100. That is the
number to keep in mind before treating a wave of simultaneous timeouts as real
failures.


## 3. The five-session fleet run, behind "Fan-Out Budget"

Measured across a five-session, roughly forty-hour agent fleet run in 2026-08.

A fleet has no running total. Each agent reports its own context and nobody
reports the sum, so the width of a fan-out is chosen without a cost figure in
front of the person choosing it. One session held roughly 839,000 tokens of live
context across seven concurrent agents sixteen minutes after launch. Four
sessions running that wide exhausted a shared usage window about twenty minutes
after launch.

The threshold is five rather than seven because those are the two widths the run
actually exercised: three explorers plus two planners ran comfortably, and the failure
appeared at seven. Nothing here establishes that five is a safe boundary, and the rule
does not claim it is. Six was never tried. The rule also counts agents while this
section measures tokens, which do not convert: five orchestrators at the figures above
would exceed the total that broke at seven. Headcount is used because it is the only
quantity observable at the moment of the spawn.

Two supporting figures. Orchestrators reached 200,000 to 265,000 tokens before
writing a line of plan; no rule is attached to that number, and it is recorded
because it bounds how much supervision capacity an orchestrator has left at the
moment it starts spending. Separately, exploration reports totalling 155,890
characters drove one orchestrator from 137,000 to 247,000 tokens before any code
existed, which is the report-length budget's reason for being a number in the
brief rather than an instruction to be brief.

On model tier: the fleet was told to reserve the expensive model for error-prone
work. It did not visibly apply that anywhere, because no spawn had to say which
tier it took or why. The one clear success was a deliberate cheap-tier choice for
an agent whose whole job was waiting on CI. A per-spawn sentence is what turns a
standing preference into an observable decision.

## 4. Briefs that were re-derived, and briefs that survived, behind "Briefs"

Same run.

Re-spawned planners received byte-identical briefs, one of them 10,187 characters
on both issues, and those briefs omitted the exploration reports already sitting
in the orchestrator's context. The planners re-read the same sources in 105 tool
calls. One design document was paged into a model context six times in a single
session. The orchestrator's own account is the useful part: the briefs did carry
the verified facts, but none of them said "these are verified, do not re-derive
them", and an agent handed facts without that line treats them as leads.

The file-pointer half has the cleanest contrast in the run. The one session that
wrote briefs to files had implementer prompts of 1,671 and 1,969 characters
pointing at 5 to 9 KB files behind a shared common-rules file. Three of its
implementers were killed and resumed without anyone re-authoring a brief, and two
red-team reviewers cut off mid-round resumed from their original briefs.
Elsewhere in the same run, two killed planning agents lost 67 and 72 tool calls
with nothing recoverable. The truncation figure is separate and easy to miss:
inter-agent messages cut off around four kilobytes without saying so, which cost
two round trips in an earlier session.

The rule's first use was writing this pull request. The agent drafting it hit the
truncation limit reporting its first milestone, wrote the draft to a file, and
sent a pointer with a one-line summary; the report arrived whole on the first
attempt. The durability clause comes from the same exchange. The briefs it was
working from lived in the orchestrator's own session scratchpad, so had the
orchestrator been killed, the pointer would have outlived the file it pointed at,
which is the failure the rule exists to prevent.

## 5. The stash that was not yours, behind "Worktree And Branch Discipline"

An agent ran `git stash push` with a pathspec that matched nothing. The command
succeeded and stashed nothing. The paired `git stash pop` therefore took the top
of the stack, which was a peer session's uncommitted draft: seven files and 2,513
insertions, applied into a tree that had never seen them. It was noticed in
thirteen seconds and recovered, which is luck rather than a control. The stack on
this clone holds ten entries from six branches going back to May, so the next
occurrence has ten different wrong answers available to it.

The second half of the rule has its own evidence. One session wrote four times to
`.git/info/exclude` in the primary checkout, once immediately after correctly
working out that a worktree's `.git` is a file and that the directory it points at
is shared. Knowing the topology did not change the behaviour, so the contract
names the shared surfaces explicitly rather than leaving them to be inferred.

## 6. Pull requests that outran their oracle, behind "Pull Request Review Gate"

Same run. Pull requests landed at 3,744 insertions across 24 files, and one branch
reached 6,282 insertions across 88 files, about six times the figure in the
contract.

The practitioner's own verdict is why the figure moved rather than the practice.
The work "did not have a line threshold, and treating it as one would have been
the wrong test", and the decision to let one oversized pull request ride was "a
sunk-cost argument dressed up as a judgment". What rescued that decision was not
the line count at all: it was putting size inside the reviewing round's scope and
asking the reviewer to disagree with it.

The metric that reviewer surfaced is the one worth keeping. A pull request
carried 1,166 lines of tests that proved 7 of the 57 cases its claim covered.
That ratio, not the diff size, is what produced two consecutive findings about
claims outrunning their oracle, and it is measurable before a reviewer ever sees
the branch.

## 7. Nine hours of asserted free targets, behind "Fresh-Context Enforcement"

Same run. For about nine hours every round brief asserted that the reviewer's target
was free, and every one of those assertions was produced the same way: the author
transcribed its own last report of the target. That is recollection with the timestamp
removed, and it cannot see a build another agent started in between. Five collisions
followed.

The worst of them contaminated a gate run. A reviewer inserted a type variant into
tracked source to prove that a guard fires, which is exactly the probe the round called
for. The author's `--local` gate compiled the tree while that variant was in it and
failed with three errors unrelated to anything either agent was working on. Nothing in
the failure pointed at the cause, and the shape of it, several unrelated errors at once,
is the shape the contract elsewhere teaches you to read as contention rather than as
breakage.

The fix existed before the collisions stopped. The advisory gate lease answers "is a
gate running here" authoritatively, kernel-backed, and it shipped the same day. It was
first consulted at 05:28:37Z, and the collision class closed at that point. The nine
hours were not spent waiting for a mechanism; they were spent not asking a question that
already had an answer. Hence the rule pastes the command's output rather than the
author's belief about it.

The separate-worktree exception has the same source. Sequencing a mutating probe against
a neighbouring build narrows the window in which one agent's inserted variant reaches the
other's compile. It does not remove the window, and the collision above happened inside
one.

## 8. A gate run that described a head nobody merged, behind "Pull Request Review Gate"

The rule this replaces said the `--local` gate runs at most once per pull request, on
the committed candidate, immediately before ready-for-review. Both halves cannot hold at
once as soon as a review round changes the candidate, and the run's first pull request
showed it: the `--local` evidence described head `7ea68247`, the round-2 repairs landed
after that run, and `45b8d640` is what went ready-for-review. The evidence and the
candidate were never the same commit. Later in the same run a merge shipped with
`--local` evidence describing a head one commit behind the merged one.

Neither case was a shortcut. Both are what the rule asks for when read literally, which
is why the fix is to state the obligation as coverage of the shipped head rather than as
a count of runs.

The second rule is about the invocation rather than the cadence. The three `--local`
runs that completed took 7m51s, 9m56s and 10m02s, against a ten-minute foreground
command limit. The documented invocation therefore works by luck, and it stopped working
inside the same run: one pull request spent three attempts to obtain one result, wasting
two whole runs of the machine's time. A run that takes nearly the whole budget on a warm
target has no margin at all on a cold one, so the detached form is the default rather
than the fallback.

A history rewrite silently invalidates every evidence claim already recorded on the
branch. The pull request that added this section demonstrated that while describing it:
its conformance evidence named a commit a later autosquash had removed, so those tests
had never run on any ancestor of the candidate, and the diff between the two heads
touched exactly the files they read. What hid it is that the hosted checks did not look
wrong. The conformance job skipped while its workflow still reported success, so the
page showed green for tests that never ran. Read the job, not the workflow.
