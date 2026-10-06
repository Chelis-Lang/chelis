# Verus axis-kernel proof of functionality

This experimental `verus` branch starts from Chelis `main` at
`2cabdbef5` (`fix(effects): preserve imported effects in inferred reports`).
Its purpose is to make one
small compiler-critical Rust kernel executable by Chelis **and** verifiable by
Verus, then measure the cost of doing so. It is discussion evidence, not a
compiler-soundness or phase-completion claim.

The follow-up merges current main before freezing its comparison baseline. The
[oracle experiment](axis_oracle_comparison.md) records the post-merge boundary,
bounded-checking and mutation design, and production rank-domain audit. PR #2851
remains draft for discussion; experiment success does not authorize adoption.

The crate, compiler call sites, proof runners, and receipts form one reviewable
slice: without any one of them, a passing proof would not establish that Chelis
uses the verified executable functions.

## Why this slice

The normative axis rules are in `spec/04-type-system.md` and
`spec/05-risc-primitives.md`: tensor dimensions are ordered positions, negative
reduction axes count from the end where specified, and `permute` is a complete
permutation of nonnegative positions. Equal extents do not make two positions
interchangeable. LaCaDiLE models ordered axes in Lean, and Hull exercises
bounded reference behavior, including reductions over repeated extents; neither
proves Chelis's Rust axis-handling implementation. This slice connects a formal
contract to code that the type checker and IR actually call.

The org-wide Verus survey found prior use only in the private
`Chelis-Lang/buoy` repository: eight Rust proof files (711 lines), a pinned
Verus release, a production proof binding, and comparison spikes alongside
Aeneas/Lean. There was no Chelis-core Verus gate before this branch.

## Proved code and compiler connection

`crates/chelis-axis-core/src/verified.rs` contains four executable functions
and their Verus contracts:

| Function | Proved contract |
| --- | --- |
| `is_permutation` | Returns true **iff** the input has rank length, every index lies in `0..rank`, and all indices are distinct. |
| `normalize_axis` | Equals a signed-index mathematical model; every accepted result is in range, including `i64::MIN` and oversized ranks. |
| `reduction_survivors` | Rejects an invalid axis; otherwise returns exactly `rank - 1` positions in their original order, omitting only the chosen axis. |
| `checked_inverse` | Rejects non-permutations; otherwise returns a full inverse mapping from original position to output position. |

There are no `assume` statements, `external_body` annotations, or unproved
bridge lemmas in this source. Normal Cargo compiles the same `verus!` executable
bodies that Verus verifies. The `Permutation` value has a private axis field;
its constructors admit only lists accepted by `is_permutation`. The checker
retains the existing `i32` axis-type gate and diagnostic categories. IR
lowering checks the plan before building a permute node, IR validation uses
the same predicate, AD obtains the inverse through the checked value, and
reduction lowering uses the ordered-survivor function.

The proof covers these axis algorithms. Rust callers' extraction of literal
axes, mapping of positions onto `DimInfo`, numeric values, backend execution,
and the complete compiler pipeline remain outside the formal proof. Ordinary
tests exercise those connections. The private-field invariant is established
by Rust visibility and constructor tests; Verus does not verify the entire
`Permutation` impl or its compiler callers.

## Reproduction and baggage

`.venv/bin/python crates/chelis-axis-core/proofs/verify_axis_verus.py` is the Verus
acceptance command. It pins
[Verus release 0.2026.09.27.3cf1832](https://github.com/verus-lang/verus/releases/tag/release%2F0.2026.09.27.3cf1832),
checks its archive SHA-256, selects Rust 1.98.1 without changing Chelis's
normal 1.98.0 pin, verifies the crate, and requires rejection of a deliberately
false postcondition in a temporary copy of the production source. On arm64
macOS the release archive is about 429 MiB and expands to about 1.4 GiB. The
ordinary Cargo lockfile gains `vstd` and six supporting packages. These costs
are a reason to keep this branch experimental rather than add the command to
the default Chelis CI gate.

The normal-build checks are `cargo test -p chelis-axis-core --test axis_contract`
and `cargo test -p chelis-types -p chelis-ir --lib`; the Surf checker test pairs
an ordered permutation with duplicate, negative, out-of-range, and wrong-arity
rejections. The axis-crate tests additionally exercise rank zero, `i64::MIN`,
oversized rank, inverse, and survivor order. The false-postcondition control
must fail **as a proof failure**; a parser or toolchain failure cannot satisfy
it.

## Vermilion comparison

`.venv/bin/python crates/chelis-axis-core/proofs/verify_axis_vermilion.py` pins
`ilyasergey/vermilion` at
`696756d6b9bbaedd61d6cd743ec3165f2b639224`, stages the **entire,
byte-identical** production `verified.rs`, and runs its pinned Verus frontend
and Lean backend. A fresh run generates 75 Lean obligations. Automation
discharges 61; `crates/chelis-axis-core/proofs/axis_vermilion_proofs.py`
supplies one helper lemma and 14 hand-written tactics for the rest: three
permutation, seven signed-axis, two survivor-order, and two inverse-update
obligations. It checks every manual obligation's statement hash before
installing its proof. A fresh run kernel-checked all 75 with zero `sorry`;
the shared Rust source SHA-256 was
`6ee585b811e93abc016d80c6baa322df19fa3f38086990cf338cbbc653c19091`.
Both verifiers check the four functional contracts and the supplied progress
assertions described below. Lean checks the complete proof twin; neither the runner nor its
helper introduces an assumption. The negative control extracts the unchanged
production permutation specification and function, changes the accepted-result
postcondition to false, and requires a Lean postcondition rejection rather
than a frontend or adapter failure.

The first translation exposed an unsupported `vec![0; rank]` expansion;
replacing that with an explicit fill loop in the shared Rust source let all
four functions translate. A later trial used `axis.rs` as its case filename;
that namespace collided with the local `axis` parameter in generated Lean.
Staging the same source as `verified.rs` removed eight spurious failures.
The remaining 14 were proved interactively, without weakening the Rust
contracts or changing their executable bodies.

The Vermilion trial additionally used about 8.8 GiB for its checkout and
Mathlib artifacts, 1.0 GiB for its Verus fork, and 2.7 GiB for Lean 4.33.0.
These measurements exclude shared Cargo caches. This is the principal
merge-readiness concern for making the Lean comparison a routine CI gate.

## What the approaches buy

Verus is a good fit for this kernel: finite-width integer guards, ordered
sequences, indexed updates and inductive loops all have useful `vstd` models.
The functional contracts say what admission, normalization, survivor order and
inverse positions mean; loop invariants connect the implementation to those
contracts. After the source was adapted, the SMT proof needed no manual tactic
library. Its guarantee covers the modeled machine input domain without choosing
a loop exploration bound. Contract correctness, the Rust-to-verification
translation and the solver remain trusted; a successful proof cannot repair an
incorrect contract or establish that a caller extracts the right axis.

Vermilion preserves those Rust contracts and annotations while generating Lean
obligations. The Lean kernel checks proof terms, including terms produced by
automation, and the manual proofs can compose with Lean mathematics. That is a
different assurance boundary, not an independent implementation of the Rust
frontend: both approaches rely on Verus's elaboration and obligation placement.
Vermilion adds its SST lowering, embedding and reporting to the trusted path.
Its [trust document](https://github.com/ilyasergey/vermilion/blob/696756d6b9bbaedd61d6cd743ec3165f2b639224/docs/trust.md)
and [VC policy](https://github.com/ilyasergey/vermilion/blob/696756d6b9bbaedd61d6cd743ec3165f2b639224/docs/vcgen.md)
describe that distinction. A kernel-checked twin is valuable evidence about the
translated obligations; translation soundness remains a separate obligation.

Termination needs inspection of the emitted obligations as well as the policy.
The pinned VC policy describes its planned whole-function soundness result as
partial correctness with termination side conditions imported from Verus. In
this actual kernel, however, the `--no-verify` frontend exports five decreases
checks as ordinary assertions. Lean checks all five among the 75 obligations.
A focused zero-increment mutant rejected `verified.is_permutation.assert_0_decreases`
while the other 74 obligations closed. It is therefore incorrect to say that
Lean ignores the supplied loop progress checks here. These local arithmetic
proofs still depend on trusted lowering and VC placement; they are not a
kernel-checked soundness theorem relating the complete Rust execution to its VCs.
The campaign keeps liveness defects without a finite independent violation
witness separate from its functional detection score.

The hard part here was proof engineering and tool integration, rather than the
axis mathematics. Vermilion required an explicit fill loop instead of a `vec!`
expansion, a stable generated namespace, and 14 manual proofs for integer casts,
permutation facts, survivor order and inverse updates. Statement hashes protect
against silently retaining a proof of an old theorem. They also make regeneration
and proof maintenance visible costs. The Linux reproduction exposed executable
permissions lost during Verus ZIP extraction and upstream target-path assumptions
broken by an inherited `CARGO_TARGET_DIR`; both runners now handle those cases.
The shared Rust source is 169 lines; the maintained Lean tactic installer is 272
lines and the generated baseline proof twin is 3,551 lines. These are artifact
and maintenance observations, not a measurement of authoring hours. Verus still
needs human-written contracts and inductive invariants despite its automated
discharge. The equivalent inverse-loop mutant demonstrates why annotation
maintenance and behavioral correctness need separate evidence.

Both approaches erase proof specifications from ordinary Rust execution. That
does not make adapting the executable algorithm free: this permutation predicate
uses pairwise comparison, with quadratic work on an accepted rank, while the
previous checker used a linear seen-position vector. The constructor still does
its linear diagnostic validation before calling the predicate. The experiment's
release microbenchmark separates proof erasure from this algorithmic tradeoff;
isolated stripped binaries cannot establish the size delta of the whole compiler.

This pattern should transfer most directly to small deterministic kernels with
explicit inputs, integer metadata, sequences and local mutation. Extending it to
compiler-wide invariants requires contracts at the caller and representation
boundaries, not just additional leaf proofs. Floating-point numerical agreement,
foreign code, GPU execution, aliasing across a large IR and external effects add
modeling work that this slice has not exercised. Verus's automated inductive proof
is the lighter engineering choice demonstrated here. Vermilion is most attractive
when auditable Lean terms or composition with existing Lean models justify its
translation, manual-proof and installation costs.

The completed [mutation comparison](axis_oracle_comparison.md#results) finds
36 test catches, 36 Kani catches (plus three timeouts and one pass), and 40
witness-qualified Verus catches among 40 finite functional defects. Tests and
Kani together catch 39. The remaining high-rank normalization defect lies outside
the production ABI and is caught by full-width Kani. The experiment therefore
shows inexpensive universal checking and better completion on some allocation
models, without a production-domain functional catch requiring unbounded proof
in this inventory. The two equivalent mutants and five liveness defects remain
separate from that score. Vermilion adds no catch over Verus on its 10 selected
mutants; the Lean assurance boundary remains its distinct contribution.

The Linux [cost measurements](axis_oracle_comparison.md#measured-costs) make the
tradeoff concrete: direct Verus verification is 0.67 seconds warm, compared with
266.65 seconds and about 20 GiB RSS for four Kani harnesses at common rank 5.
Vermilion's completed twin is 27.94 seconds warm, with about 2.72 GiB RSS and
14.43 GiB of active tooling. Those figures exclude installation from check times
and cover different assurance boundaries. The ordinary release crate build took
6.67 seconds; the runtime experiment attributes the large-rank slowdown to the
quadratic permutation algorithm rather than proof annotations. Neither the local
microbenchmark nor isolated binary sizes establish whole-compiler overhead.
