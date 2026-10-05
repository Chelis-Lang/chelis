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

`python3 crates/chelis-axis-core/proofs/verify_axis_verus.py` is the Verus
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

`python3 crates/chelis-axis-core/proofs/verify_axis_vermilion.py` pins
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
The same four executable functions and contracts are now proved by both
verifiers. Lean checks the complete proof twin; neither the runner nor its
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
