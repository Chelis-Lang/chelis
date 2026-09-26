import IteratorCombinators

open Aeneas Aeneas.Std

namespace IteratorCombinatorsChecks

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def item (x : Option Usize) : Option Nat := x.map (·.val)
def pair (x : Option (Usize × Usize)) : Option (Nat × Nat) :=
  x.map fun (a, b) => (a.val, b.val)

-- This source has a genuine, nonidentity restoration lens. The constructor
-- must compose it with the final right-hand state, not save the original source.
def natIter : core.iter.traits.iterator.Iterator Nat Nat := {
  next := fun n => if n < 3 then .ok (some n, n + 1) else .ok (none, n + 1)
}

def sourceInto : core.iter.traits.collect.IntoIterator (Nat × Nat) Nat Nat := {
  iteratorInst := natIter
  into_iter := fun (start, marker) => .ok (start, fun final => (final, marker + 1))
}

def chainRestoration : Result (Nat × (Nat × Nat)) := do
  let (state, left, right) ← core.iter.traits.iterator.Iterator.chain.default
    natIter sourceInto 3 (0, 10)
  let (_, state) ← core.iter.adapters.chain.IteratorChain.next natIter natIter state
  .ok (left state, right state)

def zipRestoration : Result (Nat × (Nat × Nat)) := do
  let (state, left, right) ← core.iter.traits.iterator.Iterator.zip.default
    natIter sourceInto 0 (2, 10)
  let (_, state) ← core.iter.adapters.zip.IteratorZip.next natIter natIter state
  let (_, state) ← core.iter.adapters.zip.IteratorZip.next natIter natIter state
  .ok (left state, right state)

set_option synthInstance.maxSize 512 in
#eval do
  expect "chain fuses only its left input and gives back terminal state"
    (decode (fun ((a, b, c, d, l, r) : Option Usize × Option Usize × Option Usize × Option Usize × Usize × Usize) =>
      (item a, item b, item c, item d, l.val, r.val))
      iterator_combinators.chain_nonfused)
    (.ok (some 0, none, some 2, some 3, 1, 4))
  expect "chain any threads predicate state and resumes after matching right item"
    (decode (fun ((found, calls, next, l, r) : Bool × Usize × Option Usize × Usize × Usize) =>
      (found, calls.val, item next, l.val, r.val))
      iterator_combinators.chain_any_resume)
    (.ok (true, 3, some 2, 2, 3))
  expect "chain count selects component overrides, excluding fused left"
    (decode (fun ((a, b) : Usize × Usize) => (a.val, b.val)) iterator_combinators.chain_counts)
    (.ok (305, 202))
  expect "chain nth carries the residual index into right nth override"
    (decode (fun ((next, afterNext, l, r, calls, n) : Option Usize × Option Usize × Usize × Usize × Usize × Usize) =>
      (item next, item afterNext, l.val, r.val, calls.val, n.val))
      iterator_combinators.chain_nth_crosses_left)
    (.ok (some 1, some 2, 3, 3, 1, 1))
  expect "chain any dispatches nested Skip try_fold rather than next"
    (decode (fun ((found, next, calls, n) : Bool × Option Usize × Usize × Usize) =>
      (found, item next, calls.val, n.val)) iterator_combinators.chain_skip_try_fold)
    (.ok (true, some 4, 1, 1))
  expect "zip right exhaustion consumes the preceding left item"
    (decode (fun ((a, b, next, l, r) : Option (Usize × Usize) × Option (Usize × Usize) × Option Usize × Usize × Usize) =>
      (pair a, pair b, item next, l.val, r.val))
      iterator_combinators.zip_partial_consumption)
    (.ok (some (0, 0), none, some 2, 3, 2))
  expect "zip left None neither advances right nor fuses future calls"
    (decode (fun ((a, b, l, r) : Option (Usize × Usize) × Option (Usize × Usize) × Usize × Usize) =>
      (pair a, pair b, l.val, r.val))
      iterator_combinators.zip_left_before_right)
    (.ok (none, some (1, 0), 2, 1))
  expect "zip count ignores input count overrides"
    (decode UScalar.val iterator_combinators.zip_count) (.ok 1)
  expect "trusted zip count avoids an extra left callback at right exhaustion"
    (decode (fun ((count, l, r) : Usize × Usize × Usize) => (count.val, l.val, r.val))
      iterator_combinators.zip_trusted_count) (.ok (2, 2, 2))
  expect "exact size hints do not grant Zip a TrustedLen implementation"
    (decode (fun ((count, l, r) : Usize × Usize × Usize) => (count.val, l.val, r.val))
      iterator_combinators.zip_exact_untrusted_count) (.ok (2, 3, 3))
  expect "random-access zip nth preserves skipped callbacks but does not poll past the end"
    (decode (fun ((first, third, last, l, r) : Option (Usize × Usize) × Option (Usize × Usize) × Option (Usize × Usize) × Usize × Usize) =>
      (pair first, pair third, pair last, l.val, r.val))
      iterator_combinators.zip_random_access_resume)
    (.ok (some (0, 10), some (2, 12), none, 3, 3))
  expect "Skip random access visits its side-effecting prefix exactly once"
    (decode (fun ((first, second, last, l, r) : Option (Usize × Usize) × Option (Usize × Usize) × Option (Usize × Usize) × Usize × Usize) =>
      (pair first, pair second, pair last, l.val, r.val))
      iterator_combinators.zip_skip_random_access)
    (.ok (some (2, 0), some (3, 1), none, 4, 2))
  expect "random-access zip count starts at the current offset"
    (decode (fun ((count, l, r) : Usize × Usize × Usize) => (count.val, l.val, r.val))
      iterator_combinators.zip_count_after_next) (.ok (2, 3, 3))
  expect "TrustedLen without random access preserves the unconsumed left suffix"
    (decode (fun ((count, next, l, r) : Usize × Option Usize × Usize × Usize) =>
      (count.val, item next, l.val, r.val)) iterator_combinators.zip_borrowed_trusted_count)
    (.ok (2, some 2, 3, 2))
  expect "zip any consumes only through the matching pair"
    (decode (fun ((found, calls, next, l, r) : Bool × Usize × Option (Usize × Usize) × Usize × Usize) =>
      (found, calls.val, pair next, l.val, r.val))
      iterator_combinators.zip_any_resume)
    (.ok (true, 2, some (2, 2), 3, 3))
  expect "skip clears its pending count even when nth returns None"
    (decode (fun ((a, b, pos, calls, n) : Option Usize × Option Usize × Usize × Usize × Usize) =>
      (item a, item b, pos.val, calls.val, n.val))
      iterator_combinators.skip_nonfused)
    (.ok (none, some 2, 3, 1, 3))
  expect "skip any calls nth(n-1), threads predicate state, and resumes"
    (decode (fun ((found, calls, next, pos, nthCalls, n) : Bool × Usize × Option Usize × Usize × Usize × Usize) =>
      (found, calls.val, item next, pos.val, nthCalls.val, n.val))
      iterator_combinators.skip_any_resume)
    (.ok (true, 2, some 4, 5, 1, 1))
  expect "skip any encountering None during initial skip invokes no predicate"
    (decode (fun ((found, calls, next, nthCalls, n) : Bool × Usize × Option Usize × Usize × Usize) =>
      (found, calls.val, item next, nthCalls.val, n.val))
      iterator_combinators.skip_any_gap)
    (.ok (false, 0, some 2, 1, 2))
  expect "skip count delegates only after successful initial skipping"
    (decode (fun ((a, b, c) : Usize × Usize × Usize) => (a.val, b.val, c.val)) iterator_combinators.skip_counts)
    (.ok (103, 0, 105))
  expect "skip nth overflow stops at first None without overflowing arithmetic"
    (decode (fun ((next, pos, calls, n) : Option Usize × Usize × Usize × Usize) =>
      (item next, pos.val, calls.val, n.val))
      iterator_combinators.skip_nth_overflow)
    (.ok (none, 1, 1, Usize.max - 1))
  expect "chain constructor restores through IntoIterator lens"
    (decode id chainRestoration) (.ok (4, (1, 11)))
  expect "zip constructor restores terminal right state through IntoIterator lens"
    (decode id zipRestoration) (.ok (2, (4, 11)))
  expect "random-access zip collects enumerated fields over the shorter length"
    (decode (fun (v : alloc.vec.Vec (Usize × U8 × U8)) =>
      v.val.map fun (x : Usize × U8 × U8) => (x.1.val, x.2.1.val, x.2.2.val))
      iterator_combinators.zip_collect_fields)
    (.ok [(0, 1, 7), (1, 2, 8)])
  expect "random-access zip collect runs the left callback on zipped positions only"
    (decode (fun ((v, calls) : alloc.vec.Vec (U8 × U8) × Usize) =>
      (v.val.map fun (x : U8 × U8) => (x.1.val, x.2.val), calls.val))
      iterator_combinators.zip_collect_mapped)
    (.ok ([(1, 7)], 1))
  IO.println "Iterator combinator smoke: 24 behavioral checks passed"

end IteratorCombinatorsChecks
