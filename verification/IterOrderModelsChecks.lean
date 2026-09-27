import IterOrderModels

open Aeneas Aeneas.Std

namespace IterOrderModelsChecks

/-! Mirrors the native tests of `iter_order_models.rs`, including the counters given back
    through the captured borrows. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def nat {ty : UScalarTy} (x : UScalar ty) : Nat := x.val

def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩

def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def vals {α β : Type} (f : α → β) (l : { l : List α // l.length ≤ Usize.max }) : List β :=
  l.val.map f

def str (s : String) : Str := match Str.ofString s with | .ok v => v | _ => ⟨[], by simp⟩

def text (s : Str) : String := match Str.toString s with | .ok v => v | _ => "<invalid>"

open iter_order_models

def keyed (pairs : List (Nat × Nat)) : Slice Keyed :=
  slice (pairs.map fun (k, t) => { key := u32 k, tag := u32 t })

#eval do
  let pair := fun (o : Option (U32 × U32)) => o.map fun ((k, t) : U32 × U32) => (k.val, t.val)
  expect "representative" (decode pair (representative (keyed [(3, 0), (1, 1), (2, 2), (1, 3)])))
    (.ok (some (1, 1)))
  expect "representative 1" (decode pair (representative (keyed [(5, 9)]))) (.ok (some (5, 9)))
  expect "representative 0" (decode pair (representative (keyed []))) (.ok none)
  expect "least_index" (decode nat (least_index (slice ([2, 0, 1].map us)) (slice ([7, 9, 4].map us))))
    (.ok 4)
  expect "least_index empty" (decode nat (least_index (slice []) (slice ([7, 9, 4].map us))))
    (.ok (2 ^ 64 - 1))
  let counted := fun ((m, calls) : Option U32 × U32) => (m.map nat, calls.val)
  expect "counted_min" (decode counted (counted_min (slice ([20, 15, 9, 14].map u32)) (u32 2)))
    (.ok (some 0, 6))
  expect "counted_min empty" (decode counted (counted_min (slice []) (u32 0))) (.ok (none, 0))
  expect "earliest_even_half" (decode (Option.map nat)
    (earliest_even_half (slice ([9, 14, 3, 8, 12].map u32)))) (.ok (some 4))
  expect "earliest_even_half none" (decode (Option.map nat) (earliest_even_half (slice ([9, 3].map u32))))
    (.ok none)
  expect "counted_filter_min" (decode counted
    (counted_filter_min (slice ([9, 13, 6, 22, 7].map u32)) (u32 1))) (.ok (some 2, 6))

#eval do
  expect "identity" (decode id (is_identity (slice ([0, 1, 2, 3].map us)))) (.ok true)
  expect "not identity" (decode id (is_identity (slice ([0, 2, 1].map us)))) (.ok false)
  expect "identity empty" (decode id (is_identity (slice []))) (.ok true)
  expect "identity short" (decode id (is_identity (slice [us 1]))) (.ok false)
  let rows := fun (l : List (Nat × Nat)) => slice (l.map fun (a, v) => (us a, u32 v))
  expect "axes same" (decode id (axes_differ (rows [(0, 5), (2, 1)]) (rows [(0, 9), (2, 3)])))
    (.ok false)
  expect "axes differ" (decode id (axes_differ (rows [(0, 5), (2, 1)]) (rows [(0, 9), (1, 3)])))
    (.ok true)
  expect "axes length" (decode id (axes_differ (rows [(0, 5)]) (rows [(0, 9), (1, 3)]))) (.ok true)
  expect "axes empty" (decode id (axes_differ (rows []) (rows []))) (.ok false)
  let q := fun (a b : List Nat) => decode id (quotients_equal (slice (a.map u32)) (slice (b.map u32)))
  expect "quotients" (q [1, 2, 3] [60, 30, 20]) (.ok true)
  expect "quotients differ" (q [1, 2, 4] [60, 30, 20]) (.ok false)
  expect "quotients lengths" (q [0, 1] [60]) (.ok false)
  expect "quotients divide" (q [0, 1] [60, 60]) (.error "panic")
  let o := fun (a b : List Nat) =>
    decode id (quotients_differ_from_odd (slice (a.map u32)) (slice (b.map u32)))
  expect "odd same" (o [4, 12] [15, 2, 5]) (.ok false)
  expect "odd longer" (o [4, 12] [15, 2, 5, 7]) (.ok true)
  expect "odd shorter" (o [4] [15, 5]) (.ok true)

#eval do
  let segs := fun (l : List String) => slice (l.map str)
  let texts := fun (v : alloc.vec.Vec Str) => vals text v
  expect "module" (decode texts (module_segments (segs ["std", "io", "Error", "kind"])))
    (.ok ["Error", "kind"])
  expect "module empty segment" (decode texts (module_segments (segs ["", "Map", "x"])))
    (.ok ["Map", "x"])
  expect "module none" (decode texts (module_segments (segs ["a", "b"]))) (.ok [])
  let skipped := fun ((v, calls) : alloc.vec.Vec U32 × U32) => (vals nat v, calls.val)
  expect "counted_skip" (decode skipped (counted_skip (slice ([1, 2, 7, 1, 9].map u32)) (u32 5) (u32 3)))
    (.ok ([7, 1, 9], 6))
  expect "counted_skip all" (decode skipped (counted_skip (slice ([1, 2].map u32)) (u32 5) (u32 0)))
    (.ok ([], 2))
  let two := fun ((a, b) : Option U32 × Option U32) => (a.map nat, b.map nat)
  expect "first_two" (decode two (first_two_kept (slice ([1, 6, 2, 8].map u32)) (u32 5)))
    (.ok (some 6, some 2))
  expect "first_two one" (decode two (first_two_kept (slice ([1, 6].map u32)) (u32 5)))
    (.ok (some 6, none))
  expect "first_two none" (decode two (first_two_kept (slice ([1, 2].map u32)) (u32 5)))
    (.ok (none, none))

end IterOrderModelsChecks
