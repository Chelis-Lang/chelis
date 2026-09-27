import SliceSearchModels

open Aeneas Aeneas.Std

namespace SliceSearchModelsChecks

/-! Mirrors the native tests of `slice_search_models.rs`. -/

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "panic"
  | .div => .error "divergence"

def nat {ty : UScalarTy} (x : UScalar ty) : Nat := x.val
def int {ty : IScalarTy} (x : IScalar ty) : Int := x.val

def u8 (n : Nat) : U8 := ⟨BitVec.ofNat _ n⟩
def u32 (n : Nat) : U32 := ⟨BitVec.ofNat _ n⟩
def u64 (n : Nat) : U64 := ⟨BitVec.ofNat _ n⟩
def us (n : Nat) : Usize := ⟨BitVec.ofNat _ n⟩
def i32 (n : Int) : I32 := ⟨BitVec.ofInt _ n⟩
def i64 (n : Int) : I64 := ⟨BitVec.ofInt _ n⟩

def slice {α : Type} (l : List α) : Slice α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def vec {α : Type} (l : List α) : alloc.vec.Vec α :=
  if h : l.length ≤ Usize.max then ⟨l, h⟩ else ⟨[], by simp⟩

def vals {α : Type} (f : α → β) (l : { l : List α // l.length ≤ Usize.max }) : List β := l.val.map f

def res : core.result.Result Usize Usize → Except Nat Nat
  | .Ok i => .ok i.val
  | .Err i => .error i.val

open slice_search_models

#eval do
  let pr := fun ((ps, rest) : alloc.vec.Vec (U32 × U32) × alloc.vec.Vec U32) =>
    (vals (fun ((a, b) : U32 × U32) => (a.val, b.val)) ps, vals nat rest)
  expect "pairs 5" (decode pr (pairs (slice ([1, 2, 3, 4, 5].map u32)))) (.ok ([(1, 2), (3, 4)], [5]))
  expect "pairs 2" (decode pr (pairs (slice ([1, 2].map u32)))) (.ok ([(1, 2)], []))
  expect "pairs 0" (decode pr (pairs (slice []))) (.ok ([], []))
  expect "pairs 1" (decode pr (pairs (slice [u32 7]))) (.ok ([], [7]))
  expect "pair_seconds" (decode nat (pair_seconds (slice ([1, 10, 2, 20, 3].map u64)))) (.ok 30)
  let two := fun ((a, b) : Usize × Usize) => (a.val, b.val)
  expect "triples 7" (decode two (triples (slice (List.replicate 7 (u8 0))))) (.ok (2, 1))
  expect "triples 6" (decode two (triples (slice (List.replicate 6 (u8 0))))) (.ok (2, 0))
  expect "triples 0" (decode two (triples (slice []))) (.ok (0, 0))

#eval do
  let adjacency := slice [vec [us 0, us 2], vec [us 0, us 2], vec []]
  expect "self edge 0" (decode id (has_self_edge adjacency (us 0))) (.ok true)
  expect "self edge 1" (decode id (has_self_edge adjacency (us 1))) (.ok false)
  expect "self edge 2" (decode id (has_self_edge adjacency (us 2))) (.ok false)
  let s := slice ([0, 1, 1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55].map i32)
  expect "search 13" (decode res (search s (i32 13))) (.ok (.ok 9))
  expect "search 4" (decode res (search s (i32 4))) (.ok (.error 7))
  expect "search 100" (decode res (search s (i32 100))) (.ok (.error 13))
  expect "search -1" (decode res (search s (i32 (-1)))) (.ok (.error 0))
  expect "search 1" (decode res (search s (i32 1))) (.ok (.ok 4))
  expect "search empty" (decode res (search (slice []) (i32 1))) (.ok (.error 0))
  expect "search 1111" (decode res (search (slice ([1, 1, 1, 1].map i32)) (i32 1))) (.ok (.ok 3))
  expect "search 11111" (decode res (search (slice ([1, 1, 1, 1, 1].map i32)) (i32 1))) (.ok (.ok 4))
  expect "search 12" (decode res (search (slice ([1, 2].map i32)) (i32 1))) (.ok (.ok 0))

#eval do
  let levels := slice ([1, 3, 3, 7].map u32)
  expect "transition 3" (decode nat (transition_index levels (u32 3))) (.ok 3)
  expect "transition 0" (decode nat (transition_index levels (u32 0))) (.ok 0)
  expect "transition 9" (decode nat (transition_index levels (u32 9))) (.ok 4)
  expect "transition empty" (decode nat (transition_index (slice []) (u32 9))) (.ok 0)
  let two := fun ((i, calls) : Usize × U32) => (i.val, calls.val)
  expect "counted" (decode two (counted_partition (slice ([1, 2, 3, 4, 5, 6, 7, 8].map u32)) (u32 6) (u32 5)))
    (.ok (5, 9))
  expect "counted empty" (decode two (counted_partition (slice []) (u32 6) (u32 0))) (.ok (0, 0))

#eval do
  expect "wildcard" (decode (vals int) (wildcard (slice ([3, 4, 5].map i64)))) (.ok [-1, -1, -1])
  expect "wildcard empty" (decode (vals int) (wildcard (slice []))) (.ok [])
  let tagged := fun (t : Tagged) => (t.id.val, t.clones.val)
  let items : alloc.vec.Vec Tagged := vec [{ id := u32 1, clones := u32 9 },
    { id := u32 2, clones := u32 9 }, { id := u32 3, clones := u32 9 }]
  expect "fill tagged" (decode (vals tagged) (fill_tagged items (u32 7))) (.ok [(7, 1), (7, 1), (7, 0)])
  expect "fill none" (decode (vals tagged) (fill_tagged (vec []) (u32 7))) (.ok [])

end SliceSearchModelsChecks
