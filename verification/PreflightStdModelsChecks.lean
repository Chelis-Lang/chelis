import PreflightStdModels

open Aeneas Aeneas.Std

def expect [BEq α] [Repr α] (label : String) (actual expected : α) : IO Unit := do
  unless actual == expected do
    throw (IO.userError s!"{label}: expected {repr expected}, got {repr actual}")

def decode {α β : Type} (f : α → β) : Result α → Except String β
  | .ok value => .ok (f value)
  | .fail _ => .error "failure"
  | .div => .error "divergence"

def sl {α : Type} (l : List α) (h : l.length ≤ Usize.max := by scalar_tac) : Slice α := ⟨l, h⟩
def vec {α : Type} (l : List α) (h : l.length ≤ Usize.max := by scalar_tac) :
    alloc.vec.Vec α := ⟨l, h⟩

def nats (l : List U32) : List Nat := l.map (·.val)
def optNat (o : Option U32) : Option Nat := o.map (·.val)
def deque (l : List U32) : alloc.collections.vec_deque.VecDeque U32 := ⟨l⟩
def dequeNats (q : alloc.collections.vec_deque.VecDeque U32) : List Nat := q.items.map (·.val)
def btree (l : List (U32 × U32)) : alloc.collections.btree.map.BTreeMap U32 U32 := ⟨l⟩
def mapPairs (m : alloc.collections.btree.map.BTreeMap U32 U32) : List (Nat × Nat) :=
  m.entries.map fun (k, v) => (k.val, v.val)
/-- A `usize` value on this (64-bit) platform. -/
def usizeOf (n : Nat) : Usize :=
  if h : n < 2 ^ UScalarTy.Usize.numBits then Usize.ofNatCore n h else core.num.Usize.MAX
def vecPairs (v : alloc.vec.Vec (U32 × U32)) : List (Nat × Nat) := v.val.map fun (k, v) => (k.val, v.val)

set_option synthInstance.maxSize 512 in
#eval show IO Unit from do
  -- VecDeque
  expect "worklist FIFO with pushes while draining"
    (decode (fun ((o, p, e) : alloc.vec.Vec U32 × Option U32 × Bool) => (nats o.val, optNat p, e))
      (preflight_std_models.worklist (sl [3#u32, 25#u32, 4#u32]))) (.ok ([3, 25, 4, 2], none, true))
  expect "worklist empty"
    (decode (fun ((o, p, e) : alloc.vec.Vec U32 × Option U32 × Bool) => (nats o.val, optNat p, e))
      (preflight_std_models.worklist (sl []))) (.ok ([], none, true))
  expect "worklist chain"
    (decode (fun ((o, p, e) : alloc.vec.Vec U32 × Option U32 × Bool) => (nats o.val, optNat p, e))
      (preflight_std_models.worklist (sl [120#u32]))) (.ok ([120, 12, 1], none, true))
  expect "deque ends, From<[T; N]>, clear, Default"
    (decode (fun ((b, f, l, n, m, e) : Option U32 × Option U32 × Option U32 × Usize × Usize × Bool) =>
        (optNat b, optNat f, optNat l, n.val, m.val, e))
      (preflight_std_models.deque_ends 5#u32 6#u32)) (.ok (some 6, some 0, some 5, 2, 0, true))
  expect "drain_ends (values, deque given back)"
    (decode (fun (((b, f), q) : (Option U32 × Option U32) × _) => (optNat b, optNat f, dequeNats q))
      (preflight_std_models.drain_ends (deque [1#u32, 2#u32, 3#u32]))) (.ok (some 3, some 1, [7, 2]))
  expect "drain_ends single (deque given back)"
    (decode (fun (((b, f), q) : (Option U32 × Option U32) × _) => (optNat b, optNat f, dequeNats q))
      (preflight_std_models.drain_ends (deque [4#u32]))) (.ok (some 4, none, [7]))
  expect "drain_ends empty (deque given back)"
    (decode (fun (((b, f), q) : (Option U32 × Option U32) × _) => (optNat b, optNat f, dequeNats q))
      (preflight_std_models.drain_ends (deque []))) (.ok (none, none, [7]))
  -- BTreeMap::get_mut
  expect "get_mut found (map given back)"
    (decode (fun ((found, m) : Bool × _) => (found, mapPairs m))
      (preflight_std_models.bump (btree [(1#u32, 10#u32), (3#u32, 30#u32), (5#u32, 50#u32)]) 3#u32))
    (.ok (true, [(1, 10), (3, 130), (5, 50)]))
  expect "get_mut missing (map given back unchanged)"
    (decode (fun ((found, m) : Bool × _) => (found, mapPairs m))
      (preflight_std_models.bump (btree [(1#u32, 10#u32), (3#u32, 130#u32), (5#u32, 50#u32)]) 4#u32))
    (.ok (false, [(1, 10), (3, 130), (5, 50)]))
  expect "get_mut on a collected map"
    (decode (fun ((found, v) : Bool × alloc.vec.Vec (U32 × U32)) => (found, vecPairs v))
      (preflight_std_models.bump_entries (sl [(2#u32, 1#u32), (1#u32, 7#u32)]) 1#u32))
    (.ok (true, [(1, 107), (2, 1)]))
  expect "get_mut on an empty map"
    (decode (fun ((found, v) : Bool × alloc.vec.Vec (U32 × U32)) => (found, vecPairs v))
      (preflight_std_models.bump_entries (sl []) 1#u32)) (.ok (false, []))
  expect "get_mut write overflows"
    (decode (fun ((found, v) : Bool × alloc.vec.Vec (U32 × U32)) => (found, vecPairs v))
      (preflight_std_models.bump_entries (sl [(1#u32, core.num.U32.MAX)]) 1#u32)) (.error "failure")
  -- first_mut / last_mut
  expect "last_mut (slice given back)"
    (decode (fun ((b, s) : Bool × Slice U32) => (b, nats s.val))
      (preflight_std_models.bump_last (sl [1#u32, 2#u32, 3#u32]))) (.ok (true, [1, 2, 13]))
  expect "first_mut (slice given back)"
    (decode (fun ((b, s) : Bool × Slice U32) => (b, nats s.val))
      (preflight_std_models.bump_first (sl [1#u32, 2#u32, 13#u32]))) (.ok (true, [2, 2, 13]))
  expect "last_mut empty"
    (decode (fun ((b, s) : Bool × Slice U32) => (b, nats s.val))
      (preflight_std_models.bump_last (sl []))) (.ok (false, []))
  expect "first_mut empty"
    (decode (fun ((b, s) : Bool × Slice U32) => (b, nats s.val))
      (preflight_std_models.bump_first (sl []))) (.ok (false, []))
  expect "push into the innermost scope (scopes given back)"
    (decode (fun ((b, s) : Bool × alloc.vec.Vec (alloc.vec.Vec U32)) => (b, s.val.map (nats ·.val)))
      (preflight_std_models.push_innermost (vec [vec [1#u32], vec [2#u32]]) 9#u32))
    (.ok (true, [[1], [2, 9]]))
  expect "no scope (scopes given back)"
    (decode (fun ((b, s) : Bool × alloc.vec.Vec (alloc.vec.Vec U32)) => (b, s.val.map (nats ·.val)))
      (preflight_std_models.push_innermost (vec []) 9#u32)) (.ok (false, []))
  -- RangeFull
  expect "[..], get(..), get_mut(..) (vector given back)"
    (decode (fun (((n, g), v) : (Usize × Option Usize) × alloc.vec.Vec U32) =>
        (n.val, g.map (·.val), nats v.val))
      (preflight_std_models.full_range (vec [1#u32, 2#u32, 3#u32]))) (.ok (3, some 3, [2, 2, 6]))
  expect "full range of one"
    (decode (fun (((n, g), v) : (Usize × Option Usize) × alloc.vec.Vec U32) =>
        (n.val, g.map (·.val), nats v.val))
      (preflight_std_models.full_range (vec [5#u32]))) (.ok (1, some 1, [12]))
  expect "full range of none"
    (decode (fun (((n, g), v) : (Usize × Option Usize) × alloc.vec.Vec U32) =>
        (n.val, g.map (·.val), nats v.val))
      (preflight_std_models.full_range (vec []))) (.ok (0, some 0, []))
  -- String / str
  expect "with_capacity, push_str, push, len"
    (decode (fun ((s, n) : String × Usize) => (s, n.val))
      (preflight_std_models.build 4#usize (toStr "ab") 'c')) (.ok ("abc!", 4))
  expect "len counts UTF-8 bytes"
    (decode (fun ((s, n) : String × Usize) => (s, n.val))
      (preflight_std_models.build 0#usize (toStr "é") 'ß')) (.ok ("éß!", 5))
  expect "with_capacity(usize::MAX) panics"
    (decode (fun ((s, n) : String × Usize) => (s, n.val))
      (preflight_std_models.build core.num.Usize.MAX (toStr "") 'a')) (.error "failure")
  expect "with_capacity(isize::MAX + 1) panics"
    (decode (fun ((s, n) : String × Usize) => (s, n.val))
      (preflight_std_models.build (usizeOf 9223372036854775808) (toStr "") 'a')) (.error "failure")
  expect "append (string given back)"
    (decode (fun ((n, s) : Usize × String) => (n.val, s))
      (preflight_std_models.append "x" (toStr "yz"))) (.ok (4, "xyz."))
  expect "indent" (decode id (preflight_std_models.indent 3#usize (toStr "a"))) (.ok "   a")
  expect "indent 0" (decode id (preflight_std_models.indent 0#usize (toStr "b"))) (.ok "b")
  expect "repeat" (decode id (preflight_std_models.repeat_word (toStr "ab") 3#usize)) (.ok "ababab")
  expect "repeat multibyte" (decode id (preflight_std_models.repeat_word (toStr "é") 2#usize)) (.ok "éé")
  expect "repeat 0" (decode id (preflight_std_models.repeat_word (toStr "ab") 0#usize)) (.ok "")
  expect "repeat empty usize::MAX times"
    (decode id (preflight_std_models.repeat_word (toStr "") core.num.Usize.MAX)) (.ok "")
  expect "repeat overflowing usize panics"
    (decode id (preflight_std_models.repeat_word (toStr "ab") core.num.Usize.MAX)) (.error "failure")
  expect "repeat past isize::MAX bytes panics"
    (decode id (preflight_std_models.repeat_word (toStr "ab") (usizeOf 4611686018427387904)))
    (.error "failure")
  expect "contains" (decode id (preflight_std_models.contains_both (toStr "hello") (toStr "ll") 'z'))
    (.ok (true, false))
  expect "contains empty needle"
    (decode id (preflight_std_models.contains_both (toStr "") (toStr "") 'a')) (.ok (true, false))
  expect "contains multibyte"
    (decode id (preflight_std_models.contains_both (toStr "né") (toStr "é") 'é')) (.ok (true, true))
  expect "contains longer needle"
    (decode id (preflight_std_models.contains_both (toStr "abc") (toStr "abcd") 'c')) (.ok (false, true))
  expect "contains one byte"
    (decode id (preflight_std_models.contains_both (toStr "abc") (toStr "b") 'a')) (.ok (true, true))
  -- bool::then
  expect "then true" (decode id (preflight_std_models.then_label true (toStr "f"))) (.ok (some "f"))
  expect "then false" (decode id (preflight_std_models.then_label false (toStr "f"))) (.ok none)
  expect "then calls" (decode optNat (preflight_std_models.then_count true 1#u32)) (.ok (some 2))
  expect "then does not call"
    (decode optNat (preflight_std_models.then_count false core.num.U32.MAX)) (.ok none)
  expect "then overflow panics"
    (decode optNat (preflight_std_models.then_count true core.num.U32.MAX)) (.error "failure")
  -- Iterator::sum
  expect "sum of a map" (decode (·.val) (preflight_std_models.total (sl [1#u32, 2#u32, 3#u32])))
    (.ok 12)
  expect "sum of nothing" (decode (·.val) (preflight_std_models.total (sl []))) (.ok 0)
  expect "sum up to usize::MAX"
    (decode (·.val) (preflight_std_models.total_sizes (sl [core.num.Usize.MAX, 0#usize])))
    (.ok Usize.max)
  expect "sum overflow panics"
    (decode (·.val) (preflight_std_models.total_sizes (sl [core.num.Usize.MAX, 1#usize])))
    (.error "failure")
  -- AsRef for Box
  expect "Box as_ref" (decode (fun ((a, n) : U32 × Usize) => (a.val, n.val))
    (preflight_std_models.unbox (4#u32, "abc"))) (.ok (5, 3))
  IO.println "preflight std models: 49 checks passed"
