//! Lexical environment snapshots must not duplicate every function signature,
//! nor copy the name map itself: a snapshot's cost does not depend on how
//! many names are bound (chelis#3750).
use chelis_types::{
    env::Env,
    types::{Scheme, Type},
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static BYTES: AtomicUsize = AtomicUsize::new(0);
// This integration binary has one test; the counter covers only its clone window.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        BYTES.fetch_add(size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// An environment binding `count` wide signatures, which model a
/// scalar-unrolled layer's neuron helpers.
fn wide_env(count: usize) -> Env {
    let mut env = Env::new();
    for index in 0..count {
        let arguments = (0..256).map(|_| Type::reference(Type::Unit)).collect();
        env.bind(
            format!("neuron_{index}"),
            Scheme::mono(Type::Fn(arguments, Arc::new(Type::Unit))),
        );
    }
    env
}

/// The bytes one snapshot of `env` allocates.
fn snapshot_bytes(env: &Env) -> (Env, usize) {
    let before = BYTES.load(Ordering::Relaxed);
    let snapshot = std::hint::black_box(env.clone());
    (snapshot, BYTES.load(Ordering::Relaxed) - before)
}

#[test]
fn snapshots_bound_allocations_and_preserve_lexical_isolation_and_serialization() {
    let mut env = wide_env(512);
    let serialized = serde_json::to_vec(&env).unwrap();
    let (mut inner, allocated) = snapshot_bytes(&env);
    eprintln!("512-signature snapshot allocated {allocated} bytes");
    // The snapshot shares the name map, so its cost is the same for every
    // environment size; one with a single binding allocates as much.
    let (_, single) = snapshot_bytes(&wide_env(1));
    assert_eq!(
        allocated, single,
        "a snapshot's cost grew with the bound names: {single} bytes for one binding, \
         {allocated} for 512"
    );
    assert_eq!(serde_json::to_vec(&inner).unwrap(), serialized);
    let decoded: Env = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(
        decoded.lookup("neuron_0").unwrap().body,
        env.lookup("neuron_0").unwrap().body
    );
    inner.bind("neuron_0".into(), Scheme::mono(Type::Unit));
    inner.bind("inner_only".into(), Scheme::mono(Type::Unit));
    assert_ne!(
        inner.lookup("neuron_0").unwrap().body,
        env.lookup("neuron_0").unwrap().body
    );
    assert!(env.lookup("inner_only").is_none());
    env.bind("outer_only".into(), Scheme::mono(Type::Unit));
    assert!(inner.lookup("outer_only").is_none());
}
