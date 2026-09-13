//! Lexical environment snapshots must not duplicate every function signature.
use chelis_types::{
    env::Env,
    types::{Scheme, Type},
};
use std::alloc::{GlobalAlloc, Layout, System};
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

#[test]
fn snapshots_bound_allocations_and_preserve_lexical_isolation_and_serialization() {
    let mut env = Env::new();
    // Wide signatures model a scalar-unrolled layer's neuron helpers.
    for index in 0..512 {
        let arguments = (0..256).map(|_| Type::Ref(Box::new(Type::Unit))).collect();
        env.bind(
            format!("neuron_{index}"),
            Scheme::mono(Type::Fn(arguments, Box::new(Type::Unit))),
        );
    }
    let serialized = serde_json::to_vec(&env).unwrap();
    let before = BYTES.load(Ordering::Relaxed);
    let mut inner = std::hint::black_box(env.clone());
    let allocated = BYTES.load(Ordering::Relaxed) - before;
    eprintln!("512-signature snapshot allocated {allocated} bytes");
    // Leave generous room for the name map and provenance, but not duplicated bodies.
    assert!(
        allocated < 512 * 1024,
        "snapshot copied signature bodies: {allocated} bytes"
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
