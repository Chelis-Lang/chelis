use std::alloc::{GlobalAlloc, Layout, System};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use chelis_unord::UnordMap;

const TRACE_CAPACITY: usize = 4_096;
static RECORDING: AtomicBool = AtomicBool::new(false);
static TRACE_LEN: AtomicUsize = AtomicUsize::new(0);
static TRACE: [AtomicUsize; TRACE_CAPACITY] = [const { AtomicUsize::new(0) }; TRACE_CAPACITY];

struct RecordingAllocator;

// SAFETY: this allocator delegates every operation to `System` with the
// original pointer and layout. The only added behavior is lock-free recording
// of deallocation sizes into a fixed static buffer, which does not allocate.
unsafe impl GlobalAlloc for RecordingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated with the caller-provided layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if RECORDING.load(Ordering::Relaxed) {
            let index = TRACE_LEN.fetch_add(1, Ordering::Relaxed);
            if index < TRACE_CAPACITY {
                TRACE[index].store(layout.size(), Ordering::Relaxed);
            }
        }
        // SAFETY: delegated with the exact pointer and layout supplied by the
        // allocation protocol.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: RecordingAllocator = RecordingAllocator;

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
struct CollisionGroup {
    group: u8,
    member: u8,
}

impl Hash for CollisionGroup {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.group.hash(state);
    }
}

fn allocator_visible_drop_trace() -> Vec<usize> {
    let mut map = UnordMap::new();
    for group in 0..12_u8 {
        for member in 0..=group {
            map.insert(CollisionGroup { group, member }, ());
        }
    }

    TRACE_LEN.store(0, Ordering::Relaxed);
    RECORDING.store(true, Ordering::SeqCst);
    drop(map);
    RECORDING.store(false, Ordering::SeqCst);

    let length = TRACE_LEN.load(Ordering::Relaxed);
    assert!(
        length <= TRACE_CAPACITY,
        "drop emitted {length} allocator callbacks, exceeding the fixed trace capacity"
    );
    TRACE[..length]
        .iter()
        .map(|size| size.load(Ordering::Relaxed))
        .collect()
}

#[test]
fn raw_bucket_destruction_has_one_allocator_visible_order() {
    let reference = allocator_visible_drop_trace();
    assert!(
        !reference.is_empty(),
        "the negative control must observe drop"
    );
    for attempt in 1..64 {
        assert_eq!(
            allocator_visible_drop_trace(),
            reference,
            "randomized raw-bucket destruction escaped through allocator callbacks on attempt {attempt}"
        );
    }
}
