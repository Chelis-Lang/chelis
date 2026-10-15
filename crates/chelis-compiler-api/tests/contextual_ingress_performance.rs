//! Bounded public-ingress allocation/scaling witnesses for chelis#3231.
use chelis_compiler_api::schema::{WIRE_DAG_SCHEMA_VERSION, WireDag};
use chelis_types::{scalar_from_f64, types::Prim};
use serde_json::{Value, json};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
struct AllocationWork;
static ENABLED: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
fn allocated(size: usize) {
    let live = LIVE.fetch_add(size, Relaxed) + size;
    if ENABLED.load(Relaxed) {
        BYTES.fetch_add(size, Relaxed);
        CALLS.fetch_add(1, Relaxed);
        PEAK.fetch_max(live, Relaxed);
    }
}
unsafe impl GlobalAlloc for AllocationWork {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            allocated(layout.size());
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let next = unsafe { System.realloc(ptr, layout, size) };
        if !next.is_null() {
            LIVE.fetch_sub(layout.size(), Relaxed);
            allocated(size);
        }
        next
    }
}
#[global_allocator]
static ALLOCATOR: AllocationWork = AllocationWork;
fn node(id: usize, op: Value, inputs: Vec<usize>, deps: Vec<usize>, precision: &str) -> Value {
    json!({"id":id,"declaration":0,"activation":null,"shape_deps":deps,"span_id":null,"merged_spans":[],"inputs":inputs,"op":op,"output_type":{"dims":[{"kind":"named","name":"*","size":null}],"precision":precision}})
}
fn payload(n: usize, changing: bool, witnesses: bool, alternating: bool) -> String {
    let loads = if alternating {
        2
    } else if changing {
        n
    } else {
        1
    };
    let mut nodes = (0..loads)
        .map(|id| {
            node(
                id,
                json!({"kind":"load","name":format!("x{id}")}),
                vec![],
                vec![],
                "f32",
            )
        })
        .collect::<Vec<_>>();
    nodes.push(node(nodes.len(),json!({"kind":"const","value":scalar_from_f64("contextual ingress fixture",Prim::F32,0.0).unwrap()}),vec![],(0..loads).collect(),"f32"));
    for _ in 0..n {
        let id = nodes.len();
        nodes.push(node(
            id,
            json!({"kind":"add"}),
            vec![id - 1, id - 1],
            vec![],
            "f32",
        ));
    }
    let prefix = nodes.len() - 1;
    let mut first_witness = None;
    let mut roots = Vec::new();
    for i in 0..n {
        let source = if alternating {
            i % 2
        } else if changing {
            i
        } else {
            0
        };
        if witnesses {
            let claims = if first_witness.is_some() {
                vec![json!({"claim":"n","requirement_declares":true})]
            } else {
                vec![]
            };
            let mut inputs = vec![source];
            if let Some(first) = first_witness {
                inputs.push(first);
            }
            let id = nodes.len();
            let mut witness = node(
                id,
                json!({"kind":"extent_witness","site":"caller","parameter":format!("x{source}"),"axis":{"axis":"lit","value":0},"requirements":[],"claims":claims}),
                inputs,
                vec![],
                "int64",
            );
            witness["output_type"]["dims"] = json!([]);
            nodes.push(witness);
            first_witness.get_or_insert(id);
        }
        let id = nodes.len();
        nodes.push(node(
            id,
            json!({"kind":"compare","comparison":"eq"}),
            vec![prefix, source],
            vec![prefix],
            "bool",
        ));
        roots.push(id);
        if witnesses {
            // Force actual witness connectivity, rather than merely changing
            // the cutoff while both contextual origins already agree.
            let id = nodes.len();
            nodes.push(node(
                id,
                json!({"kind":"compare","comparison":"eq"}),
                vec![0, source],
                vec![0],
                "bool",
            ));
            roots.push(id);
        }
    }
    json!({"schema_version":WIRE_DAG_SCHEMA_VERSION,"declarations":["entry"],"nodes":nodes,"roots":roots}).to_string()
}
#[test]
fn contextual_public_ingress_allocation_and_context_scaling() {
    for (changing, witnesses, alternating) in [
        (false, false, false),
        (true, false, false),
        (true, true, false),
        (false, false, true),
    ] {
        for n in [32, 64, 128] {
            let json = payload(n, changing, witnesses, alternating);
            let before = LIVE.load(Relaxed);
            PEAK.store(before, Relaxed);
            BYTES.store(0, Relaxed);
            CALLS.store(0, Relaxed);
            ENABLED.store(true, Relaxed);
            let start = Instant::now();
            let decoded = WireDag::from_validated_json(&json);
            let wall = start.elapsed();
            ENABLED.store(false, Relaxed);
            let allocated = BYTES.load(Relaxed);
            let calls = CALLS.load(Relaxed);
            let peak = PEAK.load(Relaxed).saturating_sub(before);
            let decoded = decoded.expect("full public contextual graph validates");
            eprintln!(
                "contextual ingress changing={changing} witnesses={witnesses} alternating={alternating} n={n} nodes={} bytes={} allocated={allocated} allocation_calls={calls} peak_heap_delta={peak} wall={wall:?}",
                decoded.nodes.len(),
                json.len()
            );
            assert!(
                peak <= decoded.nodes.len() * 2048 + 64 * 1024,
                "context authority must not retain every prefix memo: {peak}"
            );
            if !changing && !witnesses {
                assert!(
                    allocated <= decoded.nodes.len() * 4096 + 64 * 1024,
                    "repeated contextual ingress allocates expanded prefix paths: {allocated}"
                );
            }
            let mut bad: Value = serde_json::from_str(&json).unwrap();
            bad["nodes"].as_array_mut().unwrap().last_mut().unwrap()["inputs"][1] = json!(u64::MAX);
            assert!(WireDag::from_validated_json(&bad.to_string()).is_err());
        }
    }
}
