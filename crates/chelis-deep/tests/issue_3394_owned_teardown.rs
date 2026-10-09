//! Owned Deep trees release on a constrained worker stack, including
//! diagnostic carriers that are not validated vocabulary nodes.

use chelis_deep::annotations::{MacroSource, MetadataValue, Spanned, TypeSyntax};
use chelis_deep::{
    Atom, DeepTag, Expr, MetaExpr, Metadata, RawAtom, RawExpr, Span, UnknownFormData,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static LIVE_BYTES: Cell<isize> = const { Cell::new(0) };
}

struct CountLive;
unsafe impl GlobalAlloc for CountLive {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            count_bytes(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count_bytes(-(layout.size() as isize));
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            count_bytes(new_size as isize - layout.size() as isize);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountLive = CountLive;

fn count_bytes(delta: isize) {
    let _ = TRACK_ALLOCATIONS.try_with(|tracking| {
        if tracking.get() {
            let _ = LIVE_BYTES.try_with(|bytes| bytes.set(bytes.get() + delta));
        }
    });
}

fn tracked_live_bytes(f: impl FnOnce()) -> isize {
    LIVE_BYTES.with(|bytes| bytes.set(0));
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    f();
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    LIVE_BYTES.with(|bytes| bytes.get())
}

fn span() -> Span {
    Span::new(0, 0)
}

fn var(name: &str) -> Expr {
    Expr::node(
        DeepTag::Var,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name(name.into()), span())],
        span(),
    )
}

fn cons(head: Expr, tail: Expr) -> Expr {
    Expr::node(
        DeepTag::App,
        Metadata::default(),
        vec![var("Cons"), head, tail],
        span(),
    )
}

fn on_small_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(f)
        .expect("spawn constrained worker")
        .join()
        .expect("owned tree release must complete");
}

#[test]
fn canonical_cons_spine_releases_without_recursive_native_teardown() {
    on_small_stack(|| {
        let mut chain = var("Nil");
        for _ in 0..16_686 {
            chain = cons(var("head"), chain);
        }
        drop(chain);
    });
}

#[test]
fn mixed_non_node_and_diagnostic_carriers_release() {
    on_small_stack(|| {
        let mut tree = Expr::Atom(Atom::Bool(true), span());
        for index in 0..16_686 {
            tree = match index % 3 {
                0 => Expr::BareList(vec![tree], span()),
                1 => Expr::UnknownForm(Box::new(UnknownFormData {
                    head: "unknown".into(),
                    meta: Metadata::default(),
                    children: vec![tree],
                    span: span(),
                })),
                _ => Expr::MetaExpr(
                    MetaExpr {
                        metadata: Metadata::default(),
                        expr: Box::new(tree),
                    },
                    span(),
                ),
            };
        }
        drop(tree);
    });
}

#[test]
fn partial_unknown_form_with_an_improper_tail_releases() {
    on_small_stack(|| {
        let mut tail = Expr::UnknownForm(Box::new(UnknownFormData {
            head: "broken".into(),
            meta: Metadata::default(),
            children: Vec::new(),
            span: span(),
        }));
        for _ in 0..16_686 {
            tail = cons(var("head"), tail);
        }
        drop(tail);
    });
}

#[test]
fn expression_valued_annotations_release_even_when_storage_was_shared() {
    on_small_stack(|| {
        let mut current = Expr::node(
            DeepTag::TPrim,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("f32".into()), span())],
            span(),
        );
        for _ in 0..12_000 {
            let mut annotations = Metadata::default();
            annotations
                .insert(MetadataValue::Type(TypeSyntax::try_new(current).unwrap()))
                .unwrap();
            current = Expr::node(
                DeepTag::TPrim,
                annotations,
                vec![Expr::Atom(Atom::Name("f32".into()), span())],
                span(),
            );
        }
        let shared_annotations = match &current {
            Expr::Node(node, _) => node.meta().clone(),
            _ => unreachable!(),
        };
        drop(current);
        drop(shared_annotations);
    });
}

#[test]
fn historical_raw_source_annotation_releases_with_its_deep_expression() {
    on_small_stack(|| {
        let mut raw = RawExpr::Atom(RawAtom::Symbol("leaf".into()), span());
        for index in 0..16_686 {
            raw = match index % 3 {
                0 => RawExpr::List(vec![raw], span()),
                1 => RawExpr::Map(vec![("key".into(), raw)], span()),
                _ => RawExpr::MetaExpr {
                    entries: Vec::new(),
                    expr: Box::new(raw),
                    span: span(),
                },
            };
        }
        let source = MacroSource::new(Spanned::new("macro".into(), span()), vec![raw], span());
        let mut annotations = Metadata::default();
        annotations.insert(MetadataValue::Source(source)).unwrap();
        drop(Expr::Map(annotations, span()));
    });
}

#[test]
fn releasing_a_long_owned_tree_balances_its_heap_allocations() {
    on_small_stack(|| {
        let live = tracked_live_bytes(|| {
            let mut tree = var("Nil");
            for _ in 0..5_000 {
                tree = cons(var("head"), tree);
            }
            drop(tree);
        });
        assert_eq!(live, 0, "owned tree leaked {live} allocated bytes");
    });
}

#[test]
fn shared_annotation_storage_can_be_released_by_either_worker() {
    use std::sync::{Arc, Barrier};

    let mut tree = Expr::node(
        DeepTag::TPrim,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name("f32".into()), span())],
        span(),
    );
    for _ in 0..12_000 {
        let mut nested = Metadata::default();
        nested
            .insert(MetadataValue::Type(TypeSyntax::try_new(tree).unwrap()))
            .unwrap();
        tree = Expr::node(
            DeepTag::TPrim,
            nested,
            vec![Expr::Atom(Atom::Name("f32".into()), span())],
            span(),
        );
    }
    let mut annotations = Metadata::default();
    annotations
        .insert(MetadataValue::Type(TypeSyntax::try_new(tree).unwrap()))
        .unwrap();
    let other = annotations.clone();
    let barrier = Arc::new(Barrier::new(3));
    let workers: Vec<_> = [annotations, other]
        .into_iter()
        .map(|annotations| {
            let barrier = barrier.clone();
            std::thread::Builder::new()
                .stack_size(2 * 1024 * 1024)
                .spawn(move || {
                    barrier.wait();
                    drop(annotations);
                })
                .expect("spawn constrained worker")
        })
        .collect();
    barrier.wait();
    for worker in workers {
        worker.join().expect("shared storage release must complete");
    }
}

#[test]
fn owned_extraction_preserves_a_mismatched_carrier() {
    let original = Expr::BareList(vec![var("item")], span());
    let returned = original
        .into_node_parts()
        .expect_err("a bare list is not a stamped node");
    assert!(matches!(&returned, Expr::BareList(children, _) if children.len() == 1));
}
