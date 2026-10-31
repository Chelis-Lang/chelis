fn host_checks() {
    let facts = exercise();
    assert!(!facts.annotated_exprs().is_empty());
    let resources = Resources::new(&CallerStack);
    let library = parse("(def {} shared_id (fn {} (params {} x) (var {} x)))");
    let (context, _) = chelis_checker_core::build_compiled_library_context(&library, &resources).unwrap();
    let extension = parse("(def {} use_shared (app {} (var {} shared_id) (lit {type: (t-prim {} i32)} 1)))");
    let expected = chelis_checker_core::check_ir_with_context(&context, &extension, &resources).unwrap().into_facts();
    std::thread::scope(|scope| {
        let handles = (0..8).map(|index| {
            let context = &context;
            let extension = &extension;
            scope.spawn(move || {
                let stack = CallerStack;
                let mut resources = Resources::new(&stack);
                let cancel = CancelAfter { polls: Cell::new(0), limit: 0 };
                if index == 0 {
                    resources.cancellation = Some(&cancel);
                    assert!(chelis_checker_core::check_ir_with_context(context, extension, &resources).is_err());
                    resources.cancellation = None;
                }
                chelis_checker_core::check_ir_with_context(context, extension, &resources).unwrap().into_facts()
            })
        }).collect::<Vec<_>>();
        for handle in handles {
            let actual = handle.join().expect("The independent check must finish.");
            assert_eq!(actual.annotated_exprs(), expected.annotated_exprs());
            assert_eq!(actual.type_env(), expected.type_env());
            assert_eq!(actual.local_ascriptions(), expected.local_ascriptions());
            assert_eq!(actual.infer_stats().typed_nodes, expected.infer_stats().typed_nodes);
            assert_eq!(actual.infer_stats().total_nodes, expected.infer_stats().total_nodes);
        }
    });
    let repeated = chelis_checker_core::check_ir_with_context(&context, &extension, &resources).unwrap();
    assert_eq!(repeated.facts().annotated_exprs(), expected.annotated_exprs());
    println!("CHECKER_CORE_CONSUMER accepted");
}

fn main() {
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(host_checks).unwrap()
        .join().expect("The independent consumer must finish.");
}
