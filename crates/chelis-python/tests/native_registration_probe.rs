//! C6 joins the actual four native descriptors to compiled Rust class identity.
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[path = "../../../tests/support/pyo3_native_registration.rs"]
mod native_registration;

#[::pyo3::pyclass(name = "NativeTensor")]
struct ForeignTensor;

fn with_module(check: impl FnOnce(&Bound<'_, PyModule>)) {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_native_registration_control").unwrap();
        chelis_python::register_module(&module).unwrap();
        check(&module);
    });
}

#[test]
fn native_registration_binds_all_four_compiled_descriptors() {
    with_module(|module| {
        let report = native_registration::probe(module).unwrap();
        assert_eq!(report.registrations.len(), 4);
        assert_eq!(report.descriptors.len(), 4);
        assert_eq!(report.classes.len(), 2);
        assert_eq!(report.source_path, "crates/chelis-python/src/lib.rs");
        assert_eq!(report.source_sha256.len(), 64);
        assert_eq!(
            report
                .descriptors
                .iter()
                .map(|row| (
                    row.owner.as_str(),
                    row.python_name.as_str(),
                    row.kind.as_str(),
                    row.descriptor.as_str()
                ))
                .collect::<Vec<_>>(),
            vec![
                ("CompiledModel", "__call__", "method", "wrapper_descriptor"),
                ("NativeTensor", "__dlpack__", "method", "method_descriptor"),
                (
                    "NativeTensor",
                    "__dlpack_device__",
                    "method",
                    "method_descriptor"
                ),
                ("NativeTensor", "shape", "getter", "getset_descriptor"),
            ]
        );
    });
}

#[test]
fn native_registration_rejects_a_same_named_foreign_class() {
    with_module(|module| {
        module.add_class::<ForeignTensor>().unwrap();
        assert!(
            native_registration::probe(module)
                .unwrap_err()
                .contains("compiled class identity")
        );
    });
}

#[test]
fn native_registration_rejects_missing_or_swapped_classes() {
    for changed in ["missing", "swapped", "nonclass"] {
        with_module(|module| {
            match changed {
                "missing" => module.delattr("NativeTensor").unwrap(),
                "swapped" => module
                    .add("NativeTensor", module.getattr("CompiledModel").unwrap())
                    .unwrap(),
                _ => module.add("NativeTensor", 17).unwrap(),
            }
            assert!(native_registration::probe(module).is_err(), "{changed}");
        });
    }
}

#[test]
fn native_registration_rejects_incomplete_or_reclassified_provenance() {
    with_module(|module| {
        let report = native_registration::probe(module).unwrap();
        native_registration::check_registrations(&report.registrations).unwrap();
        for changed in [
            "missing",
            "duplicate",
            "kind",
            "implementation",
            "owner",
            "position",
        ] {
            let mut rows = report.registrations.clone();
            match changed {
                "missing" => {
                    rows.pop();
                }
                "duplicate" => rows.push(rows[0].clone()),
                "kind" => rows[0].kind = "staticmethod".into(),
                "implementation" => rows[0].rust_name = "same_named_helper".into(),
                "owner" => rows[0].owner = Some("ForeignTensor".into()),
                _ => rows[0].line = 0,
            }
            assert!(
                native_registration::check_registrations(&rows).is_err(),
                "{changed}"
            );
        }
    });
}
