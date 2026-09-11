//! Current compiled descriptor identity; registration is an obligation, not authority.
use std::collections::BTreeMap;

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[path = "pyo3_registration.rs"]
mod source_registration;
pub use source_registration::Registration;

const SOURCE: &str = include_str!("../../crates/chelis-python/src/lib.rs");
const SLOTS: [(&str, &str, &str, &str, &str); 4] = [
    (
        "CompiledModel",
        "NativeCompiledModel",
        "__call__",
        "method",
        "wrapper_descriptor",
    ),
    (
        "NativeTensor",
        "NativeTensor",
        "__dlpack__",
        "method",
        "method_descriptor",
    ),
    (
        "NativeTensor",
        "NativeTensor",
        "__dlpack_device__",
        "method",
        "method_descriptor",
    ),
    (
        "NativeTensor",
        "NativeTensor",
        "shape",
        "getter",
        "getset_descriptor",
    ),
];

#[derive(Debug, Serialize)]
pub struct Descriptor {
    pub owner: String,
    pub rust_owner: String,
    pub python_name: String,
    pub kind: String,
    pub descriptor: String,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub source_path: &'static str,
    pub source_sha256: String,
    pub registrations: Vec<Registration>,
    pub classes: BTreeMap<String, String>,
    pub descriptors: Vec<Descriptor>,
}

pub fn check_registrations(rows: &[Registration]) -> Result<(), String> {
    if rows.len() != SLOTS.len() {
        return Err("native registration requires exactly four source owners".into());
    }
    for (_, rust_owner, name, kind, _) in SLOTS {
        let matching: Vec<_> = rows
            .iter()
            .filter(|row| row.owner.as_deref() == Some(rust_owner) && row.python_name == name)
            .collect();
        if matching.len() != 1
            || matching[0].rust_name != name
            || matching[0].kind != kind
            || matching[0].line == 0
            || matching[0].column == 0
        {
            return Err("missing, duplicate or changed native source registration".into());
        }
    }
    Ok(())
}

pub fn probe(module: &Bound<'_, PyModule>) -> Result<Report, String> {
    let compiled = chelis_python::capacity_census_class_types(module.py());
    let mut classes = BTreeMap::new();
    for (name, rust_owner, class) in compiled {
        let actual = module.getattr(name).map_err(|error| error.to_string())?;
        if !actual.is_instance_of::<PyType>() || !actual.is(&class) {
            return Err(format!("native compiled class identity changed: {name}"));
        }
        if classes.insert(name.into(), rust_owner.into()).is_some() {
            return Err("duplicate native compiled class identity".into());
        }
    }
    let registrations: Vec<_> = source_registration::registrations(SOURCE)?
        .into_iter()
        .filter(|row| {
            SLOTS.iter().any(|(_, rust_owner, name, _, _)| {
                row.owner.as_deref() == Some(*rust_owner) && row.python_name == *name
            })
        })
        .collect();
    check_registrations(&registrations)?;
    let mut descriptors = Vec::new();
    for (owner, rust_owner, name, kind, expected) in SLOTS {
        if classes.get(owner).map(String::as_str)
            != Some(format!("chelis_python::{rust_owner}").as_str())
        {
            return Err("native compiled class provenance changed".into());
        }
        let class = module.getattr(owner).map_err(|error| error.to_string())?;
        let raw = class
            .getattr("__dict__")
            .and_then(|dict| dict.get_item(name))
            .map_err(|error| error.to_string())?;
        let descriptor = raw
            .get_type()
            .name()
            .map_err(|error| error.to_string())?
            .to_string();
        if descriptor != expected
            || !raw
                .getattr("__objclass__")
                .map_err(|error| error.to_string())?
                .is(&class)
        {
            return Err(format!(
                "native descriptor kind or defining class changed: {owner}::{name}"
            ));
        }
        descriptors.push(Descriptor {
            owner: owner.into(),
            rust_owner: format!("chelis_python::{rust_owner}"),
            python_name: name.into(),
            kind: kind.into(),
            descriptor,
        });
    }
    Ok(Report {
        source_path: "crates/chelis-python/src/lib.rs",
        source_sha256: format!("{:x}", Sha256::digest(SOURCE.as_bytes())),
        registrations,
        classes,
        descriptors,
    })
}
