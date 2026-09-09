//! A typed Python string conversion for the two nonnumeric compiler results.
//!
//! Keep the result until PyO3 converts it. Returning an already serialized
//! String would erase the carrier graph from the registered signature.

use chelis_compiler_api::schema::{DecompileResult, ValidateResult};
use pyo3::prelude::*;
use pyo3::types::PyString;
use serde::Serialize;

mod sealed {
    pub trait Sealed {}
    impl Sealed for chelis_compiler_api::schema::DecompileResult {}
    impl Sealed for chelis_compiler_api::schema::ValidateResult {}
}

// The private supertrait prevents sibling modules from adding result types.
pub(super) trait SourceResult: Serialize + sealed::Sealed {}
impl SourceResult for DecompileResult {}
impl SourceResult for ValidateResult {}

pub(super) struct SourceJson<T> {
    value: T,
}

impl<T: SourceResult> SourceJson<T> {
    pub(super) fn new(value: T) -> Self {
        Self { value }
    }
}

impl<'py, T: SourceResult> IntoPyObject<'py> for SourceJson<T> {
    type Target = PyString;
    type Output = Bound<'py, PyString>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let encoded = serde_json::to_string(&self.value).map_err(|error| {
            super::ChelisError::new_err(format!("serialization failed: {error}"))
        })?;
        Ok(PyString::new(py, &encoded))
    }
}
