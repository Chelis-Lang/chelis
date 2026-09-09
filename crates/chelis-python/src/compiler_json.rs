//! Concrete Python string conversions for spec/11 §1.1's compiler payloads.
//!
//! Private fields retain each exact producer until its codec runs. These
//! adapters neither accept already encoded strings nor authorize other types.
use std::collections::BTreeMap;

use chelis_compiler_api::schema::{
    CheckResult, CompileResult, DesugarResult, EvalResult, TensorValue,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyString};

pub(super) struct CheckJson {
    value: CheckResult,
}

impl CheckJson {
    pub(super) fn new(value: CheckResult) -> Self {
        Self { value }
    }
}

impl<'py> IntoPyObject<'py> for CheckJson {
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

pub(super) struct CompileJson {
    value: CompileResult,
}

impl CompileJson {
    pub(super) fn new(value: CompileResult) -> Self {
        Self { value }
    }
}

impl<'py> IntoPyObject<'py> for CompileJson {
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

pub(super) struct DesugarJson {
    value: DesugarResult,
}

impl DesugarJson {
    pub(super) fn new(value: DesugarResult) -> Self {
        Self { value }
    }
}

impl<'py> IntoPyObject<'py> for DesugarJson {
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

pub(super) struct EvalJson {
    value: EvalResult,
}

impl EvalJson {
    pub(super) fn new(value: EvalResult) -> Self {
        Self { value }
    }
}

impl<'py> IntoPyObject<'py> for EvalJson {
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

pub(super) struct EvalBindingsJson {
    value: BTreeMap<String, TensorValue>,
}

impl EvalBindingsJson {
    pub(super) fn empty() -> Self {
        Self {
            value: BTreeMap::new(),
        }
    }

    pub(super) fn into_bindings(self) -> BTreeMap<String, TensorValue> {
        self.value
    }
}

impl<'py> FromPyObject<'py> for EvalBindingsJson {
    fn extract_bound(object: &Bound<'py, PyAny>) -> PyResult<Self> {
        let text = object.extract::<String>()?;
        let value = serde_json::from_str::<BTreeMap<String, TensorValue>>(&text)
            .map_err(|error| PyValueError::new_err(format!("invalid bindings json: {error}")))?;
        Ok(Self { value })
    }
}
