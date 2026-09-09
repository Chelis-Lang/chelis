//! Compiled positive and adversarial registrations for the binding census.

use pyo3::prelude::*;
use pyo3::types::{PyModule, PyType};

#[::pyo3::pyfunction(name = "public_function")]
fn numeric_function(value: f64) -> f64 {
    value
}

fn public_function() -> bool {
    true
}

#[::pyo3::pyclass]
pub struct NativeTensor {
    value: i32,
}

impl NativeTensor {
    fn dtype(&self) -> PyResult<&'static str> {
        Ok("f32")
    }
    fn new() -> Self {
        Self { value: 0 }
    }
    fn run(&self) -> bool {
        true
    }
}

#[::pyo3::pymethods]
impl NativeTensor {
    #[new]
    fn create(dtype: i32) -> Self {
        Self { value: dtype }
    }

    #[getter(dtype)]
    fn dtype_code(&self) -> i32 {
        self.value
    }

    #[setter(dtype)]
    fn assign_dtype(&mut self, dtype: i32) {
        self.value = dtype;
    }

    #[pyo3(name = "run")]
    fn compute(&self, value: f64) -> f64 {
        value
    }

    #[classmethod]
    #[pyo3(name = "from_code")]
    fn build(_cls: &Bound<'_, PyType>, dtype: i32) -> Self {
        Self { value: dtype }
    }

    #[staticmethod]
    #[pyo3(name = "accept")]
    fn accept_value(value: f64) -> f64 {
        value
    }

    fn __call__(&self, value: f64) -> f64 {
        value
    }
}

pub fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTensor>()?;
    module.add_function(::pyo3::wrap_pyfunction!(numeric_function, module)?)?;
    Ok(())
}

pub fn verify_execution() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_registration_probe").unwrap();
        register_module(&module).unwrap();
        let class = module.getattr("NativeTensor").unwrap();
        let value = class.call1((7,)).unwrap();
        assert_eq!(value.getattr("dtype").unwrap().extract::<i32>().unwrap(), 7);
        assert!(value.getattr("dtype").unwrap().extract::<String>().is_err());
        assert!(value.getattr("dtype_code").is_err());
        assert!(class.call0().is_err());
        assert!(class.call1(("not a dtype",)).is_err());
        value.setattr("dtype", 9).unwrap();
        assert_eq!(value.getattr("dtype").unwrap().extract::<i32>().unwrap(), 9);
        assert!(value.setattr("dtype", "not a dtype").is_err());
        for callable in [
            value.getattr("run").unwrap(),
            value.clone(),
            class.getattr("accept").unwrap(),
            module.getattr("public_function").unwrap(),
        ] {
            assert_eq!(
                callable.call1((2.5,)).unwrap().extract::<f64>().unwrap(),
                2.5
            );
            assert!(callable.call1(("not a number",)).is_err());
        }
        assert!(module.getattr("numeric_function").is_err());
        let made = class.call_method1("from_code", (11,)).unwrap();
        assert_eq!(made.getattr("dtype").unwrap().extract::<i32>().unwrap(), 11);
        assert!(class.call_method1("from_code", ("not a dtype",)).is_err());
        assert_eq!(NativeTensor::new().dtype().unwrap(), "f32");
        assert!(NativeTensor::new().run());
        assert!(public_function());
    });
}
