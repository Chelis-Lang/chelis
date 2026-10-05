//! Fixed-dtype JSON-number adapters over the canonical sealed scalar carrier.
//!
//! These adapters enforce representation and domain. Source syntax admission,
//! report provenance, and report counter relationships remain separate checks.

use chelis_types::{ElementRef, ScalarValue, scalar_from_f64, scalar_from_i64, types::Prim};
use schemars::{JsonSchema, r#gen::SchemaGenerator, schema::Schema};
use serde::{Deserialize, Serialize};

macro_rules! float_adapter {
    ($name:ident, $doc:literal, $admit:expr, $rule:literal, $min:expr, $max:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $name(ScalarValue);

        impl $name {
            pub fn new(value: f64) -> Result<Self, String> {
                let _fp_env = chelis_runtime::FpEnvGuard::enter();
                if !value.is_finite() || !($admit)(value) {
                    return Err($rule.to_string());
                }
                // F64 admission is an identity on a finite binary64 value.
                Ok(Self(
                    scalar_from_f64("wire-number", Prim::F64, value)
                        .expect("finite f64 stores exactly in f64"),
                ))
            }

            pub fn get(self) -> f64 {
                let _fp_env = chelis_runtime::FpEnvGuard::enter();
                match self.0.element_ref() {
                    ElementRef::F64(value) => value,
                    _ => unreachable!("fixed f64 carrier admission"),
                }
            }

            pub fn scalar(self) -> ScalarValue {
                let _fp_env = chelis_runtime::FpEnvGuard::enter();
                self.0
            }
        }

        impl TryFrom<ScalarValue> for $name {
            type Error = String;
            fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
                let ElementRef::F64(number) = value.element_ref() else {
                    return Err("fixed f64 number adapter requires exact f64 dtype".to_string());
                };
                if !number.is_finite() || !($admit)(number) {
                    return Err($rule.to_string());
                }
                Ok(Self(value))
            }
        }

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_f64(self.get())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                Self::new(f64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
            }
        }

        impl JsonSchema for $name {
            fn schema_name() -> String {
                stringify!($name).to_string()
            }
            fn json_schema(generator: &mut SchemaGenerator) -> Schema {
                let mut schema = f64::json_schema(generator).into_object();
                schema.number().minimum = Some($min);
                schema.number().maximum = Some($max);
                schema.into()
            }
        }
    };
}

float_adapter!(
    UnitInterval,
    "Finite binary64 report score or severity in [0,1].",
    |value: f64| (0.0..=1.0).contains(&value),
    "report number must be a finite f64 in [0,1]",
    0.0,
    1.0
);
float_adapter!(
    SourceFloat,
    "Finite lexical binary64 value, before source dtype admission.",
    |_: f64| true,
    "source float must be finite binary64",
    f64::MIN,
    f64::MAX
);

/// Exact signed lexical int64, before source dtype admission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceInteger(ScalarValue);

impl SourceInteger {
    pub fn new(value: i64) -> Self {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        Self(
            scalar_from_i64("wire-integer", Prim::Int64, value)
                .expect("int64 stores exactly in int64"),
        )
    }

    pub fn get(self) -> i64 {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        match self.0.element_ref() {
            ElementRef::I64(value) => value,
            _ => unreachable!("fixed int64 carrier admission"),
        }
    }

    pub fn scalar(self) -> ScalarValue {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        self.0
    }
}

impl TryFrom<ScalarValue> for SourceInteger {
    type Error = String;
    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        if value.prim() != Prim::Int64 {
            return Err("source integer requires exact int64 dtype".to_string());
        }
        Ok(Self(value))
    }
}

impl Serialize for SourceInteger {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i64(self.get())
    }
}

impl<'de> Deserialize<'de> for SourceInteger {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(i64::deserialize(deserializer)?))
    }
}

impl JsonSchema for SourceInteger {
    fn schema_name() -> String {
        "SourceInteger".to_string()
    }
    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        i64::json_schema(generator)
    }
}

/// Nonnegative exact int64 fixed count. This is not allocation authority.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonnegativeCount(ScalarValue);

impl NonnegativeCount {
    pub fn new(value: i64) -> Result<Self, String> {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        if value < 0 {
            return Err("fixed count must be a nonnegative int64".to_string());
        }
        Ok(Self(SourceInteger::new(value).scalar()))
    }

    pub fn get(self) -> i64 {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        match self.0.element_ref() {
            ElementRef::I64(value) => value,
            _ => unreachable!("fixed int64 carrier admission"),
        }
    }

    pub fn scalar(self) -> ScalarValue {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        self.0
    }
}

impl TryFrom<usize> for NonnegativeCount {
    type Error = String;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(i64::try_from(value).map_err(|_| "fixed count exceeds int64".to_string())?)
    }
}

impl TryFrom<ScalarValue> for NonnegativeCount {
    type Error = String;
    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        let number = SourceInteger::try_from(value)?.get();
        if number < 0 {
            return Err("fixed count must be a nonnegative int64".to_string());
        }
        Ok(Self(value))
    }
}

impl Serialize for NonnegativeCount {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i64(self.get())
    }
}

impl<'de> Deserialize<'de> for NonnegativeCount {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(i64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for NonnegativeCount {
    fn schema_name() -> String {
        "NonnegativeCount".to_string()
    }
    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let mut schema = i64::json_schema(generator).into_object();
        schema.number().minimum = Some(0.0);
        schema.into()
    }
}

/// Observed nonnegative int64 dimension. This carries a numeric extent;
/// it does not establish allocation capacity or representation equality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonnegativeExtent(ScalarValue);

impl NonnegativeExtent {
    pub fn new(value: i64) -> Result<Self, String> {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        if value < 0 {
            return Err("dimension extent must be a nonnegative int64".to_string());
        }
        Ok(Self(SourceInteger::new(value).scalar()))
    }

    pub fn get(self) -> i64 {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        match self.0.element_ref() {
            ElementRef::I64(value) => value,
            _ => unreachable!("fixed int64 extent admission"),
        }
    }

    pub fn scalar(self) -> ScalarValue {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        self.0
    }
}

impl TryFrom<usize> for NonnegativeExtent {
    type Error = String;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        Self::new(i64::try_from(value).map_err(|_| "dimension extent exceeds int64".to_string())?)
    }
}

impl TryFrom<ScalarValue> for NonnegativeExtent {
    type Error = String;
    fn try_from(value: ScalarValue) -> Result<Self, Self::Error> {
        let number = SourceInteger::try_from(value)?.get();
        if number < 0 {
            return Err("dimension extent must be a nonnegative int64".to_string());
        }
        Ok(Self(value))
    }
}

impl Serialize for NonnegativeExtent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_i64(self.get())
    }
}

impl<'de> Deserialize<'de> for NonnegativeExtent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(i64::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for NonnegativeExtent {
    fn schema_name() -> String {
        "NonnegativeExtent".to_string()
    }
    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let mut schema = i64::json_schema(generator).into_object();
        schema.number().minimum = Some(0.0);
        schema.into()
    }
}
